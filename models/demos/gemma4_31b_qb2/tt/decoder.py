# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Gemma4 31B TP4 decoder for four Blackhole chips.

Mesh [1,4], local Q heads8, local KV heads4(sliding)/1(full). Setup transforms
HF weights once. Prefill [B,1,S,Hr] has Hr5376 replicated for S<=128 and
Hr1344 hidden-sharded for S>128; decode [1,1,B,5376] is replicated. Choose
ownership once at stack entry using prefill_input_width(S). Every layer's
output has its input ownership. Page tables and absolute positions replicate.
Caches contain local heads; bounded sliding page tables repeat exactly nine
distinct private physical pages/user, while full attention requires one physical page
per logical page. All valid logical S1..262144 and B1..32 are preserved.
No tensor host boundary occurs in either runtime forward.
The bounded ring supports forward decoding with complete live history, not
arbitrary rollback into evicted tokens. Page-ID changes require live KV migration.
"""

import ttnn
from models.demos.gemma4_31b_qb2.tt.sdpa_l1 import BLACKHOLE_L1_BYTES, full_sdpa_fp32_l1_end
from models.demos.gpt_oss.tt.ccl import CCLManager


class MeshCCL(CCLManager):
    """Ping-pong ownership over the actual QB2 worker grid, including columns8–10."""

    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


# Input cores, readers per DRAM bank, and K tiles per decode matmul block.
DECODE_GEOMETRY = {
    "qkv": (8, 2, 21),
    "gate_up": (56, 2, 3),
    "o": (8, 2, 8),
    "down": (8, 2, 21),
}


class Decoder:
    PAGE_SIZE = 128
    # An unaligned 1024-token live window can straddle nine 128-token pages.
    SLIDING_WINDOW_PAGES = 1024 // PAGE_SIZE + 1
    PREFILL_CHUNK = 1024

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, shared_setup=None):
        """Load one layer and share the stack's collective and rotary buffers."""
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        c = hf_config
        if (
            mesh_device.get_num_devices() != 4
            or tuple(mesh_device.shape) != (1, 4)
            or mesh_device.arch() != ttnn.Arch.BLACKHOLE
            or ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.P300_X2
        ):
            raise ValueError("Gemma4 31B QB2 requires a P300_X2 cluster with four Blackhole devices in a [1, 4] mesh")
        assert (c.hidden_size, c.intermediate_size, c.num_attention_heads) == (5376, 21504, 32)
        assert not c.enable_moe_block and (not c.hidden_size_per_layer_input) and (not c.num_kv_shared_layers)
        assert not c.attention_bias and c.attention_k_eq_v and (not c.use_double_wide_mlp)
        assert c.use_bidirectional_attention == "vision" and c.hidden_activation == "gelu_pytorch_tanh"
        assert c.max_position_embeddings == 262144 and c.sliding_window == 1024
        self = cls()
        self.device = mesh_device
        self.kind = c.layer_types[layer_idx]
        self.activation_dtype = ttnn.bfloat16
        self.residual_dtype = ttnn.bfloat16
        self.norms_dtype = ttnn.bfloat16
        self.PREFILL_CHUNK = 1024 if self.kind == "sliding_attention" else 6656
        if self.PREFILL_CHUNK <= 0 or self.PREFILL_CHUNK % self.PAGE_SIZE:
            raise ValueError("Compute chunks must contain a positive whole number of cache pages")
        if self.kind == "sliding_attention" and self.PREFILL_CHUNK != 1024:
            raise ValueError("Sliding attention requires its fixed 1024-row history chunk")
        self.residual_width = 5376
        shared_setup = {} if shared_setup is None else shared_setup
        if "ccl" not in shared_setup:
            shared_setup["ccl"] = MeshCCL(mesh_device, 2, ttnn.Topology.Ring)
        self.ccl = shared_setup["ccl"]
        assert self.ccl.mesh_device is mesh_device
        assert self.ccl.num_links == 2
        assert self.ccl.topology == ttnn.Topology.Ring
        self.matmul_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.hidden_size = c.hidden_size
        self.max_context = c.max_position_embeddings
        if "prefill_output_page_table" not in shared_setup:
            shared_setup["prefill_output_page_table"] = ttnn.from_torch(
                torch.arange(self.max_context // 32, dtype=torch.int32).reshape(1, -1),
                device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        self.prefill_output_page_table = shared_setup["prefill_output_page_table"]
        self.eps = c.rms_norm_eps
        self.sliding = self.kind == "sliding_attention"
        assert self.kind in ("sliding_attention", "full_attention")
        self.head_dim = c.head_dim if self.sliding else c.global_head_dim
        self.kv_heads = (c.num_key_value_heads if self.sliding else c.num_global_key_value_heads) // 4
        self.window = c.sliding_window if self.sliding else None
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.pref_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.decode_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.pref_sdpa_by_q = {
            q: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*[11, 10]),
                q_chunk_size=q,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
            for q in (32, 64, 128, 256, 512, 1024)
            if q <= (128 if self.sliding else 256)
        }
        self.pref_sdpa_selections = {}
        prefix = f"model.language_model.layers.{layer_idx}."
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sd:
            sd = state_dict

        def tensor(w, dtype=ttnn.bfloat16, dim=None, decode_only=None):
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            if decode_only is not None:
                # These packed matrices have no prefill consumer. Construct
                # their final native DRAM layout directly so releasing a
                # temporary interleaved copy cannot fragment the weight arena.
                assert dim == 1
                assert w.dtype == torch.bfloat16, "Direct packed setup requires the native BF16 checkpoint values"
                kdim, global_width = w.shape
                ndim = global_width // 4
                dg = mesh_device.dram_grid_size()
                banks = dg.x * dg.y
                readers = DECODE_GEOMETRY[decode_only][1]
                shard_width = (ndim + banks * readers * 32 - 1) // (banks * readers * 32) * readers * 32
                # The selected decode-only matrices have exact bank geometry;
                # other weights retain the existing padded-reader conversion.
                assert shard_width * banks == ndim
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
                memory_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(grid, (kdim, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
                )
            return ttnn.from_torch(
                w.contiguous(),
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh_device)
                    if dim is None
                    else ttnn.ShardTensorToMesh(mesh_device, dim=dim)
                ),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
            )

        def linear(name):
            return tensor(
                sd[name + ".weight"].T, ttnn.bfloat4_b, dim=0 if name in ("self_attn.o_proj", "mlp.down_proj") else 1
            )

        q, k = (sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"])
        v = sd["self_attn.v_proj.weight"] if self.sliding else k

        def packed(*weights):
            # Mesh rank owns complete local Q/K/V or gate/up blocks.
            chunks = [w.chunk(4, dim=0) for w in weights]
            return torch.cat([torch.cat([w[r] for w in chunks], dim=0) for r in range(4)], dim=0).T

        self.qkv = tensor(
            packed(q, k, v),
            ttnn.bfloat4_b,
            dim=1,
            decode_only="qkv" if not self.sliding else None,
        )
        if not self.sliding:
            self.qk = tensor(packed(q, k), ttnn.bfloat4_b, dim=1)
        self.o = linear("self_attn.o_proj")
        self.gate, self.up, self.down = (linear("mlp." + name + "_proj") for name in ("gate", "up", "down"))
        self.gate_up = tensor(
            packed(sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"]),
            ttnn.bfloat4_b,
            dim=1,
            decode_only="gate_up",
        )
        self.norms = {
            name: tensor(sd[name + ".weight"].reshape(1, 1, 1, -1))
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            )
        }
        self.norm_rm = {
            name: ttnn.from_torch(
                sd[name + ".weight"].reshape(1, 1, -1, 32).contiguous(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=self.norms_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if (name not in ["self_attn.q_norm", "self_attn.k_norm"])
                    else ttnn.L1_MEMORY_CONFIG
                ),
            )
            for name in self.norms
        }
        self.local_norms = {
            name: tensor(sd[name + ".weight"].reshape(1, 1, 1, -1), dim=3)
            for name in self.norms
            if not name.startswith("self_attn")
        }
        # Non-AR candidate buffers remain layer-owned. Minimal AR scratch uses
        # two paired slots: the intervening all-rank reduction fences reuse.
        # This contract requires complete decoders, ordered O/down on CQ0 and
        # the common subdevice; neither host enqueue order nor semaphore reset
        # alone would make a single slot safe. Keep the pool alive with traces.
        self.decode_ccl_buffers = {}
        self.decode_ar_slots = {}
        assert self.residual_width == 5376
        dtype = ttnn.bfloat8_b
        signature = (dtype, 28, self.ccl.num_links, ttnn.Topology.Ring)
        pools = shared_setup.setdefault("decode_ar_pools", {})
        if signature not in pools:
            mem = self._mem(self.hidden_size * 4, 28)
            pools[signature] = {
                "mesh": mesh_device,
                "signature": signature,
                "slots": {
                    name: {
                        "scratch": ttnn.empty(
                            [1, 1, 32, self.hidden_size * 4],
                            dtype=dtype,
                            layout=ttnn.TILE_LAYOUT,
                            device=mesh_device,
                            memory_config=mem,
                        ),
                        "semaphore": ttnn.create_global_semaphore(mesh_device, self.ccl.ccl_cores, 0),
                    }
                    for name in ("o", "down")
                },
            }
        pool = pools[signature]
        assert pool["mesh"] is mesh_device and pool["signature"] == signature
        self.decode_ar_slots = pool["slots"]
        self.decode_weights = {}
        dg = mesh_device.dram_grid_size()
        banks = dg.x * dg.y
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
        for name in DECODE_GEOMETRY:
            weight = getattr(self, name)
            kdim, ndim = list(weight.shape)[-2:]
            readers = DECODE_GEOMETRY[name][1]
            shard_width = ((ndim + banks * readers * 32 - 1) // (banks * readers * 32)) * readers * 32
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (kdim, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.decode_weights[name] = ttnn.to_memory_config(weight, mem)
            if weight.memory_config().is_sharded():
                assert weight.memory_config() == mem
                assert weight.buffer_unique_id() == self.decode_weights[name].buffer_unique_id()
        # Separate prefill gate/up weights are already resident. Decode
        # owns its DRAM-sharded packed copy; retain no unused duplicate.
        del self.gate_up
        if not self.sliding:
            # Tied full-attention prefill consumes qk. Decode already owns
            # its separate DRAM-sharded qkv; this setup copy has no reader.
            del self.qkv
        self.layer_scalar = sd["layer_scalar"].item()
        table_key = ("rope", self.kind)
        if table_key not in shared_setup:
            rotary = Gemma4TextRotaryEmbedding(c)
            cos, sin = rotary(
                torch.empty(1, dtype=torch.bfloat16),
                torch.arange(self.max_context).reshape(1, -1),
                layer_type=self.kind,
            )
            self.cos, self.sin = (tensor(cos[0]), tensor(sin[0]))
            # Embedding requires row-major tables. Hoist full-table untilizes out
            # of decode; retain tiled tables for prefill slicing.
            self.decode_cos = ttnn.to_layout(self.cos, ttnn.ROW_MAJOR_LAYOUT)
            self.decode_sin = ttnn.to_layout(self.sin, ttnn.ROW_MAJOR_LAYOUT)
            if self.sliding:
                qi = torch.arange(1024)[:, None]
                ki = torch.arange(2048)[None, :]
                self.history_mask = tensor(
                    torch.where((ki > qi) & (ki <= 1024 + qi), 0.0, float("-inf")).bfloat16().reshape(1, 1, 1024, 2048)
                )
            names = ["cos", "sin", "decode_cos", "decode_sin"] + (["history_mask"] if self.sliding else [])
            shared_setup[table_key] = {name: getattr(self, name) for name in names}
        for name, value in shared_setup[table_key].items():
            setattr(self, name, value)
        self.decode_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*[8, 4]),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.decode_sdpa_medium = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4), q_chunk_size=0, k_chunk_size=64, exp_approx_mode=False
        )
        self.decode_sdpa_large = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            q_chunk_size=0,
            k_chunk_size=64,
            exp_approx_mode=False,
        )
        self.decode_sdpa_maximum = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        return self

    def _prepare_prefill_norm_stats(self):
        """Native maximum owners, with exact native geometry views per shape.

        Keep four distinct owners per layer: reuse of a norm name is separated
        by another all-rank norm/reduction on CQ0 and the common subdevice.
        Prefix views retain the FP32 DRAM owner, page geometry and topology.
        All owners and views are prepared before cache allocation or tracing.
        """
        names = (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        )
        # The native stats factory reads geometry only. Borrow existing BF16
        # tile storage for this descriptor instead of allocating and releasing
        # a large unused template between persistent norm owners. The view's
        # values are intentionally never consumed by a numerical operation.
        elements = self.PREFILL_CHUNK * 1344
        prefix_shape = ttnn.Shape([1, 1, elements // self.head_dim, self.head_dim])
        prefix = ttnn.reshape(self.cos, prefix_shape, prefix_shape)
        x = ttnn.experimental.view(prefix, [1, 1, self.PREFILL_CHUNK, 1344])
        assert x.buffer_unique_id() == self.cos.buffer_unique_id()
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
        assert x.memory_config() == ttnn.DRAM_MEMORY_CONFIG

        def native_stats(inp, name):
            return ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                inp,
                cluster_axis=1,
                mesh_device=self.device,
                num_links=self.ccl.num_links,
                weight=self.local_norms[name],
            )

        # Query the native factory rather than duplicating worker/round
        # sizing. Its geometry does not depend on the norm weight's values.
        shapes = {}
        for rows in range(self.PAGE_SIZE, self.PREFILL_CHUNK + 1, self.PAGE_SIZE):
            shape = ttnn.Shape([1, 1, rows, 1344])
            probe = native_stats(ttnn.reshape(x, shape, shape), names[0])
            assert probe.dtype == ttnn.float32 and probe.layout == ttnn.ROW_MAJOR_LAYOUT
            assert probe.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            assert probe.buffer_aligned_page_size() == 8192
            shapes[rows] = (probe.shape, probe.padded_shape)
            if rows == self.PREFILL_CHUNK:
                first_owner = probe
            del probe
        self.prefill_norm_stats_owners = {}
        for name in names:
            owner = first_owner if name == names[0] else native_stats(x, name)
            self.prefill_norm_stats_owners[name] = owner
            for rows, (logical, padded) in shapes.items():
                assert logical[-1] == owner.shape[-1] and logical[-2] <= owner.shape[-2]
                view = ttnn.reshape(owner, logical, padded)
                assert view.buffer_unique_id() == owner.buffer_unique_id()
                assert view.buffer_address() == owner.buffer_address()
                self.decode_ccl_buffers["fused_norm", name, (1, 1, rows, 1344)] = view

    def _gather(self, x, memory_config=None, *, output=None):
        memory_config = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=output,
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            topology=self.ccl.topology,
            memory_config=memory_config,
            num_workers_per_link=None,
            chunks_per_sync=None,
            num_buffers_per_channel=None,
        )

    def _validate_precise_bf8_row(self, x, dtype):
        """Check actual storage; logical batch 1..32 occupies one padded tile."""
        shape = tuple(x.shape)
        if (
            len(shape) != 4
            or shape[:2] != (1, 1)
            or not 1 <= shape[2] <= 32
            or shape[3] != 5376
            or tuple(x.padded_shape) != (1, 1, 32, 5376)
            or x.dtype != dtype
            or x.layout != ttnn.TILE_LAYOUT
            or not x.is_sharded()
            or tuple(x.get_tile().tile_shape) != (32, 32)
        ):
            raise ValueError("Precise BF8 rows require tiled [1,1,B,5376], B1..32 padded to32")
        mem = x.memory_config()
        shard = mem.shard_spec
        if (
            mem.buffer_type != ttnn.BufferType.L1
            or mem.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
            or tuple(shard.shape) != (32, 96)
            or shard.grid.num_cores() != 56
            or shard.orientation != ttnn.ShardOrientation.ROW_MAJOR
        ):
            raise ValueError("Precise BF8 rows require actual 56-core L1 width shards [32,96]")

    def _decode_reduce(self, x, name):
        memory = self._mem(self.hidden_size, 28)
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, memory)
        slot = self.decode_ar_slots[name]
        # Sliding rows retain the BF16 completed sum. Full-attention rows retain
        # the measured BF8 sum; both communicate and receive BF8 payloads.
        result = ttnn.experimental.all_reduce_async(
            x,
            slot["scratch"],
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=slot["semaphore"],
            memory_config=memory,
            dtype=ttnn.bfloat16 if self.sliding else ttnn.bfloat8_b,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            fp32_dest_acc=False,
        )
        return result if self.sliding else ttnn.typecast(result, ttnn.bfloat16)

    def prefill_input_width(self, sequence_length):
        """Residual ownership chosen once at entry to the entire layer stack."""
        return 1344 if sequence_length > 128 else 5376

    def _reduce(self, x, residual_width=None):
        residual_width = self.residual_width if residual_width is None else residual_width
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        dtype = ttnn.bfloat16
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        if residual_width == self.hidden_size:
            y = ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.typecast(y, ttnn.bfloat16) if y.dtype != ttnn.bfloat16 else y
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            topology=self.ccl.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_workers_per_link=None,
        )
        return ttnn.typecast(y, ttnn.bfloat16) if y.dtype != ttnn.bfloat16 else y

    def _pref_linear(self, x, name, activation=None):
        import math

        weight = getattr(self, name)
        grid = (8, 8) if self.sliding and name in ("qkv", "qk") else (11, 8) if self.sliding else (11, 10)
        wide_qkv = (
            self.sliding
            and name == "qkv"
            and x.shape[2] == 1024
            and weight.dtype == ttnn.bfloat4_b
            and x.dtype == ttnn.bfloat16
        )
        if wide_qkv:
            grid = (11, 8)
        gx, gy = grid
        m, n = (x.shape[2] + 31) // 32, weight.shape[-1] // 32
        gy = min(gy, m)
        pm, pn = math.ceil(m / gy), math.ceil(n / gx)
        sw = max(i for i in range(1, min(pn, 8) + 1) if pn % i == 0)
        sh = 1 if sw < pn else max(i for i in range(1, min(pm, 8 // sw) + 1) if pm % i == 0)
        cap = (8 if name == "o" else 14) if self.sliding else (4 if name == "o" else 6)
        expected_input = ttnn.bfloat8_b if name == "o" and not self.sliding else ttnn.bfloat16
        if not wide_qkv and weight.dtype == ttnn.bfloat4_b and x.dtype == expected_input:
            if not self.sliding and x.shape[2] in (128, 256):
                cap = 64 if name == "o" else 42
            elif x.shape[2] == 1024:
                cap = 32 if name == "o" else 28
        if wide_qkv:
            cap = 42
        kt = weight.shape[-2] // 32
        block = max(i for i in range(1, min(kt, cap) + 1) if kt % i == 0)
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=block,
            per_core_M=pm,
            per_core_N=pn,
            out_subblock_h=sh,
            out_subblock_w=sw,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH) if activation else None,
        )
        return ttnn.linear(
            x,
            weight,
            program_config=config,
            compute_kernel_config=self.matmul_compute,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _dlinear(self, x, name):
        weight = self.decode_weights[name]
        kdim, ndim = weight.shape[-2], weight.shape[-1]
        cores, readers, k_block = DECODE_GEOMETRY[name]
        row = name in ("o", "down")
        staged = row and self.sliding
        x = ttnn.to_memory_config(x, self._mem(kdim, cores))
        config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k_block,
            per_core_M=1,
            per_core_N=3 if staged else (ndim + cores * 32 - 1) // (cores * 32),
            num_workers_per_dram_bank=readers,
        )
        y = ttnn.linear(
            x,
            weight,
            program_config=config,
            compute_kernel_config=self.matmul_compute,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b if row and not staged else ttnn.bfloat16,
        )
        if staged:
            # Keep the BF16 projection in its 56-core owner until the native
            # precise BF8 pack. Direct BF8 matmul output has different rounding.
            self._validate_precise_bf8_row(y, ttnn.bfloat16)
            shape, memory = tuple(y.shape), y.memory_config()
            y = ttnn.typecast(y, ttnn.bfloat8_b, memory_config=memory)
            self._validate_precise_bf8_row(y, ttnn.bfloat8_b)
            if tuple(y.shape) != shape or y.memory_config() != memory:
                raise ValueError("BF8 packing changed the projection's shape or shard ownership")
        return y

    def _resnorm(self, x, name, decode):
        if decode:
            return self._dnorm(x, name)
        if x.shape[-1] == self.hidden_size:
            rows = x.padded_shape[-2]
            mem = ttnn.create_sharded_memory_config(
                (rows, 192),
                ttnn.CoreGrid(x=7, y=4),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.to_memory_config(x, mem)
            cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[7, 4],
                subblock_w=2,
                block_h=rows // 32,
                block_w=6,
                inplace=False,
            )
            y = ttnn.rms_norm(
                x,
                weight=self.norm_rm[name],
                epsilon=self.eps,
                program_config=cfg,
                compute_kernel_config=self.compute,
                memory_config=mem,
            )
            return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        key = ("fused_norm", name, tuple(x.shape))
        if key not in self.decode_ccl_buffers:
            raise ValueError(f"Prefill norm shape {tuple(x.shape)} has no prepared persistent view")
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x,
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            topology=self.ccl.topology,
            epsilon=self.eps,
            weight=self.local_norms[name],
            persistent_output_buffer=self.decode_ccl_buffers[key],
            num_preferred_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )

    def _input_norm(self, x, name, decode):
        if not decode and name == "pre_feedforward_layernorm" and x.shape[-1] != self.hidden_size:
            # Place the larger gather output before its temporary local norm
            # output. The latter can then retire into the free tail needed by
            # gate/up rather than leave a small hole between live MLP owners.
            shape = list(x.shape)
            shape[-1] *= 4
            output = ttnn.empty(
                shape, dtype=x.dtype, layout=x.layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            n = self._resnorm(x, name, False)
            return self._gather(n, output=output)
        n = self._resnorm(x, name, decode)
        return self._gather(n) if x.shape[-1] != self.hidden_size else n

    def _prefill_mlp(self, x):
        n = self._input_norm(x, "pre_feedforward_layernorm", False)
        gate = self._pref_linear(n, "gate", "gelu_tanh")
        up = self._pref_linear(n, "up")
        del n
        product = ttnn.multiply(gate, up)
        del gate, up
        local = self._pref_linear(product, "down")
        del product
        reduced = self._reduce(local, x.shape[-1])
        del local
        post = self._resnorm(reduced, "post_feedforward_layernorm", False)
        del reduced
        return ttnn.add(
            x,
            post,
            dtype=self.residual_dtype,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)],
        )

    def _decode_finish(self, residual, attention):
        post = self._resnorm(self._decode_reduce(self._dlinear(attention, "o"), "o"), "post_attention_layernorm", True)
        mem = self._mem(self.hidden_size, 28)
        residual = ttnn.to_memory_config(residual, mem)
        x = ttnn.add(residual, post, memory_config=mem, dtype=self.residual_dtype)
        n = self._input_norm(x, "pre_feedforward_layernorm", True)
        packed = self._dlinear(n, "gate_up")
        gate, up = packed[..., :5376], packed[..., 5376:]
        product = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.GELU_TANH])
        post = self._resnorm(
            self._decode_reduce(self._dlinear(product, "down"), "down"), "post_feedforward_layernorm", True
        )
        return ttnn.add(
            x,
            post,
            memory_config=mem,
            dtype=self.residual_dtype,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)],
        )

    def _prefill_sdpa_config(self, q, cache, page_table, *, start=0):
        k_chunk = 128
        if not self.sliding and start % k_chunk:
            raise ValueError(f"Full-attention SDPA start {start} must align to K chunk {k_chunk}")
        eligible = sorted((c for c in self.pref_sdpa_by_q if q.shape[2] % c == 0 and start % c == 0), reverse=True)
        if not eligible:
            raise ValueError(f"No SDPA Q chunk aligns with rows {q.shape[2]} and start {start}")
        if self.sliding or not self.pref_compute.fp32_dest_acc_en:
            return self.pref_sdpa_by_q[eligible[0]]
        tile_bytes = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048, ttnn.float32: 4096}
        grid = [11, 10]
        page_bytes = page_table.buffer_aligned_page_size()
        for chunk in eligible:
            end = full_sdpa_fp32_l1_end(
                rows=q.shape[2],
                q_chunk=chunk,
                k_chunk=128,
                heads=q.shape[0] * q.shape[1],
                cores=grid[0] * grid[1],
                head_dim=q.shape[-1],
                q_tile_bytes=tile_bytes[q.dtype],
                kv_tile_bytes=tile_bytes[cache.dtype],
                page_table_bytes=page_bytes,
            )
            if end <= BLACKHOLE_L1_BYTES:
                key = (q.shape[2], start, str(q.dtype), str(cache.dtype), page_bytes)
                self.pref_sdpa_selections[key] = dict(
                    rows=q.shape[2],
                    start=start,
                    q_dtype=str(q.dtype),
                    kv_dtype=str(cache.dtype),
                    page_table_bytes=page_bytes,
                    q_chunk=chunk,
                    k_chunk=128,
                    cb_region_end=end,
                )
                return self.pref_sdpa_by_q[chunk]
        raise ValueError("No configured full-attention SDPA Q chunk fits Blackhole L1")

    def _concat_heads(self, attn):
        return ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _decode_rope(self, x, cos, sin):
        # The native single-user RoPE path exposes its 32 padded head rows.
        # SDPA derives the GQA ratio from logical Q heads: restore local head
        # ownership before attention (8 Q heads, not 32 padding rows).
        result = self._apply_decode_rope(x, cos, sin)
        return ttnn.reshape(result, ttnn.Shape(list(x.shape)), ttnn.Shape(list(x.padded_shape)))

    def _prefill_qkv(self, x, start):
        tied = not self.sliding
        fused = self._pref_linear(self._input_norm(x, "input_layernorm", False), "qk" if tied else "qkv")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            fused,
            num_heads=8,
            num_kv_heads=self.kv_heads,
            transpose_k_heads=False,
            kv_tied=tied,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del fused
        if tied:
            # Full attention derives rotated K from normalized V below.
            # The raw K output has no consumer after head creation.
            del k
        # Only a start-zero, whole-width prefix can alias the table buffer.
        rope_shape = ttnn.Shape([1, 1, x.shape[2], self.head_dim])
        if start == 0:
            cos = ttnn.reshape(self.cos, rope_shape, rope_shape)
            sin = ttnn.reshape(self.sin, rope_shape, rope_shape)
        else:
            cos = ttnn.reshape(self.cos[start : start + x.shape[2], :], rope_shape)
            sin = ttnn.reshape(self.sin[start : start + x.shape[2], :], rope_shape)
        q = self._norm(q, "self_attn.q_norm")
        q = ttnn.experimental.rotary_embedding(q, cos, sin)
        if not self.sliding:
            v = self._norm(v)
            k = ttnn.experimental.rotary_embedding(ttnn.multiply(v, self.norms["self_attn.k_norm"]), cos, sin)
        else:
            k = self._norm(k, "self_attn.k_norm")
            k = ttnn.experimental.rotary_embedding(k, cos, sin)
            v = self._norm(v)
        return (q, k, v)

    def prefill_forward(
        self, x, *, page_table, kv_cache, consume_input=False, start_pos=0, history=None, return_history=False
    ):
        """Prefill a page-aligned span, optionally retaining sliding history for a continuation."""
        if start_pos < 0 or start_pos % self.PAGE_SIZE:
            raise ValueError("Prefill start must be a nonnegative cache-page boundary")
        if start_pos and self.sliding and history is None:
            raise ValueError("Sliding continuation requires its preceding K/V history")
        next_history = None
        batch, one, seq, width = x.shape
        assert one == 1 and width == self.prefill_input_width(seq) and (0 < seq <= self.max_context)
        assert page_table.shape[0] == batch and page_table.shape[1] * self.PAGE_SIZE >= start_pos + seq
        if consume_input:
            assert batch == 1 and x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
            assert x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        users = []
        for user in range(batch):
            parts = []
            stream = seq > self.PREFILL_CHUNK
            if stream:
                output_shape = ttnn.Shape([(seq + 31) // 32, 1, 32, width])
                if consume_input:
                    # Same-width BF16 tile view. Only completed chunk pages
                    # are written; future hidden rows remain untouched.
                    output = ttnn.reshape(x, output_shape, output_shape)
                    assert output.buffer_unique_id() == x.buffer_unique_id()
                else:
                    output = ttnn.empty(
                        output_shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
            previous_kv = history
            pt = (
                ttnn.reshape(
                    page_table, ttnn.Shape([1, page_table.shape[1]]), ttnn.Shape([1, page_table.padded_shape[1]])
                )
                if user == 0
                else page_table[user : user + 1, :]
            )
            for start in range(0, seq, self.PREFILL_CHUNK):
                end = min(start + self.PREFILL_CHUNK, seq)
                inp = (
                    ttnn.reshape(x, ttnn.Shape([1, 1, end, width]), ttnn.Shape([1, 1, (end + 31) // 32 * 32, width]))
                    if user == 0 and start == 0
                    else x[user : user + 1, :, start:end, :]
                )
                pad = -inp.shape[2] % self.PAGE_SIZE
                if pad:
                    inp = ttnn.pad(inp, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
                absolute_start = start_pos + start
                q, k, v = self._prefill_qkv(inp, absolute_start)
                if not self.sliding:
                    q = ttnn.typecast(q, ttnn.bfloat8_b)
                sdpa_config = self._prefill_sdpa_config(q, kv_cache[0], pt, start=absolute_start)
                length = inp.shape[2]
                fill_pt = pt[:, absolute_start // self.PAGE_SIZE : (absolute_start + length) // self.PAGE_SIZE]
                ttnn.experimental.paged_fill_cache(
                    kv_cache[0], ttnn.typecast(k, kv_cache[0].dtype), fill_pt, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    kv_cache[1], ttnn.typecast(v, kv_cache[1].dtype), fill_pt, batch_idx=0
                )
                if not self.sliding:
                    # Full attention consumes the paged cache, not these
                    # temporary heads. CQ0 orders cache fill before SDPA.
                    del k, v
                if self.sliding and previous_kv is not None:
                    both_k = ttnn.concat([previous_kv[0], k], dim=2)
                    both_v = ttnn.concat([previous_kv[1], v], dim=2)
                    attn = ttnn.transformer.scaled_dot_product_attention(
                        q,
                        both_k,
                        both_v,
                        attn_mask=self.history_mask[
                            :, :, : inp.shape[2], 1024 - previous_kv[0].shape[2] : 1024 + inp.shape[2]
                        ],
                        is_causal=False,
                        scale=1.0,
                        program_config=sdpa_config,
                        compute_kernel_config=self.pref_compute,
                    )
                    del both_k, both_v
                elif self.sliding:
                    attn = ttnn.transformer.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=True,
                        scale=1.0,
                        sliding_window_size=self.window,
                        program_config=sdpa_config,
                        compute_kernel_config=self.pref_compute,
                    )
                else:
                    attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                        q,
                        kv_cache[0],
                        kv_cache[1],
                        pt,
                        chunk_start_idx=absolute_start,
                        scale=1.0,
                        program_config=sdpa_config,
                        compute_kernel_config=self.pref_compute,
                    )
                if self.sliding:
                    if return_history and end == seq:
                        # Keep one extra page so a later unaligned scheduler boundary
                        # can replay its partial page without losing the prior window.
                        tails = []
                        for index, value in enumerate((k, v)):
                            live = value[:, :, : end - start, :]
                            joined = ttnn.concat([previous_kv[index], live], dim=2) if previous_kv is not None else live
                            tail_start = max(0, joined.shape[2] - (self.window + self.PAGE_SIZE))
                            tails.append(joined[:, :, tail_start:, :])
                        next_history = tuple(tails)
                    previous_kv = (k, v)
                if attn.dtype != ttnn.bfloat16 and self.sliding:
                    attn = ttnn.typecast(attn, ttnn.bfloat16)
                joined = self._concat_heads(attn)
                del q, attn
                local = self._pref_linear(joined, "o")
                del joined
                reduced = self._reduce(local, width)
                del local
                post = self._resnorm(reduced, "post_attention_layernorm", False)
                # Reuse the earliest private owner: reduced has finished its
                # norm read on CQ0, while post and a nonaliasing input slice
                # are also disposable after this add. This lets later MLP
                # allocations coalesce without mutating caller x.
                reuse_post = width != self.hidden_size
                private_input = (reuse_post) and inp.buffer_unique_id() != x.buffer_unique_id()
                residual_output = None
                if reuse_post:
                    candidates = [reduced, post]
                    if private_input:
                        candidates.append(inp)
                    residual_output = min(candidates, key=lambda tensor: tensor.buffer_address())
                    del candidates
                mlp_input = ttnn.add(inp, post, output_tensor=residual_output, dtype=self.residual_dtype)
                del residual_output
                del inp, post, reduced
                result = self._prefill_mlp(mlp_input)
                del mlp_input
                # Keep canonical tile padding while aliasing leading output rows.
                # The storage view retains the result allocation owner.
                logical = end - start
                if stream:
                    # Raw BF16 tile copies into the final sequence allocation.
                    # A32-row block preserves exactly the canonical tile tail.
                    if consume_input:
                        assert result.buffer_unique_id() != x.buffer_unique_id()
                    if start == 0:
                        output.update_tensor_topology(result.tensor_topology())
                    tiled_shape = ttnn.Shape([1, 1, (logical + 31) // 32 * 32, width])
                    output_part = ttnn.reshape(result, tiled_shape, tiled_shape)
                    output_pages = self.prefill_output_page_table[:, start // 32 : (end + 31) // 32]
                    ttnn.experimental.paged_fill_cache(output, output_part, output_pages, batch_idx=0)
                    del output_part, output_pages
                else:
                    parts.append(
                        ttnn.reshape(
                            result,
                            ttnn.Shape([1, 1, logical, width]),
                            ttnn.Shape([1, 1, (logical + 31) // 32 * 32, width]),
                        )
                    )
                del result
            if self.sliding:
                del previous_kv, k, v
            if stream:
                users.append(
                    ttnn.reshape(
                        output, ttnn.Shape([1, 1, seq, width]), ttnn.Shape([1, 1, (seq + 31) // 32 * 32, width])
                    )
                )
                del output
            elif len(parts) > 1:
                # Concatenating a partial tile would untilize every chunk and
                # allocate extra full-sequence copies. Only the final chunk
                # has padding: join physical tile views, then restore true S.
                tiled_parts = [ttnn.reshape(part, part.padded_shape, part.padded_shape) for part in parts]
                joined_parts = ttnn.concat(tiled_parts, dim=2)
                users.append(
                    ttnn.reshape(
                        joined_parts, ttnn.Shape([1, 1, seq, width]), ttnn.Shape([1, 1, (seq + 31) // 32 * 32, width])
                    )
                )
                del tiled_parts, joined_parts
            else:
                users.append(parts[0])
        result = ttnn.concat(users, dim=0) if batch > 1 else users[0]
        return (result, next_history) if return_history else result

    def decode_forward(self, x, *, positions, page_table, kv_cache, rope_positions=None, cyclic_cache=True):
        """One token per user; all runtime data inputs and outputs are device tensors."""
        batch = x.shape[2]
        assert x.shape[0] == x.shape[1] == 1 and x.shape[3] == self.residual_width
        assert 1 <= batch <= 32 and page_table.shape[0] == batch and (positions.shape[0] == batch)
        normalized = self._input_norm(x, "input_layernorm", True)
        fused = self._dlinear(normalized, "qkv")
        spec = fused.memory_config().shard_spec
        assert spec.shape[1] * spec.grid.num_cores() == fused.shape[-1], "QKV head reader requires exact output shards"
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=8, num_kv_heads=self.kv_heads, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(fused)
        ttnn.deallocate(normalized)
        cache_mem = k.memory_config()
        head_mem = ttnn.L1_MEMORY_CONFIG
        if batch != 1:
            q, k, v = (ttnn.to_memory_config(t, head_mem) for t in (q, k, v))
        # External fixed-slot schedulers may use -1 to suppress cache/SDPA rows.
        # Their separate nonnegative RoPE indices avoid an out-of-bounds lookup.
        idx = ttnn.reshape(
            ttnn.typecast(positions if rope_positions is None else rope_positions, ttnn.uint32), (1, batch)
        )
        # A full32-index row selects the embedding kernel with native TILE
        # output, avoiding separate small tilize operations.
        idx = ttnn.pad(idx, [(0, 0), (0, 32 - batch)], value=0)
        cos = ttnn.embedding(idx, self.decode_cos, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.decode_sin, layout=ttnn.TILE_LAYOUT)
        # Explicit padded shapes crop logical rows without a device copy.
        logical = ttnn.Shape([1, 1, batch, self.head_dim])
        padded = ttnn.Shape([1, 1, 32, self.head_dim])
        cos = ttnn.reshape(cos, logical, padded)
        sin = ttnn.reshape(sin, logical, padded)
        q = self._decode_rope(self._norm(q, "self_attn.q_norm"), cos, sin)
        if not self.sliding:
            v = self._norm(v)
            k = self._decode_rope(ttnn.multiply(v, self.norms["self_attn.k_norm"]), cos, sin)
        else:
            k = self._decode_rope(self._norm(k, "self_attn.k_norm"), cos, sin)
            v = self._norm(v)
        k = ttnn.to_memory_config(k, cache_mem)
        # Fused cache update requires disjoint K/V cores. K occupies rows0-3
        # at batch32; V starts at row4 and supports every logical batch1-32.
        vgrid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(i % 8, 4 + i // 8), ttnn.CoreCoord(i % 8, 4 + i // 8)) for i in range(batch)}
        )
        vmem = ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            vgrid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        v = ttnn.to_memory_config(v, vmem)
        bounded = cyclic_cache and self.sliding and page_table.padded_shape[1] > kv_cache[0].shape[0]
        modulo = {"cache_position_modulo": self.SLIDING_WINDOW_PAGES * self.PAGE_SIZE} if bounded else {}
        if bounded:
            # Fused update still enforces logical pages <= physical pages.
            # Native separate updates support a bounded pool explicitly. The
            # documented cyclic table repeats its first nine private pages.
            ttnn.experimental.paged_update_cache(
                kv_cache[0], k, update_idxs_tensor=positions, page_table=page_table, **modulo
            )
            ttnn.experimental.paged_update_cache(
                kv_cache[1], v, update_idxs_tensor=positions, page_table=page_table, **modulo
            )
        else:
            ttnn.experimental.paged_fused_update_cache(
                kv_cache[0], k, kv_cache[1], v, update_idxs_tensor=positions, page_table=page_table
            )
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        sdpa_config = self.decode_sdpa_config
        if not self.sliding:
            horizon = page_table.shape[1] * self.PAGE_SIZE
            # The maximum-capacity sweep slightly favors K128; preserve K64
            # for the separately measured intermediate/long horizons.
            if horizon >= self.max_context:
                sdpa_config = self.decode_sdpa_maximum
            elif horizon > 4096:
                sdpa_config = self.decode_sdpa_large
            elif horizon > 256:
                sdpa_config = self.decode_sdpa_medium
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            cur_pos_tensor=positions,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=self.window,
            program_config=sdpa_config,
            compute_kernel_config=self.decode_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **modulo,
        )
        # GQA SDPA requires interleaved output in this snapshot. Decode concat
        # requires height sharding and emits the width-sharded projection input.
        # This concat kernel requires a rectangular user-core grid. Prime
        # batches larger than either grid axis need internal user padding.
        grid = self.device.compute_with_storage_grid_size()
        widths = [w for w in range(1, grid.x + 1) if batch % w == 0 and batch // w <= grid.y]
        concat_batch = batch if widths else (batch + 7) // 8 * 8
        grid_width = max(widths) if widths else 8
        if concat_batch != batch:
            attn = ttnn.pad(attn, [(0, 0), (0, concat_batch - batch), (0, 0), (0, 0)], value=0.0)
        concat_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_width - 1, concat_batch // grid_width - 1))}
        )
        mem = ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            concat_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        joined = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(attn, mem), num_heads=8)
        joined = ttnn.reshape(
            joined, ttnn.Shape([1, 1, batch, joined.shape[3]]), ttnn.Shape([1, 1, 32, joined.shape[3]])
        )
        return self._decode_finish(x, joined)

    def _mem(self, width, cores):
        grid = self.device.compute_with_storage_grid_size()
        # Rectangular grids keep RMSNorm's multicast contract explicit.
        widths = [i for i in range(1, grid.x + 1) if cores % i == 0 and cores // i <= grid.y]
        gx = max(widths) if widths else grid.x
        core_grid = ttnn.CoreGrid(x=gx, y=cores // gx) if widths else ttnn.num_cores_to_corerangeset(cores, grid, True)
        return ttnn.create_sharded_memory_config(
            (32, ((width + cores * 32 - 1) // (cores * 32)) * 32),
            core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _dnorm(self, x, name):
        cores = 28
        mem = self._mem(self.hidden_size, cores)
        x = ttnn.to_memory_config(x, mem)
        block = self.hidden_size // cores // 32
        sub = max(i for i in range(1, 5) if block % i == 0)
        grid = mem.shard_spec.grid.bounding_box().end
        cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x + 1, grid.y + 1],
            subblock_w=sub,
            block_h=1,
            block_w=block,
            inplace=False,
        )
        return ttnn.rms_norm(
            x,
            weight=self.norm_rm[name],
            epsilon=self.eps,
            program_config=cfg,
            compute_kernel_config=self.compute,
            memory_config=mem,
        )

    def allocate_cache(self, *, physical_pages):
        """Setup helper. Capacity is controlled by the caller's page table."""
        shape = (physical_pages, self.kv_heads, self.PAGE_SIZE, self.head_dim)
        return tuple(
            (
                ttnn.zeros(
                    shape,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
        )

    def _norm(self, x, name=None):
        if (x.is_sharded()) and x.shape[1] == 1:
            original_mem = x.memory_config()
            mem = ttnn.MemoryConfig(
                (ttnn.TensorMemoryLayout.BLOCK_SHARDED),
                ttnn.BufferType.L1,
                original_mem.shard_spec,
            )
            x = ttnn.to_memory_config(x, mem)
            end = mem.shard_spec.grid.bounding_box().end
            cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[end.x + 1, end.y + 1],
                subblock_w=4,
                block_h=1,
                block_w=self.head_dim // 32,
                inplace=False,
            )
            result = ttnn.rms_norm(
                x,
                weight=self.norm_rm[name] if name else None,
                epsilon=self.eps,
                program_config=cfg,
                compute_kernel_config=self.compute,
                memory_config=mem,
            )
            return result
        small_l1 = x.padded_shape[2] <= 32
        return ttnn.rms_norm(
            x,
            weight=self.norms[name] if name else None,
            epsilon=self.eps,
            compute_kernel_config=self.compute,
            memory_config=ttnn.L1_MEMORY_CONFIG if small_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )

    def _apply_decode_rope(self, x, cos, sin):
        batch = x.shape[1]
        cos = ttnn.reshape(cos, (1, 1, batch, self.head_dim))
        sin = ttnn.reshape(sin, (1, 1, batch, self.head_dim))
        if batch == 1:
            return ttnn.experimental.rotary_embedding(
                x,
                cos,
                sin,
                token_index=0,
                memory_config=(ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG if x.shape[2] != 32 else (ttnn.DRAM_MEMORY_CONFIG)),
            )
        # Interleaved decode Q must reside in DRAM for SDPA. Write the
        # required transpose there directly; K can remain in L1 until update.
        result = ttnn.transpose(
            ttnn.experimental.rotary_embedding(ttnn.transpose(x, 1, 2), cos, sin),
            1,
            2,
            memory_config=(ttnn.DRAM_MEMORY_CONFIG if (x.shape[2] == 32) else x.memory_config()),
        )
        result = ttnn.reshape(
            result, ttnn.Shape([1, batch, x.shape[2], self.head_dim]), ttnn.Shape([1, batch, 32, self.head_dim])
        )
        return result
