# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QB2 TP4 decoder derived from the completed single-chip OptimizedDecoder.

Mesh order owns Q heads [0:8,8:16,16:24,24:32], KV heads [0,0,1,1].
Page tables and positions are replicated; caches store one local KV head.
The residual is either replicated H6656 or hidden-sharded H1664, selected at
construction. Input and output use the same contract in both execution modes.
Inherited chunking keeps arbitrary logical lengths and append positions valid.
"""

import math

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.muse_glimmer_30b_qb2.tt.optimized_decoder import OptimizedDecoder


class MeshCCLManager(CCLManager):
    """Initialize semaphore words on every core eligible for CCL placement."""

    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


class MultichipDecoder(OptimizedDecoder):
    POLICY = dict(
        OptimizedDecoder.POLICY,
        sharded_residual=False,
        minimal_k=4,
        minimal_n=32,
        weight_bank_multiple=2,
        prefill_norm_block=True,
        topology="Ring",
        links=2,
        persistent_ccl=True,
        direct_all_reduce=True,
        ccl_dtype="bfloat8_b",
        prefill_sdpa_q=64,
        prefill_sdpa_k=128,
        **{
            "qkv.block": 16,
            "qkv.readers": 3,
            "self_attn.o_proj.block": 4,
            "self_attn.o_proj.readers": 3,
            "mlp.gate_up.block": 4,
            "mlp.gate_up.readers": 2,
            "mlp.down_proj.block": 6,
            "mlp.down_proj.readers": 2,
            "qkv.minimal_k": 8,
            "self_attn.o_proj.minimal_k": 8,
            "mlp.gate_up.minimal_k": 8,
        },
    )
    KIND_POLICY = {
        "sliding_attention": {
            "ccl_dtype": "bfloat16",
            "qkv.fp32": True,
            "qkv.minimal_n": 16,
            "norm_fidelity": "HiFi2",
            "input_layernorm.fidelity": "HiFi4",
            "pre_feedforward_layernorm.fidelity": "LoFi",
            "pre_feedforward_layernorm.fp32": False,
            "decode.mlp.down_proj.ccl_dtype": "bfloat8_b",
            "prefill_norm_sharded": True,
            "prefill_norm_inplace": True,
            "release_normalized": True,
            "prefill_norm_max_rows": 1024,
            "prefill_norm_min_rows": 64,
        },
        "full_attention": {
            "attention_act_dtype": "bfloat8_b",
            "norm_fidelity": "LoFi",
            "norm_fp32": False,
            "prefill_attention_act_dtype": "bfloat8_b",
            "prefill_mlp_act_dtype": "bfloat8_b",
            "prefill_input_l1": True,
            "prefill_input_l1_max_rows": 1024,
            "prefill_norm_sharded": True,
            "prefill_norm_inplace": True,
            "release_normalized": True,
            "prefill_norm_max_rows": 1024,
            "prefill_norm_min_rows": 64,
        },
    }

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, ccl=None, policy=None):
        import torch

        self = cls()
        self.policy = dict(cls.POLICY)
        c = getattr(hf_config, "text_config", hf_config)
        if (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (
            6656,
            19968,
            32,
            2,
            128,
        ):
            raise ValueError("Requires Muse-Glimmer-30B text configuration")
        if mesh_device.get_num_devices() != 4 or tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Requires the authorized QB2 1x4 mesh")
        if c.attention_bias or c.hidden_activation != "silu":
            raise ValueError("Unsupported bias / activation")
        self.config, self.device, self.layer_idx = c, mesh_device, layer_idx
        self.kind = c.layer_types[layer_idx]
        self.policy.update(cls.KIND_POLICY[self.kind])
        self.policy.update(policy or {})
        self.CHUNK_SIZE = self.policy.get("chunk_size", 4096)
        self.window = c.sliding_window if self.kind == "sliding_attention" else None
        self.use_rope = c.layer_rope_theta[layer_idx] != 0
        self.attention_scale = torch.tensor(128**-0.5 * c.qk_scale_factor, dtype=torch.float32).item()
        self.kv_dtype = getattr(ttnn, self.policy["kv_dtype"])
        self.sharded_residual = self.policy["sharded_residual"]
        self.ccl = ccl or MeshCCLManager(
            mesh_device, self.policy.get("links", 1), getattr(ttnn.Topology, self.policy.get("topology", "Linear"))
        )
        grid = mesh_device.compute_with_storage_grid_size()
        if self.ccl.mesh_device is not mesh_device or not all(
            self.ccl.ccl_cores.contains(ttnn.CoreCoord(x, y)) for y in range(grid.y) for x in range(grid.x)
        ):
            raise ValueError("Shared CCL manager must initialize semaphores on the full mesh worker grid")
        # Layer stacks pass one CCLManager. All buffers here are sequentially
        # consumed scratch; no residual/cache tensor aliases them across layers.
        # Pooling avoids one L1 all-reduce intermediate per decoder layer.
        if not hasattr(self.ccl, "muse_scratch_buffers"):
            self.ccl.muse_scratch_buffers = {}
        self.ccl_buffers = self.ccl.muse_scratch_buffers
        self.ar_semaphores = [ttnn.create_global_semaphore(mesh_device, self.ccl.ccl_cores, 0) for _ in range(2)]
        self.ar_index = 0
        self.grid = tuple(self.policy["grid"])
        self.core_grid = ttnn.CoreGrid(x=self.grid[0], y=self.grid[1])
        self.dram_sharded, self.sharded_norm = True, True
        self.residual_mem = self._width_mem(1664 if self.sharded_residual else 6656, exact=True)
        self.ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.attention_ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.policy["sdpa_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_configs = {}
        for phase in ("prefill", "decode"):
            self.norm_configs[phase] = {}
            for name in (
                None,
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                role = name or "qk_norm"

                def norm_option(field, default):
                    return self.policy.get(
                        f"{phase}.{role}.{field}",
                        self.policy.get(
                            f"{phase}.norm_{field}",
                            self.policy.get(f"{role}.{field}", self.policy.get(f"norm_{field}", default)),
                        ),
                    )

                self.norm_configs[phase][name] = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, norm_option("fidelity", "HiFi4")),
                    math_approx_mode=False,
                    fp32_dest_acc_en=norm_option("fp32", True),
                    packer_l1_acc=True,
                )
        self.sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("prefill_sdpa_grid", (8, 8))),
            q_chunk_size=self.policy["prefill_sdpa_q"],
            k_chunk_size=self.policy["prefill_sdpa_k"],
            exp_approx_mode=False,
        )
        self.sdpa_decode = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("sdpa_grid", (8, 8))),
            q_chunk_size=0,
            k_chunk_size=self.policy["sdpa_k"],
            exp_approx_mode=False,
        )
        prefix = f"model.language_model.layers.{layer_idx}."

        def raw(name):
            key = name + ".weight"
            return state_dict[key] if key in state_dict else state_dict[prefix + key]

        def upload(t, dtype, dim=None, memory=None, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh_device)
                    if dim is None
                    else ttnn.ShardTensorToMesh(mesh_device, dim=dim)
                ),
                memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
            )

        self.weights, self.norm_weights, self.mm_ck, self.decode_mem, self.decode_pc = {}, {}, {}, {}, {}
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            w = (1 + raw(name).float()).reshape(1, 1, 1, 6656)
            self.weights[name] = upload(
                w,
                getattr(ttnn, self.policy.get("norm_weight_dtype", "bfloat16")),
                3 if self.sharded_residual else None,
            )
            self.norm_weights[name] = upload(
                w.reshape(1, 1, 208, 32),
                getattr(ttnn, self.policy.get("norm_weight_dtype", "bfloat16")),
                2 if self.sharded_residual else None,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        packed = []
        for rank in range(4):
            q = raw("self_attn.q_proj")[rank * 1024 : (rank + 1) * 1024]
            k = raw("self_attn.k_proj")[(rank // 2) * 128 : (rank // 2 + 1) * 128]
            v = raw("self_attn.v_proj")[(rank // 2) * 128 : (rank // 2 + 1) * 128]
            g = raw("self_attn.gate_proj")[rank * 1024 : (rank + 1) * 1024]
            packed.append(torch.cat((q, k, v) if self.policy.get("separate_attention_gate") else (q, k, v, g), 0).t())
        tensors = {
            "qkv": (torch.cat(packed, 1), 1),
            "self_attn.o_proj": (raw("self_attn.o_proj").t(), 0),
            "mlp.down_proj": (raw("mlp.down_proj").t(), 0),
        }
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            if self.policy.get(name + ".topology", self.policy.get("output_topology", "reduce")) in (
                "gather",
                "fused_gather",
            ):
                tensors[name] = (tensors[name][0], 1)
        if self.policy.get("separate_attention_gate"):
            tensors["self_attn.gate_proj"] = (raw("self_attn.gate_proj").t(), 1)
        gate, up = raw("mlp.gate_proj").t(), raw("mlp.up_proj").t()
        if self.policy.get("separate_mlp"):
            tensors["mlp.gate_proj"] = (gate, 1)
            tensors["mlp.up_proj"] = (up, 1)
        tensors["mlp.gate_up"] = (
            torch.cat(
                [
                    torch.cat((gate[:, r * 4992 : (r + 1) * 4992], up[:, r * 4992 : (r + 1) * 4992]), 1)
                    for r in range(4)
                ],
                1,
            ),
            1,
        )
        self.prefill_weights = {k: v for k, v in self.weights.items()}
        for name, (w, dim) in tensors.items():
            k, n = w.shape
            k, n = (k // 4, n) if dim == 0 else (k, n // 4)
            original_multiple = self.policy.get("weight_bank_multiple", 1)
            self.policy["weight_bank_multiple"] = self.policy.get(name + ".readers", self.policy["readers"])
            memory = self._weight_mem(k, n)
            self.policy["weight_bank_multiple"] = original_multiple
            self.weights[name] = upload(w, self._weight_dtype(name), dim, memory)
            if name != "mlp.gate_up" or not self.policy.get("minimal_mlp"):
                self.prefill_weights[name] = ttnn.to_memory_config(self.weights[name], ttnn.DRAM_MEMORY_CONFIG)
            group = "mlp" if name.startswith("mlp") else "attention"
            self.mm_ck[name] = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(
                    ttnn.MathFidelity, self.policy.get(name + ".fidelity", self.policy[group + "_fidelity"])
                ),
                math_approx_mode=False,
                fp32_dest_acc_en=self.policy.get(name + ".fp32", self.policy["fp32"]),
                packer_l1_acc=True,
            )
            grid = tuple(self.policy.get(name + ".grid", self.grid))
            self.decode_mem[name] = self._width_mem(k, grid, exact=True)
            block = self.policy.get(name + ".block", self.policy["block"])
            self.decode_pc[name] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=math.ceil(n / (32 * self.decode_mem[name].shard_spec.grid.num_cores())),
                num_workers_per_dram_bank=self.policy.get(name + ".readers", self.policy["readers"]),
                fused_activation=(
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
                    if name == "mlp.gate_proj" and self.policy.get("fuse_silu")
                    else None
                ),
            )
        paired = torch.stack((gate.reshape(6656, 624, 32), up.reshape(6656, 624, 32)), 2).reshape(6656, 39936)
        self.minimal_gate_up = upload(paired, self._weight_dtype("mlp.gate_up"), 1)
        self.positions = upload(
            torch.arange(c.max_position_embeddings, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        if self.use_rope:
            inv = 1 / c.layer_rope_theta[layer_idx] ** (torch.arange(0, 128, 2).float() / 128)
            freq = torch.arange(c.max_position_embeddings).float()[:, None] * inv
            freq = torch.cat((freq, freq), -1)[None, None]
            self.cos = upload(freq.cos(), getattr(ttnn, self.policy.get("rope_dtype", "bfloat16")))
            self.sin = upload(freq.sin(), getattr(ttnn, self.policy.get("rope_dtype", "bfloat16")))
        return self

    def prefill_forward(self, x, *, kv_cache, page_table, start_pos=0, slots=None):
        # Establish a DRAM residual boundary before any inplace L1 norm. A
        # full-range slice and same-memory conversion can otherwise alias a
        # caller's already block-sharded L1 tensor. This is a no-op for the
        # normal prefill stack boundary and preserves caller-owned input data.
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return super().prefill_forward(x, kv_cache=kv_cache, page_table=page_table, start_pos=start_pos, slots=slots)

    def allocate_cache(self, *, batch_size=1, num_pages=None):
        """Caller may supply a pooled physical-page budget for a future layer stack."""
        pages = batch_size * self.config.max_position_embeddings // 32 if num_pages is None else num_pages
        if not 1 <= batch_size <= 32 or pages <= 0:
            raise ValueError("Requires batch1..32 and a positive physical-page budget")
        shape = (pages, 1, 32, 128)
        return tuple(
            ttnn.zeros(
                shape,
                dtype=self.kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _buffer(self, kind, shape, dtype):
        key = (kind, tuple(shape), dtype)
        if key not in self.ccl_buffers:
            self.ccl_buffers[key] = ttnn.empty(
                shape, device=self.device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        return self.ccl_buffers[key]

    def _gather(self, x, *, dtype=None):
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        dtype = dtype or getattr(ttnn, self.policy.get("ccl_dtype", "bfloat16"))
        if x.shape[-1] > 128 and x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        out = None
        if self._decode and self.policy.get("persistent_ccl"):
            out = self._buffer("ag", list(x.shape)[:-1] + [x.shape[-1] * 4], x.dtype)
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            persistent_output_buffer=out,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            **self._ccl_options("ag"),
        )

    def _ccl_options(self, kind):
        phase = "decode" if self._decode else "prefill"
        return {
            field: self.policy[f"{phase}.{kind}.{field}"]
            for field in ("chunks_per_sync", "num_workers_per_link", "num_buffers_per_channel")
            if f"{phase}.{kind}.{field}" in self.policy
        }

    def _reduce(self, x):
        phase = "decode" if self._decode else "prefill"
        dtype = getattr(
            ttnn,
            self.policy.get(
                f"{phase}.{self._active_projection}.ccl_dtype",
                self.policy.get(self._active_projection + ".ccl_dtype", self.policy.get("ccl_dtype", "bfloat16")),
            ),
        )
        if (
            (self._decode or (self.policy.get("short_all_reduce") and x.padded_shape[-2] == 32))
            and self.policy.get("direct_all_reduce")
            and not self.sharded_residual
        ):
            x = ttnn.to_memory_config(x, self.residual_mem)
            if x.dtype != dtype:
                x = ttnn.typecast(x, dtype)
            key = ("ar", tuple(x.shape), dtype)
            if key not in self.ccl_buffers:
                spec = self.residual_mem.shard_spec
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(spec.grid, [32, spec.shape[1] * 4], ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.ccl_buffers[key] = ttnn.empty(
                    list(x.shape)[:-1] + [x.shape[-1] * 4],
                    device=self.device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory,
                )
            semaphore = self.ar_semaphores[self.ar_index]
            self.ar_index = 1 - self.ar_index
            return ttnn.experimental.all_reduce_async(
                x,
                self.ccl_buffers[key],
                cluster_axis=1,
                mesh_device=self.device,
                multi_device_global_semaphore=semaphore,
                memory_config=self.residual_mem,
                dtype=dtype,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
            )
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        buffers = {}
        if self._decode and self.policy.get("persistent_ccl"):
            output = self._buffer("rs_output", list(x.shape)[:-1] + [x.shape[-1] // 4], dtype)
            if self.ccl.topology == ttnn.Topology.Ring:
                key = ("rs_staging", tuple(x.shape), dtype)
                if key not in self.ccl_buffers:
                    self.ccl_buffers[key] = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                        x, dim=3, topology=self.ccl.topology, cluster_axis=1
                    )
                staging = self.ccl_buffers[key]
                buffers = dict(persistent_output_buffers=[staging[0], output, staging[1]])
            else:
                buffers = dict(
                    persistent_output_buffers=[self._buffer("rs_intermediate", list(x.shape), dtype), output]
                )
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            **self._ccl_options("rs"),
            **buffers,
        )
        return y if self.sharded_residual else self._gather(y, dtype=dtype)

    def _linear(self, x, name):
        self._active_projection = name
        fused_dtype = getattr(ttnn, self.policy.get("fused_output_dtype", "bfloat16"))
        row = name in ("self_attn.o_proj", "mlp.down_proj")
        topology = self.policy.get(name + ".topology", self.policy.get("output_topology", "reduce")) if row else "none"
        if topology in ("gather", "fused_gather"):
            if topology == "fused_gather":
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                if x.dtype != ttnn.bfloat16:
                    x = ttnn.typecast(x, ttnn.bfloat16)
                m = x.padded_shape[-2] // 32
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=m,
                    per_core_N=2,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                _, y = ttnn.experimental.all_gather_matmul_async(
                    x,
                    ttnn.reshape(self.prefill_weights[name], (1, 1, *self.prefill_weights[name].shape)),
                    dim=3,
                    multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                    all_gather_core_grid_offset=(0, 6),
                    num_links=self.ccl.num_links,
                    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl.topology,
                    barrier_semaphore=self.ccl.get_barrier_semaphore(),
                    program_config=pc,
                    compute_kernel_config=self.mm_ck[name],
                    dtype=fused_dtype,
                )
            else:
                y = super()._linear(self._gather(x), name)
            return y if self.sharded_residual else self._gather(y)
        if topology == "fused_reduce":
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            shape = list(x.shape)[:-1] + [6656]
            intermediate = self._buffer("fused_rs_intermediate", shape, fused_dtype)
            output = self._buffer("fused_rs_output", shape[:-1] + [1664], fused_dtype)
            m = x.padded_shape[-2] // 32
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, min(m, 4)),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=math.ceil(m / min(m, 4)),
                per_core_N=26,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            _, y = ttnn.experimental.matmul_reduce_scatter_async(
                x,
                self.prefill_weights[name],
                persistent_intermediate_buffer=intermediate,
                persistent_output_buffer=output,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                reduce_scatter_core_grid_offset=(0, 6),
                num_links=self.ccl.num_links,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl.topology,
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                program_config=pc,
                compute_kernel_config=self.mm_ck[name],
                dtype=fused_dtype,
            )
            return y if self.sharded_residual else self._gather(y)
        y = super()._linear(x, name)
        return self._reduce(y) if row else y

    def _norm(self, x, name=None, *, eps=None):
        if name is None or not self.sharded_residual:
            chunk = self.policy.get("prefill_norm_chunk", 1024)
            if name and self._sharded_prefill_norm(x) and self.policy.get("prefill_norm_block"):
                height = x.padded_shape[-2]
                gy = min(8, height // 32)
                while (height // 32) % gy:
                    gy -= 1
                if height // gy > 128:
                    # The WIDTH fallback also needs bounded RMSNorm scratch.
                    # Split only irregular eager heights; prepared buckets
                    # retain their existing geometry and trace addresses.
                    chunk = min(chunk, 512)
            if name and self._sharded_prefill_norm(x) and x.shape[-2] > chunk:
                # Rows are independent in RMSNorm. Bound its L1 scratch without
                # shrinking the matmul/attention chunk or the public sequence.
                parts = []
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                for start in range(0, x.shape[-2], chunk):
                    stop = min(start + chunk, x.shape[-2])
                    part = ttnn.slice(x, [0, 0, start, 0], list(x.shape)[:2] + [stop, x.shape[-1]])
                    part = super()._norm(part, name, eps=eps)
                    parts.append(ttnn.to_memory_config(part, ttnn.DRAM_MEMORY_CONFIG))
                    del part
                return ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return super()._norm(x, name, eps=eps)
        ck = self.norm_configs["decode" if self._decode else "prefill"][name]
        if self._decode and self.policy.get("distributed_norm_sharded"):
            # Distributed norm requires a rectangle. H/TP=1664 has 52 tiles;
            # four width shards of 13 tiles avoid the ragged 52-core boundary.
            norm_grid = tuple(self.policy.get("distributed_norm_grid", (4, 1)))
            norm_mem = self._width_mem(1664, norm_grid, exact=True)
            x = ttnn.to_memory_config(x, norm_mem)
            if x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            width = norm_mem.shard_spec.shape[1] // 32
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=norm_grid,
                subblock_w=next(b for b in (4, 3, 2, 1) if width % b == 0),
                block_h=1,
                block_w=width,
                inplace=False,
            )
            stats = ttnn.rms_norm_pre_all_gather(x, program_config=pc, compute_kernel_config=ck)
            mem = self._width_mem(128, (1, 1), exact=True)
            key = ("norm_stats", tuple(stats.shape))
            if key not in self.ccl_buffers:
                self.ccl_buffers[key] = ttnn.empty(
                    list(stats.shape)[:-1] + [128],
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                )
            stats = ttnn.experimental.all_gather_async(
                stats,
                dim=3,
                cluster_axis=1,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
                memory_config=mem,
                persistent_output_buffer=self.ccl_buffers[key],
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
            )
            y = ttnn.rms_norm_post_all_gather(
                x,
                stats,
                epsilon=eps or self.config.rms_norm_eps,
                weight=self.norm_weights[name],
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=norm_mem,
            )
            return self._gather(y) if name in ("input_layernorm", "pre_feedforward_layernorm") else y
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=ck, dtype=ttnn.bfloat16)
        stats = self._gather(stats)
        y = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=eps or self.config.rms_norm_eps,
            weight=self.norm_weights[name],
            compute_kernel_config=ck,
        )
        return self._gather(y) if name in ("input_layernorm", "pre_feedforward_layernorm") else y

    def _project_attention(self, normalized):
        packed = self._linear(normalized, "qkv")
        if self.policy.get("separate_attention_gate"):
            return packed, None
        shape = list(packed.shape)
        gate = ttnn.slice(packed, [0, 0, 0, 1280], shape)
        return ttnn.slice(packed, [0, 0, 0, 0], shape[:-1] + [1280]), gate

    def _finish(self, residual, normalized, attn, gate=None):
        if not self._sharded_prefill_norm(residual):
            return super()._finish(residual, normalized, attn, gate)
        gate = self._linear(normalized, "self_attn.gate_proj") if gate is None else gate
        del normalized
        gated = ttnn.multiply(
            gate,
            ttnn.to_memory_config(attn, gate.memory_config()),
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        del gate, attn
        projected = self._linear(gated, "self_attn.o_proj")
        del gated
        x = self._add(residual, self._norm(projected, "post_attention_layernorm", eps=self.config.post_norm_eps))
        del projected
        normalized = self._norm(x, "pre_feedforward_layernorm")
        mlp = self._mlp(normalized)
        # Block-sharded prefill norms otherwise retain both the previous norm
        # output and the postnorm input in L1, colliding with RMSNorm's buffers.
        del normalized
        return self._add(x, self._norm(mlp, "post_feedforward_layernorm", eps=self.config.post_norm_eps))

    def _minimal_config(self, x):
        name = getattr(self, "_active_projection", "")

        def value(key, default):
            return self.policy.get(name + "." + key, self.policy.get(key, default))

        bm = min(value("minimal_m", 4), x.padded_shape[-2] // 32)
        return ttnn.MinimalMatmulConfig(
            M_block_size=bm,
            K_block_size=value("minimal_k", 4),
            N_block_size=value("minimal_n", 32),
            subblock_h=2 if bm % 2 == 0 else 1,
            subblock_w=2 if value("fp32", False) else 4,
            compute_with_storage_grid_size=ttnn.CoreCoord(*value("minimal_grid", (11, 10))),
        )

    def _mlp(self, normalized):
        self._active_projection = "mlp.gate_up"
        if self._decode and self.policy.get("separate_mlp"):
            gate = self._linear(normalized, "mlp.gate_proj")
            up = self._linear(normalized, "mlp.up_proj")
            product = ttnn.multiply(
                gate, up, input_tensor_a_activations=[] if self.policy.get("fuse_silu") else [ttnn.UnaryOpType.SILU]
            )
            return self._linear(product, "mlp.down_proj")
        minimal = self.policy.get("minimal_mlp")
        if not self._decode and self.policy.get("short_prefill_dram") and normalized.padded_shape[-2] == 32:
            minimal = False
        if minimal and (minimal == "both" or (minimal == "decode") == self._decode):
            product = ttnn.experimental.minimal_matmul(
                (
                    normalized
                    if (
                        self._sharded_prefill_norm(normalized)
                        and normalized.is_sharded()
                        and not self.policy.get("prefill_norm_to_dram")
                    )
                    else ttnn.to_memory_config(
                        normalized,
                        (ttnn.L1_MEMORY_CONFIG if self._prefill_input_l1(normalized) else ttnn.DRAM_MEMORY_CONFIG),
                    )
                ),
                self.minimal_gate_up,
                config=self._minimal_config(normalized),
                fuse_swiglu=True,
                compute_kernel_config=self.mm_ck["mlp.gate_up"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=(
                    getattr(ttnn, self.policy.get("prefill_mlp_act_dtype", "bfloat16"))
                    if not self._decode
                    else ttnn.bfloat16
                ),
            )
            return self._linear(product, "mlp.down_proj")
        if self.policy.get("pack_mlp"):
            packed = self._linear(normalized, "mlp.gate_up")
            shape = list(packed.shape)
            gate = ttnn.slice(packed, [0, 0, 0, 0], shape[:-1] + [4992])
            up = ttnn.slice(packed, [0, 0, 0, 4992], shape)
        else:
            gate = self._linear(normalized, "mlp.gate_proj")
            up = self._linear(normalized, "mlp.up_proj")
        product = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=(
                [] if self._decode and self.policy.get("fuse_silu") else [ttnn.UnaryOpType.SILU]
            ),
        )
        return self._linear(product, "mlp.down_proj")

    def decode_forward(self, x, *, kv_cache, page_table, current_pos, rope_cos=None, rope_sin=None):
        """One on-device step; position/page/RoPE inputs can change between replays."""
        self._decode = True
        if self.policy.get("keep_residual"):
            x = ttnn.to_memory_config(x, self.residual_mem)
        batch = x.shape[2]
        partial_row = batch > 8 and batch % 8 != 0
        normalized = self._norm(x, "input_layernorm")
        packed, attention_gate = self._project_attention(normalized)
        heads_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        if self.policy.get("aligned_heads"):
            cores = ttnn.num_cores_to_corerangeset(10, ttnn.CoreCoord(8, 8), row_wise=True)
            packed = ttnn.to_memory_config(
                packed,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
        if packed.dtype != ttnn.bfloat16:
            packed = ttnn.typecast(packed, ttnn.bfloat16)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            packed if self.policy.get("sharded_heads") else ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG),
            num_heads=8,
            num_kv_heads=1,
            memory_config=heads_mem,
        )
        # Native RMSNorm cannot consume height shards; V stays in its cache layout.
        q, k = (self._norm(q), self._norm(k))
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        if self.use_rope:
            if rope_cos is None or rope_sin is None:
                raise ValueError("RoPE device tensors are required for sliding layers")
            if partial_row:
                # Native decode RoPE launches on a bounding rectangle. For an
                # incomplete core row, make users its sequence axis instead.
                cos, sin = (
                    ttnn.reshape(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG), (1, 1, batch, 128))
                    for t in (rope_cos, rope_sin)
                )
                q, k = (ttnn.transpose(self._rope(ttnn.transpose(t, 1, 2), cos, sin), 1, 2) for t in (q, k))
            else:
                cos, sin = (ttnn.to_memory_config(t, mem) for t in (rope_cos, rope_sin))
                q, k = (
                    ttnn.experimental.rotary_embedding_hf(
                        ttnn.to_memory_config(t, mem), cos, sin, is_decode_mode=True, compute_kernel_config=self.ck
                    )
                    for t in (q, k)
                )
        # Fused cache update needs disjoint K/V cores, including at batch32.
        kgrid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(i % 8, 4 + i // 8), ttnn.CoreCoord(i % 8, 4 + i // 8)) for i in range(batch)}
        )
        kmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(kgrid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        k = ttnn.to_memory_config(k, kmem)
        # V already uses these cores; head creation writes the cache layout directly.
        ttnn.experimental.paged_fused_update_cache(
            kv_cache[0], k, kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table
        )
        # Both NoPE and the partial-core-row RoPE path produce interleaved Q.
        # SDPA accepts L1 only for a sharded Q tensor.
        if not q.is_sharded() or self.policy.get("sdpa_q_dram_control"):
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        if self.policy.get("full_tile_sdpa_control"):
            # Diagnostic: query rows are independent for this one local KV head.
            # Q32 selects the full-tile accurate-exp path; only Q0..7 are used.
            q = ttnn.pad(
                ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG),
                ((0, 0), (0, 0), (0, 24), (0, 0)),
                0.0,
            )
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table,
            cur_pos_tensor=current_pos,
            sliding_window_size=self.window,
            scale=self.attention_scale,
            program_config=self.sdpa_decode,
            compute_kernel_config=self.attention_ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.policy.get("full_tile_sdpa_control"):
            attn = ttnn.slice(attn, [0, 0, 0, 0], [1, batch, 8, 128])
        # GQA SDPA requires DRAM output; concat-heads requires L1 height shards.
        attn = ttnn.to_memory_config(attn, mem)
        # A partial input row selects the subgrid kernel, which requires an
        # explicit 32-core destination grid before output allocation.
        attn = ttnn.experimental.nlp_concat_heads_decode(
            attn, num_heads=8, sub_core_grids=heads_mem.shard_spec.grid if partial_row else None
        )
        if not self.policy.get("keep_attn_l1"):
            attn = ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, ttnn.Shape([1, 1, batch, 1024]), ttnn.Shape([1, 1, 32, 1024]))
        if self.policy.get("release_decode"):
            del q, k, v, packed
            if attention_gate is not None:
                del normalized
                normalized = None
        return self._finish(x, normalized, attn, attention_gate)

    def _prefill_chunk(
        self,
        x,
        *,
        start_pos,
        kv_cache,
        page_table,
        slot,
        rope_cos=None,
        rope_sin=None,
        fill_page_table=None,
        chunk_start_tensor=None,
    ):
        self._decode = False
        length = x.shape[2]
        normalized = self._norm(x, "input_layernorm")

        packed, attention_gate = self._project_attention(normalized)
        if attention_gate is not None and self.policy.get("release_normalized"):
            del normalized
            normalized = None
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(packed, (1, length, 1280)), num_heads=8, num_kv_heads=1, transpose_key=False
        )
        q, k = (self._norm(q), self._norm(k))
        if self.use_rope:
            cos = (
                rope_cos
                if rope_cos is not None
                else ttnn.slice(self.cos, [0, 0, start_pos, 0], [1, 1, start_pos + length, 128])
            )
            sin = (
                rope_sin
                if rope_sin is not None
                else ttnn.slice(self.sin, [0, 0, start_pos, 0], [1, 1, start_pos + length, 128])
            )
            q, k = (self._rope(q, cos, sin), self._rope(k, cos, sin))
        pages = (
            fill_page_table
            if fill_page_table is not None
            else ttnn.slice(page_table, [slot, start_pos // 32], [slot + 1, (start_pos + length) // 32])
        )
        for cache, new in zip(kv_cache, (k, v)):
            ttnn.experimental.paged_fill_cache(cache, ttnn.typecast(new, self.kv_dtype), pages, batch_idx=0)
        row = (
            page_table
            if chunk_start_tensor is not None
            else ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
        )
        sdpa = self.sdpa
        q_chunk = self.policy.get("prefill_sdpa_q", 32)
        k_chunk = self.policy.get("prefill_sdpa_k", 32)
        if start_pos % q_chunk or start_pos % k_chunk:
            # Chunked SDPA indexes blocks from the absolute cache position.
            # Append prefixes are page-aligned, not necessarily 256-aligned.
            while start_pos % q_chunk:
                q_chunk //= 2
            while start_pos % k_chunk:
                k_chunk //= 2
            sdpa = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("prefill_sdpa_grid", (8, 8))),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
        position_args = (
            {"chunk_start_idx_tensor": chunk_start_tensor}
            if chunk_start_tensor is not None
            else {"chunk_start_idx": start_pos}
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            row,
            **position_args,
            scale=self.attention_scale,
            sliding_window_size=self.window,
            program_config=sdpa,
            compute_kernel_config=self.attention_ck,
        )
        attn = ttnn.reshape(ttnn.transformer.concatenate_heads(attn), (1, 1, length, 1024))
        return self._finish(x, normalized, attn, attention_gate)
