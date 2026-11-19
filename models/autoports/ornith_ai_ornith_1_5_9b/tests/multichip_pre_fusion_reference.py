# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ornith Qwen3.5 text decoder, four-chip TP implementation based on the optimized decoder.

Default inputs/outputs are TP4 replicated residual BF16 [1, 1, logical_tokens, 4096].
Decode returns local width-sharded L1; packed batch rows stay there across layers.
The sharded residual candidate uses local width 1024 at both boundaries. Each request owns
its state; full attention uses 32-token paged KV, linear attention uses FP32
recurrent state and three convolution history rows. Setup performs conversion;
forward methods contain only TTNN operations and shape/chunk planning.
"""

import math
from dataclasses import dataclass, field

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.tt.ccl import CCLManager


def exact_k_shards(k, requested_cores, block=1):
    """Keep DRAM-matmul sender count consistent with the true, unpadded K.

    The single-reader kernel reconstructs its coordinate-table length from K
    blocks / blocks-per-shard. Oversized storage grids silently truncate that
    quotient, so choose input storage independently of output geometry.
    """
    assert k % 32 == 0 and requested_cores in (16, 32, 64)
    tiles = k // 32
    choices = [s for s in range(8, requested_cores + 1, 8) if tiles % s == 0 and (tiles // s) % block == 0]
    assert choices, (k, requested_cores, block)
    cores = max(choices)
    return cores, tiles // cores


@dataclass
class DecoderState:
    conv: object = None
    recurrent: object = None
    key: object = None
    value: object = None
    state_row: int | None = field(default=None, metadata={"host_metadata": True})
    page_offset: int = field(default=0, metadata={"host_metadata": True})
    page_count: int | None = field(default=None, metadata={"host_metadata": True})
    page_offsets: object = field(default=None, metadata={"host_metadata": True})


class MultichipCollectives:
    """One setup-owned workspace shared by a serial decoder stack on one mesh.

    Each layer issues two decode reductions, so ping-pong buffers are safely reused
    by the next layer on the same command queue. Concurrent stacks need separate
    workspaces. Sharing avoids 32 layers reserving 2 MiB/core on the same L1 grid.
    """

    def __init__(self, mesh_device, ccl=None):
        self.device = mesh_device
        self.ccl = ccl or CCLManager(
            mesh_device, num_links=ttnn.get_num_links(mesh_device, 1), topology=ttnn.Topology.Linear
        )
        self.ar_mem = ttnn.create_sharded_memory_config(
            (32, 128), ttnn.CoreGrid(x=8, y=4), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        buffer_mem = ttnn.create_sharded_memory_config(
            (32, 512), ttnn.CoreGrid(x=8, y=4), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        self.ar_buffers = [
            ttnn.empty(
                (1, 1, 32, 16384),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=buffer_mem,
            )
            for _ in range(2)
        ]
        self.ar_semaphores = [ttnn.create_global_semaphore(self.device, self.ccl.ccl_cores, 0) for _ in range(2)]


class MultichipDecoder(LightweightModule):
    attention_dtype = ttnn.bfloat8_b
    mlp_dtype = ttnn.bfloat8_b
    attention_fidelity = ttnn.MathFidelity.HiFi2
    mlp_fidelity = ttnn.MathFidelity.LoFi
    kv_dtype = ttnn.bfloat8_b
    decode_wave_heads = 48
    decode_wave_producers = 1

    def weight_dtype(self, name):
        return self.mlp_dtype if name.startswith(("mlp.", "gate_up", "packed_mlp")) else self.attention_dtype

    def matmul_compute(self, name):
        return self.mlp_compute if name.startswith(("mlp.", "gate_up", "packed_mlp")) else self.attention_compute

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        chunk_size=1024,
        residual_layout="replicated",
        collective="direct",
        fused_prefill=False,
        native_local=True,
        movement_optimized=True,
        collectives=None,
        readers=None,
        cores=None,
    ):
        import torch

        self = cls()
        c = getattr(hf_config, "text_config", hf_config)
        assert (c.hidden_size, c.intermediate_size, c.head_dim) == (4096, 12288, 256)
        assert tuple(mesh_device.shape) == (1, 4)
        assert chunk_size > 0 and chunk_size % 32 == 0
        assert residual_layout in ("sharded", "replicated")
        self.config, self.device, self.chunk_size = c, mesh_device, chunk_size
        self.kind = c.layer_types[layer_idx]
        self.residual_layout = residual_layout
        self.hidden_width = 1024 if residual_layout == "sharded" else 4096
        self.collective = collective
        self.fused_prefill = fused_prefill
        if fused_prefill:
            # The resident fused kernel uses both NoCs under dedicated ownership.
            # A 64-KiB plan excludes residency for local K>=1024 and keeps reads
            # on NoC0. This optional safe fusion is slower than the default path.
            bank = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank
            self.prefill_headroom = bank - 64 * 1024
        self.native_local = native_local and collective == "direct" and residual_layout == "replicated"
        self.movement_optimized = movement_optimized and self.native_local
        assert collective in ("direct", "separate")
        self.ccl = (
            collectives.ccl
            if collectives is not None
            else CCLManager(mesh_device, num_links=ttnn.get_num_links(mesh_device, 1), topology=ttnn.Topology.Linear)
        )
        if collective == "direct" and residual_layout == "replicated":
            workspace = collectives or MultichipCollectives(mesh_device, self.ccl)
            assert workspace.device is mesh_device
            self.collectives = workspace
            self.ar_mem = workspace.ar_mem
            self.ar_buffers = workspace.ar_buffers
            self.ar_semaphores = workspace.ar_semaphores
            self.ar_index = 0
            self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4), subblock_w=2, block_h=1, block_w=4, inplace=False
            )
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.attention_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.attention_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.mlp_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.mlp_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        prefix = f"model.language_model.layers.{layer_idx}."
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict

        def tensor(x, dtype=ttnn.bfloat16, dim=None, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh_device) if dim is None else ttnn.ShardTensorToMesh(mesh_device, dim=dim)
            )
            return ttnn.from_torch(
                x.contiguous(), device=mesh_device, dtype=dtype, layout=layout, memory_config=mem, mesh_mapper=mapper
            )

        self.weights = {}
        for name, value in sd.items():
            if name.endswith("norm.weight"):
                offset = not name.endswith("linear_attn.norm.weight")
                value = (value.float() + int(offset)).reshape(1, 1, 1, -1)
                dim = (
                    3
                    if residual_layout == "sharded"
                    and name in ("input_layernorm.weight", "post_attention_layernorm.weight")
                    else None
                )
                self.weights[name] = tensor(value, dim=dim)
        hosts = {
            name: []
            for name in (
                "packed_input",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj",
                "packed_mlp",
            )
        }
        taps, decays, biases = [], [], []
        for rank in range(4):

            def column(name, width):
                return sd[name + ".weight"][rank * width : (rank + 1) * width].T

            if self.kind == "linear_attention":
                q, k, v = sd["linear_attn.in_proj_qkv.weight"].split([2048, 2048, 4096], dim=0)
                local = torch.cat(
                    [
                        q[rank * 512 : (rank + 1) * 512],
                        k[rank * 512 : (rank + 1) * 512],
                        v[rank * 1024 : (rank + 1) * 1024],
                    ],
                    dim=0,
                ).T
                z = column("linear_attn.in_proj_z", 1024)
                bb = torch.nn.functional.pad(column("linear_attn.in_proj_b", 8), (0, 24))
                aa = torch.nn.functional.pad(column("linear_attn.in_proj_a", 8), (0, 24))
                hosts["packed_input"].append(torch.cat([local, z, bb, aa], dim=1))
                q, k, v = sd["linear_attn.conv1d.weight"][:, 0, :].split([2048, 2048, 4096], dim=0)
                taps.append(
                    torch.cat(
                        [
                            q[rank * 512 : (rank + 1) * 512],
                            k[rank * 512 : (rank + 1) * 512],
                            v[rank * 1024 : (rank + 1) * 1024],
                        ],
                        dim=0,
                    )
                )
                decays.append(-sd["linear_attn.A_log"].float()[rank * 8 : (rank + 1) * 8].exp())
                biases.append(sd["linear_attn.dt_bias"].float()[rank * 8 : (rank + 1) * 8])
            else:
                qg = sd["self_attn.q_proj.weight"].reshape(16, 2, 256, 4096)[rank * 4 : (rank + 1) * 4]
                qg = qg.permute(1, 0, 2, 3).reshape(2048, 4096).T
                hosts["packed_input"].append(
                    torch.cat([qg, column("self_attn.k_proj", 256), column("self_attn.v_proj", 256)], dim=1)
                )
            gate, up = column("mlp.gate_proj", 3072), column("mlp.up_proj", 3072)
            hosts["mlp.gate_proj"].append(gate)
            hosts["mlp.up_proj"].append(up)
            hosts["packed_mlp"].append(
                torch.stack([gate.reshape(4096, 96, 32), up.reshape(4096, 96, 32)], dim=2).reshape(4096, 6144)
            )
            hosts["mlp.down_proj"].append(sd["mlp.down_proj.weight"][:, rank * 3072 : (rank + 1) * 3072].T)
            name = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
            hosts[name].append(sd[name + ".weight"][:, rank * 1024 : (rank + 1) * 1024].T)
        self.dram_weights, self.dram_layouts, self.decode_geometry = {}, {}, {}
        banks = mesh_device.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for name, pieces in hosts.items():
            k, n = pieces[0].shape
            joined = torch.cat(pieces, dim=0)
            self.weights[name + ".weight"] = tensor(joined, self.weight_dtype(name), dim=0)
            if name == "packed_mlp":
                continue
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.dram_weights[name] = tensor(joined, self.weight_dtype(name), dim=0, mem=mc)

            def layout(width, layout_cores):
                sw = math.ceil(width / (32 * layout_cores))
                return (
                    ttnn.create_sharded_memory_config(
                        (32, sw * 32),
                        ttnn.CoreGrid(x=8, y=layout_cores // 8),
                        ttnn.ShardStrategy.WIDTH,
                        use_height_and_width_as_shard_shape=True,
                    ),
                    sw,
                )

            # Real-weight TP4 sweep: preserve precision and select per-role geometry.
            selected = (64, 1, 4)
            if name == "packed_input":
                selected = (32 if self.kind == "linear_attention" else 64, 2, 2)
            elif name in ("mlp.gate_proj", "mlp.up_proj"):
                selected = (64, 2, 2)
            out_cores = selected[0] if cores is None else cores
            block = selected[1] if cores is None and readers is None else 1
            role_readers = selected[2] if readers is None else readers
            input_cores, _ = exact_k_shards(k, out_cores, block)
            im, _ = layout(k, input_cores)
            om, ow = layout(n, out_cores)
            if self.movement_optimized:
                actual_grid = ttnn.num_cores_to_corerangeset(
                    math.ceil(n / (ow * 32)), mesh_device.compute_with_storage_grid_size(), True
                )
                om = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(actual_grid, (32, ow * 32), ttnn.ShardOrientation.ROW_MAJOR),
                )
                if k == 4096:
                    im = self.ar_mem
                    input_cores = 32
                if name == "mlp.down_proj":
                    im = self.dram_layouts["mlp.gate_proj"][1]
                    input_cores = math.ceil(k / im.shard_spec.shape[1])
            self.decode_geometry[name] = dict(
                input_cores=input_cores,
                output_cores=out_cores,
                actual_output_cores=math.ceil(n / (ow * 32)),
                block=block,
                readers=role_readers,
            )
            cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=ow,
                num_workers_per_dram_bank=role_readers,
                use_multicore_pipeline=role_readers > 1,
            )
            self.dram_layouts[name] = (im, om, cfg)
        self.prefill_program = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=16,
            N_block_size=8,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=(11, 10),
        )
        if self.kind == "linear_attention":
            self.taps = [tensor(torch.cat([t[:, i] for t in taps]).reshape(1, 1, 1, -1), dim=3) for i in range(4)]
            self.float_taps = [ttnn.typecast(t, ttnn.float32) for t in self.taps]
            self.recurrent_write_mem = ttnn.create_sharded_memory_config(
                (32, 128), ttnn.CoreGrid(x=8, y=4), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            self.decay = tensor(torch.cat(decays).reshape(1, 1, 32), ttnn.float32, dim=2)
            self.dt_bias = tensor(torch.cat(biases).reshape(1, 1, 32), ttnn.float32, dim=2)
            ii = torch.arange(32)[:, None] < 16
            jj = torch.arange(32)[None, :] < 16
            const = [
                torch.eye(32),
                torch.ones(32, 32).tril(),
                torch.ones(32, 32),
                torch.cat([(ii & jj).float(), (~ii & ~jj).float(), (~ii & jj).float()], dim=1),
            ]
            self.gdn_constants = dict(
                zip(("eye", "tril", "ones", "masks"), [tensor(v.reshape(1, 1, *v.shape), ttnn.float32) for v in const])
            )
            self.conv_compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self.conv_start = tensor(torch.tensor([0], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            inv = 1.0 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
            angle = torch.arange(c.max_position_embeddings).float()[:, None] * inv[None, :]
            self.cos = tensor(torch.cat([angle, angle], -1).cos())
            self.sin = tensor(torch.cat([angle, angle], -1).sin())
            self.rope_decode = ttnn.to_layout(ttnn.concat([self.cos, self.sin], dim=-1), ttnn.ROW_MAJOR_LAYOUT)
            self.positions = tensor(
                torch.arange(c.max_position_embeddings, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.shard = ttnn.create_sharded_memory_config(
                (32, 256), ttnn.CoreGrid(x=1, y=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            self.value_shard = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                    (32, 256),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        return self

    def gather(self, x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            memory_config=x.memory_config(),
        )

    def reduce(self, x):
        # Fused row projection has already summed and scattered these columns.
        if x.shape[-1] == 1024:
            return x if self.residual_layout == "sharded" else self.gather(x)
        if x.shape[2] == 1 and self.collective == "direct" and self.residual_layout == "replicated":
            return self.decode_reduce(x)

        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            memory_config=x.memory_config(),
        )
        return result if self.residual_layout == "sharded" else self.gather(result)

    def decode_reduce(self, x):
        """Two-link asynchronous AR using the shared setup-owned receive buffers."""
        i = self.ar_index
        self.ar_index = 1 - i
        out = ttnn.experimental.all_reduce_async(
            x if self.movement_optimized and x.is_sharded() else ttnn.to_memory_config(x, self.ar_mem),
            self.ar_buffers[i],
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat16,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
        )
        return out if self.native_local else ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

    def allocate_state(self, *, context=None):
        context = context or self.config.max_position_embeddings
        assert 0 < context <= self.config.max_position_embeddings and context % 32 == 0

        def zero(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.zeros(
                shape, dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        if self.kind == "linear_attention":
            return DecoderState(
                conv=zero((1, 1, 3, 2048), layout=ttnn.ROW_MAJOR_LAYOUT),
                recurrent=zero((1, 8, 128, 128), ttnn.float32),
            )
        return DecoderState(
            key=zero((context // 32, 1, 32, 256), self.kv_dtype), value=zero((context // 32, 1, 32, 256), self.kv_dtype)
        )

    def allocate_batch_state(self, *, batch, context=None):
        """Allocate disjoint request regions in setup-owned pools.

        Keep all returned states alive with their traces. Page tables contain local
        page indices; this decoder adds each request's pool offset on device.
        To restart full attention, overwrite its valid prefix starting at zero and
        reset its position. Reset a linear request by writing zero conv/recurrent
        rows through write_linear_state before its new prefill. Never clear an
        entire pooled tensor to reset one request. Batch entries must identify
        distinct requests; duplicate state regions are invalid.
        allocate_state retains the original independently owned allocation API.
        """
        context = context or self.config.max_position_embeddings
        assert 1 <= batch <= 32 and 0 < context <= self.config.max_position_embeddings and context % 32 == 0
        if batch == 1:
            return [self.allocate_state(context=context)]
        if self.kind == "linear_attention":
            conv = ttnn.zeros(
                (batch, 1, 3, 2048),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            recurrent = ttnn.zeros(
                (batch, 8, 128, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return [DecoderState(conv=conv, recurrent=recurrent, state_row=i) for i in range(batch)]
        pages = context // 32
        key = ttnn.zeros(
            (batch * pages, 1, 32, 256),
            dtype=self.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value = ttnn.zeros_like(key)
        import torch

        # Immutable setup-owned offsets travel with the pool, not a decoder-global
        # cache. Captured traces retain their state's backing and metadata tensors.
        offsets = ttnn.from_torch(
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1) * pages,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return [
            DecoderState(key=key, value=value, page_offset=i * pages, page_count=pages, page_offsets=offsets)
            for i in range(batch)
        ]

    def write_linear_state(self, state, name, value):
        """Write only this request's rows, preserving every pooled backing allocation."""
        destination = getattr(state, name)
        row = getattr(state, "state_row", None)
        if row is None:
            ttnn.copy(value, destination)
            return
        if name == "recurrent":
            value = ttnn.to_memory_config(value, self.recurrent_write_mem)
        ttnn.experimental.slice_write(
            value, destination, (row, 0, 0, 0), (row + 1, *tuple(destination.shape)[1:]), (1, 1, 1, 1)
        )

    def linear(self, x, name):
        if (
            self.fused_prefill
            and x.shape[2] > 1
            and name in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")
        ):
            n = x.shape[2]
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if n % 32:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            return ttnn.experimental.matmul_reduce_scatter_prefill(
                x,
                self.weights[name + ".weight"],
                cluster_axis=1,
                scatter_dim=-1,
                topology=self.ccl.topology,
                num_links=1,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.mlp_fidelity if name.startswith("mlp.") else self.attention_fidelity
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                l1_headroom=self.prefill_headroom,
            )[:, :, :n, :]

        if x.shape[2] == 1:
            return self.decode_linear(x, name)
        return ttnn.experimental.minimal_matmul(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
            self.weights[name + ".weight"],
            config=self.prefill_program,
            compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_linear(self, x, name):
        """DRAM-sharded projection for 1..32 logical rows in one physical tile."""
        im, om, cfg = self.dram_layouts[name]
        out = ttnn.linear(
            ttnn.to_memory_config(x, im),
            self.dram_weights[name],
            program_config=cfg,
            compute_kernel_config=self.matmul_compute(name),
            memory_config=om,
        )
        # Attention's head/conv consumers require interleaved packed columns.
        return (
            ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            if name == "packed_input" or not self.native_local
            else out
        )

    def norm(self, x, name):
        if self.native_local and x.shape[2] == 1 and name in ("input_layernorm", "post_attention_layernorm"):
            return self.decode_norm(x, name)
        if name in ("input_layernorm", "post_attention_layernorm") and self.residual_layout == "sharded":
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG)
            stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
            stats = self.gather(stats)
            result = ttnn.rms_norm_post_all_gather(
                x,
                stats,
                epsilon=self.config.rms_norm_eps,
                weight=self.weights[name + ".weight"],
                compute_kernel_config=self.compute,
                memory_config=x.memory_config(),
            )
            return self.gather(result)
        out = ttnn.rms_norm(
            x,
            weight=self.weights[name + ".weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
        )
        if x.shape[2] > 1 and name in ("input_layernorm", "post_attention_layernorm"):
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        return out

    def decode_norm(self, x, name):
        return ttnn.rms_norm(
            ttnn.to_memory_config(x, self.ar_mem),
            weight=self.weights[name + ".weight"],
            epsilon=self.config.rms_norm_eps,
            program_config=self.norm_program,
            compute_kernel_config=self.compute,
            memory_config=self.ar_mem,
        )

    def finish(self, x, mixed):
        if self.native_local and x.shape[2] == 1:
            x = ttnn.to_memory_config(x, self.ar_mem)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        if x.shape[2] == 1:
            gate = self.linear(h, "mlp.gate_proj")
            up = self.linear(h, "mlp.up_proj")
            h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        else:
            h = ttnn.experimental.minimal_matmul(
                h,
                self.weights["packed_mlp.weight"],
                fuse_swiglu=True,
                config=self.prefill_program,
                compute_kernel_config=self.mlp_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))

    def decode_finish(self, x, mixed):
        residual = ttnn.add(x, mixed)
        h = self.decode_norm(residual, "post_attention_layernorm")
        gate = self.decode_linear(h, "mlp.gate_proj")
        up = self.decode_linear(h, "mlp.up_proj")
        h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual, self.decode_reduce(self.decode_linear(h, "mlp.down_proj")))

    def linear_attention(self, x, state, *, decode, packed=None, project=True):
        row = getattr(state, "state_row", None)
        if row is not None:
            local = DecoderState(conv=state.conv[row : row + 1], recurrent=state.recurrent[row : row + 1])
            out = self.linear_attention(x, local, decode=decode, packed=packed, project=project)
            self.write_linear_state(state, "conv", local.conv)
            self.write_linear_state(state, "recurrent", local.recurrent)
            return out
        n = x.shape[2]
        packed = self.linear(x, "packed_input") if packed is None else packed
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (1, n, 8))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.multiply(
            ttnn.add(a, self.dt_bias),
            self.decay,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        old_history = ttnn.reshape(ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT), (1, 3, 2048))
        padded = packed
        if n % 32:
            padded = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(padded, (1, padded.shape[2], 3136)),
            old_history,
            *self.taps,
            512,
            512,
            1024,
            actual_start=self.conv_start,
            predecessor_carry=old_history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
            compute_kernel_config=self.conv_compute,
        )
        # Keep the three logical history rows in row-major storage. Reuse the
        # convolution view instead of a concat that tilizes and immediately untilizes.
        if n >= 3:
            history = ttnn.to_layout(packed[:, :, n - 3 : n, :2048], ttnn.ROW_MAJOR_LAYOUT)
        else:
            history = ttnn.concat(
                [
                    ttnn.reshape(old_history, (1, 1, 3, 2048)),
                    ttnn.untilize_with_unpadding(packed, (0, 0, n - 1, 2047)),
                ],
                dim=2,
            )
            ttnn.slice(history, (0, 0, n, 0), (1, 1, n + 3, 2048), (1, 1, 1, 1), output_tensor=state.conv)
        if n >= 3:
            ttnn.copy(history, state.conv)

        if n % 32:
            # Inactive tail rows leave recurrence unchanged: exp(g)=1 and beta=0.
            g = ttnn.pad(g, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        if decode:
            # Each receiver reads its complete, disjoint head/V state slab before
            # publishing the final slab. The initial/final views share backing storage;
            # scratch is consumed within this forward and never escapes a live trace.
            recurrent = ttnn.reshape(state.recurrent, (8, 128, 128))
            scratch = ttnn.empty(
                (8, 1, 32, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out, _ = ttnn._ttnn.operations.transformer.chunk_gdn_fused(
                q,
                k,
                v,
                g,
                beta,
                self.gdn_constants["eye"],
                self.gdn_constants["tril"],
                self.gdn_constants["ones"],
                self.gdn_constants["masks"],
                recurrent,
                chunk_size=32,
                output_final_state=True,
                compute_kernel_config=self.compute,
                program_config=ttnn.ChunkGdnFusedProgramConfig(num_receivers=1),
                v_flat=True,
                HV=8,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=4,
                o_out=scratch,
                final_state_out=recurrent,
            )
            out = ttnn.reshape(out, (8, 32, 128))
        else:
            out, new_state = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state.recurrent,
                output_final_state=True,
                compute_kernel_config=self.compute,
                chunk_size=32,
                num_key_heads=4,
                output_head_major=True,
                **self.gdn_constants,
            )
            ttnn.copy(new_state, state.recurrent)

        out = ttnn.typecast(out, ttnn.bfloat16)
        out = self.norm(out, "linear_attn.norm")
        out = ttnn.transformer.concatenate_heads(ttnn.reshape(out, (1, 8, padded.shape[2], 128)))
        out = ttnn.reshape(out, (1, 1, padded.shape[2], 1024))[:, :, :n, :]
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return self.reduce(self.linear(out, "linear_attn.out_proj")) if project else out

    def rope(self, x, cos, sin):
        rot = x[:, :, :, :64]
        rotated = ttnn.experimental.rotary_embedding(
            rot,
            cos,
            sin,
            token_index=0 if cos.shape[2] == 1 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        return ttnn.concat([rotated[:, :, : x.shape[2], :], x[:, :, :, 64:]], dim=-1)

    def full_attention(self, x, state, *, page_table, position, decode, packed=None, project=True):
        # SDPA reads complete key chunks, including masked future pages. Own
        # safe local page-zero tails separately from the cache-update table.
        attention_table = page_table
        if page_table.shape[1] % 8:
            attention_table = ttnn.pad(page_table, [(0, 0), (0, 8 - page_table.shape[1] % 8)], 0)
        if state.page_offset:
            page_table = ttnn.add(page_table, state.page_offset)
            attention_table = ttnn.add(attention_table, state.page_offset)
        n = x.shape[2]
        packed = self.linear(x, "packed_input") if packed is None else packed
        if decode:
            q = self.norm(ttnn.reshape(packed[:, :, :, :1024], (1, n, 4, 256)), "self_attn.q_norm")
            gate = packed[:, :, :, 1024:2048]
            k = self.norm(ttnn.reshape(packed[:, :, :, 2048:2304], (1, n, 1, 256)), "self_attn.k_norm")
            v = ttnn.reshape(packed[:, :, :, 2304:2560], (1, n, 1, 256))
        else:
            qg, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn.reshape(packed, (1, n, 2560)),
                num_heads=8,
                num_kv_heads=1,
                transpose_key=False,
            )
            q = self.norm(qg[:, :4, :, :], "self_attn.q_norm")
            k = self.norm(k, "self_attn.k_norm")
            gate = ttnn.reshape(ttnn.permute(qg[:, 4:8, :, :], (0, 2, 1, 3)), (1, 1, n, 1024))
        if decode:
            # A full index tile lets embedding emit tiled output directly; the
            # extra indices are inactive lookup padding, never additional users.
            rope_position = ttnn.pad(ttnn.typecast(position, ttnn.uint32), [(0, 31)], 0)
            rope_values = ttnn.reshape(
                ttnn.embedding(rope_position, self.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, 32, 128)
            )[:, :, :1, :]
            cos, sin = rope_values[:, :, :, :64], rope_values[:, :, :, 64:]
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            q = ttnn.to_memory_config(q, self.shard)
            k = ttnn.to_memory_config(k, self.shard)
            v = ttnn.to_memory_config(v, self.value_shard)
            ttnn.experimental.paged_fused_update_cache(
                state.key, k, state.value, v, update_idxs_tensor=position, page_table=page_table
            )
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                state.key,
                state.value,
                attention_table,
                cur_pos_tensor=position,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(out, self.shard), num_heads=4)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)[:, :, :1, :]
        else:
            cos = ttnn.reshape(self.cos[position : position + n, :], (1, 1, n, 64))
            sin = ttnn.reshape(self.sin[position : position + n, :], (1, 1, n, 64))
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            local_table = page_table[:, position // 32 : (position + n) // 32]
            ttnn.experimental.paged_fill_cache(state.key, ttnn.typecast(k, self.kv_dtype), local_table)
            ttnn.experimental.paged_fill_cache(state.value, ttnn.typecast(v, self.kv_dtype), local_table)
            out = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                state.key,
                state.value,
                attention_table,
                chunk_start_idx_tensor=self.positions[position : position + 1],
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, n, 1024))
        out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self.reduce(self.linear(out, "self_attn.o_proj")) if project else out

    def decode_forward(self, x, *, state, page_table=None, current_position=None):
        """One token per state; current_position is a persistent int32 device tensor.

        The caller owns position advancement and state restoration at request boundaries.
        This function is the entire trace-captured layer pass.
        """
        assert x.shape[2] == 1 and x.shape[-1] == self.hidden_width
        x = ttnn.to_memory_config(x, self.ar_mem if self.native_local else ttnn.L1_MEMORY_CONFIG)
        h = self.norm(x, "input_layernorm")
        mixed = (
            self.linear_attention(h, state, decode=True)
            if self.kind == "linear_attention"
            else self.full_attention(h, state, page_table=page_table, position=current_position, decode=True)
        )
        return self.finish(x, mixed)

    def prefill_forward(self, x, *, state, start_position=0, page_table=None):
        """Consume a logical [1,1,S,4096] segment and update its request state.

        For continuation the caller supplies state from exactly start_position.
        Padded rows are never part of the logical recurrent sequence.
        """
        n = x.shape[2]
        assert x.shape[-1] == self.hidden_width
        assert 0 < n and 0 <= start_position and start_position + n <= self.config.max_position_embeddings
        outputs = []
        offset = 0
        while offset < n:
            absolute = start_position + offset
            if self.kind == "full_attention" and absolute % 32:
                chunk = x[:, :, offset : offset + 1, :]
                outputs.append(
                    ttnn.to_memory_config(
                        self.decode_forward(
                            chunk,
                            state=state,
                            page_table=page_table,
                            current_position=self.positions[absolute : absolute + 1],
                        ),
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                offset += 1
                continue
            count = min(n - offset, self.chunk_size)
            chunk = x[:, :, offset : offset + count, :]
            if self.kind == "full_attention" and count % 32:
                chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, 32 - count % 32), (0, 0)], 0.0)
            h = self.norm(chunk, "input_layernorm")
            mixed = (
                self.linear_attention(h, state, decode=False)
                if self.kind == "linear_attention"
                else self.full_attention(h, state, page_table=page_table, position=absolute, decode=False)
            )
            outputs.append(ttnn.to_memory_config(self.finish(chunk, mixed), ttnn.DRAM_MEMORY_CONFIG)[:, :, :count, :])
            offset += count
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def prefill_batch(self, rows, *, states, start_positions, page_tables):
        """Ragged batch: independent device rows, starts and page maps."""
        assert len(rows) == len(states) == len(start_positions) == len(page_tables)
        return [
            self.prefill_forward(x, state=st, start_position=start, page_table=pt)
            for x, st, start, pt in zip(rows, states, start_positions, page_tables)
        ]

    def pooled_linear_attention_decode_batch(self, packed, states):
        """Neutral tail tokens and disjoint head waves update pooled state in place."""
        n = len(states)
        raw = packed[:, :, :, :2048]
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.reshape(
            ttnn.multiply(
                ttnn.add(a, self.dt_bias),
                self.decay,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            ),
            (n, 1, 8),
        )
        history = states[0].conv
        conv = ttnn.multiply(ttnn.typecast(raw, ttnn.float32), self.float_taps[3])
        for j in range(3):
            old = ttnn.to_layout(ttnn.reshape(history[:, :, j : j + 1, :], (1, 1, n, 2048)), ttnn.TILE_LAYOUT)
            conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(old, ttnn.float32), self.float_taps[j]))
        conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
        new_history = ttnn.concat(
            [history[:, :, 1:, :], ttnn.reshape(ttnn.to_layout(raw, ttnn.ROW_MAJOR_LAYOUT), (n, 1, 1, 2048))], dim=2
        )
        ttnn.copy(new_history, states[0].conv)

        def padded_flat(value, width):
            value = ttnn.reshape(value, (n, 1, width))
            return ttnn.pad(value, [(0, 0), (0, 31), (0, 0)], 0.0)

        q = padded_flat(conv[:, :, :, :512], 512)
        k = padded_flat(conv[:, :, :, 512:1024], 512)
        v = padded_flat(conv[:, :, :, 1024:], 1024)
        g = ttnn.pad(g, [(0, 0), (0, 31), (0, 0)], 0.0)
        beta = ttnn.pad(beta, [(0, 0), (0, 31), (0, 0)], 0.0)
        recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
        scratch = ttnn.empty(
            (n * 8, 1, 32, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wave_heads = self.decode_wave_heads
        for head_offset in range(0, n * 8, wave_heads):
            out, _ = ttnn._ttnn.operations.transformer.chunk_gdn_fused(
                q,
                k,
                v,
                g,
                beta,
                self.gdn_constants["eye"],
                self.gdn_constants["tril"],
                self.gdn_constants["ones"],
                self.gdn_constants["masks"],
                recurrent,
                chunk_size=32,
                output_final_state=True,
                compute_kernel_config=self.compute,
                program_config=ttnn.ChunkGdnFusedProgramConfig(
                    num_receivers=1,
                    **({"num_producers": self.decode_wave_producers} if self.decode_wave_producers else {}),
                ),
                num_heads=min(wave_heads, n * 8 - head_offset),
                head_offset=head_offset,
                v_flat=True,
                HV=8,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=4,
                o_out=scratch,
                final_state_out=recurrent,
            )
        out = ttnn.permute(ttnn.reshape(out, (n, 8, 32, 128))[:, :, :1, :], (0, 2, 1, 3))
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out

    def linear_attention_decode_batch(self, packed, states):
        """Independent recurrent users, vectorized in core-fitting groups of twelve."""
        n = len(states)
        raw = packed[:, :, :, :2048]
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.reshape(
            ttnn.multiply(
                ttnn.add(a, self.dt_bias),
                self.decay,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            ),
            (n, 1, 8),
        )
        history = ttnn.concat([st.conv for st in states], dim=0) if n > 1 else states[0].conv
        conv = ttnn.multiply(ttnn.typecast(raw, ttnn.float32), self.float_taps[3])
        for j in range(3):
            old = ttnn.to_layout(ttnn.reshape(history[:, :, j : j + 1, :], (1, 1, n, 2048)), ttnn.TILE_LAYOUT)
            conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(old, ttnn.float32), self.float_taps[j]))
        conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
        new_history = ttnn.concat(
            [history[:, :, 1:, :], ttnn.reshape(ttnn.to_layout(raw, ttnn.ROW_MAJOR_LAYOUT), (n, 1, 1, 2048))], dim=2
        )
        for i, st in enumerate(states):
            ttnn.copy(new_history[i : i + 1, :, :, :], st.conv)
        q = ttnn.typecast(ttnn.reshape(conv[:, :, :, :512], (n, 1, 4, 128)), ttnn.float32)
        k = ttnn.typecast(ttnn.reshape(conv[:, :, :, 512:1024], (n, 1, 4, 128)), ttnn.float32)
        v = ttnn.reshape(conv[:, :, :, 1024:], (n, 1, 8, 128))
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6 / 128, compute_kernel_config=self.compute), 128**-0.5)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6 / 128, compute_kernel_config=self.compute), 128**-0.5)
        recurrent = ttnn.concat([st.recurrent for st in states], dim=0) if n > 1 else states[0].recurrent
        parts, final_parts = [], []
        # One core per value head: twelve requests use96 of110 available cores.
        for start in range(0, n, 12):
            end = min(start + 12, n)
            part, last = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q[start:end],
                k[start:end],
                v[start:end],
                g[start:end],
                beta[start:end],
                initial_state=recurrent[start:end],
                output_final_state=True,
                compute_kernel_config=self.compute,
            )
            parts.append(part)
            final_parts.append(last)
        out = ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]
        final = ttnn.concat(final_parts, dim=0) if len(final_parts) > 1 else final_parts[0]
        for i, st in enumerate(states):
            ttnn.copy(final[i : i + 1, :, :, :], st.recurrent)
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out

    def full_attention_decode_batch(self, packed, states, current_positions, page_tables):
        """Batch SDPA with private cache-pool packing and explicit scatter-back."""
        n = len(states)
        q = self.norm(ttnn.reshape(packed[:, :, :, :1024], (1, n, 4, 256)), "self_attn.q_norm")
        gate = packed[:, :, :, 1024:2048]
        k = self.norm(ttnn.reshape(packed[:, :, :, 2048:2304], (1, n, 1, 256)), "self_attn.k_norm")
        v = ttnn.reshape(packed[:, :, :, 2304:2560], (1, n, 1, 256))
        positions = ttnn.concat(current_positions, dim=0)
        rp = ttnn.typecast(positions, ttnn.uint32)
        if n < 32:
            rp = ttnn.pad(rp, [(0, 32 - n)], 0)
        values = ttnn.reshape(ttnn.embedding(rp, self.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, 32, 128))[
            :, :, :n, :
        ]
        cos, sin = values[:, :, :, :64], values[:, :, :, 64:]
        q = ttnn.permute(self.rope(ttnn.permute(q, (0, 2, 1, 3)), cos, sin), (0, 2, 1, 3))
        k = ttnn.permute(self.rope(ttnn.permute(k, (0, 2, 1, 3)), cos, sin), (0, 2, 1, 3))

        def shard(y_offset):
            ranges = set()
            rows, tail = divmod(n, 8)
            if rows:
                ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, y_offset), ttnn.CoreCoord(7, y_offset + rows - 1)))
            if tail:
                ranges.add(
                    ttnn.CoreRange(ttnn.CoreCoord(0, y_offset + rows), ttnn.CoreCoord(tail - 1, y_offset + rows))
                )
            # SDPA enumerates ranges before row-major cores within each range.
            # Singleton ranges sort by x first and permute independent users.
            cores = ttnn.CoreRangeSet(ranges)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, (32, 256), ttnn.ShardOrientation.ROW_MAJOR),
            )

        q = ttnn.to_memory_config(q, shard(0))
        k = ttnn.to_memory_config(k, shard(0))
        v = ttnn.to_memory_config(v, shard(4))
        ordered_pool = (
            states[0].page_offsets is not None
            and all(
                st.key is states[0].key and st.value is states[0].value and st.page_offset == i * st.page_count
                for i, st in enumerate(states)
            )
            and n * states[0].page_count == states[0].key.shape[0]
            and len({table.shape[-1] for table in page_tables}) == 1
        )
        if ordered_pool:
            key, value = states[0].key, states[0].value
            pools = [(key, value)]
            local_table = ttnn.concat(page_tables, dim=0)
            attention_table = local_table
            width = local_table.shape[-1]
            if width % 8:
                attention_table = ttnn.pad(local_table, [(0, 0), (0, 8 - width % 8)], 0)
            table = ttnn.add(local_table, states[0].page_offsets)
            attention_table = (
                table if attention_table is local_table else ttnn.add(attention_table, states[0].page_offsets)
            )
        else:
            pools, pool_offsets, total_pages = [], {}, 0
            for state in states:
                identity = (id(state.key), id(state.value))
                if identity not in pool_offsets:
                    pools.append((state.key, state.value))
                    pool_offsets[identity] = total_pages
                    total_pages += state.key.shape[0]
            key = pools[0][0] if len(pools) == 1 else ttnn.concat([pair[0] for pair in pools], dim=0)
            value = pools[0][1] if len(pools) == 1 else ttnn.concat([pair[1] for pair in pools], dim=0)
            tables = []
            max_pages = max(table.shape[-1] for table in page_tables)
            for state, table in zip(states, page_tables):
                # Different allocated contexts are valid. Padded local page zero is
                # inactive for valid positions within each request's own context.
                if table.shape[-1] < max_pages:
                    table = ttnn.pad(table, [(0, 0), (0, max_pages - table.shape[-1])], 0)
                offset = pool_offsets[(id(state.key), id(state.value))] + getattr(state, "page_offset", 0)
                tables.append(ttnn.add(table, offset) if offset else table)
            table = ttnn.concat(tables, dim=0)
            attention_table = table
            if max_pages % 8:
                attention_pages = ((max_pages + 7) // 8) * 8
                attention_rows = []
                for state, local_table in zip(states, page_tables):
                    padded = ttnn.pad(local_table, [(0, 0), (0, attention_pages - local_table.shape[-1])], 0)
                    offset = pool_offsets[(id(state.key), id(state.value))] + state.page_offset
                    attention_rows.append(ttnn.add(padded, offset) if offset else padded)
                attention_table = ttnn.concat(attention_rows, dim=0)
        ttnn.experimental.paged_fused_update_cache(key, k, value, v, update_idxs_tensor=positions, page_table=table)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            key,
            value,
            attention_table,
            cur_pos_tensor=positions,
            scale=1 / 16,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
            ),
            compute_kernel_config=self.compute,
        )
        out = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(out, shard(0)),
            num_heads=4,
            **({"sub_core_grids": shard(0).shard_spec.grid} if n >= 4 else {}),
        )
        out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)[:, :, :n, :]
        if len(pools) > 1:
            offset = 0
            for old_key, old_value in pools:
                end = offset + old_key.shape[0]
                ttnn.slice(key, (offset, 0, 0, 0), (end, 1, 32, 256), (1, 1, 1, 1), output_tensor=old_key)
                ttnn.slice(value, (offset, 0, 0, 0), (end, 1, 32, 256), (1, 1, 1, 1), output_tensor=old_value)
                offset = end
        out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return out

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        """TP4 BF16 [1,1,B,4096], 1<=B<=32, local L1 residual across layers.

        Request attention/cache state stays independent. Dense projections, norms,
        residual adds and reductions share the physical tile. The caller packs at
        stack entry and unpacks only at its exit, outside decoder layer boundaries.
        """
        n = x.shape[2]
        assert self.native_local and 1 <= n <= 32
        assert n == len(states) == len(current_positions) == len(page_tables)
        regions = [
            (
                (id(st.conv), id(st.recurrent), st.state_row)
                if self.kind == "linear_attention"
                else (id(st.key), id(st.value), st.page_offset)
            )
            for st in states
        ]
        assert len(set(regions)) == n, "A decode batch must contain distinct request state regions"
        x = ttnn.to_memory_config(x, self.ar_mem)
        packed = self.decode_linear(self.decode_norm(x, "input_layernorm"), "packed_input")
        pooled_linear = (
            self.kind == "linear_attention"
            and all(
                st.conv is states[0].conv
                and st.recurrent is states[0].recurrent
                and getattr(st, "state_row", None) == i
                for i, st in enumerate(states)
            )
            and states[0].conv.shape[0] == n
        )
        independent_linear = all(getattr(st, "state_row", None) is None for st in states)
        if n > 1 and (self.kind == "full_attention" or pooled_linear or independent_linear):
            if self.kind == "full_attention":
                local = self.full_attention_decode_batch(packed, states, current_positions, page_tables)
            elif pooled_linear:
                local = self.pooled_linear_attention_decode_batch(packed, states)
            else:
                local = self.linear_attention_decode_batch(packed, states)
        else:
            local = []
            for i, (state, position, table) in enumerate(zip(states, current_positions, page_tables)):
                row = packed[:, :, i : i + 1, :]
                local.append(
                    self.linear_attention(row, state, decode=True, packed=row, project=False)
                    if self.kind == "linear_attention"
                    else self.full_attention(
                        row, state, page_table=table, position=position, decode=True, packed=row, project=False
                    )
                )
            local = ttnn.concat(local, dim=2) if n > 1 else local[0]
        name = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
        return self.decode_finish(x, self.decode_reduce(self.decode_linear(local, name)))

    def decode_batch(self, rows, *, states, current_positions, page_tables):
        """List compatibility wrapper; stacked decoders use decode_batch_packed."""
        assert len(rows) == len(states) == len(current_positions) == len(page_tables)
        if self.native_local and 1 <= len(rows) <= 32:
            packed = ttnn.concat([ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) for x in rows], dim=2)
            out = self.decode_batch_packed(
                packed, states=states, current_positions=current_positions, page_tables=page_tables
            )
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            return [out[:, :, i : i + 1, :] for i in range(len(rows))]
        return [
            self.decode_forward(x, state=st, current_position=pos, page_table=pt)
            for x, st, pos, pt in zip(rows, states, current_positions, page_tables)
        ]
