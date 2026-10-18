# SPDX-License-Identifier: Apache-2.0
"""Four-Blackhole tensor-parallel decoder. See doc/multichip_decoder.

The stack residual is [1,B,S,2560] locally: four 640-wide hidden shards.
Setup transforms checkpoint tensors; runtime controls and collectives stay on device.
The numerical and performance control is optimized_decoder.OptimizedDecoder.
Configure FABRIC_1D before opening the fixed 1x4 mesh.
"""

import math
from dataclasses import dataclass

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .fused_decoder import FusedQSA, _PrefixView
from .optimized_decoder import OptimizedDecoder, OptimizedGatedDeltaNet, OptimizedQSA


@dataclass(frozen=True)
class MeshPolicy:
    tp: int = 4
    links: int = 2
    residual: str = "hidden_sharded"

    def validate(self, mesh):
        if mesh.get_num_devices() != 4 or tuple(mesh.shape) != (1, 4):
            raise ValueError("This decoder requires the qualified 1x4 Blackhole mesh")
        if self.residual != "hidden_sharded":
            raise ValueError("Unqualified residual policy")


class MeshOps:
    def __init__(self, mesh, policy=MeshPolicy()):
        policy.validate(mesh)
        self.mesh, self.policy = mesh, policy
        self.precision_ccl = "mixed"
        self.ccl = TT_CCL(mesh)
        self.gather_outputs = {}
        self.reduce_outputs = {}
        self.large_prefill = False
        self.l1_prefill = False
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def linear(self, x, weight, *, compute_kernel_config, dtype=None, max_cores=10):
        """Measured decode and prefill geometry; logical tails remain legal."""
        shape = tuple(x.shape)
        rows = math.prod(shape[:-2])
        if (
            shape[-2] == 1
            and rows > 1
            and math.prod(tuple(weight.shape)[:-2]) == 1
            and x.dtype in (ttnn.bfloat16, ttnn.float32)
            and dtype in (None, ttnn.bfloat16, ttnn.float32)
        ):
            # Independent singleton rows otherwise each consume a padded tile.
            # This also covers singleton prefill without changing its caller's
            # selected weight, fidelity, output dtype or memory policy.
            flat = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, 1, rows, shape[-1])), ttnn.TILE_LAYOUT
            )
            out = self.linear(
                flat, weight, compute_kernel_config=compute_kernel_config, dtype=dtype, max_cores=max_cores
            )
            return ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), (*shape[:-1], weight.shape[-1])),
                ttnn.TILE_LAYOUT,
            )
        prefill_memory = ttnn.L1_MEMORY_CONFIG if self.l1_prefill else ttnn.DRAM_MEMORY_CONFIG
        if self.l1_prefill and x.shape[-2] > 1 and math.prod(tuple(x.shape)[:-2]) == 1:
            x = ttnn.to_memory_config(x, prefill_memory)
        if (
            x.shape[-2] >= 256 or (self.large_prefill and x.shape[-2] > 1) or (self.l1_prefill and x.shape[-2] >= 128)
        ) and math.prod(tuple(x.shape)[:-2]) == 1:
            mt = math.ceil(x.shape[-2] / 32)
            kt, nt = math.ceil(weight.shape[-2] / 32), math.ceil(weight.shape[-1] / 32)
            gx, gy = min(10, nt), min(8, mt)
            pn, pm = math.ceil(nt / gx), math.ceil(mt / gy)
            block = min(20, kt)
            while kt % block:
                block -= 1
            subblock = min(4, pn)
            while pn % subblock:
                subblock -= 1
            program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=subblock,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fuse_batch=False,
            )
            return ttnn.linear(
                x,
                weight,
                program_config=program,
                compute_kernel_config=compute_kernel_config,
                dtype=dtype,
                memory_config=prefill_memory,
            )
        if x.shape[-2] != 1 or math.prod(tuple(x.shape)[:-2]) != 1:
            return ttnn.linear(
                x, weight, compute_kernel_config=compute_kernel_config, dtype=dtype, memory_config=prefill_memory
            )
        k, n = tuple(weight.shape)[-2:]
        cores = min(max_cores, math.ceil(n / 32))
        per_n = math.ceil(n / 32 / cores)
        subblock = min(4, per_n)
        while per_n % subblock:
            subblock -= 1
        block = min(20, k // 32)
        while (k // 32) % block:
            block -= 1
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(min(10, cores), math.ceil(cores / 10)),
            in0_block_w=block,
            out_subblock_h=1,
            out_subblock_w=subblock,
            per_core_M=1,
            per_core_N=per_n,
            fuse_batch=True,
            mcast_in0=True,
        )
        return ttnn.linear(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
            weight,
            program_config=program,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=dtype,
        )

    def upload(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def shards(self, values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        import torch  # Setup only.

        return ttnn.from_torch(
            torch.cat(values, dim=0).contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def gather(self, value, dim=-1):
        original_dtype = value.dtype
        transfer = self.precision_ccl == "bfloat8_b" and value.dtype == ttnn.bfloat16 and value.shape[-1] >= 128
        if transfer:
            value = ttnn.typecast(value, ttnn.bfloat8_b)
        dim %= len(value.shape)
        shape = tuple(value.shape)
        # Only B1 decode payloads persist. Arbitrary prefill shapes must not
        # accumulate a permanent buffer for every logical sequence length.
        persistent = None
        if (
            len(shape) == 4
            and math.prod(shape[:-2]) == 1
            and (shape[-2] == 1 or (shape[-2] == 4 and (value.dtype == ttnn.float32 or shape[-1] == 32)))
        ):
            key = (shape, dim, value.dtype)
            if key not in self.gather_outputs:
                outshape = list(shape)
                outshape[dim] *= 4
                self.gather_outputs[key] = ttnn.empty(
                    outshape, dtype=value.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh
                )
            persistent = self.gather_outputs[key]
        result = ttnn.experimental.all_gather_async(
            value,
            dim=dim,
            mesh_device=self.mesh,
            cluster_axis=1,
            persistent_output_tensor=persistent,
            num_links=self.policy.links,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return ttnn.typecast(result, original_dtype) if transfer else result

    def reduce(self, value):
        # Preserve individual scalar/statistic rows; gathering the non-spatial
        # leading dimension needs no replication of logical tile padding.
        return ttnn.sum(self.gather(value, 0), dim=0, keepdim=True)

    def scatter_reduce(self, value):
        original_dtype = value.dtype
        transfer = self.precision_ccl == "bfloat8_b" and value.dtype == ttnn.bfloat16 and value.shape[-1] >= 128
        if transfer:
            value = ttnn.typecast(value, ttnn.bfloat8_b)
        shape = tuple(value.shape)
        flat = ttnn.reshape(value, (1, 1, math.prod(shape[:-1]), shape[-1]))
        persistent = None
        if math.prod(shape[:-1]) == 1:
            key = (tuple(flat.shape), flat.dtype)
            if key not in self.reduce_outputs:
                self.reduce_outputs[key] = [
                    ttnn.empty((2, 1, 1, shape[-1]), dtype=flat.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh),
                    ttnn.empty((1, 1, 1, shape[-1] // 4), dtype=flat.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh),
                ]
            persistent = self.reduce_outputs[key]
        out = ttnn.experimental.reduce_scatter_minimal_async(
            flat,
            dim=3,
            persistent_output_buffers=persistent,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.links,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if transfer:
            out = ttnn.typecast(out, original_dtype)
        return ttnn.reshape(out, (*shape[:-1], shape[-1] // 4))

    def norm(self, value, weight, eps, *, reuse_output=False):
        if tuple(value.shape) == (1, 1, 4, 640):
            value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)
            if value.dtype == ttnn.bfloat16:
                # HC singleton working shard. The sharded TTNN norm accumulates
                # in FP32 but communicates BF16 statistics; qualified separately
                # from PLE's FP32 input/statistics contract.
                mem = ttnn.create_sharded_memory_config(
                    (32, 640),
                    ttnn.CoreGrid(x=1, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(1, 1),
                    subblock_w=4,
                    block_h=1,
                    block_w=20,
                    inplace=False,
                )
                x = ttnn.to_memory_config(value, mem)
                stats = ttnn.rms_norm_pre_all_gather(
                    x, dtype=ttnn.bfloat16, compute_kernel_config=self.compute, program_config=program
                )
                stats = self.gather(ttnn.to_memory_config(stats, ttnn.DRAM_MEMORY_CONFIG))
                stats_mem = ttnn.create_sharded_memory_config(
                    (32, 128),
                    ttnn.CoreGrid(x=1, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                normalized = ttnn.rms_norm_post_all_gather(
                    x,
                    ttnn.to_memory_config(stats, stats_mem),
                    epsilon=eps,
                    compute_kernel_config=self.compute,
                    program_config=program,
                    memory_config=mem,
                )
                normalized = ttnn.multiply(normalized, ttnn.to_memory_config(weight, mem), memory_config=mem)
                return ttnn.to_memory_config(normalized, ttnn.L1_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(value, dtype=ttnn.float32, compute_kernel_config=self.compute)
        gathered_stats = self.gather(stats)
        if reuse_output:
            del stats
        normalized = ttnn.rms_norm_post_all_gather(
            value, gathered_stats, epsilon=eps, compute_kernel_config=self.compute
        )
        # PLE owns this fresh normalized temporary. Reuse it for gamma to
        # avoid a third 40MiB FP32 tensor at the512-token L1 stack boundary.
        # Neither the incoming residual nor the broadcast weight is aliased.
        return ttnn.multiply(normalized, weight, output_tensor=normalized if reuse_output else None)


def _hidden_parts(value):
    """Split flattened four-stream hidden columns, preserving stream order."""
    return [
        value.reshape(*value.shape[:-1], 4, 2560)[..., r * 640 : (r + 1) * 640]
        .reshape(*value.shape[:-1], 2560)
        .contiguous()
        for r in range(4)
    ]


class ParallelHyperConnection:
    def __init__(self, state, prefix, cfg, ops):
        import torch

        self.ops, self.eps = ops, cfg.rms_norm_eps
        self.norm = ops.shards(
            [x.reshape(1, 1, 4, 640) for x in _hidden_parts(1 + state[prefix + ".hc_norm.weight"].float())]
        )
        packed = torch.cat(
            (
                state[prefix + ".input_mix_weight_down.weight"].T / 4,
                torch.nn.functional.pad(state[prefix + ".block_inject_weight.weight"].T / 4, (0, 28)),
            ),
            -1,
        )
        self.read_weight = ops.shards([x.T for x in _hidden_parts(packed.T)], ttnn.bfloat8_b)
        self.up = ops.shards(_hidden_parts(state[prefix + ".input_mix_weight_up.weight"].T), ttnn.bfloat8_b)
        self.compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def read(self, x, *, decode=False):
        if tuple(x.shape) == (1, 1, 1, 2560) or self.ops.l1_prefill:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        b, s = tuple(x.shape)[1:3]
        streams = self.ops.norm(ttnn.reshape(x, (1, b * s, 4, 640)), self.norm, self.eps)
        normalized = ttnn.reshape(streams, (1, b, s, 2560))
        # Prefill rounds once after summing TP partials, including singleton
        # prefill. B1 decode uses the separately qualified BF16 partial path.
        partial_dtype = ttnn.bfloat16 if decode and b == 1 else ttnn.float32
        packed = ttnn.typecast(
            self.ops.reduce(
                self.ops.linear(
                    normalized, self.read_weight, compute_kernel_config=self.compute, dtype=partial_dtype, max_cores=5
                )
            ),
            ttnn.bfloat16,
        )
        mix = ttnn.sigmoid(
            self.ops.linear(ttnn.silu(packed[..., :320]), self.up, compute_kernel_config=self.compute, max_cores=5)
        )
        mix = ttnn.reshape(mix, (1, b * s, 4, 640))
        mixed = ttnn.mean(ttnn.multiply(mix, streams), dim=2, keepdim=True)
        mixed = ttnn.reshape(mixed, (1, b, s, 640))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self.ops.gather(mixed), inject

    def write(self, x, update, inject):
        if tuple(x.shape) == (1, 1, 1, 2560) or self.ops.l1_prefill:
            # Keep the local four-stream residual in L1 through the next HC
            # consumer. Mesh ownership remains four 640-wide hidden shards.
            b, s = tuple(x.shape)[1:3]
            scaled = ttnn.multiply(
                ttnn.reshape(update, (1, b * s, 1, 640)),
                ttnn.reshape(inject, (1, b * s, 4, 1)),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            result = ttnn.add(ttnn.reshape(x, (1, b * s, 4, 640)), scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
            return ttnn.reshape(result, x.shape)
        b, s = tuple(x.shape)[1:3]
        result = ttnn.add(
            ttnn.reshape(x, (1, b * s, 4, 640)),
            ttnn.multiply(ttnn.reshape(update, (1, b * s, 1, 640)), ttnn.reshape(inject, (1, b * s, 4, 1))),
        )
        return ttnn.reshape(result, x.shape)


class ParallelMoE:
    def __init__(self, state, cfg, ops):
        import torch

        self.ops = ops
        self.router = ops.upload(
            torch.cat(
                (
                    state["mlp.gate.weight"].T,
                    torch.nn.functional.pad(state["mlp.shared_expert_gate.weight"].T, (0, 31)),
                ),
                -1,
            )
        )
        self.shared_gu = ops.shards(
            [
                torch.cat(
                    (
                        state["mlp.shared_expert.gate_proj.weight"][r * 160 : (r + 1) * 160].T,
                        state["mlp.shared_expert.up_proj.weight"][r * 160 : (r + 1) * 160].T,
                    ),
                    -1,
                )
                for r in range(4)
            ]
        )
        self.shared_down = ops.shards(
            [state["mlp.shared_expert.down_proj.weight"][:, r * 160 : (r + 1) * 160].T for r in range(4)]
        )
        mask = torch.full((1, 1, 1, 32), float("-inf"))
        mask[..., :10] = 0
        self.mask = ops.upload(mask)
        self.shared_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.router_compute = self.shared_compute
        self.routed_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )
        from .expert_weight_cache import cached_experts

        cached = cached_experts(state, ops)
        if cached is None:
            gu = state["mlp.experts.gate_up_proj"]
            down = state["mlp.experts.down_proj"]
            self.gates = []
            self.ups = []
            self.downs = []
            for e in range(128):
                self.gates.append(ops.shards([gu[r * 128 + e, :640].T for r in range(4)], ttnn.bfloat4_b))
                self.ups.append(ops.shards([gu[r * 128 + e, 640:].T for r in range(4)], ttnn.bfloat4_b))
                self.downs.append(ops.shards([down[r * 128 + e].T for r in range(4)], ttnn.bfloat4_b))
        else:
            self.gates, self.ups, self.downs = cached
        self.offsets128 = ops.upload(torch.arange(128, dtype=torch.int32) * 128, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.ids = ops.upload(torch.arange(128, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.offsets = ops.upload(torch.arange(128, dtype=torch.int32) * 32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.rank = ops.shards(
            [torch.tensor([r], dtype=torch.int32) for r in range(4)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.capacity = ttnn.empty(
            (1, 1, 128 * 32, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=ops.mesh
        )

    def group_native(self, x):
        n = x.shape[2]
        router = self.ops.linear(x, self.router, compute_kernel_config=self.router_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(
            self.ops.mesh,
            [source, ids, scores, self.rank, counts, self.capacity],
            _EXPERT_DISPATCH_SELECTED,
            [n],
            workers=32,
            cb_size=5888,
        )
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            self.capacity,
            self.offsets,
            counts,
            self.ids,
            self.gates,
            self.ups,
            self.downs,
            32,
            compute_kernel_config=self.routed_compute,
        )
        compact = ttnn.empty(
            (1, 10, 32, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh, [down, ids, scores, self.rank, compact], _EXPERT_COMPACT, [n], workers=80, cb_size=8448
        )
        padded_scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, 32 - n), (0, 0)], 0) if n < 32 else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(padded_scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        shared = self.ops.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = self.ops.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )

    def group_packed(self, x):
        n = x.shape[2]
        capacity_rows = 512 if n > 256 else 256 if n > 128 else 128
        if not 1 <= n <= 512 or n > getattr(self, "prefill_group_size", 128):
            raise ValueError("Packed groups above128 require the corresponding prepared prefill view")
        offsets = getattr(self, f"offsets{capacity_rows}")
        dispatch_source = {
            128: _EXPERT_DISPATCH_PACKED,
            256: _EXPERT_DISPATCH_PACKED_256,
            512: _EXPERT_DISPATCH_PACKED_512,
        }[capacity_rows]
        compact_source = {
            128: _EXPERT_COMPACT_PACKED,
            256: _EXPERT_COMPACT_PACKED_256,
            512: _EXPERT_COMPACT_PACKED_512,
        }[capacity_rows]
        padded = math.ceil(n / 32) * 32
        router = self.ops.linear(x, self.router, compute_kernel_config=self.router_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        capacity = ttnn.empty(
            (1, 1, 128 * capacity_rows, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
        )
        counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(
            self.ops.mesh,
            [source, ids, scores, self.rank, counts, mapping, capacity],
            dispatch_source,
            [n],
            workers=32,
            cb_size=5120 + n * 192 + 64,
        )
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            capacity,
            offsets,
            counts,
            self.ids,
            self.gates,
            self.ups,
            self.downs,
            capacity_rows,
            compute_kernel_config=self.routed_compute,
        )
        compact = ttnn.empty(
            (1, 10, padded, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh,
            [down, ids, scores, self.rank, mapping, compact],
            compact_source,
            [n],
            workers=80,
            cb_size=2176 + n * 320 + 64,
        )
        scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, padded - n), (0, 0)], 0)
            if n < padded
            else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        shared = self.ops.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = self.ops.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )

    def __call__(self, x):
        b, s = tuple(x.shape)[1:3]
        if s == 1 and 1 < b <= 128:
            # Pack independent decode rows before routing so shared experts
            # and repeated routed experts execute together. Remove each lane's
            # singleton tile padding before joining its logical row.
            flat = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, 1, b, 2560)), ttnn.TILE_LAYOUT
            )
            result = self.ops.scatter_reduce(self.group_packed(flat))
            return ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT), (1, b, 1, 640)), ttnn.TILE_LAYOUT
            )
        if s == 1:
            lanes = [self.group_native(x[:, lane : lane + 1, :, :]) for lane in range(b)]
            return self.ops.scatter_reduce(lanes[0] if len(lanes) == 1 else ttnn.concat(lanes, dim=1))
        group_size = getattr(self, "prefill_group_size", 128)
        if group_size not in (128, 256, 512) or (group_size > 128 and b != 1):
            raise ValueError("Larger expert grouping is restricted to the serial B1 prefill view")
        lanes = []
        for lane in range(b):
            chunks = [
                self.group_packed(x[:, lane : lane + 1, start : min(start + group_size, s), :])
                for start in range(0, s, group_size)
            ]
            lanes.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
        return self.ops.scatter_reduce(lanes[0] if len(lanes) == 1 else ttnn.concat(lanes, dim=1))


def _expert_kernel(mesh, tensors, source, extra, workers, cb_size):
    coords = [ttnn.CoreCoord(i % 10, i // 10) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = [*extra, workers]
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=cb_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=cb_size)],
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


_EXPERT_HEADER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
"""
_EXPERT_DISPATCH_SELECTED = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    static_assert(N==1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(4));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(5));
    const uint32_t worker=get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    noc.async_read(ids,CoreLocalMem<uint32_t>(base+5120),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+5248),64,{.page_id=0},{});
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+5312),4,{.page_id=0},{});
    noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5312);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5120);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+5248);
    auto tmp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5328);
    for(uint32_t e=worker;e<128;e+=W) {
        uint32_t active=0;
        for(uint32_t route=0;route<10;++route) {
            ASSERT(id[route]<512);
            if(id[route]==rank*128+e && score[route]!=0)active=1;
        }
        if(active) {
            noc.async_read(x,CoreLocalMem<uint32_t>(base),5120,{.page_id=0},{});noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base),output,5120,{}, {.page_id=e*32});
        }
        const uint32_t lane=(e*4)&15;tmp[lane/4]=active;
        noc.async_write(CoreLocalMem<uint32_t>(base+5328+lane),counts,4,{}, {.page_id=0,.offset_bytes=e*4});
        noc.async_write_barrier();
    }
}
"""
)

_EXPERT_DISPATCH = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ca=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(1));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    noc.async_read(counts,CoreLocalMem<uint32_t>(base+5120),512,{.page_id=0},{});noc.async_read_barrier();
    auto active=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5120);
    for(uint32_t e=worker;e<128;e+=W) if(active[e]) {
        ASSERT(active[e]==N);
        for(uint32_t row=0;row<N;++row) {
            noc.async_read(x,CoreLocalMem<uint32_t>(base),5120,{.page_id=row},{});noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base),output,5120,{}, {.page_id=e*32+row});noc.async_write_barrier();
        }
    }
}
"""
)
_EXPERT_COMPACT = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0);
    constexpr auto da=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(4));
    const uint32_t col=get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+2176+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+6272+row*64),64,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+8320),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+8320);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+2176);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+6272);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+1088);
    for(uint32_t route=0;route<10;++route) {
        for(uint32_t i=0;i<1088;++i)out[i]=0;
        for(uint32_t row=0;row<N;++row) {
            const uint32_t e=id[row*32+route];ASSERT(e<512);
            if(e/128!=rank || score[row*32+route]==0)continue;
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=(e%128)*80+col},{});noc.async_read_barrier();
            for(uint32_t side=0;side<2;++side) {
                const uint32_t face=(row/16)*2+side,exp=face*16+row%16,data=64+face*256+(row%16)*16;
                out[exp]=src[exp];for(uint32_t j=0;j<16;++j)out[data+j]=src[data+j];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+1088),output,1088,{}, {.page_id=route*80+col});noc.async_write_barrier();
    }
}
"""
)

_EXPERT_DISPATCH_PACKED = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(4));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(5));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(6));
    const uint32_t worker=get_arg_val<uint32_t>(7);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t scratch=5120+N*192;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+5120+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+5120+N*128+row*64),64,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+scratch),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+scratch);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5120);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+5120+N*128);
    auto tmp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+scratch);
    for(uint32_t e=worker;e<128;e+=W) {
        uint32_t count=0;
        for(uint32_t row=0;row<N;++row)for(uint32_t route=0;route<10;++route) {
            if(id[row*32+route]!=rank*128+e || score[row*32+route]==0)continue;
            noc.async_read(x,CoreLocalMem<uint32_t>(base),5120,{.page_id=row},{});noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base),output,5120,{}, {.page_id=e*128+count});
            const uint32_t map_lane=(route*4)&15;
            tmp[map_lane/4]=count;
            noc.async_write(CoreLocalMem<uint32_t>(base+scratch+map_lane),mapping,4,{}, {.page_id=row,.offset_bytes=route*4});noc.async_write_barrier();
            ++count;
        }
        const uint32_t count_lane=(e*4)&15;
        tmp[count_lane/4]=count;noc.async_write(CoreLocalMem<uint32_t>(base+scratch+count_lane),counts,4,{}, {.page_id=0,.offset_bytes=e*4});noc.async_write_barrier();
    }
}
"""
)
_EXPERT_COMPACT_PACKED = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),T=(N+31)/32;
    constexpr auto da=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(4));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(5));
    const uint32_t col=get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t ib=2176,sb=ib+N*128,mb=sb+N*64,rb=mb+N*128;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+ib+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+sb+row*64),64,{.page_id=row},{});
        noc.async_read(mapping,CoreLocalMem<uint32_t>(base+mb+row*128),128,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+rb),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+rb);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+ib);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+sb);
    auto map=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+mb);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+1088);
    for(uint32_t route=0;route<10;++route)for(uint32_t tile=0;tile<T;++tile) {
        for(uint32_t i=0;i<1088;++i)out[i]=0;
        for(uint32_t row=tile*32;row<N && row<(tile+1)*32;++row) {
            const uint32_t e=id[row*32+route];ASSERT(e<512);
            if(e/128!=rank || score[row*32+route]==0)continue;
            const uint32_t packed=map[row*32+route];ASSERT(packed<128);
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=((e%128)*4+packed/32)*80+col},{});noc.async_read_barrier();
            for(uint32_t side=0;side<2;++side) {
                const uint32_t sf=(packed%32/16)*2+side,df=(row%32/16)*2+side;
                out[df*16+row%16]=src[sf*16+packed%16];
                for(uint32_t j=0;j<16;++j)out[64+df*256+(row%16)*16+j]=src[64+sf*256+(packed%16)*16+j];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+1088),output,1088,{}, {.page_id=(route*T+tile)*80+col});noc.async_write_barrier();
    }
}
"""
)


def _expert_capacity_source(source, old, new):
    """Derive only the three qualified capacity strides; reject source drift."""
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one qualified expert kernel fragment: {old}")
    return source.replace(old, new)


_EXPERT_DISPATCH_PACKED_256 = _expert_capacity_source(_EXPERT_DISPATCH_PACKED, "e*128+count", "e*256+count")
_EXPERT_COMPACT_PACKED_256 = _expert_capacity_source(_EXPERT_COMPACT_PACKED, "ASSERT(packed<128)", "ASSERT(packed<256)")
_EXPERT_COMPACT_PACKED_256 = _expert_capacity_source(
    _EXPERT_COMPACT_PACKED_256, "(e%128)*4+packed/32", "(e%128)*8+packed/32"
)
_EXPERT_DISPATCH_PACKED_512 = _expert_capacity_source(_EXPERT_DISPATCH_PACKED, "e*128+count", "e*512+count")
_EXPERT_COMPACT_PACKED_512 = _expert_capacity_source(_EXPERT_COMPACT_PACKED, "ASSERT(packed<128)", "ASSERT(packed<512)")
_EXPERT_COMPACT_PACKED_512 = _expert_capacity_source(
    _EXPERT_COMPACT_PACKED_512, "(e%128)*4+packed/32", "(e%128)*16+packed/32"
)


class AttentionOutputProjection:
    """Local 1536 attention channels to a 640-wide hidden shard.

    B1/S1 gathers BFP8 activations into a column-parallel projection. Other
    shapes use the row-parallel weight and reduce-scatter each logical chunk.
    Singleton output stays BFP8 through the next HC multiply. Convert singleton
    tails to BF16 only when joining the BF16 chunks of one attention invocation.
    The enclosing decoder always returns BF16 residuals.
    Two weight representations trade 4,177,920 bytes/layer/rank for the
    measured singleton latency improvement without regressing prefill.
    """

    def __init__(self, full_weight, row_weight, ops, compute, *, allocate_column_weight=True):
        self.ops, self.row_weight, self.compute = ops, row_weight, compute
        self.column_weight = (
            ops.shards([v.unsqueeze(0).unsqueeze(0) for v in full_weight.T.chunk(4, -1)], ttnn.bfloat8_b)
            if allocate_column_weight
            else None
        )
        self.program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(10, 1),
            in0_block_w=48,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            mcast_in0=True,
            allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]),
        )

    def __call__(self, x, *, concatenated=False):
        shape = tuple(x.shape)
        if shape[-2:] != (1, 1536) or math.prod(shape[:-2]) != 1:
            return self.ops.scatter_reduce(self.ops.linear(x, self.row_weight, compute_kernel_config=self.compute))
        if self.column_weight is None:
            raise ValueError("Singleton projection was not allocated for this fixed-batch serving model")
        payload = ttnn.typecast(ttnn.reshape(x, (1, 1, 1, 1536)), ttnn.bfloat8_b)
        # The native fused family hardcodes eight transfer slices and trips
        # its receiver invariant on this four-chip mesh. Keep the legal AG
        # and matmul decomposition (doc/multichip_decoder/triage_watcher).
        out = ttnn.linear(
            self.ops.gather(payload),
            self.column_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self.program,
            compute_kernel_config=self.compute,
            dtype=ttnn.bfloat8_b,
        )
        if concatenated:
            out = ttnn.typecast(out, ttnn.bfloat16)
        return ttnn.reshape(out, (*shape[:-1], 640))


class ParallelGatedDeltaNet(OptimizedGatedDeltaNet):
    """Twelve local value heads and four key heads; FP32 recurrent updates."""

    def __init__(self, state, cfg, ops, batch_size=1, max_requests=1, chunk_size=512, *, allocate_column_weight=True):
        import torch

        self.ops, self.device = ops, ops.mesh
        self.batch_size, self.max_requests, self.chunk_size = batch_size, max_requests, chunk_size
        self.recurrence_broadcast = False
        self.norm_eps = cfg.rms_norm_eps
        self.compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.projection_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        qkv = state["in_proj_qkv.weight"]
        parts = []
        for r in range(4):
            parts.append(
                torch.cat(
                    (
                        qkv[r * 512 : (r + 1) * 512],
                        qkv[2048 + r * 512 : 2048 + (r + 1) * 512],
                        qkv[4096 + r * 1536 : 4096 + (r + 1) * 1536],
                        state["in_proj_z.weight"][r * 1536 : (r + 1) * 1536],
                        torch.nn.functional.pad(state["in_proj_a.weight"][r * 12 : (r + 1) * 12], (0, 0, 0, 20)),
                        torch.nn.functional.pad(state["in_proj_b.weight"][r * 12 : (r + 1) * 12], (0, 0, 0, 20)),
                    ),
                    0,
                ).T
            )
        self.packed_weight = ops.shards(parts, ttnn.bfloat8_b)
        self.out_weight = ops.shards(
            [state["out_proj.weight"][:, r * 1536 : (r + 1) * 1536].T for r in range(4)], ttnn.bfloat8_b
        )
        self.output_projection = AttentionOutputProjection(
            state["out_proj.weight"],
            self.out_weight,
            ops,
            self.projection_compute,
            allocate_column_weight=allocate_column_weight,
        )
        conv = state["conv1d.weight"][:, 0, :]
        self.conv_taps = [
            ops.shards(
                [
                    torch.cat(
                        (
                            conv[r * 512 : (r + 1) * 512, t],
                            conv[2048 + r * 512 : 2048 + (r + 1) * 512, t],
                            conv[4096 + r * 1536 : 4096 + (r + 1) * 1536, t],
                        )
                    ).reshape(1, 1, 2560)
                    for r in range(4)
                ]
            )
            for t in range(4)
        ]
        self.conv_decode_history_enabled = batch_size == 32 and not allocate_column_weight
        self.conv_decode_predecessor = None
        self._conv_decode_phase = False
        self._decode_conv_batch_packed = False
        self._decode_conv_history_packed = False
        if self.conv_decode_history_enabled:
            self.conv_decode_predecessor = ops.upload(torch.zeros(1, 3, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.a_neg = ops.shards(
            [-state["A_log"][r * 12 : (r + 1) * 12].float().exp().reshape(1, 1, 12) for r in range(4)], ttnn.float32
        )
        self.dt_bias = ops.shards(
            [state["dt_bias"][r * 12 : (r + 1) * 12].float().reshape(1, 1, 12) for r in range(4)], ttnn.float32
        )
        self.norm_weight_flat = ops.upload(state["norm.weight"].reshape(128))
        self.qscale = ops.upload(torch.full((1, 1, 1, 128), 1 / 128), ttnn.float32)
        self.kscale = ops.upload(torch.full((1, 1, 1, 128), 128**-0.5), ttnn.float32)
        self.conv_zero = ops.upload(torch.zeros(1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.recurrent_state = ops.upload(torch.zeros(batch_size, 12, 128, 128), ttnn.float32)
        self.zero_recurrent = ops.upload(torch.zeros(batch_size, 12, 128, 128), ttnn.float32)
        self.recurrent_bank = ops.upload(torch.zeros(max_requests, 12, 128, 128), ttnn.float32)
        self.conv_state = ops.upload(torch.zeros(batch_size, 3, 2560))
        self.zero_conv = ops.upload(torch.zeros(batch_size, 3, 2560))
        self.conv_bank = ops.upload(torch.zeros(max_requests, 3, 2560))

    def _linear(self, x, weight):
        return self.ops.linear(
            x,
            weight,
            compute_kernel_config=self.projection_compute,
            max_cores=20 if weight is self.packed_weight else 10,
        )

    def prefill_forward(self, hidden_states, *, current_positions=None, slots=None, _decode=False):
        self._conv_decode_phase = _decode
        rank4 = len(hidden_states.shape) == 4
        if rank4 and hidden_states.shape[0] != 1:
            raise ValueError("Rank-4 input must have shape [1,B,S,2560]")
        x = ttnn.reshape(hidden_states, tuple(hidden_states.shape)[1:]) if rank4 else hidden_states
        # Logical invocation shape is captured into the graph; later eager
        # prefill calls do not alter the already-captured decode program.
        self.output_sequence = x.shape[-2]
        # The inherited rank-three path preserves validation, chunking and
        # slot routing without its replicated-width rank-four return reshape.
        out = super().prefill_forward(x, current_positions=current_positions, slots=slots)
        return ttnn.reshape(out, (1, *tuple(out.shape))) if rank4 else out

    def decode_forward(self, hidden_states, *, current_positions=None, slots=None):
        if hidden_states.shape[-2] != 1:
            raise ValueError("decode_forward requires exactly one token per lane")
        return self.prefill_forward(hidden_states, current_positions=current_positions, slots=slots, _decode=True)

    def _route_state(self, slots, positions, *, store):
        """Gather/scatter complete raw tiles using device request IDs.

        Unique slot IDs within each batch are a caller precondition. Position
        zero initializes the selected lane on gather; scatter updates only the
        selected slot. Banks and work buffers are persistent across replay.
        """
        for bank, work, tiles, tile_bytes in (
            (self.recurrent_bank, self.recurrent_state, 12 * 4 * 4, 4096),
            (self.conv_bank, self.conv_state, 80, 2048),
        ):
            grid = self.device.compute_with_storage_grid_size()
            coords = [
                ttnn.CoreCoord(i % grid.x, i // grid.x)
                for i in range(min(grid.x * grid.y, self.batch_size * self.state_workers))
            ]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            tensors = [bank, work, slots, positions]
            workers = max(1, len(coords) // self.batch_size)
            coords = coords[: workers * self.batch_size]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            args = [tiles, tile_bytes, self.max_requests, int(store), workers]
            for tensor in tensors:
                args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            cb = ttnn.CBDescriptor(
                total_size=tile_bytes + 256,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=tile_bytes + 256)
                ],
            )
            kernel = ttnn.KernelDescriptor(
                kernel_source=_PARALLEL_STATE_SOURCE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=args,
                runtime_args=[
                    ((c.x, c.y), [*[t.buffer_address() for t in tensors], i // workers, i % workers])
                    for i, c in enumerate(coords)
                ],
                config=ttnn.ReaderConfigDescriptor(),
            )
            ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def _chunk(self, x):
        b, s, _ = x.shape
        padded = (s + 31) // 32 * 32
        projections = self._linear(x, self.packed_weight)
        projected = projections[..., :2560]
        self._decode_conv_history_packed = (
            self._conv_decode_phase and b in (8, 16, 32) and s == 1 and self.conv_decode_history_enabled
        )
        if self._decode_conv_history_packed:
            from .gdn_conv_pack import history_convolution

            q, k, v = history_convolution(self, projected)
        else:
            proj = ttnn.pad(projected, [(0, 0), (0, padded - s), (0, 0)], 0) if padded != s else projected
            lanes = []
            for lane in range(b):
                incoming = ttnn.to_layout(self.conv_state[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT)
                lanes.append(
                    ttnn.experimental.kda.qkv_causal_conv1d_silu(
                        ttnn.to_layout(proj[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT),
                        incoming,
                        *self.conv_taps,
                        512,
                        512,
                        1536,
                        actual_start=self.conv_zero,
                        predecessor_carry=incoming,
                        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                        compute_kernel_config=self.compute,
                    )
                )
            q, k, v = [ttnn.concat([row[i] for row in lanes], dim=0) if b > 1 else lanes[0][i] for i in range(3)]
            q, k, v = [ttnn.to_layout(t[:, :s, :], ttnn.TILE_LAYOUT) for t in (q, k, v)]
        self._roll_conv(projected)
        a = ttnn.typecast(ttnn.to_layout(projections[..., 4096:4108], ttnn.TILE_LAYOUT), ttnn.float32)
        g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)), activations=[ttnn.UnaryOpType.EXP])
        beta = ttnn.typecast(ttnn.sigmoid(ttnn.to_layout(projections[..., 4128:4140], ttnn.TILE_LAYOUT)), ttnn.float32)
        o, new_state = self._recurrence(q, k, v, g, beta)
        o = ttnn.typecast(o, ttnn.bfloat16)
        gate = projections[..., 2560:4096]
        if padded != s:
            o = ttnn.pad(o, [(0, 0), (0, 0), (0, padded - s), (0, 0)], 0)
            gate = ttnn.pad(gate, [(0, 0), (0, padded - s), (0, 0)], 0)
        o = ttnn.reshape(o, (b * 12, padded, 128))
        gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o,
            gate,
            self.norm_weight_flat,
            12,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute,
            output_dtype=ttnn.bfloat16,
        )
        gated = ttnn.to_layout(gated[:, :s, :], ttnn.TILE_LAYOUT)
        return self.output_projection(gated, concatenated=self.output_sequence > 1)

    def _recurrence(self, q, k, v, g, beta):
        b, s, _ = q.shape
        q = self._heads(q, b, s, 4)
        k = self._heads(k, b, s, 4)
        q = ttnn.rms_norm(
            ttnn.typecast(q, ttnn.float32), weight=self.qscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        k = ttnn.rms_norm(
            ttnn.typecast(k, ttnn.float32), weight=self.kscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        v = ttnn.typecast(self._heads(v, b, s, 12), ttnn.float32)
        return _local_recurrence(self, q, k, v, g, beta)

    def _roll_conv(self, projected):
        b, s, _ = projected.shape
        workers = 80
        coords = [ttnn.CoreCoord(i % 10, i // 10) for i in range(workers)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tensors = [self.conv_state, projected]
        args = [b, s, workers]
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k = ttnn.KernelDescriptor(
            kernel_source=_LOCAL_ROLL_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=6144,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=6144)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[k], cbs=[cb]))


def _local_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 12, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = self.recurrent_state
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    jobs = batch * 48
    workers = min(jobs, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [(i * jobs // workers, (i + 1) * jobs // workers - i * jobs // workers) for i in range(workers)]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    broadcast = self.recurrence_broadcast
    reader = (
        (_BROADCAST_GDN_READER if length == 1 else _BROADCAST_PREFILL_READER)
        if broadcast
        else (_LOCAL_GDN_READER if length == 1 else _LOCAL_PREFILL_READER)
    )
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=reader,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            config=ttnn.ReaderConfigDescriptor(),
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length])
                for c, job in zip(coords, assignments)
            ],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_BROADCAST_DECODE_COMPUTE if broadcast else _LOCAL_DECODE_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1]
    if length > 1:
        sizes.append(1)
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, count in enumerate(sizes)
        if not (length > 1 and i in (0, 12))
    ]
    if length > 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=4 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return output, final


from .optimized_decoder import _PARALLEL_STATE_SOURCE

_LOCAL_DECODE_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,cb,tile); tile_regs_release();
}
template<uint32_t Out> void reduce_product() {
    cb_wait_front(8,4);
    compute_kernel_lib::reduce<PoolType::SUM,ReduceDim::REDUCE_COL,8,6,Out,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(4,1));
    cb_pop_front(8,4);
}
void product(uint32_t state,uint32_t column) {
    cb_wait_front(state,4); cb_wait_front(column,4); cb_reserve_back(8,4);
    pack_reconfig_data_format(8);
    for(uint32_t i=0;i<4;++i) {
        tile_regs_acquire(); load(state,i,0); load(column,i,1);
        mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(8,i);
    }
    cb_push_back(8,4);
}
void kernel_main() {
    compute_kernel_hw_startup(0,4,7);
    const uint32_t count=get_arg_val<uint32_t>(0);
    for(uint32_t job=0;job<count;++job) {
        cb_wait_front(0,4); cb_wait_front(4,1); cb_reserve_back(7,4);
        pack_reconfig_data_format(7);
        for(uint32_t i=0;i<4;++i) {
            tile_regs_acquire(); load(0,i,0); load(4,0,1);
            mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(7,i);
        }
        cb_push_back(7,4); cb_pop_front(0,4); cb_pop_front(4,1);
        product(7,2); reduce_product<9>();
        cb_wait_front(9,1); cb_wait_front(3,1); cb_wait_front(5,1);
        cb_reserve_back(11,1); pack_reconfig_data_format(11);
        for(uint32_t i=0;i<1;++i) {
            tile_regs_acquire(); load(3,i,0); load(9,i,1); load(5,0,2);
            sub_binary_tile_init(); sub_binary_tile(0,1,0);
            mul_binary_tile_init(); mul_binary_tile(0,2,0); finish(11,i);
        }
        cb_push_back(11,1); cb_pop_front(9,1); cb_pop_front(3,1); cb_pop_front(5,1);
        cb_wait_front(11,1); cb_reserve_back(12,4); pack_reconfig_data_format(12);
        for(uint32_t i=0;i<4;++i) {
            tile_regs_acquire(); load(2,i,0); load(11,0,1); load(7,i,2);
            sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);
            add_binary_tile_init(); add_binary_tile(0,2,0); finish(12,i);
        }
        cb_push_back(12,4); cb_pop_front(11,1); cb_pop_front(7,4); cb_pop_front(2,4);
        product(12,1); reduce_product<13>(); cb_pop_front(1,4);
    }
    cb_pop_front(6,1);
}
"""

_LOCAL_GDN_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t head=job/4, column=job%4;
        const uint32_t b = head/12, h = head%12;
        cb_reserve_back(0,4);
        for (uint32_t i=0;i<4;++i)
            noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=head*16+i*4+column},{});
        noc.async_read_barrier();
        cb_push_back(0,4);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,which==3?1:4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {
                uint32_t page = which==3 ? (b+h/32)*4+column : b*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,which==3?1:4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=b+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=b+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(0,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,4);
        for(uint32_t i=0;i<4;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=head*16+i*4+column});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,1);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        for(uint32_t t=0;t<1;++t) {
            for(uint32_t r=1;r<32;++r) for(uint32_t c=0;c<32;++c) out[t*1024+off(r,c)]=0;
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(13)+t*4096),output,4096,{},
                {.page_id=head*4+column});
        }
        noc.async_write_barrier();
        cb_pop_front(13,1);
        cb_pop_front(12,4);
    }
}
"""

_LOCAL_PREFILL_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    cb_reserve_back(15,1);
    auto assembled=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t head=job/4, column=job%4;
        const uint32_t b = head/12, h = head%12;
        for(uint32_t token=0;token<length;++token) {
        cb_reserve_back(0,4);
        for (uint32_t i=0;i<4;++i) {
            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=head*16+i*4+column},{});
            // Later tokens reuse the FP32 bytes through aliased CB0/CB12.

        }
        noc.async_read_barrier();
        cb_push_back(0,4);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,which==3?1:4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {
                uint32_t page = which==3 ? ((b*length+token)+h/32)*4+column : (b*length+token)*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,which==3?1:4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,4);
        if(token+1==length) for(uint32_t i=0;i<4;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=head*16+i*4+column});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,1);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        if(token%32==0) for(uint32_t i=0;i<1024;++i) assembled[i]=0;
        for(uint32_t t=0;t<1;++t) for(uint32_t c=0;c<32;++c)
            assembled[t*1024+off(token%32,c)]=out[t*1024+off(0,c)];
        if(token%32==31 || token+1==length) {
            for(uint32_t t=0;t<1;++t)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+t*4096),output,4096,{},
                    {.page_id=(head*st+token/32)*4+column});
            noc.async_write_barrier();
        }
        cb_pop_front(13,1);
        cb_pop_front(12,4);
        }
    }
}
"""


def _broadcast_recurrence_reader(source):
    # FP32 unpack-to-DST broadcasts replace only RISC replication stores.
    # Keep DRAM reads, state layout, arithmetic and reduction order unchanged.
    old = """for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }"""
    assert source.count(old) == 1
    source = source.replace(old, "dst[tile*1024+off(r,0)]=value;")
    old = "for(uint32_t i=0;i<1024;++i) dst[i]=value;"
    assert source.count(old) == 1
    return source.replace(old, "dst[0]=value;")


def _broadcast_recurrence_compute(source):
    old = """    copy_init(cb);
    copy_tile(cb,tile,dst);"""
    assert source.count(old) == 1
    source = '#include "api/compute/bcast.h"\n' + source
    return source.replace(
        old,
        """    if(cb==1 || cb==2) {
        unary_bcast_init<BroadcastType::COL>(cb);
        unary_bcast<BroadcastType::COL>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::COL>(cb);
    } else if(cb==4 || cb==5) {
        unary_bcast_init<BroadcastType::SCALAR>(cb);
        unary_bcast<BroadcastType::SCALAR>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::SCALAR>(cb);
    } else {
        copy_init(cb);
        copy_tile(cb,tile,dst);
    }""",
    )


_BROADCAST_GDN_READER = _broadcast_recurrence_reader(_LOCAL_GDN_READER)
_BROADCAST_PREFILL_READER = _broadcast_recurrence_reader(_LOCAL_PREFILL_READER)
_BROADCAST_DECODE_COMPUTE = _broadcast_recurrence_compute(_LOCAL_DECODE_COMPUTE)


_LOCAL_ROLL_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), S=get_compile_time_arg_val(1), W=get_compile_time_arg_val(2);
    constexpr auto ca=TensorAccessorArgs<3>();
    constexpr auto pa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto cache=TensorAccessor(ca,get_arg_val<uint32_t>(0));
    const auto proj=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    auto old=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+2048);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+4096);
    for(uint32_t page=get_arg_val<uint32_t>(2);page<B*80;page+=W) {
        const uint32_t b=page/80,c=page%80;
        for(uint32_t i=0;i<1024;++i)out[i]=0;
        if(S<3){noc.async_read(cache,CoreLocalMem<uint32_t>(base),2048,{.page_id=page},{});noc.async_read_barrier();}
        uint32_t loaded=0xffffffff;
        for(uint32_t r=0;r<3;++r) {
            const uint32_t history=S+r;
            const bool from_old=history<3;
            const uint32_t row=from_old?history:history-3;
            if(!from_old && loaded!=row/32) {
                noc.async_read(proj,CoreLocalMem<uint32_t>(base+2048),2048,{.page_id=(b*((S+31)/32)+row/32)*80+c},{});
                noc.async_read_barrier();loaded=row/32;
            }
            for(uint32_t side=0;side<2;++side)for(uint32_t j=0;j<16;++j) {
                const uint32_t offset=(row%32/16)*512+side*256+(row%16)*16+j;
                out[side*256+r*16+j]=from_old?old[offset]:src[offset];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+4096),cache,2048,{}, {.page_id=page});noc.async_write_barrier();
    }
}
"""


class ParallelQSA(OptimizedQSA):
    cache_dtype = ttnn.fp8_e4m3
    decode_chunk = 128

    @classmethod
    def build(cls, state, cfg, ops, layer_idx, batch_size=1, max_requests=1, attention_pool_tokens=None):
        import torch

        obj = FusedQSA.from_state_dict.__func__(
            cls,
            state,
            hf_config=cfg,
            mesh_device=ops.mesh,
            layer_idx=layer_idx,
            batch_size=batch_size,
            max_requests=max_requests,
            allocate_kv_cache=False,
            allocate_block_ids=attention_pool_tokens is None,
            attention_pool_tokens=attention_pool_tokens,
        )
        obj.ops = ops
        qg = state["q_proj.weight"].reshape(24, 512, 2560)
        projections = []
        for r in range(4):
            g = r // 2
            projections.append(
                torch.cat(
                    (
                        qg[r * 6 : (r + 1) * 6, :256].reshape(1536, 2560),
                        state["k_proj.weight"][g * 256 : (g + 1) * 256],
                        state["v_proj.weight"][g * 256 : (g + 1) * 256],
                        qg[r * 6 : (r + 1) * 6, 256:].reshape(1536, 2560),
                        state["indexer.index_qk_proj.weight"],
                    ),
                    0,
                ).T
            )
        obj.packed_projection = ops.shards(projections, ttnn.bfloat8_b)
        # Separate immutable decode weights preserve the faster BF8 prefill
        # policy and the qualified BF8-to-BF4 quantization used during decode.
        obj.decode_projection = ttnn.typecast(obj.packed_projection, ttnn.bfloat4_b) if batch_size == 1 else None
        obj._decode_phase = False
        obj.wo = ops.shards([state["o_proj.weight"][:, r * 1536 : (r + 1) * 1536].T for r in range(4)], ttnn.bfloat8_b)
        obj.projection_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        obj.output_projection = AttentionOutputProjection(
            state["o_proj.weight"],
            obj.wo,
            ops,
            obj.projection_compute,
            allocate_column_weight=attention_pool_tokens is None or batch_size == 1,
        )
        obj.prefill_projection_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        obj.packed_singleton_compute = obj.projection_compute
        obj.chunk_size = 512 if batch_size == 1 else 128
        obj.chunk_offsets = ops.upload(torch.arange(obj.chunk_size).float().reshape(1, 1, -1, 1), ttnn.float32)
        obj.pool_tokens = attention_pool_tokens if obj.paged_pooled else batch_size * obj.max_context
        if batch_size == 1:
            del obj._physical_update_indices, obj._index_core
        obj.sparse_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        obj.indexer_prefix = ops.upload(torch.zeros(batch_size, 4, 31, 128))
        obj.indexer_weights = ops.upload(torch.ones(batch_size, 1, obj.chunk_size + 32, 4))
        # Qualify only the B32 serving owner. Its shallow serial-prefill view
        # inherits this flag and scalar; the original B1 profile stays unchanged.
        # Allocate before generator capture. CQ0 serializes each lane's bound
        # producer and indexer reader before the next lane can overwrite it.
        obj.indexer_prefix_cap = batch_size == 32 and attention_pool_tokens is not None
        obj.indexer_prefix_bound = (
            ttnn.empty(
                (1,),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=ops.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if obj.indexer_prefix_cap
            else None
        )
        obj.indexer_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        obj.packed_cache = ttnn.empty(
            (1, 1, obj.pool_tokens, 512), dtype=obj.cache_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=ops.mesh
        )
        return obj

    def prefill_forward(self, hidden_states, **kwargs):
        self._decode_phase = False
        return super().prefill_forward(hidden_states, **kwargs)

    def decode_forward(self, hidden_states, **kwargs):
        self._decode_phase = True
        return super().decode_forward(hidden_states, **kwargs)

    def _linear(self, x, weight):
        # Explicit phase, not sequence shape: singleton prefill remains BF8.
        if self._decode_phase and self.batch_size == 1 and weight is self.packed_projection:
            return self.ops.linear(
                x, self.decode_projection, compute_kernel_config=self.projection_compute, max_cores=22
            )
        compute = self.projection_compute
        if weight is self.packed_projection:
            compute = self.prefill_projection_compute if x.shape[-2] > 1 else self.packed_singleton_compute
        return self.ops.linear(
            x, weight, compute_kernel_config=compute, max_cores=20 if weight is self.packed_projection else 10
        )

    def _adapter(self, source, output, positions, page_table, mode, seq, index_width=None):
        width = self.index_width if index_width is None else index_width
        work = self.batch_size * seq
        grid = self.device.compute_with_storage_grid_size()
        _, cores, g1, _, n1, n2 = ttnn.split_work_to_cores(grid, work, True)
        tensors = [source, output, positions, page_table]
        args = [
            self.batch_size,
            seq,
            self.page_size,
            self.max_context,
            self.pool_tokens,
            width,
            mode,
            1024 if self.cache_dtype == ttnn.bfloat16 else 512,
        ]
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = []
        offset = 0
        for i, c in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
            n = n1 if i < g1.num_cores() else n2
            rt.append(((c.x, c.y), [*[t.buffer_address() for t in tensors], offset, n]))
            offset += n
        scratch = 128 + self.max_context // self.page_size * 4 + 2048 + width * 4
        scratch = (scratch + 63) // 64 * 64
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_LOCAL_QSA_ADAPTER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def _sparse_attention(self, q, selected, positions, page_table):
        b, _, s, _ = tuple(q.shape)
        k_chunk = self.decode_chunk if s == 1 and self.decode_chunk else self.sparse_chunk
        width = (2051 + k_chunk - 1) // k_chunk * k_chunk
        indices = ttnn.empty(
            (1, 1, b * s, width),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        selected = ttnn.to_layout(ttnn.typecast(selected, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        self._adapter(selected, indices, positions, page_table, 1, s, index_width=width)
        vectorized_decode = self._decode_phase and b > 1 and s == 1
        self._decode_fold_vectorized = vectorized_decode
        if vectorized_decode:
            # Logical permutation excludes implicit sequence padding when batch
            # becomes the sparse attention token axis. Never use a physical view.
            chunk = ttnn.permute(q, (2, 1, 0, 3))
            chunk = ttnn.pad(chunk, [(0, 0), (0, 26), (0, 0), (0, 0)], 0)
            folded = ttnn.to_layout(ttnn.concat((ttnn.zeros_like(chunk), chunk), dim=-1), ttnn.ROW_MAJOR_LAYOUT)
        else:
            groups = []
            for batch in range(b):
                for g in range(1):
                    chunk = q[batch : batch + 1, g * 6 : (g + 1) * 6, :, :]
                    chunk = ttnn.pad(chunk, [(0, 0), (0, 26), (0, 0), (0, 0)], 0)
                    groups.append(ttnn.concat((ttnn.zeros_like(chunk), chunk), dim=-1))
            folded = ttnn.to_layout(ttnn.concat(groups, dim=2), ttnn.ROW_MAJOR_LAYOUT)
        fmt = (
            ttnn.transformer.SparseKVFormat.BF16
            if self.cache_dtype == ttnn.bfloat16
            else ttnn.transformer.SparseKVFormat.FP8_E4M3
        )
        out = ttnn.transformer.sparse_sdpa(
            folded,
            self.packed_cache,
            indices,
            256,
            kv_format=fmt,
            scale=0.0625,
            k_chunk_size=k_chunk,
            compute_kernel_config=self.sparse_compute,
        )
        if vectorized_decode:
            # sparse_sdpa returns unpadded DRAM ROW_MAJOR rows; six heads can
            # therefore be flattened without introducing tile padding as data.
            out = ttnn.permute(out[:, :6, :, :], (2, 0, 1, 3))
        else:
            rows = []
            for batch in range(b):
                heads = [out[:, :6, (batch + g) * s : (batch + g + 1) * s, :] for g in range(1)]
                rows.append(ttnn.concat(heads, dim=1))
            out = rows[0] if b == 1 else ttnn.concat(rows, dim=0)
            out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (b, 1, s, 1536))
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)

    def _route_pooled(self, slots, position, *, store):
        # In serving mode page ownership carries index state across lane changes.
        if not self.paged_pooled:
            return super()._route_pooled(slots, position, store=store)

    def _select_groups(self, pooled, iq, positions_packed, page_table, seq):
        batch = self.batch_size
        if self.paged_pooled:
            from .paged_qsa import select_groups

            return select_groups(self, pooled, iq, positions_packed, page_table, seq)
        self._transfer(pooled, self.pooled_keys, positions_packed, page_table, mode=2, sequence=seq, heads=1)
        qq = ttnn.to_layout(ttnn.concat((self.indexer_prefix, iq), dim=2), ttnn.TILE_LAYOUT)
        padded = (seq + 31 + 31) // 32 * 32
        qq = ttnn.pad(qq, [(0, 0), (0, 0), (0, padded - seq - 31), (0, 0)], 0)
        score_rows = []
        for lane in range(batch):
            score_rows.append(
                ttnn.experimental.indexer_score_dsa(
                    qq[lane : lane + 1],
                    self.pooled_keys,
                    self.indexer_weights[lane : lane + 1, :, :padded, :],
                    chunk_start_idx=self.blocks - 32,
                    key_compression_ratio=1,
                    cache_batch_idx=lane,
                    program_config=ttnn.IndexerScoreProgramConfig(
                        q_chunk_size=32, k_chunk_size=self.indexer_chunk, head_group_size=4
                    ),
                    compute_kernel_config=self.indexer_compute,
                )
            )
        scores = score_rows[0] if batch == 1 else ttnn.concat(score_rows, dim=0)
        # Release the padded per-lane score owners before slicing/copying the
        # selected rows; retaining them crosses the BF16 full-context DRAM peak.
        del score_rows
        scores = scores[:, :, 31 : 31 + seq, :]
        scores = ttnn.to_layout(ttnn.typecast(scores, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        bound = self._mask_scores(scores, positions_packed, seq)
        selected = ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=bound)
        return selected

    def _chunk_forward(self, hidden_states, *, current_pos, page_table):
        """Vectorized projections/selection with bounded selected-KV attention."""
        batch, seq = (self.batch_size, hidden_states.shape[-2])
        x = self._reshape_logical(hidden_states, (batch, 1, seq, 2560))
        positions_packed = self._packed_position(current_pos)
        start = ttnn.permute(self._position_float(self._canonical_position(current_pos)), (1, 0, 2, 3))
        positions = ttnn.add(start, self.chunk_offsets[:, :, :seq, :])

        def heads(value, count, width):
            value = self._reshape_logical(value, (batch, seq, count, width))
            return ttnn.permute(value, (0, 2, 1, 3))

        projections = self._linear(x, self.packed_projection)
        qkv = ttnn.reshape(projections[..., :2048], (batch, seq, 2048))
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=6, num_kv_heads=1, transpose_key=False
        )
        rope_cos, rope_sin = _shared_rope_tables(self, positions, batch, seq)
        q = _shared_apply_rope(self, self._norm(q, self.nq), rope_cos, rope_sin)
        k = _shared_apply_rope(self, self._norm(k, self.nk), rope_cos, rope_sin)
        gate = projections[..., 2048:3584]
        index = projections[..., 3584:]
        iq = _shared_apply_rope(self, self._norm(heads(index[..., :512], 4, 128), self.niq), rope_cos, rope_sin)
        raw = index[..., 512:640]
        self._write_packed(k, v, positions_packed, page_table)
        self._transfer(raw, self.raw_key_cache, positions_packed, page_table, mode=0, sequence=seq, heads=1)
        groups = self._transfer(
            self.raw_key_cache,
            self._allocate((batch * seq, 1, 4, 128)),
            positions_packed,
            page_table,
            mode=1,
            sequence=seq,
            heads=1,
        )
        pooled = ttnn.typecast(ttnn.mean(ttnn.typecast(groups, ttnn.float32), dim=2, keepdim=True), ttnn.bfloat16)
        pooled = self._reshape_logical(pooled, (batch, 1, seq, 128))
        block_start = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4.0)
        pooled = self._chunk_rope(self._norm(pooled, self.nik), block_start)
        selected = self._select_groups(pooled, iq, positions_packed, page_table, seq)
        out = self._sparse_attention(q, selected, positions_packed, page_table)
        out = self.output_projection(
            ttnn.multiply(out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
            concatenated=self.route_sequence > 1,
        )
        return self._reshape_logical(out, (1, batch, seq, 640))


_LOCAL_QSA_ADAPTER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), S=get_compile_time_arg_val(1);
    constexpr uint32_t P=get_compile_time_arg_val(2), C=get_compile_time_arg_val(3);
    constexpr uint32_t T=get_compile_time_arg_val(4), L=get_compile_time_arg_val(5);
    constexpr uint32_t mode=get_compile_time_arg_val(6), rowbytes=get_compile_time_arg_val(7);
    constexpr auto aa=TensorAccessorArgs<8>();
    constexpr auto oa=TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto src=TensorAccessor(aa,get_arg_val<uint32_t>(0));
    const auto dst=TensorAccessor(oa,get_arg_val<uint32_t>(1));
    const auto pos=TensorAccessor(pa,get_arg_val<uint32_t>(2));
    const auto table=TensorAccessor(ta,get_arg_val<uint32_t>(3));
    const uint32_t begin=get_arg_val<uint32_t>(4), count=get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();
    constexpr uint32_t tablebytes=C/P*4;
    const uint32_t paddr=base, taddr=base+128, iaddr=taddr+tablebytes, oaddr=iaddr+2048;
    auto pp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(paddr);
    auto pages=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(taddr);
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iaddr);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(oaddr);
    Noc noc;
    for(uint32_t job=begin;job<begin+count;++job) {
        const uint32_t b=job/(S);
        const uint32_t s=job%S;
        ASSERT(b<B);
        noc.async_read(pos,CoreLocalMem<uint32_t>(paddr),128,{.page_id=b},{});
        noc.async_read(table,CoreLocalMem<uint32_t>(taddr),tablebytes,{.page_id=b},{});
        noc.async_read_barrier();
        const uint32_t absolute=pp[0]+s;ASSERT(absolute<C);
        if constexpr(mode==0) {
            const uint32_t g=0;
            ASSERT(pages[absolute/P]<T/P);
            const uint32_t physical=pages[absolute/P]*P+absolute%P;ASSERT(physical<T);
            noc.async_read(src,CoreLocalMem<uint32_t>(iaddr),rowbytes,{.page_id=job},{});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(iaddr),dst,rowbytes,{}, {.page_id=physical});
            noc.async_write_barrier();
        } else {
            noc.async_read(src,CoreLocalMem<uint32_t>(iaddr),2048,{.page_id=b*S+s},{});
            noc.async_read_barrier();
            const uint32_t complete=(absolute+1)/4;
            uint32_t cursor=0;
            for(uint32_t e=0;e<512;++e) {
                const uint32_t block=ids[e];
                if(block>=complete) continue;
                for(uint32_t j=0;j<4;++j) {
                    const uint32_t logical=block*4+j;
                    ASSERT(pages[logical/P]<T/P);
                    const uint32_t physical=pages[logical/P]*P+logical%P;ASSERT(physical<T);
                    out[cursor++]=physical;
                }
            }
            for(uint32_t logical=complete*4;logical<=absolute;++logical) {
                ASSERT(pages[logical/P]<T/P);
                const uint32_t physical=pages[logical/P]*P+logical%P;ASSERT(physical<T);
                out[cursor++]=physical;
            }
            ASSERT(cursor>0 && cursor<=2051);
            for(uint32_t j=cursor;j<L;++j) out[j]=0xffffffff;
            noc.async_write(CoreLocalMem<uint32_t>(oaddr),dst,L*4,{}, {.page_id=b*S+s});
            noc.async_write_barrier();
        }
    }
}
"""

from pathlib import Path

from .fused_decoder import _shared_apply_rope, _shared_rope_tables
from .ple import PLECache, PLEWorkspace
from .ple_hash import ngram_indices


class ParallelPLE:
    """Full-domain table sharding, with no prompt-dependent weight selection."""

    def __init__(self, state, cfg, ops, cache_dir=None):
        import torch

        self.ops, self.mesh, self.eps = ops, ops.mesh, cfg.rms_norm_eps
        self.compute = ops.compute
        self.eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
        self.workspaces = {}
        self.multipliers = state["ple_embedding.layer_multipliers"].tolist()
        self.primes = state["ple_embedding.ngram_heads_vocab_sizes"].tolist()
        self.offsets = state["ple_embedding.ngram_heads_offsets"].tolist()
        self.rank = ops.shards(
            [torch.tensor([r], dtype=torch.int32) for r in range(4)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.tables = []
        dest = None if cache_dir is None else Path(cache_dir) / "tp4_tables"
        if dest:
            dest.mkdir(parents=True, exist_ok=True)
        for i in range(32):
            path = None if dest is None else dest / f"group_{i}.tensorbin"
            if path and path.exists():
                table = ttnn.load_tensor(path, device=ops.mesh)
            else:
                table = ops.shards(
                    [state[f"ple_embedding.ngram_embedding.shard_{r*32+i}.weight"] for r in range(4)], ttnn.bfloat4_b
                )
                if path:
                    ttnn.dump_tensor(path, table)
            self.tables.append(table)
        key_parts = _hidden_parts(state["key_proj.weight"].T)
        value = state["value_proj.weight"].T
        self.projection = ops.shards(
            [torch.cat((key_parts[r], value[:, r * 640 : (r + 1) * 640]), -1) for r in range(4)], ttnn.bfloat8_b
        )
        self.projection_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._norm_host_parts = {}
        self.flat_norm_weights = {}
        for name in ("norm_key", "norm_query", "norm_conv"):
            weight = 1 + state[name + ".weight"].float()
            parts = [x.reshape(1, 1, 4, 640) for x in _hidden_parts(weight)]
            setattr(self, name, ops.shards(parts, ttnn.float32))
            self._norm_host_parts[id(getattr(self, name))] = parts
        conv = state["conv1d.weight"].squeeze(1).T.reshape(4, 1, 10240).float()
        self.conv = ops.shards(_hidden_parts(conv), ttnn.float32)

    def make_cache(self, max_requests):
        import torch

        token = torch.zeros((max_requests, 32), dtype=torch.int32)
        token[:, :2] = self.eos
        return PLECache(
            self.ops.upload(token, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            self.ops.upload(torch.zeros(max_requests * 9, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        )

    def prepare(self, *, batch, physical_seq_len):
        key = (batch, physical_seq_len)
        if key not in self.workspaces:
            n = batch * physical_seq_len
            for identity, parts in self._norm_host_parts.items():
                if (identity, n) not in self.flat_norm_weights:
                    self.flat_norm_weights[identity, n] = self.ops.shards(
                        [part.repeat(1, 1, n, 1) for part in parts], ttnn.float32
                    )

            def alloc(shape, dtype):
                return ttnn.empty(shape, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)

            self.workspaces[key] = PLEWorkspace(
                alloc((n, 16), ttnn.uint32), alloc((n, 2560), ttnn.bfloat16), alloc((4, n, 2560), ttnn.bfloat16)
            )
        return self.workspaces[key]

    def norm(self, x, weight, n):
        # Four stream rows per token have the same independent global-width
        # RMS reduction. Flattening them avoids padding each group of4 to32
        # before FP32 conversion (40MiB versus5MiB at512 tokens).
        x = ttnn.typecast(ttnn.reshape(x, (1, 1, n * 4, 640)), ttnn.float32)
        normalized = self.ops.norm(x, self.flat_norm_weights[id(weight), n], self.eps, reuse_output=True)
        return ttnn.reshape(ttnn.typecast(normalized, ttnn.bfloat16), (1, n, 4, 640))

    def forward(self, x, *, tokens, positions, slots, valid_lengths, cache):
        batch, seq = tokens.shape
        n = batch * seq
        ws = self.workspaces[(batch, seq)]
        ngram_indices(
            tokens,
            positions,
            slots,
            cache.token_history,
            valid_lengths=valid_lengths,
            output=ws.indices,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
            eos=self.eos,
        )
        _local_ple_lookup(self.tables, ws.indices, output=ws.embeddings, rank=self.rank)
        emb = self.ops.reduce(ttnn.reshape(ttnn.to_layout(ws.embeddings, ttnn.TILE_LAYOUT), (1, 1, n, 2560)))
        projections = self.ops.linear(emb, self.projection, compute_kernel_config=self.projection_compute, max_cores=20)
        key = self.norm(projections[..., :2560], self.norm_key, n)
        # Keep only the compact value slice while the query norm runs. The
        # padded four-stream query and key norms otherwise compete with dead
        # projection storage for the static kernel-buffer region.
        value_compact = projections[..., 2560:]
        del projections
        query = self.norm(x, self.norm_query, n)
        dot = self.ops.reduce(ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True))
        del key, query
        gate = ttnn.multiply(dot, 2560**-0.5)
        gate = ttnn.multiply(
            gate,
            gate,
            input_tensor_a_activations=[
                ttnn.UnaryOpType.ABS,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 1e-6),
                ttnn.UnaryOpType.SQRT,
            ],
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGN],
        )
        del dot
        value = ttnn.reshape(value_compact, (1, n, 1, 640))
        del value_compact
        gated = ttnn.reshape(
            ttnn.multiply(gate, value, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), (1, 1, n, 2560)
        )
        del gate, value
        normalized = self.norm(gated, self.norm_conv, n)
        normalized = ttnn.to_layout(ttnn.reshape(normalized, (n, 2560)), ttnn.ROW_MAJOR_LAYOUT)
        _local_ple_conv(
            normalized, positions, slots, valid_lengths, cache.conv_history, output=ws.taps, batch=batch, seq=seq
        )
        taps = ttnn.to_layout(ws.taps, ttnn.TILE_LAYOUT)
        convolved = ttnn.experimental.fast_reduce_nc(
            ttnn.multiply(ttnn.typecast(taps, ttnn.float32), self.conv),
            dims=[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        convolved = ttnn.typecast(convolved[..., :n, :], ttnn.bfloat16)
        gated_rm = ttnn.to_layout(ttnn.reshape(gated, (n, 2560)), ttnn.ROW_MAJOR_LAYOUT)
        _local_ple_mask(gated_rm, valid_lengths, batch=batch, seq=seq)
        result = ttnn.add(
            ttnn.to_layout(gated_rm, ttnn.TILE_LAYOUT),
            ttnn.reshape(convolved, (n, 2560)),
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.reshape(result, (1, batch, seq, 2560))


def _local_ple_lookup(tables, indices, *, output, rank):
    rows = tables[0].shape[-2]
    n = indices.shape[0]
    mesh = indices.device()
    grid = mesh.compute_with_storage_grid_size()
    _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, n * 16, True)
    work = []
    start = 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        count = work1 if i < group1.num_cores() else work2
        work.append(
            (
                (core.x, core.y),
                [
                    indices.buffer_address(),
                    output.buffer_address(),
                    start,
                    count,
                    *[t.buffer_address() for t in tables],
                    rank.buffer_address(),
                ],
            )
        )
        start += count
    args = [
        rows,
        len(tables),
        *ttnn.TensorAccessorArgs(indices).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(tables[0]).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(rank).get_compile_time_args(),
    ]
    cb = ttnn.CBDescriptor(
        total_size=1024,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=1024)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_LOCAL_PLE_LOOKUP,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=work,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([*tables, indices, output, rank], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return output


_LOCAL_PLE_LOOKUP = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Decode BFP4_b exactly as blockfloat_common.cpp, directly into BF16 bits.
static uint16_t decode(uint8_t nibble, uint8_t exponent) {
    uint32_t mantissa = nibble & 7;
    if (!mantissa) {
        return 0;
    }
    uint32_t shifts = 0;
    while (!(mantissa & 4)) {
        mantissa <<= 1;
        ++shifts;
    }
    mantissa = (mantissa << 1) & 7;
    return ((nibble >> 3) << 15) | ((exponent - shifts) << 7) | (mantissa << 4);
}

void kernel_main() {
    Noc noc;
    constexpr uint32_t rows_per_shard = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr auto indices_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto indices = TensorAccessor(indices_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t scratch = cb.get_write_ptr();
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto tile = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch + 64);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 640);
    constexpr auto rank_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    const auto rank_tensor = TensorAccessor(rank_args, get_arg_val<uint32_t>(4 + num_shards));
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(scratch+960),4,{.page_id=0},{});
    noc.async_read_barrier();
    const uint32_t rank = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch+960);
    for (uint32_t work = start; work < start + count; ++work) {
        const uint32_t token=work/16, head=work%16;
        noc.async_read(indices, CoreLocalMem<uint32_t>(scratch), 64, {.page_id = token}, {});
        noc.async_read_barrier();
        {
            const uint32_t index = ids[head];
            const uint32_t shard = index / rows_per_shard;
            ASSERT(shard < num_shards * 4);
            if (shard / num_shards != rank) {
                for (uint32_t c=0;c<160;++c) out[c]=0;
            } else {
            const uint32_t row = index % rows_per_shard;
            const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(4 + shard % num_shards));
            for (uint32_t col_tile = 0; col_tile < 5; ++col_tile) {
                noc.async_read(
                    table, CoreLocalMem<uint32_t>(scratch + 64), 576, {.page_id = (row / 32) * 5 + col_tile}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t face = (row % 32 / 16) * 2 + c / 16;
                    const uint32_t face_row = row % 16;
                    const uint32_t elem = face * 256 + face_row * 16 + c % 16;
                    const uint8_t packed = tile[64 + elem / 2];
                    const uint8_t nibble = (packed >> (4 * (elem % 2))) & 15;
                    out[col_tile * 32 + c] = decode(nibble, tile[face * 16 + face_row]);
                }
            }
        }
        }
        noc.async_write(CoreLocalMem<uint32_t>(scratch + 640), output, 320, {}, {.page_id = token, .offset_bytes = head * 320});
        noc.async_write_barrier();
    }
}

"""


def _local_ple_conv(x, positions, slots, valid_lengths, history, *, output, batch, seq):
    """Device gather/cache update, separated from tensor-core arithmetic."""
    grid = x.device().compute_with_storage_grid_size()
    if batch > grid.x * grid.y:
        raise ValueError("PLE convolution needs one worker per batch lane")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    tensors = [x, positions, slots, valid_lengths, history, output]
    args = [batch, seq]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=5184,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=5184)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_LOCAL_PLE_CONV,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


_LOCAL_PLE_CONV = r"""// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t seq = get_compile_time_arg_val(1);
    constexpr uint32_t bytes = 2560 * 2;
    constexpr auto xa = TensorAccessorArgs<2>();
    constexpr auto pa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(1));
    const auto slots = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto valid = TensorAccessor(va, get_arg_val<uint32_t>(3));
    const auto history = TensorAccessor(ha, get_arg_val<uint32_t>(4));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(5));
    const uint32_t lane = get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto control = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + bytes);
    Noc noc;
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + bytes), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t position = control[0];
    noc.async_read(slots, CoreLocalMem<uint32_t>(base + bytes), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t slot = control[0];
    noc.async_read(valid, CoreLocalMem<uint32_t>(base + bytes), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t length = control[0];
    ASSERT(length <= seq);
    for (uint32_t tap = 0; tap < 4; ++tap) {
        for (uint32_t t = 0; t < seq; ++t) {
            const int32_t source = int32_t(t) - 9 + int32_t(tap) * 3;
            if (t >= length || (source < 0 && position == 0)) {
                for (uint32_t j = 0; j < bytes / 4; ++j) {
                    words[j] = 0;
                }
            } else if (source < 0) {
                noc.async_read(
                    history, CoreLocalMem<uint32_t>(base), bytes, {.page_id = slot * 9 + uint32_t(source + 9)}, {});
                noc.async_read_barrier();
            } else {
                noc.async_read(x, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * seq + uint32_t(source)}, {});
                noc.async_read_barrier();
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(base), output, bytes, {}, {.page_id = tap * batch * seq + lane * seq + t});
            noc.async_write_barrier();
        }
    }
    // Advance by logical length. Ascending writes are safe for overlapping
    // cache shifts because every old source row is >= its destination row.
    for (uint32_t j = 0; j < 9; ++j) {
        const int32_t source = int32_t(length) - 9 + int32_t(j);
        if (source < 0 && position == 0) {
            for (uint32_t k = 0; k < bytes / 4; ++k) {
                words[k] = 0;
            }
        } else if (source < 0) {
            noc.async_read(
                history, CoreLocalMem<uint32_t>(base), bytes, {.page_id = slot * 9 + uint32_t(source + 9)}, {});
            noc.async_read_barrier();
        } else {
            noc.async_read(x, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * seq + uint32_t(source)}, {});
            noc.async_read_barrier();
        }
        noc.async_write(CoreLocalMem<uint32_t>(base), history, bytes, {}, {.page_id = slot * 9 + j});
        noc.async_write_barrier();
    }
}
"""


def _local_ple_mask(x, valid_lengths, *, batch, seq):
    grid = x.device().compute_with_storage_grid_size()
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = [
        seq,
        *ttnn.TensorAccessorArgs(x).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(valid_lengths).get_compile_time_args(),
    ]
    cb = ttnn.CBDescriptor(
        total_size=5184,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=5184)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_LOCAL_PLE_MASK,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[
            ((c.x, c.y), [x.buffer_address(), valid_lengths.buffer_address(), i]) for i, c in enumerate(coords)
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([x, valid_lengths], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


_LOCAL_PLE_MASK = r"""// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t seq = get_compile_time_arg_val(0);
    constexpr auto xa = TensorAccessorArgs<1>();
    constexpr auto va = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto valid = TensorAccessor(va, get_arg_val<uint32_t>(1));
    const uint32_t lane = get_arg_val<uint32_t>(2);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    noc.async_read(valid, CoreLocalMem<uint32_t>(base), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t length = words[0];
    ASSERT(length <= seq);
    for (uint32_t j = 0; j < 1280; ++j) {
        words[j] = 0;
    }
    for (uint32_t t = length; t < seq; ++t) {
        noc.async_write(CoreLocalMem<uint32_t>(base), x, 5120, {}, {.page_id = lane * seq + t});
        noc.async_write_barrier();
    }
}
"""


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch_size=1,
        max_requests=None,
        chunk_size=512,
        cache_dir=None,
        policy=MeshPolicy(),
        attention_pool_tokens=None,
    ):
        obj = cls()
        cfg = getattr(hf_config, "text_config", hf_config)
        if (cfg.hidden_size, cfg.hc_count, cfg.num_experts) != (2560, 4, 512):
            raise ValueError("Expected Qwen3.8-Flash-Next dimensions")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        prefix = f"model.language_model.layers.{layer_idx}."
        state = _PrefixView(state_dict, prefix) if prefix + "mlp.gate.weight" in state_dict else state_dict
        obj.mesh, obj.ops = mesh_device, MeshOps(mesh_device, policy)
        obj.batch_size, obj.chunk_size, obj.context = batch_size, chunk_size, cfg.max_position_embeddings
        obj.kind = cfg.layer_types[layer_idx]
        requests = max_requests or batch_size
        obj.attn_hyper = ParallelHyperConnection(state, "attn_hyper_connection", cfg, obj.ops)
        obj.mlp_hyper = ParallelHyperConnection(state, "mlp_hyper_connection", cfg, obj.ops)
        obj.mlp = ParallelMoE(state, cfg, obj.ops)
        if obj.kind == "linear_attention":
            obj.attention = ParallelGatedDeltaNet(
                _PrefixView(state, "linear_attn."),
                cfg,
                obj.ops,
                batch_size,
                requests,
                chunk_size,
                allocate_column_weight=attention_pool_tokens is None or batch_size == 1,
            )
            # Serving B32 owns this execution choice; its shallow B1 prefill
            # view inherits it. Standalone Stage 10 B1 keeps its original path.
            obj.attention.recurrence_broadcast = batch_size == 32 and attention_pool_tokens is not None
        else:
            obj.attention = ParallelQSA.build(
                _PrefixView(state, "self_attn."), cfg, obj.ops, layer_idx, batch_size, requests, attention_pool_tokens
            )
        obj.ple = None
        if layer_idx + 1 in cfg.ple_layer_ids:
            obj.ple = ParallelPLE(_PrefixView(state, "ple."), cfg, obj.ops, cache_dir)
            obj.ple_cache = obj.ple.make_cache(requests)
        obj.lengths = {}
        # Serving's generator prepares its actual buckets before warmup/capture.
        # Eager standalone S128 preparation otherwise retains unused PLE buffers.
        if attention_pool_tokens is None:
            obj.prepare([1, chunk_size])
        return obj

    def _chunk(self, x, *, positions, slots, tokens, page_table, decode):
        # A large physical chunk contains 128-token MoE groups. Apply its
        # selected prefill geometry to those groups too, without changing
        # short prompts, singleton tails, decode or later independent calls.
        previous = self.ops.large_prefill
        previous_l1 = self.ops.l1_prefill
        self.ops.large_prefill = not decode and x.shape[2] >= 256
        # The measured L1 family fits B1 chunks through512 rows. Larger custom
        # chunks and larger batches retain their existing DRAM working layout.
        self.ops.l1_prefill = not decode and x.shape[1] == 1 and x.shape[2] <= 512
        try:
            return self._chunk_impl(
                x, positions=positions, slots=slots, tokens=tokens, page_table=page_table, decode=decode
            )
        finally:
            self.ops.large_prefill = previous
            self.ops.l1_prefill = previous_l1

    def _chunk_impl(self, x, *, positions, slots, tokens, page_table, decode):
        size = x.shape[2]
        if self.ple is not None:
            x = ttnn.add(
                x,
                self.ple.forward(
                    x,
                    tokens=tokens,
                    positions=positions,
                    slots=slots,
                    valid_lengths=self.lengths[size],
                    cache=self.ple_cache,
                ),
            )
        mixed, inject = self.attn_hyper.read(x, decode=decode)
        forward = self.attention.decode_forward if decode else self.attention.prefill_forward
        if self.kind == "linear_attention":
            update = forward(mixed, current_positions=positions, slots=slots)
        else:
            update = forward(mixed, current_pos=positions, page_table=page_table, slots=slots)
        x = self.attn_hyper.write(x, update, inject)
        mixed, inject = self.mlp_hyper.read(x, decode=decode)
        return self.mlp_hyper.write(x, self.mlp(mixed), inject)

    def prefill_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if tuple(hidden_states.shape)[:2] != (1, self.batch_size) or hidden_states.shape[-1] != 2560:
            raise ValueError("Expected [1,B,S,2560] residual streams")
        length = hidden_states.shape[2]
        if not 1 <= length <= self.context:
            raise ValueError("Sequence length is outside the advertised context")
        if self.ple is not None and (tokens is None or slots is None):
            raise ValueError("PLE requires token IDs and request slots")
        # Reject missing setup before the first chunk can mutate request state.
        if self.ple is not None:
            sizes = {min(self.chunk_size, length - start) for start in range(0, length, self.chunk_size)}
            if not sizes.issubset(self.lengths):
                raise ValueError("Call prepare([sequence_length]) before executing this PLE prefill shape")
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            position = current_positions if start == 0 else ttnn.add(current_positions, start)
            x = hidden_states[:, :, start:stop, :]
            ids = None if tokens is None else tokens[:, start:stop]
            out = self._chunk(x, positions=position, slots=slots, tokens=ids, page_table=page_table, decode=False)
            # Retaining a full context of local residuals requires1.34GB/rank,
            # beyond L1 capacity. Stage completed chunks locally; intermediate
            # HC/matmul consumers stay in L1, and mesh ownership never changes.
            outputs.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if length > self.chunk_size else out)
            del out
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if tuple(hidden_states.shape) != (1, self.batch_size, 1, 2560):
            raise ValueError("Decode requires [1,B,1,2560]")
        return self._chunk(
            hidden_states, positions=current_positions, slots=slots, tokens=tokens, page_table=page_table, decode=True
        )
