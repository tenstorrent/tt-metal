"""Experimental on-device sharding; not an accepted production configuration."""

import math

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_base_candidate import CandidateBase


class LayoutCandidate(CandidateBase):
    @classmethod
    def from_state_dict(cls, state_dict, **kw):
        geometry = kw.pop("geometry", {})
        obj = super().from_state_dict(state_dict, **kw)
        obj.geometry = geometry
        obj.decoding = False
        obj.carry = geometry.get("carry", False)
        grid = obj.mesh_device.dram_grid_size()
        obj.banks = grid.x * grid.y
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        for name, weight in list(obj.weights.items()):
            if not name.endswith("proj.weight") and not name.endswith("prefill_qkv.weight"):
                continue
            k, n = tuple(weight.shape)[-2:]
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(cores, (k, math.ceil(n / (32 * obj.banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            obj.weights[name] = ttnn.to_memory_config(weight, mem)
        return obj

    def decode_forward(self, *a, **kw):
        self.decoding = True
        if self.carry:
            a = (
                ttnn.to_memory_config(a[0], self._working_memory(a[0], self.geometry.get("residual_cores", 8))),
                *a[1:],
            )
        try:
            return super().decode_forward(*a, **kw)
        finally:
            self.decoding = False

    def _working_memory(self, x, count):
        gx = max(v for v in range(1, min(11, count) + 1) if count % v == 0 and count // v <= 10)
        cores = ttnn.CoreGrid(x=gx, y=count // gx)
        return ttnn.create_sharded_memory_config(
            (math.ceil(x.shape[-2] / 32) * 32, x.shape[-1] // count),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _norm(self, x, name=None):
        if self.decoding and self.carry and x.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            shard = x.memory_config().shard_spec
            box = shard.grid.bounding_box().end
            width = shard.shape[1] // 32
            sub = max(v for v in (1, 2, 4, 8) if width % v == 0)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(box.x + 1, box.y + 1),
                subblock_w=sub,
                block_h=shard.shape[0] // 32,
                block_w=width,
                inplace=False,
            )
            return ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                weight=self.weights[name + ".weight"] if name else None,
                compute_kernel_config=self.kernel,
                program_config=pc,
                memory_config=x.memory_config(),
            )
        return super()._norm(x, name)

    def _linear(self, x, name):
        weight = self.weights[name + ".weight"]
        k, n = tuple(weight.shape)[-2:]
        group = "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
        ck = self.linear_kernels[group]
        if self.geometry.get("attn_act8" if group == "attention" else "mlp_act8") and x.dtype != ttnn.bfloat8_b:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        if self.decoding and x.shape[-2] > 32:
            # The width-sharded decode factory validates M==1 tile. Use its
            # large-M 2D counterpart, preserving weights and precision.
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            gy = min(10, math.ceil(x.shape[-2] / 32))
            pn = math.ceil(n / 32 / self.banks)
            sub = max(v for v in (1, 2, 4, 8) if pn % v == 0)
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(self.banks, gy),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=sub,
                per_core_M=math.ceil(x.shape[-2] / 32 / gy),
                per_core_N=pn,
                transpose_mcast=False,
                fuse_batch=False,
            )
            out = ttnn.linear(
                x,
                weight,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            if self.carry and name in ("self_attn.o_proj", "mlp.down_proj"):
                out = ttnn.to_memory_config(out, self._working_memory(out, self.geometry.get("residual_cores", 8)))
            return out
        if self.decoding:
            # A role may override geometry without changing the precision pairing.
            g = self.geometry.get(name, self.geometry.get("default", {}))
            count = g.get("cores", math.gcd(k // 32, n // 32, 8))
            mem = self._working_memory(x, count)
            x = ttnn.to_memory_config(x, mem)
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=g.get("block", k // 32 // count),
                per_core_M=math.ceil(x.shape[-2] / 32),
                per_core_N=math.ceil(n / 32 / count),
                num_workers_per_dram_bank=g.get("readers", 1),
                fused_activation=(
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
                    if self.geometry.get("fused_gate") and name == "mlp.gate_proj"
                    else None
                ),
            )
            out = ttnn.linear(
                x,
                weight,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            if self.carry and name in ("self_attn.o_proj", "mlp.down_proj"):
                return ttnn.to_memory_config(out, self._working_memory(out, self.geometry.get("residual_cores", 8)))
            if self.carry and name.startswith("mlp."):
                return out
            return ttnn.to_memory_config(
                out, ttnn.L1_MEMORY_CONFIG if self.geometry.get("head_l1") else ttnn.DRAM_MEMORY_CONFIG
            )
        m = x.padded_shape[-2]
        # N workers match bank shards; M workers fit the physical worker height.
        gx, gy = self.banks, min(8, max(1, m // 32))
        pn = math.ceil(n / 32 / gx)
        sub = max(v for v in (1, 2, 4, 8) if pn % v == 0)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=sub,
            per_core_M=math.ceil(m / 32 / gy),
            per_core_N=pn,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=(
                ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
                if self.geometry.get("fused_gate") and name == "mlp.gate_proj"
                else None
            ),
        )
        return ttnn.linear(
            x,
            weight,
            program_config=pc,
            compute_kernel_config=ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
