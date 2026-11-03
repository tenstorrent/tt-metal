# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the explicit multi-core DRAM pipeline on each projection role."""
import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _linear(self, x, name, activation=None):
        if name == "attention_packed" and self.options.get("split_attention"):
            return ttnn.concat([self._linear(x, part) for part in self.attention_names], dim=-1)
        w = self.weights[name + ".weight"]
        k, n = tuple(w.shape)[-2:]
        compute = self.matmul_compute[self._weight_group(name)]
        if self.options.get("output_fidelity") and name == "self_attn.o_proj":
            compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, self.options["output_fidelity"]),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        if name == "mlp.gate" and self.options.get("router_interleaved", True):
            return ttnn.linear(
                x,
                w,
                compute_kernel_config=self.compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG if x.shape[-2] == 32 else ttnn.DRAM_MEMORY_CONFIG,
            )
        if x.shape[-2] == 32:
            cores = self.options.get(self._role(name) + "_cores", self.options.get("dense_cores", 8))
            grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, k // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )
            x = ttnn.to_memory_config(x, mem)
            role = self._role(name)
            block = self.options.get("dram_" + role + "_block", k // 32 // cores)
            prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                use_multicore_pipeline=role in self.options.get("multicore_roles", []),
                in0_block_w=block,
                per_core_M=1,
                per_core_N=math.ceil(self.decode_weights[name + ".weight"].shape[-1] / 32 / cores),
                num_workers_per_dram_bank=self.options.get(role + "_readers", self.options.get("readers", 1)),
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )
            out = ttnn.linear(
                x,
                self.decode_weights[name + ".weight"],
                program_config=prog,
                compute_kernel_config=compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            if (
                self.options.get("keep_sharded")
                and out.shape[-1] == n
                and (
                    name == "self_attn.o_proj"
                    or (self.options.get("keep_down_sharded", False) and name.endswith("down_proj"))
                    or (not self.sparse and name.startswith("mlp.") and not self.options.get("packed_dense"))
                )
            ):
                return out
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            return out[..., :n] if out.shape[-1] != n else out
        if self.options.get("prefill_2d"):
            grid = self.options.get("prefill_grid", [8, 8])
            if x.shape[-2] >= 2048:
                grid = self.options.get("prefill_grid_large", grid)
            m = math.ceil(x.shape[-2] / 32 / grid[1])
            nt = math.ceil(n / 32 / grid[0])
            sub = next(v for v in range(8, 0, -1) if nt % v == 0)
            prog = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=tuple(grid),
                in0_block_w=self.options.get("prefill_block", 8),
                out_subblock_h=1,
                out_subblock_w=sub,
                per_core_M=m,
                per_core_N=nt,
                transpose_mcast=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )
            return ttnn.linear(
                x,
                w,
                program_config=prog,
                compute_kernel_config=compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.linear(
            x,
            w,
            compute_kernel_config=compute,
            dtype=ttnn.bfloat16,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
