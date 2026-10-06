# SPDX-License-Identifier: Apache-2.0
"""Evidence-only per-role projection/layout candidates; no production selector."""
import json
import math
import os

import torch

import ttnn

from .optimized_candidate_base import DRAM
from .optimized_candidate_base import CandidateBase as OptimizedDecoder


class ProjectionCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.options = json.loads(os.environ.get("OPT_PROJECTIONS", "{}"))
        prefix = f"model.layers.{self.layer_idx}."
        w = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        values = {
            "qkv": torch.cat([w["self_attn." + k + ".weight"].T for k in ["q_proj", "k_proj", "v_proj"]], -1),
            "o_proj": w["self_attn.o_proj.weight"].T,
            "gate_up": torch.cat([w["mlp.shared_experts." + k + ".weight"].T for k in ["gate_proj", "up_proj"]], -1),
            "down_proj": w["mlp.shared_experts.down_proj.weight"].T,
        }
        self.role_info = {}
        dram_grid = self.device.dram_grid_size()
        banks = dram_grid.x * dram_grid.y
        dg = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))})
        for role, value in values.items():
            opts = self.options.get(role, {})
            target = self.projections if role in ("qkv", "o_proj") else self.shared
            weight = ttnn.from_torch(
                value.contiguous(),
                device=self.device,
                dtype=getattr(ttnn, opts.get("dtype", "bfloat16")),
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
            )
            target[role] = weight
            compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, opts.get("fidelity", "HiFi4")),
                math_approx_mode=False,
                fp32_dest_acc_en=opts.get("fp32", True),
                packer_l1_acc=True,
            )
            info = dict(role=role, compute=compute, options=opts)
            if opts.get("dram", False):
                k, n = value.shape
                readers = opts.get("readers", 1)
                shard_n = math.ceil(n / 32 / banks / readers) * 32 * readers
                physical_n = shard_n * banks
                padded_value = torch.nn.functional.pad(value, (0, physical_n - n))
                info["logical_n"] = n
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dg, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
                )
                info["weight"] = ttnn.from_torch(
                    padded_value.contiguous(),
                    device=self.device,
                    dtype=weight.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                )
                cores = opts.get("cores", 5 if k == 2560 else 8 if k == 6144 else 4)
                assert k % (32 * cores) == 0
                grid = ttnn.num_cores_to_corerangeset(
                    cores, self.device.compute_with_storage_grid_size(), row_wise=True
                )
                info["input_mem"] = ttnn.create_sharded_memory_config(
                    (32, k // cores), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                )
                info["program"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=opts.get("kblock", k // 32 // cores),
                    per_core_M=1,
                    per_core_N=math.ceil(physical_n / 32 / cores),
                    num_workers_per_dram_bank=opts.get("readers", 1),
                )
            self.role_info[id(weight)] = info
        return self

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        info = self.role_info.get(id(w))
        if info is None:
            return super()._linear(x, w, dtype=dtype, activation=activation)
        if "weight" in info and x.shape[-2] <= 32:
            a = ttnn.to_memory_config(x, info["input_mem"])
            result = ttnn.linear(
                a,
                info["weight"],
                dtype=dtype,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=info["program"],
                compute_kernel_config=info["compute"],
            )
            # Compatibility control: subsequent experiments carry this output through consumers.
            result = ttnn.to_memory_config(result, DRAM)
            if result.shape[-1] != info["logical_n"]:
                result = ttnn.slice(result, (0,) * len(result.shape), (*tuple(result.shape)[:-1], info["logical_n"]))
            return result
        return ttnn.linear(
            x,
            w,
            activation=activation,
            dtype=dtype,
            memory_config=DRAM,
            compute_kernel_config=info["compute"],
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
