# SPDX-License-Identifier: Apache-2.0
"""Sharded residual and projection-consumer compatibility experiments."""
import json
import os

import ttnn

from ..tt.optimized_decoder import DRAM
from .optimized_projection_candidates import ProjectionCandidate


class LayoutCandidate(ProjectionCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.layout_options = json.loads(os.environ.get("OPT_LAYOUT", "{}"))
        gx, gy = self.layout_options.get("grid", [5, 2])
        width = 2560 // (gx * gy)
        assert width % 32 == 0
        self.residual_mem = ttnn.create_sharded_memory_config(
            (32, width), ttnn.CoreGrid(x=gx, y=gy), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        bw = width // 32
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=max(s for s in [1, 2, 4] if bw % s == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )
        return self

    def _norm(self, x, name):
        if self.layout_options.get("norm", True) and x.shape[-1] == 2560 and x.shape[-2] <= 32:
            x = ttnn.to_memory_config(x, self.residual_mem)
            normalized = ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                program_config=self.norm_program,
                compute_kernel_config=self.compute,
                memory_config=self.residual_mem,
            )
            return ttnn.multiply(normalized, self.norms[name], memory_config=self.residual_mem)
        if x.is_sharded():
            x = ttnn.to_memory_config(x, DRAM)
        return super()._norm(x, name)

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        info = self.role_info.get(id(w))
        if (
            self.layout_options.get("carry_projection", True)
            and info
            and info["role"] in self.layout_options.get("carry_roles", ["qkv", "o_proj", "gate_up", "down_proj"])
            and "weight" in info
            and x.shape[-2] <= 32
        ):
            a = ttnn.to_memory_config(x, info["input_mem"])
            out = ttnn.linear(
                a,
                info["weight"],
                dtype=dtype,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=info["program"],
                compute_kernel_config=info["compute"],
            )
            if out.shape[-1] != info["logical_n"]:
                out = ttnn.slice(out, (0,) * len(out.shape), (*tuple(out.shape)[:-1], info["logical_n"]))
            return out
        return super()._linear(x, w, dtype=dtype, activation=activation)

    def _moe(self, x, *, prefill=False):
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return super()._moe(x, prefill=prefill)

    def _finish(self, residual, attention, *, prefill=False):
        if not prefill and self.layout_options.get("norm", True):
            residual = ttnn.to_memory_config(residual, self.residual_mem)
            x = ttnn.add(residual, self._norm(attention, "post_attn_norm"), memory_config=self.residual_mem)
            moe = self._moe(self._norm(x, "post_attention_layernorm"))
            return ttnn.add(x, self._norm(moe, "post_ffn_norm"), memory_config=self.residual_mem)
        return super()._finish(residual, attention, prefill=prefill)
