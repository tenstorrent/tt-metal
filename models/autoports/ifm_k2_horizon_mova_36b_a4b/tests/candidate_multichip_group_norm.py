# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep each complete RMS group contiguous and fuse its learned scale."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.group_norm_weights = {
            name: [self.weights[name + ".weight"][..., i * 1280 : (i + 1) * 1280] for i in range(2)]
            for name in ("input_layernorm", "post_attention_layernorm")
        }
        return self

    def _norm(self, x, name):
        decode = x.shape[-2] == 32
        if self.options.get("group_norm_phase", "both") == ("prefill" if decode else "decode"):
            return super()._norm(x, name)
        mem = ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, mem)
        outputs = []
        for i in range(2):
            group = x[..., i * 1280 : (i + 1) * 1280]
            if not decode and not self.options.get("prefill_bf16_norm", False):
                group = ttnn.typecast(group, ttnn.float32)
            kwargs = {}
            if decode and self.options.get("group_norm_cores", 8):
                cores = self.options.get("group_norm_cores", 8)
                grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
                sharded = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, [32, 1280 // cores], ttnn.ShardOrientation.ROW_MAJOR),
                )
                group = ttnn.to_memory_config(group, sharded)
                width = 40 // cores
                kwargs = dict(
                    memory_config=sharded,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                        block_h=1,
                        block_w=width,
                        subblock_w=next(v for v in range(min(4, width), 0, -1) if width % v == 0),
                        inplace=False,
                    ),
                )
            out = ttnn.rms_norm(
                group,
                weight=self.group_norm_weights[name][i],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.compute,
                **kwargs,
            )
            outputs.append(ttnn.to_memory_config(out, mem))
        out = ttnn.concat(outputs, dim=-1, memory_config=mem)
        out = ttnn.typecast(
            out, getattr(ttnn, self.options["attention_activation" if name == "input_layernorm" else "mlp_activation"])
        )
        return ttnn.to_memory_config(out, self._residual_memory()) if decode else out
