# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Norm output placement and explicit fused-SwiGLU prefill configuration trials."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def _norm(self, x, name):
        if x.shape[-2] != 32 or not self.options.get("sharded_norm"):
            return self._norm_base(x, name)
        # Grouping moves each token's two1280-wide groups into separate rows.
        shape = tuple(x.shape)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if not self.options.get("bf16_norm"):
            x = ttnn.typecast(x, ttnn.float32)
        grouped = ttnn.reshape(x, (1, 1, 64, 1280))
        cores = self.options.get("norm_cores", 8)
        grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [64, 1280 // cores], ttnn.ShardOrientation.ROW_MAJOR),
        )
        grouped = ttnn.to_memory_config(grouped, mem)
        width = 40 // cores
        sub = next(v for v in range(min(4, width), 0, -1) if width % v == 0)
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            block_h=2,
            block_w=width,
            subblock_w=sub,
            inplace=False,
        )
        y = ttnn.rms_norm(
            grouped,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
            program_config=prog,
            memory_config=mem,
        )
        y = ttnn.reshape(ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG), shape)
        return ttnn.mul(
            y,
            self.weights[name + ".weight"],
            dtype=getattr(
                ttnn, self.options["attention_activation" if name == "input_layernorm" else "mlp_activation"]
            ),
            memory_config=self._residual_memory() if self.options.get("norm_output_sharded") else ttnn.L1_MEMORY_CONFIG,
        )

    def _mlp_base(self, x, *, valid_tokens=None):
        prefix = "mlp.shared_experts" if self.sparse else "mlp"
        router_logits = None
        if x.shape[-2] > 32 and not self.sparse:
            hidden = ttnn.experimental.minimal_matmul(
                x,
                self.weights[prefix + ".swiglu.weight"],
                fuse_swiglu=True,
                config=(
                    ttnn.MinimalMatmulConfig(
                        **self.options["minimal_config"],
                        compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    )
                    if self.options.get("minimal_config")
                    else None
                ),
                compute_kernel_config=self.matmul_compute["mlp"],
            )
        elif self.sparse:
            gu = self._linear(x, "mlp.shared_router")
            router_logits = gu[..., 1536:]
            hidden = ttnn.mul(gu[..., :768], gu[..., 768:1536], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        else:
            gate = ttnn.linear(
                x,
                self.weights[prefix + ".gate_proj.weight"],
                activation="silu",
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.matmul_compute["mlp"],
            )
            hidden = ttnn.mul(gate, self._linear(x, prefix + ".up_proj"))
        dense = self._linear(hidden, prefix + ".down_proj")
        return (
            ttnn.add(self._experts(x, valid_tokens=valid_tokens, logits=router_logits), dense) if self.sparse else dense
        )

    def _experts(self, x, **kwargs):
        return super()._experts(
            (
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
                if self.options.get("norm_output_sharded") and not self.options.get("expert_keep_sharded")
                else x
            ),
            **kwargs,
        )
