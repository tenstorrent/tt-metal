# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct L1 prefill producers under the final TP/EP topology."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _norm(self, x, name):
        if x.shape[-2] == 32 or not self.options["prefill_group_norm"]:
            return super()._norm(x, name)
        # Each half is one complete HF RMS group. Native RMSNorm fuses the
        # corresponding learned scale; the concat restores channel order.
        groups = []
        for i in range(2):
            group = x[..., i * 1280 : (i + 1) * 1280]
            if group.dtype != getattr(ttnn, self.options["prefill_norm_dtype"]):
                group = ttnn.typecast(group, getattr(ttnn, self.options["prefill_norm_dtype"]))
            groups.append(
                ttnn.rms_norm(
                    group,
                    weight=self.group_norm_weights[name][i],
                    epsilon=self.config.rms_norm_eps,
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            )
        return ttnn.typecast(
            ttnn.concat(groups, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG),
            getattr(ttnn, self.options["attention_activation" if name == "input_layernorm" else "mlp_activation"]),
        )

    def _mlp(self, x, *, valid_tokens=None):
        if x.shape[-2] == 32:
            return super()._mlp(x, valid_tokens=valid_tokens)
        if self.sparse:
            gu = self._linear(x, "mlp.shared_router")
            hidden = ttnn.mul(
                gu[..., :256],
                gu[..., 256:512],
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            local = ttnn.add(
                self._linear(hidden, "mlp.shared_experts.down_proj"),
                self._experts(x, logits=gu[..., 512:], valid_tokens=valid_tokens),
            )
        elif x.shape[-2] > 32:
            hidden = ttnn.experimental.minimal_matmul(
                x,
                self.weights["mlp.swiglu.weight"],
                fuse_swiglu=True,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=16,
                    N_block_size=8,
                    subblock_h=2,
                    subblock_w=4,
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                ),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.matmul_compute["mlp"],
            )
            local = self._linear(hidden, "mlp.down_proj")
        else:
            gu = self._linear(x, "mlp.gate_up")
            hidden = ttnn.mul(
                gu[..., :1536],
                gu[..., 1536:],
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            local = self._linear(hidden, "mlp.down_proj")
        return self._reduce(local, slot=1)

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        if attention.shape[-2] == 32:
            return super()._finish(residual, gate_logits, attention, valid_tokens=valid_tokens)
        gated = ttnn.mul(
            attention,
            gate_logits,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
        )
        projected = self._reduce(self._linear(gated, "self_attn.o_proj"))
        x = ttnn.add(residual, projected)
        normed = self._norm(self._gather(x) if self.residual_sharded else x, "post_attention_layernorm")
        return ttnn.add(x, self._mlp(normed, valid_tokens=valid_tokens))
