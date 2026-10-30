"""Remove dead projection temporaries before the down projection."""

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_attention_candidate import AttentionCandidate


class LifetimeCandidate(AttentionCandidate):
    def _finish(self, x, attention, *, decode):
        if not decode:
            if self.head_dim == 512:
                # A single 32-head concatenation needs >2MiB L1 at D512.
                # Two 16-head groups stay within the hardware's per-core L1.
                attention = ttnn.concat(
                    [
                        ttnn.transformer.concatenate_heads(
                            ttnn.slice(
                                attention,
                                (0, start, 0, 0),
                                (attention.shape[0], start + 16, attention.shape[2], self.head_dim),
                            )
                        )
                        for start in (0, 16)
                    ],
                    dim=-1,
                )
            else:
                attention = ttnn.transformer.concatenate_heads(attention)
        attention = ttnn.reshape(attention, (*tuple(x.shape)[:-1], self.nq * self.head_dim))
        add = ttnn.add_ if not decode and self.geometry.get("prefill_l1_inplace") else ttnn.add
        x = add(x, self._norm(self._linear(attention, "self_attn.o_proj"), "post_attention_layernorm"))
        z = self._norm(x, "pre_feedforward_layernorm")
        if self.geometry.get("mlp_act8"):
            z = ttnn.typecast(z, ttnn.bfloat8_b)
        if self.decoding and x.shape[-2] > 32:
            z = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
        elif self.decoding:
            role = "mlp.gate_up_proj" if self.pack_mlp else "mlp.gate_proj"
            count = self.geometry.get(role, self.geometry.get("default", {})).get("cores", 8)
            z = ttnn.to_memory_config(z, self._working_memory(z, count))
        gate, up = self._linear(z, "mlp.gate_proj"), self._linear(z, "mlp.up_proj")
        multiply = ttnn.multiply_ if not decode and self.geometry.get("prefill_l1_inplace") else ttnn.multiply
        intermediate = multiply(
            gate, up, input_tensor_a_activations=[] if self.geometry.get("fused_gate") else [ttnn.UnaryOpType.GELU_TANH]
        )
        del gate, up, z
        if self.geometry.get("mlp_act8"):
            intermediate = ttnn.typecast(intermediate, ttnn.bfloat8_b)
        mlp = self._linear(intermediate, "mlp.down_proj")
        return ttnn.add(
            x,
            self._norm(mlp, "post_feedforward_layernorm"),
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.scalar)],
        )
