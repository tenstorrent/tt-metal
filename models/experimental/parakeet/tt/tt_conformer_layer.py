import ttnn


class TtConformerLayer:
    """TTNN Conformer layer (inference). Mirrors NeMo’s ConformerLayer forward order."""

    def __init__(self, device, d_model, ffn, conv, mha):
        self.device = device
        self.d_model = d_model
        self.ffn1 = ffn
        self.conv = conv
        self.mha = mha
        self.ffn2 = ffn  # reuse instance; weights differ per layer via parameters

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _layer_norm(self, x, weight, bias, eps=1e-5):
        weight_tt = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
        bias_tt = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        return ttnn.layer_norm(x, weight=weight_tt, bias=bias_tt, epsilon=eps)

    def forward(self, x, pos_emb, att_mask=None, pad_mask=None, parameters=None):
        residual = x

        # FFN1 + 0.5 residual
        x = self._layer_norm(x, parameters.ffn1.layer_norm_weight, parameters.ffn1.layer_norm_bias)
        x = self.ffn1(x, parameters.ffn1)
        residual = ttnn.add(residual, ttnn.multiply(x, 0.5))

        # Self-attention (rel-pos)
        x = self._layer_norm(residual, parameters.self_attn.layer_norm_weight, parameters.self_attn.layer_norm_bias)
        x = self.mha.forward(x, x, x, mask=att_mask, pos_emb=pos_emb)
        residual = ttnn.add(residual, x)

        # Convolution
        x = self._layer_norm(residual, parameters.conv.layer_norm_weight, parameters.conv.layer_norm_bias)
        x = self.conv(x, pad_mask=pad_mask, parameters=parameters.conv)
        residual = ttnn.add(residual, x)

        # FFN2 + 0.5 residual
        x = self._layer_norm(residual, parameters.ffn2.layer_norm_weight, parameters.ffn2.layer_norm_bias)
        x = self.ffn2(x, parameters.ffn2)
        residual = ttnn.add(residual, ttnn.multiply(x, 0.5))

        # Final LayerNorm
        out = self._layer_norm(residual, parameters.final_layer_norm_weight, parameters.final_layer_norm_bias)
        return out
