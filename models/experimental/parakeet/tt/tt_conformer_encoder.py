import ttnn
from types import SimpleNamespace
from ttnn_conformer_layer import TtConformerLayer
from tt_relposition import RelPositionMultiHeadAttentionTTNN, RelPositionalEncodingTTNN
from ttnn_conf_layer import TtConformerFeedForward, TtConformerConvolution


class TtConformerEncoder:
    """TTNN Conformer encoder (inference). Builds a list of TtConformerLayer."""

    def __init__(self, device, config):
        self.device = device
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.pos_enc = RelPositionalEncodingTTNN(
            device, d_model=config.d_model, max_len=config.max_len, return_two_values=False
        )

        # Shared submodules (weights are per-layer via parameters)
        # self.ffn = TtConformerFeedForward(device, dtype=ttnn.bfloat16)
        # self.conv = TtConformerConvolution(config.d_model, config.conv_kernel_size, device, ttnn.bfloat16)

        # One MHA per layer to match NeMo’s per-layer weights
        self.mhas = [RelPositionMultiHeadAttentionTTNN(device, config) for _ in range(config.n_layers)]

        self.layers = []

        for i in range(config.n_layers):
            ffn1 = TtConformerFeedForward(device, dtype=ttnn.bfloat16)
            conv = TtConformerConvolution(
                config.d_model,
                config.conv_kernel_size,
                device,
                ttnn.bfloat16,
            )
            mha = self.mhas[i]

            layer = TtConformerLayer(
                device,
                config.d_model,
                ffn1,
                conv,
                mha,
            )

            self.layers.append(layer)

    def prepare_weights(self, torch_model):
        """Convert NeMo ConformerEncoder weights to per-layer TTNN parameters."""
        params_list = []
        for i, torch_layer in enumerate(torch_model.layers):
            # Prepare per-layer MHA weights
            self.mhas[i].prepare_weights(
                torch_layer.self_attn.linear_q.weight,
                torch_layer.self_attn.linear_k.weight,
                torch_layer.self_attn.linear_v.weight,
                torch_layer.self_attn.linear_q.bias,
                torch_layer.self_attn.linear_k.bias,
                torch_layer.self_attn.linear_v.bias,
                torch_layer.self_attn.linear_out.weight,
                torch_layer.self_attn.linear_out.bias,
                torch_layer.self_attn.linear_pos.weight,
                torch_layer.self_attn.pos_bias_u,
                torch_layer.self_attn.pos_bias_v,
            )
            p = SimpleNamespace()
            # FFN1
            p.ffn1 = SimpleNamespace()
            p.ffn1.layer_norm_weight = ttnn.from_torch(
                torch_layer.norm_feed_forward1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.ffn1.layer_norm_bias = ttnn.from_torch(
                torch_layer.norm_feed_forward1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.ffn1.linear1 = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.feed_forward1.linear1.weight.transpose(-1, -2),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            p.ffn1.linear2 = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.feed_forward1.linear2.weight.transpose(-1, -2),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            # Self-attn norm
            p.self_attn = SimpleNamespace()
            p.self_attn.layer_norm_weight = ttnn.from_torch(
                torch_layer.norm_self_att.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.self_attn.layer_norm_bias = ttnn.from_torch(
                torch_layer.norm_self_att.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            # Conv (no transpose on conv weights to match test expectations)
            p.conv = SimpleNamespace()
            p.conv.layer_norm_weight = ttnn.from_torch(
                torch_layer.norm_conv.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.conv.layer_norm_bias = ttnn.from_torch(
                torch_layer.norm_conv.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.conv.pointwise1 = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.conv.pointwise_conv1.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            p.conv.pointwise1.bias = getattr(torch_layer.conv.pointwise_conv1, "bias", None)
            if p.conv.pointwise1.bias is not None:
                p.conv.pointwise1.bias = ttnn.from_torch(
                    p.conv.pointwise1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            p.conv.depthwise = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.conv.depthwise_conv.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            p.conv.depthwise.bias = getattr(torch_layer.conv.depthwise_conv, "bias", None)
            if p.conv.depthwise.bias is not None:
                p.conv.depthwise.bias = ttnn.from_torch(
                    p.conv.depthwise.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            p.conv.pointwise2 = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.conv.pointwise_conv2.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            p.conv.pointwise2.bias = getattr(torch_layer.conv.pointwise_conv2, "bias", None)
            if p.conv.pointwise2.bias is not None:
                p.conv.pointwise2.bias = ttnn.from_torch(
                    p.conv.pointwise2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            p.conv.bn = SimpleNamespace()
            p.conv.bn.running_mean = ttnn.from_torch(
                torch_layer.conv.batch_norm.running_mean,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            p.conv.bn.running_var = ttnn.from_torch(
                torch_layer.conv.batch_norm.running_var,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            p.conv.bn.weight = ttnn.from_torch(
                torch_layer.conv.batch_norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.conv.bn.bias = ttnn.from_torch(
                torch_layer.conv.batch_norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            # FFN2
            p.ffn2 = SimpleNamespace()
            p.ffn2.layer_norm_weight = ttnn.from_torch(
                torch_layer.norm_feed_forward2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.ffn2.layer_norm_bias = ttnn.from_torch(
                torch_layer.norm_feed_forward2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.ffn2.linear1 = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.feed_forward2.linear1.weight.transpose(-1, -2),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            p.ffn2.linear2 = SimpleNamespace(
                weight=ttnn.from_torch(
                    torch_layer.feed_forward2.linear2.weight.transpose(-1, -2),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            )
            # Final norm
            p.final_layer_norm_weight = ttnn.from_torch(
                torch_layer.norm_out.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            p.final_layer_norm_bias = ttnn.from_torch(
                torch_layer.norm_out.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

            params_list.append(p)
        return params_list

    def forward(self, x, att_mask=None, pad_mask=None, parameters_list=None):
        pos_emb = self.pos_enc(x)  # (1, 2*T-1, d_model) in TILE_LAYOUT
        for layer, params in zip(self.layers, parameters_list):
            x = layer.forward(x, pos_emb, att_mask=att_mask, pad_mask=pad_mask, parameters=params)
        return x
