# SPDX-License-Identifier: Apache-2.0
"""TTNN port of RF-DETR's windowed DINOv2-S/14 backbone.

Embeddings (patch conv + cls + interpolated pos-embed + window partition) and the
final feature-map shaping (layernorm + drop-cls + window unpartition) run on host
for the baseline — they are cheap, reshape-heavy, and not tile-friendly. The 12
transformer layers (the bulk of the FLOPs) run on device.

Windowing: 560/14 = 40 patch grid, num_windows=4 -> 16 windows of 10x10=100 patches.
Windowed layers operate on [16, 101, 384] (1 cls + 100 patches per window).
Global layers (2,5,8,11) attend over the merged [1, 1616, 384] sequence.
"""

import torch
import ttnn


def _lin(linear, device):
    """torch nn.Linear -> (ttnn weight [in,out], ttnn bias [1,out] or None)."""
    w = ttnn.from_torch(
        linear.weight.detach().t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = None
    if linear.bias is not None:
        b = ttnn.from_torch(
            linear.bias.detach().reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    return w, b


def _vec(t, device):
    return ttnn.from_torch(t.detach().reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class TtDinoBackbone:
    def __init__(self, ref_model, device):
        self.device = device
        wb = ref_model.backbone[0].encoder.encoder  # WindowedDinoBackbone
        self.wb = wb  # kept for host-side embeddings + feature shaping
        self.cfg = wb.cfg
        self.num_heads = self.cfg.num_attention_heads
        self.head_dim = self.cfg.hidden_size // self.num_heads
        self.eps = self.cfg.layer_norm_eps
        self.num_windows = self.cfg.num_windows
        self.nw2 = self.num_windows ** 2

        self.layers = []
        for layer in wb.encoder.layer:
            att = layer.attention.attention
            qkv_w = torch.cat([att.query.weight, att.key.weight, att.value.weight], dim=0)  # [3*384,384]
            qkv_b = torch.cat([att.query.bias, att.key.bias, att.value.bias], dim=0)
            qkv_w_tt = ttnn.from_torch(
                qkv_w.detach().t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            qkv_b_tt = ttnn.from_torch(
                qkv_b.detach().reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            proj_w, proj_b = _lin(layer.attention.output.dense, device)
            fc1_w, fc1_b = _lin(layer.mlp.fc1, device)
            fc2_w, fc2_b = _lin(layer.mlp.fc2, device)
            self.layers.append(
                {
                    "global": layer.global_attention,
                    "norm1_w": _vec(layer.norm1.weight, device),
                    "norm1_b": _vec(layer.norm1.bias, device),
                    "qkv_w": qkv_w_tt,
                    "qkv_b": qkv_b_tt,
                    "proj_w": proj_w,
                    "proj_b": proj_b,
                    "ls1": _vec(layer.layer_scale1.lambda1, device),
                    "norm2_w": _vec(layer.norm2.weight, device),
                    "norm2_b": _vec(layer.norm2.bias, device),
                    "fc1_w": fc1_w,
                    "fc1_b": fc1_b,
                    "fc2_w": fc2_w,
                    "fc2_b": fc2_b,
                    "ls2": _vec(layer.layer_scale2.lambda1, device),
                }
            )

    def _layer(self, x, p):
        # ---- attention (norm1 -> MHA -> layerscale1 -> residual) ----
        residual = x
        if p["global"]:
            b, s, c = x.shape
            x = ttnn.reshape(x, (b // self.nw2, self.nw2 * s, c))
        normed = ttnn.layer_norm(x, weight=p["norm1_w"], bias=p["norm1_b"], epsilon=self.eps)
        qkv = ttnn.linear(normed, p["qkv_w"], bias=p["qkv_b"])
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, transpose_key=True
        )
        scores = ttnn.matmul(q, k)
        scores = ttnn.multiply(scores, self.head_dim ** -0.5)
        probs = ttnn.softmax(scores, dim=-1)
        ctx = ttnn.matmul(probs, v)  # [b,h,s,d]
        ctx = ttnn.transformer.concatenate_heads(ctx)  # [b,s,384]
        ttnn.deallocate(qkv)
        attn_out = ttnn.linear(ctx, p["proj_w"], bias=p["proj_b"])
        if p["global"]:
            attn_out = ttnn.reshape(attn_out, (residual.shape[0], residual.shape[1], residual.shape[2]))
        attn_out = ttnn.multiply(attn_out, p["ls1"])
        x = ttnn.add(attn_out, residual)

        # ---- mlp (norm2 -> fc1 -> gelu -> fc2 -> layerscale2 -> residual) ----
        residual = x
        h = ttnn.layer_norm(x, weight=p["norm2_w"], bias=p["norm2_b"], epsilon=self.eps)
        h = ttnn.linear(h, p["fc1_w"], bias=p["fc1_b"])
        h = ttnn.gelu(h, fast_and_approximate_mode=False)
        h = ttnn.linear(h, p["fc2_w"], bias=p["fc2_b"])
        h = ttnn.multiply(h, p["ls2"])
        x = ttnn.add(h, residual)
        return x

    def run_layers(self, embed_windowed_torch):
        """embed_windowed_torch: [16, 101, 384]. Returns dict idx->ttnn hidden after layers 1,4,7,10."""
        x = ttnn.from_torch(
            embed_windowed_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        out = {}
        for i, p in enumerate(self.layers):
            x = self._layer(x, p)
            if i in (1, 4, 7, 10):
                out[i] = ttnn.to_torch(x).float()
        return out

    def feature_maps(self, pixel_values):
        """Full backbone: host embeddings -> device layers -> host feature shaping.
        Returns list of 4 torch tensors [1,384,40,40]."""
        embed = self.wb.embeddings(pixel_values)  # [16,101,384] host
        hidden = self.run_layers(embed)
        _, _, H, W = pixel_values.shape
        feats = []
        for i in (1, 4, 7, 10):
            hs = torch.from_numpy(hidden[i].numpy()) if not torch.is_tensor(hidden[i]) else hidden[i]
            hs = self.wb.layernorm(hs)
            hs = hs[:, 1:]  # drop cls per window
            hs = self.wb.window_unpartition(hs, H, W)
            hs = hs.reshape(pixel_values.shape[0], H // self.cfg.patch_size, W // self.cfg.patch_size, -1)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            feats.append(hs)
        return feats
