# SPDX-License-Identifier: Apache-2.0
"""ttnn DINOv2 ViT-L/14 encoder for MoGe-2 (faithful baseline, no optimization).

The 24 transformer blocks (the dominant compute) run on device. Token
preparation (patch embed + cls + interpolated pos-embed) and the final
output-projection + sum reuse the exact torch reference modules on host, so the
encoder output matches `DINOv2Encoder.forward` bit-for-bit up to bf16 device math.

Public API mirrors the reference encoder:
    TtMoGeEncoder(ref_moge_model, device)(image, base_h, base_w) -> (x, cls_token)
        x:         torch [B, 1024, base_h, base_w]   (summed projected features)
        cls_token: torch [B, 1024]                   (normed cls of layer 23)
"""
import torch
import torch.nn.functional as F

import ttnn

# Faithful-baseline math config: high fidelity, fp32 accumulation. Optimization
# iterations may lower this (HiFi2 / bf8 / approx) and re-check PCC.
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

INTERMEDIATE_LAYERS = (5, 11, 17, 23)
NUM_HEADS = 16
HEAD_DIM = 64
EPS = 1e-6  # DINOv2 LayerNorm eps


def _lin_w(w, dtype=ttnn.bfloat16):
    return ttnn.from_torch(w.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _lin_b(b, dtype=ttnn.bfloat16):
    return ttnn.from_torch(b.reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _norm(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _gamma(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


class TtMoGeEncoder:
    def __init__(self, ref_moge_model, device, weight_dtype=ttnn.bfloat16):
        self.device = device
        self.enc = ref_moge_model.encoder            # DINOv2Encoder (host: prep + projections)
        self.backbone = self.enc.backbone
        self.intermediate_layers = list(self.enc.intermediate_layers)

        blocks = []
        for blk in self.backbone.blocks:
            a = blk.attn
            blocks.append({
                "norm1_w": _norm(blk.norm1.weight.data), "norm1_b": _norm(blk.norm1.bias.data),
                "norm2_w": _norm(blk.norm2.weight.data), "norm2_b": _norm(blk.norm2.bias.data),
                "qkv_w": _lin_w(a.qkv.weight.data, weight_dtype), "qkv_b": _lin_b(a.qkv.bias.data),
                "proj_w": _lin_w(a.proj.weight.data, weight_dtype), "proj_b": _lin_b(a.proj.bias.data),
                "ls1": _gamma(blk.ls1.gamma.data), "ls2": _gamma(blk.ls2.gamma.data),
                "fc1_w": _lin_w(blk.mlp.fc1.weight.data, weight_dtype), "fc1_b": _lin_b(blk.mlp.fc1.bias.data),
                "fc2_w": _lin_w(blk.mlp.fc2.weight.data, weight_dtype), "fc2_b": _lin_b(blk.mlp.fc2.bias.data),
            })
        self.blocks = [{k: ttnn.to_device(v, device) for k, v in b.items()} for b in blocks]
        self.final_norm_w = ttnn.to_device(_norm(self.backbone.norm.weight.data), device)
        self.final_norm_b = ttnn.to_device(_norm(self.backbone.norm.bias.data), device)

    def _attention(self, x, p):
        qkv = ttnn.linear(x, p["qkv_w"], bias=p["qkv_b"], compute_kernel_config=COMPUTE_CONFIG)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=NUM_HEADS, transpose_key=False)
        ttnn.deallocate(qkv)
        kt = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        scores = ttnn.matmul(q, kt, compute_kernel_config=COMPUTE_CONFIG)
        ttnn.deallocate(q)
        ttnn.deallocate(kt)
        scores = ttnn.multiply(scores, HEAD_DIM ** -0.5)
        scores = ttnn.softmax(scores, dim=-1)        # respects logical width (seq need not be tile-aligned)
        out = ttnn.matmul(scores, v, compute_kernel_config=COMPUTE_CONFIG)
        ttnn.deallocate(scores)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(out)
        out = ttnn.linear(out, p["proj_w"], bias=p["proj_b"], compute_kernel_config=COMPUTE_CONFIG)
        return out

    def _block(self, x, p):
        n1 = ttnn.layer_norm(x, weight=p["norm1_w"], bias=p["norm1_b"], epsilon=EPS,
                             compute_kernel_config=COMPUTE_CONFIG)
        attn = self._attention(n1, p)
        ttnn.deallocate(n1)
        attn = ttnn.multiply(attn, p["ls1"])
        x = ttnn.add(x, attn)
        ttnn.deallocate(attn)
        n2 = ttnn.layer_norm(x, weight=p["norm2_w"], bias=p["norm2_b"], epsilon=EPS,
                             compute_kernel_config=COMPUTE_CONFIG)
        h = ttnn.linear(n2, p["fc1_w"], bias=p["fc1_b"], activation="gelu", compute_kernel_config=COMPUTE_CONFIG)
        ttnn.deallocate(n2)
        h = ttnn.linear(h, p["fc2_w"], bias=p["fc2_b"], compute_kernel_config=COMPUTE_CONFIG)
        h = ttnn.multiply(h, p["ls2"])
        x = ttnn.add(x, h)
        ttnn.deallocate(h)
        return x

    @torch.inference_mode()
    def to_tokens(self, image, base_h, base_w):
        """Host token prep (faithful to DINOv2Encoder.forward) -> tokens torch [B, 1+H*W, 1024]."""
        onnx = getattr(self.enc, "onnx_compatible_mode", False)
        image_14 = F.interpolate(image, (base_h * 14, base_w * 14), mode="bilinear",
                                 align_corners=False, antialias=not onnx)
        image_14 = (image_14 - self.enc.image_mean) / self.enc.image_std
        return self.backbone.prepare_tokens_with_masks(image_14)

    def device_region(self, tt_x):
        """Pure-device transformer (traceable): tt tokens -> list of 4 normed tt tensors.
        Does NOT deallocate the returned tensors (they are trace outputs)."""
        captured = []
        for i, p in enumerate(self.blocks):
            tt_x = self._block(tt_x, p)
            if i in self.intermediate_layers:
                captured.append(ttnn.layer_norm(tt_x, weight=self.final_norm_w, bias=self.final_norm_b,
                                                epsilon=EPS, compute_kernel_config=COMPUTE_CONFIG))
        ttnn.deallocate(tt_x)
        return captured

    @torch.inference_mode()
    def project_sum(self, normed_list, base_h, base_w):
        """Host output-projection + sum (faithful) -> (x [B,1024,bh,bw], cls_token [B,1024])."""
        nreg = self.backbone.num_register_tokens
        feats = [(out[:, 1 + nreg:], out[:, 0]) for out in normed_list]
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (base_h, base_w)).contiguous())
            for proj, (feat, _cls) in zip(self.enc.output_projections, feats)
        ], dim=1).sum(dim=1)
        return x, feats[-1][1]

    @torch.inference_mode()
    def __call__(self, image, base_h, base_w):
        tokens = self.to_tokens(image, base_h, base_w)
        tt_x = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        normed = [ttnn.to_torch(t).float() for t in self.device_region(tt_x)]
        return self.project_sum(normed, base_h, base_w)
