# SPDX-License-Identifier: Apache-2.0
"""TtMoGe — MoGe-2 on Blackhole.

On device: DINOv2 ViT-L encoder (TtMoGeEncoder) + the full ConvStack decoder
(neck + points/normal/mask heads via TtConvStack). On host: token prep,
encoder output-projection+sum, UV-coordinate maps, scale_head MLP, the final
bilinear resize to image resolution, and the output remap. Faithful to
MoGeModel.forward.

API:  TtMoGe(ref_moge_model, device)(image, num_tokens) -> dict(points, normal, mask, metric_scale)
"""
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.moge2.tt.ttnn_moge_encoder import TtMoGeEncoder
from models.experimental.moge2.tt.ttnn_moge_decoder import TtConvStack, _to_cl

from moge.utils.geometry_torch import normalized_view_plane_uv


def _cl_to_nchw(t, H, W):
    return ttnn.to_torch(t).float().reshape(1, H, W, -1).permute(0, 3, 1, 2)


class TtMoGe:
    def __init__(self, ref_moge_model, device):
        self.ref = ref_moge_model
        self.device = device
        self.encoder = TtMoGeEncoder(ref_moge_model, device)
        self.cc = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True)
        self.neck = TtConvStack(ref_moge_model.neck, device, self.cc)
        self.heads = {}
        for h in ["points_head", "normal_head", "mask_head"]:
            if hasattr(ref_moge_model, h):
                self.heads[h] = TtConvStack(getattr(ref_moge_model, h), device, self.cc)
        self._uv_cache = {}   # (base_h, base_w) -> [None, uv1_cl, uv2_cl, uv3_cl, uv4_cl]

    def _uv_features(self, base_h, base_w, aspect_ratio, dtype):
        """UV coordinate maps are fixed per (geometry) — cache the uploaded device
        tensors for levels 1..4 (the expensive high-res uploads). Level 0's UV is
        concatenated with the per-image encoder feature on host (cheap)."""
        key = (base_h, base_w)
        if key not in self._uv_cache:
            cached = [None, None, None, None, None]
            for level in range(1, 5):
                uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level,
                                              aspect_ratio=aspect_ratio, dtype=dtype, device="cpu")
                uv = uv.permute(2, 0, 1).unsqueeze(0)
                cached[level] = _to_cl(uv, self.device)
            self._uv_cache[key] = cached
        return self._uv_cache[key]

    @torch.inference_mode()
    def __call__(self, image, num_tokens):
        ref = self.ref
        B, _, img_h, img_w = image.shape
        dtype = image.dtype
        aspect_ratio = img_w / img_h
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        # ---- encoder (device transformer) ----
        x, cls_token = self.encoder(image, base_h, base_w)   # x: torch [B,1024,bh,bw]

        # ---- input features: level 0 = [encoder x ; UV] (host, cheap); levels 1..4 cached UV ----
        uv0 = normalized_view_plane_uv(width=base_w, height=base_h, aspect_ratio=aspect_ratio,
                                       dtype=dtype, device=x.device).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        in_feats = [_to_cl(torch.cat([x, uv0], dim=1), self.device)]
        in_feats += self._uv_features(base_h, base_w, aspect_ratio, dtype)[1:]

        # ---- neck + heads on device ----
        neck_out = self.neck(in_feats)
        head_out = {}
        for name, head in self.heads.items():
            t, H, W = head(neck_out)[-1]
            head_out[name] = _cl_to_nchw(t, H, W)
        points = head_out.get("points_head")
        normal = head_out.get("normal_head")
        mask = head_out.get("mask_head")
        metric_scale = ref.scale_head(cls_token) if hasattr(ref, "scale_head") else None

        # ---- resize + remap on host (faithful) ----
        points, normal, mask = (
            F.interpolate(v, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False)
            if v is not None else None for v in [points, normal, mask]
        )
        if points is not None:
            points = ref._remap_points(points.permute(0, 2, 3, 1))
        if normal is not None:
            normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        out = {"points": points, "normal": normal, "mask": mask, "metric_scale": metric_scale}
        return {k: v for k, v in out.items() if v is not None}
