# SPDX-License-Identifier: Apache-2.0
"""TtMoGe — MoGe-2 on Blackhole (faithful baseline, no optimization).

Baseline split: the DINOv2 ViT-L encoder (24 transformer blocks — the dominant
compute) runs on device via TtMoGeEncoder; the convolutional decoder (neck +
points/normal/mask heads + scale MLP + output remap) runs on host using the
exact torch reference modules. This yields a correct end-to-end geometry output
with the heavy transformer on the NPU. Optimization iterations move the decoder
on-device.

API mirrors MoGeModel.forward:
    TtMoGe(ref_moge_model, device)(image, num_tokens) -> dict(points, normal, mask, metric_scale)
"""
import torch
import torch.nn.functional as F

from models.experimental.moge2.tt.ttnn_moge_encoder import TtMoGeEncoder

# vendored reference geometry helper (added to sys.path by the caller/harness)
from moge.utils.geometry_torch import normalized_view_plane_uv


class TtMoGe:
    def __init__(self, ref_moge_model, device):
        self.ref = ref_moge_model
        self.device = device
        self.encoder = TtMoGeEncoder(ref_moge_model, device)

    @torch.inference_mode()
    def __call__(self, image, num_tokens):
        ref = self.ref
        B, _, img_h, img_w = image.shape
        dtype = image.dtype
        aspect_ratio = img_w / img_h
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        # ---- encoder (device transformer) ----
        x, cls_token = self.encoder(image, base_h, base_w)

        # ---- decoder tail on host (faithful MoGeModel.forward) ----
        features = [x, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level,
                                          aspect_ratio=aspect_ratio, dtype=dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
            features[level] = uv if features[level] is None else torch.cat([features[level], uv], dim=1)

        features = ref.neck(features)

        points, normal, mask = (
            getattr(ref, head)(features)[-1] if hasattr(ref, head) else None
            for head in ["points_head", "normal_head", "mask_head"]
        )
        metric_scale = ref.scale_head(cls_token) if hasattr(ref, "scale_head") else None

        points, normal, mask = (
            F.interpolate(v, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False)
            if v is not None else None for v in [points, normal, mask]
        )

        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = ref._remap_points(points)
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        out = {"points": points, "normal": normal, "mask": mask, "metric_scale": metric_scale}
        return {k: v for k, v in out.items() if v is not None}
