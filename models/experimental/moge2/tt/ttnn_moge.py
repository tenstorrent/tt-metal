# SPDX-License-Identifier: Apache-2.0
"""TtMoGe — MoGe-2 on Blackhole.

On device: DINOv2 ViT-L encoder (TtMoGeEncoder) + the full ConvStack decoder
(neck + points/normal/mask heads via TtConvStack). On host: token prep,
encoder output-projection+sum, UV-coordinate maps, scale_head MLP, the final
bilinear resize to image resolution, and the output remap.

Both device regions (encoder transformer, conv decoder) are captured as metal
traces on the first call and replayed afterwards, collapsing per-op dispatch
overhead. Set trace=False for the eager path. API mirrors MoGeModel.forward.
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
    def __init__(self, ref_moge_model, device, trace=True):
        self.ref = ref_moge_model
        self.device = device
        self.trace = trace
        self.encoder = TtMoGeEncoder(ref_moge_model, device)
        self.cc = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True)
        self.neck = TtConvStack(ref_moge_model.neck, device, self.cc)
        self.heads = {}
        for h in ["points_head", "normal_head", "mask_head"]:
            if hasattr(ref_moge_model, h):
                self.heads[h] = TtConvStack(getattr(ref_moge_model, h), device, self.cc)
        self._uv_cache = {}
        # decoder trace state (encoder stays eager — rf-detr pattern: trace the
        # dispatch-bound conv tail, keep the matmul backbone eager)
        self._dec_tid = self._dec_in0 = self._dec_out = None
        self._dec_hw = None

    def _uv_features(self, base_h, base_w, aspect_ratio, dtype):
        key = (base_h, base_w)
        if key not in self._uv_cache:
            cached = [None, None, None, None, None]
            for level in range(1, 5):
                uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level,
                                              aspect_ratio=aspect_ratio, dtype=dtype, device="cpu")
                cached[level] = _to_cl(uv.permute(2, 0, 1).unsqueeze(0), self.device)
            self._uv_cache[key] = cached
        return self._uv_cache[key]

    # ---- device regions (closures over cached/persistent inputs) ----
    def _decoder_region(self, in_feats):
        neck_out = self.neck(in_feats)
        return [self.heads[n](neck_out)[-1] for n in self.heads]   # list of (tt, H, W)

    @torch.inference_mode()
    def __call__(self, image, num_tokens):
        ref = self.ref
        B, _, img_h, img_w = image.shape
        dtype = image.dtype
        aspect_ratio = img_w / img_h
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        # ===== encoder device region (eager) =====
        tokens = self.encoder.to_tokens(image, base_h, base_w)
        tt_x = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        normed = [ttnn.to_torch(t).float() for t in self.encoder.device_region(tt_x)]
        x, cls_token = self.encoder.project_sum(normed, base_h, base_w)

        # ===== decoder device region =====
        uv0 = normalized_view_plane_uv(width=base_w, height=base_h, aspect_ratio=aspect_ratio,
                                       dtype=dtype, device=x.device).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        level0_nchw = torch.cat([x, uv0], dim=1)
        uv_cached = self._uv_features(base_h, base_w, aspect_ratio, dtype)

        if not self.trace:
            in_feats = [_to_cl(level0_nchw, self.device)] + uv_cached[1:]
            outs = self._decoder_region(in_feats)
            head_t = {n: _cl_to_nchw(t, H, W) for n, (t, H, W) in zip(self.heads, outs)}
        else:
            l0 = level0_nchw.permute(0, 2, 3, 1).reshape(1, 1, base_h * base_w, -1).contiguous()
            host_l0 = ttnn.from_torch(l0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if self._dec_tid is None:
                self._dec_in0 = ttnn.to_device(host_l0, self.device)
                in_feats = [(self._dec_in0, base_h, base_w)] + uv_cached[1:]
                _ = self._decoder_region(in_feats)                    # compile
                self._dec_tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                out = self._decoder_region(in_feats)
                ttnn.end_trace_capture(self.device, self._dec_tid, cq_id=0)
                self._dec_out = [t for (t, _h, _w) in out]
                self._dec_hw = [(h, w) for (_t, h, w) in out]
            else:
                ttnn.copy_host_to_device_tensor(host_l0, self._dec_in0)
            ttnn.execute_trace(self.device, self._dec_tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            head_t = {n: _cl_to_nchw(t, h, w) for n, t, (h, w) in zip(self.heads, self._dec_out, self._dec_hw)}

        points = head_t.get("points_head")
        normal = head_t.get("normal_head")
        mask = head_t.get("mask_head")
        metric_scale = ref.scale_head(cls_token) if hasattr(ref, "scale_head") else None

        # ===== resize + remap on host (faithful) =====
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
