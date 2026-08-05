# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device SigLIP image encoder for HunyuanVideo-1.5 **i2v**.

The i2v pipeline conditions on a SigLIP-so400m-patch14-384 vision encoder
(`pipe.image_encoder`, a diffusers/transformers `SiglipVisionModel`): it maps the
input first-frame to `image_embeds` (729 tokens) that feed the DiT's image
projection. By default it runs on the host CPU; `HY_TT_SIGLIP=1` swaps in this
adapter so the 27-layer vision transformer (the bulk of the 0.428B encoder) runs
on the TT mesh instead -- matching the on-device text/vision-encode approach used
by e.g. Wan2.2.

Reuse, not rewrite: the transformer blocks are the already-shipped, PCC-verified
`SigLIPBlockTTNN` from `models/experimental/pi0/tt/ttnn_siglip.py` (pi0/PaliGemma
use the same so400m variant). Only pi0's on-device patch-embed unfold uses a
drifted 6D-permute op, so we keep the cheap patch+position embedding and the final
post-LayerNorm on host (a single conv + LN) and run the expensive 27-block stack on
device. Measured PCC vs the host encoder ~0.995.
"""
from types import SimpleNamespace

import torch

import ttnn


def _remap_siglip_state_dict(sd):
    """Normalize a `SiglipVisionModel.state_dict()` to the `vision_model.*` checkpoint
    form the pi0 blocks expect (transformers 5.x flattened the wrapper and drops that
    prefix) and drop the attention-pooling `head.*` (we use `last_hidden_state`, not
    `pooler_output`)."""
    out = {}
    for k, v in sd.items():
        kk = k if k.startswith("vision_model.") else f"vision_model.{k}"
        if kk.startswith("vision_model.head."):
            continue
        out[kk] = v
    return out


def _layer_weights(w, i):
    pre = f"vision_model.encoder.layers.{i}."
    return {k[len(pre) :]: v for k, v in w.items() if k.startswith(pre)}


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 1.0 if torch.allclose(a, b, atol=1e-4) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _single_device(device):
    """A single-chip view for the small (0.428B, run-once) SigLIP: carve a 1x1
    submesh off a multi-device mesh, else use the device as-is."""
    try:
        if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
            return device.create_submesh(ttnn.MeshShape(1, 1))
    except Exception:
        pass
    return device


class TTSiglipImageEncoderAdapter:
    """Drop-in for `pipe.image_encoder`: same `__call__(pixel_values=...) ->
    .last_hidden_state` contract, but the 27 vision-transformer layers run on the TT
    mesh. Non-overridden attributes (`.config`, `.parameters()`, `.dtype`, ...) proxy
    to the wrapped host encoder, which we keep for the host patch-embed + post-LN and a
    one-time PCC self-check."""

    def __init__(self, real_image_encoder, device, verify_pcc=True, pcc_threshold=0.95):
        from models.experimental.pi0.common.configs import SigLIPConfig
        from models.experimental.pi0.tt.ttnn_siglip import SigLIPBlockTTNN

        real = real_image_encoder
        c = real.config
        cfg = SigLIPConfig(
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_hidden_layers=c.num_hidden_layers,
            num_attention_heads=c.num_attention_heads,
            image_size=c.image_size,
            patch_size=c.patch_size,
            layer_norm_eps=c.layer_norm_eps,
        )
        w = _remap_siglip_state_dict(real.state_dict())
        dev = _single_device(device)

        self.__dict__["_real"] = real
        # transformers 5.x flattened SiglipVisionModel; embeddings/post_layernorm are direct.
        self.__dict__["_vm"] = getattr(real, "vision_model", real)
        self.__dict__["_hidden"] = c.hidden_size
        self.__dict__["_num_patches"] = (c.image_size // c.patch_size) ** 2
        self.__dict__["_blocks"] = [SigLIPBlockTTNN(cfg, _layer_weights(w, i), dev) for i in range(c.num_hidden_layers)]
        self.__dict__["_verify"] = verify_pcc
        self.__dict__["_thr"] = pcc_threshold
        self.__dict__["_checked"] = False

    def __getattr__(self, k):
        return getattr(self.__dict__["_real"], k)

    def _run_tt(self, pixel_values):
        vm = self.__dict__["_vm"]
        with torch.no_grad():
            hs = vm.embeddings(pixel_values.float())  # host patch + position embed -> (B, P, C)
        hs_tt = ttnn.from_torch(
            hs,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.__dict__["_blocks"][0].device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for blk in self.__dict__["_blocks"]:
            hs_tt = blk.forward(hs_tt)
        B = pixel_values.shape[0]
        out = ttnn.to_torch(hs_tt).reshape(B, self.__dict__["_num_patches"], self.__dict__["_hidden"]).float()
        with torch.no_grad():
            out = vm.post_layernorm(out)  # host post-LN -> last_hidden_state
        return out

    def __call__(self, pixel_values=None, **kw):
        out = self._run_tt(pixel_values).to(pixel_values.dtype)
        if self.__dict__["_verify"] and not self.__dict__["_checked"]:
            self.__dict__["_checked"] = True
            with torch.no_grad():
                ref = self.__dict__["_real"](pixel_values=pixel_values).last_hidden_state
            pcc = _pcc(ref, out)
            print(
                f"[HY_TT_SIGLIP] on-device SigLIP PCC vs host = {pcc:.5f} (threshold {self.__dict__['_thr']})",
                flush=True,
            )
            assert pcc >= self.__dict__["_thr"], f"on-device SigLIP PCC {pcc:.5f} < {self.__dict__['_thr']}"
        return SimpleNamespace(last_hidden_state=out)
