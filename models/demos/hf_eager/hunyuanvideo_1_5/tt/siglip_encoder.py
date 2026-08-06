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

# Set by the Hunyuan mesh fixture when a genuinely disjoint 1x1 chip is
# available. Never carve this from the resident DiT/VAE/Qwen mesh: overlapping
# mesh contexts can deadlock even though the stages execute sequentially.
HY_SIGLIP_SUBMESH = None


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


def _mesh_size(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


def _build_blocks_replicated(cfg, w, device, SigLIPBlockTTNN):
    """Build the pi0 SigLIP transformer blocks REPLICATED across the full mesh, shared
    with the resident DiT (the way the VAE-decode / text-encode adapters run) rather than
    carving a co-resident submesh -- an overlapping `create_submesh(1,1)` deadlocks against
    the resident DiT. The pi0 blocks' __init__ does single-device `from_torch`/`to_torch`
    weight-prep round-trips, so during construction we wrap those two ops to (a) replicate
    every weight upload across the mesh and (b) read weights back from device 0 (all replicas
    are identical). The blocks' forward already uses mesh-safe ops (layer_norm / matmul /
    SDPA / nlp_*_heads)."""
    n = _mesh_size(device)
    num_layers = cfg.num_hidden_layers
    if n <= 1:
        return [SigLIPBlockTTNN(cfg, _layer_weights(w, i), device) for i in range(num_layers)]

    orig_from, orig_to = ttnn.from_torch, ttnn.to_torch

    def _from(t, *a, **k):
        if k.get("device", None) is device and "mesh_mapper" not in k:
            k["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        return orig_from(t, *a, **k)

    def _to(t, *a, **k):
        if isinstance(t, ttnn.Tensor):
            try:
                k.pop("mesh_composer", None)
                return orig_to(ttnn.get_device_tensors(t)[0], *a, **k)
            except Exception:
                pass
        return orig_to(t, *a, **k)

    ttnn.from_torch, ttnn.to_torch = _from, _to
    try:
        return [SigLIPBlockTTNN(cfg, _layer_weights(w, i), device) for i in range(num_layers)]
    finally:
        ttnn.from_torch, ttnn.to_torch = orig_from, orig_to


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

        self.__dict__["_real"] = real
        self.__dict__["_device"] = device
        self.__dict__["_n"] = _mesh_size(device)
        # transformers 5.x flattened SiglipVisionModel; embeddings/post_layernorm are direct.
        self.__dict__["_vm"] = getattr(real, "vision_model", real)
        self.__dict__["_hidden"] = c.hidden_size
        self.__dict__["_num_patches"] = (c.image_size // c.patch_size) ** 2
        self.__dict__["_blocks"] = _build_blocks_replicated(cfg, w, device, SigLIPBlockTTNN)
        self.__dict__["_verify"] = verify_pcc
        self.__dict__["_thr"] = pcc_threshold
        self.__dict__["_checked"] = False

    def __getattr__(self, k):
        return getattr(self.__dict__["_real"], k)

    def _run_tt(self, pixel_values):
        vm = self.__dict__["_vm"]
        dev = self.__dict__["_device"]
        n = self.__dict__["_n"]
        # Match the host embed/LN module dtype (bf16 in the real pipeline, fp32 in the PCC
        # test) at the CPU boundaries; the on-device blocks run bf16 regardless.
        emb_dtype = next(vm.embeddings.parameters()).dtype
        ln_dtype = next(vm.post_layernorm.parameters()).dtype
        with torch.no_grad():
            hs = vm.embeddings(pixel_values.to(emb_dtype))  # host patch + position embed -> (B, P, C)
        # Replicate the activation across the mesh, run the 27 blocks (replicated -> every chip
        # computes the same result alongside its DiT shard), then read one replica back.
        mapper = ttnn.ReplicateTensorToMesh(dev) if n > 1 else None
        hs_tt = ttnn.from_torch(
            hs.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        for blk in self.__dict__["_blocks"]:
            hs_tt = blk.forward(hs_tt)
        B = pixel_values.shape[0]
        if n > 1:
            out = ttnn.to_torch(hs_tt, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:B]
        else:
            out = ttnn.to_torch(hs_tt)
        out = out.reshape(B, self.__dict__["_num_patches"], self.__dict__["_hidden"])
        with torch.no_grad():
            out = vm.post_layernorm(out.to(ln_dtype))  # host post-LN -> last_hidden_state
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
