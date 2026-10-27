# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF reference for the EmbeddingGemma 2 vision path (fp32, CPU): `Gemma4VisionModel` + `embed_vision`.

Needs transformers >= 5.20 (overlay on PYTHONPATH, see hf_reference.py). The modules are the
unmodified HF classes the full `EmbeddingGemma2Model` builds (`AutoModel.from_config(vision_config)`
and `EmbeddingGemma2MultimodalEmbedder`), loaded from the checkpoint's `vision_tower.*` /
`embed_vision.*` tensors and upcast to fp32. `run_vision_reference` records activations with
forward hooks; it never changes the model. The soft tokens equal
`EmbeddingGemma2Model.get_image_features(...).pooler_output` (checked by test_vision_reference_matches_full_model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from models.demos.blackhole.embedding_gemma2.reference.hf_reference import _check_transformers


def load_reference_vision(snapshot, attn_implementation: str = "sdpa"):
    """(vision_tower, embed_vision, config) as fp32 eval modules with the real weights."""
    _check_transformers()
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder

    cfg = AutoConfig.from_pretrained(str(snapshot))
    vcfg = cfg.vision_config
    vcfg._attn_implementation = attn_implementation
    tower = AutoModel.from_config(vcfg)
    embed = EmbeddingGemma2MultimodalEmbedder(vcfg, cfg.text_config)
    sd_t, sd_e = {}, {}
    for f in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith("vision_tower."):
                    sd_t[k[len("vision_tower.") :]] = sf.get_tensor(k).float()
                elif k.startswith("embed_vision."):
                    sd_e[k[len("embed_vision.") :]] = sf.get_tensor(k).float()
    for mod, sd, name in ((tower, sd_t, "vision_tower"), (embed, sd_e, "embed_vision")):
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "inv_freq" not in m]
        if missing or unexpected:
            raise RuntimeError(f"{name} load mismatch: missing={missing} unexpected={unexpected}")
    return tower.float().eval(), embed.float().eval(), cfg


@dataclass
class VisionReferenceOutputs:
    patch_embeds: torch.Tensor  # [B, N, 768] patch embedder output (encoder input)
    layer_in: dict[int, torch.Tensor] = field(default_factory=dict)  # [B, N, 768]
    layer_out: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_in: dict[int, torch.Tensor] = field(default_factory=dict)  # input_layernorm(x)
    attn_out: dict[int, torch.Tensor] = field(default_factory=dict)  # o_proj output
    mlp_in: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_out: dict[int, torch.Tensor] = field(default_factory=dict)
    encoder_out: torch.Tensor | None = None  # [B, N, 768] last layer output
    pooled: torch.Tensor | None = None  # [B, N/9, 768] fp32 pooler output before stripping
    pooled_mask: torch.Tensor | None = None  # [B, N/9] True = kept
    soft_tokens: list[torch.Tensor] = field(default_factory=list)  # per image [n_b, 512] (embed_vision out)
    valid: torch.Tensor | None = None  # [B, N] True = valid patch


@torch.no_grad()
def run_vision_reference(
    tower,
    embed,
    pixel_values: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    capture_layers: tuple[int, ...] | None = None,
    capture_sublayers: tuple[int, ...] = (),
) -> VisionReferenceOutputs:
    layers = tower.encoder.layers
    capture_layers = tuple(range(len(layers))) if capture_layers is None else capture_layers
    store: dict[str, dict] = {k: {} for k in ("li", "lo", "ai", "ao", "mi", "mo", "pe", "pool")}
    handles = []

    def layer_hook(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            store["li"][i] = x.detach().clone()
            store["lo"][i] = (out[0] if isinstance(out, tuple) else out).detach().clone()

        return hook

    def attn_hook(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            store["ai"][i] = x.detach().clone()
            store["ao"][i] = out[0].detach().clone()

        return hook

    def mlp_hook(i):
        def hook(mod, args, out):
            store["mi"][i] = args[0].detach().clone()
            store["mo"][i] = out.detach().clone()

        return hook

    def pe_hook(mod, args, out):
        store["pe"][0] = out.detach().clone()

    def pool_hook(mod, args, kwargs, out):
        store["pool"]["h"] = out[0].detach().clone()
        store["pool"]["m"] = out[1].detach().clone()

    for i in capture_layers:
        handles.append(layers[i].register_forward_hook(layer_hook(i), with_kwargs=True))
    for i in capture_sublayers:
        handles.append(layers[i].self_attn.register_forward_hook(attn_hook(i), with_kwargs=True))
        handles.append(layers[i].mlp.register_forward_hook(mlp_hook(i)))
    handles.append(tower.patch_embedder.register_forward_hook(pe_hook))
    handles.append(tower.pooler.register_forward_hook(pool_hook, with_kwargs=True))
    last_layer = len(layers) - 1
    if last_layer not in capture_layers:
        handles.append(layers[last_layer].register_forward_hook(layer_hook(last_layer), with_kwargs=True))
    try:
        out = tower(pixel_values=pixel_values.float(), pixel_position_ids=position_ids.long())
        soft = embed(inputs_embeds=out.last_hidden_state)
    finally:
        for h in handles:
            h.remove()
    valid = (position_ids != -1).all(dim=-1)
    k2 = tower.config.pooling_kernel_size**2
    counts = (valid.sum(-1) // k2).tolist()
    return VisionReferenceOutputs(
        patch_embeds=store["pe"][0],
        layer_in={i: v for i, v in store["li"].items() if i in capture_layers},
        layer_out={i: v for i, v in store["lo"].items() if i in capture_layers},
        attn_in=store["ai"],
        attn_out=store["ao"],
        mlp_in=store["mi"],
        mlp_out=store["mo"],
        encoder_out=store["lo"][last_layer],
        pooled=store["pool"]["h"],
        pooled_mask=store["pool"]["m"],
        soft_tokens=list(torch.split(soft, counts)),
        valid=valid,
    )
