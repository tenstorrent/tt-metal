# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF reference for the EmbeddingGemma 2 audio path (fp32, CPU): `Gemma4AudioModel` + `embed_audio`.

Needs transformers >= 5.20 (overlay on PYTHONPATH, see hf_reference.py). The modules are the
unmodified HF classes the full `EmbeddingGemma2Model` builds (`AutoModel.from_config(audio_config)`
and `EmbeddingGemma2MultimodalEmbedder`), loaded from the checkpoint's `audio_tower.*` /
`embed_audio.*` tensors (clip bounds included) and upcast to fp32. `run_audio_reference` records
activations with forward hooks; it never changes the model. The soft tokens equal
`EmbeddingGemma2Model.get_audio_features(...).pooler_output[mask]`
(checked by test_audio_model.py::test_reference_matches_full_model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from models.demos.embedding_gemma2.reference.hf_reference import _check_transformers


def load_reference_audio(snapshot):
    """(audio_tower, embed_audio, config) as fp32 eval modules with the real weights."""
    _check_transformers()
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder

    cfg = AutoConfig.from_pretrained(str(snapshot))
    acfg = cfg.audio_config
    acfg._attn_implementation = "eager"  # the only implementation Gemma4AudioModel supports
    # Construct in fp32: from_config otherwise builds in config.dtype (bf16), which rounds the
    # non-persistent `rel_pos_enc.inv_timescales` buffer (not in the checkpoint) to bf16 and makes
    # the reference differ from `EmbeddingGemma2Model.from_pretrained(dtype=float32)`.
    tower = AutoModel.from_config(acfg, dtype=torch.float32)
    embed = EmbeddingGemma2MultimodalEmbedder(acfg, cfg.text_config)
    sd_t, sd_e = {}, {}
    for f in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith("audio_tower."):
                    sd_t[k[len("audio_tower.") :]] = sf.get_tensor(k).float()
                elif k.startswith("embed_audio."):
                    sd_e[k[len("embed_audio.") :]] = sf.get_tensor(k).float()
    for mod, sd, name in ((tower, sd_t, "audio_tower"), (embed, sd_e, "embed_audio")):
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "inv_timescales" not in m and not m.endswith("softcap")]
        if missing or unexpected:
            raise RuntimeError(f"{name} load mismatch: missing={missing} unexpected={unexpected}")
    return tower.float().eval(), embed.float().eval(), cfg


@dataclass
class AudioReferenceOutputs:
    subsample_out: torch.Tensor  # [B, N, 1024] input_proj output (layer-0 input)
    conv0_out: torch.Tensor  # [B, 128, T/2, 64] subsample layer0 output (after LN + ReLU)
    conv1_out: torch.Tensor  # [B, 32, T/4, 32] subsample layer1 output
    proj_in: torch.Tensor  # [B, N, 1024] flattened conv output (input_proj input)
    token_mask: torch.Tensor  # [B, N] True = valid token
    attn_mask_5d: torch.Tensor | None  # the blocked mask HF passes to every layer
    layer_in: dict[int, torch.Tensor] = field(default_factory=dict)  # [B, N, 1024]
    layer_out: dict[int, torch.Tensor] = field(default_factory=dict)
    ffn1_in: dict[int, torch.Tensor] = field(default_factory=dict)
    ffn1_out: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_in: dict[int, torch.Tensor] = field(default_factory=dict)  # norm_pre_attn(x)
    attn_out: dict[int, torch.Tensor] = field(default_factory=dict)  # post() output
    lconv_in: dict[int, torch.Tensor] = field(default_factory=dict)
    lconv_out: dict[int, torch.Tensor] = field(default_factory=dict)
    ffn2_in: dict[int, torch.Tensor] = field(default_factory=dict)
    ffn2_out: dict[int, torch.Tensor] = field(default_factory=dict)
    hidden: torch.Tensor | None = None  # [B, N, 1536] output_proj output (last_hidden_state)
    soft_all: torch.Tensor | None = None  # [B, N, 512] embed_audio over every row
    soft_tokens: list[torch.Tensor] = field(default_factory=list)  # per clip [n_b, 512] (valid rows)
    max_abs_activation: float = 0.0  # largest |x| entering any clamp(+-1e10) site (layer in/out, ffn out)


@torch.no_grad()
def run_audio_reference(
    tower,
    embed,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
    *,
    capture_layers: tuple[int, ...] | None = None,
    capture_sublayers: tuple[int, ...] = (),
) -> AudioReferenceOutputs:
    layers = tower.layers
    capture_layers = tuple(range(len(layers))) if capture_layers is None else capture_layers
    store: dict[str, dict] = {
        k: {} for k in ("li", "lo", "f1i", "f1o", "ai", "ao", "ci", "co", "f2i", "f2o", "sub", "mask")
    }
    handles = []
    peak = [0.0]

    def track(t):
        peak[0] = max(peak[0], float(t.abs().max()))

    def layer_hook(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            o = out[0] if isinstance(out, tuple) else out
            track(x)
            track(o)
            if i in capture_layers:
                store["li"][i] = x.detach().clone()
                store["lo"][i] = o.detach().clone()
            if i == 0:
                store["mask"][0] = kwargs.get("attention_mask")

        return hook

    def io_hook(kin, kout, i, kwarg="hidden_states"):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs[kwarg]
            o = out[0] if isinstance(out, tuple) else out
            track(o)
            store[kin][i] = x.detach().clone()
            store[kout][i] = o.detach().clone()

        return hook

    def sub_hook(name):
        def hook(mod, args, out):
            store["sub"][name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            if name == "proj":
                store["sub"]["proj_in"] = args[0].detach().clone()

        return hook

    for i in range(len(layers)):
        handles.append(layers[i].register_forward_hook(layer_hook(i), with_kwargs=True))
    for i in capture_sublayers:
        L = layers[i]
        handles.append(L.feed_forward1.register_forward_hook(io_hook("f1i", "f1o", i), with_kwargs=True))
        handles.append(L.self_attn.register_forward_hook(io_hook("ai", "ao", i), with_kwargs=True))
        handles.append(L.lconv1d.register_forward_hook(io_hook("ci", "co", i), with_kwargs=True))
        handles.append(L.feed_forward2.register_forward_hook(io_hook("f2i", "f2o", i), with_kwargs=True))
    sub = tower.subsample_conv_projection
    handles.append(sub.layer0.register_forward_hook(sub_hook("conv0")))
    handles.append(sub.layer1.register_forward_hook(sub_hook("conv1")))
    handles.append(sub.input_proj_linear.register_forward_hook(sub_hook("proj")))
    try:
        out = tower(input_features.float(), input_features_mask.bool(), return_dict=True)
        hidden = out.last_hidden_state
        soft = embed(inputs_embeds=hidden)
    finally:
        for h in handles:
            h.remove()
    tmask = out.attention_mask.bool()
    return AudioReferenceOutputs(
        subsample_out=store["sub"]["proj"],
        conv0_out=store["sub"]["conv0"],
        conv1_out=store["sub"]["conv1"],
        proj_in=store["sub"]["proj_in"],
        token_mask=tmask,
        attn_mask_5d=store["mask"].get(0),
        layer_in=store["li"],
        layer_out=store["lo"],
        ffn1_in=store["f1i"],
        ffn1_out=store["f1o"],
        attn_in=store["ai"],
        attn_out=store["ao"],
        lconv_in=store["ci"],
        lconv_out=store["co"],
        ffn2_in=store["f2i"],
        ffn2_out=store["f2o"],
        hidden=hidden,
        soft_all=soft,
        soft_tokens=[soft[b][tmask[b]] for b in range(soft.shape[0])],
        max_abs_activation=peak[0],
    )
