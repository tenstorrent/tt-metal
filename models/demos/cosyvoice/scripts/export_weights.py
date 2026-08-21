# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Export CosyVoice weights to a flat .npz the TTNN side can load on its own.

The tt-metal environment must not need the CosyVoice package. That is the same
two-environment rule the demo README states and that the Docker image enforces
(installing whisper into tt-metal's python_env broke torch outright), and it
applies just as much to weights: building a TTNN module from a live
`HiFTGenerator` would drag hyperpyyaml, the cosyvoice package and its torch pin
into the runtime that has to stay clean.

So this runs ONCE in the CosyVoice venv and emits a flat array dictionary:

    PYTHONPATH=/mnt/CosyVoice:/mnt/CosyVoice/third_party/Matcha-TTS \
    /mnt/cosyvoice_env/bin/python export_weights.py --out hift_weights.npz

weight_norm is folded with torch's own machinery rather than by reimplementing
`w = g*v/||v||` -- that arithmetic is easy to get subtly wrong (the wrong norm
axis yields a scaled-but-plausible weight) and the failure would be a uniform
distortion across the whole vocoder rather than an obvious break. The fold is
verified by comparing a forward pass before and after.

NOTE the reference's own `HiFTGenerator.remove_weight_norm()` does NOT work on any
torch >= 2.1. `generator.py:25` imports the LEGACY `torch.nn.utils.remove_weight_norm`
unconditionally, while `:27-29` prefer the NEW
`torch.nn.utils.parametrizations.weight_norm` for applying. On torch >= 2.1 those
are different mechanisms, so removal raises

    ValueError: weight_norm of 'weight' not found in ParametrizedConvTranspose1d(...)

`fold_weight_norm_inplace()` below handles both spellings, so this exporter works
regardless of which API the installed torch selected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

DEFAULT_ROOT = os.environ.get("COSYVOICE_ROOT", "/mnt/CosyVoice")


def load_module(model_dir: str, which: str):
    """`which` is 'hift' or 'flow'; both come from the same cosyvoice.yaml."""
    from hyperpyyaml import load_hyperpyyaml

    with open(os.path.join(model_dir, "cosyvoice.yaml")) as fh:
        configs = load_hyperpyyaml(fh)
    module = configs[which]
    sd = torch.load(os.path.join(model_dir, f"{which}.pt"), map_location="cpu", weights_only=True)
    if which == "hift":
        # hift.pt is saved with a `generator.` prefix; flow.pt is not.
        sd = {k.replace("generator.", ""): v for k, v in sd.items()}
    module.load_state_dict(sd, strict=True)
    module.eval()
    return module


def load_hift(model_dir: str):
    return load_module(model_dir, "hift")


def flow_meta(flow) -> dict:
    """Architecture constants the TTNN side needs and cannot read off a tensor."""
    enc = flow.encoder
    layer0 = enc.encoders[0]
    attn = layer0.self_attn
    return {
        "module": "flow",
        "n_layers": len(enc.encoders),
        "n_head": int(attn.h),
        "d_k": int(attn.d_k),
        "d_model": int(attn.h * attn.d_k),
        "ffn_dim": int(layer0.feed_forward.w_1.out_features),
        "ff_scale": float(layer0.ff_scale),
        "normalize_before": bool(layer0.normalize_before),
        "layer_norm_eps": 1e-12,
        "has_macaron": layer0.feed_forward_macaron is not None,
        "has_conv_module": layer0.conv_module is not None,
        "input_frame_rate": int(flow.input_frame_rate),
        "output_size": int(flow.output_size),
        "vocab_size": int(flow.input_embedding.num_embeddings),
        "n_timesteps": 10,  # hardcoded at flow.py:inference
        "inference_cfg_rate": float(flow.decoder.inference_cfg_rate),
        "t_scheduler": flow.decoder.t_scheduler,
    }


def llm_meta(llm) -> dict:
    """Constants for both encoders in TransformerLM.

    Two stacks with different shapes share one checkpoint: a 6-block Conformer
    text encoder (`text_encoder`) and a 14-block Transformer AR decoder (`llm`).
    They differ in more than depth -- the decoder's input layer is
    `LegacyLinearNoSubsampling`, which appends a **ReLU** after the LayerNorm that
    the plain `LinearNoSubsampling` does not have. Missing that produces a network
    that runs and drifts.
    """

    def stack(enc):
        layer0 = enc.encoders[0]
        attn = layer0.self_attn
        return {
            "n_layers": len(enc.encoders),
            "n_head": int(attn.h),
            "d_k": int(attn.d_k),
            "d_model": int(attn.h * attn.d_k),
            "ffn_dim": int(layer0.feed_forward.w_1.out_features),
            "normalize_before": bool(layer0.normalize_before),
            # encoder_layer.py pins 1e-12 on the block norms; subsampling.py pins
            # 1e-5 on the embedding norm. They are genuinely different.
            "layer_norm_eps": 1e-12,
            "embed_norm_eps": 1e-5,
            "embed_has_relu": bool(enc.embed.__class__.__name__.startswith("Legacy")),
            "input_size": int(enc.embed.out[0].in_features),
            # ConformerEncoder defaults activation_type to "swish" (SiLU);
            # TransformerEncoder defaults it to "relu". Both defaults are taken --
            # cosyvoice.yaml overrides neither -- so the two stacks in this one
            # checkpoint have *different* feed-forward activations.
            "ffn_activation": type(layer0.feed_forward.activation).__name__.lower(),
        }

    text = stack(llm.text_encoder)
    text.update(
        {
            "has_macaron": llm.text_encoder.encoders[0].feed_forward_macaron is not None,
            "has_conv_module": llm.text_encoder.encoders[0].conv_module is not None,
            "ff_scale": float(llm.text_encoder.encoders[0].ff_scale),
        }
    )
    return {
        "module": "llm",
        "text_encoder": text,
        "ar_decoder": stack(llm.llm),
        "llm_input_size": int(llm.llm_input_size),
        "speech_token_size": int(llm.speech_token_size),
        "text_token_size": int(llm.text_embedding.num_embeddings),
        "sos": int(llm.sos),
        "task_id": int(llm.task_id),
        "eos_token": int(llm.eos_token),
        # ras_sampling defaults from cosyvoice.yaml
        "sampling": {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1, "sampling": 25},
        "max_token_text_ratio": 20,
        "min_token_text_ratio": 2,
    }


def fold_weight_norm_inplace(model: torch.nn.Module) -> int:
    """Bake weight_norm into `.weight` for every submodule, either API. Returns
    how many modules were folded."""
    from torch.nn.utils import parametrize
    from torch.nn.utils import remove_weight_norm as legacy_remove

    folded = 0
    for module in model.modules():
        params = getattr(module, "parametrizations", None)
        if params is not None and "weight" in params:
            # torch >= 2.1: leave_parametrized=True writes the computed weight back
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            folded += 1
        elif hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            legacy_remove(module)
            folded += 1
    return folded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice-root", default=DEFAULT_ROOT)
    ap.add_argument("--checkpoint", default="CosyVoice-300M")
    ap.add_argument("--module", default="hift", choices=["hift", "flow", "llm"], help="which submodule to export")
    ap.add_argument("--out", default=None, help="default <this file>/../tests/golden/hift_weights.npz")
    ap.add_argument("--fp16", action="store_true", help="halve the file; the device carries bf16 anyway")
    args = ap.parse_args()

    root = args.cosyvoice_root
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "third_party", "Matcha-TTS"))
    model_dir = os.path.join(root, "pretrained_models", args.checkpoint)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", f"{args.module}_weights.npz"
    )
    out = os.path.abspath(out)

    model = load_module(model_dir, args.module)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.checkpoint} {args.module}: {n_params/1e6:.2f} M params")

    if args.module == "hift":
        # Prove the fold is a no-op numerically before trusting it.
        torch.manual_seed(0)
        mel = torch.randn(1, 80, 64)
        with torch.no_grad():
            before = model.f0_predictor(mel)
        n_folded = fold_weight_norm_inplace(model)
        with torch.no_grad():
            after = model.f0_predictor(mel)
        delta = (before - after).abs().max().item()
        print(f"weight_norm folded in {n_folded} modules; max|Δ| on a forward pass = {delta:.3e}")
        if delta > 1e-4:
            raise SystemExit(f"fold changed the output by {delta} -- refusing to export")
    else:
        # Neither flow nor llm uses weight_norm anywhere; assert rather than assume.
        assert fold_weight_norm_inplace(model) == 0, f"{args.module} unexpectedly has weight_norm"

    arrays, total = {}, 0
    for name, tensor in model.state_dict().items():
        a = tensor.detach().cpu().float().numpy()
        if args.fp16 and a.dtype == np.float32 and a.size > (1 << 16):
            a = a.astype(np.float16)
        arrays[name] = a
        total += a.nbytes

    meta = {"checkpoint": args.checkpoint, "n_params": int(n_params), "module": args.module}
    if args.module == "hift":
        meta.update(
            {
                "istft_params": dict(model.istft_params),
                "lrelu_slope": float(model.lrelu_slope),
                "audio_limit": float(model.audio_limit),
                "num_kernels": int(model.num_kernels),
                "num_upsamples": int(model.num_upsamples),
                "sampling_rate": int(model.sampling_rate),
                # The Hann window the reference builds with scipy get_window(..., fftbins=True).
                # Exported rather than recomputed so the device cannot disagree about it.
                "stft_window": model.stft_window.detach().cpu().numpy().tolist(),
                "weight_norm_folded": True,
            }
        )
    elif args.module == "flow":
        meta.update(flow_meta(model))
    else:
        meta.update(llm_meta(model))
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(
        f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB compressed, "
        f"{total/1e6:.1f} MB raw, {len(arrays)-1} tensors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
