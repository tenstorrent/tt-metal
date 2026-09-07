#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write a SYNTHETIC Mistral-Medium-3.5 checkpoint in the published checkpoint's exact on-disk shape.

Why this exists. The bring-up needs the P1 loader path — safetensors shards, the
``Mistral3ForConditionalGeneration`` wrapper prefix, per-tensor fp8 weights with ``weight_scale`` /
``input_scale`` siblings, ``modules_to_not_convert`` — exercised for real, and the published 128 B
checkpoint is not reachable from this machine. A synthetic checkpoint reproduces every one of those
FORMAT properties at a size a host can build in seconds, so the loader, the dequantizer, the key
mapping and the golden trace are all validated against real files rather than against a mock.

What it does NOT give you is the model's learned weights, so any accuracy number measured against it
is a number about this code, not about Mistral-Medium-3.5. See README "Known gaps".

Produced layout (the published checkpoint's, minus the vision tower's real weights):

    {out}/config.json                      the vendored config, with dims overridden and
                                           quantization_config preserved
    {out}/model-0000N-of-0000N.safetensors fp8 weights + weight_scale/input_scale siblings
    {out}/model.safetensors.index.json     the weight map
    {out}/reference_bf16.pt                the pre-quantization bf16 weights, so a test can measure
                                           the loader's round-trip error against ground truth

Usage:
    python models/demos/mistral_3_5_d_p/scripts/make_synthetic_checkpoint.py \
        --out /tmp/mistral_synth --layers 2 --hidden 12288 --intermediate 28672 --vocab 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.demos.mistral_3_5_d_p.reference import model as reference  # noqa: E402
from models.demos.mistral_3_5_d_p.reference.mistral_config import raw_config, reduced_text_config  # noqa: E402
from models.demos.mistral_3_5_d_p.tt.fp8_dequant import (  # noqa: E402
    INPUT_SCALE_SUFFIX,
    WEIGHT_SCALE_SUFFIX,
    is_unquantized_module,
)

# The wrapper prefix the published checkpoint uses for its text backbone (verified against
# transformers 5.12's Mistral3ForConditionalGeneration state dict).
WRAPPER_PREFIX = "model.language_model."


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=None, help="default: the config's 12288")
    ap.add_argument("--intermediate", type=int, default=None, help="default: the config's 28672")
    ap.add_argument("--vocab", type=int, default=None, help="default: the config's 131072")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=2, help="how many safetensors files to split across")
    ap.add_argument(
        "--unquantized",
        action="store_true",
        help="write bf16 weights with no fp8 scales (for isolating the loader from the dequantizer)",
    )
    return ap.parse_args()


def quantize_per_tensor_fp8(weight: torch.Tensor):
    """``(fp8_weight, weight_scale)`` such that ``fp8_weight.float() * weight_scale ~= weight``.

    The scale is ``amax / e4m3_max``, which is what compressed-tensors / vLLM emit for a per-tensor
    fp8 checkpoint: it puts the largest magnitude at the top of the representable range.
    """
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    amax = weight.abs().max().float()
    scale = (amax / e4m3_max).clamp(min=1e-12)
    return (weight.float() / scale).to(torch.float8_e4m3fn), scale.reshape(())


def build_bf16_state_dict(args):
    """A randomly-initialised text backbone, in the wrapper's key naming."""
    hf_config = reduced_text_config(
        num_hidden_layers=args.layers,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        vocab_size=args.vocab,
    )
    model = reference.build_reference_model(hf_config, seed=args.seed)
    backbone = {}
    for key, value in model.state_dict().items():
        # model.layers.* / model.embed_tokens.weight / model.norm.weight -> the wrapper prefix.
        # lm_head.weight sits outside the wrapper in the real checkpoint too.
        if key.startswith("model."):
            key = WRAPPER_PREFIX + key[len("model.") :]
        backbone[key] = value.to(torch.bfloat16).contiguous()
    return hf_config, backbone


def quantize_state_dict(bf16_state: dict, *, unquantized: bool):
    """Turn the bf16 backbone into the checkpoint's on-disk tensors.

    Only 2-D projection weights are quantized, which is what the published checkpoint does: norms
    and the embedding table stay bf16, and ``modules_to_not_convert`` (``lm_head`` here) is left
    alone even though it is a projection.
    """
    out: dict[str, torch.Tensor] = {}
    n_quantized = 0
    for key, value in bf16_state.items():
        quantizable = (
            not unquantized
            and value.ndim == 2
            and key.endswith(".weight")
            and not is_unquantized_module(key)
            and "embed_tokens" not in key
        )
        if not quantizable:
            out[key] = value
            continue
        fp8, scale = quantize_per_tensor_fp8(value)
        base = key[: -len(".weight")]
        out[key] = fp8
        out[f"{base}.{WEIGHT_SCALE_SUFFIX}"] = scale
        # The static activation scale the checkpoint also ships. The device computes activations in
        # bf16 and the loader drops these; they are written so the loader's dropping is exercised.
        out[f"{base}.{INPUT_SCALE_SUFFIX}"] = torch.tensor(1.0)
        n_quantized += 1
    print(f"[quantize] {n_quantized} projections -> per-tensor fp8 (+ weight_scale, input_scale)", flush=True)
    return out


def write_config(out_dir: Path, hf_config, *, unquantized: bool):
    """The vendored config with the run's dims, keeping ``quantization_config`` and the vision block
    so ``AutoConfig`` sees the same wrapper shape the published checkpoint has."""
    cfg = json.loads(json.dumps(raw_config()))  # deep copy
    text = cfg["text_config"]
    text["num_hidden_layers"] = hf_config.num_hidden_layers
    text["hidden_size"] = hf_config.hidden_size
    text["intermediate_size"] = hf_config.intermediate_size
    text["vocab_size"] = hf_config.vocab_size
    if unquantized:
        cfg.pop("quantization_config", None)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def write_shards(out_dir: Path, tensors: dict, n_shards: int):
    """Split the tensors across ``n_shards`` safetensors files and write the index, as HF does."""
    from safetensors.torch import save_file

    keys = sorted(tensors)
    n_shards = max(1, min(n_shards, len(keys)))
    per_shard = -(-len(keys) // n_shards)
    weight_map, total_bytes = {}, 0
    for shard_idx in range(n_shards):
        shard_keys = keys[shard_idx * per_shard : (shard_idx + 1) * per_shard]
        if not shard_keys:
            continue
        name = f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file({k: tensors[k].contiguous() for k in shard_keys}, str(out_dir / name))
        for k in shard_keys:
            weight_map[k] = name
        total_bytes += (out_dir / name).stat().st_size
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2)
    )
    print(f"[write] {len(weight_map)} tensors in {n_shards} shard(s), {total_bytes / 1e9:.2f} GB", flush=True)


def main():
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_config, bf16_state = build_bf16_state_dict(args)
    tensors = quantize_state_dict(bf16_state, unquantized=args.unquantized)
    write_config(out_dir, hf_config, unquantized=args.unquantized)
    write_shards(out_dir, tensors, args.shards)
    # Ground truth for the loader test: the weights BEFORE quantization, in the text-backbone naming
    # the LOADER emits (``model.*`` / ``lm_head.*``), so a test can compare loaded-vs-original by key
    # without re-implementing the prefix mapping.
    torch.save(
        {
            ("model." + k[len(WRAPPER_PREFIX) :]) if k.startswith(WRAPPER_PREFIX) else k: v
            for k, v in bf16_state.items()
        },
        out_dir / "reference_bf16.pt",
    )
    print(f"[write] pre-quantization bf16 ground truth -> {out_dir / 'reference_bf16.pt'}", flush=True)
    print(f"\nSynthetic checkpoint ready: {out_dir}\n  export HF_MODEL={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
