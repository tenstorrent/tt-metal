# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only sampled HF oracle using every K/V row of a recorded text fixture."""

import argparse
import json
import sys
import time
from collections import UserDict
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, apply_rotary_pos_emb

from models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_activation_fixture import sha256
from models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls import (
    MODEL,
    REVISION,
    cached_file,
    load_real_layer,
)


def load_input_fixture(path, config, layer_idx, length):
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    metadata = saved["metadata"]
    if (metadata["model"], metadata["revision"], metadata["layer"]) != (MODEL, REVISION, layer_idx):
        raise ValueError("Fixture model, revision or layer does not match the requested oracle")
    values = saved["prefill"]
    if tuple(values.shape) != (1, metadata["length"], config.hidden_size) or values.dtype != torch.float32:
        raise ValueError("Fixture prefill must be FP32 with the recorded [1, length, hidden] shape")
    if not 0 < length <= min(values.shape[1], config.max_position_embeddings):
        raise ValueError("Requested context exceeds the fixture or model context")
    if metadata["source"]["kind"] != "recorded_real_text_hf_layer_inputs":
        raise ValueError("Long-context input must contain recorded text-derived HF layer inputs")
    values = values[:, :length]
    for chunk in values.split(1024, dim=1):
        if not torch.isfinite(chunk).all() or not torch.equal(chunk, chunk.bfloat16().float()):
            raise ValueError("Fixture transport must be finite and exactly BF16-roundtripped")
    return values, {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "available_length": metadata["length"],
        "used_length": length,
        "slice_start": 0,
        "source": metadata["source"],
    }


def load_reference(path, layer_idx, length, input_fixture):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["layer"], saved["length"], saved["revision"]) != (layer_idx, length, REVISION):
        raise ValueError("Saved reference layer, length or revision does not match")
    saved_fixture = saved.get("input_fixture")
    if input_fixture is not None:
        if saved_fixture is None or any(
            saved_fixture.get(key) != input_fixture[key] for key in ("sha256", "used_length", "slice_start")
        ):
            raise ValueError("Saved reference does not match the actual-input fixture hash and slice")
    elif saved_fixture is not None:
        raise ValueError("A text-fixture reference requires the matching --input-fixture")
    return saved["reference"], saved["samples"]


@torch.inference_mode()
def sampled_reference(hf, x, config, layer_idx, cos, sin):
    """Keep long_context's sampled rows and complete causal K/V attention."""
    length = x.shape[1]
    kind = config.layer_types[layer_idx]
    keys, values = [], []
    for start in range(0, length, 1024):
        end = min(start + 1024, length)
        norm = hf.input_layernorm(x[:, start:end])
        k = hf.self_attn.k_proj(norm).view(1, end - start, -1, hf.self_attn.head_dim)
        v = hf.self_attn.v_proj(norm).view_as(k) if hf.self_attn.v_proj is not None else k
        k = apply_rotary_pos_emb(hf.self_attn.k_norm(k), cos[:, start:end], sin[:, start:end], unsqueeze_dim=2)
        keys.append(k.transpose(1, 2))
        values.append(hf.self_attn.v_norm(v).transpose(1, 2))
        if end % 16384 == 0 or end == length:
            print(f"HF_REFERENCE_KV_PROGRESS layer={layer_idx} tokens={end}/{length}", flush=True)
    keys, values = torch.cat(keys, dim=2), torch.cat(values, dim=2)

    class FixedCache:
        def update(self, k, v, layer_idx):
            return keys, values

    samples = sorted(
        set(
            [
                0,
                min(31, length - 1),
                min(32, length - 1),
                *range(1023, length, 1024),
                *range(max(0, length - 33), length),
            ]
        )
    )
    refs = []
    positions = torch.arange(length)
    for offset in range(0, len(samples), 16):
        idx = torch.tensor(samples[offset : offset + 16])
        allowed = positions[None, :] <= idx[:, None]
        if kind == "sliding_attention":
            allowed &= positions[None, :] > idx[:, None] - config.sliding_window
        mask = torch.zeros(len(idx), length).masked_fill(~allowed, float("-inf"))[None, None]
        refs.append(
            hf(
                x[:, idx],
                position_embeddings=(cos[:, idx], sin[:, idx]),
                attention_mask=mask,
                shared_kv_states=UserDict(),
                past_key_values=FixedCache(),
            )
        )
        print(
            f"HF_REFERENCE_QUERY_PROGRESS layer={layer_idx} rows={min(offset + 16, len(samples))}/{len(samples)}",
            flush=True,
        )
    return torch.cat(refs, dim=1), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fixture", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True, choices=(0, 5))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = AutoConfig.from_pretrained(Path(cached_file("config.json")).parent, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    x, provenance = load_input_fixture(args.input_fixture, config, args.layer, args.length)
    hf = load_real_layer(config, args.layer)
    extent = (args.length + 1023) // 1024 * 1024
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    reference, samples = sampled_reference(hf, x, config, args.layer, cos, sin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": MODEL,
            "revision": REVISION,
            "layer": args.layer,
            "length": args.length,
            "reference": reference,
            "samples": samples,
            "input_fixture": provenance,
            "reference_helper_sha256": sha256(__file__),
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "layer": args.layer,
                "length": args.length,
                "input_fixture": provenance,
                "reference_file": str(args.output),
                "reference_sha256": sha256(args.output),
                "reference_shape": list(reference.shape),
                "compared_query_rows": samples,
                "all_kv_positions": args.length,
                "scope": "subset",
                "threads": args.threads,
                "seconds": time.monotonic() - started,
                "helper_sha256": sha256(__file__),
                "command": [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"HF_SAMPLED_REFERENCE_READY {args.output} rows={len(samples)}", flush=True)


if __name__ == "__main__":
    main()
