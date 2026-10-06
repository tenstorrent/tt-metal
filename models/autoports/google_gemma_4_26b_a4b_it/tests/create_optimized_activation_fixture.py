# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture text-driven HF layer inputs without loading the complete model.

Only layers preceding the requested boundary are evaluated. Each checkpoint
layer is loaded once, evaluated over prefill and teacher-forced decode, and
released before the next layer. This reorders independent layer/token work;
it preserves causal attention, the HF sliding cache, RoPE, and layer scaling.
The saved transport tensors explicitly round the captured boundary to BF16,
matching the layer-only device harness. Raw FP32 boundaries are also retained.
"""

import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
from collections import UserDict
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from transformers.models.gemma4 import modeling_gemma4
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

from models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls import (
    MODEL,
    REVISION,
    cached_file,
    load_real_layer,
)

REPO = Path(__file__).resolve().parents[4]
DEFAULT_TEXT = (
    "README.md",
    "docs/realtime_profiler_architecture.md",
    "docs/L1_ACCUMULATION_FP32_ANALYSIS.md",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def embedding_rows(token_ids, weight_map, hidden_size):
    """Gather only used vocabulary rows, retaining FP32 HF embedding semantics."""
    key = "model.language_model.embed_tokens.weight"
    unique, inverse = token_ids.unique(sorted=True, return_inverse=True)
    with safe_open(cached_file(weight_map[key]), framework="pt", device="cpu") as weights:
        source = weights.get_slice(key)
        rows = torch.stack([source[index : index + 1, :].squeeze(0) for index in unique.tolist()]).float()
    # Gemma4TextScaledWordEmbedding casts its nonpersistent scale to weight dtype.
    # All preceding checkpoint weights are FP32 in this reference computation.
    scale = torch.tensor(hidden_size**0.5, dtype=torch.float32)
    return rows[inverse].reshape(1, token_ids.numel(), hidden_size) * scale


def attention_mask(config, cache, layer_idx, positions):
    # Ask this layer's HF cache for the exact K extent/offset before its update.
    # The global mask helper selects a representative layer; during streaming
    # only the current layer's cache is populated, so use its geometry directly.
    kv_length, kv_offset = cache.get_mask_sizes(positions.numel(), layer_idx)
    keys = torch.arange(kv_offset, kv_offset + kv_length)
    allowed = positions[:, None] >= keys[None, :]
    if config.layer_types[layer_idx] == "sliding_attention":
        allowed &= positions[:, None] - keys[None, :] < config.sliding_window
    return torch.zeros((positions.numel(), kv_length), dtype=torch.float32).masked_fill(
        ~allowed, torch.finfo(torch.float32).min
    )[None, None]


@torch.inference_mode()
def evaluate_layer(layer, hidden, config, layer_idx, length, steps, prefill_chunk_size=None):
    if prefill_chunk_size is not None:
        if prefill_chunk_size < 1 or config.layer_types[layer_idx] != "sliding_attention":
            raise ValueError("Streamed fixture capture requires sliding layers and a positive chunk size")
        # In this Transformers version DynamicCache(config) slices layer_types
        # with [:-num_kv_shared_layers]; zero shared layers therefore selects an
        # unbounded cache. Construct the HF sliding layers explicitly.
        cache = Cache(
            layers=[DynamicSlidingWindowLayer(sliding_window=config.sliding_window) for _ in range(layer_idx + 1)]
        )
    else:
        cache = DynamicCache(config=config)
    rotary = Gemma4TextRotaryEmbedding(config)
    positions = torch.arange(length + steps)
    if prefill_chunk_size is None:
        cos, sin = rotary(hidden, positions[None], layer_type=config.layer_types[layer_idx])
    shared_kv_states = UserDict()
    output = torch.empty_like(hidden)
    chunk = prefill_chunk_size or length
    ranges = [(start, min(start + chunk, length)) for start in range(0, length, chunk)]
    ranges.extend((position, position + 1) for position in range(length, length + steps))
    for start, stop in ranges:
        pos = positions[start:stop]
        rope = (
            rotary(hidden[:, start:stop], pos[None], layer_type=config.layer_types[layer_idx])
            if prefill_chunk_size is not None
            else (cos[:, start:stop], sin[:, start:stop])
        )
        output[:, start:stop] = layer(
            hidden[:, start:stop],
            per_layer_input=None,
            shared_kv_states=shared_kv_states,
            position_embeddings=rope,
            attention_mask=attention_mask(config, cache, layer_idx, pos),
            position_ids=pos[None],
            past_key_values=cache,
        )
        if prefill_chunk_size is not None and (stop % 16384 == 0 or stop == length + steps):
            print(f"HF_STREAM_PROGRESS layer={layer_idx} tokens={stop}/{length + steps}", flush=True)
    if not torch.isfinite(output).all():
        raise ValueError(f"Nonfinite HF activations after layer {layer_idx}")
    return output


def save_fixture(output_dir, hidden, tokens, layer_idx, length, steps, provenance):
    stem = f"actual_text_layer{layer_idx}_{length}_{steps}"
    path = output_dir / f"{stem}.pt"
    raw_prefill = hidden[:, :length].clone()
    raw_decode = hidden[:, length : length + steps].clone()
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "layer": layer_idx,
        "length": length,
        "steps": steps,
        "source": provenance,
        "raw_dtype": "float32",
        "transport_dtype": "bfloat16_roundtripped_float32",
        "teacher_forced": True,
    }
    fixture = {
        "metadata": metadata,
        "prefill": raw_prefill.bfloat16().float(),
        "decode": raw_decode.bfloat16().float(),
        "raw_prefill": raw_prefill,
        "raw_decode": raw_decode,
        "token_ids": tokens[: length + steps].clone(),
    }
    torch.save(fixture, path)
    manifest = {
        **metadata,
        "fixture": str(path),
        "fixture_sha256": sha256(path),
        "prefill_shape": list(fixture["prefill"].shape),
        "decode_shape": list(fixture["decode"].shape),
        "token_count": length + steps,
        "decode_first_token_ids": tokens[length : length + min(16, steps)].tolist(),
        "finite": bool(torch.isfinite(hidden).all()),
        "raw_rms": float(hidden.square().mean().sqrt()),
        "max_transport_rounding_error": float((hidden - hidden.bfloat16().float()).abs().max()),
    }
    write_json(output_dir / f"{stem}.json", manifest)
    print(f"FIXTURE_READY {path}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", type=int, nargs=2, action="append", metavar=("PREFILL", "DECODE"))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--text-file", type=Path, action="append")
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument(
        "--prefill-chunk-size", type=int, help="Stream preceding sliding layers with a bounded HF cache"
    )
    parser.add_argument(
        "--repeat-text-to-length", action="store_true", help="Repeat the source text before tokenization"
    )
    args = parser.parse_args()
    cases = args.case or [(4096, 128), (1025, 512)]
    layers = sorted(set(args.layers))
    if not layers or min(layers) < 0 or max(layers) > 5:
        parser.error("This bounded fixture generator supports layer boundaries 0 through 5")
    if any(length < 1 or steps < 0 for length, steps in cases):
        parser.error("Each case requires positive prefill and nonnegative decode lengths")
    if args.prefill_chunk_size is not None and args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config_path = Path(cached_file("config.json"))
    config = AutoConfig.from_pretrained(config_path.parent, local_files_only=True).text_config
    config._attn_implementation = "eager"
    # These dependencies are absent in the pinned target. Never silently omit
    # them if this helper is accidentally pointed at a different configuration.
    if config.hidden_size_per_layer_input or config.num_kv_shared_layers:
        raise ValueError("This fixture requires a checkpoint without PLE or shared KV layers")
    if any(length + steps > config.max_position_embeddings for length, steps in cases):
        parser.error("A fixture cannot exceed the checkpoint's context length")
    index_path = Path(cached_file("model.safetensors.index.json"))
    weight_map = json.loads(index_path.read_text())["weight_map"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=not args.allow_tokenizer_download
    )
    text_paths = args.text_file or [REPO / name for name in DEFAULT_TEXT]
    corpus = "\n\n".join(path.read_text() for path in text_paths)
    encoded = tokenizer(corpus, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    total = max(length + steps for length, steps in cases)
    repetitions = 1
    if len(encoded) < total and args.repeat_text_to_length:
        original_corpus = corpus
        while len(encoded) < total:
            repetitions = max(repetitions + 1, (total + len(encoded) - 1) // len(encoded) * repetitions)
            corpus = "\n\n".join([original_corpus] * repetitions)
            encoded = tokenizer(corpus, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(encoded) < total:
        raise ValueError(f"Text supplies {len(encoded)} tokens; {total} needed. Supply additional --text-file inputs.")
    tokens = torch.tensor(encoded[:total], dtype=torch.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_path = args.output_dir / "actual_text_source.txt"
    tokens_path = args.output_dir / "actual_text_tokens.json"
    text_path.write_text(corpus)
    write_json(tokens_path, {"model": MODEL, "revision": REVISION, "token_ids": tokens.tolist()})
    model_source = Path(inspect.getfile(modeling_gemma4))
    provenance = {
        "kind": "recorded_real_text_hf_layer_inputs",
        "text_files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in text_paths],
        "corpus_path": str(text_path),
        "corpus_sha256": sha256(text_path),
        "tokens_path": str(tokens_path),
        "tokens_sha256": sha256(tokens_path),
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_revision": REVISION,
        "transformers_version": transformers.__version__,
        "fixture_generator_sha256": sha256(__file__),
        "hf_source_sha256": sha256(model_source),
        "config_sha256": sha256(config_path),
        "weight_index_sha256": sha256(index_path),
        "preceding_computation": "HF FP32 eager, FP32 checkpoint weights and KV, original scaled embedding and layer scalars",
        "embedding_scale_fp32": float(torch.tensor(config.hidden_size**0.5, dtype=torch.float32)),
        "embedding_scale_bfloat16_control": float(torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16)),
        "per_layer_embeddings": False,
        "shared_kv_layers": 0,
        "sampling": "Actual corpus token continuation; no generated or random activations",
        "text_repetitions_before_tokenization": repetitions,
        "prefill_chunk_size": args.prefill_chunk_size,
        "preceding_cache": "explicit HF DynamicSlidingWindowLayer" if args.prefill_chunk_size else "HF DynamicCache",
    }
    hidden = embedding_rows(tokens, weight_map, config.hidden_size)
    streams = {(length, steps): hidden[:, : length + steps].clone() for length, steps in cases}
    del hidden
    manifests, timings = [], []
    started = time.monotonic()
    for boundary in range(max(layers) + 1):
        if boundary in layers:
            for (length, steps), hidden in streams.items():
                manifests.append(save_fixture(args.output_dir, hidden, tokens, boundary, length, steps, provenance))
        if boundary == max(layers):
            break
        layer_start = time.monotonic()
        print(f"LOAD_HF_LAYER {boundary}", flush=True)
        layer = load_real_layer(config, boundary)
        for (length, steps), hidden in streams.items():
            stream_start = time.monotonic()
            streams[(length, steps)] = evaluate_layer(
                layer, hidden, config, boundary, length, steps, args.prefill_chunk_size
            )
            elapsed = time.monotonic() - stream_start
            timings.append({"layer": boundary, "length": length, "steps": steps, "seconds": elapsed})
            print(f"HF_LAYER_COMPLETE layer={boundary} length={length} steps={steps} seconds={elapsed:.2f}", flush=True)
        del layer, hidden
        gc.collect()
        print(f"HF_LAYER_RELEASED layer={boundary} seconds={time.monotonic() - layer_start:.2f}", flush=True)
    write_json(
        args.output_dir / "actual_text_fixture_manifest.json",
        {
            "command": [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
            "source": provenance,
            "threads": args.threads,
            "seconds": time.monotonic() - started,
            "layers_evaluated": list(range(max(layers))),
            "timings": timings,
            "fixtures": [{"path": row["fixture"], "sha256": row["fixture_sha256"]} for row in manifests],
        },
    )


if __name__ == "__main__":
    main()
