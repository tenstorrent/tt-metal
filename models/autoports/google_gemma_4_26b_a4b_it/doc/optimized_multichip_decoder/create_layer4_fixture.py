# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Continue the recorded raw layer-0 fixture through HF layers 0..3 on CPU.

Produces the actual layer-4 input for adjacent 4->5 decoder validation. A
shorter requested prefix is followed by the original recorded continuation;
HF recomputes all contextual states at the selected sequence's new positions.
Loads one real HF layer at a time; no TTNN import or complete-model execution.
"""

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig
from transformers.models.gemma4 import modeling_gemma4

from models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_activation_fixture import (
    evaluate_layer,
    save_fixture,
    sha256,
    write_json,
)
from models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls import (
    MODEL,
    REVISION,
    cached_file,
    load_real_layer,
)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-fixture", type=Path, default=root / "doc/optimized_decoder/actual_text_layer0_4096_128.pt"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    args = parser.parse_args()
    if args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    source = torch.load(args.source_fixture, map_location="cpu", weights_only=True, mmap=True)
    metadata = source["metadata"]
    if (metadata["model"], metadata["revision"], metadata["layer"]) != (MODEL, REVISION, 0):
        raise ValueError("Source must be the pinned target model's layer-0 input fixture")
    if metadata["source"]["kind"] != "recorded_real_text_hf_layer_inputs" or not metadata["teacher_forced"]:
        raise ValueError("Source must contain recorded text and teacher-forced continuation")
    source_length, source_steps = metadata["length"], metadata["steps"]
    if (source_length, source_steps) != (4096, 128):
        raise ValueError("This bounded continuation expects the recorded 4096/128 fixture")
    length, steps = args.length, args.steps
    if not 1 <= length <= source_length or not 0 <= steps <= source_steps:
        parser.error("Requested prefix/continuation exceeds the source fixture")
    config_path = Path(cached_file("config.json"))
    index_path = Path(cached_file("model.safetensors.index.json"))
    config = AutoConfig.from_pretrained(config_path.parent, local_files_only=True).text_config
    config._attn_implementation = "eager"
    if config.hidden_size_per_layer_input or config.num_kv_shared_layers:
        raise ValueError("This fixture requires the pinned checkpoint without PLE or shared KV layers")
    if any(config.layer_types[layer] != "sliding_attention" for layer in range(4)):
        raise ValueError("Streamed boundary-4 capture expects preceding sliding layers 0..3")
    hf_source = Path(inspect.getfile(modeling_gemma4))
    current_hashes = dict(
        config_sha256=sha256(config_path),
        weight_index_sha256=sha256(index_path),
        hf_source_sha256=sha256(hf_source),
    )
    for name, value in current_hashes.items():
        if metadata["source"].get(name) != value:
            raise ValueError(f"Current {name} differs from the recorded source fixture")
    for name, count in (("raw_prefill", source_length), ("raw_decode", source_steps)):
        value = source[name]
        if value.dtype != torch.float32 or tuple(value.shape) != (1, count, config.hidden_size):
            raise ValueError(f"Source {name} must be the original FP32 boundary")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite source {name}")
    source_tokens = source["token_ids"]
    if source_tokens.dtype != torch.int64 or tuple(source_tokens.shape) != (source_length + source_steps,):
        raise ValueError("Source token IDs must preserve the exact prefill and continuation")
    tokens = torch.cat((source_tokens[:length], source_tokens[source_length : source_length + steps]))
    hidden = torch.cat((source["raw_prefill"][:, :length], source["raw_decode"][:, :steps]), dim=1)
    source_metadata = metadata
    del source
    provenance = {
        **source_metadata["source"],
        **current_hashes,
        "transformers_version": transformers.__version__,
        "fixture_generator_sha256": sha256(__file__),
        "prefill_chunk_size": args.prefill_chunk_size,
        "preceding_cache": "explicit HF DynamicSlidingWindowLayer",
        "sampling": "Selected recorded corpus prefix followed by recorded continuation; HF recomputes new positions",
        "continuation": {
            "source_fixture": str(args.source_fixture.resolve()),
            "source_fixture_sha256": sha256(args.source_fixture),
            "source_boundary": 0,
            "source_fields": ["raw_prefill", "raw_decode"],
            "source_token_ranges": [[0, length], [source_length, source_length + steps]],
            "skipped_corpus_token_range": [length, source_length] if length < source_length else None,
            "target_prefill_position_range": [0, length],
            "target_decode_position_range": [length, length + steps],
            "sequence_kind": "spliced_recorded_tokens" if length < source_length else "original_recorded_tokens",
            "source_tokens_path": source_metadata["source"]["tokens_path"],
            "source_tokens_sha256": source_metadata["source"]["tokens_sha256"],
            "target_boundary": 4,
            "layers_evaluated": [0, 1, 2, 3],
            "selected_token_ids_preserved": True,
            "boundary_transport_rounding": "BF16 only when saving target boundary; intervening HF states remain FP32",
            "evaluate_layer_source_sha256": sha256(inspect.getfile(evaluate_layer)),
            "load_real_layer_source_sha256": sha256(inspect.getfile(load_real_layer)),
        },
    }
    timings = []
    for layer_idx in range(4):
        layer_started = time.monotonic()
        print(f"LOAD_HF_LAYER {layer_idx}", flush=True)
        layer = load_real_layer(config, layer_idx)
        hidden = evaluate_layer(layer, hidden, config, layer_idx, length, steps, args.prefill_chunk_size)
        del layer
        gc.collect()
        seconds = time.monotonic() - layer_started
        timings.append(dict(layer=layer_idx, seconds=seconds))
        print(f"HF_LAYER_RELEASED layer={layer_idx} seconds={seconds:.2f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = args.output_dir / f"actual_text_layer4_{length}_{steps}_tokens.json"
    write_json(tokens_path, dict(model=MODEL, revision=REVISION, token_ids=tokens.tolist()))
    provenance.update(tokens_path=str(tokens_path), tokens_sha256=sha256(tokens_path))
    manifest = save_fixture(args.output_dir, hidden, tokens, 4, length, steps, provenance)
    command = (
        [sys.executable, "-m", __spec__.name, *sys.argv[1:]] if __spec__ is not None else [sys.executable, *sys.argv]
    )
    write_json(
        args.output_dir / f"actual_text_layer4_{length}_{steps}_capture.json",
        dict(
            command=command,
            source=provenance,
            threads=args.threads,
            seconds=time.monotonic() - started,
            layers_evaluated=[0, 1, 2, 3],
            timings=timings,
            fixture=manifest["fixture"],
            fixture_sha256=manifest["fixture_sha256"],
            intended_validation="adjacent real-model layers 4->5 with PCC >= 0.995",
        ),
    )
    print("ADJACENT_STACK_FIXTURE_READY", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
