# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Reuse saved GQA traces, generating missing prefixes with the independent HF CPU reference."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from models.demos.common.prefill.runners.trace_utils import ensure_trace, validate_gqa_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Writable cache for generated traces and trace_paths.json"
    )
    parser.add_argument(
        "--reuse-trace-dirs", default="", help="One shared trace or one comma-separated trace per active slot"
    )
    parser.add_argument("--num-slots", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.seq_len < 2 or args.threads < 1:
        parser.error("seq-len must be >= 2 and threads must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = args.output_dir / "trace_paths.json"
    cached = [Path(p) for p in json.loads(index.read_text())] if index.exists() else []
    requested = [Path(p.strip()) for p in args.reuse_trace_dirs.split(",") if p.strip()]
    if requested and len(requested) not in (1, args.num_slots):
        parser.error("provide one shared trace or one trace per active slot")
    model = None
    generated_dir = None

    def validate(path, length):
        validate_gqa_trace(path, length, num_layers=32, num_kv_heads=8, head_dim=128)

    def generate(slot):
        # Reuse never imports transformers or loads weights. New output lives in a
        # separate directory, so a failed generation cannot damage the saved golden.
        nonlocal model, generated_dir, slots, load_seconds
        import torch
        import transformers
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model is None:
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
            tokens = tokenizer.encode(args.prompt_file.read_bytes().decode("utf-8"), add_special_tokens=False)
            passage_len = args.seq_len - 1
            if len(tokens) < args.num_slots * passage_len:
                raise ValueError(f"prompt needs at least {args.num_slots * passage_len} tokens; got {len(tokens)}")
            slots = [
                [tokenizer.bos_token_id] + tokens[s * passage_len : (s + 1) * passage_len]
                for s in range(args.num_slots)
            ]
            if len({tuple(ids) for ids in slots}) != len(slots):
                raise ValueError("generated passages must differ for distinct-slot coverage")
            start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
            ).eval()
            if model.config.num_hidden_layers != 32 or model.config.num_key_value_heads != 8:
                raise ValueError("expected the Llama-3.1-8B checkpoint")
            load_seconds = time.perf_counter() - start
            generated_dir = Path(tempfile.mkdtemp(prefix=f"tokens{args.seq_len}_", dir=args.output_dir))
            print(f"Loaded FP32 CPU reference in {load_seconds:.2f}s", flush=True)

        directory = generated_dir / f"slot{slot}"
        cache_dir = directory / "kv_cache"
        cache_dir.mkdir(parents=True)
        started = time.perf_counter()
        ids = slots[slot]
        with torch.inference_mode():
            # Skip the vocabulary projection: this trace validates post-RoPE K and raw V.
            result = model.model(input_ids=torch.tensor([ids]), use_cache=True, return_dict=True)
            cache = result.past_key_values
            if hasattr(cache, "to_legacy_cache"):
                cache = cache.to_legacy_cache()
            forward_seconds = time.perf_counter() - started
            if len(cache) != 32:
                raise ValueError("reference did not return all 32 layers")
            for layer, entry in enumerate(cache):
                key, value = entry[:2]
                expected = (1, 8, args.seq_len, 128)
                if key.shape != expected or value.shape != expected:
                    raise ValueError(f"layer {layer}: unexpected cache shape")
                if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                    raise ValueError(f"layer {layer}: non-finite reference")
                save_file(
                    {f"key_cache_layer_{layer}": key.contiguous(), f"value_cache_layer_{layer}": value.contiguous()},
                    str(cache_dir / f"layer_{layer}.safetensors"),
                )
        metadata = {
            "token_ids": ids,
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "num_layers": 32,
            "seq_len": args.seq_len,
            "slot": slot,
            "key_rotary_frame": "hf_half_split",
            "dtype": "float32",
            "reference": "transformers.AutoModelForCausalLM.model",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
        }
        # Metadata is published last, after every layer was saved successfully.
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"slot{slot}: generated {args.seq_len} tokens, 32 layers, forward {forward_seconds:.2f}s", flush=True)
        return directory

    slots, load_seconds = [], 0.0
    selected = []
    for slot in range(args.num_slots):
        fallback = cached[slot] if slot < len(cached) else args.output_dir / f"slot{slot}"
        candidate = requested[slot % len(requested)] if requested else fallback
        selected.append(
            ensure_trace(
                candidate,
                args.seq_len,
                lambda slot=slot, fallback=fallback: ensure_trace(
                    fallback, args.seq_len, lambda: generate(slot), validate=validate
                ),
                validate=validate,
            )
        )
    temporary = index.with_suffix(".tmp")
    temporary.write_text(json.dumps([str(path.resolve()) for path in selected], indent=2) + "\n")
    temporary.replace(index)


if __name__ == "__main__":
    main()
