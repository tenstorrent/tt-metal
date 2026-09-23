# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Generate two independent Hugging Face KV traces for shared producer PCC.

Use a local text file with at least twice ``seq_len - 1`` tokens. Each slot
gets a different passage, with one BOS token. Saved keys use HF's half-split
rotary frame; the common GQA producer converts them to the device frame.
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.seq_len < 2 or args.threads < 1:
        parser.error("seq-len must be >= 2 and threads must be positive")

    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    # Preserve line endings: changing CRLF to LF changes token IDs for the book fixture.
    tokens = tokenizer.encode(args.prompt_file.read_bytes().decode("utf-8"), add_special_tokens=False)
    passage_len = args.seq_len - 1
    if len(tokens) < 2 * passage_len:
        parser.error(f"prompt needs at least {2 * passage_len} tokens; got {len(tokens)}")
    slots = [[tokenizer.bos_token_id] + tokens[s * passage_len : (s + 1) * passage_len] for s in range(2)]
    if slots[0] == slots[1]:
        parser.error("the two passages must differ to detect slot contamination")
    for slot in range(2):
        if (args.output_dir / f"slot{slot}" / "metadata.json").exists():
            parser.error("output already contains a complete trace; use a new directory")

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).eval()
    if model.config.num_hidden_layers != 32 or model.config.num_key_value_heads != 8:
        raise ValueError("expected the Llama-3.1-8B checkpoint")
    load_seconds = time.perf_counter() - start
    print(f"Loaded FP32 CPU reference in {load_seconds:.2f}s", flush=True)
    with torch.inference_mode():
        for slot, ids in enumerate(slots):
            directory = args.output_dir / f"slot{slot}"
            cache_dir = directory / "kv_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            # The HF body computes all layers and returns post-RoPE K and unrotated
            # V. Skip the vocabulary projection: this trace validates cache bytes.
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
            # Publish metadata last: its presence denotes a complete trace.
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"slot{slot}: {args.seq_len} tokens, 32 layers, forward {forward_seconds:.2f}s", flush=True)
            del result, cache


if __name__ == "__main__":
    main()
