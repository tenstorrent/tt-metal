#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate golden KV cache for GPT-OSS prefill validation.

Runs the stock HuggingFace ``GptOssForCausalLM`` (dense GQA — no MSA / no sparse indexer) on a
one-shot prefill and saves the post-RoPE K + raw V per layer in the format the prefill producer's
KV PCC check expects.

Unlike MiniMax M3 (which needed a custom reference because its HF checkpoint ships no modeling
code), GPT-OSS is a normal HF model — we load it directly with ``AutoModelForCausalLM``.

Output format (matches ``prefill_producer._read_slot_kv_and_check_pcc_gpt_oss``)::

    {trace_dir}/
        metadata.json                    - prompt, token_ids, model info
        kv_cache/
            layer_0.safetensors          - key_cache_layer_0 / value_cache_layer_0
            layer_1.safetensors            each [1, num_kv_heads, seq_len, head_dim]
            ...                            (post-RoPE K in HF half-split convention; raw V)
            layer_N.safetensors

Usage::

    # Short smoke test first (small prompt, faster iteration)
    export HF_MODEL=/data/.../gpt-oss-120b
    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt "The capital of France is" \\
        --out /tmp/gpt_oss_golden_smoke \\
        --max-tokens 512

    # Full run matching the Gate 2 YAML seq_len
    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt-json prompt.json \\
        --out /data/jmalone/gpt-oss-120b/prefill_traces/longbook_qa_eng_prefill_56320 \\
        --max-tokens 56320
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _raise_cpu_time_limit():
    """CPU inference across many threads drains RLIMIT_CPU ~N× wall-clock. Raise the soft limit
    to the hard limit so a long multi-layer prefill isn't killed with SIGXCPU mid-run."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            print(f"[limit] raised RLIMIT_CPU soft {soft}s -> {hard}")
        except (ValueError, OSError) as e:
            print(
                f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s); if run dies with "
                f"'CPU time limit exceeded', run `ulimit -t unlimited` first: {e}",
                file=sys.stderr,
            )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate golden KV cache for GPT-OSS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    prompt_group = ap.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-json", type=Path, help='JSON file with {"prompt": "..."}')
    prompt_group.add_argument("--prompt", type=str, help="Direct prompt text")

    ap.add_argument("--out", type=Path, required=True, help="Output trace dir (will create kv_cache/ subdir)")
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HF model dir with config.json + *.safetensors (default: $HF_MODEL / $PREFILL_HF_MODEL)",
    )
    ap.add_argument("--max-tokens", type=int, default=None, help="Truncate prompt to N tokens")
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize as raw text instead of chat-template wrapping. MUST match the tokenization "
        "your tests use — if tests apply chat template, generate golden with chat template.",
    )
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32", "float16"],
        default="bfloat16",
        help="Stored KV dtype (default: bfloat16, matches device cache).",
    )
    ap.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float32", "float16"],
        default="bfloat16",
        help="Model compute dtype (default: bfloat16). fp32 is a higher-precision golden but roughly "
        "2× RAM + slower; bf16 is what the device runs and is usually fine for PCC.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"cpu", "cuda", "cuda:0", or "auto" (default). "auto" uses device_map="auto" — GPU if '
        "available with CPU/disk offload, otherwise pure CPU.",
    )
    return ap.parse_args()


def load_prompt(args) -> str:
    if args.prompt_json:
        with open(args.prompt_json) as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompt" in data:
            return data["prompt"]
        if isinstance(data, str):
            return data
        raise ValueError(f"{args.prompt_json}: expected dict with 'prompt' key or string")
    return args.prompt


def tokenize_prompt(tokenizer, prompt: str, max_tokens, use_chat_template: bool):
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        ids = tokenizer(prompt)["input_ids"]
    if max_tokens and len(ids) > max_tokens:
        print(f"[tokenize] truncating {len(ids)} tokens -> {max_tokens}")
        ids = ids[:max_tokens]
    return ids, len(ids)


def _cache_layer_kv(past_key_values, layer_idx: int):
    """HF's Cache API has drifted across transformers versions. Try the modern ``layers`` list first,
    then fall back to the legacy tuple/list form."""
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        entry = layers[layer_idx]
        k = getattr(entry, "keys", None)
        v = getattr(entry, "values", None)
        if k is not None and v is not None:
            return k, v
    # DynamicCache older API: .key_cache / .value_cache lists
    k_list = getattr(past_key_values, "key_cache", None)
    v_list = getattr(past_key_values, "value_cache", None)
    if k_list is not None and v_list is not None:
        return k_list[layer_idx], v_list[layer_idx]
    # Legacy tuple: past_key_values[layer_idx] = (k, v)
    return past_key_values[layer_idx][0], past_key_values[layer_idx][1]


def main():
    args = parse_args()
    _raise_cpu_time_limit()

    model_path = args.model_path or os.environ.get("HF_MODEL") or os.environ.get("PREFILL_HF_MODEL")
    if not model_path:
        print("ERROR: provide --model-path or set $HF_MODEL / $PREFILL_HF_MODEL", file=sys.stderr)
        return 1

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    save_dtype = dtype_map[args.dtype]
    compute_dtype = dtype_map[args.compute_dtype]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    kv_cache_dir = out_dir / "kv_cache"
    kv_cache_dir.mkdir(exist_ok=True)

    print(f"[load] reading prompt...", flush=True)
    prompt = load_prompt(args)
    print(f"[load] prompt length: {len(prompt)} characters", flush=True)

    print(f"[load] loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    use_chat_template = not args.no_chat_template
    token_ids, seq_len = tokenize_prompt(tokenizer, prompt, args.max_tokens, use_chat_template)
    print(f"[load] tokenized to {seq_len} tokens (chat_template={use_chat_template})", flush=True)

    # Peek at the HF config for reporting; the model load will re-read it.
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = getattr(cfg, "text_config", cfg)
    print(
        f"[load] model shape: layers={cfg.num_hidden_layers} num_kv_heads={cfg.num_key_value_heads} "
        f"head_dim={cfg.head_dim}",
        flush=True,
    )

    print(f"\n{'=' * 70}", flush=True)
    print(f"[load] Loading GptOssForCausalLM from {model_path}", flush=True)
    print(f"[load] compute_dtype={args.compute_dtype} device={args.device}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    torch.set_num_threads(os.cpu_count() or 32)

    load_kwargs = dict(
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.device == "auto":
        load_kwargs["device_map"] = "auto"
    elif args.device == "cpu":
        load_kwargs["device_map"] = {"": "cpu"}
    else:
        load_kwargs["device_map"] = {"": args.device}

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    print(f"[load] ✓ Model ready in {time.time() - t0:.1f}s", flush=True)

    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    expected_shape = (1, num_kv_heads, seq_len, head_dim)

    print(f"\n{'=' * 70}", flush=True)
    print(f"[forward] Running prefill forward pass for {seq_len} tokens", flush=True)
    print(f"[forward] WARNING: CPU inference is SLOW - can take 10-60+ minutes for long prompts", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    # Route input to the model's own device (device_map="auto" may place embeddings on GPU).
    first_param_device = next(model.parameters()).device
    input_ids = input_ids.to(first_param_device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    forward_time = time.time() - t0

    past_key_values = outputs.past_key_values
    del outputs  # free the logits/etc. before iterating KV
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[forward] ✓ Completed in {int(forward_time // 60)}m {int(forward_time % 60)}s "
          f"({seq_len / max(forward_time, 1e-9):.2f} tok/s)", flush=True)
    print(f"[save] writing per-layer safetensors to {kv_cache_dir}/", flush=True)

    saved = {}
    progress = tqdm(total=num_layers, desc="Save KV per layer", unit="layer")
    for layer in range(num_layers):
        k, v = _cache_layer_kv(past_key_values, layer)
        k = k.detach().to("cpu", dtype=save_dtype).contiguous()
        v = v.detach().to("cpu", dtype=save_dtype).contiguous()
        if tuple(k.shape) != expected_shape or tuple(v.shape) != expected_shape:
            tqdm.write(
                f"[save] WARNING: layer {layer} KV shape mismatch! "
                f"K: {tuple(k.shape)}, V: {tuple(v.shape)}, expected: {expected_shape}"
            )
        save_file(
            {f"key_cache_layer_{layer}": k, f"value_cache_layer_{layer}": v},
            str(kv_cache_dir / f"layer_{layer}.safetensors"),
        )
        saved["key_shape"] = list(k.shape)
        saved["value_shape"] = list(v.shape)
        # Release GPU/CPU refs asap so the next layer isn't stacked on top.
        del k, v
        progress.update(1)
    progress.close()

    metadata = {
        "model_path": str(model_path),
        "reference": "transformers.GptOssForCausalLM (stock HF, dense GQA, post-RoPE K + raw V)",
        "prompt_source": str(args.prompt_json) if args.prompt_json else "direct",
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        "prompt_length_chars": len(prompt),
        "token_ids": token_ids,
        "n_tokens": seq_len,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "compute_dtype": args.compute_dtype,
        "device": args.device,
        "chat_template": use_chat_template,
        "forward_time_seconds": forward_time,
        "tokens_per_second": seq_len / max(forward_time, 1e-9),
        "kv_cache_format": "separate_k_v",
        "key_cache_shape": saved.get("key_shape"),
        "value_cache_shape": saved.get("value_shape"),
    }

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_size_gb = sum(f.stat().st_size for f in kv_cache_dir.glob("*.safetensors")) / (1024**3)

    print(f"\n{'=' * 70}", flush=True)
    print(f"✅ Golden KV cache generation complete!", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Metadata:         {metadata_path}", flush=True)
    print(f"KV cache:         {kv_cache_dir}/ ({num_layers} layer files)", flush=True)
    print(f"Total size:       {total_size_gb:.2f} GB", flush=True)
    print(f"", flush=True)
    print(f"Each layer contains:", flush=True)
    print(f"  - key_cache_layer_{{N}}:   {saved.get('key_shape')}", flush=True)
    print(f"  - value_cache_layer_{{N}}: {saved.get('value_shape')}", flush=True)
    print(f"", flush=True)
    print(f"Use in tests: export PREFILL_TRACE_DIR={out_dir}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
