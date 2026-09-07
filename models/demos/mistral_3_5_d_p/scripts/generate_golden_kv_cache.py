#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the CPU golden for Mistral-Medium-3.5 prefill validation.

Pattern: ``gpt_oss_d_p/scripts/generate_golden_kv_cache.py`` +
``deepseek_v3_d_p/tt/runners/generate_prompt_trace.py``. This is the ONE place a CPU reference
forward runs; every test asserts against what it produced (recipe §3: "assert rather than recompute
where a CPU run is expensive").

Two output shapes, because two consumers want different things:

``--mode trace`` (default) — the ``PREFILL_TRACE_DIR`` layout the device harnesses read:

    {out}/metadata.json                          token_ids + dims + provenance
    {out}/kv_cache/layer_N.safetensors           key_cache_layer_N / value_cache_layer_N,
                                                 each [1, num_kv_heads, seq_len, head_dim]
    {out}/reference_top1.safetensors             HF's per-position argmax (top-1 sign-off)

  K is post-RoPE in HF (concat-halves) convention and V is raw — exactly what the device cache holds
  once the device K is permuted HF->Meta (``tt/rope.hf_to_meta_head_permutation``). Consumed by
  ``tests/galaxy_prefill_kv_pcc.py`` and by the shared producer/runner e2e test.

``--mode cache`` — the frozen-key golden cache (``reference/golden_cache.py``): per-layer residual
  snapshots plus KV, keyed on every field that changes the result, for the host-side tests.

Weights: ``--weights <dir>`` loads a real checkpoint through the production loader (so the loader is
part of what gets validated); with no ``--weights`` the model is randomly initialised under
``--seed``, which is what every pre-P1 test uses. Random init is not a lesser mode here — the whole
point of the recipe's "random weights until P1" rule is that the two sides share weights exactly.

Usage:
    # golden trace from a checkpoint (real or synthetic), for the device KV-PCC harness
    python models/demos/mistral_3_5_d_p/scripts/generate_golden_kv_cache.py \
        --weights /path/to/checkpoint --isl 5120 --layers 4 --out /tmp/mistral_golden_5120

    # random-weight trace at reduced depth/width, no checkpoint needed
    python models/demos/mistral_3_5_d_p/scripts/generate_golden_kv_cache.py \
        --isl 512 --layers 2 --hidden 12288 --out /tmp/mistral_golden_512

    # the frozen-key host golden cache
    python models/demos/mistral_3_5_d_p/scripts/generate_golden_kv_cache.py \
        --mode cache --layers 4 --isl 512 --hidden 1024 --intermediate 2048 --vocab 2048
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.demos.mistral_3_5_d_p.reference import golden_cache, model as reference  # noqa: E402
from models.demos.mistral_3_5_d_p.reference.mistral_config import (  # noqa: E402
    MistralMedium35Config as C,
)
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config  # noqa: E402


def raise_cpu_time_limit():
    """A long CPU forward can hit RLIMIT_CPU; raise the soft limit to the hard one up front so the
    run does not die hours in with 'CPU time limit exceeded'."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            print(f"[limit] raised RLIMIT_CPU soft {soft}s -> {hard}", flush=True)
        except (ValueError, OSError) as e:
            print(f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s): {e}", file=sys.stderr)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate the Mistral-Medium-3.5 CPU golden",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mode", choices=["trace", "cache"], default="trace", help="output shape (see the docstring)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (required for --mode trace)")
    ap.add_argument("--weights", type=str, default=None, help="checkpoint dir; omit for random weights")
    ap.add_argument("--isl", type=int, default=512, help="prompt length in tokens")
    ap.add_argument("--layers", type=int, default=2, help="decoder layers to build and capture")
    ap.add_argument("--hidden", type=int, default=None, help="override hidden_size (host-affordable runs)")
    ap.add_argument("--intermediate", type=int, default=None, help="override intermediate_size")
    ap.add_argument("--vocab", type=int, default=None, help="override vocab_size")
    ap.add_argument("--seed", type=int, default=0, help="random-weight / token seed")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="stored KV dtype")
    ap.add_argument(
        "--compute-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help=(
            "reference COMPUTE precision. float32 (default) is the strict oracle. bfloat16 matches "
            "what the sibling harnesses (gpt_oss_d_p, minimax_m3) generate their goldens in, and is "
            "the apples-to-apples comparison for a device whose activations are bf16 — see README."
        ),
    )
    return ap.parse_args()


def build_model_and_tokens(args):
    """The reference model plus this run's token ids, from a checkpoint or from ``--seed``."""
    hf_config = reduced_text_config(
        num_hidden_layers=args.layers,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        vocab_size=args.vocab,
    )
    compute_bf16 = args.compute_dtype == "bfloat16"
    if args.weights:
        from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs

        print(f"[load] loading the checkpoint at {args.weights} through the production loader ...", flush=True)
        # convert_to_meta_format=False: the reference is HF-convention, and the Meta swizzle is the
        # DEVICE's requirement. Swizzling here would silently rotate the golden's q/k.
        state_dict = ModelArgs.load_state_dict(args.weights, convert_to_meta_format=False)
        state_dict = {k: v for k, v in state_dict.items() if _keep_for_layers(k, args.layers)}
        model = reference.build_reference_model(hf_config, state_dict=state_dict)
        tokenizer_dir = args.weights
    else:
        print(f"[load] building a randomly initialised reference (seed={args.seed}) ...", flush=True)
        model = reference.build_reference_model(hf_config, seed=args.seed)
        tokenizer_dir = None
    if compute_bf16:
        print("[load] reference COMPUTE in bfloat16 (HF still reduces norms/softmax in fp32)", flush=True)
        model = model.to(torch.bfloat16).eval()

    torch.manual_seed(args.seed + 1)
    token_ids = torch.randint(0, hf_config.vocab_size, (1, args.isl))
    return model, hf_config, token_ids, tokenizer_dir


def _keep_for_layers(key: str, num_layers: int) -> bool:
    """Drop layers beyond ``num_layers`` so a reduced-depth run can load a full checkpoint."""
    marker = "model.layers."
    if marker not in key:
        return True
    idx = int(key.split(marker, 1)[1].split(".", 1)[0])
    return idx < num_layers


def write_trace(args, model, hf_config, token_ids):
    """Write the ``PREFILL_TRACE_DIR`` layout the device harnesses read."""
    from safetensors.torch import save_file

    out_dir = args.out
    kv_dir = out_dir / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)
    store_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(
        f"[forward] one-shot CPU prefill over {args.isl} tokens, {args.layers} layers "
        f"(compute {args.compute_dtype})",
        flush=True,
    )
    print("[forward] WARNING: CPU inference is SLOW — expect minutes at real dims", flush=True)
    torch.set_num_threads(os.cpu_count() or 32)
    t0 = time.time()
    out = reference.model_reference_forward(model, token_ids)
    elapsed = time.time() - t0
    print(f"[forward] done in {elapsed:.1f}s ({args.isl / max(elapsed, 1e-9):.2f} tok/s)", flush=True)

    expected = (1, hf_config.num_key_value_heads, args.isl, hf_config.head_dim)
    for layer_idx, (k, v) in enumerate(out.kv):
        k, v = k.float(), v.float()
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"layer {layer_idx} KV shape K={tuple(k.shape)} V={tuple(v.shape)} != {expected}")
        save_file(
            {
                f"key_cache_layer_{layer_idx}": k.to(store_dtype).contiguous(),
                f"value_cache_layer_{layer_idx}": v.to(store_dtype).contiguous(),
            },
            str(kv_dir / f"layer_{layer_idx}.safetensors"),
        )
    save_file(
        {"top1_token_ids": out.logits[0].float().argmax(-1).to(torch.int32).contiguous()},
        str(out_dir / "reference_top1.safetensors"),
    )

    metadata = {
        "model_name": C.MODEL_NAME,
        "reference": "transformers.models.ministral3 (imported, not vendored) via reference/model.py",
        "weights": args.weights or f"random(seed={args.seed})",
        "token_ids": token_ids[0].tolist(),
        "n_tokens": args.isl,
        "num_layers": hf_config.num_hidden_layers,
        "n_layers": hf_config.num_hidden_layers,
        "num_kv_heads": hf_config.num_key_value_heads,
        "head_dim": hf_config.head_dim,
        "hidden_size": hf_config.hidden_size,
        "intermediate_size": hf_config.intermediate_size,
        "vocab_size": hf_config.vocab_size,
        "sliding_window": hf_config.sliding_window,
        "rope_parameters": dict(hf_config.rope_parameters),
        "dtype": args.dtype,
        "compute_dtype": args.compute_dtype,
        "kv_cache_format": "separate_k_v",
        "kv_convention": "post-RoPE K in HF concat-halves convention; raw V. Permute the DEVICE K "
        "HF->Meta with tt/rope.hf_to_meta_head_permutation before comparing.",
        "prefill_mode": "one_shot",
        "forward_time_seconds": elapsed,
        "reduced": {
            "hidden": hf_config.hidden_size != C.HIDDEN_SIZE,
            "intermediate": hf_config.intermediate_size != C.INTERMEDIATE_SIZE,
            "vocab": hf_config.vocab_size != C.VOCAB_SIZE,
            "layers": hf_config.num_hidden_layers != C.NUM_LAYERS,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # The weights the trace was produced from, so a device run can load exactly them. HF-convention
    # (unswizzled): the device loader applies the Meta swizzle itself.
    torch.save({k: v.float() for k, v in model.state_dict().items()}, out_dir / "reference_weights.pt")

    total_gb = sum(f.stat().st_size for f in kv_dir.glob("*.safetensors")) / (1024**3)
    print(f"[save] {hf_config.num_hidden_layers} layers -> {kv_dir} ({total_gb:.2f} GB)", flush=True)
    print(f"[save] metadata -> {out_dir / 'metadata.json'}", flush=True)
    print(f"[save] weights  -> {out_dir / 'reference_weights.pt'}", flush=True)
    print(f"\nNext: export PREFILL_TRACE_DIR={out_dir}", flush=True)


def write_cache(args):
    """Compute + persist the frozen-key golden cache the host-side tests assert against."""
    spec = golden_cache.GoldenSpec(
        num_layers=args.layers,
        isl=args.isl,
        hidden_size=args.hidden or C.HIDDEN_SIZE,
        intermediate_size=args.intermediate or C.INTERMEDIATE_SIZE,
        vocab_size=args.vocab or C.VOCAB_SIZE,
        seed=args.seed,
    )
    if golden_cache.exists(spec):
        print(f"[cache] already present for {spec.cache_key}; nothing to do", flush=True)
        return
    torch.set_num_threads(os.cpu_count() or 32)
    t0 = time.time()
    golden = golden_cache.compute(spec)
    golden_cache.save(golden)
    print(
        f"[cache] computed + saved {spec.cache_key} in {time.time() - t0:.1f}s "
        f"({len(golden.snapshots)} snapshots, {len(golden.kv)} KV layers)",
        flush=True,
    )


def main():
    args = parse_args()
    raise_cpu_time_limit()
    if args.mode == "cache":
        write_cache(args)
        return 0
    if args.out is None:
        print("ERROR: --mode trace requires --out", file=sys.stderr)
        return 1
    model, hf_config, token_ids, _ = build_model_and_tokens(args)
    write_trace(args, model, hf_config, token_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
