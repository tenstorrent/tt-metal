# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU HuggingFace reference for the prefill accuracy gate (no device).

Runs the HF GPT-OSS model on the frozen real-text prompt (first L tokens) and stores the last-token logits as
GOLDENS_DIR/hf_ref_L{L}.pt. Run once; the gate reads the stored vectors.

    HF_MODEL=/home/ttuser/models/gpt-oss-20b python models/demos/gpt_oss/tests/perf/prefill_hf_ref.py --lengths 128,1024
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.demos.gpt_oss.tests.perf.prefill_common import (  # noqa: E402
    GOLDENS_DIR,
    git_info,
    load_or_build_prompt_tokens,
    parse_lengths,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,1024")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument(
        "--last32",
        action="store_true",
        help="store full-vocab logits of the last 32 positions (L-32..L-1) as GOLDENS_DIR/hf_last32/hf_last32_L{L}.pt "
        "(prefill_gate.py --mode v2) instead of the last-token vector",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="store fp16 full-vocab logits of ALL positions 0..L-1 as GOLDENS_DIR/hf_full/hf_full_L{L}.pt "
        "(prefill_gate.py --mode v3) instead of the last-token vector",
    )
    args = ap.parse_args()
    lengths = parse_lengths(args.lengths)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ["HF_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = load_or_build_prompt_tokens(tokenizer, create=True)

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.config._attn_implementation = "eager"
    model.eval()
    print(f"HF model loaded in {time.time() - t0:.0f}s", flush=True)

    for L in lengths:
        t0 = time.time()
        with torch.no_grad():
            if args.last32:
                # logits_to_keep: the LM head runs on the last 32 positions only.
                out = model(tokens[:L].unsqueeze(0), use_cache=False, logits_to_keep=32)
            else:
                out = model(tokens[:L].unsqueeze(0), use_cache=False)
        if args.last32:
            last32 = out.logits[0, -32:].float()
            assert last32.shape[0] == 32, last32.shape
            path = GOLDENS_DIR / "hf_last32" / f"hf_last32_L{L}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "length": L,
                    "positions": list(range(L - 32, L)),
                    "logits": last32,
                    "dtype": args.dtype,
                    "git": git_info(),
                },
                path,
            )
            print(f"L={L}: forward {time.time() - t0:.0f}s last-32 logits {tuple(last32.shape)} -> {path}", flush=True)
            continue
        if args.full:
            full = out.logits[0].to(torch.float16)  # [L, vocab]; bf16 logits fit fp16 range, fp16 keeps more mantissa
            assert full.shape[0] == L and torch.isfinite(full).all(), (full.shape, L)
            path = GOLDENS_DIR / "hf_full" / f"hf_full_L{L}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"length": L, "logits": full, "dtype": args.dtype, "git": git_info()}, path)
            print(
                f"L={L}: forward {time.time() - t0:.0f}s all-position logits {tuple(full.shape)} -> {path}", flush=True
            )
            continue
        logits = out.logits[0, -1].float()
        top = torch.topk(logits, 10)
        path = GOLDENS_DIR / f"hf_ref_L{L}.pt"
        torch.save(
            {
                "length": L,
                "logits": logits,
                "top10_ids": top.indices,
                "top10_vals": top.values,
                "dtype": args.dtype,
                "git": git_info(),
            },
            path,
        )
        print(
            f"L={L}: forward {time.time() - t0:.0f}s top5={top.indices[:5].tolist()} "
            f"tok={tokenizer.decode(top.indices[:1])!r} -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
