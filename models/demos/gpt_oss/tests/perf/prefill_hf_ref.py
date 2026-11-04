# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU HuggingFace reference for the prefill accuracy gate (no device).

Runs the HF GPT-OSS model on the frozen real-text prompt (first L tokens) and stores the last-token logits as
GOLDENS_DIR/hf_ref_L{L}.pt. Run once; the gate reads the stored vectors.

    HF_MODEL=/path/to/gpt-oss-20b python models/demos/gpt_oss/tests/perf/prefill_hf_ref.py --lengths 128,1024

--full / --last / --last32 / --v4 store multi-position references for prefill_gate.py --mode v2 / v3 / v4.
"""

import argparse
import os
import re
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

# Gate v4 prompts (prefill_gate.py --mode v4): one 1024-token window of each of these prose Markdown docs, tokenized
# like the v3 stream (encode, add_special_tokens=False). Each window starts at the doc's first line with >= 20 words
# (skips titles, tables of contents and front matter) and is frozen in GOLDENS_DIR/hf_v4/prompts_v4.pt, so later edits
# to the docs never change the gate. No doc is part of the v3 stream; no two windows share an 8-gram (checked).
V4_DOCS = (
    "METALIUM_GUIDE.md",
    "CONTRIBUTING.md",
    "tech_reports/SubDevices/SubDevices.md",
    "ttnn/cpp/ttnn/operations/reduction/topk/docs/TopK.md",
    "tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md",
    "tech_reports/TT-Distributed/MultiHostMeshRuntime.md",
    "tt_metal/tt-llk/docs/llk/l2/top_level_overview.md",
    "tech_reports/TT-Fabric/TT-Fabric-Architecture.md",
)
V4_PROMPT_LEN = 1024
V4_MIN_WORDS = 20


def _ngrams(ids, n=8):
    return {tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)}


def load_or_build_v4_prompts(tokenizer):
    """GOLDENS_DIR/hf_v4/prompts_v4.pt: {'tokens': [8, 1024] long, 'docs': [per-doc metadata], ...}."""
    path = GOLDENS_DIR / "hf_v4" / "prompts_v4.pt"
    if path.exists():
        return torch.load(path)
    root = Path(__file__).resolve().parents[5]
    v3_stream = _ngrams(load_or_build_prompt_tokens().tolist())
    prompts, docs = [], []
    for d in V4_DOCS:
        text = (root / d).read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        start = next(i for i, line in enumerate(lines) if len(re.findall(r"[A-Za-z]+", line)) >= V4_MIN_WORDS)
        start_char = sum(len(line) for line in lines[:start])
        ids = tokenizer.encode(text[start_char:], add_special_tokens=False)
        assert len(ids) >= V4_PROMPT_LEN, (d, len(ids))
        window = ids[:V4_PROMPT_LEN]
        prompts.append(torch.tensor(window, dtype=torch.long))
        docs.append(
            {
                "file": d,
                "start_line": start + 1,
                "start_char": start_char,
                "doc_tokens_from_start": len(ids),
                "v3_stream_8gram_overlap": round(len(_ngrams(window) & v3_stream) / (V4_PROMPT_LEN - 7), 4),
            }
        )
    grams = [_ngrams(p.tolist()) for p in prompts]
    shared = max(len(grams[i] & grams[j]) for i in range(len(grams)) for j in range(i + 1, len(grams)))
    assert shared == 0, f"v4 prompt windows share {shared} 8-grams"
    record = {"tokens": torch.stack(prompts), "docs": docs, "length": V4_PROMPT_LEN, "git": git_info()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    print(f"wrote {len(prompts)} v4 prompts -> {path}", flush=True)
    return record


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
    ap.add_argument(
        "--last",
        type=int,
        default=0,
        help="with --full: keep only the last N positions (L-N..L-1; the LM head runs on those rows only) and store "
        "them as GOLDENS_DIR/hf_full/hf_full_L{L}_last{N}.pt (the gate v3 long window, --v3-window L:N)",
    )
    ap.add_argument(
        "--v4",
        action="store_true",
        help="gate v4: build (once) the 8 independent 1024-token prompts GOLDENS_DIR/hf_v4/prompts_v4.pt from V4_DOCS "
        "and store fp16 full-vocab logits of all their positions as GOLDENS_DIR/hf_v4/hf_v4_p{i}.pt (--lengths unused)",
    )
    args = ap.parse_args()
    assert args.last == 0 or args.full, "--last needs --full"
    lengths = [] if args.v4 else parse_lengths(args.lengths)

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

    if args.v4:
        v4 = load_or_build_v4_prompts(tokenizer)
        for i, (toks, doc) in enumerate(zip(v4["tokens"], v4["docs"])):
            t0 = time.time()
            with torch.no_grad():
                out = model(toks.unsqueeze(0), use_cache=False)
            full = out.logits[0].to(torch.float16)
            assert full.shape[0] == toks.numel() and torch.isfinite(full).all(), full.shape
            path = GOLDENS_DIR / "hf_v4" / f"hf_v4_p{i}.pt"
            record = {"prompt": i, "doc": doc["file"], "tokens": toks.clone(), "logits": full, "dtype": args.dtype}
            torch.save({**record, "git": git_info()}, path)
            print(f"v4 p{i} {doc['file']}: forward {time.time() - t0:.0f}s {tuple(full.shape)} -> {path}", flush=True)

    for L in lengths:
        t0 = time.time()
        with torch.no_grad():
            if args.last32:
                # logits_to_keep: the LM head runs on the last 32 positions only.
                out = model(tokens[:L].unsqueeze(0), use_cache=False, logits_to_keep=32)
            elif args.full and args.last:
                out = model(tokens[:L].unsqueeze(0), use_cache=False, logits_to_keep=args.last)
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
            rows = args.last or L
            assert full.shape[0] == rows and torch.isfinite(full).all(), (full.shape, L, rows)
            name = f"hf_full_L{L}_last{rows}.pt" if args.last else f"hf_full_L{L}.pt"
            path = GOLDENS_DIR / "hf_full" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"length": L, "logits": full, "dtype": args.dtype, "git": git_info()}
            if args.last:
                record["positions"] = [L - rows, L]  # half-open row range of the prompt
            torch.save(record, path)
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
