# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model readiness harness for Laguna-XS-2.1 on a P150 D=1/2/4 profile.

Subcommands:
  prefill   — prefill_forward(return_all_logits=True) top-1/5/100 vs the AIME24 reference (faithful
              to models.common.readiness_check.run_prefill_check scoring, but builds a mesh-correct
              page table / cache instead of the shared runner's single-device placeholder).
  teacher   — drives the official models.common.readiness_check.run_teacher_forcing programmatically
              (traced token-out decode; top-1/5/100 + TTFT + decode t/s/u).
  autoreg   — free-running generate for one prompt; writes hf/tt completions + meta for the
              degeneracy check.

Env: cd /tmp, TT_METAL_HOME=<installed tree>, PYTHONPATH=<repo>.
"""
from __future__ import annotations

import argparse
import json

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    MODEL_DIR,
    add_profile_args,
    assert_memory_margin,
    close_mesh,
    open_mesh,
    print_memory_snapshot,
    profile_from_args,
    profile_summary,
)

REF = MODEL_DIR / "tests" / "reference_outputs" / "readiness_aime24_chat.refpt"
QUALITY_BARS = {"top1": 0.90, "top5": 0.98, "top100": 1.0}


def _build(mesh, profile, num_layers=None, host_sampling=False, max_seq_len=None):
    from models.autoports.poolside_laguna_xs_2_1.tt.generator import build_generator

    return build_generator(
        MODEL_DIR,
        mesh,
        num_layers=num_layers,
        host_sampling=host_sampling,
        max_seq_len=max_seq_len or profile.max_context,
    )


def _snapshot(args, mesh, label):
    snapshot = print_memory_snapshot(ttnn, mesh, label)
    if args.enforce_memory_margin:
        assert_memory_margin(snapshot)
    return snapshot


def _build_kwargs(args):
    kwargs = {"max_seq_len": args.max_seq_len or args.profile_spec.max_context}
    if args.num_layers:
        kwargs["num_layers"] = args.num_layers
    return kwargs


def _score(tt_pred, topk_ref):
    """Identical scoring to run_prefill_check: tt_pred [G], topk_ref [G,K]."""
    G = len(tt_pred)
    m1 = m5 = mk = 0
    for i in range(G):
        row = topk_ref[i].tolist()
        pred = int(tt_pred[i])
        if pred == row[0]:
            m1 += 1
        if pred in row[:5]:
            m5 += 1
        if pred in row:
            mk += 1
    return dict(
        top1=m1 / G,
        top5=m5 / G,
        top100=mk / G,
        matches_top1=m1,
        matches_top5=m5,
        matches_top100=mk,
        total=G,
        k=len(topk_ref[0]),
    )


def _assert_quality(rows, label):
    """Turn readiness metrics into an acceptance gate instead of a print-only report."""
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        raise AssertionError(f"{label} produced no accuracy rows")
    for index, stats in enumerate(rows):
        for metric, floor in QUALITY_BARS.items():
            value = float(stats[metric])
            assert value >= floor, f"{label}[{index}] {metric}={value:.4f} < required {floor:.4f}"


def _assert_acceptance_profile(args):
    if not args.acceptance:
        return
    assert args.num_layers is None, "acceptance requires the full 40-layer model"
    requested = args.max_seq_len or args.profile_spec.max_context
    assert requested == args.profile_spec.max_context, (
        f"acceptance requires the exact {args.profile_spec.name} context cap "
        f"{args.profile_spec.max_context}, got {requested}"
    )


def cmd_prefill(args):
    from models.common.readiness_check.schema import load_reference

    ref = load_reference(REF)
    e = ref.entries[0]
    prompt = e.prompt_tokens[0].tolist()
    gen = e.generated_tokens[0].tolist()
    full = prompt + gen
    P, Gn = len(prompt), len(gen)
    mesh = open_mesh(ttnn, args.profile_spec)
    try:
        g = _build(mesh, args.profile_spec, num_layers=args.num_layers, max_seq_len=args.max_seq_len)
        _snapshot(args, mesh, "weights")
        logits = g.prefill_forward(
            tokens=torch.tensor([full], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[len(full)],
            return_all_logits=True,
        )  # [1, full, vocab]
        pred_logits = logits[0, P - 1 : P + Gn - 1, :]
        tt_pred = torch.argmax(pred_logits, dim=-1).tolist()
        stats = _score(tt_pred, e.topk_tokens)
        print("PREFILL", json.dumps(stats))
        _assert_quality(stats, "prefill")
        _snapshot(args, mesh, "prefill_warmup")
    finally:
        try:
            g.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


def cmd_prefill_official(args):
    """Drive the OFFICIAL models.common.readiness_check.run_prefill_check programmatically on the
    selected mesh. Documents whether the shared runner can drive a
    multi-device model given its single-device placeholder page-table setup."""
    from models.common.readiness_check.run_prefill_check import run_prefill_check

    mesh = open_mesh(ttnn, args.profile_spec)
    try:
        per = run_prefill_check(
            model_dir=MODEL_DIR,
            reference_path=REF,
            mesh_device=mesh,
            build_kwargs=_build_kwargs(args),
        )
        print("PREFILL_OFFICIAL", json.dumps(per))
        _assert_quality(per, "official prefill")
        _snapshot(args, mesh, "official_prefill")
    finally:
        close_mesh(ttnn, mesh)


def cmd_teacher(args):
    from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

    mesh = open_mesh(ttnn, args.profile_spec)
    try:
        per = run_teacher_forcing(
            model_dir=MODEL_DIR,
            reference_path=REF,
            mesh_device=mesh,
            build_kwargs=_build_kwargs(args),
        )
        print("TEACHER", json.dumps(per))
        _assert_quality(per, "teacher forcing")
        _snapshot(args, mesh, "teacher_forcing")
    finally:
        close_mesh(ttnn, mesh)


def cmd_autoreg(args):
    from models.common.readiness_check.schema import load_reference

    ref = load_reference(REF)
    e = ref.entries[0]
    prompt = e.prompt_tokens[0].tolist()
    mesh = open_mesh(ttnn, args.profile_spec)
    try:
        g = _build(mesh, args.profile_spec, num_layers=args.num_layers, max_seq_len=args.max_seq_len)
        _snapshot(args, mesh, "weights")
        tt_ids = g.generate(prompt, args.gen_len, next_input=None, enable_trace=True, stop_on_eos=True)
        _snapshot(args, mesh, "trace_warmup")
        tok = g.tokenizer
        tt_text = tok.decode(tt_ids, skip_special_tokens=False)
        outdir = MODEL_DIR / args.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "tt_completion.txt").write_text(tt_text)
        hf_ids = ref.entries[0].generated_tokens[0].tolist()
        (outdir / "hf_completion.txt").write_text(tok.decode(hf_ids, skip_special_tokens=False))
        meta = {
            "hf_model_id": ref.hf_model_id,
            "prompt_text": e.prompt_text,
            "prompt_token_ids": prompt,
            "max_new_tokens": args.gen_len,
            "hf": {"token_ids": hf_ids, "num_tokens": len(hf_ids)},
            "tt": {"token_ids": tt_ids, "num_tokens": len(tt_ids)},
            "counters": dict(g.counters),
        }
        (outdir / "autoregressive_meta.json").write_text(json.dumps(meta, indent=2))
        print("AUTOREG tt_tokens", len(tt_ids), "first16", tt_ids[:16])
        print("AUTOREG counters", dict(g.counters))
    finally:
        try:
            g.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


def cmd_prefill_autoreg(args):
    """Prefill top-k + free-running autoregressive + qualitative artifacts in ONE model build."""
    from models.common.readiness_check.schema import load_reference

    ref = load_reference(REF)
    e = ref.entries[0]
    prompt = e.prompt_tokens[0].tolist()
    gen_ref = e.generated_tokens[0].tolist()
    full = prompt + gen_ref
    P, Gn = len(prompt), len(gen_ref)
    outdir = MODEL_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    mesh = open_mesh(ttnn, args.profile_spec)
    try:
        g = _build(mesh, args.profile_spec, num_layers=args.num_layers, max_seq_len=args.max_seq_len)
        _snapshot(args, mesh, "weights")
        # --- prefill top-k ---
        logits = g.prefill_forward(
            tokens=torch.tensor([full], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[len(full)],
            return_all_logits=True,
        )
        pred_logits = logits[0, P - 1 : P + Gn - 1, :]
        tt_pred = torch.argmax(pred_logits, dim=-1).tolist()
        pstats = _score(tt_pred, e.topk_tokens)
        print("PREFILL", json.dumps(pstats))
        _assert_quality(pstats, "prefill")
        _snapshot(args, mesh, "prefill_warmup")
        (outdir / "prefill_accuracy.json").write_text(json.dumps(pstats, indent=2))

        # --- free-running autoregressive ---
        g.reset()
        tt_ids = g.generate(prompt, args.gen_len, next_input=None, enable_trace=True, stop_on_eos=True)
        _snapshot(args, mesh, "trace_warmup")
        tok = g.tokenizer
        tt_text = tok.decode(tt_ids, skip_special_tokens=False)
        hf_text = tok.decode(gen_ref, skip_special_tokens=False)
        (outdir / "tt_completion.txt").write_text(tt_text)
        (outdir / "hf_completion.txt").write_text(hf_text)
        meta = {
            "hf_model_id": ref.hf_model_id,
            "prompt_text": e.prompt_text,
            "prompt_token_ids": prompt,
            "max_new_tokens": args.gen_len,
            "hf": {"token_ids": gen_ref, "num_tokens": len(gen_ref)},
            "tt": {"token_ids": tt_ids, "num_tokens": len(tt_ids)},
            "counters": dict(g.counters),
        }
        (outdir / "autoregressive_meta.json").write_text(json.dumps(meta, indent=2))
        fmt = {
            "hf_model_id": ref.hf_model_id,
            "tokenizer_class": type(tok).__name__,
            "chat_template_present": bool(getattr(tok, "chat_template", None)),
            "prompt_mode": "chat",
            "rendering": "tokenizer.apply_chat_template(add_generation_prompt=True) via AIME24 reference",
            "gen_params": "greedy (top-k k=1 on-device split sampling)",
            "prompt_source": "models/demos/deepseek_v3/demo/aime_under_8k_prompts.json[0]",
        }
        (outdir / "qualitative_prompt_format.json").write_text(json.dumps(fmt, indent=2))
        agree = sum(int(a == b) for a, b in zip(tt_ids, gen_ref))
        print("AUTOREG tt_tokens", len(tt_ids), "hf-agree", f"{agree}/{min(len(tt_ids), len(gen_ref))}")
        print("AUTOREG counters", dict(g.counters))
        print("TT_HEAD:", repr(tt_text[:220]))
        print("HF_HEAD:", repr(hf_text[:220]))
    finally:
        try:
            g.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefill", "prefill_official", "teacher", "autoreg", "prefill_autoreg"])
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--gen-len", type=int, default=100)
    ap.add_argument(
        "--acceptance",
        action="store_true",
        help="require the full 40-layer model at the selected profile's exact context cap",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="model/KV addressability (default: selected profile's qualified context)",
    )
    ap.add_argument(
        "--enforce-memory-margin",
        action="store_true",
        help="fail unless the final DRAM snapshot has 10%% free and 128 MiB contiguous per bank",
    )
    add_profile_args(ap)
    ap.add_argument(
        "--outdir",
        type=str,
        default="doc/full_model",
        help="Artifact output dir relative to the model dir (autoreg/prefill_autoreg write here).",
    )
    a = ap.parse_args()
    a.profile_spec = profile_from_args(a)
    _assert_acceptance_profile(a)
    print("PROFILE", json.dumps(profile_summary(a.profile_spec), sort_keys=True))
    {
        "prefill": cmd_prefill,
        "prefill_official": cmd_prefill_official,
        "teacher": cmd_teacher,
        "autoreg": cmd_autoreg,
        "prefill_autoreg": cmd_prefill_autoreg,
    }[a.cmd](a)
    # The mesh is already closed in each cmd's finally; skip the multi-minute interpreter GC of the
    # full-model tensor graph (which otherwise hangs after results are printed).
    import os as _os
    import sys as _sys

    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
