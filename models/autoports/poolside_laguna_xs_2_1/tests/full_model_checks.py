# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model readiness harness for Laguna-XS-2.1 on the 1×4 Blackhole mesh.

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
from pathlib import Path

import torch

import ttnn

MODEL_DIR = Path("/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1")
REF = MODEL_DIR / "readiness_aime24_chat.refpt"


def _open_mesh():
    # Mesh shape is env-parameterized so the same gates run on the 1x4 mesh (default) or a smaller
    # mesh (e.g. TT_LAGUNA_MESH="1,2" for a 2-chip host). The shard factors in model.py/decoder derive
    # from get_num_devices(), so any factor of the head/expert/vocab dims works.
    import os as _os

    r, c = (int(x) for x in _os.environ.get("TT_LAGUNA_MESH", "1,4").split(","))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    return ttnn.open_mesh_device(ttnn.MeshShape(r, c), trace_region_size=1_500_000_000)


def _close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _build(mesh, num_layers=None, host_sampling=False, max_seq_len=16384):
    from models.autoports.poolside_laguna_xs_2_1.tt.generator import build_generator

    return build_generator(MODEL_DIR, mesh, num_layers=num_layers, host_sampling=host_sampling, max_seq_len=max_seq_len)


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


def cmd_prefill(args):
    from models.common.readiness_check.schema import load_reference

    ref = load_reference(REF)
    e = ref.entries[0]
    prompt = e.prompt_tokens[0].tolist()
    gen = e.generated_tokens[0].tolist()
    full = prompt + gen
    P, Gn = len(prompt), len(gen)
    mesh = _open_mesh()
    try:
        g = _build(mesh, num_layers=args.num_layers)
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
    finally:
        try:
            g.teardown()
        except Exception:
            pass
        _close_mesh(mesh)


def cmd_prefill_official(args):
    """Drive the OFFICIAL models.common.readiness_check.run_prefill_check programmatically on the
    (1,4) mesh (its CLI has no 1x4 label). Documents whether the shared runner can drive a
    multi-device model given its single-device placeholder page-table setup."""
    from models.common.readiness_check.run_prefill_check import run_prefill_check

    mesh = _open_mesh()
    try:
        per = run_prefill_check(
            model_dir=MODEL_DIR,
            reference_path=REF,
            mesh_device=mesh,
            build_kwargs={"num_layers": args.num_layers} if args.num_layers else {},
        )
        print("PREFILL_OFFICIAL", json.dumps(per))
    finally:
        _close_mesh(mesh)


def cmd_teacher(args):
    from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

    mesh = _open_mesh()
    try:
        per = run_teacher_forcing(
            model_dir=MODEL_DIR,
            reference_path=REF,
            mesh_device=mesh,
            build_kwargs={"num_layers": args.num_layers} if args.num_layers else {},
        )
        print("TEACHER", json.dumps(per))
    finally:
        _close_mesh(mesh)


def cmd_autoreg(args):
    from models.common.readiness_check.schema import load_reference

    ref = load_reference(REF)
    e = ref.entries[0]
    prompt = e.prompt_tokens[0].tolist()
    mesh = _open_mesh()
    try:
        g = _build(mesh, num_layers=args.num_layers)
        tt_ids = g.generate(prompt, args.gen_len, next_input=None, enable_trace=True, stop_on_eos=True)
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
        _close_mesh(mesh)


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
    mesh = _open_mesh()
    try:
        g = _build(mesh, num_layers=args.num_layers)
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
        (outdir / "prefill_accuracy.json").write_text(json.dumps(pstats, indent=2))

        # --- free-running autoregressive ---
        g.reset()
        tt_ids = g.generate(prompt, args.gen_len, next_input=None, enable_trace=True, stop_on_eos=True)
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
        _close_mesh(mesh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefill", "prefill_official", "teacher", "autoreg", "prefill_autoreg"])
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--gen-len", type=int, default=100)
    ap.add_argument(
        "--outdir",
        type=str,
        default="doc/full_model",
        help="Artifact output dir relative to the model dir (autoreg/prefill_autoreg write here). "
        "Default doc/full_model preserves stage-05; pass doc/optimized_full_model for the BFP8 stage.",
    )
    a = ap.parse_args()
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
