# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``publish-hf`` — publish a tool-optimized model to Hugging Face.

Assembles a model repo from an optimize run: the optimized model implementation (the demo dir, which
holds the committed wins after ``commit-wins``) plus an auto-generated model card whose perf table is
drawn straight from the run's own metrics (throughput per-user, batch, TTFT/TPOT/E2E, PCC gate,
tt-metal commit). Weights are REFERENCED (``base_model`` pointer), never re-uploaded.

Pushes with ``huggingface_hub`` (token from ``--token`` / ``HF_TOKEN`` / the CLI login file). Use
``--dry-run`` to stage + preview the card without pushing. A ``tt-model.yaml`` manifest is written so
the repo can later be turned into a servable ``tt-model`` bundle.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..optimize_dashboard import (
    collect_state,
    find_run_dir,
    repo_root_for_run,
    run_slug,
    state_dir_candidates,
)
from .optimize import _repo_root, _resolve_target


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _fmt(v, nd=2) -> str:
    return "—" if v is None else f"{float(v):.{nd}f}"


def _build_card(state: dict, slug: str, base_weights: str | None, commit: str | None) -> str:
    cfg = state.get("config") or {}
    thr = state.get("throughput") or {}
    sv = state.get("serving") or {}
    m = state.get("metric") or {}
    batch = state.get("batch")
    pt = sv.get("per_token") or {}
    ft = sv.get("first_token") or {}
    e2 = sv.get("e2e_latency") or {}

    L: list[str] = ["---"]
    if base_weights:
        L.append(f"base_model: {base_weights}")
        L.append("base_model_relation: finetune")
    L += [
        "library_name: tt-metal",
        "pipeline_tag: text-generation",
        "tags:",
        "  - tenstorrent",
        "  - tt-metal",
        "  - ttnn",
        "  - blackhole",
        "---",
        "",
        f"# {slug} — Tenstorrent-optimized",
        "",
        "Optimized for Tenstorrent hardware with `tt_hw_planner` (kernel/lever autotuning "
        "against a PCC-gated end-to-end pipeline).",
        "",
    ]
    if base_weights:
        L.append(f"- **Base weights:** [{base_weights}](https://huggingface.co/{base_weights})")
    if commit:
        L.append(f"- **tt-metal commit:** `{commit}`")
    if cfg.get("pcc_test"):
        L.append(f"- **Accuracy gate:** `{str(cfg['pcc_test']).split('::')[-1]}` (PCC)")
    if cfg.get("devices"):
        L.append(f"- **Devices:** {cfg['devices']}")
    L += ["", "## Performance", "", "| Metric | Value |", "| --- | --- |"]
    if batch is not None:
        L.append(f"| Batch (concurrent users) | {batch} |")
    if thr.get("current") is not None:
        L.append(f"| Throughput | {_fmt(thr['current'])} tok/s/user |")
        if thr.get("baseline"):
            gain = (thr["current"] - thr["baseline"]) / thr["baseline"] * 100.0
            L.append(
                f"| Throughput vs. baseline | {_fmt(thr['baseline'])} → {_fmt(thr['current'])} tok/s/user (+{gain:.0f}%) |"
            )
    if ft.get("ms") is not None:
        L.append(f"| TTFT | {_fmt(ft['ms'], 1)} ms |")
    if pt.get("ms") is not None:
        L.append(f"| TPOT / ITL | {_fmt(pt['ms'], 1)} ms |")
    if e2.get("ms") is not None:
        L.append(f"| E2E latency | {_fmt(e2['ms'], 1)} ms |")
    if m.get("name") and m.get("current") is not None:
        L.append(f"| {m['name']} | {_fmt(m.get('current'), 1)} {m.get('unit', '')} |")
    L += [
        "",
        "## Run on Tenstorrent",
        "",
        "```bash",
        "# vLLM, OpenAI-compatible API via tt-inference-server, from an optimized tt-metal checkout:",
        f"python3 run.py --model {slug} --tt-device <device> --workflow server \\",
        "  --local-server --tt-metal-home <your tt-metal checkout>",
        "```",
        "",
        "_Published with `tt_hw_planner publish-hf`. The optimized model implementation lives under "
        "`models/`; weights are referenced from the base repo above (not stored here)._",
        "",
    ]
    return "\n".join(L)


def cmd_publish_hf(args) -> int:
    repo_root = _repo_root()
    slug = None
    demo_dir = None
    target = getattr(args, "target", None)
    if target:
        demo_dir = _resolve_target(target, repo_root)
        if demo_dir is not None:
            slug = demo_dir.name

    run_dir = find_run_dir(repo_root, slug=slug, run_ref=getattr(args, "run", None))
    if run_dir is None:
        what = f"for '{slug}' " if slug else ""
        print(f"  [publish-hf] no optimize run found {what}under {repo_root}. Pass a target or --run.")
        return 2
    slug = slug or run_slug(run_dir)
    state_root = repo_root_for_run(run_dir, repo_root)
    state = collect_state(run_dir, state_dir_candidates(state_root, slug), slug)

    # The publishable model source is the demo dir in the MAIN checkout — after `commit-wins` it holds
    # the optimized code. Fall back to the run's own model_root only if the checkout lacks it.
    if demo_dir is None:
        try:
            from ..bringup_loop import find_demo_dir

            d = find_demo_dir(slug, repo_root)
            demo_dir = d.resolve() if d else None
        except Exception:
            demo_dir = None
    if demo_dir is None or not Path(demo_dir).is_dir():
        mr = (state.get("model") or {}).get("root")
        if mr and Path(mr).is_dir():
            demo_dir = Path(mr)
    if demo_dir is None or not Path(demo_dir).is_dir():
        print(f"  [publish-hf] could not locate the model demo dir for '{slug}'. Run `commit-wins` first.")
        return 2

    commit = _git_commit(state_root)
    base_weights = getattr(args, "weights", None)
    card = _build_card(state, slug, base_weights, commit)

    stage = Path(tempfile.mkdtemp(prefix="tt_publish_"))
    (stage / "README.md").write_text(card)
    manifest = {
        "repo": args.repo,
        "name": slug,
        "source": {"tt_metal": str(state_root), "code": [f"models/demos/{slug}", "models/common"]},
        "weights": base_weights,
        "serve": {"block_size": 64, "max_num_seqs": state.get("batch") or 32},
        "provenance": {
            "tt_metal_commit": commit,
            "run": run_dir.name,
            "throughput_tok_s_user": (state.get("throughput") or {}).get("current"),
            "batch": state.get("batch"),
        },
    }
    (stage / "tt-model.yaml").write_text(json.dumps(manifest, indent=2))

    if not getattr(args, "card_only", False):
        dst = stage / "models" / "demos" / slug
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            demo_dir,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.pt", "*.bin", "*.safetensors"),
        )

    print(f"  [publish-hf] repo:   {args.repo}")
    print(f"  [publish-hf] model:  {slug}   (source: {demo_dir})")
    print(
        f"  [publish-hf] commit: {commit}   batch: {state.get('batch')}   "
        f"throughput: {_fmt((state.get('throughput') or {}).get('current'))} tok/s/user"
    )
    print(f"  [publish-hf] staged: {stage}")

    if getattr(args, "dry_run", False):
        print("  [publish-hf] --dry-run: NOT pushing. Model card preview:")
        print("  " + "-" * 60)
        for ln in card.splitlines():
            print("  | " + ln)
        print("  " + "-" * 60)
        return 0

    try:
        from huggingface_hub import create_repo, upload_folder
    except Exception:
        print(
            "  [publish-hf] huggingface_hub is not installed. `pip install huggingface_hub`, "
            "or re-run with --dry-run to preview."
        )
        return 3

    token = getattr(args, "token", None) or os.environ.get("HF_TOKEN")
    try:
        create_repo(
            args.repo, repo_type="model", private=bool(getattr(args, "private", False)), exist_ok=True, token=token
        )
        upload_folder(
            repo_id=args.repo,
            folder_path=str(stage),
            repo_type="model",
            token=token,
            commit_message=f"Publish {slug} (tt_hw_planner; tt-metal {commit})",
        )
    except Exception as e:
        print(f"  [publish-hf] push failed: {e}")
        return 4

    url = f"https://huggingface.co/{args.repo}"
    print(f"  [publish-hf] published: {url}")
    return 0
