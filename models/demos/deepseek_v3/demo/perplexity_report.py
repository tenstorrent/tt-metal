# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Post-demo perplexity report: score each TT generation with an external judge LM.

Writes ``<demo_artifact>.json`` sibling ``<demo_artifact>_perplexity.json`` and prints
``PERPLEXITY_SUMMARY`` lines for log scraping. Triggered when ``DEEPSEEK_PPL_REPORT``
is enabled (see ``maybe_write_demo_perplexity_report``).
"""

from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import statistics
from pathlib import Path

import torch
from loguru import logger


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _resolve_judge_device() -> torch.device:
    raw = os.getenv("DEEPSEEK_PPL_JUDGE_DEVICE", "cuda")
    if raw.lower() == "cuda" and not torch.cuda.is_available():
        logger.warning("DEEPSEEK_PPL_JUDGE_DEVICE=cuda but CUDA is not available; using CPU for judge")
        return torch.device("cpu")
    return torch.device(raw)


def _global_rank_from_env() -> int | None:
    """Best-effort distributed global rank detection across MPI launchers."""
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK", "RANK", "SLURM_PROCID"):
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid {}={!r}; expected integer rank", key, raw)
    return None


def _continuation_for_generation(
    gen: dict,
    *,
    ds_tokenizer=None,
) -> str:
    text = gen.get("text")
    if isinstance(text, str) and text.strip():
        return text
    toks = gen.get("tokens") or []
    if not toks or ds_tokenizer is None:
        return ""
    return ds_tokenizer.decode([int(t) for t in toks], skip_special_tokens=True)


def maybe_write_demo_perplexity_report(
    *,
    prompts: list[str],
    results: dict,
    demo_json_path: Path,
    model_path: Path,
) -> Path | None:
    """
    If ``DEEPSEEK_PPL_REPORT`` is set, load the judge LM and write perplexity for
    each entry in ``results["generations"]`` (paired with ``prompts[i]``).

    Returns the path to ``*_perplexity.json``, or ``None`` if skipped.
    """
    if not _env_truthy("DEEPSEEK_PPL_REPORT", "0"):
        return None
    reporter_rank = _global_rank_from_env()
    if reporter_rank is not None and reporter_rank != 0:
        logger.info("DEEPSEEK_PPL_REPORT: skipping on non-primary distributed rank {}", reporter_rank)
        return None

    from models.demos.deepseek_v3.demo.perplexity_judge import (
        DEFAULT_JUDGE_MODEL,
        load_judge_lm,
        mean_nll_continuation_with_prompt,
        perplexity_from_mean_nll,
    )

    generations = results.get("generations") or []
    if not generations:
        logger.warning("DEEPSEEK_PPL_REPORT set but no generations to score")
        return None
    if len(prompts) != len(generations):
        logger.warning(
            "DEEPSEEK_PPL_REPORT: prompts/generations length mismatch (prompts={}, generations={}); "
            "scoring uses index-aligned first {} prompts.",
            len(prompts),
            len(generations),
            len(generations),
        )

    judge_id = os.getenv("DEEPSEEK_PPL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    out_path = demo_json_path.with_name(f"{demo_json_path.stem}_perplexity.json")

    try:
        judge_device = _resolve_judge_device()
        judge_model, judge_tokenizer, _ = load_judge_lm(model_id=judge_id, device=judge_device)
    except Exception as exc:
        logger.error("DEEPSEEK_PPL_REPORT: failed to load judge '{}': {}", judge_id, exc)
        raise

    needs_decode_fallback = any(not (isinstance(g.get("text"), str) and g.get("text").strip()) for g in generations)
    if needs_decode_fallback:
        from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer

        ds_tokenizer = load_tokenizer(str(model_path))
    else:
        ds_tokenizer = None
    rows: list[dict] = []
    nlls: list[float] = []

    for i, gen in enumerate(generations):
        prompt = prompts[i] if i < len(prompts) else ""
        cont = _continuation_for_generation(gen, ds_tokenizer=ds_tokenizer)
        prompt_preview = prompt[:120].replace("\n", " ")
        continuation_preview = cont[:120].replace("\n", " ")
        if not cont.strip():
            rows.append(
                {
                    "index": i + 1,
                    "prompt_preview": prompt_preview,
                    "continuation_preview": continuation_preview,
                    "mean_nll": None,
                    "perplexity": None,
                    "skipped": True,
                    "reason": "empty_continuation",
                }
            )
            continue
        nll = mean_nll_continuation_with_prompt(
            judge_model,
            judge_tokenizer,
            judge_device,
            prompt=prompt,
            continuation=cont,
        )
        ppl = perplexity_from_mean_nll(nll)
        rows.append(
            {
                "index": i + 1,
                "prompt_preview": prompt_preview,
                "continuation_preview": continuation_preview,
                "mean_nll": nll if math.isfinite(nll) else None,
                "perplexity": ppl if math.isfinite(ppl) else None,
                "skipped": not math.isfinite(nll),
                "continuation_chars": len(cont),
            }
        )
        if math.isfinite(nll):
            nlls.append(nll)
        ppl_s = f"{ppl:.2f}" if math.isfinite(ppl) else "nan"
        print(
            f"PERPLEXITY_ROW index={i + 1} judge={judge_id!r} mean_nll={nll:.4f} perplexity={ppl_s}",
            flush=True,
        )

    mean_nll_all = statistics.mean(nlls) if nlls else float("nan")
    # exp(mean NLL) equals the geometric mean of per-row perplexities (since ppl_i = exp(nll_i)).
    ppl_geom = math.exp(mean_nll_all) if nlls and math.isfinite(mean_nll_all) else float("nan")
    ppl_arith_mean = statistics.mean([math.exp(n) for n in nlls]) if nlls else float("nan")

    payload = {
        "judge_model": judge_id,
        "judge_device": str(judge_device),
        "reporter_rank": reporter_rank,
        "demo_json": str(demo_json_path),
        "demo_json_resolved": str(demo_json_path.resolve()),
        "prompt_count": len(prompts),
        "generation_count": len(generations),
        "per_prompt": rows,
        "aggregate": {
            "count_scored": len(nlls),
            "count_generations": len(generations),
            "mean_nll": mean_nll_all if nlls and math.isfinite(mean_nll_all) else None,
            "perplexity_exp_mean_nll": ppl_geom if nlls and math.isfinite(ppl_geom) else None,
            "mean_perplexity": ppl_arith_mean if nlls and math.isfinite(ppl_arith_mean) else None,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_name(f"{out_path.name}.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        try:
            # Extra defense against accidental multi-writer races.
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning("DEEPSEEK_PPL_REPORT: another process is writing {}; skipping", out_path)
            return None
        tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    summary = (
        f"PERPLEXITY_SUMMARY judge={judge_id!r} rank={reporter_rank} scored={len(nlls)}/{len(generations)} "
        f"mean_nll={mean_nll_all:.4f} perplexity_exp_mean_nll={ppl_geom:.2f} mean_perplexity={ppl_arith_mean:.2f} "
        f"path={out_path}"
    )
    print(summary, flush=True)
    logger.info(summary)

    del judge_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path
