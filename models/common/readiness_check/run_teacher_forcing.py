# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run the teacher-forcing portion of the model-readiness check.

Given a model directory containing a `tt/generator.py` that satisfies the
contract in `contract.py`, and a reference file produced by
`models.common.readiness_check.generate`, this runner:

  1. Builds the generator via `<model_dir>/tt/generator.py::build_generator`.
  2. For each entry in the reference, requires `generator.generate()` to
     explicitly accept `enable_trace` and calls
     `generator.generate(..., next_input=acc.collect_predicted_tokens,
     enable_trace=True)`.
  3. Resets the generator between entries.
  4. Prints per-entry and aggregate top-1 / top-5 / top-K accuracy plus
     directly timed TTFT and traced decode t/s/u.

CLI:
    python -m models.common.readiness_check.run_teacher_forcing \\
        --model-dir models/autoports/<model_name> \\
        --reference models/common/readiness_check/references/<model>.refpt \\
        --mesh-device N150
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from models.common.readiness_check.contract import (
    BUILD_GENERATOR_FUNCTION_NAME,
    GENERATOR_MODULE_RELPATH,
    BuildGeneratorFn,
    Generator,
)
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)
from models.common.readiness_check.teacher_forcing import TokenAccuracy


def _import_build_generator(model_dir: Path) -> BuildGeneratorFn:
    """
    Load `<model_dir>/tt/generator.py` and return its `build_generator`
    function. Uses importlib.util.spec_from_file_location so the model
    does not need to be on sys.path or a proper package.
    """
    generator_path = model_dir / GENERATOR_MODULE_RELPATH
    if not generator_path.exists():
        raise FileNotFoundError(
            f"Expected generator at {generator_path}. The readiness check requires "
            f"<model_dir>/{GENERATOR_MODULE_RELPATH} to exist and expose `{BUILD_GENERATOR_FUNCTION_NAME}`."
        )

    # Stable module name so repeat imports hit the cache.
    module_name = f"_readiness_generator_{model_dir.resolve().name}"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    fn = getattr(module, BUILD_GENERATOR_FUNCTION_NAME, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{generator_path} does not export a callable `{BUILD_GENERATOR_FUNCTION_NAME}`. "
            f"See models/common/readiness_check/contract.py."
        )
    return fn  # type: ignore[return-value]


def _run_one_entry(
    *,
    generator: Generator,
    acc: TokenAccuracy,
    entry_idx: int,
) -> Dict[str, Any]:
    prompt_ids = acc.get_prompt_token_ids(entry_idx)
    n_steps = acc.num_gt_tokens(entry_idx)
    timing: Dict[str, Any] = {
        "start_s": None,
        "first_token_s": None,
        "last_decode_token_s": None,
        "callback_count": 0,
    }

    def next_input(step: int, predicted: int) -> int:
        now = time.perf_counter()
        if timing["first_token_s"] is None:
            timing["first_token_s"] = now
        else:
            timing["last_decode_token_s"] = now
        timing["callback_count"] += 1
        return acc.collect_predicted_tokens(predicted, user_idx=entry_idx)

    _require_explicit_generate_kwarg(generator, "enable_trace")

    timing["start_s"] = time.perf_counter()
    generator.generate(
        prompt_token_ids=prompt_ids,
        max_new_tokens=n_steps,
        next_input=next_input,
        enable_trace=True,
    )
    end_s = time.perf_counter()

    _require_full_teacher_forcing(
        acc=acc,
        entry_idx=entry_idx,
        expected_tokens=n_steps,
        callback_count=int(timing["callback_count"]),
    )

    stats = acc.compute_accuracy(user_idx=entry_idx)
    stats.update(_compute_perf_stats(timing=timing, end_s=end_s, token_count=stats["total"]))
    generate_stats = getattr(generator, "last_generate_stats", None)
    if generate_stats:
        stats["generate_stats"] = generate_stats
    return stats


def _require_full_teacher_forcing(
    *,
    acc: TokenAccuracy,
    entry_idx: int,
    expected_tokens: int,
    callback_count: int,
) -> None:
    predicted_tokens = acc.num_pred_tokens(entry_idx)
    if predicted_tokens == expected_tokens and callback_count == expected_tokens:
        return

    raise RuntimeError(
        f"Teacher-forcing run for entry[{entry_idx}] produced {predicted_tokens}/{expected_tokens} "
        f"predictions via {callback_count} next_input callback(s). The readiness reference covers the "
        "full generated length, so generate() must call next_input once per requested token and must not "
        "stop early during teacher forcing."
    )


def _require_explicit_generate_kwarg(generator: Generator, name: str) -> None:
    signature = inspect.signature(generator.generate)
    parameter = signature.parameters.get(name)
    if parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return

    raise TypeError(
        f"{generator.__class__.__name__}.generate() must explicitly declare an `{name}` keyword "
        "and the readiness teacher-forcing runner always calls it with `enable_trace=True`. "
        "A catch-all `**kwargs` parameter is not sufficient because it can silently ignore the "
        "tracing requirement."
    )


def _compute_perf_stats(*, timing: Dict[str, Any], end_s: float, token_count: int) -> Dict[str, float]:
    start_s = timing["start_s"]
    first_token_s = timing["first_token_s"]
    elapsed_s = max(end_s - start_s, 0.0)
    perf: Dict[str, float] = {
        "elapsed_s": elapsed_s,
        "e2e_t/s/u": (token_count / elapsed_s) if elapsed_s > 0 else 0.0,
    }

    if first_token_s is None:
        return perf

    ttft_s = max(first_token_s - start_s, 0.0)
    perf["ttft_ms"] = ttft_s * 1000.0

    decode_tokens = max(token_count - 1, 0)
    if decode_tokens > 0:
        decode_end_s = timing["last_decode_token_s"] if timing["last_decode_token_s"] is not None else end_s
        decode_elapsed_s = max(decode_end_s - first_token_s, 0.0)
        perf["decode_tokens"] = float(decode_tokens)
        perf["decode_elapsed_s"] = decode_elapsed_s
        perf["decode_t/s/u"] = (decode_tokens / decode_elapsed_s) if decode_elapsed_s > 0 else 0.0

    return perf


def _format_row(label: str, stats: Dict[str, Any]) -> str:
    row = (
        f"{label:<20} "
        f"top1={stats['top1']:.3f} ({stats['matches_top1']}/{stats['total']})  "
        f"top5={stats['top5']:.3f} ({stats['matches_top5']}/{stats['total']})  "
        f"top{stats['k']}={stats['top100']:.3f} ({stats['matches_top100']}/{stats['total']})"
    )
    perf_parts = []
    if stats.get("ttft_ms") is not None:
        perf_parts.append(f"TTFT={stats['ttft_ms']:.2f}ms")
    if stats.get("decode_t/s/u") is not None:
        perf_parts.append(f"decode={stats['decode_t/s/u']:.2f} t/s/u")
    if stats.get("e2e_t/s/u") is not None:
        perf_parts.append(f"e2e={stats['e2e_t/s/u']:.2f} t/s/u")
    if perf_parts:
        row += "  " + "  ".join(perf_parts)
    return row


def run_teacher_forcing(
    *,
    model_dir: Path,
    reference_path: Path,
    mesh_device,
    build_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Programmatic entry point. Builds the generator, runs teacher forcing
    over all reference entries, and returns the per-entry accuracy dicts.
    Decode must be traced: `generate()` is required to explicitly accept
    `enable_trace`, and this runner always passes `enable_trace=True`.
    """
    build_kwargs = build_kwargs or {}
    build_generator = _import_build_generator(model_dir)
    generator: Generator = build_generator(model_dir=model_dir, mesh_device=mesh_device, **build_kwargs)

    acc = TokenAccuracy(reference_path)
    per_entry: List[Dict[str, Any]] = []
    try:
        for entry_idx in range(acc.num_entries):
            if entry_idx > 0:
                generator.reset()
            stats = _run_one_entry(
                generator=generator,
                acc=acc,
                entry_idx=entry_idx,
            )
            per_entry.append(stats)
            print(_format_row(f"entry[{entry_idx}]", stats))
            if stats.get("generate_stats"):
                print(f"entry[{entry_idx}] trace stats: {stats['generate_stats']}")
    finally:
        teardown = getattr(generator, "teardown", None)
        if callable(teardown):
            teardown()

    total = sum(s["total"] for s in per_entry)
    if total:
        total_elapsed_s = sum(s.get("elapsed_s", 0.0) for s in per_entry)
        ttft_values = [s["ttft_ms"] for s in per_entry if s.get("ttft_ms") is not None]
        decode_tokens = sum(s.get("decode_tokens", 0.0) for s in per_entry)
        decode_elapsed_s = sum(s.get("decode_elapsed_s", 0.0) for s in per_entry)
        agg = {
            "top1": sum(s["matches_top1"] for s in per_entry) / total,
            "top5": sum(s["matches_top5"] for s in per_entry) / total,
            "top100": sum(s["matches_top100"] for s in per_entry) / total,
            "matches_top1": sum(s["matches_top1"] for s in per_entry),
            "matches_top5": sum(s["matches_top5"] for s in per_entry),
            "matches_top100": sum(s["matches_top100"] for s in per_entry),
            "total": total,
            "k": per_entry[0]["k"],
            "elapsed_s": total_elapsed_s,
            "e2e_t/s/u": (total / total_elapsed_s) if total_elapsed_s > 0 else 0.0,
        }
        if ttft_values:
            agg["ttft_ms"] = sum(ttft_values) / len(ttft_values)
        if decode_elapsed_s > 0:
            agg["decode_tokens"] = decode_tokens
            agg["decode_elapsed_s"] = decode_elapsed_s
            agg["decode_t/s/u"] = decode_tokens / decode_elapsed_s
        print(_format_row("AGGREGATE", agg))

    return per_entry


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the teacher-forcing readiness check against a reference file.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the model directory.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to the .refpt reference file.")
    add_mesh_device_args(parser)
    args = parser.parse_args()

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config, args.trace_region_size)
    try:
        run_teacher_forcing(
            model_dir=args.model_dir.resolve(),
            reference_path=args.reference.resolve(),
            mesh_device=mesh_device,
        )
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    _main()
