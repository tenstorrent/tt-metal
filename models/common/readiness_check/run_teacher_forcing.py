# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run the teacher-forcing portion of the model-readiness check.

Given a model directory containing a `tt/generator.py` that satisfies the
contract in `contract.py`, and a reference file produced by
`models.common.readiness_check.generate`, this runner:

  1. Builds the generator via `<model_dir>/tt/generator.py::build_generator`.
  2. For each entry in the reference, calls
     `generator.generate(prompt_ids, n_steps, next_input=acc.collect_predicted_tokens)`.
  3. Resets the generator between entries.
  4. Prints per-entry and aggregate top-1 / top-5 / top-K accuracy.

CLI:
    python -m models.common.readiness_check.run_teacher_forcing \\
        --model-dir models/autoports/<model_name> \\
        --reference models/common/readiness_check/references/<model>.refpt \\
        --mesh-device N150
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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

    def next_input(step: int, predicted: int) -> int:
        return acc.collect_predicted_tokens(predicted, user_idx=entry_idx)

    generator.generate(
        prompt_token_ids=prompt_ids,
        max_new_tokens=n_steps,
        next_input=next_input,
    )
    return acc.compute_accuracy(user_idx=entry_idx)


def _format_row(label: str, stats: Dict[str, Any]) -> str:
    return (
        f"{label:<20} "
        f"top1={stats['top1']:.3f} ({stats['matches_top1']}/{stats['total']})  "
        f"top5={stats['top5']:.3f} ({stats['matches_top5']}/{stats['total']})  "
        f"top{stats['k']}={stats['top100']:.3f} ({stats['matches_top100']}/{stats['total']})"
    )


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
            stats = _run_one_entry(generator=generator, acc=acc, entry_idx=entry_idx)
            per_entry.append(stats)
            print(_format_row(f"entry[{entry_idx}]", stats))
    finally:
        teardown = getattr(generator, "teardown", None)
        if callable(teardown):
            teardown()

    total = sum(s["total"] for s in per_entry)
    if total:
        agg = {
            "top1": sum(s["matches_top1"] for s in per_entry) / total,
            "top5": sum(s["matches_top5"] for s in per_entry) / total,
            "top100": sum(s["matches_top100"] for s in per_entry) / total,
            "matches_top1": sum(s["matches_top1"] for s in per_entry),
            "matches_top5": sum(s["matches_top5"] for s in per_entry),
            "matches_top100": sum(s["matches_top100"] for s in per_entry),
            "total": total,
            "k": per_entry[0]["k"],
        }
        print(_format_row("AGGREGATE", agg))

    return per_entry


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the teacher-forcing readiness check against a reference file.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the model directory.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to the .refpt reference file.")
    add_mesh_device_args(parser)
    args = parser.parse_args()

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
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
