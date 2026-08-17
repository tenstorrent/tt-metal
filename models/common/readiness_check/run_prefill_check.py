# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run prefill-based readiness check against a reference file.

Validates TT model prefill accuracy by:
  1. Loading the generator from `<model_dir>/tt/generator.py`
  2. Running `prefill_forward(return_all_logits=True)` on full sequences
  3. Comparing predictions against reference top-K at each position
  4. Reporting top-1, top-5, and top-K accuracy

Use alongside `run_teacher_forcing.py` to validate both prefill and decode paths.

CLI:
    python -m models.common.readiness_check.run_prefill_check \\
        --model-dir models/autoports/<model_name> \\
        --reference references/<model>.refpt \\
        --mesh-device N150
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

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
from models.common.readiness_check.schema import Reference, ReferenceEntry, load_reference


def _import_build_generator(model_dir: Path) -> BuildGeneratorFn:
    """
    Load `<model_dir>/tt/generator.py` and return its `build_generator` function.
    """
    generator_path = model_dir / GENERATOR_MODULE_RELPATH
    if not generator_path.exists():
        raise FileNotFoundError(
            f"Expected generator at {generator_path}. The readiness check requires "
            f"<model_dir>/{GENERATOR_MODULE_RELPATH} to exist and expose `{BUILD_GENERATOR_FUNCTION_NAME}`."
        )

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


def _run_one_entry_prefill(
    *,
    generator: Generator,
    entry: ReferenceEntry,
    reference: Reference,
) -> Dict[str, Any]:
    """
    Run batch prefill check for one entry.

    Args:
        generator: TT generator instance
        entry: Reference entry with prompt, generated tokens, and top-K predictions
        reference: Reference object (for metadata)

    Returns:
        Accuracy dict with top1, top5, top100, matches, and total
    """
    prompt_tokens = entry.prompt_tokens[0].tolist()  # [P]
    gen_tokens = entry.generated_tokens[0].tolist()  # [G]
    topk_reference = entry.topk_tokens  # [G, K]

    # Concatenate prompt + generated for full sequence
    full_sequence = prompt_tokens + gen_tokens
    prompt_len = len(prompt_tokens)
    gen_len = len(gen_tokens)
    full_len = prompt_len + gen_len

    # Convert to tensor [1, full_len]
    tokens_tensor = torch.tensor([full_sequence], dtype=torch.long)

    # Prepare KV cache and page table (implementation-specific)
    # This is a simplified version - real implementations need proper setup
    # For now, we'll call prefill_forward with minimal setup and let the
    # generator handle defaults through **kwargs

    # Call prefill with return_all_logits=True to get logits at all positions
    # Note: This requires the generator implementation to support return_all_logits
    try:
        # Most generators will need proper page_table and kv_cache setup
        # This is a placeholder - real usage requires model-specific initialization
        import ttnn

        # Allocate dummy page table and kv cache
        # Real implementations should use generator's initialization methods
        batch_size = 1
        max_blocks = 1024
        page_table = torch.arange(max_blocks).reshape(batch_size, max_blocks)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=generator.mesh_device if hasattr(generator, "mesh_device") else None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Placeholder kv_cache - real implementation needs proper initialization
        kv_cache = None  # Generator should handle None gracefully or we need to init properly

        logits = generator.prefill_forward(
            tokens=tokens_tensor,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            prompt_lens=[full_len],
            return_all_logits=True,
        )
    except TypeError as e:
        if "return_all_logits" in str(e):
            raise NotImplementedError(
                f"Generator {type(generator).__name__} does not support return_all_logits parameter. "
                "Please update the generator to support the new contract."
            ) from e
        raise

    # logits should be [1, full_len, vocab_size]
    if logits.dim() == 2:
        # [batch, vocab] - only got last position, not all positions
        raise RuntimeError(
            f"Generator returned logits with shape {logits.shape}, but return_all_logits=True "
            "should return shape [batch, seq_len, vocab]. Generator may not support return_all_logits."
        )

    # Extract logits at positions that predict gen_tokens
    # logits[0, i] predicts token at position i+1
    # So logits[0, prompt_len-1:prompt_len+gen_len-1] predicts gen_tokens
    prediction_logits = logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [gen_len, vocab]

    # Get top-1 predictions (argmax)
    tt_predictions = torch.argmax(prediction_logits, dim=-1).cpu()  # [gen_len]

    # Compare against reference
    matches_top1 = 0
    matches_top5 = 0
    matches_topk = 0
    k_cols = min(5, topk_reference.shape[1])

    for i in range(gen_len):
        tt_pred = int(tt_predictions[i].item())
        ref_topk = topk_reference[i]

        if tt_pred == int(ref_topk[0].item()):
            matches_top1 += 1
        if tt_pred in ref_topk[:k_cols].tolist():
            matches_top5 += 1
        if tt_pred in ref_topk.tolist():
            matches_topk += 1

    return {
        "top1": matches_top1 / gen_len,
        "top5": matches_top5 / gen_len,
        "top100": matches_topk / gen_len,
        "matches_top1": matches_top1,
        "matches_top5": matches_top5,
        "matches_top100": matches_topk,
        "total": gen_len,
        "k": reference.k,
    }


def _format_row(label: str, stats: Dict[str, Any]) -> str:
    return (
        f"{label:<20} "
        f"top1={stats['top1']:.3f} ({stats['matches_top1']}/{stats['total']})  "
        f"top5={stats['top5']:.3f} ({stats['matches_top5']}/{stats['total']})  "
        f"top{stats['k']}={stats['top100']:.3f} ({stats['matches_top100']}/{stats['total']})"
    )


def run_prefill_check(
    *,
    model_dir: Path,
    reference_path: Path,
    mesh_device,
    build_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Programmatic entry point. Builds the generator, runs batch prefill
    over all reference entries, and returns the per-entry accuracy dicts.
    """
    build_kwargs = build_kwargs or {}
    build_generator = _import_build_generator(model_dir)
    generator: Generator = build_generator(model_dir=model_dir, mesh_device=mesh_device, **build_kwargs)

    reference = load_reference(reference_path)
    per_entry: List[Dict[str, Any]] = []

    try:
        for entry_idx, entry in enumerate(reference.entries):
            if entry_idx > 0:
                generator.reset()
            stats = _run_one_entry_prefill(generator=generator, entry=entry, reference=reference)
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
    parser = argparse.ArgumentParser(description="Run the batch prefill readiness check against a reference file.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the model directory.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to the .refpt reference file.")
    add_mesh_device_args(parser)
    args = parser.parse_args()

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config, args.trace_region_size)
    try:
        run_prefill_check(
            model_dir=args.model_dir.resolve(),
            reference_path=args.reference.resolve(),
            mesh_device=mesh_device,
        )
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    _main()
