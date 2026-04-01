#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tests.sweep_framework.framework.constants import (
    format_hardware_suffix,
    parse_hardware_suffix,
    strip_grouping_suffix,
)
from tests.sweep_framework.framework.compute_sweep_matrix import _hw_label, _runner_for_hardware_group
from tests.sweep_framework.framework.execution_capabilities import (
    load_execution_capability_profiles,
    load_vector_file_summaries,
    resolve_active_profile,
    select_eligible_vector_summaries,
)


def compute_validation_matrix(vectors_dir: Path, suite_name: str) -> dict:
    vector_summaries = load_vector_file_summaries(vectors_dir)
    if not vector_summaries:
        raise RuntimeError(f"No vector JSON files found in {vectors_dir}")

    profiles = load_execution_capability_profiles()
    try:
        active_profile = resolve_active_profile(profiles=profiles)
        print(f"Active execution capability profile: {active_profile.name}", file=sys.stderr)
        vector_summaries = select_eligible_vector_summaries(vector_summaries, active_profile)
        if not vector_summaries:
            raise RuntimeError(f"No vector JSON files are eligible for capability profile {active_profile.name}")
    except RuntimeError as exc:
        print(f"Capability profile selection skipped: {exc}", file=sys.stderr)

    hardware_summaries: dict[tuple[str, str, int] | None, list] = defaultdict(list)
    for summary in vector_summaries:
        hardware_summaries[parse_hardware_suffix(summary.module_name)].append(summary)

    include_entries = []
    batches = []
    grouped_items = sorted(hardware_summaries.items(), key=lambda item: (item[0] is None, item[0]))
    for hardware_group, grouped_summaries in grouped_items:
        base_modules = sorted({strip_grouping_suffix(summary.module_name) for summary in grouped_summaries})
        if not base_modules:
            continue

        runner_config = _runner_for_hardware_group(hardware_group)
        hardware_label = _hw_label(hardware_group)
        hardware_slug = (
            format_hardware_suffix(*hardware_group).replace(".hw_", "", 1) if hardware_group is not None else "default"
        )
        trace_ids = sorted({trace_id for summary in grouped_summaries for trace_id in summary.trace_ids})
        batch = ",".join(base_modules)
        batches.append(batch)
        include_entries.append(
            {
                **runner_config,
                "test_group_name": f"validation-{hardware_slug}",
                "module_selector": batch,
                "batch_display": hardware_label,
                "suite_name": suite_name,
                "trace_ids": trace_ids,
                "hardware_group": hardware_slug,
            }
        )

    total_base_modules = len({strip_grouping_suffix(summary.module_name) for summary in vector_summaries})
    print(
        f"Validation run: {len(vector_summaries)} vector files ({total_base_modules} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for hardware_group, grouped_summaries in grouped_items:
        unique_base = len({strip_grouping_suffix(summary.module_name) for summary in grouped_summaries})
        print(
            f"  hardware {_hw_label(hardware_group)}: {len(grouped_summaries)} vectors ({unique_base} unique modules)",
            file=sys.stderr,
        )

    return {
        "module": [summary.module_name for summary in vector_summaries],
        "batches": batches,
        "include": include_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute validation sweep matrix from generated vector files.")
    parser.add_argument(
        "--vectors-dir", required=True, type=Path, help="Directory containing generated vector JSON files"
    )
    parser.add_argument("--suite-name", default="model_traced", help="Suite name stored in each matrix row")
    args = parser.parse_args()

    matrix = compute_validation_matrix(args.vectors_dir, args.suite_name)
    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
