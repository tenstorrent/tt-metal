# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The KDA single-op test checklist: the one source of test names, selection groups and reasons.

A test named ``test_<binding>_<suffix>`` belongs to the checklist item whose ``suffix`` pattern matches the
whole suffix. The KDA conftest marks each test with its item's ``group``, and the conformance test checks
every op module against ``enforced`` items. Plain Python on purpose: collection and the conformance test
must not need a device or ttnn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistItem:
    leaf: str
    group: str
    suffix: str
    enforced: bool
    why: str

    def matches(self, suffix: str) -> bool:
        return re.fullmatch(self.suffix, suffix) is not None


GROUPS = {
    "accuracy": "computes the right values",
    "stability": "error stays bounded as outputs feed back",
    "determinism": "same answer on every run, eager and traced",
    "contract": "accepts exactly what it supports, returns what it promises",
    "caching": "stays correct when programs and traces are reused",
    "safety": "stays inside its memory and execution bounds",
    "perf": "as fast as its measured reference",
}

SHAPES = {
    "unit": "a device test on small development shapes (any test without a production case)",
    "production": "a device test on a model's production geometry",
    "host": "a test that needs no device",
}

# Enforced items must exist in every op module or be declared in its NOT_APPLICABLE with a reason. The others
# are candidates (bead tt_metal_tracker-6u9h.2): their names are reserved so tests written now already conform.
CHECKLIST = (
    ChecklistItem(
        "1a",
        "accuracy",
        r"accuracy",
        True,
        "PCC is blind to scale; NRMSE against a measured threshold sees precision loss.",
    ),
    ChecklistItem(
        "1b",
        "accuracy",
        r"matches_\w+",
        False,
        "An independent path catches errors a golden written from the same understanding shares.",
    ),
    ChecklistItem(
        "1c",
        "accuracy",
        r"edge_\w+",
        False,
        "Ties, sentinels, padding and partial tiles are where implementations legitimately differ.",
    ),
    ChecklistItem(
        "1d",
        "accuracy",
        r"compute_config_\w+",
        False,
        "A precision setting that is silently ignored changes accuracy with no test noticing.",
    ),
    ChecklistItem(
        "1e",
        "accuracy",
        r"extreme_inputs",
        False,
        "Random normal inputs rarely reach the regimes where precision breaks.",
    ),
    ChecklistItem(
        "2a",
        "stability",
        r"long_horizon_\w+",
        False,
        "Systematic rounding accumulates over a recurrence's memory length.",
    ),
    ChecklistItem(
        "3",
        "determinism",
        r"is_deterministic",
        True,
        "Work split across cores can race; a race shows as run-to-run variation, eager or under trace replay.",
    ),
    ChecklistItem(
        "4a",
        "contract",
        r"rejects_\w+",
        True,
        "A guard without a test can be removed silently; the caller then gets wrong results instead of an error.",
    ),
    ChecklistItem(
        "4b",
        "contract",
        r"default_compute_config_matches_explicit",
        True,
        "The default path is a separate code path the model may never exercise.",
    ),
    ChecklistItem(
        "4d",
        "contract",
        r"does_not_expose_removed_modes",
        False,
        "A deleted mode must not come back through a stale keyword.",
    ),
    ChecklistItem(
        "4f",
        "contract",
        r"output_contract",
        True,
        "Callers rely on dtype, layout, placement and shape; outputs must not alias inputs or modify them.",
    ),
    ChecklistItem(
        "5a",
        "caching",
        r"cache_hit_rebinds_fresh_tensors",
        True,
        "An address not rebound on a cache hit makes the op read or write the old buffer.",
    ),
    ChecklistItem(
        "5b",
        "caching",
        r"program_key_includes_\w+",
        False,
        "A missing attribute in the program hash silently reuses another configuration's program.",
    ),
    ChecklistItem(
        "5c",
        "caching",
        r"program_key_shares_\w+",
        False,
        "Over-keyed programs compile needlessly.",
    ),
    ChecklistItem(
        "5d",
        "caching",
        r"trace_follows_\w+",
        False,
        "Replays that change only a device-tensor parameter's contents must give the scalar result.",
    ),
    ChecklistItem(
        "6a",
        "safety",
        r"writes_only_outputs",
        False,
        "A stray write into a pooled buffer corrupts someone else's memory.",
    ),
    ChecklistItem(
        "6b",
        "safety",
        r"ignores_stale_input",
        False,
        "Reused buffers always hold stale data past the valid length.",
    ),
    ChecklistItem(
        "6c",
        "safety",
        r"at_resource_limit",
        False,
        "Limits are where layouts change and where silent overflow corrupts memory.",
    ),
    ChecklistItem(
        "6d",
        "safety",
        r"completes_at_degenerate_sizes",
        False,
        "Handshakes deadlock where one side has no work.",
    ),
    ChecklistItem(
        "6f",
        "safety",
        r"after_resource_reset",
        False,
        "Ops that assume zeroed scratch or semaphores pass alone and fail inside a model.",
    ),
    ChecklistItem(
        "6g",
        "safety",
        r"no_leaks_over_repeated_calls",
        False,
        "A slow leak passes every short test and fails a long serve.",
    ),
    ChecklistItem(
        "7a",
        "perf",
        r"performance",
        True,
        "A two-sided band around a measured device time catches regressions and stale references.",
    ),
    ChecklistItem(
        "7b",
        "perf",
        r"performance_beats_\w+",
        False,
        "A fused op that is not faster than the composed path has no reason to exist.",
    ),
    ChecklistItem(
        "7c",
        "perf",
        r"work_golden",
        True,
        "Utilization reports are only as good as the work model behind them.",
    ),
)

LEAVES = {item.leaf: item for item in CHECKLIST}
# Device-time gates: they carry the perf decorator (host-IOMMU runner, no watcher or LLK asserts).
DEVICE_TIME_LEAVES = frozenset({"7a", "7b"})


def binding_of(module_name: str) -> str:
    """``test_<binding>`` (a module name or file stem) -> ``<binding>``."""
    return module_name.rsplit(".", 1)[-1].removesuffix(".py").removeprefix("test_")


def item_for(test_name: str, binding: str) -> ChecklistItem | None:
    """The checklist item of ``test_<binding>_<suffix>``, or None if the name is outside the vocabulary."""
    prefix = f"test_{binding}_"
    if not test_name.startswith(prefix):
        return None
    suffix = test_name.removeprefix(prefix)
    return next((item for item in CHECKLIST if item.matches(suffix)), None)
