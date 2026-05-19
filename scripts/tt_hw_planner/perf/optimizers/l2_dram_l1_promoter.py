# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L2 dram_l1_promoter — DRAM tensors that fit -> L1-sharded."""

from __future__ import annotations

import re
from typing import List

from ..ceilings import BoxSpec
from ..join import JoinedRow
from .base import (
    Finding,
    ModelSource,
    Patch,
    PatchKind,
    Severity,
    SourceLocation,
    VerificationResult,
)


_DIM_RE = re.compile(r"(\d+)")


def _estimate_input_bytes(row: JoinedRow) -> int:
    """Crude estimate of total input tensor bytes."""
    total = 0
    for t in row.inputs:
        dims_str = "x".join(str(t.get(k, "")) for k in ("W", "Z", "Y", "X"))
        nums = [int(m.group()) for m in _DIM_RE.finditer(dims_str)]
        elems = 1
        for n in nums:
            elems *= n
        # Assume 2 bytes/element (bf16). Refined later by mapping DATATYPE.
        total += elems * 2
    return total


class DramL1Promoter:
    name = "dram_l1_promoter"
    level = 2
    requires: List[str] = []

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box: BoxSpec) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        l1_budget = box.cores_per_chip * box.l1_per_core_b
        seen = set()
        for r in joined:
            if r.cluster_id in seen:
                continue
            seen.add(r.cluster_id)
            if r.region != "E":
                continue
            bytes_needed = _estimate_input_bytes(r)
            if bytes_needed <= 0 or bytes_needed > l1_budget:
                continue
            cost_ms = (r.device_kernel_ns or 0) / 1e6
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=r.cluster_id or "?",
                    block_path=r.block_path,
                    severity=Severity.WARN,
                    cost_ms=cost_ms,
                    evidence={
                        "estimated_bytes": bytes_needed,
                        "l1_budget_bytes": l1_budget,
                        "dram_util_pct": r.dram_bw_util_pct,
                        "region_reason": r.region_reason,
                    },
                    suggestion=(
                        f"Tensor fits in {bytes_needed / 1024:.1f} KiB of L1 "
                        f"(<{l1_budget / 1024 / 1024:.1f} MiB available across cores). "
                        f"Promote `memory_config=DRAM_INTERLEAVED` -> `ShardedL1(...)`."
                    ),
                    source_locations=[
                        SourceLocation(
                            path="models/tt_transformers/tt/model_config.py",
                            variable=f"memory_config[{r.op_code}]",
                        )
                    ],
                    expected_gain_ms=cost_ms * 0.5,
                )
            )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.TUNING_TABLE_ENTRY,
                    target=SourceLocation(
                        path="models/tt_transformers/tt/model_config.py",
                        variable=f"memory_config[{f.cluster_id}]",
                    ),
                    new_kwargs={
                        "memory_config": "SHARDED_L1",
                        "shard_strategy": "block",
                        "cluster_id": f.cluster_id,
                    },
                    rationale=(
                        f"Region E DRAM-bound; tensor fits ({f.evidence.get('estimated_bytes')} B). "
                        f"Expected -{f.expected_gain_ms:.2f} ms"
                    ),
                    extra={"finding": f.cluster_id, "auto_revert_on_slowdown": True},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        b = sum(r.device_kernel_ns or 0 for r in before)
        a = sum(r.device_kernel_ns or 0 for r in after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("L1 promotion helped" if delta_ms < 0 else "L1 promotion slowed things; reverting"),
        )
