# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L6 cache_warmer — fix low program_cache_hit and PROGRAM HASH thrash."""

from __future__ import annotations

from collections import defaultdict
from typing import List

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


CACHE_HIT_TARGET = 0.99


class CacheWarmer:
    name = "cache_warmer"
    level = 6
    requires: List[str] = []

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        # cache-hit rate by op
        op_hits = defaultdict(lambda: [0, 0])
        # shape-thrash detection: a single cluster shouldn't have multiple
        # PROGRAM HASH values across iterations.
        cluster_program_hashes = defaultdict(set)
        cluster_cost_ms = defaultdict(float)
        for r in joined:
            if r.program_cache_hit is not None:
                op_hits[r.op_code][0] += 1 if r.program_cache_hit else 0
                op_hits[r.op_code][1] += 1
            if r.program_hash:
                cluster_program_hashes[r.cluster_id or "?"].add(r.program_hash)
                cluster_cost_ms[r.cluster_id or "?"] += (r.device_kernel_ns or 0) / 1e6

        for op, (hits, total) in op_hits.items():
            if total == 0:
                continue
            rate = hits / total
            if rate >= CACHE_HIT_TARGET:
                continue
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=op,
                    block_path="root",
                    severity=Severity.WARN,
                    cost_ms=0.0,
                    evidence={"cache_hit_rate": rate, "total_calls": total, "kind": "cold_cache"},
                    suggestion=(
                        f"{op} has only {rate*100:.0f}% cache hits ({hits}/{total}); "
                        "insert a warm-up iteration before measurement."
                    ),
                    source_locations=[
                        SourceLocation(path=source.demo_test_path, func="iteration", variable="warmup_iters")
                    ],
                    expected_gain_ms=0.0,
                )
            )

        for cid, hashes in cluster_program_hashes.items():
            if len(hashes) <= 1:
                continue
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=cid,
                    block_path="root",
                    severity=Severity.WARN,
                    cost_ms=cluster_cost_ms[cid],
                    evidence={"program_hashes": list(hashes), "kind": "shape_thrash"},
                    suggestion=(
                        f"Cluster {cid} has {len(hashes)} distinct PROGRAM HASH values; "
                        "the inner loop is recompiling. Fix the iteration to use a stable shape."
                    ),
                    source_locations=[SourceLocation(path=source.demo_test_path, func="iteration")],
                    expected_gain_ms=cluster_cost_ms[cid] * 0.2,
                )
            )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.KWARG_REPLACE,
                    target=SourceLocation(path=source.demo_test_path, variable="warmup_iters"),
                    new_kwargs={"warmup_iters": 1 if f.evidence.get("kind") == "cold_cache" else 2},
                    rationale=f"{f.evidence.get('kind')} -> add warm-up iterations",
                    extra={"finding": f.cluster_id, "kind": f.evidence.get("kind")},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        def hit_rate(rows):
            yes = sum(1 for r in rows if r.program_cache_hit)
            tot = sum(1 for r in rows if r.program_cache_hit is not None)
            return yes / tot if tot else 0.0

        b = hit_rate(before)
        a = hit_rate(after)
        return VerificationResult(
            ok=a > b,
            delta_ms=0.0,
            accuracy_held=True,
            reason=f"cache hit {b*100:.1f}% -> {a*100:.1f}%",
        )
