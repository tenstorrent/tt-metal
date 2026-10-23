# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stable candidate selection and provenance for calculator-driven hardware trials."""

import hashlib
import json
import math
from pathlib import Path


def file_hashes(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if p is not None}


def make_shortlist(rows, count, provenance):
    if count < 1:
        raise ValueError("top N must be positive")
    # Identical estimates do not imply identical schedules. Remove only repeated
    # configurations; preserve distinct groups and reader assignments even on ties.
    unique = {}
    for row in rows:
        identity = row["configuration"]
        if not isinstance(identity, dict) or not identity:
            raise ValueError("configuration must be a nonempty operation/schedule description")
        if not math.isfinite(row["estimated_us"]) or row["estimated_us"] < 0:
            raise ValueError("estimate must be finite and nonnegative")
        signature = json.dumps(identity, sort_keys=True, allow_nan=False)
        if signature in unique and unique[signature]["estimated_us"] != row["estimated_us"]:
            raise ValueError("duplicate configuration has inconsistent estimates")
        unique.setdefault(signature, row)
    ranked = sorted(unique.items(), key=lambda item: (item[1]["estimated_us"], item[0]))
    candidates = [
        dict(rank=i + 1, candidate_id=hashlib.sha256(key.encode()).hexdigest()[:16], prediction=row)
        for i, (key, row) in enumerate(ranked[:count])
    ]
    return dict(
        status="predicted_shortlist_not_hardware_validated",
        requested_candidates=count,
        distinct_candidates=len(ranked),
        candidates=candidates,
        provenance=provenance,
        next_step="Implement and correctness-check every shortlisted configuration, then select using comparable device measurements.",
    )
