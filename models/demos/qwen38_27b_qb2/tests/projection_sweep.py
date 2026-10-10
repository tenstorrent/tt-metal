# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded BFP8 projection candidates and conservative timing acceptance."""

import math
import statistics

ROLES = {
    "output": dict(name="linear_attn.out_proj", k=1536, n=5120, layers=48, block=6),
    "down": dict(name="mlp.down_proj", k=4352, n=5120, layers=64, block=17),
}
BATCHES = (16, 32)


def candidates(role):
    """Vary one knob first, then reader/block combinations; no precision changes."""
    block = ROLES[role]["block"]
    # The pinned kernel supports at most three readers per DRAM bank. Larger
    # blocks gather adjacent activation shards before multicast (M is one tile).
    values = [(2, 8, block), (1, 8, block), (3, 8, block), (2, 8, 1), (2, 8, block * 2)]
    if role == "output":
        values += [(2, 8, 24), (3, 8, 12), (3, 8, 24), (2, 16, 3), (2, 24, 2)]
    else:
        values += [(3, 8, 34), (2, 17, 8), (2, 17, 4), (2, 34, 4), (3, 17, 8)]
    return [dict(readers=r, cores=c, block=b) for r, c, b in values]


def geometry(role, config, *, banks=8):
    spec = ROLES[role]
    readers, cores, block = (config[k] for k in ("readers", "cores", "block"))
    if readers not in (1, 2, 3) or cores <= 0 or block <= 0:
        raise ValueError("Unsupported projection geometry")
    kt = spec["k"] // 32
    shard_tiles = math.ceil(kt / cores)
    if kt % block or (block > shard_tiles and block % shard_tiles):
        raise ValueError("K/block/shard divisibility is incompatible")
    padded_n = math.ceil(spec["n"] / (32 * banks * readers)) * 32 * banks * readers
    return dict(
        padded_n=padded_n,
        weight_shard_width=padded_n // banks,
        activation_shard_tiles=shard_tiles,
        encoded_weight_bytes=spec["k"] * padded_n // 1024 * 1088,
        multicast_blocks=kt // block,
    )


def compare(before, candidate, after, *, role):
    """Projection plus layout/collective time; never label this model throughput."""
    timings = [r["samples_us"] for r in (before, candidate, after)]
    if any(len(v) != 5 for v in timings) or any(not math.isfinite(t) or t <= 0 for v in timings for t in v):
        raise ValueError("Require five finite positive timing samples in each bracket")
    base = statistics.median(timings[0] + timings[2])
    current = statistics.median(timings[1])
    drift = abs(statistics.median(timings[2]) / statistics.median(timings[0]) - 1)
    control_match = before["output_sha256"] == after["output_sha256"]
    accuracy = all(
        r.get("accuracy_passed") is True and r.get("changed_input_trace_passed") is True
        for r in (before, candidate, after)
    )
    accepted = drift <= 0.03 and control_match and accuracy
    return dict(
        control_us=base,
        candidate_us=current,
        control_drift_fraction=drift,
        controls_match=control_match,
        accuracy_passed=accuracy,
        comparison_qualified=accepted,
        speedup=base / current if accepted else None,
        projected_model_saving_ms=(base - current) * ROLES[role]["layers"] / 1000 if accepted else None,
        multiplicity=ROLES[role]["layers"],
        full_model_measured=False,
        promoted_to_serving=False,
    )
