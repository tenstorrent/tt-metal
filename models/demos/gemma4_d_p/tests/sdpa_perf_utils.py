# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only configuration, layout and statistics for the isolated Gemma SDPA sweep."""

import json
import math
import statistics
from dataclasses import asdict, dataclass, replace

PREFIXES = (0, 49152, 98304, 196608, 253952)
CP, TP, LOCAL_Q, CAPACITY = 8, 4, 1024, 262144
CHUNK = CP * LOCAL_Q
PCC_THRESHOLD, RMSE_THRESHOLD = 0.999, 0.05


@dataclass(frozen=True)
class Candidate:
    q: int = 64
    k: int = 256
    exp_approx: bool = False
    math_approx: bool = False
    fp32_dest: bool = False
    packer_l1: bool = False
    fidelity: str = "HiFi2"

    @property
    def id(self):
        return (
            f"q{self.q}-k{self.k}-exp{int(self.exp_approx)}-math{int(self.math_approx)}"
            f"-fp32{int(self.fp32_dest)}-pack{int(self.packer_l1)}-{self.fidelity}"
        )

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def baseline(layer):
    if layer not in ("global", "swa"):
        raise ValueError(f"Unknown layer: {layer}")
    return Candidate(k=256 if layer == "global" else 128)


def tilings(layer):
    base = baseline(layer)
    qs, ks = ((32, 64, 128), (64, 128, 256, 512)) if layer == "global" else ((64, 128), (128,))
    return list(dict.fromkeys([base] + [replace(base, q=q, k=k) for q in qs for k in ks]))


def family_candidates(winner, family):
    values = {
        "exp_approx": (False, True),
        "math_approx": (False, True),
        "fp32_dest": (False, True),
        "packer_l1": (False, True),
        "fidelity": ("LoFi", "HiFi2", "HiFi3", "HiFi4"),
    }
    return [replace(winner, **{family: value}) for value in values[family]]


def cache_to_rank_major(tensor, cp=CP, slab=LOCAL_Q):
    """Chronological [B,H,S,D] -> contiguous per-rank block-cyclic cache slabs."""
    b, h, s, d = tensor.shape
    if s % (cp * slab):
        raise ValueError("Cache capacity must contain complete ring groups")
    return tensor.reshape(b, h, s // (cp * slab), cp, slab, d).permute(0, 1, 3, 2, 4, 5).reshape(b, h, s, d)


def cache_to_chronological(tensor, cp=CP, slab=LOCAL_Q):
    b, h, s, d = tensor.shape
    if s % (cp * slab):
        raise ValueError("Cache capacity must contain complete ring groups")
    return tensor.reshape(b, h, cp, s // (cp * slab), slab, d).permute(0, 1, 3, 2, 4, 5).reshape(b, h, s, d)


def pack_queries(q, prefix, cp=CP, slab=LOCAL_Q):
    """Pack valid query rows in rank order, including wrapped starts and partial ends."""
    import torch

    b, h, length, d = q.shape
    if prefix < 0 or prefix % 32 or length <= 0 or length > cp * slab or length % 32:
        raise ValueError("Query prefix/length must be tile aligned and fit one ring group")
    positions = torch.arange(prefix, prefix + length)
    owners = (positions % (cp * slab)) // slab
    packed = q.new_zeros((b, h, cp * slab, d))
    rank_positions = []
    for rank in range(cp):
        indices = torch.where(owners == rank)[0]
        packed[:, :, rank * slab : rank * slab + len(indices)] = q[:, :, indices]
        rank_positions.append(positions[indices])
    return packed, rank_positions


def reference_attention(q, k, v, positions, window=None, block=32):
    """FP32 PyTorch reference on every valid row, with bounded score-matrix memory."""
    import torch
    import torch.nn.functional as F

    if q.shape[1] % k.shape[1] or k.shape[1] != v.shape[1]:
        raise ValueError("Expected grouped-query attention with matching K/V head counts")
    head_indices = torch.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    result = torch.empty_like(q, dtype=torch.float32)
    for start in range(0, len(positions), block):
        end = min(start + block, len(positions))
        pos = positions[start:end]
        lo = max(0, int(pos.min()) - window + 1) if window else 0
        hi = int(pos.max()) + 1
        keys = torch.arange(lo, hi)
        allowed = keys[None, :] <= pos[:, None]
        if window:
            allowed &= keys[None, :] >= pos[:, None] - window + 1
        result[:, :, start:end] = F.scaled_dot_product_attention(
            q[:, :, start:end].float(),
            k[:, head_indices, lo:hi].float(),
            v[:, head_indices, lo:hi].float(),
            attn_mask=allowed,
            dropout_p=0.0,
            scale=1.0,
        )
    return result


def accuracy_metrics(expected, actual):
    import torch

    x, y = expected.double().flatten(), actual.double().flatten()
    if x.numel() != y.numel() or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise AssertionError("Nonfinite or mismatched reference/output")
    rmse = (x - y).square().mean().sqrt().item()
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        raise AssertionError("Degenerate reference/output cannot establish PCC")
    return (x.dot(y) / denom).item(), rmse


def sdpa_duration_ns(records, expected_chips):
    """Require exactly one target invocation and a complete participating-chip record set."""
    programs = {}
    for record in records:
        sources = [source.replace("\\", "/") for source in record["kernel_sources"]]
        if not any("/sdpa/" in source and "ring_joint" in source for source in sources):
            continue
        runtime_id = record["runtime_id"]
        if not runtime_id:
            raise AssertionError("Target SDPA record has no runtime ID")
        chips = programs.setdefault(runtime_id, {})
        chip = record["chip_id"]
        if chip in chips:
            raise AssertionError(f"Duplicate SDPA record for chip {chip}, runtime {runtime_id}")
        duration = record["duration_ns"]
        if not math.isfinite(duration) or duration <= 0:
            raise AssertionError("Invalid device duration")
        chips[chip] = duration
    if len(programs) != 1:
        raise AssertionError(f"Expected one SDPA invocation, found {len(programs)}")
    chips = next(iter(programs.values()))
    if set(chips) != set(expected_chips):
        raise AssertionError(f"Incomplete SDPA chip records: got {sorted(chips)}, expected {sorted(expected_chips)}")
    return max(chips.values())


def timing_summary(durations_ns):
    if len(durations_ns) < 20:
        raise ValueError("At least 20 measured invocations are required")
    samples = sorted(duration / 1000 for duration in durations_ns)
    return statistics.median(samples), samples[math.ceil(0.9 * len(samples)) - 1]
