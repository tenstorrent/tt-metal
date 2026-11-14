# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.58 (owner 2026-11-14, TP experts go/no-go, STEP 1): the CPU model of the routed experts' compute under expert
parallelism (EP, today) and tensor-parallel experts (TP4 / TP8). No device; tests/test_moe_tp4_check.py feeds it the
recorded router counts and the device's calibration.

Terms
-----
* chip k = r C + c on the R x C = 8 x 4 mesh (ConcatMeshToTensor order, tt/moe_block_main_adapter.flush_counts); slot j
  of chip k holds expert e = k * 12 + j (EP=32: 384 experts, 12 a chip). counts[k, j]: tokens routed to that expert in
  one layer of one chunk (MOE_COUNTS).
* tile: 32 rows. An expert with n tokens costs ceil(n / 32) row tiles (the flat expert's 32-row blocks).
* EP chip time (P.55a's fit): t_k = max(A x active experts_k, B x row tiles_k); layer = C0 + max_k t_k.
  A = one expert's weight read (19.9 MB bfp4 at the DRAM rate), B = one row tile of one whole expert's GLU FFN.
* TP-g (g = 4: a group = one mesh ROW of 4 chips, 8 groups of 48 experts; g = 8: one mesh COLUMN of 8 chips, 4 groups
  of 96 experts): each expert split g ways along its FFN width (I = 2304 -> 576 / 288); every chip of a group computes
  ALL of the group's tokens on its 1/g slice; the g partial outputs are summed in the combine. Chip time (the same on
  every chip of a group):  t_G = max(A_g x active experts_G, B_g x row tiles_G), row tiles at full rows (each slice
  still pads every expert's rows to 32). Two variants:
    "scaled": A_g = A / g, B_g = B / g (per byte and per FLOP the same as EP);
    "fitted": A_g, B_g, C0_g from the device's own flat expert at the slice width (calibration sweeps and real-count
              layers, test_moe_tp4_check.py).
  The layer's experts program = C0_g + max over groups t_G.
* busy/mean: the busiest chip's (group's) time over the mean chip's, compute only.
"""

from __future__ import annotations

import math

import numpy as np

EPC = 12  # experts a chip (EP=32)


def tiles(n) -> np.ndarray:
    return (np.asarray(n) + 31) // 32


def groups(shape: tuple, tp: int) -> list[list[int]]:
    """The chips of each TP group: tp == C -> mesh rows; tp == R -> mesh columns; tp == 1 -> one chip each."""
    R, C = shape
    if tp == 1:
        return [[k] for k in range(R * C)]
    if tp == C:
        return [[r * C + c for c in range(C)] for r in range(R)]
    if tp == R:
        return [[r * C + c for r in range(R)] for c in range(C)]
    raise ValueError(f"tp {tp} on a {R} x {C} mesh: a row ({C}) or a column ({R})")


def group_counts(own: np.ndarray, shape: tuple, tp: int) -> list[np.ndarray]:
    """[chips, EPC] own counts -> per group the counts of its experts (its chips' slots, in chip order)."""
    return [np.concatenate([own[k] for k in g]) for g in groups(shape, tp)]


def chip_times(own: np.ndarray, shape: tuple, tp: int, A: float, B: float) -> np.ndarray:
    """Per chip compute time (ms): EP (tp 1) each chip its own experts; TP every chip of a group the same."""
    out = np.zeros(own.shape[0])
    for g, n in zip(groups(shape, tp), group_counts(own, shape, tp)):
        t = max(A * float((n > 0).sum()), B * float(tiles(n).sum()))
        out[g] = t
    return out


def layer_ms(own: np.ndarray, shape: tuple, tp: int, A: float, B: float, C0: float) -> dict:
    t = chip_times(own, shape, tp, A, B)
    return {"ms": C0 + float(t.max()), "busy_ms": float(t.max()), "mean_ms": float(t.mean()),
            "busy_over_mean": float(t.max() / t.mean()) if t.mean() > 0 else float("nan")}  # fmt: skip


def padding_share(own: np.ndarray, shape: tuple, tp: int) -> float:
    """Padded rows / real rows of the busiest group (32-row padding a slot)."""
    best = None
    for n in group_counts(own, shape, tp):
        real = float(n.sum())
        pad = float((tiles(n) * 32).sum())
        if best is None or real > best[0]:
            best = (real, pad)
    return best[1] / best[0] - 1.0 if best and best[0] else 0.0


def fit_linear(samples: list, fixed_A: float | None = None) -> dict:
    """samples [(measured ms, active, tiles)] of one config's slowest chip -> ms = C0 + max(A act, B tiles), least squares
    over a grid for (A, B) (A fixed when given), C0 the mean residual."""
    T = np.array([s[0] for s in samples], dtype=float)
    act = np.array([s[1] for s in samples], dtype=float)
    til = np.array([s[2] for s in samples], dtype=float)

    def sse(A, B):
        W = np.maximum(A * act, B * til)
        c0 = float((T - W).mean())
        r = T - W - c0
        return float((r**2).sum()), c0

    As = [fixed_A] if fixed_A is not None else list(np.linspace(0.001, 0.08, 80))
    best = None
    for A in As:
        for B in np.linspace(0.0002, 0.02, 100):
            e, c0 = sse(A, B)
            if best is None or e < best[0]:
                best = (e, A, B, c0)
    e, A, B, c0 = best
    for A2 in [A] if fixed_A is not None else np.linspace(max(A - 0.001, 1e-5), A + 0.001, 41):
        for B2 in np.linspace(max(B - 0.0002, 1e-6), B + 0.0002, 41):
            e2, c2 = sse(A2, B2)
            if e2 < e:
                e, A, B, c0 = e2, A2, B2, c2
    ss = float(((T - T.mean()) ** 2).sum())
    return {"A_ms": float(A), "B_ms": float(B), "C0_ms": float(c0), "r2": 1.0 - e / ss if ss > 0 else float("nan"),
            "rmse_ms": math.sqrt(e / len(T)), "samples": len(T)}  # fmt: skip


def chunk_table(layers: dict, shape: tuple, params: dict) -> dict:
    """layers {L: own [chips, EPC]}; params {name: (tp, A, B, C0)} -> {name: {"ms": sum over layers, "busy_over_mean":
    mean over layers, "per_layer": [...]}}."""
    out = {}
    for name, (tp, A, B, C0) in params.items():
        rows = [layer_ms(own, shape, tp, A, B, C0) for _, own in sorted(layers.items())]
        out[name] = {
            "ms": float(sum(r["ms"] for r in rows)),
            "busy_over_mean": float(np.mean([r["busy_over_mean"] for r in rows])),
            "per_layer": [round(r["ms"], 4) for r in rows],
        }
    return out


# ------------------------------------------------------------------------------------------- routing for the benchmark
def steer_bias(scores: np.ndarray, target: np.ndarray, K: int, iters: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A router bias making top-K of (scores + bias) give about `target` tokens an expert (scores [T, E] fp32, the
    head's selection rule): iterative log-ratio steps. -> (bias [E], the counts it gives)."""
    T, E = scores.shape
    tgt = np.asarray(target, dtype=float)
    tgt = tgt * (T * K / tgt.sum())
    b = np.zeros(E, dtype=np.float32)
    step = float(scores.std()) * 0.5
    best = (None, None, np.inf)
    for i in range(iters):
        top = np.argpartition(-(scores + b), K - 1, axis=1)[:, :K]
        cnt = np.bincount(top.reshape(-1), minlength=E).astype(float)
        err = float(np.abs(cnt - tgt).sum())
        if err < best[2]:
            best = (b.copy(), cnt.copy(), err)
        lr = step * (0.5 if i < iters // 2 else 0.15)
        b = (b + lr * (np.log(tgt + 1.0) - np.log(cnt + 1.0))).astype(np.float32)
    return best[0], best[1]


def sqrtsoftplus(z: np.ndarray) -> np.ndarray:
    return np.sqrt(np.logaddexp(0.0, z))
