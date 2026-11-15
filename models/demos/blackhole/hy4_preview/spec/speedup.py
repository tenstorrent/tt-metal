# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expected speculative-decode speedup of Hy4 at B=4 x 1M (1 user per mesh row; global rows T = 4 (1+k) in the verify step).

Every constant is a cited MEASUREMENT or a labelled ASSUMPTION; the 'other' per-layer time is the big unknown (no full decode layer is measured yet): the table is
printed for three values. Run ``python -m models.demos.blackhole.hy4_preview.spec.speedup`` (needs no device).

Measured inputs (all per layer, per chip, traced):
  * routed+shared MoE decode, bfp4 experts, real activations, global batch T = 4 / 8 / 16 rows: 363 / 430 / 584 us   [layer-moe, TICKLOG 2026-11-13T20:09:05Z, logs layer-moe/logs]
  * sparse attention core, DSV4 key-split kernel at K576/topk2048, H32 padded, S query rows on the chip:
    S=1 52.5, S=2 61 (kc256), S=4 61 (G8 kc128), S=8 91 (G4 kc256) us                                              [attn-op-probe/RESULT.md split table]; S=3 interpolated 61, S=5 ~70 (ASSUMPTION)
Assumptions (UNMEASURED, to be replaced by the device stage):
  * other = projections q_a/q_b/kv_a/o/linear_gate + indexer scan/top-k (21 of 78 layers) + norms + iHC + shared-expert + CCL, us/layer at 1 row/user. DEFAULT 130 (sweep 80-200).
    Rows scaling: other(r) = other * (1 + ROW_SLOPE (r-1)), r = rows per user.
  * lm_head + sampler 0.5 ms per call (154880 x 6144 bf16 / 8 columns ~ 238 MB per chip at ~420 GB/s), replicated over the 4 rows: one call per verify step (the
    k+1 rows share the weight read) and one per draft step (1 row per user: the last accepted row is selected first, as DSV4 does with the one-hot).
  * round overhead (host launch of the traced round, accept logic): 0.3 ms.
MTP layer cost per draft step = one more 'layer' with its own MoE (256 experts: same cost table), its own full attention (+ its own indexer scan on every step:
index_share_for_mtp_iteration is not set in the Hy4 config, so step 0 and the recursive steps each scan the keys), plus eh_proj (6144 x 12288 matmul, ~10 us).
"""
from typing import Sequence

from models.demos.blackhole.hy4_preview.spec.policy import expected_tokens

MOE_US = {4: 363.0, 8: 430.0, 16: 584.0}  # measured, global rows T
ATTN_US = {1: 52.5, 2: 61.0, 3: 61.0, 4: 61.0, 5: 70.0, 8: 91.0}  # S rows per chip
ROW_SLOPE = 0.05
LAYERS = 78
LM_HEAD_MS = 0.5
ROUND_OVERHEAD_MS = 0.3
EH_PROJ_US = 10.0
ROWS = 4  # mesh rows = users at B=4


def moe_us(T: int) -> float:
    ts = sorted(MOE_US)
    if T <= ts[0]:
        return MOE_US[ts[0]]
    for a, b in zip(ts, ts[1:]):
        if T <= b:
            return MOE_US[a] + (MOE_US[b] - MOE_US[a]) * (T - a) / (b - a)
    a, b = ts[-2], ts[-1]  # extrapolate with the last slope
    return MOE_US[b] + (MOE_US[b] - MOE_US[a]) * (T - b) / (b - a)


def attn_us(S: int) -> float:
    ks = sorted(ATTN_US)
    for k in ks:
        if S <= k:
            return ATTN_US[k]
    return ATTN_US[ks[-1]]


def layer_us(r: int, other: float) -> float:
    """One main-model layer for r rows per user (global rows T = 4 r)."""
    return moe_us(ROWS * r) + attn_us(r) + other * (1 + ROW_SLOPE * (r - 1))


def plain_ms(other: float = 130.0) -> float:
    return LAYERS * layer_us(1, other) / 1e3 + LM_HEAD_MS


def verify_ms(k: int, other: float = 130.0) -> float:
    return LAYERS * layer_us(k + 1, other) / 1e3 + LM_HEAD_MS


def draft_ms(k: int, other: float = 130.0) -> float:
    """k recursive MTP passes: step 0 over the k+1 verify rows, steps 1..k-1 over one row per user; k lm_head calls."""
    if k == 0:
        return 0.0
    step0 = (layer_us(k + 1, other) + EH_PROJ_US) / 1e3 + LM_HEAD_MS
    rest = (k - 1) * ((layer_us(1, other) + EH_PROJ_US) / 1e3 + LM_HEAD_MS)
    return step0 + rest


def round_ms(k: int, other: float = 130.0) -> float:
    if k == 0:
        # plain round with the MTP state kept current (no draft): verify of 1 row + MTP step 0 on 1 row (writes the MTP KV so a later k>0 round is valid)
        return plain_ms(other) + (layer_us(1, other) + EH_PROJ_US) / 1e3 + ROUND_OVERHEAD_MS
    return verify_ms(k, other) + draft_ms(k, other) + ROUND_OVERHEAD_MS


def alphas(a1: float, decay: float, k: int) -> list:
    """Conditional acceptance of draft i: a1 * decay^(i-1) (decay < 1 models the recursion: the single MTP layer is trained for depth 1)."""
    return [a1 * decay**i for i in range(k)]


def speedup(k: int, a1: float, decay: float = 1.0, other: float = 130.0) -> float:
    """Tokens/s of speculative rounds at k over the plain decode step. k=0 -> plain round / plain step (< 1: the MTP bookkeeping tax)."""
    e = expected_tokens(alphas(a1, decay, max(k, 1)), k)
    return plain_ms(other) / (round_ms(k, other) / e)


def tokens_per_s_user(k: int, a1: float, decay: float = 1.0, other: float = 130.0) -> float:
    e = expected_tokens(alphas(a1, decay, max(k, 1)), k)
    return 1e3 * e / round_ms(k, other)


def breakeven_alpha(k: int, decay: float = 1.0, other: float = 130.0) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if speedup(k, mid, decay, other) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def table(
    other: float = 130.0,
    ks: Sequence[int] = (1, 2, 3),
    a1s: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    decays: Sequence[float] = (1.0, 0.85),
) -> str:
    lines = [
        f"other={other:.0f} us/layer: plain step {plain_ms(other):.1f} ms; "
        + "; ".join(
            f"k={k}: verify {verify_ms(k, other):.1f} + draft {draft_ms(k, other):.1f} = round {round_ms(k, other):.1f} ms"
            for k in (0,) + tuple(ks)
            if k
        )
    ]
    for dec in decays:
        lines.append(f"decay {dec}: speedup over plain (E tokens/round in parentheses)")
        lines.append("a1   " + "  ".join(f"k={k}".rjust(14) for k in ks))
        for a in a1s:
            cells = [f"{speedup(k, a, dec, other):5.2f}x ({expected_tokens(alphas(a, dec, k), k):.2f})" for k in ks]
            lines.append(f"{a:.2f} " + "  ".join(c.rjust(14) for c in cells))
    return "\n".join(lines)


if __name__ == "__main__":
    for o in (80.0, 130.0, 200.0):
        print(table(o))
        print(
            "break-even a1 (decay 1.0 / 0.85): "
            + ", ".join(f"k={k}: {breakeven_alpha(k, 1.0, o):.2f}/{breakeven_alpha(k, 0.85, o):.2f}" for k in (1, 2, 3))
        )
        print()
