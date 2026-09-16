# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Explaining a golden-vs-device mismatch.

A bare "assert failed" says only that the numbers differ. What tells you *why*
is the shape of the disagreement: whether it is spread thinly over every tile
(a precision model that is slightly off) or concentrated in a few (a tile
index, a stride, a feedback order). This builds that picture, plus the chain
the golden ran, so a failure names its own suspect.
"""

from typing import Optional, Sequence

import torch

from .operations.chain import Chain, StageRecord


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[finite], b[finite]
    if a.numel() < 2:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float("nan") if denom == 0 else (a @ b / denom).item()


def local_step(
    golden: torch.Tensor, magnitude: torch.Tensor, mantissa_bits: int
) -> torch.Tensor:
    """The MX lattice step at each element's own magnitude.

    Mirrors ``_mxfp_block_aware_compare``: an MX-float element carries its own
    exponent above the block scale, so the spacing between representable values
    depends on the value. A comparison tolerance of N steps is therefore
    relative, and ranking failures by absolute error hides the ones that
    actually fail.
    """
    safe = magnitude > 0
    step = torch.ones_like(magnitude)
    step[safe] = torch.pow(
        2.0, torch.floor(torch.log2(magnitude[safe])) - mantissa_bits
    )
    return step


def describe_mismatch(
    golden: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str = "",
    mantissa_bits: int = 0,
    datums_per_tile: int = 1024,
    chain: Optional[Chain] = None,
    trace: Optional[Sequence[StageRecord]] = None,
    dest: Optional[torch.Tensor] = None,
    worst: int = 8,
    max_steps: int = 2,
) -> str:
    """Describe how `actual` differs from `golden`, as a failure message."""
    g = golden.flatten().float()
    a = actual.flatten().float()
    if g.numel() != a.numel():
        return f"GOLDEN MISMATCH {context}\n  length {g.numel()} vs {a.numel()}"

    err = (g - a).abs()
    a_mag = torch.maximum(g.abs(), a.abs())
    bad = err > 0
    lines = [f"GOLDEN MISMATCH  {context}".rstrip()]
    lines.append(
        f"  {int(bad.sum())} / {g.numel()} datums differ "
        f"({100.0 * bad.float().mean():.2f}%)   PCC {_pcc(g, a):.9f}"
    )
    lines.append(
        f"  max |err| {err.max():.6g}   mean |err| {err.mean():.6g}   "
        f"golden range [{g.min():.4g}, {g.max():.4g}]"
    )

    # Spread across tiles separates a precision error from an indexing one.
    tiles = g.numel() // datums_per_tile
    if tiles > 1:
        per_tile = err.reshape(tiles, datums_per_tile)
        counts = (per_tile > 0).sum(dim=1)
        worst_per_tile = per_tile.max(dim=1).values
        lines.append(
            "  per tile:  "
            + "  ".join(
                f"t{i}:{int(c)}/{datums_per_tile}@{w:.3g}"
                for i, (c, w) in enumerate(zip(counts, worst_per_tile))
            )
        )
        touched = int((counts > 0).sum())
        lines.append(
            f"  -> {touched}/{tiles} tiles affected"
            + (
                "; concentrated, suspect tile indexing or ordering"
                if touched < tiles
                else "; spread evenly, suspect the precision model"
            )
        )

    # Rank by lattice steps, not absolute error. MX-float tolerance is relative
    # to each element's own magnitude, so the largest absolute error is often a
    # datum that comfortably passes while a much smaller error at a small
    # magnitude is the one that fails.
    # Differing is not failing. The MX compare accepts `max_steps` lattice
    # steps, so most of what shows up here is within tolerance and is only
    # listed because it is non-zero. Say which datums actually break the test,
    # or the top of this table reads as the cause when it is noise.
    steps = err / local_step(g, a_mag, mantissa_bits) if mantissa_bits else err
    if mantissa_bits:
        over = steps > max_steps
        lines.append(
            f"  {int(over.sum())} of those exceed the {max_steps}-step tolerance "
            f"-- those are the failures; the rest differ but pass"
        )
    else:
        lines.append(
            "  no lattice model for this output format, so the 'steps' column "
            "below is absolute error and the tolerance is not shown"
        )
    order = torch.argsort(steps, descending=True)[:worst]
    # The golden's Dest before the pack, when the caller collected it: a datum
    # can disagree because the math left a different value in Dest, or because
    # the packer treated the same value differently. Only the pair tells you
    # which, and they call for different fixes.
    d = (
        dest.flatten().float()
        if dest is not None and dest.numel() == g.numel()
        else None
    )
    lines.append(f"  worst {len(order)} datums, by lattice steps:")
    lines.append(
        f"    {'index':>8} {'tile':>5} {'golden':>14} {'device':>14} "
        f"{'abs err':>12} {'steps':>8}"
        + (f" {'golden dest':>14}" if d is not None else "")
    )
    for i in order.tolist():
        if err[i] == 0:
            break
        lines.append(
            f"    {i:>8} {i // datums_per_tile:>5} "
            f"{g[i].item():>14.7g} {a[i].item():>14.7g} {err[i].item():>12.6g} "
            f"{steps[i].item():>8.2f}"
            + (f" {d[i].item():>14.7g}" if d is not None else "")
        )

    # A device zero against a nonzero golden is the one case where the Dest
    # value is decisive: hardware writes no denormal (RES_A2), and a sweep
    # confirmed it holds everything at or above the fp16 normal floor. So a
    # golden Dest at or above that floor rules Dest flushing out.
    if d is not None:
        zeroed = (a == 0) & (g != 0)
        if zeroed.any():
            floor = 2.0**-14
            above = int((d[zeroed].abs() >= floor).sum())
            lines.append(
                f"  {int(zeroed.sum())} datums the device zeroed; golden Dest is "
                f">= 2^-14 for {above} of them"
                + (
                    " -> the device would have held those, so the divergence is "
                    "in the math or the pack, not Dest FTZ"
                    if above
                    else " -> consistent with Dest FTZ"
                )
            )

    if chain is not None:
        lines.append(f"  golden chain: {chain}")
    if trace:
        lines.append("  last tile's trace:")
        lines.extend(f"    {record}" for record in trace)
    return "\n".join(lines)
