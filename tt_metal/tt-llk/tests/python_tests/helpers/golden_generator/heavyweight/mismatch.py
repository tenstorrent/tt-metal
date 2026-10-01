# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The two things a golden failure needs that a value dump cannot give.

``passed_test`` already prints the failing tiles with the bad datums
highlighted, which answers *where* in the tile. What it cannot answer is:

* **how badly** -- for an MX output the tolerance is a number of lattice steps,
  and the step is relative to each element's own magnitude. Ranked by absolute
  error, the top of a table is large-magnitude datums that comfortably pass
  while the one that actually failed sits further down.
* **which stage** -- a datum can disagree because the math left a different
  value in Dest, or because the packer treated the same value differently.
  Those need different fixes and the packed output cannot distinguish them.

So this adds a step-ranked table, the golden's pre-pack Dest beside it, and the
chain that produced it. Use it alongside ``passed_test``, not instead of it.
"""

from typing import Optional, Sequence

import torch
from helpers.utils import calculate_pcc

from .operations.chain import Chain, StageRecord


def local_step(magnitude: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    """The MX-float lattice step at each element's own magnitude.

    Mirrors ``_mxfp_block_aware_compare``: an MX-float element carries its own
    exponent above the block scale, so the spacing between representable values
    depends on the value. Zero and negatives fall back to a step of 1, which
    only affects ranking.
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
    """Rank a golden-vs-device disagreement by lattice steps.

    `mantissa_bits` is the MX-float element width (``MXFP_MANTISSA_BITS``); 0
    means no lattice model, and then the ranking falls back to absolute error,
    the steps column is omitted and the tolerance is not shown. `dest` is the
    golden's pre-pack Dest, when the caller collected it.
    """
    g = golden.flatten().float()
    a = actual.flatten().float()
    if g.numel() != a.numel():
        return f"GOLDEN MISMATCH {context}\n  length {g.numel()} vs {a.numel()}"

    err = (g - a).abs()
    steps = (
        err / local_step(torch.maximum(g.abs(), a.abs()), mantissa_bits)
        if mantissa_bits
        else err
    )
    differ = err > 0

    lines = [f"GOLDEN MISMATCH  {context}".rstrip()]
    # Differing is not failing. Most of what shows up here is within tolerance
    # and is only listed because it is non-zero -- say which datums actually
    # break the test, or the top of the table reads as the cause when it is
    # noise.
    if mantissa_bits:
        over = steps > max_steps
        lines.append(
            f"  {int(differ.sum())} / {g.numel()} datums differ, of which "
            f"{int(over.sum())} exceed the {max_steps}-step tolerance -- those "
            f"are the failures; the rest differ but pass"
            f"   PCC {calculate_pcc(golden, actual):.9f}"
        )
    else:
        lines.append(
            f"  {int(differ.sum())} / {g.numel()} datums differ"
            f"   PCC {calculate_pcc(golden, actual):.9f}\n"
            f"  No lattice model for this output format, so the ranking below "
            f"is by absolute error and the tolerance is not shown"
        )

    order = torch.argsort(steps, descending=True)[:worst]
    d = (
        dest.flatten().float()
        if dest is not None and dest.numel() == g.numel()
        else None
    )
    # Without a lattice model `steps` is just `err` again, so leave the column
    # out rather than print a second copy of a number under a name it does not
    # mean.
    lines.append(
        "  worst datums, by lattice steps:"
        if mantissa_bits
        else "  worst datums, by absolute error:"
    )
    lines.append(
        f"    {'index':>8} {'tile':>5} {'golden':>14} {'device':>14} {'abs err':>12}"
        + (f" {'steps':>8}" if mantissa_bits else "")
        + (f" {'golden dest':>14}" if d is not None else "")
    )
    for i in order.tolist():
        if err[i] == 0:
            break
        lines.append(
            f"    {i:>8} {i // datums_per_tile:>5} "
            f"{g[i].item():>14.7g} {a[i].item():>14.7g} {err[i].item():>12.6g}"
            + (f" {steps[i].item():>8.2f}" if mantissa_bits else "")
            + (f" {d[i].item():>14.7g}" if d is not None else "")
        )

    # A device zero against a nonzero golden is the one case where the Dest
    # value is decisive: hardware writes no denormal, and a sweep confirmed it
    # holds everything at or above the fp16 normal floor. So a golden Dest at or
    # above that floor rules Dest flushing out.
    if d is not None:
        zeroed = (a == 0) & (g != 0)
        if zeroed.any():
            above = int((d[zeroed].abs() >= 2.0**-14).sum())
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
        lines.append("  trace (all blocks, index restarts per output tile):")
        lines.extend(f"    {record}" for record in trace)
    return "\n".join(lines)
