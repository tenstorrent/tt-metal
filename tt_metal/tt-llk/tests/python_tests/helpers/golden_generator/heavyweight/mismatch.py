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

from typing import Optional, Sequence, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import format_dict
from helpers.utils import _MXFP_COMPARE_PARAMS, calculate_pcc, mxfp_local_step

from .operations.chain import Chain, StageRecord


def lattice_step(
    magnitude: torch.Tensor, output_format: DataFormat
) -> Optional[Tuple[torch.Tensor, int]]:
    """``(step, max_steps)`` for an MX-float output, or ``None`` if unmodelled.

    Delegates to :func:`helpers.utils.mxfp_local_step`, the same calculation
    ``_mxfp_block_aware_compare`` uses to decide pass/fail, so a datum this
    report calls a failure is one the comparator rejects. Computing the step
    here independently went wrong in exactly the way that invites: the
    normal-value formula alone reports the smallest E4M3 subnormal as eight
    steps from zero, where the comparator has them adjacent.
    """
    params = _MXFP_COMPARE_PARAMS.get(output_format)
    if params is None:
        return None
    mantissa_bits, max_steps, element_max_normal, element_min_subnormal = params
    step = mxfp_local_step(
        magnitude, mantissa_bits, element_max_normal, element_min_subnormal
    )
    return step, max_steps


def describe_mismatch(
    golden: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str = "",
    output_format: Optional[DataFormat] = None,
    datums_per_tile: int = 1024,
    chain: Optional[Chain] = None,
    trace: Optional[Sequence[StageRecord]] = None,
    dest: Optional[torch.Tensor] = None,
    dest_format: Optional[DataFormat] = None,
    worst: int = 8,
) -> str:
    """Rank a golden-vs-device disagreement by lattice steps.

    `output_format` selects the lattice the result landed on; a format with no
    MX-float model falls back to ranking by absolute error, omits the steps
    column and does not claim a tolerance. `dest` is the golden's pre-pack
    Dest, when the caller collected it, and `dest_format` is the format it was
    held in -- needed to say anything about flush-to-zero, since the floor is
    the Dest format's smallest normal and differs by ~2^112 between fp16 and
    bf16. Without it the FTZ line is omitted rather than guessed.
    """
    g = golden.flatten().float()
    a = actual.flatten().float()
    if g.numel() != a.numel():
        return f"GOLDEN MISMATCH {context}\n  length {g.numel()} vs {a.numel()}"

    err = (g - a).abs()
    lattice = lattice_step(torch.maximum(g.abs(), a.abs()), output_format)
    if lattice:
        # A step of 0 means both values were 0, so the error is 0 too. Dividing
        # would give NaN, and torch sorts NaN first descending -- which empties
        # the table below, since it stops at the first zero-error datum.
        steps = torch.zeros_like(err)
        nonzero = lattice[0] > 0
        steps[nonzero] = err[nonzero] / lattice[0][nonzero]
    else:
        steps = err
    differ = err > 0

    lines = [f"GOLDEN MISMATCH  {context}".rstrip()]
    # Differing is not failing. Most of what shows up here is within tolerance
    # and is only listed because it is non-zero -- say which datums actually
    # break the test, or the top of the table reads as the cause when it is
    # noise.
    if lattice:
        max_steps = lattice[1]
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
        if lattice
        else "  worst datums, by absolute error:"
    )
    lines.append(
        f"    {'index':>8} {'tile':>5} {'golden':>14} {'device':>14} {'abs err':>12}"
        + (f" {'steps':>8}" if lattice else "")
        + (f" {'golden dest':>14}" if d is not None else "")
    )
    for i in order.tolist():
        if err[i] == 0:
            break
        lines.append(
            f"    {i:>8} {i // datums_per_tile:>5} "
            f"{g[i].item():>14.7g} {a[i].item():>14.7g} {err[i].item():>12.6g}"
            + (f" {steps[i].item():>8.2f}" if lattice else "")
            + (f" {d[i].item():>14.7g}" if d is not None else "")
        )

    # A device zero against a nonzero golden is the one case where the Dest
    # value is decisive: hardware writes no denormal, so a golden Dest at or
    # above the Dest format's smallest normal rules Dest flushing out. The floor
    # has to come from that format -- using fp16's 2^-14 for a bf16 Dest calls
    # everything below it "consistent with FTZ" when the golden Dest held it
    # perfectly well and the loss was really in the pack.
    floor = None
    if dest_format is not None:
        dtype = format_dict[dest_format]
        if dtype.is_floating_point:
            floor = torch.finfo(dtype).smallest_normal
    if d is not None and floor is not None:
        zeroed = (a == 0) & (g != 0)
        if zeroed.any():
            above = int((d[zeroed].abs() >= floor).sum())
            lines.append(
                f"  {int(zeroed.sum())} datums the device zeroed; golden Dest is "
                f">= {floor:.3g} ({dest_format}'s smallest normal) for {above} "
                f"of them"
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
