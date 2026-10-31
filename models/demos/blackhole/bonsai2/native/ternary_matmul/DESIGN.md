# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Design note: how the per-group scale is applied, and what it costs.

The engine needs ``y[m, n] = sum_k x[m, k] * code[n, k] * scale[n, k // G]``.
Because ``scale`` depends on ``n`` (the output column) *and* on the K-group, it
cannot be folded into the activation (shared by all ``n``) nor into the weight
(codes are +-1, exact). So the sum has to be split per group and each group's
partial scaled before it is accumulated::

    acc += (sum over the group's tiles of x*w) * scale_row

Accumulating a group partial in Dst and scaling it in place would be ideal, and
the MVMUL can accumulate across K tiles in Dst perfectly well. The problem is
the scale multiply: every elementwise op exposed by the compute API takes its
operands from CBs and its result from Dst, or takes one operand from Dst via
``add_reuse_dest_tiles`` -- but there is **no** broadcast-with-Dst-reuse form, so
``acc += partial * scale_row`` with both operands in Dst does not exist.

Three ways to get the scale in, with the cost of each:

**S1  partial through L1 (chosen).** Accumulate the group's tiles into Dst, pack
     the partial to an L1 CB, then run ``mul_tiles_bcast_rows`` reading the
     partial back from that CB while adding into the accumulator held in another
     CB tile. Cost: one pack + one unpack per group per output column. For G=128
     that is one extra round trip per 4 K-tiles, i.e. every ~1.3 KB of streaming.

**S2  scale the accumulator, not the partial.** Reorder to
     ``acc = acc * (s[i]/s[i-1]) + partial * ...`` -- only works if consecutive
     scales form a telescoping product, which they do not.

**S3  one scale per pass.** Force all groups in a subblock to share a scale
     (the plan's "forced shared exponent" idea, applied to scales rather than
     tile exponents). Halves the number of scale applications at the cost of
     accuracy; that is exactly the A2 variant, so S1 is the honest baseline to
     measure A2 against.

S1 is implemented and measured first. If the extra round trip costs more than a
few percent of the streaming time, S3 becomes attractive for other reasons and
A2 will show it.
"""

#: Extra pack+unpack operations per (output column, group) in the S1 scheme.
S1_EXTRA_ROUND_TRIPS_PER_GROUP = 1
