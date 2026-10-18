# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared by the dFlash fused decoder, its width ladder and the contract adapter."""

# Positions kept free between a verify block's last write and the captured width
# or generation horizon that must contain it. The width ladder, width_for, the
# generation budget and the contract's coverage check all add this same margin.
#
# The value is headroom, not a derived bound. Every width and horizon is rounded
# up to a multiple of 1024, so it is also a multiple of the 64-token KV block and
# the 32-row tile, and ``start + P_v <= width`` alone keeps every verify write in
# a whole block inside the width. What the margin changes is which width a start
# selects: a start within ``P_v + margin`` of a width's end moves to the next one.
# A smaller value has not been measured. The ladder prepared in warmup and the
# widths selected while serving read this one constant, so they agree whatever
# its value.
VERIFY_WIDTH_MARGIN = 64
