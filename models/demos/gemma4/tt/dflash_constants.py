# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared by the dFlash fused decoder, its width ladder and the contract adapter."""

# Positions kept free between a verify block's last write and the captured width
# or generation horizon that must contain it. The width ladder, width_for, the
# generation budget and the contract's coverage check all add this same margin.
VERIFY_WIDTH_MARGIN = 64
