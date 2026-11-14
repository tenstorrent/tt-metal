# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side constants of kernels/step_prefetch.hpp."""

TB, NBK, HT = 2048, 5, 4
NCH = 3 * HT
PF_BYTES = 2 * NBK * TB + TB + TB + NCH * 256 + HT * 128 + NCH * 256 + HT * HT * TB
