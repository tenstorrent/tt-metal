# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Segment sizes of the KDA state migration contract (``KDA_STATE_MIGRATION.md``).

Both are clean divisions of Kimi-K3's head_dim 128. The convolution segment is also the page of KDA's native
convolution tail, so a slab copy of the convolution state is a page copy.
"""

KDA_RECURRENT_BAND_WIDTH = 32
KDA_CONVOLUTION_PAGE_CHANNELS = 64
