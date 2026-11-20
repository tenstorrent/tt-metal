# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill: the runtime the ``hy4_preview`` adapter builds (``runtime.py``) and its mesh layouts
(``layout.py``: TP32 / DP1, plus the GLM-5.3 dev / numerics layout).

Hy4 runs the GLM-5.3 prefill stack (``TtPrefillTransformer``); this package holds only what differs.
The attention sink and output gate (#65921) live in the shared ttMLA; the iHC residual (#65891) will land
here or in the shared modules.
"""
