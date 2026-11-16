# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill: the runtime the ``hy4_preview`` adapter builds (``runtime.py``).

Hy4 runs the GLM-5.3 prefill stack (``TtPrefillTransformer``); this package holds only what differs.
The iHC residual (#65891) and the attention sink (#65921) will land here or in the shared modules.
"""
