# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch references for Kimi K3 attention residuals (AttnRes), in two tiers.

``attn_res_reference.py`` is the naive fp64 ground truth, written from the
published definition and deliberately taking none of the algebraic shortcuts the
implementations take.

``attn_res.py`` is the folded torch form the device modules mirror: one query
carrying ``res_norm.weight * res_proj.weight``, ``rsqrt`` pulled out of the dot.
Every device PCC gate is measured against this tier, and the tier above is what
pins it — see ``tests/attn_res/model/test_attn_res_reference.py``.
"""
