# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch references for Kimi K3 attention residuals (AttnRes), in three tiers.

``hf_attn_res.py`` is the vendored upstream HF modeling code, under the Kimi K3
License in ``LICENSE-Kimi-K3`` rather than Apache-2.0.

``attn_res_reference.py`` is the naive fp64 ground truth, written from the
published definition and deliberately taking none of the algebraic shortcuts the
implementations take.

``attn_res.py`` is the folded torch form the device composite mirrors: one query
carrying ``res_norm.weight * res_proj.weight``, ``rsqrt`` pulled out of the dot.
"""
