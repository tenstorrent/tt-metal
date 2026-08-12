# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch references for Kimi K3 attention residuals (AttnRes), in three tiers.

``hf_attn_res.py`` is upstream's ``_apply_attn_res``, vendored byte-identical
under ``LICENSE-Kimi-K3``. Nothing imports it outside the test that scores the
tier below against it, which is its entire job: it is the only tier not written
here, and so the only evidence that the definition below was read correctly. It
widens with ``.float()`` and therefore computes in fp32 whatever it is handed,
which is why it anchors the ladder rather than heading it.

``attn_res_reference.py`` is the naive fp64 ground truth, written from the
published definition and deliberately taking none of the algebraic shortcuts the
implementations take.

``attn_res.py`` is the folded torch form the device modules mirror: one query
carrying ``res_norm.weight * res_proj.weight``, ``rsqrt`` pulled out of the dot.
Every device PCC gate is measured against this tier, and the tiers above are what
pin it — see ``tests/attn_res/model/test_attn_res_reference.py``.
"""
