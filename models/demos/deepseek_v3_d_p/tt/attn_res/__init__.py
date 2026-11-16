# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of Kimi K3 attention residuals (AttnRes).

``attn_res.py`` holds the read itself, split so the sealed half amortizes across a
whole block: ``inter_block`` once, ``merge`` per read site.
``attn_res_stream.py`` drives the residual stream over a stack of layers with the
real seal cadence.
"""
