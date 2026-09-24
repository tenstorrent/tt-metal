# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash checkpoint loading for the pure-ttnn prefill: dequant (FP4 experts, FP8 block-scaled
projections), the per-layer taxonomy, and the native -> reference name map. ``dequant.py`` and ``layer_weights.py``
are copies of ``tt-blaze/blaze/weights/deepseek_v4_flash/{dequant,layer_weights}.py`` (tt-metal must not import
blaze); keep them in sync by copy, not by divergence."""
