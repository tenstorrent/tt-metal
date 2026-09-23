# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: WanAttention must read its SDPA compute config at call time.

apply_quant_config (pipelines/wan/quant_config.py) replaces
attn1.sdpa_compute_kernel_config after construction; a dict captured in
__init__ would silently keep the old fidelity.
"""

from models.tt_dit.models.transformers.wan2_2.attention_wan import WanAttention


def _bare_attention():
    # Bypass __init__, which needs a mesh device; only the kwargs helper is under test.
    return object.__new__(WanAttention)


def test_legacy_sdpa_kwargs_follow_reassigned_compute_config():
    attention = _bare_attention()
    attention.sdpa_recipe_kwargs = None
    attention.sdpa_compute_kernel_config = "initial"
    assert attention._self_sdpa_kwargs() == {"compute_kernel_config": "initial"}
    attention.sdpa_compute_kernel_config = "quantized"
    assert attention._self_sdpa_kwargs() == {"compute_kernel_config": "quantized"}


def test_recipe_sdpa_kwargs_ignore_compute_config():
    attention = _bare_attention()
    attention.sdpa_recipe_kwargs = {"precision": "recipe", "inputs_prepared": False}
    attention.sdpa_compute_kernel_config = "quantized"
    assert attention._self_sdpa_kwargs() == {"precision": "recipe", "inputs_prepared": False}
