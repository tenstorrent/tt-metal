# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.7-Code Model Configuration (text tower only).

K2.7-Code is architecturally identical to Kimi-K2.6 -- same 61 layers, 384 routed experts, same
MLA and MoE dimensions, same RoPE/YaRN parameters -- so every value is inherited rather than
restated. Verified field-by-field against ``text_config`` of
``/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized/config.json``; see
``tests/torch/test_kimi_k2_7_config.py``, which fails the day that stops being true.

Kept as a distinct name so K2.7 tests read K2.7, and so a real divergence becomes an override here
instead of a fork of the whole class. ``KimiK27Adapter`` deliberately keeps ``model_config =
KimiK26Config`` -- the runner resolves dims from the checkpoint, and the two are the same object's
worth of numbers either way.
"""

from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config


class KimiK27Config(KimiK26Config):
    """Kimi K2.7-Code model dimensions. Identical to K2.6; see the module docstring."""
