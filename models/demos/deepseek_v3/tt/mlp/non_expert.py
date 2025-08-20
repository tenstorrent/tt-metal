# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


from models.demos.deepseek_v3.tt.mlp.mlp_1d import MLP1D


class NonExpert(MLP1D):
    """
    Non-expert layer for Mixture-of-Experts (MoE) models.
    This MLP layer is used for the first 3 layers of DeepSeek.
    """
