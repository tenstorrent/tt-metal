# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from models.demos.deepseek_v3.tt.mlp.mlp_dequant import MLPDequant


class NonExpert(MLPDequant):
    """
    Non-expert layer for Mixture-of-Experts (MoE) models.
    This MLP layer is used for the first 3 layers of DeepSeek.
    """
