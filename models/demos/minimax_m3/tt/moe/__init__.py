# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 expert-parallel (EP) MoE.

The deployment routed-expert path. Import the pieces directly from their submodules:
- ``tt_minimax_moe.TtMiniMaxMoE`` — the EP MoE block (DeepSeek dispatch/combine + the fused
  ``unified_routed_expert_ffn`` kernel with M3's clamped swigluoai activation).
- ``activation.apply_swiglu`` — the clamped gpt-oss SwiGLU (used by the dense MLP).
"""
