# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 expert-parallel (EP) MoE.

The deployment routed-expert path. Import the pieces directly from their submodules:
- ``tt_minimax_moe.TtMiniMaxMoE`` — the EP MoE block (DeepSeek dispatch/combine + M3 experts).
- ``composite_routed_expert.CompositeRoutedExpert`` — per-local-expert clamped-swigluoai FFN.
- ``activation.apply_swiglu`` — the clamped gpt-oss SwiGLU (shared with the dense MLP).
"""
