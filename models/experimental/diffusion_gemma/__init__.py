# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma 26B-A4B-it bring-up on tt-metal.

A discrete text-diffusion LLM fine-tuned from Gemma-4 26B-A4B (MoE). The text
backbone is identical to ``models/demos/gemma4``; the net-new work is the
block-autoregressive multi-canvas *generation procedure* — bidirectional canvas
attention, a three-phase KV-cache state machine, entropy-budget acceptance
sampling, and self-conditioning.

See ``plan.md`` for the implementation plan and ``AGENTS.md`` for working
context. ``STATUS.md`` tracks what is implemented vs blocked-on-environment.
"""
