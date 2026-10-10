# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash pure-ttnn prefill (the encoder half: layers 0..19 + layer 20's KV) for the disaggregated
prefill -> KV hand-off -> 48-stage decode ring (tt-blaze DS41F-0037). Plan: tt-blaze docs/deepseek_v4_1_flash/
disagg_prefill_plan.md."""
