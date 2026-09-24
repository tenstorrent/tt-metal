# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash prefill in pure ttnn: layer-kind schedule, KV geometry, mHC, block, transformer, runtime.

Built on the V4 pieces already in this package (``tt/mla/heavily_compressed_attention.py``, the V4 MoE gate,
``reference/deepseek_v4``) and driven by the model-agnostic engine in ``models/demos/common/prefill``. Plan:
tt-blaze ``docs/plans/deepseek_v4_flash_prefill_ttnn_plan.md``.
"""
