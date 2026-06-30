# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3VLVisionModel integration notes.

Phase 1 (tt-symbiote MVP): the vision encoder is `transformers.Qwen3VLVisionModel`
loaded by `Cosmos3OmniDiffusersPipeline.from_pretrained(...)`. tt-symbiote
replaces its `nn.Linear` layers via the standard replacement dict; everything
else (patch embedding, layer norm, GELU, attention softmax, deepstack
gather at layers 8/16/24) runs on PyTorch.

Phase 2 (native tt-nn): port as a stand-alone module under
`models/tt_dit/encoders/qwen3_vl/`. ≈ 1–2 weeks per the plan.
"""

from __future__ import annotations
