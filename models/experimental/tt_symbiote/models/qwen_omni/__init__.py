# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-MoE–specific helpers (HF integration patches, etc.)."""

from models.experimental.tt_symbiote.models.qwen_omni.hf_generation_compat import (
    apply_qwen3_omni_talker_prepare_inputs_fix,
)

__all__ = ["apply_qwen3_omni_talker_prepare_inputs_fix"]
