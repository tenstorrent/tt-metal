# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen2-7B construction entry point over the shared Qwen2 executor."""

from models.common.models.qwen2_7b.hf_adaptor import Qwen2ForCausalLM
from models.common.models.qwen2_executor import (
    Qwen2Executor,
    Qwen2ExecutorConfig,
    build_qwen2_7b_executor as _build_qwen2_7b_executor,
)


def build_qwen2_7b_executor(llm: Qwen2ForCausalLM, config: Qwen2ExecutorConfig) -> Qwen2Executor:
    """Build one executor around an already-loaded Qwen2 model adapter."""

    return _build_qwen2_7b_executor(llm, config)
