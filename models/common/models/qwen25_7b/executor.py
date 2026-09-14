# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-7B construction entry point over the shared Qwen2 executor."""

from models.common.models.qwen2_executor import Qwen25Executor, Qwen25ExecutorConfig
from models.common.models.qwen2_executor import build_qwen25_7b_executor as _build_qwen25_7b_executor
from models.common.models.qwen25_7b.hf_adaptor import Qwen25ForCausalLM


def build_qwen25_7b_executor(llm: Qwen25ForCausalLM, config: Qwen25ExecutorConfig) -> Qwen25Executor:
    """Build one executor around an already-loaded Qwen2.5 model adapter."""

    return _build_qwen25_7b_executor(llm, config)
