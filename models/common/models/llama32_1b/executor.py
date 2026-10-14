# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.2-1B executor construction entry point."""

from models.common.models.llama3_executor import Llama32_1BExecutor, Llama32_1BExecutorConfig
from models.common.models.llama32_1b.hf_adaptor import Llama32_1BForCausalLM


def build_llama32_1b_executor(llm: Llama32_1BForCausalLM, config: Llama32_1BExecutorConfig) -> Llama32_1BExecutor:
    """Build one executor around an already-loaded Llama 3.2-1B adapter."""

    return Llama32_1BExecutor(llm.model, llm.runtime_config, config)
