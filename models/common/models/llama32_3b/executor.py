# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.2-3B executor construction entry point."""

from models.common.models.llama32_3b.hf_adaptor import Llama32_3BForCausalLM
from models.common.models.llama3_executor import Llama32_3BExecutor, Llama32_3BExecutorConfig


def build_llama32_3b_executor(llm: Llama32_3BForCausalLM, config: Llama32_3BExecutorConfig) -> Llama32_3BExecutor:
    """Build one executor around an already-loaded Llama 3.2-3B adapter."""

    return Llama32_3BExecutor(llm.model, llm.runtime_config, config)
