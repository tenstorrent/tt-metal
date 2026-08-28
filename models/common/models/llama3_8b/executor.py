# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1-8B executor construction entry point."""

from models.common.models.llama3_8b.hf_adaptor import Llama3ForCausalLM
from models.common.models.llama3_executor import Llama3Executor, Llama3ExecutorConfig
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.sampling_state_1d import SamplingState1D


def build_llama3_executor(llm: Llama3ForCausalLM, config: Llama3ExecutorConfig) -> Llama3Executor:
    """Build one executor around an already-loaded Llama 3.1-8B adapter."""

    return Llama3Executor(llm.model, llm.runtime_config, config)
