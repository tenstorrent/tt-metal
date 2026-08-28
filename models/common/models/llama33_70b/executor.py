# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.3-70B executor construction entry point."""

from models.common.models.llama33_70b.hf_adaptor import Llama33_70BForCausalLM
from models.common.models.llama3_executor import Llama33_70BExecutor, Llama33_70BExecutorConfig
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.sampling_state_1d import SamplingState1D


def build_llama33_70b_executor(llm: Llama33_70BForCausalLM, config: Llama33_70BExecutorConfig) -> Llama33_70BExecutor:
    """Build one executor around an already-loaded Llama 3.3-70B adapter."""

    return Llama33_70BExecutor(llm.model, llm.runtime_config, config)
