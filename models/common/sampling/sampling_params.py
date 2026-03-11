# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    """
    Used in Generator decode forward functions for greedy decoding / sampling on device.
    The same data class exists in vLLM at vllm/v1/worker/tt_model_runner.py.
    """

    temperature: float | list[float]
    top_k: int | list[int]
    top_p: float | list[float]
    presence_penalty: float | list[float] = 0.0
    frequency_penalty: float | list[float] = 0.0
    repetition_penalty: float | list[float] = 1.0
    seed: int | list[int] | None = None
    # Number of top log-probs to return per token (1-20), or 0/None to disable.
    # Values above 20 are capped to 20. Values below 1 (except 0/None) will assert.
    num_logprobs: int | list[int] = 0
