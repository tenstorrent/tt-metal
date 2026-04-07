# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from itertools import product

import torch
from loguru import logger

from models.common.sampling.sampling_params import SamplingParams


class WarmupForwardMixin:
    """
    This class is used by vLLM.

    Mixin class that provides decode warmup functionality for generator classes.

    This class should be inherited by any generator class that needs to warm up
    the decode forward pass. It requires the following to be defined in the
    inheriting class:
    - self.decode_forward(): method to perform decode forward pass
    """

    def _create_sampling_params(self, can_sample_on_device, non_greedy_decoding_on_device, batch_size):
        """
        non_greedy_decoding_on_device: when True, device supports non-greedy sampling (temperature,
        top_k, top_p, presence/frequency/repetition penalties, log_probs); warmup then includes
        those configs. When False, only greedy decoding is warmed up (temperature=0.0, top_k=1, top_p=1.0).
        """
        if not can_sample_on_device:
            return [None]

        sampling_configs = []

        if non_greedy_decoding_on_device:
            for penalties, log_probs in product([True, False], repeat=2):
                presence_penalty, frequency_penalty, repetition_penalty = None, None, None

                if penalties:
                    presence_penalty = [1.2] * batch_size
                    frequency_penalty = [1.2] * batch_size
                    repetition_penalty = [1.5] * batch_size

                enable_log_probs = [log_probs] * batch_size

                temperature = [1.0] * batch_size
                top_k = [10] * batch_size
                top_p = [0.9] * batch_size

                sampling_configs.append(
                    SamplingParams(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        repetition_penalty=repetition_penalty,
                        enable_log_probs=enable_log_probs,
                    )
                )

        sampling_configs.append(
            SamplingParams(
                temperature=[0.0] * batch_size,
                top_k=[1] * batch_size,
                top_p=[1.0] * batch_size,
            )
        )

        sampling_configs.append(None)

        return sampling_configs

    def _create_decode_warmup_inputs(self, max_batch_size, num_blocks):
        tokens = torch.zeros(max_batch_size, 1, dtype=torch.int32)
        start_pos = torch.zeros(max_batch_size, dtype=torch.int32)
        page_table = torch.zeros(max_batch_size, num_blocks, dtype=torch.int32)
        return tokens, start_pos, page_table

    def warmup_model_decode(
        self,
        kv_cache,
        enable_trace,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        non_greedy_decoding_on_device,
    ):
        """
        This function is called by vLLM
        """
        sampling_params = self._create_sampling_params(
            can_sample_on_device, non_greedy_decoding_on_device, max_batch_size
        )

        tokens, start_pos, page_table = self._create_decode_warmup_inputs(max_batch_size, num_blocks)

        logger.info("Starting decode warmup")
        logger.info(f"Tokens shape: {tokens.shape}")
        logger.info(f"Start pos shape: {start_pos.shape}")
        logger.info(f"Page table shape: {page_table.shape}")

        trace_values = [False, True] if enable_trace else [False]
        for trace_value in trace_values:
            for param in sampling_params:
                # summarize sampling params for simple logging
                if param is None:
                    param_summary = "None"
                else:
                    temp0 = (
                        param.temperature[0]
                        if isinstance(param.temperature, list) and len(param.temperature) > 0
                        else param.temperature
                    )
                    topk0 = param.top_k[0] if isinstance(param.top_k, list) and len(param.top_k) > 0 else param.top_k
                    topp0 = param.top_p[0] if isinstance(param.top_p, list) and len(param.top_p) > 0 else param.top_p
                    penalties_on = any(
                        x is not None
                        and (
                            (isinstance(x, list) and len(x) > 0 and x[0] not in (0.0, 1.0))
                            or (not isinstance(x, list) and x not in (0.0, 1.0, None))
                        )
                        for x in (param.presence_penalty, param.frequency_penalty, param.repetition_penalty)
                    )
                    log_probs_on = (
                        any(param.enable_log_probs)
                        if isinstance(param.enable_log_probs, list)
                        else bool(param.enable_log_probs)
                    )
                    param_summary = (
                        f"temperature={temp0}, top_k={topk0}, top_p={topp0}, "
                        f"penalties_on={penalties_on}, log_probs_on={log_probs_on}"
                    )

                warmup_mode = not trace_value
                logger.info(
                    f"Warming up decode for sampling params: {param_summary} with enable_trace={trace_value} and warmup_mode={warmup_mode}"
                )
                self.decode_forward(
                    tokens=tokens,
                    start_pos=start_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=trace_value,
                    read_from_device=True,
                    sampling_params=param,
                    warmup_mode=warmup_mode,
                )

        logger.info("Decode warmup completed")
