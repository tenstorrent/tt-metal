# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
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

    def _create_sampling_params(
        self,
        can_sample_on_device,
        batch_size,
        greedy_only: bool = False,
        include_greedy_penalties: bool = False,
    ):
        """
        greedy_only: when True, warmup only covers greedy decoding on device (temperature=0.0,
        top_k=1, top_p=1.0). When False (the default), warmup also exercises non-greedy variants
        — temperature/top_k/top_p, presence/frequency/repetition penalties, and log_probs.
        """
        if not can_sample_on_device:
            return [None]

        sampling_configs = []

        if not greedy_only:
            # Full warmup pre-captures every penalties × log_probs permutation so
            # no on-device-sampling request ever pays a one-time trace-capture
            # cost on first use. Each permutation is a *separate resident trace*,
            # though, and for large MoE models the combined trace region can run
            # into the gigabytes (e.g. Gemma4-26B-A4B). ``TT_LEAN_DECODE_WARMUP``
            # restricts the sweep to the plain (no-penalty, no-logprob) sampling
            # config — what a throughput benchmark with default sampling actually
            # exercises — trading a one-time runtime capture for the rarer
            # penalty/logprob request shapes in exchange for a much smaller trace
            # region. Device grammar still needs every no-logprobs penalty family
            # because lazily capturing one behind a live model trace is unsafe.
            # Greedy and ``None`` are still captured below.
            if os.environ.get("TT_LEAN_DECODE_WARMUP"):
                penalty_logprob_combos = [(False, False)]
                if include_greedy_penalties:
                    penalty_logprob_combos.append((True, False))
            else:
                penalty_logprob_combos = list(product([True, False], repeat=2))

            for penalties, log_probs in penalty_logprob_combos:
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

        if include_greedy_penalties:
            sampling_configs.append(
                SamplingParams(
                    temperature=[0.0] * batch_size,
                    top_k=[1] * batch_size,
                    top_p=[1.0] * batch_size,
                    presence_penalty=[1.2] * batch_size,
                    frequency_penalty=[1.2] * batch_size,
                    repetition_penalty=[1.5] * batch_size,
                    enable_log_probs=[False] * batch_size,
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
        can_sample_device_grammar: bool = False,
        read_from_device=True,
        greedy_only: bool = False,
        skip_trace_precompile: bool = False,
        sampling_trace_variants_prepared: bool = False,
    ):
        """Compile or capture every decode and sampling variant used by vLLM.

        Device grammar traces require the normal two-phase invocation: first
        ``enable_trace=False`` compiles grammar-on sampling, then
        ``enable_trace=True`` records it without an allocation-producing
        precompile pass behind live traces.
        """
        sampling_params = self._create_sampling_params(
            can_sample_on_device,
            max_batch_size,
            greedy_only=greedy_only,
            include_greedy_penalties=can_sample_device_grammar,
        )

        tokens, start_pos, page_table = self._create_decode_warmup_inputs(max_batch_size, num_blocks)

        logger.info("Starting decode warmup")
        logger.info(f"Tokens shape: {tokens.shape}")
        logger.info(f"Start pos shape: {start_pos.shape}")
        logger.info(f"Page table shape: {page_table.shape}")

        trace_variants_prepared = sampling_trace_variants_prepared
        for index, param in enumerate(sampling_params):
            logger.info(f"Warming up decode for sampling params: {param}")
            decode_kwargs = dict(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                read_from_device=read_from_device,
                sampling_params=param,
            )
            if enable_trace and param is not None and can_sample_device_grammar and not trace_variants_prepared:
                # The common generator uses the first traced call to allocate
                # the persistent decode feedback buffer and precompile every
                # sampler variant against that exact buffer before recording
                # any trace. End with this call's params so its trace key
                # matches the state captured immediately afterwards.
                decode_kwargs["sampling_trace_warmup_params"] = (
                    sampling_params[index + 1 :] + sampling_params[: index + 1]
                )
                decode_kwargs["sampling_trace_warmup_device_grammar"] = can_sample_device_grammar
                trace_variants_prepared = True
            elif (enable_trace and trace_variants_prepared) or skip_trace_precompile:
                decode_kwargs["skip_trace_precompile"] = True
            self.decode_forward(**decode_kwargs)
            enable_log_probs = getattr(param, "enable_log_probs", False) if param is not None else False
            has_logprobs = (
                bool(enable_log_probs.any())
                if isinstance(enable_log_probs, torch.Tensor)
                else (any(enable_log_probs) if isinstance(enable_log_probs, (list, tuple)) else bool(enable_log_probs))
            )
            if param is not None and can_sample_device_grammar and not has_logprobs:
                grammar_kwargs = dict(
                    decode_kwargs,
                    grammar_bitmask=self._create_warmup_grammar_bitmask(max_batch_size),
                )
                grammar_kwargs.pop("sampling_trace_warmup_params", None)
                grammar_kwargs.pop("sampling_trace_warmup_device_grammar", None)
                if enable_trace:
                    # The non-traced warmup phase already compiled the grammar
                    # path. Avoid allocating a precompile pass behind live
                    # model/sampling traces before recording grammar-on.
                    grammar_kwargs["skip_trace_precompile"] = True
                self.decode_forward(**grammar_kwargs)

        logger.info("Decode warmup completed")
