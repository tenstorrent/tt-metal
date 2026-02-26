# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import random
import secrets
from dataclasses import dataclass, fields, replace
from typing import List, Optional

import torch
from loguru import logger

import ttnn

from ._utils import clamp, is_default_value, split_list
from .tt_penalties import TTPenalties
from .tt_sampling import TTSampling


@dataclass(frozen=True)
class SamplingParams:
    """
    Sampling parameters for on-device greedy decoding / sampling.

    Used by Generator decode/prefill functions. vLLM has its own duck-type-compatible
    TTSamplingParams (in vllm/worker/tt_model_runner.py) that works with the same
    format_sampling_params / chunk_sampling_params functions.
    """

    temperature: float | list[float]
    top_k: int | list[int]
    top_p: float | list[float]
    presence_penalty: float | list[float] = 0.0
    frequency_penalty: float | list[float] = 0.0
    repetition_penalty: float | list[float] = 1.0
    seed: int | list[int] | None = None
    enable_log_probs: bool | list[bool] = False


SAMPLING_PARAM_FIELDS = tuple(f.name for f in fields(SamplingParams))


@dataclass(frozen=True)
class _TraceKey:
    penalties_on: bool
    log_probs_on: bool
    force_argmax: bool


class SamplingGenerator:
    """
    High-level sampling helper that owns both `TTSampling` and `TTPenalties`
    modules and optionally manages TTNN trace capture/execution for sampling.

    Typical usage:
        generator = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=tt_ccl)
        generator.reset_sampling_params(k=..., p=..., temp=...)
        tokens = generator.sample(logits, enable_trace=True)
    """

    _DEFAULT_PENALTIES = {
        "presence": 0.0,
        "frequency": 0.0,
        "repetition": 1.0,
    }

    def __init__(
        self,
        *,
        args,
        mesh_device,
        tt_ccl,
        enable_internal_trace: bool = True,
        cq_id: int = 0,
    ):
        self.mesh_device = mesh_device
        self.cq_id = cq_id
        self.args = args
        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self.enable_internal_trace = enable_internal_trace

        self.tt_sampling = TTSampling(mesh_device=mesh_device, tt_ccl=tt_ccl, args=args)
        self.tt_penalties = TTPenalties(mesh_device=mesh_device, args=args)

        self._penalties_active = False

        self._trace_states: dict[_TraceKey, dict] = {}
        seed_batch_size = self.tt_sampling.max_batch_size * self.tt_sampling._sampling_dp
        self.seed_manager = SeedManager(self.tt_sampling, max_batch_size=seed_batch_size)

    def _new_trace_state(self):
        return {"id": None, "input": None, "output": None, "kwargs": {}}

    def _trace_slot(self, penalties_on: bool, log_probs_on: bool, force_argmax: bool):
        key = _TraceKey(penalties_on=penalties_on, log_probs_on=log_probs_on, force_argmax=force_argmax)
        slot = self._trace_states.get(key)
        if slot is None:
            slot = self._new_trace_state()
            self._trace_states[key] = slot
        return key, slot

    def reset_trace(self):
        """
        Drop any cached trace metadata for both penalties/no-penalties and log-probs/no-log-probs paths.
        """
        for key, slot in self._trace_states.items():
            if slot["id"] is not None:
                logger.debug(
                    f"Resetting sampling trace (penalties={key.penalties_on}, log_probs={key.log_probs_on}, force_argmax={key.force_argmax}, trace_id={slot['id']})"
                )
        self._trace_states.clear()

    def reset_prompt_tokens(self, prompt_tokens):
        if not self._penalties_active:
            return
        self.tt_penalties.reset_prompt_tokens(prompt_tokens)

    def reset_output_state(self, tokens=None):
        if not self._penalties_active:
            return
        self.tt_penalties.reset_output_tokens(tokens)

    # ---------------------------------------------------------------------
    # Prefill / decode state helpers
    # ---------------------------------------------------------------------
    def apply_prefill_state(
        self,
        *,
        sampling_params,
        prompt_tokens: torch.Tensor | None,
        empty_slots: list[int],
    ):
        """Prepare sampling state for a prefill request.

        Resets params, seeds, prompt tokens, and output state in the correct order.
        """
        self.reset_sampling_params(sampling_params)
        if getattr(sampling_params, "seed", None) is not None:
            self.seed_manager.reset_seed(sampling_params.seed, empty_slots)
        self.seed_manager.get_new_values(empty_slots, replicate_seeds=True)
        if prompt_tokens is not None:
            self.reset_prompt_tokens(prompt_tokens)
        self.reset_output_state()

    def apply_decode_state(
        self,
        sampling_params_chunks: list,
        *,
        reset_batch: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
    ):
        """Format, merge (if row-sharded), and apply sampling params for one model instance.

        Args:
            sampling_params_chunks: List of SamplingParams assigned to this instance.
                Length-1 for simple cases; >1 for row-sharded (sampling_dp > data_parallel).
            reset_batch: Also reset prompt tokens and output state (first decode step).
            prompt_tokens: Prompt tokens for penalty tracking.
            output_tokens: Output tokens for penalty tracking.

        Does NOT call ``seed_manager.get_new_values()`` — callers manage seed
        advancement separately since generators call it at different points.
        """
        chunks_per_model = len(sampling_params_chunks)

        max_batch_size = self.tt_sampling.max_batch_size

        if chunks_per_model == 1:
            formatted_params = format_sampling_params(sampling_params_chunks[0], max_batch_size)
            self.reset_sampling_params(formatted_params)
        else:
            # Row-sharded case: format each chunk to max_batch_size, concatenate.
            # After (0, None) sharding each row gets its own chunk of max_batch_size entries.
            # Both TTSampling and TTPenalties use the same concatenated params.
            formatted_chunks = [format_sampling_params(chunk, max_batch_size) for chunk in sampling_params_chunks]
            concat_fields = {}
            for field in SAMPLING_PARAM_FIELDS:
                lists = [getattr(fc, field) for fc in formatted_chunks]
                if all(v is None for v in lists):
                    concat_fields[field] = None
                else:
                    concat_fields[field] = sum((v if isinstance(v, list) else [v] for v in lists), [])
            formatted_params = SamplingParams(**concat_fields)
            self.reset_sampling_params(formatted_params)

        if reset_batch:
            self.reset_prompt_tokens(prompt_tokens)
            self.reset_output_state(output_tokens)

    # ---------------------------------------------------------------------
    # Sampling helpers
    # ---------------------------------------------------------------------
    def reset_sampling_params(self, sampling_params):
        old_force_argmax_sampling = self.tt_sampling.force_argmax_sampling
        self.tt_sampling.reset_params(
            k=sampling_params.top_k,
            p=sampling_params.top_p,
            temp=sampling_params.temperature,
            enable_log_probs=sampling_params.enable_log_probs,
        )
        if self.tt_sampling.force_argmax_sampling != old_force_argmax_sampling:
            self.reset_trace()

        old_penalties_active = self._penalties_active
        self._penalties_active = not (
            is_default_value(sampling_params.presence_penalty, self._DEFAULT_PENALTIES["presence"])
            and is_default_value(sampling_params.frequency_penalty, self._DEFAULT_PENALTIES["frequency"])
            and is_default_value(sampling_params.repetition_penalty, self._DEFAULT_PENALTIES["repetition"])
        )
        if (
            not self.tt_sampling.force_argmax_sampling
            or self._penalties_active
            or self._penalties_active != old_penalties_active
        ):
            self.tt_penalties.reset_params(
                sampling_params.presence_penalty, sampling_params.frequency_penalty, sampling_params.repetition_penalty
            )
        self._log_probs_active = self.tt_sampling.log_probs_calculator.enable_log_probs

    def _validate_trace_inputs(self, slot, logits: ttnn.Tensor, tt_out_tok: Optional[ttnn.Tensor]):
        if slot["input"] is None or slot["output"] is None:
            raise RuntimeError("Trace metadata missing. Call capture_trace first.")

        if logits is not slot["input"]:
            raise ValueError(
                "The provided logits tensor does not match the tensor used during trace capture. "
                "Call `reset_trace()` before tracing with new tensors."
            )
        if isinstance(slot["output"], tuple):
            if tt_out_tok is not None and tt_out_tok is not slot["output"][0]:
                raise ValueError(
                    "The provided output tensor does not match the tensor used during trace capture. "
                    "Call `reset_trace()` before tracing with new tensors."
                )
        else:
            if tt_out_tok is not None and tt_out_tok is not slot["output"]:
                raise ValueError(
                    "The provided output tensor does not match the tensor used during trace capture. "
                    "Call `reset_trace()` before tracing with new tensors."
                )

    def _run_sampling(
        self,
        logits,
        *,
        penalties_on: bool,
        tt_out_tok: Optional[ttnn.Tensor],
    ):
        if penalties_on:
            logits = self.tt_penalties.apply(logits)
        tt_tokens, tt_log_probs = self.tt_sampling(logits, tt_out_tok=tt_out_tok)
        return tt_tokens, tt_log_probs

    def capture_trace(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Capture a trace of the sampling pipeline for the given configuration.
        """
        penalties_on = self._penalties_active
        log_probs_on = getattr(self, "_log_probs_active", False)
        force_argmax = self.tt_sampling.force_argmax_sampling

        key, slot = self._trace_slot(penalties_on, log_probs_on, force_argmax)

        logger.debug(
            f"Pre-compiling sampling path before trace capture (penalties={penalties_on},log_probs_on={log_probs_on},force_argmax={force_argmax})"
        )
        self._run_sampling(
            logits,
            penalties_on=penalties_on,
            tt_out_tok=tt_out_tok,
        )

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)
        sampled = self._run_sampling(
            logits,
            penalties_on=penalties_on,
            tt_out_tok=tt_out_tok,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=self.cq_id)
        ttnn.synchronize_device(self.mesh_device)

        if tt_out_tok is not None:
            if isinstance(sampled, tuple):
                output = (tt_out_tok, sampled[-1])
            else:
                output = (tt_out_tok, sampled)
        else:
            output = sampled

        slot["id"] = trace_id
        slot["input"] = logits
        slot["output"] = output
        slot["kwargs"] = {"tt_out_tok": tt_out_tok}

        return slot["output"]

    def _execute_trace(self, key: _TraceKey) -> ttnn.Tensor:
        slot = self._trace_states.get(key)
        if slot is None:
            raise RuntimeError("Trace has not been captured yet.")
        if slot["id"] is None or slot["output"] is None:
            raise RuntimeError("Trace has not been captured yet.")

        ttnn.execute_trace(self.mesh_device, slot["id"], cq_id=self.cq_id, blocking=False)
        return slot["output"]

    def sample(
        self,
        logits: ttnn.Tensor,
        *,
        enable_trace: bool = True,
        tt_out_tok: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Convenience wrapper that either runs the sampling module directly or
        replays a captured trace.
        """

        penalties_on = self._penalties_active
        log_probs_on = getattr(self, "_log_probs_active", False)
        force_argmax = self.tt_sampling.force_argmax_sampling
        use_internal_trace = enable_trace and self.enable_internal_trace

        if not use_internal_trace:
            tt_out = self._run_sampling(
                logits,
                penalties_on=penalties_on,
                tt_out_tok=tt_out_tok,
            )
        else:
            key, slot = self._trace_slot(penalties_on, log_probs_on, force_argmax)
            if slot["id"] is None:
                return self.capture_trace(
                    logits,
                    tt_out_tok=tt_out_tok,
                )

            self._validate_trace_inputs(slot, logits, tt_out_tok)
            tt_out = self._execute_trace(key)

        if penalties_on and tt_out is not None:
            if isinstance(tt_out, tuple):
                self.tt_penalties.update_output_tokens(tt_out[0])
            else:
                self.tt_penalties.update_output_tokens(tt_out)
        return tt_out


def format_sampling_params(sampling_params, max_batch_size):
    """
    Format sampling parameters for on-device use.

    Converts scalar fields to lists, pads all lists to ``max_batch_size``,
    inverts temperature, clamps top-p/top-k, and normalises penalties.

    Returns a **new** SamplingParams — the input is never mutated.
    """
    if not isinstance(sampling_params.temperature, List):
        update_dict = {field.name: [getattr(sampling_params, field.name)] for field in fields(sampling_params)}
        sampling_params = replace(sampling_params, **update_dict)

    target_len = max_batch_size
    assert target_len % 32 == 0, f"Sampling batch size must be a multiple of 32, got {target_len}"

    # Defaults used when padding short lists to target_len
    defaults = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": random.randint(0, 1000000),
    }

    def _pad(lst, name):
        """Return a new list padded to target_len with the default for *name*."""
        if len(lst) >= target_len:
            return list(lst)
        return list(lst) + [defaults[name]] * (target_len - len(lst))

    # Pad core sampling fields
    temperature = _pad(sampling_params.temperature, "temperature")
    top_p = _pad(sampling_params.top_p, "top_p")
    top_k = _pad(sampling_params.top_k, "top_k")

    # Normalise and pad penalty / seed fields
    def _normalise_and_pad(name):
        value = getattr(sampling_params, name, None)
        if value is None:
            lst = [defaults[name]]
        elif isinstance(value, List):
            lst = list(value)
        else:
            lst = [value]
        return _pad(lst, name)

    presence_penalty = _normalise_and_pad("presence_penalty")
    frequency_penalty = _normalise_and_pad("frequency_penalty")
    repetition_penalty = _normalise_and_pad("repetition_penalty")
    seed = _normalise_and_pad("seed")

    # Clamp / transform values in the new lists (no mutation of the input)
    TOP_P_MIN = 0.0
    TOP_P_MAX = 1.0

    for i in range(len(temperature)):
        top_p[i] = clamp(top_p[i], TOP_P_MIN, TOP_P_MAX)

        if temperature[i] == 0:
            temperature[i] = 1.0
            top_k[i] = 1
        else:
            temperature[i] = 1 / temperature[i]

        # top_k contract: TT sampling supports up to 32 today.
        # k < 1 means "no restriction" → max (32); k > 32 → capped to 32.
        if top_k[i] < 1:
            top_k[i] = 32
        if top_k[i] > 32:
            top_k[i] = 32

        if repetition_penalty[i] == 0:
            repetition_penalty[i] = defaults["repetition_penalty"]

    return replace(
        sampling_params,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )


def broadcast_sampling_params(
    formatted_sampling_params,
    idx: int,
    slot_len: int = 32,
):
    """
    Create a new SamplingParams where each list field is broadcast to a full list of length
    ``slot_len``, taking the value from ``idx``. Does not mutate the input.
    """
    kwargs = {}
    for f in fields(formatted_sampling_params):
        value = getattr(formatted_sampling_params, f.name)
        if isinstance(value, List):
            chosen = value[idx] if idx < len(value) else value[0]
        else:
            chosen = value
        if chosen is None:
            kwargs[f.name] = None
        else:
            kwargs[f.name] = [chosen] * slot_len
    return SamplingParams(**kwargs)


def chunk_sampling_params(sampling_params, sampling_dp: int) -> list:
    """
    Chunk a SamplingParams (or duck-type-compatible object) into ``sampling_dp`` pieces.

    List fields are split evenly (length must be divisible by ``sampling_dp``).
    Scalar fields are replicated to all chunks.  Falls back to dataclass defaults
    for missing attributes so that vLLM's TTSamplingParams works transparently.

    Returns a list of SamplingParams.
    """
    if sampling_dp == 1:
        return [sampling_params]

    chunked_fields = {}
    for field_name in SAMPLING_PARAM_FIELDS:
        try:
            val = getattr(sampling_params, field_name)
        except AttributeError:
            if hasattr(SamplingParams, field_name):
                val = getattr(SamplingParams, field_name)
            else:
                raise
        if isinstance(val, list):
            assert (
                len(val) % sampling_dp == 0
            ), f"Sampling param '{field_name}' length {len(val)} not divisible by sampling_dp {sampling_dp}"
            chunked_fields[field_name] = split_list(val, sampling_dp)
        else:
            chunked_fields[field_name] = [val] * sampling_dp

    return [
        SamplingParams(**{field: chunked_fields[field][i] for field in SAMPLING_PARAM_FIELDS})
        for i in range(sampling_dp)
    ]


class SeedManager:
    def __init__(self, tt_sampling, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.seeds = [secrets.randbits(64) for _ in range(max_batch_size)]
        self.rngs = [random.Random(seed) for seed in self.seeds]
        self.tt_sampling = tt_sampling
        # Mesh mapper for sharding seeds across rows when sampling_dp > 1
        if tt_sampling._sampling_dp > 1:
            self._seed_mapper = ttnn.ShardTensor2dMesh(
                tt_sampling.mesh_device, dims=tt_sampling._param_dims, mesh_shape=tt_sampling.cluster_shape
            )
        else:
            self._seed_mapper = None

    def reset_seed(self, seeds, user_ids):
        for i, user in enumerate(user_ids):
            self.rngs[user].seed(seeds[i])
            self.seeds[user] = seeds[i]

    def get_new_values(self, empty_slots=None, replicate_seeds=False):
        if empty_slots is None:
            empty_slots = range(self.max_batch_size)
        # get new seeds for each user in empty_slots otherwise 0
        new_seeds = [rng.randint(0, 1000000) if i in empty_slots else 0 for i, rng in enumerate(self.rngs)]

        if replicate_seeds:
            assert len(empty_slots) == 1, "Cannot replicate seeds if empty_slots is not length 1"
            new_seeds = self.max_batch_size * [new_seeds[empty_slots[0]]]
        # send new seeds to sampling module
        new_seed_tt = ttnn.from_torch(
            torch.tensor(new_seeds), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._seed_mapper
        )
        ttnn.copy_host_to_device_tensor(new_seed_tt, self.tt_sampling.seeds_tt_tensor)
