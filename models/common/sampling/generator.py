# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import random
import secrets
from dataclasses import dataclass, fields, replace
from typing import List, Optional

import torch
from loguru import logger

import ttnn

from .tt_penalties import TTPenalties
from .tt_sampling import TTSampling


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
        self.seed_manager = SeedManager(self.tt_sampling)

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

    def _is_default_penalty(self, values, default):
        if values is None:
            return True
        if isinstance(values, (int, float)):
            return values == default
        return all(value == default for value in values)

    def reset_prompt_tokens(self, prompt_tokens):
        if not self._penalties_active:
            return
        self.tt_penalties.reset_prompt_tokens(prompt_tokens)

    def reset_output_state(self, tokens=None):
        if not self._penalties_active:
            return
        self.tt_penalties.reset_output_tokens(tokens)

    # ---------------------------------------------------------------------
    # Sampling helpers
    # ---------------------------------------------------------------------
    def reset_sampling_params(self, sampling_params):
        old_force_argmax_sampling = self.tt_sampling._force_argmax_sampling
        self.tt_sampling.reset_params(
            k=sampling_params.top_k,
            p=sampling_params.top_p,
            temp=sampling_params.temperature,
            enable_log_probs=sampling_params.enable_log_probs,
        )
        if self.tt_sampling._force_argmax_sampling != old_force_argmax_sampling:
            self.reset_trace()

        old_penalties_active = self._penalties_active
        self._penalties_active = not (
            self._is_default_penalty(sampling_params.presence_penalty, self._DEFAULT_PENALTIES["presence"])
            and self._is_default_penalty(sampling_params.frequency_penalty, self._DEFAULT_PENALTIES["frequency"])
            and self._is_default_penalty(sampling_params.repetition_penalty, self._DEFAULT_PENALTIES["repetition"])
        )
        if (
            not self.tt_sampling._force_argmax_sampling
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
        force_argmax = self.tt_sampling._force_argmax_sampling

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
        force_argmax = self.tt_sampling._force_argmax_sampling
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


def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    return value


def format_sampling_params(sampling_params, max_batch_size):
    """
    Format sampling parameters to a dictionary.
    """
    if not isinstance(sampling_params.temperature, List):
        # convert all sampling_params to lists
        update_dict = {field.name: [getattr(sampling_params, field.name)] for field in fields(sampling_params)}
        sampling_params = replace(sampling_params, **update_dict)

    # Must pad sampling_params to max_batch_size
    default_params = {
        "temp": 0.0,
        "p": 1.0,
        "k": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": random.randint(0, 1000000),  # set to random seed to have variability while using tensor manual_seed
    }
    target_len = max_batch_size
    assert target_len == 32, "Sampling only support batch_size=32"
    for name, tensor in zip(
        ("temp", "p", "k"), (sampling_params.temperature, sampling_params.top_p, sampling_params.top_k)
    ):
        current_len = len(tensor)
        if current_len < target_len:
            tensor.extend([default_params[name]] * (target_len - current_len))

    params = {}
    for name in ("presence_penalty", "frequency_penalty", "repetition_penalty", "seed"):
        value = getattr(sampling_params, name, None)
        if value is None:
            params[name] = [default_params[name]]
        elif isinstance(value, List):
            params[name] = list(value)
        else:
            params[name] = [value]

    sampling_params = replace(
        sampling_params,
        presence_penalty=params["presence_penalty"],
        frequency_penalty=params["frequency_penalty"],
        repetition_penalty=params["repetition_penalty"],
        seed=params["seed"],
    )

    for name in ("presence_penalty", "frequency_penalty", "repetition_penalty", "seed"):
        tensor = getattr(sampling_params, name)
        current_len = len(tensor)
        if current_len < target_len:
            tensor.extend([default_params[name]] * (target_len - current_len))

    # We must clamp top-p in range [0.0, 1.0]
    # Cannot rely on external SamplingParams to be clamped
    TOP_P_MIN = 0.0
    TOP_P_MAX = 1.0

    for i, (top_p, temp) in enumerate(zip(sampling_params.top_p, sampling_params.temperature)):
        # Clamp top-p
        clamped_top_p = clamp(top_p, TOP_P_MIN, TOP_P_MAX)
        if clamped_top_p != top_p:
            sampling_params.top_p[i] = clamped_top_p

        # Process temperature
        if temp == 0:
            sampling_params.temperature[i] = 1.0
            sampling_params.top_k[i] = 1
        else:
            sampling_params.temperature[i] = 1 / temp

        if sampling_params.top_k[i] < 1:
            sampling_params.top_k[i] = 32  # k<1 means no restriction so set it to max k (32)

        if sampling_params.repetition_penalty[i] == 0:
            sampling_params.repetition_penalty[i] = default_params["repetition_penalty"]

        if sampling_params.top_k[i] < 1:
            sampling_params.top_k[i] = 32  # k<1 means no restriction so set it to max k (32)
    return sampling_params


class SeedManager:
    def __init__(self, tt_sampling):
        self.seeds = [secrets.randbits(64) for _ in range(32)]
        self.rngs = [random.Random(seed) for seed in self.seeds]
        self.tt_sampling = tt_sampling

    def reset_seed(self, seeds, user_ids):
        for i, user in enumerate(user_ids):
            self.rngs[user].seed(seeds[i])
            self.seeds[user] = seeds[i]

    def get_new_values(self, empty_slots=range(32), replicate_seeds=False):
        # get new seeds for each user in empty_slots otherwise 0
        new_seeds = [rng.randint(0, 1000000) if i in empty_slots else 0 for i, rng in enumerate(self.rngs)]

        if replicate_seeds:
            assert len(empty_slots) == 1, "Cannot replicate seeds if empty_slots is not length 1"
            new_seeds = 32 * [new_seeds[empty_slots[0]]]
        # send new seeds to sampling module
        new_seed_tt = ttnn.from_torch(torch.tensor(new_seeds), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(new_seed_tt, self.tt_sampling.seeds_tt_tensor)
