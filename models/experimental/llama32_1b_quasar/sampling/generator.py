# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import random
import secrets
from dataclasses import dataclass, fields, replace
from typing import List, Optional

import torch
from loguru import logger

import ttnn

from ._utils import clamp
from ._utils import compact_debug_list as _compact_debug_list
from ._utils import is_default_value, is_llama33_70b_model
from ._utils import log_sampling_debug as _log_sampling_debug
from ._utils import split_list
from .tt_penalties import TTPenalties
from .tt_sampling import TTSampling

MAX_UINT32 = 2**32 - 1
# MAX_UINT32 is reserved as the device skip sentinel; keep real seeds in a bounded positive range.
DEVICE_SEED_MAX = 1_000_000
_UINT64_MASK = (1 << 64) - 1


def _hash_request_seed_to_device_seed(seed: int, counter: int) -> int:
    """Derive a stable per-token device seed from a request seed.

    The device sampling op accepts bounded positive seeds, while vLLM
    request seeds can be any integer and must be reproducible regardless
    of batch slot. Hashing (request seed, token counter) gives each token
    a deterministic but well-mixed device seed without relying on mutable
    per-slot RNG state. The constants below are the SplitMix64 finalizer.
    """
    value = (int(seed) & _UINT64_MASK) ^ ((int(counter) + 0x9E3779B97F4A7C15) & _UINT64_MASK)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value = (value ^ (value >> 31)) & _UINT64_MASK
    return (value % DEVICE_SEED_MAX) + 1


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
    num_logprobs: int | list[int] = 0


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
        cq_id: int = 0,
    ):
        self.mesh_device = mesh_device
        self.cq_id = cq_id
        self.args = args
        self._sampling_debug_enabled = is_llama33_70b_model(args)
        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self.tt_sampling = TTSampling(mesh_device=mesh_device, tt_ccl=tt_ccl, args=args)
        self.tt_penalties = TTPenalties(mesh_device=mesh_device, args=args)

        self._penalties_active = False

        self._trace_states: dict[_TraceKey, dict] = {}
        seed_batch_size = self.tt_sampling.max_batch_size * self.tt_sampling._sampling_dp
        self.seed_manager = SeedManager(
            self.tt_sampling,
            max_batch_size=seed_batch_size,
        )

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
            if slot["id"] is None:
                continue
            logger.debug(
                f"Resetting sampling trace (penalties={key.penalties_on}, log_probs={key.log_probs_on}, force_argmax={key.force_argmax}, trace_id={slot['id']})"
            )
            try:
                ttnn.release_trace(self.mesh_device, slot["id"])
            except Exception as e:
                logger.warning(f"Failed to release trace {slot['id']} : {e}")
                continue
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
        replicate_seeds: bool = True,
    ):
        """Prepare sampling state for a prefill request.

        Resets params, seeds, prompt tokens, and output state in the correct order.
        """
        self.reset_sampling_params(sampling_params, empty_slots=empty_slots)
        seed = getattr(sampling_params, "seed", None)
        # assert on condition that seed is not None
        assert seed is not None, "sampling_params must be formatted (seed should be a list, not None)"
        self.seed_manager.reset_seed(seed, empty_slots)
        self.seed_manager.get_new_values(empty_slots, replicate_seeds=replicate_seeds)
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
    def reset_sampling_params(self, sampling_params, empty_slots: list[int] | None = None):
        old_force_argmax_sampling = self.tt_sampling.force_argmax_sampling
        num_logprobs = getattr(sampling_params, "num_logprobs", None)
        self.tt_sampling.reset_params(
            k=sampling_params.top_k,
            p=sampling_params.top_p,
            temp=sampling_params.temperature,
            enable_log_probs=sampling_params.enable_log_probs,
            num_logprobs=num_logprobs,
            empty_slots=empty_slots,
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
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SamplingGenerator reset params",
            empty_slots=_compact_debug_list(empty_slots),
            force_argmax=self.tt_sampling.force_argmax_sampling,
            force_argmax_changed=self.tt_sampling.force_argmax_sampling != old_force_argmax_sampling,
            penalties_active=self._penalties_active,
            log_probs_active=self._log_probs_active,
            temperature=_compact_debug_list(sampling_params.temperature),
            top_k=_compact_debug_list(sampling_params.top_k),
            top_p=_compact_debug_list(sampling_params.top_p),
            presence_penalty=_compact_debug_list(sampling_params.presence_penalty),
            frequency_penalty=_compact_debug_list(sampling_params.frequency_penalty),
            repetition_penalty=_compact_debug_list(sampling_params.repetition_penalty),
            seed=_compact_debug_list(getattr(sampling_params, "seed", None)),
        )

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
        skip_precompile: bool = False,
    ) -> ttnn.Tensor:
        """
        Capture a trace of the sampling pipeline for the given configuration.
        """
        penalties_on = self._penalties_active
        log_probs_on = getattr(self, "_log_probs_active", False)
        force_argmax = self.tt_sampling.force_argmax_sampling

        key, slot = self._trace_slot(penalties_on, log_probs_on, force_argmax)

        if not skip_precompile:
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
        skip_precompile: bool = False,
    ) -> ttnn.Tensor:
        """
        Convenience wrapper that either runs the sampling module directly or
        replays a captured trace.
        """

        penalties_on = self._penalties_active
        log_probs_on = getattr(self, "_log_probs_active", False)
        force_argmax = self.tt_sampling.force_argmax_sampling
        # Explicit request seeds update a persistent seed tensor every token;
        # run them directly so trace replay cannot observe stale seed state.
        use_internal_trace = enable_trace and not self.seed_manager.has_active_request_seed()
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SamplingGenerator sample",
            enable_trace=enable_trace,
            use_internal_trace=use_internal_trace,
            penalties_on=penalties_on,
            log_probs_on=log_probs_on,
            force_argmax=force_argmax,
            logits_shape=list(logits.shape),
            tt_out_tok_shape=list(tt_out_tok.shape) if tt_out_tok is not None else None,
        )

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
                    skip_precompile=skip_precompile,
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
        "seed": None,
        "num_logprobs": 0,
        "enable_log_probs": False,
    }

    def _pad(lst, name):
        """Return a new list padded to target_len with the default for *name*."""
        if len(lst) >= target_len:
            return list(lst)
        return list(lst) + [defaults[name]] * (target_len - len(lst))

    # Pad core sampling fields (scalar→list already done above)
    temperature = _pad(sampling_params.temperature, "temperature")
    top_p = _pad(sampling_params.top_p, "top_p")
    top_k = _pad(sampling_params.top_k, "top_k")

    # enable_log_probs / num_logprobs: scalar → broadcast to all users.
    # Multi-element list → pad with default (False/0) for inactive slots.
    # Single-element list (from scalar→list conversion) → broadcast to all.
    def _broadcast_pad(lst, name):
        if not isinstance(lst, list):
            return [lst] * target_len
        if len(lst) == 1:
            return lst * target_len
        return _pad(lst, name)

    enable_log_probs = _broadcast_pad(sampling_params.enable_log_probs, "enable_log_probs")
    if getattr(sampling_params, "num_logprobs", None) is not None:
        num_logprobs = _broadcast_pad(sampling_params.num_logprobs, "num_logprobs")
    else:
        num_logprobs = None

    # Normalise and pad penalty / seed fields (may still be None/scalar)
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
            # Device sampling treats p=0 as a first-token cutoff; with k=1
            # this is the compact argmax representation for greedy rows.
            top_p[i] = 0.0
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

    kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    # Only include logprobs fields if the input dataclass has them
    # (vLLM's TTSamplingParams may not have these fields)
    input_fields = {f.name for f in fields(sampling_params)}
    if "num_logprobs" in input_fields:
        kwargs["num_logprobs"] = num_logprobs
    if "enable_log_probs" in input_fields:
        kwargs["enable_log_probs"] = enable_log_probs

    return replace(sampling_params, **kwargs)


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
        value_is_list = isinstance(value, List)
        if value_is_list:
            chosen = value[idx] if idx < len(value) else value[0]
        else:
            chosen = value
        if value_is_list:
            # Preserve list fields as lists even when the selected value is None.
            kwargs[f.name] = [chosen] * slot_len
        elif chosen is None:
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
    """Manages per-user RNG seeds for on-device sampling.

    Tracks which users have explicit seeds set (``_seed_active``) and avoids
    unnecessary host-to-device copies during decode when no seeds are active.
    """

    def __init__(self, tt_sampling, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.seeds = [None for _ in range(max_batch_size)]
        self.seed_counters = [0 for _ in range(max_batch_size)]
        # Pre-allocate RNG objects; actual request seeds are set via reset_seed().
        self.rngs = [random.Random(secrets.randbits(64)) for _ in range(max_batch_size)]
        self.tt_sampling = tt_sampling
        self._sampling_debug_enabled = getattr(tt_sampling, "_sampling_debug_enabled", False)
        # True when at least one user slot has a non-None request seed.
        self._seed_active = False
        # Set to True by reset_seed() so the next get_new_values() pushes
        # fresh values to the device. When _seed_active is True this pushes
        # per-user seeds; when False it pushes varied per-user
        # values to diversify the device RNG state. Cleared after the push.
        self._reseted = False
        # When True, the next get_new_values() must push MAX_UINT32 (SKIP) so
        # the device transitions from rand_tile_init to rand_tile advance.
        self._needs_skip = False
        # True only for the most recent get_new_values() call when at least
        # one active slot used an explicit request seed.
        self._active_request_seed = False
        # Mesh mapper for sharding seeds across rows when sampling_dp > 1.
        if tt_sampling._sampling_dp > 1:
            self._seed_mapper = ttnn.ShardTensor2dMesh(
                tt_sampling.mesh_device, dims=tt_sampling._param_dims, mesh_shape=tt_sampling.cluster_shape
            )
        else:
            self._seed_mapper = None

    def _next_unseeded_rng_seed(self) -> int:
        return secrets.randbits(64)

    def _next_unseeded_device_seed(self) -> int:
        return secrets.randbelow(DEVICE_SEED_MAX) + 1

    def _next_device_seed_from_rng(self, rng: random.Random) -> int:
        return rng.randint(1, DEVICE_SEED_MAX)

    def _next_device_seed_for_slot(self, slot: int) -> int:
        request_seed = self.seeds[slot]
        if request_seed is None:
            return self._next_device_seed_from_rng(self.rngs[slot])
        device_seed = _hash_request_seed_to_device_seed(int(request_seed), self.seed_counters[slot])
        self.seed_counters[slot] += 1
        return device_seed

    def _seed_from_slot_params(self, seeds, slot: int):
        if seeds is None:
            return None
        if isinstance(seeds, torch.Tensor):
            flat = seeds.reshape(-1)
            if slot < 0 or slot >= flat.numel():
                return None
            seed = flat[slot]
        elif isinstance(seeds, list):
            if slot < 0 or slot >= len(seeds):
                return None
            seed = seeds[slot]
        else:
            seed = seeds

        if seed is None:
            return None
        if isinstance(seed, torch.Tensor):
            if seed.numel() == 0:
                return None
            seed = seed.reshape(-1)[0].item()
        return int(seed)

    def reset_seed_from_slots(self, seeds, user_ids):
        """Reset decode seed state from slot-indexed sampling params."""
        if user_ids is None:
            user_ids = range(self.max_batch_size)
        for user in user_ids:
            slot = int(user)
            seed = self._seed_from_slot_params(seeds, slot)
            self.seeds[slot] = seed
            self.seed_counters[slot] = 0
            if seed is None:
                self.rngs[slot].seed(self._next_unseeded_rng_seed())
            else:
                self.rngs[slot].seed(int(seed))
        self._seed_active = any(s is not None for s in self.seeds)
        self._reseted = True

    def reset_seed_from_slots_if_needed(self, seeds, user_ids) -> bool:
        """Reset only active slots whose slot-indexed seed changed."""
        if user_ids is None:
            user_ids = range(self.max_batch_size)
        reset_slots = []
        for user in user_ids:
            slot = int(user)
            if self._seed_from_slot_params(seeds, slot) != self.seeds[slot]:
                reset_slots.append(slot)
        if not reset_slots:
            return False
        self.reset_seed_from_slots(seeds, reset_slots)
        return True

    def align_seed_counters_to_positions(self, seeds, user_ids, positions, offset: int = 1):
        """Make explicit-seed decode independent of persistent slot lifetime.

        vLLM can temporarily remove running requests from the persistent batch
        while admitting another prefill batch, then re-add them in different
        slots. For explicit request seeds, deriving the per-token device seed
        from the absolute decode position keeps the stream reproducible even
        when the Python-side slot counter was reset or moved.
        """
        if positions is None:
            return
        if user_ids is None:
            user_ids = range(self.max_batch_size)

        if isinstance(positions, torch.Tensor):
            flat_positions = positions.reshape(-1)

            def _position(slot):
                if slot < 0 or slot >= flat_positions.numel():
                    return None
                pos = flat_positions[slot]
                return int(pos.item())

        elif isinstance(positions, list):

            def _position(slot):
                if slot < 0 or slot >= len(positions):
                    return None
                return int(positions[slot])

        else:

            def _position(_slot):
                return int(positions)

        for user in user_ids:
            slot = int(user)
            seed = self._seed_from_slot_params(seeds, slot)
            if seed is None:
                continue
            position = _position(slot)
            if position is None or position < 0:
                continue
            self.seed_counters[slot] = max(0, position + offset)

    def has_active_request_seed(self) -> bool:
        return self._active_request_seed

    def _debug_state(self, slots=None):
        if slots is None:
            slots = range(self.max_batch_size)
        state = []
        for slot in slots:
            slot = int(slot)
            if slot < 0 or slot >= self.max_batch_size:
                continue
            seed = self.seeds[slot]
            if seed is not None:
                state.append((slot, seed))
        return _compact_debug_list(state)

    def apply_slot_remap(self, remap):
        """Reindex RNG state after batch condense.

        ``remap`` is a 1-D int tensor of length ``max_batch_size`` where
        ``remap[i] = j`` means slot *i* now holds the request that was
        previously at slot *j*. Identity entries (``remap[i] == i``) are
        no-ops. Only non-identity entries trigger a move.
        """
        if not self._seed_active:
            return
        moves = [(int(remap[i]), i) for i in range(len(remap)) if int(remap[i]) != i]
        if not moves:
            _log_sampling_debug(
                self._sampling_debug_enabled, "SeedManager slot remap identity", seed_active=self._seed_active
            )
            return
        # Snapshot the state we're about to overwrite.
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SeedManager slot remap",
            moves=_compact_debug_list(moves),
            state_before=self._debug_state(),
        )
        old_seeds = list(self.seeds)
        old_counters = list(self.seed_counters)
        old_rngs = list(self.rngs)
        moved_sources = {old_slot for old_slot, _ in moves}
        moved_destinations = {new_slot for _, new_slot in moves}
        for old_slot, new_slot in moves:
            self.seeds[new_slot] = old_seeds[old_slot]
            self.seed_counters[new_slot] = old_counters[old_slot]
            # copy.copy preserves internal RNG state but creates an
            # independent object so the old slot reference does not alias
            # the new one.
            self.rngs[new_slot] = copy.copy(old_rngs[old_slot])
        for old_slot in moved_sources - moved_destinations:
            self.seeds[old_slot] = None
            self.seed_counters[old_slot] = 0
        self._seed_active = any(s is not None for s in self.seeds)
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SeedManager slot remap done",
            seed_active=self._seed_active,
            state_after=self._debug_state(),
        )

    def reset_seed(self, seeds, user_ids):
        """Update RNG state for the given user slots after a prefill.

        Args:
            seeds: Seed values in request order. Accepts a list, tensor, scalar,
                or None (treated as all unseeded).
            user_ids: Batch slot indices being prefilled.
        """
        user_ids = [int(user) for user in user_ids]
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SeedManager reset prefill",
            user_ids=_compact_debug_list(user_ids),
            requested_seeds=_compact_debug_list(seeds),
            state_before=self._debug_state(user_ids),
        )
        for i, user in enumerate(user_ids):
            slot = int(user)
            seed = self._seed_from_slot_params(seeds, i)
            self.seeds[slot] = seed
            self.seed_counters[slot] = 0
            if seed is None:
                self.rngs[slot].seed(self._next_unseeded_rng_seed())
            else:
                self.rngs[slot].seed(int(seed))
        self._seed_active = any(s is not None for s in self.seeds)
        self._reseted = True
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SeedManager reset prefill done",
            seed_active=self._seed_active,
            state_after=self._debug_state(user_ids),
        )

    def get_new_values(self, empty_slots=None, replicate_seeds=False):
        """Generate and push new seed values to the device.

        **Seeded path** (``_seed_active=True``):
        Advances each active slot seed state and copies the new values to
        the device every step. Explicit request seeds produce slot-independent
        device seeds derived from the request seed and the slot counter. Some
        decode callers align that counter to the absolute token position so
        vLLM batch-layout changes cannot reset a request's random stream.

        **Unseeded path** (``_seed_active=False``):
        Uses a three-state machine to ensure each user gets a unique device
        RNG state without redundant host-to-device copies during decode:

          State 1 - **init** (``_reseted=True``):
            Push varied per-user values from system entropy.

          State 2 - **transition** (``_needs_skip=True``):
            Push MAX_UINT32 (SKIP) so the device stops reinitializing and
            starts advancing via rand_tile().

          State 3 - **steady** (both flags clear):
            Early-return with no device copy.
        """
        if empty_slots is None:
            empty_slots = list(range(self.max_batch_size))
        else:
            empty_slots = [int(slot) for slot in empty_slots]
        empty_slot_set = set(empty_slots)
        self._active_request_seed = any(self.seeds[i] is not None for i in empty_slot_set)

        if not self._seed_active:
            self._active_request_seed = False
            if self._reseted:
                new_seeds = [self._next_unseeded_device_seed() for _ in range(self.max_batch_size)]
                self._needs_skip = True
            elif self._needs_skip:
                new_seeds = [MAX_UINT32] * self.max_batch_size
                self._needs_skip = False
            else:
                # State 3 (steady): device already has SKIP, rand_tile
                # advances on its own, so no host-to-device copy is needed.
                _log_sampling_debug(
                    self._sampling_debug_enabled,
                    "SeedManager seed update skipped",
                    active_slots=_compact_debug_list(empty_slots),
                    seed_active=self._seed_active,
                    reseted=self._reseted,
                    needs_skip=self._needs_skip,
                )
                return
        else:
            new_seeds = [
                self._next_device_seed_for_slot(i) if i in empty_slot_set else MAX_UINT32
                for i in range(self.max_batch_size)
            ]
            if replicate_seeds:
                assert len(empty_slots) == 1, "Cannot replicate seeds if empty_slots is not length 1"
                new_seeds = self.max_batch_size * [new_seeds[empty_slots[0]]]

        _log_sampling_debug(
            self._sampling_debug_enabled,
            "SeedManager seed update",
            active_slots=_compact_debug_list(empty_slots),
            replicate_seeds=replicate_seeds,
            seed_active=self._seed_active,
            reseted=self._reseted,
            needs_skip=self._needs_skip,
            new_device_seeds=_compact_debug_list(new_seeds),
            state_after_counter_advance=self._debug_state(empty_slots),
        )

        new_seed_tt = ttnn.from_torch(
            torch.tensor(new_seeds), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._seed_mapper
        )
        ttnn.copy_host_to_device_tensor(new_seed_tt, self.tt_sampling.seeds_tt_tensor)
        self._reseted = False
