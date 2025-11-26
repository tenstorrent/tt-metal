import logging
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from .tt_penalties import TTPenalties
from .tt_sampling import TTSampling
from .tt_sampling import format_sampling_params as _format_sampling_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TraceKey:
    penalties_on: bool
    num_outputs: int


class SamplingGenerator:
    """
    High-level sampling helper that owns both `TTSampling` and `TTPenalties`
    modules and optionally manages TTNN trace capture/execution for sampling.

    Typical usage:
        generator = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=tt_ccl)
        generator.reset_sampling_params(k=..., p=..., temp=...)
        generator.reset_penalty_params(...)
        tokens = generator.sample(logits, seed=seed, enable_trace=True)
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
        enable_penalties: bool = True,
        enable_internal_trace: bool = True,
        cq_id: int = 0,
    ):
        self.mesh_device = mesh_device
        self.cq_id = cq_id
        self.args = args
        self.enable_internal_trace = enable_internal_trace

        self.tt_sampling = TTSampling(mesh_device=mesh_device, tt_ccl=tt_ccl, args=args)
        self.tt_penalties = (
            TTPenalties(mesh_device=mesh_device, max_batch_size=args.max_batch_size, vocab_size=args.vocab_size)
            if enable_penalties
            else None
        )
        self._penalties_active = False

        self._trace_states: dict[_TraceKey, dict] = {}

    def _new_trace_state(self):
        return {"id": None, "input": None, "output": None, "kwargs": {}, "batch_size": None}

    def _trace_slot(self, penalties_on: bool, num_outputs: int):
        key = _TraceKey(penalties_on=penalties_on, num_outputs=num_outputs)
        slot = self._trace_states.get(key)
        if slot is None:
            slot = self._new_trace_state()
            self._trace_states[key] = slot
        return key, slot

    def reset_trace(self):
        """
        Drop any cached trace metadata for both penalties/no-penalties paths.
        """
        for key, slot in self._trace_states.items():
            if slot["id"] is not None:
                logger.debug(
                    "Resetting sampling trace (penalties=%s, num_outputs=%s, trace_id=%s)",
                    key.penalties_on,
                    key.num_outputs,
                    slot["id"],
                )
        self._trace_states.clear()

    # ---------------------------------------------------------------------
    # Penalties helpers
    # ---------------------------------------------------------------------
    def _penalties_enabled(self) -> bool:
        return self.tt_penalties is not None and self._penalties_active

    def _normalize_penalty_values(self, values):
        if values is None:
            return None
        if isinstance(values, (int, float)):
            return [float(values)]
        if isinstance(values, torch.Tensor):
            return values.flatten().tolist()
        return list(values)

    def reset_penalty_params(self, *, presence, frequency, repetition):
        if self.tt_penalties is None:
            return
        presence = self._normalize_penalty_values(presence)
        frequency = self._normalize_penalty_values(frequency)
        repetition = self._normalize_penalty_values(repetition)
        self.tt_penalties.reset_params(presence, frequency, repetition)
        self._penalties_active = not (
            self._is_default_penalty(presence, self._DEFAULT_PENALTIES["presence"])
            and self._is_default_penalty(frequency, self._DEFAULT_PENALTIES["frequency"])
            and self._is_default_penalty(repetition, self._DEFAULT_PENALTIES["repetition"])
        )

    def _is_default_penalty(self, values, default):
        if values is None:
            return True
        if isinstance(values, (int, float)):
            return values == default
        return all(value == default for value in values)

    def reset_prompt_tokens(self, prompt_tokens):
        if not self._penalties_enabled():
            return
        self.tt_penalties.reset_prompt_tokens(prompt_tokens)

    def reset_output_state(self):
        if not self._penalties_enabled():
            return
        self.tt_penalties.reset_output_tokens()

    def update_output_tokens(self, new_tokens):
        if not self._penalties_enabled():
            return
        self.tt_penalties.update_output_tokens(new_tokens)

    def apply_penalties(self, tt_logits, batch_size: int):
        if not self._penalties_enabled():
            return tt_logits
        return self.tt_penalties.apply(tt_logits, batch_size)

    # ---------------------------------------------------------------------
    # Sampling helpers
    # ---------------------------------------------------------------------
    def reset_sampling_params(self, *, k, p, temp, calculate_log_probs=None):
        self.tt_sampling.reset_params(k=k, p=p, temp=temp, calculate_log_probs=calculate_log_probs)

    def _validate_trace_inputs(self, slot, logits: ttnn.Tensor, tt_out_tok: Optional[ttnn.Tensor]):
        if slot["input"] is None or slot["output"] is None:
            raise RuntimeError("Trace metadata missing. Call capture_trace first.")

        if logits is not slot["input"]:
            raise ValueError(
                "The provided logits tensor does not match the tensor used during trace capture. "
                "Call `reset_trace()` before tracing with new tensors."
            )
        if tt_out_tok is not None and tt_out_tok is not slot["output"]:
            raise ValueError(
                "The provided output tensor does not match the tensor used during trace capture. "
                "Call `reset_trace()` before tracing with new tensors."
            )

    def _extract_tokens_host(self, tt_tokens: ttnn.Tensor, batch_size: int) -> torch.Tensor:
        host = ttnn.to_torch(tt_tokens).view(-1)
        return host[:batch_size].to(torch.int64)

    def _update_penalties_from_tokens(self, tt_tokens: ttnn.Tensor, batch_size: int):
        # TODO: penalty histograms disabled for now
        return

    def _run_sampling(
        self,
        logits,
        *,
        penalties_on: bool,
        batch_size: int,
        seed: Optional[int],
        tt_out_tok: Optional[ttnn.Tensor],
        update_state: bool,
        num_outputs: int,
    ):
        if penalties_on:
            self.tt_penalties.apply(logits, batch_size)
        tt_tokens, tt_log_probs = self.tt_sampling(logits, seed=seed, tt_out_tok=tt_out_tok)
        target_tokens = tt_out_tok or tt_tokens
        if penalties_on and update_state and target_tokens is not None:
            self._update_penalties_from_tokens(target_tokens, batch_size)
        if num_outputs != 1:
            raise NotImplementedError("num_outputs > 1 not yet supported on device")
        return tt_tokens, tt_log_probs

    def capture_trace(
        self,
        logits: ttnn.Tensor,
        *,
        seed: Optional[int] = None,
        tt_out_tok: Optional[ttnn.Tensor] = None,
        penalties_on: Optional[bool] = None,
        batch_size: Optional[int] = None,
        update_state: bool = False,
        num_outputs: int = 1,
    ) -> ttnn.Tensor:
        """
        Capture a trace of the sampling pipeline for the given configuration.
        """
        if penalties_on is None:
            penalties_on = self._penalties_enabled()
        batch_size = batch_size or self.args.max_batch_size

        key, slot = self._trace_slot(penalties_on, num_outputs)

        logger.debug("Pre-compiling sampling path before trace capture (penalties=%s)", penalties_on)
        self._run_sampling(
            logits,
            penalties_on=penalties_on,
            batch_size=batch_size,
            seed=seed,
            tt_out_tok=tt_out_tok,
            update_state=False,
            num_outputs=num_outputs,
        )

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)
        sampled = self._run_sampling(
            logits,
            penalties_on=penalties_on,
            batch_size=batch_size,
            seed=seed,
            tt_out_tok=tt_out_tok,
            update_state=update_state,
            num_outputs=num_outputs,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=self.cq_id)
        ttnn.synchronize_device(self.mesh_device)

        slot["id"] = trace_id
        slot["input"] = logits
        slot["output"] = tt_out_tok or sampled
        slot["kwargs"] = {"seed": seed, "tt_out_tok": tt_out_tok}
        slot["batch_size"] = batch_size

        return slot["output"]

    def _execute_trace(self, key: _TraceKey, update_state: bool) -> ttnn.Tensor:
        slot = self._trace_states.get(key)
        if slot is None:
            raise RuntimeError("Trace has not been captured yet.")
        if slot["id"] is None or slot["output"] is None:
            raise RuntimeError("Trace has not been captured yet.")

        ttnn.execute_trace(self.mesh_device, slot["id"], cq_id=self.cq_id, blocking=False)
        if key.penalties_on and update_state:
            self._update_penalties_from_tokens(slot["output"], slot["batch_size"] or self.args.max_batch_size)
        return slot["output"]

    def execute_trace(self, *, num_outputs: int = 1) -> ttnn.Tensor:
        penalties_on = self._penalties_enabled()
        key = _TraceKey(penalties_on=penalties_on, num_outputs=num_outputs)
        return self._execute_trace(key, update_state=True)

    def sample(
        self,
        logits: ttnn.Tensor,
        *,
        enable_trace: bool = False,
        seed: Optional[int] = None,
        tt_out_tok: Optional[ttnn.Tensor] = None,
        batch_size: Optional[int] = None,
        update_state: bool = True,
        num_outputs: int = 1,
    ) -> ttnn.Tensor:
        """
        Convenience wrapper that either runs the sampling module directly or
        replays a captured trace.
        """
        batch_size = batch_size or self.args.max_batch_size
        penalties_on = self._penalties_enabled()
        use_internal_trace = enable_trace and self.enable_internal_trace

        if not use_internal_trace:
            return self._run_sampling(
                logits,
                penalties_on=penalties_on,
                batch_size=batch_size,
                seed=seed,
                tt_out_tok=tt_out_tok,
                update_state=update_state,
                num_outputs=num_outputs,
            )

        key, slot = self._trace_slot(penalties_on, num_outputs)
        if slot["id"] is None:
            return self.capture_trace(
                logits,
                seed=seed,
                tt_out_tok=tt_out_tok,
                penalties_on=penalties_on,
                batch_size=batch_size,
                update_state=update_state,
                num_outputs=num_outputs,
            )

        self._validate_trace_inputs(slot, logits, tt_out_tok)
        slot["batch_size"] = batch_size
        return self._execute_trace(key, update_state=update_state)


def format_sampling_params(*args, **kwargs):
    """
    Re-export helper so callers can import params formatting from the same module.
    """
    return _format_sampling_params(*args, **kwargs)
