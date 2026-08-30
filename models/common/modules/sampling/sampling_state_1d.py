# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-lane TTTv2 sampling state orchestration.

``Sampling1D`` owns sampling programs and a stable seed buffer.  This module
borrows that sampler, constructs the matching ``Penalties1D`` module, and
returns all mutable request state to the caller in ``SamplingState1DState``.
It does not import or fall back to the legacy sampling generator, penalties,
or seed manager.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from models.common.modules.lazy_buffer import LazyBuffer
from models.common.modules.sampling.params import PreparedSamplingParams, place_prepared_sampling_params
from models.common.modules.sampling.penalties_1d import (
    Penalties1D,
    Penalties1DConfig,
    PenaltyAccumulator,
    PenaltyParams,
    _materialize,
)
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.seed_manager_1d import SeedManager1D, SeedState


@dataclass(frozen=True)
class SamplingStaticIdentity1D:
    """Sampling choices that must participate in trace/program identity."""

    sampling_path: str
    penalties_enabled: bool
    log_probs_enabled: bool
    logprob_modes: tuple[str, ...]


@dataclass
class SamplingState1DState:
    """Mutable per-lane state owned by the runtime caller."""

    seed_state: SeedState
    penalty_params: PenaltyParams
    penalty_accumulator: PenaltyAccumulator
    active_mask: tuple[bool, ...]
    static_identity: SamplingStaticIdentity1D | None = None
    pending_sample_id: int | None = None
    next_sample_id: int = 0
    penalty_history_valid: bool = True
    released: bool = False

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(slot for slot, active in enumerate(self.active_mask) if active)


class SamplingState1D:
    """Compose one borrowed sampler with native seed and penalty state.

    One instance belongs to one executor lane.  The ``Sampling1D`` argument is
    borrowed and is never released here.  The topology-matched ``Penalties1D``
    instance is owned by this controller.  Mutable request state is returned by
    :meth:`create_state` rather than retained on the controller.
    """

    def __init__(
        self,
        sampling: Sampling1D,
        *,
        penalties_factory: Callable[[Penalties1DConfig], Penalties1D] | None = None,
        seed_manager_factory: Callable[[Any], SeedManager1D] | None = None,
    ) -> None:
        sampling_config = getattr(sampling, "config", None)
        if sampling_config is None:
            raise TypeError("sampling must expose a resolved Sampling1D config")
        self._validate_1d_topology(sampling_config.mesh_device)

        penalties_config = Penalties1DConfig(
            vocab_size=int(sampling_config.vocab_size),
            mesh_device=sampling_config.mesh_device,
            max_batch_size=int(sampling_config.max_batch_size),
            sub_core_grids=sampling_config.sub_core_grids,
        )
        make_penalties = penalties_factory or Penalties1D.from_config
        make_seed_manager = seed_manager_factory or SeedManager1D

        self.sampling = sampling
        self.penalties = make_penalties(penalties_config)
        try:
            self.seed_manager = make_seed_manager(sampling_config)
            self._validate_component_contracts()
        except BaseException as primary:
            try:
                self.penalties.release()
            except BaseException as cleanup_error:
                self._attach_cleanup_failures(primary, (cleanup_error,))
            raise
        self._state_leased = False

    @property
    def sampling_config(self):
        return self.sampling.config

    @property
    def penalties_config(self):
        return self.penalties.config

    def create_state(self) -> SamplingState1DState:
        """Materialize buffers and return fresh caller-owned lane state."""

        if self._state_leased:
            raise RuntimeError("one SamplingState1DState is already live for this lane")
        state = None
        try:
            self.penalties.load_device_buffers()
            self._write_noop_penalty_params()
            state = SamplingState1DState(
                seed_state=self.seed_manager.create_state(),
                penalty_params=PenaltyParams(
                    prompt_mask=_materialize(self.penalties.config.prompt_mask),
                    presence_penalties=_materialize(self.penalties.config.presence_penalties),
                    frequency_penalties=_materialize(self.penalties.config.frequency_penalties),
                    repetition_penalties=_materialize(self.penalties.config.repetition_penalties),
                    inverse_repetition_penalties=_materialize(self.penalties.config.inverse_repetition_penalties),
                ),
                penalty_accumulator=PenaltyAccumulator(
                    output_mask=_materialize(self.penalties.config.output_mask),
                    output_counts=_materialize(self.penalties.config.output_counts),
                    output_counts_gathered=_materialize(self.penalties.config.output_counts_gathered),
                ),
                active_mask=(False,) * int(self.sampling.config.max_batch_size),
            )
            self.seed_manager.reset(state.seed_state)
            self._rebuild_penalty_history(
                state,
                prompt_tokens=None,
                output_tokens=None,
                active_mask=state.active_mask,
            )
        except BaseException as primary:
            cleanup_failures = []
            if state is not None:
                try:
                    self.seed_manager.reset(state.seed_state)
                except BaseException as error:
                    cleanup_failures.append(error)
            try:
                self.penalties.release()
            except BaseException as error:
                cleanup_failures.append(error)
            self._attach_cleanup_failures(primary, cleanup_failures)
            raise
        self._state_leased = True
        return state

    def static_identity(self, prepared: PreparedSamplingParams) -> SamplingStaticIdentity1D:
        """Return the trace/program identity for one prepared request."""

        self._validate_prepared(prepared)
        active_modes = tuple(mode for active, mode in zip(prepared.active_mask, prepared.logprob_modes) if active)
        return SamplingStaticIdentity1D(
            sampling_path=prepared.sampling_path,
            penalties_enabled=prepared.penalties_enabled,
            log_probs_enabled=prepared.log_probs_enabled,
            logprob_modes=active_modes,
        )

    def admit(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        slots: Iterable[int] | None = None,
    ) -> None:
        """Register prefill admissions and rebuild complete penalty history."""

        self._require_idle(state)
        self._validate_prepared(prepared)
        if prepared.slot_remap is not None:
            raise ValueError("consume slot_remap with apply_slot_remap before admission")
        self._validate_prepared_history(prepared)
        active_slots = self._active_slots(prepared)
        admission_slots = active_slots if slots is None else tuple(int(slot) for slot in slots)
        if len(set(admission_slots)) != len(admission_slots):
            raise ValueError("admission slots must be unique")
        if any(slot not in active_slots for slot in admission_slots):
            raise ValueError("admission slots must be active prepared-sampling rows")

        request_seeds = tuple(prepared.seeds[slot] for slot in admission_slots)
        self.seed_manager.admit(state.seed_state, request_seeds, admission_slots)
        self.seed_manager.synchronize(
            state.seed_state,
            prepared.seeds,
            active_slots,
            reset_batch=True,
        )
        self._synchronize_penalties_and_identity(state, prepared, rebuild_history=True)

    def admit_prefill(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        slots: Iterable[int],
        positions: Sequence[int] | None = None,
    ) -> None:
        """Admit one request-ordered prefill into its persistent decode slot."""

        self._require_idle(state)
        self._validate_prepared(prepared)
        self._validate_prepared_history(prepared)
        active_sources = self._active_slots(prepared)
        destination_slots = tuple(int(slot) for slot in slots)
        if len(active_sources) != 1 or len(destination_slots) != 1:
            raise ValueError("native device-sampled prefill currently requires exactly one active request")
        destination = destination_slots[0]
        if prepared.slot_remap is not None:
            self.seed_manager.apply_slot_remap(state.seed_state, prepared.slot_remap)
        placed = place_prepared_sampling_params(prepared, destination_slots)
        execution_prepared = _broadcast_prefill_prepared(prepared, active_sources[0])
        self.seed_manager.admit(
            state.seed_state,
            (prepared.seeds[active_sources[0]],),
            destination_slots,
        )
        self._write_penalty_params(execution_prepared)
        self._rebuild_penalty_history(
            state,
            prompt_tokens=execution_prepared.prompt_tokens,
            output_tokens=execution_prepared.output_tokens,
            active_mask=execution_prepared.active_mask,
        )
        position = None if positions is None else int(tuple(positions)[0])
        if prepared.sampling_path == "topk":
            self.seed_manager.refresh_prefill_replicated(
                state.seed_state,
                destination,
                position=position,
            )
        else:
            self.seed_manager.restore_defaults(state.seed_state)
        state.active_mask = tuple(state.seed_state.active)
        state.static_identity = self.static_identity(placed)
        state.penalty_history_valid = not prepared.penalties_enabled

    def prefill_forward(
        self,
        logits,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        k=None,
        p=None,
        temp=None,
        tt_out_tok=None,
        count_tokens: bool = True,
    ):
        """Apply request-ordered prefill penalties, sample, and count once."""

        self._require_idle(state)
        self._validate_prepared(prepared)
        if prepared.penalties_enabled:
            logits = self.penalties.decode_forward(logits, state.penalty_params, state.penalty_accumulator)
        output = self.sampling.decode_forward(
            logits,
            k=k,
            p=p,
            temp=temp,
            seeds=self.seed_manager.get_seed_device_buffer() if prepared.sampling_path == "topk" else None,
            tt_out_tok=tt_out_tok,
            enable_log_probs=list(prepared.enable_log_probs),
        )
        if prepared.penalties_enabled and count_tokens:
            self.penalties.update_output_tokens(state.penalty_accumulator, output[0])
            state.penalty_history_valid = False
        return output

    def apply_slot_remap(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
    ) -> PreparedSamplingParams:
        """Move state, rebuild histories, and return a consumed-remap value."""

        self._require_idle(state)
        self._validate_prepared(prepared)
        if prepared.slot_remap is None:
            raise ValueError("prepared sampling state does not contain a slot_remap")
        self._validate_prepared_history(prepared)
        active_slots = self._active_slots(prepared)
        projected_active, projected_seeds = self._project_seed_membership(state.seed_state, prepared.slot_remap)
        new_slots = self._validate_remapped_decode_membership(
            prepared,
            active_slots,
            projected_active=projected_active,
            projected_seeds=projected_seeds,
        )
        self.seed_manager.apply_slot_remap(state.seed_state, prepared.slot_remap)
        if new_slots:
            self.seed_manager.admit(
                state.seed_state,
                tuple(prepared.seeds[slot] for slot in new_slots),
                new_slots,
            )
        self.seed_manager.synchronize(
            state.seed_state,
            prepared.seeds,
            active_slots,
            reset_batch=False,
        )
        self._synchronize_penalties_and_identity(state, prepared, rebuild_history=True)
        return replace(prepared, slot_remap=None)

    def synchronize_decode(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        reset_batch: bool,
    ) -> PreparedSamplingParams:
        """Synchronize one decode boundary without resetting survivor streams.

        New requests are admitted by prefill.  ``reset_batch`` permits changed
        active membership and rebuilds host-provided prompt/output histories;
        unchanged slots retain their SeedState counters, salts, and RNG state.
        """

        self._require_idle(state)
        self._validate_prepared(prepared)
        if reset_batch:
            self._validate_prepared_history(prepared)
        if not state.penalty_history_valid and not reset_batch:
            raise RuntimeError("decode after sampled prefill requires reset_batch=True with complete output history")
        if prepared.slot_remap is not None:
            prepared = self.apply_slot_remap(state, prepared)
        active_slots = self._active_slots(prepared)
        self.seed_manager.synchronize(
            state.seed_state,
            prepared.seeds,
            active_slots,
            reset_batch=bool(reset_batch),
        )
        identity = self.static_identity(prepared)
        rebuild_history = bool(reset_batch)
        if state.static_identity is None and active_slots:
            if not reset_batch:
                raise RuntimeError("initial decode sampling state requires prefill admission or reset_batch=True")
            rebuild_history = True
        if rebuild_history:
            self._synchronize_penalties_and_identity(state, prepared, rebuild_history=True)
        else:
            state.active_mask = prepared.active_mask
            state.static_identity = identity
            self._write_penalty_params(prepared)
        return prepared

    @staticmethod
    def _project_seed_membership(seed_state: SeedState, remap: Sequence[int] | torch.Tensor):
        """Project active/seed membership through one remap without mutation."""

        flat = remap.reshape(-1).tolist() if isinstance(remap, torch.Tensor) else list(remap)
        capacity = len(seed_state.active)
        if len(flat) != capacity:
            raise ValueError(f"slot remap must contain {capacity} entries")
        normalized = tuple(int(slot) for slot in flat)
        if any(slot < 0 or slot >= capacity for slot in normalized):
            raise ValueError("slot remap source is outside the seed-state capacity")
        moves = tuple((old_slot, new_slot) for new_slot, old_slot in enumerate(normalized) if old_slot != new_slot)
        moved_sources = tuple(old_slot for old_slot, _ in moves)
        if len(set(moved_sources)) != len(moved_sources):
            raise ValueError("slot remap cannot copy one seed stream into multiple destinations")

        active_before = tuple(seed_state.active)
        seeds_before = tuple(seed_state.request_seeds)
        projected_active = list(active_before)
        projected_seeds = list(seeds_before)
        for old_slot, new_slot in moves:
            projected_active[new_slot] = active_before[old_slot]
            projected_seeds[new_slot] = seeds_before[old_slot] if active_before[old_slot] else None
        moved_destinations = {new_slot for _, new_slot in moves}
        for old_slot in set(moved_sources).difference(moved_destinations):
            projected_active[old_slot] = False
            projected_seeds[old_slot] = None
        return tuple(projected_active), tuple(projected_seeds)

    @staticmethod
    def _validate_remapped_decode_membership(
        prepared: PreparedSamplingParams,
        active_slots: Sequence[int],
        *,
        projected_active: Sequence[bool],
        projected_seeds: Sequence[int | None],
    ) -> tuple[int, ...]:
        """Return genuinely new destinations or reject a changed survivor."""

        changed_slots = tuple(
            slot for slot in active_slots if projected_active[slot] and projected_seeds[slot] != prepared.seeds[slot]
        )
        if changed_slots:
            raise RuntimeError(
                "changed active seed slots require reset_batch=True or an explicit prefill admission: "
                f"{list(changed_slots)}"
            )
        new_slots = tuple(slot for slot in active_slots if not projected_active[slot])
        if new_slots and not any(projected_active[slot] for slot in active_slots):
            raise RuntimeError(
                "new or changed active seed slots require reset_batch=True or an explicit prefill admission: "
                f"{list(new_slots)}"
            )
        return new_slots

    def cleanup(
        self,
        state: SamplingState1DState,
        live_slots: Iterable[int],
        *,
        prepared: PreparedSamplingParams | None = None,
    ) -> None:
        """Remove completed requests and clear or rebuild their penalty state."""

        self._require_idle(state)
        live = tuple(int(slot) for slot in live_slots)
        if len(set(live)) != len(live):
            raise ValueError("live_slots must be unique")
        capacity = int(self.sampling.config.max_batch_size)
        if any(slot < 0 or slot >= capacity for slot in live):
            raise ValueError("live_slots contain a row outside the lane capacity")
        if live and prepared is None:
            raise ValueError("prepared sampling state is required to preserve live penalty history")
        if prepared is not None:
            self._validate_prepared(prepared)
            if prepared.slot_remap is not None:
                raise ValueError("consume slot_remap before cleanup")
            self._validate_prepared_history(prepared)
            if set(live) != set(self._active_slots(prepared)):
                raise ValueError("live_slots must match the active rows in prepared sampling state")
        self.seed_manager.cleanup(state.seed_state, live)
        if not live:
            self._write_noop_penalty_params()
            inactive_mask = (False,) * capacity
            self._rebuild_penalty_history(
                state,
                prompt_tokens=None,
                output_tokens=None,
                active_mask=inactive_mask,
            )
            state.active_mask = inactive_mask
            state.static_identity = None
            state.penalty_history_valid = True
            return
        self._synchronize_penalties_and_identity(state, prepared, rebuild_history=True)

    def reset(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams | None = None,
    ) -> None:
        """Clear the lane, optionally admitting one complete replacement batch."""

        self._require_idle(state)
        self.seed_manager.reset(state.seed_state)
        self._write_noop_penalty_params()
        inactive_mask = (False,) * int(self.sampling.config.max_batch_size)
        self._rebuild_penalty_history(
            state,
            prompt_tokens=None,
            output_tokens=None,
            active_mask=inactive_mask,
        )
        state.active_mask = inactive_mask
        state.static_identity = None
        state.penalty_history_valid = True
        if prepared is not None:
            self.admit(state, prepared)

    def refresh_dynamic_inputs(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        positions=None,
        advance_seeds: bool = True,
    ) -> None:
        """Refresh penalty constants and stable seeds before eager/trace replay."""

        self._require_idle(state)
        self._validate_prepared(prepared)
        if prepared.slot_remap is not None:
            raise RuntimeError("apply slot_remap before refreshing dynamic sampling inputs")
        identity = self.static_identity(prepared)
        if state.static_identity is None:
            raise RuntimeError("sampling state must be admitted before refresh")
        if identity != state.static_identity:
            raise RuntimeError(
                "sampling static identity changed; select the matching trace/program and "
                "admit, remap, or reset the lane before replay"
            )

        active_slots = self._active_slots(prepared)
        self.seed_manager.synchronize(
            state.seed_state,
            prepared.seeds,
            active_slots,
            reset_batch=False,
        )
        self._write_penalty_params(prepared)
        if prepared.sampling_path == "topk" and advance_seeds:
            self.seed_manager.refresh(state.seed_state, active_slots, positions=positions)
        else:
            self.seed_manager.restore_defaults(state.seed_state)
        state.active_mask = prepared.active_mask

    def before_sampling(
        self,
        logits,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        positions=None,
        advance_seeds: bool = True,
    ):
        """Apply penalties before sampling and open one exactly-once update."""

        self._require_idle(state)
        if advance_seeds:
            self.refresh_dynamic_inputs(
                state,
                prepared,
                positions=positions,
                advance_seeds=True,
            )
        else:
            # Compile-only execution and trace capture prepare state before
            # entering the device body.  Do not issue host-to-device writes
            # here: a captured write would replay after the runtime refresh and
            # restore capture-time penalty/seed values over the live request.
            self._validate_prepared(prepared)
            if prepared.slot_remap is not None:
                raise RuntimeError("apply slot_remap before entering the sampling body")
            identity = self.static_identity(prepared)
            if state.static_identity != identity:
                raise RuntimeError("sampling body state does not match its static trace/program identity")
        if prepared.penalties_enabled:
            logits = self.penalties.decode_forward(
                logits,
                state.penalty_params,
                state.penalty_accumulator,
            )
        sample_id = state.next_sample_id
        state.next_sample_id += 1
        state.pending_sample_id = sample_id
        return logits, sample_id

    def after_sampling(
        self,
        state: SamplingState1DState,
        sampled_tokens,
        *,
        sample_id: int,
        count_tokens: bool = True,
    ) -> None:
        """Record sampled tokens exactly once for the matching sampling step."""

        self._validate_state(state)
        if state.pending_sample_id != int(sample_id):
            raise RuntimeError(
                f"sample_id {sample_id} is not the pending sampling step " f"({state.pending_sample_id})"
            )
        if count_tokens and state.static_identity is not None and state.static_identity.penalties_enabled:
            # Leave the step pending if the device update fails: retrying blindly
            # could count a partially applied token twice.
            self.penalties.update_output_tokens(state.penalty_accumulator, sampled_tokens)
        state.pending_sample_id = None

    def cancel_sampling(self, state: SamplingState1DState, *, sample_id: int) -> None:
        """Cancel a step only when sampling produced no token."""

        self._validate_state(state)
        if state.pending_sample_id != int(sample_id):
            raise RuntimeError(f"cannot cancel non-pending sample_id {sample_id}")
        state.pending_sample_id = None

    def decode_forward(
        self,
        logits,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        k=None,
        p=None,
        temp=None,
        positions=None,
        tt_out_tok=None,
        count_tokens: bool = True,
        advance_seeds: bool = True,
    ):
        """Apply penalties, sample, and update output history in order.

        Eager compile-only warmups set ``count_tokens=False`` so they do not
        admit a phantom generated token.  Trace capture keeps it true because
        capture records the update for replay without executing it immediately.
        """

        if prepared.sampling_path == "argmax":
            if any(value is not None for value in (k, p, temp)):
                raise ValueError("argmax sampling requires k, p, and temp to be None")
        elif any(value is None for value in (k, p, temp)):
            raise ValueError("topk sampling requires k, p, and temp device tensors")

        penalized_logits, sample_id = self.before_sampling(
            logits,
            state,
            prepared,
            positions=positions,
            advance_seeds=advance_seeds,
        )
        try:
            output = self.sampling.decode_forward(
                penalized_logits,
                k=k,
                p=p,
                temp=temp,
                seeds=(self.seed_manager.get_seed_device_buffer() if prepared.sampling_path == "topk" else None),
                tt_out_tok=tt_out_tok,
                enable_log_probs=list(prepared.enable_log_probs),
            )
        except BaseException:
            self.cancel_sampling(state, sample_id=sample_id)
            raise
        sampled_tokens = output[0]
        self.after_sampling(
            state,
            sampled_tokens,
            sample_id=sample_id,
            count_tokens=count_tokens,
        )
        return output

    def release(self, state: SamplingState1DState | None = None) -> None:
        """Release controller-owned penalties, never the borrowed sampler."""

        if state is None:
            if self._state_leased:
                raise ValueError("the live caller-owned state is required for release")
            self.penalties.release()
            return
        if state.released:
            self.penalties.release()
            self._state_leased = False
            return
        self._require_idle(state)
        failures = []
        try:
            self.seed_manager.reset(state.seed_state)
        except BaseException as error:
            failures.append(error)
        finally:
            state.active_mask = (False,) * int(self.sampling.config.max_batch_size)
            state.static_identity = None
            state.penalty_history_valid = True
            state.released = True
        try:
            self.penalties.release()
        except BaseException as error:
            failures.append(error)
        finally:
            self._state_leased = False
        if failures:
            primary = failures[0]
            self._attach_cleanup_failures(primary, failures[1:])
            raise primary

    # Internal helpers -----------------------------------------------------

    @staticmethod
    def _validate_1d_topology(mesh_device) -> None:
        shape = tuple(int(value) for value in mesh_device.shape)
        if len(shape) != 2 or min(shape) != 1:
            raise ValueError(f"SamplingState1D only supports 1D mesh topologies, got shape {shape}")

    def _validate_component_contracts(self) -> None:
        sampling = self.sampling.config
        penalties = self.penalties.config
        if penalties.mesh_device is not sampling.mesh_device:
            raise ValueError("Penalties1D must borrow the Sampling1D mesh_device")
        if int(penalties.vocab_size) != int(sampling.vocab_size):
            raise ValueError("Penalties1D and Sampling1D vocab_size must match")
        if int(penalties.max_batch_size) != int(sampling.max_batch_size):
            raise ValueError("Penalties1D and Sampling1D max_batch_size must match")
        if penalties.sub_core_grids is not sampling.sub_core_grids:
            raise ValueError("Penalties1D must borrow the Sampling1D sub_core_grids")
        if int(self.seed_manager.max_batch_size) != int(sampling.max_batch_size):
            raise ValueError("SeedManager1D and Sampling1D max_batch_size must match")
        if self.seed_manager.seed_buffer is not sampling.seeds:
            raise ValueError("SeedManager1D must borrow the Sampling1D seeds LazyBuffer")

    def _validate_state(self, state: SamplingState1DState) -> None:
        if not isinstance(state, SamplingState1DState):
            raise TypeError("state must be a caller-owned SamplingState1DState")
        if state.released:
            raise RuntimeError("sampling state was released")
        capacity = int(self.sampling.config.max_batch_size)
        if len(state.active_mask) != capacity or state.seed_state.capacity != capacity:
            raise ValueError("sampling state capacity does not match the lane")

    def _require_idle(self, state: SamplingState1DState) -> None:
        self._validate_state(state)
        if state.pending_sample_id is not None:
            raise RuntimeError(f"sampling step {state.pending_sample_id} must be completed or cancelled first")

    def _validate_prepared(self, prepared: PreparedSamplingParams) -> None:
        if not isinstance(prepared, PreparedSamplingParams):
            raise TypeError("prepared must be PreparedSamplingParams")
        sampling = self.sampling.config
        if prepared.batch_size != int(sampling.max_batch_size):
            raise ValueError("prepared batch_size does not match the Sampling1D lane capacity")
        if prepared.max_device_top_k != int(sampling.max_top_k):
            raise ValueError("prepared max_device_top_k does not match Sampling1D.max_top_k")
        for slot, active in enumerate(prepared.active_mask):
            if not active:
                continue
            repetition = float(prepared.repetition_penalty[slot])
            if repetition <= 0.0:
                raise ValueError("active repetition penalties must be positive")

    @staticmethod
    def _active_slots(prepared: PreparedSamplingParams) -> tuple[int, ...]:
        return tuple(slot for slot, active in enumerate(prepared.active_mask) if active)

    def _validate_prepared_history(self, prepared: PreparedSamplingParams) -> None:
        capacity = int(self.sampling.config.max_batch_size)
        prompt = self._history_tensor(
            prepared.prompt_tokens,
            name="prompt_tokens",
            capacity=capacity,
            active_mask=prepared.active_mask,
        )
        if prompt is None and any(
            active and float(value) != 1.0 for active, value in zip(prepared.active_mask, prepared.repetition_penalty)
        ):
            raise ValueError("prompt_tokens are required when repetition penalty is enabled")
        self._history_tensor(
            prepared.output_tokens,
            name="output_tokens",
            capacity=capacity,
            active_mask=prepared.active_mask,
        )

    def _synchronize_penalties_and_identity(
        self,
        state: SamplingState1DState,
        prepared: PreparedSamplingParams,
        *,
        rebuild_history: bool,
    ) -> None:
        self._write_penalty_params(prepared)
        if rebuild_history:
            self._rebuild_penalty_history(
                state,
                prompt_tokens=prepared.prompt_tokens,
                output_tokens=prepared.output_tokens,
                active_mask=prepared.active_mask,
            )
        state.active_mask = prepared.active_mask
        state.static_identity = self.static_identity(prepared)
        state.penalty_history_valid = True

    def _write_noop_penalty_params(self) -> None:
        capacity = int(self.sampling.config.max_batch_size)
        self._update_penalty_buffer("presence_penalties", torch.zeros(capacity, 1))
        self._update_penalty_buffer("frequency_penalties", torch.zeros(capacity, 1))
        self._update_penalty_buffer("repetition_penalties", torch.ones(capacity, 1))
        self._update_penalty_buffer("inverse_repetition_penalties", torch.ones(capacity, 1))

    def _write_penalty_params(self, prepared: PreparedSamplingParams) -> None:
        presence = []
        frequency = []
        repetition = []
        for slot, active in enumerate(prepared.active_mask):
            presence.append(float(prepared.presence_penalty[slot]) if active else 0.0)
            frequency.append(float(prepared.frequency_penalty[slot]) if active else 0.0)
            repetition.append(float(prepared.repetition_penalty[slot]) if active else 1.0)
        inverse = [1.0 / value for value in repetition]
        self._update_penalty_buffer("presence_penalties", torch.tensor(presence, dtype=torch.float32).reshape(-1, 1))
        self._update_penalty_buffer("frequency_penalties", torch.tensor(frequency, dtype=torch.float32).reshape(-1, 1))
        self._update_penalty_buffer(
            "repetition_penalties", torch.tensor(repetition, dtype=torch.float32).reshape(-1, 1)
        )
        self._update_penalty_buffer(
            "inverse_repetition_penalties", torch.tensor(inverse, dtype=torch.float32).reshape(-1, 1)
        )

    def _update_penalty_buffer(self, name: str, source: torch.Tensor) -> None:
        specification = getattr(self.penalties.config, name)
        if not isinstance(specification, LazyBuffer) and not callable(getattr(specification, "update", None)):
            raise TypeError(f"Penalties1DConfig.{name} must be a mutable LazyBuffer")
        specification.update(source)

    def _rebuild_penalty_history(
        self,
        state: SamplingState1DState,
        *,
        prompt_tokens,
        output_tokens,
        active_mask: tuple[bool, ...],
    ) -> None:
        capacity = int(self.sampling.config.max_batch_size)
        prompt = self._history_tensor(
            prompt_tokens,
            name="prompt_tokens",
            capacity=capacity,
            active_mask=active_mask,
        )
        repetition_needs_prompt = any(
            active and float(value) != 1.0 for active, value in zip(active_mask, self._current_repetition_values())
        )
        if prompt is None and repetition_needs_prompt:
            raise ValueError("prompt_tokens are required when repetition penalty is enabled")
        if prompt is None:
            prompt = torch.full((capacity, 1), -1, dtype=torch.int64)
        output = self._history_tensor(
            output_tokens,
            name="output_tokens",
            capacity=capacity,
            active_mask=active_mask,
        )
        self.penalties.init_prompt_penalties(
            state.penalty_params,
            state.penalty_accumulator,
            prompt,
        )
        self.penalties.reset_output_tokens(state.penalty_accumulator, output)

    @staticmethod
    def _history_tensor(value, *, name: str, capacity: int, active_mask: tuple[bool, ...]):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) == 0:
                return None
            tensor = torch.as_tensor(value)
        else:
            raise TypeError("sampling token history must be a torch.Tensor or sequence")
        if tensor.numel() == 0:
            return None
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        rows = int(tensor.shape[0])
        if rows > capacity:
            raise ValueError(f"{name} has {rows} rows, exceeding lane capacity {capacity}")
        active_slots = tuple(slot for slot, active in enumerate(active_mask) if active)
        if active_slots and rows <= max(active_slots):
            raise ValueError(f"{name} does not cover active slot {max(active_slots)}")
        return tensor

    def _current_repetition_values(self) -> tuple[float, ...]:
        source = getattr(self.penalties.config.repetition_penalties, "source", None)
        if not isinstance(source, torch.Tensor):
            raise TypeError("repetition penalty buffer must retain a torch source")
        return tuple(float(value) for value in source.reshape(-1).tolist())

    @staticmethod
    def _attach_cleanup_failures(primary, failures) -> None:
        if not failures:
            return
        previous = tuple(getattr(primary, "cleanup_failures", ()))
        primary.cleanup_failures = previous + tuple(failures)
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(f"cleanup also encountered {len(failures)} failure(s)")


def _broadcast_prefill_prepared(
    prepared: PreparedSamplingParams,
    source_row: int,
) -> PreparedSamplingParams:
    """Broadcast one request over every physical prefill sampling row."""

    capacity = prepared.batch_size

    def broadcast(values):
        return (values[int(source_row)],) * capacity

    return replace(
        prepared,
        top_k=broadcast(prepared.top_k),
        top_p=broadcast(prepared.top_p),
        temperature=broadcast(prepared.temperature),
        presence_penalty=broadcast(prepared.presence_penalty),
        frequency_penalty=broadcast(prepared.frequency_penalty),
        repetition_penalty=broadcast(prepared.repetition_penalty),
        seeds=broadcast(prepared.seeds),
        enable_log_probs=broadcast(prepared.enable_log_probs),
        num_logprobs=broadcast(prepared.num_logprobs),
        logprob_modes=broadcast(prepared.logprob_modes),
        greedy_mask=broadcast(prepared.greedy_mask),
        row_paths=broadcast(prepared.row_paths),
        active_mask=(True,) * capacity,
        active_rows=capacity,
        prompt_tokens=_broadcast_prefill_history(prepared.prompt_tokens, source_row, capacity),
        output_tokens=_broadcast_prefill_history(prepared.output_tokens, source_row, capacity),
        slot_remap=None,
    )


def _broadcast_prefill_history(value, source_row: int, capacity: int):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            value = value.reshape(1, 1)
        row = 0 if int(value.shape[0]) == 1 else int(source_row)
        return value[row : row + 1].expand((capacity, *value.shape[1:])).clone()
    values = list(value)
    row = values[0 if len(values) == 1 else int(source_row)]
    repeated = [row for _ in range(capacity)]
    return tuple(repeated) if isinstance(value, tuple) else repeated
