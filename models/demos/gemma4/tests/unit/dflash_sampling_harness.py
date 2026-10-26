# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Production sampling state transitions over recording device writes and seeds."""

from __future__ import annotations

from types import MethodType, SimpleNamespace


def sampling_functions():
    from models.common.sampling.generator import SamplingGenerator
    from models.tt_transformers.tt.generator import Generator

    return {
        "_get_sampling_dp": Generator._get_sampling_dp,
        "_prepare_decode_sampling_state": Generator._prepare_decode_sampling_state,
        "_apply_sampling_slot_remap": Generator._apply_sampling_slot_remap,
        **{
            name: getattr(SamplingGenerator, name)
            for name in (
                "apply_decode_state",
                "validate_decode_state_commands",
                "commit_decode_state_commands",
                "reset_sampling_params",
                "reset_prompt_tokens",
                "reset_output_state",
                "apply_slot_remap",
            )
        },
    }


class RecordingSeeds:
    def __init__(self, rows):
        self.max_batch_size = rows
        self.seeds = [None] * rows
        self.positions = [None] * rows
        self.events = []

    def apply_slot_remap(self, remap):
        self.events.append(("remap", list(remap)))
        self.seeds = [self.seeds[row] for row in remap]
        self.positions = [self.positions[row] for row in remap]

    def deactivate_slots_except(self, slots):
        self.events.append(("deactivate", list(slots)))

    def reset_seed_from_slots(self, seeds, slots):
        self.events.append(("reset", list(slots)))
        for slot in slots:
            self.seeds[slot] = seeds[slot]
            self.positions[slot] = 0

    def reset_seed_from_slots_if_needed(self, seeds, slots):
        changed = [slot for slot in slots if self.seeds[slot] != seeds[slot]]
        self.reset_seed_from_slots(seeds, changed)
        return changed

    def align_seed_counters_to_positions(self, seeds, slots, positions):
        self.events.append(("align", list(slots)))
        for slot in slots:
            self.positions[slot] = positions[slot]

    def get_new_values(self, slots=None):
        self.events.append(("draw", None if slots is None else list(slots)))


class RecordingSampler:
    _DEFAULT_PENALTIES = {"presence": 0.0, "frequency": 0.0, "repetition": 1.0}

    def __init__(self, sampling_dp=1):
        self.uploads = []
        self.penalty_uploads = []
        self.prompt_resets = []
        self.output_resets = []
        self._penalties_active = False
        self._slot_state_requires_authoritative_reload = False
        self.seed_manager = RecordingSeeds(32 * sampling_dp)
        self.tt_sampling = SimpleNamespace(
            max_batch_size=32,
            force_argmax_sampling=False,
            log_probs_calculator=SimpleNamespace(enable_log_probs=False),
            reset_params=lambda **kwargs: self.uploads.append(kwargs),
        )
        self.tt_penalties = SimpleNamespace(
            reset_params=lambda *args: self.penalty_uploads.append(args),
            reset_prompt_tokens=lambda tokens, **kwargs: self.prompt_resets.append(tokens),
            reset_output_tokens=lambda tokens, **kwargs: self.output_resets.append(tokens),
        )
        production = sampling_functions()
        for name in (
            "apply_decode_state",
            "validate_decode_state_commands",
            "commit_decode_state_commands",
            "reset_sampling_params",
            "reset_prompt_tokens",
            "reset_output_state",
            "apply_slot_remap",
        ):
            setattr(self, name, MethodType(production[name], self))


def attach_sampling_state(model):
    production = sampling_functions()
    model.data_parallel = len(model.model)
    model.sampling_commands = []
    for target in model.model:
        target.sampling = RecordingSampler(getattr(target, "sampling_dp", 1))
    model._get_sampling_dp = MethodType(production["_get_sampling_dp"], model)
    model._apply_sampling_slot_remap = MethodType(production["_apply_sampling_slot_remap"], model)

    def prepare(self, sampling_params, **kwargs):
        self.sampling_commands.append(dict(kwargs))
        return production["_prepare_decode_sampling_state"](self, sampling_params, **kwargs)

    model._prepare_decode_sampling_state = MethodType(prepare, model)
