# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Request packing shared by the model adapters; no model construction here."""

from contextlib import ExitStack, contextmanager
from dataclasses import fields
from unittest.mock import patch

import torch

from tests.model_behavior.driver import Sampling


class PagedAdapter:
    duplicate_seed_policy = "identical"
    stochastic_prefill = True
    logprob_phases = ("prefill", "decode")
    supports_chunked_prefill = True
    max_seq_len = 8192
    block_size = 64
    physical_blocks = 1024

    @property
    def sampling_model(self):
        return self.generator.model

    def _params(self, states, *, by_slot):
        # Explicit vectors are essential: a scalar temperature in this API only
        # activates row zero. Prefill params are request-ordered, decode params
        # are physical-slot-ordered.
        params = [Sampling() for _ in range(self.capacity)] if by_slot else []
        for state in states:
            if by_slot:
                params[state.slot] = state.request.sampling
            else:
                params.append(state.request.sampling)
        return self.sampling_params_type(**{f.name: [getattr(p, f.name) for p in params] for f in fields(Sampling)})

    def release_request(self, slot):
        # The serving runner calls this optional public hook at completion,
        # while its request-to-state-slot mapping is still available.
        release = getattr(self.generator, "release_request", None)
        if callable(release):
            release(slot)

    def _history(self, states, attribute):
        width = max(1, max(len(getattr(state, attribute)) for state in states))
        result = torch.full((self.capacity, width), -1, dtype=torch.long)
        for state in states:
            values = getattr(state, attribute)
            result[state.slot, : len(values)] = torch.tensor(values, dtype=torch.long)
        return result

    @contextmanager
    def _observe_prefill_offsets(self):
        calls = []

        def observer(original, method):
            def observe(*args, **kwargs):
                if not getattr(self.generator, "warming_up_prefill", False):
                    calls.append(dict(method=method, slot=kwargs["user_id"], start=kwargs["num_cached_tokens"]))
                return original(*args, **kwargs)

            return observe

        with ExitStack() as stack:
            for method in ("prefill_forward_single_user_text", "_easy_trace_prefill"):
                original = getattr(self.generator, method)
                stack.enter_context(patch.object(self.generator, method, observer(original, method)))
            yield calls

    def _checked_prefill_call(self, admitted, lengths, offsets, params):
        with self._observe_prefill_offsets() as calls:
            result = self._prefill_call(admitted, lengths, offsets, params)
        for state, end, start in zip(admitted, lengths, offsets):
            observed = [call for call in calls if call["slot"] == state.slot]
            self.prefill_events.append(
                dict(request_id=state.request.request_id, slot=state.slot, start=start, end=end, observed=observed)
            )
            if start and not any(call["start"] == start for call in observed):
                raise AssertionError(f"Chunked prefill did not resume at {start=} for slot {state.slot}: {calls}")
        return result

    def prefill(self, admitted):
        for state in admitted:
            limit = self.slot_token_capacity[state.slot]
            if len(state.prompt_tokens) + state.request.max_tokens > limit:
                raise ValueError(f"Request {state.request.request_id} exceeds slot {state.slot}'s {limit} tokens")
            ends = state.request.prefill_chunk_ends
            if (
                any(end <= 0 or end >= len(state.prompt_tokens) or end % 128 for end in ends)
                or tuple(sorted(set(ends))) != ends
            ):
                raise ValueError(f"Chunk ends must be increasing, 128-aligned offsets inside the prompt: {ends}")
        chunked = any(state.request.prefill_chunk_ends for state in admitted)
        offsets = [0] * len(admitted)
        # Interleave prefix construction across requests sharing the same cache.
        # sampling_params=None returns logits; intermediate tokens never enter the
        # request's output or penalty history. Final prefill alone samples token 0.
        for stage in range(max((len(state.request.prefill_chunk_ends) for state in admitted), default=0)):
            for row, state in enumerate(admitted):
                if stage < len(state.request.prefill_chunk_ends):
                    end = state.request.prefill_chunk_ends[stage]
                    self._checked_prefill_call([state], [end], [offsets[row]], None)
                    offsets[row] = end
        with self._observe_sampling(admitted) as references:
            call = self._checked_prefill_call if chunked else self._prefill_call
            sampled = call(
                admitted,
                [len(state.prompt_tokens) for state in admitted],
                offsets,
                self._params(admitted, by_slot=False),
            )
        logprobs = None
        if isinstance(sampled, tuple):
            sampled, logprobs = sampled
        return self._map_samples(admitted, sampled, logprobs, references, prefill=True)

    def decode(self, active, *, reset_batch):
        tokens = torch.zeros((self.capacity, 1), dtype=torch.int32)
        positions = torch.full((self.capacity,), -1, dtype=torch.int32)
        for state in active:
            tokens[state.slot, 0] = state.output_tokens[-1]
            positions[state.slot] = state.position
        with self._observe_sampling(active) as references:
            sampled, logprobs = self.generator.decode_forward(
                tokens,
                positions,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                sampling_params=self._params(active, by_slot=True),
                enable_trace=self.enable_trace,
                read_from_device=True,
                async_read=False,
                reset_batch=reset_batch,
                # As in serving, a layout change hands the generator complete active
                # histories. Steady decode advances its resident state itself.
                prompt_tokens=self._history(active, "prompt_tokens") if reset_batch else None,
                output_tokens=self._history(active, "output_tokens") if reset_batch else None,
            )
        if self.enable_trace and not any(t is not None for t in self.generator.trace_ids_decode.values()):
            raise RuntimeError("Traced decode returned without creating a model decode trace")
        if self.enable_trace and all(state.request.sampling.seed is None for state in active):
            sampling = self.sampling_model.sampling
            if sampling.seed_manager.has_active_request_seed():
                raise RuntimeError("An unseeded batch retained a previous request's seed and bypassed sampling traces")
            if not getattr(self.sampling_model, "_tt_disable_sampling_trace", False) and not any(
                state["id"] is not None for state in sampling._trace_states.values()
            ):
                raise RuntimeError("Unseeded traced decode returned without creating a sampling trace")
        return self._map_samples(active, sampled, logprobs, references, prefill=False)
