# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from vllm.sampling_params import SamplingType
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm_tt_plugin.logprobs import build_device_logprobs
from vllm_tt_plugin.model_input import (
    TTModelInput,
    TTSamplingParams,
    slice_tt_sampling_params,
)
from vllm_tt_plugin.structured_output import (
    has_structured_outputs,
    reorder_grammar_bitmask_for_tt_batch,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm_tt_plugin.lane_scheduler import TTStepPlan
    from vllm_tt_plugin.model_runner import TTModelRunner

# Sentinel value for None seed. vLLM treats -1 as equivalent to None
# (see SamplingParams.__post_init__), so we use -1 as the sentinel.
SEED_NONE_SENTINEL = -1

# Sentinel for logprobs=None (disabled). Can't use 0 because
# SamplingParams.logprobs=0 means "return the sampled token's logprob".
# Can't use -1 because SamplingParams.logprobs=-1 means "all vocab logprobs".
# although -1 gets remapped before writing to SamplingInputBatch.num_logprobs
LOGPROBS_NONE_SENTINEL = -2


def build_cached_request_state(new_req_data) -> CachedRequestState:
    """Build a ``CachedRequestState`` for one newly-scheduled request.

    Shared by the front-packed (``TTModelRunner._update_states``) and lane-DP
    (``TTLaneInputBatch.apply_step_plan``) state updates so request-cache
    construction -- prompt validation and per-request generator seeding -- lives
    in exactly one place.
    """
    assert new_req_data.sampling_params is not None, (
        "Pooling is not supported for TT yet"
    )
    if new_req_data.prompt_token_ids is None:
        raise NotImplementedError("TT backend does not support prompt_embeds yet")
    sampling_params = new_req_data.sampling_params
    if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sampling_params.seed)
    else:
        generator = None
    return CachedRequestState(
        req_id=new_req_data.req_id,
        prompt_token_ids=new_req_data.prompt_token_ids,
        mm_features=new_req_data.mm_features,
        sampling_params=sampling_params,
        pooling_params=None,
        generator=generator,
        block_ids=new_req_data.block_ids,
        num_computed_tokens=new_req_data.num_computed_tokens,
        output_token_ids=[],
        lora_request=new_req_data.lora_request,
        prompt_embeds=new_req_data.prompt_embeds,
    )


def apply_cached_req_state_update(
    req_state: CachedRequestState,
    num_computed_tokens: int,
    new_block_ids,
    resumed_from_preemption: bool,
    *,
    num_output_tokens: int,
    all_token_ids: list[int] | None = None,
) -> None:
    """Apply a ``scheduled_cached_reqs`` update to a request's cached state.

    Identical for front-packed and lane-DP: a request resumed from preemption
    had its KV freed and rebuilt (replace block IDs), otherwise the newly
    allocated blocks are appended. Persistent-batch row bookkeeping is left to
    the caller.
    """
    req_state.num_computed_tokens = num_computed_tokens
    if resumed_from_preemption:
        assert new_block_ids is not None
        req_state.block_ids = new_block_ids
        # Async readbacks can outlive preemption. The scheduler discards those
        # outputs, while the worker may already have appended them locally.
        # Only a resumed prefill makes its retained history authoritative;
        # ordinary decode metadata can lag behind the worker's live history.
        if num_output_tokens < 0:
            raise ValueError("Resumed output token count must be nonnegative")
        if all_token_ids is not None:
            prompt = req_state.prompt_token_ids
            assert prompt is not None
            if (
                len(all_token_ids) != len(prompt) + num_output_tokens
                or all_token_ids[: len(prompt)] != prompt
            ):
                raise ValueError(
                    "Resumed token IDs disagree with the prompt or retained output count"
                )
            retained = all_token_ids[len(prompt) :]
        else:
            if num_output_tokens > len(req_state.output_token_ids):
                raise ValueError(
                    "Resumed output history is missing scheduler-retained tokens"
                )
            retained = req_state.output_token_ids[:num_output_tokens]
        # Persistent batches and logits processors can share this list.
        req_state.output_token_ids[:] = retained
    elif new_block_ids is not None:
        for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
            block_ids.extend(new_ids)


def clone_torch_generator(generator: torch.Generator) -> torch.Generator:
    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state())
    return clone


class SamplingInputBatch:
    # Default values for padding sampling parameters in decode mode.
    DEFAULTS = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": SEED_NONE_SENTINEL,  # Sentinel represents None (no seed)
        "num_logprobs": LOGPROBS_NONE_SENTINEL,
    }

    def __init__(self, max_num_reqs: int, logitsprocs: LogitsProcessors | None = None):
        self.max_num_reqs = max_num_reqs
        # Initialize sampling parameter tensors with default values.
        default_tensors = self.create_default_tensors()
        # Set attributes explicitly for each parameter.
        self.temperature = default_tensors["temperature"]
        self.top_p = default_tensors["top_p"]
        self.top_k = default_tensors["top_k"]
        self.presence_penalty = default_tensors["presence_penalty"]
        self.frequency_penalty = default_tensors["frequency_penalty"]
        self.repetition_penalty = default_tensors["repetition_penalty"]
        self.seed = default_tensors["seed"]
        self.num_logprobs = default_tensors["num_logprobs"]
        # Asserting that all defaults have corresponding attributes.
        for name in self.DEFAULTS:
            assert hasattr(self, name), (
                f"Missing attribute '{name}' in SamplingInputBatch"
            )

        # req_index -> generator
        # NOTE: The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # Loaded logits processors (builtin + optional custom), initialized by
        # the model runner and passed in here.
        self.logitsprocs = logitsprocs or LogitsProcessors()

        # Allowed token IDs tracking
        self.has_allowed_token_ids: set[str] = set()
        # NOTE: In the mask tensor, if the corresponding token is allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

    def has_active_logitsprocs(self) -> bool:
        """True if any logits processors have active per-request state."""
        for proc in self.logitsprocs.all:
            if isinstance(proc, MinPLogitsProcessor) and proc.min_p_count:
                return True
            if isinstance(proc, LogitBiasLogitsProcessor) and proc.biases:
                return True
            if isinstance(proc, MinTokensLogitsProcessor) and proc.min_toks:
                return True
        return False

    def create_default_tensors(self) -> dict[str, torch.Tensor]:
        """Create tensors filled with default values for all parameters in
        DEFAULTS."""
        # Map Python types to PyTorch dtypes
        # Note: torch.full infers dtype, but int defaults to int64, so we
        # explicitly specify int32.
        dtype_map = {
            float: torch.float32,
            int: torch.int32,
            bool: torch.bool,
        }
        result: dict[str, torch.Tensor] = {}
        for name, default_value in self.DEFAULTS.items():
            dtype = dtype_map[type(default_value)]
            result[name] = torch.full((self.max_num_reqs,), default_value, dtype=dtype)
        return result


class InputBatch:
    """Persistent input batch, based on InputBatch for GPU/TPU backends."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[int],
        logitsprocs: LogitsProcessors | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}
        # Sampling fast-path bookkeeping (track by req_id like GPUInputBatch).
        # These are used to answer common "batch-wide" queries in O(1).
        self.random_reqs: set[str] = set()
        self.presence_penalties_reqs: set[str] = set()
        self.frequency_penalties_reqs: set[str] = set()
        self.repetition_penalties_reqs: set[str] = set()

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            dtype=torch.int32,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()

        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

        # Block table.
        # vLLM 0.26 accepts per-group capacities instead of max_model_len.
        capacity_kwargs: dict[str, Any]
        if "max_model_len" in inspect.signature(MultiGroupBlockTable).parameters:
            capacity_kwargs = {"max_model_len": max_model_len}
        else:
            capacity_kwargs = {
                "max_num_blocks": [
                    (max_model_len + block_size - 1) // block_size
                    for block_size in block_sizes
                ]
            }
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device="cpu",
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            **capacity_kwargs,
        )

        self.req_output_token_ids: list[list[int] | None] = []

        # Sampling-related.
        self.sampling = SamplingInputBatch(max_num_reqs, logitsprocs=logitsprocs)

        # Condense-move remap: remap[i] = j means slot i's data came from slot j.
        # ``condense`` is the only writer and ``TTLaneInputBatch`` overrides it to a
        # no-op, so ``pop_slot_remap``'s one caller always reads the identity; the
        # non-lane path resets it and uses ``_req_state_slot``, which subsumes it.
        self._slot_remap = torch.arange(max_num_reqs, dtype=torch.int32)

    def reset_slot_remap(self) -> None:
        """Drop any pending slot remap; the identity from here."""
        self._slot_remap = torch.arange(self.max_num_reqs, dtype=torch.int32)

    def pop_slot_remap(self) -> torch.Tensor:
        """Return pending slot remap and reset to identity."""
        remap = self._slot_remap
        self.reset_slot_remap()
        return remap

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        """True iff all active requests are greedy (temperature == 0.0)."""
        return len(self.random_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        """True iff no active request has sampling penalties."""
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: int | None = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs, (
            f"req_index={req_index} >= max_num_reqs={self.max_num_reqs}"
        )

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        prompt_token_ids = request.prompt_token_ids
        assert prompt_token_ids is not None, "prompt_embeds are not supported for TT"
        num_prompt_tokens = len(prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[req_index, :num_prompt_tokens] = prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        self.num_tokens[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # Sampling-related.
        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"

        # Register with batch update builder for logits processors
        self.sampling.batch_update_builder.added.append(
            (
                req_index,
                sampling_params,
                request.prompt_token_ids,
                request.output_token_ids,
            )
        )

        self.sampling.temperature[req_index] = sampling_params.temperature
        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        if not (0 < top_k < self.vocab_size):
            # Normalize top_k <= 0 or >= vocab_size to vocab_size
            # (consider all tokens)
            top_k = self.vocab_size
        # Workaround for https://github.com/tenstorrent/tt-metal/issues/46827
        # top_k == 1 means greedy/argmax for this request. The on-device sampler
        # always builds a fixed top-32 candidate set and its per-user top_k does
        # NOT collapse that set to a single token before the RNG draw, so with
        # any top_p < 1.0 a multi-token nucleus survives and the random seed makes
        # top_k=1 non-deterministic (e.g. Qwen3's generation_config defaults
        # top_p=0.95). Force top_p to 0 so the nucleus keeps exactly the single
        # most-probable token (cum_prob > 0 keeps one), i.e. exact argmax and
        # RNG-independent. Per-request, so mixed-k batches are unaffected.
        if top_k == 1:
            top_p = 0.0
        self.sampling.top_p[req_index] = top_p
        self.sampling.top_k[req_index] = top_k
        self.sampling.presence_penalty[req_index] = sampling_params.presence_penalty
        self.sampling.frequency_penalty[req_index] = sampling_params.frequency_penalty
        self.sampling.repetition_penalty[req_index] = sampling_params.repetition_penalty
        # Store seed, using sentinel value for None
        self.sampling.seed[req_index] = (
            sampling_params.seed
            if sampling_params.seed is not None
            else SEED_NONE_SENTINEL
        )

        # Update fast-path bookkeeping sets.
        # NOTE: Use `discard()` because `req_id` can be reused (abort+resubmit)
        # and slots can be overwritten.
        if sampling_params.temperature == 0.0:
            self.random_reqs.discard(req_id)
        else:
            self.random_reqs.add(req_id)
        if sampling_params.presence_penalty == 0.0:
            self.presence_penalties_reqs.discard(req_id)
        else:
            self.presence_penalties_reqs.add(req_id)
        if sampling_params.frequency_penalty == 0.0:
            self.frequency_penalties_reqs.discard(req_id)
        else:
            self.frequency_penalties_reqs.add(req_id)
        if sampling_params.repetition_penalty == 1.0:
            self.repetition_penalties_reqs.discard(req_id)
        else:
            self.repetition_penalties_reqs.add(req_id)

        # Generator for random sampling
        if request.generator is not None:
            self.sampling.generators[req_index] = request.generator

        # Logprobs (-1 means all vocab logprobs, remap to vocab_size)
        if sampling_params.logprobs is not None:
            self.sampling.num_logprobs[req_index] = (
                self.vocab_size
                if sampling_params.logprobs == -1
                else sampling_params.logprobs
            )
        else:
            self.sampling.num_logprobs[req_index] = LOGPROBS_NONE_SENTINEL

        # Allowed token IDs
        if sampling_params.allowed_token_ids:
            self.sampling.has_allowed_token_ids.add(req_id)
            if self.sampling.allowed_token_ids_mask is None:
                # Lazy allocation for this tensor, which can be large.
                # True means we fill with -inf (disallowed).
                self.sampling.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool, device="cpu"
                )
            self.sampling.allowed_token_ids_mask[req_index] = True
            # False means we don't fill with -inf (allowed).
            self.sampling.allowed_token_ids_mask[req_index][
                sampling_params.allowed_token_ids
            ] = False
        elif self.sampling.allowed_token_ids_mask is not None:
            # This request has no allowlist. The slot may have been reused from
            # a previous request that did, so its mask row could hold stale
            # "disallowed" bits. The mask is read as ``mask[req_indices]``
            # whenever *any* batched request has an allowlist, so a stale row
            # would wrongly constrain this request. Reset it.
            self.sampling.allowed_token_ids_mask[req_index] = False

        # Bad words
        if sampling_params.bad_words_token_ids:
            self.sampling.bad_words_token_ids[req_index] = (
                sampling_params.bad_words_token_ids
            )

    def remove_request(self, req_id: str) -> int | None:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.sampling.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # Update fast-path bookkeeping sets.
        self.random_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)

        # Clean up host-only sampling param tracking
        self.sampling.generators.pop(req_index, None)
        self.sampling.has_allowed_token_ids.discard(req_id)
        self.sampling.bad_words_token_ids.pop(req_index, None)
        # Clear the allowlist mask row so a stale "disallowed" set can never
        # survive into a request that later reuses this slot.
        if self.sampling.allowed_token_ids_mask is not None:
            self.sampling.allowed_token_ids_mask[req_index] = False

        return req_index

    def condense(self, empty_req_indices: list[int]) -> None:
        """Move non-empty requests down into lower, empty indices.

        Args:
            empty_req_indices: empty batch indices, sorted descending.
        """
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Track the move for logits processors
            self.sampling.batch_update_builder.moved.append(
                (last_req_index, empty_index, MoveDirectionality.UNIDIRECTIONAL)
            )
            # Condense-move tracking (see ``self._slot_remap``): the only write.
            self._slot_remap[empty_index] = self._slot_remap[last_req_index]

            # Swap the states.
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            self.num_tokens[empty_index] = num_tokens
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            self.block_table.move_row(last_req_index, empty_index)

            # Sampling-related.
            sampling = self.sampling
            sampling.temperature[empty_index] = sampling.temperature[last_req_index]
            sampling.top_p[empty_index] = sampling.top_p[last_req_index]
            sampling.top_k[empty_index] = sampling.top_k[last_req_index]
            sampling.presence_penalty[empty_index] = sampling.presence_penalty[
                last_req_index
            ]
            sampling.frequency_penalty[empty_index] = sampling.frequency_penalty[
                last_req_index
            ]
            sampling.repetition_penalty[empty_index] = sampling.repetition_penalty[
                last_req_index
            ]
            sampling.seed[empty_index] = sampling.seed[last_req_index]
            sampling.num_logprobs[empty_index] = sampling.num_logprobs[last_req_index]

            # Move host-only sampling params
            if last_req_index in self.sampling.generators:
                self.sampling.generators[empty_index] = self.sampling.generators.pop(
                    last_req_index
                )

            if last_req_index in self.sampling.bad_words_token_ids:
                self.sampling.bad_words_token_ids[empty_index] = (
                    self.sampling.bad_words_token_ids.pop(last_req_index)
                )

            # Move allowed_token_ids_mask row
            if self.sampling.allowed_token_ids_mask is not None:
                self.sampling.allowed_token_ids_mask[empty_index] = (
                    self.sampling.allowed_token_ids_mask[last_req_index]
                )

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs :]
        del self.req_output_token_ids[self.num_reqs :]

    @property
    def max_num_logprobs(self) -> int | None:
        """Returns the max logprobs across requests, or None if none need logprobs."""
        if self.num_reqs == 0:
            return None
        max_val = int(self.sampling.num_logprobs[: self.num_reqs].max().item())
        if max_val < 0:
            return None
        return max_val

    @property
    def no_allowed_token_ids(self) -> bool:
        """True if no requests have allowed_token_ids set."""
        return len(self.sampling.has_allowed_token_ids) == 0

    def refresh_logitsprocs(self) -> None:
        """Update logits processors with batch state changes."""

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        batch_update = self.sampling.batch_update_builder.get_and_reset(self.num_reqs)
        for logit_proc in self.sampling.logitsprocs.all:
            logit_proc.update_state(batch_update)

    def make_prompt_token_ids_tensor(
        self, req_indices: list[int] | None = None
    ) -> torch.Tensor:
        """Create a tensor of prompt token IDs, padded with -1.

        ``req_indices`` selects which rows of the persistent batch to emit (one
        row per index, in order). ``None`` means the whole local batch
        (``range(num_reqs)``). Lane-DP passes one lane's indices so the result
        is attributed to that lane's requests rather than the merged batch's
        leading rows.

        NOTE: TT device sampling relies on -1 as the padding sentinel.
        If these tokens are passed to the host sampler for penalties, they must
        be canonicalized (cast to int64 and -1 replaced with vocab_size) before
        scatter operations.
        """
        rows = list(range(self.num_reqs)) if req_indices is None else list(req_indices)
        n = len(rows)
        idx = np.asarray(rows, dtype=np.int64)
        max_prompt_len = int(self.num_prompt_tokens[idx].max()) if n > 0 else 0
        prompt_token_ids_tensor = torch.full(
            (n, max_prompt_len),
            -1,
            device="cpu",
            dtype=torch.int32,
        )
        prompt_token_ids = prompt_token_ids_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[idx, :max_prompt_len]
        # Pad with -1 for positions beyond actual prompt length
        for row, i in enumerate(rows):
            prompt_token_ids[row, self.num_prompt_tokens[i] :] = -1
        return prompt_token_ids_tensor

    def make_output_token_ids_tensor(
        self, req_indices: list[int] | None = None
    ) -> torch.Tensor:
        """Create a tensor of output token IDs, padded with -1.

        ``req_indices`` selects which rows of the persistent batch to emit (see
        ``make_prompt_token_ids_tensor``).

        NOTE: TT device sampling relies on -1 as the padding sentinel.
        If these tokens are used by the host sampler penalties logic, -1 padding
        should be removed/handled before use.
        """
        rows = list(range(self.num_reqs)) if req_indices is None else list(req_indices)
        n = len(rows)
        idx = np.asarray(rows, dtype=np.int64)
        output_lens = self.num_tokens[idx] - self.num_prompt_tokens[idx]
        max_output_len = int(output_lens.max()) if n > 0 else 0

        output_token_ids_tensor = torch.full(
            (n, max_output_len),
            -1,
            device="cpu",
            dtype=torch.int32,
        )
        output_token_ids = output_token_ids_tensor.numpy()
        # Copy output tokens from token_ids_cpu
        for row, i in enumerate(rows):
            prompt_len = self.num_prompt_tokens[i]
            total_len = self.num_tokens[i]
            output_len = total_len - prompt_len
            if output_len > 0:
                output_token_ids[row, :output_len] = self.token_ids_cpu[
                    i, prompt_len:total_len
                ]
        return output_token_ids_tensor

    def block_tables_for_rows(
        self, rows: torch.Tensor | list[int], width: int
    ) -> list[torch.Tensor]:
        """Per-group block tables sliced to ``rows`` and right-padded on the
        block dimension to ``width``.

        Constant ``width`` (``max_num_blocks_per_req``) is required for ttnn
        tracing: runtime block tables must match the traced width even when
        their underlying group is narrower.
        """
        out: list[torch.Tensor] = []
        for bt in self.block_table.block_tables:
            bt_cpu = bt.get_cpu_tensor()[rows, :width].clone()
            if bt_cpu.shape[1] < width:
                pad = torch.zeros(
                    bt_cpu.shape[0], width - bt_cpu.shape[1], dtype=bt_cpu.dtype
                )
                bt_cpu = torch.cat([bt_cpu, pad], dim=1)
            out.append(bt_cpu)
        return out

    def advance_generators(self, req_indices: list[int] | None = None) -> None:
        # This relies on the fact, that for a torch all_gather_object,
        # the local object is also copied,
        # so the original object is not modified.
        # Otherwise, the generator at local_rank 0
        # would get out of sync with the others.
        #
        # ``req_indices`` restricts advancement to the build's own requests.
        # Each generator belongs to a single request, so lane-DP (which calls
        # this once per lane) passes the lane's indices to advance every
        # generator exactly once per step rather than once per lane. ``None``
        # advances all generators (whole-batch build, called once per step).
        if req_indices is None:
            generators = list(self.sampling.generators.values())
        else:
            generators = [
                self.sampling.generators[i]
                for i in req_indices
                if i in self.sampling.generators
            ]
        for generator in generators:
            # Sample once from the generator to advance its state.
            torch.rand(1, generator=generator)


class TTLaneInputBatch(InputBatch):
    """Persistent input batch for single-process multi-lane (lane-DP) execution.

    One engine process drives ``num_lanes`` data-parallel KV-cache replicas
    ("lanes") that execute in lockstep against a single gathered device batch.
    This batch owns the lane layout so the model runner does not: it lays the
    persistent rows out as ``num_lanes`` contiguous chunks of ``per_lane`` rows
    and binds each request to a stable row for its whole lifetime.

    Layout: lane ``l`` owns rows ``[l * per_lane, (l + 1) * per_lane)``. A
    request placed at lane-local slot ``s`` lives at persistent row
    ``l * per_lane + s``. **That persistent row IS the request's device decode
    slot**, so the merged device input is the batch's own row layout -- no
    scatter, and no separate ``req_id -> slot`` map. ``max_num_reqs`` is
    ``num_lanes * per_lane`` (the global ``max_num_seqs`` in lane mode).

    Stable slots: a request never moves once placed. Removing a request leaves
    its row as an empty gap to be reused by a later request in the same lane.
    There is no condense (``condense`` is a no-op): keeping every live request
    pinned to its row keeps the on-device per-slot seed RNG correct and makes
    the seed manager's ``slot_remap`` the identity. Gaps are reset to neutral
    sampling defaults so they cannot perturb batch-wide flags (``all_greedy`` /
    ``no_penalties``) or sample an invalid value.

    Merged host sampling: because rows are the device slots and gaps carry
    neutral defaults, the runner samples the whole ``max_num_reqs`` slot batch
    in one call against one :class:`SamplingMetadata` built here over every row
    (``build_merged_sampling_metadata``). The builtin/custom logits processors
    keep per-row state over this full slot batch (``refresh_logitsprocs`` passes
    ``max_num_reqs`` as the batch size), exactly like a normal single-engine
    vLLM batch -- so there is no per-lane slicing, no per-lane generator/penalty
    remap, and custom logits processors work unchanged. Pad rows sample greedy
    garbage that the runner drops when reading back the occupied rows.
    """

    def __init__(
        self,
        num_lanes: int,
        per_lane: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        logitsprocs: LogitsProcessors | None = None,
    ):
        if num_lanes < 1 or per_lane < 1:
            raise ValueError(
                f"num_lanes and per_lane must be >= 1, got num_lanes={num_lanes}, "
                f"per_lane={per_lane}"
            )
        self.num_lanes = num_lanes
        self.per_lane = per_lane
        super().__init__(
            max_num_reqs=num_lanes * per_lane,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            logitsprocs=logitsprocs,
        )
        # Rows are a fixed slot grid (lane-chunked), not a front-packed list:
        # pre-size so a request can occupy any slot in its lane's chunk, with
        # gaps, instead of always appending at ``num_reqs``.
        self._req_ids = [None] * self.max_num_reqs
        self.req_output_token_ids = [None] * self.max_num_reqs

    # ------------------------------------------------------------------
    # Lane geometry / membership
    # ------------------------------------------------------------------

    def lane_of(self, req_id: str) -> int:
        """Return the lane a request is bound to.

        Derived from the request's row: lane ``l`` owns the contiguous chunk
        ``[l * per_lane, (l + 1) * per_lane)``, so the row alone determines the
        lane and no separate ``req_id -> lane`` map is needed.
        """
        return self.req_id_to_index[req_id] // self.per_lane

    def occupied_rows(self) -> list[int]:
        """Persistent rows holding a live request, in ascending (lane-major,
        slot) order. This is the canonical merged order used for output."""
        return [row for row, rid in enumerate(self._req_ids) if rid is not None]

    # ------------------------------------------------------------------
    # Placement (stable lane-local slots)
    # ------------------------------------------------------------------

    def add_request_to_row(self, request: "CachedRequestState", row: int) -> int:
        """Materialize a scheduler-owned stable row assignment.

        :class:`~vllm_tt_plugin.lane_scheduler.TTLaneCoordinator` owns slot
        allocation (which lane, which free row); this batch only places the
        request at the row the coordinator already chose.
        """
        if not (0 <= row < self.max_num_reqs):
            raise ValueError(f"row {row} out of range [0, {self.max_num_reqs})")
        if self._req_ids[row] is not None and self._req_ids[row] != request.req_id:
            raise ValueError(f"row {row} is already occupied")
        # If this row was freed earlier in the same step it is still in the
        # logitsproc batch-update ``removed`` list. Drop it so the reused row is
        # recorded only as an ``added`` update, not both -- mirroring upstream
        # ``gpu_input_batch._register_add_request``'s ``pop_removed()`` so the
        # builtin logits processors do not first set then clear the new
        # request's per-row state.
        builder = self.sampling.batch_update_builder
        if row in builder._removed:
            builder._removed.remove(row)
        super().add_request(request, row)
        return row

    def remove_request(self, req_id: str) -> int | None:
        row = super().remove_request(req_id)
        if row is not None:
            self._reset_slot(row)
        return row

    def _reset_slot(self, row: int) -> None:
        """Reset a freed row to neutral defaults so a gap never perturbs the
        merged batch's sampling. ``super().remove_request`` already clears the
        generator, bad-words and allowed-token-ids entries; this also resets the
        per-row sampling tensors and token counts (so the row reads as an empty,
        greedy, no-penalty request until it is reused)."""
        sampling = self.sampling
        for name, default in sampling.DEFAULTS.items():
            getattr(sampling, name)[row] = default
        self.num_tokens[row] = 0
        self.num_prompt_tokens[row] = 0
        self.num_computed_tokens_cpu[row] = 0

    def condense(self, empty_req_indices: list[int]) -> None:
        """No-op: lane slots are stable.

        The base class condenses by moving the highest live request into the
        lowest empty index. That would move requests across lane boundaries and
        shift their device slots, corrupting the on-device per-slot seed RNG.
        Lane mode instead leaves freed rows as gaps (reused in place by later
        requests in the same lane), so live requests never move and the seed
        manager's ``slot_remap`` stays the identity.
        """
        return

    # ------------------------------------------------------------------
    # State update (lane step plan -> batch + request map)
    # ------------------------------------------------------------------

    def apply_step_plan(
        self,
        scheduler_output: "SchedulerOutput",
        plan: "TTStepPlan",
        requests: dict[str, CachedRequestState],
        encoder_cache: dict,
    ) -> bool:
        """Apply one lane step plan to this batch and the runner's request map.

        This is the lane-DP variant of ``TTModelRunner._update_states``. Unlike
        the front-packed batch it does **not** evict merely-unscheduled
        requests: a prefill step can leave running decodes unscheduled, and
        freeing their stable device slot would disturb the on-device per-slot
        seed RNG. Only finished requests, and requests resumed from preemption
        (whose KV was rebuilt), release their slot. There is no condense.

        ``requests`` (the runner's canonical ``req_id -> CachedRequestState``
        map) and ``encoder_cache`` are mutated in place. Placement rows come
        from ``plan.req_id_to_row`` -- the scheduler owns slot allocation, this
        batch only materializes it. Returns whether the decode layout changed
        (a placement or a freed slot), which the caller uses to reset the device
        decode batch.
        """
        layout_changed = False

        # Finished requests release their slot.
        for req_id in scheduler_output.finished_req_ids:
            requests.pop(req_id, None)
            if self.remove_request(req_id) is not None:
                layout_changed = True

        # Free cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            encoder_cache.pop(mm_hash, None)

        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            requests[req_id] = build_cached_request_state(new_req_data)
            req_ids_to_add.append(req_id)

        # Running / resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            apply_cached_req_state_update(
                req_state,
                num_computed_tokens,
                new_block_ids,
                resumed_from_preemption,
                num_output_tokens=req_data.num_output_tokens[i],
                all_token_ids=req_data.all_token_ids.get(req_id),
            )
            if resumed_from_preemption:
                # KV was freed and is being rebuilt; re-add fresh (drop the
                # stale slot first). The slot may differ afterwards --
                # acceptable under the exceptional preemption path, which
                # re-prefills the request anyway.
                if self.remove_request(req_id) is not None:
                    layout_changed = True
                req_ids_to_add.append(req_id)
                continue
            req_index = self.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue
            self.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.block_table.append_row(new_block_ids, req_index)
                if any(new_block_ids):
                    # The lane runner drains pending decode before this reset
                    # reloads tokens, positions and newly allocated pages.
                    layout_changed = True

        # Place new / resumed requests at scheduler-owned stable rows.
        for req_id in req_ids_to_add:
            self.add_request_to_row(requests[req_id], plan.req_id_to_row[req_id])
            layout_changed = True

        self.refresh_logitsprocs()
        return layout_changed

    # ------------------------------------------------------------------
    # Sampling layout (merged, over the full slot batch)
    # ------------------------------------------------------------------

    @property
    def max_num_logprobs(self) -> int | None:
        """Max logprobs across live requests, or None if none need logprobs.

        Computed over every slot row rather than ``[:num_reqs]`` because live
        rows are not front-packed; gap rows carry the ``LOGPROBS_NONE_SENTINEL``
        default (the minimum value), so the max over all rows equals the max
        over the live rows.
        """
        if self.num_reqs == 0:
            return None
        max_val = int(self.sampling.num_logprobs[: self.max_num_reqs].max().item())
        if max_val < 0:
            return None
        return max_val

    def refresh_logitsprocs(self) -> None:
        """Apply batch state changes to logits processors over the full slot
        batch. Passes ``max_num_reqs`` (not ``num_reqs``) as the batch size so
        each processor's per-row state spans every slot, matching the full slot
        logits the runner samples."""
        batch_update = self.sampling.batch_update_builder.get_and_reset(
            self.max_num_reqs
        )
        for logit_proc in self.sampling.logitsprocs.all:
            logit_proc.update_state(batch_update)

    def build_merged_sampling_metadata(
        self,
        scheduled_rows: list[int] | None = None,
        non_sampling_rows: list[int] | None = None,
    ) -> SamplingMetadata:
        """Build one :class:`SamplingMetadata` over every slot row.

        Mirrors a normal single-engine vLLM ``SamplingMetadata`` build, but over
        the full ``max_num_reqs`` slot batch (live rows interleaved with neutral
        gap rows) so it lines up row-for-row with the full slot logits the
        runner hands the host sampler and with the per-row logits-processor
        state. Gap rows carry neutral defaults (greedy, no penalties), so they
        do not change ``all_greedy`` / ``no_penalties`` and sample harmless
        greedy tokens the runner discards.

        ``all_random`` is intentionally computed over the full tensor, so it is
        False whenever any gap exists; that keeps the sampler's div-by-zero
        guard active for the default-``temperature=0`` gap rows.

        ``scheduled_rows`` are the rows actually producing a token this step. The
        host sampler advances *every* per-request generator handed to it (one
        ``exponential_`` draw per row in ``generators``), so only the scheduled
        rows' generators are passed through: a seeded request occupying a slot
        but not scheduled this step (e.g. a running decode request during a
        prefill-only step) must not have its RNG advanced, or its token stream
        would drift by one draw per step it sat out. ``non_sampling_rows`` are
        scheduled intermediate-prefill rows; their generator clones preserve the
        fixed slot layout without advancing the request's real RNG state.
        """
        n = self.max_num_reqs
        sampling = self.sampling
        temperature = sampling.temperature[:n]
        all_greedy = bool((temperature == 0.0).all())
        all_random = bool((temperature != 0.0).all())
        presence = sampling.presence_penalty[:n]
        frequency = sampling.frequency_penalty[:n]
        repetition = sampling.repetition_penalty[:n]
        no_penalties = bool(
            (presence == 0.0).all()
            and (frequency == 0.0).all()
            and (repetition == 1.0).all()
        )
        rows = list(range(n))
        if not no_penalties:
            prompt_token_ids = self.make_prompt_token_ids_tensor(rows).to(torch.int64)
            prompt_token_ids = prompt_token_ids.masked_fill(
                prompt_token_ids == -1, self.vocab_size
            )
            output_rows = self.make_output_token_ids_tensor(rows)
            output_token_ids = [
                [tok for tok in row.tolist() if tok != -1] for row in output_rows
            ]
        else:
            prompt_token_ids = None
            output_token_ids = [[] for _ in range(n)]
        # Only hand the sampler an allowlist mask when some live request
        # actually constrains its tokens. The mask tensor is allocated lazily
        # and never freed, and freed rows are reset to all-False, so once any
        # request has ever used an allowlist the tensor lingers as an all-False
        # no-op; passing it would just make the sampler do wasted masked_fill
        # work. Gate on ``no_allowed_token_ids`` exactly like upstream.
        if self.no_allowed_token_ids:
            allowed_token_ids_mask = None
        else:
            allowed_token_ids_mask = sampling.allowed_token_ids_mask
            if allowed_token_ids_mask is not None:
                allowed_token_ids_mask = allowed_token_ids_mask[:n]
        if scheduled_rows is None:
            generators = dict(sampling.generators)
        else:
            scheduled = set(scheduled_rows)
            non_sampling = set(non_sampling_rows or ())
            generators = {
                row: clone_torch_generator(gen) if row in non_sampling else gen
                for row, gen in sampling.generators.items()
                if row in scheduled
            }
        return SamplingMetadata(
            temperature=temperature if not all_greedy else None,
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=sampling.top_p[:n],
            top_k=sampling.top_k[:n],
            generators=generators,
            max_num_logprobs=self.max_num_logprobs,
            no_penalties=no_penalties,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=frequency,
            presence_penalties=presence,
            repetition_penalties=repetition,
            output_token_ids=output_token_ids,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=dict(sampling.bad_words_token_ids),
            logitsprocs=sampling.logitsprocs,
        )

    # ------------------------------------------------------------------
    # Per-row tensor views (the lane batch owns its own slicing/padding)
    # ------------------------------------------------------------------

    def slot_block_tables(
        self, rows: list[int], zero_gaps: bool, total: int, width: int
    ) -> list[torch.Tensor]:
        """Per-group block tables for ``rows`` (one row per slot), each padded
        to ``width`` (``max_num_blocks_per_req``). When ``zero_gaps`` is set,
        rows of ``range(total)`` not in ``rows`` are zeroed (empty decode slots
        carry no blocks)."""
        occupied = set(rows)
        sel = list(range(total)) if zero_gaps else rows
        out = self.block_tables_for_rows(sel, width)
        if zero_gaps and len(occupied) < total:
            gap = torch.ones(total, dtype=torch.bool)
            gap[list(occupied)] = False
            for bt_cpu in out:
                bt_cpu[gap] = 0
        return [bt.contiguous() for bt in out]

    def slot_sampling_params(self, rows: list[int]) -> TTSamplingParams:
        """Slice the slot-ordered sampling tensors to ``rows``."""
        return slice_tt_sampling_params(
            self.sampling, torch.as_tensor(rows, dtype=torch.long)
        )

    def slot_grammar_bitmask(
        self, grammar_output: "GrammarOutput | None", batch_length: int
    ) -> torch.Tensor | None:
        """Reorder the scheduler grammar bitmask into a full slot-batch tensor.

        Each structured-output request's bitmask row is placed at its device
        slot row; every other slot is left all-ones (all tokens allowed), so the
        mask lines up with the full slot logits the host sampler reads.
        """
        if grammar_output is None or grammar_output.grammar_bitmask is None:
            return None
        bitmask = torch.from_numpy(grammar_output.grammar_bitmask)
        return reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=grammar_output.structured_output_request_ids,
            req_id_to_index=self.req_id_to_index,
            req_indices=list(range(batch_length)),
            batch_length=batch_length,
        )

    # ------------------------------------------------------------------
    # Merged model-input assembly (decode/prefill) and output extraction
    # ------------------------------------------------------------------
    #
    # The lane batch materializes the whole merged device input from its own
    # slot layout and reads the merged device output back into per-request
    # tokens. ``runner`` is passed in for the genuinely runner/model-owned
    # helpers (KV-layer block tables, device-sampling policy, multi-modal
    # gather, the host sampler); everything row/slot-shaped lives here.

    def build_model_input(
        self,
        runner: "TTModelRunner",
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
        plan: "TTStepPlan",
    ) -> TTModelInput:
        """Build the merged device input for one lane step (decode or prefill)."""
        if plan.is_decode:
            return self._build_decode_input(
                runner, scheduler_output, grammar_output, plan
            )
        return self._build_prefill_input(runner, scheduler_output, grammar_output, plan)

    def _build_decode_input(
        self,
        runner: "TTModelRunner",
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
        plan: "TTStepPlan",
    ) -> TTModelInput:
        """Build the merged decode input straight from the slot-ordered batch.

        Every slot is present (gaps padded), so this is the device decode batch
        with no scatter: row == device slot.
        """
        lane_batch = self
        total = plan.capacity
        occupied = lane_batch.occupied_rows()

        num_tokens = lane_batch.num_tokens
        positions_np = num_tokens[:total].astype(np.int32) - 1  # gaps -> -1
        input_positions = torch.from_numpy(positions_np)
        tokens_np = np.zeros((total, 1), dtype=np.int32)
        for row in occupied:
            tokens_np[row, 0] = lane_batch.token_ids_cpu[row, num_tokens[row] - 1]
        input_tokens = torch.from_numpy(tokens_np)

        block_tables_per_group = lane_batch.slot_block_tables(
            occupied, zero_gaps=True, total=total, width=runner.max_num_blocks_per_req
        )
        rows_all = list(range(total))
        tt_sampling_params = lane_batch.slot_sampling_params(rows_all)

        bitmask = lane_batch.slot_grammar_bitmask(grammar_output, total)
        has_structured = has_structured_outputs(
            runner.requests, scheduler_output, bitmask
        )
        perform_device_sampling = runner.check_perform_device_sampling(
            is_decode=True, has_structured_outputs=has_structured
        )

        # The prompt/output token tensors feed device-side penalties only. Host
        # sampling rebuilds them itself in ``build_merged_sampling_metadata``, so
        # building them here too would be dead work on the host path.
        prompt_tokens = output_tokens = None
        if perform_device_sampling and not lane_batch.no_penalties:
            prompt_tokens = lane_batch.make_prompt_token_ids_tensor(rows_all)
            output_tokens = lane_batch.make_output_token_ids_tensor(rows_all)
        reset_batch = runner._decode_layout_changed_since_last_decode
        runner._decode_layout_changed_since_last_decode = False
        slot_remap = lane_batch.pop_slot_remap()  # identity for stable slots

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=None,
            block_tables=block_tables_per_group[0],
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=runner._block_tables_per_layer(
                block_tables_per_group
            ),
            # Device decodes every slot; only used for the empty-batch guard.
            unpadded_batch_size=list(plan.batch_size_per_dp),
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs={},
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=reset_batch,
            slot_remap=slot_remap,
            # Host sampling reads the merged batch directly (see
            # ``extract_output``); the per-rank sidecars are unused here.
            allowed_token_ids_mask_list=[None],
            bad_words_token_ids_list=[{}],
            max_num_logprobs=[lane_batch.max_num_logprobs],
            logitsprocs_list=[None],
            generators_list=[{}],
            prefill_empty_slots=None,
        )

    def _build_prefill_input(
        self,
        runner: "TTModelRunner",
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
        plan: "TTStepPlan",
    ) -> TTModelInput:
        """Build the prefill input for the requests scheduled this step.

        Prefill rows are front-packed in scheduler plan order. The plan carries
        the stable slots (``prefill_empty_slots``) so ``submit_prefill`` seeds
        each user at the device row decode will later read from. The output is
        one token per prefilled request, in this same order.
        """
        lane_batch = self
        rows = list(plan.input_rows)
        rows_np = np.asarray(rows, dtype=np.int64)
        input_positions_np = lane_batch.num_computed_tokens_cpu[rows_np]
        input_positions = torch.from_numpy(input_positions_np.astype(np.int32))
        chunk_lens = np.asarray(
            [
                scheduler_output.num_scheduled_tokens[lane_batch.req_ids[row]]
                for row in rows
            ],
            dtype=np.int64,
        )
        prompt_lens = input_positions_np + chunk_lens
        intermediate_prefill_mask = torch.from_numpy(
            prompt_lens < lane_batch.num_tokens[rows_np]
        )
        max_prefill = int(prompt_lens.max())
        input_tokens = lane_batch.token_ids_cpu_tensor[rows_np, :max_prefill]

        block_tables_per_group = lane_batch.slot_block_tables(
            rows, zero_gaps=False, total=0, width=runner.max_num_blocks_per_req
        )
        tt_sampling_params = lane_batch.slot_sampling_params(rows)
        batch_size_per_dp = list(plan.batch_size_per_dp)

        bitmask = lane_batch.slot_grammar_bitmask(
            grammar_output, lane_batch.max_num_reqs
        )
        has_structured = has_structured_outputs(
            runner.requests, scheduler_output, bitmask
        )
        perform_device_sampling = runner.check_perform_device_sampling(
            is_decode=False, has_structured_outputs=has_structured
        )
        if intermediate_prefill_mask.any():
            perform_device_sampling = False

        # Device-side penalties only; host sampling rebuilds these in
        # ``build_merged_sampling_metadata`` (over the full slot batch), so
        # building them here on the host path would be dead work.
        prompt_tokens = output_tokens = None
        if perform_device_sampling and not lane_batch.no_penalties:
            prompt_tokens = lane_batch.make_prompt_token_ids_tensor(rows)
            output_tokens = lane_batch.make_output_token_ids_tensor(rows)

        multi_modal_kwargs = (
            runner._gather_multi_modal_inputs(req_indices=list(rows))
            if runner.model_config.is_multimodal_model
            else {}
        )

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables_per_group[0],
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=runner._block_tables_per_layer(
                block_tables_per_group
            ),
            unpadded_batch_size=batch_size_per_dp,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=False,
            slot_remap=None,
            allowed_token_ids_mask_list=[None],
            bad_words_token_ids_list=[{}],
            max_num_logprobs=[lane_batch.max_num_logprobs],
            logitsprocs_list=[None],
            generators_list=[{}],
            prefill_empty_slots=(
                list(plan.prefill_empty_slots)
                if plan.prefill_empty_slots is not None
                else None
            ),
            intermediate_prefill_mask=intermediate_prefill_mask,
        )

    def extract_output(
        self,
        runner: "TTModelRunner",
        tt_out: Any,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        is_decode: bool,
    ) -> tuple[torch.Tensor, LogprobsLists | None]:
        """Read back one merged lane step into per-request sampled tokens.

        Returns ``(sampled_token_ids[n, 1], logprobs)`` for the ``n``
        ``scheduled_rows`` in order. Device sampling reads the sampled tokens
        directly from each slot; host sampling runs **one** sampler call over
        the whole slot batch (so the builtin/custom logits processors stay
        row-aligned, with no per-lane slicing) and then picks the scheduled
        rows out of the result. Also called from ``TTAsyncDecodeController`` to
        finalize an async lane-decode step.
        """
        n = len(scheduled_rows)
        rows_t = torch.as_tensor(scheduled_rows, dtype=torch.long)
        intermediate_prefill_mask = getattr(
            model_input, "intermediate_prefill_mask", None
        )
        intermediate_rows: list[int] = []
        if not is_decode and intermediate_prefill_mask is not None:
            assert intermediate_prefill_mask.numel() == n
            intermediate_rows = [
                row
                for row, is_intermediate in zip(
                    scheduled_rows, intermediate_prefill_mask.tolist(), strict=True
                )
                if is_intermediate
            ]
            if len(intermediate_rows) == n:
                return torch.zeros((n, 1), dtype=torch.int32), None
        if model_input.perform_device_sampling:
            assert not intermediate_rows, (
                "Intermediate prefill rows must use host sampling so their "
                "device RNG state is not advanced."
            )
            tokens = tt_out.reshape(-1) if isinstance(tt_out, torch.Tensor) else tt_out
            # Decode reads each scheduled slot; prefill returns one token per
            # scheduled request, already in row order.
            sampled = tokens[rows_t] if is_decode else tokens[:n]
            sampled = sampled.reshape(n, 1).to(torch.int32)
            logprobs = self._device_logprobs(
                tt_log_probs, model_input, scheduled_rows, sampled, is_decode
            )
            return sampled, logprobs

        # Host sampling over the full slot batch.
        total = self.max_num_reqs
        logits = self._host_logits(tt_out, scheduled_rows, is_decode, total)
        bitmask = model_input.grammar_bitmask[0]
        if bitmask is not None:
            runner.apply_grammar_bitmask(logits, bitmask)
        sampling_metadata = self.build_merged_sampling_metadata(
            scheduled_rows, non_sampling_rows=intermediate_rows
        )
        sampler_output = runner.host_sampler(
            logits=logits, sampling_metadata=sampling_metadata
        )
        sampled = sampler_output.sampled_token_ids.reshape(-1)[rows_t].reshape(n, 1)
        logprobs = self._host_logprobs(sampler_output.logprobs_tensors, scheduled_rows)
        return sampled.to(torch.int32), logprobs

    def _host_logits(
        self, tt_out: Any, scheduled_rows: list[int], is_decode: bool, total: int
    ) -> torch.Tensor:
        """Full ``[total, vocab]`` slot logits for host sampling.

        Decode logits already cover every slot. Prefill logits cover only the
        scheduled requests (row order), so scatter them onto their slot rows;
        the unscheduled / gap rows are sampled harmlessly and dropped.
        """
        logits = tt_out[:, -1, :] if tt_out.dim() == 3 else tt_out
        if is_decode:
            return logits
        full = torch.zeros((total, logits.shape[-1]), dtype=logits.dtype)
        full[torch.as_tensor(scheduled_rows, dtype=torch.long)] = logits[
            : len(scheduled_rows)
        ]
        return full

    def _host_logprobs(
        self, logprobs_tensors: LogprobsTensors | None, scheduled_rows: list[int]
    ) -> LogprobsLists | None:
        if logprobs_tensors is None:
            return None
        rows_t = torch.as_tensor(scheduled_rows, dtype=torch.long)
        return LogprobsTensors(
            logprob_token_ids=logprobs_tensors.logprob_token_ids[rows_t],
            logprobs=logprobs_tensors.logprobs[rows_t],
            selected_token_ranks=logprobs_tensors.selected_token_ranks[rows_t],
        ).tolists()

    def _device_logprobs(
        self,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        sampled: torch.Tensor,
        is_decode: bool,
    ) -> LogprobsLists | None:
        """Build logprobs for device-sampled tokens, mirroring the gather-DP
        device logprobs path but over the scheduled slot rows."""
        n = len(scheduled_rows)
        assert isinstance(model_input.tt_sampling_params.enable_log_probs, torch.Tensor)
        enable = model_input.tt_sampling_params.enable_log_probs
        sel = (
            torch.as_tensor(scheduled_rows, dtype=torch.long)
            if is_decode
            else torch.arange(n, dtype=torch.long)
        )
        if not enable[sel].any():
            return None
        assert tt_log_probs is not None, "model should return logprobs when requested"
        return build_device_logprobs(
            tt_log_probs=tt_log_probs,
            sampled_token_ids=sampled.reshape(n),
            rows=sel,
            max_num_logprobs=model_input.max_num_logprobs[0] or 0,
        ).tolists()
