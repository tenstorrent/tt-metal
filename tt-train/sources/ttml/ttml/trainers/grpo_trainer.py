# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
import csv
import inspect
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

import os
import numpy as np
import torch
import ttml
import ttnn
from safetensors.numpy import save_file
from ttml.common.utils import build_causal_mask, create_optimizer, no_grad, round_up_to_tile

from .callback import TrainerCallback

try:
    import wandb as _wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    _wandb = None


class GRPOCompleter(ABC):
    """Abstract base for model-specific completion engines used in GRPO training.

    Subclass this for each model architecture (Llama, Qwen, etc.).
    """

    @abstractmethod
    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        """Generate completions for a batch of tokenised prompts.

        For N prompts returns N * completions_per_prompt completions.
        """

    @abstractmethod
    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        """Generate completions from string prompts, returning decoded strings.
        For N strs returns N * completions_per_prompt strs.
        """

    @abstractmethod
    def compute_nlog_probs(self, prompts: List[List[int]], completions: List[List[int]]) -> tuple:
        """Compute per-token negative log probabilities for prompt+completion pairs.

        Each prompt[i] and completion[i] are concatenated, and the standard
        next-token-prediction shift is applied (input = seq[:-1],
        target = seq[1:]).  The model runs a forward pass and returns
        cross-entropy at every position.

        Dimension glossary:
            B: Global batch size (number of prompt+completion pairs).
            B_local: Per-device batch size (``B // total_devices``).
                On a single device B_local == B.
            T: ``max(len(prompt[i]) + len(completion[i])) - 1`` across the
                batch — the sequence length after the next-token shift.
            T_padded: ``T`` rounded up to the tile boundary (multiple of 32).

        Args:
            prompts: B lists of token IDs (the original prompts).
            completions: B lists of token IDs (the generated completions).

        Returns:
            nlog_probs: Tensor [B_local, T_padded] — negative log-probability
                of each target token.  Prompt and padding positions contain
                meaningless values; use ``mask`` to ignore them.
            mask: Tensor [B_local, T_padded] — binary mask where 1.0 marks
                completion-token positions and 0.0 marks prompt tokens,
                left-padding, and tile-padding.
        """

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """The tokenizer used by this completion engine."""

    @property
    @abstractmethod
    def model(self) -> Any:
        """The underlying tt model used for forward passes and optimization."""


@dataclass
class RolloutBatch:
    """Describes multiple rollout samples (not a specific count).

    Fields:
        batch_id: Monotonic counter picked by the producer.
        weight_version: Which theta version produced this batch. The trainer
            can use it to detect / down-weight stale samples.
        prompts: B ragged prompt token IDs.
        completions: B ragged completion token IDs (one per prompt in
            single-generation mode; per prompt-completion pair otherwise).
        logprobs: [B, max_completion_length] float32. Per-generated-token
            log pi_old(a_t | s_t). Padding positions are don't-care; the
            trainer masks them out.
    """

    batch_id: int
    weight_version: int
    prompts: List[List[int]]
    completions: List[List[int]]
    logprobs: np.ndarray


class RolloutSampler(ABC):
    """Abstract base for producers of :class:`RolloutBatch`."""

    @abstractmethod
    def generate(self, prompts: List[List[int]]) -> RolloutBatch:
        """Generate completions for a batch of tokenised prompts and return them
        packaged (with per-token log pi_old and producer metadata) as a
        :class:`RolloutBatch`.
        """


# Query-axis tile width for the Llama decode's tile-pad trick. Kept as a
# module-level constant so both the mask helper and the decode loop below
# read the same value.
_TTML_ROLLOUT_TILE_SIZE = 32

# Chunked async d2h readback cadence for stop-token detection during decode.
# Every ``_TTML_ROLLOUT_CHUNK`` decode steps a non-blocking cpu() is issued for
# the just-finished token columns; the check is then synchronized on the next
# chunk boundary so the readback lags compute by ``_TTML_ROLLOUT_CHUNK`` steps.
# Matches the ``CHUNK = 32`` value both existing GRPO completers use.
_TTML_ROLLOUT_CHUNK = 32


def _ttml_async_read_to_host(tensors: List[Any], mesh_device: Any) -> Tuple[List[Any], Any]:
    """Non-blocking d2h reads for ``tensors`` on the default command queue.

    Returns ``(host_tensors, event)``. The caller must
    ``ttnn.event_synchronize(event)`` before consuming ``host_tensors``;
    deallocating any of the source ``tensors`` before then races with the
    in-flight DMA.
    """
    hosts = [t.cpu(blocking=False) for t in tensors]
    done = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
    return hosts, done


class TTMLRolloutSampler(RolloutSampler):
    """Concrete :class:`RolloutSampler` for the ttml Llama and Qwen3 models.

    Runs a KV-cached, right-padded prefill + decode loop and returns a
    :class:`RolloutBatch` with per-token ``log pi_old(a_t | s_t)`` captured on
    device via a fused ``cross_entropy_loss`` gather.

    Deliberately does **not** own device setup or model loading — those stay
    with whatever built the model. The caller passes an already-open
    ``mesh_device`` plus the resolved ``dp_mapper`` / ``dp_composer`` /
    ``num_devices`` / ``seed_axes`` state, and per-model KV-cache dimensions.

    Model kind is passed explicitly as a string (``"llama"`` or ``"qwen3"``)
    rather than detected via ``isinstance``. ``LlamaCompositeKV`` lives under
    ``sources/examples/`` and has multiple class definitions in the tree; a
    class-identity check would fragilely wire core trainer code into an
    example-only type. The caller is the natural place to know which
    architecture it built.
    """

    _SUPPORTED_KINDS = ("llama", "qwen3")

    def __init__(
        self,
        *,
        model_kind: str,
        model: Any,
        tokenizer: Any,
        mesh_device: Any,
        dp_mapper: Any,
        dp_composer: Any,
        num_devices: int,
        seed_axes: Optional[List[int]],
        num_layers: int,
        max_seq_len: int,
        max_tokens_to_complete: int,
        max_completion_length: int,
        temperature: float,
        completions_per_prompt: int = 1,
        pad_token: Optional[int] = None,
        num_kv_groups: Optional[int] = None,
        head_dim: Optional[int] = None,
    ) -> None:
        if model_kind not in self._SUPPORTED_KINDS:
            raise ValueError(
                f"TTMLRolloutSampler: model_kind must be one of {self._SUPPORTED_KINDS}, " f"got {model_kind!r}"
            )
        if model_kind == "llama" and (num_kv_groups is None or head_dim is None):
            raise ValueError(
                "TTMLRolloutSampler: model_kind='llama' requires 'num_kv_groups' and 'head_dim' "
                "for the C++ KvCache constructor"
            )

        self._kind = model_kind
        self._model = model
        self._tokenizer = tokenizer
        self._mesh_device = mesh_device
        self._dp_mapper = dp_mapper
        self._dp_composer = dp_composer
        self._num_devices = num_devices
        self._seed_axes = seed_axes
        self._num_layers = num_layers
        self._max_seq_len = max_seq_len
        self._max_tokens_to_complete = max_tokens_to_complete
        self._max_completion_length = max_completion_length
        self._temperature = temperature
        self._completions_per_prompt = completions_per_prompt

        resolved_pad = pad_token
        if resolved_pad is None:
            resolved_pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if resolved_pad is None:
            raise ValueError("TTMLRolloutSampler: could not resolve pad_token from tokenizer")
        self._pad_token = int(resolved_pad)

        self._num_kv_groups = num_kv_groups
        self._head_dim = head_dim

        self._kv_cache: Any = None
        self._kv_cache_B: int = 0

        self._batch_id_counter: int = 0
        self._weight_version: int = 0

    # --------------------------------------------------------------
    # Producer metadata
    # --------------------------------------------------------------

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def set_weight_version(self, version: int) -> None:
        """Publish a new theta version. Subsequent :meth:`generate` calls stamp
        their :class:`RolloutBatch` with this value.
        """
        self._weight_version = int(version)

    def _next_batch_id(self) -> int:
        current = self._batch_id_counter
        self._batch_id_counter += 1
        return current

    # --------------------------------------------------------------
    # Per-model dispatch: KV cache + forward call signature
    # --------------------------------------------------------------

    def _get_kv_cache(self, B_local: int) -> Any:
        """Return a KV cache sized for the current per-device batch.

        Llama uses the C++ ``ttml.models.KvCache`` directly (matches
        ``LlamaGRPOCompleter._get_kv_cache``); Qwen3 uses the Python
        ``ttml.models.qwen3.kv_cache.KVCache`` wrapper which lazy-inits on
        the first ``update`` call and is what the Qwen3 model expects as
        ``past_key_values=``.
        """
        if self._kind == "llama":
            if self._kv_cache is None or self._kv_cache_B != B_local:
                self._kv_cache = ttml.models.KvCache(
                    self._num_layers,
                    B_local,
                    self._num_kv_groups,
                    self._max_seq_len,
                    self._head_dim,
                )
                self._kv_cache_B = B_local
            self._kv_cache.reset()
            return self._kv_cache

        # Qwen3: fresh instance per generate() so the lazy-init picks up any
        # batch-size change and .clear() at the end returns DRAM cleanly.
        from ttml.models.qwen3.kv_cache import KVCache as Qwen3KVCache

        self._kv_cache = Qwen3KVCache(self._num_layers, self._max_seq_len)
        self._kv_cache_B = B_local
        return self._kv_cache

    def _forward(self, x: Any, mask: Any, kv: Any, new_tokens: int) -> Any:
        """Call the model with the right kwarg for its KV interface."""
        if self._kind == "llama":
            return self._model(x, mask, kv_cache=kv, new_tokens=new_tokens)
        return self._model(x, mask, past_key_values=kv)

    def _cache_position(self, kv: Any) -> int:
        """How many tokens the cache currently holds (per row)."""
        if self._kind == "llama":
            return kv.get_cache_position()
        return kv.get_seq_length()

    def _cache_release(self, kv: Any) -> None:
        """Return per-run cache resources. Llama keeps DRAM allocated across
        calls (``reset`` just rewinds the position); Qwen3 frees the underlying
        C++ cache so the training forward that runs right after generation has
        DRAM headroom (matches ``ttml.models.qwen3.KVCache.clear``).
        """
        if self._kind == "llama":
            kv.reset()
        else:
            kv.clear()

    # --------------------------------------------------------------
    # Mask + tokenization helpers
    # --------------------------------------------------------------

    def _decode_causal_mask_right_padded(
        self,
        cur_pos: int,
        cache_len: int,
        pad_positions: List[Tuple[int, int]],
    ) -> "ttml.autograd.Tensor":
        """Per-row decode mask under right-padded prompts.

        Kind-dispatched shape (matches each model's current decode mask):

          * Llama: ``[B, 1, TILE_SIZE=32, cache_len]``. Row 0 carries the
            causal pattern; rows 1..31 stay zero. This matches
            ``LlamaGRPOCompleter._create_causal_mask`` where only
            ``mask_one_token[:query_len]`` gets filled — the padded query
            rows are left zero and their SDPA outputs are discarded
            downstream by the row-0 slice in :meth:`generate`.
          * Qwen3: ``[B, 1, 1, cache_len]``. Single query row.

        Only the REAL query row (row 0) carries an attention pattern: start
        allow-all on row 0, then zero two regions:

          (1) future decode positions ``(cur_pos+1, cache_len)`` — causal cap,
              same for every row on row 0.
          (2) trailing prompt pads ``[len_b, prefill_end)`` — per row on row 0.

        Everything else on row 0 (real prompt ``[0, len_b)`` and decoded
        tokens ``[prefill_end, cur_pos]``) stays 1.
        """
        B = len(pad_positions)
        padded_q = _TTML_ROLLOUT_TILE_SIZE if self._kind == "llama" else 1

        mask = np.zeros((B, 1, padded_q, cache_len), dtype=np.float32)
        # Row 0: allow positions [0, cur_pos] (causal cap by construction).
        mask[:, 0, 0, : cur_pos + 1] = 1.0
        # Row 0: zero per-row trailing prompt pads.
        for b in range(B):
            len_b, prefill_end = pad_positions[b]
            mask[b, 0, 0, len_b:prefill_end] = 0.0
        # Rows 1..padded_q-1 stay zero (Llama tile-pad; matches today's behavior).

        return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, self._dp_mapper)

    def _tokens_to_tensor(self, tokens_np: np.ndarray, B: int) -> "ttml.autograd.Tensor":
        """Reshape ``tokens_np`` from ``[B, S]`` to ``[B, 1, 1, S]`` and upload
        as ``UINT32 / ROW_MAJOR``.
        """
        return ttml.autograd.Tensor.from_numpy(
            tokens_np.reshape(B, 1, 1, tokens_np.shape[1]).astype(np.uint32),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            self._dp_mapper,
        )

    def _get_stop_ids(self) -> set:
        """Union of the tokenizer's EOS/PAD ids plus any chat-template stop
        tokens present in the vocab (across llama-3 and qwen3 templates).
        """
        tokenizer = self._tokenizer
        stop_ids: set = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        if tokenizer.pad_token_id is not None:
            stop_ids.add(int(tokenizer.pad_token_id))
        for tok in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>", "<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                stop_ids.add(int(tid))
        return stop_ids

    # --------------------------------------------------------------
    # Main entrypoint
    # --------------------------------------------------------------

    def generate(self, prompts: List[List[int]]) -> RolloutBatch:  # noqa: PLR0915
        """Right-padded, split prefill/decode with per-token log pi_old capture.

        Prefill uses the Qwen3-style shard-safe pattern: one ``sample_op`` on
        the full prefill logits, one CE pass for the logprob, both readbacks
        via a composer-reassembled ``gather_to_host`` inner helper, then the
        first completion token per row is picked on host by ``pred_pos[b]``.
        Decode is a fixed-shape per-step loop that stores raw ttnn token and
        nlog columns and reads them back at the end, with chunked async
        readbacks along the way for stop-token detection.
        """
        pad_token = self._pad_token
        temperature = self._temperature
        seed_axes = self._seed_axes
        dp_composer = self._dp_composer
        mesh_device = self._mesh_device

        G = self._completions_per_prompt
        rows: List[List[int]] = [list(p) for p in prompts for _ in range(G)]
        B = len(rows)
        assert B % self._num_devices == 0, f"batch {B} must be divisible by num_devices {self._num_devices}"
        B_local = B // self._num_devices

        lengths = [len(r) for r in rows]
        W = max(lengths)
        # Tile-aligned right-padded prompt window: real at [0, len_b), pad after.
        prefill_width = round_up_to_tile(W)
        cache_advance = W if self._kind == "llama" else prefill_width
        pred_pos = np.asarray([length - 1 for length in lengths], dtype=np.int64)
        pad_positions: List[Tuple[int, int]] = [(int(length), cache_advance) for length in lengths]

        tokens_to_complete = max(0, min(self._max_tokens_to_complete, self._max_seq_len - cache_advance))

        prompts_x: List[List[int]] = [list(r) for r in rows]
        max_len = self._max_completion_length

        # Nothing to generate (prompt already fills the window): return empty
        # completions with the padded logprob buffer.
        if tokens_to_complete <= 0:
            return RolloutBatch(
                batch_id=self._next_batch_id(),
                weight_version=self._weight_version,
                prompts=prompts_x,
                completions=[[] for _ in range(B)],
                logprobs=np.zeros((B, max_len), dtype=np.float32),
            )

        stop_ids = self._get_stop_ids()
        stop_arr = np.fromiter(stop_ids, dtype=np.int32) if stop_ids else np.empty(0, dtype=np.int32)

        # --------------------------------------------------------------
        # Inner helpers: closed over dp_composer + the local B_local so
        # they stay off the class surface. Both are used in prefill AND
        # in the decode loop below.
        # --------------------------------------------------------------

        def nlog_probs(logits_t: "ttml.autograd.Tensor", tokens_t: "ttml.autograd.Tensor") -> "ttml.autograd.Tensor":
            """Fused log_softmax + gather-at-index: -log p(tokens) under logits.

            logits_t: [B_local, 1, S, V] autograd tensor (per-device local shape).
            tokens_t: [B_local, 1, S, 1] autograd tensor from sample_op (UINT32 / ROW_MAJOR).
            Returns [B_local, S] autograd tensor of -log p(tokens_t) at every (row, pos).

            Both dimensions are read from ``tokens_t.shape()`` (per-device local)
            so this works correctly under axis-0 sharding — closing over the
            enclosing ``B`` (which is the host batch count) would fail on any
            multi-device run because each shard holds only
            ``B_local = B // num_devices`` rows and its tensors are sized
            accordingly. Matches the same ``[B_local, ...]`` reshape pattern used
            in ``LlamaGRPOCompleter.compute_nlog_probs``.

            ``cross_entropy_loss`` requires target rank == 2 (see
            ``ttml::ops::cross_entropy_loss`` at
            ``tt-train/sources/ttml/ops/losses.cpp``), so tokens are reshaped
            ``[B_local, 1, S, 1] -> [B_local, S]`` on the way in and the result
            is reshaped ``[B_local, S]`` on the way out.
            """
            shape = tokens_t.shape()
            bl = int(shape[0])
            s = int(shape[2])
            tokens_2d = ttml.ops.reshape.reshape(tokens_t, [bl, s])
            nlog = ttml.ops.loss.cross_entropy_loss(logits_t, tokens_2d, ttml.ops.ReduceType.NONE)
            return ttml.ops.reshape.reshape(nlog, [bl, s])

        def gather_to_host(t: "ttnn.Tensor", shape: Tuple[int, ...], dtype: "torch.dtype") -> np.ndarray:
            """Read a sharded ttnn tensor to host in host row order.

            ``dp_composer`` stitches every device's local axis-0 shard back into
            the full ``[B, ...]`` tensor, so ``arr[b, ...]`` on host pairs with
            host row ``b`` regardless of which device it came from — the
            invariant we need for per-row host-side indexing by ``pred_pos``.
            """
            return ttnn.to_torch(t, mesh_composer=dp_composer).reshape(*shape).to(dtype).numpy()

        kv = self._get_kv_cache(B_local)

        # Per-decode-step column stores (raw ttnn.Tensor). Read back at the end.
        generated_columns: List[Any] = []
        nlog_columns: List[Any] = []
        # Rolling per-chunk token-column list that's fed to the async d2h at
        # every ``_TTML_ROLLOUT_CHUNK`` boundary. Reset each boundary to bound
        # the async readback payload — matches ``LlamaGRPOCompleter`` and
        # ``Qwen3GRPOCompleter``.
        chunk_columns: List[Any] = []

        # Async chunk state (used only for stop detection during decode).
        pending_hosts: List[Any] = []
        pending_event: Any = None
        done = np.zeros(B, dtype=bool)

        self._model.eval()
        try:
            with no_grad():
                # -------- Prefill: shard-safe host-side per-row pick --------
                prompt_np = np.full((B, prefill_width), pad_token, dtype=np.uint32)
                for b in range(B):
                    seq = rows[b]
                    prompt_np[b, : len(seq)] = np.asarray(seq, dtype=np.uint32)

                input_tensor = self._tokens_to_tensor(prompt_np, B)
                prefill_mask = build_causal_mask(prefill_width, device=True)
                logits = self._forward(input_tensor, prefill_mask, kv, new_tokens=cache_advance)

                prefill_seed = int(np.random.randint(low=1, high=int(1e7)))
                sampled_full = ttml.ops.sample.sample_op(logits, temperature, prefill_seed, None, seed_axes)
                nlog_full = nlog_probs(logits, sampled_full)

                sampled_host = gather_to_host(sampled_full.get_value(), (B, prefill_width), torch.int32)
                nlog_host = gather_to_host(nlog_full.get_value(), (B, prefill_width), torch.float32)

                rows_arr = np.arange(B)
                first_tokens = sampled_host[rows_arr, pred_pos].astype(np.int32)
                # `first_completion_nlogs` is still -log p(sampled); host column 0
                # of the logprobs accumulator (sign-flipped at assembly).
                first_nlogs = nlog_host[rows_arr, pred_pos].astype(np.float32)

                for b in range(B):
                    if int(first_tokens[b]) in stop_ids:
                        done[b] = True

                _deallocate_tensors([sampled_full, nlog_full, logits, input_tensor, prefill_mask])
                ttml.autograd.AutoContext.get_instance().reset_graph()

                # -------- Decode: fixed-shape per-step loop --------
                cache_len = self._max_seq_len
                last_input = self._tokens_to_tensor(first_tokens.reshape(B, 1).astype(np.uint32), B)

                for i in range(tokens_to_complete - 1):
                    if done.all():
                        break

                    cur_pos = self._cache_position(kv)
                    decode_mask = self._decode_causal_mask_right_padded(cur_pos, cache_len, pad_positions)

                    # Llama's tile-pad trick: pad the single-token input up to a
                    # full tile so the model + attention kernel keep their query
                    # tile shape. Real token lives at index 0; pads at 1..31 —
                    # matches LlamaGRPOCompleter._completion_batched_impl.
                    if self._kind == "llama":
                        token_raw = ttnn.pad(
                            last_input.get_value(),
                            [(0, 0), (0, 0), (0, 0), (0, _TTML_ROLLOUT_TILE_SIZE - 1)],
                            pad_token,
                        )
                        token_input = ttml.autograd.Tensor(token_raw, False)
                    else:
                        token_input = last_input

                    logits = self._forward(token_input, decode_mask, kv, new_tokens=1)

                    step_seed = int(np.random.randint(low=1, high=int(1e7)))
                    sampled = ttml.ops.sample.sample_op(logits, temperature, step_seed, None, seed_axes)
                    nlog_2d = nlog_probs(logits, sampled)

                    # The real predicted-next-token is at query row 0 for both
                    # kinds (Llama's tile-pad places real at 0, pads at 1..31;
                    # Qwen3 only has one row). Match today's Llama slice pattern
                    # and clone so the stored column is independent of
                    # ``sampled`` / ``nlog_2d`` (which get freed right below).
                    last_token_col = ttnn.clone(ttnn.slice(sampled.get_value(), [0, 0, 0, 0], [B_local, 1, 1, 1]))
                    last_nlog_col = ttnn.clone(ttnn.slice(nlog_2d.get_value(), [0, 0], [B_local, 1]))

                    generated_columns.append(last_token_col)
                    nlog_columns.append(last_nlog_col)
                    chunk_columns.append(last_token_col)

                    # Dealloc per-step intermediates. For Llama, ``token_input``
                    # is a fresh ``ttnn.pad`` output distinct from ``last_input``;
                    # safe to dealloc. For Qwen3, ``token_input is last_input``
                    # and after step 0 that wraps the previous step's still-
                    # in-``generated_columns`` column, so we must NOT dealloc it
                    # on the Qwen3 path (matches ``Qwen3GRPOCompleter``'s
                    # comment at line 393).
                    if self._kind == "llama":
                        _deallocate_tensors([token_input])
                    _deallocate_tensors([decode_mask, logits, sampled, nlog_2d])
                    last_input = ttml.autograd.Tensor(last_token_col, False)

                    # Chunked async stop detection: every CHUNK steps, sync the
                    # previous chunk's async d2h and check for stop tokens;
                    # kick off the next chunk's async d2h. Only tokens need
                    # this — nlog columns are read back synchronously at end.
                    if (i + 1) % _TTML_ROLLOUT_CHUNK == 0:
                        if pending_event is not None:
                            ttnn.event_synchronize(mesh_event=pending_event)
                            chunk_np = np.stack(
                                [h.to_numpy(mesh_composer=dp_composer).reshape(B) for h in pending_hosts],
                                axis=1,
                            )
                            done |= np.isin(chunk_np, stop_arr).any(axis=1)
                            if done.all():
                                break

                        pending_hosts, pending_event = _ttml_async_read_to_host(chunk_columns, mesh_device)
                        chunk_columns = []
        finally:
            self._cache_release(kv)
            ttml.autograd.AutoContext.get_instance().reset_graph()

        # -------- Assemble RolloutBatch on host --------

        def _columns_to_np(cols: List[Any], torch_dtype: "torch.dtype", np_dtype: Any) -> np.ndarray:
            """Sync readback for the full decode-column list (host row order)."""
            if not cols:
                return np.empty((B, 0), dtype=np_dtype)
            arr = np.empty((B, len(cols)), dtype=np_dtype)
            for j, col in enumerate(cols):
                arr[:, j] = ttnn.to_torch(col, mesh_composer=dp_composer).reshape(B).to(torch_dtype).numpy()
            return arr

        decode_tokens_np = _columns_to_np(generated_columns, torch.int32, np.int32)
        decode_nlog_np = _columns_to_np(nlog_columns, torch.float32, np.float32)

        # Free device columns now that we have host copies.
        _deallocate_tensors(generated_columns + nlog_columns)

        # For each row: prepend prefill's first token / first -log p, trim at
        # first stop token, negate nlog to obtain log_pi_old, right-pad with
        # zeros into a fixed-width [B, max_completion_length] float32 buffer.
        completions: List[List[int]] = []
        logprobs_padded = np.zeros((B, max_len), dtype=np.float32)
        for b in range(B):
            row_tokens = [int(first_tokens[b])] + [int(t) for t in decode_tokens_np[b]]
            row_nlogs = np.concatenate(([first_nlogs[b].astype(np.float32)], decode_nlog_np[b].astype(np.float32)))
            cut = len(row_tokens)
            for j, tok in enumerate(row_tokens):
                if tok in stop_ids:
                    cut = j
                    break
            trimmed_tokens = row_tokens[:cut]
            trimmed_nlogs = row_nlogs[:cut]
            completions.append(trimmed_tokens)
            # Negate: cross_entropy_loss emits -log p; RolloutBatch.logprobs is
            # documented as log pi_old(a_t | s_t). Skipping this sign flip would
            # invert every downstream ratio exp(log_pi_new - log_pi_old).
            fit = min(len(trimmed_nlogs), max_len)
            if fit > 0:
                logprobs_padded[b, :fit] = -trimmed_nlogs[:fit]

        return RolloutBatch(
            batch_id=self._next_batch_id(),
            weight_version=self._weight_version,
            prompts=prompts_x,
            completions=completions,
            logprobs=logprobs_padded,
        )


@dataclass
class GRPOConfig:
    epsilon: float
    # Number of completions resident on a single device within one micro-batch.
    # The across-mesh micro-batch size is per_device_train_batch_size *
    # num_devices, and the per micro-batch prompt count is derived from it (see
    # GRPOTrainer.train).
    per_device_train_batch_size: int
    num_iterations: int
    # Number of micro-batches per generation (effective) batch and per optimizer
    # step. The generation batch generates gradient_accumulation_steps *
    # per_device_train_batch_size * num_devices completions, then the trainer
    # accumulates gradients over micro-batches of size per_device_train_batch_size * num_devices
    # before each optimizer step. Larger values mean a larger effective batch per step.
    gradient_accumulation_steps: int
    # Metrics are accumulated every step and emitted every ``logging_steps``
    # steps as an interval-mean (except ``min_completion_len``,
    # ``max_completion_len``, ``lr`` where minimum, maximum, last value is
    # emitted respectively).
    logging_steps: int
    output_dir: str
    checkpointing: bool
    checkpoint_interval: int
    prompts_to_train: int
    temperature: float
    max_completion_length: int
    num_generations: int
    warmup_steps: int
    # LR schedule shape AFTER warmup. Names match HuggingFace transformers / TRL
    # so users familiar with those configs can map yamls directly:
    #   "constant" -> flat at base_lr; ``min_lr_rate`` is ignored.
    #   "cosine"   -> half-cosine anneal from base_lr down to
    #                 base_lr * ``min_lr_rate`` over the remaining steps.
    # Default preserves today's behavior (linear warmup then flat).
    lr_scheduler_type: str = "constant"
    min_lr_rate: float = 0.0
    # Gradient clipping. Off by default (matches today).
    use_clip_grad_norm: bool = False
    clip_grad_norm_max_norm: float = 1.0
    log_completions: bool = False
    num_completions_to_print: int = 0
    report_to: str = "none"
    # Optional wandb run name, mirrored from TRL's ``TrainingArguments.run_name``.
    # ``None`` lets wandb auto-generate a name. Project / entity / mode
    # come from the ``WANDB_PROJECT`` / ``WANDB_ENTITY`` / ``WANDB_MODE`` env
    # vars, matching TRL + transformers conventions.
    run_name: Optional[str] = None
    # Escape hatch: when True the trainer does NOT auto-append a GRPOMonitor
    # callback (users can still add their own). Kept off by default so the
    # default experience prints step metrics + writes ``grpo_metrics.csv``.
    disable_default_monitor: bool = False
    # Deprecated/unused: the number of prompts per generation batch is now
    # derived at runtime from per_device_train_batch_size, num_devices, and
    # num_generations. Kept only so older configs that still set it construct
    # without error; the trainer ignores any value provided here.
    batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        # num_generations is the GRPO group size: each prompt must produce at least
        # one completion to form a group (advantages are computed within a group,
        # and the loss normalizes by the completion count). A value <= 0 yields
        # empty batches and divide-by-zero downstream, so fail fast here.
        if self.num_generations <= 0:
            raise ValueError(
                f"grpo_config: 'num_generations' must be > 0 (got {self.num_generations}); "
                "GRPO needs at least one completion per prompt."
            )

        # Other count fields that must be strictly positive: a value <= 0 produces
        # empty batches / divide-by-zero in batch sizing, loss normalization, or
        # the dataset loop. Fail fast at config-construction time.
        for _name, _val in (
            ("per_device_train_batch_size", self.per_device_train_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("prompts_to_train", self.prompts_to_train),
        ):
            if _val <= 0:
                raise ValueError(f"grpo_config: '{_name}' must be > 0 (got {_val}).")

        # checkpoint_interval is only consulted when checkpointing is enabled, where
        # it drives ``num_steps % checkpoint_interval`` -- a value <= 0 would be a
        # modulo-by-zero. (When checkpointing is off the value is unused, so don't
        # constrain it.)
        if self.checkpointing and self.checkpoint_interval <= 0:
            raise ValueError(
                f"grpo_config: 'checkpoint_interval' must be > 0 when checkpointing is enabled "
                f"(got {self.checkpoint_interval})."
            )

        # LR scheduler name must be one of the two shapes _build_lr_factor_fn
        # knows about.
        _allowed_lr_scheduler = {"constant", "cosine"}
        if self.lr_scheduler_type not in _allowed_lr_scheduler:
            raise ValueError(
                f"grpo_config: 'lr_scheduler_type' must be one of {sorted(_allowed_lr_scheduler)} "
                f"(got {self.lr_scheduler_type!r})."
            )
        # ``min_lr_rate`` is a fraction of base_lr (TRL semantics). Only
        # consulted by the cosine schedule; still validate the range so a
        # misconfigured yaml fails at construction rather than silently.
        if not (0.0 <= self.min_lr_rate <= 1.0):
            raise ValueError(f"grpo_config: 'min_lr_rate' must be in [0.0, 1.0] (got {self.min_lr_rate}).")
        # Grad clip norm is only consulted when clipping is enabled; guard the
        # positive-only invariant so a misconfigured yaml fails loudly.
        if self.use_clip_grad_norm and self.clip_grad_norm_max_norm <= 0.0:
            raise ValueError(
                f"grpo_config: 'clip_grad_norm_max_norm' must be > 0 when 'use_clip_grad_norm' is True "
                f"(got {self.clip_grad_norm_max_norm})."
            )

        # ``report_to`` is intentionally a plain string in this framework
        # (unlike TRL, which accepts a list). Reject lists/tuples early so a
        # copy-pasted TRL config surfaces the difference immediately.
        if not isinstance(self.report_to, str):
            raise TypeError(
                f"grpo_config: 'report_to' must be a str, got {type(self.report_to).__name__}. "
                "Supported values: 'none', 'wandb'."
            )
        _allowed_report_to = {"none", "wandb"}
        if self.report_to not in _allowed_report_to:
            raise ValueError(
                f"grpo_config: 'report_to' must be one of {sorted(_allowed_report_to)} " f"(got {self.report_to!r})."
            )

        if not isinstance(self.num_completions_to_print, int) or isinstance(self.num_completions_to_print, bool):
            raise TypeError(
                "grpo_config: 'num_completions_to_print' must be an int, "
                f"got {type(self.num_completions_to_print).__name__}."
            )
        if self.num_completions_to_print < 0:
            raise ValueError(
                f"grpo_config: 'num_completions_to_print' must be >= 0 (got {self.num_completions_to_print})."
            )
        if self.log_completions and self.num_completions_to_print == 0:
            logging.warning(
                "grpo_config: 'log_completions' is True but 'num_completions_to_print' is 0; "
                "no completions will be logged. Set 'num_completions_to_print' > 0 to enable."
            )

        # Warn (once per construction) when a deprecated field is explicitly set.
        # TODO: remove this field and warning once all configs have migrated.
        if self.batch_size is not None:
            logging.warning(
                "grpo_config: 'batch_size' is deprecated and ignored; the generation batch "
                "size is now derived from per_device_train_batch_size, num_devices, "
                "num_generations, and gradient_accumulation_steps. Remove it from your config."
            )


def get_grpo_config(yaml_config: dict, output_dir: str = "") -> GRPOConfig:
    """Build a :class:`GRPOConfig` from a top-level YAML config dict.

    Looks for ``training_config.grpo_config`` in ``yaml_config`` and constructs
    a :class:`GRPOConfig` from it. ``output_dir`` defaults to an empty string so
    callers can fill it in once they have picked a run directory.
    """
    tc = yaml_config.get("training_config", {})
    grpo_section = tc.get("grpo_config")
    if grpo_section is None:
        raise ValueError("training_config must contain a 'grpo_config' section")
    fields = dict(grpo_section)
    fields.setdefault("output_dir", output_dir)

    # Backwards-compatibility shim for the transition period.
    # ``micro_batch_size`` was renamed to ``per_device_train_batch_size``. Accept
    # the old name so existing configs keep working, mapping its value onto the
    # new field. TODO: deprecated — remove this shim (and the warning) once all
    # configs have migrated to ``per_device_train_batch_size``.
    if "micro_batch_size" in fields:
        old_value = fields.pop("micro_batch_size")
        if "per_device_train_batch_size" in fields and fields["per_device_train_batch_size"] != old_value:
            raise ValueError(
                "grpo_config: both 'micro_batch_size' (deprecated) and 'per_device_train_batch_size' are set with different values; "
                "remove 'micro_batch_size' and keep only 'per_device_train_batch_size'."
            )
        logging.warning(
            "grpo_config: 'micro_batch_size' is deprecated and will be removed; "
            "use 'per_device_train_batch_size' instead."
        )
        fields.setdefault("per_device_train_batch_size", old_value)

    return GRPOConfig(**fields)


# Fixed base column order in the CSV. Callback-timing columns are appended after
# these; any extra scalar keys populated on ``trainer.metrics`` follow. The
# header line is materialised together with the first row on the first logging
# step (see :meth:`GRPOMonitor._write_csv_header`).
_CSV_BASE_COLUMNS = (
    "step",
    "reward_mean",
    "reward_std",
    "mean_completion_len",
    "min_completion_len",
    "max_completion_len",
    "lr",
    "step_time_s",
)


# Metric keys that carry per-step sample payloads (not scalars) and should
# never be forwarded to the CSV or to wandb as scalars. Kept as a module-level
# constant so both the CSV writer and the wandb sink agree.
_NON_CSV_KEYS = frozenset({"completions", "prompts", "rewards"})


class _MetricStats:
    """Constant-memory running stats for a single metric key."""

    __slots__ = ("sum", "count", "min", "max", "last")

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.min = math.inf
        self.max = -math.inf
        self.last = float("nan")

    def push(self, value: float) -> None:
        self.sum += value
        self.count += 1
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        self.last = value


# Per-key aggregation policy for interval emission. Anything not listed defaults
# to ``"mean"`` (matching TRL, which averages every metric it logs).
_AGG_POLICY: dict[str, str] = {
    "min_completion_len": "min",
    "max_completion_len": "max",
    "lr": "last",
}


def _aggregate(key: str, r: _MetricStats) -> float:
    policy = _AGG_POLICY.get(key, "mean")
    if policy == "min":
        return r.min
    if policy == "max":
        return r.max
    if policy == "last":
        return r.last
    return r.sum / max(r.count, 1)


class GRPOMonitor(TrainerCallback):
    """CSV + console + optional wandb logger for GRPO training.

    The trainer auto-appends a :class:`GRPOMonitor` to its callback list unless
    the config sets ``disable_default_monitor=True``. It writes a
    ``grpo_metrics.csv`` under the config's ``output_dir``, prints a per-step
    log line, and — when ``report_to == "wandb"`` — calls ``wandb.init(...)``
    itself (matching the TRL / ``transformers`` convention) and forwards scalar
    metrics + a small completions table.

    Args:
        config: The :class:`GRPOConfig` in use. ``output_dir`` (where the CSV
            lands), ``report_to`` (``"none"`` or ``"wandb"``), ``run_name``
            (wandb run name), ``log_completions`` and
            ``num_completions_to_print`` are read from it. The trainer wires
            this up automatically.
    """

    def __init__(self, config: GRPOConfig) -> None:
        self._config = config
        self._output_dir = config.output_dir
        self._report_to = config.report_to
        self._run_name = config.run_name
        self._log_completions = config.log_completions
        self._num_completions_to_print = config.num_completions_to_print

        self._csv_path = os.path.join(self._output_dir, "grpo_metrics.csv") if self._output_dir else None
        # Frozen list of CSV columns; populated on the first ``_write_csv_row``
        # call so that the header reflects the actual contents of ``trainer.metrics``
        # at the first logging step (including keys populated by user callbacks
        # in their ``on_step_end``). Empty means "header not written yet".
        self._columns: list[str] = []
        # Snapshot of the callback classes present at ``on_train_begin`` — used
        # to seed the ``<Callback>_time_s`` columns in the header. Captured then
        # rather than at write time so mutations to ``trainer.callbacks`` after
        # training starts do not silently churn the CSV schema.
        self._callback_time_columns: list[str] = []
        # One-time-warning flags to avoid spamming logs on every step.
        self._warned_missing_wandb = False
        self._warned_unknown_columns: set[str] = set()

        self._wandb_active = False
        # Whether this callback owns the wandb run (i.e. it called
        # ``wandb.init`` itself). If a caller already started a run before the
        # trainer was constructed, we log into it but don't finish it in
        # ``on_train_end`` — that stays the caller's responsibility.
        self._wandb_owned = False

        # Per-key running stats accumulated every step; flushed to CSV / wandb
        # (as interval-mean, or the policy in ``_AGG_POLICY``) every
        # ``logging_steps`` steps.
        self._running: dict[str, _MetricStats] = defaultdict(_MetricStats)

    # -- lifecycle -----------------------------------------------------------

    def on_train_begin(self, trainer: Any) -> None:
        # Skip GRPOMonitor: its cost is deliberately outside step_time_s and it
        # never writes a ``{Callback}_time_s`` entry, so a column would always be empty.
        self._callback_time_columns = [
            f"{type(cb).__name__}_time_s" for cb in trainer.callbacks if not isinstance(cb, GRPOMonitor)
        ]

        if self._csv_path is not None:
            os.makedirs(self._output_dir, exist_ok=True)

        if self._report_to == "wandb":
            self._start_wandb(trainer)

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        metrics: dict[str, Any] = getattr(trainer, "metrics", None) or dict(kwargs)
        self._accumulate(metrics)

        if not self._is_logging_step(step):
            return

        row: dict[str, Any] = {k: _aggregate(k, r) for k, r in self._running.items()}
        for key in _NON_CSV_KEYS:
            if key in metrics:
                row[key] = metrics[key]

        self._log_console(step, row)
        self._write_csv_row(step, row)
        self._maybe_log_completions(step, row)
        self._maybe_log_wandb(step, row)
        self._running.clear()

    def _accumulate(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if key in _NON_CSV_KEYS:
                continue
            # ``step`` is the row identifier, always passed positionally to
            # this hook and consumed by _log_console / _write_csv_row / wandb
            # directly. Skip it here so it does not get cast to float by the
            # aggregator (which would make the console log show ``step: 4.0``
            # and wandb receive a float ``grpo/step``).
            if key == "step":
                continue
            if isinstance(value, (bool, int, float)) and not (isinstance(value, float) and math.isnan(value)):
                self._running[key].push(float(value))

    def _is_logging_step(self, step: int) -> bool:
        ls = self._config.logging_steps
        return ls > 0 and step % ls == 0

    def on_train_end(self, trainer: Any) -> None:
        logging.info("Training complete.")
        if self._wandb_active and self._wandb_owned:
            _wandb.finish()

    # -- helpers -------------------------------------------------------------

    def _start_wandb(self, trainer: Any) -> None:
        if _wandb is None:
            logging.warning(
                "GRPOMonitor: report_to='wandb' but the 'wandb' package is not installed; "
                "falling back to console + CSV logging only."
            )
            self._warned_missing_wandb = True
            return

        if getattr(_wandb, "run", None) is not None:
            # A run was already opened by the caller — log into it but don't
            # own its lifecycle. This preserves the escape hatch for users who
            # want full control over ``wandb.init`` (custom tags, groups, etc.).
            self._wandb_active = True
            self._wandb_owned = False
            logging.info("GRPOMonitor: reusing existing wandb run (%s).", _wandb.run.name)
            return

        # Project / entity / mode follow the TRL + transformers convention of
        # coming from ``WANDB_*`` env vars; only ``run_name`` is a config field.
        _wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name=self._run_name,
            entity=os.environ.get("WANDB_ENTITY"),
            mode=os.environ.get("WANDB_MODE"),
            config=_build_wandb_config(trainer),
        )
        self._wandb_active = True
        self._wandb_owned = True
        logging.info(
            "GRPOMonitor: wandb.init() called (project=%s, run=%s).",
            os.environ.get("WANDB_PROJECT") or "<default>",
            self._run_name or "<auto>",
        )

    def _log_console(self, step: int, metrics: dict[str, Any]) -> None:
        logs = {"step": step, **{k: v for k, v in metrics.items() if k not in _NON_CSV_KEYS}}
        logging.info(logs)

    def _write_csv_row(self, step: int, metrics: dict[str, Any]) -> None:
        if self._csv_path is None:
            return

        if not self._columns:
            self._write_csv_header(metrics)

        row_values = [step if col == "step" else _format_cell(metrics.get(col, float("nan"))) for col in self._columns]
        with open(self._csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(row_values)

        self._warn_unknown_keys(metrics)

    def _write_csv_header(self, metrics: dict[str, Any]) -> None:
        """Derive the CSV column list from the first row's metrics and write the header.

        Called once, on the first logging step. Columns are the base scalars,
        then the ``<Callback>_time_s`` columns captured in ``on_train_begin``,
        then any other numeric keys already populated on ``trainer.metrics``.
        """
        extras = [
            k
            for k in metrics
            if k not in _NON_CSV_KEYS and k not in _CSV_BASE_COLUMNS and k not in self._callback_time_columns
        ]
        self._columns = list(_CSV_BASE_COLUMNS) + self._callback_time_columns + extras
        with open(self._csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(self._columns)

    def _warn_unknown_keys(self, metrics: dict[str, Any]) -> None:
        """Log a one-time warning for any metric key that appeared after the header froze.

        On the first row this is a no-op — ``_write_csv_header`` derives the
        columns from ``metrics`` so every key present is already in
        ``self._columns``.
        """
        for key in metrics:
            if key in _NON_CSV_KEYS or key in self._columns or key in self._warned_unknown_columns:
                continue
            self._warned_unknown_columns.add(key)
            logging.warning(
                "GRPOMonitor: metric %r first appeared after the CSV header was frozen; "
                "it will not be written to %s.",
                key,
                self._csv_path,
            )

    def _maybe_log_completions(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._log_completions or self._num_completions_to_print <= 0:
            return
        completions: Iterable[str] = metrics.get("completions", []) or []
        prompts: Iterable[str] = metrics.get("prompts", []) or []
        rewards: Iterable[float] = metrics.get("rewards", []) or []
        for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards)):
            if i >= self._num_completions_to_print:
                break
            logging.info(
                "[completion %d @ step %d] reward=%.4f\n  prompt=%r\n  completion=%r",
                i,
                step,
                _as_float(reward),
                _preview(prompt),
                _preview(completion),
            )

    def _maybe_log_wandb(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._wandb_active:
            return
        # Only forward numeric scalars to wandb — string / list metrics (like
        # ``completions``) are handled separately as a ``wandb.Table`` below.
        scalar_payload: dict[str, Any] = {}
        for key, value in metrics.items():
            if key in _NON_CSV_KEYS:
                continue
            if isinstance(value, (bool, int, float)) and not (isinstance(value, float) and math.isnan(value)):
                scalar_payload[f"grpo/{key}"] = value
        if scalar_payload:
            _wandb.log(scalar_payload, step=step)

        if self._log_completions and self._num_completions_to_print > 0:
            prompts = list(metrics.get("prompts", []) or [])[: self._num_completions_to_print]
            completions = list(metrics.get("completions", []) or [])[: self._num_completions_to_print]
            rewards = list(metrics.get("rewards", []) or [])[: self._num_completions_to_print]
            if prompts and completions:
                table = _wandb.Table(columns=["step", "prompt", "completion", "reward"])
                for prompt, completion, reward in zip(prompts, completions, rewards):
                    table.add_data(step, _preview(prompt), _preview(completion), _as_float(reward))
                _wandb.log({"grpo/completions": table}, step=step)


def _build_wandb_config(trainer: Any) -> dict:
    """Assemble the ``config`` payload logged to wandb at run start.

    Includes the full :class:`GRPOConfig` (via ``asdict``), the model source,
    and the raw optimizer dict — enough to reproduce a run from the wandb
    settings tab.
    """
    payload: dict = {}
    cfg = getattr(trainer, "config", None)
    if cfg is not None:
        try:
            payload.update(asdict(cfg))
        except TypeError:
            pass
    model_source = getattr(trainer, "model_source", None)
    if model_source is not None:
        payload["model_source"] = model_source
    optimizer_dict = getattr(trainer, "optimizer_dict", None)
    if optimizer_dict:
        payload["optimizer"] = dict(optimizer_dict)
    return payload


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_cell(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _preview(text: Any, max_len: int = 300) -> str:
    s = str(text).replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"


def _deallocate_tensors(tensors: Any) -> None:
    if tensors is None:
        return
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, ttml.autograd.Tensor):
            ttnn.deallocate(t.get_value(), force=True)
        elif isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t, force=True)


def dispatch_reward(
    reward_func: Callable[..., List[float]],
    completions: List[str],
    prompts: List[str],
    batch_columns: dict,
) -> List[float]:
    sig = inspect.signature(reward_func)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    data_pool = {"completions": completions, "prompts": prompts, **batch_columns}

    if has_kwargs:
        return reward_func(**data_pool)

    call_kwargs = {name: data_pool[name] for name in params if name in data_pool}
    return reward_func(**call_kwargs)


def _derive_reward_names(funcs: List[Callable]) -> List[str]:
    """Derive a unique, human-readable name per reward function.

    Uses ``fn.__name__`` when available; falls back to ``reward_{i}`` for
    lambdas and other unnamed callables. Duplicate names are disambiguated
    with a ``_2`` / ``_3`` suffix so per-component metric keys stay unique.
    """
    names: List[str] = []
    seen: dict[str, int] = {}
    for i, fn in enumerate(funcs):
        raw = getattr(fn, "__name__", None) or f"reward_{i}"
        if raw == "<lambda>":
            raw = f"reward_{i}"
        if raw in seen:
            seen[raw] += 1
            names.append(f"{raw}_{seen[raw]}")
        else:
            seen[raw] = 1
            names.append(raw)
    return names


def compute_advantages_host(rewards_np: np.ndarray, group_size: int) -> np.ndarray:
    """Compute group-relative advantages on the host, kept in host order.

    ``rewards_np`` has shape ``[B]`` with ``B = num_groups * group_size`` and
    contiguous groups of length ``group_size`` (all completions of prompt 0,
    then prompt 1, ...). Returns an array of the same shape and order where
    each element has had its group (per-prompt) mean subtracted.

    Doing the group reduction on the host means the advantages never need to be
    co-located by group on a device, so groups are free to straddle devices.
    The advantages are deliberately returned in host order (NOT regrouped per
    device) so that each micro-batch slice can later be sharded along axis 0 in
    the exact same group-agnostic, host-order way that
    :meth:`GRPOCompleter.compute_nlog_probs` shards its token tensors. That
    alignment is what keeps every completion paired with its own advantage on
    every device; see :func:`upload_micro_advantages`.
    """
    B = rewards_np.shape[0]
    assert B % group_size == 0, "rewards length must be divisible by group_size"
    grouped = rewards_np.reshape(-1, group_size).astype(np.float32)
    advantages = grouped - grouped.mean(axis=1, keepdims=True)
    return advantages.reshape(B)


def upload_micro_advantages(adv_np: np.ndarray, mapper: Any, num_devices: int) -> Any:
    """Upload one micro-batch's advantages, sharded to match ``compute_nlog_probs``.

    ``adv_np`` is the host-order advantage slice for a single micro-batch (shape
    ``[mb]``, where ``mb`` is the micro-batch size). It is sharded along axis 0
    across the mesh, so device ``d`` receives host rows
    ``[d * mb_local : (d + 1) * mb_local]`` — the SAME contiguous,
    group-agnostic split that :meth:`GRPOCompleter.compute_nlog_probs` applies
    to its ``[mb, T]`` token tensors for the very same micro-batch. Because both
    tensors are sharded the same way over the same host-order list, device-local
    row ``r`` of the advantages corresponds to device-local row ``r`` of the
    log-probs, i.e. the same completion.

    Returns a ``ttnn.Tensor`` of global shape ``[mb, 1]`` (per device
    ``[mb_local, 1]``), ready to broadcast-multiply the per-completion loss.
    """
    mb = adv_np.shape[0]
    assert mb % num_devices == 0, f"micro-batch size ({mb}) must be divisible by num_devices ({num_devices})"
    mb_local = mb // num_devices
    adv_4d = adv_np.reshape(mb, 1, 1, 1).astype(np.float32)
    adv_ttml = ttml.autograd.Tensor.from_numpy(adv_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    adv_rm = ttnn.to_layout(adv_ttml.get_value(), ttnn.Layout.ROW_MAJOR)
    return ttnn.reshape(adv_rm, [mb_local, 1])


def iter_micro_batch(
    prompts: List[List[int]],
    completions: List[List[int]],
    micro_batch_size: int = 16,
) -> Iterator[Tuple[List[List[int]], List[List[int]]]]:
    for start in range(0, len(completions), micro_batch_size):
        end = min(start + micro_batch_size, len(completions))

        yield prompts[start:end], completions[start:end]


def save_checkpoint(
    model: Any,
    step: int,
    output_dir: str,
    dp_composer: Any = None,
    tokenizer: Any = None,
    grpo_config: Optional[GRPOConfig] = None,
    optimizer: Any = None,
    model_source: Optional[str] = None,
) -> None:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"grpo_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    tensors = {name: param.to_numpy(ttnn.DataType.FLOAT32, dp_composer) for name, param in model.parameters().items()}
    save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))

    if model_source:
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_source)
            hf_config.save_pretrained(ckpt_dir)
        except Exception as exc:
            logging.warning("Could not save HF config for %s: %s", model_source, exc)

    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    if grpo_config is not None:
        gen_config = {
            "temperature": grpo_config.temperature,
            "max_new_tokens": grpo_config.max_completion_length,
        }
        if tokenizer is not None:
            gen_config["eos_token_id"] = tokenizer.eos_token_id
            gen_config["pad_token_id"] = tokenizer.pad_token_id
        with open(os.path.join(ckpt_dir, "generation_config.json"), "w") as f:
            json.dump(gen_config, f, indent=2)

    trainer_state = {"global_step": step}
    if optimizer is not None:
        trainer_state["learning_rate"] = optimizer.get_lr()
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f, indent=2)

    scheduler_state = {
        "base_lr": optimizer.get_lr() if optimizer else None,
        "warmup_steps": grpo_config.warmup_steps if grpo_config else 0,
        "last_step": step,
    }
    torch.save(scheduler_state, os.path.join(ckpt_dir, "scheduler.pt"))

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    torch.save(rng_state, os.path.join(ckpt_dir, "rng_state.pth"))

    if grpo_config is not None:
        torch.save(asdict(grpo_config), os.path.join(ckpt_dir, "training_args.bin"))

    with open(os.path.join(ckpt_dir, "timestamp.txt"), "w") as f:
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC\n"))


def _build_lr_factor_fn(
    *,
    lr_scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    min_lr_rate: float,
) -> Callable[[int], float]:
    """Return ``lr_lambda(step) -> factor`` for a ``LambdaScheduler``.

    ``step`` is 1-indexed (``LambdaScheduler`` bumps its counter to 1 before
    calling us for the first time). The returned factor multiplies base_lr.

    Combos:
      * type="constant", warmup_steps=0  -> flat at base_lr for the whole run.
      * type="constant", warmup_steps>0  -> linear warmup, then flat at base_lr.
                                             ``min_lr_rate`` is ignored (matches TRL).
      * type="cosine",   warmup_steps=0  -> half-cosine anneal base_lr -> base_lr*min_lr_rate.
      * type="cosine",   warmup_steps>0  -> linear warmup, then half-cosine anneal.

    Cosine formula matches HuggingFace ``get_cosine_with_min_lr_schedule_with_warmup``
    with ``num_cycles=0.5`` (TRL's default), simplified from
    ``0.5*(1+cos(pi*progress)) * (1 - min_lr_rate) + min_lr_rate`` to
    ``min_lr_rate + 0.5*(1 - min_lr_rate) * (1 + cos(pi*progress))``.

    Edge case: ``warmup_steps > total_steps`` -> the ramp caps at
    ``total_steps / warmup_steps < 1.0``; the schedule never reaches base_lr.
    Matches today's silent behavior in the inline warmup this replaces.
    """
    if lr_scheduler_type not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_scheduler_type: {lr_scheduler_type!r}; expected 'constant' or 'cosine'")

    warmup = max(0, warmup_steps)
    decay_steps = max(1, total_steps - warmup)

    def factor(step: int) -> float:
        if warmup > 0 and step <= warmup:
            return step / warmup
        if lr_scheduler_type == "constant":
            return 1.0
        progress = min((step - warmup) / decay_steps, 1.0)
        return min_lr_rate + 0.5 * (1.0 - min_lr_rate) * (1.0 + math.cos(math.pi * progress))

    return factor


class GRPOTrainer:
    def __init__(
        self,
        completer: GRPOCompleter,
        dataset: Any,
        config: GRPOConfig,
        reward_func: Optional[Callable[..., List[float]]] = None,
        optimizer_dict: Optional[dict] = None,
        callbacks: Optional[List[Any]] = None,
        model_source: Optional[str] = None,
        reward_funcs: Optional[List[Callable[..., List[float]]]] = None,
    ) -> None:
        if optimizer_dict is None:
            raise ValueError("GRPOTrainer: 'optimizer_dict' is required.")

        self._init_rewards(reward_func, reward_funcs)

        # Constructor inputs (immutable during ``train``).
        self.completer = completer
        self.dataset = dataset
        self.config = config
        self.optimizer_dict = optimizer_dict
        self.callbacks: List[Any] = list(callbacks or [])
        self.model_source = model_source

        # Model handle — bound in ``_setup`` from ``completer.model``.
        self.model: Any = None

        # Per-step accumulator rebuilt every optimizer step by
        # ``_reset_step_metrics``. Callbacks can inject additional keys here
        # (e.g. an eval callback writing ``trainer.metrics["eval_similarity"]``)
        # and later callbacks (notably ``GRPOMonitor``) read the merged view.
        self.metrics: dict = {}

        # Transient timing state used to bracket ``step_time_s`` — only touched
        # by ``_reset_step_metrics`` (write) and ``_publish_step_metrics`` (read).
        self._step_start_time: float = 0.0

        # Resolved-at-setup state. Pre-declared with ``None`` sentinels so the
        # full trainer lifecycle is visible in one place; populated by
        # ``_setup()`` on the first ``train()`` call.
        self._tokenizer: Any = None
        self._optimizer: Any = None
        self._lr_scheduler: Any = None
        self._autograd_ctx: Any = None
        self._mesh: Any = None
        self._num_devices: int = 1
        self._fsdp_enabled: bool = False
        self._ddp_enabled: bool = False
        self._ddp_context_enabled: bool = False
        self._fsdp_sync_axes: Tuple[str, ...] = ()
        self._dp_mapper: Any = None
        self._dp_composer: Any = None
        self._grad_sync_world_size: int = 1
        self._completions_per_microbatch: int = 0
        self._prompts_per_microbatch: int = 0
        self._generation_batch_prompts: int = 0
        self._prompts: List[List[int]] = []
        self._extra_dataset_columns: dict[str, list] = {}
        self._total_optimizer_steps: int = 0

        # Auto-append the framework's default GRPOMonitor unless the config
        # opts out. Placed last so any user-supplied callbacks (e.g. eval)
        # get a chance to populate ``self.metrics`` before the monitor writes
        # a row.
        if not self.config.disable_default_monitor:
            # Detect the framework GRPOMonitor (or a subclass) by isinstance, and
            # a legacy local class also named ``GRPOMonitor`` by class name — pre-
            # refactor forks used to define their own monitor with that name.
            # Either way we skip auto-appending to avoid duplicate CSV writes.
            if not any(isinstance(cb, GRPOMonitor) or type(cb).__name__ == "GRPOMonitor" for cb in self.callbacks):
                self.callbacks.append(GRPOMonitor(self.config))

    def _init_rewards(
        self,
        reward_func: Optional[Callable[..., List[float]]],
        reward_funcs: Optional[List[Callable[..., List[float]]]],
    ) -> None:
        """Validate reward inputs (exactly one of ``reward_func`` /
        ``reward_funcs`` must be provided) and normalize onto a list.

        When ``reward_funcs`` is given, per-completion rewards from each
        function are summed element-wise at train time; per-component means
        are logged separately under ``{fn.__name__}_mean`` keys.
        """
        if (reward_func is None) == (reward_funcs is None):
            raise ValueError(
                "GRPOTrainer: pass exactly one of 'reward_func' (single callable) "
                "or 'reward_funcs' (non-empty list of callables)."
            )
        if reward_funcs is not None and not reward_funcs:
            raise ValueError("GRPOTrainer: 'reward_funcs' must be a non-empty list.")

        self.reward_funcs = list(reward_funcs) if reward_funcs is not None else [reward_func]
        self.reward_func = reward_func
        self._reward_func_names = _derive_reward_names(self.reward_funcs)

    def _time_callback(self, cb: Any, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Fire ``cb.<method_name>(*args, **kwargs)`` and accumulate its wall-clock
        time into ``self.metrics[<ClassName>_time_s]`` for the current step.

        ``GRPOMonitor`` is intentionally excluded from the accumulator: it runs
        last, after ``step_time_s`` has been sealed, and its own cost sits
        outside the step wall time by design.
        """
        cb_t0 = time.perf_counter()
        getattr(cb, method_name)(*args, **kwargs)
        if not isinstance(cb, GRPOMonitor):
            key = f"{type(cb).__name__}_time_s"
            self.metrics[key] = self.metrics.get(key, 0.0) + (time.perf_counter() - cb_t0)

    def _compute_grpo_loss(
        self,
        nlog_probs_old: ttml.autograd.Tensor,
        nlog_probs_new: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        adv_ttml: ttml.autograd.Tensor,
        completions_batch_len: int,
        eps: float,
        ddp_world_size: int = 1,
    ) -> ttml.autograd.Tensor:
        """Compute the clipped GRPO surrogate loss.

        ``completions_batch_len`` is the *global* number of completions in the
        optimizer step. Under DDP, ``ttml.core.distributed.synchronize_gradients``
        *averages* (not sums) the per-device gradients, so normalising the loss
        by the global count would leave an extra ``1 / ddp_world_size`` factor
        after that averaging. To keep gradients invariant to the device count we
        normalise by the *per-device* completion count instead
        (``completions_batch_len / ddp_world_size``); the gradient averaging then
        restores the intended global-mean gradient. ``ddp_world_size`` is 1 when
        DDP is disabled, leaving the single-device path unchanged.
        """
        B_local, Tp = nlog_probs_old.shape()
        ratio = ttml.ops.unary.exp(nlog_probs_old - nlog_probs_new)
        clipped_ratio = ttml.ops.unary.clip(ratio, 1.0 - eps, 1.0 + eps)

        surr1 = ratio * adv_ttml
        surr2 = clipped_ratio * adv_ttml
        surr = ttml.ops.binary.min(surr1, surr2)

        # Per-completion normalised weight: w[i,t] = mask[i,t] / max(sum_t(mask[i,t]), 1)
        mask_val = mask.get_value()
        tokens_per_completion = ttnn.maximum(ttnn.sum(mask_val, dim=1, keepdim=True), 1.0)
        weight_tt = ttml.autograd.create_tensor(ttnn.div(mask_val, tokens_per_completion), requires_grad=False)

        weighted_surr = surr * weight_tt
        weighted_surr_4d = ttml.ops.reshape.reshape(weighted_surr, [1, 1, B_local, Tp])
        per_device_batch_len = completions_batch_len / ddp_world_size
        return ttml.ops.unary.mean(weighted_surr_4d) * (-float(B_local) * float(Tp) / per_device_batch_len)

    def _setup(self) -> None:
        """One-shot training setup: validate config, build optimizer, resolve
        the device-parallelism topology, and tokenize the dataset.

        Populates the ``self._foo`` attributes pre-declared in ``__init__`` (all
        the parallelism-topology and batching state that every phase helper
        below reads) plus the public ``self.model`` handle.
        """
        grpo_cfg = self.config
        completer = self.completer
        tt_model = completer.model
        tokenizer = completer.tokenizer
        self.model = tt_model

        optimizer = create_optimizer(tt_model, self.optimizer_dict)
        base_lr = optimizer.get_lr()

        # Device-parallelism state. The trainer currently only handles either
        # single-device or DDP; tensor parallelism is not supported here. We
        # gate the multi-device sharding paths on ``ddp_enabled`` rather than
        # ``num_devices > 1`` so this assumption is explicit at the call sites.
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        device = autograd_ctx.get_device()
        num_devices: int = device.get_num_devices()
        mesh = ttml.maybe_mesh()
        # ttml has two coexisting distributed backends, and DDP can be signalled
        # through EITHER, so both must count here:
        #   * Parallelism-context DDP — set up with
        #     ``initialize_parallelism_context`` and synced with
        #     ``synchronize_gradients``. This is the general-purpose DDP/TP
        #     mechanism used across ttml (the shared trainer, the non-GRPO qwen3
        #     examples, etc.); the Llama GRPO completer initializes it.
        #   * Named-mesh DDP — the completer opened a named mesh (via
        #     ``ttml.open_device_mesh``) with a "dp" axis of size > 1 and synced
        #     with ``sync_gradients`` over that axis. This is the FSDP-oriented
        #     backend; the Qwen3 GRPO completer takes ONLY this route and never
        #     initializes a parallelism context.
        # Checking the context alone (as the code originally did) leaves
        # ``ddp_enabled`` False on the Qwen3 path — which then trips
        # ``num_devices = 1`` below while the completer still shards the batch
        # across the whole mesh, blowing up with "batch N must be divisible by
        # num_devices". Detecting the "dp" axis mirrors the exact DDP signal the
        # completer uses (``mesh.has_axis("dp")``).
        ddp_context_enabled: bool = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        ddp_enabled: bool = ddp_context_enabled or (
            mesh is not None and mesh.has_axis("dp") and mesh.axis_size("dp") > 1
        )
        # FSDP is configured through a named mesh (axis "fsdp"), opened via
        # ``ttml.open_device_mesh`` by the completer — not the parallelism
        # context that context-based DDP uses. When an "fsdp" axis is present the
        # batch is sliced across the whole mesh (dim 0) exactly like DDP, and
        # gradients are synchronised with ``ttml.sync_gradients`` over the
        # ("dp", "fsdp") axes (FSDP-managed params are skipped per-axis because
        # the FSDP backward hook already reduce-scattered them).
        fsdp_enabled: bool = mesh is not None and mesh.has_axis("fsdp") and mesh.axis_size("fsdp") > 1
        fsdp_sync_axes: Tuple[str, ...] = (
            tuple(name for name in ("dp", "fsdp") if mesh.has_axis(name) and mesh.axis_size(name) > 1)
            if mesh is not None
            else ()
        )
        batch_sharded: bool = ddp_enabled or fsdp_enabled
        dp_mapper: Any = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0) if batch_sharded else None
        dp_composer: Any = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0) if batch_sharded else None
        if not batch_sharded:
            num_devices = 1

        # World size for the loss normalization. ``synchronize_gradients`` /
        # ``sync_gradients`` all-reduce (sum) each replicated gradient and then
        # divide by the product of the sizes of the axes they reduce over,
        # leaving the mean. ``_compute_grpo_loss`` must divide the loss by that
        # SAME factor (see its docstring) so the two cancel to the intended
        # global-mean gradient. Derive it from the exact axes each branch syncs,
        # rather than blanket ``get_num_devices()``, so the normalization stays
        # correct even on a mesh whose reduced axes don't span every device
        # (e.g. a "dp" axis narrower than the full mesh).
        if fsdp_enabled:
            # sync_gradients over fsdp_sync_axes -> divide by their size product.
            grad_sync_world_size = 1
            for _name in fsdp_sync_axes:
                grad_sync_world_size *= mesh.axis_size(_name)
        elif ddp_context_enabled:
            # C++ synchronize_gradients reduces each replicated grad over ONLY
            # the context's DDP axis (and CP, if enabled) — never TP: TP-sharded
            # params hold per-shard-correct grads and TP-replicated params
            # already match across TP ranks. So the divisor is the DDP-axis
            # device count, NOT get_num_devices(). Using the whole-mesh count
            # would over-divide by the TP factor on a DP+TP mesh and normalize
            # the loss incorrectly. Read the exact DDP size from the context.
            #
            # NOTE (CP / forward-looking correctness): the C++ divisor is
            # ddp_size * cp_size (synchronize_gradients pushes BOTH the CP and
            # DDP axes into cluster_axes — see core/distributed/distributed.cpp).
            # get_ddp_size() alone is the *complete* divisor here ONLY because CP
            # cannot currently be turned on from Python: the DistributedConfig
            # nanobind binding exposes just enable_ddp / enable_tp (no enable_cp
            # ctor arg or settable field — see nanobind/nb_autograd.cpp), so
            # is_cp_enabled() is always false on any Python-driven run and no CP
            # axis is ever reduced. If enable_cp is ever bound to Python, this
            # line must become get_ddp_size() * get_cp_size() (and a get_cp_size
            # accessor would need binding), or the loss will silently mis-
            # normalize by the CP factor — the same bug this branch fixes for TP.
            grad_sync_world_size = autograd_ctx.get_parallelism_context().get_ddp_size()
        elif ddp_enabled:
            # sync_gradients over ("dp",) -> divide by the "dp" axis size.
            grad_sync_world_size = mesh.axis_size("dp")
        else:
            grad_sync_world_size = 1

        # Derive the across-mesh micro-batch size (in completions), the per
        # micro-batch prompt count, and the generation (effective) batch size up
        # front, and validate the divisibility relationships so misconfigurations
        # fail with a clear message instead of a cryptic shard assert deep in
        # ``compute_nlog_probs`` (or a silently ragged final micro-batch).
        #
        # ``per_device_train_batch_size`` is the number of completions resident
        # on a single device within one micro-batch, so the whole mesh handles
        # ``completions_per_microbatch = per_device_train_batch_size *
        # num_devices`` completions per micro-batch. The per micro-batch prompt
        # count is ``completions_per_microbatch // num_generations``. By
        # construction ``completions_per_microbatch`` is divisible by
        # ``num_devices``, so each micro-batch always shards evenly along axis 0.
        #
        # The generation (effective) batch spans ``gradient_accumulation_steps``
        # micro-batches: each batch generates ``grad_accum`` times the per
        # micro-batch prompt count, then the trainer runs one forward/backward
        # pass per micro-batch accumulating gradients before a single optimizer
        # step. Increasing ``gradient_accumulation_steps`` therefore generates
        # proportionally more completions per batch and trains over that many
        # micro-batches between optimizer steps.
        if grpo_cfg.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be positive, got {grpo_cfg.per_device_train_batch_size}"
            )
        grad_accum = grpo_cfg.gradient_accumulation_steps
        if grad_accum <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {grad_accum}")
        if grpo_cfg.num_generations <= 0:
            raise ValueError(f"num_generations must be positive, got {grpo_cfg.num_generations}")
        completions_per_microbatch = grpo_cfg.per_device_train_batch_size * num_devices
        if completions_per_microbatch % grpo_cfg.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size * num_devices ({grpo_cfg.per_device_train_batch_size} * "
                f"{num_devices} = {completions_per_microbatch}) must be divisible by "
                f"num_generations ({grpo_cfg.num_generations}) so the per micro-batch prompt count is an integer"
            )
        prompts_per_microbatch = completions_per_microbatch // grpo_cfg.num_generations
        generation_batch_prompts = prompts_per_microbatch * grad_accum

        total_prompts = min(int(grpo_cfg.prompts_to_train), len(self.dataset))
        if total_prompts % generation_batch_prompts != 0:
            raise ValueError(
                f"prompts_to_train ({total_prompts}) must be divisible by the generation batch size "
                f"(prompts_per_microbatch * gradient_accumulation_steps = {prompts_per_microbatch} * "
                f"{grad_accum} = {generation_batch_prompts}) to avoid a ragged final batch that can break "
                "micro-batch sharding"
            )
        dataset = self.dataset.select(range(total_prompts))
        prompts = [tokenizer.encode(row["prompt"]) for row in dataset]
        extra_columns = {k: list(dataset[k]) for k in dataset.column_names if k != "prompt"}

        # Publish outputs onto self for the per-batch helpers.
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self._autograd_ctx = autograd_ctx
        self._mesh = mesh
        self._num_devices = num_devices
        self._fsdp_enabled = fsdp_enabled
        self._ddp_enabled = ddp_enabled
        self._ddp_context_enabled = ddp_context_enabled
        self._fsdp_sync_axes = fsdp_sync_axes
        self._dp_mapper = dp_mapper
        self._dp_composer = dp_composer
        self._grad_sync_world_size = grad_sync_world_size
        self._completions_per_microbatch = completions_per_microbatch
        self._prompts_per_microbatch = prompts_per_microbatch
        self._generation_batch_prompts = generation_batch_prompts
        self._prompts = prompts
        self._extra_dataset_columns = extra_columns
        # Total optimizer steps this run will take. Read by the LR scheduler.
        self._total_optimizer_steps = (total_prompts // generation_batch_prompts) * self.config.num_iterations

        # LR scheduler. One ``LambdaScheduler`` over the closure returned by
        # ``_build_lr_factor_fn`` handles all four (warmup, kind) combos in one
        # branch. Backward compat: with default ``lr_scheduler_type="constant"``
        # and ``warmup_steps > 0`` the emitted LR trace is bit-identical to
        # today's inline ``warmup_factor = min(1, (step+1)/warmup_steps)``.
        from ttml.common.schedulers import LambdaScheduler

        self._lr_scheduler = LambdaScheduler(
            optimizer,
            _build_lr_factor_fn(
                lr_scheduler_type=self.config.lr_scheduler_type,
                warmup_steps=int(self.config.warmup_steps),
                total_steps=self._total_optimizer_steps,
                min_lr_rate=float(self.config.min_lr_rate),
            ),
        )

    # -- phase helpers -------------------------------------------------------
    #
    # Each helper maps to one phase of GRPO training. They take 1-3 raw args,
    # return one primitive or nothing, and write any metrics they produce
    # directly into ``self.metrics``. ``train()`` composes them.

    def _iter_prompt_batches(self) -> Iterator[Tuple[List[List[int]], dict]]:
        """Yield ``(prompts, extra_dataset_columns)`` per generation batch.

        Prompts are UNEXPANDED (one entry per prompt); the completer's
        ``generate`` fans them out to ``num_generations`` completions each.
        """
        gbp = self._generation_batch_prompts
        prompts = self._prompts
        extra_cols = self._extra_dataset_columns
        for start in range(0, len(prompts), gbp):
            end = min(start + gbp, len(prompts))
            yield (
                list(prompts[start:end]),
                {k: list(col[start:end]) for k, col in extra_cols.items()},
            )

    def _rollout(self, prompts: List[List[int]]) -> List[List[int]]:
        """Sync rollout: block on ``completer.generate(prompts)`` and return
        ``N * num_generations`` completions. Writes ``generation_time_s`` to
        ``self.metrics``.
        """
        gen_t0 = time.perf_counter()
        completions = self.completer.generate(prompts)
        self.metrics["generation_time_s"] = time.perf_counter() - gen_t0
        return completions

    def _expand_prompts_and_columns(
        self,
        prompts: List[List[int]],
        extra_dataset_columns: dict,
    ) -> Tuple[List[List[int]], dict]:
        """Replicate each prompt and each dataset-column entry
        ``num_generations`` times so that index ``i`` of the returned lists
        aligns 1:1 with completion ``i``. Called AFTER ``_rollout``.
        """
        g = self.config.num_generations
        prompts_x = [p for p in prompts for _ in range(g)]
        cols_x = {k: [v for v in col for _ in range(g)] for k, col in extra_dataset_columns.items()}
        return prompts_x, cols_x

    def _compute_rewards(
        self,
        prompts_x: List[List[int]],
        completions: List[List[int]],
        extra_cols_x: dict,
    ) -> np.ndarray:
        """Decode strings, dispatch each reward function, sum element-wise.

        Writes the following to ``self.metrics``: ``reward_mean``,
        ``reward_std``, per-function ``{name}_mean`` (only when >1 reward
        function is configured), ``mean/min/max_completion_len``, and — when
        ``log_completions`` is enabled — ``prompts`` / ``completions`` /
        ``rewards`` display slices.
        """
        prompt_strs = [self._tokenizer.decode(p) for p in prompts_x]
        completion_strs = [self._tokenizer.decode(c, skip_special_tokens=True) for c in completions]

        per_func = [
            np.array(dispatch_reward(fn, completion_strs, prompt_strs, extra_cols_x), dtype=np.float32)
            for fn in self.reward_funcs
        ]
        if len(self.reward_funcs) > 1:
            for name, arr in zip(self._reward_func_names, per_func):
                self.metrics[f"{name}_mean"] = float(arr.mean()) if arr.size else 0.0
        rewards_np = np.sum(per_func, axis=0).astype(np.float32)

        self.metrics["reward_mean"] = float(rewards_np.mean())
        self.metrics["reward_std"] = float(rewards_np.std())

        lens = [len(c) for c in completions]
        self.metrics["mean_completion_len"] = (sum(lens) / len(lens)) if lens else 0.0
        self.metrics["min_completion_len"] = min(lens) if lens else 0
        self.metrics["max_completion_len"] = max(lens) if lens else 0

        cfg = self.config
        if cfg.log_completions and cfg.num_completions_to_print > 0:
            k = cfg.num_completions_to_print
            self.metrics["prompts"] = prompt_strs[:k]
            self.metrics["completions"] = completion_strs[:k]
            self.metrics["rewards"] = rewards_np[:k].tolist()

        return rewards_np

    def _compute_advantages(self, rewards_np: np.ndarray) -> np.ndarray:
        """Group-relative advantages on host (per-prompt mean subtracted)."""
        return compute_advantages_host(rewards_np, self.config.num_generations)

    def _optimize(
        self,
        prompts_x: List[List[int]],
        completions: List[List[int]],
        advantages_np: np.ndarray,
    ) -> None:
        """Reference log-probs (once) + per-micro-batch forward-with-grad + GRPO
        loss + backward. Accumulates gradients across the whole generation
        batch; does NOT call ``optimizer.step()`` — that is ``_apply_gradients``.
        """
        ref_logprobs = self._compute_ref_logprobs(prompts_x, completions)
        try:
            self.model.train()
            self._optimizer.zero_grad()
            global_len = len(prompts_x)
            mb = self._completions_per_microbatch
            for i, (p, c, ref_nlog, ref_mask) in enumerate(
                self._iter_micro_batches(prompts_x, completions, ref_logprobs),
            ):
                adv_slice = advantages_np[i * mb : i * mb + len(c)]
                self._compute_loss_and_backward(p, c, adv_slice, ref_nlog, ref_mask, global_len)
        finally:
            for nlog, mask in ref_logprobs:
                _deallocate_tensors([nlog, mask])

    def _compute_ref_logprobs(
        self,
        prompts_x: List[List[int]],
        completions: List[List[int]],
    ) -> List[Tuple[Any, Any]]:
        """Eval-mode + ``no_grad`` forward pass: ratio-denominator log-probs
        for every micro-batch of the current generation batch. Reused across
        every mini-epoch of ``_optimize``; freed on ``_optimize`` exit.
        """
        out: List[Tuple[Any, Any]] = []
        self.model.eval()
        with no_grad():
            for p, c in iter_micro_batch(prompts_x, completions, self._completions_per_microbatch):
                nlog, mask = self.completer.compute_nlog_probs(p, c)
                nlog.set_requires_grad(False)
                mask.set_requires_grad(False)
                out.append((nlog, mask))
        return out

    def _iter_micro_batches(
        self,
        prompts_x: List[List[int]],
        completions: List[List[int]],
        ref_logprobs: List[Tuple[Any, Any]],
    ) -> Iterator[Tuple[List[List[int]], List[List[int]], Any, Any]]:
        """Yield ``(prompts_slice, completions_slice, ref_nlog, ref_mask)`` per
        micro-batch, pairing each token slice with its cached reference
        log-probs (by index into ``ref_logprobs``).
        """
        for i, (p, c) in enumerate(
            iter_micro_batch(prompts_x, completions, self._completions_per_microbatch),
        ):
            nlog, mask = ref_logprobs[i]
            yield p, c, nlog, mask

    def _compute_loss_and_backward(
        self,
        prompts_slice: List[List[int]],
        completions_slice: List[List[int]],
        adv_slice: np.ndarray,
        ref_nlog: Any,
        ref_mask: Any,
        global_len: int,
    ) -> None:
        """One autograd transaction: upload advantages, run the current-policy
        forward pass with grad, build the clipped GRPO surrogate, backward,
        and free the intermediates.

        ``global_len`` is the completion count across the whole generation
        batch (all ``grad_accum`` micro-batches); the loss normalization uses
        it to yield a mean-over-effective-batch gradient after all micro-batch
        contributions accumulate.
        """
        # Advantages and completion token tensors are both sharded along axis
        # 0 over the identical host-order slice, so each device pairs a
        # completion's log-probs with that same completion's advantage.
        adv_slice_val = upload_micro_advantages(adv_slice, self._dp_mapper, self._num_devices)
        adv_ttml = ttml.autograd.create_tensor(adv_slice_val, requires_grad=False)

        nlog_new, mask_new = self.completer.compute_nlog_probs(prompts_slice, completions_slice)
        loss = self._compute_grpo_loss(
            ref_nlog,
            nlog_new,
            ref_mask,
            adv_ttml,
            global_len,
            self.config.epsilon,
            ddp_world_size=self._grad_sync_world_size,
        )
        loss.backward(retain_graph=False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        _deallocate_tensors([nlog_new, mask_new, adv_ttml, loss])

    def _apply_gradients(self) -> None:
        """LR scheduler step + grad sync + optional grad clip +
        ``on_before_optimizer_step`` + ``optimizer.step()`` + ``zero_grad``.
        Writes ``lr`` and per-callback timings to ``self.metrics``.

        Ordering:
          1. ``self._lr_scheduler.step()`` bumps its internal step counter
             and writes the new LR onto the optimizer.
          2. Grad sync (FSDP / parallelism-context DDP / named-mesh DDP).
          3. ``ttml.core.clip_grad_norm`` AFTER grad sync so every rank clips
             the identical reduced grads. Guarded by ``config.use_clip_grad_norm``.
          4. Callback ``on_before_optimizer_step`` fires.
          5. ``optimizer.step()`` + ``zero_grad``.
        """
        self._lr_scheduler.step()

        if self._fsdp_enabled:
            ttml.sync_gradients(self.model.parameters(), axis_names=self._fsdp_sync_axes)
        elif self._ddp_context_enabled:
            # Parallelism-context DDP (Llama completer): a parallelism context
            # is initialized, so use its gradient sync. ``sync_gradients`` would
            # be a silent no-op here — it reduces over named mesh axes, and this
            # path opens no named mesh (``maybe_mesh()`` is None), so it would
            # leave gradients un-averaged.
            ttml.core.distributed.synchronize_gradients(self.model.parameters())
        elif self._ddp_enabled:
            # Named-mesh DDP (Qwen3 completer): no parallelism context exists,
            # so all-reduce + average grads over the "dp" mesh axis — the same
            # primitive FSDP uses. The loss normalization divides by
            # ``grad_sync_world_size`` (= the "dp" axis size here), matched to
            # this reduction.
            ttml.sync_gradients(self.model.parameters(), axis_names=("dp",))

        if self.config.use_clip_grad_norm:
            ttml.core.clip_grad_norm(
                self.model.parameters(),
                self.config.clip_grad_norm_max_norm,
            )

        for cb in self.callbacks:
            self._time_callback(cb, "on_before_optimizer_step", self)

        self._optimizer.step()
        self._optimizer.zero_grad()
        self.metrics["lr"] = self._optimizer.get_lr()

    def _publish_step_metrics(self) -> None:
        """Fire ``on_step_end`` for every non-monitor callback (their timings
        fold into ``self.metrics`` via ``_time_callback``), seal
        ``step_time_s``, and finally fire the monitor callback's
        ``on_step_end`` OUTSIDE the ``step_time_s`` window.

        Reads step from ``self.metrics["step"]`` and passes it positionally to
        callbacks so the ``on_step_end(trainer, step, **metrics)`` signature
        is unchanged.
        """
        step = self.metrics["step"]
        monitor_cb = next((cb for cb in self.callbacks if isinstance(cb, GRPOMonitor)), None)

        # ``step`` lives on ``self.metrics`` but is also passed positionally to
        # match the callback signature ``on_step_end(trainer, step, **metrics)``.
        # Filter it out of the splat so we don't hit "got multiple values for
        # argument 'step'".
        def _step_kwargs() -> dict:
            return {k: v for k, v in self.metrics.items() if k != "step"}

        for cb in self.callbacks:
            if cb is monitor_cb:
                continue
            self._time_callback(cb, "on_step_end", self, step, **_step_kwargs())

        # Seal step_time_s after all non-monitor work is done, so it covers
        # the full per-step wall time (rollout, host post-gen, reference
        # log-probs, training loop, non-monitor callbacks). GRPOMonitor's own
        # cost is deliberately outside this window.
        self.metrics["step_time_s"] = time.perf_counter() - self._step_start_time

        if monitor_cb is not None:
            self._time_callback(monitor_cb, "on_step_end", self, step, **_step_kwargs())

    def _maybe_checkpoint(self) -> None:
        """Save a checkpoint on interval + fire ``on_save`` for non-monitor
        callbacks (their timings fold into ``self.metrics``).
        """
        step = self.metrics["step"]
        cfg = self.config
        if not cfg.checkpointing or step % cfg.checkpoint_interval != 0:
            return
        save_checkpoint(
            self.model,
            step,
            cfg.output_dir,
            dp_composer=self._dp_composer,
            tokenizer=self._tokenizer,
            grpo_config=cfg,
            optimizer=self._optimizer,
            model_source=self.model_source,
        )
        ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"grpo_step_{step}")
        monitor_cb = next((cb for cb in self.callbacks if isinstance(cb, GRPOMonitor)), None)
        for cb in self.callbacks:
            if cb is monitor_cb:
                continue
            self._time_callback(cb, "on_save", self, step, ckpt_dir)

    def _reset_step_metrics(self) -> None:
        """Start-of-step reset: preserve ``step``, reseed NaN reward-component
        keys so GRPOMonitor freezes them into the CSV header, and reseed
        ``self._step_start_time`` (the timer used to bracket ``step_time_s``).
        This is the ONLY method that writes ``self._step_start_time``.
        """
        step = self.metrics.get("step", 0)
        seed = {f"{name}_mean": float("nan") for name in self._reward_func_names} if len(self.reward_funcs) > 1 else {}
        self.metrics = {"step": step, **seed}
        self._step_start_time = time.perf_counter()

    # -- training loop -------------------------------------------------------

    def train(self) -> None:
        """Synchronous GRPO training loop.

        Walks the phase helpers above once per generation batch, then per
        mini-epoch runs ``_optimize`` (ref logprobs + fwd-grad + loss +
        backward), ``_apply_gradients``, and the metrics/checkpoint bookkeeping.
        """
        self._setup()
        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.metrics = {"step": 0}
        self._reset_step_metrics()

        for prompts, extra_cols in self._iter_prompt_batches():
            completions = self._rollout(prompts)
            prompts_x, cols_x = self._expand_prompts_and_columns(prompts, extra_cols)
            rewards_np = self._compute_rewards(prompts_x, completions, cols_x)
            advantages_np = self._compute_advantages(rewards_np)

            for _ in range(self.config.num_iterations):
                self._optimize(prompts_x, completions, advantages_np)
                self._apply_gradients()
                self.metrics["step"] += 1
                self._publish_step_metrics()
                self._maybe_checkpoint()
                self._reset_step_metrics()

        for cb in self.callbacks:
            cb.on_train_end(self)
