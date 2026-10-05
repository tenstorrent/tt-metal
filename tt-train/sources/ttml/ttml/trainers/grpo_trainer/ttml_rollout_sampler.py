# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch
import ttnn

import ttml
from ttml.common.utils import build_causal_mask, no_grad, round_up_to_tile, run_mode
from ttml.modules import RunMode
from ttml.models.llama import Llama
from ttml.models.qwen3 import Qwen3
from ttml.models.qwen3.kv_cache import KVCache as Qwen3KVCache

from .grpo_trainer import RolloutBatch, RolloutSampler

TILE_SIZE = 32

# Chunked async d2h readback cadence for stop-token detection during decode.
CHUNK = 32


def deallocate_tensors(tensors: Any) -> None:
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


def async_read_to_host(tensors: List[Any], mesh_device: Any) -> Tuple[List[Any], Any]:
    """Issue non-blocking d2h reads for ``tensors`` on the single command queue.

    Returns ``(host_tensors, event)``. The caller must call
    ``event_synchronize(event)`` before consuming ``host_tensors``; deallocating
    the source ``tensors`` before then races with the in-flight DMA.
    """
    hosts = [t.cpu(blocking=False) for t in tensors]
    done = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
    return hosts, done


class TTMLRolloutSampler(RolloutSampler):
    """Concrete :class:`RolloutSampler` for the ttml Llama and Qwen3 models.

    Generates with a model and tokenizer it is given (built by
    :func:`~ttml.trainers.grpo_trainer.grpo_ttml_model.setup_ttml_model` and
    owned by the trainer) on the already-open device. Runs a KV-cached,
    right-padded prefill + decode loop that returns a :class:`RolloutBatch` with
    per-token ``log pi_old(a_t | s_t)`` captured on device via a fused
    ``cross_entropy_loss`` gather.

    Args:
        model: A ttml ``Llama`` (e.g. ``LlamaCompositeKV``) or ``Qwen3``.
        tokenizer: The HuggingFace tokenizer matching ``model``.
        max_completion_length: Maximum generated tokens per completion, and
            the width of ``RolloutBatch.logprobs``.
        temperature: Sampling temperature (0 = greedy).
        completions_per_prompt: Completions generated per prompt.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_completion_length: int,
        temperature: float,
        completions_per_prompt: int,
    ) -> None:
        if isinstance(model, Qwen3):
            self._kind = "qwen3"
        elif isinstance(model, Llama):
            self._kind = "llama"
        else:
            raise TypeError(f"TTMLRolloutSampler supports ttml Llama and Qwen3 models, got {type(model).__name__}")

        self._model = model
        self._tokenizer = tokenizer
        self._max_completion_length = max_completion_length
        self._temperature = temperature
        self._completions_per_prompt = completions_per_prompt

        self._mesh_device: Any = ttml.autograd.AutoContext.get_instance().get_device()
        self._num_devices = self._mesh_device.get_num_devices()
        if self._kind == "llama":
            self._init_llama_layout()
        else:
            self._init_qwen3_layout()

        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token is None:
            raise ValueError("TTMLRolloutSampler: could not resolve pad_token from tokenizer")
        self._pad_token = int(pad_token)

        self._kv_cache: Any = None
        self._kv_cache_B: int = 0

        self._batch_id_counter: int = 0
        self._weight_version: int = 0

    # --------------------------------------------------------------
    # Batch sharding + KV-cache sizes, read from the open device and model
    # --------------------------------------------------------------

    def _init_llama_layout(self) -> None:
        mesh_device = self._mesh_device
        autograd_ctx = ttml.autograd.AutoContext.get_instance()

        ddp_enabled = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        self._dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if ddp_enabled else None
        self._dp_composer = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if ddp_enabled else None
        )
        ddp_axis = autograd_ctx.get_parallelism_context().get_ddp_axis() if ddp_enabled else None
        self._seed_axes = [int(ddp_axis)] if ddp_axis is not None else None

        cfg = self._model.config
        self._num_layers = cfg.num_hidden_layers
        self._num_kv_groups = cfg.num_key_value_heads
        self._head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._max_seq_len = cfg.max_position_embeddings

    def _init_qwen3_layout(self) -> None:
        mesh = ttml.mesh()
        mesh_device = self._mesh_device

        # build_mesh only names an axis "fsdp" / "dp" when enable_fsdp / enable_ddp is set.
        fsdp_enabled = mesh.has_axis("fsdp") and mesh.axis_size("fsdp") > 1
        ddp_enabled = mesh.has_axis("dp") and mesh.axis_size("dp") > 1

        batch_sharded = fsdp_enabled or ddp_enabled
        self._dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if batch_sharded else None
        self._dp_composer = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if batch_sharded else None
        )
        if not batch_sharded:
            self._num_devices = 1

        self._seed_axes = []
        if ddp_enabled:
            self._seed_axes.append(mesh.axis_index("dp"))
        if fsdp_enabled:
            self._seed_axes.append(mesh.axis_index("fsdp"))

        cfg = self._model.config
        self._num_layers = cfg.num_hidden_layers
        self._num_kv_groups = None
        self._head_dim = None
        self._max_seq_len = cfg.max_position_embeddings

    @property
    def temperature(self) -> float:
        """Sampling temperature used by :meth:`generate` (0 = greedy)."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = float(value)

    @property
    def completions_per_prompt(self) -> int:
        """Completions :meth:`generate` produces per prompt."""
        return self._completions_per_prompt

    @completions_per_prompt.setter
    def completions_per_prompt(self, value: int) -> None:
        self._completions_per_prompt = int(value)

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
        padded_q = TILE_SIZE if self._kind == "llama" else 1

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

        max_len = self._max_completion_length
        tokens_to_complete = max(0, min(max_len, self._max_seq_len - cache_advance))

        prompts_x: List[List[int]] = [list(r) for r in rows]

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
        # every ``CHUNK`` boundary. Reset each boundary to bound the async
        # readback payload — matches ``LlamaGRPOCompleter`` and
        # ``Qwen3GRPOCompleter``.
        chunk_columns: List[Any] = []

        # Async chunk state (used only for stop detection during decode).
        pending_hosts: List[Any] = []
        pending_event: Any = None
        done = np.zeros(B, dtype=bool)

        try:
            with run_mode(self._model, RunMode.EVAL), no_grad():
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

                deallocate_tensors([sampled_full, nlog_full, logits, input_tensor, prefill_mask])
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
                            [(0, 0), (0, 0), (0, 0), (0, TILE_SIZE - 1)],
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
                    # on the Qwen3 path.
                    if self._kind == "llama":
                        deallocate_tensors([token_input])
                    deallocate_tensors([decode_mask, logits, sampled, nlog_2d])
                    last_input = ttml.autograd.Tensor(last_token_col, False)

                    # Chunked async stop detection: every CHUNK steps, sync the
                    # previous chunk's async d2h and check for stop tokens;
                    # kick off the next chunk's async d2h. Only tokens need
                    # this — nlog columns are read back synchronously at end.
                    if (i + 1) % CHUNK == 0:
                        if pending_event is not None:
                            ttnn.event_synchronize(mesh_event=pending_event)
                            chunk_np = np.stack(
                                [h.to_numpy(mesh_composer=dp_composer).reshape(B) for h in pending_hosts],
                                axis=1,
                            )
                            done |= np.isin(chunk_np, stop_arr).any(axis=1)
                            if done.all():
                                break

                        pending_hosts, pending_event = async_read_to_host(chunk_columns, mesh_device)
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
        deallocate_tensors(generated_columns + nlog_columns)

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
