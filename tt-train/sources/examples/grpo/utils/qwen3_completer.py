# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-specific GRPO completion engine with FSDP support.

Unlike :class:`LlamaGRPOCompleter` (which drives the C++ Llama binding), this
completer runs the pure-Python ttml Qwen3 model (``ttml.models.qwen3.Qwen3``)
and shards it across the ``"fsdp"`` mesh axis with :func:`ttml.fsdp.fully_shard`.

Generation uses a KV-cache decode: the (tile-aligned, right-padded) prompt
window is prefilled once with a plain broadcast causal mask, then one token is
decoded per step against the shared KV cache, sampling at each row's current
position. The fixed cache length keeps tensor shapes constant across steps (no
kernel recompiles).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import ttnn

import ttml
from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import no_grad, round_up_to_tile, build_causal_mask
from ttml.models import RunnerType
from ttml.models.qwen3 import Qwen3, create_qwen3_config_from_hf

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ttml.trainers.grpo_trainer import GRPOCompleter
from .completer_common import deallocate_tensors, async_read_to_host
from ttml.common.utils import build_mesh
from ttml.models.qwen3.weights import load_weights_from_hf
from ttml.models.qwen3.kv_cache import KVCache

# Chunked async readback during decode: instead of a blocking device->host sync
# every token, sampled token columns are read back non-blocking every CHUNK
# steps for stop-token detection. The check thus lags compute by CHUNK steps,
# which is the cost of keeping the device pipeline full.
CHUNK = 32


@dataclass
class Qwen3CompletionCtx:
    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


class Qwen3GRPOCompleter(GRPOCompleter):
    """Qwen3 completion engine that shards the model with FSDP.

    Args:
        ctx: Generation parameters. ``_tokenizer`` and ``_pad_token`` are filled
            in automatically from ``model_source``.
        transformer_config: Present for parity with ``LlamaGRPOCompleter``;
            only ``max_sequence_length`` is consulted (to bound the generation
            horizon). The architecture itself is read from the HF config of
            ``model_source``.
        device_config: Device mesh config. ``setup_device`` opens a *named*
            mesh (so an ``"fsdp"`` axis exists) when ``enable_fsdp`` is set.
        model_source: HuggingFace model ID or local directory.
        memory_efficient: When ``True`` (the default), build the model with
            ``RunnerType.MemoryEfficient`` (gradient checkpointing): per-block
            activations are recomputed in the backward pass instead of being
            retained, which keeps the training forward within DRAM at large
            micro-batch / sequence lengths. When ``False``, use
            ``RunnerType.Default`` (retain activations): faster backward, much
            higher peak memory.
    """

    def setup_device(self, device_config: DeviceConfig) -> Any:
        """Open a named device mesh (with an ``"fsdp"`` axis when enabled)."""
        mesh = build_mesh(device_config)
        device_ids = tuple(device_config.device_ids) if device_config.device_ids else None
        ttml.open_device_mesh(mesh, device_ids)
        self._mesh = mesh
        return ttml.autograd.AutoContext.get_instance().get_device()

    def __init__(
        self,
        ctx: Qwen3CompletionCtx,
        transformer_config: TransformerConfig,
        device_config: DeviceConfig,
        model_source: str,
        memory_efficient: bool = True,
    ) -> None:
        self._mesh: Any = None
        mesh_device: Any = self.setup_device(device_config)
        self._mesh_device = mesh_device
        self._num_devices = mesh_device.get_num_devices()

        self._fsdp_enabled = (
            bool(device_config.enable_fsdp) and self._mesh.has_axis("fsdp") and (self._mesh.axis_size("fsdp") > 1)
        )
        self._ddp_enabled = bool(
            device_config.enable_ddp and self._mesh.has_axis("dp") and (self._mesh.axis_size("dp") > 1)
        )

        # Batch sharding: when FSDP (or DDP) is active the batch is sliced
        # across the whole mesh along dim 0, matching the GRPO trainer.
        batch_sharded = self._fsdp_enabled or self._ddp_enabled
        self._dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if batch_sharded else None
        self._dp_composer = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if batch_sharded else None
        )
        if not batch_sharded:
            self._num_devices = 1

        # Mesh axes to seed UNIQUELY in the sample op: only the batch-sharded data-parallel axes
        # (dp / fsdp) that are actually active, whose devices hold DISTINCT prompts and must draw
        # independent Gumbel noise. Gated on the same enable flags as the batch mapper so seeding
        # always mirrors the batch sharding; a tp axis (replicated logits) is never included.
        self._seed_axes = []
        if self._ddp_enabled:
            self._seed_axes.append(self._mesh.axis_index("dp"))
        if self._fsdp_enabled:
            self._seed_axes.append(self._mesh.axis_index("fsdp"))

        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

        max_seq_len = int(getattr(transformer_config, "max_sequence_length", 2048) or 2048)
        hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        # MemoryEfficient = gradient checkpointing: per-block activations are
        # recomputed in the backward pass instead of being retained, which keeps
        # the training forward (compute_nlog_probs) within DRAM at large
        # micro-batch / sequence lengths. Matches the Llama GRPO path, which uses
        # memory_efficient. Pass memory_efficient=False (--no-memory_efficient on
        # the CLI) for the retain-activations runner (faster backward, much higher
        # peak memory).
        runner_type = RunnerType.MemoryEfficient if memory_efficient else RunnerType.Default
        qwen_config = create_qwen3_config_from_hf(hf_config, max_seq_len, runner_type=runner_type)
        tie = bool(getattr(hf_config, "tie_word_embeddings", False))

        logging.info(
            "Building ttml Qwen3 model (hidden=%d, layers=%d)", qwen_config.hidden_size, qwen_config.num_hidden_layers
        )

        # Lazy-init FSDP requires both the opt-in flag and an active FSDP axis;
        # without FSDP there is nothing to shard and the eager path is simpler.
        lazy_fsdp = self._fsdp_enabled and bool(device_config.lazy_parameter_init)

        if lazy_fsdp:
            # Lazy-init FSDP path (required for large models like 32B): build the
            # module tree without allocating any weights, let fully_shard rewrite
            # each lazy parameter's mapper to add Shard{dim} on the 'fsdp' axis,
            # materialize the parameters already-sharded, then stream the HF
            # weights in sharded. The full unsharded model is never present on a
            # single device. Contract: fully_shard -> materialize_module ->
            # (load weights) -> create_optimizer (the optimizer is created later
            # in GRPOTrainer.train()).
            logging.info(
                "Applying lazy-init FSDP fully_shard across the 'fsdp' axis " "(size=%d, reshard_after_forward=True)",
                self._mesh.axis_size("fsdp"),
            )
            with ttml.lazy_init():
                tt_model = Qwen3(qwen_config)
            for block in tt_model.blocks:
                ttml.fsdp.fully_shard(block, reshard_after_forward=True)
            ttml.fsdp.fully_shard(tt_model, reshard_after_forward=True)
            ttml.materialize_module(tt_model)

            # Weights are uploaded already-sharded to match each materialized
            # parameter's placements (full tensor stays in host RAM).
            hf_state_dict = self._load_hf_state_dict(model_source)
            load_weights_from_hf(tt_model, hf_state_dict, qwen_config, tie_word_embeddings=tie, sharded=True)
            del hf_state_dict
        else:
            tt_model = Qwen3(qwen_config)

            # Load HF weights into the (still replicated) model, then shard.
            hf_state_dict = self._load_hf_state_dict(model_source)
            load_weights_from_hf(tt_model, hf_state_dict, qwen_config, tie_word_embeddings=tie)
            del hf_state_dict

            if self._fsdp_enabled:
                logging.info(
                    "Applying FSDP fully_shard across the 'fsdp' axis " "(size=%d, reshard_after_forward=True)",
                    self._mesh.axis_size("fsdp"),
                )
                for block in tt_model.blocks:
                    ttml.fsdp.fully_shard(block, reshard_after_forward=True)
                ttml.fsdp.fully_shard(tt_model, reshard_after_forward=True)

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model
        self._config = qwen_config
        self._max_seq_len = max_seq_len

    @staticmethod
    def _load_hf_state_dict(model_source: str) -> dict:
        """Return a HuggingFace float state-dict for ``model_source``."""
        import torch

        if os.path.isdir(model_source):
            path = model_source
        else:
            path = snapshot_download(
                repo_id=model_source,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
            )
        hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
        state_dict = hf_model.state_dict()
        del hf_model
        return state_dict

    @property
    def tokenizer(self) -> Any:
        return self._ctx._tokenizer

    @property
    def model(self) -> Any:
        return self._model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_causal_mask(self, cur_pos: int, width: int) -> ttml.autograd.Tensor:
        """Single-query decode mask ``[1, 1, 1, width]`` (broadcast over batch).

        The new query at absolute position ``cur_pos`` attends to cache columns
        ``[0, cur_pos]`` (causal). Mirrors ``KVCache.get_attn_mask`` in
        ``ttml.models.qwen3.kv_cache`` (a row of the full causal mask).
        """
        mask = np.zeros((1, 1, 1, width), dtype=np.float32)
        hi = min(cur_pos, width - 1)
        mask[0, 0, 0, : hi + 1] = 1.0
        return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)

    def _tokens_to_tensor(self, tokens_np: np.ndarray, B: int) -> ttml.autograd.Tensor:
        return ttml.autograd.Tensor.from_numpy(
            tokens_np.reshape(B, 1, 1, tokens_np.shape[1]).astype(np.uint32),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            self._dp_mapper,
        )

    def _get_stop_ids(self) -> set:
        tokenizer = self._ctx._tokenizer
        stop_ids: set = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        if tokenizer.pad_token_id is not None:
            stop_ids.add(int(tokenizer.pad_token_id))
        for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                stop_ids.add(int(tid))
        return stop_ids

    # ------------------------------------------------------------------
    # GRPOCompleter API
    # ------------------------------------------------------------------

    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        """KV-cache generation following the validated examples/qwen3/generate.py
        recipe: left-ALIGNED (right-padded) prompts, a plain broadcast causal
        mask (so prefill uses the fused SDPA), and a per-row prediction position
        (``len_b - 1``). Prefill once, then decode one token per step against the
        shared KV cache. Decode keeps the speed wins (device-resident token feed +
        chunked async stop readback) but uses the same plain broadcast masks.
        """
        ctx = self._ctx
        G = ctx.completions_per_prompt
        rows: List[List[int]] = [list(p) for p in prompts for _ in range(G)]
        B = len(rows)
        assert B % self._num_devices == 0, f"batch {B} must be divisible by num_devices {self._num_devices}"

        lengths = [len(r) for r in rows]
        max_prompt_len = max(lengths)
        # Tile-aligned prompt window; prompts are left-aligned within it (real
        # tokens at [0, len_b), pad after). Per-row prediction at len_b - 1.
        Np = round_up_to_tile(max_prompt_len)
        pred_pos = [length - 1 for length in lengths]

        tokens_to_complete = min(ctx.max_tokens_to_complete, self._max_seq_len - Np)
        tokens_to_complete = max(tokens_to_complete, 0)

        stop_ids = self._get_stop_ids()

        logging.info(
            "[qwen3] generate B=%d prompts=%d Np=%d tokens_to_complete=%d",
            B,
            len(prompts),
            Np,
            tokens_to_complete,
        )
        # Log decode prompt[0] to confirm the chat-templated input is sane (not itself
        # garbage) -- rules the prompt in/out as a gibberish source.
        try:
            preview = self._ctx._tokenizer.decode(rows[0], skip_special_tokens=False)
            logging.info("[qwen3] prompt[0] (%d toks) = %r", len(rows[0]), preview[:300])
        except Exception:  # noqa: BLE001
            pass

        if tokens_to_complete <= 0:
            return [[] for _ in range(B)]

        # Fixed cache length (model max) keeps the decode cache / mask / SDPA
        # shapes constant across GRPO steps (no per-step kernel recompilation).
        # Freed after each generate(); generation is no_grad + memory-efficient,
        # so the full-size allocation only coexists with generation activations.
        cache_len = self._max_seq_len
        kv = KVCache(self._config.num_hidden_layers, cache_len)

        B_local = B // self._num_devices
        composer = self._dp_composer
        mesh_device = self._mesh_device
        stop_arr = np.fromiter(stop_ids, dtype=np.int32) if stop_ids else np.empty(0, dtype=np.int32)

        # Decode token columns stay resident on device (each [B_local, 1, 1, 1]);
        # read back to host in chunks for stop detection and once at the end. This
        # avoids a blocking d2h sync per token.
        generated_columns: List[Any] = []
        chunk_columns: List[Any] = []
        pending_hosts: List[Any] = []
        pending_event: Any = None
        done = np.zeros(B, dtype=bool)
        first_tokens = np.zeros(B, dtype=np.int64)

        def _columns_to_np(column_list: List[Any]) -> np.ndarray:
            if not column_list:
                return np.empty((B, 0), dtype=np.int32)
            arr = np.empty((B, len(column_list)), dtype=np.int32)
            for j, column in enumerate(column_list):
                arr[:, j] = column.to_numpy(mesh_composer=composer).reshape(B)
            return arr

        self._model.eval()
        try:
            with no_grad():
                # --- Prefill: left-aligned prompt, plain causal mask (fused
                #     SDPA), sample each row at its own last position len_b - 1. ---
                prompt_np = np.full((B, Np), ctx._pad_token, dtype=np.uint32)
                for b in range(B):
                    seq = rows[b]
                    prompt_np[b, : len(seq)] = np.asarray(seq, dtype=np.uint32)

                input_tensor = self._tokens_to_tensor(prompt_np, B)
                prefill_mask = build_causal_mask(Np, device=True)
                logits = self._model(input_tensor, prefill_mask, past_key_values=kv)

                seed = int(np.random.randint(low=1, high=int(1e7)))
                # Seed uniquely only over the batch-sharded axes (dp/fsdp) so each device's rollout
                # draws independent noise; a replicated (tp) axis, if any, is excluded via _seed_axes.
                sampled = ttml.ops.sample.sample_op(logits, ctx.temperature, seed, None, self._seed_axes)
                # Per-row prediction position: read the whole sampled column once
                # on host and pick row b's token at pred_pos[b].
                sampled_host = ttnn.to_torch(sampled.get_value(), mesh_composer=composer)
                sampled_host = sampled_host.reshape(B, 1, Np, 1).to(int).numpy()
                for b in range(B):
                    tok = int(sampled_host[b, 0, pred_pos[b], 0])
                    first_tokens[b] = tok
                    if tok in stop_ids:
                        done[b] = True
                deallocate_tensors([input_tensor, prefill_mask, logits, sampled])
                ttml.autograd.AutoContext.get_instance().reset_graph()

                # --- Decode: feed one token per step. The first decode input is
                #     the prefill token (uploaded once); thereafter the sampled
                #     token is reused straight from device. ---
                last_input = self._tokens_to_tensor(first_tokens.reshape(B, 1).astype(np.uint32), B)
                decode_t0 = time.perf_counter()
                for i in range(tokens_to_complete - 1):
                    if done.all():
                        break

                    cur_pos = kv.get_seq_length()
                    decode_mask = self._decode_causal_mask(cur_pos, cache_len)
                    logits = self._model(last_input, decode_mask, past_key_values=kv)

                    seed = int(np.random.randint(low=1, high=int(1e7)))
                    sampled = ttml.ops.sample.sample_op(logits, ctx.temperature, seed, None, self._seed_axes)
                    # Clone so the column is independent of the deallocated sampled.
                    last_token_column = ttnn.clone(ttnn.slice(sampled.get_value(), [0, 0, 0, 0], [B_local, 1, 1, 1]))
                    generated_columns.append(last_token_column)
                    chunk_columns.append(last_token_column)
                    # Do NOT deallocate ``last_input``: after step 0 it wraps the
                    # previous step's still-referenced column.
                    deallocate_tensors([decode_mask, logits, sampled])
                    last_input = ttml.autograd.Tensor(last_token_column, False)

                    # Chunked async stop detection.
                    if (i + 1) % CHUNK == 0:
                        if pending_event is not None:
                            ttnn.event_synchronize(mesh_event=pending_event)
                            chunk_np = np.stack(
                                [h.to_numpy(mesh_composer=composer).reshape(B) for h in pending_hosts],
                                axis=1,
                            )
                            done |= np.isin(chunk_np, stop_arr).any(axis=1)
                            if done.all():
                                break
                        pending_hosts, pending_event = async_read_to_host(chunk_columns, mesh_device)
                        chunk_columns = []

                        n_done = i + 1
                        elapsed = time.perf_counter() - decode_t0
                        rate = n_done / elapsed if elapsed > 0 else 0.0
                        logging.debug(
                            "[qwen3] decode %d/%d cache_pos=%d done=%d/%d %.1f tok/s (%.0fs)",
                            n_done,
                            tokens_to_complete - 1,
                            kv.get_seq_length(),
                            int(done.sum()),
                            B,
                            rate,
                            elapsed,
                        )

                decode_np = _columns_to_np(generated_columns)
        finally:
            # Free the K/V DRAM and token columns before the trainer's next forward.
            deallocate_tensors(generated_columns)
            kv.clear()
            ttml.autograd.AutoContext.get_instance().reset_graph()

        # Assemble [first token + decode tokens] per row, trim at first stop.
        completions: List[List[int]] = []
        for b in range(B):
            seq = [int(first_tokens[b])] + [int(t) for t in decode_np[b]]
            cut = len(seq)
            for j, tok in enumerate(seq):
                if tok in stop_ids:
                    cut = j
                    break
            completions.append(seq[:cut])
        return completions

    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        prompts = [self._ctx._tokenizer.encode(s) for s in prompt_strs]
        completions = self.generate(prompts)
        return [self._ctx._tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    def compute_nlog_probs(
        self, prompts: List[List[int]], completions: List[List[int]]
    ) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
        assert len(completions) == len(prompts)
        B = len(completions)
        pad_token = self._ctx._pad_token

        total_devices = self._num_devices
        assert B % total_devices == 0
        B_local = B // total_devices

        lengths = [len(p) + len(c) - 1 for p, c in zip(prompts, completions)]
        T = max(lengths)
        assert T >= 1
        Tp = round_up_to_tile(T)

        inputs_np = np.full((B, Tp), pad_token, dtype=np.uint32)
        targets_np = np.full((B, Tp), pad_token, dtype=np.uint32)
        loss_mask_np = np.zeros((B, Tp), dtype=np.float32)

        for i, (p, c) in enumerate(zip(prompts, completions)):
            if len(p) < 2:
                raise ValueError("Prompt is too short")
            sequence = p + c
            if len(sequence) < 2:
                raise ValueError("Sequence is too short")
            L = len(sequence) - 1

            inputs_np[i, :L] = np.asarray(sequence[:-1], dtype=np.uint32)
            targets_np[i, :L] = np.asarray(sequence[1:], dtype=np.uint32)

            if c:
                start = len(p) - 1
                end = min(start + len(c), L)
                if start < end:
                    loss_mask_np[i, start:end] = 1.0

        input_tensor = self._tokens_to_tensor(inputs_np, B)
        mask = build_causal_mask(Tp, device=True)
        logits = self._model(input_tensor, mask)

        targets_tt = ttml.autograd.Tensor.from_numpy(
            targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )
        nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
        nlog = ttml.ops.reshape.reshape(nlog, [B_local, Tp])

        loss_mask_tt = ttml.autograd.Tensor.from_numpy(
            loss_mask_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, self._dp_mapper
        )

        return nlog, loss_mask_tt
