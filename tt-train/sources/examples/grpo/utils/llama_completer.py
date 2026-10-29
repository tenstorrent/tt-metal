# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import ttnn

import ttml
from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import no_grad, round_up_to_tile
from ttml.models import RunnerType
from ttml.models.llama import Llama, LlamaConfig, load_from_safetensors
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from ttml.trainers.grpo_trainer import GRPOCompleter
from ttml.common.sampling import positions_to_tensor

from .completer_common import deallocate_tensors, async_read_to_host

TILE_SIZE = 32
SAMPLE_SEED = 42

# Chunked async readback: every CHUNK decode steps the loop fires a
# non-blocking d2h for the just-finished chunk, then on the next chunk
# boundary host-synchronises that read and checks for stop tokens. The check
# thus lags compute by CHUNK steps, which is the cost of avoiding any
# per-step host sync.
CHUNK = 32


@dataclass
class LlamaCompletionCtx:
    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


def load_checkpoint(model: Any, checkpoint_path: str, dp_mapper: Any = None) -> None:
    """Load a ``model.safetensors`` the GRPO trainer saved, under TTML's parameter names, into *model*.
    Raise before writing anything if a parameter is missing, e.g. in a checkpoint saved before a rename."""
    from safetensors.numpy import load_file
    import ml_dtypes

    checkpoint = load_file(checkpoint_path)
    parameters = model.parameters()
    missing = sorted(name for name in parameters if name not in checkpoint)
    if missing:
        raise RuntimeError(
            f"{checkpoint_path} lacks {len(missing)} of the model's {len(parameters)} parameters: " + ", ".join(missing)
        )

    for name, param in parameters.items():
        arr = checkpoint[name].astype(ml_dtypes.bfloat16)
        if arr.ndim == 1:
            arr = arr.reshape(1, 1, 1, -1)
        elif arr.ndim == 2:
            arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
        restored = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
        param.assign(restored)

    print(f"Loaded {len(parameters)} parameters from {checkpoint_path}")


class LlamaGRPOCompleter(GRPOCompleter):
    """Llama-specific completion engine.

    Handles autoregressive generation and forward passes for Llama models.
    All mutable generation state (KV cache, batch dims) lives here, not
    on :class:`LlamaCompletionCtx`.

    Args:
        ctx: Generation parameters. ``_tokenizer`` and ``_pad_token`` are set
            automatically from ``model_source``.
        transformer_config: A :class:`ttml.common.config.TransformerConfig`.
            The architecture comes from the checkpoint's ``config.json``;
            this sets only ``max_sequence_length``, ``dropout_prob`` and ``runner_type``.
        device_config: Device mesh config (a
            :class:`ttml.common.config.DeviceConfig` instance). Device
            initialisation (``enable_fabric``, ``open_device``) is delegated
            to :meth:`setup_device`, which the constructor calls. Subclasses
            and tests may override ``setup_device`` to swap that behaviour
            (e.g. reuse an already-open device).
        model_source: A HuggingFace model ID; a local directory holding a HuggingFace checkpoint
            (``.safetensors``, ``config.json`` and the tokenizer); or a checkpoint directory saved
            by the GRPO trainer, recognised by its ``trainer_state.json``.
    """

    def setup_device(self, device_config: DeviceConfig) -> Any:
        """Enable fabric (multi-device) and open the AutoContext mesh device.

        Returns the resulting ``ttnn.MeshDevice``. Tests may override this
        method (e.g. via ``monkeypatch.setattr``) to reuse a device that was
        already opened earlier in the process by returning
        ``ttml.autograd.AutoContext.get_instance().get_device()`` without
        calling ``open_device`` again. Overrides must return a mesh whose
        topology matches ``device_config``; otherwise the cached
        ``_mesh_device``/``_num_devices`` will be inconsistent with the rest
        of the completer.
        """
        if device_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(device_config.total_devices())
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
        return autograd_ctx.get_device()

    def __init__(
        self,
        ctx: LlamaCompletionCtx,
        transformer_config: TransformerConfig,
        device_config: DeviceConfig,
        model_source: str,
    ) -> None:
        tf_config = transformer_config
        dev_config = device_config

        # Cache the device + parallelism state on ``self`` rather than going
        # through ``AutoContext`` on every tensor upload. The completer (and
        # the trainer that drives it) only handles single-device or DDP today;
        # tensor parallelism is intentionally not supported here.
        mesh_device: Any = self.setup_device(dev_config)
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        self._mesh_device: Any = mesh_device
        self._num_devices: int = mesh_device.get_num_devices()

        tokenizer = AutoTokenizer.from_pretrained(model_source)

        # save_checkpoint writes trainer_state.json beside the weights, which use TTML's parameter names;
        # any other local directory is a HuggingFace checkpoint, which may also be a single model.safetensors.
        trainer_checkpoint = os.path.isfile(os.path.join(model_source, "trainer_state.json"))
        if os.path.isdir(model_source):
            checkpoint_dir = model_source
        else:
            logging.info("Downloading model from HuggingFace: %s", model_source)
            checkpoint_dir = snapshot_download(
                repo_id=model_source,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
            )

        llama_cfg = LlamaConfig.from_hf(
            checkpoint_dir,
            max_position_embeddings=tf_config.max_sequence_length,
            attention_dropout=tf_config.dropout_prob,
            mlp_dropout=tf_config.dropout_prob,
            runner_type=RunnerType.from_string(str(tf_config.runner_type)),
            # The completer emits non-broadcast (B, 1, S, S) masks, which only the composite kernel accepts.
            use_composite_sdpa=True,
        )
        if len(tokenizer) > llama_cfg.vocab_size:
            raise ValueError(
                f"the tokenizer has {len(tokenizer)} tokens but the checkpoint embeds {llama_cfg.vocab_size}"
            )

        tt_model = Llama(llama_cfg)

        if dev_config.enable_ddp:
            # NOTE: TP is intentionally disabled here. The cross_entropy_loss call below
            # assumes full-vocab logits;
            autograd_ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False)
            )

        # Resolve DDP state once after the parallelism context is (maybe)
        # initialised so callers can use the cached mapper/composer/flags
        # without re-querying ``AutoContext``.
        self._ddp_enabled: bool = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        self._dp_mapper: Any = (
            ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if self._ddp_enabled else None
        )
        self._dp_composer: Any = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if self._ddp_enabled else None
        )
        # Mesh axis to seed UNIQUELY in the sample op: the DDP (data-parallel) axis, whose devices hold
        # DISTINCT prompts and must draw independent Gumbel noise. TP is disabled on this path (see
        # above), so the ddp axis is the only sharded axis; None => no per-device seeding (identical noise).
        ddp_axis = autograd_ctx.get_parallelism_context().get_ddp_axis() if self._ddp_enabled else None
        self._seed_axes: Any = [int(ddp_axis)] if ddp_axis is not None else None

        if trainer_checkpoint:
            logging.info("Resuming from a GRPO trainer checkpoint: %s", model_source)
            load_checkpoint(tt_model, os.path.join(model_source, "model.safetensors"), dp_mapper=self._dp_mapper)
        else:
            print(load_from_safetensors(tt_model, checkpoint_dir))

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model

        self._kv_cache: Any = None
        self._kv_cache_B: int = 0

    @property
    def tokenizer(self) -> Any:
        return self._ctx._tokenizer

    @property
    def model(self) -> Any:
        """The underlying tt model."""
        return self._model

    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        """Generate completions for a batch of tokenised prompts.

        For N prompts, returns N * completions_per_prompt completions.
        """
        ctx = self._ctx
        max_len = max(len(row) for row in prompts)
        pad_lengths = [max_len - len(row) for row in prompts for _ in range(ctx.completions_per_prompt)]
        B = ctx.completions_per_prompt * len(prompts)
        N = max_len

        prompt_tokens_np = np.full((B, max_len), ctx._pad_token)
        for i, row in enumerate(prompts):
            start = i * ctx.completions_per_prompt
            end = start + ctx.completions_per_prompt
            prompt_tokens_np[start:end, max_len - len(row) :] = np.asarray(row)

        self._model.eval()
        with no_grad():
            return self._completion_batched_impl(prompt_tokens_np, pad_lengths, B, N)

    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        """Generate completions from string prompts, returning decoded strings."""
        prompts = [self._ctx._tokenizer.encode(s) for s in prompt_strs]
        completions = self.generate(prompts)
        return [self._ctx._tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    def compute_nlog_probs(
        self, prompts: List[List[int]], completions: List[List[int]]
    ) -> tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
        assert len(completions) == len(prompts)

        B = len(completions)
        pad_token = self._ctx._pad_token

        total_devices = self._num_devices
        assert B % total_devices == 0
        B_local = B // total_devices

        lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
        T = max(lengths) - 1
        assert T >= 1

        inputs_np = np.full((B, T), pad_token, dtype=np.uint32)
        targets_np = np.full((B, T), pad_token, dtype=np.uint32)
        loss_mask_np = np.zeros((B, T), dtype=np.float32)
        pad_lengths: List[int] = []

        for i, (p, c) in enumerate(zip(prompts, completions)):
            sequence = p + c
            L = len(sequence) - 1
            shift = T - L
            pad_lengths.append(shift)

            if len(p) < 2:
                raise ValueError("Prompt is too short")
            if len(sequence) < 2:
                raise ValueError("Sequence is too short")

            inputs_np[i, -L:] = np.asarray(sequence[:-1], dtype=np.uint32)
            targets_np[i, -L:] = np.asarray(sequence[1:], dtype=np.uint32)

            if c:
                start = -1 + shift + len(p)
                end = min(start + len(c), T)
                if start < end:
                    loss_mask_np[i, start:end] = 1.0

        logits = self._forward(inputs_np, pad_lengths, B)

        Tp = round_up_to_tile(T)
        targets_pad = np.full((B, Tp), pad_token, dtype=np.uint32)
        targets_pad[:, :T] = targets_np

        targets_tt = ttml.autograd.Tensor.from_numpy(
            targets_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )

        nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
        nlog = ttml.ops.reshape.reshape(nlog, [B_local, Tp])

        loss_mask_pad = np.zeros((B, Tp), dtype=np.float32)
        loss_mask_pad[:, :T] = loss_mask_np

        loss_mask_tt = ttml.autograd.Tensor.from_numpy(
            loss_mask_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, self._dp_mapper
        )

        return nlog, loss_mask_tt

    def _forward(self, input_ids_np: np.ndarray, pad_lengths: List[int], B: int) -> ttml.autograd.Tensor:
        """Run a full forward pass (no KV cache) and return logits."""
        T = input_ids_np.shape[1]
        x_tt = self._tokens_to_tensor(input_ids_np, B)
        mask_tt = self._create_causal_mask(prompt_len=0, query_len=T, pad_lengths=pad_lengths, B=B)
        return self._model(x_tt, mask_tt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_to_tensor(self, tokens_np: np.ndarray, B: int) -> ttml.autograd.Tensor:
        padded_len = round_up_to_tile(tokens_np.shape[1])
        padded = np.full((B, padded_len), self._ctx._pad_token, dtype=np.uint32)
        padded[:, : tokens_np.shape[1]] = tokens_np
        return ttml.autograd.Tensor.from_numpy(
            padded.reshape(B, 1, 1, padded_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )

    def _create_causal_mask(
        self, prompt_len: int, query_len: int, pad_lengths: List[int], B: int
    ) -> ttml.autograd.Tensor:
        assert len(pad_lengths) == B

        whole_len = prompt_len + query_len
        padded_q = round_up_to_tile(query_len)
        padded_w = round_up_to_tile(whole_len)

        mask_one_token = np.zeros((padded_q, padded_w), dtype=np.float32)
        mask_one_token[:query_len, :padded_w] = np.tri(query_len, padded_w, k=prompt_len, dtype=np.float32)

        mask_3d = np.tile(mask_one_token, (B, 1, 1))
        for i in range(B):
            mask_3d[i, :, 0 : pad_lengths[i]] = 0

        mask_4d = mask_3d[:, np.newaxis, :, :]
        assert mask_4d.shape == (B, 1, padded_q, padded_w)

        return ttml.autograd.Tensor.from_numpy(mask_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, self._dp_mapper)

    def _build_logits_mask(self, vocab_size: int, padded_vocab_size: int, dtype: ttnn.DataType) -> ttml.autograd.Tensor:
        """``dtype`` should match the logits dtype -- the fused sampler typecasts a mismatched mask on
        every call, so the caller passes the dtype of the actual logits tensor to skip that."""
        logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
        logits_mask[:, :, :, vocab_size:] = 1e4
        return ttml.autograd.Tensor.from_numpy(logits_mask, ttnn.Layout.TILE, dtype)

    def _get_stop_ids(self) -> set[int]:
        tokenizer = self._ctx._tokenizer
        stop_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        if tokenizer.pad_token_id is not None:
            stop_ids.add(int(tokenizer.pad_token_id))
        for tok in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                stop_ids.add(int(tid))
        return stop_ids

    def _get_kv_cache(self, B: int) -> ttml.models.KvCache:
        cfg = self._model.config
        if self._kv_cache is None or self._kv_cache_B != B:
            self._kv_cache = ttml.models.KvCache(
                cfg.num_hidden_layers,
                B,
                cfg.num_key_value_heads,
                cfg.max_position_embeddings,
                cfg.hidden_size // cfg.num_attention_heads,
            )
            self._kv_cache_B = B
        self._kv_cache.reset()
        return self._kv_cache

    def _completion_batched_impl(
        self,
        prompt_tokens_np: np.ndarray,
        pad_lengths: List[int],
        B: int,
        N: int,
    ) -> List[List[int]]:
        ctx = self._ctx
        assert prompt_tokens_np.shape == (B, N)
        assert len(pad_lengths) == B
        total_devices = self._num_devices
        assert B % total_devices == 0

        B_local = B // total_devices

        V = len(ctx._tokenizer)
        padded_V = self._model.config.vocab_size

        kv_cache = self._get_kv_cache(B_local)
        # Built lazily on the first step: the mask should match the LOGITS dtype (a mismatch costs
        # a typecast on every sample call), and that dtype is only knowable from an actual forward pass.
        logits_mask_tensor = None

        tokens_to_complete = min(
            ctx.max_tokens_to_complete,
            self._model.config.max_position_embeddings - N,
        )

        mesh_device = self._mesh_device
        composer = self._dp_composer
        stop_ids = self._get_stop_ids()
        stop_arr = np.fromiter(stop_ids, dtype=np.int32) if stop_ids else np.empty(0, dtype=np.int32)

        generated_columns: List[Any] = []
        chunk_columns: List[Any] = []
        pending_hosts: List[Any] = []
        pending_event: Any = None
        done = np.zeros(B, dtype=bool)

        def to_np(column_list: List[Any]) -> np.ndarray:
            arr = np.empty((B, len(column_list)), dtype=np.int32)
            for j, column in enumerate(column_list):
                arr[:, j] = column.to_numpy(composer).reshape(B)
            return arr

        prefill_positions = positions_to_tensor([N - 1] * B, B, N, self._dp_mapper)
        decode_positions = positions_to_tensor([0] * B, B, 1, self._dp_mapper)

        for i in range(tokens_to_complete):
            is_prefill = kv_cache.get_cache_position() == 0
            if is_prefill:
                processed = 0
                new_tokens = prompt_tokens_np.shape[1]
                token_tensor = self._tokens_to_tensor(prompt_tokens_np, B)
            else:
                processed = N - 1
                new_tokens = 1
                token_tensor = ttnn.pad(
                    last_token_column,
                    [(0, 0), (0, 0), (0, 0), (0, TILE_SIZE - 1)],
                    ctx._pad_token,
                )
                token_tensor = ttml.autograd.Tensor(token_tensor, False)

            mask = self._create_causal_mask(processed, new_tokens, pad_lengths, B)
            logits = self._model(token_tensor, mask, kv_cache=kv_cache, new_tokens=new_tokens)

            if logits_mask_tensor is None and padded_V != V:
                logits_mask_tensor = self._build_logits_mask(V, padded_V, logits.get_value().dtype)

            next_token_tensor = ttml.ops.sample.sample_op(
                logits,
                ctx.temperature,
                np.random.randint(low=1e7),
                logits_mask_tensor,
                self._seed_axes,
                prefill_positions if is_prefill else decode_positions,
            )

            last_token_column = next_token_tensor.get_value()

            generated_columns.append(last_token_column)
            chunk_columns.append(last_token_column)
            N += 1

            deallocate_tensors([token_tensor, mask, logits])

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

        completions_np = to_np(generated_columns)
        deallocate_tensors(generated_columns + [prefill_positions, decode_positions, logits_mask_tensor])
        kv_cache.reset()

        completions: List[List[int]] = []
        for i in range(B):
            to = completions_np.shape[1]
            for j, token in enumerate(completions_np[i]):
                if token in stop_ids:
                    to = j
                    break
            completions.append(completions_np[i, :to].tolist())

        return completions
