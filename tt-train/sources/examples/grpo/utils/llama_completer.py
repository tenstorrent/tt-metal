# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import ttnn

import ttml
from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import no_grad
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from ttml.trainers.grpo_trainer import GRPOCompleter
from .llama_overrides import LlamaCompositeKV


TILE_SIZE = 32
SAMPLE_SEED = 42


@dataclass
class LlamaCompletionCtx:
    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: object = None
    _pad_token: int = None


def deallocate_tensors(tensors) -> None:
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


def load_checkpoint(model, checkpoint_path, dp_mapper=None):
    from safetensors.numpy import load_file
    import ml_dtypes

    checkpoint = load_file(checkpoint_path)
    parameters = model.parameters()
    loaded, missing = 0, []

    for name, param in parameters.items():
        if name in checkpoint:
            arr = checkpoint[name].astype(ml_dtypes.bfloat16)
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, 1, -1)
            elif arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            restored = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
            param.assign(restored)
            loaded += 1
        else:
            missing.append(name)

    print(f"Loaded {loaded}/{len(parameters)} parameters from {checkpoint_path}")
    if missing:
        print(f"Warning: {len(missing)} parameters not found in checkpoint:")
        for n in missing:
            print(f"  - {n}")


def _round_up(x: int) -> int:
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


class LlamaGRPOCompleter(GRPOCompleter):
    """Llama-specific completion engine.

    Handles autoregressive generation and forward passes for Llama models.
    All mutable generation state (KV cache, batch dims) lives here, not
    on :class:`LlamaCompletionCtx`.

    Args:
        ctx: Generation parameters. ``_tokenizer`` and ``_pad_token`` are set
            automatically from ``model_source``.
        transformer_config: Model architecture config dict (same format as
            used in ``GRPOConfig``).
        device_config: Device mesh config dict (``enable_ddp``, ``mesh_shape``,
            etc.). Device initialisation (``enable_fabric``, ``open_device``)
            is performed inside this constructor.
        model_source: HuggingFace model ID or path to a local directory
            containing ``model.safetensors``.
    """

    def __init__(
        self,
        ctx: LlamaCompletionCtx,
        transformer_config: dict,
        device_config: dict,
        model_source: str,
    ):
        tf_config = TransformerConfig({"transformer_config": transformer_config})
        dev_config = DeviceConfig({"device_config": device_config})

        if dev_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(dev_config.total_devices())
        ttml.autograd.AutoContext.get_instance().open_device(dev_config.mesh_shape, dev_config.device_ids)

        tokenizer = AutoTokenizer.from_pretrained(model_source)
        tf_config.vocab_size = len(tokenizer)

        rope_scaling = LlamaRopeScalingConfig(
            scaling_factor=getattr(tf_config, "scaling_factor", 0.0) or 0.0,
            high_freq_factor=getattr(tf_config, "high_freq_factor", 4.0) or 4.0,
            low_freq_factor=getattr(tf_config, "low_freq_factor", 1.0) or 1.0,
            original_context_length=getattr(tf_config, "original_context_length", 0) or 0,
        )

        runner_type = RunnerType.from_string(str(tf_config.runner_type))
        weight_tying = WeightTyingType.Disabled
        if tf_config.weight_tying:
            weight_tying = WeightTyingType.from_string(str(tf_config.weight_tying))

        llama_cfg = LlamaConfig(
            hidden_size=tf_config.embedding_dim,
            intermediate_size=tf_config.intermediate_dim,
            num_hidden_layers=tf_config.num_blocks,
            num_attention_heads=tf_config.num_heads,
            num_key_value_heads=tf_config.num_groups,
            vocab_size=len(tokenizer),
            max_position_embeddings=tf_config.max_sequence_length,
            rope_theta=tf_config.theta or 10000.0,
            attention_dropout=tf_config.dropout_prob,
            mlp_dropout=tf_config.dropout_prob,
            runner_type=runner_type,
            weight_tying=weight_tying,
            rope_scaling=rope_scaling,
        )

        tt_model = LlamaCompositeKV(llama_cfg)

        _dp_mapper = None
        _dp_composer = None
        _total_devices = 1
        if dev_config.enable_ddp:
            autograd_ctx = ttml.autograd.AutoContext.get_instance()
            autograd_ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False)
            )
            device = autograd_ctx.get_device()
            _dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
            _dp_composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
            _total_devices = dev_config.total_devices()

        local_safetensors = os.path.isdir(model_source) and any(
            f == "model.safetensors" for f in os.listdir(model_source)
        )
        if local_safetensors:
            logging.info("Loading model from local safetensors: %s", model_source)
            load_checkpoint(tt_model, model_source, dp_mapper=_dp_mapper)
        else:
            logging.info("Downloading model from HuggingFace: %s", model_source)
            model_repo_path = snapshot_download(
                repo_id=model_source,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
            )
            load_from_safetensors(tt_model, model_repo_path, llama_cfg)

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model
        self.transformer_config = tf_config
        self._dp_mapper = _dp_mapper
        self._dp_composer = _dp_composer
        self._total_devices = _total_devices

        self._kv_cache = None
        self._kv_cache_B: int = 0

    @property
    def tokenizer(self):
        return self._ctx._tokenizer

    @property
    def dp_mapper(self):
        return self._dp_mapper

    @property
    def dp_composer(self):
        return self._dp_composer

    @property
    def model(self):
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

        assert B % self._total_devices == 0
        B_local = B // self._total_devices

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

        Tp = _round_up(T)
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
        padded_len = _round_up(tokens_np.shape[1])
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
        padded_q = _round_up(query_len)
        padded_w = _round_up(whole_len)

        mask_one_token = np.zeros((padded_q, padded_w), dtype=np.float32)
        mask_one_token[:query_len, :padded_w] = np.tri(query_len, padded_w, k=prompt_len, dtype=np.float32)

        mask_3d = np.tile(mask_one_token, (B, 1, 1))
        for i in range(B):
            mask_3d[i, :, 0 : pad_lengths[i]] = 0

        mask_4d = mask_3d[:, np.newaxis, :, :]
        assert mask_4d.shape == (B, 1, padded_q, padded_w)

        return ttml.autograd.Tensor.from_numpy(mask_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, self._dp_mapper)

    def _build_logits_mask(self, vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
        logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
        logits_mask[:, :, :, vocab_size:] = 1e4
        return ttml.autograd.Tensor.from_numpy(logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)

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
        cfg = self.transformer_config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.embedding_dim // cfg.num_heads)
        if self._kv_cache is None or self._kv_cache_B != B:
            self._kv_cache = ttml.models.KvCache(
                cfg.num_blocks,
                B,
                cfg.num_groups,
                cfg.max_sequence_length,
                head_dim,
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
        assert B % self._total_devices == 0

        B_local = B // self._total_devices

        V = len(ctx._tokenizer)
        padded_V = _round_up(V)

        kv_cache = self._get_kv_cache(B_local)
        logits_mask_tensor = self._build_logits_mask(V, padded_V) if padded_V != V else None

        tokens_to_complete = min(
            ctx.max_tokens_to_complete,
            self.transformer_config.max_sequence_length - N,
        )

        generated_columns: list = []

        def to_np(column_list):
            arr = np.empty((B, len(column_list)), dtype=np.int32)
            for j, column in enumerate(column_list):
                arr[:, j] = column.to_numpy(self._dp_composer).reshape(B)
            return arr

        for i in range(tokens_to_complete):
            if kv_cache.get_cache_position() == 0:
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

            next_token_tensor = ttml.ops.sample.sample_op(
                logits, ctx.temperature, np.random.randint(low=1e7), logits_mask_tensor
            )

            last_token_column = ttnn.slice(
                next_token_tensor.get_value(),
                [0, 0, new_tokens - 1, 0],
                [B_local, 1, new_tokens, 1],
            )

            generated_columns.append(last_token_column)
            N += 1

            deallocate_tensors([token_tensor, mask, logits, next_token_tensor])

        completions_np = to_np(generated_columns)
        deallocate_tensors(generated_columns)
        deallocate_tensors([logits_mask_tensor])
        kv_cache.reset()

        stop_ids = self._get_stop_ids()

        completions: List[List[int]] = []
        for i in range(B):
            to = ctx.max_tokens_to_complete
            for j, token in enumerate(completions_np[i]):
                if token in stop_ids:
                    to = j
                    break
            completions.append(completions_np[i, :to].tolist())

        return completions
