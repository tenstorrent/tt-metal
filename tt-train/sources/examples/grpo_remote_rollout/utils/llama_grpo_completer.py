# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttml-backed :class:`GRPOCompleter` that delegates generation over RPC.

Runs on the ttml rank: ``compute_nlog_probs`` runs locally (gradient path);
``generate`` / ``generate_str`` proxy to the remote :class:`TttGenerationWorker`
via :class:`MPIRolloutClient`; :meth:`push_weights` ships the ttml weights over.
Push once before :meth:`GRPOTrainer.train` to overwrite the worker's dummy boot
weights; the caller drives any periodic re-sync during training.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import ttnn

import ttml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from ttml.common.config import TransformerConfig
from ttml.common.utils import no_grad
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from ttml.trainers.grpo_trainer import GRPOCompleter

from .mpi_rollout import MPIRolloutClient
from .llama_overrides import LlamaCompositeKV


TILE_SIZE = 32


@dataclass
class LlamaCompletionCtx:
    """Generation parameters shared with the remote ttt worker.

    ``_tokenizer`` / ``_pad_token`` are populated by
    :class:`LlamaCompleterRemoteRollout`; callers should not set them manually.
    """

    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


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


def _load_checkpoint(model: Any, checkpoint_path: str, dp_mapper: Any = None) -> None:
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


def _ensure_safetensors_dir(model_dir: str) -> str:
    """Ensure ``model_dir`` has a ``*.safetensors`` file, converting a legacy
    ``pytorch_model.bin`` once on first use if needed."""
    p = Path(model_dir)
    if list(p.glob("*.safetensors")):
        return model_dir

    bin_files = sorted(p.glob("pytorch_model*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"Neither *.safetensors nor pytorch_model*.bin found in {model_dir}")

    import torch
    from safetensors.torch import save_file

    state_dict: dict = {}
    for bin_file in bin_files:
        logging.info("Converting legacy weights to safetensors: %s", bin_file)
        sd = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            state_dict[k] = v.contiguous()

    out_path = p / "model.safetensors"
    print(f"[ttml] writing converted safetensors to {out_path}")
    save_file(state_dict, str(out_path))
    return model_dir


class LlamaCompleterRemoteRollout(GRPOCompleter):
    """ttml-side :class:`GRPOCompleter`: remote ttt worker for generation,
    local ttml ``Llama`` for nlog-prob computation.

    Does NOT open/close the device: the caller opens the ttml ``AutoContext``,
    passes the resulting ``mesh_device``, and owns ``close_device()``.
    """

    def __init__(
        self,
        ctx: LlamaCompletionCtx,
        transformer_config: TransformerConfig,
        *,
        mesh_device: Any,
        model_source: str,
        inference_client: MPIRolloutClient,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        enable_ddp: bool = False,
    ) -> None:
        tf_config = transformer_config

        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        self._mesh_device: Any = mesh_device
        self._num_devices: int = mesh_device.get_num_devices()

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

        if enable_ddp:
            autograd_ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False)
            )

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

        local_safetensors = os.path.isdir(model_source) and any(
            f == "model.safetensors" for f in os.listdir(model_source)
        )
        if local_safetensors:
            logging.info("Loading model from local safetensors: %s", model_source)
            _load_checkpoint(tt_model, model_source, dp_mapper=self._dp_mapper)
        else:
            logging.info("Downloading model from HuggingFace: %s", model_source)
            model_repo_path = snapshot_download(
                repo_id=model_source,
                allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
            )
            model_repo_path = _ensure_safetensors_dir(model_repo_path)
            load_from_safetensors(tt_model, model_repo_path, llama_cfg)

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model
        self.transformer_config = tf_config

        self._client = inference_client
        self._top_p = float(top_p)
        self._seed = seed

    @property
    def tokenizer(self) -> Any:
        return self._ctx._tokenizer

    @property
    def model(self) -> Any:
        """The underlying ttml model (NOT the remote tt-transformers worker)."""
        return self._model

    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        """Generate remotely via the ttt worker.

        For N prompts, returns N * ``completions_per_prompt`` completions.
        """
        ctx = self._ctx
        if ctx.completions_per_prompt > 1:
            expanded = [list(p) for p in prompts for _ in range(ctx.completions_per_prompt)]
        else:
            expanded = [list(p) for p in prompts]
        return self._client.remote_generate(
            expanded,
            max_new_tokens=int(ctx.max_tokens_to_complete),
            temperature=float(ctx.temperature),
            top_p=self._top_p,
            seed=self._seed,
        )

    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        """Generate from strings: tokenise locally, ship IDs, decode locally."""
        tok = self._ctx._tokenizer
        prompts = [tok.encode(s) for s in prompt_strs]
        completions = self.generate(prompts)
        return [tok.decode(c, skip_special_tokens=False) for c in completions]

    def compute_nlog_probs(
        self, prompts: List[List[int]], completions: List[List[int]]
    ) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
        """Local-only cross-entropy of (prompt + completion) on ttml (gradient
        path, stays in-process)."""
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

    def push_weights(self) -> None:
        """Export the ttml model to an HF-keyed dict and send_weights it once.

        Call once before ``GRPOTrainer.train()`` to overwrite the worker's dummy
        boot weights; the caller re-invokes it to re-sync during training.
        """
        hf_dict = self._model.weights_ref_hf_dict()
        try:
            self._client.send_weights(hf_dict)
        finally:
            del hf_dict
            gc.collect()

    def _forward(self, input_ids_np: np.ndarray, pad_lengths: List[int], B: int) -> ttml.autograd.Tensor:
        """Run a full forward pass (no KV cache) and return logits."""
        T = input_ids_np.shape[1]
        x_tt = self._tokens_to_tensor(input_ids_np, B)
        mask_tt = self._create_causal_mask(prompt_len=0, query_len=T, pad_lengths=pad_lengths, B=B)
        return self._model(x_tt, mask_tt)

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
