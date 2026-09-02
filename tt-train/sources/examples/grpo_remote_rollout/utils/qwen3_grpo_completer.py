# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttml-backed :class:`GRPOCompleter` for Qwen3, with remote rollout.

Sibling to :mod:`utils.llama_grpo_completer`. Runs on the ttml rank:
``compute_nlog_probs`` runs locally against a full-forward ttml Qwen3 (gradient
path); ``generate`` / ``generate_str`` proxy to the remote
:class:`TttGenerationWorker` via :class:`MPIRolloutClient`; :meth:`push_weights`
exports the ttml Qwen3 model to an HF-keyed dict via
:func:`qwen3_weights_ref_hf_dict` and ships it over the bridge.

Push once before :meth:`GRPOTrainer.train` to overwrite the worker's dummy boot
weights; the caller drives any periodic re-sync during training.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import ttnn

import ttml
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ttml.common.config import TransformerConfig
from ttml.common.utils import build_causal_mask, no_grad, round_up_to_tile
from ttml.models import RunnerType
from ttml.models.qwen3 import Qwen3, create_qwen3_config_from_hf
from ttml.models.qwen3.weights import load_weights_from_hf
from ttml.trainers.grpo_trainer import GRPOCompleter

from .mpi_rollout import MPIRolloutClient
from .qwen3_overrides import qwen3_weights_ref_hf_dict


@dataclass
class Qwen3CompletionCtx:
    """Generation parameters shared with the remote ttt worker.

    ``_tokenizer`` / ``_pad_token`` are populated by
    :class:`Qwen3CompleterRemoteRollout`; callers should not set them manually.
    """

    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


class Qwen3CompleterRemoteRollout(GRPOCompleter):
    """ttml-side :class:`GRPOCompleter`: remote ttt worker for generation,
    local ttml Qwen3 for nlog-prob computation.

    Does NOT open/close the device: the caller opens the ttml ``AutoContext``,
    passes the resulting ``mesh_device``, and owns ``close_device()``.

    Single-device / DDP only: :func:`qwen3_weights_ref_hf_dict` (used by
    :meth:`push_weights`) requires the ttml parameters to stay replicated on
    the mesh. DDP shards the batch, not the params, so it is compatible; FSDP
    / TP shard mappers on the parameters are not.
    """

    def __init__(
        self,
        ctx: Qwen3CompletionCtx,
        transformer_config: TransformerConfig,
        *,
        mesh_device: Any,
        model_source: str,
        inference_client: Optional[MPIRolloutClient] = None,
        enable_ddp: bool = False,
        memory_efficient: bool = True,
    ) -> None:
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        self._mesh_device: Any = mesh_device
        self._num_devices: int = mesh_device.get_num_devices()

        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

        if enable_ddp:
            autograd_ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False)
            )

        self._ddp_enabled: bool = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        # DDP shards the BATCH across the mesh (params stay replicated). Without
        # DDP the input tensors are replicated too; either way the weight
        # bridge only cares that the params are replicated.
        self._dp_mapper: Any = (
            ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if self._ddp_enabled else None
        )
        self._dp_composer: Any = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if self._ddp_enabled else None
        )
        if not self._ddp_enabled:
            # ``compute_nlog_probs`` divides the batch by ``self._num_devices``;
            # when replicated (no batch mapper), the whole batch is on every
            # device -- treat that as a single logical device for the divide.
            self._num_devices = 1

        max_seq_len = int(getattr(transformer_config, "max_sequence_length", 2048) or 2048)
        hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        # MemoryEfficient = gradient checkpointing: per-block activations are
        # recomputed in the backward pass instead of being retained, which
        # keeps the training forward within DRAM at the reverse-text batch
        # size (16 completions x 4 grad-accum micro-batches).
        runner_type = RunnerType.MemoryEfficient if memory_efficient else RunnerType.Default
        qwen_config = create_qwen3_config_from_hf(hf_config, max_seq_len, runner_type=runner_type)
        self._tie_word_embeddings: bool = bool(getattr(hf_config, "tie_word_embeddings", False))

        logging.info(
            "Building ttml Qwen3 model (hidden=%d, layers=%d)", qwen_config.hidden_size, qwen_config.num_hidden_layers
        )
        tt_model = Qwen3(qwen_config)

        hf_state_dict = self._load_hf_state_dict(model_source)
        load_weights_from_hf(tt_model, hf_state_dict, qwen_config, tie_word_embeddings=self._tie_word_embeddings)
        del hf_state_dict

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model
        self._config = qwen_config
        self.transformer_config = transformer_config

        self._client: Optional[MPIRolloutClient] = inference_client

    @staticmethod
    def _load_hf_state_dict(model_source: str) -> dict:
        """Return a HuggingFace float state-dict for ``model_source`` (mirrors
        ``Qwen3GRPOCompleter._load_hf_state_dict``)."""
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
        """The underlying ttml Qwen3 (NOT the remote tt-transformers worker)."""
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
        """Local ttml Qwen3 cross-entropy of (prompt + completion) on the
        training rank (gradient path, stays in-process).

        Mirrors ``Qwen3GRPOCompleter.compute_nlog_probs`` from the on-main
        example -- left-aligned tokens, full causal mask, sum-reduced later
        by the trainer against the completion-only loss mask.
        """
        assert len(completions) == len(prompts)
        B = len(completions)
        pad_token = self._ctx._pad_token

        total_devices = self._num_devices
        assert B % total_devices == 0, f"batch {B} must be divisible by num_devices {total_devices}"
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

    def push_weights(self) -> None:
        """Export the ttml Qwen3 model to an HF-keyed dict and send it once.

        Call once before ``GRPOTrainer.train()`` to overwrite the worker's
        dummy boot weights; the caller re-invokes it to re-sync during training.
        """
        hf_dict = qwen3_weights_ref_hf_dict(self._model, tie_word_embeddings=self._tie_word_embeddings)
        try:
            self._client.send_weights(hf_dict)
        finally:
            del hf_dict
            gc.collect()

    def _tokens_to_tensor(self, tokens_np: np.ndarray, B: int) -> ttml.autograd.Tensor:
        return ttml.autograd.Tensor.from_numpy(
            tokens_np.reshape(B, 1, 1, tokens_np.shape[1]).astype(np.uint32),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            self._dp_mapper,
        )
