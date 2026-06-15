# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 GRPO completion engine (single-device and tensor-parallel).

Mirrors the generic structure of :class:`utils.llama_completer.LlamaGRPOCompleter`
(prompt batching, next-token shift, completion loss-mask, tile padding) but
targets the Qwen3 architecture and supports tensor parallelism (TP) so large
models (e.g. Qwen3-32B) fit across a device mesh.

Key differences vs the Llama path:
  - Model is :class:`Qwen3` (single device) or
    :class:`utils.qwen_distributed.DistributedQwen3ForCausalLM` (TP).
  - ``compute_nlog_probs`` right-pads each prompt+completion at the front of a
    fixed-width buffer and uses a single shared ``[1, 1, Tp, Tp]`` causal mask,
    which works with Qwen3's regular SDPA path (no per-row composite mask /
    attention override needed).
  - Under TP the LM head emits vocab-sharded logits, so the per-token loss uses
    ``ttml.ops.distributed.vocab_parallel_cross_entropy_loss(reduce=NONE)``.
  - Generation all-gathers the vocab-sharded logits before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
import ttnn

import ttml
from ttml.common.config import DeviceConfig
from ttml.common.utils import no_grad
from ttml.models import RunnerType
from ttml.models.qwen3 import Qwen3, create_qwen3_config_from_hf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ttml.trainers.grpo_trainer import GRPOCompleter
from .qwen_distributed import (
    DistributedQwen3ForCausalLM,
    load_weights_from_hf_distributed,
    load_weights_from_hf_single,
)

TILE_SIZE = 32
# Mesh layout convention: dim 0 = data-parallel (DP), dim 1 = tensor-parallel (TP).
TP_SHARD_DIM = 1


@dataclass
class QwenCompletionCtx:
    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


def _round_up(x: int) -> int:
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


class QwenGRPOCompleter(GRPOCompleter):
    """Qwen3 completion engine for GRPO training (single-device or TP).

    Args:
        ctx: Generation parameters (``_tokenizer``/``_pad_token`` are filled in).
        device_config: Device mesh config. ``mesh_shape`` is interpreted as
            ``[dp_size, tp_size]``.
        model_source: HuggingFace model id (e.g. ``"Qwen/Qwen3-32B"``).
        max_seq_len: Maximum sequence length (prompt + completion) the model is
            built for. Drives RoPE/positional sizing and the generation window.
    """

    def setup_device(self, device_config: DeviceConfig) -> Any:
        """Open the mesh device and initialise the DP/TP parallelism context."""
        dp_size, tp_size = self._dp_size, self._tp_size
        total = dp_size * tp_size
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        if total > 1:
            ttml.core.distributed.enable_fabric(total)
            if device_config.device_ids:
                autograd_ctx.open_device([dp_size, tp_size], device_config.device_ids)
            else:
                autograd_ctx.open_device([dp_size, tp_size])
            autograd_ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=dp_size > 1, enable_tp=tp_size > 1)
            )
        else:
            autograd_ctx.open_device()
        return autograd_ctx.get_device()

    def __init__(
        self,
        ctx: QwenCompletionCtx,
        device_config: DeviceConfig,
        model_source: str,
        max_seq_len: int,
        use_checkpoint: bool = True,
    ) -> None:
        mesh_shape = list(device_config.mesh_shape)
        if len(mesh_shape) == 1:
            mesh_shape = [1, mesh_shape[0]]
        self._dp_size = int(mesh_shape[0])
        self._tp_size = int(mesh_shape[1])

        mesh_device = self.setup_device(device_config)
        self._mesh_device = mesh_device
        self._num_devices = mesh_device.get_num_devices()
        self._max_seq_len = max_seq_len

        tokenizer = AutoTokenizer.from_pretrained(model_source)
        hf_config = AutoConfig.from_pretrained(model_source)
        # Gradient checkpointing (recompute activations in backward) is essential
        # to fit the 32B forward+backward; it sets the runner for the single-device
        # core Qwen3 and is passed explicitly to the distributed model below.
        runner_type = RunnerType.MemoryEfficient if use_checkpoint else RunnerType.Default
        config = create_qwen3_config_from_hf(hf_config, max_seq_len, runner_type=runner_type)
        tie = bool(getattr(hf_config, "tie_word_embeddings", False))

        # Load HF weights on host (bf16 to halve host memory for large models),
        # then free the torch module immediately after extracting the state dict.
        hf_model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.bfloat16)
        hf_state_dict = hf_model.state_dict()
        del hf_model

        if self._tp_size > 1:
            tt_model = DistributedQwen3ForCausalLM(
                config, tie_word_embeddings=tie, shard_dim=TP_SHARD_DIM, use_checkpoint=use_checkpoint
            )
            load_weights_from_hf_distributed(tt_model, hf_state_dict, config, tie, shard_dim=TP_SHARD_DIM)
        else:
            tt_model = Qwen3(config)
            load_weights_from_hf_single(tt_model, hf_state_dict, config, tie)
        del hf_state_dict

        # Batch dim is sharded across the DP axis (mesh dim 0) and replicated
        # across the TP axis. With dp_size == 1 there is nothing to shard.
        self._dp_mapper = (
            ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0, 0) if self._dp_size > 1 else None
        )

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model
        self._config = config
        self._tie = tie

        # Report per-device DRAM occupied by the (TP-sharded) weights before any
        # forward pass. This is the baseline the activations + transient buffers
        # (e.g. the embedding untilize) have to fit on top of.
        from ttml.common.utils import log_device_dram_usage

        log_device_dram_usage("after weight load (pre-forward)")

    # ------------------------------------------------------------------
    # GRPOCompleter interface
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        return self._ctx._tokenizer

    @property
    def model(self) -> Any:
        return self._model

    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        ctx = self._ctx
        self._model.eval()
        with no_grad():
            return self._generate_impl(prompts)

    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        prompts = [self._ctx._tokenizer.encode(s) for s in prompt_strs]
        completions = self.generate(prompts)
        return [self._ctx._tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    def compute_nlog_probs(self, prompts: List[List[int]], completions: List[List[int]]):
        assert len(completions) == len(prompts)

        B = len(completions)
        pad_token = self._ctx._pad_token
        assert B % self._dp_size == 0, "batch must be divisible by dp_size"
        B_local = B // self._dp_size

        lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
        T = max(lengths) - 1
        assert T >= 1
        Tp = _round_up(T)

        # Right-pad: place sequence[:-1]/sequence[1:] at positions [0, L) and
        # pad the tail. A shared lower-triangular causal mask then works without
        # per-row left-padding masks (Qwen3 uses regular SDPA when q==k).
        inputs_np = np.full((B, Tp), pad_token, dtype=np.uint32)
        targets_np = np.full((B, Tp), pad_token, dtype=np.uint32)
        loss_mask_np = np.zeros((B, Tp), dtype=np.float32)

        for i, (p, c) in enumerate(zip(prompts, completions)):
            sequence = p + c
            if len(p) < 2:
                raise ValueError("Prompt is too short")
            if len(sequence) < 2:
                raise ValueError("Sequence is too short")
            L = len(sequence) - 1
            inputs_np[i, :L] = np.asarray(sequence[:-1], dtype=np.uint32)
            targets_np[i, :L] = np.asarray(sequence[1:], dtype=np.uint32)
            if c:
                start = len(p) - 1
                end = min(start + len(c), Tp)
                if start < end:
                    loss_mask_np[i, start:end] = 1.0

        logits = self._forward(inputs_np, B)

        targets_tt = ttml.autograd.Tensor.from_numpy(
            targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )

        if self._tp_size > 1:
            nlog = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
                logits, targets_tt, cluster_axis=TP_SHARD_DIM, reduce=ttml.ops.ReduceType.NONE
            )
        else:
            nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
        nlog = ttml.ops.reshape.reshape(nlog, [B_local, Tp])

        loss_mask_tt = ttml.autograd.Tensor.from_numpy(
            loss_mask_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, self._dp_mapper
        )
        return nlog, loss_mask_tt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_to_tensor(self, tokens_np: np.ndarray) -> ttml.autograd.Tensor:
        """numpy ``[B, S]`` uint32 -> ``[B, 1, 1, S]`` ttml tensor (DP-sharded)."""
        B, S = tokens_np.shape
        return ttml.autograd.Tensor.from_numpy(
            tokens_np.reshape(B, 1, 1, S).astype(np.uint32),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            self._dp_mapper,
        )

    def _causal_mask(self, seq_len: int) -> ttml.autograd.Tensor:
        """Shared lower-triangular causal mask ``[1, 1, seq_len, seq_len]`` (replicated)."""
        mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))
        return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)

    def _model_forward(self, x_tt, mask_tt, input_ids_np):
        """Forward through either the TP or single-device model (no KV cache)."""
        if self._tp_size > 1:
            return self._model(x_tt, mask_tt, input_ids_np=input_ids_np)
        return self._model(x_tt, mask_tt)

    def _forward(self, input_ids_np: np.ndarray, B: int) -> ttml.autograd.Tensor:
        Tp = input_ids_np.shape[1]
        x_tt = self._tokens_to_tensor(input_ids_np)
        mask_tt = self._causal_mask(Tp)
        return self._model_forward(x_tt, mask_tt, input_ids_np.reshape(B, 1, 1, Tp))

    def _get_stop_ids(self) -> set:
        tokenizer = self._ctx._tokenizer
        stop_ids: set = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        if tokenizer.pad_token_id is not None:
            stop_ids.add(int(tokenizer.pad_token_id))
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                stop_ids.add(int(tid))
        return stop_ids

    def _to_host_batch(self, tt_tensor: ttml.autograd.Tensor, B: int) -> np.ndarray:
        """Bring a (DP-sharded, TP-replicated) tensor back to host as full-batch numpy.

        After concat along dim 0 the rows are ordered as blocks per device in
        mesh order ``(dp0,tp0), (dp0,tp1), ..., (dp1,tp0), ...``; TP replicas are
        identical so we keep only the tp-rank-0 copy of each DP block.
        """
        if self._num_devices == 1:
            return tt_tensor.to_numpy()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(self._mesh_device, 0)
        arr = tt_tensor.to_numpy(composer=composer)
        if self._dp_size > 1:
            b_local = B // self._dp_size
            rest = arr.shape[1:]
            arr = arr.reshape(self._dp_size, self._tp_size, b_local, *rest)
            return arr[:, 0].reshape(B, *rest)
        return arr[:B]

    def _generate_impl(self, prompts: List[List[int]]) -> List[List[int]]:
        ctx = self._ctx
        cpp = ctx.completions_per_prompt

        # Replicate each prompt ``completions_per_prompt`` times (contiguous groups).
        expanded: List[List[int]] = [list(p) for p in prompts for _ in range(cpp)]
        B = len(expanded)
        assert B % self._dp_size == 0, "expanded batch must be divisible by dp_size"

        max_prompt_len = max(len(p) for p in expanded)
        tokens_to_complete = min(
            ctx.max_tokens_to_complete,
            self._max_seq_len - max_prompt_len,
        )
        if tokens_to_complete <= 0:
            return [[] for _ in range(B)]

        window = _round_up(min(max_prompt_len + tokens_to_complete, self._max_seq_len))
        mask_tt = self._causal_mask(window)

        V = self._config.vocab_size
        stop_ids = self._get_stop_ids()

        current: List[List[int]] = [list(p) for p in expanded]
        generated: List[List[int]] = [[] for _ in range(B)]
        done = np.zeros(B, dtype=bool)

        for _ in range(tokens_to_complete):
            padded = np.full((B, 1, 1, window), ctx._pad_token, dtype=np.uint32)
            pred_pos = np.empty(B, dtype=np.int64)
            for b in range(B):
                toks = current[b][-window:]
                padded[b, 0, 0, : len(toks)] = np.asarray(toks, dtype=np.uint32)
                pred_pos[b] = len(toks) - 1

            x_tt = ttml.autograd.Tensor.from_numpy(padded, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper)
            logits = self._model_forward(x_tt, mask_tt, padded)

            if self._tp_size > 1:
                logits = ttml.ops.distributed.all_gather(logits, dim=3, cluster_axis=TP_SHARD_DIM)

            gathered_width = int(logits.shape()[-1])
            logits_mask = self._build_logits_mask(V, gathered_width) if gathered_width != V else None

            sampled = ttml.ops.sample.sample_op(
                logits, ctx.temperature, int(np.random.randint(low=1, high=int(1e7))), logits_mask
            )
            sampled_np = self._to_host_batch(sampled, B).reshape(B, -1)

            for b in range(B):
                if done[b]:
                    continue
                tok = int(sampled_np[b, pred_pos[b]])
                generated[b].append(tok)
                current[b].append(tok)
                if tok in stop_ids:
                    done[b] = True

            ttml.autograd.AutoContext.get_instance().reset_graph()
            if done.all():
                break

        completions: List[List[int]] = []
        for b in range(B):
            seq = generated[b]
            to = len(seq)
            for j, token in enumerate(seq):
                if token in stop_ids:
                    to = j
                    break
            completions.append(seq[:to])
        return completions

    def _build_logits_mask(self, vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
        logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
        logits_mask[:, :, :, vocab_size:] = 1e4
        return ttml.autograd.Tensor.from_numpy(logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
