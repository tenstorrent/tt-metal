# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-specific GRPO completion engine with FSDP support.

Unlike :class:`LlamaGRPOCompleter` (which drives the C++ Llama binding), this
completer runs the pure-Python ttml Qwen3 model (``ttml.models.qwen3.Qwen3``)
and shards it across the ``"fsdp"`` mesh axis with :func:`ttml.fsdp.fully_shard`.

Generation uses a fixed-horizon full-recompute decode (no KV cache): every
step re-runs the forward over a fixed-length, right-padded window and samples
at each row's current position. This keeps tensor shapes constant (no kernel
recompiles) and avoids the per-row KV-cache mask bookkeeping, at the cost of
recomputing the prefix each step. It is intentionally simple; the priority
here is validating the Qwen3 + FSDP + GRPO pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import ttnn

import ttml
from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import no_grad
from ttml.models import RunnerType
from ttml.models.qwen3 import Qwen3, create_qwen3_config_from_hf

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ttml.trainers.grpo_trainer import GRPOCompleter
from .qwen3_support import build_mesh, load_weights_from_hf

TILE_SIZE = 32


@dataclass
class Qwen3CompletionCtx:
    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


def _round_up(x: int) -> int:
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


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
    ) -> None:
        self._mesh: Any = None
        mesh_device: Any = self.setup_device(device_config)
        self._mesh_device = mesh_device
        self._num_devices = mesh_device.get_num_devices()

        self._fsdp_enabled = (
            bool(device_config.enable_fsdp) and self._mesh.has_axis("fsdp") and (self._mesh.axis_size("fsdp") > 1)
        )

        # Batch sharding: when FSDP (or DDP) is active the batch is sliced
        # across the whole mesh along dim 0, matching the GRPO trainer.
        batch_sharded = self._fsdp_enabled or bool(device_config.enable_ddp)
        self._dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if batch_sharded else None
        self._dp_composer = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if batch_sharded else None
        )
        if not batch_sharded:
            self._num_devices = 1

        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

        max_seq_len = int(getattr(transformer_config, "max_sequence_length", 2048) or 2048)
        hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        qwen_config = create_qwen3_config_from_hf(hf_config, max_seq_len, runner_type=RunnerType.Default)
        tie = bool(getattr(hf_config, "tie_word_embeddings", False))

        logging.info(
            "Building ttml Qwen3 model (hidden=%d, layers=%d)", qwen_config.hidden_size, qwen_config.num_hidden_layers
        )
        tt_model = Qwen3(qwen_config)

        # Load HF weights into the (still replicated) model, then shard.
        hf_state_dict = self._load_hf_state_dict(model_source)
        load_weights_from_hf(tt_model, hf_state_dict, qwen_config, tie_word_embeddings=tie)
        del hf_state_dict

        if self._fsdp_enabled:
            # reshard_after_forward keeps peak memory low by resharding weights
            # between forward and backward and re-gathering them in the backward
            # pre-hook. That re-gather path is sensitive to autograd-callback
            # ordering across a deep block stack + the tied-embedding root
            # wrapper; until that is validated for this model, default to
            # keeping weights gathered between forward and backward (params are
            # still sharded at rest and gradients are still reduce-scattered, so
            # the FSDP grad path is exercised). Set GRPO_QWEN_FSDP_RESHARD=1 to
            # opt into the memory-efficient reshard path.
            reshard = os.environ.get("GRPO_QWEN_FSDP_RESHARD", "0") == "1"
            # The root wrapper manages the tied tok_emb/fc weight (used at both
            # ends of the network) + ln_fc, and its backward window spans the
            # whole block stack. That combination breaks the reshard re-gather
            # contract (the first fc backward closure crashes). Per-block
            # wrapping is the tested-good configuration. Set
            # GRPO_QWEN_FSDP_WRAP_ROOT=0 to shard only the blocks (the bulk of
            # params) and leave embeddings/lm_head/final-norm replicated, which
            # sidesteps the root-wrapper issue while keeping memory-efficient
            # sharding of the layers.
            wrap_root = os.environ.get("GRPO_QWEN_FSDP_WRAP_ROOT", "1") == "1"
            logging.info(
                "Applying FSDP fully_shard across the 'fsdp' axis " "(size=%d, reshard_after_forward=%s, wrap_root=%s)",
                self._mesh.axis_size("fsdp"),
                reshard,
                wrap_root,
            )
            for block in tt_model.blocks:
                ttml.fsdp.fully_shard(block, reshard_after_forward=reshard)
            if wrap_root:
                ttml.fsdp.fully_shard(tt_model, reshard_after_forward=reshard)

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

    def _causal_mask(self, seq_len: int) -> ttml.autograd.Tensor:
        mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))
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
        ctx = self._ctx
        G = ctx.completions_per_prompt
        rows: List[List[int]] = [list(p) for p in prompts for _ in range(G)]
        B = len(rows)
        assert B % self._num_devices == 0, f"batch {B} must be divisible by num_devices {self._num_devices}"

        max_prompt_len = max(len(r) for r in rows)
        tokens_to_complete = min(ctx.max_tokens_to_complete, self._max_seq_len - max_prompt_len)
        tokens_to_complete = max(tokens_to_complete, 0)
        horizon = _round_up(min(max_prompt_len + tokens_to_complete, self._max_seq_len))

        stop_ids = self._get_stop_ids()
        causal_mask = self._causal_mask(horizon)

        generated: List[List[int]] = [[] for _ in range(B)]
        done = [False] * B

        _dbg = os.environ.get("GRPO_QWEN_DEBUG")
        if _dbg:
            print(
                f"[qwen3] generate B={B} prompts={len(prompts)} horizon={horizon} "
                f"tokens_to_complete={tokens_to_complete}",
                flush=True,
            )

        self._model.eval()
        with no_grad():
            for _ in range(tokens_to_complete):
                tokens_np = np.full((B, horizon), ctx._pad_token, dtype=np.uint32)
                pred_pos = [0] * B
                for b in range(B):
                    seq = rows[b]
                    tokens_np[b, : len(seq)] = np.asarray(seq, dtype=np.uint32)
                    pred_pos[b] = len(seq) - 1

                input_tensor = self._tokens_to_tensor(tokens_np, B)
                logits = self._model(input_tensor, causal_mask)

                seed = int(np.random.randint(low=1, high=int(1e7)))
                sampled = ttml.ops.sample.sample_op(logits, ctx.temperature, seed, None)
                sampled_host = ttnn.to_torch(sampled.get_value(), mesh_composer=self._dp_composer)
                sampled_np = sampled_host.reshape(B, 1, horizon, 1).to(int).numpy()

                deallocate_tensors([input_tensor, logits, sampled])

                for b in range(B):
                    if done[b]:
                        continue
                    tok = int(sampled_np[b, 0, pred_pos[b], 0])
                    generated[b].append(tok)
                    rows[b].append(tok)
                    if tok in stop_ids:
                        done[b] = True

                ttml.autograd.AutoContext.get_instance().reset_graph()
                if all(done):
                    break

        deallocate_tensors(causal_mask)

        # Trim trailing stop token(s).
        completions: List[List[int]] = []
        for b in range(B):
            seq = generated[b]
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
        Tp = _round_up(T)

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

        _dbg = os.environ.get("GRPO_QWEN_DEBUG")
        if _dbg:
            print(f"[qwen3] compute_nlog_probs B={B} B_local={B_local} T={T} Tp={Tp} -> forward", flush=True)

        input_tensor = self._tokens_to_tensor(inputs_np, B)
        mask = self._causal_mask(Tp)
        logits = self._model(input_tensor, mask)
        if _dbg:
            print(f"[qwen3] compute_nlog_probs forward done, logits shape={logits.shape()}", flush=True)

        targets_tt = ttml.autograd.Tensor.from_numpy(
            targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )
        nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
        nlog = ttml.ops.reshape.reshape(nlog, [B_local, Tp])
        if _dbg:
            print(f"[qwen3] compute_nlog_probs cross_entropy done", flush=True)

        loss_mask_tt = ttml.autograd.Tensor.from_numpy(
            loss_mask_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, self._dp_mapper
        )

        # NOTE: do NOT deallocate ``input_tensor`` / ``mask`` here. When this runs
        # for the *new* log-probs (grad enabled) they are still referenced by the
        # autograd graph and are needed by ``loss.backward()``; the trainer's
        # ``reset_graph()`` releases them after the backward pass.

        # awliu: deleted in failed attempt to find crash which was actually in loss.backwards()
        return nlog, loss_mask_tt
