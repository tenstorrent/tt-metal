# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Gemma4 vLLM integration.

Supported today:
- single TT data-parallel instance;
- batch=1 prefill/decode;
- vLLM-provided paged-attention page tables and KV caches;
- device-side decode sampling when the Gemma4 model was created on a TP mesh
  that supports ``models.common.sampling``.

Continuous batching and prefix caching need separate work because Gemma4 mixes
sliding/global attention cache geometry across layers.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams, format_sampling_params
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.model_config import DEFAULT_GEMMA4_MODEL
from models.tt_transformers.tt.common import PagedAttentionConfig, copy_host_to_device


def _mesh_shape_tuple(mesh_device_or_shape) -> tuple[int, int]:
    if hasattr(mesh_device_or_shape, "shape"):
        return tuple(mesh_device_or_shape.shape)
    return tuple(mesh_device_or_shape)


def layer_kv_cache_shape(
    hf_config,
    layer_idx: int,
    mesh_device_or_shape,
    *,
    max_num_blocks: int,
    block_size: int,
) -> tuple[int, int, int, int]:
    """Return Gemma4's per-layer paged KV cache shape.

    Gemma4 cannot use one vLLM cache shape for every layer: sliding layers use
    ``head_dim=256`` and full-attention layers use ``global_head_dim=512``.
    """

    mesh_shape = _mesh_shape_tuple(mesh_device_or_shape)
    tp = mesh_shape[1] if len(mesh_shape) > 1 else 1
    attn_cfg = Gemma4AttentionConfig(hf_config, layer_idx)
    num_local_kv_heads = 1 if attn_cfg.num_key_value_heads < tp else attn_cfg.num_key_value_heads // tp
    return (max_num_blocks, num_local_kv_heads, block_size, attn_cfg.head_dim)


def validate_page_table(page_table: torch.Tensor, *, block_size: int, max_seq_len: int, batch_size: int) -> None:
    """Validate the vLLM page table metadata Gemma4 forwards to TTNN ops."""

    if not isinstance(page_table, torch.Tensor):
        raise TypeError(f"page_table must be a torch.Tensor, got {type(page_table)!r}")
    if page_table.dtype != torch.int32:
        raise TypeError(f"page_table dtype must be torch.int32, got {page_table.dtype}")
    if page_table.dim() != 2:
        raise ValueError(f"page_table must be rank 2 [batch, blocks], got shape {tuple(page_table.shape)}")
    if page_table.shape[0] != batch_size:
        raise ValueError(f"page_table batch {page_table.shape[0]} does not match batch_size {batch_size}")
    required_blocks = (max_seq_len + block_size - 1) // block_size
    if page_table.shape[1] < required_blocks:
        raise ValueError(
            f"page_table has {page_table.shape[1]} blocks, but max_seq_len={max_seq_len} "
            f"with block_size={block_size} requires at least {required_blocks}"
        )
    if page_table.numel() and int(page_table.min().item()) < 0:
        raise ValueError("page_table contains negative physical block ids")


def _tokens_from_device(tt_tokens, mesh_device, batch_size=1) -> torch.Tensor:
    if isinstance(tt_tokens, tuple):
        tt_tokens = tt_tokens[0]
    if hasattr(mesh_device, "shape"):
        tt_tokens = ttnn.get_device_tensors(tt_tokens)[0]
    return ttnn.to_torch(tt_tokens).reshape(-1)[:batch_size].to(torch.int64)


def _logits_from_device(tt_logits, mesh_device, batch_size=1) -> torch.Tensor:
    if hasattr(mesh_device, "shape"):
        tt_logits = ttnn.get_device_tensors(tt_logits)[0]
    logits = ttnn.to_torch(tt_logits)
    return logits.squeeze(0).squeeze(0)[:batch_size].unsqueeze(1)


def _sample_host(logits: torch.Tensor, sampling_params: SamplingParams | None) -> torch.Tensor:
    if sampling_params is None or sampling_params.temperature == 0 or sampling_params.top_k == 1:
        return torch.argmax(logits, dim=-1).to(torch.int64)
    temp = float(sampling_params.temperature)
    top_k = int(sampling_params.top_k)
    top_p = float(sampling_params.top_p)
    scores = logits.float() / temp
    if top_k > 0:
        values, indices = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1)
        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(-1, indices, values)
    probs = torch.softmax(scores, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs[remove] = 0
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int64)


class Gemma4ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(
        self,
        *,
        model,
        model_args,
        mesh_device,
        state_dict,
        model_path,
        max_batch_size,
        max_seq_len,
        paged_attention_config,
    ):
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.model_path = str(model_path)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.paged_attention_config = paged_attention_config
        self._sampling_params = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=4096,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        if optimizations is not None:
            raise ValueError("Gemma4 vLLM integration does not support custom optimization presets yet")
        if tt_data_parallel != 1:
            raise ValueError("Gemma4 vLLM integration currently supports tt_data_parallel=1 only")
        if max_batch_size != 1:
            raise ValueError("Gemma4 vLLM integration currently supports batch=1 only")

        model_path = os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH")
        model_path = model_path or getattr(hf_config, "_name_or_path", None) or DEFAULT_GEMMA4_MODEL
        page_block_size = int(os.getenv("GEMMA4_PAGE_BLOCK_SIZE", "64"))
        paged_attention_config = PagedAttentionConfig(
            block_size=page_block_size,
            max_num_blocks=max_seq_len // page_block_size,
        )
        model_args, model, _, state_dict = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=n_layers,
            model_path=model_path,
            create_kv_cache=False,
            paged_attention_config=paged_attention_config,
        )
        model_args.max_batch_size = max_batch_size
        model_args.max_seq_len = max_seq_len
        return cls(
            model=model,
            model_args=model_args,
            mesh_device=mesh_device,
            state_dict=state_dict,
            model_path=model_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
        )

    @property
    def cache_path(self):
        return Path(self.model_args.weight_cache_path(self.model_path, ttnn.bfloat16)).parent

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate Gemma4 per-layer paged KV caches for vLLM.

        ``kv_cache_shape`` is accepted for vLLM API compatibility.  Gemma4
        derives actual per-layer shapes from the HF config because full-attention
        layers use a wider head dimension than sliding layers.
        """

        if kv_cache_shape is not None:
            self.paged_attention_config = PagedAttentionConfig(
                block_size=int(kv_cache_shape[2]),
                max_num_blocks=int(kv_cache_shape[0]),
            )
        tt_dtype = ttnn.bfloat16
        caches = []
        for layer_idx in range(num_layers):
            attn_cfg = Gemma4AttentionConfig(self.model_args, layer_idx)
            caches.append(
                init_kv_cache(
                    mesh_device=self.mesh_device,
                    config=attn_cfg,
                    max_batch_size=self.max_batch_size,
                    max_seq_len=self.max_seq_len,
                    paged_attention_config=self.paged_attention_config,
                    cache_dtype=tt_dtype,
                )
            )
        return caches

    def _page_table_to_device(
        self, page_table: torch.Tensor | None, required_seq_len: int | None = None
    ) -> ttnn.Tensor | None:
        if page_table is None:
            return None
        required_seq_len = required_seq_len or self.max_seq_len
        validate_page_table(
            page_table,
            block_size=self.paged_attention_config.block_size,
            max_seq_len=required_seq_len,
            batch_size=page_table.shape[0],
        )
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(self.mesh_device, "shape") else None
        return ttnn.from_torch(
            page_table,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )

    def _apply_sampling_params(self, sampling_params):
        self._sampling_params = sampling_params
        if sampling_params is None or self.model.sampling is None:
            return False
        formatted = format_sampling_params(sampling_params, self.model.sampling.tt_sampling.max_batch_size)
        self.model.sampling.reset_sampling_params(formatted)
        self.model.sampling.seed_manager.reset_seed(formatted.seed, [0])
        self.model.sampling.seed_manager.get_new_values([0])
        return True

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=False,
        sampling_params=None,
        start_pos=None,
        **kwargs,
    ):
        if tokens.shape[0] != 1:
            raise ValueError(f"Gemma4 vLLM prefill currently supports batch=1, got {tokens.shape[0]}")
        if start_pos is not None and any(int(x) != 0 for x in start_pos):
            raise ValueError(f"Gemma4 prefix caching is not supported, got start_pos={start_pos}")
        if enable_trace:
            logger.warning("Gemma4 vLLM prefill tracing is not supported; running untraced prefill")

        prompt_len = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[-1])
        if prompt_len <= 128:
            padded_len = 128
        elif prompt_len <= 1024:
            padded_len = 1024
        else:
            padded_len = 2 ** (prompt_len - 1).bit_length()

        input_ids = tokens[0:1, :prompt_len].to(torch.int64)
        input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
        page_table_tt = self._page_table_to_device(page_table[0:1] if page_table is not None else None, prompt_len)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(self.mesh_device, "shape") else None
        tokens_tt = ttnn.from_torch(
            input_ids_padded.to(torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = self.model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, padded_len, self.model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

        embed_weight = self.state_dict.get(
            "model.language_model.embed_tokens.weight",
            self.state_dict.get("model.embed_tokens.weight"),
        )
        embeds_torch = torch.nn.functional.embedding(input_ids_padded.long(), embed_weight) * self.model.embed_scale
        get_last_token = ((prompt_len - 1) // 32) * 32
        tt_logits = self.model.ttnn_prefill_forward(
            embeds,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            input_ids_torch=input_ids_padded,
            embeds_torch=embeds_torch.float(),
        )
        logits = _logits_from_device(tt_logits, self.mesh_device, batch_size=32)
        tt_logits.deallocate(True)
        last_logits = logits[(prompt_len - 1) - get_last_token : (prompt_len - 1) - get_last_token + 1]
        if sampling_params is not None:
            return _sample_host(last_logits.squeeze(1), sampling_params)
        return last_logits

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=False,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        if tokens.shape[0] != 1:
            raise ValueError(f"Gemma4 vLLM decode currently supports batch=1, got {tokens.shape[0]}")
        if enable_trace:
            logger.warning("Gemma4 vLLM decode tracing is not supported; running untraced decode")

        sample_on_device = self._apply_sampling_params(sampling_params)
        if sampling_params is not None and not sample_on_device:
            raise RuntimeError("Gemma4 vLLM device sampling requested, but model.sampling is unavailable")

        token_step = tokens[:, -1].to(torch.int64)
        if page_table is not None:
            required_seq_len = int(start_pos.max().item()) + 1
            validate_page_table(
                page_table,
                block_size=self.paged_attention_config.block_size,
                max_seq_len=required_seq_len,
                batch_size=tokens.shape[0],
            )
        host_inputs = self.model.prepare_decode_inputs_host(token_step, start_pos.to(torch.int64), page_table)
        embeds_tt, pos_tt, pos_int32_tt, page_table_tt, pli_tt = copy_host_to_device(
            host_inputs,
            mesh_device=self.mesh_device,
        )
        if sample_on_device:
            self.model.sampling.seed_manager.get_new_values([0])
        tt_out, _ = self.model.ttnn_decode_forward(
            embeds_tt,
            pos_tt,
            rot_mat_idxs=pos_int32_tt,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            sampling_on_device=sample_on_device,
            pli_combined=pli_tt,
        )
        if not read_from_device:
            return tt_out
        if sample_on_device:
            return _tokens_from_device(tt_out, self.mesh_device, batch_size=1)
        return _logits_from_device(tt_out, self.mesh_device, batch_size=1)

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        if self._sampling_params is not None:
            tokens = _tokens_from_device(tt_out, self.mesh_device, batch_size=1)
            return (tokens, []) if async_read else tokens
        logits = _logits_from_device(tt_out, self.mesh_device, batch_size=1)
        return (logits, []) if async_read else logits
