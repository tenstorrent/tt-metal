# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for DeepSeek-V4-Flash (Phase 1: functional bringup).

Each vLLM slot is one paged decode session on the model (see
``DeepSeekV4Model.activate_session``): the batch is served by activating a slot's
session and replaying the shared decode trace, so slots share one block pool and one
capture. Prefill replays decode one token at a time. Sampling is done on host by vLLM.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
_BLOCK_SIZE = 32
_WEIGHT_DTYPE = ttnn.bfloat4_b


def _build_rope(config, max_seq: int) -> dict:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    dummy = torch.zeros(1, max_seq, 1, dtype=torch.float32)
    rotary = M.DeepseekV4RotaryEmbedding(config).to(torch.float32)

    def half(layer_type: str, position_ids: torch.Tensor):
        cos, sin = rotary(dummy, position_ids=position_ids, layer_type=layer_type)
        return cos[0].contiguous(), sin[0].contiguous()

    positions = torch.arange(max_seq).unsqueeze(0)
    rope = {"main": half("main", positions), "compress": half("compress", positions), "win": {}}
    for cr in sorted({int(v) for v in config.compress_rates.values()}):
        win_pos = (torch.arange(max_seq // cr) * cr).unsqueeze(0)
        rope["win"][cr] = half("compress", win_pos)
    return rope


def _round_max_seq(config, max_seq_len: int) -> int:
    return round_context(max_seq_len, set(config.compress_rates.values()), _BLOCK_SIZE)


class DeepseekV4FlashForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }

    def __init__(self, model, lm_head, tokenizer, hf_config, rope, max_seq, max_batch_size, slots):
        self.model = model
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.rope = rope
        self.max_seq = max_seq
        self.max_batch_size = max_batch_size
        self.slots = slots  # vLLM slot index -> session id

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError(
                "DeepSeek-V4-Flash is pipeline-parallel across the whole mesh; only tt_data_parallel=1 is supported, "
                f"got {tt_data_parallel}"
            )
        from transformers import AutoTokenizer
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        loader = DeepseekV4WeightLoader(os.environ.get("DEEPSEEK_V4_HF_MODEL", _DEFAULT_MODEL_DIR))
        cache_dir = os.environ.get("DEEPSEEK_V4_CACHE_DIR")
        config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
        config._attn_implementation = "eager"
        tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

        max_seq = _round_max_seq(config, max_seq_len)
        rope = _build_rope(config, max_seq)
        max_layers = min(
            int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
        )
        cache = WeightCache(os.path.join(cache_dir, "full_decode", "ttnn")) if cache_dir else None

        model = DeepSeekV4Model(
            config,
            loader,
            mesh_device,
            cache=cache,
            weight_dtype=_WEIGHT_DTYPE,
            max_layers=max_layers,
            use_submeshes=True,
        )
        lm_head = Linear(
            lambda: dequantize_weight(loader.get_tensor("lm_head.weight"), loader.get_scale("lm_head.weight")),
            model.last_device,
            cache.file("lm_head") if cache else None,
            dtype=_WEIGHT_DTYPE,
        )
        # One session per vLLM slot, all sharing the block pool and (lazily captured)
        # decode traces. ``lm_head`` is folded into the last submesh's trace, so a step
        # returns logits without a separate host-dispatched matmul.
        model.prepare_static_decode(
            rope,
            max_seq,
            lm_head=lm_head,
            num_sessions=max_batch_size,
            total_tokens=max_batch_size * max_seq,
            block_size=_BLOCK_SIZE,
        )
        slots = [model.open_session() for _ in range(max_batch_size)]

        return cls(model, lm_head, tokenizer, config, rope, max_seq, max_batch_size, slots)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        # V4's CSA/HCA compressor caches are per-token projections sized to
        # max_seq // compress_rate, which cannot be expressed in vLLM's uniform
        # (blocks, heads, block_size, head_dim) paged layout. All caches are therefore
        # owned by the model (its own block pools) and the kv_cache / page_table
        # arguments vLLM passes into forward are ignored.
        return [None] * num_layers

    def _logits(self, slot: int, token_id: int, pos: int) -> torch.Tensor:
        self.model.activate_session(self.slots[slot])
        logits = self.model.decode_traced(int(token_id), int(pos))  # lm_head is in-trace
        return ttnn.to_torch(logits).reshape(-1).float()

    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs["tokens"]
        prompt_lens = kwargs["prompt_lens"]
        empty_slots = kwargs.get("empty_slots")
        assert kwargs.get("sampling_params") is None, "on-device sampling is not supported"

        max_padded_len = max(int(l) for l in prompt_lens)
        out = torch.zeros(tokens.shape[0], max_padded_len, self.hf_config.vocab_size)
        for i in range(tokens.shape[0]):
            slot = int(empty_slots[i]) if empty_slots is not None else i
            # A slot handed back by vLLM carries the previous sequence's cache; the
            # prefill of a new one starts at position 0, so rewind it first.
            self.model.reset_session(self.slots[slot])
            logits = None
            for pos in range(int(prompt_lens[i])):
                logits = self._logits(slot, tokens[i, pos], pos)
            if logits is not None:
                out[i, :] = logits
        return out

    def decode_forward(self, *args, **kwargs):
        tokens = kwargs["tokens"].squeeze(1)
        start_pos = kwargs["start_pos"]
        assert kwargs.get("sampling_params") is None, "on-device sampling is not supported"

        out = torch.zeros(tokens.shape[0], 1, self.hf_config.vocab_size)
        for slot in range(tokens.shape[0]):
            pos = int(start_pos[slot])
            if pos <= 0 or pos >= self.max_seq:
                continue
            out[slot, 0] = self._logits(slot, tokens[slot], pos)
        return out

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out
