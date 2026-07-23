# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for DeepSeek-V4-Flash (Phase 1: functional bringup).

Batch is handled by looping ``DeepSeekV4Model.decode_user`` over user slots and
prefill replays decode one token at a time. Sampling is done on host by vLLM.
"""

from __future__ import annotations

import math
import os

import torch

import ttnn
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
_BLOCK_SIZE = 64
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
    crs = {int(v) for v in config.compress_rates.values()}
    step = math.lcm(32, _BLOCK_SIZE, *crs) if crs else math.lcm(32, _BLOCK_SIZE)
    return ((max_seq_len + step - 1) // step) * step


class DeepseekV4FlashForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }

    def __init__(self, model, lm_head, tokenizer, hf_config, rope, max_seq, max_batch_size):
        self.model = model
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.rope = rope
        self.max_seq = max_seq
        self.max_batch_size = max_batch_size

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
        model.reset_multi_user_paged_caches(max_seq, num_users=max_batch_size, block_size=_BLOCK_SIZE)

        return cls(model, lm_head, tokenizer, config, rope, max_seq, max_batch_size)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        # V4's CSA/HCA compressor caches are per-token projections sized to
        # max_seq // compress_rate, which cannot be expressed in vLLM's uniform
        # (blocks, heads, block_size, head_dim) paged layout. All caches are therefore
        # owned by the model (reset_multi_user_paged_caches) and the kv_cache /
        # page_table arguments vLLM passes into forward are ignored.
        return [None] * num_layers

    def _logits(self, user_id: int, token_id: int, pos: int) -> torch.Tensor:
        hidden = self.model.decode_user(user_id, int(token_id), int(pos), self.rope)
        return ttnn.to_torch(self.lm_head(hidden)).reshape(-1).float()

    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs["tokens"]
        prompt_lens = kwargs["prompt_lens"]
        empty_slots = kwargs.get("empty_slots")
        assert kwargs.get("sampling_params") is None, "on-device sampling is not supported"

        max_padded_len = max(int(l) for l in prompt_lens)
        out = torch.zeros(tokens.shape[0], max_padded_len, self.hf_config.vocab_size)
        for i in range(tokens.shape[0]):
            user_id = int(empty_slots[i]) if empty_slots is not None else i
            logits = None
            for pos in range(int(prompt_lens[i])):
                logits = self._logits(user_id, tokens[i, pos], pos)
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
