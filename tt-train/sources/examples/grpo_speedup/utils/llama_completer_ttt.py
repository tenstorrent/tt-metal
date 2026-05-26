# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import ttnn

import ttml
from ttml.common.config import DeviceConfig


def _bf16_decoders_precision(num_decoders: int, model_name: str) -> Any:
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    conf.__name__ = "bf16_attn_bfp8_mlp"
    inst = DecodersPrecision(num_decoders, model_name, decoder_conf=conf)
    inst.__name__ = "bf16_attn_bfp8_mlp"
    return inst


class LlamaGRPOCompleter:
    """Llama completer backed by a tt-transformers ``Transformer``."""

    def setup_device(self, device_config: DeviceConfig) -> Any:
        if device_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(device_config.total_devices())
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
        return autograd_ctx.get_device()

    def __init__(
        self,
        device_config: DeviceConfig,
        model_source: str,
        max_batch_size: int,
        *,
        max_seq_len: int = 2048,
        instruct: bool = True,
        dummy_weights: bool = False,
    ) -> None:
        import torch

        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        self.mesh_device: Any = self.setup_device(device_config)
        self._model_source: str = model_source

        os.environ["HF_MODEL"] = model_source

        self._dtype: Any = ttnn.bfloat16
        self._max_batch_size: int = max_batch_size

        self.model_args = ModelArgs(
            self.mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=lambda ma: _bf16_decoders_precision(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            cache_hf=True,
        )
        self.model_args.lm_head_dtype = ttnn.bfloat16
        self.model_args.ccl_dtype = ttnn.bfloat16
        self.tokenizer: Any = self.model_args.tokenizer

        block_size = 32
        MIN_NUM_BLOCKS = 1024
        required_blocks_per_user = (max_seq_len + block_size - 1) // block_size
        max_num_blocks = max(MIN_NUM_BLOCKS, max_batch_size * required_blocks_per_user)
        blocks_per_user = max_num_blocks // max_batch_size
        max_num_blocks = blocks_per_user * max_batch_size
        self._paged_attention_config = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )
        self._paged_cache_max_seq_len = block_size * blocks_per_user
        self.page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, blocks_per_user)

        if dummy_weights:
            self.model_args.dummy_weights = True
            weight_cache_path: Optional[Path] = None
            cache_dir_tmp: Optional[tempfile.TemporaryDirectory] = None
        else:
            cache_dir_tmp = tempfile.TemporaryDirectory(prefix="ttt_grpo_completer_")
            weight_cache_path = Path(cache_dir_tmp.name)

        state_dict = self.model_args.load_state_dict()

        self.model = Transformer(
            args=self.model_args,
            mesh_device=self.mesh_device,
            dtype=self._dtype,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=self._paged_attention_config,
        )

        self.kv_cache = [l.attention.layer_past for l in self.model.layers]

        self.generator = Generator(
            model=[self.model],
            model_args=[self.model_args],
            mesh_device=self.mesh_device,
            tokenizer=self.tokenizer,
        )

        self._cache_dir_tmp: Optional[tempfile.TemporaryDirectory] = cache_dir_tmp

    def _reset_kv_cache(self) -> None:
        for layer in self.model.layers:
            k_cache, v_cache = layer.attention.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)
        self.generator.prev_page_table = None

    def _stop_token_ids(self) -> set:
        tok = self.tokenizer
        ids: set = set()
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        if tok.pad_token_id is not None:
            ids.add(int(tok.pad_token_id))
        for s in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid >= 0 and tid != tok.unk_token_id:
                ids.add(int(tid))
        return ids

    def _prepare_prompt_batch(
        self, prompts: List[List[int]], max_new_tokens: int
    ) -> tuple[List[List[int]], List[int], int]:
        assert max_new_tokens >= 0, "max_new_tokens must be non-negative"

        active_batch_size = len(prompts)
        assert 0 < active_batch_size <= self._max_batch_size, (
            f"generate() got {active_batch_size} prompts but completer was built with "
            f"max_batch_size={self._max_batch_size}"
        )

        normalized_prompts = [[int(tok) for tok in prompt] for prompt in prompts]
        prompt_lens = [len(p) for p in normalized_prompts]
        assert min(prompt_lens) > 0, "empty prompts are not supported"

        max_prefill_len = self.model_args.max_seq_len - max_new_tokens
        assert (
            max_prefill_len > 0
        ), f"max_new_tokens ({max_new_tokens}) must be smaller than max_seq_len ({self.model_args.max_seq_len})"

        if max(prompt_lens) > max_prefill_len:
            normalized_prompts = [p[-max_prefill_len:] for p in normalized_prompts]
            prompt_lens = [len(p) for p in normalized_prompts]

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len + max_new_tokens <= self.model_args.max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= max_seq_len ({self.model_args.max_seq_len})"
        )
        assert max_prompt_len + max_new_tokens <= self._paged_cache_max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= paged-cache capacity ({self._paged_cache_max_seq_len})"
        )

        if active_batch_size < self._max_batch_size:
            filler_token = self.tokenizer.pad_token_id
            if filler_token is None:
                filler_token = self.tokenizer.eos_token_id
            if filler_token is None or filler_token < 0:
                filler_token = 0
            filler_prompt = [int(filler_token)]
            pad_slots = self._max_batch_size - active_batch_size
            normalized_prompts.extend([filler_prompt] * pad_slots)
            prompt_lens.extend([1] * pad_slots)

        return normalized_prompts, prompt_lens, active_batch_size

    def generate(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> List[List[int]]:
        import torch

        from models.tt_transformers.tt.common import sample_host

        if max_new_tokens == 0:
            return [[] for _ in prompts]

        prompts, prompt_lens, active_batch_size = self._prepare_prompt_batch(prompts, max_new_tokens)
        batch_size = len(prompts)
        max_prompt_len = max(prompt_lens)

        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0
        input_tokens_prefill_pt = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            input_tokens_prefill_pt[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

        if seed is not None:
            torch.manual_seed(seed)

        kv_cache = [self.kv_cache]

        self._reset_kv_cache()

        prefill_logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=None,
            warmup_prefill=False,
            enable_trace=enable_trace,
        )

        _, prefilled_token = sample_host(prefill_logits, temperature=temperature, top_p=top_p, on_host=True)

        completions: List[List[int]] = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size
        for u in range(active_batch_size, batch_size):
            user_done[u] = True
        stop_ids = self._stop_token_ids() if stop_at_eos else set()

        for u in range(batch_size):
            if user_done[u]:
                continue
            tok = int(prefilled_token[u].item())
            if stop_at_eos and tok in stop_ids:
                user_done[u] = True
            else:
                completions[u].append(tok)

        if all(user_done) or max_new_tokens <= 1:
            return completions[:active_batch_size]

        current_pos = torch.tensor(prompt_lens, dtype=torch.int32)
        out_tok = prefilled_token

        for step in range(max_new_tokens - 1):
            decode_logits, _ = self.generator.decode_forward(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                sampling_params=None,
                reset_batch=(step == 0),
            )
            _, sampled = sample_host(decode_logits, temperature=temperature, top_p=top_p, on_host=True)
            out_tok = sampled
            current_pos = current_pos + 1

            for u in range(batch_size):
                if user_done[u]:
                    continue
                tok = int(out_tok[u].item())
                if stop_at_eos and tok in stop_ids:
                    user_done[u] = True
                else:
                    completions[u].append(tok)

            if stop_at_eos and all(user_done):
                break

        return completions[:active_batch_size]
