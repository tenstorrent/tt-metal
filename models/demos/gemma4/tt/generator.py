# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import determine_device_name


def _patch_model_args(model_args, mesh_device, max_batch_size, max_seq_len, model_path, tokenizer):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_context_len = max_seq_len
    model_args.max_prefill_chunk_size = max_seq_len
    model_args.trace_prefill_supported_seq_lens = [128, 512]
    model_args.mesh_device = mesh_device
    model_args.device_name = determine_device_name(mesh_device)
    model_args.model_name = model_path
    model_args.base_model_name = Path(model_path).name
    model_args.tokenizer = tokenizer
    model_args.processor = None
    model_args.can_enable_trace = (
        lambda prefill_seq_len, num_cached_tokens=0: num_cached_tokens == 0
        and prefill_seq_len in model_args.trace_prefill_supported_seq_lens
    )
    model_args.is_llama_vision = lambda: False
    model_args.encode_prompt = lambda prompt, instruct=False: (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if instruct and getattr(tokenizer, "chat_template", None)
        else tokenizer.encode(prompt, add_special_tokens=True)
    )


class Gemma4Generator(Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gemma4 decode already returns sampled tokens when on-device sampling is enabled.
        self.enable_split_sampling = False

    def _clear_prefill_traces(self):
        for trace_key, trace_id in list(self.trace_id_prefill.items()):
            if trace_id is not None:
                parts = trace_key.split("_")
                model_id = int(parts[1]) if len(parts) >= 2 else 0
                ttnn.release_trace(self.model_args[model_id].mesh_device, trace_id)
            self.trace_id_prefill[trace_key] = None
            self.trace_inputs_prefill[trace_key] = None
            self.trace_output_prefill[trace_key] = None

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device):
        super().warmup_model_prefill(
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            non_greedy_decoding_on_device=non_greedy_decoding_on_device,
        )
        if enable_trace:
            # Gemma4 prefill depends on prompt-specific per-layer inputs.
            # Warmup traces are only for compile coverage and must not be reused
            # for a different prompt at runtime.
            self._clear_prefill_traces()

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        **kwargs,
    ):
        batch_size = tokens.shape[0]
        if batch_size > 1 and enable_trace:
            # Batched prefill uses prompt-specific per-layer inputs; trace capture
            # cannot host-read/write those buffers (TT_FATAL during capture).
            logger.info("Disabling prefill trace for batched prefill (batch_size={})", batch_size)
            enable_trace = False
        return super().prefill_forward_text(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        model_path,
        max_batch_size=1,
        max_seq_len=4096,
        num_layers=None,
        paged_attention_config=None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not hasattr(tokenizer, "stop_tokens"):
            tokenizer.stop_tokens = [tokenizer.eos_token_id]

        model_args, model, tt_kv_cache, _ = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            model_path=model_path,
            create_kv_cache=True,
            paged_attention_config=paged_attention_config,
        )
        _patch_model_args(
            model_args,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            model_path=model_path,
            tokenizer=tokenizer,
        )
        generator = cls([model], [model_args], mesh_device, processor=None, tokenizer=tokenizer)
        return generator, [tt_kv_cache], tokenizer
