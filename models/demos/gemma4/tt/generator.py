# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer

from models.demos.gemma4.tt.common import create_tt_model
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import MAX_BATCHED_PREFILL_SEQ_LEN, SUPPORTED_PREFILL_BATCH_SIZES, Generator
from models.tt_transformers.tt.model_config import determine_device_name

# Same 128k batched-prefill token ceiling as the shared Generator
# (padded_batch × padded_prefill_seq_len).
GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN = MAX_BATCHED_PREFILL_SEQ_LEN


def _patch_model_args(model_args, mesh_device, max_batch_size, max_seq_len, model_path, tokenizer):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_context_len = max_seq_len
    model_args.max_prefill_chunk_size = max_seq_len
    model_args.trace_prefill_supported_seq_lens = [128, 512, 1024, 2048]
    model_args.mesh_device = mesh_device
    model_args.device_name = determine_device_name(mesh_device)
    model_args.model_name = model_path
    model_args.base_model_name = Path(model_path).name
    model_args.tokenizer = tokenizer
    model_args.processor = None
    uses_pli = bool(model_args.hidden_size_per_layer_input)
    model_args.can_enable_trace = (
        lambda prefill_seq_len, num_cached_tokens=0: not uses_pli
        and num_cached_tokens == 0
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

    @staticmethod
    def _model_uses_pli(model) -> bool:
        """True for E2B/E4B-style models with per-layer-input embeddings."""
        return bool(getattr(model, "hidden_size_per_layer_input", 0))

    def _maybe_disable_pli_prefill_trace(self, enable_trace: bool, batch_size: int = 1) -> bool:
        """PLI prefill uploads per-layer inputs via ttnn.from_torch inside forward.

        That host-device traffic during trace capture triggers TT_FATAL
        (``Writes are not supported during trace capture``). Decode trace is
        unaffected — PLI is prepared on host and copied in out-of-trace.
        """
        if enable_trace and self._model_uses_pli(self.model[0]):
            logger.info(
                "Disabling prefill trace on PLI model (batch_size={}): "
                "in-forward ttnn.from_torch PLI upload is incompatible with trace capture",
                batch_size,
            )
            return False
        return enable_trace

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device):
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace)
        super().warmup_model_prefill(
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            non_greedy_decoding_on_device=non_greedy_decoding_on_device,
        )

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos: list[int] = None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        batch_size, batch_seq_len = tokens.shape
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)

        prompt_lens_list = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        if not isinstance(prompt_lens_list, list):
            prompt_lens_list = prompt_lens_list.tolist()
        num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
        prefill_seq_lens = [
            get_padded_prefill_len(seq_len - num_cached)
            for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
        ]
        is_harmony = tokens.shape[1] > 0 and int(tokens[0, 0]) == 200006
        can_batch_prefill = (
            page_table is not None
            and batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and all(n == 0 for n in num_cached_per_user)
            and not (getattr(self.model[0], "users_row_sharded", False) and sampling_params is not None and is_harmony)
        )
        if sampling_params is not None and can_batch_prefill:
            sampling_module, sampling_dp, _, _ = self._get_sampling_contract(0)
            if sampling_module is not None and sampling_dp > 1:
                can_batch_prefill = False

        if can_batch_prefill:
            padded_batch = next(
                (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
                self.model_args[0].max_batch_size,
            )
            if (
                padded_batch <= self.model_args[0].max_batch_size
                and padded_batch * prefill_seq_lens[0] >= GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN
            ):
                max_users_per_chunk = min(
                    max(1, GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN // prefill_seq_lens[0]),
                    padded_batch,
                )
                while (
                    max_users_per_chunk > 1
                    and max_users_per_chunk * prefill_seq_lens[0] >= GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN
                ):
                    max_users_per_chunk //= 2

                logger.info(
                    "Chunking Gemma4 batched prefill: batch_size={} padded_batch={} prefill_seq_len={} chunk_size={}",
                    batch_size,
                    padded_batch,
                    prefill_seq_lens[0],
                    max_users_per_chunk,
                )

                merged_output = None
                merged_tokens = None
                merged_log_probs = None
                for chunk_start in range(0, batch_size, max_users_per_chunk):
                    chunk_end = min(chunk_start + max_users_per_chunk, batch_size)
                    chunk_size = chunk_end - chunk_start
                    # Each chunk is an independent batched-prefill call with local
                    # slot indices 0..chunk_size-1; global user identity comes
                    # from the sliced tokens/page_table/prompt_lens.
                    chunk_result = super().prefill_forward_text(
                        tokens=tokens[chunk_start:chunk_end],
                        page_table=page_table[chunk_start:chunk_end] if page_table is not None else None,
                        kv_cache=kv_cache,
                        prompt_lens=prompt_lens_list[chunk_start:chunk_end],
                        empty_slots=list(range(chunk_size)),
                        enable_trace=enable_trace,
                        model_id_warmup=model_id_warmup,
                        sampling_params=sampling_params,
                        start_pos=num_cached_per_user[chunk_start:chunk_end] if start_pos is not None else None,
                        return_hidden_states=return_hidden_states,
                        warmup_prefill=warmup_prefill and chunk_start == 0,
                        **kwargs,
                    )

                    if sampling_params is not None:
                        chunk_tokens, chunk_log_probs = chunk_result
                        if merged_tokens is None:
                            merged_tokens = torch.zeros(
                                (batch_size, *chunk_tokens.shape[1:]),
                                dtype=chunk_tokens.dtype,
                                device=chunk_tokens.device,
                            )
                        merged_tokens[chunk_start:chunk_end] = chunk_tokens

                        if isinstance(chunk_log_probs, tuple):
                            if merged_log_probs is None:
                                merged_log_probs = (
                                    torch.zeros(
                                        (batch_size, *chunk_log_probs[0].shape[1:]),
                                        dtype=chunk_log_probs[0].dtype,
                                        device=chunk_log_probs[0].device,
                                    ),
                                    torch.zeros(
                                        (batch_size, *chunk_log_probs[1].shape[1:]),
                                        dtype=chunk_log_probs[1].dtype,
                                        device=chunk_log_probs[1].device,
                                    ),
                                )
                            merged_log_probs[0][chunk_start:chunk_end] = chunk_log_probs[0]
                            merged_log_probs[1][chunk_start:chunk_end] = chunk_log_probs[1]
                        else:
                            if merged_log_probs is None:
                                merged_log_probs = torch.zeros(
                                    (batch_size, *chunk_log_probs.shape[1:]),
                                    dtype=chunk_log_probs.dtype,
                                    device=chunk_log_probs.device,
                                )
                            merged_log_probs[chunk_start:chunk_end] = chunk_log_probs
                    else:
                        if merged_output is None:
                            merged_output = torch.zeros(
                                (batch_size, *chunk_result.shape[1:]),
                                dtype=chunk_result.dtype,
                                device=chunk_result.device,
                            )
                        merged_output[chunk_start:chunk_end] = chunk_result

                if sampling_params is not None:
                    return merged_tokens, merged_log_probs
                return merged_output

        return super().prefill_forward_text(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
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
