# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer

from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.generator_trace import (
    apply_gemma4_prefill_trace_policy,
    maybe_disable_pli_prefill_trace,
    patch_gemma4_trace_model_args,
    resolve_gemma4_prefill_trace_enable,
    warmup_gemma4_model_prefill,
)
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import MAX_BATCHED_PREFILL_SEQ_LEN, SUPPORTED_PREFILL_BATCH_SIZES, Generator
from models.tt_transformers.tt.model_config import determine_device_name

# Same 128k batched-prefill token ceiling as the shared Generator
# (padded_batch × padded_prefill_seq_len).
GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN = MAX_BATCHED_PREFILL_SEQ_LEN


def _trace_prefill_supported_seq_lens(max_seq_len, has_per_layer_inputs, bounded_sliding):
    """Padded prefill buckets for which capturing a prefill trace is correct AND
    a net win for Gemma4.

    Tracing removes host op-dispatch overhead, a meaningful fraction of TTFT
    across short *and* medium/high ISLs (the model body is ~1k dispatched ops per
    prefill). The lm_head is deferred OUTSIDE the trace — the trace returns
    post-norm hidden states and ``process_logits_after_prefill_trace`` runs
    lm_head on just the last-token tile — so the 262k-vocab matmul no longer
    scales with sequence length and these buckets stay a net win at higher ISL.

    Disabled entirely when:
      * the model has per-layer inputs (E2B/E4B): prefill uploads per-layer
        tensors via ``ttnn.from_torch`` inside the layer loop, which is not
        allowed during trace capture and would freeze warmup values.
      * bounded sliding KV cache is on (long context, >16k): the traced path runs
        with ``get_last_token=-1`` so the host-side ``valid_seq_len`` fill cap is
        skipped, which silently corrupts the circular cache with prompt padding.
        Unlocking >16k traced prefill needs the device-tensor fill cap in
        paged_fill_cache (Phase 2).
    """
    if has_per_layer_inputs or bounded_sliding:
        return []
    override = os.environ.get("GEMMA4_TRACE_PREFILL_SEQ_LENS")
    if override is not None:
        lens = [int(x) for x in override.split(",") if x.strip()]
    else:
        lens = [128, 1024, 4096, 8192, 16384, 32768, 65536]
    return [n for n in lens if n <= max_seq_len]


def _patch_model_args(
    model_args,
    mesh_device,
    max_batch_size,
    max_seq_len,
    model_path,
    tokenizer,
    has_per_layer_inputs=False,
    bounded_sliding=False,
):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_context_len = max_seq_len
    # Gemma4's prefill doesn't yet honor the Generator's per-chunk offset
    # (chunk_start_idx/chunk_page_table are discarded), so multi-chunk prefill
    # mis-handles every chunk after the first (wrong RoPE positions + no
    # cross-chunk attention) — this is the real cause of the >32k garbage. To
    # keep prefill correct we force a SINGLE chunk by making max_prefill_chunk_size
    # a power of 2 >= the padded prompt; the in-call SDPA then chunks internally
    # (correct past 32768). NOTE: a single chunk uses prefill memory proportional
    # to the full prompt, so very long contexts (>~64k) can OOM — proper
    # chunk_start_idx support is the follow-up for bounded-memory long prefill.
    model_args.max_prefill_chunk_size = 1 << max(int(max_seq_len - 1).bit_length(), 11)
    model_args.mesh_device = mesh_device
    model_args.device_name = determine_device_name(mesh_device)
    model_args.model_name = model_path
    model_args.base_model_name = Path(model_path).name
    model_args.tokenizer = tokenizer
    model_args.processor = None
    patch_gemma4_trace_model_args(model_args, prefill_trace_enabled=True)
    model_args.is_llama_vision = lambda: False

    def _encode_prompt(prompt, instruct=False):
        if instruct and getattr(tokenizer, "chat_template", None):
            # tokenize=True can return a BatchEncoding (dict) depending on the
            # transformers/tokenizer version; return_dict=False forces a plain
            # List[int], which is what preprocess_inputs_prefill / torch.tensor
            # expect.
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            # Defensive: some versions still hand back a dict-like with input_ids.
            return out["input_ids"] if isinstance(out, dict) else out
        return tokenizer.encode(prompt, add_special_tokens=True)

    model_args.encode_prompt = _encode_prompt


class Gemma4Generator(Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gemma4 decode already returns sampled tokens when on-device sampling is enabled.
        self.enable_split_sampling = False

    def _maybe_disable_pli_prefill_trace(self, enable_trace: bool, batch_size: int = 1) -> bool:
        return maybe_disable_pli_prefill_trace(enable_trace, self.model[0], batch_size=batch_size)

    def _easy_trace_prefill(self, *args, **kwargs):
        """Capture/replay a prefill trace with the lm_head deferred outside it.

        Sets ``_prefill_trace_mode`` on every model for the duration of the
        super() call so that ``Gemma4Model.__call__`` returns post-norm hidden
        states (skipping the full-sequence lm_head) during trace capture. Replay
        (``execute_trace``) runs no Python model code, so the flag only affects
        the capture/compile passes; the deferred lm_head runs afterward in
        ``process_logits_after_prefill_trace`` on the last-token tile.
        """
        for m in self.model:
            m._prefill_trace_mode = True
        try:
            return super()._easy_trace_prefill(*args, **kwargs)
        finally:
            for m in self.model:
                m._prefill_trace_mode = False

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        greedy_only: bool = False,
    ):
        warmup_gemma4_model_prefill(
            self,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
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
        if model_id_warmup is not None:
            warmup_prefill = False

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
                    chunk_enable_trace = apply_gemma4_prefill_trace_policy(
                        enable_trace,
                        prefill_seq_lens[0],
                        chunk_size,
                        self.model[0],
                    )
                    chunk_result = super().prefill_forward_text(
                        tokens=tokens[chunk_start:chunk_end],
                        page_table=page_table[chunk_start:chunk_end] if page_table is not None else None,
                        kv_cache=kv_cache,
                        prompt_lens=prompt_lens_list[chunk_start:chunk_end],
                        empty_slots=list(range(chunk_size)),
                        enable_trace=chunk_enable_trace,
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

        enable_trace = resolve_gemma4_prefill_trace_enable(
            enable_trace,
            self.model[0],
            self.model_args[0],
            batch_size=batch_size,
            prefill_seq_lens=prefill_seq_lens,
            can_batch_prefill=can_batch_prefill,
        )

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
        bounded_sliding_kv_cache=False,
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
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        _patch_model_args(
            model_args,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            model_path=model_path,
            tokenizer=tokenizer,
            has_per_layer_inputs=bool(getattr(model, "hidden_size_per_layer_input", 0)),
            bounded_sliding=bounded_sliding_kv_cache,
        )
        generator = cls([model], [model_args], mesh_device, processor=None, tokenizer=tokenizer)
        return generator, [tt_kv_cache], tokenizer
