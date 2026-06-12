# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from transformers import AutoTokenizer

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import determine_device_name


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
    # Gemma4 prefill is single-user (attention head-split asserts seq batch == 1),
    # so force the Generator's per-user sequential prefill instead of its batched
    # prefill path. Decode is batched separately.
    model_args.disable_batched_prefill = True
    model_args.trace_prefill_supported_seq_lens = _trace_prefill_supported_seq_lens(
        max_seq_len, has_per_layer_inputs, bounded_sliding
    )
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

    def _clear_prefill_traces(self):
        for trace_key, trace_id in list(self.trace_id_prefill.items()):
            if trace_id is not None:
                parts = trace_key.split("_")
                model_id = int(parts[1]) if len(parts) >= 2 else 0
                ttnn.release_trace(self.model_args[model_id].mesh_device, trace_id)
            self.trace_id_prefill[trace_key] = None
            self.trace_inputs_prefill[trace_key] = None
            self.trace_output_prefill[trace_key] = None

    def prefill_forward_text(self, *args, **kwargs):
        if kwargs.get("model_id_warmup") is not None:
            kwargs["warmup_prefill"] = False
        return super().prefill_forward_text(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
        super().warmup_model_prefill(
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
        )
        if enable_trace and bool(getattr(self.model[0], "hidden_size_per_layer_input", 0)):
            # E2B/E4B only: prefill uploads prompt-specific per-layer inputs via
            # ttnn.from_torch inside the layer loop, so a warmup-captured trace
            # would replay stale (warmup) per-layer values. Drop it so the runtime
            # path re-captures per prompt. (Trace is gated OFF for these models
            # today, so this is defensive.) For non-PLI models (31B/12B) the
            # captured graph is prompt-independent — tokens/page_table are
            # refreshed input buffers — so the warmup trace is reused as-is.
            self._clear_prefill_traces()

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
