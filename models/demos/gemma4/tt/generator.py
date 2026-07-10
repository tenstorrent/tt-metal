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
    resolve_gemma4_prefill_chunk_size,
    resolve_gemma4_prefill_trace_enable,
    warmup_gemma4_model_prefill,
)
from models.tt_transformers.tt.common import get_block_size, get_padded_prefill_len, num_blocks_in_seq
from models.tt_transformers.tt.generator import MAX_BATCHED_PREFILL_SEQ_LEN, SUPPORTED_PREFILL_BATCH_SIZES, Generator
from models.tt_transformers.tt.model_config import determine_device_name

# Same 128k batched-prefill token ceiling as the shared Generator
# (padded_batch × padded_prefill_seq_len).
GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN = MAX_BATCHED_PREFILL_SEQ_LEN


def _load_text_tokenizer(model_path):
    # The 12B tokenizer config can advertise multimodal extra_special_tokens as
    # a list (for example ["<|video|>"]), while this transformers version expects
    # a dict. The text-only demo does not need those model-specific aliases.
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, extra_special_tokens={})
    except (ValueError, OSError, EnvironmentError) as e:
        # The whole Gemma4 family shares one identical tokenizer, but some
        # checkpoints (e.g. gemma-4-31B-it) ship without local tokenizer files.
        # On an offline box AutoTokenizer can't fetch them and raises a misleading
        # "need sentencepiece/tiktoken" error. Fall back to an explicit source
        # (GEMMA4_TOKENIZER) or the 12B tokenizer, which is byte-identical.
        fallback = os.environ.get("GEMMA4_TOKENIZER", "google/gemma-4-12B-it")
        if os.path.normpath(str(fallback)) == os.path.normpath(str(model_path)):
            raise
        logger.warning(
            f"Tokenizer load from '{model_path}' failed ({type(e).__name__}: {e}); "
            f"falling back to the shared Gemma4 tokenizer '{fallback}'. "
            f"Override with GEMMA4_TOKENIZER."
        )
        return AutoTokenizer.from_pretrained(fallback, trust_remote_code=True, extra_special_tokens={})


def _trace_prefill_supported_seq_lens(max_seq_len, has_per_layer_inputs, bounded_sliding=False):
    """Padded prefill buckets for which capturing a prefill trace is correct AND
    a net win for Gemma4.

    Tracing removes host op-dispatch overhead, a meaningful fraction of TTFT
    across short *and* medium/high ISLs (the model body is ~1k dispatched ops per
    prefill). The lm_head is deferred OUTSIDE the trace — the trace returns
    post-norm hidden states and ``process_logits_after_prefill_trace`` runs
    lm_head on just the last-token tile — so the 262k-vocab matmul no longer
    scales with sequence length and these buckets stay a net win at higher ISL.

    Disabled when the model has per-layer inputs (E2B/E4B): prefill uploads
    per-layer tensors via ``ttnn.from_torch`` inside the layer loop, which is not
    allowed during trace capture and would freeze warmup values.

    Bounded sliding is fine: the generator refreshes a persistent
    ``valid_seq_len`` device tensor out-of-trace and ``paged_fill_cache``'s
    writer caps the circular fill at runtime (``get_last_token=-1`` no longer
    skips the cap). ``bounded_sliding`` is kept for call-site compatibility.
    """
    del bounded_sliding  # unlocked by kernel-side valid_seq_len fill cap
    if has_per_layer_inputs:
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
    # Prefill chunking: with bounded sliding (64k+), use generator-level chunks
    # (4096 on QB2 / GEMMA4_GEN_PREFILL_CHUNK) so 128k/256k fit in DRAM — a
    # single full-length chunk OOMs at 256k (~5.6GB scratch). Short-context /
    # unbounded demos keep a single power-of-2 chunk. Multi-chunk long-context
    # output is only partially coherent today ("la la..." fragments); that is a
    # known correctness gap, separate from the DRAM fit requirement.
    # GEMMA4_DEMO_SINGLE_CHUNK=1 forces the old single-chunk path under bounded.
    _force_single = os.environ.get("GEMMA4_DEMO_SINGLE_CHUNK", "0") != "0"
    if bounded_sliding and not _force_single:
        model_args.max_prefill_chunk_size = resolve_gemma4_prefill_chunk_size(
            max_seq_len, mesh_device=mesh_device, non_qb2_default=4096
        )
        logger.info(
            f"Bounded sliding + chunked prefill: max_prefill_chunk_size=" f"{model_args.max_prefill_chunk_size}"
        )
    else:
        _chunk_override = int(os.environ.get("GEMMA4_GEN_PREFILL_CHUNK", "0"))
        model_args.max_prefill_chunk_size = (
            _chunk_override if _chunk_override > 0 else 1 << max(int(max_seq_len - 1).bit_length(), 11)
        )
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


class ChunkedPrefillPageTableGuardMixin:
    """Guard the shared ``Generator.prefill_forward_single_user_text`` against an
    over-wide page table, without modifying ``models/tt_transformers``.

    vLLM's block manager can hand over a page table with more block columns than
    the prompt needs. The shared method then computes a negative pad width
    (``num_blocks_in_seq(...) - page_table.shape[1] < 0``) and crashes on
    ``torch.cat``. Trim the page table to exactly the blocks this prompt needs so
    the base method's pad becomes a no-op. Mirrors the per-model guard in
    ``models/demos/llama3_70b_galaxy/tt/generator.py`` (kept in the gemma4 model
    so the shared generator stays untouched). Mixed into both the demo
    (:class:`Gemma4Generator`) and vLLM (``Gemma4ForCausalLM``) generators, whose
    class hierarchies both reach the shared method but do not share a gemma4 base.
    """

    def _effective_paged_block_size(self, kv_cache):
        """Effective block_size the paged ops address this model's K/V cache with.

        Under vLLM hybrid kv-cache groups the K/V buffer is HMA-shared: a full-attention
        layer views a buffer whose declared head_dim belongs to a sliding layer
        (e.g. declared block 64 / head_dim 256 shared with full-attn head_dim 512 →
        eff_bs 64). ``paged_fill_cache`` / chunked SDPA address that view, so page-table
        math must use it too.

        When every layer owns a matching cache (Option A / non-hybrid), declared
        block_size is correct — do **not** scale by max(head_dim) just because sliding
        and full layers differ. That wrongly halves the block size, doubles the page
        table width, and makes later chunks' ``chunk_page_table`` slices land on the
        zero-pad (clobbering earlier chunks' KV).
        """
        block_size = get_block_size(kv_cache)
        for i, layer in enumerate(getattr(self.model[0], "layers", [])):
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            if cfg is None or i >= len(kv_cache) or kv_cache[i] is None:
                continue
            cache = kv_cache[i][0]
            cache_hd = int(cache.shape[-1])
            if cache_hd != int(cfg.head_dim) and cache_hd > 0:
                # HMA-shared buffer: byte-invariant reinterpret for this layer's head_dim.
                return int(cache.shape[2]) * cache_hd // int(cfg.head_dim)
        return block_size

    def _paged_prefill_block_size(self, kv_cache):
        # Base Generator hook: chunked-prefill page-table padding/slicing uses this so
        # it matches the HMA effective block_size instead of the declared shape.
        return self._effective_paged_block_size(kv_cache)

    def _chunk_prefill_get_last_token(self, *, is_last_chunk, last_token_idx_in_chunk, chunk_size):
        """Per-chunk fill length for Gemma4 multi-chunk prefill.

        Intermediate chunks are fully real tokens (padding lives only in the last
        chunk). The legacy default reuses the last-chunk's short index for every
        chunk, which under-fills intermediate KV and breaks cross-chunk /
        bounded-sliding attention. Return ``-1`` for intermediate chunks so:
          * ``valid_seq_len`` stays unset → the whole chunk is written to KV
          * the model skips the last-token lm_head slice (logits are discarded)
        Keep the real (tile-aligned) index on the last chunk.
        """
        del chunk_size
        if is_last_chunk:
            return (last_token_idx_in_chunk // 32) * 32
        return -1

    def _refresh_prefill_valid_seq_len(self, *, model_id=-1, last_token_idx=None, num_cached_tokens=0):
        """Refresh the persistent bounded-fill cap tensor out of any active trace.

        Traced prefill captures ``paged_fill_cache`` with ``get_last_token=-1``, so
        the host-side ``valid_seq_len`` slice is skipped. The writer kernel instead
        reads ``model.prefill_valid_len_dev``; this method copies the real
        (unpadded) length into that buffer before capture/replay.
        """
        model = self.model[model_id]
        update = getattr(model, "update_prefill_valid_seq_len", None)
        if update is None or getattr(model, "prefill_valid_len_dev", None) is None:
            return
        if last_token_idx is None:
            return
        # Batched traced prefill passes a per-slot list; the shared fill-cap
        # tensor is single-element, so only the single-user (scalar) path is
        # supported here. Batched+bounded is not a current serving shape.
        if isinstance(last_token_idx, (list, tuple)):
            return
        valid_len = int(last_token_idx) - int(num_cached_tokens) + 1
        if valid_len > 0:
            update(valid_len)

    def _easy_trace_prefill(self, *args, **kwargs):
        # Refresh before capture *and* replay so the writer kernel sees the
        # current request's real length (trace binds the buffer address; this
        # updates its contents out-of-trace).
        self._refresh_prefill_valid_seq_len(
            model_id=kwargs.get("model_id", -1),
            last_token_idx=kwargs.get("last_token_idx"),
            num_cached_tokens=kwargs.get("num_cached_tokens", 0),
        )
        # Non-APC traced prefill: ``full_page_table`` is sized to the raw prompt
        # (e.g. 61 blocks for a 3896-token prompt) while the captured device
        # buffer is sized to the padded bucket (64 for 4096). Under bounded
        # sliding, layer-0's cache has only ``sliding_window/block_size`` blocks,
        # so ``_pad_or_create_page_table`` cannot grow the short table up to the
        # captured width and ``copy_host_to_device`` TT_FATALs. Drop
        # ``full_page_table`` so the already-padded ``page_table`` is used.
        # APC (num_cached_tokens > 0) still needs the full mapping for chunk
        # slicing — leave it alone there.
        if not kwargs.get("num_cached_tokens") and kwargs.get("full_page_table") is not None:
            kwargs["full_page_table"] = None
        # Defer lm_head outside the trace: capture returns post-norm hidden
        # states; ``process_logits_after_prefill_trace`` runs lm_head on the
        # last-token tile. Must live on this mixin (not only Gemma4Generator)
        # so the vLLM ``Gemma4ForCausalLM`` path also sets the flag — otherwise
        # ``get_last_token=-1`` (trace default) hits the intermediate-chunk
        # ``return None`` and warmup crashes in process_logits_after_prefill_trace.
        for m in self.model:
            m._prefill_trace_mode = True
        try:
            return super()._easy_trace_prefill(*args, **kwargs)
        finally:
            for m in self.model:
                m._prefill_trace_mode = False

    def prefill_forward_single_user_text(
        self, tokens, page_table=None, *, kv_cache=None, num_cached_tokens=0, **kwargs
    ):
        if page_table is not None and kv_cache is not None:
            block_size = self._effective_paged_block_size(kv_cache)
            needed_blocks = num_blocks_in_seq(tokens.shape[-1] + num_cached_tokens, block_size)
            # Bounded sliding KV cache: sliding layers pass ``cache_position_modulo``
            # to ``paged_fill_cache``, which requires ``page_table`` to span the whole
            # window (``cols * block_size >= cache_position_modulo``) so the circular
            # wrap can address every slot — even for a short prompt whose own block
            # count is far smaller. Trimming to the prompt's ``needed_blocks`` (as the
            # over-wide guard does) would undo the widening applied upstream in
            # ``_get_prefill_user_page_table`` and TT_FATAL the fill. Floor
            # ``needed_blocks`` at the window's block count when any layer runs bounded
            # (read from the per-layer configs so this holds for both the demo and vLLM
            # generators without a generator-level flag).
            modulos = []
            for layer in getattr(self.model[0], "layers", []):
                cfg = getattr(getattr(layer, "self_attn", None), "config", None)
                modulo = getattr(cfg, "cache_position_modulo", None)
                if modulo:
                    modulos.append(modulo)
            if modulos and block_size:
                needed_blocks = max(needed_blocks, num_blocks_in_seq(max(modulos), block_size))
            if page_table.shape[1] > needed_blocks:
                page_table = page_table[:, :needed_blocks]
        # Eager single-chunk still prefers the host-side slice when get_last_token
        # is known; refreshing here keeps the persistent tensor current for any
        # path that falls through to the kernel cap (and for multi-chunk's last
        # chunk if a future hook uses it).
        self._refresh_prefill_valid_seq_len(
            model_id=kwargs.get("model_id", -1),
            last_token_idx=kwargs.get("last_token_idx"),
            num_cached_tokens=num_cached_tokens,
        )
        return super().prefill_forward_single_user_text(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            num_cached_tokens=num_cached_tokens,
            **kwargs,
        )


class Gemma4Generator(ChunkedPrefillPageTableGuardMixin, Generator):
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
        tokenizer = _load_text_tokenizer(model_path)
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
