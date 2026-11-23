# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR generator with paged KV and batch > 1, in the vLLM generator API.

Prefill runs the audio encoder per request, splices its output on device over the
audio placeholder tokens and prefills each user from those embeddings. Decode is the
shared tt_transformers paged, batched, traced decode with on-device sampling.

This module does not import vLLM, so it can be driven directly by tests; the vLLM
class in ``generator_vllm.py`` adds the processor registration on top.
"""
import collections
import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.audio.qwen3_asr.tt import audio_encoder as tt_enc
from models.demos.audio.qwen3_asr.tt.model_args import Qwen3ASRModelArgs
from models.demos.audio.qwen3_asr.tt.qwen3_asr_decoder import Qwen3ASRDecoder
from models.tt_transformers.tt.common import get_block_size, get_padded_prefill_len
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import TensorGroup

AUDIO_TOKEN_ID = 151676
# Chunked prefill SDPA takes its Q/K chunk from the largest power of two dividing the start
# (tt_transformers get_attn_sdpa_prefill_program_config); a multiple of 64 keeps the same
# chunk as a prefill from 0.
CONTINUATION_ALIGN = 64
# Recent audio items' mel features (about 256 KB per 5 s), for continued prefills.
MEL_CACHE_ITEMS = 64
# Multimodal fields vLLM gathers for every multimodal model besides the audio features.
MULTIMODAL_KWARGS = (
    "pixel_values",
    "image_grid_thw",
    "mm_placeholder_offsets",
    "mm_placeholder_lengths",
    "mm_identifiers",
)


def split_audio_features(features, lengths):
    """Per-request audio features -> list of (n_mels, T_i) mels, one per audio item.

    Accepts a list of mels, a flat (n_mels, sum T_i) tensor with per-item lengths, or a
    padded (n_items, n_mels, T_max) tensor with per-item lengths."""
    if features is None:
        return []
    if isinstance(features, (list, tuple)):
        return [m for f in features for m in split_audio_features(f, None)]
    if lengths is not None:
        lengths = [int(n) for n in torch.as_tensor(lengths).reshape(-1)]
    if features.dim() == 3:
        if lengths is None:
            return list(features)
        return [features[i, :, :n] for i, n in enumerate(lengths)]
    if lengths is None:
        return [features]
    return list(torch.split(features, lengths, dim=-1))


class Qwen3ASRGenerator(Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        """KV-cache tokens across all users. The shared fallback (131072) does not fit
        next to the weights in one N150's DRAM; 32768 is 16 users at 2048 tokens."""
        return 32_768

    def __init__(self, model, model_args, mesh_device, tokenizer=None, encoder_params=None):
        super().__init__(model, model_args, mesh_device, tokenizer=tokenizer)
        self.encoder_params = encoder_params
        self._mel_cache = collections.OrderedDict()

    @classmethod
    def build(cls, mesh_device, max_batch_size, max_seq_len, snap_dir, text_decoder_dir, optimizations=None):
        """snap_dir: Qwen3-ASR HF snapshot (audio tower). text_decoder_dir: the extracted
        Qwen3-1.7B checkpoint (see demo/transcribe.find_text_decoder)."""
        from models.demos.audio.qwen3_asr.reference.audio_encoder_ref import load_audio_tower_weights

        # ModelArgs takes its checkpoint only from HF_MODEL.
        os.environ["HF_MODEL"] = text_decoder_dir
        args = Qwen3ASRModelArgs(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, optimizations=optimizations
        )
        # Each user's prefill must reach prepare_inputs_prefill on its own to take its audio.
        args.disable_batched_prefill = True
        state_dict = args.load_state_dict()
        model = Qwen3ASRDecoder(
            args,
            ttnn.bfloat16,
            mesh_device,
            state_dict,
            args.weight_cache_path(ttnn.bfloat16),
            use_paged_kv_cache=True,
        )
        encoder_params = tt_enc.preprocess_weights(
            load_audio_tower_weights(snap_dir=snap_dir, dtype=torch.float32), mesh_device
        )
        # Decode traces are captured later; the encoder's per-shape device tensors must exist first.
        tt_enc.warmup(encoder_params, mesh_device)
        return cls(
            [model],
            [args],
            mesh_device,
            tokenizer=args.tokenizer,
            encoder_params=encoder_params,
        )

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """vLLM paged KV: per layer [k, v] of shape (num_blocks, kv_heads, block_size, head_dim)."""
        model = self.model[0]
        layers = []
        for layer_num in range(num_layers):
            kv_dtype = None
            if model.args.optimizations is not None:
                kv_dtype = model.args.optimizations.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
            kv_dtype = ttnn.bfloat8_b if kv_dtype is None else kv_dtype
            zeros = torch.zeros(kv_cache_shape, dtype=dtype)
            layers.append(
                [
                    ttnn.as_tensor(
                        zeros,
                        device=model.mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=kv_dtype,
                        cache_file_name=self.cache_path / f"empty_{kv}cache_paged_attention{tuple(kv_cache_shape)}",
                    )
                    for kv in ["k", "v"]
                ]
            )
        return [layers]

    def encode_audio(self, features, lengths=None):
        """Per audio item, in prompt order: (device embeds (1, 1, bucket rows, dim), number
        of valid rows)."""
        return [
            (
                tt_enc.encode_mel(mel.float(), self.encoder_params, self.mesh_device, to_host=False),
                tt_enc.feat_out_len(mel.shape[-1]),
            )
            for mel in split_audio_features(features, lengths)
        ]

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only=False):
        """Also builds, for every warmed prefill length, the audio splice for each encoder
        bucket, the last-token slice at each 32-row offset (its offset is part of the
        program) and the continued prefill from each aligned start, so untraced prefill
        builds no program once a decode trace exists."""
        if self.already_warmed_up_prefill:
            return
        super().warmup_model_prefill(kv_cache, enable_trace, can_sample_on_device, greedy_only=greedy_only)
        model = self.model[0]
        args = self.model_args[0]
        for seq_len in args.get_warmup_prefill_supported_seq_lens():
            if seq_len > args.max_prefill_chunk_size:
                continue
            hidden = ttnn.from_torch(
                torch.zeros(1, 1, seq_len, args.dim),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for row in range(0, seq_len, 32):
                ttnn.deallocate(ttnn.slice(hidden, (0, 0, row, 0), (1, 1, row + 32, args.dim)))
            ttnn.deallocate(hidden)
            for n in tt_enc.WARM_BUCKETS:
                rows = n * tt_enc.TOKENS_PER_CHUNK
                audio = ttnn.from_torch(
                    torch.zeros(1, 1, rows, args.dim),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                model.set_prefill_audio([[(0, min(rows, seq_len), audio)]])
                try:
                    out = model.prepare_inputs_prefill(torch.zeros(1, seq_len, dtype=torch.long), page_table=None)
                finally:
                    model.set_prefill_audio(None)
                for t in out[:1]:
                    ttnn.deallocate(t)
                ttnn.deallocate(audio)
        self._warmup_continued_prefill(kv_cache)

    def _warmup_continued_prefill(self, kv_cache):
        """A continued prefill's attention reads the page table of its whole prompt, so its
        programs depend on the start as well as the length: run one per aligned start and
        warmed length."""
        if kv_cache is None:
            return
        args = self.model_args[0]
        block = get_block_size(kv_cache[0])
        align = math.lcm(block, CONTINUATION_ALIGN)
        started = time.time()
        count = 0
        for seq_len in args.get_warmup_prefill_supported_seq_lens():
            if seq_len > args.max_prefill_chunk_size or get_padded_prefill_len(seq_len) != seq_len:
                continue
            # The shortest prompt padded to seq_len bounds the starts a request can reach.
            shortest = next(n for n in range(1, seq_len + 1) if get_padded_prefill_len(n) == seq_len)
            for start in range(align, args.max_seq_len - shortest + 1, align):
                count += 1
                total = start + seq_len
                self.prefill_forward_text(
                    torch.zeros(1, total, dtype=torch.long),
                    page_table=torch.zeros(1, math.ceil(total / block), dtype=torch.int32),
                    kv_cache=kv_cache,
                    prompt_lens=torch.tensor([total]),
                    start_pos=[start],
                    empty_slots=[0],
                    enable_trace=False,
                    warmup_prefill=False,
                )
        logger.info(f"Warmed {count} continued prefills in {time.time() - started:.1f}s")

    def _audio_items(self, features, lengths, identifiers):
        """Mel features per audio item, in prompt order. vLLM sends an item's features only
        with the prefill that first covers it, and a continued prefill can restart inside
        that item, so features are kept by vLLM's item identifier (None if unknown)."""
        if not isinstance(features, (list, tuple)):
            return split_audio_features(features, lengths)
        mels = []
        for k, item in enumerate(features):
            key = identifiers[k] if identifiers is not None else None
            if item is None:
                mels.append(self._mel_cache.get(key) if key is not None else None)
                continue
            mels.extend(split_audio_features(item, None))
            if key is not None:
                self._mel_cache[key] = mels[-1]
                self._mel_cache.move_to_end(key)
                while len(self._mel_cache) > MEL_CACHE_ITEMS:
                    self._mel_cache.popitem(last=False)
        return mels

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len, **kwargs):
        """The base trims a user's untraced page table to the padded length of the tokens it
        prefills, which drops the blocks past that length of a continued prefill. Keep
        blocks for the whole prompt (prefill_len counts the cached prefix)."""
        if not kwargs.get("use_batched_prefill", False) and kwargs.get("prefill_seq_len") is not None:
            kwargs["prefill_seq_len"] = max(kwargs["prefill_seq_len"], prefill_len)
        return super()._get_prefill_user_page_table(page_table, kv_cache, prefill_len, **kwargs)

    @staticmethod
    def audio_runs(input_ids):
        """(offset, length) of each run of audio placeholder tokens, in prompt order."""
        is_audio = (input_ids == AUDIO_TOKEN_ID).to(torch.int8)
        edges = torch.diff(torch.nn.functional.pad(is_audio, (1, 1)))
        starts = torch.nonzero(edges == 1).flatten().tolist()
        ends = torch.nonzero(edges == -1).flatten().tolist()
        return [(a, b - a) for a, b in zip(starts, ends)]

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace=False,
        input_audio_features=None,
        audio_feature_lengths=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        """vLLM prefill. input_audio_features / audio_feature_lengths hold one entry per
        user (None for a user without audio), covering every audio item of its prompt.

        start_pos[i] > 0 continues user i on top of its KV cache, as a vLLM streaming
        session does for each new input. A session's cached length is arbitrary, but a
        continuation must start on a KV block and an SDPA chunk boundary, so it restarts
        at the boundary below and recomputes the tokens in between: their KV is rewritten
        unchanged, and only audio from the boundary on is encoded."""
        start_pos = kwargs.pop("start_pos", None)
        kwargs.pop("warmup_prefill", None)
        identifiers = kwargs.get("mm_identifiers")
        # The audio reaches the model through the splice; a continued prefill would pass any
        # other multimodal field on to the decoder forward, which takes none.
        for key in MULTIMODAL_KWARGS:
            kwargs.pop(key, None)
        batch, seq = tokens.shape
        lens = [seq] * batch if prompt_lens is None else [int(n) for n in prompt_lens]
        starts = [0] * batch if start_pos is None else [int(s) for s in start_pos]
        if any(starts):
            align = math.lcm(get_block_size(kv_cache[0]), CONTINUATION_ALIGN)
            starts = [s - s % align for s in starts]
        audio_per_user = []
        for i in range(batch):
            feats = input_audio_features[i] if input_audio_features is not None else None
            flens = audio_feature_lengths[i] if audio_feature_lengths is not None else None
            mels = self._audio_items(feats, flens, identifiers[i] if identifiers is not None else None)
            runs = self.audio_runs(tokens[i, : lens[i]])
            cached = sum(off + length <= starts[i] for off, length in runs)
            out_lens = [
                length if mel is None else tt_enc.feat_out_len(mel.shape[-1]) for (_, length), mel in zip(runs, mels)
            ]
            if len(mels) != len(runs) or [length for _, length in runs] != out_lens:
                raise ValueError(f"user {i}: audio placeholder runs {runs} do not match encoder outputs {out_lens}")
            if any(mel is None for mel in mels[cached:]):
                raise ValueError(f"user {i}: an audio item after position {starts[i]} has no features")
            encoded = self.encode_audio(mels[cached:])
            audio_per_user.append([(off, length, emb) for (off, length), (emb, _) in zip(runs[cached:], encoded)])

        model = self.model[0]
        model.set_prefill_audio(audio_per_user, starts)
        try:
            # Traced prefill feeds token ids, and warmup prefills would take the queued
            # audio, so this path runs untraced with warmup done beforehand.
            out = self.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                start_pos=None if start_pos is None else starts,
                enable_trace=False,
                warmup_prefill=False,
                **kwargs,
            )
            assert model.pending_prefill_audio() == 0, "a user's prefill did not consume its audio"
        finally:
            model.set_prefill_audio(None)
            for items in audio_per_user:
                for _, _, emb in items:
                    ttnn.deallocate(emb)
        return out
