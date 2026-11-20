# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR generator with paged KV and batch > 1, in the vLLM generator API.

Prefill runs the audio encoder per request, splices its output on device over the
audio placeholder tokens and prefills each user from those embeddings. Decode is the
shared tt_transformers paged, batched, traced decode with on-device sampling.

This module does not import vLLM, so it can be driven directly by tests; the vLLM
class in ``generator_vllm.py`` adds the processor registration on top.
"""
import os

import torch

import ttnn
from models.demos.audio.qwen3_asr.tt import audio_encoder as tt_enc
from models.demos.audio.qwen3_asr.tt.model_args import Qwen3ASRModelArgs
from models.demos.audio.qwen3_asr.tt.qwen3_asr_decoder import Qwen3ASRDecoder
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import TensorGroup

AUDIO_TOKEN_ID = 151676


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
        bucket and the last-token slice at each 32-row offset (its offset is part of the
        program), so untraced prefill builds no program once a decode trace exists."""
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
        user (None for a user without audio)."""
        start_pos = kwargs.get("start_pos")
        assert start_pos is None or all(int(s) == 0 for s in start_pos), "prefix caching is not supported"
        kwargs.pop("warmup_prefill", None)
        batch, seq = tokens.shape
        lens = [seq] * batch if prompt_lens is None else [int(n) for n in prompt_lens]
        audio_per_user = []
        for i in range(batch):
            feats = input_audio_features[i] if input_audio_features is not None else None
            flens = audio_feature_lengths[i] if audio_feature_lengths is not None else None
            encoded = self.encode_audio(feats, flens)
            runs = self.audio_runs(tokens[i, : lens[i]])
            if [length for _, length in runs] != [n for _, n in encoded]:
                raise ValueError(
                    f"user {i}: audio placeholder runs {runs} do not match encoder outputs "
                    f"{[n for _, n in encoded]}"
                )
            audio_per_user.append([(off, length, emb) for (off, length), (emb, _) in zip(runs, encoded)])

        model = self.model[0]
        model.set_prefill_audio(audio_per_user)
        try:
            # Traced prefill feeds token ids, and warmup prefills would take the queued
            # audio, so this path runs untraced with warmup done beforehand.
            out = self.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
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
