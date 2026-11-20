# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR generator with paged KV and batch > 1, in the vLLM generator API.

Prefill runs the audio encoder per request, splices its output over the audio
placeholder tokens and prefills each user from those embeddings. Decode is the
shared tt_transformers paged, batched, traced decode with on-device sampling.

This module does not import vLLM, so it can be driven directly by tests; the vLLM
class in ``generator_vllm.py`` adds the processor registration on top.
"""
import os

import torch
from safetensors import safe_open

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

    def __init__(self, model, model_args, mesh_device, tokenizer=None, encoder_params=None, embed_table=None):
        super().__init__(model, model_args, mesh_device, tokenizer=tokenizer)
        self.encoder_params = encoder_params
        self.embed_table = embed_table

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
        # Each user's prefill must reach prepare_inputs_prefill on its own to take its embeddings.
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
        with safe_open(os.path.join(text_decoder_dir, "model.safetensors"), "pt") as h:
            embed_table = h.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
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
            embed_table=embed_table,
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
        """Audio embeddings (feat_out_len(T_i), dim) for each audio item, in prompt order."""
        return [
            tt_enc.encode_mel(mel.float(), self.encoder_params, self.mesh_device).to(torch.bfloat16)
            for mel in split_audio_features(features, lengths)
        ]

    def merge_embeds(self, input_ids, audio_embeds):
        """Token embeddings with each audio item's embeddings over its placeholder tokens."""
        x = self.embed_table[input_ids.long()]
        mask = input_ids == AUDIO_TOKEN_ID
        n_audio = sum(int(a.shape[0]) for a in audio_embeds)
        assert int(mask.sum()) == n_audio, f"{int(mask.sum())} audio placeholders, {n_audio} audio embeddings"
        if n_audio:
            x[mask] = torch.cat(audio_embeds).to(x.dtype)
        return x

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
        embeds = []
        for i in range(batch):
            feats = input_audio_features[i] if input_audio_features is not None else None
            flens = audio_feature_lengths[i] if audio_feature_lengths is not None else None
            embeds.append(self.merge_embeds(tokens[i, : lens[i]], self.encode_audio(feats, flens)))

        model = self.model[0]
        model.set_prefill_embeds(embeds)
        try:
            # Traced prefill feeds token ids, and warmup prefills would take the queued
            # embeddings, so this path runs untraced with warmup done beforehand.
            out = self.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                enable_trace=False,
                warmup_prefill=False,
                **kwargs,
            )
            assert model.pending_prefill_embeds() == 0, "a user's prefill did not consume its embeddings"
        finally:
            model.set_prefill_embeds(None)
        return out
