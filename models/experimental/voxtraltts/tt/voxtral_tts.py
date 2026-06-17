# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-level TT orchestration for Voxtral TTS: text backbone, acoustic head, audio tokenizer."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.voxtraltts.reference.audio_tokenizer_ops import (
    audio_tokenizer_encode_tokens_reference,
)
from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import AudioSpecialTokens
from models.experimental.voxtraltts.reference.voxtral_config import (
    DEFAULT_VOXTRAL_MODEL,
    load_voxtral_config,
    parse_csv_ints,
)
from models.experimental.voxtraltts.reference.voxtral_request import (
    compose_speech_request,
)
from models.experimental.voxtraltts.tt.acoustic_model import VoxtralTTAcousticModel
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.text_model import VoxtralTTTextModel, patch_text_model_fp32_rms_norms
from models.experimental.voxtraltts.tt.voxtral_tt_args import (
    _load_safetensors_state_dict,
    voxtral_text_default_optimizations,
)
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    AudioTokenizerOptimizations,
    voxtral_audio_tokenizer_default_optimizations,
)
from models.experimental.voxtraltts.utils.mesh import (
    voxtral_from_torch,
    voxtral_is_multi_device_mesh,
    voxtral_replicate_mesh_mapper,
    voxtral_to_torch_replicated,
    voxtral_tp_shard_dim3_mapper,
)
from models.experimental.voxtraltts.utils.rng import acoustic_fm_noise_seed
from models.tt_transformers.tt.common import PagedAttentionConfig

ACOUSTIC_CFG_ALPHA_DEFAULT = 1.2


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


@dataclass
class VoxtralTTSGenerateOutput:
    """Debuggable output from the full free-running TTS path."""

    waveform: torch.Tensor
    codes_b37t: torch.Tensor
    shifted_codes_t37: torch.Tensor
    hit_end_audio: bool
    # Time-to-first-audio: wall-seconds from generation entry (incl. prompt embed + prefill) until the
    # first acoustic frame's codes are produced. None if no frame was generated.
    first_frame_s: float | None = None
    debug: VoxtralTTSDebugTrace | None = None


@dataclass
class VoxtralTTSDeviceGenerateOutput:
    """Device-resident output from the TT generation path.

    The caller owns all returned TT tensors. Use ``to_host_output`` only at an
    explicit export/test boundary.
    """

    waveform_tt: ttnn.Tensor | None
    waveform_chunks_tt: list[ttnn.Tensor] | None
    codes_b37t_tt: ttnn.Tensor | None
    shifted_codes_t37_tt: ttnn.Tensor | None
    n_frames: int
    expected_samples: int
    hit_end_audio: bool | None
    debug: VoxtralTTSDebugTrace | None = None

    def to_host_output(self) -> VoxtralTTSGenerateOutput:
        if self.waveform_tt is None and not self.waveform_chunks_tt:
            waveform = torch.zeros(1, 1, 0, dtype=torch.float32)
        elif self.waveform_chunks_tt:
            parts = [voxtral_to_torch_replicated(chunk).float().reshape(-1) for chunk in self.waveform_chunks_tt]
            waveform = torch.cat(parts, dim=0)[: self.expected_samples].reshape(1, 1, -1)
        else:
            wav = voxtral_to_torch_replicated(self.waveform_tt).float()
            waveform = wav.reshape(-1)[: self.expected_samples].reshape(1, 1, -1)

        if self.codes_b37t_tt is None:
            codes_b37t = torch.empty((1, 37, 0), dtype=torch.long)
        else:
            codes_b37t = voxtral_to_torch_replicated(self.codes_b37t_tt).long().reshape(1, 37, self.n_frames)

        if self.shifted_codes_t37_tt is None:
            shifted_codes_t37 = torch.empty((0, 37), dtype=torch.long)
        else:
            shifted_codes_t37 = voxtral_to_torch_replicated(self.shifted_codes_t37_tt).long().reshape(self.n_frames, 37)

        return VoxtralTTSGenerateOutput(
            waveform=waveform,
            codes_b37t=codes_b37t,
            shifted_codes_t37=shifted_codes_t37,
            hit_end_audio=bool(self.hit_end_audio),
            debug=self.debug,
        )

    def deallocate(self) -> None:
        for tensor in (self.waveform_tt, self.codes_b37t_tt, self.shifted_codes_t37_tt):
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)
        for tensor in self.waveform_chunks_tt or []:
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)


@dataclass
class VoxtralTTSPipeline:
    """Text + acoustic + audio-tokenizer on TT. CPU: tokenize, voice file, embedding lookup."""

    mesh_device: Any
    model_name_or_path: str
    config: Any
    text: VoxtralTTTextModel
    acoustic: VoxtralTTAcousticModel
    audio_tokenizer: VoxtralTTAudioTokenizer
    audio_tokenizer_sd: dict[str, torch.Tensor]
    tok_embedding_weight: torch.Tensor
    mm_embedding_weight: torch.Tensor
    audio_token_id: int
    end_audio_id: int
    paged_attention_config: Any  # PagedAttentionConfig | None

    @property
    def _downsample_factor(self) -> int:
        cfg = self.config.audio_tokenizer_args
        return cfg.pretransform_patch_size * math.prod(parse_csv_ints(cfg.decoder_convs_strides_str))

    @property
    def _shifted_end_audio_id(self) -> int:
        """Semantic END token after special-token offset (matches shifted code rows)."""
        return self.end_audio_id + self.acoustic._acoustic_special_token_offset

    def _semantic_is_end_audio(self, semantic_token: int) -> bool:
        token = int(semantic_token)
        if token > 32:
            return False
        return token == self.end_audio_id or token == self._shifted_end_audio_id

    @classmethod
    def from_model_name(
        cls,
        mesh_device,
        model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
        *,
        text_max_seq_len: int = 256,
        text_dtype: ttnn.DataType = ttnn.bfloat16,
        text_optimizations=voxtral_text_default_optimizations,
        acoustic_dtype: ttnn.DataType = ttnn.bfloat16,
        tokenizer_dtype: ttnn.DataType = ttnn.bfloat16,
        audio_tokenizer_optimizations: AudioTokenizerOptimizations | None = None,
        use_paged_kv_cache: bool = False,
        paged_block_size: int = 32,
    ) -> "VoxtralTTSPipeline":
        """Build TT TTS pipeline using the production default text optimization profile.

        ``use_paged_kv_cache=True`` enables paged KV attention, which breaks the attention
        CB page from ``seq_len × head_dim`` down to ``block_size × head_dim``.  This removes
        the L1 SRAM CB size constraint that limits ``text_max_seq_len`` to 4096 on Blackhole.
        With paged KV you can safely use ``text_max_seq_len=10000`` for ≤1500-word texts.
        ``paged_block_size`` must be a multiple of 32 (TILE row size); 32 is the safest choice.
        """
        full = _load_safetensors_state_dict(model_name_or_path)
        cfg = load_voxtral_config(model_name_or_path)
        sd_at = extract_audio_tokenizer_state_dict(full)

        tok_emb_w = full.get("mm_audio_embeddings.tok_embeddings.weight")
        if tok_emb_w is None:
            raise RuntimeError("Missing 'mm_audio_embeddings.tok_embeddings.weight' in checkpoint.")
        mm_emb_w = full.get("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight")
        if mm_emb_w is None:
            raise RuntimeError("Missing 'mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight'.")

        paged_cfg = None
        if use_paged_kv_cache:
            import math

            max_num_blocks = math.ceil(text_max_seq_len / paged_block_size)
            paged_cfg = PagedAttentionConfig(block_size=paged_block_size, max_num_blocks=max_num_blocks)

        text = VoxtralTTTextModel.create_from_model_name(
            mesh_device=mesh_device,
            model_name_or_path=model_name_or_path,
            dtype=text_dtype,
            max_batch_size=1,
            max_seq_len=text_max_seq_len,
            preloaded_state_dict=full,
            optimizations=text_optimizations,
            paged_attention_config=paged_cfg,
            use_paged_kv_cache=False,  # False = model manages its own paged KV blocks internally.
            # True is only for vLLM where KV cache is provided externally — skips init_kv_cache
            # and layer_past is never set, causing AttributeError on the first decode step.
        )
        if text.inner.args.num_devices > 1:
            patch_text_model_fp32_rms_norms(
                text,
                mesh_device=mesh_device,
                state_dict=full,
                dim=cfg.dim,
                norm_eps=cfg.norm_eps,
            )
        acoustic = VoxtralTTAcousticModel.create_from_model_name(
            mesh_device,
            model_name_or_path=model_name_or_path,
            dtype=acoustic_dtype,
            preloaded_state_dict=full,
        )
        audio_tokenizer = VoxtralTTAudioTokenizer(
            mesh_device,
            state_dict=sd_at,
            tokenizer_cfg=cfg.audio_tokenizer_args,
            dtype=tokenizer_dtype,
            full_checkpoint=full,
            optimizations=audio_tokenizer_optimizations or voxtral_audio_tokenizer_default_optimizations(),
        )
        return cls(
            mesh_device=mesh_device,
            model_name_or_path=model_name_or_path,
            config=cfg,
            text=text,
            acoustic=acoustic,
            audio_tokenizer=audio_tokenizer,
            audio_tokenizer_sd=sd_at,
            tok_embedding_weight=tok_emb_w.to(dtype=torch.bfloat16),
            mm_embedding_weight=mm_emb_w.to(dtype=torch.bfloat16),
            audio_token_id=int(cfg.audio_model_args.audio_token_id),
            end_audio_id=int(AudioSpecialTokens.id(AudioSpecialTokens.end_audio)),
            paged_attention_config=paged_cfg,
        )

    def _build_page_table(self, max_seq_len: int) -> "ttnn.Tensor | None":
        """Build a sequential page_table ``[1, max_num_blocks]`` for paged KV attention.

        Returns ``None`` when paged KV is disabled (``paged_attention_config is None``).
        The page_table MUST be sized to ``max_num_blocks`` (the full KV block pool),
        NOT just the number of blocks needed for this sequence.  The paged SDPA op
        indexes into the KV block pool using page_table entries — a truncated
        page_table causes a shape mismatch / out-of-bounds access → crash.
        """
        if self.paged_attention_config is None:
            return None
        max_num_blocks = self.paged_attention_config.max_num_blocks
        page_table_host = torch.arange(max_num_blocks, dtype=torch.int32).unsqueeze(0)  # [1, max_num_blocks]
        return ttnn.from_torch(
            page_table_host,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _resolve_model_file(self, filename: str) -> Path:
        p = Path(self.model_name_or_path)
        if p.is_dir():
            return p / filename
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub required for HF model files.") from exc
        return Path(hf_hub_download(self.model_name_or_path, filename=filename))

    def _load_voice_embedding(self, voice: str) -> torch.Tensor:
        """Load voice ``.pt`` once; cached in ``self._voice_emb_cache``."""
        if not hasattr(self, "_voice_emb_cache"):
            self._voice_emb_cache: dict[str, torch.Tensor] = {}
        cached = self._voice_emb_cache.get(voice)
        if cached is not None:
            return cached
        voice_path = self._resolve_model_file(f"voice_embedding/{voice}.pt")
        try:
            voice_emb = torch.load(str(voice_path), map_location="cpu", weights_only=True)
        except TypeError:
            voice_emb = torch.load(str(voice_path), map_location="cpu")
        voice_emb = voice_emb.to(dtype=torch.bfloat16)
        self._voice_emb_cache[voice] = voice_emb
        return voice_emb

    def _build_voice_injected_embeds(
        self,
        prompt_token_ids: list[int],
        voice: str,
    ) -> torch.Tensor:
        """Prompt token IDs + voice injection → ``[S, dim]`` bfloat16 CPU embeddings."""
        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long)
        embeds = F.embedding(input_ids, self.tok_embedding_weight)
        voice_emb = self._load_voice_embedding(voice)
        audio_mask = input_ids == self.audio_token_id
        n_audio = int(audio_mask.sum().item())
        if n_audio != voice_emb.shape[0]:
            raise RuntimeError(
                f"Voice embedding length mismatch: {voice_emb.shape[0]} voice tokens vs "
                f"{n_audio} audio placeholder positions in prompt."
            )
        embeds[audio_mask] = voice_emb
        return embeds

    def _tok_embedding_weight_device(self) -> ttnn.Tensor:
        cached = getattr(self, "_tok_embedding_weight_tt", None)
        if cached is not None and cached.is_allocated():
            return cached
        args = self.text.inner.args
        self._tok_embedding_weight_tt = ttnn.as_tensor(
            self.tok_embedding_weight.unsqueeze(0).unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device,
                dims=(None, 3),
                mesh_shape=args.cluster_shape,
            ),
        )
        return self._tok_embedding_weight_tt

    def _voice_embedding_device(self, voice: str) -> ttnn.Tensor:
        if not hasattr(self, "_voice_emb_tt_cache"):
            self._voice_emb_tt_cache: dict[str, ttnn.Tensor] = {}
        cached = self._voice_emb_tt_cache.get(voice)
        if cached is not None and cached.is_allocated():
            return cached
        voice_emb = self._load_voice_embedding(voice)
        args = self.text.inner.args
        dim = int(voice_emb.shape[1])
        voice_4d = voice_emb.reshape(int(voice_emb.shape[0]), 1, 1, dim).contiguous()
        tp_mapper = voxtral_tp_shard_dim3_mapper(self.mesh_device, args.cluster_shape)
        if tp_mapper is not None:
            voice_tt = ttnn.from_torch(
                voice_4d,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=tp_mapper,
            )
        else:
            voice_tt = voxtral_from_torch(
                voice_4d,
                self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self._voice_emb_tt_cache[voice] = voice_tt
        return voice_tt

    def _embed_prompt_token_chunk_tt(self, token_ids: list[int]) -> ttnn.Tensor:
        """``token_ids`` → ``[n, 1, 1, local_dim]`` ROW_MAJOR via ``ttnn.embedding`` (slice trims padding)."""
        n = int(len(token_ids))
        ids = torch.tensor(token_ids, dtype=torch.uint32).reshape(1, 1, n).contiguous()
        ids_tt = voxtral_from_torch(
            ids,
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        emb = ttnn.embedding(
            ids_tt,
            self._tok_embedding_weight_device(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ids_tt)
        if len(emb.shape) == 3:
            emb_w = int(emb.shape[2])
            emb = ttnn.slice(emb, [0, 0, 0], [int(emb.shape[0]), n, emb_w])
            emb = ttnn.reshape(emb, (n, 1, 1, emb_w))
        elif len(emb.shape) == 4:
            emb_w = int(emb.shape[3])
            emb = ttnn.slice(emb, [0, 0, 0, 0], [int(emb.shape[0]), int(emb.shape[1]), n, emb_w])
            emb = ttnn.permute(emb, (2, 0, 1, 3))
        else:
            raise RuntimeError(f"Unexpected ttnn.embedding rank {len(emb.shape)}: {tuple(emb.shape)}")
        return ttnn.to_layout(emb, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _build_voice_injected_embeds_tt(
        self,
        prompt_token_ids: list[int],
        voice: str,
        *,
        return_cpu_debug: bool = False,
    ) -> tuple[torch.Tensor | None, ttnn.Tensor]:
        """Prompt token IDs + voice injection → device ``[S, 1, 1, dim]`` (TT embedding + concat)."""
        voice_emb = self._load_voice_embedding(voice)
        n_audio = sum(1 for token_id in prompt_token_ids if token_id == self.audio_token_id)
        if n_audio != int(voice_emb.shape[0]):
            raise RuntimeError(
                f"Voice embedding length mismatch: {voice_emb.shape[0]} voice tokens vs "
                f"{n_audio} audio placeholder positions in prompt."
            )

        chunks: list[ttnn.Tensor] = []
        voice_tt = self._voice_embedding_device(voice)
        pos = 0
        voice_pos = 0
        total = len(prompt_token_ids)
        while pos < total:
            is_audio = prompt_token_ids[pos] == self.audio_token_id
            end = pos + 1
            while end < total and (prompt_token_ids[end] == self.audio_token_id) == is_audio:
                end += 1
            if is_audio:
                voice_dim = int(voice_tt.shape[-1])
                chunk = ttnn.slice(voice_tt, [voice_pos, 0, 0, 0], [voice_pos + (end - pos), 1, 1, voice_dim])
                voice_pos += end - pos
            else:
                chunk = self._embed_prompt_token_chunk_tt(prompt_token_ids[pos:end])
            chunks.append(chunk)
            pos = end

        embeds_tt = chunks[0]
        for chunk in chunks[1:]:
            merged = ttnn.concat([embeds_tt, chunk], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(embeds_tt)
            ttnn.deallocate(chunk)
            embeds_tt = merged

        embeds_cpu = self._build_voice_injected_embeds(prompt_token_ids, voice) if return_cpu_debug else None
        return embeds_cpu, embeds_tt

    def _audio_codes_to_mm_embed(self, audio_codes_1_37: torch.Tensor) -> torch.Tensor:
        """``[1, 37]`` codes → ``[dim]`` MM embedding (CPU lookup+sum)."""
        emb = audio_tokenizer_encode_tokens_reference(
            audio_codes_1_37.unsqueeze(-1),
            self.mm_embedding_weight,
            self.config.audio_model_args,
        )
        return emb.squeeze(0).to(dtype=torch.bfloat16)

    def _audio_codes_to_mm_embed_tt(self, audio_codes_1_37: torch.Tensor) -> ttnn.Tensor:
        """``[1, 37]`` codes → device ``[1, 1, 1, dim]`` MM embedding for ``decode_step_from_embeds_tt``."""
        emb = self._audio_codes_to_mm_embed(audio_codes_1_37)
        dim = self.text.inner.args.dim
        return self._mm_embed_host_to_device(emb.reshape(1, 1, 1, dim).contiguous())

    def _mm_embed_host_to_device(self, x_embed_4d: torch.Tensor) -> ttnn.Tensor:
        """Upload ``[1,1,1,dim]`` MM embed; column-shard for TP text, replicate on 1×1."""
        from models.experimental.voxtraltts.demo.decode_trace_2cq import embed_host

        args = self.text.inner.args
        embed_mem_cfg = self.text._decode_embed_input_mem_cfg
        x_bf16 = x_embed_4d.to(dtype=torch.bfloat16).contiguous()
        if args.num_devices > 1:
            host_tt = embed_host(x_bf16, self.mesh_device, tuple(args.cluster_shape))
            out = ttnn.to_device(host_tt, self.mesh_device, embed_mem_cfg)
            if host_tt.is_allocated():
                ttnn.deallocate(host_tt)
            return out
        return ttnn.from_torch(
            x_bf16,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=embed_mem_cfg,
            mesh_mapper=voxtral_replicate_mesh_mapper(self.mesh_device),
        )

    def _shard_mm_embed_4d_for_text_decode(self, emb_4d: ttnn.Tensor) -> ttnn.Tensor:
        """Column-shard replicated ``[1,1,1,dim]`` MM embed for TP text decode buffers."""
        args = self.text.inner.args
        embed_mem_cfg = self.text._decode_embed_input_mem_cfg
        if emb_4d.layout != ttnn.TILE_LAYOUT:
            emb_4d = ttnn.to_layout(emb_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if args.num_devices <= 1:
            return ttnn.to_memory_config(emb_4d, embed_mem_cfg)

        # Audio tokenizer emits replicated full-width MM embeds; take one rank and column-shard for TP text.
        device_tensors = ttnn.get_device_tensors(emb_4d)
        host = ttnn.to_torch(device_tensors[0]).to(torch.bfloat16)
        if emb_4d.is_allocated():
            ttnn.deallocate(emb_4d)
        while host.dim() < 4:
            host = host.unsqueeze(0)
        return self._mm_embed_host_to_device(host.reshape(1, 1, 1, -1).contiguous())

    def _audio_codes_to_mm_embed_device(self, audio_codes_1_37: torch.Tensor) -> ttnn.Tensor:
        """``[1, 37]`` codes → ``[1, 1, 1, dim]`` TT embed; CPU embedding + TT upload.

        ``mm_audio_encode_tokens_summed_forward`` contains TILE rank-change reshapes that
        fail for T=1 (single AR step). Using the proven CPU-side ``F.embedding`` path and a
        single ``ttnn.from_torch`` upload is reliable and produces the correct mesh mapping
        for the text model's ``_decode_single_token_to_tt``.
        """
        emb = self._audio_codes_to_mm_embed(audio_codes_1_37)  # CPU F.embedding + sum → [dim]
        dim = self.text.inner.args.dim
        return self._mm_embed_host_to_device(emb.reshape(1, 1, 1, dim).contiguous())

    def _audio_codes_tt_to_mm_embed_device(self, audio_codes_b1_1_37_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Device ``[B,1,37]`` shifted codes → TT ``[1,1,1,dim]`` MM embedding."""
        codes_rm = ttnn.to_layout(audio_codes_b1_1_37_tt, ttnn.ROW_MAJOR_LAYOUT)
        codes_b37t = ttnn.permute(codes_rm, (0, 2, 1))
        if codes_rm is not audio_codes_b1_1_37_tt and codes_rm.is_allocated():
            ttnn.deallocate(codes_rm)
        emb_td = self.audio_tokenizer.mm_audio_encode_tokens_summed_forward(codes_b37t)
        if codes_b37t.is_allocated():
            ttnn.deallocate(codes_b37t)
        local_dim = int(emb_td.shape[-1])
        emb_4d = ttnn.reshape(emb_td, (1, 1, 1, local_dim))
        if emb_td is not emb_4d and emb_td.is_allocated():
            ttnn.deallocate(emb_td)
        return self._shard_mm_embed_4d_for_text_decode(emb_4d)

    @staticmethod
    def _codes_tt_to_cpu_row(codes_tt: ttnn.Tensor) -> torch.Tensor:
        """ROW_MAJOR readback for ``[B,1,37]`` uint32 code rows (TILE→host corrupts col 0)."""
        if codes_tt.layout != ttnn.ROW_MAJOR_LAYOUT:
            codes_rm = ttnn.to_layout(codes_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            row = voxtral_to_torch_replicated(codes_rm).long().reshape(-1).cpu()
            if codes_rm is not codes_tt and codes_rm.is_allocated():
                ttnn.deallocate(codes_rm)
            return row
        return voxtral_to_torch_replicated(codes_tt).long().reshape(-1).cpu()

    def _codes_tt_is_end_audio(self, codes_tt: ttnn.Tensor) -> bool:
        row = self._codes_tt_to_cpu_row(codes_tt)
        return self._semantic_is_end_audio(int(row[0].item()))

    def _concat_generated_codes_tt(self, generated_codes_tt: list[ttnn.Tensor]) -> ttnn.Tensor:
        """``[B,1,37]`` per-frame rows → ``[B, 37]`` ROW_MAJOR (one bulk readback, no per-frame TILE host)."""
        if not generated_codes_tt:
            raise ValueError("generated_codes_tt is empty")
        n_frames = len(generated_codes_tt)
        shifted_bt37_tt = generated_codes_tt[0]
        for chunk in generated_codes_tt[1:]:
            merged = ttnn.concat([shifted_bt37_tt, chunk], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if shifted_bt37_tt is not generated_codes_tt[0] and shifted_bt37_tt.is_allocated():
                ttnn.deallocate(shifted_bt37_tt)
            if chunk.is_allocated():
                ttnn.deallocate(chunk)
            shifted_bt37_tt = merged
        shifted_t37_tt = ttnn.reshape(shifted_bt37_tt, (n_frames, 37))
        if shifted_t37_tt is not shifted_bt37_tt and shifted_bt37_tt.is_allocated():
            ttnn.deallocate(shifted_bt37_tt)
        return ttnn.to_layout(shifted_t37_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _finalize_generated_codes(
        self, generated_codes_tt: list[ttnn.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Device concat + one ROW_MAJOR readback → trimmed shifted/audio code tables."""
        shifted_rm = self._concat_generated_codes_tt(generated_codes_tt)
        stacked = voxtral_to_torch_replicated(shifted_rm).long().reshape(-1, 37)
        if shifted_rm.is_allocated():
            ttnn.deallocate(shifted_rm)

        sem_col = stacked[:, 0]
        eoa = ((sem_col == self.end_audio_id) | (sem_col == self._shifted_end_audio_id)).nonzero(as_tuple=False)
        hit_end_audio = len(eoa) > 0
        cut = int(eoa[0].item()) if len(eoa) else int(stacked.shape[0])
        shifted_audio_tokens = stacked[:cut]
        audio_tokens = shifted_audio_tokens - 2
        codes_b37t = audio_tokens.T.unsqueeze(0).long()
        return shifted_audio_tokens, audio_tokens, codes_b37t, hit_end_audio

    def _waveform_from_tt_chunks(self, wav_chunks: list[ttnn.Tensor], expected_samples: int) -> torch.Tensor:
        """Explicit host export boundary for TT waveform chunks."""
        parts = [voxtral_to_torch_replicated(chunk).float().reshape(-1) for chunk in wav_chunks]
        for chunk in wav_chunks:
            if chunk.is_allocated():
                ttnn.deallocate(chunk)
        if not parts:
            return torch.zeros(1, 1, 0, dtype=torch.float32)
        return torch.cat(parts, dim=0)[:expected_samples].reshape(1, 1, -1)

    @torch.no_grad()
    def forward_device_resident(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 65536,
        seed: int = 0,
        *,
        fixed_step_count: bool = False,
        include_waveform_decode: bool = True,
        return_device_tensors: bool = False,
        return_debug: bool = False,
    ) -> VoxtralTTSGenerateOutput | VoxtralTTSDeviceGenerateOutput:
        """Same AR loop as :meth:`forward`; final waveform decode on TT device.

        ``return_debug`` only controls trace collection. It must not change numerics;
        staged PCC and ``test_ttnn_trial.py`` share the same compute path.
        """
        torch.manual_seed(seed)
        _t_entry = time.perf_counter()  # for time-to-first-audio (TTFA)
        first_frame_s: float | None = None
        debug = VoxtralTTSDebugTrace() if return_debug else None

        request = compose_speech_request(text, self.model_name_or_path, voice=voice)
        prompt_token_ids: list[int] = request["prompt_token_ids"]
        S_prompt = len(prompt_token_ids)
        # Device prompt embeddings; CPU reference only for explicit debug capture.
        inputs_embeds_cpu, inputs_embeds_tt = self._build_voice_injected_embeds_tt(
            prompt_token_ids,
            voice,
            return_cpu_debug=debug is not None,
        )
        if debug is not None:
            debug.set("embeds.prompt", inputs_embeds_cpu)

        # Production prefill path only; debug must not call collect_layer_hiddens here
        # (that path reads every layer to host and can change the last-token hidden).
        # hidden stays on device (ttnn.Tensor) throughout the AR loop; acoustic model reads it via
        # forward_from_tt. Convert to torch only for debug trace (hidden_tt_to_torch).
        page_table = self._build_page_table(S_prompt + max_tokens)
        cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)
        cfg_scalar = float(cfg_alpha.item())
        generated_codes: list[torch.Tensor] = []
        generated_codes_tt: list[ttnn.Tensor] = []
        # The acoustic FM trace re-binds fixed buffer addresses each replay, which clobbers per-frame
        # device code clones accumulated across the loop: a stored clone reads back as a later step's
        # raw float FM sample (uint32 garbage on 1x1; mostly-silence + clipping spikes on 1x4 once a
        # chunk runs enough frames). Reading each frame's codes to host immediately (tiny [1,37] copy;
        # the FM-trace loop already does a per-step synchronize_device, so this adds no real cost on
        # either mesh) sidesteps the aliasing entirely. The host-accumulated codes are re-uploaded once
        # for the tokenizer/waveform decode.
        _codes_host_accum = True

        def _stash_frame_codes(codes_tt: ttnn.Tensor) -> None:
            """Accumulate one frame's shifted ``[37]`` codes on host (trace-clobber-safe for 1x1 and 1x4)."""
            if _codes_host_accum:
                generated_codes.append(voxtral_to_torch_replicated(codes_tt).long().reshape(-1).cpu())
            else:
                generated_codes_tt.append(ttnn.clone(codes_tt))

        current_pos = S_prompt
        env_noise_rng = os.environ.get("VOXTRAL_ACOUSTIC_NOISE_RNG")
        acoustic_noise_rng = (
            env_noise_rng.strip().lower() if env_noise_rng is not None else ("ttnn" if debug is None else "torch")
        )
        acoustic_noise_scale = _env_float("VOXTRAL_ACOUSTIC_NOISE_SCALE", 1.0)

        # Staged trace replay: text-decode trace (+ acoustic FM trace on multi-device). On BH 1×1
        # submesh, text trace diverges numerically; default trace is OFF there (see decode_trace_enabled).
        # Debug (return_debug) keeps the legacy prefill_from_embeds + forward path for host traces.
        from models.experimental.voxtraltts.demo.decode_trace_2cq import (
            AcousticFMBuffers,
            TracedAcousticFM,
            TracedTextDecode,
            VoxtralDecodeBuffers,
            VoxtralDecodeTrace2CQ,
            decode_trace_2cq_enabled,
            decode_trace_enabled,
            signal_decode_step_done,
            stage_acoustic_inputs,
            stage_decode_inputs_tt,
            stage_prompt_embed_tt,
        )

        _staged_decode = debug is None and decode_trace_enabled()
        # BH 1×1 submesh: text-decode trace is stable; acoustic FM trace diverges (noise / bad codes).
        # 1×4 needs both traces for throughput. Override with VOXTRAL_ACOUSTIC_FM_TRACE=1 on 1×1.
        _acoustic_fm_trace = _staged_decode and (
            self.text.inner.args.num_devices > 1 or os.environ.get("VOXTRAL_ACOUSTIC_FM_TRACE", "0") == "1"
        )
        _trace = _bufs = _dec2cq = None
        _ac_trace = _ac_bufs = None
        _captured = _ac_captured = False
        _cluster_shape = self.text.inner.args.cluster_shape
        _dim = int(self.text.inner.args.dim)
        if _staged_decode:
            _bufs = VoxtralDecodeBuffers.create(
                self.mesh_device,
                _dim,
                _cluster_shape,
                self.text.inner.rope_setup,
                0,  # seed pos 0 (prefill start); overwritten by per-token staging
                torch.zeros(_dim, dtype=torch.bfloat16),
            )
            _trace = TracedTextDecode(self.text, _bufs, page_table, self.mesh_device)
            _dec2cq = (
                VoxtralDecodeTrace2CQ.create(self.mesh_device, _bufs, _cluster_shape)
                if decode_trace_2cq_enabled()
                else None
            )
            if _acoustic_fm_trace:
                # Acoustic FM trace (the 78%/step bottleneck on multi-device).
                _ac_bufs = AcousticFMBuffers.create(self.mesh_device, self.acoustic, 1, _dim)
                _ac_trace = TracedAcousticFM(self.acoustic, _ac_bufs, cfg_scalar, self.mesh_device)

        # ── Prefill ──────────────────────────────────────────────────────────────────────────
        # Staged: per-token DECODE loop over the prompt (capture on token 0, replay thereafter).
        # Debug: standard prefill_from_embeds path.
        if _staged_decode:
            last_hidden_tt = None
            for i in range(S_prompt):
                stage_prompt_embed_tt(_dec2cq, _bufs, self.mesh_device, _cluster_shape, inputs_embeds_tt, i, i, _dim)
                if not _captured:
                    _pf_compile = _trace.compile()
                    if _pf_compile is not None and _pf_compile.is_allocated():
                        ttnn.deallocate(_pf_compile)
                    _trace.capture()
                    _captured = True
                    last_hidden_tt = _trace.hidden_dev
                else:
                    last_hidden_tt = _trace.execute(blocking=False)
                ttnn.synchronize_device(self.mesh_device)
                signal_decode_step_done(_dec2cq)
            if inputs_embeds_tt.is_allocated():
                ttnn.deallocate(inputs_embeds_tt)
        else:
            last_hidden_tt = self.text.prefill_from_embeds(inputs_embeds_tt, start_pos=0, page_table=page_table)
            if inputs_embeds_tt.is_allocated():
                ttnn.deallocate(inputs_embeds_tt)
        if inputs_embeds_cpu is not None:
            del inputs_embeds_cpu
        if debug is not None:
            debug.set("text.prefill.hidden", self.text.hidden_tt_to_torch(last_hidden_tt))

        for step_idx in range(max_tokens):
            next_mm_embed_tt = None
            if debug is not None:
                last_hidden = self.text.hidden_tt_to_torch(last_hidden_tt)
                debug.set(f"step.{step_idx}.text.hidden_in", last_hidden)
            # TT-native acoustic forward — no CPU round-trip per step
            _timing = os.environ.get("VOXTRAL_TRACE_TIMING") == "1"
            if _timing:
                ttnn.synchronize_device(self.mesh_device)
                _t0 = time.perf_counter()
            _noise_seed = acoustic_fm_noise_seed(seed, step_idx)
            if _acoustic_fm_trace:
                # Traced acoustic FM core (the 78%/step bottleneck). Stage hidden+noise into the
                # persistent buffers, replay the Euler-FM trace, then finalize codes (semantic argmax +
                # end-audio mask + concat) outside the trace. ``out_dev`` is persistent → consumed by
                # codes_from_fm here before the next replay.
                stage_acoustic_inputs(
                    self, self.acoustic, _ac_bufs, last_hidden_tt, _noise_seed, noise_rng=acoustic_noise_rng
                )
                if not _ac_captured:
                    _ac_compile = _ac_trace.compile()
                    if _ac_compile is not None and _ac_compile.is_allocated():
                        ttnn.deallocate(_ac_compile)
                    _ac_trace.capture()
                    _ac_captured = True
                    acoustic_tt = _ac_trace.out_dev
                else:
                    acoustic_tt = _ac_trace.execute(blocking=False)
                ttnn.synchronize_device(self.mesh_device)
                # Copy the FM trace output out of the (persistent, trace-region) buffer BEFORE the
                # finalize runs — codes_from_fm's semantic matmul allocates, and with an active trace
                # that would clobber the trace output (TT_THROW "Tensor is not allocated" at concat).
                acoustic_safe = ttnn.clone(acoustic_tt)
                codes_tt = self.acoustic.codes_from_fm(_ac_bufs.llm_dev, acoustic_safe)
                if acoustic_safe.is_allocated():
                    ttnn.deallocate(acoustic_safe)
                _stash_frame_codes(codes_tt)
                if debug is not None:
                    ac_out = voxtral_to_torch_replicated(codes_tt).long().reshape(1, -1)
                    debug.set(f"step.{step_idx}.acoustic.codes", ac_out.squeeze(0))
            elif _staged_decode:
                llm_tile = self._acoustic_hidden_tile_copy(last_hidden_tt)
                noise_tt = self.acoustic.fm_noise_tt(
                    1,
                    _noise_seed,
                    rng=acoustic_noise_rng,
                    scale=acoustic_noise_scale,
                )
                codes_tt = self.acoustic.forward(llm_tile, noise_tt, cfg_scalar)
                ttnn.deallocate(llm_tile)
                ttnn.deallocate(noise_tt)
                _stash_frame_codes(codes_tt)
                if debug is not None:
                    ac_out = voxtral_to_torch_replicated(codes_tt).long().reshape(1, -1)
                    debug.set(f"step.{step_idx}.acoustic.codes", ac_out.squeeze(0))
            else:
                llm_tile = self._acoustic_hidden_tile_copy(last_hidden_tt)
                noise_tt = self.acoustic.fm_noise_tt(
                    1,
                    _noise_seed,
                    rng=acoustic_noise_rng,
                    scale=acoustic_noise_scale,
                )
                codes_tt = self.acoustic.forward(llm_tile, noise_tt, cfg_scalar)
                ttnn.deallocate(llm_tile)
                ttnn.deallocate(noise_tt)
                _stash_frame_codes(codes_tt)
                if debug is not None:
                    ac_out = ttnn.to_torch(codes_tt).long().reshape(1, -1)
                    debug.set(f"step.{step_idx}.acoustic.codes", ac_out.squeeze(0))
            if _timing:
                _t_ac = (time.perf_counter() - _t0) * 1000.0
            if debug is not None:
                sem_tile = self._acoustic_hidden_tile_copy(last_hidden_tt)
                sem_tt = self.acoustic.semantic_logits_tt(sem_tile)
                ttnn.deallocate(sem_tile)
                sem_host = ttnn.to_torch(sem_tt).float()
                ttnn.deallocate(sem_tt)
                while sem_host.dim() > 2:
                    sem_host = sem_host.squeeze(1)
                debug.set(f"step.{step_idx}.acoustic.semantic_logits", sem_host.squeeze(0))

            if first_frame_s is None:
                first_frame_s = time.perf_counter() - _t_entry
            if not fixed_step_count and self._codes_tt_is_end_audio(codes_tt):
                break

            next_mm_embed_tt = self._audio_codes_tt_to_mm_embed_device(codes_tt)

            if _staged_decode:
                # Stage (embed, pos, rot_idxs) into the persistent buffers (CQ1 when 2CQ on), then
                # compile+capture on the first step and replay thereafter. ``hidden_dev`` is a
                # persistent buffer the next iteration's acoustic copy consumes before the next replay.
                if _timing:
                    _ts = time.perf_counter()
                stage_decode_inputs_tt(_dec2cq, _bufs, self.mesh_device, _cluster_shape, next_mm_embed_tt, current_pos)
                if next_mm_embed_tt.is_allocated():
                    ttnn.deallocate(next_mm_embed_tt)
                if _timing:
                    ttnn.synchronize_device(self.mesh_device)
                    _t_stage = (time.perf_counter() - _ts) * 1000.0
                    _te = time.perf_counter()
                if not _captured:
                    out_compile = _trace.compile()
                    if out_compile is not None and out_compile.is_allocated():
                        ttnn.deallocate(out_compile)
                    _trace.capture()
                    _captured = True
                    next_hidden_tt = _trace.hidden_dev
                else:
                    next_hidden_tt = _trace.execute(blocking=False)
                # 2CQ needs a per-step sync/event handoff so CQ1 cannot overwrite trace inputs early.
                # In 1CQ, command ordering plus the next code readback/final sync is sufficient.
                if _dec2cq is not None:
                    ttnn.synchronize_device(self.mesh_device)
                signal_decode_step_done(_dec2cq)
                # Free the previous hidden only if it is NOT the persistent trace output (the prefill
                # hidden on the first traced step is a one-off and must be freed; hidden_dev is reused).
                if (
                    last_hidden_tt is not None
                    and last_hidden_tt is not _trace.hidden_dev
                    and last_hidden_tt.is_allocated()
                ):
                    ttnn.deallocate(last_hidden_tt)
                last_hidden_tt = next_hidden_tt
                if _timing:
                    ttnn.synchronize_device(self.mesh_device)
                    _t_exec = (time.perf_counter() - _te) * 1000.0
                    if step_idx < 10:
                        from loguru import logger as _lg

                        _lg.info(
                            f"[trace-timing] step{step_idx} acoustic={_t_ac:.0f}ms "
                            f"stage={_t_stage:.0f}ms textdecode={_t_exec:.0f}ms"
                        )
            else:
                next_hidden_tt = self.text._decode_single_token_to_tt(
                    next_mm_embed_tt, current_pos, page_table=page_table
                )
                if next_mm_embed_tt.is_allocated():
                    ttnn.deallocate(next_mm_embed_tt)
                if last_hidden_tt.is_allocated():
                    ttnn.deallocate(last_hidden_tt)
                last_hidden_tt = next_hidden_tt
                ttnn.synchronize_device(self.mesh_device)
            if debug is not None:
                debug.set(f"step.{step_idx}.text.hidden_out", self.text.hidden_tt_to_torch(last_hidden_tt))
            current_pos += 1

        ttnn.synchronize_device(self.mesh_device)
        if last_hidden_tt is not None and last_hidden_tt.is_allocated():
            ttnn.deallocate(last_hidden_tt)  # frees the persistent trace hidden_dev too (same tensor)
        if _trace is not None:
            _trace.hidden_dev = None  # already freed above; avoid double-free
            _trace.release()
        if _bufs is not None:
            _bufs.deallocate()
        if _ac_trace is not None:
            _ac_trace.release()
        if _ac_bufs is not None:
            _ac_bufs.deallocate()
        if page_table is not None and page_table.is_allocated():
            ttnn.deallocate(page_table)

        if fixed_step_count:
            if not generated_codes_tt and not generated_codes:
                empty_wav = torch.tensor([], dtype=torch.float32)
                return VoxtralTTSGenerateOutput(
                    waveform=empty_wav,
                    codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                    shifted_codes_t37=torch.empty((0, 37), dtype=torch.long),
                    hit_end_audio=False,
                    debug=debug,
                )

            shifted_bt37_tt = None
            if _codes_host_accum:
                # 1x1: codes accumulated on host (trace-clobber-safe). Rebuild the device code tables
                # with a single upload so the tokenizer/waveform path below is unchanged.
                stacked_host = torch.stack(generated_codes, dim=0)  # [n_frames, 37] shifted
                n_frames = int(stacked_host.shape[0])
                shifted_codes_t37_tt = voxtral_from_torch(
                    stacked_host.to(torch.uint32).contiguous(),
                    self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                audio_2d_tt = voxtral_from_torch(
                    (stacked_host - 2).to(torch.uint32).contiguous(),
                    self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                codes_t_tt = ttnn.permute(audio_2d_tt, (1, 0))
                if audio_2d_tt.is_allocated():
                    ttnn.deallocate(audio_2d_tt)
                codes_b37t_tt = ttnn.reshape(codes_t_tt, (1, 37, n_frames))
                if codes_t_tt is not codes_b37t_tt and codes_t_tt.is_allocated():
                    ttnn.deallocate(codes_t_tt)
            else:
                n_frames = len(generated_codes_tt)
                shifted_bt37_tt = generated_codes_tt[0]
                for chunk in generated_codes_tt[1:]:
                    merged = ttnn.concat([shifted_bt37_tt, chunk], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(shifted_bt37_tt)
                    ttnn.deallocate(chunk)
                    shifted_bt37_tt = merged
                shifted_codes_t37_tt = ttnn.reshape(shifted_bt37_tt, (n_frames, 37))
                audio_bt37_tt = ttnn.subtract(shifted_bt37_tt, 2)
                codes_b37t_tt = ttnn.permute(audio_bt37_tt, (0, 2, 1))
                if audio_bt37_tt.is_allocated():
                    ttnn.deallocate(audio_bt37_tt)

            if include_waveform_decode:
                # Pass a clone: latent_from_codes_tt internally does to_layout(codes, ROW_MAJOR) then
                # deallocates the result — which, when codes_b37t_tt is already row-major, aliases and
                # frees the caller's tensor. We still need codes_b37t_tt below (to_torch / device return),
                # so hand the tokenizer a disposable copy.
                latent_tt = self.audio_tokenizer.latent_from_codes_tt(ttnn.clone(codes_b37t_tt))
                if debug is not None:
                    debug.set("tokenizer.latent", ttnn.to_torch(latent_tt).squeeze(1).float())
                mel_tt = self.audio_tokenizer.decode_latent_to_mel_b1tc(latent_tt)
                if debug is not None:
                    debug.set("tokenizer.mel", ttnn.to_torch(mel_tt).squeeze(1).float())
                ttnn.deallocate(latent_tt)
                wav_chunks = self.audio_tokenizer.pretransform_decode_tt(mel_tt, return_chunks=True)
                ttnn.deallocate(mel_tt)
                if return_device_tensors:
                    assert isinstance(wav_chunks, list)
                    wav_tt = wav_chunks[0] if len(wav_chunks) == 1 else None
                    waveform_chunks_tt = None if len(wav_chunks) == 1 else wav_chunks
                else:
                    assert isinstance(wav_chunks, list)
                    wav_tt = wav_chunks[0] if len(wav_chunks) == 1 else None
                    waveform_chunks_tt = None
                expected_samples = n_frames * self._downsample_factor
            else:
                wav_tt = None
                waveform_chunks_tt = None
                expected_samples = 0

            if return_device_tensors:
                return VoxtralTTSDeviceGenerateOutput(
                    waveform_tt=wav_tt,
                    waveform_chunks_tt=waveform_chunks_tt,
                    codes_b37t_tt=codes_b37t_tt,
                    shifted_codes_t37_tt=shifted_codes_t37_tt,
                    n_frames=n_frames,
                    expected_samples=expected_samples,
                    hit_end_audio=False,
                    debug=debug,
                )

            shifted_audio_tokens = voxtral_to_torch_replicated(shifted_codes_t37_tt).long().reshape(n_frames, 37)
            codes_b37t = voxtral_to_torch_replicated(codes_b37t_tt).long().reshape(1, 37, n_frames)
            if wav_tt is None:
                waveform = self._waveform_from_tt_chunks(wav_chunks, expected_samples)
            else:
                wav = voxtral_to_torch_replicated(wav_tt).float()
                waveform = wav.reshape(-1)[:expected_samples].reshape(1, 1, -1)
                ttnn.deallocate(wav_tt)
            ttnn.deallocate(codes_b37t_tt)
            ttnn.deallocate(shifted_codes_t37_tt)
            if shifted_bt37_tt is not None and shifted_bt37_tt.is_allocated():
                ttnn.deallocate(shifted_bt37_tt)
            if debug is not None:
                debug.set("output.codes", codes_b37t.float())
                debug.set("output.waveform", waveform)
            return VoxtralTTSGenerateOutput(
                waveform=waveform,
                codes_b37t=codes_b37t,
                shifted_codes_t37=shifted_audio_tokens,
                hit_end_audio=False,
                debug=debug,
            )

        if not generated_codes and generated_codes_tt:
            shifted_audio_tokens, audio_tokens, codes_b37t, hit_end_audio = self._finalize_generated_codes(
                generated_codes_tt
            )
            for tensor in generated_codes_tt:
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            generated_codes_tt.clear()
        elif generated_codes:
            stacked = torch.stack(generated_codes, dim=0)
            sem_col = stacked[:, 0]
            eoa = ((sem_col == self.end_audio_id) | (sem_col == self._shifted_end_audio_id)).nonzero(as_tuple=False)
            hit_end_audio = len(eoa) > 0
            cut = int(eoa[0].item()) if len(eoa) else stacked.shape[0]
            shifted_audio_tokens = stacked[:cut]
            audio_tokens = shifted_audio_tokens - 2
            codes_b37t = audio_tokens.T.unsqueeze(0).long()
            for tensor in generated_codes_tt:
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
        else:
            empty_wav = torch.tensor([], dtype=torch.float32)
            return VoxtralTTSGenerateOutput(
                waveform=empty_wav,
                codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                shifted_codes_t37=torch.empty((0, 37), dtype=torch.long),
                hit_end_audio=False,
                first_frame_s=first_frame_s,
                debug=debug,
            )

        if audio_tokens.numel() == 0:
            empty_wav = torch.tensor([], dtype=torch.float32)
            return VoxtralTTSGenerateOutput(
                waveform=empty_wav,
                codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                shifted_codes_t37=shifted_audio_tokens.long(),
                hit_end_audio=hit_end_audio,
                first_frame_s=first_frame_s,
                debug=debug,
            )

        if include_waveform_decode:
            ttnn.synchronize_device(self.mesh_device)
            codes_2d_tt = voxtral_from_torch(
                audio_tokens.to(torch.uint32).contiguous(),
                self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            codes_t_tt = ttnn.permute(codes_2d_tt, (1, 0))
            if codes_2d_tt.is_allocated():
                ttnn.deallocate(codes_2d_tt)
            codes_b37t_tt = ttnn.reshape(codes_t_tt, (1, 37, int(audio_tokens.shape[0])))
            if codes_t_tt is not codes_b37t_tt and codes_t_tt.is_allocated():
                ttnn.deallocate(codes_t_tt)

            latent_tt = self.audio_tokenizer.latent_from_codes_tt(codes_b37t_tt)
            if debug is not None:
                debug.set("tokenizer.latent", ttnn.to_torch(latent_tt).squeeze(1).float())
            mel_tt = self.audio_tokenizer.decode_latent_to_mel_b1tc(latent_tt)
            if debug is not None:
                debug.set("tokenizer.mel", ttnn.to_torch(mel_tt).squeeze(1).float())
            ttnn.deallocate(latent_tt)
            wav_chunks = self.audio_tokenizer.pretransform_decode_tt(mel_tt, return_chunks=True)
            ttnn.deallocate(mel_tt)
            if return_device_tensors:
                assert isinstance(wav_chunks, list)
                wav_tt = wav_chunks[0] if len(wav_chunks) == 1 else None
                waveform_chunks_tt = None if len(wav_chunks) == 1 else wav_chunks
            else:
                assert isinstance(wav_chunks, list)
                wav_tt = wav_chunks[0] if len(wav_chunks) == 1 else None
                waveform_chunks_tt = None

            expected_samples = audio_tokens.shape[0] * self._downsample_factor
            if return_device_tensors:
                shifted_tt = voxtral_from_torch(
                    shifted_audio_tokens.to(torch.uint32).contiguous(),
                    self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                return VoxtralTTSDeviceGenerateOutput(
                    waveform_tt=wav_tt,
                    waveform_chunks_tt=waveform_chunks_tt,
                    codes_b37t_tt=codes_b37t_tt,
                    shifted_codes_t37_tt=shifted_tt,
                    n_frames=int(audio_tokens.shape[0]),
                    expected_samples=int(expected_samples),
                    hit_end_audio=hit_end_audio,
                    debug=debug,
                )

            if wav_tt is None:
                waveform = self._waveform_from_tt_chunks(wav_chunks, expected_samples)
            else:
                wav = voxtral_to_torch_replicated(wav_tt).float()
                waveform = wav.reshape(-1)[:expected_samples].reshape(1, 1, -1)
                if wav_tt.is_allocated():
                    ttnn.deallocate(wav_tt)
            if codes_b37t_tt.is_allocated():
                ttnn.deallocate(codes_b37t_tt)
        else:
            if return_device_tensors:
                return VoxtralTTSDeviceGenerateOutput(
                    waveform_tt=None,
                    waveform_chunks_tt=None,
                    codes_b37t_tt=None,
                    shifted_codes_t37_tt=None,
                    n_frames=int(audio_tokens.shape[0]),
                    expected_samples=0,
                    hit_end_audio=hit_end_audio,
                    debug=debug,
                )
            waveform = torch.zeros(1, 1, 0, dtype=torch.float32)

        if debug is not None:
            debug.set("output.codes", codes_b37t.float())
            debug.set("output.waveform", waveform)

        return VoxtralTTSGenerateOutput(
            waveform=waveform,
            codes_b37t=codes_b37t,
            shifted_codes_t37=shifted_audio_tokens.long(),
            hit_end_audio=hit_end_audio,
            first_frame_s=first_frame_s,
            debug=debug,
        )

    __call__ = forward_device_resident

    @torch.no_grad()
    def generate(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 2500,
        seed: int = 0,
        *,
        fixed_step_count: bool = False,
        include_waveform_decode: bool = True,
        return_device_tensors: bool = False,
    ) -> torch.Tensor:
        """Compatibility wrapper returning only the final waveform."""
        out = self.forward_device_resident(
            text=text,
            voice=voice,
            max_tokens=max_tokens,
            seed=seed,
            fixed_step_count=fixed_step_count,
            include_waveform_decode=include_waveform_decode,
            return_device_tensors=return_device_tensors,
        )
        if isinstance(out, VoxtralTTSDeviceGenerateOutput):
            host = out.to_host_output()
            out.deallocate()
            return host.waveform
        return out.waveform

    @torch.no_grad()
    def generate_with_codes(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 2500,
        seed: int = 0,
        *,
        fixed_step_count: bool = False,
        include_waveform_decode: bool = True,
        return_device_tensors: bool = False,
    ) -> VoxtralTTSGenerateOutput | VoxtralTTSDeviceGenerateOutput:
        """Run the device-resident path (:meth:`forward_device_resident`), returning generated codes."""
        return self.forward_device_resident(
            text=text,
            voice=voice,
            max_tokens=max_tokens,
            seed=seed,
            fixed_step_count=fixed_step_count,
            include_waveform_decode=include_waveform_decode,
            return_device_tensors=return_device_tensors,
        )

    def decode_waveform_from_codes_tt(self, codes_b37t: torch.Tensor) -> torch.Tensor:
        """``[B,37,T]`` int CPU codes → float32 waveform (latent + decoder + pretransform on TT)."""
        codes_tt = voxtral_from_torch(
            codes_b37t.to(torch.uint32).contiguous(),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        latent_tt = self.audio_tokenizer.latent_from_codes_tt(codes_tt)
        if codes_tt.is_allocated():
            ttnn.deallocate(codes_tt)
        mel_tt = self.audio_tokenizer.decode_latent_to_mel_b1tc(latent_tt)
        ttnn.deallocate(latent_tt)
        wav_tt = self.audio_tokenizer.pretransform_decode_tt(mel_tt)
        ttnn.deallocate(mel_tt)
        wav = voxtral_to_torch_replicated(wav_tt).float()
        if wav_tt.is_allocated():
            ttnn.deallocate(wav_tt)
        return wav

    def _acoustic_hidden_tile_copy(self, llm_hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Prepare llm hidden → [bsz, 1, dim] TILE DRAM for acoustic.forward.

        Two paths depending on the input rank:

        * 3D input (trace path, tensor already [bsz,1,dim] DRAM TILE):
          ``ttnn.clone`` — always works because source and dest are same layout type.

        * 4D input (AR loop, [1,1,1,dim] L1-sharded TILE from decode):
          TTNN has no 4D→3D reshape op, and ``ttnn.clone`` across sharded/interleaved
          boundaries is unsupported.  Use a minimal ``to_torch → reshape → from_torch``
          (8 KB round-trip) to produce the correct 3D shape.
        """
        if len(llm_hidden_tt.shape) == 4:
            # AR loop decode path — must go via CPU for the 4D→3D shape change.
            if voxtral_is_multi_device_mesh(self.mesh_device):
                host_vec = self.text.hidden_tt_to_torch(llm_hidden_tt)  # [dim]
                return voxtral_from_torch(
                    host_vec.unsqueeze(0).unsqueeze(1),  # [1, 1, dim]
                    self.mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            # TILE_LAYOUT pads H to 32, so to_torch returns [bsz, 1, 32, dim] not [bsz, 1, 1, dim].
            # Slice host[: , 0, 0, :] to extract the single real token row → [bsz, dim].
            work = ttnn.to_memory_config(llm_hidden_tt, ttnn.DRAM_MEMORY_CONFIG)
            host = ttnn.to_torch(work).to(torch.bfloat16)
            ttnn.deallocate(work)
            bsz = int(host.shape[0])
            dim = int(host.shape[-1])
            token_vec = host[:, 0, 0, :dim]  # [bsz, dim] — first tile row is the real data
            return voxtral_from_torch(
                token_vec.unsqueeze(1),  # [bsz, 1, dim]
                self.mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Trace path — tensor is already [bsz, 1, dim]; gather + replicate on multi-device meshes.
        if voxtral_is_multi_device_mesh(self.mesh_device):
            host_vec = self.text.hidden_tt_to_torch(llm_hidden_tt)
            return voxtral_from_torch(
                host_vec.unsqueeze(0).unsqueeze(1),
                self.mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        work = ttnn.clone(llm_hidden_tt)
        if work.layout != ttnn.TILE_LAYOUT:
            tile_hidden = ttnn.to_layout(work, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(work)
            return tile_hidden
        return work

    def _stage_acoustic_hidden_to_fm_buffer(self, llm_hidden_tt: ttnn.Tensor, dest: ttnn.Tensor) -> None:
        """Device copy of text hidden into ``[bsz, 1, dim]`` FM trace buffer (no host round-trip)."""
        tile = self._acoustic_hidden_tile_copy(llm_hidden_tt)
        shape = tuple(tile.shape)
        if len(shape) == 4:
            dim = int(shape[-1])
            sliced = ttnn.slice(tile, [0, 0, 0, 0], [shape[0], 1, 1, dim])
            src = ttnn.reshape(sliced, (shape[0], 1, dim))
            if sliced is not tile and sliced.is_allocated():
                ttnn.deallocate(sliced)
        elif len(shape) == 3:
            src = tile
        else:
            raise RuntimeError(f"Unexpected hidden rank {len(shape)} for acoustic staging: {shape}")
        if src.layout != ttnn.TILE_LAYOUT:
            src_tile = ttnn.to_layout(src, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if src is not tile and src.is_allocated():
                ttnn.deallocate(src)
            src = src_tile
        ttnn.copy(src, dest)
        if src is not tile and src.is_allocated():
            ttnn.deallocate(src)
        if tile is not llm_hidden_tt and tile.is_allocated():
            ttnn.deallocate(tile)

    def _acoustic_hidden_host_torch(self, llm_hidden_tt: ttnn.Tensor) -> torch.Tensor:
        """Host ``[bsz, 1, dim]`` bf16 hidden tile for staging into the acoustic FM trace buffer."""
        work = ttnn.to_memory_config(llm_hidden_tt, ttnn.DRAM_MEMORY_CONFIG)
        host = ttnn.to_torch(work).to(torch.bfloat16)
        ttnn.deallocate(work)
        dim = int(host.shape[-1])
        if host.dim() == 4:
            token_vec = host[:, 0, 0, :dim]
        elif host.dim() == 3:
            token_vec = host[:, 0, :dim]
        else:
            raise RuntimeError(f"Unexpected hidden rank {host.dim()} for acoustic staging: {tuple(host.shape)}")
        return token_vec.unsqueeze(1).contiguous()

    def forward_tts_generation_trace(
        self,
        steps: list[Any],
        *,
        cfg_scalar: float = ACOUSTIC_CFG_ALPHA_DEFAULT,
    ) -> ttnn.Tensor:
        """Fixed-step traced loop; inputs materialized before capture. Returns last-step acoustic codes."""
        last_codes_tt = None
        for step in steps:
            llm_tile = self._acoustic_hidden_tile_copy(step.llm_hidden_tt)
            codes_tt = self.acoustic.forward(llm_tile, step.noise_tt, cfg_scalar)
            if llm_tile.is_allocated():
                ttnn.deallocate(llm_tile)
            self.text.decode_step_from_embeds_tt(
                step.text_step.x_embed_tt,
                step.text_step.current_pos_tt,
                step.text_step.rot_mats_global,
                step.text_step.rot_mats_local,
                step.text_step.page_table,
            )
            if last_codes_tt is not None and last_codes_tt.is_allocated():
                ttnn.deallocate(last_codes_tt)
            last_codes_tt = codes_tt
        assert last_codes_tt is not None
        return last_codes_tt

    def acoustic_codes_forward(
        self,
        llm_hidden_bf16: torch.Tensor,
        cfg_alpha: torch.Tensor | None = None,
        *,
        noise_seed: int = 0,
    ) -> torch.Tensor:
        """Host wrapper: ``from_torch`` → acoustic ``forward`` → ``to_torch``."""
        if cfg_alpha is None:
            cfg_alpha = torch.tensor(
                ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=llm_hidden_bf16.dtype, device=llm_hidden_bf16.device
            )
        bsz = llm_hidden_bf16.shape[0]
        cfg_scalar = float(cfg_alpha.flatten()[0].item())
        llm_tt = ttnn.from_torch(
            llm_hidden_bf16.unsqueeze(1).to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        noise_tt = self.acoustic.fm_noise_tt(bsz, noise_seed)
        codes_tt = self.acoustic.forward(llm_tt, noise_tt, cfg_scalar)
        ttnn.deallocate(llm_tt)
        ttnn.deallocate(noise_tt)
        codes = voxtral_to_torch_replicated(codes_tt).long().reshape(bsz, -1)
        ttnn.deallocate(codes_tt)
        return codes

    def acoustic_pre_round_scaled_forward(
        self,
        llm_hidden_bf16: torch.Tensor,
        cfg_alpha: torch.Tensor | None = None,
        *,
        noise_seed: int = 0,
    ) -> torch.Tensor:
        """Continuous pre-``round`` FSQ values ``[bsz, n_acoustic]`` from the acoustic FM.

        Numerical-accuracy probe: ``round`` of this tensor is the acoustic code, so its PCC vs the
        reference (same hidden + same host-drawn noise) measures op accuracy independently of FSQ
        code flips at the rounding boundaries.
        """
        if cfg_alpha is None:
            cfg_alpha = torch.tensor(
                ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=llm_hidden_bf16.dtype, device=llm_hidden_bf16.device
            )
        bsz = llm_hidden_bf16.shape[0]
        cfg_scalar = float(cfg_alpha.flatten()[0].item())
        llm_tt = ttnn.from_torch(
            llm_hidden_bf16.unsqueeze(1).to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        noise_tt = self.acoustic.fm_noise_tt(bsz, noise_seed)
        scaled_tt = self.acoustic.fm_pre_round_scaled_codes_tt(llm_tt, noise_tt, cfg_scalar)
        ttnn.deallocate(llm_tt)
        ttnn.deallocate(noise_tt)
        scaled = voxtral_to_torch_replicated(scaled_tt).float().reshape(bsz, -1)
        ttnn.deallocate(scaled_tt)
        return scaled

    def cleanup_all(self) -> None:
        """Drop TT_CCL semaphores so profiler-enabled ``close_device`` does not segfault."""
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception:
            pass

        self._release_tt_ccl_handles()

        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception:
            pass

    def _release_tt_ccl_handles(self) -> None:
        """Clear ``TT_CCL`` ``GlobalSemaphore`` lists on the text transformer."""
        inner = getattr(self.text, "inner", None) if self.text is not None else None
        ccl = getattr(inner, "tt_ccl", None) if inner is not None else None
        if ccl is None:
            return

        for attr in ("barrier_semaphore_handles", "ag_semaphore_handles", "rs_semaphore_handles"):
            container = getattr(ccl, attr, None)
            if not isinstance(container, list):
                continue
            for sub in container:
                if isinstance(sub, list):
                    for i in range(len(sub)):
                        sub[i] = None
                    sub.clear()
            container.clear()

        try:
            setattr(inner, "tt_ccl", None)
        except Exception:
            pass

    def __enter__(self) -> "VoxtralTTSPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup_all()
