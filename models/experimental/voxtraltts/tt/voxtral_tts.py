# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-level TT orchestration for Voxtral TTS: text backbone, acoustic head, audio tokenizer."""

from __future__ import annotations

import math
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
from models.experimental.voxtraltts.tt.text_model import VoxtralTTTextModel
from models.experimental.voxtraltts.tt.voxtral_tt_args import (
    _load_safetensors_state_dict,
    voxtral_text_high_accuracy_optimizations,
)

ACOUSTIC_CFG_ALPHA_DEFAULT = 1.2


@dataclass
class VoxtralTTSGenerateOutput:
    """Debuggable output from the full free-running TTS path."""

    waveform: torch.Tensor
    codes_b37t: torch.Tensor
    shifted_codes_t37: torch.Tensor
    hit_end_audio: bool


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

    @property
    def _downsample_factor(self) -> int:
        cfg = self.config.audio_tokenizer_args
        return cfg.pretransform_patch_size * math.prod(parse_csv_ints(cfg.decoder_convs_strides_str))

    @classmethod
    def from_model_name(
        cls,
        mesh_device,
        model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
        *,
        text_max_seq_len: int = 256,
        text_dtype: ttnn.DataType = ttnn.bfloat16,
        text_optimizations=voxtral_text_high_accuracy_optimizations,
        acoustic_dtype: ttnn.DataType = ttnn.bfloat16,
        tokenizer_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> "VoxtralTTSPipeline":
        full = _load_safetensors_state_dict(model_name_or_path)
        cfg = load_voxtral_config(model_name_or_path)
        sd_at = extract_audio_tokenizer_state_dict(full)

        tok_emb_w = full.get("mm_audio_embeddings.tok_embeddings.weight")
        if tok_emb_w is None:
            raise RuntimeError("Missing 'mm_audio_embeddings.tok_embeddings.weight' in checkpoint.")
        mm_emb_w = full.get("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight")
        if mm_emb_w is None:
            raise RuntimeError("Missing 'mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight'.")

        text = VoxtralTTTextModel.create_from_model_name(
            mesh_device=mesh_device,
            model_name_or_path=model_name_or_path,
            dtype=text_dtype,
            max_batch_size=1,
            max_seq_len=text_max_seq_len,
            preloaded_state_dict=full,
            optimizations=text_optimizations,
        )
        acoustic = VoxtralTTAcousticModel.create_from_model_name(
            mesh_device,
            model_name_or_path=model_name_or_path,
            dtype=acoustic_dtype,
        )
        audio_tokenizer = VoxtralTTAudioTokenizer(
            mesh_device,
            state_dict=sd_at,
            tokenizer_cfg=cfg.audio_tokenizer_args,
            dtype=tokenizer_dtype,
            full_checkpoint=full,
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

    def _build_voice_injected_embeds(
        self,
        prompt_token_ids: list[int],
        voice: str,
    ) -> torch.Tensor:
        """Prompt token IDs + voice injection → ``[S, dim]`` bfloat16 CPU embeddings."""
        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long)
        embeds = F.embedding(input_ids, self.tok_embedding_weight)
        voice_path = self._resolve_model_file(f"voice_embedding/{voice}.pt")
        try:
            voice_emb = torch.load(str(voice_path), map_location="cpu", weights_only=True)
        except TypeError:
            voice_emb = torch.load(str(voice_path), map_location="cpu")
        voice_emb = voice_emb.to(dtype=torch.bfloat16)
        audio_mask = input_ids == self.audio_token_id
        n_audio = int(audio_mask.sum().item())
        if n_audio != voice_emb.shape[0]:
            raise RuntimeError(
                f"Voice embedding length mismatch: {voice_emb.shape[0]} voice tokens vs "
                f"{n_audio} audio placeholder positions in prompt."
            )
        embeds[audio_mask] = voice_emb
        return embeds

    def _audio_codes_to_mm_embed(self, audio_codes_1_37: torch.Tensor) -> torch.Tensor:
        """``[1, 37]`` codes → ``[dim]`` MM embedding (CPU lookup+sum)."""
        emb = audio_tokenizer_encode_tokens_reference(
            audio_codes_1_37.unsqueeze(-1),
            self.mm_embedding_weight,
            self.config.audio_model_args,
        )
        return emb.squeeze(0).to(dtype=torch.bfloat16)

    @torch.no_grad()
    def forward(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 2500,
        seed: int = 0,
        *,
        fixed_step_count: bool = False,
        include_waveform_decode: bool = True,
    ) -> VoxtralTTSGenerateOutput:
        """TT TTS: prefill → acoustic loop → optional waveform decode. Returns codes + waveform."""
        torch.manual_seed(seed)

        request = compose_speech_request(text, self.model_name_or_path, voice=voice)
        prompt_token_ids: list[int] = request["prompt_token_ids"]
        S_prompt = len(prompt_token_ids)

        inputs_embeds = self._build_voice_injected_embeds(prompt_token_ids, voice)
        last_hidden = self.text.prefill_from_embeds(inputs_embeds, start_pos=0)

        cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)
        generated_codes: list[torch.Tensor] = []
        current_pos = S_prompt

        for _ in range(max_tokens):
            audio_codes = self.acoustic.forward(last_hidden.unsqueeze(0), cfg_alpha).to(torch.long)

            generated_codes.append(audio_codes[0].detach().cpu())
            if not fixed_step_count and int(audio_codes[0, 0].item()) == self.end_audio_id:
                break

            mm_embed = self._audio_codes_to_mm_embed(audio_codes)
            last_hidden = self.text.decode_step_from_embeds(mm_embed, current_pos)
            current_pos += 1

        if not generated_codes:
            empty_wav = torch.tensor([], dtype=torch.float32)
            return VoxtralTTSGenerateOutput(
                waveform=empty_wav,
                codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                shifted_codes_t37=torch.empty((0, 37), dtype=torch.long),
                hit_end_audio=False,
            )

        stacked = torch.stack(generated_codes, dim=0)
        eoa = (stacked[:, 0] == self.end_audio_id).nonzero(as_tuple=False)
        hit_end_audio = len(eoa) > 0
        cut = int(eoa[0].item()) if len(eoa) else stacked.shape[0]
        shifted_audio_tokens = stacked[:cut]
        audio_tokens = shifted_audio_tokens - 2
        if audio_tokens.numel() == 0:
            empty_wav = torch.tensor([], dtype=torch.float32)
            return VoxtralTTSGenerateOutput(
                waveform=empty_wav,
                codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                shifted_codes_t37=shifted_audio_tokens.long(),
                hit_end_audio=hit_end_audio,
            )
        codes_b37t = audio_tokens.T.unsqueeze(0).long()

        if include_waveform_decode:
            wav = self.decode_waveform_from_codes_tt(codes_b37t)
            expected_samples = audio_tokens.shape[0] * self._downsample_factor
            wav_flat = wav.detach().cpu().float().reshape(-1)
            waveform = wav_flat[:expected_samples].reshape(1, 1, -1)
        else:
            waveform = torch.zeros(1, 1, 0, dtype=torch.float32)
        return VoxtralTTSGenerateOutput(
            waveform=waveform,
            codes_b37t=codes_b37t,
            shifted_codes_t37=shifted_audio_tokens.long(),
            hit_end_audio=hit_end_audio,
        )

    __call__ = forward

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
    ) -> torch.Tensor:
        """Compatibility wrapper returning only the final waveform."""
        return self.forward(
            text=text,
            voice=voice,
            max_tokens=max_tokens,
            seed=seed,
            fixed_step_count=fixed_step_count,
            include_waveform_decode=include_waveform_decode,
        ).waveform

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
    ) -> VoxtralTTSGenerateOutput:
        """Run the same free-running path as :meth:`generate`, returning generated codes for diagnostics."""
        return self.forward(
            text=text,
            voice=voice,
            max_tokens=max_tokens,
            seed=seed,
            fixed_step_count=fixed_step_count,
            include_waveform_decode=include_waveform_decode,
        )

    def decode_waveform_from_codes_tt(self, codes_b37t: torch.Tensor) -> torch.Tensor:
        """``[B,37,T]`` int CPU codes → float32 waveform (TT latent + decoder + pretransform)."""
        latent_tt = self.audio_tokenizer.latent_from_codes(codes_b37t)
        mel_tt = self.audio_tokenizer.decode_latent_to_mel_b1tc(latent_tt)
        ttnn.deallocate(latent_tt)
        wav = self.audio_tokenizer.pretransform_decode_torch(mel_tt)
        ttnn.deallocate(mel_tt)
        return wav

    def _acoustic_hidden_tile_copy(self, llm_hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Clone + TILE layout for acoustic/trace; does not free trace inputs."""
        work = ttnn.clone(llm_hidden_tt)
        if len(work.shape) == 4:
            bsz, dim = int(work.shape[0]), int(work.shape[-1])
            reshaped = ttnn.reshape(work, (bsz, 1, dim))
            if work.is_allocated():
                ttnn.deallocate(work)
            work = reshaped
        if work.layout != ttnn.TILE_LAYOUT:
            tile_hidden = ttnn.to_layout(
                work,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if work.is_allocated():
                ttnn.deallocate(work)
            work = tile_hidden
        return work

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
            codes_tt = self.acoustic.forward_acoustic_trace_codes(llm_tile, step.noise_tt, cfg_scalar)
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
    ) -> torch.Tensor:
        """``[B, input_dim]`` LLM hidden → ``[B, 1+n_acoustic]`` discrete codes (TT acoustic ``forward``)."""
        if cfg_alpha is None:
            cfg_alpha = torch.tensor(
                ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=llm_hidden_bf16.dtype, device=llm_hidden_bf16.device
            )
        return self.acoustic.forward(llm_hidden_bf16, cfg_alpha)

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
