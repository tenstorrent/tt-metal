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
from models.experimental.voxtraltts.tt.text_model import VoxtralTTTextModel, patch_text_model_fp32_rms_norms
from models.experimental.voxtraltts.tt.voxtral_tt_args import (
    _load_safetensors_state_dict,
    voxtral_text_default_optimizations,
)
from models.experimental.voxtraltts.utils.debug_trace import VoxtralTTSDebugTrace
from models.experimental.voxtraltts.utils.rng import acoustic_fm_noise_seed

ACOUSTIC_CFG_ALPHA_DEFAULT = 1.2


@dataclass
class VoxtralTTSGenerateOutput:
    """Debuggable output from the full free-running TTS path."""

    waveform: torch.Tensor
    codes_b37t: torch.Tensor
    shifted_codes_t37: torch.Tensor
    hit_end_audio: bool
    debug: VoxtralTTSDebugTrace | None = None


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
        text_optimizations=voxtral_text_default_optimizations,
        acoustic_dtype: ttnn.DataType = ttnn.bfloat16,
        tokenizer_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> "VoxtralTTSPipeline":
        """Build TT TTS pipeline using the production default text optimization profile."""
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

    def _build_voice_injected_embeds_tt(
        self,
        prompt_token_ids: list[int],
        voice: str,
    ) -> tuple[torch.Tensor, ttnn.Tensor]:
        """Prompt token IDs + voice injection → (CPU ``[S, dim]``, device ``[S, 1, 1, dim]``).

        Builds the voice-injected embedding on CPU (F.embedding + boolean-mask scatter),
        then uploads the full sequence in **one** ``ttnn.from_torch`` call as ROW_MAJOR DRAM.

        Returning the CPU tensor avoids rebuilding it for debug traces.
        The device tensor is in the exact format ``prefill_from_embeds`` expects, so the
        internal CPU-reshape branch (item #3 in the torch-fallback audit) is never entered.

        The caller owns the returned ``ttnn.Tensor`` and must deallocate it after
        ``prefill_from_embeds`` returns.
        """
        embeds_cpu = self._build_voice_injected_embeds(prompt_token_ids, voice)  # [S, dim] CPU
        dim = self.text.inner.args.dim
        S = int(embeds_cpu.shape[0])
        embeds_4d = embeds_cpu.reshape(S, 1, 1, dim).contiguous()  # [S,1,1,dim] CPU
        embeds_tt = ttnn.from_torch(
            embeds_4d,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR allows non-tile-aligned slicing
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
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
        x_4d = emb.reshape(1, 1, 1, dim).contiguous()
        return ttnn.from_torch(
            x_4d,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _audio_codes_to_mm_embed_device(self, audio_codes_1_37: torch.Tensor) -> ttnn.Tensor:
        """``[1, 37]`` codes → ``[1, 1, 1, dim]`` TT embed; CPU embedding + TT upload.

        ``mm_audio_encode_tokens_summed_forward`` contains TILE rank-change reshapes that
        fail for T=1 (single AR step). Using the proven CPU-side ``F.embedding`` path and a
        single ``ttnn.from_torch`` upload is reliable and produces the correct mesh mapping
        for the text model's ``_decode_single_token_to_tt``.
        """
        emb = self._audio_codes_to_mm_embed(audio_codes_1_37)  # CPU F.embedding + sum → [dim]
        dim = self.text.inner.args.dim
        return ttnn.from_torch(
            emb.reshape(1, 1, 1, dim).to(dtype=torch.bfloat16).contiguous(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    @torch.no_grad()
    def forward_device_resident(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 2500,
        seed: int = 0,
        *,
        fixed_step_count: bool = False,
        include_waveform_decode: bool = True,
        return_debug: bool = False,
    ) -> VoxtralTTSGenerateOutput:
        """Same AR loop as :meth:`forward`; final waveform decode on TT device.

        ``return_debug`` only controls trace collection. It must not change numerics;
        staged PCC and ``test_ttnn_trial.py`` share the same compute path.
        """
        torch.manual_seed(seed)
        debug = VoxtralTTSDebugTrace() if return_debug else None

        request = compose_speech_request(text, self.model_name_or_path, voice=voice)
        prompt_token_ids: list[int] = request["prompt_token_ids"]
        S_prompt = len(prompt_token_ids)
        # Build CPU embeds (for debug trace) and upload once to device as [S,1,1,dim].
        # prefill_from_embeds receives ttnn.Tensor → skips the internal CPU-reshape branch.
        inputs_embeds_cpu, inputs_embeds_tt = self._build_voice_injected_embeds_tt(prompt_token_ids, voice)
        if debug is not None:
            debug.set("embeds.prompt", inputs_embeds_cpu)

        # Production prefill path only; debug must not call collect_layer_hiddens here
        # (that path reads every layer to host and can change the last-token hidden).
        # hidden stays on device (ttnn.Tensor) throughout the AR loop; acoustic model reads it via
        # forward_from_tt. Convert to torch only for debug trace (hidden_tt_to_torch).
        last_hidden_tt = self.text.prefill_from_embeds(inputs_embeds_tt, start_pos=0)
        if inputs_embeds_tt.is_allocated():
            ttnn.deallocate(inputs_embeds_tt)
        del inputs_embeds_cpu  # allow CPU memory to be reclaimed
        if debug is not None:
            debug.set("text.prefill.hidden", self.text.hidden_tt_to_torch(last_hidden_tt))
        cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)
        generated_codes: list[torch.Tensor] = []
        current_pos = S_prompt

        for step_idx in range(max_tokens):
            last_hidden = self.text.hidden_tt_to_torch(last_hidden_tt)
            if debug is not None:
                debug.set(f"step.{step_idx}.text.hidden_in", last_hidden)
            ac_out = self.acoustic_codes_forward(
                last_hidden.unsqueeze(0),
                cfg_alpha,
                noise_seed=acoustic_fm_noise_seed(seed, step_idx),
            )
            audio_codes = ac_out.to(torch.long)
            if debug is not None:
                debug.set(f"step.{step_idx}.acoustic.codes", audio_codes.squeeze(0))
                llm_tt = ttnn.from_torch(
                    last_hidden.unsqueeze(0).unsqueeze(1).to(torch.bfloat16),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                sem_tt = self.acoustic.semantic_logits_tt(llm_tt)
                ttnn.deallocate(llm_tt)
                sem_host = ttnn.to_torch(sem_tt).float()
                ttnn.deallocate(sem_tt)
                while sem_host.dim() > 2:
                    sem_host = sem_host.squeeze(1)
                debug.set(f"step.{step_idx}.acoustic.semantic_logits", sem_host.squeeze(0))

            generated_codes.append(audio_codes[0].detach().cpu())
            if not fixed_step_count and int(audio_codes[0, 0].item()) == self.end_audio_id:
                break

            # MM embed: use on-device embedding table (saves 6 KB upload + CPU F.embedding per step).
            mm_embed_tt = self._audio_codes_to_mm_embed_device(audio_codes)
            next_hidden_tt = self.text._decode_single_token_to_tt(mm_embed_tt, current_pos)
            if mm_embed_tt.is_allocated():
                ttnn.deallocate(mm_embed_tt)
            if last_hidden_tt.is_allocated():
                ttnn.deallocate(last_hidden_tt)
            last_hidden_tt = next_hidden_tt
            ttnn.synchronize_device(self.mesh_device)
            if debug is not None:
                debug.set(f"step.{step_idx}.text.hidden_out", self.text.hidden_tt_to_torch(last_hidden_tt))
            current_pos += 1

        ttnn.synchronize_device(self.mesh_device)
        if last_hidden_tt is not None and last_hidden_tt.is_allocated():
            ttnn.deallocate(last_hidden_tt)

        if not generated_codes:
            empty_wav = torch.tensor([], dtype=torch.float32)
            return VoxtralTTSGenerateOutput(
                waveform=empty_wav,
                codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                shifted_codes_t37=torch.empty((0, 37), dtype=torch.long),
                hit_end_audio=False,
                debug=debug,
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
                debug=debug,
            )
        codes_b37t = audio_tokens.T.unsqueeze(0).long()

        if include_waveform_decode:
            ttnn.synchronize_device(self.mesh_device)
            codes_2d_tt = ttnn.from_torch(
                audio_tokens.to(torch.uint32).contiguous(),
                device=self.mesh_device,
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
            wav_tt = self.audio_tokenizer.pretransform_decode_tt(mel_tt)
            ttnn.deallocate(mel_tt)
            wav = ttnn.to_torch(wav_tt).float()
            if wav_tt.is_allocated():
                ttnn.deallocate(wav_tt)

            expected_samples = audio_tokens.shape[0] * self._downsample_factor
            waveform = wav.reshape(-1)[:expected_samples].reshape(1, 1, -1)
        else:
            waveform = torch.zeros(1, 1, 0, dtype=torch.float32)

        if debug is not None:
            debug.set("output.codes", codes_b37t.float())
            debug.set("output.waveform", waveform)

        return VoxtralTTSGenerateOutput(
            waveform=waveform,
            codes_b37t=codes_b37t,
            shifted_codes_t37=shifted_audio_tokens.long(),
            hit_end_audio=hit_end_audio,
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
    ) -> torch.Tensor:
        """Compatibility wrapper returning only the final waveform."""
        return self.forward_device_resident(
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
        """Run the device-resident path (:meth:`forward_device_resident`), returning generated codes."""
        return self.forward_device_resident(
            text=text,
            voice=voice,
            max_tokens=max_tokens,
            seed=seed,
            fixed_step_count=fixed_step_count,
            include_waveform_decode=include_waveform_decode,
        )

    def decode_waveform_from_codes_tt(self, codes_b37t: torch.Tensor) -> torch.Tensor:
        """``[B,37,T]`` int CPU codes → float32 waveform (latent + decoder + pretransform on TT)."""
        latent_tt = self.audio_tokenizer.latent_from_codes(codes_b37t)
        mel_tt = self.audio_tokenizer.decode_latent_to_mel_b1tc(latent_tt)
        ttnn.deallocate(latent_tt)
        wav_tt = self.audio_tokenizer.pretransform_decode_tt(mel_tt)
        ttnn.deallocate(mel_tt)
        wav = ttnn.to_torch(wav_tt).float()
        if wav_tt.is_allocated():
            ttnn.deallocate(wav_tt)
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
        codes = ttnn.to_torch(codes_tt).long().reshape(bsz, -1)
        ttnn.deallocate(codes_tt)
        return codes

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
