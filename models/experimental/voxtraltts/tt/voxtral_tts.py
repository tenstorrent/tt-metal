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
    audio_tokenizer_decode_reference,
    audio_tokenizer_encode_tokens_reference,
)
from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import AudioSpecialTokens
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    compute_rope_frequencies as reference_compute_rope_frequencies,
    extract_layer_weights,
    rms_norm as reference_rms_norm,
    text_decoder_layer as reference_text_decoder_layer,
)
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
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict

ACOUSTIC_CFG_ALPHA_DEFAULT = 1.2


def _prefill_tile_start(token_index: int) -> int:
    return (int(token_index) // 32) * 32


def reference_text_last_token_logits(state_dict: dict, args: Any, tokens: torch.Tensor) -> torch.Tensor:
    """Last-token logits from the in-repo PyTorch text backbone (same construction as ``test_text_model``)."""
    seq_len = tokens.shape[1]
    ref_cfg = VoxtralTextConfig(
        hidden_size=args.dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.hidden_dim,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.norm_eps,
    )
    ref_hidden = F.embedding(tokens, state_dict["tok_embeddings.weight"])
    ref_cos, ref_sin = reference_compute_rope_frequencies(
        head_dim=ref_cfg.head_dim,
        max_seq_len=seq_len,
        theta=ref_cfg.rope_theta,
        device=ref_hidden.device,
    )
    ref_attn_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float32)
    ref_attn_mask = torch.triu(ref_attn_mask, diagonal=1)
    for layer_idx in range(ref_cfg.num_hidden_layers):
        layer_weights = extract_layer_weights(state_dict, layer_idx, prefix="layers.")
        ref_hidden = reference_text_decoder_layer(
            hidden_states=ref_hidden,
            layer_weights=layer_weights,
            cos=ref_cos,
            sin=ref_sin,
            config=ref_cfg,
            attention_mask=ref_attn_mask,
        )
    ref_hidden = reference_rms_norm(ref_hidden, state_dict["norm.weight"], eps=ref_cfg.rms_norm_eps)
    return F.linear(ref_hidden[:, -1, :], state_dict["output.weight"]).squeeze(0).float()


@dataclass
class VoxtralTTSGenerateOutput:
    """Debuggable output from the full free-running TTS path."""

    waveform: torch.Tensor
    codes_b37t: torch.Tensor
    shifted_codes_t37: torch.Tensor
    hit_end_audio: bool


@dataclass
class VoxtralTTSPipeline:
    """Loads and wires ``VoxtralTTTextModel``, ``VoxtralTTAcousticModel``, and ``VoxtralTTAudioTokenizer``.

    ``generate(text, voice)`` runs the full TTS pipeline entirely on TT:
    1. Mistral-common tokenization (CPU, tokenization only — not model inference).
    2. Voice-embedding injection + prefill on TT text transformer.
    3. Autoregressive TT acoustic decode loop (hidden → codes).
    4. TT audio tokenizer decode (codes → waveform).
    """

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

    # ------------------------------------------------------------------
    # Internal helpers for generate()
    # ------------------------------------------------------------------

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
        """Token IDs + voice embedding injection → ``[S, dim]`` bfloat16 CPU embeddings."""
        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long)
        embeds = F.embedding(input_ids, self.tok_embedding_weight)  # [S, dim]
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
        return embeds  # [S, dim] bfloat16

    def _audio_codes_to_mm_embed(self, audio_codes_1_37: torch.Tensor) -> torch.Tensor:
        """``[1, 37]`` discrete codes → ``[dim]`` multimodal embedding (CPU, lookup+sum)."""
        emb = audio_tokenizer_encode_tokens_reference(
            audio_codes_1_37.unsqueeze(-1),  # [1, 37, 1]
            self.mm_embedding_weight,
            self.config.audio_model_args,
        )  # [1, dim]  (T=1, squeezed)
        return emb.squeeze(0).to(dtype=torch.bfloat16)  # [dim]

    # ------------------------------------------------------------------
    # End-to-end TT generation (fully on TT — no reference model)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 2500,
        seed: int = 0,
    ) -> torch.Tensor:
        """Full TT TTS pipeline: text → audio waveform.

        All neural-network forward passes run on TT:
        - text prefill + each decode step uses ``VoxtralTTTextModel``.
        - acoustic head uses ``VoxtralTTAcousticModel``.
        - audio tokenizer decode uses ``VoxtralTTAudioTokenizer``.

        Tokenization (mistral-common) and embedding lookup / voice-file I/O run on CPU —
        these are not model inference, equivalent to a tokenizer ``encode()`` call.

        Returns a float32 waveform tensor ``[1, 1, T*patch]`` on CPU.
        """
        return self.generate_with_codes(text=text, voice=voice, max_tokens=max_tokens, seed=seed).waveform

    @torch.no_grad()
    def generate_with_codes(
        self,
        text: str,
        voice: str = "casual_male",
        max_tokens: int = 2500,
        seed: int = 0,
    ) -> VoxtralTTSGenerateOutput:
        """Run the same free-running path as :meth:`generate`, returning generated codes for diagnostics."""
        torch.manual_seed(seed)

        request = compose_speech_request(text, self.model_name_or_path, voice=voice)
        prompt_token_ids: list[int] = request["prompt_token_ids"]
        S_prompt = len(prompt_token_ids)

        # --- 1. TT prefill with voice-injected embeddings ---
        inputs_embeds = self._build_voice_injected_embeds(prompt_token_ids, voice)
        last_hidden = self.text.prefill_from_embeds(inputs_embeds, start_pos=0)  # [dim]

        # --- 2. Autoregressive acoustic decode loop ---
        cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)
        generated_codes: list[torch.Tensor] = []
        current_pos = S_prompt

        for _ in range(max_tokens):
            # TT acoustic: [1, dim] hidden → [1, 37] codes
            audio_codes = self.acoustic.forward(last_hidden.unsqueeze(0), cfg_alpha).to(torch.long)  # [1, 37]

            generated_codes.append(audio_codes[0].detach().cpu())  # [37]
            if int(audio_codes[0, 0].item()) == self.end_audio_id:
                break

            # MM embedding for next text step: [dim] on CPU
            mm_embed = self._audio_codes_to_mm_embed(audio_codes)  # [dim]

            # TT decode step: updates KV cache at current_pos, returns [dim] hidden
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

        # --- 3. Trim to end-of-audio and unshift codes ---
        stacked = torch.stack(generated_codes, dim=0)  # [T, 37]
        eoa = (stacked[:, 0] == self.end_audio_id).nonzero(as_tuple=False)
        hit_end_audio = len(eoa) > 0
        cut = int(eoa[0].item()) if len(eoa) else stacked.shape[0]
        shifted_audio_tokens = stacked[:cut]
        audio_tokens = shifted_audio_tokens - 2  # un-shift special-token offset
        if audio_tokens.numel() == 0:
            empty_wav = torch.tensor([], dtype=torch.float32)
            return VoxtralTTSGenerateOutput(
                waveform=empty_wav,
                codes_b37t=torch.empty((1, 37, 0), dtype=torch.long),
                shifted_codes_t37=shifted_audio_tokens.long(),
                hit_end_audio=hit_end_audio,
            )
        codes_b37t = audio_tokens.T.unsqueeze(0).long()  # [1, 37, T]

        # --- 4. TT audio tokenizer decode: codes → waveform ---
        wav = self.decode_waveform_from_codes_tt(codes_b37t)
        expected_samples = audio_tokens.shape[0] * self._downsample_factor
        wav_flat = wav.detach().cpu().float().reshape(-1)
        waveform = wav_flat[:expected_samples].reshape(1, 1, -1)
        return VoxtralTTSGenerateOutput(
            waveform=waveform,
            codes_b37t=codes_b37t,
            shifted_codes_t37=shifted_audio_tokens.long(),
            hit_end_audio=hit_end_audio,
        )

    def decode_waveform_from_codes_tt(self, codes_b37t: torch.Tensor) -> torch.Tensor:
        """``[B,37,T]`` int CPU codes → float32 waveform (TT latent + decoder + pretransform)."""
        latent_tt = self.audio_tokenizer.latent_from_codes(codes_b37t)
        mel_tt = self.audio_tokenizer.decode_latent_to_mel_b1tc(latent_tt)
        ttnn.deallocate(latent_tt)
        wav = self.audio_tokenizer.pretransform_decode_torch(mel_tt)
        ttnn.deallocate(mel_tt)
        return wav

    def decode_waveform_from_codes_reference(self, codes_b37t: torch.Tensor) -> torch.Tensor:
        """PyTorch bf16 golden waveform for the same codes (``audio_tokenizer_decode_reference``)."""
        return audio_tokenizer_decode_reference(codes_b37t, self.audio_tokenizer_sd, self.config.audio_tokenizer_args)

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

    def text_decode_multistep_compare_reference(
        self,
        *,
        prompt_tokens: torch.Tensor,
        decode_tokens: torch.Tensor,
        pcc: float = 0.98,
    ) -> None:
        """Teacher-forced decode steps: TT last-token logits vs :func:`reference_text_last_token_logits` each step.

        Mirrors ``tests/test_text_model.py::test_text_model_decode_multistep_reference_pcc`` /
        ``models/tt_transformers/tests/test_model.py`` (run reference then compare PCC to TT).
        """
        from models.common.utility_functions import comp_pcc

        model = self.text
        args = model.inner.args
        state_dict = args.load_state_dict()
        prompt_len = int(prompt_tokens.shape[1])
        decode_steps = int(decode_tokens.shape[1])

        tt_prompt_x, prompt_rot_global, prompt_rot_local, _, _ = model.prepare_inputs_prefill(
            prompt_tokens, start_pos=0
        )
        _ = model.inner.ttnn_prefill_forward(
            tt_prompt_x,
            rot_mats_global=prompt_rot_global,
            rot_mats_local=prompt_rot_local,
            get_last_token=-1,
        )

        for step in range(decode_steps):
            current_pos = prompt_len + step
            step_token = decode_tokens[:, step]
            tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
                step_token, torch.tensor([current_pos], dtype=torch.int64)
            )
            tt_decode_logits, _ = model.inner.ttnn_decode_forward(
                tt_tokens,
                tt_current_pos,
                rot_mat_idxs=tt_rope_idxs,
                page_table=tt_page_table,
                kv_cache=None,
                sampling_on_device=False,
            )
            tt_last_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[
                0, 0
            ].float()

            ref_tokens = torch.cat([prompt_tokens, decode_tokens[:, : step + 1]], dim=1)
            ref_last_logits = reference_text_last_token_logits(state_dict, args, ref_tokens)

            passing, msg = comp_pcc(ref_last_logits, tt_last_logits, pcc=pcc)
            assert passing, f"text decode step={step} pos={current_pos} PCC failed: {msg}"


def main() -> None:
    import argparse
    import os

    p = argparse.ArgumentParser(description="Load Voxtral TT pipeline (text + acoustic + audio tokenizer).")
    p.add_argument("--model", type=str, default=os.environ.get("VOXTRAL_TTS_MODEL") or DEFAULT_VOXTRAL_MODEL)
    cli = p.parse_args()
    try:
        from tests.scripts.common import get_updated_device_params
    except ImportError as exc:
        raise SystemExit(f"Run from tt-metal with tests on PYTHONPATH: {exc}") from exc

    did = 0
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG:
        did = 4
    updated = get_updated_device_params({})
    orig = ttnn.GetDefaultDevice()
    dev = ttnn.CreateDevice(device_id=did, **updated)
    ttnn.SetDefaultDevice(dev)
    try:
        pipe = VoxtralTTSPipeline.from_model_name(dev, model_name_or_path=cli.model)
        print(f"Loaded VoxtralTTSPipeline for {cli.model!r}")
        print(f"  text layers={pipe.text.inner.args.n_layers} dim={pipe.text.inner.args.dim}")
        print("  acoustic + audio_tokenizer OK")
    finally:
        ttnn.SetDefaultDevice(orig)
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
