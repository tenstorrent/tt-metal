# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component pipeline PCC (not full E2E — see ``tests/pcc/test_ttnn_voxtral_tts_e2e.py``)."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import (
    audio_tokenizer_decode_reference,
    audio_tokenizer_latent_from_codes,
    decoder_blocks_stack_reference,
    output_proj_mel_ncl_reference_bf16,
    pretransform_decode,
)
from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import (
    FlowMatchingAudioTransformerRef,
    build_audio_model_args_from_voxtral_config,
)
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    compute_rope_frequencies as reference_compute_rope_frequencies,
    extract_layer_weights,
    rms_norm as reference_rms_norm,
    text_decoder_layer as reference_text_decoder_layer,
)
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

PREFILL_HIDDEN_PCC = 0.99
TEXT_DECODE_STEP_PCC = 0.98
ACOUSTIC_MATCH_FRAC = 0.88
WAVEFORM_PCC = 0.99
FULL_GENERATE_WAVEFORM_PCC_SHORT = 0.99
FULL_GENERATE_WAVEFORM_PCC_LONG = 0.99
FULL_GENERATE_SHORT_MAX_FRAMES = 32

_DEMO_TEXT = "Hello from the Voxtral Tenstorrent demo."
_DEMO_VOICE = "casual_male"


def _log_stage_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


def _log_generated_code_tokenizer_diagnostics(pipe: VoxtralTTSPipeline, codes_b37t: torch.Tensor) -> None:
    """Per-stage tokenizer PCC for long generated-code sequences."""
    cfg = pipe.config.audio_tokenizer_args
    sd = pipe.audio_tokenizer_sd

    _log_stage_header("GENERATED-CODE TOKENIZER DIAGNOSTICS")
    logger.info(f"  codes shape={tuple(codes_b37t.shape)}")

    ref_latent_ncl = audio_tokenizer_latent_from_codes(
        codes_b37t.cpu(),
        sd,
        n_acoustic_levels=cfg.acoustic_codebook_size,
    ).to(torch.bfloat16)
    latent_tt = pipe.audio_tokenizer.latent_from_codes(codes_b37t)
    tt_latent_btc = ttnn.to_torch(latent_tt).squeeze(1).float()
    ref_latent_btc = ref_latent_ncl.permute(0, 2, 1).contiguous().float()
    _, msg = comp_pcc(ref_latent_btc, tt_latent_btc, pcc=0.99)
    _log_pcc("latent_from_codes", float(msg), 0.99)

    ref_hidden_btd = decoder_blocks_stack_reference(ref_latent_ncl, sd, cfg)
    hidden_tt = pipe.audio_tokenizer.decode_full_forward(latent_tt)
    ttnn.deallocate(latent_tt)
    tt_hidden_btd = ttnn.to_torch(hidden_tt).squeeze(1).float()
    _, msg = comp_pcc(ref_hidden_btd.float(), tt_hidden_btd, pcc=0.99)
    _log_pcc("decoder stack output", float(msg), 0.99)

    ref_mel_ncl = output_proj_mel_ncl_reference_bf16(ref_hidden_btd.to(torch.bfloat16), sd)
    ref_mel_btc = ref_mel_ncl.permute(0, 2, 1).contiguous().float()
    mel_tt = pipe.audio_tokenizer.output_proj_forward(hidden_tt)
    ttnn.deallocate(hidden_tt)
    tt_mel_btc = ttnn.to_torch(mel_tt).squeeze(1).contiguous().float()
    _, msg = comp_pcc(ref_mel_btc, tt_mel_btc, pcc=0.99)
    _log_pcc("output_proj mel", float(msg), 0.99)

    ref_wav = pretransform_decode(ref_mel_ncl, channels=cfg.channels).float()
    tt_wav = pipe.audio_tokenizer.pretransform_decode_torch(mel_tt)
    ttnn.deallocate(mel_tt)
    _, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=0.99)
    _log_pcc("pretransform waveform", float(msg), 0.99)


def _waveform_pcc_target_for_frames(num_frames: int) -> float:
    if num_frames <= FULL_GENERATE_SHORT_MAX_FRAMES:
        return FULL_GENERATE_WAVEFORM_PCC_SHORT
    return FULL_GENERATE_WAVEFORM_PCC_LONG


def _align_to_ref_shape(ref_t: torch.Tensor, tt_t: torch.Tensor) -> torch.Tensor:
    """Trim TT tensor to reference shape (ttnn padding)."""
    out = tt_t
    for dim, size in enumerate(ref_t.shape):
        if dim < out.dim() and out.shape[dim] > size:
            sl = [slice(None)] * out.dim()
            sl[dim] = slice(0, size)
            out = out[tuple(sl)]
    return out.reshape(ref_t.shape)


def _reference_text_last_token_logits(state_dict: dict, args, tokens: torch.Tensor) -> torch.Tensor:
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


def _text_decode_multistep_compare_reference(
    pipe: VoxtralTTSPipeline,
    *,
    prompt_tokens: torch.Tensor,
    decode_tokens: torch.Tensor,
    pcc: float,
) -> None:
    model = pipe.text
    args = model.inner.args
    state_dict = args.load_state_dict()
    prompt_len = int(prompt_tokens.shape[1])
    decode_steps = int(decode_tokens.shape[1])

    tt_prompt_x, prompt_rot_global, prompt_rot_local, _, _ = model.prepare_inputs_prefill(prompt_tokens, start_pos=0)
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
        tt_last_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()

        ref_tokens = torch.cat([prompt_tokens, decode_tokens[:, : step + 1]], dim=1)
        ref_last_logits = _reference_text_last_token_logits(state_dict, args, ref_tokens)

        passing, msg = comp_pcc(ref_last_logits, tt_last_logits, pcc=pcc)
        assert passing, f"text decode step={step} pos={current_pos} PCC failed: {msg}"


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_loads(device, reset_seeds):
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")
    assert pipe.text.inner.args.dim > 0
    assert pipe.acoustic.dim > 0
    assert pipe.audio_tokenizer.cfg.dim > 0


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_waveform_codes_pcc(device, reset_seeds):
    """TT decode_waveform_from_codes_tt vs CPU for the same codes."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    cfg = pipe.config.audio_tokenizer_args
    b, t = 1, 32
    semantic = torch.randint(0, cfg.semantic_codebook_size, (b, 1, t))
    acoustic = torch.randint(0, cfg.acoustic_codebook_size, (b, cfg.acoustic_dim, t))
    codes = torch.cat([semantic, acoustic], dim=1).long()

    try:
        ref_wav = audio_tokenizer_decode_reference(codes, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
        tt_wav = pipe.decode_waveform_from_codes_tt(codes)
    except RuntimeError as exc:
        if "requires the full decoder stack" in str(exc) or "output_proj" in str(exc):
            pytest.skip(str(exc))
        raise

    assert ref_wav.shape == tt_wav.shape
    ok, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=WAVEFORM_PCC)
    _log_stage_header("AUDIO TOKENIZER - same-code waveform PCC")
    _log_pcc("decode_waveform_from_codes_tt", float(msg), WAVEFORM_PCC)
    logger.info(f"  codes shape={tuple(codes.shape)} waveform shape={tuple(tt_wav.shape)}")
    assert ok, f"Pipeline waveform PCC: {msg}"


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_acoustic_forward_matches_reference(device, reset_seeds):
    """TT acoustic forward vs CPU ref with synced FM RNG."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    cfg = load_voxtral_config(name)
    ref = FlowMatchingAudioTransformerRef(build_audio_model_args_from_voxtral_config(cfg)).to(torch.bfloat16).eval()
    full = _load_safetensors_state_dict(name)
    for k, v in full.items():
        if k.startswith("acoustic_transformer."):
            ref.load_weight((k.removeprefix("acoustic_transformer."), v))

    torch.manual_seed(42)
    bsz = 1
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    llm_h = torch.randn(bsz, d_llm, dtype=torch.bfloat16)
    cfg_alpha = torch.tensor(0.73, dtype=torch.bfloat16)

    torch.manual_seed(12345)
    ref_out = ref.forward(llm_h, cfg_alpha)
    torch.manual_seed(12345)
    tt_raw = pipe.acoustic_codes_forward(llm_h, cfg_alpha)
    tt_out = _align_to_ref_shape(ref_out, tt_raw)

    assert (
        ref_out.shape == tt_out.shape
    ), f"forward shape mismatch after align: ref={tuple(ref_out.shape)} tt_raw={tuple(tt_raw.shape)}"
    assert torch.equal(ref_out[:, :1], tt_out[:, :1]), "semantic token mismatch"

    _log_stage_header("ACOUSTIC MODEL - synced-RNG code agreement")
    logger.info(f"  semantic token exact: ref={ref_out[:, :1].tolist()} tt={tt_out[:, :1].tolist()}")
    n_acoustic = ref_out.shape[1] - 1
    if n_acoustic > 0:
        acoustic_ok = ref_out[:, 1:] == tt_out[:, 1:]
        match_frac = float(acoustic_ok.float().mean().item())
        logger.info(
            f"  acoustic code agreement: {match_frac:.4f}  "
            f"target>={ACOUSTIC_MATCH_FRAC:.4f}  matched={int(acoustic_ok.sum().item())}/{n_acoustic}"
        )
        assert (
            match_frac >= ACOUSTIC_MATCH_FRAC
        ), f"acoustic code agreement {match_frac:.4f} < {ACOUSTIC_MATCH_FRAC} (TT vs CPU FM drift at round)"


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("decode_steps", [4], ids=["4_steps"])
def test_voxtral_tts_pipeline_text_multistep_decode_pcc(device, reset_seeds, decode_steps):
    """Teacher-forced text decode logits PCC over multiple steps."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=512)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    request = compose_speech_request(_DEMO_TEXT, name, voice=_DEMO_VOICE, ref_audio=None)
    prompt_ids = request["prompt_token_ids"]
    prompt_tokens = torch.tensor(prompt_ids[:128], dtype=torch.int64).unsqueeze(0)

    vocab_size = pipe.text.inner.vocab_size
    decode_tokens = torch.randint(0, vocab_size, (1, decode_steps), dtype=torch.int64)

    _log_stage_header("TEXT MODEL - teacher-forced multistep decode PCC")
    logger.info(
        f"  prompt_len={prompt_tokens.shape[1]} decode_steps={decode_steps} "
        f"logits_pcc_target>={TEXT_DECODE_STEP_PCC:.4f}"
    )
    _text_decode_multistep_compare_reference(
        pipe,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        pcc=TEXT_DECODE_STEP_PCC,
    )
    logger.info("  text decode logits: all steps passed")


def _cpu_text_decode_step(
    cpu: VoxtralCPUReference,
    *,
    audio_codes_b37: torch.Tensor,
    past_key_values,
):
    """One HF text step from MM embedding input."""
    next_input = cpu._audio_codes_to_input_embeds(audio_codes_b37)
    outputs = cpu.text_model(
        inputs_embeds=next_input,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    return hidden, outputs.past_key_values


def _acoustic_codes_with_rng(
    pipe: VoxtralTTSPipeline,
    cpu: VoxtralCPUReference,
    *,
    hidden_bf16: torch.Tensor,
    cfg_alpha: torch.Tensor,
    rng_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU and TT acoustic forward on the same hidden with synced FM RNG."""
    hidden_in = hidden_bf16.unsqueeze(0)
    torch.manual_seed(rng_seed)
    ref_codes = cpu.acoustic_transformer(hidden_in, cfg_alpha).long()
    torch.manual_seed(rng_seed)
    tt_codes = pipe.acoustic_codes_forward(hidden_in, cfg_alpha).long()
    tt_codes = _align_to_ref_shape(ref_codes, tt_codes)
    return ref_codes, tt_codes


def _run_pipeline_inference_pcc_loop(
    pipe: VoxtralTTSPipeline,
    cpu: VoxtralCPUReference,
    *,
    prompt_ids: list[int],
    prompt_len: int,
    generate_steps: int,
    acoustic_hidden_source: Literal["cpu", "tt"],
    feedback_codes: Literal["reference", "tt"],
) -> None:
    """Chained text->acoustic->text loop with per-step PCC."""
    _log_stage_header("FULL PIPELINE - chained text -> acoustic -> text PCC")
    logger.info(
        f"  acoustic_hidden_source={acoustic_hidden_source} feedback_codes={feedback_codes} "
        f"prompt_len={prompt_len} max_steps={generate_steps}"
    )
    _, cpu_embeds = cpu._prompt_embeddings(prompt_ids, _DEMO_VOICE)
    cpu_prefill = cpu.text_model(
        inputs_embeds=cpu_embeds.unsqueeze(0),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cpu_hidden = cpu_prefill.hidden_states[-1][:, -1, :].squeeze(0).float()
    cpu_pkv = cpu_prefill.past_key_values

    tt_embeds = pipe._build_voice_injected_embeds(prompt_ids, _DEMO_VOICE)
    tt_hidden = pipe.text.prefill_from_embeds(tt_embeds, start_pos=0).float()
    current_pos = prompt_len

    ok, msg = comp_pcc(cpu_hidden, tt_hidden, pcc=PREFILL_HIDDEN_PCC)
    _log_pcc("prefill hidden", float(msg), PREFILL_HIDDEN_PCC)
    assert ok, f"prefill hidden PCC failed: {msg}"

    cfg_alpha = torch.tensor(cpu._acoustic_cfg_alpha, device=cpu_hidden.device, dtype=cpu.dtype)
    stacked_codes: list[torch.Tensor] = []
    acoustic_matches = 0
    acoustic_total = 0

    for step in range(generate_steps):
        if acoustic_hidden_source == "cpu":
            acoustic_hidden = cpu_hidden
        else:
            acoustic_hidden = tt_hidden
        hidden_bf16 = acoustic_hidden.to(torch.bfloat16)

        ref_codes, tt_codes = _acoustic_codes_with_rng(
            pipe,
            cpu,
            hidden_bf16=hidden_bf16,
            cfg_alpha=cfg_alpha,
            rng_seed=10_000 + step,
        )
        assert torch.equal(ref_codes[:, :1], tt_codes[:, :1]), f"step={step} semantic token mismatch"
        n_acoustic = ref_codes.shape[1] - 1
        if n_acoustic > 0:
            step_matches = int((ref_codes[:, 1:] == tt_codes[:, 1:]).sum().item())
            step_match_frac = step_matches / n_acoustic
            acoustic_matches += step_matches
            acoustic_total += n_acoustic
            logger.info(
                f"  step={step} acoustic semantic={int(tt_codes[0, 0].item())} "
                f"agreement={step_match_frac:.4f} ({step_matches}/{n_acoustic})"
            )

        feedback = tt_codes if feedback_codes == "tt" else ref_codes
        stacked_codes.append(feedback[0].detach().cpu())

        if int(feedback[0, 0].item()) == cpu.end_audio_id:
            break

        mm_embed = pipe._audio_codes_to_mm_embed(feedback)
        cpu_hidden, cpu_pkv = _cpu_text_decode_step(cpu, audio_codes_b37=feedback, past_key_values=cpu_pkv)
        tt_hidden = pipe.text.decode_step_from_embeds(mm_embed, current_pos).float()
        current_pos += 1

        ok, msg = comp_pcc(cpu_hidden.float(), tt_hidden, pcc=TEXT_DECODE_STEP_PCC)
        _log_pcc(f"text hidden step={step} pos={current_pos - 1}", float(msg), TEXT_DECODE_STEP_PCC)
        assert ok, f"text decode step={step} pos={current_pos - 1} hidden PCC failed: {msg}"

    assert stacked_codes, "inference loop produced no acoustic frames"
    if acoustic_total > 0:
        match_frac = acoustic_matches / acoustic_total
        logger.info(
            f"  acoustic code agreement summary: {match_frac:.4f}  "
            f"target>={ACOUSTIC_MATCH_FRAC:.4f}  matched={acoustic_matches}/{acoustic_total}"
        )
        assert (
            match_frac >= ACOUSTIC_MATCH_FRAC
        ), f"acoustic code agreement {match_frac:.4f} < {ACOUSTIC_MATCH_FRAC} over {generate_steps} steps"

    stacked = torch.stack(stacked_codes, dim=0)
    eoa = (stacked[:, 0] == cpu.end_audio_id).nonzero(as_tuple=False)
    cut = int(eoa[0].item()) if len(eoa) else stacked.shape[0]
    shifted = stacked[:cut]
    audio_tokens = shifted - 2
    codes_b37t = audio_tokens.T.unsqueeze(0).long()

    ref_wav = audio_tokenizer_decode_reference(codes_b37t, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
    tt_wav = pipe.decode_waveform_from_codes_tt(codes_b37t)
    ref_wav = ref_wav.reshape(1, 1, -1)
    tt_wav = tt_wav.reshape(1, 1, -1)[:, :, : ref_wav.shape[-1]]

    ok, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=WAVEFORM_PCC)
    _log_pcc("tokenizer waveform from stacked codes", float(msg), WAVEFORM_PCC)
    logger.info(f"  generated frames used for tokenizer PCC={codes_b37t.shape[2]}")
    assert ok, f"tokenizer waveform PCC failed: {msg}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("generate_steps", [8], ids=["8_steps"])
def test_voxtral_tts_pipeline_inference(device, reset_seeds, generate_steps):
    """Full pipeline PCC: CPU hidden + reference acoustic codes (teacher-forced)."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=512)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    request = compose_speech_request(_DEMO_TEXT, name, voice=_DEMO_VOICE, ref_audio=None)
    prompt_ids = request["prompt_token_ids"]
    _run_pipeline_inference_pcc_loop(
        pipe,
        cpu,
        prompt_ids=prompt_ids,
        prompt_len=len(prompt_ids),
        generate_steps=generate_steps,
        acoustic_hidden_source="cpu",
        feedback_codes="reference",
    )


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_generate_smoke(device, reset_seeds):
    """Full generate_with_codes to [END_AUDIO]; tokenizer PCC on TT-generated codes."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=512)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    out = pipe.generate_with_codes(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=512,
        seed=0,
    )

    assert out.codes_b37t.dim() == 3 and tuple(out.codes_b37t.shape[:2]) == (1, 37)
    assert out.codes_b37t.shape[2] > 0, "full generate produced no acoustic frames"
    assert torch.isfinite(out.waveform).all(), "full generate produced non-finite waveform samples"
    assert out.hit_end_audio, "full generate did not reach [END_AUDIO] within max_tokens"

    num_frames = int(out.codes_b37t.shape[2])
    pcc_target = _waveform_pcc_target_for_frames(num_frames)
    if num_frames > FULL_GENERATE_SHORT_MAX_FRAMES:
        _log_generated_code_tokenizer_diagnostics(pipe, out.codes_b37t)
    ref_wav = audio_tokenizer_decode_reference(
        out.codes_b37t, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args
    )
    ref_wav = ref_wav.reshape(1, 1, -1)[:, :, : out.waveform.shape[-1]]
    ok, msg = comp_pcc(ref_wav.float(), out.waveform.float(), pcc=pcc_target)
    _log_stage_header("FULL GENERATE - output-complete tokenizer self-consistency")
    _log_pcc("TT waveform vs CPU decode of TT codes", float(msg), pcc_target)
    logger.info(
        f"  frames={num_frames} pcc_target={pcc_target:.4f} "
        f"waveform shape={tuple(out.waveform.shape)} hit_end_audio={out.hit_end_audio}"
    )
    assert ok, f"generate smoke waveform PCC failed (frames={num_frames}, target={pcc_target}): {msg}"
