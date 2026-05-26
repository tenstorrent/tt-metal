# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forced diffusion PCC test: reference vs TTNN prediction_head only.

This is NOT an end-to-end generate test.  It isolates the diffusion head
(prediction_head) by teacher-forcing: per-frame initial noise and conditions
are captured from one reference run, then replayed through the TT DPM-Solver.
The AR loop and acoustic decoder are entirely the reference implementation.

For the true end-to-end test (reference AR+diffusion vs TT AR+diffusion,
no teacher forcing) see test_e2e_full_generate_pcc.py.

Strategy:
  Phase 1: Run reference generate(seed=0).
    - Capture per-frame: initial noise x_T, pos_cond, neg_cond at the first
      DPM step of each diffusion frame (new frame detected when DPM timestep
      resets upward, e.g. 100 → 999).
    - Capture ref_speech (ground truth audio).

  Phase 2: For each captured frame, run TT DPM-Solver (10 steps) with the
    same initial noise and conditions (float32 → bfloat16). Decode TT latents
    with the reference streaming acoustic decoder (fresh streaming cache).

  The ONLY difference between ref_speech and tt_speech is the prediction_head
  compute path:
    - Reference: float32 CPU
    - TT: bfloat16 TTNN device
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import (
    MODEL_PATH,
    DEFAULT_TXT_PATH,
    VOICES_DIR,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import sample_speech_latents

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"
_TEXT_PATH = DEFAULT_TXT_PATH

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).is_dir(),
    reason=f"VibeVoice weights not found at {MODEL_PATH} (set VIBEVOICE_MODEL_PATH)",
)

CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
SPEECH_PCC = 0.99
MAX_NEW_TOKENS = 128


def _load_script() -> str:
    assert _TEXT_PATH.is_file()
    with open(_TEXT_PATH, encoding="utf-8") as f:
        return f.read().strip().replace("’", "'")


def _voice_path() -> str:
    if _VOICE_PATH.is_file():
        return str(_VOICE_PATH)
    wavs = list(VOICES_DIR.glob("*.wav"))
    assert wavs, f"No voice WAV in {VOICES_DIR}"
    return str(wavs[0])


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.as_tensor(
        t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_generate_speech_pcc(mesh_device):
    """TT diffusion head PCC vs reference — teacher-forced per-frame comparison.

    Phase 1: Run reference generate (seed=0), capture:
      - ref_speech (ground truth audio)
      - Per-frame: initial noise x_T, pos_cond, neg_cond at the first DPM step.
        (DPM timesteps decrease within a frame; a new frame starts when t resets
        to the maximum value.)

    Phase 2: For each captured frame, run TT DPM-Solver (num_steps) with the
      same initial noise and conditions (converted float32 → bfloat16). Decode
      TT latents with the reference streaming acoustic decoder.

    The ONLY difference is the prediction_head compute path (float32 vs bfloat16).
    """
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    script = _load_script()
    voice_path = _voice_path()

    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)

    inputs = processor(
        text=[script],
        voice_samples=[[voice_path]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # ── Phase 1: Reference run — capture per-frame initial noise + conditions ─
    per_frame_captures = []
    _prev_t = [None]
    original_forward = ref_model.model.prediction_head.forward

    def _capture_forward(x, t, condition=None):
        t_val = float(t[0].item()) if hasattr(t, "dim") else float(t)
        # A new diffusion frame starts when t resets upward (previous t was lower).
        is_new_frame = _prev_t[0] is None or t_val > _prev_t[0]
        if is_new_frame:
            B = x.shape[0]
            half = max(1, B // 2)  # CFG batch=2 → half=1; non-CFG batch=1 → half=1
            pos_cond_cpu = condition[:half].detach().cpu().float() if condition is not None else None
            neg_cond_cpu = (
                condition[half:].detach().cpu().float() if condition is not None and condition.shape[0] > half else None
            )
            per_frame_captures.append(
                {
                    "x_T": x[:half].detach().cpu().float(),
                    "pos_cond": pos_cond_cpu,
                    "neg_cond": neg_cond_cpu,
                }
            )
        _prev_t[0] = t_val
        return original_forward(x, t, condition)

    ref_model.model.prediction_head.forward = _capture_forward

    torch.manual_seed(0)
    with torch.no_grad():
        ref_out = ref_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            cfg_scale=CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=True,
        )

    ref_model.model.prediction_head.forward = original_forward

    assert ref_out.speech_outputs and ref_out.speech_outputs[0] is not None
    ref_speech = ref_out.speech_outputs[0].to(torch.float32).reshape(-1)
    assert ref_speech.numel() > 1000, "Reference produced no speech"

    n_frames = len(per_frame_captures)
    assert n_frames > 0, "No diffusion frames captured from reference run"

    # ── Build TT model ────────────────────────────────────────────────────────
    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        MODEL_PATH,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
    )
    generator = tt_model._make_generator(processor.tokenizer, CFG_SCALE, NUM_DIFFUSION_STEPS, MAX_NEW_TOKENS)
    tt_diffusion_head = generator.diffusion_head
    tt_scheduler = generator.scheduler

    scale = tt_model._speech_scaling_factor
    bias = tt_model._speech_bias_factor
    if scale is None or bias is None:
        # Compute scale/bias from the voice input (matches reference model)
        generator._process_speech_prefill(inputs["speech_tensors"], inputs["speech_masks"])
        scale = generator.speech_scaling_factor or 1.0
        bias = generator.speech_bias_factor or 0.0

    # ── Phase 2: TT DPM-Solver per captured frame ─────────────────────────────
    try:
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerStreamingCache,
        )

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        _use_cache = True
    except ImportError:
        acoustic_cache = None
        _use_cache = False

    acoustic_decoder = ref_model.model.acoustic_tokenizer
    decoder_device = next(acoustic_decoder.parameters()).device
    sample_idx = torch.tensor([0], dtype=torch.long).to(decoder_device)

    audio_chunks = []
    for frame_data in per_frame_captures:
        x_T = frame_data["x_T"]  # [1, D] float32 initial noise
        pos_cond = frame_data["pos_cond"]  # [1, D] float32
        neg_cond = frame_data["neg_cond"]  # [1, D] float32 or None

        x_T_tt = _to_tt(x_T.view(1, 1, 1, -1).to(torch.bfloat16), mesh_device)
        pos_cond_tt = _to_tt(pos_cond.view(1, 1, 1, -1).to(torch.bfloat16), mesh_device)
        if neg_cond is not None:
            neg_cond_tt = _to_tt(neg_cond.view(1, 1, 1, -1).to(torch.bfloat16), mesh_device)
        else:
            neg_cond_tt = pos_cond_tt  # fallback: no CFG

        # Run TT DPM-Solver (10 steps) with captured initial noise + conditions.
        # sample_speech_latents batches as [neg, pos] internally for CFG.
        latent_tt = sample_speech_latents(
            tt_diffusion_head,
            pos_cond_tt,  # condition (positive)
            neg_cond_tt,  # neg_condition (negative / uncond)
            tt_scheduler,
            x_T_tt,
            cfg_scale=CFG_SCALE,
            num_steps=NUM_DIFFUSION_STEPS,
        )

        # Un-normalize: diffusion space → acoustic VAE space
        latent_cpu = ttnn.to_torch(latent_tt).to(torch.float32)  # [1, 1, 1, D]
        latent_unscaled = (latent_cpu / scale - bias).view(1, 1, -1)  # [1, 1, D]

        # Decode with reference streaming acoustic decoder (streaming cache carries
        # causal-conv state across frames, matching the reference's exact decode path).
        with torch.no_grad():
            if _use_cache:
                chunk = acoustic_decoder.decode(
                    latent_unscaled.to(decoder_device),
                    cache=acoustic_cache,
                    sample_indices=sample_idx,
                    use_cache=True,
                )
            else:
                chunk = acoustic_decoder.decode(latent_unscaled.to(decoder_device))
        audio_chunks.append(chunk.to(torch.float32).cpu())

    tt_speech = torch.cat(audio_chunks, dim=-1).reshape(-1)
    assert tt_speech.numel() > 1000, "TT produced no speech"

    n = min(ref_speech.numel(), tt_speech.numel())
    passed, pcc_val = comp_pcc(ref_speech[:n], tt_speech[:n], pcc=SPEECH_PCC)
    assert passed, (
        f"Speech PCC {pcc_val:.6f} < {SPEECH_PCC} "
        f"(ref len={ref_speech.numel()}, tt len={tt_speech.numel()}, n_frames={n_frames})"
    )
