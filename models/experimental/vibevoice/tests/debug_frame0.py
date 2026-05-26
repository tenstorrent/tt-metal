# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Debug frame 0 diffusion conditions.
Run via pytest:
  pytest models/experimental/vibevoice/tests/debug_frame0.py -s -v
"""
import sys
from pathlib import Path
import pytest
import torch

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.experimental.vibevoice.common.config import MODEL_PATH, DEFAULT_TXT_PATH, VOICES_DIR
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).is_dir(),
    reason=f"VibeVoice weights not found at {MODEL_PATH}",
)

CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
MAX_NEW_TOKENS = 5  # just get first couple of frames


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_debug_frame0(mesh_device):
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    script = open(DEFAULT_TXT_PATH).read().strip().replace("'", "'")
    voice_path = str(_VOICE_PATH) if _VOICE_PATH.is_file() else str(next(iter(VOICES_DIR.glob("*.wav"))))

    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(NUM_DIFFUSION_STEPS)

    inputs = processor(
        text=[script], voice_samples=[[voice_path]], padding=True, return_tensors="pt", return_attention_mask=True
    )

    # ── Reference: capture frame 0 conditions ────────────────────────────────
    import types

    # Patch acoustic tokenizer encode to capture fix-std noise shape
    orig_encode = ref_model.model.acoustic_tokenizer.encode

    def patched_encode(audio, **kw):
        result = orig_encode(audio, **kw)
        print(f"[DEBUG REF] acoustic encode mean shape: {result.mean.shape}, numel={result.mean.numel()}")
        return result

    ref_model.model.acoustic_tokenizer.encode = patched_encode

    # Patch VibeVoiceTokenizerEncoderOutput.sample to capture noise
    import vibevoice.modular.modular_vibevoice_tokenizer as vtok

    orig_sample_enc = vtok.VibeVoiceTokenizerEncoderOutput.sample

    def patched_sample_enc(self, dist_type="fix"):
        result = orig_sample_enc(self, dist_type)
        if dist_type == "fix":
            noise_size = self.mean.numel()
            print(f"[DEBUG REF] fix-std noise drawn: {noise_size} values (mean shape={self.mean.shape})")
        return result

    vtok.VibeVoiceTokenizerEncoderOutput.sample = patched_sample_enc

    captured_ref = {}
    orig_sample = ref_model.sample_speech_tokens.__func__

    def patched_sample(self, cond_pos, cond_neg, cfg_scale=3.0):
        if "pos" not in captured_ref:
            captured_ref["pos"] = cond_pos.detach().clone()
            captured_ref["neg"] = cond_neg.detach().clone()
            print(f"[DEBUG REF frame0] cond_pos norm={cond_pos.norm():.4f} first5={cond_pos.reshape(-1)[:5].tolist()}")
            print(f"[DEBUG REF frame0] cond_neg norm={cond_neg.norm():.4f} first5={cond_neg.reshape(-1)[:5].tolist()}")
        result = orig_sample(self, cond_pos, cond_neg, cfg_scale)
        if "latent" not in captured_ref:
            captured_ref["latent"] = result.detach().clone()
            print(f"[DEBUG REF frame0] raw latent first5={result.reshape(-1)[:5].tolist()}")
        return result

    ref_model.sample_speech_tokens = types.MethodType(patched_sample, ref_model)

    torch.manual_seed(0)
    with torch.no_grad():
        ref_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            cfg_scale=CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=True,
        )

    # ── TT model ─────────────────────────────────────────────────────────────
    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        MODEL_PATH,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
    )
    tt_model.set_cpu_acoustic_decoder(ref_model.model.acoustic_tokenizer)

    torch.manual_seed(0)
    with torch.no_grad():
        tt_out = tt_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            speech_tensors=inputs.get("speech_tensors"),
            speech_masks=inputs.get("speech_masks"),
            speech_input_mask=inputs.get("speech_input_mask"),
            tokenizer=processor.tokenizer,
            cfg_scale=CFG_SCALE,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # Compute condition PCC
    if "pos" in captured_ref:
        pass

        ref_pos = captured_ref["pos"].reshape(-1).float()
        ref_neg = captured_ref["neg"].reshape(-1).float()
        # Need TT conditions — already printed, but also need them as tensors
        # They're captured in the debug print in generator — let's use norms as proxy
        print(f"\n[DEBUG] ref_pos norm={ref_pos.norm():.4f}, ref_neg norm={ref_neg.norm():.4f}")
        if "latent" in captured_ref:
            ref_lat = captured_ref["latent"].reshape(-1).float()
            print(f"[DEBUG] ref latent norm={ref_lat.norm():.4f}")

    print("[DEBUG] Test complete — check [DEBUG TT frame0] and [DEBUG REF frame0] lines above")
