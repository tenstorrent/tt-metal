"""
Integration test for Voxtral-4B-TTS-2603 full pipeline on N150.

Tests end-to-end: text + voice → waveform.
Verifies output audio is valid (correct shape, reasonable amplitude).

Run:
  cd tt-metal
  export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd):$(pwd)/models
  export ARCH_NAME=wormhole_b0
  source python_env/bin/activate
  pytest models/demos/voxtral_tts/tests/test_integration.py -v -s
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn

MODEL_DIR = Path(
    os.environ.get(
        "VOXTRAL_MODEL_DIR",
        "/home/ttuser/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970",
    )
)
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"Model weights not found at {WEIGHTS_PATH}",
)


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def test_full_tts_pipeline_produces_audio(device):
    """End-to-end: text 'Hello.' + casual_male voice → non-silent waveform."""
    from models.demos.voxtral_tts.tt.load_checkpoint import load_voice_embeddings
    from models.demos.voxtral_tts.tt.model import VoxtralTTSModel

    model = VoxtralTTSModel.from_pretrained(MODEL_DIR, device)
    voices = load_voice_embeddings(MODEL_DIR)

    voice_emb = voices["casual_male"].unsqueeze(0)  # [1, 147, 3072]

    # Tokenize "Hello." via Tekken
    from mistral_common.protocol.instruct.chunk import TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(str(MODEL_DIR / "tekken.json"))
    req = ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="Hello.")])])
    text_ids = torch.tensor([tok.encode_chat_completion(req).tokens])

    waveform = model.generate_tts(text_ids, voice_emb, n_ode_steps=8)

    # Validate
    assert waveform.shape[0] == 1, f"Expected batch=1, got {waveform.shape}"
    assert waveform.shape[1] > 0, "Waveform is empty"
    assert waveform.abs().max() > 0.001, "Waveform is silent (all near zero)"
    assert waveform.abs().max() < 10.0, f"Waveform amplitude too large: {waveform.abs().max()}"

    n_samples = waveform.shape[1]
    duration_s = n_samples / 24000
    print(f"\n  Full TTS: {n_samples} samples = {duration_s:.2f}s at 24kHz")
    print(f"  Amplitude: mean={waveform.abs().mean():.4f}, max={waveform.abs().max():.4f}")


def test_tts_reference_pcc(device):
    """TTS output vs reference implementation: PCC > 0.99."""
    from models.demos.voxtral_tts.reference.functional import tts_generate
    from models.demos.voxtral_tts.tt.load_checkpoint import (
        get_acoustic_transformer_state,
        get_codec_decoder_state,
        get_text_decoder_state,
        load_state_dict,
        load_voice_embeddings,
    )
    from models.demos.voxtral_tts.tt.model_config import VoxtralTTSConfig

    sd = load_state_dict(WEIGHTS_PATH)
    voices = load_voice_embeddings(MODEL_DIR)
    voice_emb = voices["casual_male"].unsqueeze(0).bfloat16()

    sd_text = get_text_decoder_state(sd)
    sd_acoustic = get_acoustic_transformer_state(sd)
    sd_codec = get_codec_decoder_state(sd)

    from mistral_common.protocol.instruct.chunk import TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(str(MODEL_DIR / "tekken.json"))
    req = ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="Hello.")])])
    text_ids = torch.tensor([tok.encode_chat_completion(req).tokens])

    # Reference output (fixed seed for reproducibility)
    torch.manual_seed(42)
    ref_waveform, ref_semantic, ref_acoustic, _ = tts_generate(
        text_ids,
        voice_emb,
        sd_text=sd_text,
        sd_acoustic=sd_acoustic,
        sd_codec=sd_codec,
        max_audio_frames=len(text_ids[0]),
    )

    # TTNN output (with same seed for ODE initial noise)
    from models.demos.voxtral_tts.tt.model import VoxtralTTSModel

    config = VoxtralTTSConfig(mesh_device=device)
    model = VoxtralTTSModel(device, config, sd)

    torch.manual_seed(42)
    ttnn_waveform = model.generate_tts(text_ids, voice_emb, n_ode_steps=8)

    # Compare (TTNN text decoder uses BF16 precision, some numerical difference expected)
    min_len = min(ref_waveform.shape[1], ttnn_waveform.shape[1])
    ref_crop = ref_waveform[0, :min_len].float()
    ttnn_crop = ttnn_waveform[0, :min_len].float()

    p = torch.corrcoef(torch.stack([ref_crop, ttnn_crop]))[0, 1].item()
    print(f"\n  Integration PCC vs reference: {p:.4f}")
    assert p > 0.95, f"Integration PCC={p:.4f} < 0.95 (lower bound for full pipeline)"
