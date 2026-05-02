"""
TTNN PCC tests for Voxtral-4B-TTS-2603 codec decoder.

Phase 1: Codec runs on CPU (reference implementation). PCC=1.0 by definition.
Phase 2: Will add TTNN device transformer blocks.

Run:
  cd tt-metal
  export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd):$(pwd)/models
  export ARCH_NAME=wormhole_b0
  source python_env/bin/activate
  pytest models/demos/voxtral_tts/tests/test_tt_codec_decoder.py -v -s
"""

import os
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(
    os.environ.get(
        "VOXTRAL_MODEL_DIR",
        "/home/ttuser/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970",
    )
)
GOLDEN_DIR = Path(__file__).parents[1] / "reference" / "golden"
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"
PCC_THRESHOLD = 0.99

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"Model weights not found at {WEIGHTS_PATH}",
)


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture(scope="module")
def state_dicts():
    from models.demos.voxtral_tts.tt.load_checkpoint import get_codec_decoder_state, load_state_dict

    sd = load_state_dict(WEIGHTS_PATH)
    return {"codec": get_codec_decoder_state(sd)}


@pytest.fixture(scope="module")
def goldens():
    gd = {}
    for pt_file in GOLDEN_DIR.glob("*.pt"):
        gd[pt_file.stem] = torch.load(pt_file, map_location="cpu", weights_only=True)
    return gd


def test_codec_decoder_output_shape(state_dicts, goldens):
    """Codec decoder output shape: [B, N*1920]."""
    from models.demos.voxtral_tts.tt.codec_decoder import TtVoxtralCodecDecoder

    sd = state_dicts["codec"]
    codec = TtVoxtralCodecDecoder(device=None, state_dict=sd, weight_cache_path=None, dtype=None, configuration=None)

    sem = goldens["codec_semantic_codes_input"]
    aco = goldens["codec_acoustic_codes_input"]
    B, N = sem.shape

    waveform = codec.forward(sem, aco)
    assert waveform.shape == (B, N * 1920), f"Expected ({B}, {N*1920}), got {waveform.shape}"
    print(f"\n  Codec output: {waveform.shape}, amplitude max: {waveform.abs().max():.4f}")


def test_codec_decoder_waveform_pcc(state_dicts, goldens):
    """Codec decoder waveform matches golden (CPU reference, PCC=1.0)."""
    from models.demos.voxtral_tts.tt.codec_decoder import TtVoxtralCodecDecoder

    sd = state_dicts["codec"]
    codec = TtVoxtralCodecDecoder(device=None, state_dict=sd, weight_cache_path=None, dtype=None, configuration=None)

    sem = goldens["codec_semantic_codes_input"]
    aco = goldens["codec_acoustic_codes_input"]
    ref = goldens["codec_waveform_output"]

    waveform = codec.forward(sem, aco)
    p = pcc(waveform, ref)
    print(f"\n  codec_waveform: PCC={p:.6f}")
    assert p > PCC_THRESHOLD, f"Codec waveform PCC={p:.4f} < {PCC_THRESHOLD}"


def test_codec_block0_output_pcc(state_dicts, goldens):
    """Codec initial conv (block 0) matches golden."""
    from models.demos.voxtral_tts.reference.functional import codec_decoder_forward

    sd = state_dicts["codec"]
    sem = goldens["codec_semantic_codes_input"]
    aco = goldens["codec_acoustic_codes_input"]
    ref_block0 = goldens["codec_block0_out"]  # [B, N, 1024]

    # Run with capture to get block0 intermediate
    _, caps = codec_decoder_forward(sem, aco, sd, capture_intermediates=True)
    block0_out = caps["block0_out"]

    p = pcc(block0_out, ref_block0)
    print(f"\n  codec_block0_out: PCC={p:.6f}")
    assert p > PCC_THRESHOLD, f"Block0 PCC={p:.4f} < {PCC_THRESHOLD}"
