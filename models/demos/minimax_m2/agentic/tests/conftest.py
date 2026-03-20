# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest fixtures for device-level agentic tests.

Session-scoped fixtures:
  mesh_device   - opened N300 MeshDevice (closed at session teardown)
  all_models    - full ModelBundle loaded from device
  llm_tool      - LLMTool only (for LLM-specific tests)
  whisper_tool  - WhisperTool only
  speecht5_tool - SpeechT5Tool only
  owlvit_tool   - OWLViTTool only
  bert_tool     - BERTTool only

Function-scoped fixtures (cheap):
  test_audio_path       - path to a synthetic 2-second 16 kHz mono WAV
  test_audio_path_long  - path to a synthetic 5-second WAV with spoken-ish content
  test_image_path       - path to a synthetic 512×512 RGB image (coloured blocks)
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_sine_wave_wav(path: str, duration: float = 2.0, sample_rate: int = 16000, freq: float = 440.0):
    """Write a mono sine-wave WAV at *path* and return the path."""
    import soundfile as sf

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    sf.write(path, audio, sample_rate)
    return path


def _make_test_image(path: str, width: int = 512, height: int = 512):
    """Write a synthetic RGB image with a few coloured blocks and return the path."""
    from PIL import Image

    data = np.zeros((height, width, 3), dtype=np.uint8)
    # Red block top-left
    data[40:200, 40:200] = [220, 30, 30]
    # Blue block top-right
    data[40:200, 312:472] = [30, 30, 220]
    # Green block bottom-centre
    data[312:472, 176:336] = [30, 180, 30]
    Image.fromarray(data, "RGB").save(path)
    return path


# ---------------------------------------------------------------------------
# Session-scoped device
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mesh_device():
    """Open the N300 MeshDevice once for the entire test session."""
    import ttnn
    from models.demos.minimax_m2.agentic.loader import open_n300_device

    device = open_n300_device()
    yield device
    ttnn.close_mesh_device(device)


# ---------------------------------------------------------------------------
# Session-scoped individual model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llm_tool(mesh_device):
    """Llama 3.2 3B Instruct on TTNN – loaded once."""
    from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

    return LLMTool(mesh_device=mesh_device)


@pytest.fixture(scope="session")
def whisper_tool(mesh_device):
    """Whisper distil-large-v3 on TTNN – loaded once."""
    from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

    return WhisperTool(mesh_device=mesh_device)


@pytest.fixture(scope="session")
def speecht5_tool(mesh_device):
    """SpeechT5 TTS on TTNN – loaded once."""
    from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

    return SpeechT5Tool(mesh_device=mesh_device)


@pytest.fixture(scope="session")
def owlvit_tool(mesh_device):
    """OWL-ViT zero-shot detection on TTNN – loaded once."""
    from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

    return OWLViTTool(mesh_device=mesh_device)


@pytest.fixture(scope="session")
def bert_tool(mesh_device):
    """BERT Large QA on TTNN – loaded once."""
    from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

    return BERTTool(mesh_device=mesh_device)


@pytest.fixture(scope="session")
def all_models(mesh_device):
    """All five models loaded into one ModelBundle – loaded once."""
    from models.demos.minimax_m2.agentic.loader import load_all_models

    return load_all_models(mesh_device)


# ---------------------------------------------------------------------------
# Function-scoped synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_audio_path(tmp_path):
    """2-second 16 kHz mono sine-wave WAV."""
    wav = str(tmp_path / "test_audio.wav")
    _make_sine_wave_wav(wav, duration=2.0, sample_rate=16000, freq=440.0)
    return wav


@pytest.fixture
def test_audio_path_long(tmp_path):
    """5-second 16 kHz WAV with a chord (more Whisper-friendly)."""
    import soundfile as sf

    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Simple chord: 261 Hz (C4) + 329 Hz (E4) + 392 Hz (G4)
    audio = (
        np.sin(2 * np.pi * 261 * t) * 0.2 + np.sin(2 * np.pi * 329 * t) * 0.2 + np.sin(2 * np.pi * 392 * t) * 0.2
    ).astype(np.float32)
    path = str(tmp_path / "test_audio_long.wav")
    sf.write(path, audio, sample_rate)
    return path


@pytest.fixture
def test_image_path(tmp_path):
    """512×512 RGB image with coloured blocks."""
    img = str(tmp_path / "test_image.png")
    _make_test_image(img)
    return img


@pytest.fixture
def tts_output_path(tmp_path):
    """Output path for TTS-generated audio."""
    return str(tmp_path / "tts_output.wav")


# ---------------------------------------------------------------------------
# BERT test data
# ---------------------------------------------------------------------------

BERT_CONTEXT = (
    "The Tenstorrent N300 is an AI accelerator board featuring two Wormhole B0 chips "
    "connected via a high-bandwidth Ethernet link. Each chip has 80 Tensix cores and "
    "12 GB of GDDR6 memory, giving the board 24 GB total. "
    "The N300 is designed for inference workloads at the edge and in the data centre."
)

BERT_QUESTION = "How many chips does the N300 have?"
BERT_EXPECTED_SUBSTRING = "two"  # extractive answer should contain "two"
