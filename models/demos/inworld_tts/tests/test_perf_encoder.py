"""Performance profiling test for the full audio encoder.

Runs a single e2e pass: W2V → Semantic → Acoustic → Fusion → FSQ.
All encoder blocks are chained on device via forward_on_device().

Usage:
    # Profile with Tracy:
    python3 -m tracy -p -v -r --op-support-count 30000 -m pytest models/demos/inworld_tts/tests/test_perf_encoder.py -v -s

    # Run without profiler:
    pytest models/demos/inworld_tts/tests/test_perf_encoder.py -v -s
"""

import pytest
import torch
from vector_quantize_pytorch import ResidualFSQ

import ttnn
from models.demos.inworld_tts.tt.codec_encoder import TtCodecEncoder
from models.demos.inworld_tts.tt.model_config import ENCODER_TOTAL_STRIDE
from models.demos.inworld_tts.tt.profile_e2e import make_acoustic_encoder_state_dict


def _bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    yield dev
    ttnn.close_device(dev)


def _build_encoder(device, seq_len=64):
    """Build the full encoder with random weights."""
    C = 1024
    prefix = "SemanticEncoder_module."
    sd = {
        prefix + "initial_conv.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "initial_conv.bias": _bf16(torch.randn(C)),
        prefix + "residual_blocks.1.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "residual_blocks.1.bias": _bf16(torch.randn(C)),
        prefix + "residual_blocks.3.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "residual_blocks.3.bias": _bf16(torch.randn(C)),
        prefix + "final_conv.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "final_conv.bias": _bf16(torch.randn(C)),
        "fc_prior.weight": _bf16(torch.randn(2048, 2048)),
        "fc_prior.bias": _bf16(torch.randn(2048)),
    }
    asd = make_acoustic_encoder_state_dict()
    for k, v in asd.items():
        sd["CodecEnc." + k] = v

    quantizer = ResidualFSQ(levels=[4, 4, 4, 4, 4, 4, 4, 4], dim=2048, num_quantizers=1)
    return TtCodecEncoder(device, sd, quantizer=quantizer)


class TestPerfEncoder:
    def test_full_encoder_single_pass(self, device):
        """Single e2e encoder pass for profiling.

        Pipeline: mel → W2V(16L) → Semantic → waveform → Acoustic(5 blocks) → Fusion → FSQ
        All blocks chained on device via forward_on_device().
        """
        torch.manual_seed(42)
        seq_len = 64
        n_samples = seq_len * ENCODER_TOTAL_STRIDE

        encoder = _build_encoder(device, seq_len)

        waveform = _bf16(torch.randn(1, 1, n_samples))
        mel_features = torch.randn(1, seq_len, 160).to(torch.bfloat16).to(torch.float32)

        with torch.no_grad():
            vq_codes = encoder.forward_on_device(waveform, mel_features=mel_features)

        assert vq_codes.shape == (1, 1, seq_len), f"Expected (1, 1, {seq_len}), got {vq_codes.shape}"
        print(f"Encoder output: {vq_codes.shape} VQ codes from {n_samples/16000*1000:.0f}ms audio")
