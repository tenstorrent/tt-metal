"""Performance profiling test for the full audio decoder.

Runs a single e2e pass: FSQ dequant → fc_post_a → VocosBackbone(12L) → ISTFT.
All decoder blocks are chained on device.

Usage:
    # Profile with Tracy:
    python3 -m tracy -p -v -r --op-support-count 30000 -m pytest models/demos/inworld_tts/tests/test_perf_decoder.py -v -s

    # Run without profiler:
    pytest models/demos/inworld_tts/tests/test_perf_decoder.py -v -s
"""

import pytest
import torch
from vector_quantize_pytorch import ResidualFSQ

import ttnn
from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder
from models.demos.inworld_tts.tt.profile_e2e import make_decoder_state_dict


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    yield dev
    ttnn.close_device(dev)


class TestPerfDecoder:
    def test_full_decoder_single_pass(self, device):
        """Single e2e decoder pass for profiling.

        Pipeline: VQ codes → FSQ dequant → fc_post_a Linear(2048,1024)
                  → VocosBackbone(12L) → ISTFT head Linear(1024,1282) → ISTFT FFT → audio
        All blocks chained on device.
        """
        torch.manual_seed(42)
        seq_len = 64
        depth = 12

        sd = make_decoder_state_dict(depth=depth)
        quantizer = ResidualFSQ(levels=[4, 4, 4, 4, 4, 4, 4, 4], dim=2048, num_quantizers=1)
        decoder = TtCodecDecoder(device=device, state_dict=sd, quantizer=quantizer, depth=depth)

        vq_codes = torch.randint(0, 65536, (1, 1, seq_len))

        with torch.no_grad():
            audio = decoder(vq_codes)

        expected_samples = seq_len * 320  # hop_length=320
        assert audio.shape == (1, 1, expected_samples), f"Expected (1, 1, {expected_samples}), got {audio.shape}"
        print(f"Decoder output: {audio.shape} samples = {expected_samples/16000*1000:.0f}ms audio")
