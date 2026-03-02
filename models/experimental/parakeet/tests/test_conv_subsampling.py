# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test: PyTorch ConvSubsampling reference vs TTNNConvSubsampling with same weights.

Uses real (pretrained) weights from a NeMo Parakeet model when available; otherwise skips.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.parakeet.reference.pytorch_pre_enc import ConvSubsampling
from models.experimental.parakeet.tt.ttnn_pre_enc import TTNNConvSubsampling
from tests.ttnn.utils_for_testing import check_with_pcc


# ttnn.conv2d (sliding window/halo) needs L1; default device may have l1_small_size=0
CONV_SUBSAMPLING_L1_SMALL_SIZE = 32768

# NeMo pretrained model for real ConvSubsampling weights (encoder uses dw_striding)
PARAKEET_PRETRAINED_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def _load_pretrained_conv_subsampling(model_name=PARAKEET_PRETRAINED_MODEL):
    """Load ConvSubsampling with real weights from NeMo pretrained model.

    Returns (ref_module, feat_in, feat_out, conv_channels) or None if loading fails
    (e.g. no network, model uses different subsampling, or key mismatch).
    """
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        return None
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
    except Exception as e:
        logger.warning(f"Could not load pretrained model {model_name}: {e}")
        return None
    pre_encode = getattr(asr_model.encoder, "pre_encode", None)
    if pre_encode is None or not hasattr(pre_encode, "conv") or not hasattr(pre_encode, "out"):
        logger.warning("Pretrained encoder has no pre_encode ConvSubsampling")
        return None
    # Expect dw_striding layout: conv.0, 2, 3, 5, 6 and out
    sd = pre_encode.state_dict()
    required = {"conv.0.weight", "conv.0.bias", "out.weight", "out.bias"}
    if not required.issubset(sd.keys()):
        logger.warning(f"Pretrained subsampling state_dict missing required keys: {required - sd.keys()}")
        return None
    feat_in = getattr(pre_encode, "_feat_in", 80)
    feat_out = getattr(pre_encode, "_feat_out", 1024)
    conv_channels = getattr(pre_encode, "_conv_channels", 256)
    ref = ConvSubsampling(feat_in=feat_in, feat_out=feat_out, conv_channels=conv_channels)
    ref.load_state_dict(sd, strict=False)
    ref = ref.to(torch.bfloat16)
    ref.eval()
    return (ref, feat_in, feat_out, conv_channels)


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONV_SUBSAMPLING_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_ttnn_conv_subsampling_pcc(device, batch_size):
    """Compare ConvSubsampling output: PyTorch reference vs TTNN with same (real) weights."""
    torch.manual_seed(0)

    loaded = _load_pretrained_conv_subsampling()
    if loaded is None:
        pytest.skip(
            "Pretrained ConvSubsampling not available (install nemo, network, or use nvidia/parakeet-tdt-0.6b-v2)"
        )
    ref, feat_in, feat_out, conv_channels = loaded
    time_steps = 64

    # TTNN module (same config as reference)
    tt_module = TTNNConvSubsampling(
        device=device,
        feat_in=feat_in,
        feat_out=feat_out,
        conv_channels=conv_channels,
    )
    # Copy weights from reference so both use identical parameters
    tt_module.set_weights_from_reference(ref)

    # Same input for both (variable lengths per batch when batch_size > 1)
    pt_input = torch.randn(batch_size, time_steps, feat_in, dtype=torch.bfloat16)
    lengths = torch.tensor([time_steps] + [time_steps // 2] * (batch_size - 1), dtype=torch.long)[:batch_size]

    # Reference forward
    with torch.no_grad():
        ref_out, ref_lengths = ref(pt_input, lengths)

    # TTNN forward: upload input to device
    tt_input = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, device=device)
    tt_out, tt_lengths = tt_module.forward(tt_input, lengths)

    # Convert TTNN output to torch
    tt_out_torch = ttnn.to_torch(tt_out)
    # TTNN may return different layout; ensure shape matches (batch, time, feat_out)
    if tt_out_torch.dim() == 4:
        tt_out_torch = tt_out_torch.squeeze(1)
    if tt_out_torch.shape != ref_out.shape:
        tt_out_torch = tt_out_torch.reshape(ref_out.shape)

    # PCC check
    passed, msg = check_with_pcc(ref_out.float(), tt_out_torch.float(), pcc=0.99)
    logger.info(f"ConvSubsampling PCC: {passed}, {msg}")

    assert passed, f"ConvSubsampling PCC failed: {msg}"
    assert ref_out.shape == tt_out_torch.shape, f"Shape mismatch: ref {ref_out.shape} vs tt {tt_out_torch.shape}"
    # Lengths should match (both are per-batch output time steps)
    if hasattr(tt_lengths, "cpu"):
        tt_len = tt_lengths.cpu()
    else:
        tt_len = torch.tensor(tt_lengths)
    assert torch.equal(ref_lengths, tt_len), f"Lengths mismatch: ref {ref_lengths.tolist()} vs tt {tt_len.tolist()}"
