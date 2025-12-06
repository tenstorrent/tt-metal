# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from models.experimental.maskformer_swin.fallback import MaskFormerFallbackPipeline
from models.experimental.maskformer_swin.ttnn_compat import ttnn
from models.experimental.maskformer_swin.weights import (
    WeightConversionConfig,
    convert_state_dict_to_tt,
    download_reference_weights,
)


_RUN_TT = os.environ.get("MASKFORMER_RUN_TT_TESTS") == "1"


@pytest.mark.skipif(not _RUN_TT, reason="Set MASKFORMER_RUN_TT_TESTS=1 to enable TT smoke test.")
@pytest.mark.skipif(ttnn is None, reason="TT-NN runtime not available.")
def test_tt_forward_rough_parity():
    """Minimal TT smoke: ensure TT path executes and roughly matches CPU outputs."""

    try:
        device = ttnn.open_device(device_id=0)
    except Exception as exc:  # pragma: no cover - depends on hardware
        pytest.skip(f"TT device unavailable: {exc}")

    weight_cfg = WeightConversionConfig()
    reference = download_reference_weights(weight_cfg)
    tt_state = convert_state_dict_to_tt(reference.state_dict, weight_cfg)

    pixel_values = torch.rand(1, 3, 128, 128)

    cpu_pipeline = MaskFormerFallbackPipeline.from_reference(reference, tt_state, device=None)
    tt_pipeline = MaskFormerFallbackPipeline.from_reference(reference, tt_state, device=device)

    with torch.no_grad():
        cpu_out = cpu_pipeline.forward(pixel_values)
        os.environ["MASKFORMER_TT_DECODER"] = "1"
        tt_out = tt_pipeline.forward(pixel_values)

    cpu_logits = cpu_out.class_logits.detach().to(torch.float32)
    tt_logits = tt_out.class_logits.detach().to(torch.float32)
    diff = (cpu_logits - tt_logits).abs()

    assert torch.isfinite(diff).all()
    # Loose tolerance: TT path is expected to be close, not identical.
    assert float(diff.mean()) < 3.0
    assert float(diff.max()) < 10.0

    os.environ["MASKFORMER_TT_DECODER"] = "0"
    try:
        ttnn.close_device(device)
    except Exception:
        pass
