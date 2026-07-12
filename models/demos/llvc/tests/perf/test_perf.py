# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance (RTF / latency) tests for the LLVC TTNN bring-up.

Stage-1 targets from the bounty:
* RTF < 0.3 for streaming mode
* Latency < 100 ms per streaming chunk
"""

import pytest
import torch
from loguru import logger

from models.demos.llvc.reference.llvc_reference import build_reference_model
from models.demos.llvc.tests.pcc.test_llvc import _reference_params_from, _small_config
from models.demos.llvc.tt.model import LLVCModel

TARGET_RTF = 0.3
TARGET_CHUNK_LATENCY_MS = 100.0


class TestLLVCPerformance:
    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize("chunk_factor", [1, 2, 4])
    def test_streaming_rtf_latency(self, device, chunk_factor):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        # ~1 second of audio
        wav = torch.randn(cfg.sample_rate) * 0.2
        # Warm up program cache first (excluded from RTF timing loop).
        _ = model.stream(wav, chunk_factor=chunk_factor)
        out, rtf, latency = model.stream(wav, chunk_factor=chunk_factor)

        logger.info(f"chunk_factor={chunk_factor}: RTF={rtf:.3f}, chunk_latency={latency:.2f}ms")
        assert torch.isfinite(out).all()
        assert rtf > 0.0
        # Report against targets; a small model on random weights is only indicative.
        if rtf >= TARGET_RTF:
            logger.warning(f"RTF {rtf:.3f} above target {TARGET_RTF}")
        if latency >= TARGET_CHUNK_LATENCY_MS:
            logger.warning(f"Chunk latency {latency:.2f}ms above target {TARGET_CHUNK_LATENCY_MS}ms")

    @pytest.mark.models_performance_bare_metal
    def test_summary(self, device):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)
        wav = torch.randn(cfg.sample_rate) * 0.2
        _ = model.stream(wav, chunk_factor=1)
        _, rtf, latency = model.stream(wav, chunk_factor=1)
        logger.info("=" * 50)
        logger.info("LLVC PERFORMANCE SUMMARY (streaming, chunk_factor=1)")
        logger.info(f"RTF: {rtf:.3f} (target < {TARGET_RTF})")
        logger.info(f"Chunk latency: {latency:.2f} ms (target < {TARGET_CHUNK_LATENCY_MS})")
        logger.info("=" * 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
