# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance (RTF / latency) tests for the LLVC TTNN bring-up.

Stage-1 targets from the bounty:
* RTF < 0.3 for streaming mode
* Latency < 100 ms per streaming chunk

Primary assertions use *end-to-end* RTF/latency (per-chunk H2D upload + device
execute + D2H download). Device-only figures are logged for comparison; see
``StreamMetrics`` / README for the transfer-cost caveat.
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
        out, metrics = model.stream(wav, chunk_factor=chunk_factor)

        logger.info(
            "chunk_factor={}: e2e_RTF={:.3f} e2e_latency={:.2f}ms | device_RTF={:.3f} device_latency={:.2f}ms",
            chunk_factor,
            metrics.rtf,
            metrics.latency_ms,
            metrics.device_rtf,
            metrics.device_latency_ms,
        )
        assert torch.isfinite(out).all()
        assert metrics.rtf > 0.0
        assert metrics.rtf < TARGET_RTF, f"e2e RTF {metrics.rtf:.3f} >= target {TARGET_RTF}"
        assert (
            metrics.latency_ms < TARGET_CHUNK_LATENCY_MS
        ), f"e2e chunk latency {metrics.latency_ms:.2f}ms >= target {TARGET_CHUNK_LATENCY_MS}ms"

    @pytest.mark.models_performance_bare_metal
    def test_summary(self, device):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)
        wav = torch.randn(cfg.sample_rate) * 0.2
        _ = model.stream(wav, chunk_factor=1)
        _, metrics = model.stream(wav, chunk_factor=1)
        logger.info("=" * 50)
        logger.info("LLVC PERFORMANCE SUMMARY (streaming, chunk_factor=1)")
        logger.info(f"e2e RTF: {metrics.rtf:.3f} (target < {TARGET_RTF})")
        logger.info(f"e2e chunk latency: {metrics.latency_ms:.2f} ms (target < {TARGET_CHUNK_LATENCY_MS})")
        logger.info(f"device RTF: {metrics.device_rtf:.3f}")
        logger.info(f"device chunk latency: {metrics.device_latency_ms:.2f} ms")
        logger.info("=" * 50)
        assert metrics.rtf < TARGET_RTF, f"e2e RTF {metrics.rtf:.3f} >= target {TARGET_RTF}"
        assert (
            metrics.latency_ms < TARGET_CHUNK_LATENCY_MS
        ), f"e2e chunk latency {metrics.latency_ms:.2f}ms >= target {TARGET_CHUNK_LATENCY_MS}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
