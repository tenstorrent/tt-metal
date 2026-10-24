# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC / correctness tests for the LLVC TTNN bring-up.

These compare the TTNN model against the PyTorch reference sharing the *same*
random weights, so any divergence is due to the TTNN op implementations, not
weight mismatch.
"""

import pytest
import torch

from models.demos.llvc.reference.llvc_reference import build_reference_model
from models.demos.llvc.tt.config import LLVCConfig
from models.demos.llvc.tt.model import LLVCModel

# Streaming and offline share the chunked graph; allow a small numeric gap.
STREAM_VS_OFFLINE_PCC = 0.99
BATCHED_VS_SEQUENTIAL_PCC = 0.99


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().float()
    y = b.flatten().float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum()) * torch.sqrt((vy * vy).sum())
    if denom.item() == 0.0:
        return 1.0
    corr = (vx * vy).sum().item() / (denom.item() + 1e-8)
    return max(-1.0, min(1.0, corr))


def _small_config() -> LLVCConfig:
    """A smaller LLVC variant that keeps runtime/L1 modest for CI."""
    from models.demos.llvc.tt.config import ConvNetConfig

    return LLVCConfig(
        L=16,
        enc_dim=128,
        num_enc_layers=3,
        dec_dim=64,
        num_dec_layers=1,
        dec_buf_len=13,
        dec_chunk_size=13,
        out_buf_len=4,
        nhead=4,  # must match the PyTorch reference (no longer hardcoded to 8)
        convnet=ConvNetConfig(out_channels=tuple([1] * 3), kernel_sizes=tuple([3] * 3), dilations=tuple([1] * 3)),
    )


def _reference_params_from(config: LLVCConfig) -> dict:
    return dict(
        label_len=1,
        L=config.L,
        enc_dim=config.enc_dim,
        num_enc_layers=config.num_enc_layers,
        dec_dim=config.dec_dim,
        num_dec_layers=config.num_dec_layers,
        dec_buf_len=config.dec_buf_len,
        dec_chunk_size=config.dec_chunk_size,
        out_buf_len=config.out_buf_len,
        nhead=config.nhead,
        use_pos_enc=config.use_pos_enc,
        decoder_dropout=0.0,
        convnet_config=dict(
            convnet_prenet=config.convnet.convnet_prenet,
            out_channels=list(config.convnet.out_channels),
            kernel_sizes=list(config.convnet.kernel_sizes),
            dilations=list(config.convnet.dilations),
            dropout=0.0,
            combine_residuals=config.convnet.combine_residuals,
            skip_connection=config.convnet.skip_connection,
            use_residual_blocks=config.convnet.use_residual_blocks,
        ),
    )


class TestLLVCReference:
    def test_reference_forward_shape(self, torch_seed):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        wav = torch.randn(1, 1, cfg.dec_chunk_size * cfg.L * 2)
        with torch.no_grad():
            out = ref(wav)
        assert out.shape[0] == 1 and out.shape[1] == 1
        assert torch.isfinite(out).all()


class TestLLVCEndToEnd:
    @pytest.mark.parametrize("chunk_factor", [1, 2])
    def test_e2e_pcc_vs_reference(self, device, torch_seed, chunk_factor):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        n_samples = cfg.dec_chunk_size * cfg.L * chunk_factor
        wav = torch.randn(1, 1, n_samples) * 0.2

        with torch.no_grad():
            ref_out = ref(wav)
        tt_out = model(wav)

        assert tt_out.shape == ref_out.shape, f"{tt_out.shape} vs {ref_out.shape}"
        pcc = compute_pcc(tt_out, ref_out)
        print(f"E2E PCC (chunk_factor={chunk_factor}): {pcc:.4f}")
        assert pcc > 0.90, f"E2E PCC {pcc:.4f} < 0.90"

    def test_streaming_matches_nonstreaming(self, device, torch_seed):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        wav = torch.randn(cfg.dec_chunk_size * cfg.L * 4) * 0.2
        stream_out, metrics = model.stream(wav, chunk_factor=1)
        offline_out = model(wav)
        assert torch.isfinite(stream_out).all()
        assert stream_out.shape == offline_out.shape, f"{stream_out.shape} vs {offline_out.shape}"
        pcc = compute_pcc(stream_out, offline_out)
        print(
            f"streaming vs offline PCC={pcc:.4f} e2e_RTF={metrics.rtf:.3f} "
            f"e2e_latency={metrics.latency_ms:.2f}ms (device_RTF={metrics.device_rtf:.3f})"
        )
        assert pcc > STREAM_VS_OFFLINE_PCC, f"streaming vs non-streaming PCC {pcc:.4f} <= {STREAM_VS_OFFLINE_PCC}"

    def test_batched_streams_match_sequential(self, device, torch_seed):
        """Public API concurrent streams: batched [B, T] matches per-stream B=1."""
        cfg = _small_config()
        cfg.use_trace = False
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        n = cfg.dec_chunk_size * cfg.L * 4
        a = torch.randn(n) * 0.2
        b = torch.randn(n) * 0.2
        out_a, _ = model.stream(a, chunk_factor=1)
        out_b, _ = model.stream(b, chunk_factor=1)
        batched, _ = model.stream(torch.stack([a, b], dim=0), chunk_factor=1)
        assert batched.shape[0] == 2
        pcc_a = compute_pcc(batched[0], out_a)
        pcc_b = compute_pcc(batched[1], out_b)
        print(f"batched vs sequential PCC: stream0={pcc_a:.4f} stream1={pcc_b:.4f}")
        assert pcc_a > BATCHED_VS_SEQUENTIAL_PCC, f"batched[0] PCC {pcc_a:.4f} <= {BATCHED_VS_SEQUENTIAL_PCC}"
        assert pcc_b > BATCHED_VS_SEQUENTIAL_PCC, f"batched[1] PCC {pcc_b:.4f} <= {BATCHED_VS_SEQUENTIAL_PCC}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
