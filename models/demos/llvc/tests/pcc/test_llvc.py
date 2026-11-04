# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC / correctness tests for the LLVC TTNN bring-up.

These compare the TTNN model against the PyTorch reference sharing the *same*
random weights, so any divergence is due to the TTNN op implementations, not
weight mismatch.
"""

import pytest
import torch

from models.demos.llvc.eval.evaluate import reference_stream
from models.demos.llvc.reference.llvc_reference import build_reference_model
from models.demos.llvc.tt.config import LLVCConfig
from models.demos.llvc.tt.model import LLVCModel

# bf16 device numerics vs fp32 torch; identical weights and chunking.
E2E_VS_REFERENCE_PCC = 0.99
TRACED_VS_EAGER_PCC = 0.99
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


def _fused_sdpa_config() -> LLVCConfig:
    """Like ``_small_config`` but with head_dim 32 so the fused SDPA kernel runs.

    ``_small_config`` has dec_dim=64 / nhead=4 -> head_dim 16, which always takes
    the manual matmul+softmax fallback in ``ops.scaled_dot_product_attention``
    (head_dim not a TILE_SIZE multiple). dec_dim=128 / nhead=4 -> head_dim 32
    exercises the fused ``ttnn.transformer`` SDPA path against the reference.
    """
    cfg = _small_config()
    cfg.dec_dim = 128
    return cfg


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
    @pytest.mark.parametrize("config_builder", [_small_config, _fused_sdpa_config], ids=["manual-sdpa", "fused-sdpa"])
    @pytest.mark.parametrize("chunk_factor", [1, 2])
    def test_e2e_pcc_vs_reference(self, device, torch_seed, config_builder, chunk_factor):
        """TTNN streaming vs the PyTorch reference under *identical* chunking.

        Scores against ``reference_stream`` (same chunk split / lookahead
        alignment as ``LLVCModel.stream``) rather than the offline
        ``reference(wav)`` forward, whose ``pad=True`` alignment differs and
        previously forced a loose 0.90 gate. Same-chunking comparison isolates
        real numerical divergence, so the gate is tight. ``chunk_factor=2``
        gives multiple decoder windows per chunk (nw >= 2).
        """
        cfg = config_builder()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        # Several chunks, so the traced path (capture on chunk 1, replay after)
        # is exercised, not just the eager warmup chunk.
        n_samples = cfg.dec_chunk_size * cfg.L * chunk_factor * 4
        wav = torch.randn(n_samples) * 0.2

        ref_out = reference_stream(ref, wav, L=cfg.L, dec_chunk_size=cfg.dec_chunk_size, chunk_factor=chunk_factor)
        tt_out, _ = model.stream(wav, chunk_factor=chunk_factor)

        assert tt_out.shape == ref_out.shape, f"{tt_out.shape} vs {ref_out.shape}"
        pcc = compute_pcc(tt_out, ref_out)
        print(f"E2E PCC vs reference_stream ({config_builder.__name__}, chunk_factor={chunk_factor}): {pcc:.4f}")
        assert pcc > E2E_VS_REFERENCE_PCC, f"E2E PCC {pcc:.4f} <= {E2E_VS_REFERENCE_PCC}"

    def test_traced_stream_matches_eager(self, device, torch_seed):
        """Trace-replayed streaming matches the eager per-chunk path.

        A previous version compared ``model.stream(wav, chunk_factor=1)`` with
        ``model(wav)``, but ``__call__`` *is* ``stream(chunk_factor=1)`` — the
        same function twice. Comparing a traced model against an eager
        (``use_trace=False``) one with identical weights actually validates
        trace capture/replay equivalence.
        """
        cfg = _small_config()
        assert cfg.use_trace, "default config must exercise the traced path"
        ref = build_reference_model(_reference_params_from(cfg))
        model_traced = LLVCModel(cfg, ref, device=device)

        cfg_eager = _small_config()
        cfg_eager.use_trace = False
        model_eager = LLVCModel(cfg_eager, ref, device=device)

        wav = torch.randn(cfg.dec_chunk_size * cfg.L * 4) * 0.2
        traced_out, metrics = model_traced.stream(wav, chunk_factor=1)
        eager_out, _ = model_eager.stream(wav, chunk_factor=1)
        assert torch.isfinite(traced_out).all()
        assert traced_out.shape == eager_out.shape, f"{traced_out.shape} vs {eager_out.shape}"
        pcc = compute_pcc(traced_out, eager_out)
        print(
            f"traced vs eager streaming PCC={pcc:.4f} steady-state_RTF={metrics.rtf:.3f} "
            f"latency={metrics.latency_ms:.2f}ms (device_RTF={metrics.device_rtf:.3f})"
        )
        assert pcc > TRACED_VS_EAGER_PCC, f"traced vs eager PCC {pcc:.4f} <= {TRACED_VS_EAGER_PCC}"

    @pytest.mark.parametrize("chunk_factor", [1, 2])
    def test_batched_streams_match_sequential(self, device, torch_seed, chunk_factor):
        """Public API concurrent streams: batched [B, T] matches per-stream B=1.

        ``chunk_factor=2`` covers the B >= 2 with multiple windows per chunk
        case: ``_build_windows`` stacks windows window-major, and the decoder
        fold must convert to batch-major or frames interleave across streams
        (the bug previously measured as PCC ~0.5 on real checkpoints).
        """
        cfg = _small_config()
        cfg.use_trace = False
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        n = cfg.dec_chunk_size * cfg.L * chunk_factor * 4
        a = torch.randn(n) * 0.2
        b = torch.randn(n) * 0.2
        out_a, _ = model.stream(a, chunk_factor=chunk_factor)
        out_b, _ = model.stream(b, chunk_factor=chunk_factor)
        batched, _ = model.stream(torch.stack([a, b], dim=0), chunk_factor=chunk_factor)
        assert batched.shape[0] == 2
        pcc_a = compute_pcc(batched[0], out_a)
        pcc_b = compute_pcc(batched[1], out_b)
        print(f"batched vs sequential (chunk_factor={chunk_factor}) PCC: stream0={pcc_a:.4f} stream1={pcc_b:.4f}")
        assert pcc_a > BATCHED_VS_SEQUENTIAL_PCC, f"batched[0] PCC {pcc_a:.4f} <= {BATCHED_VS_SEQUENTIAL_PCC}"
        assert pcc_b > BATCHED_VS_SEQUENTIAL_PCC, f"batched[1] PCC {pcc_b:.4f} <= {BATCHED_VS_SEQUENTIAL_PCC}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
