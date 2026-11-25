# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC / correctness tests for the LLVC TTNN bring-up.

These compare the TTNN model against the PyTorch reference sharing the *same*
random weights, so any divergence is due to the TTNN op implementations, not
weight mismatch.
"""

import pytest
import torch

import ttnn
from models.demos.llvc.eval.evaluate import reference_stream
from models.demos.llvc.reference.llvc_reference import build_reference_model
from models.demos.llvc.tt.config import LLVCConfig
from models.demos.llvc.tt.model import LLVCModel, reference_params_from_config

# bf16 device numerics vs fp32 torch; identical weights and chunking.
E2E_VS_REFERENCE_PCC = 0.99
TRACED_VS_EAGER_PCC = 0.99
BATCHED_VS_SEQUENTIAL_PCC = 0.99
# Scored at the decoder output: with random weights the final waveform barely
# depends on the decoder, but here zeroed attention (~0.84-0.90) or a
# window-major fold (~0.80) falls well below the gate.
DECODER_VS_REFERENCE_PCC = 0.99


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().float()
    y = b.flatten().float()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return 0.0  # NaN/Inf output must fail the PCC gates, not pass them
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum()) * torch.sqrt((vy * vy).sum())
    if denom.item() == 0.0:
        # Zero variance: identical constants are a (degenerate) match, a
        # constant vs non-constant signal is a mismatch.
        return float(torch.equal(x, y))
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
    (head_dim not a TILE_SIZE multiple). nhead=2 -> head_dim 32 exercises the
    fused ``ttnn.transformer`` SDPA path. dec_dim stays 64: raising it to enc_dim
    would turn the e2d/d2e projections into one-channel-per-group convs that the
    shipped 512->256 model never runs.
    """
    cfg = _small_config()
    cfg.nhead = 2
    return cfg


# Single source of truth for config -> reference kwargs lives next to
# ``create_llvc`` so tests and production construction cannot drift.
_reference_params_from = reference_params_from_config


class TestLLVCReference:
    def test_reference_forward_shape(self, torch_seed):
        cfg = _small_config()
        ref = build_reference_model(_reference_params_from(cfg))
        wav = torch.randn(1, 1, cfg.dec_chunk_size * cfg.L * 2)
        with torch.no_grad():
            out = ref(wav)
        assert out.shape[0] == 1 and out.shape[1] == 1
        assert torch.isfinite(out).all()


def _device_btc(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(x.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


class TestLLVCDecoder:
    def test_window_fold_is_batch_major(self, device):
        """Decoder windows fold back to time per stream, not interleaved across streams.

        Each value encodes ``(stream, frame)`` exactly in bf16 (< 256), so any
        row mix-up in the fold shows up as an exact mismatch.
        """
        cfg = _small_config()
        model = LLVCModel(cfg, build_reference_model(_reference_params_from(cfg)), device=device)
        batch, num_windows, cs, d = 2, 3, cfg.dec_chunk_size, cfg.dec_dim
        frames = torch.arange(num_windows * cs, dtype=torch.float32)
        expected = (torch.arange(batch, dtype=torch.float32)[:, None] * 64 + frames[None, :])[:, :, None]
        expected = expected.expand(batch, num_windows * cs, d).contiguous()
        window_major = expected.reshape(batch, num_windows, cs, d).transpose(0, 1).reshape(num_windows * batch, cs, d)

        folded = model._fold_windows_to_time(_device_btc(window_major, device), batch, num_windows)
        folded = ttnn.to_torch(folded).float()

        assert folded.shape == expected.shape, f"{tuple(folded.shape)} vs {tuple(expected.shape)}"
        assert torch.equal(folded, expected), "decoder window fold mixes frames across streams"

    @pytest.mark.parametrize("config_builder", [_small_config, _fused_sdpa_config], ids=["manual-sdpa", "fused-sdpa"])
    @pytest.mark.parametrize("batch", [1, 2])
    def test_decoder_matches_reference(self, device, torch_seed, config_builder, batch):
        """TTNN causal decoder vs ``ref.mask_gen.decoder`` over two windows, per stream."""
        cfg = config_builder()
        ref = build_reference_model(_reference_params_from(cfg))
        model = LLVCModel(cfg, ref, device=device)

        num_frames = cfg.dec_chunk_size * 2
        m = torch.relu(torch.randn(batch, num_frames, cfg.dec_dim))
        e = torch.relu(torch.randn(batch, num_frames, cfg.dec_dim))

        decoder = ref.mask_gen.decoder
        with torch.no_grad():
            ref_out, _ = decoder(m.transpose(1, 2), e.transpose(1, 2), decoder.init_ctx_buf(batch, "cpu"))
        ref_out = ref_out.transpose(1, 2)

        tt_out = model._decoder(_device_btc(m, device), _device_btc(e, device), model.init_state(batch))
        tt_out = ttnn.to_torch(tt_out).float()

        assert tt_out.shape == ref_out.shape, f"{tuple(tt_out.shape)} vs {tuple(ref_out.shape)}"
        for b in range(batch):
            pcc = compute_pcc(tt_out[b], ref_out[b])
            print(f"decoder PCC vs reference ({config_builder.__name__}, stream {b}/{batch}): {pcc:.4f}")
            assert pcc > DECODER_VS_REFERENCE_PCC, f"stream {b} decoder PCC {pcc:.4f} <= {DECODER_VS_REFERENCE_PCC}"


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

        With random weights the waveform barely depends on the decoder, so this
        cannot detect a cross-stream window fold; ``TestLLVCDecoder`` and
        ``tests/perf/test_checkpoint_ci.py`` (real weights) cover that.
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
