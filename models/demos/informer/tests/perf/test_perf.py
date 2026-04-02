# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.informer.reference.torch_reference import TorchInformerModel, compute_metrics, ttnn_state_dict
from models.demos.informer.tests.perf.perf_common import (
    LONG_SEQ_MAX_LATENCY_MS,
    TARGET_CORRELATION,
    TARGET_LATENCY_MS,
    TARGET_MAE,
    TARGET_MSE,
    TARGET_THROUGHPUT,
    WIDE_MAX_LATENCY_MS,
    BenchmarkConfig,
    build_config,
    make_inputs,
    run_benchmark,
)
from models.demos.informer.tt.config import InformerConfig
from models.demos.informer.tt.model import InformerModel, create_informer
from models.demos.informer.tt.ops import make_causal_mask, to_torch


class TestInformerPerformance:
    """Core performance checks for the fixed high-optimization profile."""

    @pytest.mark.models_performance_bare_metal
    def test_throughput_latency_sweep(self, device):
        cfg = build_config(seq_len=96, pred_len=24, label_len=48)
        model = create_informer(cfg, device=device)

        results = []
        for batch in [1, 2, 4, 8, 16]:
            past_values, past_time, future_time = make_inputs(batch, cfg)
            throughput, latency_ms = run_benchmark(
                model,
                past_values,
                past_time,
                future_time,
                device=device,
                warmup=3,
                iters=20,
            )
            results.append((batch, throughput, latency_ms))
            logger.info(f"Batch {batch}: {throughput:.1f} seq/s, {latency_ms:.2f}ms")

        max_throughput = max(r[1] for r in results)
        batch1_latency = next(lat for (b, _, lat) in results if b == 1)

        logger.info(f"Max throughput: {max_throughput:.1f} seq/s (target: >= {TARGET_THROUGHPUT})")
        logger.info(f"Batch=1 latency: {batch1_latency:.2f}ms (target: < {TARGET_LATENCY_MS}ms)")

        assert max_throughput >= TARGET_THROUGHPUT, f"Throughput {max_throughput:.1f} < {TARGET_THROUGHPUT}"
        assert batch1_latency < TARGET_LATENCY_MS, f"Latency {batch1_latency:.2f}ms > {TARGET_LATENCY_MS}ms"
        model.release_trace()

    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "seq_len,pred_len,batch",
        [
            (96, 24, 8),
            (336, 96, 8),
            (720, 96, 4),
        ],
    )
    def test_sequence_scaling(self, device, seq_len, pred_len, batch):
        cfg = build_config(seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2)
        model = create_informer(cfg, device=device)
        past_values, past_time, future_time = make_inputs(batch, cfg)

        throughput, latency_ms = run_benchmark(
            model,
            past_values,
            past_time,
            future_time,
            device=device,
            warmup=2,
            iters=10,
        )

        logger.info(f"Seq {seq_len} pred {pred_len} batch {batch}: {throughput:.1f} seq/s, {latency_ms:.2f}ms")
        assert throughput > 0
        model.release_trace()

    @pytest.mark.models_performance_bare_metal
    def test_trace_replay_benchmark_stability(self, device):
        # Both models are forced to high profile; this checks replay stability only.
        cfg = build_config(seq_len=96, pred_len=24, label_len=48)
        past_values, past_time, future_time = make_inputs(8, cfg)

        warm_model = create_informer(cfg, device=device)
        replay_model = create_informer(cfg, device=device)

        warm_throughput, warm_latency = run_benchmark(
            warm_model,
            past_values,
            past_time,
            future_time,
            device=device,
            warmup=2,
            iters=20,
        )
        _ = replay_model(past_values, past_time, future_time)
        replay_throughput, replay_latency = run_benchmark(
            replay_model,
            past_values,
            past_time,
            future_time,
            device=device,
            warmup=2,
            iters=20,
        )

        logger.info(f"trace_warm: {warm_throughput:.1f} seq/s, {warm_latency:.2f}ms")
        logger.info(f"trace_replay: {replay_throughput:.1f} seq/s, {replay_latency:.2f}ms")
        ratio = replay_throughput / max(warm_throughput, 1e-6)
        if ratio < 0.9:
            logger.warning(f"Trace replay throughput is {ratio:.2f}x of warm traced run; investigate overhead.")
        assert ratio >= 0.5, "Trace replay regressed severely vs warm traced run"

        warm_model.release_trace()
        replay_model.release_trace()

    @pytest.mark.models_performance_bare_metal
    def test_accuracy_against_reference(self, device):
        cfg = build_config(seq_len=96, pred_len=24, label_len=48)
        torch_model = TorchInformerModel(cfg)
        torch_model.eval()
        state = ttnn_state_dict(torch_model)

        ttnn_model = InformerModel(cfg, device=device, seed=42)
        ttnn_model.load_state_dict(state, strict=True)

        past_values, past_time, future_time = make_inputs(2, cfg)
        with torch.no_grad():
            torch_out = torch_model(past_values, past_time, future_time)
        ttnn_out = to_torch(ttnn_model(past_values, past_time, future_time))

        mse, mae, corr = compute_metrics(ttnn_out, torch_out)
        logger.info(f"MSE: {mse:.6f}, MAE: {mae:.6f}, Corr: {corr:.4f}")
        assert mse < TARGET_MSE
        assert mae < TARGET_MAE
        assert corr > TARGET_CORRELATION

    @pytest.mark.models_performance_bare_metal
    def test_benchmark_summary(self, device):
        cfg = build_config(seq_len=96, pred_len=24, label_len=48)
        model = create_informer(cfg, device=device)

        bench = BenchmarkConfig(seq_len=cfg.seq_len, label_len=cfg.label_len, pred_len=cfg.pred_len, batch=8)
        past_values, past_time, future_time = make_inputs(bench.batch, cfg)
        throughput, _ = run_benchmark(
            model,
            past_values,
            past_time,
            future_time,
            device=device,
            warmup=5,
            iters=20,
        )

        past_values_1, past_time_1, future_time_1 = make_inputs(1, cfg)
        _, latency_ms = run_benchmark(
            model,
            past_values_1,
            past_time_1,
            future_time_1,
            device=device,
            warmup=5,
            iters=20,
        )

        torch_model = TorchInformerModel(cfg)
        torch_model.eval()
        state = ttnn_state_dict(torch_model)
        ttnn_model = InformerModel(cfg, device=device, seed=42)
        ttnn_model.load_state_dict(state, strict=True)
        acc_past_values, acc_past_time, acc_future_time = make_inputs(2, cfg)
        ttnn_out = to_torch(ttnn_model(acc_past_values, acc_past_time, acc_future_time))
        with torch.no_grad():
            torch_out = torch_model(acc_past_values, acc_past_time, acc_future_time)
        mse, mae, corr = compute_metrics(ttnn_out, torch_out)

        logger.info("=" * 50)
        logger.info("INFORMER PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Throughput (batch={bench.batch}): {throughput:.1f} seq/s")
        logger.info(f"Latency (batch=1): {latency_ms:.2f}ms")
        logger.info(f"MSE vs Reference: {mse:.6f}")
        logger.info(f"MAE vs Reference: {mae:.6f}")
        logger.info(f"Correlation: {corr:.4f}")
        logger.info("=" * 50)

        assert throughput >= TARGET_THROUGHPUT
        assert latency_ms < TARGET_LATENCY_MS
        assert mse < TARGET_MSE
        assert mae < TARGET_MAE
        assert corr > TARGET_CORRELATION

        model.release_trace()


class TestInformerAdvancedCapabilities:
    """Advanced capability checks under the same fixed high profile."""

    @pytest.mark.models_performance_bare_metal
    def test_prob_sparse_query_reduction(self, device):
        cfg = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=512,
            label_len=256,
            pred_len=96,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=2,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
            attention_type="prob",
            use_sdpa=True,
        )
        model = InformerModel(cfg, device=device, seed=0)
        past_values, past_time, future_time = make_inputs(1, cfg)
        _ = model(past_values, past_time, future_time)

        stats = model.encoder.layers[0].attn.last_attention_stats
        assert stats.get("mode") == "prob_sparse"
        assert stats.get("q_valid_len", 0) >= 512
        assert stats.get("top_u", 0) < stats.get("q_valid_len", 0)
        reduction_ratio = stats["top_u"] / max(1, stats["q_valid_len"])
        assert reduction_ratio <= 0.25, f"ProbSparse reduction too weak: ratio={reduction_ratio:.4f}"
        assert stats.get("used_sdpa") is True, "ProbSparse path did not use SDPA kernel."

    @pytest.mark.models_performance_bare_metal
    def test_decode_cache_path_matches_eager_sdpa(self, device):
        cfg = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=96,
            label_len=48,
            pred_len=24,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=1,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
            attention_type="prob",
            use_sdpa=True,
        )
        model = InformerModel(cfg, device=device, seed=0)
        attn = model.decoder.layers[0].self_attn
        seq = 8
        torch.manual_seed(0)
        x = torch.randn(1, seq, cfg.d_model, dtype=torch.float32)
        x_tt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        full_mask = make_causal_mask(
            seq,
            batch=1,
            heads=1,
            device=device,
            dtype=ttnn.bfloat16,
            mask_value=cfg.attn_mask_value,
        )
        eager = attn(x_tt, x_tt, x_tt, full_mask, q_valid_len=seq, k_valid_len=seq)
        eager_t = to_torch(eager).float()

        cache = {"k": None, "v": None, "valid_len": 0}
        chunks = []
        decode_steps = 2
        for token_idx in range(decode_steps):
            token_tt = ttnn.slice(x_tt, [0, token_idx, 0], [1, token_idx + 1, cfg.d_model])
            out_tt, cache = attn.call_with_cache(
                token_tt,
                token_tt,
                token_tt,
                None,
                kv_cache=cache,
                q_valid_len=1,
                k_valid_len=1,
                is_causal=True,
            )
            chunks.append(out_tt)
        cached = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=1)
        cached_t = to_torch(cached).float()
        eager_prefix = eager_t[:, :decode_steps, :]

        mse, mae, corr = compute_metrics(cached_t, eager_prefix)
        assert mse < 5e-3, f"Decode-cache MSE {mse:.6f} >= 5e-3"
        assert mae < 5e-2, f"Decode-cache MAE {mae:.6f} >= 5e-2"
        assert math.isfinite(corr), f"Decode-cache corr is non-finite: {corr}"
        assert bool(attn.last_attention_stats.get("used_sdpa")), "Decode cache path did not use SDPA."

    @pytest.mark.models_performance_bare_metal
    def test_trace_replay_matches_eager(self, device):
        cfg = build_config(seq_len=96, pred_len=24, label_len=48)
        traced_model = create_informer(cfg, device=device)
        eager_model = InformerModel(cfg, device=device)
        past_values, past_time, future_time = make_inputs(2, cfg)

        eager = to_torch(
            eager_model.forward_ttnn(
                eager_model.to_device(past_values),
                eager_model.to_device(past_time),
                eager_model.to_device(future_time),
            )
        ).float()
        traced_first = to_torch(traced_model(past_values, past_time, future_time)).float()
        traced_replay = to_torch(traced_model(past_values, past_time, future_time)).float()
        traced_model.release_trace()

        mse_first, _, corr_first = compute_metrics(traced_first, eager)
        mse_replay, _, corr_replay = compute_metrics(traced_replay, eager)
        assert mse_first < 1e-3
        assert corr_first > 0.999
        assert mse_replay < 1e-3
        assert corr_replay > 0.999

    @pytest.mark.models_performance_bare_metal
    def test_single_factory_profile_contract(self, device):
        cfg = build_config(seq_len=96, pred_len=24, label_len=48)
        factory_model = create_informer(cfg, device=device)
        direct_model = InformerModel(cfg, device=device, seed=0)

        for model in (factory_model, direct_model):
            assert model.config.use_trace is True
            assert model.config.use_sharded is True
            assert model.config.use_sdpa is True
            assert model.config.use_l1 is True
            assert model.config.use_program_cache is True

    @pytest.mark.models_performance_bare_metal
    def test_streaming_decoder_matches_full(self, device):
        cfg = build_config(seq_len=336, pred_len=96, label_len=168)
        model = InformerModel(cfg, device=device, seed=0)
        past_values, past_time, future_time = make_inputs(2, cfg)

        out_full = model(past_values, past_time, future_time)
        out_stream = model.stream_forecast(past_values, past_time, future_time, chunk_size=64)

        full_torch = to_torch(out_full).float()
        stream_torch = to_torch(out_stream).float()
        assert full_torch.shape == stream_torch.shape
        diff = torch.mean((full_torch - stream_torch) ** 2).item()
        assert diff < 1e-3

    @pytest.mark.models_performance_bare_metal
    def test_long_sequence_support(self, device):
        cfg = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=20000,
            label_len=512,
            pred_len=96,
            d_model=32,
            n_heads=1,
            d_ff=128,
            e_layers=2,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
            attention_type="prob",
            use_sdpa=True,
        )
        model = InformerModel(cfg, device=device, seed=0)
        past_values, past_time, future_time = make_inputs(1, cfg)
        # Warm one streaming pass to exclude one-time compile/program-cache overhead from latency gate.
        _ = model.stream_forecast(past_values, past_time, future_time, chunk_size=128)
        ttnn.synchronize_device(device)
        start = time.time()
        out_chunk_128 = model.stream_forecast(past_values, past_time, future_time, chunk_size=128)
        latency_ms = (time.time() - start) * 1000.0
        out_chunk_256 = model.stream_forecast(past_values, past_time, future_time, chunk_size=256)

        out_128 = to_torch(out_chunk_128).float()
        out_256 = to_torch(out_chunk_256).float()
        assert out_128.shape == (1, cfg.pred_len, cfg.c_out)
        assert torch.isfinite(out_128).all()
        assert out_256.shape == out_128.shape

        mse_stream, _, corr_stream = compute_metrics(out_128, out_256)
        assert mse_stream < 1e-3
        assert corr_stream > 0.99
        assert latency_ms <= LONG_SEQ_MAX_LATENCY_MS

    @pytest.mark.models_performance_bare_metal
    def test_high_dimensional_inputs(self, device):
        cfg = InformerConfig(
            enc_in=128,
            dec_in=128,
            c_out=128,
            seq_len=96,
            label_len=48,
            pred_len=24,
            d_model=128,
            n_heads=4,
            d_ff=256,
            e_layers=2,
            d_layers=1,
            time_feature_dim=8,
            dtype="bfloat16",
            attention_type="prob",
            use_sdpa=True,
        )
        torch_model = TorchInformerModel(cfg)
        torch_model.eval()
        state = ttnn_state_dict(torch_model)
        model = InformerModel(cfg, device=device, seed=0)
        model.load_state_dict(state, strict=True)
        past_values = torch.randn(2, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
        past_time = torch.randn(2, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(2, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)
        start = time.time()
        out = model(past_values, past_time, future_time)
        latency_ms = (time.time() - start) * 1000.0
        out_torch = to_torch(out)
        with torch.no_grad():
            ref_torch = torch_model(past_values, past_time, future_time)
        assert out_torch.shape == (2, cfg.pred_len, cfg.c_out)
        assert torch.isfinite(out_torch).all()
        mse, mae, corr = compute_metrics(out_torch, ref_torch)
        assert mse < 0.1
        assert mae < 0.1
        assert corr > 0.95
        assert latency_ms <= WIDE_MAX_LATENCY_MS

    @pytest.mark.models_performance_bare_metal
    def test_multi_horizon_consistency(self, device):
        cfg_max = build_config(seq_len=96, pred_len=720, label_len=48)
        cfg_short = build_config(seq_len=96, pred_len=168, label_len=48)
        model_max = InformerModel(cfg_max, device=device, seed=0)
        model_short = InformerModel(cfg_short, device=device, seed=0)
        past_values, past_time, future_time = make_inputs(1, cfg_max)

        out_max = model_max(past_values, past_time, future_time)
        out_short = model_short(past_values, past_time, future_time[:, : cfg_short.pred_len, :])
        out_max_torch = to_torch(out_max)[:, : cfg_short.pred_len, :].float()
        out_short_torch = to_torch(out_short).float()
        assert out_short_torch.shape == out_max_torch.shape
        diff = torch.mean((out_short_torch - out_max_torch) ** 2).item()
        assert diff < 1e-3
