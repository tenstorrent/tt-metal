# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import numpy as np
import pytest

from models.demos.ace_step_v1_5.tt_device import (
    ace_step_dit_rope_max_seq_len,
    ace_step_mesh_is_2d,
    ace_step_mesh_perf_log_default,
    ace_step_mesh_shape,
    ace_step_mesh_use_adg,
    ace_step_mesh_use_host_latent_sampler,
    ace_step_mesh_use_host_temb_precompute,
    ace_step_mesh_use_sequential_cfg,
    ace_step_mesh_use_split_ttnn_preprocess,
    ace_step_needs_split_device,
    ace_step_replicate_mesh_mapper,
    ace_step_resolve_vae_tiling,
    resolve_ace_step_mesh_sku,
)
from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import _host_gaussian_latents_f32


def test_resolve_mesh_sku_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MESH_DEVICE", "P150")
    assert resolve_ace_step_mesh_sku(cli_value="BH_QB") == "BH_QB"


def test_mesh_shape_bh_qb():
    assert ace_step_mesh_shape("BH_QB") == (2, 2)
    assert ace_step_mesh_shape("BH_LB") == (2, 4)
    assert ace_step_mesh_shape(None) == (1, 1)


def test_split_preprocess_only_for_multi_device():
    assert not ace_step_needs_split_device(None)
    assert not ace_step_needs_split_device("P150")
    assert ace_step_needs_split_device("BH_QB")
    assert ace_step_needs_split_device("BH_LB")


def test_unknown_mesh_sku_raises():
    with pytest.raises(ValueError, match="Unknown ACE-Step mesh SKU"):
        resolve_ace_step_mesh_sku(cli_value="NOT_A_SKU")


class _FakeMesh2d:
    shape = (2, 2)

    def get_num_devices(self):
        return 4


def test_mesh_is_2d():
    assert ace_step_mesh_is_2d(_FakeMesh2d())
    assert not ace_step_mesh_is_2d(_FakeMesh(4))


def test_replicate_mesh_mapper_none_for_single_device():
    assert ace_step_replicate_mesh_mapper(None) is None


def test_dit_rope_max_seq_capped_for_duration():
    # 15 s @ 25 Hz = 375 frames; patch 2 -> 188 patches + 128 margin = 316 (below 4096)
    cap = ace_step_dit_rope_max_seq_len(expected_input_length=375, patch_size=2, hf_max=4096)
    assert cap < 4096
    assert cap >= 188


def test_host_gaussian_latents_reproducible():
    a = _host_gaussian_latents_f32((1, 16, 64), seed=7)
    b = _host_gaussian_latents_f32((1, 16, 64), seed=7)
    c = _host_gaussian_latents_f32((1, 16, 64), seed=8)
    assert a.shape == (1, 16, 64)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_host_latent_sampler_on_multi_device_only():
    assert not ace_step_mesh_use_host_latent_sampler(_FakeMesh(1), use_trace=False)
    assert ace_step_mesh_use_host_latent_sampler(_FakeMesh(4), use_trace=False)
    assert not ace_step_mesh_use_host_latent_sampler(_FakeMesh(4), use_trace=True)


def test_sequential_cfg_on_multi_device_only():
    assert not ace_step_mesh_use_sequential_cfg(_FakeMesh(1), do_cfg=True)
    assert ace_step_mesh_use_sequential_cfg(_FakeMesh(4), do_cfg=True)
    assert not ace_step_mesh_use_sequential_cfg(_FakeMesh(4), do_cfg=False)


def test_dit_body_trace_safe_30s_with_sequential_cfg():
    """P300 mesh uses B=1 sequential CFG; trace must stay enabled at 30 s (fused_M=12, not 24)."""
    from models.demos.ace_step_v1_5.tt_device import ace_step_dit_pipe_batch_size
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
        ace_step_dit_body_trace_safe,
        ace_step_dit_fused_m_tiles,
    )

    patch_sz = 2
    frames_30s = 750
    patch_seq = (frames_30s + patch_sz - 1) // patch_sz
    assert ace_step_mesh_use_sequential_cfg(_FakeMesh(2), do_cfg=True)
    pipe_batch = ace_step_dit_pipe_batch_size(_FakeMesh(2), do_cfg=True)
    assert pipe_batch == 1
    fused_m = ace_step_dit_fused_m_tiles(batch_size=pipe_batch, seq_len=patch_seq)
    assert fused_m == 12
    assert ace_step_dit_body_trace_safe(batch_size=pipe_batch, patch_seq_len=patch_seq)
    assert not ace_step_dit_body_trace_safe(batch_size=2, patch_seq_len=patch_seq)


def test_host_temb_precompute_on_multi_device_only():
    assert not ace_step_mesh_use_host_temb_precompute(_FakeMesh(1))
    assert ace_step_mesh_use_host_temb_precompute(_FakeMesh(4))


def test_mesh_split_ttnn_preprocess_on_multi_device():
    assert ace_step_mesh_use_split_ttnn_preprocess("BH_QB")
    assert not ace_step_mesh_use_split_ttnn_preprocess("P150")
    assert not ace_step_mesh_use_split_ttnn_preprocess(None)


def test_mesh_use_adg_defaults():
    assert ace_step_mesh_use_adg(mesh_sku=None, variant="acestep-v15-base", cli_use_adg=None)
    assert not ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=None)
    assert ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=True)
    assert not ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-turbo", cli_use_adg=None)


def test_resolve_vae_tiling_mesh_long_clip():
    chunk, overlap = ace_step_resolve_vae_tiling(frames=375, mesh_sku="BH_QB", chunk_cli=32, overlap_cli=4)
    assert chunk == 32
    assert overlap >= 8
    chunk2, overlap2 = ace_step_resolve_vae_tiling(frames=1500, mesh_sku="BH_QB", chunk_cli=32, overlap_cli=4)
    assert chunk2 == 32
    assert overlap2 >= 14


def test_mesh_perf_log_default():
    assert ace_step_mesh_perf_log_default(mesh_sku="BH_QB")
    assert not ace_step_mesh_perf_log_default(mesh_sku="P150")
    assert not ace_step_mesh_perf_log_default(mesh_sku=None)


def test_cached_preprocess_reuse():
    from models.demos.ace_step_v1_5.demo_session import AceStepDemoSession

    session = AceStepDemoSession()
    session.store_preprocess(
        prompt="main prompt",
        duration_sec=15.0,
        seed=0,
        frames=375,
        enc_hs=object(),
        enc_mask=object(),
        ctx_lat=object(),
        null_emb=object(),
    )
    assert session.can_reuse_preprocess(prompt="main prompt", duration_sec=15.0, seed=0)
    assert not session.can_reuse_preprocess(prompt="other", duration_sec=15.0, seed=0)


def test_emit_session_summary_rollup(monkeypatch):
    monkeypatch.setenv("ACE_STEP_DEMO_PERF_LOG", "1")
    from models.demos.ace_step_v1_5.ace_step_perf_log import (
        SessionPassSnapshot,
        SessionPerfState,
        ace_step_extract_key_metrics,
        ace_step_rtf_per_step,
        emit_session_summary,
    )

    assert ace_step_rtf_per_step(wall_s=60.0, duration_sec=60.0, infer_steps=50) == pytest.approx(0.02)
    assert ace_step_rtf_per_step(wall_s=10.0, duration_sec=10.0, infer_steps=8) == pytest.approx(0.125)

    metrics = ace_step_extract_key_metrics(
        [
            ("five_hz_lm_generate", 2100.0),
            ("dit_denoise_loop", 6500.0),
            ("vae_decode", 2200.0),
        ],
        wall_ms=13000.0,
        params={"lm_num_tokens": 300, "lm_gen_time_s": 2.0},
    )
    assert metrics["wall_time_s"] == pytest.approx(13.0)
    assert metrics["lm_total_time_s"] == pytest.approx(2.0)
    assert metrics["dit_total_time_s"] == pytest.approx(6.5)
    assert metrics["vae_decode_time_s"] == pytest.approx(2.2)
    assert metrics["tokens_per_sec"] == pytest.approx(150.0)

    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_tile_physical_m_dim

    assert ace_step_tile_physical_m_dim(batch=128, one=1, seq=5) == 4096
    assert ace_step_tile_physical_m_dim(batch=1, one=1, seq=256) == 256

    state = SessionPerfState(session_t0=time.perf_counter())
    state.note_init("handler_init", 32000.0)
    state.add_pass_snapshot(
        SessionPassSnapshot(
            label="warmup_total",
            session_pass=0,
            is_warmup=True,
            total_ms=45000.0,
            modules_ms=[("dit_denoise_loop", 12000.0)],
        )
    )
    state.add_pass_snapshot(
        SessionPassSnapshot(
            label="demo_total",
            session_pass=1,
            is_warmup=False,
            total_ms=13000.0,
            modules_ms=[("dit_denoise_loop", 6500.0), ("vae_decode", 2200.0)],
        )
    )
    emit_session_summary(state, params={"duration_sec": 60.0, "infer_steps": 50, "llm_debug": True})


def test_five_hz_lm_bfloat8_optimizations_default():
    import ttnn
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
        ace_step_five_hz_lm_bfloat8_weights_enabled,
        ace_step_five_hz_lm_optimizations,
    )
    from models.tt_transformers.tt.model_config import TensorGroup

    class _FakeModelArgs:
        n_layers = 28
        model_name = "acestep-5Hz-lm-1.7B"

    assert ace_step_five_hz_lm_bfloat8_weights_enabled()
    dec = ace_step_five_hz_lm_optimizations(_FakeModelArgs())
    bf8 = getattr(ttnn, "bfloat8_b", None)
    assert bf8 is not None
    for layer in range(_FakeModelArgs.n_layers):
        for tg in (TensorGroup.FF1_FF3, TensorGroup.FF2, TensorGroup.WQKV, TensorGroup.WO, TensorGroup.KV_CACHE):
            assert dec.get_tensor_dtype(layer, tg) == bf8


def test_five_hz_lm_bfloat8_weights_env_opt_out(monkeypatch):
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_five_hz_lm_bfloat8_weights_enabled

    monkeypatch.setenv("ACE_STEP_LM_BFLOAT8_WEIGHTS", "0")
    assert not ace_step_five_hz_lm_bfloat8_weights_enabled()


def test_five_hz_lm_hifi2_optimizations_default():
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_five_hz_lm_optimizations
    from models.tt_transformers.tt.model_config import MathFidelitySetting, OpGroup

    class _FakeModelArgs:
        n_layers = 28
        model_name = "acestep-5Hz-lm-1.7B"

    dec = ace_step_five_hz_lm_optimizations(_FakeModelArgs())
    for layer in range(_FakeModelArgs.n_layers):
        for op in (
            OpGroup.LI_FF1_FF3,
            OpGroup.LI_FF2,
            OpGroup.LI_QKV_DECODE,
            OpGroup.LI_QKV_PREFILL,
            OpGroup.SDPA_DECODE,
            OpGroup.SDPA_PREFILL,
            OpGroup.LI_O_DECODE,
            OpGroup.LI_O_PREFILL,
        ):
            fid = dec.decoder_optimizations[layer].op_fidelity_settings[op]
            assert fid == MathFidelitySetting.HIFI2, f"layer={layer} op={op} got {fid}"
