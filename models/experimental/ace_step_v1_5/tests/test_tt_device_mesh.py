# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from models.experimental.ace_step_v1_5.utils.tt_device import (
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
from models.experimental.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import _host_gaussian_latents_f32


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
    from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_dit_pipe_batch_size
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
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
    assert ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=None)
    assert ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=True)
    assert not ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=False)
    assert not ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-turbo", cli_use_adg=None)


def test_dit_prefers_dram_at_30s_patch_seq():
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_dit_prefers_dram_activations

    # 15 s → patch_seq 188 @ patch_size=2
    assert not ace_step_dit_prefers_dram_activations(batch_size=1, seq_len=188)
    # 30 s → patch_seq 375
    assert ace_step_dit_prefers_dram_activations(batch_size=1, seq_len=375)


def test_dit_long_clip_quality_defaults(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl import math_perf_env as m

    monkeypatch.delenv("ACE_STEP_DIT_LONG_CLIP_QUALITY", raising=False)
    assert m.ace_step_dit_long_clip_quality_enabled(latent_frames=750, duration_sec=30.0, mesh_sku="BH_QB")
    assert not m.ace_step_dit_long_clip_quality_enabled(latent_frames=375, duration_sec=15.0, mesh_sku="BH_QB")
    assert not m.ace_step_dit_long_clip_quality_enabled(latent_frames=250, duration_sec=10.0, mesh_sku="BH_QB")
    assert not m.ace_step_dit_long_clip_quality_enabled(latent_frames=750, duration_sec=30.0, mesh_sku=None)

    monkeypatch.setenv("ACE_STEP_DIT_LONG_CLIP_QUALITY", "0")
    assert not m.ace_step_dit_long_clip_quality_enabled(latent_frames=750, duration_sec=30.0, mesh_sku="BH_QB")


def test_audio_code_limit_for_duration():
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
        ace_step_audio_code_limit_for_duration,
        ace_step_expected_audio_code_count,
    )

    assert ace_step_expected_audio_code_count(60.0) == 300
    assert ace_step_audio_code_limit_for_duration(60.0) == 350
    assert ace_step_audio_code_limit_for_duration(15.0) == 200


def test_configure_audio_code_limits_sets_env(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl import math_perf_env as m

    monkeypatch.delenv("ACE_STEP_MAX_AUDIO_CODES", raising=False)
    monkeypatch.delenv("ACE_STEP_DETOK_CHUNK_CODES", raising=False)
    assert m.ace_step_configure_audio_code_limits(60.0) == 350
    assert os.environ["ACE_STEP_MAX_AUDIO_CODES"] == "350"
    assert "ACE_STEP_DETOK_CHUNK_CODES" not in os.environ
    assert m.ace_step_detok_chunk_n() == m.ACE_STEP_DETOK_L1_CHUNK_CODES


def test_detok_chunk_n_default():
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
        ACE_STEP_DETOK_L1_CHUNK_CODES,
        ace_step_detok_chunk_n,
    )

    assert ace_step_detok_chunk_n() == ACE_STEP_DETOK_L1_CHUNK_CODES


def test_configure_audio_code_limits_respects_existing_env(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl import math_perf_env as m

    monkeypatch.setenv("ACE_STEP_MAX_AUDIO_CODES", "400")
    monkeypatch.setenv("ACE_STEP_DETOK_CHUNK_CODES", "400")
    assert m.ace_step_configure_audio_code_limits(60.0) == 400


def test_configure_dit_long_clip_quality_sets_env(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl import math_perf_env as m

    monkeypatch.delenv("ACE_STEP_DIT_LONG_CLIP_QUALITY", raising=False)
    monkeypatch.delenv("ACE_STEP_DIT_BFLOAT8_ATTN_QO", raising=False)
    monkeypatch.delenv("ACE_STEP_DIT_BFLOAT4_WEIGHTS", raising=False)
    assert m.ace_step_configure_dit_long_clip_quality(latent_frames=750, duration_sec=30.0, mesh_sku="BH_QB")
    assert m.ace_step_dit_long_clip_quality_active()
    assert os.environ.get("ACE_STEP_DIT_BFLOAT8_ATTN_QO") is None
    assert os.environ.get("ACE_STEP_DIT_BFLOAT4_WEIGHTS") is None


def test_vae_quality_clarity_30s_on_mesh():
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_vae_quality_decode_enabled

    assert ace_step_vae_quality_decode_enabled(
        latent_frames=750,
        mesh_sku="BH_QB",
        duration_sec=30.0,
        clarity_mode=True,
    )
    assert ace_step_vae_quality_decode_enabled(
        latent_frames=750,
        mesh_sku="BH_QB",
        duration_sec=30.0,
        clarity_mode=False,
    )
    assert not ace_step_vae_quality_decode_enabled(
        latent_frames=375,
        mesh_sku="BH_QB",
        duration_sec=15.0,
        clarity_mode=False,
    )
    assert ace_step_vae_quality_decode_enabled(
        latent_frames=750,
        mesh_sku="BH_QB",
        duration_sec=30.0,
        clarity_mode=False,
    )
    assert ace_step_vae_quality_decode_enabled(
        latent_frames=1000,
        mesh_sku="BH_QB",
        duration_sec=40.0,
        clarity_mode=False,
    )


def test_resolve_vae_tiling_mesh_long_clip():
    chunk, overlap = ace_step_resolve_vae_tiling(frames=375, mesh_sku="BH_QB", chunk_cli=32, overlap_cli=4)
    assert chunk == 32
    assert overlap >= 8
    chunk30, overlap30 = ace_step_resolve_vae_tiling(frames=750, mesh_sku="BH_QB", chunk_cli=32, overlap_cli=4)
    assert chunk30 == 32
    assert overlap30 >= 14
    chunk2, overlap2 = ace_step_resolve_vae_tiling(frames=1500, mesh_sku="BH_QB", chunk_cli=32, overlap_cli=4)
    assert chunk2 == 32
    assert overlap2 >= 15


def test_mesh_use_pytorch_dit_opt_in(monkeypatch):
    from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_mesh_use_pytorch_dit

    assert not ace_step_mesh_use_pytorch_dit(mesh_sku="BH_QB", duration_sec=60.0, latent_frames=1500)
    assert not ace_step_mesh_use_pytorch_dit(mesh_sku="BH_QB", duration_sec=30.0, latent_frames=750)
    assert not ace_step_mesh_use_pytorch_dit(mesh_sku="BH_QB", duration_sec=15.0, latent_frames=375)
    monkeypatch.setenv("ACE_STEP_PYTORCH_DIT", "1")
    assert ace_step_mesh_use_pytorch_dit(mesh_sku="BH_QB", duration_sec=60.0, latent_frames=1500)


def test_mesh_use_pytorch_condition_opt_in(monkeypatch):
    from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_mesh_use_pytorch_condition

    assert not ace_step_mesh_use_pytorch_condition(mesh_sku="BH_QB", duration_sec=60.0, latent_frames=1500)
    assert not ace_step_mesh_use_pytorch_condition(mesh_sku="BH_QB", duration_sec=30.0, latent_frames=750)
    assert not ace_step_mesh_use_pytorch_condition(mesh_sku="BH_QB", duration_sec=15.0, latent_frames=375)
    assert not ace_step_mesh_use_pytorch_condition(mesh_sku=None, duration_sec=60.0, latent_frames=1500)
    monkeypatch.setenv("ACE_STEP_PYTORCH_CONDITION", "1")
    assert ace_step_mesh_use_pytorch_condition(mesh_sku="BH_QB", duration_sec=60.0, latent_frames=1500)
    assert ace_step_mesh_use_pytorch_condition(mesh_sku="P150", duration_sec=15.0, latent_frames=375)


def test_mesh_audio_cover_strength_defaults(monkeypatch):
    from models.experimental.ace_step_v1_5.demo.run_prompt_to_wav import _mesh_effective_audio_cover_strength

    monkeypatch.delenv("ACE_STEP_AUDIO_COVER_STRENGTH", raising=False)
    assert _mesh_effective_audio_cover_strength(split_device=True, duration_sec=60.0, has_lm_codes=True) == 1.0
    assert _mesh_effective_audio_cover_strength(split_device=False, duration_sec=60.0, has_lm_codes=True) == 1.0
    assert _mesh_effective_audio_cover_strength(split_device=True, duration_sec=30.0, has_lm_codes=True) == 1.0
    # Short LM hints → keep full cover (no non-cover switch that drops guitar timbre).
    assert (
        _mesh_effective_audio_cover_strength(
            split_device=True,
            duration_sec=60.0,
            has_lm_codes=True,
            lm_hint_frames=250,
            total_frames=1500,
        )
        == 1.0
    )
    monkeypatch.setenv("ACE_STEP_AUDIO_COVER_STRENGTH", "0.5")
    assert _mesh_effective_audio_cover_strength(split_device=True, duration_sec=60.0, has_lm_codes=True) == 0.5


def test_repair_degenerate_lm_hint_tail():
    import torch

    from models.experimental.ace_step_v1_5.utils.official_lm_preprocess import (
        find_degenerate_code_prefix_len,
        repair_degenerate_lm_hint_tail,
    )

    head = list(range(80))
    tail = [head[-1]] * 220
    codes = head + tail
    prefix = find_degenerate_code_prefix_len(codes)
    assert prefix is not None and prefix >= 70

    hints = torch.randn(1, 1500, 64)
    sil = torch.zeros(1, 1500, 64)
    payload = {"precomputed_lm_hints_25Hz": hints.clone(), "silence_latent": sil}
    code_str = "".join(f"<|audio_code_{c}|>" for c in codes)
    repair_degenerate_lm_hint_tail(payload, code_str, 1500)
    assert int(payload["precomputed_lm_hints_25Hz"].shape[1]) == 1500
    good_frames = int(prefix * 5)
    assert torch.allclose(payload["precomputed_lm_hints_25Hz"][:, good_frames:, :], sil[:, : 1500 - good_frames, :])


def test_build_non_cover_condition_payload():
    import torch

    from models.experimental.ace_step_v1_5.utils.official_lm_preprocess import build_non_cover_condition_payload

    src = torch.randn(1, 100, 64)
    sil = torch.randn(1, 256, 64)
    nc_text = torch.randn(1, 32, 1024)
    payload = {
        "src_latents": src,
        "silence_latent": sil,
        "is_covers": torch.ones(1, dtype=torch.long),
        "non_cover_text_hidden_states": nc_text,
        "non_cover_text_attention_masks": torch.ones(1, 32),
        "precomputed_lm_hints_25Hz": torch.randn(1, 100, 64),
    }
    nc = build_non_cover_condition_payload(payload)
    assert nc is not None
    assert nc["precomputed_lm_hints_25Hz"] is None
    assert int(nc["is_covers"].sum()) == 0
    assert nc["text_hidden_states"] is nc_text


def test_configure_dit_ultra_long_clip_quality(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl import math_perf_env as m

    monkeypatch.delenv("ACE_STEP_DIT_ULTRA_LONG_CLIP_QUALITY", raising=False)
    monkeypatch.delenv("ACE_STEP_DIT_LONG_CLIP_QUALITY", raising=False)
    monkeypatch.delenv("ACE_STEP_COND_LONG_CLIP_QUALITY", raising=False)
    assert m.ace_step_configure_dit_ultra_long_clip_quality(latent_frames=1500, duration_sec=60.0, mesh_sku="BH_QB")
    assert m.ace_step_dit_ultra_long_clip_quality_active()
    assert m.ace_step_dit_long_clip_quality_active()
    assert m.ace_step_cond_long_clip_quality_active()
    monkeypatch.delenv("ACE_STEP_DIT_ULTRA_LONG_CLIP_QUALITY", raising=False)
    assert not m.ace_step_dit_ultra_long_clip_quality_enabled(latent_frames=750, duration_sec=30.0, mesh_sku="BH_QB")


def test_configure_cond_long_clip_quality(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl import math_perf_env as m

    monkeypatch.delenv("ACE_STEP_COND_LONG_CLIP_QUALITY", raising=False)
    assert m.ace_step_configure_cond_long_clip_quality(latent_frames=750, duration_sec=30.0, mesh_sku="BH_QB")
    assert m.ace_step_cond_long_clip_quality_active()
    monkeypatch.delenv("ACE_STEP_COND_LONG_CLIP_QUALITY", raising=False)
    assert not m.ace_step_cond_long_clip_quality_enabled(latent_frames=375, duration_sec=15.0, mesh_sku="BH_QB")


def test_mesh_perf_log_default():
    assert ace_step_mesh_perf_log_default(mesh_sku="BH_QB")
    assert ace_step_mesh_perf_log_default(mesh_sku="P150")
    assert ace_step_mesh_perf_log_default(mesh_sku=None)


def test_mesh_perf_log_opt_out(monkeypatch):
    from models.experimental.ace_step_v1_5.utils.ace_step_perf_log import ace_step_perf_logging_enabled

    monkeypatch.setenv("ACE_STEP_DEMO_PERF_LOG", "0")
    assert not ace_step_mesh_perf_log_default(mesh_sku="P150")
    assert not ace_step_perf_logging_enabled()


def test_cached_preprocess_reuse():
    from models.experimental.ace_step_v1_5.demo.demo_session import AceStepDemoSession

    session = AceStepDemoSession()
    session.store_preprocess(
        prompt="main prompt",
        duration_sec=15.0,
        seed=0,
        lyrics="[Instrumental]",
        instrumental=True,
        frames=375,
        enc_hs=object(),
        enc_mask=object(),
        ctx_lat=object(),
        null_emb=object(),
    )
    assert session.can_reuse_preprocess(
        prompt="main prompt",
        duration_sec=15.0,
        seed=0,
        lyrics="[Instrumental]",
        instrumental=True,
    )
    assert not session.can_reuse_preprocess(
        prompt="other",
        duration_sec=15.0,
        seed=0,
        lyrics="[Instrumental]",
        instrumental=True,
    )
    assert not session.can_reuse_preprocess(
        prompt="main prompt",
        duration_sec=15.0,
        seed=0,
        lyrics="verse one",
        instrumental=False,
    )


def test_emit_session_summary_rollup(monkeypatch):
    monkeypatch.setenv("ACE_STEP_DEMO_PERF_LOG", "1")
    from models.experimental.ace_step_v1_5.utils.ace_step_perf_log import (
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

    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_tile_physical_m_dim

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


def test_five_hz_lm_bfloat8_weights_default_off():
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_five_hz_lm_bfloat8_weights_enabled

    assert not ace_step_five_hz_lm_bfloat8_weights_enabled()


def test_five_hz_lm_bfloat8_optimizations_when_opt_in(monkeypatch):
    import ttnn
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
        ace_step_five_hz_lm_bfloat8_weights_enabled,
        ace_step_five_hz_lm_optimizations,
    )
    from models.tt_transformers.tt.model_config import TensorGroup

    class _FakeModelArgs:
        n_layers = 28
        model_name = "acestep-5Hz-lm-1.7B"

    monkeypatch.setenv("ACE_STEP_LM_BFLOAT8_WEIGHTS", "1")
    assert ace_step_five_hz_lm_bfloat8_weights_enabled()
    dec = ace_step_five_hz_lm_optimizations(_FakeModelArgs())
    bf8 = getattr(ttnn, "bfloat8_b", None)
    assert bf8 is not None
    for layer in range(_FakeModelArgs.n_layers):
        for tg in (TensorGroup.FF1_FF3, TensorGroup.FF2, TensorGroup.WQKV, TensorGroup.WO, TensorGroup.KV_CACHE):
            assert dec.get_tensor_dtype(layer, tg) == bf8


def test_five_hz_lm_bfloat8_weights_env_opt_out(monkeypatch):
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_five_hz_lm_bfloat8_weights_enabled

    monkeypatch.setenv("ACE_STEP_LM_BFLOAT8_WEIGHTS", "0")
    assert not ace_step_five_hz_lm_bfloat8_weights_enabled()


def test_five_hz_lm_accuracy_optimizations_from_ace_model_params():
    import ttnn
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
        ace_step_five_hz_lm_accuracy_decoder_config_path,
        ace_step_five_hz_lm_accuracy_optimizations,
    )
    from models.tt_transformers.tt.model_config import MathFidelitySetting, OpGroup, TensorGroup

    class _FakeModelArgs:
        n_layers = 28
        model_name = "acestep-5Hz-lm-1.7B"

    path = ace_step_five_hz_lm_accuracy_decoder_config_path(_FakeModelArgs.model_name)
    assert path is not None
    assert path.name == "accuracy_decoder_config.json"
    assert "experimental/ace_step_v1_5/model_params" in str(path)

    dec = ace_step_five_hz_lm_accuracy_optimizations(_FakeModelArgs())
    bf16 = ttnn.bfloat16
    for layer in range(_FakeModelArgs.n_layers):
        for tg in (TensorGroup.FF1_FF3, TensorGroup.FF2, TensorGroup.WQKV, TensorGroup.WO, TensorGroup.KV_CACHE):
            assert dec.get_tensor_dtype(layer, tg) == bf16
        for op in (OpGroup.LI_FF1_FF3, OpGroup.LI_FF2, OpGroup.LI_QKV_PREFILL, OpGroup.LI_O_PREFILL):
            assert dec.decoder_optimizations[layer].op_fidelity_settings[op] == MathFidelitySetting.HIFI4


def test_five_hz_lm_hifi2_optimizations_default():
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_five_hz_lm_optimizations
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
            OpGroup.SDPA_DECODE,
            OpGroup.SDPA_PREFILL,
            OpGroup.LI_O_DECODE,
            OpGroup.LI_O_PREFILL,
        ):
            fid = dec.decoder_optimizations[layer].op_fidelity_settings[op]
            assert fid == MathFidelitySetting.HIFI2, f"layer={layer} op={op} got {fid}"
        assert (
            dec.decoder_optimizations[layer].op_fidelity_settings[OpGroup.LI_QKV_PREFILL] == MathFidelitySetting.HIFI4
        )
