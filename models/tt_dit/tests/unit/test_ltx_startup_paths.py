# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.tt_dit.layers import audio_ops
from models.tt_dit.models.audio_vae import audio_decoder_ltx
from models.tt_dit.models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx import pipeline_ltx_distilled
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline


def _patch_depthwise_setup(monkeypatch):
    mesh = SimpleNamespace(arch=lambda: None)
    monkeypatch.setattr(ttnn, "from_torch", lambda *args, **kwargs: "weight")
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *args, **kwargs: "mesh-mapper")
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *args, **kwargs: "compute")
    monkeypatch.setattr(ttnn, "Conv1dConfig", lambda **kwargs: "conv")
    return mesh


def test_depthwise_recovery_disabled_uses_only_direct(monkeypatch):
    mesh = _patch_depthwise_setup(monkeypatch)
    calls = []

    def direct(*args, **kwargs):
        calls.append("direct")
        return "direct-output"

    monkeypatch.setattr(audio_ops, "_depthwise_tap_conv1d", direct)
    monkeypatch.setattr(
        audio_ops,
        "_depthwise_tap_conv1d_chunked",
        lambda *args, **kwargs: pytest.fail("LTX direct mode must not try channel chunks"),
    )
    monkeypatch.setattr(
        audio_ops,
        "_depthwise_tap_mac",
        lambda *args, **kwargs: pytest.fail("LTX direct mode must not try the MAC fallback"),
    )

    cache = {}
    output = audio_ops.depthwise_tap_filter(
        SimpleNamespace(shape=(1, 64, 256)),
        [1.0] * 7,
        1,
        mesh_device=mesh,
        dtype=ttnn.float32,
        cache=cache,
        allow_recovery=False,
    )

    assert output == "direct-output"
    assert calls == ["direct"]
    assert not any(isinstance(key, tuple) and key and key[0] == "tap_path" for key in cache)


def test_depthwise_recovery_remains_enabled_by_default(monkeypatch):
    mesh = _patch_depthwise_setup(monkeypatch)
    calls = []

    def direct(*args, **kwargs):
        calls.append("direct")
        raise RuntimeError("full-channel shape does not fit")

    def chunked(*args, **kwargs):
        calls.append(("chunk", kwargs["chunk"]))
        return "chunk-output"

    monkeypatch.setattr(audio_ops, "_depthwise_tap_conv1d", direct)
    monkeypatch.setattr(audio_ops, "_depthwise_tap_conv1d_chunked", chunked)
    monkeypatch.setattr(
        audio_ops,
        "_depthwise_tap_mac",
        lambda *args, **kwargs: pytest.fail("the chunk recovery should succeed before MAC"),
    )

    cache = {}
    output = audio_ops.depthwise_tap_filter(
        SimpleNamespace(shape=(1, 166, 256)),
        [1.0] * 7,
        1,
        mesh_device=mesh,
        dtype=ttnn.float32,
        cache=cache,
    )

    assert output == "chunk-output"
    assert calls == ["direct", ("chunk", 128)]
    assert cache[("tap_path", 256, 1, 7)] == 128


def test_ltx_tail_path_avoids_inplace_update_and_multishard_gather(monkeypatch):
    x = SimpleNamespace(shape=(1, 32, 4), get_dtype=lambda: ttnn.float32)
    mask, inverse, masked, last, fill, output = (object() for _ in range(6))

    monkeypatch.setattr(
        audio_ops,
        "_set_tpad_tail_local",
        lambda *args, **kwargs: pytest.fail("LTX traces must not capture the in-place suffix update"),
    )
    monkeypatch.setattr(
        audio_ops,
        "_all_gather_t",
        lambda *args, **kwargs: pytest.fail("LTX traces must not capture the multi-shard tail gather"),
    )
    monkeypatch.setattr(audio_ops, "_tpad_mask", lambda *args, **kwargs: (mask, inverse))
    monkeypatch.setattr(ttnn, "slice", lambda *args, **kwargs: last)
    monkeypatch.setattr(ttnn, "deallocate", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn, "multiply", lambda lhs, rhs: masked if rhs is mask else fill)
    monkeypatch.setattr(ttnn, "add", lambda *args, **kwargs: output)

    actual = audio_ops._set_tpad_tail(
        x,
        136,
        mode="replicate",
        mesh_device=object(),
        parallel_config=SimpleNamespace(factor=8),
        cache={},
        use_local_tail=False,
        legacy_replicate_tail=True,
    )

    assert actual is output


def test_ltx_vocoder_disables_depthwise_recovery(monkeypatch):
    params = inspect.signature(audio_decoder_ltx.Vocoder).parameters
    assert params["use_local_tpad_tail"].default is True
    assert params["legacy_replicate_tail"].default is False
    captured = {}

    def fake_vocoder(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(audio_decoder_ltx, "Vocoder", fake_vocoder)
    adapter = object.__new__(audio_decoder_ltx.LTXAudioDecoderAdapter)
    adapter._mesh_device = object()

    adapter._build_vocoder(
        {},
        apply_final_activation=True,
        parallel_config=None,
        ccl_manager=None,
    )

    assert captured["allow_depthwise_recovery"] is False
    assert captured["use_local_tpad_tail"] is False
    assert captured["legacy_replicate_tail"] is True
    assert captured["use_persistent_neighbor_pad"] is True


@pytest.mark.parametrize(
    ("has_encoder", "image_conditioning", "expected"),
    [(False, False, False), (True, False, False), (True, True, True)],
)
def test_image_encoder_warmup_only_for_i2v(has_encoder, image_conditioning, expected):
    pipeline = object.__new__(LTXDistilledPipeline)
    pipeline.vae = SimpleNamespace(encoder=object()) if has_encoder else None
    pipeline._image_conditioning = image_conditioning

    assert pipeline._needs_image_encoder_warmup() is expected


def test_component_trace_gate_prefers_explicit_value_over_legacy(monkeypatch):
    monkeypatch.setenv("LTX_VAE_TRACE", "1")
    monkeypatch.delenv("LTX_VIDEO_VAE_TRACE", raising=False)
    assert LTXDistilledPipeline._component_trace_enabled("LTX_VIDEO_VAE_TRACE", legacy="LTX_VAE_TRACE")

    monkeypatch.setenv("LTX_VIDEO_VAE_TRACE", "0")
    assert not LTXDistilledPipeline._component_trace_enabled("LTX_VIDEO_VAE_TRACE", legacy="LTX_VAE_TRACE")


def test_video_rope_materializer_writes_existing_rung_buffers(monkeypatch):
    pipeline = object.__new__(LTXDistilledPipeline)
    pipeline._postprocess_shapes_locked = True
    pipeline._warmed_rope_materializer_rungs = frozenset({40960})
    pipeline.inner_dim = 4096
    pipeline.positional_embedding_theta = 10000.0
    pipeline.positional_embedding_max_pos = [20, 2048, 2048]
    pipeline.parallel_config = SimpleNamespace(
        sequence_parallel=SimpleNamespace(mesh_axis=1),
        tensor_parallel=SimpleNamespace(mesh_axis=0),
    )
    state = SimpleNamespace(
        tt_video_rope_cos="self-cos-out",
        tt_video_rope_sin="self-sin-out",
        tt_video_cross_pe_cos="cross-cos-out",
        tt_video_cross_pe_sin="cross-sin-out",
    )
    compact = SimpleNamespace(
        self_cos=torch.zeros(3, 64, 682),
        self_sin=torch.zeros(3, 64, 682),
        cross_cos=torch.zeros(64, 1024),
        cross_sin=torch.zeros(64, 1024),
    )
    monkeypatch.setattr(pipeline_ltx_distilled, "prepare_compact_video_rope", lambda *args, **kwargs: compact)
    uploads = {}

    def refresh(name, value, **kwargs):
        uploads[name] = value
        return f"{name}-input"

    pipeline._refresh_rope_compact_input = refresh
    calls = []
    monkeypatch.setattr(
        ttnn.experimental, "ltx_rope_materialize", lambda *args, **kwargs: calls.append((args, kwargs)), raising=False
    )

    pipeline._materialize_video_rope(
        state,
        latent_frames=20,
        latent_h=34,
        latent_w=60,
        video_N=40960,
        video_N_real=40800,
        fps=25,
    )

    assert uploads["self_cos"].shape == (1, 3, 64, 682)
    assert uploads["cross_cos"].shape == (1, 1, 64, 1024)
    assert uploads["metadata"].flatten().tolist() == [40800, 20, 34, 60]
    args, kwargs = calls[0]
    assert args[-4:] == (
        "self-cos-out",
        "self-sin-out",
        "cross-cos-out",
        "cross-sin-out",
    )
    assert kwargs == {"sp_axis": 1, "tp_axis": 0}

    pipeline._warmed_rope_materializer_rungs = frozenset()
    with pytest.raises(ValueError, match="not compiled before traces became live"):
        pipeline._materialize_video_rope(
            state,
            latent_frames=20,
            latent_h=34,
            latent_w=60,
            video_N=40960,
            video_N_real=40800,
            fps=25,
        )


def test_traced_upsampler_reuses_prebuilt_shape_and_rejects_new_shape(monkeypatch):
    events = []

    class FakeUpsampler:
        def __init__(self, name):
            self.name = name

        def is_loaded(self):
            return True

    pipeline = object.__new__(LTXDistilledPipeline)
    current = FakeUpsampler("current")
    cached = FakeUpsampler("cached")
    pipeline.upsampler = current
    pipeline._upsampler_shape = (20, 17, 30)
    pipeline._upsampler_cache = {(20, 17, 30): current, (19, 11, 20): cached}
    pipeline._warmed_upsampler_shapes = frozenset({(20, 17, 30), (19, 11, 20)})
    pipeline._warmed_upsampler_trace_shapes = pipeline._warmed_upsampler_shapes
    pipeline._traced = True
    pipeline._postprocess_shapes_locked = True
    pipeline.vae_parallel_config = SimpleNamespace(
        height_parallel=SimpleNamespace(factor=1),
        width_parallel=SimpleNamespace(factor=1),
    )
    pipeline.mesh_device = object()
    pipeline._sync_and_reset_vae_ccl = lambda: events.append("reset")
    pipeline._register_coresident_exclusions = lambda: None
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append("sync"))
    monkeypatch.setenv("LTX_UPSAMPLER_TRACE", "1")

    assert pipeline._ensure_upsampler_shape(145, 704, 1280)
    assert pipeline.upsampler is cached
    assert events == []

    with pytest.raises(ValueError, match="not initialized before trace capture"):
        pipeline._ensure_upsampler_shape(153, 704, 1280)


def test_upsampler_weight_layout_key_is_frozen_at_construction():
    upsampler = object.__new__(LTXLatentUpsampler)
    upsampler._prepared_weight_layout_key = "construction-layout"

    assert upsampler.weight_layout_key() == "construction-layout"


def test_upsampler_keeps_three_conv_weight_layouts_resident():
    events = []

    class FakeUpsampler:
        def __init__(self, name, layout):
            self.name = name
            self.layout = layout

        def weight_layout_key(self):
            return self.layout

        def reload_weights(self, *, conv_weight_bank=None):
            events.append((self.name, None if conv_weight_bank is None else conv_weight_bank.name))

    pipeline = object.__new__(LTXDistilledPipeline)
    first = FakeUpsampler("first", "layout-a")
    first_alias = FakeUpsampler("first-alias", "layout-a")
    second = FakeUpsampler("second", "layout-b")
    third = FakeUpsampler("third", "layout-c")
    fourth = FakeUpsampler("fourth", "layout-d")

    pipeline.upsampler = first
    pipeline._prepare_upsampler()
    pipeline.upsampler = first_alias
    pipeline._prepare_upsampler()
    pipeline.upsampler = second
    pipeline._prepare_upsampler()
    pipeline.upsampler = third
    pipeline._prepare_upsampler()

    assert events == [("first", None), ("first-alias", "first"), ("second", None), ("third", None)]
    assert pipeline._upsampler_weight_banks == {"layout-a": first, "layout-b": second, "layout-c": third}

    pipeline.upsampler = fourth
    with pytest.raises(RuntimeError, match="more than 3 resident Conv3D layouts"):
        pipeline._prepare_upsampler()


def test_reset_global_semaphores_restarts_host_ping_pong_selectors(monkeypatch):
    manager = object.__new__(CCLManager)
    manager.np_ping_pong_semaphores = {0: ["np0"], 1: ["np1"]}
    manager.sr_ping_pong_semaphores = {0: ["sr0"], 1: ["sr1"]}
    manager.rs_ping_pong_semaphores = {0: ["rs0"], 1: ["rs1"]}
    manager.rs_ping_pong_semaphores_fused = {0: ["rsf0"], 1: ["rsf1"]}
    manager.ag_ping_pong_semaphores = {0: ["ag0"], 1: ["ag1"]}
    manager.exp_ring_ping_pong_semaphores = {0: ["exp0"], 1: ["exp1"]}
    manager.barrier_semaphores = {0: ["bar0"], 1: ["bar1"]}
    manager.np_fused_ping_pong_semaphores = None
    manager.barrier_fused_semaphores = None
    manager.np_region_progress_semaphores = None
    for name in (
        "rs_ping_pong_idx",
        "rs_ping_pong_idx_fused",
        "ag_ping_pong_idx",
        "exp_ring_ping_pong_idx",
        "np_ping_pong_idx",
        "np_fused_ping_pong_idx",
        "sr_ping_pong_idx",
        "barrier_idx",
        "barrier_fused_idx",
    ):
        setattr(manager, name, [1, 1])

    reset = []
    monkeypatch.setattr(ttnn, "reset_global_semaphore_value", lambda semaphore, value: reset.append(semaphore))
    manager.reset_global_semaphores()

    assert {"np0", "rsf1", "ag0", "exp1", "bar1"} <= set(reset)
    assert manager.ag_ping_pong_idx == [0, 0]
    assert manager.barrier_idx == [0, 0]
