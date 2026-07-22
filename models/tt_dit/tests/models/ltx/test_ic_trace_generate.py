# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the IC-LoRA s1_ref/s2_ref trace family + generate wiring (Item 4).

Item 4 threads reference conditioning through the three methods that own the trace lifecycle:

  * ``warmup_buffers`` — accepts ``ref_num_frames``; ``valid`` stages grow to include
    ``s1_ref``/``s2_ref``; ``_prepare_transformer(0)`` runs at the TOP (so a base-then-ref
    two-pass warmup has the DiT resident before the ref denoise); the ``_prealloc_trace_io``
    calls are stage-conditional; and the s1_ref/s2_ref denoise blocks are CAPTURED HERE in
    warmup (``traced=capture_traced``, ``trace_key="s*_ref"``) exactly like s1/s2 — NOT deferred
    to gen#0 (deferred capture is the known blank-video bug).
  * ``_prealloc_trace_io`` — ``ref_num_frames=0`` is an exact no-op; ``>0`` grows the trace-baked
    buffers to the COMBINED (target+reference) length and feeds the combined length as the V→A
    pad-mask real region (``video_N_grid``), matching the ``_denoise_no_guidance`` decouple.
  * ``generate`` — ``reference_video=(path, strength)`` builds the ref encoders, encodes the
    looped still per stage (R from ``ref_frames``), routes ``trace_key`` to ``s*_ref``, keeps
    ``upsampled_flat`` GRID-sized (``_denoise`` allocates ``base_v`` combined and the ref pin fills
    ``[T,T+R)``), and ASSERTS reference + append-token keyframe never combine.

Everything is gated on ``has_ref`` / ``reference_video`` / ``ref_num_frames`` so base / i2v /
keyframe stay byte-identical (the no-ref cases below assert this).

``pipeline_ltx_distilled`` imports the full tt_dit device stack (``ttnn.experimental`` etc.),
which needs a matching device build, so — like the sibling IC host tests — the real method
sources are AST-extracted and exec'd into an isolated shim whose device dependencies are
stubbed/mocked. The three methods under test run VERBATIM from source; their collaborators
(the transformer denoise, stage-statics builder, upsample/VAE/audio decode, encoders, device
round-trip) are mocked, so what is asserted is the real routing / sizing / capture math.

Run: PYTHONPATH=<worktree> python -m pytest <file> -v -p no:cacheprovider
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

TILE_SIZE = 32
DISTILLED_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"

_TEMPORAL_COMPRESSION = 8
_SPATIAL_COMPRESSION = 32


def _latent_grid(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    return (num_frames - 1) // _TEMPORAL_COMPRESSION + 1, height // _SPATIAL_COMPRESSION, width // _SPATIAL_COMPRESSION


def _sp_pad(n_real: int, sp_factor: int = 1) -> int:
    divisor = TILE_SIZE * sp_factor
    return ((n_real + divisor - 1) // divisor) * divisor


# ------------------------------------------------------------------------------------------
# Isolated shim: the three real methods exec'd with stubbed device globals.
# ------------------------------------------------------------------------------------------
def _build_shim() -> tuple[type, dict]:
    """Exec the real ``warmup_buffers`` + ``_prealloc_trace_io`` + ``generate`` (and the pure
    ``pixel_to_latent_frame`` helper) into a namespace whose device dependencies are stubbed.
    ``from __future__ import annotations`` keeps every ``ttnn.Tensor`` annotation a lazy string,
    so no annotation is ever evaluated. Returns ``(Pipe, ns)`` — ``ns`` is the live globals dict
    (mutate ``ns["upsample_latent"]`` etc. to control module-level collaborators)."""
    src = DISTILLED_PATH.read_text()
    src_lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    def _top(name: str) -> ast.AST:
        return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)

    cls = _top("LTXDistilledPipeline")

    def _method(name: str) -> ast.FunctionDef:
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def _lines(node: ast.AST) -> str:
        return "".join(src_lines[node.lineno - 1 : node.end_lineno])

    p2l = _lines(_top("pixel_to_latent_frame"))
    methods = ("warmup_buffers", "_prealloc_trace_io", "generate")
    method_blocks = [_lines(_method(n)) for n in methods]

    shim_src = (
        "from __future__ import annotations\n\n" + p2l + "\n\nclass LTXDistilledPipeline:\n" + "\n".join(method_blocks)
    )

    ns: dict = {
        "torch": torch,
        "os": __import__("os"),
        "time": __import__("time"),
        "logger": mock.MagicMock(name="logger"),
        "walltime": mock.MagicMock(name="walltime"),  # .timed(...) is a CM; .record(...) a no-op
        "latent_grid": _latent_grid,
        "TEMPORAL_COMPRESSION": _TEMPORAL_COMPRESSION,
        "SPATIAL_COMPRESSION": _SPATIAL_COMPRESSION,
        "LTX_DIT_PREP_RUN": False,
        # Short 2-node schedules; generate/warmup only use these for len() and pass-through.
        "DISTILLED_SIGMA_VALUES": [1.0, 0.0],
        "STAGE_2_DISTILLED_SIGMA_VALUES": [1.0, 0.0],
        "_DEFAULT_S1_SIGMAS": [1.0, 0.99, 0.0],
        "_keyframe_s1_sigmas": lambda lf: [1.0, 0.0],
        "_keyframe_s1_strength": lambda h: 1.0,
        "VideoPixelShape": lambda **kw: SimpleNamespace(**kw),
        "AudioLatentShape": SimpleNamespace(from_video_pixel_shape=staticmethod(lambda vps: SimpleNamespace(frames=8))),
        "LTXTransformerState": lambda: mock.MagicMock(name="state"),
        "upsample_latent": mock.MagicMock(name="upsample_latent"),
        "export_video_audio": mock.MagicMock(name="export_video_audio"),
    }
    exec(compile(shim_src, str(DISTILLED_PATH), "exec"), ns)
    return ns["LTXDistilledPipeline"], ns


# ==========================================================================================
# warmup_buffers
# ==========================================================================================
def _make_warmup_pipe():
    """A shim instance with every warmup collaborator mocked. Returns (obj, mocks)."""
    Pipe, _ns = _build_shim()
    obj = Pipe()
    obj._traced = True
    obj.dynamic_load = False
    obj.in_channels = 128
    obj.vae_encoder = mock.MagicMock(name="vae_encoder")
    obj.parallel_config = SimpleNamespace(sequence_parallel=SimpleNamespace(factor=1, mesh_axis=2))
    obj.gemma_encoder_pair = SimpleNamespace(
        sequence_length=4, video_dim=8, audio_dim=8, ensure_loaded=mock.MagicMock()
    )
    obj.encode_prompts = mock.MagicMock(name="encode_prompts")
    obj._kf_trace_anchors = mock.MagicMock(name="_kf_trace_anchors", return_value=[])

    for name in (
        "_prepare_transformer",
        "_prealloc_trace_io",
        "_denoise_no_guidance",
        "_warmup_upsample",
        "_warmup_decode",
        "_warmup_audio_decode",
        "_warmup_encode",
        "_warmup_ref_encode",
        "_ensure_upsampler_frames",
        "_ensure_vae_decoder_frames",
        "decode_audio",
    ):
        setattr(obj, name, mock.MagicMock(name=name))
    return obj


def _denoise_calls_by_key(obj):
    return {c.kwargs.get("trace_key"): c for c in obj._denoise_no_guidance.call_args_list}


def test_warmup_accepts_ref_num_frames_and_ref_stages(monkeypatch):
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    monkeypatch.setenv("LTX_WARMUP_ENCODERS", "1")
    obj = _make_warmup_pipe()
    # A dedicated -ref worker: warm ONLY the reference family.
    obj.warmup_buffers(
        num_frames=17, height=64, width=64, stages=("s1_ref", "s2_ref"), ref_num_frames=17, capture_traced=True
    )

    # _prepare_transformer(0) runs at the TOP (before any denoise) so the DiT is resident.
    assert obj._prepare_transformer.call_args_list[0].args == (0,)

    # prealloc is stage-conditional: only the ref stages, each carrying ref_num_frames.
    pre = {c.args[0]: c.kwargs for c in obj._prealloc_trace_io.call_args_list}
    assert set(pre) == {"s1_ref", "s2_ref"}, "ref-only worker must prealloc only the ref stages"
    assert pre["s1_ref"]["ref_num_frames"] == 17
    assert pre["s2_ref"]["ref_num_frames"] == 17

    # Both ref denoises CAPTURED in warmup (traced=capture_traced), NOT deferred to gen#0.
    calls = _denoise_calls_by_key(obj)
    assert {"s1_ref", "s2_ref"}.issubset(calls), "warmup must run the s1_ref/s2_ref denoise blocks"
    for key in ("s1_ref", "s2_ref"):
        assert calls[key].kwargs["traced"] is True, f"{key} must be captured in warmup (traced=capture_traced)"
        assert calls[key].kwargs["ref_latent"] is not None, f"{key} must feed a (dummy) ref_latent"
    assert "s1" not in calls and "s2" not in calls, "ref-only worker must not run base denoise"

    # The ref encoders are warmed last (post-DiT ordering).
    obj._warmup_ref_encode.assert_called_once_with(17, 64, 64)


def test_warmup_ref_denoise_uses_grid_initial_latent(monkeypatch):
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    obj = _make_warmup_pipe()
    obj.warmup_buffers(
        num_frames=17, height=64, width=64, stages=("s1_ref", "s2_ref"), ref_num_frames=17, capture_traced=True
    )
    calls = _denoise_calls_by_key(obj)

    # latent_frames=3, hw_full=(64/32)^2=4 => GRID-sized stage-2 init = 3*4 = 12 (target only). _denoise
    # allocates base_v at the combined length and the ref pin fills [T,T+R); the init stays grid-sized.
    s2 = calls["s2_ref"]
    init = s2.kwargs["initial_video_latent"]
    assert init.shape[1] == 3 * 4, "s2_ref warmup initial latent must be GRID-sized (target only)"
    # s2_ref ref_latent is a full-res dummy with R latent frames on the temporal axis.
    assert s2.kwargs["ref_latent"].shape[2] == 3


def test_warmup_ref_stage_without_ref_num_frames_asserts(monkeypatch, expect_error):
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj = _make_warmup_pipe()
    with expect_error(AssertionError, r"ref trace stages .* require ref_num_frames"):
        obj.warmup_buffers(num_frames=17, height=64, width=64, stages=("s1_ref",))  # ref_num_frames omitted


def test_warmup_rejects_unknown_stage(monkeypatch, expect_error):
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj = _make_warmup_pipe()
    with expect_error(AssertionError, r"stages must be subset of"):
        obj.warmup_buffers(num_frames=17, height=64, width=64, stages=("s3",))


def test_warmup_base_path_unchanged_no_ref(monkeypatch):
    """Base worker (no ref): valid stages accepted, prealloc for s1/s2 only, and NO ref work — the
    ref family stays byte-identical-absent so base/i2v/keyframe traces are untouched."""
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    obj = _make_warmup_pipe()
    obj.warmup_buffers(num_frames=17, height=64, width=64, stages=("s1", "s2"), capture_traced=True)

    pre = {c.args[0]: c.kwargs for c in obj._prealloc_trace_io.call_args_list}
    assert set(pre) == {"s1", "s2"}
    assert "ref_num_frames" not in pre["s1"] and "ref_num_frames" not in pre["s2"]

    calls = _denoise_calls_by_key(obj)
    assert set(calls) == {"s1", "s2"}, "base worker runs only the s1/s2 denoise"
    for key in ("s1", "s2"):
        assert calls[key].kwargs.get("ref_latent") is None
    obj._warmup_ref_encode.assert_not_called()


# ==========================================================================================
# _prealloc_trace_io
# ==========================================================================================
def _make_prealloc_pipe(sp_factor=1):
    Pipe, _ns = _build_shim()
    obj = Pipe()
    obj.in_channels = 128
    obj.mesh_device = mock.MagicMock(name="mesh_device")
    obj.parallel_config = SimpleNamespace(sequence_parallel=SimpleNamespace(factor=sp_factor, mesh_axis=2))
    obj._trace_state = {}
    obj._kf_trace_anchors = mock.MagicMock(return_value=[])
    obj._sp_pad_len = lambda n: _sp_pad(n, sp_factor)
    captured: list[dict] = []
    obj._prepare_stage_statics = mock.MagicMock(side_effect=lambda state, **kw: captured.append(kw))
    return obj, captured


# Geometry: num_frames 17 -> latent_frames 3; 64/32 = 2 -> hw = 4; grid = 3*4 = 12.
GRID = 3 * (2 * 2)


def test_prealloc_ref_zero_is_noop():
    obj, cap = _make_prealloc_pipe()
    obj._prealloc_trace_io("s1", num_frames=17, height=64, width=64)  # ref_num_frames defaults to 0
    kw = cap[-1]
    assert kw["video_N_real"] == GRID, "ref=0 self-attn length must be the target grid"
    assert kw["video_N_grid"] == GRID, "ref=0 V->A pad-mask real region must be the target grid"
    assert kw["video_N"] == _sp_pad(GRID)
    assert kw["ref_latent_frames"] == 0
    assert kw["anchor_frames"] == []


def test_prealloc_ref_zero_explicit_matches_default():
    obj_a, cap_a = _make_prealloc_pipe()
    obj_a._prealloc_trace_io("s2", num_frames=17, height=64, width=64)
    obj_b, cap_b = _make_prealloc_pipe()
    obj_b._prealloc_trace_io("s2", num_frames=17, height=64, width=64, ref_num_frames=0)
    assert cap_a[-1] == cap_b[-1], "ref_num_frames=0 must be byte-identical to omitting it"


@pytest.mark.parametrize("ref_nf,expected_ref_lf", [(9, 2), (17, 3), (33, 5)])
def test_prealloc_ref_grows_to_combined(ref_nf, expected_ref_lf):
    obj, cap = _make_prealloc_pipe()
    obj._prealloc_trace_io("s1_ref", num_frames=17, height=64, width=64, ref_num_frames=ref_nf)
    kw = cap[-1]
    combined = GRID + expected_ref_lf * 4
    assert kw["ref_latent_frames"] == expected_ref_lf
    assert kw["video_N_real"] == combined, "ref>0 self-attn length must be the COMBINED count"
    assert kw["video_N_grid"] == combined, "ref>0 V->A pad-mask real region must be the COMBINED count"
    assert kw["video_N"] == _sp_pad(combined)
    assert kw["anchor_frames"] == [], "reference is not an append-token anchor"


def test_prealloc_ref_sp_padding():
    obj, cap = _make_prealloc_pipe(sp_factor=4)
    obj._prealloc_trace_io("s2_ref", num_frames=17, height=64, width=64, ref_num_frames=17)
    kw = cap[-1]
    combined = GRID + 3 * 4  # 24
    assert kw["video_N_real"] == combined
    assert kw["video_N"] == _sp_pad(combined, 4)


# ==========================================================================================
# generate
# ==========================================================================================
def _real_sheet_digest(path):
    """The production content digest. Kept identical to LTXPipeline._sheet_digest so the memoize is
    exercised the way it runs, not through a stand-in that could key differently."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sheet(tmp_path):
    """A real file on disk: the reference memoize keys on sheet CONTENT, so it reads the bytes."""
    p = tmp_path / "ref.png"
    p.write_bytes(b"SHEET-BYTES")
    return str(p)


def _make_generate_pipe():
    """A shim instance with every generate collaborator mocked. Returns (obj, ns)."""
    Pipe, ns = _build_shim()
    obj = Pipe()
    obj._traced = True
    obj.dynamic_load = False
    obj.in_channels = 128
    obj.vae_encoder = mock.MagicMock(name="vae_encoder")
    obj.vae_ref_encoder_s1 = mock.MagicMock(name="ref_enc_s1")
    obj.vae_ref_encoder_full = mock.MagicMock(name="ref_enc_full")
    obj.upsampler = mock.MagicMock(name="upsampler")
    obj._i2v_cond_cache = {}
    obj._ref_latent_cache = {}
    obj._sheet_digest = staticmethod(_real_sheet_digest)

    obj._device_embed_cache_path = mock.MagicMock(return_value="/nonexistent/embed/cache")
    obj.gemma_encoder_pair = SimpleNamespace(ensure_loaded=mock.MagicMock())
    obj.encode_prompts = mock.MagicMock(return_value=[(torch.zeros(4, 16), torch.zeros(4, 16))])
    obj._prepare_transformer = mock.MagicMock()
    obj._kf_trace_anchors = mock.MagicMock(return_value=[])

    # Reference encode collaborators.
    obj._build_ref_vae_encoders = mock.MagicMock()

    def _load_ref(path, h, w, ref_pixel_frames, **kw):
        return torch.zeros(1, 3, ref_pixel_frames, h, w)

    obj._load_reference_video = mock.MagicMock(side_effect=_load_ref)

    def _encode_image(clip, encoder=None):
        lf = (clip.shape[2] - 1) // _TEMPORAL_COMPRESSION + 1
        return torch.zeros(1, 128, lf, clip.shape[3] // 32, clip.shape[4] // 32)

    obj.encode_image = mock.MagicMock(side_effect=_encode_image)

    # The two denoise stages: return latents shaped exactly like the real target-only strip so the
    # stage-1 reshape and the upsample flatten line up.
    def _denoise(v_embeds, a_embeds, *, num_frames, height, width, **kw):
        lf = (num_frames - 1) // _TEMPORAL_COMPRESSION + 1
        v = torch.zeros(1, lf * (height // 32) * (width // 32), 128)
        a = torch.zeros(1, 8, 128)
        return v, a

    obj._denoise_no_guidance = mock.MagicMock(side_effect=_denoise)

    # Upsample: full-res spatial latent (1, 128, lf, H/32, W/32).
    def _upsample(upsampler, s1_spatial, *stats):
        lf = s1_spatial.shape[2]
        return torch.zeros(1, 128, lf, 64 // 32, 64 // 32)

    ns["upsample_latent"].side_effect = _upsample

    obj._ensure_upsampler_frames = mock.MagicMock()
    obj._prepare_upsampler = mock.MagicMock()
    obj._vae_per_channel_stats = mock.MagicMock(return_value=())
    obj._ensure_vae_decoder_frames = mock.MagicMock()
    obj._prepare_vae = mock.MagicMock()
    obj.decode_latents = mock.MagicMock(side_effect=lambda v, lf, lh, lw, output_type: torch.zeros(1, 3, 17, 64, 64))
    obj.decode_audio = mock.MagicMock(return_value=mock.MagicMock(name="audio"))
    return obj, ns


def _stage_calls(obj):
    return obj._denoise_no_guidance.call_args_list


def test_generate_routes_trace_key_to_ref_family(monkeypatch, tmp_path):
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    monkeypatch.delenv("LTX_GEN_EAGER_STAGES", raising=False)
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj, _ns = _make_generate_pipe()
    obj.generate("a prompt", num_frames=17, height=64, width=64, reference_video=(_sheet(tmp_path), 0.9))

    calls = _stage_calls(obj)
    assert len(calls) == 2
    s1, s2 = calls
    assert s1.kwargs["trace_key"] == "s1_ref", "reference gen must route stage 1 to s1_ref"
    assert s2.kwargs["trace_key"] == "s2_ref", "reference gen must route stage 2 to s2_ref"
    # Reference latents + strength threaded into both stages.
    assert s1.kwargs["ref_latent"] is not None and s1.kwargs["ref_strength"] == 0.9
    assert s2.kwargs["ref_latent"] is not None and s2.kwargs["ref_strength"] == 0.9
    # Ref encoders built + looped-still encoded per stage.
    obj._build_ref_vae_encoders.assert_called_once_with(17, 64, 64)
    assert obj.encode_image.call_count == 2


def test_generate_keeps_grid_upsampled_latent(monkeypatch, tmp_path):
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    monkeypatch.delenv("LTX_GEN_EAGER_STAGES", raising=False)
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj, _ns = _make_generate_pipe()
    obj.generate("a prompt", num_frames=17, height=64, width=64, reference_video=(_sheet(tmp_path), 1.0))

    # latent_frames=3, hw_full=4 => stage-2 init stays GRID-sized (target only) = 3*4 = 12; NOT +R padded.
    # _denoise allocates base_v combined-sized and the ref pin fills [T,T+R) — matching the s2/keyframe path.
    s2 = _stage_calls(obj)[1]
    assert s2.kwargs["initial_video_latent"].shape[1] == 3 * 4, "upsampled latent must stay grid-sized (target-only)"


def test_generate_no_ref_keeps_base_trace_keys(monkeypatch):
    """No reference: trace_key stays s1/s2, no ref_latent, and the upsampled latent is target-only
    (byte-identical to the pre-Item-4 path)."""
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    monkeypatch.delenv("LTX_GEN_EAGER_STAGES", raising=False)
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj, _ns = _make_generate_pipe()
    obj.generate("a prompt", num_frames=17, height=64, width=64)

    calls = _stage_calls(obj)
    s1, s2 = calls
    assert s1.kwargs["trace_key"] == "s1"
    assert s2.kwargs["trace_key"] == "s2"
    assert s1.kwargs["ref_latent"] is None and s2.kwargs["ref_latent"] is None
    assert s2.kwargs["initial_video_latent"].shape[1] == 3 * 4, "no-ref stage-2 init stays target-only"
    obj._build_ref_vae_encoders.assert_not_called()
    obj.encode_image.assert_not_called()


def test_generate_ref_plus_append_token_asserts(monkeypatch, expect_error, tmp_path):
    """Reference + append-token keyframe must assert (they contradict on the KV-logical/decode
    roles), never silently pick one."""
    monkeypatch.setenv("LTX_KF_APPEND_TOKEN", "1")
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj, _ns = _make_generate_pipe()
    with expect_error(
        AssertionError,
        r"reference conditioning .*reference_video.* is mutually exclusive with append-token keyframe conditioning",
    ):
        obj.generate("a prompt", num_frames=17, height=64, width=64, reference_video=(_sheet(tmp_path), 1.0))


def test_second_reference_gen_reuses_the_cached_latents(monkeypatch, tmp_path):
    """A repeat sheet must NOT re-encode: the eager VAE encode deadlocks the device when it follows a
    denoise trace replay, which is exactly what the second gen of a long-lived worker does.

    The server decrypts each upload to a FRESH temp path per job, so the same sheet arrives under a
    new filename every time — the memoize has to key on content or it never hits in production.
    """
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    monkeypatch.delenv("LTX_GEN_EAGER_STAGES", raising=False)
    monkeypatch.delenv("LTX_ITER_FAST", raising=False)
    obj, _ns = _make_generate_pipe()
    sheets = []
    for i in range(3):
        p = tmp_path / f"job{i}.ref.png"  # same bytes, different path, as the worker writes them
        p.write_bytes(b"SHEET-BYTES")
        sheets.append(str(p))
    for p in sheets:
        obj.generate("a prompt", num_frames=17, height=64, width=64, reference_video=(p, 1.0))
    assert obj.encode_image.call_count == 2, "one encode per stage, once — not once per generation"

    other = tmp_path / "other.ref.png"
    other.write_bytes(b"DIFFERENT-SHEET")  # different content is a genuine miss
    obj.generate("a prompt", num_frames=17, height=64, width=64, reference_video=(str(other), 1.0))
    assert obj.encode_image.call_count == 4
