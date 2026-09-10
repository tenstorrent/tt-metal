# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the IC-LoRA reference-conditioning length decouple (Item 3).

Item 3 is the riskiest reconciliation in the port: ``_denoise_no_guidance`` tracks three
sequence lengths that COINCIDE for base / i2v / keyframe but SPLIT for a reference gen:

  * ``decode_N``            — the return/decode strip: always TARGET-ONLY (``latent_frames*hw``).
  * ``video_kv_logical_n``  — audio cross-attention's logical view of the video K/V AND the V→A
                              pad-mask real region. Keyframe: the target grid (anchors hidden).
                              Reference: the COMBINED (target + reference) length — audio MUST
                              attend to the reference block, matching the proven SOURCE.
  * ``video_N_real``        — the extended self-attention (Q) length: grid + anchor blocks
                              (keyframe) OR grid + reference block (reference).

The trap: leaving ``video_kv_logical_n`` at the target grid in the reference path compiles,
runs, and logs clean while silently dropping the reference from audio cross-attention. These
tests assert the length SEMANTICS without a device.

``pipeline_ltx_distilled`` imports the full tt_dit device stack (``ttnn.experimental`` etc.),
which needs a matching device build, so — like the other IC host tests — the real method
sources are AST-extracted and exec'd into an isolated shim whose device dependencies are
stubbed. The methods under test (``_denoise_no_guidance`` + ``_prepare_stage_statics``) and
their pure-python helpers run VERBATIM from source; the transformer, statics builder, tensor
ops, and device round-trip are mocked, so what is asserted is the real length math.

Run: PYTHONPATH=<worktree> python -m pytest <file> -v -p no:cacheprovider
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def _shim_ttnn() -> mock.MagicMock:
    """The ``ttnn`` the exec'd method sources see. Always a mock, never the real module: every
    ``ttnn`` op the sources reach here is a device op fed mocked tensors, and the real bindings
    reject them. Only ``TILE_SIZE`` (used by ``_sp_pad_len``) needs a true value. Host-only."""
    fake = mock.MagicMock(name="ttnn")
    fake.TILE_SIZE = 32
    return fake


TILE_SIZE = 32
DISTILLED_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"

# Latent grid geometry (mirrors utils.ltx, kept in sync with TEMPORAL/SPATIAL_COMPRESSION).
_TEMPORAL_COMPRESSION = 8
_SPATIAL_COMPRESSION = 32


def _latent_grid(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    return (num_frames - 1) // _TEMPORAL_COMPRESSION + 1, height // _SPATIAL_COMPRESSION, width // _SPATIAL_COMPRESSION


def _sp_pad(n_real: int, sp_factor: int) -> int:
    divisor = TILE_SIZE * sp_factor
    return ((n_real + divisor - 1) // divisor) * divisor


# ------------------------------------------------------------------------------------------
# Isolated shim: real method sources exec'd with stubbed device globals.
# ------------------------------------------------------------------------------------------
def _build_shim_namespace() -> dict:
    """Exec the real ``build_conditioning_tensors`` + ``_I2VConditioning`` + the four target methods
    (``_post_process_latent_tt``, ``_noise_video_latent``, ``_build_i2v_conditioning``,
    ``_denoise_no_guidance``) into a namespace with device dependencies stubbed. ``from __future__
    import annotations`` keeps every ``ttnn.Tensor`` / ``torch.Tensor`` annotation a lazy string, so
    no annotation is ever evaluated."""
    src = DISTILLED_PATH.read_text()
    src_lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    def _top(name: str) -> ast.AST:
        return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)

    cls = _top("LTXDistilledPipeline")

    def _method(name: str) -> ast.FunctionDef:
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def _lines(node: ast.AST) -> str:
        # node.lineno is the def/class line (decorators excluded); slice preserves original indent.
        return "".join(src_lines[node.lineno - 1 : node.end_lineno])

    bct = _lines(_top("build_conditioning_tensors"))
    i2v_dc = "@dataclass\n" + _lines(_top("_I2VConditioning"))
    static_methods = ("_post_process_latent_tt", "_noise_video_latent")
    plain_methods = ("_build_i2v_conditioning", "_denoise_no_guidance")
    method_blocks = ["    @staticmethod\n" + _lines(_method(n)) for n in static_methods]
    method_blocks += [_lines(_method(n)) for n in plain_methods]

    shim_src = (
        "from __future__ import annotations\n\n"
        + bct
        + "\n\n"
        + i2v_dc
        + "\n\n"
        + "class LTXDistilledPipeline:\n"
        + "\n".join(method_blocks)
    )

    captured: dict = {}

    def _device_to_host(x, **kwargs):
        # Return a real, generously sized tensor so the target-only return slice `[:, :decode_N, :]`
        # yields exactly decode_N rows (and audio's `[:, :audio_N_real, :]` yields audio_N_real).
        n = captured.get("video_N", 100_000)
        return torch.zeros(1, 1, n, captured.get("in_channels", 8))

    ns: dict = {
        "torch": torch,
        "os": __import__("os"),
        "ttnn": _shim_ttnn(),
        "dataclass": __import__("dataclasses").dataclass,
        "field": __import__("dataclasses").field,
        "logger": mock.MagicMock(name="logger"),
        "latent_grid": _latent_grid,
        "VideoPixelShape": lambda **kw: SimpleNamespace(**kw),
        # Audio length is independent of the video decouple under test; a fixed real frame count
        # keeps the audio path exercised without coupling the assertions to audio geometry.
        "AudioLatentShape": SimpleNamespace(from_video_pixel_shape=staticmethod(lambda vps: SimpleNamespace(frames=8))),
        "LTXTransformerState": lambda: mock.MagicMock(name="state"),
        "LTXTransformerModel": SimpleNamespace(device_to_host=staticmethod(_device_to_host)),
        "bf16_tensor": lambda t, device=None, **kw: t,
    }
    exec(compile(shim_src, str(DISTILLED_PATH), "exec"), ns)
    ns["_captured"] = captured
    return ns


def _make_pipeline(ns: dict, *, in_channels: int, sp_factor: int, image_conditioning: bool):
    """Instantiate the shim and wire the mocked collaborators. Returns (obj, captured, inner_step)."""
    Pipe = ns["LTXDistilledPipeline"]
    obj = Pipe()
    captured = ns["_captured"]
    captured["in_channels"] = in_channels

    obj.in_channels = in_channels
    obj.parallel_config = SimpleNamespace(sequence_parallel=SimpleNamespace(factor=sp_factor, mesh_axis=2))
    obj.mesh_device = mock.MagicMock(name="mesh_device")
    obj.ccl_manager = mock.MagicMock(name="ccl_manager")
    obj._trace_state = {}
    obj._prepare_prompt = mock.MagicMock(name="_prepare_prompt")

    inner_step = mock.MagicMock(name="inner_step", return_value=(mock.MagicMock(), mock.MagicMock()))
    obj.transformer = SimpleNamespace(image_conditioning=image_conditioning, inner_step=inner_step)

    def _sp_pad_len(n_real):
        divisor = TILE_SIZE * sp_factor
        return ((n_real + divisor - 1) // divisor) * divisor

    obj._sp_pad_len = _sp_pad_len

    # Capture every kwarg _prepare_stage_statics is called with (the V→A pad-mask real region is its
    # `video_N_grid`; `video_N` sizes the device round-trip return above).
    def _capture_statics(state, **kwargs):
        captured.update(kwargs)

    obj._prepare_stage_statics = mock.MagicMock(name="_prepare_stage_statics", side_effect=_capture_statics)
    return obj, captured, inner_step


def _run_denoise(obj, *, num_frames, height, width, image_conds=None, ref_latent=None, ref_strength=1.0):
    seq, dim = 4, 16
    return obj._denoise_no_guidance(
        torch.zeros(seq, dim),
        torch.zeros(seq, dim),
        num_frames=num_frames,
        height=height,
        width=width,
        sigma_values=[1.0, 0.0],
        seed=0,
        image_conds=image_conds,
        ref_latent=ref_latent,
        ref_strength=ref_strength,
        traced=False,
    )


# Geometry: latent_frames=3 (num_frames 17), latent_h=latent_w=3 (96/32), hw=9, grid=27.
NUM_FRAMES, HEIGHT, WIDTH = 17, 96, 96
IN_CH, SP = 8, 1
GRID = 3 * 3 * 3  # latent_frames * hw = 27


def _cond(lat_idx, strength, lh=3, lw=3, c=IN_CH):
    return (lat_idx, torch.randn(1, c, 1, lh, lw), strength)


# ==========================================================================================
# No-op cases: reference OFF ⇒ decode / kv-logical / pad-mask all collapse to the grid.
# ==========================================================================================
def test_t2v_no_ref_collapses_to_grid(monkeypatch):
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    ns = _build_shim_namespace()
    obj, cap, inner = _make_pipeline(ns, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    (v_out, _a_out) = _run_denoise(obj, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH)

    kw = inner.call_args.kwargs
    assert kw["video_kv_logical_n"] == GRID, "no-ref KV-logical must be the target grid"
    assert kw["video_N"] == GRID, "no-ref self-attn (Q) length must be the grid (no anchors, no ref)"
    assert cap["video_N_grid"] == GRID, "no-ref V→A pad-mask real region must be the grid"
    assert cap["ref_latent_frames"] == 0
    assert cap["anchor_frames"] == []
    assert cap["video_N"] == _sp_pad(GRID, SP)
    assert v_out.shape[1] == GRID, "return strip (decode_N) must be the grid"


def test_i2v_frame0_collapses_to_grid(monkeypatch):
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    ns = _build_shim_namespace()
    obj, cap, inner = _make_pipeline(ns, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    (v_out, _a) = _run_denoise(obj, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, image_conds=[_cond(0, 1.0)])

    kw = inner.call_args.kwargs
    assert kw["video_kv_logical_n"] == GRID
    assert kw["video_N"] == GRID
    assert cap["video_N_grid"] == GRID
    assert cap["ref_latent_frames"] == 0
    assert cap["anchor_frames"] == []
    assert v_out.shape[1] == GRID


def test_keyframe_append_token_kv_and_decode_stay_grid(monkeypatch):
    """Append-token keyframe: the anchor block extends the self-attn (Q) length ONLY. The
    KV-logical, the V→A pad-mask real region, and the decode strip all stay the target grid —
    the anchors are hidden from audio and stripped before decode. This is the keyframe corner of
    the same decouple: kv-logical == decode == grid, while video_N_real == grid + anchors."""
    monkeypatch.setenv("LTX_KF_APPEND_TOKEN", "1")
    ns = _build_shim_namespace()
    obj, cap, inner = _make_pipeline(ns, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    # One interior keyframe at latent frame 1 ⇒ one appended hw anchor block.
    (v_out, _a) = _run_denoise(obj, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, image_conds=[_cond(1, 1.0)])

    hw = 9
    kw = inner.call_args.kwargs
    assert cap["anchor_frames"] == [1], "interior keyframe must ride an appended anchor block"
    assert kw["video_N"] == GRID + hw, "self-attn (Q) length includes the anchor block"
    assert kw["video_kv_logical_n"] == GRID, "keyframe KV-logical stays the grid (anchors hidden from audio)"
    assert cap["video_N_grid"] == GRID, "keyframe V→A pad-mask real region stays the grid"
    assert cap["ref_latent_frames"] == 0
    assert v_out.shape[1] == GRID, "decode strip stays the grid (anchors stripped)"


# ==========================================================================================
# Reference case (R>0): decode is TARGET-ONLY while KV-logical / pad-mask are COMBINED.
# ==========================================================================================
@pytest.mark.parametrize("R", [1, 3, 5])
def test_reference_decouples_decode_from_kv_logical(monkeypatch, R):
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    ns = _build_shim_namespace()
    obj, cap, inner = _make_pipeline(ns, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    ref_latent = torch.randn(1, IN_CH, R, 3, 3)  # (1, C, R, lh, lw)
    (v_out, _a) = _run_denoise(obj, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, ref_latent=ref_latent)

    hw = 9
    combined = GRID + R * hw
    kw = inner.call_args.kwargs
    # THE decouple: the return strip is target-only, the KV-logical / pad-mask are combined.
    assert v_out.shape[1] == GRID, "decode strip (decode_N) must be TARGET-ONLY"
    assert kw["video_kv_logical_n"] == combined, "reference KV-logical MUST be combined (target+reference)"
    assert kw["video_N"] == combined, "reference self-attn (Q) length is the combined count"
    assert cap["video_N_grid"] == combined, "reference V→A pad-mask real region MUST be combined"
    assert cap["video_N_real"] == combined
    assert cap["video_N"] == _sp_pad(combined, SP)
    assert cap["ref_latent_frames"] == R, "combined length threaded to statics for the RoPE two-grid"
    assert cap["anchor_frames"] == [], "reference is NOT an append-token anchor"
    # KV-logical strictly exceeds the decode strip only because the reference is in audio's view.
    assert kw["video_kv_logical_n"] > v_out.shape[1]


def test_reference_plus_frame0_i2v(monkeypatch):
    """Frame-0 i2v conditioning may coexist with a reference (only KEYFRAME/anchor is exclusive).
    The reference block is appended after the (i2v-conditioned) target grid; decode stays
    target-only and KV-logical stays combined."""
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    R = 3
    ns = _build_shim_namespace()
    obj, cap, inner = _make_pipeline(ns, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    ref_latent = torch.randn(1, IN_CH, R, 3, 3)
    (v_out, _a) = _run_denoise(
        obj, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, image_conds=[_cond(0, 1.0)], ref_latent=ref_latent
    )
    combined = GRID + R * 9
    kw = inner.call_args.kwargs
    assert v_out.shape[1] == GRID
    assert kw["video_kv_logical_n"] == combined
    assert cap["video_N_grid"] == combined
    assert cap["ref_latent_frames"] == R
    assert cap["anchor_frames"] == []


def test_reference_and_keyframe_mutually_exclusive(monkeypatch, expect_error):
    """Reference + append-token keyframe active together must assert (they contradict on the
    KV-logical/decode roles), never silently pick one."""
    monkeypatch.setenv("LTX_KF_APPEND_TOKEN", "1")
    ns = _build_shim_namespace()
    obj, _cap, _inner = _make_pipeline(ns, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    ref_latent = torch.randn(1, IN_CH, 2, 3, 3)
    with expect_error(
        AssertionError, r"reference conditioning and append-token keyframe conditioning are mutually exclusive"
    ):
        _run_denoise(obj, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, ref_latent=ref_latent)


def test_reference_is_exact_noop_when_ref_latent_none(monkeypatch):
    """ref_latent=None must be byte-identical to the pre-Item-3 path: the transformer sees the same
    video_N / video_kv_logical_n and the statics see the same lengths as a plain i2v gen. Compares a
    ref=None call against an i2v gen with no reference kwargs threaded."""
    monkeypatch.delenv("LTX_KF_APPEND_TOKEN", raising=False)
    conds = [_cond(0, 1.0)]

    ns_a = _build_shim_namespace()
    obj_a, cap_a, inner_a = _make_pipeline(ns_a, in_channels=IN_CH, sp_factor=SP, image_conditioning=True)
    _run_denoise(obj_a, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, image_conds=conds, ref_latent=None)

    kw_a = inner_a.call_args.kwargs
    assert kw_a["video_kv_logical_n"] == GRID
    assert kw_a["video_N"] == GRID
    assert cap_a["video_N_grid"] == GRID
    assert cap_a["ref_latent_frames"] == 0
    assert cap_a["anchor_frames"] == []
