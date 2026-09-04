# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the IC-LoRA reference-conditioning port (Item 2).

These cover the non-device pieces of the LTX pipeline's reference-sheet support:

  * ``_load_reference_video`` loops a real still into a ``(1, 3, R, H, W)`` static
    reference video with every frame identical (the "loop the still to the output
    length" preprocessing);
  * ``_prepare_vae_encoder`` / ``encode_image`` gained an additive ``encoder=`` kwarg
    that overrides ``self.vae_encoder`` and is inert (identical behavior) when omitted.

The full ``pipeline_ltx`` module imports ``ttnn`` (and a large tt_dit device stack) at
import time, which needs a matching device build. To keep this a genuine *host-only*
test that runs regardless of build state, the pure-torch staticmethods are exercised by
loading only their real source out of ``pipeline_ltx.py`` (no ``ttnn`` import), and the
new-kwarg additions are verified structurally against the real source via ``ast``. A
best-effort real-import inspection runs too, skipping cleanly if ``ttnn`` is unavailable.
"""
from __future__ import annotations

import ast
import inspect
from io import BytesIO
from pathlib import Path

import pytest
import torch

PIPELINE_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx.py"
REFERENCE_SHEET = "/home/smarton/ic-lora/opt/reference_sheet.png"

# Pure-torch/PIL staticmethods with no ttnn dependency; loaded from source in isolation.
_PURE_STATICMETHODS = ("_crf_codec_roundtrip", "_load_conditioning_image", "_load_reference_video")


def _pipeline_source() -> str:
    return PIPELINE_PATH.read_text()


def _parse_pipeline() -> tuple[ast.Module, ast.ClassDef]:
    tree = ast.parse(_pipeline_source())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LTXPipeline")
    return tree, cls


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _load_isolated_pipeline_shim():
    """Build a throwaway ``LTXPipeline`` shim holding only the pure-torch staticmethods,
    lifted verbatim from the real source. This exercises the actual shipped code paths for
    ``_load_reference_video`` (and its ``_load_conditioning_image`` dependency) without
    importing ``ttnn`` or any device modules."""
    src = _pipeline_source()
    src_lines = src.splitlines(keepends=True)
    tree, cls = _parse_pipeline()

    def _node_lines(node: ast.AST) -> str:
        # Slice raw source lines (node.lineno is the def/assign line, decorators excluded),
        # preserving the original indentation of every line so it re-compiles cleanly.
        return "".join(src_lines[node.lineno - 1 : node.end_lineno])

    # DEFAULT_IMAGE_CRF is a module-level constant used as a default-arg value, so it must
    # exist in the exec namespace before the class body runs.
    crf_assign = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFAULT_IMAGE_CRF" for t in n.targets)
    )

    # Each method's raw lines are already indented for the class body; prepend the
    # @staticmethod decorator (same 4-space indent) and assemble a minimal class.
    method_blocks = ["    @staticmethod\n" + _node_lines(_find_method(cls, name)) for name in _PURE_STATICMETHODS]

    shim_src = _node_lines(crf_assign) + "\n\nclass LTXPipeline:\n" + "\n".join(method_blocks)
    ns: dict = {"torch": torch, "BytesIO": BytesIO}
    exec(compile(shim_src, str(PIPELINE_PATH), "exec"), ns)
    return ns["LTXPipeline"]


# --------------------------------------------------------------------------------------
# _load_reference_video: functional (real source, host-only)
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not Path(REFERENCE_SHEET).exists(), reason=f"reference sheet missing: {REFERENCE_SHEET}")
def test_load_reference_video_loops_still_identical_frames():
    shim = _load_isolated_pipeline_shim()
    R, H, W = 5, 64, 96
    # crf=0 skips the libx264 round-trip so the test needs no ffmpeg/av; the loop/expand
    # logic under test is identical regardless of crf.
    video = shim._load_reference_video(REFERENCE_SHEET, height=H, width=W, ref_pixel_frames=R, crf=0)

    assert isinstance(video, torch.Tensor)
    assert tuple(video.shape) == (1, 3, R, H, W)
    assert video.dtype == torch.float32
    # Static reference: every frame identical to frame 0.
    for i in range(1, R):
        assert torch.equal(video[:, :, 0], video[:, :, i]), f"frame {i} differs from frame 0"
    # Normalized into [-1, 1] by _load_conditioning_image.
    assert float(video.min()) >= -1.0 - 1e-6
    assert float(video.max()) <= 1.0 + 1e-6
    # The looped frame must equal the single-frame conditioning image (broadcast correctness).
    frame = shim._load_conditioning_image(REFERENCE_SHEET, height=H, width=W, crf=0)  # (1,3,1,H,W)
    assert tuple(frame.shape) == (1, 3, 1, H, W)
    assert torch.equal(video[:, :, 0:1], frame)


@pytest.mark.skipif(not Path(REFERENCE_SHEET).exists(), reason=f"reference sheet missing: {REFERENCE_SHEET}")
def test_load_reference_video_contiguous_and_frame_count():
    shim = _load_isolated_pipeline_shim()
    for R in (1, 3, 8):
        video = shim._load_reference_video(REFERENCE_SHEET, height=48, width=48, ref_pixel_frames=R, crf=0)
        assert tuple(video.shape) == (1, 3, R, 48, 48)
        # Reference tree returns a materialized (contiguous) tensor, not a broadcast view.
        assert video.is_contiguous()


@pytest.mark.skipif(not Path(REFERENCE_SHEET).exists(), reason=f"reference sheet missing: {REFERENCE_SHEET}")
def test_load_reference_video_default_crf_roundtrip_still_identical():
    pytest.importorskip("av", reason="PyAV/libx264 needed for the default-CRF codec round-trip")
    shim = _load_isolated_pipeline_shim()
    R, H, W = 4, 64, 64
    video = shim._load_reference_video(REFERENCE_SHEET, height=H, width=W, ref_pixel_frames=R)
    assert tuple(video.shape) == (1, 3, R, H, W)
    for i in range(1, R):
        assert torch.equal(video[:, :, 0], video[:, :, i])


# --------------------------------------------------------------------------------------
# encoder= additive kwarg: structural checks against the real source (no ttnn import)
# --------------------------------------------------------------------------------------
def _arg_default_map(fn: ast.FunctionDef) -> dict:
    """Map each positional-or-keyword arg name -> its default AST node (or _MISSING)."""
    args = fn.args.args
    defaults = fn.args.defaults
    n_no_default = len(args) - len(defaults)
    out = {}
    for i, a in enumerate(args):
        out[a.arg] = None if i < n_no_default else defaults[i - n_no_default]
    return out


def test_prepare_vae_encoder_has_optional_encoder_kwarg():
    _, cls = _parse_pipeline()
    fn = _find_method(cls, "_prepare_vae_encoder")
    dmap = _arg_default_map(fn)
    assert "encoder" in dmap, "_prepare_vae_encoder must accept an `encoder` param"
    default = dmap["encoder"]
    assert (
        isinstance(default, ast.Constant) and default.value is None
    ), "encoder must default to None (inert when omitted)"


def test_encode_image_has_optional_encoder_kwarg():
    _, cls = _parse_pipeline()
    fn = _find_method(cls, "encode_image")
    dmap = _arg_default_map(fn)
    assert "encoder" in dmap, "encode_image must accept an `encoder` param"
    default = dmap["encoder"]
    assert (
        isinstance(default, ast.Constant) and default.value is None
    ), "encoder must default to None (inert when omitted)"


def test_encode_image_threads_encoder_and_keeps_host_parity_path():
    """encode_image must fall back to self.vae_encoder when encoder is omitted, thread the
    chosen encoder into _prepare_vae_encoder, and keep the LTX_VAE_ENCODER_HOST parity path."""
    _, cls = _parse_pipeline()
    body_src = ast.get_source_segment(_pipeline_source(), _find_method(cls, "encode_image"))
    assert "enc = encoder or self.vae_encoder" in body_src
    assert "self._prepare_vae_encoder(enc)" in body_src
    assert "enc(image_BCFHW)" in body_src
    # Host-parity self-check path must remain intact (not re-inlined shard padding).
    assert "LTX_VAE_ENCODER_HOST" in body_src
    assert "_host_encode_image" in body_src
    assert "_log_encoder_parity" in body_src
    assert "pad_hw_replicate" not in body_src, "inline shard padding must NOT be re-added (lives in the module now)"


def test_prepare_vae_encoder_uses_override_not_self():
    _, cls = _parse_pipeline()
    body_src = ast.get_source_segment(_pipeline_source(), _find_method(cls, "_prepare_vae_encoder"))
    assert "enc = encoder or self.vae_encoder" in body_src
    assert "conv3d_blocking_hash(enc)" in body_src
    # load_model must load the resolved encoder, not hardcode self.vae_encoder.
    assert "load_model(\n            enc," in body_src or "load_model(enc," in body_src


def test_new_ref_encoder_attrs_and_methods_present():
    src = _pipeline_source()
    _, cls = _parse_pipeline()
    # Additive attributes declared in __init__.
    assert "self.vae_ref_encoder_s1 = None" in src
    assert "self.vae_ref_encoder_full = None" in src
    # New methods exist.
    method_names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "_build_ref_vae_encoders" in method_names
    assert "_load_reference_video" in method_names
    assert "_warmup_ref_encode" in method_names
    # Exactly one _warmup_ref_encode (do not duplicate the merge-artifact _warmup_encode).
    assert sum(1 for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_warmup_ref_encode") == 1


def test_load_reference_video_is_staticmethod_with_expected_signature():
    _, cls = _parse_pipeline()
    fn = _find_method(cls, "_load_reference_video")
    assert any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in fn.decorator_list)
    argnames = [a.arg for a in fn.args.args]
    assert argnames == ["image_path", "height", "width", "ref_pixel_frames", "crf"]


def test_register_coresident_exclusions_wires_ref_encoders():
    _, cls = _parse_pipeline()
    body_src = ast.get_source_segment(_pipeline_source(), _find_method(cls, "_register_coresident_exclusions"))
    # Ref encoders are wired the same way the image encoder is (evict/exclude peers).
    assert "vae_ref_encoder_s1" in body_src and "vae_ref_encoder_full" in body_src
    assert "register_coresident_exclusions" in body_src


# --------------------------------------------------------------------------------------
# Best-effort real-import inspection (skips cleanly on a stale/missing ttnn build)
# --------------------------------------------------------------------------------------
def test_real_import_signatures_when_ttnn_available():
    try:
        from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
    except Exception as e:  # noqa: BLE001 - ttnn build mismatch / missing device stack
        pytest.skip(f"pipeline_ltx not importable in this environment (host-only run): {e}")

    prep = inspect.signature(LTXPipeline._prepare_vae_encoder)
    assert "encoder" in prep.parameters and prep.parameters["encoder"].default is None
    enc = inspect.signature(LTXPipeline.encode_image)
    assert "encoder" in enc.parameters and enc.parameters["encoder"].default is None
    ref = inspect.signature(LTXPipeline._load_reference_video)
    assert list(ref.parameters) == ["image_path", "height", "width", "ref_pixel_frames", "crf"]
    assert hasattr(LTXPipeline, "_build_ref_vae_encoders")
    assert hasattr(LTXPipeline, "_warmup_ref_encode")
