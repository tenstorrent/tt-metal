# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""optimize may not replace a topology it was GIVEN with one it guessed.

THE CASE. Qwen-Image-Edit, 2026-10-01, T3K: `optimize --box T3K --mesh 2,4 --devices all` printed
two contradictory lines one after the other --

    mesh device : --box T3K + mesh 2x4 -> MESH_DEVICE=T3K_2x4
    topology    : 8-chip -> mesh 1x8 (TP=8 DP=1) [1D default]

-- and exported the second. The generated perf test honoured it (resolve_mesh_shape reads
TT_PERF_MESH_ROWS/COLS), opened 1x8, and the text encoder refused: its tensor-parallel width is the
mesh's SECOND axis and Qwen2.5-VL has 4 KV heads, so `4 kv heads not divisible by TP=8`. Three
perf-test regenerations died on it, discovery gave up, and the run aborted before the optimisation
loop ever started.

TWO INDEPENDENT DEFECTS PRODUCED THAT.

1. The shape was parsed and multiplied away. _derive_topology_env only ever received
   _chip_count_from_mesh's PRODUCT, so `--mesh 2,4` arrived as the scalar 8 and was rebuilt as
   `1 x chips` -- inverting the axes of the mesh that was asked for. Its sibling
   _derive_mesh_device_env parsed the same flag with the same helper and kept the pair, which is
   why MESH_DEVICE was right on the line above.

2. The demo was misclassified as hand-written. classify_pipeline keyed only on
   BRINGUP_STATUS_FILENAME, which an ASSEMBLED e2e demo does not carry: bring-up scaffolded this
   model's text encoder / transformer / VAE as separate components (each with its own status file
   and its own parallelism_manifest.json) and emit-e2e assembled the pipeline over them, leaving
   e2e_plan.json as the marker. So a demo whose own open_mesh had the graduated, gate-proven shape
   was treated as one with nothing worth preserving.

WHY THE GUESS SURVIVED THIS LONG. The [1D default] branch fires for any model the planner cannot
probe, and a composite checkpoint can NEVER be probed -- a diffusers pipeline has no root config,
only submodels, so plan_parallelism returns None by construction. It went unnoticed because every
earlier model through this path had 8 KV heads, which 1x8 divides exactly. The fallback was never
validated, only never contradicted.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

from scripts.tt_hw_planner.bringup_plan import BRINGUP_STATUS_FILENAME, E2E_PLAN_FILENAME
from scripts.tt_hw_planner.commands import optimize as O


def _args(**kw):
    kw.setdefault("mesh", None)
    kw.setdefault("devices", "")
    kw.setdefault("target", "some/model")
    return Namespace(**kw)


def _clear():
    os.environ.pop("TT_PERF_MESH_ROWS", None)
    os.environ.pop("TT_PERF_MESH_COLS", None)


def _exported():
    return (os.environ.get("TT_PERF_MESH_ROWS"), os.environ.get("TT_PERF_MESH_COLS"))


# --------------------------------------------------------------------------------------------
# 1. the shape the operator typed survives to the export
# --------------------------------------------------------------------------------------------


def test_an_explicit_mesh_shape_is_exported_not_its_product():
    """`--mesh 2,4` must export rows=2 cols=4. It used to export 1x8 -- the right chip count on the
    wrong axes, which is the one topology this model cannot run."""
    for spelling in ("2,4", "2x4", "2X4"):
        _clear()
        O._derive_topology_env(_args(mesh=spelling, devices="all"), model_dir=None)
        assert _exported() == ("2", "4"), "%r exported %s" % (spelling, _exported())


def test_the_shape_helper_is_the_single_owner_of_the_pair():
    """Both callers ask the same helper, so they cannot disagree the way MESH_DEVICE and the
    topology export did."""
    assert O._mesh_shape_arg("2,4") == (2, 4)
    assert O._mesh_shape_arg("2x4") == (2, 4)
    assert O._mesh_shape_arg(None) is None
    assert O._mesh_shape_arg("") is None
    assert O._mesh_shape_arg("8") is None  # 1-D: a chip count, not a topology
    assert O._mesh_shape_arg("2x2x2") is None  # >2-D: not a shape this exports
    assert O._mesh_shape_arg("0x8") is None  # a zero axis is not a mesh
    assert O._mesh_shape_arg("garbage") is None


def test_an_explicit_shape_outranks_the_kernel_viable_split(monkeypatch):
    """The operator named a topology; a planner preference does not overrule it."""
    from scripts.tt_hw_planner.parallelism import ParallelConfig

    monkeypatch.setattr(
        "scripts.tt_hw_planner.parallelism.plan_parallelism", lambda mid, chips: ParallelConfig(tp=8, dp=1)
    )
    _clear()
    O._derive_topology_env(_args(mesh="2,4", devices="all"), model_dir=None)
    assert _exported() == ("2", "4")


def test_an_unparseable_mesh_still_reaches_the_old_fallback():
    """Garbage in --mesh must not become a shape; the chip count then comes from --devices as before."""
    _clear()
    O._derive_topology_env(_args(mesh="garbage", devices="0,1,2,3"), model_dir="/tmp")
    assert _exported() == ("1", "4")


# --------------------------------------------------------------------------------------------
# 2. an emitted demo's own mesh is not overwritten by a guess
# --------------------------------------------------------------------------------------------


def _emitted(tmp_path, marker):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / marker).write_text("{}")
    return tmp_path


def test_an_assembled_demo_is_recognised_as_emitted(tmp_path):
    """e2e_plan.json alone marks an ASSEMBLED demo. Keying only on the component status file
    classified Qwen-Image-Edit as hand-written, which is how its graduated mesh became overwritable."""
    assert O.classify_pipeline(_emitted(tmp_path / "a", BRINGUP_STATUS_FILENAME)) == O._EMITTED
    assert O.classify_pipeline(_emitted(tmp_path / "b", E2E_PLAN_FILENAME)) == O._EMITTED
    plain = tmp_path / "c"
    plain.mkdir(parents=True, exist_ok=True)
    assert O.classify_pipeline(plain) == O._EXISTING


def test_an_emitted_demo_keeps_its_own_mesh(tmp_path):
    """Nothing exported: resolve_mesh_shape defers to the demo's own shape only while this pair is
    unset, which is exactly why emit-e2e -- which exports neither -- got this right."""
    _clear()
    O._derive_topology_env(
        _args(devices="0,1,2,3,4,5,6,7"), model_dir=None, demo_dir=_emitted(tmp_path, E2E_PLAN_FILENAME)
    )
    assert _exported() == (None, None)


def test_a_hand_written_model_still_gets_the_fallback(tmp_path):
    """The deference is scoped to demos that HAVE a graduated topology. Everything else is unchanged,
    so no existing model's behaviour moves."""
    plain = tmp_path / "hand_written"
    plain.mkdir(parents=True, exist_ok=True)
    _clear()
    O._derive_topology_env(_args(devices="0,1,2,3"), model_dir=str(plain), demo_dir=plain)
    assert _exported() == ("1", "4")


def test_the_flag_used_to_name_a_demo_does_not_change_what_it_is(tmp_path):
    """--model-dir makes `kind` _EXISTING for display, but whether the directory opens its own
    graduated mesh is a property of the code there, not of how the operator referred to it."""
    demo = _emitted(tmp_path, E2E_PLAN_FILENAME)
    _clear()
    O._derive_topology_env(_args(devices="0,1,2,3,4,5,6,7"), model_dir=str(demo), demo_dir=demo)
    assert _exported() == (None, None)


# --------------------------------------------------------------------------------------------
# 3. the constraints this change is held to
# --------------------------------------------------------------------------------------------


def test_the_caller_passes_the_demo_dir():
    """A default-None parameter is only a fix if the production call site actually fills it."""
    import inspect

    src = inspect.getsource(O)
    assert "_derive_topology_env(args, model_dir, demo_dir=demo_dir)" in src


def test_no_second_copy_of_the_marker_filenames():
    """The names live with the writer; this module imports them rather than retyping them."""
    import inspect

    src = inspect.getsource(O)
    for literal in ('"bringup_status.json"', '"e2e_plan.json"'):
        assert literal not in src, "optimize.py retypes %s instead of importing the constant" % literal


def test_it_names_no_model_or_stage():
    """The branch keys on a file marker and a flag, never on a model, component or stage name."""
    import inspect

    body = inspect.getsource(O._derive_topology_env) + inspect.getsource(O.classify_pipeline)
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    code = code.replace('"""', "\x00").split("\x00")
    code = "".join(code[::2])  # drop docstrings; the case history names the model deliberately
    for name in ("qwen", "image_edit", "text_encoder", "transformer", "vae", "denoise", "llama"):
        assert name not in code.lower(), "%r is hardcoded in the logic" % name
