# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of ``gdn_rows_wrap``: the registry entry (opt-in, BITWISE, off by default), what ``attach``
gives a rows state and when, the shape-only admission, the body's program sequence, and the two agreements the form
stands on -- its prim call is spelled exactly as ``gdn_rows_prims_direct``'s (which the die proved against the
composite), and the two programs it wraps are called the way their own device tests call them."""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_post_rows as post
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_pre_rows as pre
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_prims_direct as prims
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap as module

NAME = module.NAME
TILE, HEADS, HEAD_DIM = module.TILE, module.HEADS, module.HEAD_DIM


def _calls(function) -> list[str]:
    tree = ast.parse(inspect.getsource(function))
    return [
        ast.unparse(node.func)
        for node in sorted(
            (n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
    ]


# --------------------------------------------------------------------------------------- the registry entry


def test_registry_entry_is_a_bitwise_default_the_off_switch_undoes():
    entry = fused.kernel(NAME)
    assert entry.tolerance == fused.BITWISE and entry.default_on is True and NAME in fused.DEFAULT_ON
    assert entry.fused is module.rows_body_wrap and entry.composed is module.rows_body_composed
    assert entry.admits is module.admits and entry.gate is None and entry.component_proof is None
    assert "53 programs" in entry.replaces and "gdn_pre_rows" in entry.replaces
    # the served default: the admitted dispatcher (the wrap where a rows state carries its buffers, the chain else)
    default = fused.resolve_admitted(NAME, {})
    assert isinstance(default, fused.AdmittedStep) and default.fused is module.rows_body_wrap
    assert default.composed is module.rows_body_composed and default.admits is module.admits
    assert fused.resolve(NAME, {}) is module.rows_body_wrap
    # QWEN38_FUSED_OFF=gdn_rows_wrap restores the chain outright
    assert fused.resolve_admitted(NAME, {fused.OFF_ENV: NAME}) is module.rows_body_composed
    assert fused.resolve(NAME, {fused.OFF_ENV: NAME}) is module.rows_body_composed
    assert fused.resolve_admitted(NAME, {fused.OFF_ENV: "all"}) is module.rows_body_composed
    assert fused.enabled(NAME, {}) is True and fused.enabled(NAME, {fused.OFF_ENV: NAME}) is False
    # step 1 of the same lane stays registered and independent: the wrap subsumes it when both are on
    assert fused.kernel(prims.NAME).default_on is False
    assert module.SCALE == prims.SCALE == HEAD_DIM**-0.5
    assert (module.TILE, module.HEADS, module.HEAD_DIM) == (32, 12, 128)
    # the layer's verify segment: 11 programs against the chain's 58 (the 53-program window becomes 6)
    assert module.PROGRAMS_PER_LAYER == 11 and module.CHAIN_PROGRAMS_PER_LAYER == 58
    assert module.CHAIN_PROGRAMS_PER_LAYER - module.PROGRAMS_PER_LAYER == 53 - 6


def test_the_layer_resolves_the_body_once_and_forward_rows_dispatches_through_it():
    source = (gdn_module.__file__ and open(gdn_module.__file__, encoding="utf-8").read()) or ""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNGDN")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = ast.get_source_segment(source, methods["__init__"])
    assert f'self._rows_body_call = fused.resolve_admitted("{NAME}")' in init
    getter = ast.get_source_segment(source, methods["_rows_body_wrap"])  # the fold's composed callable
    assert 'self.__dict__.get("_rows_body_call")' in getter and f'fused.resolve_admitted("{NAME}")' in getter
    forward = ast.get_source_segment(source, methods["forward_rows"])
    assert "body = self._rows_body()" in forward
    assert "body(self, full_hidden, rows_state, state, full_tile=full_tile)" in forward
    # the composed body is the chain's own five calls, in the chain's order
    assert [c.removeprefix("gdn.") for c in _calls(module.rows_body_composed) if c.startswith("gdn._")] == [
        "_project_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_gate_and_project_rows",
    ]
    # and the wrap is attached where the rows state is made, so the choice precedes the warm pass
    allocate = ast.get_source_segment(source, methods["allocate_rows_state"])
    assert "fused.gdn_rows_wrap.attach(self, rows_state)" in allocate


# --------------------------------------------------------------------------------------- attach and admission


class _Fake:
    """A device tensor's surface as the admission reads it: shape, dtype, and ``buffer_address`` -- the method the two
    programs call on every input, which the no-device tests' torch-backed fakes do not have."""

    def __init__(self, shape=(1,), dtype=None):
        self.shape, self.dtype = tuple(shape), dtype

    def is_allocated(self) -> bool:
        return True

    def buffer_address(self) -> int:
        return 0


class _HostFake(_Fake):
    """The rows tests' torch-backed tensor: no ``buffer_address``."""

    buffer_address = None


class _Constants:
    def __init__(self, *, rows=5, tile_rows=TILE):
        self.rows, self.tile_rows = rows, tile_rows
        self.eye = self.tril = self.ones = self.masks = _Fake()
        self.arange_c = _Fake((HEADS, 1, TILE, 1), ttnn.float32)


class _RowsState:
    def __init__(self, *, rows=5, tile_rows=TILE, flat_qk=False, owns_body=True):
        self.constants = _Constants(rows=rows, tile_rows=tile_rows)
        self.flat_qk, self.owns_body = flat_qk, owns_body
        self.v = _Fake((1, 1, tile_rows, module.VALUE_WIDTH), ttnn.bfloat16)
        self.history = _Fake((1, 1, TILE, module.QKV_WIDTH), ttnn.bfloat16)
        self.qkv = _Fake((1, 1, tile_rows, module.QKV_WIDTH), ttnn.bfloat16)


def _gdn() -> SimpleNamespace:
    return SimpleNamespace(
        mesh_device="mesh",
        out_proj_act_memory_config="out-proj-shard",
        weights=SimpleNamespace(dt_bias=_Fake(), neg_exp_A=_Fake(), norm=_Fake(), conv_taps=(_Fake(),) * 4),
    )


@pytest.fixture
def fake_programs(monkeypatch):
    """The two programs' allocators and constant uploads, without a device."""

    freed: list[Any] = []
    monkeypatch.setattr(module, "_release", lambda *tensors: freed.extend(t for t in tensors if t is not None))
    monkeypatch.setattr(pre, "allocate_outputs", lambda mesh, rows: tuple(_Fake((rows, i)) for i in range(6)))
    monkeypatch.setattr(pre, "upload_constants", lambda mesh, dt, na: (_Fake(), _Fake(), _Fake()))
    monkeypatch.setattr(post, "scalar_tensor", lambda mesh: _Fake())
    return freed


def test_attach_serves_the_verify_form_by_default_and_not_under_the_off_switch(fake_programs):
    gdn, rows_state = _gdn(), _RowsState()
    assert module.attach(gdn, rows_state, {fused.OFF_ENV: NAME}) is None  # the off switch: the chain's layouts
    assert module.buffers_of(rows_state) is None
    buffers = module.attach(gdn, rows_state, {})  # the default
    assert buffers is not None and module.buffers_of(rows_state) is buffers
    # the pre program's six outputs, less its v: the rows state's own v is the tensor both paths use
    assert buffers.q_c is not None and buffers.k_c is not None and buffers.sig is not None
    assert len(buffers.tensors) == 9 and all(t is not None for t in buffers.tensors)
    assert buffers.gated_memory_config == "out-proj-shard"
    # every other rows form keeps the chain, whatever the switch says
    for state in (
        _RowsState(tile_rows=128),
        _RowsState(tile_rows=2048),
        _RowsState(flat_qk=True),
        _RowsState(owns_body=False),
    ):
        assert module.qualifies(state) is False
        assert module.attach(_gdn(), state, {fused.ENV: NAME}) is None
        assert module.buffers_of(state) is None
    assert module.qualifies(_RowsState(rows=32)) is True  # the whole tile is still the verify form
    assert module.qualifies(object()) is False
    # a host fake (the rows tests' torch-backed tensors have no buffer_address) keeps the chain under the default
    host = _RowsState()
    host.v = _HostFake((1, 1, TILE, module.VALUE_WIDTH), ttnn.bfloat16)
    assert module.qualifies(host) is False and module.attach(_gdn(), host, {}) is None
    assert module.buffers_of(host) is None


def test_the_gated_tile_targets_the_out_projection_shard_with_a_dram_fallback():
    gdn = _gdn()
    assert module.gated_memory_config(gdn, {}) == "out-proj-shard"
    assert module.gated_memory_config(gdn, {module.GATED_DRAM_ENV: "0"}) == "out-proj-shard"
    assert module.gated_memory_config(gdn, {module.GATED_DRAM_ENV: "1"}) is None
    body = inspect.getsource(module.rows_body_wrap)
    assert "if shard is None:" in body and "ttnn.to_memory_config(gated, gdn.out_proj_act_memory_config)" in body
    assert module.GATED_DRAM_ENV == "QWEN38_FUSED_GDN_ROWS_WRAP_GATED_DRAM"


def test_admission_is_the_attached_buffers_and_the_single_lane_state(fake_programs):
    gdn, rows_state = _gdn(), _RowsState()
    state = SimpleNamespace(recurrent=_Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32))
    assert module.admits(gdn, _Fake(), rows_state, state) is False  # nothing attached: the chain
    module.attach(gdn, rows_state, {})  # the default attaches
    assert module.admits(gdn, _Fake(), rows_state, state) is True
    assert module.admits(gdn, _Fake(), rows_state, state, full_tile=True) is True
    for recurrent in (
        _Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.bfloat16),
        _Fake((4, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32),
        _Fake((1, HEADS, HEAD_DIM), ttnn.float32),
    ):
        assert module.admits(gdn, _Fake(), rows_state, SimpleNamespace(recurrent=recurrent)) is False
    assert module.admits(gdn, _Fake(), rows_state, object()) is False
    assert module.admits(gdn, _Fake(), _RowsState(), state) is False


# ------------------------------------------------------------------------------- the body and the two agreements


def test_the_wrapped_body_is_the_nine_programs_in_order():
    """The body between the gather and the result: the projection's linear half, the qkv landing, the pre program,
    the two prims, the cast, the norm, and the chain's own out-projection."""

    calls = _calls(module.rows_body_wrap)
    programs = [
        c
        for c in calls
        if c
        in (
            "gdn._project_rows_linear",
            "gdn._land_rows_qkv",
            "pre.run",
            "gdn._chunk_rows",
            "post.post_cast",
            "post.post_norm",
            "gdn._out_proj_tile",
        )
    ]
    assert programs == [
        "gdn._project_rows_linear",
        "gdn._land_rows_qkv",
        "pre.run",
        # the two prims: _chunk_rows dispatches back into ``chunk`` here (the rows state carries the buffers), so the
        # forward pass and the commit's masked re-run take the same spelling and the same validation
        "gdn._chunk_rows",
        "post.post_cast",
        "post.post_norm",
        "gdn._out_proj_tile",
    ]
    assert _calls(module.chunk).count("ttnn.prim.chunk_gdn_prep") == 1
    assert _calls(module.chunk).count("ttnn.prim.chunk_gdn_scan") == 1
    # the body allocates only what does not outlive it; the prim layouts are the attached buffers
    assert inspect.getsource(module.rows_body_wrap).count("fp.allocate(") == 2  # o16 and the gated tile
    assert "buffers.q_c" in inspect.getsource(module.chunk) and "rows_state.v" in inspect.getsource(module.chunk)
    # the row count reaches both programs: the pre writers zero past it, the post writer zeroes the gated tail
    body = inspect.getsource(module.rows_body_wrap)
    assert "rows = rows_state.constants.rows" in body and body.count("rows=rows") == 2


def _prim_keywords(function, name: str) -> dict[str, str]:
    tree = ast.parse(inspect.getsource(function))
    call = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == f"ttnn.prim.{name}"
    )
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg is not None}


def _prim_call(function, name: str) -> ast.Call:
    tree = ast.parse(inspect.getsource(function))
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == f"ttnn.prim.{name}"
    )


def _prim_positional(function, name: str) -> list[str]:
    return [ast.unparse(a) for a in _prim_call(function, name).args]


def test_the_prim_call_is_spelled_as_the_prefill_slabs_form_spells_it():
    """One spelling of the two prims across the lane: the prefill slab's ``gdn_prefill_rows.chunk_prims`` (proven on
    one die) and this form pass the same keywords with the same values, q / k / v / g / beta positional in that order
    (g BEFORE beta), and to the scan the seven hand-offs then the state view, positional.  ``scale`` is a compile-time
    argument of the prep kernel, so both pass the composite's ``128 ** -0.5`` although ``qk_norm`` is off."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_prefill_rows as slab

    keywords = {"eye", "tril", "ones", "masks", "chunk_size", "scale", "v_flat", "HV", "qk_flat", "Hk", "qk_norm"}
    for function, names in ((module.chunk, module), (prims.chunk_rows_prims, prims), (slab.chunk_prims, slab)):
        prep = _prim_keywords(function, "chunk_gdn_prep")
        assert set(prep) == keywords, function.__module__
        assert (prep["v_flat"], prep["qk_flat"], prep["qk_norm"]) == ("True", "False", "False")
        assert getattr(names, prep["chunk_size"]) == 32
        assert getattr(names, prep["HV"]) == getattr(names, prep["Hk"]) == 12
        assert getattr(names, prep["scale"]) == 128**-0.5
        positional = _prim_positional(function, "chunk_gdn_prep")
        assert len(positional) == 5 and positional[3].endswith("g_c") and positional[4].endswith("beta_c"), positional
        scan = _prim_positional(function, "chunk_gdn_scan")
        assert len(scan) == 2 and scan[0] == "*prep", scan
        assert _prim_keywords(function, "chunk_gdn_scan") == {
            "chunk_size": prep["chunk_size"],
            "output_final_state": "True",
        }
    # the state view: theirs inline, ours bound to ``s0`` two lines up -- the same reshape either way
    assert (
        _prim_positional(slab.chunk_prims, "chunk_gdn_scan")[1]
        == "ttnn.reshape(initial_state, (HEADS, HEAD_DIM, HEAD_DIM))"
    )
    for function in (module.chunk, prims.chunk_rows_prims):
        assert _prim_positional(function, "chunk_gdn_scan")[1] == "s0"
        assert "s0 = ttnn.reshape(initial_state, (HEADS, HEAD_DIM, HEAD_DIM))" in inspect.getsource(function)


def test_the_slab_and_verify_admissions_never_both_admit():
    """The prefill slab's form (``gdn_prefill_rows``: slab rows, ``fused_prefill``) and this one (the 32-row verify
    tile with the wrap's buffers) dispatch from the same ``forward_rows``; their contracts are disjoint by row count,
    so no rows state can be admitted by both."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_prefill_rows as slab

    state = SimpleNamespace(recurrent=_Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32))
    verify = _RowsState()
    verify.wrap_buffers = object()
    assert module.admits(None, _Fake(), verify, state) is True
    assert slab.admits(None, _Fake(), verify, state.recurrent) is False  # 32 rows is not a slab
    for rows in (256, 2048, 4096):
        slab_state = _RowsState(rows=rows, tile_rows=rows)
        slab_state.fused_prefill = True
        assert module.qualifies(slab_state) is False and module.admits(None, _Fake(), slab_state, state) is False
        assert slab.is_slab_rows(rows) and not slab.is_slab_rows(TILE)


def test_the_prim_call_is_spelled_as_the_die_proved_it():
    """``gdn_rows_prims_direct`` was proved bitwise against the composite on the line; this form must hand the prims
    the same keywords, so the only difference between the two is which programs produced the layouts."""

    mine = _prim_keywords(module.chunk, "chunk_gdn_prep")
    theirs = _prim_keywords(prims.chunk_rows_prims, "chunk_gdn_prep")
    assert set(mine) == set(theirs)
    for key in ("chunk_size", "scale", "v_flat", "HV", "qk_flat", "Hk", "qk_norm"):
        assert mine[key] == theirs[key], key
    for key in ("eye", "tril", "ones", "masks"):
        assert mine[key] == theirs[key] == f"constants.{key}", key
    scan_mine, scan_theirs = _prim_keywords(module.chunk, "chunk_gdn_scan"), _prim_keywords(
        prims.chunk_rows_prims, "chunk_gdn_scan"
    )
    assert scan_mine == scan_theirs == {"chunk_size": "TILE", "output_final_state": "True"}
    for function in (module.chunk, prims.chunk_rows_prims):  # the seven hand-offs, then the state view, positional
        assert _prim_positional(function, "chunk_gdn_scan") == ["*prep", "s0"]
        assert "s0 = ttnn.reshape(initial_state, (HEADS, HEAD_DIM, HEAD_DIM))" in inspect.getsource(function)
    # and the values those names carry are the composite's own
    assert module.SCALE == 128**-0.5 and module.TILE == 32 and module.HEADS == 12


def test_the_two_programs_are_called_as_their_device_tests_call_them():
    """The pre program takes its six outputs positionally after the three constant tensors and a ``rows`` keyword;
    the post pair is ``post_cast`` into ``o16`` then ``post_norm`` with the history off (the verify keeps no history
    tile: ``commit_rows`` advances it from the window)."""

    assert list(inspect.signature(pre.run).parameters)[:12] == [
        "projected",
        "history",
        "taps",
        "constants",
        "selects",
        "scalars",
        "q_c",
        "k_c",
        "v",
        "beta_c",
        "g_c",
        "sig",
    ]
    tree = ast.parse(inspect.getsource(module.rows_body_wrap))
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pre.run")
    assert [ast.unparse(a) for a in run.args] == [
        "projected",
        "rows_state.history",
        "gdn.weights.conv_taps",
        "buffers.pre_constants",
        "buffers.pre_selects",
        "buffers.pre_scalars",
        "buffers.q_c",
        "buffers.k_c",
        "rows_state.v",  # the pre program writes the rows state's own v: the composite's flat-v form
        "buffers.beta_c",
        "buffers.g_c",
        "buffers.sig",
    ]
    assert {kw.arg for kw in run.keywords} == {"rows"}
    norm = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "post.post_norm")
    assert [ast.unparse(a) for a in norm.args] == [
        "o16",
        "buffers.sig",
        "gdn.weights.norm",
        "buffers.post_scalars",
        "projected",
        "gated",
        "None",
    ]
    assert {kw.arg: ast.unparse(kw.value) for kw in norm.keywords} == {"rows": "rows", "history": "False"}


# ------------------------------------------------------------------------------------------------ the commit


def test_the_commit_masks_the_prim_layouts_and_refuses_the_chain_anchors():
    source = open(gdn_module.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNGDN")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    chunk_rows = ast.get_source_segment(source, methods["_chunk_rows"])
    assert "fused.gdn_rows_wrap.buffers_of(rows_state)" in chunk_rows
    assert "None if committed_mask is None else committed_mask_c" in chunk_rows
    commit = ast.get_source_segment(source, methods["commit_rows"])
    assert "committed_mask_c=selectors.committed_mask_c" in commit
    assert "step_committed_rows or step_on_full_rejection" in commit
    assert "fused.gdn_rows_wrap.buffers_of(rows_state) is not None" in commit
    # the mask the commit multiplies with is built in the prims' layout, so both multiplies are element for element
    body = inspect.getsource(module.chunk)
    assert "ttnn.multiply(buffers.beta_c, committed_mask_c" in body
    assert "ttnn.multiply(buffers.g_c, committed_mask_c" in body
    selectors = ast.get_source_segment(
        source, next(n for n in tree.body if getattr(n, "name", "") == "build_rows_selectors")
    )
    assert "ttnn.le(constants.arange_c, accepted" in selectors
    assert gdn_module.Qwen38TTNNRowsSelectors.__dataclass_fields__["committed_mask_c"].default is None
    assert gdn_module.Qwen38TTNNGDNRowsConstants.__dataclass_fields__["arange_c"].default is None
    # the rows state carries the buffers and frees them with itself
    assert gdn_module.Qwen38TTNNGDNRowsState.__dataclass_fields__["wrap_buffers"].default is None
    deallocate = inspect.getsource(gdn_module.Qwen38TTNNGDNRowsState.deallocate)
    assert "self.wrap_buffers.deallocate()" in deallocate
