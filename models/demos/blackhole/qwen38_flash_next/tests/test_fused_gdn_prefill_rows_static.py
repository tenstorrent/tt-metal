# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_prefill_rows`` without a device: the registry entry and its switch, the admission predicate on host fakes,
the buffer layouts the rows state allocates under the form, the two prims call against the composite's own arguments
(a source pin on ``chunk_gated_delta_rule.cpp`` and on the binding), and the source pins that the wiring sits under a
slab condition while the 32-row, 128-row, lane and decode bodies keep their text."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_prefill_rows as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[3]
COMPOSITE = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp"
NANOBIND = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp"
PREP_KERNEL = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_prep.cpp"

NAME = "gdn_prefill_rows"
SLAB = 2048
HEADS, HEAD_DIM = module.HEADS, module.HEAD_DIM
GDN_SOURCE = inspect.getsource(gdn_module)
MODULE_SOURCE = inspect.getsource(module)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


# ------------------------------------------------------------------------------------------- the registry entry


def test_the_entry_is_bitwise_and_serves_by_default():
    entry = registry.kernel(NAME)
    assert entry.tolerance == registry.BITWISE
    # the one list decides: on by default after the line gate (identity chain vs kernel on the 4-chip line); a
    # QWEN38_FUSED_OFF=gdn_prefill_rows server runs the composed chain, and the admission keeps every non-slab call there
    assert entry.name in registry.DEFAULT_ON and entry.default_on
    assert registry.enabled(NAME, {}) is True
    assert registry.enabled(NAME, {registry.OFF_ENV: NAME}) is False
    assert registry.resolve(NAME, {registry.OFF_ENV: NAME}) is module.gdn_prefill_rows_composed
    assert entry.fused is module.gdn_prefill_rows
    assert entry.composed is module.gdn_prefill_rows_composed
    assert entry.admits is module.admits
    # the component gate's capture holds no GDN recurrent state: the proofs are the one-die microtests and the line
    assert entry.gate is None
    assert entry.component_proof is None
    assert "slab" in entry.replaces


def test_the_switch_picks_the_pair_by_default_and_the_chain_when_switched_off():
    # on by default (the one list), the production site resolving to the admitted step: the fused pair on a slab rows
    # state with the fused buffers, the composed chain on every other call
    for environ in ({}, {registry.ENV: NAME}):
        assert registry.enabled(NAME, environ)
        assert registry.resolve(NAME, environ) is module.gdn_prefill_rows
        admitted = registry.resolve_admitted(NAME, environ)
        assert isinstance(admitted, registry.AdmittedStep)
        assert admitted.fused is module.gdn_prefill_rows and admitted.composed is module.gdn_prefill_rows_composed
    # QWEN38_FUSED_OFF takes it off, even when QWEN38_FUSED names it too
    for environ in ({registry.OFF_ENV: NAME}, {registry.ENV: NAME, registry.OFF_ENV: NAME}):
        assert not registry.enabled(NAME, environ)
        assert registry.resolve(NAME, environ) is module.gdn_prefill_rows_composed
        assert registry.resolve_admitted(NAME, environ) is module.gdn_prefill_rows_composed


def test_the_name_is_reachable_from_the_package():
    from models.demos.blackhole.qwen38_flash_next.ttnn import fused

    assert fused.gdn_prefill_rows is module
    assert NAME in fused.kernels()


# ------------------------------------------------------------------------------------------- the buffer layouts


def test_the_buffer_layouts_are_the_prims_pages_and_the_pairs_hand_offs():
    layouts = module.buffer_layouts(SLAB)
    chunks = SLAB // module.CHUNK
    assert layouts["q"] == ((HEADS, chunks, module.CHUNK, HEAD_DIM), ttnn.bfloat16, 0)
    assert layouts["k"] == layouts["q"]
    assert layouts["beta"] == ((HEADS, chunks, module.CHUNK, 1), ttnn.float32, 0)
    assert layouts["g"] == layouts["beta"]
    assert layouts["sig"] == ((1, 1, SLAB, module.VALUE_WIDTH), ttnn.bfloat16, 3)
    assert layouts["gated"] == layouts["sig"]
    assert layouts["o16"] == ((HEADS, SLAB, HEAD_DIM), ttnn.bfloat16, 0)
    assert layouts["history_next"] == ((1, 1, module.TILE, module.QKV_WIDTH), ttnn.bfloat16, 3)
    assert set(layouts) == {"q", "k", "beta", "g", "sig", "o16", "gated", "history_next"}


def test_the_q_layout_is_what_the_pre_program_allocates_and_the_rows_state_takes():
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_pre_rows

    chunks = SLAB // module.CHUNK
    # the pre program's own q_c / beta_c shapes (allocate_outputs) and the rows state's
    assert module.buffer_layouts(SLAB)["q"][0] == (gdn_pre_rows.HEADS, chunks, gdn_pre_rows.TILE, gdn_pre_rows.HEAD_DIM)
    assert gdn_module.rows_qk_layout(SLAB, False, True) == ((HEADS, chunks, module.CHUNK, HEAD_DIM), 0)
    # the other two forms are untouched
    assert gdn_module.rows_qk_layout(SLAB, False) == ((1, SLAB, HEADS, HEAD_DIM), 2)
    assert gdn_module.rows_qk_layout(SLAB, True) == ((1, 1, SLAB, gdn_module.QK_WIDTH_PER_DEVICE), 3)
    assert gdn_module.rows_qk_layout(128, False) == ((1, 128, HEADS, HEAD_DIM), 2)


def test_the_extra_field_names_are_the_states_fused_fields():
    assert gdn_module._FUSED_ROWS_BUFFERS == ("sig", "o16", "gated", "history_next")
    fields = {field.name for field in dataclasses.fields(gdn_module.Qwen38TTNNGDNRowsState)}
    assert set(gdn_module._FUSED_ROWS_BUFFERS) | {"fused_prefill"} <= fields


def test_the_slab_row_predicate_matches_the_models():
    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import is_slab_rows

    for rows in (1, 5, 32, 128, 255, 129, 4224, "2048", True, None):
        assert module.is_slab_rows(rows) == is_slab_rows(rows), rows
    for rows in (256, 384, 2048, 4096):
        assert module.is_slab_rows(rows) and is_slab_rows(rows), rows


# --------------------------------------------------------------------------------------- the admission predicate


class _Fake:
    """A host tensor stand-in: shape and dtype are all ``admits`` reads."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


def _constants(rows: int, *, tiles: bool = True):
    chunk_tile = _Fake((1, 1, 32, 32), ttnn.float32) if tiles else None
    return SimpleNamespace(
        rows=rows,
        tile_rows=gdn_module.rows_tile_count(rows),
        eye=chunk_tile,
        tril=chunk_tile,
        ones=chunk_tile,
        masks=_Fake((1, 1, 32, 96), ttnn.float32) if tiles else None,
    )


def _rows_state(rows: int, *, fused_prefill: bool, flat_qk: bool = False, tiles: bool = True, drop: str = ""):
    constants = _constants(rows, tiles=tiles)
    tile_rows = constants.tile_rows
    state = SimpleNamespace(constants=constants, flat_qk=flat_qk, fused_prefill=fused_prefill)
    if fused_prefill:
        layouts = module.buffer_layouts(tile_rows)
    else:
        shape, _dim = gdn_module.rows_qk_layout(tile_rows, flat_qk)
        layouts = {
            "q": (shape, ttnn.bfloat16, 0),
            "k": (shape, ttnn.bfloat16, 0),
            "beta": ((1, 1, tile_rows, HEADS), ttnn.float32, 3),
            "g": ((1, 1, tile_rows, HEADS), ttnn.float32, 3),
        }
    for name, (shape, dtype, _shard) in layouts.items():
        setattr(state, name, None if name == drop else _Fake(shape, dtype))
    state.v = _Fake((1, 1, tile_rows, module.VALUE_WIDTH), ttnn.bfloat16)
    return state


def _initial_state():
    return _Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32)


def test_admits_a_slab_rows_state_that_carries_the_fused_buffers():
    assert module.admits(None, None, _rows_state(SLAB, fused_prefill=True), _initial_state())
    assert module.admits(None, None, _rows_state(256, fused_prefill=True), _initial_state())


@pytest.mark.parametrize(
    "state, why",
    [
        (_rows_state(SLAB, fused_prefill=False), "a slab rows state without the fused buffers"),
        (_rows_state(SLAB, fused_prefill=False, flat_qk=True), "the gdn_qk_flat slab form"),
        (_rows_state(SLAB, fused_prefill=True, tiles=False), "the chunk constant tiles missing"),
        (_rows_state(SLAB, fused_prefill=True, drop="sig"), "a hand-off buffer missing"),
        (_rows_state(SLAB, fused_prefill=True, drop="history_next"), "the history tile missing"),
        (_rows_state(32, fused_prefill=False), "the 32-row body"),
        (_rows_state(128, fused_prefill=False), "the 128-row body"),
    ],
)
def test_the_chain_serves_everything_outside_the_contract(state, why):
    assert not module.admits(None, None, state, _initial_state()), why


def test_the_contract_reads_the_state_and_the_initial_state():
    good = _rows_state(SLAB, fused_prefill=True)
    assert not module.admits(None, None, good, None)
    assert not module.admits(None, None, good, _Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.bfloat16))
    assert not module.admits(None, None, good, _Fake((HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32))
    assert not module.admits(None, None, good, _initial_state(), full_tile=True)  # a 32-row option
    assert not module.admits(None, None, SimpleNamespace(), _initial_state())  # a fake with no rows state at all


def test_a_wrong_buffer_shape_is_refused():
    state = _rows_state(SLAB, fused_prefill=True)
    state.q = _Fake((1, SLAB, HEADS, HEAD_DIM), ttnn.bfloat16)  # the chain's padded token-major q
    assert not module.admits(None, None, state, _initial_state())
    state = _rows_state(SLAB, fused_prefill=True)
    state.beta = _Fake((1, 1, SLAB, HEADS), ttnn.float32)  # the chain's token-major gate column
    assert not module.admits(None, None, state, _initial_state())


# ------------------------------------------------------------------------------------------- the two prims call


def _composite_prep_arguments() -> list[str]:
    """The positional arguments ``chunk_gated_delta_rule.cpp`` hands ``ttnn::prim::chunk_gdn_prep``."""

    source = (REPO_ROOT / COMPOSITE).read_text(encoding="utf-8")
    call = source[source.index("ttnn::prim::chunk_gdn_prep(") :]
    body = call[call.index("(") + 1 : call.index(");")]
    return [argument.strip() for argument in body.split(",")]


def _binding_arguments(prim: str) -> list[str]:
    """The argument names the nanobind declares for one prim, in order."""

    source = (REPO_ROOT / NANOBIND).read_text(encoding="utf-8")
    block = source[source.index(f'ttnn::bind_function<"{prim}", "ttnn.prim.">') :]
    block = block[: block.index(");") + 2]
    return re.findall(r'nb::arg\("([a-zA-Z_]+)"\)', block)


def test_the_prims_call_names_every_argument_the_binding_declares():
    call = MODULE_SOURCE[MODULE_SOURCE.index("ttnn.prim.chunk_gdn_prep(") :]
    call = call[: call.index("\n    )")]
    named = set(re.findall(r"\n\s+([a-zA-Z_]+)=", call))
    positional = {"q", "k", "v", "g", "beta"}
    declared = set(_binding_arguments("chunk_gdn_prep"))
    # every keyword argument of the binding is passed explicitly except the two the composite leaves unset
    assert declared - positional - named == {"memory_config", "compute_kernel_config"}
    assert named <= declared


def test_the_prep_flags_are_what_the_composite_passes_for_the_rows_shapes():
    """B = 1, T a multiple of C, HV = H = 12 (q/k GQA-expanded before the kernel), K = V = 128, v flat token-major:
    the composite's ``flat_v`` true, ``flat_qk`` false, ``qk_norm = flat_qk && C == 32`` false, ``HV`` and ``H``
    both 12, ``C`` the caller's chunk_size and ``scale`` the caller's or ``K ** -0.5``."""

    arguments = _composite_prep_arguments()
    assert arguments[-6:] == ["flat_v", "HV", "qk_norm", "scale", "flat_qk", "H"]
    call = _squash(MODULE_SOURCE[MODULE_SOURCE.index("ttnn.prim.chunk_gdn_prep(") :])
    for pin in (
        "chunk_size=CHUNK,",
        "scale=QK_SCALE,",
        "v_flat=True,",
        "HV=HEADS,",
        "qk_flat=False,",
        "Hk=HEADS,",
        "qk_norm=False,",
    ):
        assert _squash(pin) in call, pin
    assert module.CHUNK == 32 and module.HEADS == 12
    assert module.QK_SCALE == HEAD_DIM**-0.5


def test_the_prep_kernel_reads_scale_only_under_the_in_kernel_norm():
    """Why passing the composite's ``scale`` beside q that is already scaled twice is right: ``SCALE_BITS`` is read
    inside ``if constexpr (QK_NORM)`` only, so with ``qk_norm=False`` it multiplies nothing -- it is a compile-time
    argument of the prep kernel, and passing the composite's value keeps the program the composite's."""

    source = (REPO_ROOT / PREP_KERNEL).read_text(encoding="utf-8")
    assert "constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);" in source
    guarded = source[source.index("if constexpr (QK_NORM)") :]
    before = source[: source.index("if constexpr (QK_NORM)")]
    assert "SCALE_BITS" in guarded
    assert before.count("SCALE_BITS") == 1  # the declaration only


def test_the_scan_takes_the_prep_outputs_and_the_reshaped_initial_state():
    call = _squash(MODULE_SOURCE[MODULE_SOURCE.index("ttnn.prim.chunk_gdn_scan(") :])
    assert _squash("*prep,") in call  # the seven hand-off tensors, in the order the prep returned them
    assert _squash("ttnn.reshape(initial_state, (HEADS, HEAD_DIM, HEAD_DIM)),") in call
    assert _squash("chunk_size=CHUNK,") in call and _squash("output_final_state=True,") in call
    # and the composite's own folds of the two outputs
    body = MODULE_SOURCE[MODULE_SOURCE.index("def chunk_prims") :]
    assert _squash("ttnn.reshape(scan[0], (HEADS, rows_total, HEAD_DIM))") in _squash(body)
    assert _squash("ttnn.reshape(scan[1], (1, HEADS, HEAD_DIM, HEAD_DIM))") in _squash(body)


def test_the_prep_hand_off_tensors_are_freed():
    body = MODULE_SOURCE[MODULE_SOURCE.index("def chunk_prims") : MODULE_SOURCE.index("def slab_body")]
    assert "for tensor in prep:" in body and "ttnn.deallocate(tensor)" in body


# ------------------------------------------------------------------------------------------ the gdn.py wiring


def _method(name: str) -> str:
    return inspect.getsource(getattr(gdn_module.Qwen38TTNNGDN, name))


def test_the_fused_branch_sits_under_a_slab_condition():
    source = _method("forward_rows")
    slab = source.index("if is_slab_rows(rows_state.constants.tile_rows):")
    call = source.index("body = self._gdn_prefill_rows()")
    rows_body = source.index("body = self._rows_body()")
    assert slab < call < rows_body, "the fused body must be reached only through the slab condition"
    # the branch returns before the 32-row / 128-row dispatch (``_rows_body``: the verify-rows wrap, or the chain's
    # five calls, which keep their text verbatim in the registry's composed body for those row counts)
    assert _squash("return Qwen38TTNNGDNRowsResult(output, final_state, state, rows_state)") in _squash(
        source[call:rows_body]
    )
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap

    assert (
        _squash(
            """
    z, a, b = gdn._project_rows(full_hidden, rows_state)
    conv = gdn._causal_conv_rows(rows_state)
    gdn._make_chunk_inputs(conv, a, b, rows_state)
    recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=state.recurrent)
    output = gdn._gate_and_project_rows(recurrent_output, z, full_hidden, rows_state, full_tile=full_tile)
        """
        )
        in _squash(inspect.getsource(gdn_rows_wrap.rows_body_composed))
    )


def test_the_composed_callable_is_the_chains_four_calls_in_order():
    """The kernel off must run exactly what ``forward_rows`` ran before: the composed callable's body is the same
    four calls with the same arguments."""

    composed = _squash(inspect.getsource(module.gdn_prefill_rows_composed))
    for line in (
        "z, a, b = gdn._project_rows(full_hidden, rows_state)",
        "conv = gdn._causal_conv_rows(rows_state)",
        "gdn._make_chunk_inputs(conv, a, b, rows_state)",
        "recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=initial_state)",
        "output = gdn._gate_and_project_rows(recurrent_output, z, full_hidden, rows_state, full_tile=full_tile)",
    ):
        assert _squash(line) in composed, line


def test_the_other_bodies_keep_their_text():
    """The 32-row and long-chunk projections, the 32-row and long-chunk gates, the lane rows body and the decode
    step are not touched by this form: their distinguishing lines are still there, verbatim."""

    pins = {
        "_project_rows_linear": (  # the projection's linear half; _project_rows lands qkv and slices z / a / b from it
            "projected_ws = ttnn.linear(",
            "program_config=self.in_proj_program_config,",
            "for tile in dram_sharded_row_tiles(full_hidden, self.in_proj_act_memory_config):",
        ),
        "_gate_and_project_rows": (
            "normalized = ttnn.experimental.view(normalized_heads, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))",
            "output = self._out_proj_tile(gated, full_hidden)",
            "for start in range(0, tile_rows, CHUNK_SIZE):",
        ),
        "forward_decode": (
            "step = self._gdn_step()",
            "gated = step(self, projected, window, state)",
            "state.advance_conv_window()",
        ),
        "_make_chunk_inputs": (
            "expanded = ttnn.matmul(",
            "normed = ttnn.rms_norm(heads_tensor, epsilon=QK_L2_NORM_EPS / HEAD_DIM)",
            "landed = ttnn.multiply(v_slice, constants.row_mask_bf16_col, output_tensor=rows_state.v)",
        ),
        "_causal_conv_rows": (
            "conv = ttnn.multiply(pieces[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)",
            "conv = ttnn.mac(pieces[index], self.weights.conv_taps[index], previous)",
            "conv = ttnn.silu(conv, memory_config=ttnn.L1_MEMORY_CONFIG)",
        ),
    }
    for name, lines in pins.items():
        source = _squash(_method(name))
        for line in lines:
            assert _squash(line) in source, f"{name}: {line}"
    # the lane rows body and the 1-row step never mention the form
    for name in ("forward_rows_lanes", "_step_row_state", "forward_decode", "forward_prefill"):
        assert "gdn_prefill_rows" not in _method(name), name
        assert "fused_prefill" not in _method(name), name


def test_the_two_dense_linears_are_the_slab_branches_own():
    """The fused body calls the same two helpers the chain's slab branch calls, so the projection and the
    out-projection (and their program configs) cannot drift between the arms."""

    projection = _squash(_method("_slab_projection"))
    assert _squash("self._slab_program_config(tile_rows, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE)") in projection
    assert _squash('resident_weight=self.prefill_dense.resident("qkvzab")') in projection
    assert _squash("projected = self._slab_projection(full_hidden, rows_state)") in _squash(
        _method("_project_rows_linear")  # the branch moved with the linear half; _project_rows calls that half
    )

    tail = _squash(_method("_slab_out_projection"))
    assert _squash("self._slab_program_config(tile_rows, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE)") in tail
    assert _squash('resident_weight=self.prefill_dense.resident("out")') in tail
    assert _squash("output = ttnn.reduce_scatter(") in tail
    assert _squash('_copy_inplace(output, rows_state.output, label="GDN slab output")') in tail
    assert _squash("output = self._slab_out_projection(gated, full_hidden, rows_state)") in _squash(
        _method("_gate_and_project_rows")
    )
    body = _squash(inspect.getsource(module.gdn_prefill_rows))
    assert _squash("gdn._slab_projection(full_hidden, rows_state)") in body
    assert _squash("gdn._slab_out_projection(gated, full_hidden, rows_state)") in body


def test_the_commit_takes_the_epilogues_history_tile_under_the_form():
    source = _method("commit_rows_full")
    fused_at = source.index("if rows_state.fused_prefill:")
    slab_at = source.index("elif is_slab_rows(rows_state.constants.tile_rows):")
    assert fused_at < slab_at
    assert _squash('_copy_inplace(rows_state.history_next, rows_state.history, label="GDN slab history")') in _squash(
        source[fused_at:slab_at]
    )
    # the row-shift branch the slab used before, and the 32/128-row select, are both still there
    assert _squash("last_rm = ttnn.to_layout(last_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)") in _squash(source)
    assert _squash("rows_state.constants.history_select_full,") in _squash(source)


def test_the_form_is_resolved_once_and_kept():
    init = _squash(inspect.getsource(gdn_module.Qwen38TTNNGDN.__init__))
    assert _squash('self._prefill_rows = fused.resolve_admitted("gdn_prefill_rows")') in init
    assert _squash('self._prefill_rows_on = fused.enabled("gdn_prefill_rows")') in init
    accessor = _squash(_method("_gdn_prefill_rows"))
    assert _squash('self._prefill_rows = fused.resolve_admitted("gdn_prefill_rows")') in accessor


def test_the_buffers_follow_the_switch_and_exclude_the_flat_form():
    allocate = _squash(_method("allocate_rows_state"))
    assert _squash("fused_prefill = is_slab_rows(constants.rows) and self._fused_prefill_rows_buffers()") in allocate
    # the flat form's own pin (tests/test_ttnn_prefill_glue_no_device.py) keeps its text; the fused form excludes it
    assert (
        _squash('flat_qk = is_slab_rows(constants.rows) and self.glue.enabled("gdn_qk_flat") and not fused_prefill')
        in allocate
    )
    allocator = _squash(inspect.getsource(gdn_module.Qwen38TTNNGDNRowsState.allocate))
    assert _squash("gdn_qk_flat and gdn_prefill_rows are exclusive slab forms") in allocator
    assert _squash("the fused prefill rows buffers are a slab option (gdn_prefill_rows)") in allocator
    assert _squash("a shared GDN rows body and its layer disagree on the fused prefill rows form") in allocator


def test_the_note_names_the_switch_and_the_class():
    text = (MODEL_DIR / "docs/PREFILL.md").read_text(encoding="utf-8")
    assert "QWEN38_FUSED=gdn_prefill_rows" in text
    assert "gdn_pre_rows" in text and "gdn_post_rows" in text
    assert "ttnn.prim.chunk_gdn_prep" in text and "ttnn.prim.chunk_gdn_scan" in text
    # the numbers of record on one p150 die at 2048 rows: pre 332, post pair 115, the three together 447.6, the
    # chain's 3,270 us per layer (a retune moves the first two; the pair sum and the chain's figure stay pinned)
    for number in ("332", "115", "447", "3,270"):
        assert number in text, number


def test_every_buffer_a_program_writes_is_restamped_with_its_declared_topology():
    """The four-die line's placement seam: ttnn.generic_op leaves an output with the allocation's default topology
    (shard dim 0), so slab_body must give every buffer a program wrote its declared topology back before the rows
    state is validated again -- the pre program's six, post_cast's one, post_norm's two -- from buffer_layouts (v on
    the chain's dim 3)."""

    written = set(module.PRE_WRITES) | set(module.CAST_WRITES) | set(module.NORM_WRITES)
    assert written == {"q", "k", "v", "beta", "g", "sig", "o16", "gated", "history_next"}
    layouts = module.buffer_layouts(64)
    assert {name: layouts[name][2] for name in written - {"v"}} == {
        "q": 0,
        "k": 0,
        "beta": 0,
        "g": 0,
        "sig": 3,
        "o16": 0,
        "gated": 3,
        "history_next": 3,
    }
    source = inspect.getsource(module.slab_body)
    calls = [m.group(1) for m in re.finditer(r"restamp_written\(buffers, projected, ([A-Z_]+)\)", source)]
    assert calls == ["PRE_WRITES", "CAST_WRITES", "NORM_WRITES"]
    assert source.index("gdn_pre_rows.run(") < source.index("PRE_WRITES)") < source.index("chunk_prims(")
    assert source.index("post_cast(") < source.index("CAST_WRITES)") < source.index("post_norm(")
    assert (
        source.index("post_norm(")
        < source.index("NORM_WRITES)")
        < source.index("return gated, final_state, history_next")
    )
    helper = inspect.getsource(module.restamp_written)
    assert 'dim = 3 if name == "v" else layouts[name][2]' in helper and "fp.stamp_topology(" in helper
