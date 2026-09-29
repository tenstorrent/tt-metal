# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HEAD/TAIL split of the position-generic body (refresh diet B-5b), no device.

Pins the split point (layer 0 | 1: the first TAIL layer is the PLE consumer),
the handoff (one retained residual whose metadata is asserted actual vs
expected), that HEAD reads neither the PLE row nor the position while TAIL
consumes both, that the fused body is HEAD then TAIL with the handoff consumed
by layer 1 as before, the layer's ``release_input`` keyword, and the capture
API that records one residue class as one single-body trace or a HEAD and a
TAIL trace keyed ``(part, residue, regime)``.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.test_ttnn_components_static import _generic_layer, _generic_model
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    PLE_CHECKPOINT_LAYER,
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNDecoderLayerGenericState,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    EXPECTED_LAYER_PATTERN,
    GENERIC_HEAD_HANDOFF,
    GENERIC_HEAD_LAYERS,
    GENERIC_TRACE_PARTS_SINGLE,
    GENERIC_TRACE_PARTS_SPLIT,
    Qwen38TTNNGenericDecodeCapture,
    Qwen38TTNNGenericDecodeOutput,
    Qwen38TTNNGenericHead,
    Qwen38TTNNGenericTraceKey,
    Qwen38TTNNTextModel,
    validate_model_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38TTNNPLEResult

TAIL_EPILOGUE_NAMES = [
    "qsa-position-deallocate",
    "rope-deallocate",
    "final-mixer",
    "deallocate",
    "lm-head",
    "deallocate",
    "advance",
]


def _comparable(log: list) -> list:
    """The fake model's log with its per-run RoPE/position objects reduced to their type names."""

    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (str, int, bool, tuple, type(None))):
            return value
        return type(value).__name__

    return [tuple(plain(value) for value in entry) for entry in log]


def test_split_point_is_the_ple_checkpoint_layer_and_head_is_gdn_only() -> None:
    validate_model_static_contract()
    assert GENERIC_HEAD_LAYERS == 1 == PLE_CHECKPOINT_LAYER
    assert EXPECTED_LAYER_PATTERN[:GENERIC_HEAD_LAYERS] == ("linear_attention",)
    assert EXPECTED_LAYER_PATTERN[GENERIC_HEAD_LAYERS] == "linear_attention"  # layer 1 is the GDN PLE layer
    assert (GENERIC_TRACE_PARTS_SINGLE, GENERIC_TRACE_PARTS_SPLIT) == (("body",), ("head", "tail"))
    assert Qwen38TTNNGenericTraceKey("tail", 3) == Qwen38TTNNGenericTraceKey(part="tail", residue=3, regime=0)
    assert len({Qwen38TTNNGenericTraceKey(part, residue) for part in ("head", "tail") for residue in range(4)}) == 8


def test_handoff_is_the_layer_0_residual_only_with_pinned_metadata() -> None:
    assert [handoff.name for handoff in GENERIC_HEAD_HANDOFF] == ["layer_0_output_residual"]
    (handoff,) = GENERIC_HEAD_HANDOFF
    assert handoff.attribute == "residual" and hasattr(Qwen38TTNNGenericHead("x", object()), handoff.attribute)
    assert (handoff.shape, handoff.padded_shape) == ((1, 4, 1, 640), (1, 4, 32, 640))
    assert (handoff.dtype, handoff.layout, handoff.memory_config) == (
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    assert handoff.describe() == (
        f"shape=[1, 4, 1, 640] padded_shape=[1, 4, 32, 640] dtype={ttnn.bfloat16} layout={ttnn.TILE_LAYOUT} "
        f"memory_config={ttnn.DRAM_MEMORY_CONFIG}"
    )


class _FakeResidual:
    def __init__(self, *, dtype=ttnn.bfloat16, padded_rows: int = 32) -> None:
        self.shape = (1, 4, 1, 640)
        self.padded_shape = (1, 4, padded_rows, 640)
        self.dtype = dtype
        self.layout = ttnn.TILE_LAYOUT

    def memory_config(self):
        return ttnn.DRAM_MEMORY_CONFIG


def _owner_with_real_head_validation(monkeypatch, log: list):
    owner, state, prepared = _generic_model(monkeypatch, log)
    owner._validate_generic_head = Qwen38TTNNTextModel._validate_generic_head.__get__(owner)
    placements = []
    owner.mesh_contract = SimpleNamespace(
        validate_tensor=lambda tensor, *, placement, shard_dim: placements.append((tensor, placement, shard_dim))
    )
    return owner, state, prepared, placements


def test_handoff_validation_reports_actual_vs_expected_metadata(expect_error, monkeypatch) -> None:
    log: list = []
    owner, _, _, placements = _owner_with_real_head_validation(monkeypatch, log)
    good = _FakeResidual()
    owner._validate_generic_head(Qwen38TTNNGenericHead(good, owner._state_owner))
    assert placements == [(good, model_module.TensorPlacement.HIDDEN_SHARDED, 3)]

    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        RuntimeError, match="handoff layer_0_output_residual: actual .* vs expected "
    ) as info:  # allow-pytest.raises: inspect the captured exception object
        owner._validate_generic_head(Qwen38TTNNGenericHead(_FakeResidual(dtype=ttnn.float32), owner._state_owner))
    assert f"dtype={ttnn.float32}" in str(info.value) and f"dtype={ttnn.bfloat16}" in str(info.value)
    with expect_error(RuntimeError, match="actual shape=\\[1, 4, 1, 640\\] padded_shape=\\[1, 4, 64, 640\\]"):
        owner._validate_generic_head(Qwen38TTNNGenericHead(_FakeResidual(padded_rows=64), owner._state_owner))
    with expect_error(ValueError, match="not produced by this model owner"):
        owner._validate_generic_head(Qwen38TTNNGenericHead(good, object()))
    released = Qwen38TTNNGenericHead(good, owner._state_owner)
    deallocated = []
    monkeypatch.setattr(model_module.ttnn, "deallocate", deallocated.append)
    released.release_tensors()
    assert deallocated == [good] and not released.active
    with expect_error(RuntimeError, match="already released"):
        owner._validate_generic_head(released)
    with expect_error(RuntimeError, match="already released"):
        released.release_tensors()
    assert len(placements) == 1


def test_head_embeds_and_runs_layer_0_without_the_ple_row_or_the_position(monkeypatch) -> None:
    log: list = []
    owner, state, prepared = _generic_model(monkeypatch, log)

    head = owner.forward_decode_generic_head(prepared, state)

    assert isinstance(head, Qwen38TTNNGenericHead) and head.active and head.residual == "residual-0"
    assert head._owner is owner._state_owner
    assert log == [
        ("embed", "token-row"),
        ("forward", 0, "residual-in", {"prepared_ple": None, "rope": None, "qsa_position": None}),
        ("validate-head", "residual-0", True),
    ]
    source = inspect.getsource(Qwen38TTNNTextModel.forward_decode_generic_head)
    for forbidden in (
        "prepared.ple",
        "state.position",
        "rope_table",
        "derive_qsa_position_inputs",
        "advance()",
        "final_mixer",
        "lm_head",
        "release_input",
        "PLE_CHECKPOINT_LAYER",
    ):
        assert forbidden not in source, forbidden
    assert "range(GENERIC_HEAD_LAYERS)" in source and "prepared_ple=None, rope=None, qsa_position=None" in source
    assert 'self._mark_poisoned("forward_decode_generic_head"' in source


def test_tail_derives_the_position_consumes_the_ple_row_first_and_advances_last(monkeypatch) -> None:
    log: list = []
    owner, state, prepared = _generic_model(monkeypatch, log)
    head = owner.forward_decode_generic_head(prepared, state)
    log.clear()

    output = owner.forward_decode_generic_tail(head, prepared, state, release_head=False)

    assert isinstance(output, Qwen38TTNNGenericDecodeOutput) and output.logits.tensor == "logits-tensor"
    names = [entry[0] for entry in log]
    # The RoPE index rows are the device position's rows less its rotary shift (an image prompt's decode reads the
    # table at P - S); the cache, block and mask inputs stay on P.
    assert names[:7] == [
        "validate-head",
        "index_row",
        "rope_index_row",
        "rope_block_start_index_row",
        "rows",
        "deallocate",
        "derive",
    ]
    assert names[7:54] == ["forward"] * 47 and names[54:] == TAIL_EPILOGUE_NAMES
    forwards = [entry for entry in log if entry[0] == "forward"]
    assert [entry[1] for entry in forwards] == list(range(GENERIC_HEAD_LAYERS, 48))
    assert forwards[0][2] == "residual-0"
    assert forwards[0][3]["prepared_ple"] is prepared.ple and forwards[0][3]["release_input"] is False
    assert [entry[3]["prepared_ple"] is None for entry in forwards[1:]] == [True] * 46
    assert [entry[3]["release_input"] for entry in forwards[1:]] == [True] * 46
    assert head.active, "release_head=False retains the handoff for the traces"

    log.clear()
    head = owner.forward_decode_generic_head(prepared, state)
    owner.forward_decode_generic_tail(head, prepared, state)
    forwards = [entry for entry in log if entry[0] == "forward"]
    assert forwards[1][1] == 1 and forwards[1][3]["release_input"] is True
    assert not head.active, "the default consumes the handoff in layer 1, as the pre-split body did"

    source = inspect.getsource(Qwen38TTNNTextModel.forward_decode_generic_tail)
    assert "range(GENERIC_HEAD_LAYERS, BACKBONE_LAYERS)" in source
    assert "prepared_ple=prepared.ple if layer_index == PLE_CHECKPOINT_LAYER else None" in source
    assert "release_input=release_head or not first_tail_layer" in source
    assert source.count("state.position.advance()") == 1
    after_advance = source[source.index("state.position.advance()") + len("state.position.advance()") :]
    assert "ttnn." not in after_advance and "self." not in after_advance.split("except", 1)[0]
    assert "_embed_residual" not in source and "prepared.device_token" not in source


def test_fused_body_is_head_then_tail_with_the_handoff_consumed(monkeypatch) -> None:
    fused_log: list = []
    owner, state, prepared = _generic_model(monkeypatch, fused_log)
    fused = owner.forward_decode_generic(prepared, state)
    split_log: list = []
    owner, state, prepared = _generic_model(monkeypatch, split_log)
    head = owner.forward_decode_generic_head(prepared, state)
    split = owner.forward_decode_generic_tail(head, prepared, state, release_head=True)

    assert fused.logits.tensor == split.logits.tensor == "logits-tensor"
    assert _comparable(fused_log) == _comparable(split_log)
    assert not head.active
    assert inspect.signature(Qwen38TTNNTextModel.forward_decode_generic_tail).parameters["release_head"].default is True


def test_tail_refuses_a_foreign_or_released_head_before_touching_the_state(expect_error, monkeypatch) -> None:
    log: list = []
    owner, state, prepared, _ = _owner_with_real_head_validation(monkeypatch, log)
    with expect_error(ValueError, match="not produced by this model owner"):
        owner.forward_decode_generic_tail(Qwen38TTNNGenericHead(_FakeResidual(), object()), prepared, state)
    released = Qwen38TTNNGenericHead(_FakeResidual(), owner._state_owner)
    released.active = False
    with expect_error(RuntimeError, match="already released"):
        owner.forward_decode_generic_tail(released, prepared, state)
    assert log == [] and not owner.poisoned


def test_layer_release_input_keeps_the_ple_layers_input_and_is_refused_elsewhere(expect_error, monkeypatch) -> None:
    ops: list = []
    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    layer._validate_residual = lambda residual, label: None
    state = SimpleNamespace(ple="ple-state")
    layer.ple = SimpleNamespace(
        forward_prepared=lambda branch_rows, prepared, ple_state: Qwen38TTNNPLEResult("delta-rows", ple_state)
    )
    monkeypatch.setattr(layer_module.ttnn, "permute", lambda tensor, dims, memory_config: f"permuted({tensor})")
    monkeypatch.setattr(layer_module.ttnn, "add", lambda residual, delta, memory_config: f"add({residual},{delta})")
    monkeypatch.setattr(layer_module, "_deallocate_unique", lambda *tensors: ops.append(tensors))

    injected, _ = layer._apply_ple("residual", state, token_id=None, prepared_ple="row")
    assert injected == "add(residual,permuted(delta-rows))"
    assert ops == [("permuted(residual)", "delta-rows"), ("residual", "permuted(delta-rows)")]
    ops.clear()
    layer._apply_ple("residual", state, token_id=None, prepared_ple="row", release_input=False)
    assert ops == [("permuted(residual)", "delta-rows"), ("permuted(delta-rows)",)]

    layer.ple = None
    assert layer._apply_ple("residual", state, token_id=None) == ("residual", None)
    with expect_error(ValueError, match="only the PLE layer"):
        layer._apply_ple("residual", state, token_id=None, release_input=False)

    operations: list = []
    generic = _generic_layer(monkeypatch, layer_type=Qwen38TTNNLayerType.GDN, operations=operations, ple=True)
    generic.attention.forward_decode = lambda hidden, state: SimpleNamespace(hidden_sharded="gdn-hidden", state=state)
    generic_state = Qwen38TTNNDecoderLayerGenericState(
        Qwen38TTNNLayerNamespace.BACKBONE, 1, SimpleNamespace(layer_index=1), SimpleNamespace(token_context=None)
    )
    prepared_ple = SimpleNamespace(source_token_context=None)
    generic.forward_decode_generic(
        "residual-in", generic_state, prepared_ple=prepared_ple, rope=None, qsa_position=None, release_input=False
    )
    assert operations[1] == ("ple", None, prepared_ple, False)
    generic.forward_decode_generic(
        "residual-in", generic_state, prepared_ple=prepared_ple, rope=None, qsa_position=None
    )
    assert [op for op in operations if op[0] == "ple"][1] == ("ple", None, prepared_ple, True)


def _trace_runtime(monkeypatch, events: list):
    ids = iter(range(100, 200))
    monkeypatch.setattr(
        model_module.ttnn,
        "begin_trace_capture",
        lambda mesh, cq_id: events.append(("begin", mesh, cq_id)) or next(ids),
    )
    monkeypatch.setattr(
        model_module.ttnn, "end_trace_capture", lambda mesh, trace_id, cq_id: events.append(("end", trace_id, cq_id))
    )

    @contextmanager
    def scope(mesh):
        events.append(("scope-enter", mesh))
        yield
        events.append(("scope-exit", mesh))

    @contextmanager
    def guard(label: str):
        events.append(("guard-enter", label))
        attempts = [f"blocked-in:{label}"]
        yield attempts
        events.append(("guard-exit", label))

    monkeypatch.setattr(model_module, "corruptible_allocation_scope", scope)
    clock = iter(range(1000, 100000, 7))
    return guard, lambda: next(clock)


def test_capture_records_head_and_tail_traces_keyed_by_part_residue_regime(expect_error, monkeypatch) -> None:
    log: list = []
    owner, state, prepared = _generic_model(monkeypatch, log)
    events: list = []
    guard, clock_ns = _trace_runtime(monkeypatch, events)
    epilogues = []
    phases: list[str] = []

    def epilogue(output):
        epilogues.append(output)
        events.append(("epilogue", output.logits.tensor))
        return "candidates-and-token-row"

    capture = owner.capture_decode_generic(
        prepared,
        state,
        residue=2,
        split=True,
        guard=guard,
        epilogue=epilogue,
        phase_observer=phases.append,
        regime=1,
        cq_id=0,
        clock_ns=clock_ns,
    )

    assert isinstance(capture, Qwen38TTNNGenericDecodeCapture)
    assert (capture.residue, capture.regime, capture.parts) == (2, 1, ("head", "tail"))
    assert capture.trace_ids == {
        Qwen38TTNNGenericTraceKey("head", 2, 1): 100,
        Qwen38TTNNGenericTraceKey("tail", 2, 1): 101,
    }
    assert capture.head is not None and capture.head.active and capture.head.residual == "residual-0"
    assert capture.output is epilogues[0] and capture.epilogue == "candidates-and-token-row"
    assert capture.capture_ns == {"head": 7, "tail": 7} and capture.active
    assert capture.guard_attempts == (
        "blocked-in:generic head capture residue 2 regime 1",
        "blocked-in:generic tail capture residue 2 regime 1",
    )
    assert phases == ["before-head-capture", "after-head-capture", "before-tail-capture", "after-tail-capture"]
    assert events == [
        ("scope-enter", "mesh"),
        ("begin", "mesh", 0),
        ("guard-enter", "generic head capture residue 2 regime 1"),
        ("guard-exit", "generic head capture residue 2 regime 1"),
        ("end", 100, 0),
        ("scope-exit", "mesh"),
        ("scope-enter", "mesh"),
        ("begin", "mesh", 0),
        ("guard-enter", "generic tail capture residue 2 regime 1"),
        ("epilogue", "logits-tensor"),
        ("guard-exit", "generic tail capture residue 2 regime 1"),
        ("end", 101, 0),
        ("scope-exit", "mesh"),
    ]
    forwards = [entry for entry in log if entry[0] == "forward"]
    assert [entry[1] for entry in forwards] == list(range(48))
    assert forwards[1][3]["release_input"] is False, "TAIL's layer 1 keeps the retained handoff allocated"
    assert [entry[3]["release_input"] for entry in forwards[2:]] == [True] * 46
    assert log[-1] == ("advance",)

    released = []
    monkeypatch.setattr(model_module.ttnn, "deallocate", released.append)
    capture.release_tensors()
    assert released == ["logits-tensor", "residual-0"]
    assert not capture.active and not capture.head.active and not capture.output.active
    with expect_error(RuntimeError, match="already released"):
        capture.release_tensors()


def test_capture_single_body_is_the_fused_path_in_one_trace(expect_error, monkeypatch) -> None:
    log: list = []
    owner, state, prepared = _generic_model(monkeypatch, log)
    events: list = []
    guard, clock_ns = _trace_runtime(monkeypatch, events)

    capture = owner.capture_decode_generic(
        prepared, state, residue=3, split=False, guard=guard, epilogue=lambda output: None, clock_ns=clock_ns
    )

    assert (capture.parts, capture.head, capture.epilogue, capture.regime) == (("body",), None, None, 0)
    assert capture.trace_ids == {Qwen38TTNNGenericTraceKey("body", 3, 0): 100}
    assert capture.guard_attempts == ("blocked-in:generic body capture residue 3 regime 0",)
    assert [event[0] for event in events] == ["scope-enter", "begin", "guard-enter", "guard-exit", "end", "scope-exit"]
    fused_log: list = []
    fused_owner, fused_state, fused_prepared = _generic_model(monkeypatch, fused_log)
    fused_owner.forward_decode_generic(fused_prepared, fused_state)
    assert _comparable(log) == _comparable(fused_log), "split=False captures exactly the fused body"
    released = []
    monkeypatch.setattr(model_module.ttnn, "deallocate", released.append)
    capture.release_tensors()
    assert released == ["logits-tensor"] and not capture.active

    with expect_error(ValueError, match="non-negative ints"):
        owner.capture_decode_generic(prepared, state, residue=-1, split=False, guard=guard, epilogue=lambda o: None)
    with expect_error(TypeError, match="observer must be callable"):
        owner.capture_decode_generic(
            prepared, state, residue=0, split=False, guard=guard, epilogue=lambda o: None, phase_observer="x"
        )


def test_capture_api_signature_is_the_runner_contract() -> None:
    signature = inspect.signature(Qwen38TTNNTextModel.capture_decode_generic)
    assert tuple(signature.parameters) == (
        "self",
        "prepared",
        "state",
        "residue",
        "split",
        "guard",
        "epilogue",
        "phase_observer",
        "regime",
        "cq_id",
        "clock_ns",
        "retain_mtp_inputs",
    )
    keyword_only = [name for name, p in signature.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert keyword_only == [
        "residue",
        "split",
        "guard",
        "epilogue",
        "phase_observer",
        "regime",
        "cq_id",
        "clock_ns",
        "retain_mtp_inputs",
    ]
    assert (signature.parameters["regime"].default, signature.parameters["cq_id"].default) == (0, 0)
    assert signature.parameters["retain_mtp_inputs"].default is False  # the MTP server's epilogue input; off elsewhere
    assert tuple(inspect.signature(Qwen38TTNNTextModel.forward_decode_generic_head).parameters) == (
        "self",
        "prepared",
        "state",
    )
    assert tuple(inspect.signature(Qwen38TTNNTextModel.forward_decode_generic_tail).parameters) == (
        "self",
        "head",
        "prepared",
        "state",
        "return_logits",
        "release_head",
        "retain_mtp_inputs",
    )
    # The per-position eager body and the runner-facing fused body are untouched in shape.
    assert tuple(inspect.signature(Qwen38TTNNTextModel.forward_decode_generic).parameters) == (
        "self",
        "prepared",
        "state",
        "return_logits",
    )
