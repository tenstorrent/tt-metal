# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""No-device contracts for the GDN decode op diet (tile-aligned split, conv ring, 4-D recurrent step).

The device-op walk pins the exact ``ttnn`` call sequence of every method on the
``forward_decode`` path.  A tiled ``ttnn.reshape`` with an explicit ``pad_value``
enqueues a second op (fill of the implicit tile padding), every other listed call
is one device op, and the mac loop unrolls to ``CONV_KERNEL_SIZE - 1`` calls.
The block went from 76 device ops per layer (perf-integration-3) to 54, then to
52 once the hidden all-gather wrote the in-projection's activation shard and the
output reduce-scatter read the out-projection's partial shard directly, and to 53
when the decay gate's softplus became its own program (``softplus_gate``: the
fused SOFTPLUS activation of ``ttnn.add`` flushed to 0 below about -5).  Calls to
module-level helpers on the walked path are expanded into their ``ttnn`` calls.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
GDN_SOURCE = TESTS.parents[0] / "ttnn" / "gdn.py"
DEVICE_OPS = {
    "add",
    "all_gather",
    "linear",
    "mac",
    "matmul",
    "multiply",
    "reduce_scatter",
    "repeat_interleave",
    "reshape",
    "rms_norm",
    "sigmoid",
    "silu",
    "slice",
    "softplus",
    "subtract",
    "to_memory_config",
    "transpose",
    "typecast",
}
HOST_ONLY_TTNN_CALLS = {"UnaryWithParam"}
# Module-level helpers on the walked path whose bodies enqueue device ops (expanded in place by the walk).
DEVICE_HELPERS = {"softplus_gate"}

EXPECTED_DEVICE_OPS = {
    "_all_gather_hidden": ["all_gather"],
    "_project": ["linear", "to_memory_config"],
    "_split_projection": ["slice", "slice", "slice", "slice"],
    "_causal_conv_decode": ["multiply", "mac", "mac", "mac", "silu"],
    "_make_recurrent_inputs": [
        "slice",
        "slice",
        "slice",
        "reshape",
        "reshape",
        "fill_pad",
        "repeat_interleave",
        "repeat_interleave",
        "reshape",
        "typecast",
        "sigmoid",
        "reshape",
        "typecast",
        "add",
        "softplus",
        "multiply",
        "reshape",
    ],
    "_recurrent_decode": [
        "rms_norm",
        "multiply",
        "rms_norm",
        "multiply",
        "typecast",
        "typecast",
        "multiply",
        "multiply",
        "matmul",
        "subtract",
        "transpose",
        "matmul",
        "multiply",
        "add",
        "matmul",
    ],
    "_gate": [
        "typecast",
        "rms_norm",
        "reshape",
        "typecast",
        "sigmoid",
        "typecast",
        "multiply",
    ],
    "_out_project": ["linear", "reduce_scatter"],
}
# forward_decode calls these directly; the rest run inside the composed gdn_step chain (ttnn/fused/gdn_step)
FORWARD_METHODS = ["_validate_state", "_all_gather_hidden", "_project", "_gdn_step", "_out_project"]
EXPECTED_DEVICE_OPS_PER_LAYER = 53


def _module() -> ast.Module:
    return ast.parse(GDN_SOURCE.read_text(encoding="utf-8"), filename=str(GDN_SOURCE))


def _layer_class(module: ast.Module) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _method_source(name: str) -> str:
    source = GDN_SOURCE.read_text(encoding="utf-8")
    method = _method(_layer_class(ast.parse(source)), name)
    return "\n".join(source.splitlines()[method.lineno - 1 : method.end_lineno])


def _module_constants(module: ast.Module) -> dict[str, int]:
    """Evaluate the module's integer constants in order; ``ttnn.TILE_SIZE`` is the 32x32 tile."""

    namespace: dict[str, object] = {"ttnn": SimpleNamespace(TILE_SIZE=32)}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = eval(compile(ast.Expression(node.value), str(GDN_SOURCE), "eval"), {}, dict(namespace))
            except Exception:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                namespace[node.targets[0].id] = value
    return {name: value for name, value in namespace.items() if isinstance(value, int)}


def _loop_trip_count(loop: ast.For, constants: dict[str, int]) -> int:
    call = loop.iter
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "range" and 1 <= len(call.args) <= 2
    bounds = []
    for argument in call.args:
        if isinstance(argument, ast.Constant):
            bounds.append(int(argument.value))
        elif isinstance(argument, ast.Name):
            bounds.append(constants[argument.id])
        else:
            raise AssertionError(f"range bound {ast.unparse(argument)} is not a constant")
    return bounds[0] if len(bounds) == 1 else bounds[1] - bounds[0]


def _module_functions(module: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}


def _device_ops(method: ast.FunctionDef, constants: dict[str, int], helpers: dict[str, ast.FunctionDef]) -> list[str]:
    loops = [node for node in ast.walk(method) if isinstance(node, ast.For)]
    calls = sorted(
        (
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and (ast.unparse(node.func).startswith("ttnn.") or ast.unparse(node.func) in helpers)
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    sequence: list[str] = []
    for call in calls:
        target = ast.unparse(call.func)
        if target in helpers:
            ops = _device_ops(helpers[target], constants, helpers)
        else:
            name = target.split(".", 1)[1]
            if name in HOST_ONLY_TTNN_CALLS:
                continue
            assert name in DEVICE_OPS, f"{method.name} enqueues an unlisted ttnn call: ttnn.{name}"
            ops = [name]
            if name == "reshape" and any(keyword.arg == "pad_value" for keyword in call.keywords):
                ops.append("fill_pad")
        repeats = 1
        for loop in loops:
            if loop.lineno < call.lineno <= loop.end_lineno:
                repeats *= _loop_trip_count(loop, constants)
        sequence.extend(ops * repeats)
    return sequence


def test_forward_decode_calls_exactly_the_walked_methods_in_order() -> None:
    module = _module()
    forward = _method(_layer_class(module), "forward_decode")
    calls = sorted(
        (node for node in ast.walk(forward) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    names = [ast.unparse(node.func) for node in calls]
    called = [name.removeprefix("self.") for name in names if name.startswith("self._")]
    assert called == FORWARD_METHODS
    import inspect

    from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import gdn_step_composed

    composed = ast.parse(inspect.getsource(gdn_step_composed))
    chain = sorted(
        (node for node in ast.walk(composed) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    chained = [
        ast.unparse(node.func).removeprefix("gdn.") for node in chain if ast.unparse(node.func).startswith("gdn._")
    ]
    assert chained == [
        "_split_projection",
        "_causal_conv_decode",
        "_make_recurrent_inputs",
        "_gate",
        "_recurrent_decode",
    ]
    assert sorted(set(called + chained) - {"_validate_state", "_gdn_step"}) == sorted(EXPECTED_DEVICE_OPS)


def test_device_op_sequence_per_method_is_pinned() -> None:
    module = _module()
    constants = _module_constants(module)
    assert constants["CONV_KERNEL_SIZE"] == 4
    cls = _layer_class(module)
    helpers = {name: fn for name, fn in _module_functions(module).items() if name in DEVICE_HELPERS}
    walked = {name: _device_ops(_method(cls, name), constants, helpers) for name in EXPECTED_DEVICE_OPS}
    assert walked == EXPECTED_DEVICE_OPS
    assert sum(len(ops) for ops in walked.values()) == EXPECTED_DEVICE_OPS_PER_LAYER


def test_decode_path_has_no_copies_and_writes_persistent_buffers_in_place() -> None:
    cls_source = "\n".join(_method_source(name) for name in EXPECTED_DEVICE_OPS)
    assert "ttnn.copy(" not in cls_source
    assert "_copy_inplace(" not in cls_source
    project = _method_source("_split_projection")
    assert "output_tensor=newest" in project
    assert "ab = ttnn.slice(" not in project
    recurrent = _method_source("_recurrent_decode")
    assert "output_tensor=state.recurrent" in recurrent
    assert "to_memory_config" not in recurrent
    assert "ttnn.reshape(" not in recurrent
    assert "delta = ttnn.subtract(v, v_read, dtype=ttnn.float32" in recurrent
    assert "ttnn.typecast(v" not in recurrent
    assert "ttnn.typecast(log_decay" not in recurrent
    assert "k_col = ttnn.transpose(k_row, 2, 3" in recurrent
    inputs = _method_source("_make_recurrent_inputs")
    assert "k_heads = ttnn.reshape(k_slice, (1, QK_HEADS_PER_DEVICE, 1, HEAD_DIM), pad_value=0.0)" in inputs
    assert "q_heads = ttnn.reshape(q_slice, (1, QK_HEADS_PER_DEVICE, 1, HEAD_DIM))" in inputs
    assert inputs.count("dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)") == 2
    assert "dim=2" not in inputs
    forward = _method_source("forward_decode")
    order = (
        "window = state.conv_window()",
        "projected = self._project(full_hidden)",
        "step = self._gdn_step()",
        "gated = step(self, projected, window, state)",
        "state.advance_conv_window()",
    )
    offsets = [forward.index(fragment) for fragment in order]
    assert offsets == sorted(offsets)
    # the composed chain (ttnn/fused/gdn_step.gdn_step_composed) slices into the ring slot before the conv reads it
    import inspect

    from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import gdn_step_composed

    composed = inspect.getsource(gdn_step_composed)
    chain = (
        "z, a, b = gdn._split_projection(projected, window[3])",
        "conv = gdn._causal_conv_decode(window)",
        "gdn._make_recurrent_inputs(conv, a, b)",
        "gdn._recurrent_decode(q, k, v, beta, log_decay, producers, state)",
    )
    chain_offsets = [composed.index(fragment) for fragment in chain]
    assert chain_offsets == sorted(chain_offsets)


def test_collectives_read_and_write_the_matmul_shard_layouts_directly() -> None:
    """The gather writes the in-proj activation shard; the reduce-scatter reads the out-proj partial shard."""

    gather = _method_source("_all_gather_hidden")
    gather_call = gather.split("full_hidden = ttnn.all_gather(", 1)[1].split("\n        )", 1)[0]
    assert "memory_config=self.in_proj_act_memory_config" in gather_call
    assert "cluster_axis=TP_AXIS" in gather_call
    project = _method_source("_project")
    assert "to_memory_config(full_hidden" not in project
    assert "hidden_ws" not in project
    assert "projected_ws = ttnn.linear(\n            full_hidden," in project
    gate = _method_source("_out_project")
    assert "to_memory_config(partial_ws" not in gate
    assert "mark_local_partial(\n            partial_ws," in gate
    reduce = gate.split("output = ttnn.reduce_scatter(", 1)[1].split("\n        )", 1)[0]
    assert reduce.startswith("\n            partial_ws,")
    assert "memory_config=ttnn.DRAM_MEMORY_CONFIG" in reduce
    assert "topology=self.collective_topology" in reduce
    assert gate.index("output = ttnn.reduce_scatter(") < gate.index("_deallocate(partial_ws)")
    assert gate.index("_deallocate(partial_ws)") < gate.index("_deallocate(full_hidden)")


def test_gdn_module_owns_its_recurrent_step_and_leaves_the_shared_fla_module_alone() -> None:
    source = GDN_SOURCE.read_text(encoding="utf-8")
    assert "recurrent_gated_delta_rule_decode_ttnn" not in source
    assert "gated_attention_gated_deltanet" not in source


def test_decay_gate_softplus_is_its_own_program_on_every_gate_path() -> None:
    """The fused SOFTPLUS activation of ``ttnn.add`` returns exactly 0 for a + dt_bias <= -5.02 on 4x p150
    (469 of 1024 probe points in [-16, 8]; 1.5e-3 off elsewhere); ``ttnn.softplus`` never flushes and is
    4.7e-4-accurate.  The 1-row step and the chunk/rows path build the gate through the same helper."""

    source = GDN_SOURCE.read_text(encoding="utf-8")
    assert "UnaryOpType.SOFTPLUS" not in source
    gate = _module_functions(_module())["softplus_gate"]
    calls = [
        node for node in ast.walk(gate) if isinstance(node, ast.Call) and ast.unparse(node.func).startswith("ttnn.")
    ]
    assert [ast.unparse(call.func) for call in calls] == ["ttnn.add", "ttnn.softplus"]
    add, softplus = calls
    assert {keyword.arg for keyword in add.keywords} == {"memory_config"}
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in softplus.keywords} == {
        "beta": "1.0",
        "threshold": "20.0",
        "memory_config": "memory_config",
    }
    for method in ("_make_recurrent_inputs", "_make_chunk_inputs"):
        assert _method_source(method).count("softplus_gate(a_fp32, self.weights.dt_bias, memory_config=") == 1


def test_static_contract_pins_the_tile_aligned_projection_columns() -> None:
    module = _module()
    constants = _module_constants(module)
    assert constants["QKVZAB_WIDTH_PER_DEVICE"] == 4120
    assert constants["A_COLUMN"] == 4096
    assert constants["B_COLUMN"] == 4128
    assert constants["PROJECTION_WIDTH_PER_DEVICE"] == 4160
    contract = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_gdn_static_contract"
    )
    expected = next(
        node.value
        for node in contract.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "expected"
    )
    pinned = {key.value: value.value for key, value in zip(expected.keys, expected.values)}
    assert pinned["QKVZAB_WIDTH_PER_DEVICE"] == 4120
    assert pinned["A_COLUMN"] == 4096
    assert pinned["B_COLUMN"] == 4128
    assert pinned["PROJECTION_WIDTH_PER_DEVICE"] == 4160
    assert all(pinned[name] % 32 == 0 for name in ("A_COLUMN", "B_COLUMN", "PROJECTION_WIDTH_PER_DEVICE"))


def test_pack_projection_columns_moves_b_to_a_tile_boundary_with_zero_gaps(expect_error) -> None:
    torch = pytest.importorskip("torch")
    gdn = pytest.importorskip("models.demos.blackhole.qwen38_flash_next.ttnn.gdn")
    gdn.validate_gdn_static_contract()
    shards = []
    for device_index in range(gdn.TP_SIZE):
        fused = torch.arange(gdn.QKVZAB_WIDTH_PER_DEVICE * gdn.HIDDEN_SIZE, dtype=torch.float32)
        fused = (fused + device_index * 0.25).reshape(gdn.QKVZAB_WIDTH_PER_DEVICE, gdn.HIDDEN_SIZE)
        shards.append(SimpleNamespace(fused_qkvzab=fused.to(torch.bfloat16)))
    packed = gdn.pack_projection_columns(shards)
    assert tuple(packed.shape) == (1, 1, gdn.HIDDEN_SIZE, gdn.TP_SIZE * gdn.PROJECTION_WIDTH_PER_DEVICE)
    a_end = gdn.A_COLUMN + gdn.VALUE_HEADS_PER_DEVICE
    b_end = gdn.B_COLUMN + gdn.VALUE_HEADS_PER_DEVICE
    for device_index, shard in enumerate(shards):
        fused = shard.fused_qkvzab.transpose(0, 1)
        block = packed[0, 0, :, device_index * gdn.PROJECTION_WIDTH_PER_DEVICE :][:, : gdn.PROJECTION_WIDTH_PER_DEVICE]
        assert torch.equal(block[:, :a_end], fused[:, :a_end])
        assert torch.equal(block[:, gdn.B_COLUMN : b_end], fused[:, a_end : gdn.QKVZAB_WIDTH_PER_DEVICE])
        assert torch.count_nonzero(block[:, a_end : gdn.B_COLUMN]) == 0
        assert torch.count_nonzero(block[:, b_end:]) == 0
    with expect_error(RuntimeError, match="fused qkvzab shard shape"):
        gdn.pack_projection_columns([SimpleNamespace(fused_qkvzab=torch.zeros(4, 4))])


def test_conv_ring_phase_selects_the_window_and_round_trips_through_snapshots(monkeypatch) -> None:
    gdn = pytest.importorskip("models.demos.blackhole.qwen38_flash_next.ttnn.gdn")
    copies: list[tuple[object, object]] = []
    monkeypatch.setattr(gdn, "_copy_inplace", lambda source, target, *, label: copies.append((source, target)))
    monkeypatch.setattr(gdn.Qwen38TTNNGDNState, "validate", lambda self: None)
    monkeypatch.setattr(gdn.Qwen38TTNNGDNSnapshot, "validate", lambda self, contract: None)
    slots = tuple(object() for _ in range(gdn.CONV_KERNEL_SIZE))
    state = gdn.Qwen38TTNNGDNState(0, object(), slots, object(), object(), object())
    snapshot = gdn.Qwen38TTNNGDNSnapshot(0, object(), tuple(object() for _ in slots))

    assert state.conv_phase == 0
    assert state.conv_window() == (slots[1], slots[2], slots[3], slots[0])
    state.advance_conv_window()
    assert state.conv_phase == 1
    assert state.conv_window() == (slots[2], slots[3], slots[0], slots[1])
    # Four consecutive tokens visit every slot exactly once as the newest slot.
    newest = [state.conv_window()[-1]]
    for _ in range(gdn.CONV_KERNEL_SIZE - 1):
        state.advance_conv_window()
        newest.append(state.conv_window()[-1])
    assert newest == [slots[1], slots[2], slots[3], slots[0]]
    assert state.conv_phase == 0
    state.advance_conv_window()
    assert state.conv_phase == 1

    state.capture_into(snapshot)
    assert snapshot.captured and snapshot.conv_phase == 1
    assert len(copies) == 1 + gdn.CONV_KERNEL_SIZE
    state.advance_conv_window()
    state.advance_conv_window()
    assert state.conv_phase == 3
    state.restore_from(snapshot)
    assert state.conv_phase == 1
    assert len(copies) == 2 * (1 + gdn.CONV_KERNEL_SIZE)
    state.reset_inplace()
    assert state.conv_phase == 0
    assert state.conv_window() == (slots[1], slots[2], slots[3], slots[0])


def test_state_and_snapshot_validation_reject_an_out_of_range_phase(expect_error) -> None:
    gdn = pytest.importorskip("models.demos.blackhole.qwen38_flash_next.ttnn.gdn")
    ttnn = pytest.importorskip("ttnn")

    class FakeContract:
        @staticmethod
        def validate_tensor(tensor, *, placement, shard_dim):
            assert placement is gdn.TensorPlacement.HEAD_SHARDED and shard_dim in (1, 3)

    def fake(shape, dtype):
        return SimpleNamespace(shape=shape, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    recurrent_shape = (1, gdn.VALUE_HEADS_PER_DEVICE, gdn.HEAD_DIM, gdn.HEAD_DIM)
    conv_shape = (1, 1, 1, gdn.QKV_WIDTH_PER_DEVICE)
    conv = tuple(fake(conv_shape, ttnn.bfloat16) for _ in range(gdn.CONV_KERNEL_SIZE))
    snapshot = gdn.Qwen38TTNNGDNSnapshot(0, fake(recurrent_shape, ttnn.float32), conv)
    snapshot.validate(FakeContract())
    snapshot.conv_phase = gdn.CONV_KERNEL_SIZE
    with expect_error(RuntimeError, match="conv phase"):
        snapshot.validate(FakeContract())

    state = gdn.Qwen38TTNNGDNState(
        0,
        fake(recurrent_shape, ttnn.float32),
        conv,
        fake(recurrent_shape, ttnn.float32),
        fake(conv_shape, ttnn.bfloat16),
        FakeContract(),
    )
    state.validate()
    state.conv_phase = -1
    with expect_error(RuntimeError, match="conv phase"):
        state.validate()
