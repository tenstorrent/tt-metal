# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The batched-lanes GDN step, the lane resets, the final mixer rows and the lane bodies' source pins, no device.

``Qwen38TTNNGDN.forward_decode_lanes`` on B lanes (the ``[B,12,128,128]`` state, the ``[1,1,B,2560]`` ring slots
under one shared phase) runs on the step-4 fake with every matmul computed per (lane, head) tile in float64 and
rounded once, so no result depends on the batch it sits in; B separate 1-row ``forward_decode`` chains on the same
fake are the reference.  Twelve steps (three ring cycles) per lane count, lane u's recurrent state, its four ring
rows and its output row bitwise the chain's after every step; a lane admitted mid-run at a step of its residue class
(``reset_lane_inplace`` then a fresh chain) stays bitwise, the misaligned control differs and the position row
refuses it.  The final mixer rows path runs against the 1-row mixer row for row on the GR fake.  The source pins hold
the 1-row bodies untouched and the lane bodies free of host I/O, and the fused lane body (the ``gdn_step`` program on B
rows, bound over the class body under ``QWEN38_FUSED``) the 1-row fused walk on B rows in the composed ring order.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import GDN_RESIDUE_CLASSES, Qwen38TTNNDevicePositionRow
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import (
    FLAT_LOCAL_WIDTH,
    RESIDUAL_RANK,
    Qwen38TTNNFinalMixer,
    Qwen38TTNNFinalMixerWeights,
)

TESTS = Path(__file__).resolve().parent
TP = 4
STEPS = 12
ADMISSION_STEP = 8  # phase 0 again: a fresh lane (position 0) joins at a step of its residue class
LANE_COUNTS = (1, 2, 4, 8)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step4 = _load("test_mtp_v2_step4_rows_no_device")
gr_test = _load("test_gr_rows_no_device")
lanes_test = _load("test_batched_lanes_no_device")
BF16, FP32, TILE = step4.BF16, step4.FP32, step4.TILE
FakeContract, FakeTensor = step4.FakeContract, step4.FakeTensor


# --------------------------------------------------------------------------- the fake


def _matmul_per_tile(a, b, *, memory_config=None, dtype=None, program_config=None, compute_kernel_config=None):
    """Every (batch) matmul as its own float64 2-D product rounded once: the same call for a tile in a 1-row
    operand and for the same tile at batch index u of a lanes operand."""

    out_dtype = dtype or a.dtype
    results = []
    for x, y in zip(a.torch_shards(), b.torch_shards()):
        xs, ys = x.double().reshape(-1, *x.shape[-2:]), y.double().reshape(-1, *y.shape[-2:])
        if ys.shape[0] == 1 and xs.shape[0] > 1:
            ys = ys.expand(xs.shape[0], -1, -1)
        product = torch.stack([xi @ yi for xi, yi in zip(xs, ys)])
        results.append(product.reshape(*x.shape[:-2], x.shape[-2], y.shape[-1]).to(out_dtype.torch))
    return FakeTensor(results, out_dtype, a.layout)


def _multiply_device_zero(base_multiply):
    """The step-4 multiply with binary_ng's rule: the product is +0.0 whenever an input is zero (so the keep-mask
    reset of a lane leaves +0.0, the zero source's bits, as the device does; torch alone gives -0.0 for x < 0)."""

    def op(a, b, **kwargs):
        result = base_multiply(a, b, **kwargs)
        for index, x in enumerate(result.torch_shards()):
            if x.dtype.is_floating_point:
                a_shard = a.torch_shards()[index]
                zero = a_shard == 0
                if isinstance(b, FakeTensor):
                    zero = zero | (b.torch_shards()[index] == 0)
                elif b == 0:
                    zero = torch.ones_like(zero)
                x.copy_(torch.where(zero, torch.zeros_like(x), x))
        return result

    return op


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("QWEN38_FUSED_OFF", "ple")  # the fused PLE lane body serves by default; the fake runs the chain
    chunk = step4.FakeChunk()
    fake_ttnn = step4.make_fake_ttnn(chunk)
    step4.install_source_chunk_fake(monkeypatch, chunk)
    fake_ttnn.matmul = _matmul_per_tile
    fake_ttnn.multiply = _multiply_device_zero(fake_ttnn.multiply)
    fake_ttnn.empty_like = lambda t, dtype=None, layout=None, memory_config=None: FakeTensor(
        [torch.empty_like(x) for x in t.torch_shards()], dtype or t.dtype, layout or t.layout
    )
    base_reshape = fake_ttnn.reshape
    # A same-shape reshape is a view of its input on the device (the PLE lanes B = 1 defect, 2026-09-04): the
    # fake hands the input back, so releasing it after the reshape is a read of a deallocated tensor here too.
    fake_ttnn.reshape = lambda t, shape, **kwargs: t if tuple(shape) == t.shape else base_reshape(t, shape, **kwargs)
    for module in (gdn_module, ple_module):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return fake_ttnn


@pytest.fixture(scope="module")
def gdn():
    return step4._gdn_module(step4._device_gdn_weights(step4._gdn_oracle_weights()))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor.view(torch.int16)


def _equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and bool(torch.equal(_bits(actual), _bits(expected)))


def _lane_image(state: gdn_module.Qwen38TTNNGDNState, lane: int) -> dict[str, torch.Tensor]:
    """Lane u of a lanes state in the 1-row state's shapes (the recurrent over the four head shards, the ring rows)."""

    recurrent = step4._cat(state.recurrent, 1)[lane : lane + 1]
    slots = [step4._cat(slot, 3)[:, :, lane : lane + 1] for slot in state.conv]
    return {"recurrent": recurrent, **{f"conv[{index}]": slot for index, slot in enumerate(slots)}}


def _one_row_image(state: gdn_module.Qwen38TTNNGDNState) -> dict[str, torch.Tensor]:
    return _lane_image(state, 0)


# --------------------------------------------------------------------------- the lanes step against B chains


@pytest.mark.parametrize("lanes", LANE_COUNTS)
def test_gdn_lanes_equal_b_one_row_chains_over_twelve_steps_with_an_aligned_admission(
    fake, gdn, lanes: int, expect_error
) -> None:
    torch.manual_seed(100 + lanes)
    hidden = torch.randn(STEPS, lanes, 2560).to(torch.bfloat16)  # step t, lane u
    lane_state = gdn.allocate_lane_state(lanes)
    assert lane_state.batch_size == lanes and lane_state.conv_phase == 0
    chains = [gdn.allocate_state() for _ in range(lanes)]
    admitted = 1 if lanes > 1 else 0
    for step in range(STEPS):
        if step == ADMISSION_STEP:
            # Lane `admitted` is a fresh session joining at a step of its residue class (phase 0 = position 0).
            assert lane_state.conv_phase == 0
            lane_state.reset_lane_inplace(admitted)
            chains[admitted] = gdn.allocate_state()
            image = _lane_image(lane_state, admitted)
            assert all(int(torch.count_nonzero(value)) == 0 for value in image.values())
        result = gdn.forward_decode_lanes(step4._hidden_sharded(hidden[step : step + 1]), lane_state)
        assert result.state is lane_state and lane_state.conv_phase == (step + 1) % GDN_RESIDUE_CLASSES
        output_lanes = step4._cat(result.hidden_sharded, 3)
        assert output_lanes.shape == (1, 1, lanes, 2560)
        for lane, chain in enumerate(chains):
            reference = gdn.forward_decode(step4._hidden_sharded(hidden[step : step + 1, lane : lane + 1]), chain)
            assert chain.conv_phase == lane_state.conv_phase
            assert _equal(output_lanes[:, :, lane : lane + 1], step4._cat(reference.hidden_sharded, 3)), (step, lane)
            lane_image, chain_image = _lane_image(lane_state, lane), _one_row_image(chain)
            for name, value in chain_image.items():
                assert _equal(lane_image[name], value), (step, lane, name)
    if lanes > 1:
        # The lanes differ from each other (the comparison is not vacuous) and the 1-row body refuses the state.
        assert not _equal(_lane_image(lane_state, 0)["recurrent"], _lane_image(lane_state, 1)["recurrent"])
        with expect_error(ValueError, match="the 1-row GDN decode body admits a batch-1 state"):
            gdn.forward_decode(step4._hidden_sharded(hidden[:1, :1]), lane_state)


def test_misaligned_admission_differs_and_the_position_row_refuses_it(fake, gdn, monkeypatch, expect_error) -> None:
    torch.manual_seed(7)
    hidden = torch.randn(STEPS, 2, 2560).to(torch.bfloat16)
    lane_state = gdn.allocate_lane_state(2)
    misaligned_step = 5  # ring phase 1: a fresh session (position 0, residue 0) may not join here
    fresh = None
    for step in range(STEPS):
        if step == misaligned_step:
            assert lane_state.conv_phase == 1
            lane_state.reset_lane_inplace(1)
            fresh = gdn.allocate_state()  # phase 0: the chain a fresh session would run
        gdn.forward_decode_lanes(step4._hidden_sharded(hidden[step : step + 1]), lane_state)
        if fresh is not None:
            gdn.forward_decode(step4._hidden_sharded(hidden[step : step + 1, 1:2]), fresh)
    lane_image, fresh_image = _lane_image(lane_state, 1), _one_row_image(fresh)
    assert not all(_equal(lane_image[name], fresh_image[name]) for name in fresh_image)
    # The rule that forbids it: the position row admits a lane only at a step of its residue class.
    lane_fake = lanes_test.make_lane_fake()
    monkeypatch.setattr(contracts_module, "ttnn", lane_fake)
    monkeypatch.setattr(contracts_module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate")
    row = Qwen38TTNNDevicePositionRow.allocate("mesh", FakeContract(), [5, 5], lanes=2)
    with expect_error(ValueError, match="residue 0, expected the row's residue 1; admit it 3 steps later"):
        row.admit(1, 0)
    for _ in range(3):
        row.advance()
    row.admit(1, 0)
    assert row.positions[:2] == [8, 0] and row.residue == 0


# --------------------------------------------------------------------------- lane resets and the snapshot at B


def test_lane_reset_zeroes_one_lane_and_keeps_the_others_bitwise(fake, gdn, expect_error) -> None:
    lanes = 4
    torch.manual_seed(3)
    state = gdn.allocate_lane_state(lanes)
    for local in state.recurrent.locals:
        local.copy_(torch.randn(lanes, 12, 128, 128) * 0.3)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, lanes, 2560).to(torch.bfloat16))
    before = {lane: _lane_image(state, lane) for lane in range(lanes)}
    state.reset_lane_inplace(2)
    for lane in range(lanes):
        image = _lane_image(state, lane)
        for name, value in image.items():
            if lane == 2:
                assert int(torch.count_nonzero(value)) == 0 and not bool(torch.signbit(value).any()), name
            else:
                assert _equal(value, before[lane][name]), (lane, name)
    for bad in (-1, 4, True, 1.0):
        with expect_error(ValueError, match=r"lane must be an int in \[0,4\)"):  # allow-pytest.raises: contract
            state.reset_lane_inplace(bad)
    # The snapshot follows the batch: capture, mutate, restore round trip.
    snapshot = state.allocate_snapshot()
    assert snapshot.batch_size == lanes and snapshot.recurrent.shape == (lanes, 12, 128, 128)
    state.capture_into(snapshot)
    state.reset_inplace()
    assert all(int(torch.count_nonzero(v)) == 0 for lane in range(lanes) for v in _lane_image(state, lane).values())
    state.restore_from(snapshot)
    for lane in range(lanes):
        for name, value in _lane_image(state, lane).items():
            assert _equal(value, before[lane][name]) if lane != 2 else int(torch.count_nonzero(value)) == 0, name
    snapshot.batch_size = 1
    with expect_error(RuntimeError, match=r"GDN snapshot recurrent state local shape must be \(1, 12, 128, 128\)"):
        snapshot.validate(state.mesh_contract)


class _LaneLookup(step4._FakeResidentLookup):
    def lookup_lanes(self, tokens, contexts):
        payloads, next_contexts = zip(
            *(self.lookup_token(int(token), context) for token, context in zip(tokens, contexts))
        )
        return bytearray().join(payloads), tuple(next_contexts)


def test_ple_lanes_at_one_lane_equal_the_one_row_path(fake) -> None:
    """B = 1: the value row keeps its shape through the rows projection (a same-shape reshape is a view; releasing
    the input freed it on the device), so the lane is the 1-row path bitwise over three steps."""

    module = step4._ple_module()
    module._resident_lookup = _LaneLookup()
    module.host_embedding = module._resident_lookup
    torch.manual_seed(5)
    lanes_state = module.allocate_lanes_state(1)
    chain = module.allocate_state()
    for step, token in enumerate((17, 20, 16)):
        residual = torch.randn(1, 4, 2560).to(torch.bfloat16)
        prepared = module.prepare_lanes_input([token], lanes_state)
        delta = module.forward_prepared_lanes(step4._residual_rows(residual), prepared, lanes_state)
        one = module.prepare_decode_input(torch.tensor([[token]], dtype=torch.long), chain)
        reference = module.forward_prepared(step4._residual_rows(residual), one, chain)
        assert _equal(step4._cat(delta, 3), step4._cat(reference.residual_delta, 3)), step
        for index in range(ple_module.CONV_STATE_LENGTH):
            assert _equal(step4._cat(lanes_state.conv[index], 3), step4._cat(chain.conv[index], 3)), (step, index)
        assert lanes_state.token_contexts == (tuple(int(v) for v in chain.token_context[0]),)
        prepared.release()


def test_ple_lane_reset_zeroes_one_lane_and_clears_its_context(fake, expect_error) -> None:
    module = step4._ple_module()
    module._resident_lookup = _LaneLookup()
    module.host_embedding = module._resident_lookup
    lanes = 3
    torch.manual_seed(4)
    state = module.allocate_lanes_state(lanes)
    for step in range(2):
        prepared = module.prepare_lanes_input([17 + step, 20 + step, 16], state)
        delta = module.forward_prepared_lanes(
            step4._residual_rows(torch.randn(lanes, 4, 2560).to(torch.bfloat16)), prepared, state
        )
        prepared.release()
        del delta
    before = [step4._cat(slot, 3) for slot in state.conv]
    contexts = state.token_contexts
    assert all(context is not None for context in contexts)
    state.reset_lane_inplace(1)
    for index, slot in enumerate(state.conv):
        after = step4._cat(slot, 3)
        assert int(torch.count_nonzero(after[:, 1])) == 0, index
        assert _equal(after[:, 0], before[index][:, 0]) and _equal(after[:, 2], before[index][:, 2]), index
    assert state.token_contexts == (contexts[0], None, contexts[2])
    with expect_error(ValueError, match=r"PLE lane must be an int in \[0,3\), got 3"):  # allow-pytest.raises
        state.reset_lane_inplace(3)


# --------------------------------------------------------------------------- final mixer rows


def _mixer_fake():
    fake = gr_test._gr_fake()

    def all_reduce(tensor, *, cluster_axis, memory_config=None, topology=None):
        total = gr_test._sequential_sum(torch.stack(tensor.torch_shards()), 0).squeeze(0)
        return FakeTensor([total.clone() for _ in range(TP)], tensor.dtype, tensor.layout)

    fake.all_reduce = all_reduce
    return fake


def _mixer_module(fake) -> Qwen38TTNNFinalMixer:
    torch.manual_seed(9)
    weights = Qwen38TTNNFinalMixerWeights(
        norm_scale=gr_test.FakeTensor(
            [(torch.randn(1, 4, 1, 640) * 0.1 + 0.25) for _ in range(TP)], gr_test.FP32, gr_test.TILE, 3
        ),
        down=gr_test.FakeTensor(
            [gr_test._bf16(1, 1, FLAT_LOCAL_WIDTH, RESIDUAL_RANK, scale=FLAT_LOCAL_WIDTH**-0.5) for _ in range(TP)],
            gr_test.BF16,
            gr_test.TILE,
            2,
        ),
        up=gr_test.FakeTensor(
            [gr_test._bf16(1, 1, RESIDUAL_RANK, FLAT_LOCAL_WIDTH, scale=RESIDUAL_RANK**-0.5) for _ in range(TP)],
            gr_test.BF16,
            gr_test.TILE,
            3,
        ),
        replicated_anchor=gr_test.FakeTensor(
            [torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], gr_test.BF16, gr_test.TILE
        ),
        namespace="backbone",
    )
    module = object.__new__(Qwen38TTNNFinalMixer)
    module.mesh_device = "mesh"
    module.mesh_contract = gr_test.FakeContract()
    module.weights = weights
    module.collective_topology = "linear"
    module.compute_config = "compute"
    module.weight_compute_config = "weight_compute"  # the down / up linears (decode_matmul fidelity)
    module.down_act_memory_config = fake.DRAM_MEMORY_CONFIG
    module.down_program_config = "down"
    module.up_act_memory_config = fake.DRAM_MEMORY_CONFIG
    module.up_program_config = "up"
    module.norm_scale_flat = fake.experimental.view(weights.norm_scale, (1, 1, 1, FLAT_LOCAL_WIDTH))
    return module


@pytest.mark.parametrize("rows", [1, 3, 8, 32])
def test_final_mixer_rows_equal_the_one_row_mixer_row_for_row(monkeypatch, rows: int, expect_error) -> None:
    fake = _mixer_fake()
    monkeypatch.setattr(mixer_module, "ttnn", fake)
    module = _mixer_module(fake)
    torch.manual_seed(20 + rows)
    residual = gr_test.FakeTensor([gr_test._bf16(1, 4, rows, 640) for _ in range(TP)], gr_test.BF16, gr_test.TILE, 3)
    output = module.rows(residual)
    assert output.shape == (1, 1, rows, 640) and output.dtype is gr_test.BF16
    for row in range(rows):
        one = module(gr_test._rows(residual, row))
        assert one.shape == (1, 1, 1, 640)
        gr_test._equal(
            gr_test.FakeTensor([x[:, :, row : row + 1] for x in output.torch_shards()], output.dtype, output.layout),
            [one],
            dim=2,
            label=f"final mixer rows row {row}",
        )
    with expect_error(
        ValueError, match=r"final mixer residual rows must be in \[1,32\], got 33"
    ):  # allow-pytest.raises
        module.rows(
            gr_test.FakeTensor([gr_test._bf16(1, 4, 33, 640) for _ in range(TP)], gr_test.BF16, gr_test.TILE, 3)
        )
    with expect_error(ValueError, match=r"must be rank 4"):  # allow-pytest.raises: contract
        module.rows(gr_test.FakeTensor([gr_test._bf16(4, 1, 640) for _ in range(TP)], gr_test.BF16, gr_test.TILE, 2))


# --------------------------------------------------------------------------- source pins


def _self_walk(function, receiver: str = "self") -> list[str]:
    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse("\n".join(line[indent:] for line in lines))
    return [
        ast.unparse(node.func).removeprefix(f"{receiver}.")
        for node in sorted(
            (n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
        if ast.unparse(node.func).startswith(f"{receiver}.")
    ]


def _calls(function) -> list[str]:
    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse("\n".join(line[indent:] for line in lines))
    return [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


HOST_IO = ("ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor", "torch.")


def test_gdn_lanes_body_is_the_one_row_walk_with_the_lanes_suffix_and_the_one_row_body_is_untouched() -> None:
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gdn_step_module

    one_row = [name for name in _self_walk(gdn_module.Qwen38TTNNGDN.forward_decode) if name.startswith("_")]
    lanes = [name for name in _self_walk(gdn_module.Qwen38TTNNGDN.forward_decode_lanes) if name.startswith("_")]
    # The 1-row body resolves its projection -> gated-output step through the fused registry (ttnn/fused/gdn_step,
    # opt-in): the composed chain it runs by default is the walk the lane stages mirror.
    assert one_row == ["_validate_state", "_all_gather_hidden", "_project", "_gdn_step", "_out_project"]
    chain = [name for name in _self_walk(gdn_step_module.gdn_step_composed, receiver="gdn") if name.startswith("_")]
    # source order: the gate wraps the recurrence on one line (``gdn._gate(gdn._recurrent_decode(...), z)``)
    assert chain == ["_split_projection", "_causal_conv_decode", "_make_recurrent_inputs", "_gate", "_recurrent_decode"]
    assert lanes == [
        "_validate_lane_state",
        "_all_gather_hidden_lanes",
        "_project_lanes",
        "_causal_conv_lanes",
        "_make_recurrent_inputs_lanes",
        "_recurrent_decode_lanes",
        "_gate_and_project_lanes",
    ]
    # The same ttnn ops per stage (the lane stage is the 1-row stage on B rows, nothing added or removed): the 1-row
    # projection and its z/a/b split are the lanes projection (its linear split out as ``_project_lanes_unsplit``, the
    # fused program's input), its gate and out-projection the lanes gate (its tail split out as ``_out_project_lanes``,
    # the fused program's successor); the z/a/b slices are one call site in a loop on the lanes side.
    stages = (
        (("_all_gather_hidden",), ("_all_gather_hidden_lanes",)),
        (("_project", "_split_projection"), ("_project_lanes_unsplit", "_project_lanes")),
        (("_causal_conv_decode",), ("_causal_conv_lanes",)),
        (("_make_recurrent_inputs",), ("_make_recurrent_inputs_lanes",)),
        (("_recurrent_decode",), ("_recurrent_decode_lanes",)),
        (("_gate", "_out_project"), ("_out_project_lanes", "_gate_and_project_lanes")),
    )
    assert [lane_names[-1] for _, lane_names in stages] == lanes[1:]
    for one_names, lane_names in stages:
        one_calls = {
            c for name in one_names for c in _calls(getattr(gdn_module.Qwen38TTNNGDN, name)) if c.startswith("ttnn.")
        }
        lane_calls = {
            c for name in lane_names for c in _calls(getattr(gdn_module.Qwen38TTNNGDN, name)) if c.startswith("ttnn.")
        }
        assert one_calls == lane_calls, (one_names, lane_names)
    assert _self_walk(gdn_module.Qwen38TTNNGDN._project_lanes)[0] == "_project_lanes_unsplit"
    assert _self_walk(gdn_module.Qwen38TTNNGDN._gate_and_project_lanes)[-1] == "_out_project_lanes"
    for function in (
        gdn_module.Qwen38TTNNGDN.forward_decode_lanes,
        *(getattr(gdn_module.Qwen38TTNNGDN, name) for name in lanes),
        gdn_module.Qwen38TTNNGDN._project_lanes_unsplit,
        gdn_module.Qwen38TTNNGDN._out_project_lanes,
        gdn_module.Qwen38TTNNGDN._forward_decode_lanes_fused,
        layer_module.Qwen38TTNNDecoderLayer.forward_decode_lanes,
        model_module.Qwen38TTNNTextModel.forward_decode_lanes,
        model_module.Qwen38TTNNTextModel._embed_residual_lanes_from_device_token,
        model_module.Qwen38TTNNTextModel.resolve_lane_tokens,
        mixer_module.Qwen38TTNNFinalMixer.rows,
    ):
        assert not any(call.startswith(HOST_IO) for call in _calls(function)), function.__qualname__
    assert "Advance one true-global-B1 token" in inspect.getsource(gdn_module.Qwen38TTNNGDN.forward_decode)
    assert "admits a batch-1 state" in inspect.getsource(gdn_module.Qwen38TTNNGDN._validate_state)
    assert "lanes" not in inspect.getsource(gdn_module.Qwen38TTNNGDN._recurrent_decode)
    # The lane resets are keep-mask multiplies landed in place (eager host uploads), never the trace body's ops.
    for function in (
        gdn_module.Qwen38TTNNGDNState.reset_lane_inplace,
        ple_module.Qwen38TTNNPLELanesState.reset_lane_inplace,
        qsa_module.Qwen38TTNNQSA.reset_lane_inplace,
    ):
        calls = _calls(function)
        assert "ttnn.multiply" in calls and "ttnn.from_torch" in calls, function
        assert "ttnn.copy" in calls or "_copy_inplace" in calls, function


def test_gdn_fused_lanes_body_is_the_one_row_fused_walk_on_b_rows_and_binds_under_the_switch() -> None:
    """The fused lane body (``QWEN38_FUSED=gdn_step``) is ``forward_decode``'s walk with the lanes suffix around the
    fused ``gdn_step`` program at rows = B: the gather, the unsplit projection, the program, the out-projection.  Its
    ring order is the composed lanes body's (the window read before the step, the token landing in the window's last
    slot, the phase advanced once after the step); it is bound over the class body at construction only when the kernel
    is on, and the class body, the fallback, still runs the lane chain."""

    from models.demos.blackhole.qwen38_flash_next.ttnn import fused
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gdn_step_module

    cls = gdn_module.Qwen38TTNNGDN
    fused_body = cls._forward_decode_lanes_fused
    assert [name for name in _self_walk(fused_body) if name.startswith("_")] == [
        "_validate_lane_state",
        "_all_gather_hidden_lanes",
        "_project_lanes_unsplit",
        "_out_project_lanes",
    ]
    calls = _calls(fused_body)
    assert "fused.gdn_step.gdn_step" in calls and "state.conv_window" in calls and "state.advance_conv_window" in calls
    assert not any(call.startswith("ttnn.") for call in calls)  # every device op is in the helpers or the program
    assert not any(call.startswith(HOST_IO) for call in calls)
    source = inspect.getsource(fused_body)
    order = (
        "window = state.conv_window()",
        "projected = self._project_lanes_unsplit(full_hidden, lanes)",
        "gated = fused.gdn_step.gdn_step(self, projected, window, state)",
        "state.advance_conv_window()",
        "output = self._out_project_lanes(gated, full_hidden, lanes)",
    )
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    # The ring order of the 1-row body and of the composed lanes body: the window is read at the old phase, this
    # token's q/k/v land in its last slot (the program writes ``window[3]``; the chain slices into ``window[-1]``),
    # the phase advances once after the write.
    one_row = inspect.getsource(cls.forward_decode)
    assert (
        one_row.index("window = state.conv_window()")
        < one_row.index("gated = step(self, projected, window, state)")
        < one_row.index("state.advance_conv_window()")
    )
    composed = inspect.getsource(cls.forward_decode_lanes)
    assert (
        composed.index("window = state.conv_window()")
        < composed.index("z, a, b = self._project_lanes(full_hidden, window[-1], lanes)")
        < composed.index("state.advance_conv_window()")
    )
    wrapper = "".join(inspect.getsource(gdn_step_module.gdn_step).split())  # wrapping-agnostic
    assert "run(projected,window[:3],window[3]," in wrapper and "ttnn.deallocate(projected)" in wrapper
    assert "the last slot receives that token" in inspect.getsource(gdn_module.Qwen38TTNNGDNState.conv_window)
    # The unsplit projection writes no ring slot and takes no z/a/b slice (the program does both); the out-projection
    # is the one reduce-scatter of the lanes body.
    unsplit = _calls(cls._project_lanes_unsplit)
    assert "ttnn.slice" not in unsplit and unsplit.count("ttnn.linear") == 1
    assert _calls(cls._out_project_lanes).count("ttnn.reduce_scatter") == 1
    assert "lanes" in fused.kernel("gdn_step").replaces and "_forward_decode_lanes_fused" in gdn_step_module.__doc__
    # The binding: the one ``if fused.enabled("gdn_step")`` of the constructor assigns the partial over the class
    # attribute; the kernel serves by default (since 2026-09-25) and QWEN38_FUSED_OFF=gdn_step unbinds it; the class
    # body still runs the lane chain.
    init = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    binds = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "fused.enabled('gdn_step')"
    ]
    assert len(binds) == 1 and len(binds[0].body) == 1 and not binds[0].orelse
    assert ast.unparse(binds[0].body[0]) == (
        "self.forward_decode_lanes = functools.partial(type(self)._forward_decode_lanes_fused, self)"
    )
    assert fused.enabled("gdn_step", {}) and not fused.enabled("gdn_step", {"QWEN38_FUSED_OFF": "gdn_step"})
    assert "gdn_step" in fused.DEFAULT_ON and fused.kernel("gdn_step").default_on
    assert "_recurrent_decode_lanes" in _self_walk(cls.forward_decode_lanes)
    assert "fused" not in _self_walk(cls.forward_decode_lanes) and "gdn_step" not in composed


def test_layer_and_model_lane_bodies_are_the_generic_walk_over_lanes() -> None:
    layer_walk = _self_walk(layer_module.Qwen38TTNNDecoderLayer.forward_decode_lanes)
    order = [
        "ple.inject_lanes",
        "attention_gr.read_rows",
        "attention.forward_decode_lanes",
        "attention.forward_decode_lanes",
        "attention_gr.write_rows",
        "mlp_gr.read_rows",
        "expert_streamer.layer",
        "mlp_gr.write_rows",
    ]
    assert [name for name in layer_walk if name in order] == order
    source = inspect.getsource(layer_module.Qwen38TTNNDecoderLayer.forward_decode_lanes)
    # The lane GR reads and the final mixer rows walk branch-major <-> flat rows through views (the same tile pages
    # for 1..32 rows: no permute or relayout kernel in the lane body); the rows forms keep the permutes by default.
    assert source.count("read_rows(residual, flat_views=True)") == 2
    for function in (gr_module.Qwen38TTNNGatedResidual.read_rows, mixer_module.Qwen38TTNNFinalMixer.rows):
        calls = _calls(function)
        assert calls.count("ttnn.experimental.view") == 2 and calls.count("ttnn.permute") == 2, function.__qualname__
        assert inspect.signature(function).parameters["flat_views"].default is False, function.__qualname__
    assert "state.moe.forward(mlp_input, packed_experts[0], packed_experts[1])" in source
    assert "state.ple.token_contexts = (None,) * lanes" in source  # the caller owns the n-gram contexts
    assert "output_tensor=state.attention_rows" in source  # the QSA 32-row output lands in the persistent rows
    model_walk = _self_walk(model_module.Qwen38TTNNTextModel.forward_decode_lanes)
    assert [name for name in model_walk if name.startswith(("_", "final_mixer", "model_io"))] == [
        "_validate_lane_state",
        "_position_derive_lanes",  # the fused position derive's lane form (the chain lines stay the fallback)
        "_embed_residual_lanes_from_device_token",
        "final_mixer.rows",
        "model_io.lm_head",
        "_mark_poisoned",
    ]
    model_source = inspect.getsource(model_module.Qwen38TTNNTextModel.forward_decode_lanes)
    assert "self.final_mixer.rows(residual, flat_views=True)" in model_source
    assert model_source.index("qsa_module.derive_qsa_lane_inputs(") < model_source.index("].forward_decode_lanes(")
    assert model_source.index("state.position.advance()") > model_source.index("model_io.lm_head(")
    capture = inspect.getsource(model_module.Qwen38TTNNTextModel.capture_decode_lanes)
    assert "state.position.residue != residue or set(phases.values()) != {residue}" in capture
    assert "corruptible_allocation_scope" in capture and "GENERIC_TRACE_PARTS_SINGLE" in capture
    admit = inspect.getsource(model_module.Qwen38TTNNTextModel.admit_lane)
    assert admit.index("state.position.admit(lane, position, rope_shift)") < admit.index(
        "layer.reset_lane_inplace(layer_state, lane)"
    )
    # The generic 1-row model body is untouched.
    assert "derive_qsa_position_inputs(state.position.scalar" in inspect.getsource(
        model_module.Qwen38TTNNTextModel.forward_decode_generic_tail
    )
    assert "lanes" not in inspect.getsource(layer_module.Qwen38TTNNDecoderLayer.forward_decode_generic)
