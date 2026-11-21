# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The batched-lanes GDN step and PLE lanes, the lane resets, the final mixer rows and the lane bodies' source pins.

The fake computes every matmul per (lane, head) tile in float64 rounded once, so no result depends on the batch it
sits in; B separate 1-row chains are the reference.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    FP32,
    ROW_MAJOR,
    TILE,
    TP,
    FakeContract,
    FakeTensor,
    assert_rows_bitwise,
    bf16_randn,
    bits_equal,
    cat_shards,
    device_gdn_weights,
    function_calls,
    gdn_oracle_weights,
    hidden_sharded,
    make_gdn_lanes_ttnn,
    make_gdn_module,
    make_gr_ttnn,
    make_ple_module,
    ordered_calls,
    patch_ttnn,
    residual_rows,
    use_tile_padding,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import GDN_RESIDUE_CLASSES
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import (
    FLAT_LOCAL_WIDTH,
    RESIDUAL_RANK,
    Qwen38TTNNFinalMixer,
    Qwen38TTNNFinalMixerWeights,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gdn_step_module

STEPS = 12
ADMISSION_STEP = 8  # phase 0 again: a fresh lane (position 0) joins at a step of its residue class
LANE_COUNTS = (1, 2, 4, 8)
HOST_IO = ("ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor", "torch.")


# --------------------------------------------------------------------------- the fake


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("QWEN38_FUSED_OFF", "ple")  # the fused PLE lane body serves by default; the fake runs the chain
    fake_ttnn = make_gdn_lanes_ttnn()
    patch_ttnn(monkeypatch, fake_ttnn, gdn_module, ple_module)
    return fake_ttnn


@pytest.fixture(scope="module")
def gdn():
    return make_gdn_module(device_gdn_weights(gdn_oracle_weights()))


def _lane_image(state: gdn_module.Qwen38TTNNGDNState, lane: int) -> dict[str, torch.Tensor]:
    """Lane u of a lanes state in the 1-row state's shapes (the recurrent over the four head shards, the ring rows)."""

    recurrent = cat_shards(state.recurrent, 1)[lane : lane + 1]
    slots = [cat_shards(slot, 3)[:, :, lane : lane + 1] for slot in state.conv]
    return {"recurrent": recurrent, **{f"conv[{index}]": slot for index, slot in enumerate(slots)}}


# --------------------------------------------------------------------------- the GDN lanes step against B chains


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
        result = gdn.forward_decode_lanes(hidden_sharded(hidden[step : step + 1]), lane_state)
        assert result.state is lane_state and lane_state.conv_phase == (step + 1) % GDN_RESIDUE_CLASSES
        output_lanes = cat_shards(result.hidden_sharded, 3)
        assert output_lanes.shape == (1, 1, lanes, 2560)
        for lane, chain in enumerate(chains):
            reference = gdn.forward_decode(hidden_sharded(hidden[step : step + 1, lane : lane + 1]), chain)
            assert chain.conv_phase == lane_state.conv_phase
            reference_hidden = cat_shards(reference.hidden_sharded, 3)
            assert bits_equal(output_lanes[:, :, lane : lane + 1], reference_hidden), (step, lane)
            lane_image, chain_image = _lane_image(lane_state, lane), _lane_image(chain, 0)
            for name, value in chain_image.items():
                assert bits_equal(lane_image[name], value), (step, lane, name)
    if lanes > 1:
        # The lanes differ from each other (the comparison is not vacuous) and the 1-row body refuses the state.
        assert not bits_equal(_lane_image(lane_state, 0)["recurrent"], _lane_image(lane_state, 1)["recurrent"])
        with expect_error(ValueError, match=f"the 1-row GDN decode body admits a batch-1 state, got batch {lanes}"):
            gdn.forward_decode(hidden_sharded(hidden[:1, :1]), lane_state)


def test_misaligned_admission_differs_from_a_fresh_session(fake, gdn) -> None:
    # The position row refuses this admission (test_batched_lanes_no_device); here is why.
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
        gdn.forward_decode_lanes(hidden_sharded(hidden[step : step + 1]), lane_state)
        if fresh is not None:
            gdn.forward_decode(hidden_sharded(hidden[step : step + 1, 1:2]), fresh)
    lane_image, fresh_image = _lane_image(lane_state, 1), _lane_image(fresh, 0)
    assert not all(bits_equal(lane_image[name], fresh_image[name]) for name in fresh_image)


# --------------------------------------------------------------------------- GDN state, lane resets, the snapshot at B


def test_gdn_lane_state_layout_and_lane_reset_keep_the_other_lanes_bitwise(fake, gdn, expect_error) -> None:
    lanes = 4
    torch.manual_seed(3)
    state = gdn.allocate_lane_state(lanes)
    assert state.recurrent.shape == (lanes, 12, 128, 128) and state.recurrent.dtype is FP32
    assert all(slot.shape == (1, 1, lanes, 2560) and slot.dtype is BF16 for slot in state.conv)
    one = gdn.allocate_state()
    assert one.batch_size == 1 and one.recurrent.shape == (1, 12, 128, 128) and one.conv[0].shape == (1, 1, 1, 2560)
    for bad in (0, 33):
        with expect_error(ValueError, match=r"GDN state batch size must be in \[1,32\]"):  # allow-pytest.raises
            gdn_module.Qwen38TTNNGDNState.allocate("mesh", gdn.mesh_contract, layer_index=0, batch_size=bad)
    state.batch_size = lanes + 1
    with expect_error(
        RuntimeError,
        match=r"GDN recurrent state \(batch 5\) local shape must be \(5, 12, 128, 128\), got \(4, 12, 128, 128\)",
    ):
        state.validate()
    state.batch_size = lanes
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
                assert bits_equal(value, before[lane][name]), (lane, name)
    for bad in (-1, 4, True, 1.0):
        with expect_error(ValueError, match=r"lane must be an int in \[0,4\)"):  # allow-pytest.raises: contract
            state.reset_lane_inplace(bad)
    # The snapshot follows the batch: capture, mutate, restore round trip.
    snapshot = state.allocate_snapshot()
    assert snapshot.batch_size == lanes and snapshot.recurrent.shape == (lanes, 12, 128, 128)
    state.capture_into(snapshot)
    state.reset_inplace()
    assert state.conv_phase == 0 and len(state.conv_window()) == 4
    assert all(int(torch.count_nonzero(v)) == 0 for lane in range(lanes) for v in _lane_image(state, lane).values())
    state.restore_from(snapshot)
    for lane in range(lanes):
        for name, value in _lane_image(state, lane).items():
            assert bits_equal(value, before[lane][name]) if lane != 2 else int(torch.count_nonzero(value)) == 0, name
    snapshot.batch_size = 1
    with expect_error(RuntimeError, match=r"GDN snapshot recurrent state local shape must be \(1, 12, 128, 128\)"):
        snapshot.validate(state.mesh_contract)


# --------------------------------------------------------------------------- PLE lanes


@pytest.mark.parametrize("lanes", [1, 3])
def test_ple_lanes_equal_the_one_row_path_per_lane_with_independent_contexts(fake, expect_error, lanes: int) -> None:
    """Lane u is the 1-row path on its own stream, bitwise (at B = 1 a same-shape reshape is a view)."""

    module = make_ple_module()
    steps = 3
    streams = ((17, 15, 16), (16, 95859, 17), (20, 17, 17))  # lane u's tokens, step by step
    torch.manual_seed(24)
    residual = torch.randn(steps, lanes, 4, 2560).to(torch.bfloat16)

    # 1-row references: one fresh state per lane, threaded through its nine slots and its host context.
    states = [module.allocate_state() for _ in range(lanes)]
    deltas = [[None] * lanes for _ in range(steps)]
    contexts = [[None] * lanes for _ in range(steps)]
    for step in range(steps):
        for lane, state in enumerate(states):
            prepared = module.prepare_decode_input(torch.tensor([[streams[lane][step]]], dtype=torch.long), state)
            result = module.forward_prepared(residual_rows(residual[step, lane : lane + 1]), prepared, state)
            deltas[step][lane] = cat_shards(result.residual_delta, 3)
            contexts[step][lane] = tuple(int(v) for v in state.token_context[0])

    lanes_state = module.allocate_lanes_state(lanes)
    assert lanes_state.token_contexts == (None,) * lanes and lanes_state.conv[0].shape == (1, lanes, 4, 640)
    for step in range(steps):
        prepared = module.prepare_lanes_input([streams[lane][step] for lane in range(lanes)], lanes_state)
        assert prepared.embedding_rows.shape == (1, 1, lanes, 640) and prepared.embedding_rows.layout == ROW_MAJOR
        delta_lanes = cat_shards(module.forward_prepared_lanes(residual_rows(residual[step]), prepared, lanes_state), 3)
        assert delta_lanes.shape == (1, lanes, 4, 2560)
        for lane in range(lanes):
            assert bits_equal(delta_lanes[:, lane : lane + 1], deltas[step][lane]), (step, lane)
        assert lanes_state.token_contexts == tuple(contexts[step]), step
        prepared.release()
    for index in range(ple_module.CONV_STATE_LENGTH):
        lane_slots = cat_shards(lanes_state.conv[index], 3)
        for lane, state in enumerate(states):
            assert bits_equal(lane_slots[:, lane : lane + 1], cat_shards(state.conv[index], 3)), (index, lane)
    if lanes == 1:
        return
    # The streams differ, so the per-lane comparison is not vacuous.
    assert not torch.equal(deltas[2][1], deltas[2][2])

    # The lane reset zeroes lane 1 and clears its context; the other lanes keep their slots bitwise.
    before = [cat_shards(slot, 3) for slot in lanes_state.conv]
    lane_contexts = lanes_state.token_contexts
    lanes_state.reset_lane_inplace(1)
    for index, slot in enumerate(lanes_state.conv):
        after = cat_shards(slot, 3)
        assert int(torch.count_nonzero(after[:, 1])) == 0, index
        assert bits_equal(after[:, 0], before[index][:, 0]) and bits_equal(after[:, 2], before[index][:, 2]), index
    assert lanes_state.token_contexts == (lane_contexts[0], None, lane_contexts[2])
    with expect_error(ValueError, match=r"PLE lane must be an int in \[0,3\), got 3"):  # allow-pytest.raises
        lanes_state.reset_lane_inplace(3)

    # Contracts: token count, stale prepared input, lane count.
    with expect_error(ValueError, match=r"needs 3 tokens \(one per lane\), got 2"):
        module.prepare_lanes_input([1, 2], lanes_state)
    stale = module.prepare_lanes_input([1, 2, 3], lanes_state)
    lanes_state.token_contexts = ((1, 1),) * lanes
    with expect_error(RuntimeError, match="prepared PLE lanes were looked up from contexts"):
        module.forward_prepared_lanes(residual_rows(residual[0]), stale, lanes_state)
    with expect_error(ValueError, match=r"PLE lanes must be in \[1,32\], got 33"):  # allow-pytest.raises: contract
        module.allocate_lanes_state(33)


def test_ple_inject_lanes_adds_the_permuted_delta_to_the_branch_major_rows(expect_error, fake) -> None:
    module = make_ple_module()
    lanes = 2
    torch.manual_seed(25)
    residual = torch.randn(lanes, 4, 2560).to(torch.bfloat16)
    tokens = [17, 20]
    lanes_state = module.allocate_lanes_state(lanes)
    prepared = module.prepare_lanes_input(tokens, lanes_state)
    delta = cat_shards(module.forward_prepared_lanes(residual_rows(residual), prepared, lanes_state), 3)
    fresh = module.allocate_lanes_state(lanes)
    branch_major = residual.reshape(1, lanes, 4, 2560).permute(0, 2, 1, 3).contiguous()
    injected = module.inject_lanes(
        FakeTensor([p.clone() for p in torch.chunk(branch_major, TP, dim=3)], BF16, TILE, 3),
        module.prepare_lanes_input(tokens, fresh),
        fresh,
    )
    assert injected.shape == (1, 4, lanes, 640)
    expected = (branch_major.float() + delta.permute(0, 2, 1, 3).float()).to(torch.bfloat16)
    assert torch.equal(cat_shards(injected, 3).view(torch.int16), expected.view(torch.int16))
    with expect_error(ValueError, match=r"lanes residual must be BF16 TILE \(1, 4, 2, 640\)"):
        module.inject_lanes(residual_rows(residual), module.prepare_lanes_input(tokens, fresh), fresh)


# --------------------------------------------------------------------------- final mixer rows


@pytest.mark.parametrize("rows", [1, 3, 8, 32])
def test_final_mixer_rows_equal_the_one_row_mixer_row_for_row(monkeypatch, rows: int, expect_error) -> None:
    use_tile_padding(monkeypatch)
    fake = make_gr_ttnn()
    monkeypatch.setattr(mixer_module, "ttnn", fake)
    torch.manual_seed(9)
    width, rank = FLAT_LOCAL_WIDTH, RESIDUAL_RANK
    weights = Qwen38TTNNFinalMixerWeights(
        norm_scale=FakeTensor([(torch.randn(1, 4, 1, 640) * 0.1 + 0.25) for _ in range(TP)], FP32, TILE, 3),
        down=FakeTensor([bf16_randn(1, 1, width, rank, scale=width**-0.5) for _ in range(TP)], BF16, TILE, 2),
        up=FakeTensor([bf16_randn(1, 1, rank, width, scale=rank**-0.5) for _ in range(TP)], BF16, TILE, 3),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        namespace="backbone",
    )
    module = object.__new__(Qwen38TTNNFinalMixer)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = weights
    module.collective_topology = "linear"
    module.compute_config = "compute"
    module.weight_compute_config = "weight_compute"  # the down / up linears (decode_matmul fidelity)
    module.down_act_memory_config = fake.DRAM_MEMORY_CONFIG
    module.down_program_config = "down"
    module.up_act_memory_config = fake.DRAM_MEMORY_CONFIG
    module.up_program_config = "up"
    module.norm_scale_flat = fake.experimental.view(weights.norm_scale, (1, 1, 1, width))
    torch.manual_seed(20 + rows)
    residual = FakeTensor([bf16_randn(1, 4, rows, 640) for _ in range(TP)], BF16, TILE, 3)
    output = module.rows(residual)
    assert output.shape == (1, 1, rows, 640) and output.dtype is BF16
    row_inputs = [
        FakeTensor([x.narrow(2, row, 1).clone() for x in residual.torch_shards()], BF16, TILE) for row in range(rows)
    ]
    per_row = [module(row_input) for row_input in row_inputs]
    assert_rows_bitwise(output, per_row, dim=2, label=f"final mixer {rows} rows")
    with expect_error(ValueError, match=r"final mixer residual rows must be in \[1,32\], got 33"):
        module.rows(FakeTensor([bf16_randn(1, 4, 33, 640) for _ in range(TP)], BF16, TILE, 3))
    with expect_error(ValueError, match=r"must be rank 4"):  # allow-pytest.raises: contract
        module.rows(FakeTensor([bf16_randn(4, 1, 640) for _ in range(TP)], BF16, TILE, 2))


# --------------------------------------------------------------------------- source pins


def _self_walk(function, receiver: str = "self") -> list[str]:
    calls = ordered_calls(ast.parse(textwrap.dedent(inspect.getsource(function))))
    return [call.removeprefix(f"{receiver}.") for call in calls if call.startswith(f"{receiver}.")]


def test_gdn_lanes_body_is_the_one_row_walk_with_the_lanes_suffix_and_the_one_row_body_is_untouched() -> None:
    cls = gdn_module.Qwen38TTNNGDN
    one_row = [name for name in _self_walk(cls.forward_decode) if name.startswith("_")]
    lanes = [name for name in _self_walk(cls.forward_decode_lanes) if name.startswith("_")]
    # The 1-row body resolves its projection -> gated-output step through the fused registry; the composed chain is
    # the walk the lane stages mirror.
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
    # The same ttnn ops per stage: the lane stage is the 1-row stage on B rows, nothing added or removed (the z/a/b
    # slices are one call site in a loop on the lanes side).
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
        one_calls, lane_calls = (
            {call for name in names for call in function_calls(getattr(cls, name)) if call.startswith("ttnn.")}
            for names in (one_names, lane_names)
        )
        assert one_calls == lane_calls, (one_names, lane_names)
    assert _self_walk(cls._project_lanes)[0] == "_project_lanes_unsplit"
    assert _self_walk(cls._gate_and_project_lanes)[-1] == "_out_project_lanes"
    # The lane bodies touch no host: no host I/O, no per-lane host ints.
    for function in (
        cls.forward_decode_lanes,
        *(getattr(cls, name) for name in lanes),
        cls._project_lanes_unsplit,
        cls._out_project_lanes,
        cls._forward_decode_lanes_fused,
        layer_module.Qwen38TTNNDecoderLayer.forward_decode_lanes,
        model_module.Qwen38TTNNTextModel.forward_decode_lanes,
        model_module.Qwen38TTNNTextModel._embed_residual_lanes_from_device_token,
        model_module.Qwen38TTNNTextModel.resolve_lane_tokens,
        mixer_module.Qwen38TTNNFinalMixer.rows,
        contracts_module.Qwen38TTNNDevicePositionRow.advance,
        contracts_module.Qwen38TTNNDevicePositionRow.index_row,
        embedding_module.resolve_greedy_lanes_on_device,
        ple_module.Qwen38TTNNPLE.forward_prepared_lanes,
        ple_module.Qwen38TTNNPLE._convolve_lanes,
    ):
        assert not any(call.startswith(HOST_IO) for call in function_calls(function)), function.__qualname__
    assert "Advance one true-global-B1 token" in inspect.getsource(cls.forward_decode)
    assert "admits a batch-1 state" in inspect.getsource(cls._validate_state)
    assert "lanes" not in inspect.getsource(cls._recurrent_decode)
    # The lane resets are keep-mask multiplies landed in place (eager host uploads), never the trace body's ops.
    for function in (
        gdn_module.Qwen38TTNNGDNState.reset_lane_inplace,
        ple_module.Qwen38TTNNPLELanesState.reset_lane_inplace,
        qsa_module.Qwen38TTNNQSA.reset_lane_inplace,
    ):
        calls = function_calls(function)
        assert "ttnn.multiply" in calls and "ttnn.from_torch" in calls, function
        assert "ttnn.copy" in calls or "_copy_inplace" in calls, function


def test_gdn_fused_lanes_body_is_the_one_row_fused_walk_on_b_rows_and_binds_under_the_switch() -> None:
    """The fused lane body is ``forward_decode``'s walk around the ``gdn_step`` program at rows = B, in the composed
    lanes body's ring order; it is bound at construction only when the kernel is on."""

    cls = gdn_module.Qwen38TTNNGDN
    fused_body = cls._forward_decode_lanes_fused
    assert [name for name in _self_walk(fused_body) if name.startswith("_")] == [
        "_validate_lane_state",
        "_all_gather_hidden_lanes",
        "_project_lanes_unsplit",
        "_out_project_lanes",
    ]
    calls = function_calls(fused_body)
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
    unsplit = function_calls(cls._project_lanes_unsplit)
    assert "ttnn.slice" not in unsplit and unsplit.count("ttnn.linear") == 1
    assert function_calls(cls._out_project_lanes).count("ttnn.reduce_scatter") == 1
    assert "lanes" in fused.kernel("gdn_step").replaces and "_forward_decode_lanes_fused" in gdn_step_module.__doc__
    # The binding: the one ``if fused.enabled("gdn_step")`` of the constructor assigns the partial over the class
    # attribute; QWEN38_FUSED_OFF=gdn_step unbinds it and the class body runs the lane chain.
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
    # for 1..32 rows: no permute or relayout kernel in the lane body); the rows forms keep the permutes by default
    # (the GR rows pins: test_gr_rows_no_device).
    assert source.count("read_rows(residual, flat_views=True)") == 2
    mixer_calls = function_calls(mixer_module.Qwen38TTNNFinalMixer.rows)
    assert mixer_calls.count("ttnn.experimental.view") == 2 and mixer_calls.count("ttnn.permute") == 2
    assert inspect.signature(mixer_module.Qwen38TTNNFinalMixer.rows).parameters["flat_views"].default is False
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
