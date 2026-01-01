# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The batched-lanes QSA layer without a device: B lanes at their own positions bitwise the B 1-row steps.

Both bodies run on a torch-backed ttnn whose ops are deterministic per row (float64 rounded once to the op's dtype),
so no result depends on the tensor shape it sits in.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.attention import rope_tp as rope_module
from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    ROW_MAJOR,
    TILE,
    TP,
    FakeContract,
    FakeTensor,
    bits_equal,
    copy,
    make_qsa_ttnn,
    ordered_calls,
    patch_ttnn,
    replicated,
    sharded,
    use_tile_padding,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MAX_LANES,
    Qwen38TTNNDevicePosition,
    Qwen38TTNNDevicePositionRow,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    CACHE_WRITE_ROWS,
    COMPRESS_RATIO,
    HEAD_DIM,
    HIDDEN_SIZE,
    INDEX_HEAD_DIM,
    INDEXER_FORMS,
    LOCAL_QUERY_WIDTH,
    MASKED_INDEX,
    ROPE_DIM,
    Qwen38TTNNQSA,
    Qwen38TTNNQSAChunkConstants,
    Qwen38TTNNQSALaneConstants,
    Qwen38TTNNQSALaneState,
    Qwen38TTNNQSAPositionConstants,
    Qwen38TTNNQSAWeights,
    derive_qsa_lane_inputs,
    derive_qsa_position_inputs,
)

ALLOCATED_CONTEXT = 32768
BLOCKS = ALLOCATED_CONTEXT // COMPRESS_RATIO
# One lane per 4k span of the 32k cache, all in residue class 3 (the block-closing compressed write is real), each in
# its own 32-row block.
LANE_POSITIONS = (2051, 6147, 10243, 14339, 18435, 22531, 26627, 30723)
LANE_COUNTS = (1, 2, 4, 8)
STEPS = 2
LAYER = 3


# --------------------------------------------------------------------------- the QSA fake


class FakeMesh:
    def arch(self):
        return "BLACKHOLE"

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)


@pytest.fixture
def fake(monkeypatch):
    # The fake ttnn runs the op chains only, so the six QSA programs (on by default) are switched off.
    monkeypatch.setenv(
        "QWEN38_FUSED_OFF",
        "qsa_index_tail,qsa_main_tail,qsa_post_attention,qsa_widen_partial,qsa_selection_row,qsa_score_merge",
    )
    use_tile_padding(monkeypatch)
    fake = make_qsa_ttnn()
    patch_ttnn(monkeypatch, fake, qsa_module, contracts_module, model_module, rope_module, decode_matmul_module)
    # the fake's bf16 is the dense plan's bf16 for the projection weights (dense_dtype_tag / the fidelity name)
    monkeypatch.setitem(decode_matmul_module.DENSE_DTYPE_TAGS, BF16, "bf16")
    monkeypatch.setitem(decode_matmul_module.DENSE_MATH_FIDELITY_NAMES, BF16, "HiFi4")
    return fake


# --------------------------------------------------------------------------- the layer under the fake


def _dram_sharded_weight(host: torch.Tensor, dim: int, k: int, n: int) -> FakeTensor:
    """A decode linear's weight in the one-reader DRAM bank layout the module validates (decode_matmul)."""

    tensor = sharded(host, dim)
    tensor.memory_config_value = decode_matmul_module.dram_sharded_weight_memory_config(FakeMesh(), k, n)
    return tensor


def _weights(generator) -> Qwen38TTNNQSAWeights:
    def randn(*shape, scale):
        return torch.randn(*shape, generator=generator) * scale

    return Qwen38TTNNQSAWeights(
        source_kind="backbone",
        layer_index=LAYER,
        qg=_dram_sharded_weight(
            randn(1, 1, HIDDEN_SIZE, TP * 2 * LOCAL_QUERY_WIDTH, scale=0.02), 3, HIDDEN_SIZE, 2 * LOCAL_QUERY_WIDTH
        ),
        k_pair_grouped=sharded(randn(1, 1, HIDDEN_SIZE, TP * HEAD_DIM, scale=0.02), 3),
        v_pair_grouped=sharded(randn(1, 1, HIDDEN_SIZE, TP * HEAD_DIM, scale=0.02), 3),
        out=_dram_sharded_weight(
            randn(1, 1, TP * LOCAL_QUERY_WIDTH, HIDDEN_SIZE, scale=0.02), 2, LOCAL_QUERY_WIDTH, HIDDEN_SIZE
        ),
        q_norm=replicated(1.0 + randn(1, 1, 1, HEAD_DIM, scale=0.1), BF16),
        k_norm=replicated(1.0 + randn(1, 1, 1, HEAD_DIM, scale=0.1), BF16),
        index_q=sharded(randn(1, 1, HIDDEN_SIZE, TP * INDEX_HEAD_DIM, scale=0.02), 3),
        index_k=replicated(randn(1, 1, HIDDEN_SIZE, INDEX_HEAD_DIM, scale=0.02), BF16),
        index_q_norm=replicated(1.0 + randn(1, 1, 1, INDEX_HEAD_DIM, scale=0.1), BF16),
        index_k_norm=replicated(1.0 + randn(1, 1, 1, INDEX_HEAD_DIM, scale=0.1), BF16),
        weight_dtype=BF16,
    )


class _Layer:
    """The QSA layer, its RoPE table, the position constants and the warm cache images, all on the fake."""

    def __init__(self, seed: int = 20260903) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.contract = FakeContract()
        self.mesh = FakeMesh()
        self.qsa = Qwen38TTNNQSA(
            self.mesh, self.contract, _weights(self.generator), layer_index=LAYER, allocated_context=ALLOCATED_CONTEXT
        )
        cos, sin = (torch.randn(1, 1, ALLOCATED_CONTEXT, ROPE_DIM, generator=self.generator) for _ in range(2))
        self.rope_table = model_module.Qwen38TTNNRoPETable(
            replicated(cos, BF16, ROW_MAJOR), replicated(sin, BF16, ROW_MAJOR), ALLOCATED_CONTEXT, None, self.contract
        )
        self.position_constants = Qwen38TTNNQSAPositionConstants.build(self.mesh, self.contract, BLOCKS)
        self.chunk_constants = Qwen38TTNNQSAChunkConstants.build(self.mesh, self.contract, BLOCKS)
        self.kv_host = torch.randn(1, TP, ALLOCATED_CONTEXT, 2 * HEAD_DIM, generator=self.generator).to(torch.bfloat16)
        self.compressed_host = torch.cat(
            [
                torch.randn(1, 1, BLOCKS, INDEX_HEAD_DIM, generator=self.generator).to(torch.bfloat16),
                torch.zeros(1, 1, 32, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            ],
            dim=2,
        )

    def hidden(self, rows: int) -> torch.Tensor:
        return torch.randn(1, 1, rows, HIDDEN_SIZE, generator=self.generator).to(torch.bfloat16)

    def write_lanes(self, state: Qwen38TTNNQSALaneState) -> None:
        lanes = state.lanes
        for host, target in (
            (self.kv_host.repeat(1, 1, lanes, 1), state.packed_kv_cache),
            (self.compressed_host.repeat(lanes, TP, 1, 1), state.compressed_index_cache),
            (torch.zeros(1, TP * lanes, CACHE_WRITE_ROWS, 2 * HEAD_DIM), state.kv_staging),
            (torch.zeros(1, TP * lanes, CACHE_WRITE_ROWS, INDEX_HEAD_DIM), state.raw_key_ring),
        ):
            copy(sharded(host, 1), target)


def _shards(tensor: FakeTensor) -> list[torch.Tensor]:
    return [x.clone() for x in tensor.torch_shards()]


ROW_FIELDS = (
    "index_query",
    "raw_key",
    "masked_scores",
    "sparse_indices",
    "q",
    "gate",
    "k",
    "v",
    "local_attention",
    "output",
)


def _lane_view(name: str, shard: torch.Tensor, lane: int, position: int) -> torch.Tensor:
    """Lane u's 1-row image out of a lane-body field: row u of a rows tensor (the sparse ids with lane u's KV offset
    taken back off every non-sentinel slot), tile u of the slot tensors, user u of the compressed cache, lane u's
    block of the flat KV cache."""

    if name == "sparse_indices":
        row = shard[:, :, lane : lane + 1]
        return torch.where(row == MASKED_INDEX, row, row - lane * ALLOCATED_CONTEXT)
    if name in ROW_FIELDS:
        return shard[:, :, lane : lane + 1]
    if name in ("staging", "ring"):
        return shard[:, lane : lane + 1]
    if name == "compressed_cache":
        return shard[lane : lane + 1]
    assert name == "kv_cache"
    start = lane * ALLOCATED_CONTEXT + (position & ~(CACHE_WRITE_ROWS - 1))
    return shard[:, :, start : start + CACHE_WRITE_ROWS]


@pytest.mark.parametrize(
    "lanes,indexer_form", [(1, "wide"), (2, "wide"), (4, "wide"), (8, "wide"), (2, "per_lane"), (8, "per_lane")]
)
def test_lane_body_rows_and_regions_are_bitwise_the_one_row_steps(fake, lanes: int, indexer_form: str) -> None:
    layer = _Layer()
    qsa, constants = layer.qsa, layer.chunk_constants
    positions = LANE_POSITIONS[:lanes]
    lane_constants = Qwen38TTNNQSALaneConstants.build(
        layer.mesh, layer.contract, lanes=lanes, allocated_context=ALLOCATED_CONTEXT, indexer_form=indexer_form
    )
    hidden = [layer.hidden(lanes) for _ in range(STEPS)]  # step t: row u is lane u's hidden row

    # The B 1-row chains: lane u's STEPS steps of forward_decode_generic stage by stage on its own warm caches
    # (staging and ring empty at the start); every intermediate and the caches after it.
    reference: list[list[dict[str, list[torch.Tensor]]]] = [[] for _ in positions]
    generic = qsa.allocate_generic_state()
    for lane, position in enumerate(positions):
        for host, target in (
            (layer.kv_host, generic.packed_kv_cache),
            (layer.compressed_host.expand(1, TP, -1, -1), generic.compressed_index_cache),
            (torch.zeros(1, TP, CACHE_WRITE_ROWS, 2 * HEAD_DIM), generic.kv_staging),
            (torch.zeros(1, TP, CACHE_WRITE_ROWS, INDEX_HEAD_DIM), generic.raw_key_ring),
        ):
            copy(sharded(host, 1), target)
        for step in range(STEPS):
            scalar = Qwen38TTNNDevicePosition.allocate(layer.mesh, layer.contract, position=position + step)
            inputs = derive_qsa_position_inputs(scalar.scalar, layer.position_constants)
            index_row = scalar.index_row()
            rope = layer.rope_table.rows(index_row, scalar.block_start_index_row(index_row))
            fields: dict[str, list[torch.Tensor]] = {}
            full_hidden = qsa._all_gather_hidden(sharded(hidden[step][:, :, lane : lane + 1], 3))
            index_query, raw_key = qsa._index_projection(full_hidden, rope.cos, rope.sin)
            fields["index_query"], fields["raw_key"] = _shards(index_query), _shards(raw_key)
            qsa._write_compressed_index_generic(generic, raw_key, inputs, rope.block_start_cos, rope.block_start_sin)
            masked_scores = qsa._score_blocks_generic(index_query, generic, inputs)
            fields["masked_scores"] = _shards(masked_scores)
            sparse_indices = qsa._materialize_row_generic(masked_scores, inputs)
            fields["sparse_indices"] = _shards(sparse_indices)
            query, gate, key, value = qsa._main_projection(full_hidden, rope.cos, rope.sin)
            fields.update(q=_shards(query), gate=_shards(gate), k=_shards(key), v=_shards(value))
            qsa._write_packed_kv_generic(generic, key, value, inputs)
            local_attention = qsa._sparse_value_attention(query, gate, sparse_indices, generic)
            fields["local_attention"] = _shards(local_attention)
            fields["output"] = _shards(qsa._project_output(local_attention, full_hidden))
            start = (position + step) & ~(CACHE_WRITE_ROWS - 1)
            fields["kv_cache"] = [x[:, :, start : start + CACHE_WRITE_ROWS] for x in _shards(generic.packed_kv_cache)]
            fields["compressed_cache"] = _shards(generic.compressed_index_cache)
            fields["staging"], fields["ring"] = _shards(generic.kv_staging), _shards(generic.raw_key_ring)
            reference[lane].append(fields)
    qsa.release_generic_state(generic)

    # The lane body on the same warm caches, forward_decode_lanes stage by stage in the body's own order, every lane
    # in its own region, the row advanced between the steps.
    state = qsa.allocate_lane_state(lanes)
    layer.write_lanes(state)
    row = Qwen38TTNNDevicePositionRow.allocate(layer.mesh, layer.contract, positions, lanes=lanes)
    for step in range(STEPS):
        rows = torch.zeros(1, 1, MAX_LANES, HIDDEN_SIZE, dtype=torch.bfloat16)
        rows[:, :, :lanes] = hidden[step]
        inputs = derive_qsa_lane_inputs(row.row, layer.position_constants, constants, lane_constants)
        index_row = row.index_row()
        rope = layer.rope_table.rows_chunk(index_row, row.block_start_index_row(index_row))
        fields = {}
        full_hidden = qsa._all_gather_hidden_rows(sharded(rows, 3), constants)
        index_query, raw_key = qsa._index_projection_rows(full_hidden, None, rope.cos, rope.sin, constants)
        fields["index_query"], fields["raw_key"] = _shards(index_query), _shards(raw_key)
        qsa._write_compressed_index_lanes(
            state, raw_key, rope.block_start_cos, rope.block_start_sin, inputs, lane_constants
        )
        masked_scores = qsa._score_blocks_lanes(index_query, state, inputs, lane_constants)
        fields["masked_scores"] = _shards(masked_scores)
        sparse_indices = qsa._materialize_rows_lanes(masked_scores, inputs, constants, lane_constants)
        fields["sparse_indices"] = _shards(sparse_indices)
        query, gate, key, value = qsa._main_projection_rows(full_hidden, None, rope.cos, rope.sin, constants)
        fields.update(q=_shards(query), gate=_shards(gate), k=_shards(key), v=_shards(value))
        qsa._write_packed_kv_lanes(state, key, value, inputs)
        local_attention = qsa._sparse_value_attention_rows(query, gate, sparse_indices, state, constants)
        fields["local_attention"] = _shards(local_attention)
        fields["output"] = _shards(qsa._project_output_rows(local_attention, full_hidden, constants))
        # The caches themselves (read before the next step mutates them): no copies of the B x 32k regions.
        fields["kv_cache"] = list(state.packed_kv_cache.torch_shards())
        fields["compressed_cache"] = list(state.compressed_index_cache.torch_shards())
        fields["staging"] = list(state.kv_staging.torch_shards())
        fields["ring"] = list(state.raw_key_ring.torch_shards())
        inputs.deallocate()
        for lane, position in enumerate(positions):
            expected = reference[lane][step]
            for name in (*ROW_FIELDS, "kv_cache", "compressed_cache", "staging", "ring"):
                for coordinate, (got, want) in enumerate(zip(fields[name], expected[name])):
                    view = _lane_view(name, got, lane, position + step)
                    assert bits_equal(view, want), (
                        f"B={lanes} {indexer_form} step {step} lane {lane} (P={position + step}) {name} coordinate "
                        f"{coordinate}: lane {tuple(view.shape)} {view.flatten()[:6].tolist()} vs 1-row "
                        f"{tuple(want.shape)} {want.flatten()[:6].tolist()}"
                    )
        # Every other lane's region is untouched by lane u's writes: the KV rows outside every lane's block stay warm.
        for coordinate, cache in enumerate(fields["kv_cache"]):
            warm = layer.kv_host[:, coordinate : coordinate + 1].repeat(1, 1, lanes, 1)
            touched = torch.zeros(lanes * ALLOCATED_CONTEXT, dtype=torch.bool)
            for lane, position in enumerate(positions):
                start = lane * ALLOCATED_CONTEXT + ((position + step) & ~(CACHE_WRITE_ROWS - 1))
                touched[start : start + CACHE_WRITE_ROWS] = True
            assert bits_equal(cache[:, :, ~touched], warm[:, :, ~touched]), f"coordinate {coordinate}"
        row.advance()
    assert row.positions[:lanes] == [position + STEPS for position in positions]
    qsa.release_lane_state(state)
    lane_constants.deallocate()


def test_the_fused_lane_body_takes_its_score_windows_from_qsa_score_merge(expect_error, fake, monkeypatch) -> None:
    """The fused lane body is bound only with qsa_score_merge's score windows, under every admitted switch subset."""

    names = model_module.QSA_LANE_FUSED_KERNELS
    weights = _weights(torch.Generator().manual_seed(7))
    bodies = []
    for mask in range(1 << len(names)):
        off = [name for bit, name in enumerate(names) if mask >> bit & 1]
        monkeypatch.setenv("QWEN38_FUSED_OFF", ",".join(off))
        if "qsa_main_tail" in off and "qsa_post_attention" not in off:
            with expect_error(ValueError, match="enable qsa_main_tail too"):  # allow-pytest.raises: the refusal
                Qwen38TTNNQSA(
                    FakeMesh(), FakeContract(), weights, layer_index=LAYER, allocated_context=ALLOCATED_CONTEXT
                )
            continue
        layer = Qwen38TTNNQSA(
            FakeMesh(), FakeContract(), weights, layer_index=LAYER, allocated_context=ALLOCATED_CONTEXT
        )
        windows = layer._lane_score_rows_of is fused.qsa_block.lane_score_rows_of
        body = getattr(layer.forward_decode_lanes, "func", None) is Qwen38TTNNQSA._forward_decode_lanes_fused
        assert windows == ("qsa_score_merge" not in off), off
        assert windows or not body, off
        bodies.append(body)
    assert any(bodies) and not all(bodies), bodies


def test_lane_state_contract_and_the_entry_point_run_the_layer(expect_error, fake) -> None:
    layer = _Layer()
    qsa = layer.qsa
    state = qsa.allocate_lane_state(8)
    assert state.lanes == 8 and state.layer_index == LAYER
    assert state.packed_kv_cache.shape == (1, 1, 8 * ALLOCATED_CONTEXT, 2 * HEAD_DIM)
    assert state.packed_kv_cache.layout == ROW_MAJOR and state.packed_kv_cache.dtype is BF16
    assert state.compressed_index_cache.shape == (8, 1, BLOCKS + 32, INDEX_HEAD_DIM)
    assert state.compressed_index_cache_flat.shape == (1, 1, 8 * (BLOCKS + 32), INDEX_HEAD_DIM)
    assert state.kv_staging.shape == (1, 8, CACHE_WRITE_ROWS, 2 * HEAD_DIM) and state.kv_staging.layout == TILE
    assert state.raw_key_ring.shape == (1, 8, CACHE_WRITE_ROWS, INDEX_HEAD_DIM)
    # The flat view reads the users back to back: user u's rows are rows [u*R, (u+1)*R) of the view.
    layer.write_lanes(state)
    flat = fake.experimental.view(state.compressed_index_cache, (1, 1, 8 * (BLOCKS + 32), INDEX_HEAD_DIM))
    for user in range(8):
        rows = flat.torch_shards()[0][0, 0, user * (BLOCKS + 32) : (user + 1) * (BLOCKS + 32)]
        assert bits_equal(rows, state.compressed_index_cache.torch_shards()[0][user, 0])
    for x in state.kv_staging.locals:
        x.fill_(2.0)
    qsa.reset_lane_state_inplace(state)
    assert all(float(x.abs().sum()) == 0.0 for x in state.kv_staging.locals + state.raw_key_ring.locals)
    with expect_error(ValueError, match=r"QSA lane count must be in \[1,32\], got 33"):  # allow-pytest.raises
        qsa.allocate_lane_state(33)
    foreign = Qwen38TTNNQSALaneState(**{**state.__dict__, "layer_index": LAYER + 1})
    with expect_error(ValueError, match="belongs to layer 4"):  # allow-pytest.raises: pure contract test
        qsa._validate_lane_state(foreign)
    with expect_error(ValueError, match="epoch"):  # allow-pytest.raises: pure contract test
        qsa._validate_lane_state(Qwen38TTNNQSALaneState(**{**state.__dict__, "epoch": 10**6}))
    qsa.release_lane_state(state)
    with expect_error(ValueError, match="was not allocated"):  # allow-pytest.raises: released epochs are dead
        qsa.release_lane_state(state)

    # The entry point runs the layer on a 2-lane state; the lane count of the constants, the inputs and the state
    # must agree. The residue rule lives at the position row (test_batched_lanes_no_device): the derive is
    # residue-agnostic.
    two = qsa.allocate_lane_state(2)
    layer.write_lanes(two)
    lane_constants = {
        lanes: Qwen38TTNNQSALaneConstants.build(
            layer.mesh, layer.contract, lanes=lanes, allocated_context=ALLOCATED_CONTEXT
        )
        for lanes in (2, 4)
    }
    row = Qwen38TTNNDevicePositionRow.allocate(layer.mesh, layer.contract, [2051, 6147], lanes=2)
    inputs = derive_qsa_lane_inputs(row.row, layer.position_constants, layer.chunk_constants, lane_constants[2])
    index_row = row.index_row()
    rope = layer.rope_table.rows_chunk(index_row, row.block_start_index_row(index_row))
    operands = {
        "cos": rope.cos,
        "sin": rope.sin,
        "block_start_cos": rope.block_start_cos,
        "block_start_sin": rope.block_start_sin,
        "lanes": inputs,
        "constants": layer.chunk_constants,
    }
    with expect_error(ValueError, match="lane constants were built for 4 lanes"):  # allow-pytest.raises: contract
        qsa.forward_decode_lanes(
            sharded(torch.zeros(1, 1, MAX_LANES, HIDDEN_SIZE), 3), two, lane_constants=lane_constants[4], **operands
        )
    with expect_error(ValueError, match="carry 2 slab rows, expected 4"):  # allow-pytest.raises: contract
        qsa._validate_lane_inputs(inputs, 4)
    output = qsa.forward_decode_lanes(
        sharded(layer.hidden(MAX_LANES), 3), two, lane_constants=lane_constants[2], **operands
    )
    assert output.shape == (1, 1, MAX_LANES, HIDDEN_SIZE // TP) and output.dtype is BF16 and output.layout == TILE
    assert all(torch.isfinite(x.float()).all() for x in output.torch_shards())
    inputs.deallocate()
    qsa.release_lane_state(two)


# --------------------------------------------------------------------------- source pins

LANE_BODY = (
    "forward_decode_lanes",
    "_write_compressed_index_lanes",
    "_score_blocks_lanes",
    "_materialize_rows_lanes",
    "_write_packed_kv_lanes",
)


def _ttnn_op_walk(name: str) -> list[str]:
    """A QSA method's ttnn ops by name in source order, without the deallocations."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(Qwen38TTNNQSA, name))))
    ops = [call.rsplit(".", 1)[-1] for call in ordered_calls(tree) if call.startswith("ttnn.")]
    return [op for op in ops if op not in ("deallocate", "Shape")]


def test_lane_body_is_the_decode_walk_on_the_lane_layout_and_the_other_bodies_are_untouched() -> None:
    for name in LANE_BODY:
        source = inspect.getsource(getattr(Qwen38TTNNQSA, name))
        for forbidden in ("from_torch", "to_torch", ".item()", "copy_host_to_device_tensor", "synchronize", ".shape["):
            assert forbidden not in source, (name, forbidden)
        for argument in ast.parse(textwrap.dedent(source)).body[0].args.args:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else ""
            assert annotation != "int", (name, argument.arg)
    body = inspect.getsource(Qwen38TTNNQSA.forward_decode_lanes)
    order = (
        "self._all_gather_hidden_rows(hidden_rows, constants)",
        "self._index_projection_rows(full_hidden, None, cos, sin, constants)",
        "self._write_compressed_index_lanes(state, raw_key, block_start_cos, block_start_sin, lanes, lane_constants)",
        "self._score_blocks_lanes(index_query, state, lanes, lane_constants)",
        "self._materialize_rows_lanes(masked_scores, lanes, constants, lane_constants)",
        "self._main_projection_rows(full_hidden, None, cos, sin, constants)",
        "self._write_packed_kv_lanes(state, key, value, lanes)",
        "self._sparse_value_attention_rows(query, gate, sparse_indices, state, constants)",
        "self._project_output_rows(local_attention, full_hidden, constants)",
    )
    found = [body.index(fragment) for fragment in order]
    assert found == sorted(found)
    # The compressed write: one selection matmul, the in-place ring add, the 1-row reduce per lane tile, the row
    # RoPE, one paged_update_cache for every user; no per-lane loop.
    compressed = _ttnn_op_walk("_write_compressed_index_lanes")
    assert compressed.count("paged_update_cache") == 1 and compressed.count("matmul") == 1
    assert compressed.count("sum") == 1 and "for lane in range" not in inspect.getsource(
        Qwen38TTNNQSA._write_compressed_index_lanes
    )
    assert compressed.count("view") == 2 and compressed.count("to_layout") == 4
    # The KV write: one selection matmul, the in-place staging add, one untilize, then per lane one slab slice and
    # one update_padded_kv_cache read from the lane's metadata scalar (the last three in ``_write_kv_slabs_lanes``).
    kv = _ttnn_op_walk("_write_packed_kv_lanes") + _ttnn_op_walk("_write_kv_slabs_lanes")
    assert kv.count("matmul") == 1 and kv.count("update_padded_kv_cache") == 1 and kv.count("to_layout") == 1
    kv_source = inspect.getsource(Qwen38TTNNQSA._write_packed_kv_lanes) + inspect.getsource(
        Qwen38TTNNQSA._write_kv_slabs_lanes
    )
    assert "lanes.kv_row_start_lanes" in kv_source and "row_starts[lane]" in kv_source and "self.slot_zero" in kv_source
    # The scores: one wide indexer on the flat view or one per lane with cache_batch_idx, lane u's row sliced.
    score_source = inspect.getsource(Qwen38TTNNQSA._score_blocks_lanes)
    assert "state.compressed_index_cache_flat" in score_source and "cache_batch_idx=lane" in score_source
    assert (
        'lane_constants.indexer_form == "wide"' in score_source
        and _ttnn_op_walk("_score_blocks_lanes").count("all_reduce") == 1
    )
    materialize = _ttnn_op_walk("_materialize_rows_lanes")
    assert materialize == [
        "topk_large_indices",
        "bitwise_left_shift",
        "repeat_interleave",
        "add",
        "concat",
        "bitwise_and",
        "bitwise_or",
    ]
    assert "lane_constants.block_offsets_lanes" in inspect.getsource(Qwen38TTNNQSA._materialize_rows_lanes)
    # The 1-row generic body, its helpers and the chunk body are not edited by the lane path.
    for name in (
        "forward_decode_generic",
        "_write_compressed_index_generic",
        "_score_blocks_generic",
        "_materialize_row_generic",
        "_write_packed_kv_generic",
        "forward_chunk_generic",
        "_write_compressed_index_chunk",
        "_score_blocks_chunk",
        "_materialize_rows_chunk",
        "_write_packed_kv_chunk",
        "_sparse_value_attention_rows",
        "_project_output_rows",
        "_main_projection_rows",
        "_index_projection_rows",
        "_all_gather_hidden_rows",
    ):
        source = inspect.getsource(getattr(Qwen38TTNNQSA, name))
        assert "lane" not in source and "Lane" not in source, name
    assert INDEXER_FORMS == ("wide", "per_lane")
    assert tuple(Qwen38TTNNQSALaneState.__dataclass_fields__) == (
        "layer_index",
        "epoch",
        "lanes",
        "packed_kv_cache",
        "compressed_index_cache",
        "compressed_index_cache_flat",
        "kv_staging",
        "raw_key_ring",
        "kv_scratch_rows",  # the MTP lane verify's redirect region past the last lane (0 for the plain lanes)
    )
