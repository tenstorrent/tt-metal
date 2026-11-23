# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop pass-2 two-bank sparse MLA (A2-A6) on device, against the reference given the device's own inputs.

Module level (``ttMLA(loop_pass2=True)``), not through prefill_runner: G2's 7-set handoff is not wired into the runner
yet (q171.2.8). Two loop layers on one 8x4 torus-xy mesh, tier-1 MLA weights (``glm_random_mla_weights``, the
draw ``glm54_tier1_weights`` makes; layer 2 owns an indexer here so pass 1 produces a second top-k set):

* pass 1: each layer is a full sparse layer writing its KVPE into the stage cache (bank A) and returning its top-k;
  the sets are copied into their own buffers, as G2 receives them (``tt.loop_handoff``);
* pass 2: each layer is a 'shared' pass-2 layer (no indexer) fed its group's set (``pass1_index_set``), reading bank A
  from the same cache (G2's copy is bit-exact to G1's, q171.2.7) and carrying its window between chunks explicitly.

``test_two_bank_device`` runs a schedule of chunks and checks, per chunk and layer: pass 2 leaves the pass-1 KVPE
cache (bank A) byte-identical; repeated forwards are bit-identical; every chip holds the same window rows; and
``LoopPass2MLAReference``, fed the device's own inputs (bank A as pass 1 left it, the top-k set, the carry, the hidden
states), matches the output (PCC >= 0.99 overall and over each SP shard's first W rows, which read the previous
shard's rows) and the window rows (PCC >= 0.998). The reference runs in the test, about 20 s per chunk and layer on
the host. ``test_two_bank_traced`` checks a captured pass-2 forward replays bit-identically to eager across chunk
starts; ``test_two_bank_index_list`` checks the device index list row for row.
"""

import copy

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import LoopPass2MLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import GLM54_SYNTHETIC_SEED
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm54_weights import glm54_reduced_config
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_mla_weights
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.two_bank import (
    WINDOW_FORMATS,
    TwoBankTable,
    chunk_row,
    pass1_index_set,
    pass2_indexer_types,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCacheGeometry,
    init_kvpe_cache,
    init_mla_kv_cache,
    striped_cache_to_host,
)

CHUNK = 5120
MAX_SEQ = 10240
LOOP_LAYERS = (1, 2)
OUTPUT_PCC = 0.99
WINDOW_PCC = 0.998
SENTINEL = 0xFFFFFFFF

# (start, end) chunks of one request; every chunk is a 5120-row slab, `end` its real tokens.
SCHEDULES = {
    "short100": [(0, 100)],  # < W: every row's window is short (compaction on every row)
    "short1500": [(0, 1500)],  # < index_topk: bank A has a sentinel tail on every row
    "two_chunks": [(0, CHUNK), (CHUNK, 2 * CHUNK)],  # full chunks: halo across 8 SP shards, carry across chunks
    "start96": [(0, 96), (96, 96 + CHUNK)],  # a 32-aligned non-chunk start < W: partial carry + compaction
}
FORMATS = list(WINDOW_FORMATS)


def _config(window_format: str):
    """Pass-1 config: the tier-1 reduced stack with loop layers 1-2 both owning an indexer (two top-k sets)."""
    c = glm54_reduced_config(MAX_SEQ)
    c.indexer_types = ["full"] * c.num_hidden_layers
    c.loop_window_format = window_format
    return c


def _pass2_config(config):
    c = copy.deepcopy(config)
    c.indexer_types = pass2_indexer_types(config)
    return c


def _hidden(layer: int, loop_pass: int, start: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * loop_pass + 10 * layer + start)
    return torch.randn(1, CHUNK, glm_5_4_hf_config().hidden_size, generator=g).to(torch.bfloat16)


# --- host-only checks (no device) --------------------------------------------------------------------------------


def test_pass1_index_set_maps_loop_layers_to_groups(expect_error):
    """GLM-5.4: pass-2 layers L26-53 reuse the 7 pass-1 sets in groups of 4 (full layers 26, 30, ..., 50)."""
    c = glm_5_4_hf_config()
    assert [pass1_index_set(c, L) for L in range(26, 54)] == [k for k in range(7) for _ in range(4)]
    assert set(pass2_indexer_types(c)) == {"shared"}
    with expect_error(ValueError, "outside the loop"):
        pass1_index_set(c, 25)


@pytest.mark.parametrize("start", [0, 96, 640 + 96, 3 * 640 + 608, 5120, 5120 + 7 * 640 + 32])
def test_chunk_row_inverts_the_rotated_layout(start):
    """chunk_row(p) is the SP gather's row holding position p: chip s's row i holds rotated_chunk_positions[s][i]."""
    sp, local = 8, 640
    positions = rotated_chunk_positions(start, sp, local)
    for s in range(sp):
        for i in range(local):
            assert chunk_row(positions[s][i], start, sp, local) == s * local + i


INDEX_STARTS = [0, 32, 96, 160, 640 + 96, 3 * 640 + 608, 5120, 5120 + 96, 2 * 5120 + 7 * 640 + 32, 99840 + 4064]


def expected_index_list(table, start: int, bank_a: torch.Tensor, query_rows: int, sp: int, tp: int) -> torch.Tensor:
    """Host index list of chip (s, t): per query row, its window's table rows (positions >= 0, ascending), then
    bank A's ids, then sentinels; bank_a is [sp, tp, rows, k] with sentinel tails. [sp, tp, rows, W + k]."""
    W, width = table.window, table.width
    positions = rotated_chunk_positions(start, sp, table.chunk_local)
    out = torch.full((sp, tp, query_rows, width), SENTINEL, dtype=torch.int64)
    for s in range(sp):
        for t in range(tp):
            for r in range(query_rows):
                p = positions[s][t * query_rows + r]
                window = [table.table_row(q, start) for q in range(p - W + 1, p + 1) if q >= 0]
                a = [v for v in bank_a[s, t, r].tolist() if v != SENTINEL]
                row = window + a
                out[s, t, r, : len(row)] = torch.tensor(row)
    return out


# --- device phase ---# --- device phase -----------------------------------------------------------------------------------------------


def _mesh_params():
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=GLM53Adapter.l1_small_size, trace_region_size=64 * 1024 * 1024),
            id="8x4-torus-xy",
        )
    ]


@pytest.mark.parametrize("window_format", ["scaled_fp8"])
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole mesh")
@pytest.mark.timeout(0)
def test_two_bank_index_list(mesh_device, window_format):
    """The device index list (window ids from the chunk start on device, rotated layout, compaction) equals the host
    expectation from rotated_chunk_positions and table_row, row for row, over aligned, mid-block, boundary-chip and
    late starts; bank A's ids keep their own sentinel tails."""
    config = _config(window_format)
    sp, tp = tuple(mesh_device.shape)
    rows, topk = CHUNK // (sp * tp), config.index_topk
    table = TwoBankTable(
        mesh_device,
        kv_format=WINDOW_FORMATS[window_format],
        geometry=MlaKvCacheGeometry.from_config(config),
        push_rows=4096,
        window=config.loop_window_size,
        chunk_global=CHUNK,
        query_rows=rows,
        topk=topk,
        sp_axis=0,
        tp_sequence_sharded=True,
        num_links=1,
    )
    g = torch.Generator().manual_seed(7)
    bank_a = torch.randint(0, 4096, (sp, tp, rows, topk), generator=g)
    valid = torch.randint(1, topk + 1, (sp, tp, rows, 1), generator=g)
    bank_a[torch.arange(topk).view(1, 1, 1, -1).expand_as(bank_a) >= valid] = SENTINEL
    tt_bank_a = ttnn.from_torch(
        (bank_a - (bank_a >= 2**31).long() * 2**32).to(torch.int32),  # uint32 bits through int32
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1)),
    )
    for start in INDEX_STARTS:
        got_tt = table.index_list(tt_bank_a, _scalar(mesh_device, start))
        got = ttnn.to_torch(
            got_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(sp, tp))
        )
        got = got.to(torch.int64) & 0xFFFFFFFF
        expect = expected_index_list(table, start, bank_a, rows, sp, tp)
        bad = (got != expect).any(-1).nonzero()
        assert not len(
            bad
        ), f"start {start}: {len(bad)} rows differ, first {bad[0].tolist()}: {got[tuple(bad[0])][:140].tolist()} vs {expect[tuple(bad[0])][:140].tolist()}"
        logger.info(f"index list at start {start}: all {sp * tp * rows} rows match")


class _Stage:
    """Pass-1 and pass-2 ttMLA for the loop layers over one cache, and host conversions."""

    def __init__(self, mesh_device, window_format: str):
        self.mesh = mesh_device
        self.config = _config(window_format)
        self.kv_format = WINDOW_FORMATS[window_format]
        self.sp_axis, self.tp_axis = 0, 1
        self.shape = tuple(mesh_device.shape)
        topology = per_axis_topology(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
        self.weights = {L: glm_random_mla_weights(self.config, L, GLM54_SYNTHETIC_SEED) for L in LOOP_LAYERS}
        common = dict(
            seq_len=MAX_SEQ,
            sp_axis=self.sp_axis,
            tp_axis=self.tp_axis,
            topology=topology,
            is_chunked=True,
            active_seq_len=CHUNK,
            layer_num=len(LOOP_LAYERS),
            first_layer_idx=LOOP_LAYERS[0],
            sparse_kv_cache_format=self.kv_format,
        )
        self.pass1 = {L: ttMLA(self.config, self.weights[L], mesh_device, layer_idx=L, **common) for L in LOOP_LAYERS}
        p2 = _pass2_config(self.config)
        self.pass2 = {
            L: ttMLA(p2, self.weights[L], mesh_device, layer_idx=L, loop_pass2=True, **common) for L in LOOP_LAYERS
        }
        self.cache = init_mla_kv_cache(
            cache_format=self.kv_format,
            hf_config=self.config,
            mesh_device=mesh_device,
            seq_len=MAX_SEQ,
            mesh_shape=list(self.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=len(LOOP_LAYERS),
            tp_axis=self.tp_axis,
        )
        self.index_cache = init_kvpe_cache(
            kvpe_cache_head_dim=self.config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=MAX_SEQ,
            mesh_shape=list(self.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=num_full_indexer_layers(self.config),
            dtype=ttnn.bfloat8_b,
            tp_axis=self.tp_axis,
        )
        self.rope = RotarySetup(
            self.config, mesh_device, sp_axis=self.sp_axis, is_balanced=False
        ).get_rope_tensors_indexed(MAX_SEQ, CHUNK)
        self.shard_dims = [None, None]
        self.shard_dims[self.tp_axis], self.shard_dims[self.sp_axis] = -1, -2

    def device_order(self, start: int) -> torch.Tensor:
        """For each device chunk row (SP shard s, row i, in s-major order) its position minus start: the rotated
        layout a chunk at `start` has on device (rotated_chunk_positions)."""
        positions = rotated_chunk_positions(start, self.shape[self.sp_axis], CHUNK // self.shape[self.sp_axis])
        return torch.tensor(positions).flatten() - start

    def to_device(self, hidden: torch.Tensor, start: int) -> ttnn.Tensor:
        """Natural-order chunk rows [1, CHUNK, D] -> the device's rotated SP layout."""
        return ttnn.from_torch(
            hidden[:, self.device_order(start)].unsqueeze(0),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=self.shape, dims=self.shard_dims),
        )

    def output_to_torch(self, t: ttnn.Tensor, start: int) -> torch.Tensor:
        """Device chunk rows (rotated layout) -> natural order [1, CHUNK, D]."""
        dev = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, dims=self.shard_dims, mesh_shape=self.shape)
        ).to(torch.bfloat16)[0]
        out = torch.empty_like(dev)
        out[:, self.device_order(start)] = dev
        return out

    def per_chip(self, t: ttnn.Tensor) -> torch.Tensor:
        """[sp, tp, ...] host copy, one block per chip."""
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, dims=(0, 1), mesh_shape=self.shape))

    def rows_logical(self, physical: torch.Tensor) -> torch.Tensor:
        """Physical table/cache rows [..., row] -> logical bf16 [..., 576]."""
        return self.pass2[LOOP_LAYERS[0]]._two_bank.table().unpack_host(physical).to(torch.bfloat16)

    def cache_snapshot(self) -> torch.Tensor:
        """The whole pass-1 KVPE cache (bank A), physical rows in stripe order [layers, MAX_SEQ, row]."""
        return striped_cache_to_host(self.cache.storage, self.mesh, self.sp_axis)

    def bank_a(self, snapshot: torch.Tensor, layer: int) -> torch.Tensor:
        """Layer `layer`'s pass-1 KVPE from a cache snapshot, natural order, logical bf16 [MAX_SEQ, 576]."""
        stripes = self.shape[0] * self.shape[1]
        flat = snapshot[LOOP_LAYERS.index(layer)]
        nat = torch.empty_like(flat)
        nat[blockcyclic_positions(stripes, CHUNK, flat.shape[0])] = flat
        return self.rows_logical(nat)

    def indices(self, t: ttnn.Tensor, start: int) -> torch.Tensor:
        """A top-k set: chip (s, t) holds SP-shard rows t*rows..; -> natural row order [1, 1, CHUNK, k] int64."""
        host = (self.per_chip(t).to(torch.int64) & 0xFFFFFFFF).reshape(CHUNK, -1)  # device row order
        out = torch.empty_like(host)
        out[self.device_order(start)] = host
        return out.view(1, 1, CHUNK, -1)

    def window_natural(self, layer: int, window: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """The window region's chunk rows (`window` = window_rows() host rows) in position order, start .. end-1."""
        table = self.pass2[layer]._two_bank
        return window[[table.table_row(p, start) - table.window_base for p in range(start, end)]]

    def replicated(self, t: ttnn.Tensor, first_row: int = 0) -> torch.Tensor:
        """A replicated tensor's host copy, checked identical on every chip from row `first_row` on (rows of positions
        before 0 were never written and hold whatever each chip's allocation held)."""
        chips = self.per_chip(t)
        flat = chips[:, :, first_row:].reshape(-1, *chips.shape[3:]).contiguous().view(torch.uint8)
        flat = flat.view(chips.shape[0] * chips.shape[1], -1)
        for i in range(1, flat.shape[0]):
            assert torch.equal(flat[i], flat[0]), f"replicated tensor differs on chip {i}"
        return chips[0, 0]

    def pass1_chunk(self, start: int, end: int) -> list[ttnn.Tensor]:
        """Pass 1 over the chunk: writes bank A, returns the loop's top-k sets (own copies, as G2 receives them)."""
        sets = []
        for L in LOOP_LAYERS:
            _, idx = self.pass1[L].forward(
                self.to_device(_hidden(L, 1, start), start),
                self.rope,
                self.cache,
                cache_layer_idx=LOOP_LAYERS.index(L),
                actual_start=start,
                actual_end=end,
                index_kv_cache=self.index_cache,
                return_indexer_indices=True,
            )
            sets.append(ttnn.clone(idx))
        return sets

    def pass2_forward(self, layer, start, end, sets, carry, hidden=None, metadata=None):
        return self.pass2[layer].forward(
            hidden if hidden is not None else self.to_device(_hidden(layer, 2, start), start),
            self.rope,
            self.cache,
            cache_layer_idx=LOOP_LAYERS.index(layer),
            actual_start=None if metadata is not None else start,
            actual_end=None if metadata is not None else end,
            indexer_indices=sets[pass1_index_set(self.config, layer)],
            window_carry=carry,
            metadata=metadata,
        )


@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("window_format", FORMATS)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (sparse SDPA, push) are Blackhole-only")
@pytest.mark.timeout(0)
def test_two_bank_device(mesh_device, window_format, schedule):
    """Run the schedule; per chunk and layer check bank A untouched by pass 2, determinism, window replication, and
    the output and window against the reference given the device's own inputs."""
    stage = _Stage(mesh_device, window_format)
    window = stage.config.loop_window_size
    carries = {L: None for L in LOOP_LAYERS}
    for start, end in SCHEDULES[schedule]:
        sets = stage.pass1_chunk(start, end)
        before = stage.cache_snapshot()  # bank A as pass 1 left it: the reference's bank A
        set_rows = [stage.indices(s, start) for s in sets]
        # Window region row i is position start - (W - 1) + i; rows of positions < 0 are never written.
        real_from = max(0, window - 1 - start)
        results = {}
        for L in LOOP_LAYERS:
            carry_in = None if carries[L] is None else stage.rows_logical(stage.replicated(carries[L], real_from))
            out = stage.output_to_torch(stage.pass2_forward(L, start, end, sets, carries[L]), start)
            rows = stage.replicated(stage.pass2[L]._two_bank.window_rows(), real_from)
            # Determinism: the same forward again (same carry, same bank A, same set) is bit-identical.
            again = stage.output_to_torch(stage.pass2_forward(L, start, end, sets, carries[L]), start)
            assert torch.equal(out, again), f"layer {L} chunk {start}: repeated pass-2 forward differs"
            assert torch.equal(
                stage.replicated(stage.pass2[L]._two_bank.window_rows(), real_from)[real_from:].view(torch.uint8),
                rows[real_from:].view(torch.uint8),
            ), f"layer {L} chunk {start}: repeated window differs"
            results[L] = (out, stage.rows_logical(stage.window_natural(L, rows, start, end)), carry_in)
            carries[L] = stage.pass2[L].window_carry(start, end)
        ttnn.synchronize_device(mesh_device)
        after = stage.cache_snapshot()
        assert torch.equal(after.view(torch.uint8), before.view(torch.uint8)), (
            f"chunk {start}: pass 2 changed the pass-1 KVPE cache (bank A) in "
            f"{int((after.view(torch.uint8) != before.view(torch.uint8)).any(-1).sum())} rows"
        )
        for L, (out, window_rows, carry_in) in results.items():
            _check_vs_reference(
                stage.config,
                f"{window_format} {schedule}",
                L,
                start,
                end,
                out=out,
                window_rows=window_rows,
                carry_in=carry_in,
                bank_a=stage.bank_a(before, L)[:end],
                indices=set_rows[pass1_index_set(stage.config, L)],
            )
        for t in sets:
            ttnn.deallocate(t)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _check_vs_reference(config, case, layer, start, end, *, out, window_rows, carry_in, bank_a, indices):
    """LoopPass2MLAReference on the device's inputs vs the device's output (overall and each SP shard's first W rows,
    which read the previous shard's rows through the halo) and its window rows (positions start .. end-1)."""
    window, shard, n = config.loop_window_size, CHUNK // 8, end - start
    mla_ref = SparseMLAReference(config, glm_random_mla_weights(config, layer, GLM54_SYNTHETIC_SEED), seq_len=MAX_SEQ)
    ref = LoopPass2MLAReference(mla_ref, window=window, latent_dim=config.kv_lora_rank)
    need = min(window - 1, start)
    res = ref.forward(
        _hidden(layer, 2, start)[:, :n],
        kvpe_a=bank_a,
        indices_a=indices[..., :n, :],
        actual_start=start,
        window_carry=None if carry_in is None else carry_in[window - 1 - need :],
    )
    got = out[:, :n]
    out_pcc = _pearson(got, res.output)
    window_pcc = _pearson(window_rows, res.kvpe)
    shard_pccs = [
        _pearson(got[:, s : min(s + window, n)], res.output[:, s : min(s + window, n)]) for s in range(shard, n, shard)
    ]
    logger.info(
        f"[{case}] L{layer} chunk {start}-{end}: output PCC {out_pcc:.5f}, window PCC {window_pcc:.5f}, "
        f"shard-start PCC min {min(shard_pccs, default=float('nan')):.5f}"
    )
    # The shard halo first: a row near a shard start reads the previous shard's rows (all-gathered).
    for k, p in enumerate(shard_pccs, start=1):
        assert p >= OUTPUT_PCC, f"L{layer} chunk {start}: SP shard {k}'s first {window} rows PCC {p:.5f}"
    assert out_pcc >= OUTPUT_PCC, f"L{layer} chunk {start}: output PCC {out_pcc:.5f} < {OUTPUT_PCC}"
    assert window_pcc >= WINDOW_PCC, f"L{layer} chunk {start}: window PCC {window_pcc:.5f} < {WINDOW_PCC}"


def _scalar(mesh_device, value: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([[[[value]]]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("window_format", FORMATS)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (sparse SDPA, push) are Blackhole-only")
@pytest.mark.timeout(0)
def test_two_bank_traced(mesh_device, window_format):
    """One pass-2 forward captured at chunk start 5120 replays bit-identically to eager (metadata path) at start 5120
    and at start 0 (every row compacted): the push slot and plan start, the index-list compaction and the window
    writes read metadata and persistent inputs on device."""
    stage = _Stage(mesh_device, window_format)
    L = LOOP_LAYERS[0]
    chunks = [(0, CHUNK), (CHUNK, 2 * CHUNK)]
    # Eager over both chunks (metadata path); keep each chunk's inputs to replay them.
    inputs, eager = {}, {}
    row = stage.pass2[L]._two_bank.row_width
    # The metadata path always takes a carry; a chunk at start 0 reads none of its rows.
    carry = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            shape=ttnn.Shape([1, 1, stage.config.loop_window_size - 1, row]),
            dtype=stage.kv_format.storage_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_device,
    )
    for start, end in chunks:
        sets = stage.pass1_chunk(start, end)
        metadata = [_scalar(mesh_device, v) for v in (0, start, end)]
        hidden = stage.to_device(_hidden(L, 2, start), start)
        out = stage.pass2_forward(L, start, end, sets, carry, hidden=hidden, metadata=metadata)
        eager[start] = stage.output_to_torch(out, start)
        inputs[start] = (sets, carry)
        carry = stage.pass2[L].window_carry(start, end)
    # Persistent inputs the capture reads; their contents change between replays.
    p_meta = [_scalar(mesh_device, v) for v in (0, chunks[1][0], chunks[1][1])]
    p_hidden = stage.to_device(_hidden(L, 2, chunks[1][0]), chunks[1][0])
    p_sets = [ttnn.clone(s) for s in inputs[chunks[1][0]][0]]
    p_carry = ttnn.clone(inputs[chunks[1][0]][1])
    assert tuple(p_carry.shape) == (1, 1, stage.config.loop_window_size - 1, row)
    stage.pass2_forward(L, None, None, p_sets, p_carry, hidden=p_hidden, metadata=p_meta)  # warm at capture shapes
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = stage.pass2_forward(L, None, None, p_sets, p_carry, hidden=p_hidden, metadata=p_meta)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        for start, end in (chunks[1], chunks[0], chunks[1]):
            sets, carry = inputs[start]
            for dst, v in zip(p_meta, (0, start, end)):
                ttnn.copy(_scalar(mesh_device, v), dst)
            ttnn.copy(stage.to_device(_hidden(L, 2, start), start), p_hidden)
            for src, dst in zip(sets, p_sets):
                ttnn.copy(src, dst)
            if carry is not None:
                ttnn.copy(carry, p_carry)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            got = stage.output_to_torch(out, start)
            assert torch.equal(got, eager[start]), f"traced replay at start {start} differs from eager"
            logger.info(f"[{window_format}] traced replay at start {start}: bit-identical to eager")
    finally:
        ttnn.release_trace(mesh_device, tid)
