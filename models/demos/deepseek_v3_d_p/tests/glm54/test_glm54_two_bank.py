# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop pass-2 two-bank sparse MLA (A2-A6) with its window carried across chunks and user slots, on device.

Module level (``ttMLA(loop_pass2=True)``, ``forward(loop_pass2=True)``), as G2 builds it: two loop layers on one
8x4 torus-xy mesh, tier-1 MLA weights (``glm_random_mla_weights``, the draw ``glm54_tier1_weights`` makes; layer 2
owns an indexer here so pass 1 produces a second top-k set). Per chunk:

* pass 1: each layer is a full sparse layer writing its KVPE into its slot's pass-1 rows (bank A) and returning its
  top-k; the sets are copied into their own buffers, as G2 receives them (``tt.loop_handoff``);
* pass 2: each layer is a 'shared' pass-2 layer (no indexer) fed its group's set (``pass1_index_set``), reading bank A
  and writing its own KV only to the slot's window ring (two_bank.py), which carries the window to the slot's next
  chunk.

``test_two_bank_device`` runs a schedule of chunks of two user slots interleaved, and checks: pass 2 leaves every
slot's pass-1 rows byte-identical and the other slot's ring untouched; repeated forwards are bit-identical; and, per
slot at the end, ``LoopPass2MLAReference`` run single-shot over the slot's whole prompt (on the device's bank A and
top-k sets) matches the chunked device output (PCC >= 0.99 per chunk and over each SP shard's first W rows, which read
the previous shard's rows) and the ring's rows (PCC >= 0.998). The reference is chunk-invariant (q171.2.2), so this is
chunked vs single-shot. ``test_two_bank_traced`` replays one captured pass-2 forward over the whole interleaved
schedule bit-identically to eager; ``test_two_bank_index_list`` checks the device index list row for row.
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
    TwoBankIndex,
    pass1_index_set,
    pass2_indexer_types,
    ring_id,
    ring_rows,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache, init_mla_kv_cache, striped_cache_to_host

CHUNK = 5120
MAX_POSITIONS = 3 * CHUNK  # pass-1 positions per slot
CAPACITY = MAX_POSITIONS + ring_rows(CHUNK)  # cache positions per slot: pass 1, then the window ring
RING_BASE = MAX_POSITIONS
LOOP_LAYERS = (1, 2)
NUM_SLOTS = 2
OUTPUT_PCC = 0.99
WINDOW_PCC = 0.998
SENTINEL = 0xFFFFFFFF

# (slot, start, end) chunks in execution order; every chunk is a 5120-row slab, `end` its real tokens.
SCHEDULES = {
    # Slot 0: full chunks then a partial one (boundaries at 5120 and 10240; positions past the ring period 10240).
    # Slot 1: a 96-token chunk, then chunks starting mid-slab (rotated layout, partial carry at 96).
    "two_slots": [(0, 0, 5120), (1, 0, 96), (0, 5120, 10240), (1, 96, 5216), (0, 10240, 12000), (1, 5216, 9000)],
    "short": [(0, 0, 100), (1, 0, 1500)],  # < W and < index_topk: every row's window or top-k is short
}
FORMATS = list(WINDOW_FORMATS)


def _config(window_format: str):
    """The tier-1 reduced stack with loop layers 1-2 both owning an indexer (two top-k sets)."""
    c = glm54_reduced_config(CAPACITY)
    c.indexer_types = ["full"] * c.num_hidden_layers
    c.loop_window_format = window_format
    return c


def _pass2_config(config):
    c = copy.deepcopy(config)
    c.indexer_types = pass2_indexer_types(config)
    return c


def _prompt(slot: int, layer: int, loop_pass: int, length: int) -> torch.Tensor:
    """A slot's input hidden states of one layer and pass over its whole prompt, natural order [1, length, D]."""
    g = torch.Generator().manual_seed(1000 * loop_pass + 100 * slot + layer)
    return torch.randn(1, length, glm_5_4_hf_config().hidden_size, generator=g).to(torch.bfloat16)


def _chunk_rows(slot, layer, loop_pass, start, end) -> torch.Tensor:
    """One chunk's rows of the slot's prompt, zero-padded to the slab [1, CHUNK, D]."""
    rows = torch.zeros(1, CHUNK, glm_5_4_hf_config().hidden_size, dtype=torch.bfloat16)
    rows[:, : end - start] = _prompt(slot, layer, loop_pass, end)[:, start:end]
    return rows


# --- host-only checks (no device) --------------------------------------------------------------------------------


def test_pass1_index_set_maps_loop_layers_to_groups(expect_error):
    """GLM-5.4: pass-2 layers L26-53 reuse the 7 pass-1 sets in groups of 4 (full layers 26, 30, ..., 50)."""
    c = glm_5_4_hf_config()
    assert [pass1_index_set(c, L) for L in range(26, 54)] == [k for k in range(7) for _ in range(4)]
    assert set(pass2_indexer_types(c)) == {"shared"}
    with expect_error(ValueError, "outside the loop"):
        pass1_index_set(c, 25)


RING_SCHEDULES = {
    "aligned": [(0, 5120), (5120, 10240), (10240, 15360), (15360, 17000)],
    "start96": [(0, 96), (96, 5216), (5216, 9000), (9000, 14120), (14120, 19240)],
    "start4064": [(0, 4064), (4064, 9184), (9184, 14304), (14304, 19424), (19424, 20000)],
    "late": [(0, 5120), (5120, 10208), (10208, 15328)],  # start mod P = P - 32: the second copy ends at 2P + chunk - 32
}


@pytest.mark.parametrize("schedule", list(RING_SCHEDULES))
def test_ring_canonical_rows_hold_every_window_position(schedule):
    """Simulate a slot's ring writes (two copies per chunk): before each chunk's attention, the canonical row of every
    window position p - W + 1 .. p it reads holds that position, written by this or an earlier chunk; no write leaves
    the ring."""
    index = TwoBankIndex.__new__(TwoBankIndex)  # host math only: ring_writes needs ring_base and period
    index.ring_base, index.period, W = 0, 2 * CHUNK, 128
    held = {}
    for start, end in RING_SCHEDULES[schedule]:
        for v, ve in index.ring_writes(start, end):
            assert ve <= ring_rows(CHUNK), f"chunk {start}: write [{v}, {ve}) leaves the ring"
            for k in range(ve - v):
                held[v + k] = start + k
        for p in range(start, end):
            for q in range(max(0, p - W + 1), p + 1):
                got = held.get(ring_id(q, 0, CHUNK))
                assert got == q, f"chunk {start}: position {q} reads ring row {ring_id(q, 0, CHUNK)} holding {got}"


INDEX_STARTS = [0, 32, 96, 160, 640 + 96, 3 * 640 + 608, 5120, 5120 + 96, 2 * 5120 + 7 * 640 + 32, 99840 + 4064]


def expected_index_list(index, start: int, topk: torch.Tensor, query_rows: int, sp: int, tp: int) -> torch.Tensor:
    """Host index list of chip (s, t): per query row, its window's ring ids (positions >= 0, ascending), then its top-k,
    then sentinels; topk is [sp, tp, rows, k] with sentinel tails. [sp, tp, rows, W + k]."""
    W, width = index.window, index.width
    positions = rotated_chunk_positions(start, sp, index.chunk_local)
    out = torch.full((sp, tp, query_rows, width), SENTINEL, dtype=torch.int64)
    for s in range(sp):
        for t in range(tp):
            for r in range(query_rows):
                p = positions[s][t * query_rows + r]
                window = [ring_id(q, index.ring_base, index.chunk_global) for q in range(p - W + 1, p + 1) if q >= 0]
                a = [v for v in topk[s, t, r].tolist() if v != SENTINEL]
                row = window + a
                out[s, t, r, : len(row)] = torch.tensor(row)
    return out


# --- device phase -----------------------------------------------------------------------------------------------


def _mesh_params():
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=GLM53Adapter.l1_small_size, trace_region_size=64 * 1024 * 1024),
            id="8x4-torus-xy",
        )
    ]


def _scalar(mesh_device, value: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([[[[value]]]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole mesh")
@pytest.mark.timeout(0)
def test_two_bank_index_list(mesh_device):
    """The device index list (ring ids of each query row's window from the chunk start on device, rotated layout,
    compaction) equals the host expectation from rotated_chunk_positions and ring_id, row for row, over aligned,
    mid-block, boundary-chip and late starts; the top-k keeps its own sentinel tails."""
    config = _config("scaled_fp8")
    sp, tp = tuple(mesh_device.shape)
    rows, topk = CHUNK // (sp * tp), config.index_topk
    index = TwoBankIndex(
        mesh_device,
        ring_base=RING_BASE,
        window=config.loop_window_size,
        chunk_global=CHUNK,
        query_rows=rows,
        topk=topk,
        sp_axis=0,
        tp_sequence_sharded=True,
    )
    g = torch.Generator().manual_seed(7)
    sets = torch.randint(0, RING_BASE, (sp, tp, rows, topk), generator=g)
    valid = torch.randint(1, topk + 1, (sp, tp, rows, 1), generator=g)
    sets[torch.arange(topk).view(1, 1, 1, -1).expand_as(sets) >= valid] = SENTINEL
    tt_sets = ttnn.from_torch(
        (sets - (sets >= 2**31).long() * 2**32).to(torch.int32),  # uint32 bits through int32
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1)),
    )
    for start in INDEX_STARTS:
        got = ttnn.to_torch(
            index.index_list(tt_sets, _scalar(mesh_device, start), _scalar(mesh_device, start + CHUNK)),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(sp, tp)),
        )
        got = got.to(torch.int64) & 0xFFFFFFFF
        expect = expected_index_list(index, start, sets, rows, sp, tp)
        bad = (got != expect).any(-1).nonzero()
        assert not len(bad), (
            f"start {start}: {len(bad)} rows differ, first {bad[0].tolist()}: "
            f"{got[tuple(bad[0])][:140].tolist()} vs {expect[tuple(bad[0])][:140].tolist()}"
        )
        logger.info(f"index list at start {start}: all {sp * tp * rows} rows match")


class _Stage:
    """Pass-1 and pass-2 ttMLA for the loop layers over one cache of NUM_SLOTS user slots, and host conversions."""

    def __init__(self, mesh_device, window_format: str):
        self.mesh = mesh_device
        self.config = _config(window_format)
        self.kv_format = WINDOW_FORMATS[window_format]
        self.sp_axis, self.tp_axis = 0, 1
        self.shape = tuple(mesh_device.shape)
        topology = per_axis_topology(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
        self.weights = {L: glm_random_mla_weights(self.config, L, GLM54_SYNTHETIC_SEED) for L in LOOP_LAYERS}
        common = dict(
            seq_len=CAPACITY,
            sp_axis=self.sp_axis,
            tp_axis=self.tp_axis,
            topology=topology,
            is_chunked=True,
            active_seq_len=CHUNK,
            slot_num=NUM_SLOTS,
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
            seq_len=CAPACITY,
            mesh_shape=list(self.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=len(LOOP_LAYERS),
            num_users=NUM_SLOTS,
            tp_axis=self.tp_axis,
        )
        self.index_cache = init_kvpe_cache(
            kvpe_cache_head_dim=self.config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=CAPACITY,
            mesh_shape=list(self.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=num_full_indexer_layers(self.config),
            num_users=NUM_SLOTS,
            dtype=ttnn.bfloat8_b,
            tp_axis=self.tp_axis,
        )
        self.rope = RotarySetup(
            self.config, mesh_device, sp_axis=self.sp_axis, is_balanced=False
        ).get_rope_tensors_indexed(CAPACITY, CHUNK)
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

    def cache_natural(self) -> torch.Tensor:
        """The whole KVPE cache, physical rows in natural position order [slots, layers, CAPACITY, row]."""
        flat = striped_cache_to_host(self.cache.storage, self.mesh, self.sp_axis)  # [slots*layers, CAPACITY, row]
        nat = torch.empty_like(flat)
        nat[:, blockcyclic_positions(self.shape[0] * self.shape[1], CHUNK, flat.shape[1])] = flat
        return nat.view(NUM_SLOTS, len(LOOP_LAYERS), CAPACITY, -1)

    def logical(self, physical: torch.Tensor) -> torch.Tensor:
        """Physical cache rows [..., row] -> logical bf16 [..., 576]."""
        return self.cache.unpack_host(physical).to(torch.bfloat16)

    def indices(self, t: ttnn.Tensor, start: int) -> torch.Tensor:
        """A top-k set: chip (s, t) holds SP-shard rows t*rows..; -> natural row order [CHUNK, k] int64."""
        host = (self.per_chip(t).to(torch.int64) & 0xFFFFFFFF).reshape(CHUNK, -1)  # device row order
        out = torch.empty_like(host)
        out[self.device_order(start)] = host
        return out

    def pass1_chunk(self, slot: int, start: int, end: int) -> list[ttnn.Tensor]:
        """Pass 1 over the chunk: writes bank A, returns the loop's top-k sets (own copies, as G2 receives them)."""
        sets = []
        for L in LOOP_LAYERS:
            _, idx = self.pass1[L].forward(
                self.to_device(_chunk_rows(slot, L, 1, start, end), start),
                self.rope,
                self.cache,
                cache_layer_idx=LOOP_LAYERS.index(L),
                cache_user_id=slot,
                actual_start=start,
                actual_end=end,
                index_kv_cache=self.index_cache,
                return_indexer_indices=True,
            )
            sets.append(ttnn.clone(idx))
        return sets

    def pass2_forward(self, layer, slot, start, end, sets, hidden=None, metadata=None):
        return self.pass2[layer].forward(
            hidden if hidden is not None else self.to_device(_chunk_rows(slot, layer, 2, start, end), start),
            self.rope,
            self.cache,
            cache_layer_idx=LOOP_LAYERS.index(layer),
            cache_user_id=slot or 0,
            actual_start=None if metadata is not None else start,
            actual_end=None if metadata is not None else end,
            indexer_indices=sets[pass1_index_set(self.config, layer)],
            metadata=metadata,
            loop_pass2=True,
        )


def _byte_rows(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("window_format", FORMATS)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (sparse SDPA, push) are Blackhole-only")
@pytest.mark.timeout(0)
def test_two_bank_device(mesh_device, window_format, schedule):
    """Run the interleaved schedule; per chunk: pass 2 leaves every slot's pass-1 rows and the other slot's ring
    byte-identical, and a repeated forward is bit-identical; per slot at the end: chunked device output and ring rows
    vs the single-shot reference on the device's bank A and top-k."""
    stage = _Stage(mesh_device, window_format)
    outputs = {(slot, L): [] for slot in range(NUM_SLOTS) for L in LOOP_LAYERS}
    # Ring rows of each chunk's positions, read right after the chunk (the ring keeps only the latest P positions).
    rings = {(slot, L): [] for slot in range(NUM_SLOTS) for L in LOOP_LAYERS}
    set_rows = {(slot, k): [] for slot in range(NUM_SLOTS) for k in range(len(LOOP_LAYERS))}
    for slot, start, end in SCHEDULES[schedule]:
        sets = stage.pass1_chunk(slot, start, end)
        for k, s in enumerate(sets):
            set_rows[(slot, k)].append(stage.indices(s, start)[: end - start])
        before = stage.cache_natural()
        for L in LOOP_LAYERS:
            out = stage.output_to_torch(stage.pass2_forward(L, slot, start, end, sets), start)
            again = stage.output_to_torch(stage.pass2_forward(L, slot, start, end, sets), start)
            assert torch.equal(out, again), f"slot {slot} layer {L} chunk {start}: repeated pass-2 forward differs"
            outputs[(slot, L)].append(out[:, : end - start])
        ttnn.synchronize_device(mesh_device)
        after = stage.cache_natural()
        changed = (_byte_rows(after) != _byte_rows(before)).any(-1)  # [slots, layers, CAPACITY]
        assert not changed[:, :, :RING_BASE].any(), (
            f"slot {slot} chunk {start}: pass 2 changed pass-1 rows (bank A): "
            f"{changed[:, :, :RING_BASE].nonzero()[:4].tolist()}"
        )
        assert not changed[1 - slot].any(), f"slot {slot} chunk {start}: pass 2 changed the other slot's ring"
        for L in LOOP_LAYERS:
            cache = after[slot, LOOP_LAYERS.index(L)]
            rings[(slot, L)].append(stage.logical(cache[[ring_id(q, RING_BASE, CHUNK) for q in range(start, end)]]))
        for t in sets:
            ttnn.deallocate(t)
    final = stage.cache_natural()
    for slot in range(NUM_SLOTS):
        chunks = [(s, e) for sl, s, e in SCHEDULES[schedule] if sl == slot]
        total = chunks[-1][1]
        for L in LOOP_LAYERS:
            cache = final[slot, LOOP_LAYERS.index(L)]
            _check_vs_single_shot(
                stage.config,
                f"{window_format} {schedule} slot {slot}",
                L,
                slot,
                chunks,
                out=torch.cat(outputs[(slot, L)], dim=1),
                ring=torch.cat(rings[(slot, L)], dim=0),
                bank_a=stage.logical(cache[:total]),
                indices=torch.cat(set_rows[(slot, pass1_index_set(stage.config, L))], dim=0).view(1, 1, total, -1),
            )


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _check_vs_single_shot(config, case, layer, slot, chunks, *, out, ring, bank_a, indices):
    """LoopPass2MLAReference single-shot over the slot's prompt (positions 0 .. end of its last chunk) on the device's
    bank A and top-k vs the chunked device output, per chunk and over each SP shard's first W rows of a chunk (which
    read the previous shard's rows), and vs the ring's rows of every position, each read right after its chunk."""
    window, shard, total = config.loop_window_size, CHUNK // 8, chunks[-1][1]
    mla_ref = SparseMLAReference(config, glm_random_mla_weights(config, layer, GLM54_SYNTHETIC_SEED), seq_len=CAPACITY)
    ref = LoopPass2MLAReference(mla_ref, window=window, latent_dim=config.kv_lora_rank).forward(
        _prompt(slot, layer, 2, total), kvpe_a=bank_a, indices_a=indices, actual_start=0
    )
    ring_pcc = _pearson(ring, ref.kvpe)
    logger.info(f"[{case}] L{layer}: ring rows PCC {ring_pcc:.5f} over {total} positions")
    for start, end in chunks:
        got, exp = out[:, start:end], ref.output[:, start:end]
        n = end - start
        shard_pccs = [
            _pearson(got[:, s : min(s + window, n)], exp[:, s : min(s + window, n)]) for s in range(shard, n, shard)
        ]
        out_pcc = _pearson(got, exp)
        logger.info(
            f"[{case}] L{layer} chunk {start}-{end}: output PCC {out_pcc:.5f}, shard-start PCC min "
            f"{min(shard_pccs, default=float('nan')):.5f}"
        )
        # The shard halo first: a row near a shard start reads the previous shard's rows.
        for k, p in enumerate(shard_pccs, start=1):
            assert p >= OUTPUT_PCC, f"L{layer} chunk {start}: SP shard {k}'s first {window} rows PCC {p:.5f}"
        assert out_pcc >= OUTPUT_PCC, f"L{layer} chunk {start}: output PCC {out_pcc:.5f} < {OUTPUT_PCC}"
    assert ring_pcc >= WINDOW_PCC, f"L{layer}: ring rows PCC {ring_pcc:.5f} < {WINDOW_PCC}"


@pytest.mark.parametrize("window_format", FORMATS)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (sparse SDPA, push) are Blackhole-only")
@pytest.mark.timeout(0)
def test_two_bank_traced(mesh_device, window_format):
    """One pass-2 forward captured at the first chunk replays bit-identically to eager (metadata path) over the whole
    interleaved two-slot schedule: ring writes, index list, push slot and plan start all follow the metadata. Eager
    and traced passes run the same schedule from the same state (pass 1 rewrites bank A identically and every chunk
    rewrites the ring rows it reads before the next chunk of its slot)."""
    stage = _Stage(mesh_device, window_format)
    L = LOOP_LAYERS[0]
    schedule = SCHEDULES["two_slots"]
    eager = []
    for slot, start, end in schedule:
        sets = stage.pass1_chunk(slot, start, end)
        metadata = [_scalar(mesh_device, v) for v in (slot, start, end)]
        hidden = stage.to_device(_chunk_rows(slot, L, 2, start, end), start)
        eager.append(stage.output_to_torch(stage.pass2_forward(L, slot, start, end, sets, hidden, metadata), start))
    # Persistent inputs the capture reads; their contents change between replays.
    slot0, start0, end0 = schedule[0]
    p_meta = [_scalar(mesh_device, v) for v in (slot0, start0, end0)]
    p_hidden = stage.to_device(_chunk_rows(slot0, L, 2, start0, end0), start0)
    p_sets = [ttnn.clone(s) for s in stage.pass1_chunk(slot0, start0, end0)]
    stage.pass2_forward(L, None, None, None, p_sets, p_hidden, p_meta)  # warm at capture shapes
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = stage.pass2_forward(L, None, None, None, p_sets, p_hidden, p_meta)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        for k, (slot, start, end) in enumerate(schedule):
            for src, dst in zip(stage.pass1_chunk(slot, start, end), p_sets):
                ttnn.copy(src, dst)
            for dst, v in zip(p_meta, (slot, start, end)):
                ttnn.copy(_scalar(mesh_device, v), dst)
            ttnn.copy(stage.to_device(_chunk_rows(slot, L, 2, start, end), start), p_hidden)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            # Every row, padding included: padding rows read only ring rows their own chunk wrote.
            got = stage.output_to_torch(out, start)
            assert torch.equal(got, eager[k]), f"traced replay of slot {slot} chunk {start} differs from eager"
            logger.info(f"[{window_format}] traced replay slot {slot} chunk {start}-{end}: bit-identical to eager")
    finally:
        ttnn.release_trace(mesh_device, tid)
