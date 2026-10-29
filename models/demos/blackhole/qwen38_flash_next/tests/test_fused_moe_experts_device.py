# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x37: the routed-expert program (``ttnn/fused/moe_experts``, Design A of x35) against today's ``moe_compute``, BIT FOR BIT.

One die of the TP=8 mesh on a 1x1 mesh: layer ``X37_LAYER`` of the real cached BF4 experts, die ``X37_SHARD``'s 64
experts (the cache is one tensorbin of 8 die shards; the shard is moved to the one die with the cache's own memory
config).  The reference is exactly what ``Qwen38TTNNMoE._routed_partial`` runs around the op (moe.py:1033-1138):
untilize the hidden rows, reshape for several rows, move indices/scores to the drain core's L1 shard, fill the local combine
buffer with 0, ``ttnn.experimental.moe_compute`` (local-output form: cluster axis 0 of extent 1).  A routing is
random with a chosen number of experts of this die per row (0..4), same expert in two rows, etc.

Run: tests/../x37 run_dev.sh pytest this_file -s  (one die, device lock)
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import load_layer_shard, traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_experts as me

BF16, U16 = ttnn.bfloat16, ttnn.uint16
DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN, TOP_K, EXPERTS, PER_DEVICE = 2560, 10, 512, 64
LAYER = int(os.environ.get("X37_LAYER", "0"))
SHARD = int(os.environ.get("X37_SHARD", "0"))
DEVICE_PARAMS = [{"trace_region_size": 32 * 1024 * 1024}]
PLACEMENT = {
    "where": os.environ.get("X37_WHERE", "bank"),
    "x_fragments": int(os.environ.get("X37_XFRAG", "1")),
    "slots": int(os.environ.get("X37_SLOTS", "8")),
}  # "bank": 3 cores on each bank worker's row; "rect": one 24-core rectangle


def output_height_shard_dim(rows: int, ring: int) -> int:
    """moe.py ``moe_compute_output_height_shard_dim`` (rows <= 32)."""
    d = max(d for d in range(1, 5) if (HIDDEN // 32) % d == 0 and ring % d == 0)
    return -(-rows // (32 * d))


class Setup:
    """One layer of one die's experts and the constant tensors moe.py builds once per instance, for ``rows`` rows."""

    def __init__(self, mesh, rows: int, layer: int = LAYER, shard: int = SHARD, weights=None):
        self.mesh, self.rows, self.shard = mesh, rows, shard
        self.w01, self.w2 = weights if weights is not None else load_layer_shard(mesh, layer, shard)
        ring = effective_matmul_ring_size(mesh)
        assert ring == 8, ring
        self.height_shard_dim = output_height_shard_dim(rows, ring)
        drain = ttnn.experimental.get_moe_tilize_drain_core(mesh, 1, auto_output_width_shard_dim(HIDDEN), HIDDEN)
        self.drain = drain
        self.routing_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)}), [rows, TOP_K], ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        # the die's row of the owner mapping: 0 where this die (mesh coordinate 0) owns the expert
        owner = torch.ones(1, EXPERTS, dtype=torch.int32)
        owner[0, shard * PER_DEVICE : (shard + 1) * PER_DEVICE] = 0
        self.owner_map = owner
        self.expert_mapping = ttnn.from_torch(
            owner,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=U16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # moe_post's owner row: 1 where this die owns the expert
        self.owner_row = ttnn.from_torch(
            (1 - owner).to(torch.int16),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=U16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.combine = ttnn.from_torch(
            torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=BF16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def upload(self, tensor, dtype, layout, memory_config=DRAM):
        return ttnn.from_torch(
            tensor.contiguous(),
            device=self.mesh,
            layout=layout,
            dtype=dtype,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def hidden_tensor(self, hidden: torch.Tensor):
        """``[1, 1, rows, 2560]`` bf16 TILE in DRAM (the gathered hidden of the model)."""
        return self.upload(hidden.reshape(1, 1, self.rows, HIDDEN), BF16, ttnn.TILE_LAYOUT)

    def hidden_l1(self, hidden: torch.Tensor):
        """The same rows in the 5-core L1 width shard the gathered hidden lands in (moe.py ``hidden_act_memory_config``)."""
        if not hasattr(self, "hidden_cfg"):
            self.hidden_cfg, _ = dm.dram_sharded_matmul_configs(self.mesh, HIDDEN, EXPERTS, num_cores=5)
        return self.upload(hidden.reshape(1, 1, self.rows, HIDDEN), BF16, ttnn.TILE_LAYOUT, self.hidden_cfg)

    def routing_tensors(self, indices: torch.Tensor, scores: torch.Tensor):
        """The L1 drain-core shards of the model's routing rows (``_routed_partial``): RM uint16 / bf16 ``[1, rows, 10]``."""
        idx = self.upload(indices.reshape(1, 1, self.rows, TOP_K).to(torch.int32), U16, ttnn.ROW_MAJOR_LAYOUT)
        sc = self.upload(scores.reshape(1, 1, self.rows, TOP_K), BF16, ttnn.ROW_MAJOR_LAYOUT)
        idx3 = ttnn.reshape(idx, (1, self.rows, TOP_K))
        sc3 = ttnn.reshape(sc, (1, self.rows, TOP_K))
        return ttnn.to_memory_config(idx3, self.routing_cfg), ttnn.to_memory_config(sc3, self.routing_cfg)

    def reference(self, hidden_dev, indices_l1, scores_l1):
        """Today's path around ``moe_compute`` (``Qwen38TTNNMoE._routed_partial``); returns the combine buffer (torch)."""
        rows = self.rows
        sparse = ttnn.to_layout(hidden_dev, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if rows != 1:
            sparse = ttnn.reshape(sparse, (1, rows, HIDDEN))
        zeroed = ttnn.fill(self.combine, 0.0, output_tensor=self.combine)
        assert zeroed.tensor_id == self.combine.tensor_id
        outputs = ttnn.experimental.moe_compute(
            sparse,
            indices_l1,
            scores_l1,
            self.expert_mapping,
            self.w01,
            self.w2,
            layer_id=0,
            output_height_shard_dim=self.height_shard_dim,
            intermediate_size=640,
            has_bias=False,
            cluster_axis=0,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            output_memory_config=DRAM,
            optional_output_tensor=self.combine,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.SILU,
            compute_only=False,
            num_shared_experts_per_device=0,
        )
        assert len(outputs) == 6 and outputs[5].tensor_id == self.combine.tensor_id
        for t in (outputs[0], outputs[1], outputs[2], outputs[4]):
            ttnn.deallocate(t)
        return outputs[5]

    def read_combine(self, tensor) -> torch.Tensor:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(TOP_K, self.rows, HIDDEN).clone()


def make_routing(generator, rows: int, shard: int, n_local, shared_with_row0: int = 0):
    """Random top-10 per row: ``n_local[r]`` distinct experts of die ``shard`` (64 consecutive ids) and the rest elsewhere,
    in random slot order; row 1 shares ``shared_with_row0`` of row 0's local experts when asked."""
    lo = shard * PER_DEVICE
    local_ids, other_ids = list(range(lo, lo + PER_DEVICE)), [
        e for e in range(EXPERTS) if not lo <= e < lo + PER_DEVICE
    ]
    indices = torch.zeros(rows, TOP_K, dtype=torch.int64)
    prev = []
    for r in range(rows):
        n = n_local[r]
        pick = []
        if r == 1 and shared_with_row0:
            pick = prev[:shared_with_row0]
        pool = [e for e in local_ids if e not in pick]
        pick += [pool[i] for i in torch.randperm(len(pool), generator=generator)[: n - len(pick)].tolist()]
        rest = [other_ids[i] for i in torch.randperm(len(other_ids), generator=generator)[: TOP_K - n].tolist()]
        ids = pick + rest
        perm = torch.randperm(TOP_K, generator=generator).tolist()
        indices[r] = torch.tensor([ids[i] for i in perm])
        if r == 0:
            prev = [e for e in indices[r].tolist() if lo <= e < lo + PER_DEVICE]
    scores = (torch.rand(rows, TOP_K, generator=generator) * 0.5 + 0.05).to(torch.bfloat16)
    return indices, scores


def random_hidden(generator, rows: int):
    return (torch.randn(rows, HIDDEN, generator=generator) * 2.0).to(torch.bfloat16)


CASES = [  # (rows, n_local per row, rows sharing experts)
    (1, (0,), 0),
    (1, (1,), 0),
    (1, (2,), 0),
    (1, (3,), 0),
    (1, (4,), 0),
    (2, (1, 1), 0),
    (2, (2, 3), 0),
    (2, (4, 4), 0),
    (2, (0, 2), 0),
    (2, (0, 0), 0),
    (2, (2, 2), 2),
    (2, (3, 3), 1),
    (2, (4, 4), 4),
]


def case_id(case):
    rows, n_local, shared = case
    return f"r{rows}_n{'-'.join(map(str, n_local))}_s{shared}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_reference_deterministic(mesh_device):
    """Step 1: today's moe_compute path on the real layer: finite, owned slots non-zero, the rest untouched, and the same
    bits twice (the reference every later step compares with)."""
    mesh = mesh_device
    weights = load_layer_shard(mesh, LAYER, SHARD)
    for rows in (1, 2):
        setup = Setup(mesh, rows, weights=weights)
        gen = torch.Generator().manual_seed(7 + rows)
        for case in [c for c in CASES if c[0] == rows]:
            indices, scores = make_routing(gen, rows, SHARD, case[1], case[2])
            hidden = random_hidden(gen, rows)
            hidden_dev = setup.hidden_tensor(hidden)
            idx_l1, sc_l1 = setup.routing_tensors(indices, scores)
            first = setup.read_combine(setup.reference(hidden_dev, idx_l1, sc_l1))
            second = setup.read_combine(setup.reference(hidden_dev, idx_l1, sc_l1))
            assert torch.equal(first.view(torch.int16), second.view(torch.int16)), case
            owned = indices // PER_DEVICE == SHARD  # [rows, 10]
            for r in range(rows):
                for k in range(TOP_K):
                    page = first[k, r]
                    if owned[r, k]:
                        assert torch.isfinite(page.float()).all() and page.float().abs().max() > 0, (case, r, k)
                    else:
                        assert torch.count_nonzero(page) == 0, (case, r, k)
            logger.info(
                f"{case_id(case)}: reference deterministic, owned slots {int(owned.sum())}, |out|max "
                f"{first.float().abs().max():.3f}"
            )


# ---------------------------------------------------------------- golden of the gate/up phase (dequantised cached weights)


def active_experts(indices: torch.Tensor, shard: int):
    """The experts of the die the rows select, in the program's order (row-major first appearance): ``[(local expert, row
    mask, [k slot of row 0, row 1])]``."""
    out = []
    for r in range(indices.shape[0]):
        for k in range(TOP_K):
            e = int(indices[r, k])
            if e // PER_DEVICE != shard:
                continue
            local = e % PER_DEVICE
            entry = next((x for x in out if x[0] == local), None)
            if entry is None:
                entry = [local, 0, [0, 0]]
                out.append(entry)
            entry[1] |= 1 << r
            entry[2][r] = k
    return out


def dequantised_stream(setup: Setup):
    """``[64 experts, 80 blocks, 320, 128]`` fp32: the gate/up stream of every expert (the banks' pieces back to back)."""
    t = ttnn.to_torch(ttnn.get_device_tensors(setup.w01)[0]).float()  # [8, 1, 64, 10, 320, 128]
    return t[:, 0].permute(1, 0, 2, 3, 4).reshape(PER_DEVICE, 80, 320, 128)


def gate_up_golden(stream_e: torch.Tensor, x: torch.Tensor):
    """``[rows, 20]`` -> ``[rows, 640]`` fp32: silu(x W0) * (x W1) of every column from one expert's gate/up stream."""
    out = torch.zeros(x.shape[0], 640)
    xf = x.float()
    ranges = me.stream_block_ranges()
    for c in range(me.RING):
        a, b = ranges[f"P{c}"]
        w = torch.cat([stream_e[i] for i in range(a, b)], dim=0)  # [2560, 128]: W0 c | W1 c | W0 c+1 | W1 c+1
        for pair in range(2):
            g, u = xf @ w[:, 64 * pair : 64 * pair + 32], xf @ w[:, 64 * pair + 32 : 64 * pair + 64]
            col = me.COLUMN_START[c] + pair
            out[:, 32 * col : 32 * col + 32] = torch.nn.functional.silu(g) * u
        if me.COLUMNS[c] == 3:
            a, b = ranges[f"H{c}"]
            rows_k = []
            for i in range(a, b):
                blk = stream_e[i].reshape(10, 32, 128)
                for tr in range(10):
                    for h in range(2):
                        rows_k.append(blk[tr, :, 64 * h : 64 * h + 64])  # K tile 2 tr + h: [32 (k), W0 | W1]
            w = torch.cat(rows_k, dim=0)  # [2560, 64]
            g, u = xf @ w[:, :32], xf @ w[:, 32:64]
            col = me.COLUMN_START[c] + 2
            out[:, 32 * col : 32 * col + 32] = torch.nn.functional.silu(g) * u
    return out


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_gate_up_stage(mesh_device):
    """Step 2: the gate/up phase alone (``stage=2``: the in2 tiles of every active expert dumped to DRAM) against the torch
    golden of the dequantised weights (PCC; the bitwise check of the whole program is step 3)."""
    mesh = mesh_device
    weights = load_layer_shard(mesh, LAYER, SHARD)
    for rows in (1, 2):
        setup = Setup(mesh, rows, weights=weights)
        stream = dequantised_stream(setup)
        gen = torch.Generator().manual_seed(11 + rows)
        for case in [c for c in CASES if c[0] == rows]:
            indices, scores = make_routing(gen, rows, SHARD, case[1], case[2])
            hidden = random_hidden(gen, rows)
            hidden_dev = setup.hidden_l1(hidden)
            idx_l1, _ = setup.routing_tensors(indices, scores)
            dump = setup.upload(torch.zeros(1, 1, 32, 640 * 20, dtype=torch.bfloat16), BF16, ttnn.TILE_LAYOUT)
            me.moe_experts(
                hidden_dev, idx_l1, setup.owner_row, setup.w01, setup.w2, setup.combine, dump=dump, stage=2, **PLACEMENT
            )
            got = ttnn.to_torch(ttnn.get_device_tensors(dump)[0]).reshape(32, 640 * 20)
            for j, (e, mask, kslots) in enumerate(active_experts(indices, SHARD)):
                want = gate_up_golden(stream[e], hidden)
                have = got[:rows, j * 640 : (j + 1) * 640].float()
                score = pcc(have, want)
                err = (have - want).abs().max() / want.abs().max()
                logger.info(f"{case_id(case)} expert {j} (local {e}): in2 PCC {score:.6f} max rel err {err:.4f}")
                assert score > 0.999, (case, j, score)


# ---------------------------------------------------------------- the whole program against moe_compute, bit for bit


def random_cases(gen, count: int):
    cases = []
    for _ in range(count):
        rows = 1 + int(torch.randint(0, 2, (1,), generator=gen))
        n_local = tuple(int(v) for v in torch.randint(0, 5, (rows,), generator=gen))
        shared = 0
        if rows == 2 and min(n_local) > 0 and int(torch.randint(0, 2, (1,), generator=gen)):
            shared = int(torch.randint(1, min(n_local) + 1, (1,), generator=gen))
        cases.append((rows, n_local, shared))
    return cases


def run_mine(setup: Setup, combine, hidden_l1, idx_l1, **kwargs):
    ttnn.fill(combine, 0.0, output_tensor=combine)
    me.moe_experts(hidden_l1, idx_l1, setup.owner_row, setup.w01, setup.w2, combine, **kwargs, **PLACEMENT)
    return setup.read_combine(combine)


def compare(got: torch.Tensor, want: torch.Tensor):
    g, w = got.view(torch.int16), want.view(torch.int16)
    return int((g != w).sum()), g.numel()


def check_cases(mesh, cases_per_rows, seed, shard=SHARD, layer=LAYER, repeats=1):
    weights = load_layer_shard(mesh, layer, shard)
    failures = []
    totals = 0
    for rows in (1, 2):
        setup = Setup(mesh, rows, layer=layer, shard=shard, weights=weights)
        mine_combine = ttnn.from_torch(
            torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=BF16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        gen = torch.Generator().manual_seed(seed * 100 + rows)
        cases = [c for c in CASES if c[0] == rows] + [c for c in cases_per_rows(gen) if c[0] == rows]
        for case in cases:
            indices, scores = make_routing(gen, rows, shard, case[1], case[2])
            hidden = random_hidden(gen, rows)
            hidden_dev, hidden_l1 = setup.hidden_tensor(hidden), setup.hidden_l1(hidden)
            idx_l1, sc_l1 = setup.routing_tensors(indices, scores)
            want = setup.read_combine(setup.reference(hidden_dev, idx_l1, sc_l1))
            for rep in range(repeats):
                got = run_mine(setup, mine_combine, hidden_l1, idx_l1)
                bad, total = compare(got, want)
                totals += 1
                owned = int((indices // PER_DEVICE == shard).sum())
                assert (owned == 0) == (torch.count_nonzero(want) == 0), "reference has no data for an owned slot?"
                logger.info(f"{case_id(case)} owned slots {owned} rep {rep}: {bad} / {total} differing bf16 words")
                if bad:
                    failures.append((case, rep, bad))
            for t in (hidden_dev, hidden_l1, idx_l1, sc_l1):
                ttnn.deallocate(t)
    return failures, totals


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bitwise_vs_moe_compute(mesh_device):
    """Step 3: ``moe_experts`` (all stages) == ``moe_compute`` bit for bit on the whole [10, rows, 2560] buffer."""
    count = int(os.environ.get("X37_RANDOM_CASES", "30"))
    failures, totals = check_cases(
        mesh_device,
        lambda gen: random_cases(gen, count),
        seed=int(os.environ.get("X37_SEED", "1")),
        repeats=int(os.environ.get("X37_REPEATS", "1")),
    )
    logger.info(f"bitwise: {totals - len(failures)} / {totals} routings identical")
    assert not failures, failures


# ---------------------------------------------------------------- timing: today's chain vs the new program, in a trace


TIMING_CASES = [  # (rows, n_local per row, rows sharing) -> active experts n
    (1, (0,), 0),
    (1, (1,), 0),
    (1, (2,), 0),
    (1, (3,), 0),
    (1, (4,), 0),
    (2, (1, 1), 0),
    (2, (2, 2), 0),
    (2, (3, 3), 3),
    (2, (4, 4), 0),
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_timing_vs_moe_compute(mesh_device):
    """Step 4: wall us per call in a trace (slope between a 4-call and a 12-call trace): today's moe_compute alone, today's
    chain around it (untilize + reshape + fill + moe_compute, the routing already in the drain shard as in the fused tail),
    and ``moe_experts`` alone (it needs none of the untilize / fill)."""
    import json

    mesh = mesh_device
    weights = load_layer_shard(mesh, LAYER, SHARD)
    results = Path(os.environ.get("X37_RESULTS", "/tmp/x37_timing.jsonl"))
    replays = int(os.environ.get("X37_REPLAYS", "100"))
    for case in TIMING_CASES:
        rows = case[0]
        setup = Setup(mesh, rows, weights=weights)
        gen = torch.Generator().manual_seed(5)
        indices, scores = make_routing(gen, rows, SHARD, case[1], case[2])
        hidden = random_hidden(gen, rows)
        hidden_dev, hidden_l1 = setup.hidden_tensor(hidden), setup.hidden_l1(hidden)
        idx_l1, sc_l1 = setup.routing_tensors(indices, scores)
        n_active = len(active_experts(indices, SHARD))
        mine_combine = ttnn.from_torch(
            torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=BF16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sparse = ttnn.to_layout(hidden_dev, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if rows != 1:
            sparse = ttnn.reshape(sparse, (1, rows, HIDDEN))

        def moe_compute_only():
            outputs = ttnn.experimental.moe_compute(
                sparse,
                idx_l1,
                sc_l1,
                setup.expert_mapping,
                setup.w01,
                setup.w2,
                layer_id=0,
                output_height_shard_dim=setup.height_shard_dim,
                intermediate_size=640,
                has_bias=False,
                cluster_axis=0,
                topology=None,
                num_links=None,
                mux_core_range_set=None,
                output_memory_config=DRAM,
                optional_output_tensor=setup.combine,
                optional_cross_device_semaphore=None,
                activation_type=MoEActivationFunction.SILU,
                compute_only=False,
                num_shared_experts_per_device=0,
            )
            for t in (outputs[0], outputs[1], outputs[2], outputs[4]):
                ttnn.deallocate(t)

        def today_chain():
            s = ttnn.to_layout(hidden_dev, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            if rows != 1:
                s = ttnn.reshape(s, (1, rows, HIDDEN))
            ttnn.fill(setup.combine, 0.0, output_tensor=setup.combine)
            outputs = ttnn.experimental.moe_compute(
                s,
                idx_l1,
                sc_l1,
                setup.expert_mapping,
                setup.w01,
                setup.w2,
                layer_id=0,
                output_height_shard_dim=setup.height_shard_dim,
                intermediate_size=640,
                has_bias=False,
                cluster_axis=0,
                topology=None,
                num_links=None,
                mux_core_range_set=None,
                output_memory_config=DRAM,
                optional_output_tensor=setup.combine,
                optional_cross_device_semaphore=None,
                activation_type=MoEActivationFunction.SILU,
                compute_only=False,
                num_shared_experts_per_device=0,
            )
            for t in (outputs[0], outputs[1], outputs[2], outputs[4], s):
                ttnn.deallocate(t)

        def new_op():
            me.moe_experts(hidden_l1, idx_l1, setup.owner_row, setup.w01, setup.w2, mine_combine, **PLACEMENT)

        def gate_up_only():
            me.moe_experts(hidden_l1, idx_l1, setup.owner_row, setup.w01, setup.w2, mine_combine, stage=2, **PLACEMENT)

        row = dict(case=case_id(case), rows=rows, active_experts=n_active, load=os.getloadavg())
        variants = [("moe_compute_only", moe_compute_only), ("today_chain", today_chain), ("moe_experts", new_op)]
        if os.environ.get("X37_ONLY_NEW"):
            variants = [("moe_experts", new_op), ("moe_experts_stage2", gate_up_only)]
        for name, call in variants:
            slope, per_call = traced_wall_us(mesh, call, replays=replays)
            row[name] = round(slope, 2)
            row[name + "_T12_per_call"] = round(per_call, 2)
        logger.info(json.dumps(row))
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        for t in (hidden_dev, hidden_l1, idx_l1, sc_l1, sparse, mine_combine):
            ttnn.deallocate(t)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_zone_profile(mesh_device):
    """Study build (QWEN38_FUSED_ZONES=1 TT_METAL_DEVICE_PROFILER=1): runs the program for 4 local experts (rows 1) a few times
    and reads the device profiler; the per-phase zones (fz_me_*) land in <profiler dir>/.logs/profile_log_device.csv."""
    mesh = mesh_device
    weights = load_layer_shard(mesh, LAYER, SHARD)
    rows = int(os.environ.get("X37_PROFILE_ROWS", "1"))
    n_local = tuple(int(v) for v in os.environ.get("X37_PROFILE_NLOCAL", "4").split(","))[:rows]
    setup = Setup(mesh, rows, weights=weights)
    gen = torch.Generator().manual_seed(5)
    indices, scores = make_routing(gen, rows, SHARD, n_local if len(n_local) == rows else n_local * rows, 0)
    hidden = random_hidden(gen, rows)
    hidden_l1 = setup.hidden_l1(hidden)
    idx_l1, _ = setup.routing_tensors(indices, scores)
    out = ttnn.from_torch(
        torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=BF16,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    for _ in range(3):
        me.moe_experts(hidden_l1, idx_l1, setup.owner_row, setup.w01, setup.w2, out, **PLACEMENT)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    for _ in range(3):
        me.moe_experts(hidden_l1, idx_l1, setup.owner_row, setup.w01, setup.w2, out, **PLACEMENT)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_trace_replay_bitwise(mesh_device):
    """The program under a trace (what the served decode step does): three launches of different routings captured in one
    trace and replayed twice; the buffers equal today's results bit for bit (semaphores / CB state across replays)."""
    mesh = mesh_device
    weights = load_layer_shard(mesh, LAYER, SHARD)
    for rows, cases in (
        (1, [(1, (4,), 0), (1, (0,), 0), (1, (2,), 0)]),
        (2, [(2, (3, 3), 2), (2, (4, 0), 0), (2, (1, 2), 1)]),
    ):
        setup = Setup(mesh, rows, weights=weights)
        gen = torch.Generator().manual_seed(21 + rows)
        owner = setup.owner_row
        mine = [
            ttnn.from_torch(
                torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=BF16,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for _ in cases
        ]
        inputs, wants = [], []
        for case in cases:
            indices, scores = make_routing(gen, rows, SHARD, case[1], case[2])
            hidden = random_hidden(gen, rows)
            hidden_dev, hidden_l1 = setup.hidden_tensor(hidden), setup.hidden_l1(hidden)
            idx_l1, sc_l1 = setup.routing_tensors(indices, scores)
            wants.append(setup.read_combine(setup.reference(hidden_dev, idx_l1, sc_l1)))
            inputs.append((hidden_l1, idx_l1))
        for (hidden_l1, idx_l1), out in zip(inputs, mine):  # eager once: programs are cached before a capture
            ttnn.fill(out, 0.0, output_tensor=out)
            me.moe_experts(hidden_l1, idx_l1, owner, setup.w01, setup.w2, out, **PLACEMENT)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        for (hidden_l1, idx_l1), out in zip(inputs, mine):
            me.moe_experts(hidden_l1, idx_l1, owner, setup.w01, setup.w2, out, **PLACEMENT)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for replay in range(2):
            for out in mine:
                ttnn.fill(out, 0.0, output_tensor=out)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for out, want, case in zip(mine, wants, cases):
                bad, total = compare(setup.read_combine(out), want)
                logger.info(f"trace replay {replay} {case_id(case)}: {bad} / {total} differing bf16 words")
                assert bad == 0, (replay, case)
        ttnn.release_trace(mesh, trace)
