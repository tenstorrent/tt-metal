# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoeComputeCustom (custom_ops/moe_compute) vs ttnn.experimental.moe_compute at the gemma_4_26b decode shapes,
single chip (TT_VISIBLE_DEVICES=6): bit-identity of the staged expert outputs in both staging layouts, the torch
golden, the program-cache hit on fresh buffers, the pad-row case and the device time per call.

Routing model: T tokens with distinct top-8 over 128 global experts of which E are local (mapping row: local ids ->
this device, every other id -> a non-existent coordinate 1), i.e. one Galaxy column chip's share; the pad case adds
rows with ids 0..7 and score 0 (the decode frame's padding), which land on local experts 0..3.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.experimental.moe_compute_utils import (
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.gemma4.custom_ops.moe_compute.op import MoeComputeCustom

if ttnn.get_arch_name() != "blackhole":
    pytest.skip("the moe_compute port assumes the Blackhole DRAM ring and NoC burst", allow_module_level=True)

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED, "l1_small_size": 16384}, id="single-chip")]
H, N, K, GLOBAL_EXPERTS = 2816, 704, 8, 128
HS = 4


def _bf4_round(w):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _routing(T, E, pad_rows=0, seed=0):
    """tokens [T, H] bf16, indices [T, K] uint16 (distinct per token), scores [T, K] bf16 (normalised); the last
    ``pad_rows`` rows are padding: ids 0..K-1, score 0."""
    g = torch.Generator().manual_seed(seed)
    tokens = (torch.rand(T, H, generator=g) - 0.5).bfloat16()
    idx = torch.stack([torch.randperm(GLOBAL_EXPERTS, generator=g)[:K] for _ in range(T)]).to(torch.int32)
    sc = torch.rand(T, K, generator=g) + 1e-3
    sc = (sc / sc.sum(-1, keepdim=True)).bfloat16()
    if pad_rows:
        idx[T - pad_rows :] = torch.arange(K, dtype=torch.int32).repeat(pad_rows, 1)
        sc[T - pad_rows :] = 0.0
    return tokens, idx, sc


def _weights(E, seed=1):
    g = torch.Generator().manual_seed(seed)
    w0 = ((torch.rand(1, E, H, N, generator=g) - 0.5) * 0.2).bfloat16()
    w1 = ((torch.rand(1, E, H, N, generator=g) - 0.5) * 0.2).bfloat16()
    w2 = ((torch.rand(1, E, N, H, generator=g) - 0.5) * 0.2).bfloat16()
    return w0, w1, w2


class Setup:
    """Device tensors of one case, shared by the custom op and the shared op."""

    def __init__(self, dev, E, T, pad_rows=0, seed=0, mux=None):
        self.E, self.T = E, T
        self.tokens, self.idx, self.sc = _routing(T, E, pad_rows, seed)
        self.w0, self.w1, self.w2 = _weights(E, seed + 1)
        pk = {} if mux is None else {"mux_core_range_set": mux}
        drain = ttnn.experimental.get_moe_tilize_drain_core(dev, HS, 4, H, **pk)
        drain_crs = ttnn.CoreRangeSet([ttnn.CoreRange(drain, drain)])
        mapping = torch.ones(1, GLOBAL_EXPERTS, dtype=torch.int32)  # every expert elsewhere ...
        mapping[0, :E] = 0  # ... but the local ones (global ids 0..E-1 = local slots)
        self.local_ids = list(range(E))
        self.tt_mapping = ttnn.from_torch(
            mapping, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.tt_tokens = ttnn.from_torch(
            self.tokens.reshape(1, T, H),
            device=dev,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def drain_sharded(t, dtype):
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(drain_crs, [T, K], ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(
                t.reshape(1, T, K), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=mc
            )

        self.tt_idx = drain_sharded(self.idx, ttnn.uint16)
        self.tt_sc = drain_sharded(self.sc, ttnn.bfloat16)
        w01_map, w2_map, banks = get_weight_core_shard_maps(dev, H, N)
        w01_mc, w2_mc, _, _ = get_weight_mem_configs(1, E, H, N, w01_map, w2_map, banks, has_bias=False)
        packed01 = prepare_w0_w1_tensor_for_moe_compute(self.w0, self.w1, 1, E, H, N, w01_map)
        packed2 = prepare_w2_tensor_for_moe_compute(self.w2, 1, E, N, H, w2_map, w01_map)
        self.tt_w01 = ttnn.from_torch(
            packed01, dtype=ttnn.bfloat4_b, device=dev, layout=ttnn.TILE_LAYOUT, memory_config=w01_mc
        )
        self.tt_w2 = ttnn.from_torch(
            packed2, dtype=ttnn.bfloat4_b, device=dev, layout=ttnn.TILE_LAYOUT, memory_config=w2_mc
        )
        self.mux = mux

    def golden(self, skip_zero_scores=False):
        return MoeComputeCustom.golden(
            self.tokens,
            self.idx,
            self.sc,
            self.local_ids,
            _bf4_round(self.w0[0]),
            _bf4_round(self.w1[0]),
            _bf4_round(self.w2[0]),
            skip_zero_scores=skip_zero_scores,
        )

    def shared_op(self, compute_only, output=None):
        return ttnn.experimental.moe_compute(
            self.tt_tokens,
            self.tt_idx,
            self.tt_sc,
            self.tt_mapping,
            self.tt_w01,
            self.tt_w2,
            layer_id=0,
            output_height_shard_dim=HS,
            intermediate_size=N,
            has_bias=False,
            cluster_axis=None,
            topology=None,
            num_links=None,
            mux_core_range_set=self.mux,
            output_memory_config=None,
            optional_output_tensor=output,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.GELU,
            compute_only=compute_only,
        )

    def custom(self, dev, layout, **kw):
        return MoeComputeCustom(
            dev,
            hidden_size=H,
            intermediate_size=N,
            experts_per_device=self.E,
            select_experts_k=K,
            output_height_shard_dim=HS,
            mux_core_range_set=self.mux,
            activation_type=MoEActivationFunction.GELU,
            cluster_axis=None,
            staging_layout=layout,
            **kw,
        )

    def run_custom(self, op):
        return op(self.tt_tokens, self.tt_idx, self.tt_sc, self.tt_mapping, self.tt_w01, self.tt_w2, layer_id=0)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _counts_row(counts, core_index, E):
    """The per-expert counts of one worker core (the row's tail past E is allocation garbage)."""
    return ttnn.to_torch(counts)[core_index, :E].clone()


def _ring_core_index(dev, op):
    g = op.cores["grid"]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g[0] - 1, g[1] - 1))])
    cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(crs, row_wise=True)]
    return cores.index(op.cores["matmul"][0])


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _check_golden(rows_dev, rows_gold, tag, pcc_min=0.98):
    for e, (d, g) in enumerate(zip(rows_dev, rows_gold)):
        assert d.shape == g.shape, f"{tag} expert {e}: {tuple(d.shape)} vs golden {tuple(g.shape)}"
        if d.numel() == 0:
            continue
        assert torch.isfinite(d.float()).all(), f"{tag} expert {e}: non-finite"
        pcc = _pcc(d, g)
        logger.info(f"{tag} expert {e}: {d.shape[0]} rows, pcc vs golden {pcc:.6f}")
        assert pcc > pcc_min, f"{tag} expert {e}: pcc {pcc}"


# --------------------------------------------------------------------------------------------- placement (host)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_placement_matches_shared_op(mesh_device, device_params):
    """select_cores (the Python port of select_moe_compute_cores) agrees with the op's placement helpers, with and
    without the production mux range."""
    mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])
    for m in (None, mux):
        op = MoeComputeCustom(
            mesh_device,
            hidden_size=H,
            intermediate_size=N,
            experts_per_device=4,
            select_experts_k=K,
            mux_core_range_set=m,
            cluster_axis=None,
        )
        logger.info(
            f"mux={m is not None}: ring {op.cores['matmul']} tilize {op.cores['tilize']} combine {op.cores['combine']}"
        )
        assert len(op.cores["matmul"]) == op.ring_size and len(op.cores["combine"]) == 16
        assert op.w01_tiles == 560 and op.w2_tiles == 336, (op.w01_tiles, op.w2_tiles)


# --------------------------------------------------------------------------------------------- bit identity
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_bit_identity_fused_layout(mesh_device, device_params):
    """2 local experts (the fused halves hold two): custom op (fused layout) vs the shared op compute_only=True;
    counts, activation, e_t and every staged row bitwise equal; golden PCC; cache hit on fresh buffers."""
    mesh_device.enable_program_cache()
    s = Setup(mesh_device, E=2, T=256)
    op = s.custom(mesh_device, "fused")
    ci = _ring_core_index(mesh_device, op)
    ref = s.shared_op(compute_only=True)
    ref_counts = _counts_row(ref[0], ci, s.E)
    ref_rows = op.staging_rows(ttnn.to_torch(ref[4]).reshape(-1, H), ref_counts, s.T, "fused")
    rows_gold, lists = s.golden()
    # the tables are written up to the activated-token count / the expert count (+ the -1 sentinel): compare that
    n_act = len({t for lst in lists for t, _ in lst})
    act_words = n_act * (((2 * s.E + 1) * 4 + 15) // 16 * 16) // 4
    ref_act = ttnn.to_torch(ref[1]).reshape(-1)[:act_words]
    # e_t entries are 16 B: word 0 token id, word 1 k slot, words 2-3 never written
    et_words = [(len(lst) + 1) * 4 for lst in lists]
    et_view = lambda t, e, w: t[e, :w].reshape(-1, 4)[:, :2]
    ref_et_t = ttnn.to_torch(ref[2])
    ref_et = [et_view(ref_et_t, e, w) for e, w in enumerate(et_words)]
    for trial in range(2):  # cache miss, then the hit on fresh buffers
        out = s.run_custom(op)
        counts = _counts_row(out[0], ci, s.E)
        assert torch.equal(counts, ref_counts), (counts, ref_counts)
        assert torch.equal(ttnn.to_torch(out[1]).reshape(-1)[:act_words], ref_act), "activation table differs"
        et = ttnn.to_torch(out[2])
        for e, w in enumerate(et_words):
            assert torch.equal(et_view(et, e, w), ref_et[e]), f"e_t of expert {e} differs"
        rows = op.staging_rows(ttnn.to_torch(out[3]).reshape(-1, H), counts, s.T, "fused")
        for e in range(s.E):
            assert int(counts[e]) == len(lists[e])
            diff = int((_bits(rows[e]) != _bits(ref_rows[e])).sum())
            assert (
                diff == 0
            ), f"trial {trial} expert {e}: {diff} of {rows[e].numel()} bf16 values differ from the shared op"
        _check_golden(rows, rows_gold, f"fused trial {trial}")
        for t in out[:4]:  # out[4] is out[3]: the shared shard must go before the next call's allocations
            ttnn.deallocate(t)
    logger.info("fused layout: bitwise equal to the shared op on 2 experts x 256 tokens (miss + hit)")


@pytest.mark.parametrize(
    "pad_rows, skip", [(0, False), (128, False), (128, True)], ids=["no-pad", "pad128", "pad128-skip"]
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_bit_identity_standalone_layout(mesh_device, device_params, pad_rows, skip):
    """4 local experts, 256 tokens (128 real + 128 pad rows in the pad cases: 5 chunks per expert): custom op
    (standalone layout) gathered to [K, T, H] vs the shared op's SingleDevice fused local combine output, bitwise.
    With skip_zero_scores the pad routes are dropped: the real rows stay bitwise equal, the pad rows are zero."""
    mesh_device.enable_program_cache()
    s = Setup(mesh_device, E=4, T=256, pad_rows=pad_rows)
    op = s.custom(mesh_device, "standalone", skip_zero_scores=skip)
    ci = _ring_core_index(mesh_device, op)
    zero_out = lambda: ttnn.from_torch(
        torch.zeros(K, s.T, H, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ref = s.shared_op(compute_only=False, output=zero_out())
    ref_counts = _counts_row(ref[0], ci, s.E)
    ref_combined = ttnn.to_torch(ref[5])
    rows_gold, lists = s.golden(skip_zero_scores=skip)
    if skip:
        ref_combined = ref_combined.clone()
        ref_combined[:, s.T - pad_rows :] = 0  # the dropped pad rows are never written by the custom op
    for trial in range(2):
        out = s.run_custom(op)
        counts = _counts_row(out[0], ci, s.E)
        for e in range(s.E):
            assert int(counts[e]) == len(lists[e]), (e, int(counts[e]), len(lists[e]))
        if not skip:
            assert torch.equal(counts, ref_counts)
        combined = op.combined_from_staging(ttnn.to_torch(out[3]).reshape(-1, H), counts, ttnn.to_torch(out[2]), s.T)
        diff = int((_bits(combined) != _bits(ref_combined)).sum())
        assert diff == 0, f"trial {trial}: {diff} of {combined.numel()} bf16 values differ from the shared op's combine"
        rows = op.staging_rows(ttnn.to_torch(out[3]).reshape(-1, H), counts, s.T)
        _check_golden(rows, rows_gold, f"standalone trial {trial}")
        chunks = [(int(c) + 31) // 32 for c in counts.tolist()]
        logger.info(f"standalone pad={pad_rows} skip={skip} trial {trial}: chunks per expert {chunks}, bitwise equal")
        for t in out[:4]:
            ttnn.deallocate(t)


# --------------------------------------------------------------------------------------------- device time
def _device_kernel_durations(log_path, detail=None):
    """Per program (run host ID, in order): max kernel end - min kernel start over all cores, cycles -> us.
    With ``detail`` (a dict) also records per program the per-(core, RISC) kernel spans in us."""
    starts, ends = defaultdict(list), defaultdict(list)
    per = defaultdict(dict)
    with open(log_path) as f:
        head = next(f)
        freq_mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        hdr = [c.strip() for c in next(f).split(",")]
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < len(hdr):
                continue
            zone, typ = p[ci["zone name"]], p[ci["type"]]
            if "KERNEL" not in zone:
                continue
            rid = int(p[ci["run host ID"]])
            t = int(p[ci["time[cycles since reset]"]])
            (starts if typ == "ZONE_START" else ends)[rid].append(t)
            key = (int(p[ci["core_x"]]), int(p[ci["core_y"]]), p[ci["RISC processor type"]])
            d = per[rid].setdefault(key, [None, None])
            d[0 if typ == "ZONE_START" else 1] = t
    out = []
    for rid in sorted(starts):
        if rid in ends:
            t0 = min(starts[rid])
            out.append((rid, (max(ends[rid]) - t0) / freq_mhz))
            if detail is not None:
                detail[rid] = {
                    k: ((v[0] - t0) / freq_mhz, (v[1] - t0) / freq_mhz) for k, v in per[rid].items() if None not in v
                }
    return out


@pytest.mark.parametrize("pad_rows", [0, 128], ids=["one-chunk", "five-chunks"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_perf_vs_shared_op(mesh_device, device_params, pad_rows):
    """Device profiler: kernel duration per call of the shared op (compute_only) and the custom op (standalone
    layout, and with skip_zero_scores in the pad case). Run with TT_METAL_DEVICE_PROFILER=1 and a private
    TT_METAL_PROFILER_DIR; without the profiler the test only reports wall time."""
    mesh_device.enable_program_cache()
    iters = int(os.environ.get("MOE_COMPUTE_PERF_ITERS", "5"))
    s = Setup(mesh_device, E=4, T=256, pad_rows=pad_rows)
    variants = [("custom", s.custom(mesh_device, "standalone"))]
    if pad_rows:
        variants.append(("custom-skip", s.custom(mesh_device, "standalone", skip_zero_scores=True)))
    depth = os.environ.get("MOE_COMPUTE_PERF_DEPTHS")
    if depth:
        for d in depth.split(","):
            variants.append((f"custom-depth{d}", s.custom(mesh_device, "standalone", prefetch_depth=int(d))))
    order = []  # (name, count) in program order, for the device log
    # warm every program once (compile), then the timed calls
    for name, op in [("shared", None)] + variants:
        out = s.shared_op(compute_only=True) if op is None else s.run_custom(op)
        ttnn.synchronize_device(mesh_device)
        for t in out[:4]:
            ttnn.deallocate(t)
        order.append((name, 1))
    walls = {}
    for name, op in [("shared", None)] + variants:
        t0 = time.perf_counter()
        for _ in range(iters):
            out = s.shared_op(compute_only=True) if op is None else s.run_custom(op)
            for t in out[:4]:
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        walls[name] = (time.perf_counter() - t0) / iters * 1e6
        order.append((name, iters))
    ttnn.ReadDeviceProfiler(mesh_device)
    for name, us in walls.items():
        logger.info(f"pad={pad_rows} {name}: wall {us:.0f} us per call (eager, {iters} calls)")
    prof_dir = os.environ.get("TT_METAL_PROFILER_DIR")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1" or not prof_dir:
        return
    # The device log is written when the device closes: record the program order next to it and parse offline
    # (``python test_moe_compute.py <profiler dir>``), one perf case per process and profiler dir.
    import json

    Path(prof_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(prof_dir) / "order.json", "w") as f:
        json.dump({"pad_rows": pad_rows, "order": order}, f)


def report_perf(prof_dir):
    """Offline: pair the device log of one perf-test process with its order.json."""
    import json

    with open(Path(prof_dir) / "order.json") as f:
        meta = json.load(f)
    logs = sorted(Path(prof_dir).rglob("profile_log_device.csv"), key=lambda p: p.stat().st_mtime)
    assert logs, f"no profile_log_device.csv under {prof_dir}"
    detail = {}
    durs = _device_kernel_durations(logs[-1], detail)
    expected = sum(n for _, n in meta["order"])
    print(f"{logs[-1]}: {len(durs)} programs recorded, {expected} expected (pad_rows={meta['pad_rows']})")
    assert len(durs) >= expected, "the log holds fewer programs than the test ran"
    durs = durs[-expected:]
    i = 0
    for name, n in meta["order"]:
        group = durs[i : i + n]
        vals = sorted(d for _, d in group)
        i += n
        if n > 1:
            print(
                f"PERF pad={meta['pad_rows']} {name}: kernel {vals[len(vals) // 2]:.1f} us median, "
                f"min {vals[0]:.1f}, max {vals[-1]:.1f} ({n} calls)"
            )
            # the long poles of the median call: (core, RISC) kernel spans, start..end relative to the program
            rid = sorted(group, key=lambda x: x[1])[len(group) // 2][0]
            spans = sorted(detail[rid].items(), key=lambda kv: -(kv[1][1] - kv[1][0]))
            print("   top spans: " + "; ".join(f"({x},{y}) {r} {a:.0f}..{b:.0f}" for (x, y, r), (a, b) in spans[:12]))


if __name__ == "__main__":
    import sys

    report_perf(sys.argv[1])


# --------------------------------------------------------------------------------------------- multi-chip (lead)
BLOCK_2X4 = [
    pytest.param(
        (2, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 16384},
        id="block-2x4",
    )
]


@pytest.mark.parametrize("mesh_device, device_params", BLOCK_2X4, indirect=True)
def test_mesh_block_pair_vs_shared(mesh_device, device_params):
    """(2,4) block, dispatch axis 0 (extent 2, Linear): the custom op + the standalone
    ``ttnn.experimental.selective_reduce_combine`` vs the shared FullCcl ``moe_compute`` on the same inputs
    (the 6U harness data: 16 experts per chip, 32 tokens per chip, top-8, gemma shapes): the combine outputs
    must be bitwise equal on every chip, and both pass the harness' combine golden. NOT run in this pass (no
    multi-chip mesh was available): TT_VISIBLE_DEVICES=0,1,4,5,8,9,12,13 with the p150_x8 descriptor."""
    from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
        _build_moe_host_data,
        _build_quantized_weight_host_tensors,
        create_sharded_memory_config,
        validate_combine,
    )

    rows, cols = mesh_device.shape
    assert (rows, cols) == (2, 4), f"this test is written for the (2,4) block, got {(rows, cols)}"
    mesh_device.enable_program_cache()
    mesh_shape, cluster_axis = (rows, cols), 0
    experts_per_device, tokens_per_device = 16, 32
    num_devices = rows * cols
    total_tokens = tokens_per_device * mesh_shape[cluster_axis]
    host, weight_inputs = _build_moe_host_data(
        mesh_shape,
        cluster_axis,
        experts_per_device,
        tokens_per_device,
        K,
        1,
        N,
        H,
        ttnn.bfloat16,
        MoEActivationFunction.GELU,
        False,
    )
    if host.w0_w1_host is None:
        host.w0_w1_host, host.w2_host = _build_quantized_weight_host_tensors(
            mesh_device, weight_inputs, 1, experts_per_device, H, N, False
        )
    mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])
    drain = ttnn.experimental.get_moe_tilize_drain_core(mesh_device, HS, 4, H, mux_core_range_set=mux)
    drain_crs = ttnn.CoreRangeSet([ttnn.CoreRange(drain, drain)])
    tt_mapping = ttnn.from_torch(
        host.expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_sparse = ttnn.from_torch(
        host.sparse_buffers[0],
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    def replicated_drain(t, dtype):
        flat = t.reshape(total_tokens, K).unsqueeze(0).repeat(num_devices, 1, 1)
        return ttnn.from_torch(
            flat,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=create_sharded_memory_config(drain_crs, [total_tokens, K], dtype),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    tt_idx = replicated_drain(host.expert_indices_list[0], ttnn.uint16)
    tt_sc = replicated_drain(host.expert_scores_list[0], ttnn.bfloat16)
    wmc = ttnn.experimental.get_weight_mem_configs(
        mesh_device,
        num_layers=1,
        experts_per_device=experts_per_device,
        hidden_size=H,
        intermediate_size=N,
        has_bias=False,
    )
    tt_w01 = ttnn.to_device(host.w0_w1_host, mesh_device, memory_config=wmc.w0_w1)
    tt_w2 = ttnn.to_device(host.w2_host, mesh_device, memory_config=wmc.w2)
    combine_cores = ttnn.experimental.get_moe_combine_cores(mesh_device, HS, 4, H, mux_core_range_set=mux)
    combine_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in combine_cores])
    sem = ttnn.create_global_semaphore(mesh_device, combine_crs, 0)

    def combine_output():
        return ttnn.from_torch(
            torch.zeros(K, total_tokens, H, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )

    ref_out = combine_output()
    ref = ttnn.experimental.moe_compute(
        tt_sparse,
        tt_idx,
        tt_sc,
        tt_mapping,
        tt_w01,
        tt_w2,
        layer_id=0,
        output_height_shard_dim=HS,
        intermediate_size=N,
        has_bias=False,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        num_links=None,
        mux_core_range_set=mux,
        optional_output_tensor=ref_out,
        optional_cross_device_semaphore=sem,
        activation_type=MoEActivationFunction.GELU,
    )
    ttnn.synchronize_device(mesh_device)
    ref_bits = [_bits(ttnn.to_torch(t, mesh_composer=None)) for t in ttnn.get_device_tensors(ref[5])]

    op = MoeComputeCustom(
        mesh_device,
        hidden_size=H,
        intermediate_size=N,
        experts_per_device=experts_per_device,
        select_experts_k=K,
        output_height_shard_dim=HS,
        mux_core_range_set=mux,
        activation_type=MoEActivationFunction.GELU,
        cluster_axis=cluster_axis,
    )
    for trial in range(2):
        counts, activation, e_t, shared, staging = op(tt_sparse, tt_idx, tt_sc, tt_mapping, tt_w01, tt_w2, layer_id=0)
        out = ttnn.experimental.selective_reduce_combine(
            staging,
            activation,
            e_t,
            counts,
            H,
            1,
            total_tokens,
            K,
            cluster_axis,
            topology=ttnn.Topology.Linear,
            num_links=ttnn.get_num_links(mesh_device, cluster_axis),
            token_parallel_core_dim=HS,
            data_parallel_core_dim=4,
            worker_cores=combine_cores,
            mux_core_range_set=mux,
            output_tensor=combine_output(),
            optional_cross_device_semaphore=sem,
        )
        ttnn.synchronize_device(mesh_device)
        for d, (mine, theirs) in enumerate(zip(ttnn.get_device_tensors(out), ref_bits)):
            diff = int((_bits(ttnn.to_torch(mine, mesh_composer=None)) != theirs).sum())
            assert diff == 0, f"trial {trial} device {d}: {diff} bf16 values differ from the shared FullCcl op"
        assert validate_combine(0, mesh_device, cluster_axis, out, host.combine_goldens, 0.98)
        for t in (counts, activation, e_t, shared):
            ttnn.deallocate(t)
    logger.info("block (2,4): custom op + standalone combine bitwise equal to the shared FullCcl op on every chip")


def test_capacity_and_arch_contract_host():
    """The wrapper's refusal (tt/experts/moe_decode.py keeps the shared op with one warning): the standalone staging
    holds E x T <= 64 x hs x ws segment rows (the Galaxy's 4 experts x 256 frame rows fit the 360448 B shard, a
    (2,4) block's 16 experts x 128 rows do not), the frame splits over the height shards, and only Blackhole."""
    assert MoeComputeCustom.staging_capacity_reason(H, 4, 256, HS) is None
    assert MoeComputeCustom.staging_capacity_reason(H, 16, 64, HS) is None  # the (2,4) block at frame 64: exactly
    assert "exceed" in MoeComputeCustom.staging_capacity_reason(H, 16, 128, HS)
    assert "split" in MoeComputeCustom.staging_capacity_reason(H, 4, 258, HS)

    class _Dev:
        def __init__(self, arch):
            self._arch = arch

        def arch(self):
            return self._arch

    kw = dict(hidden_size=H, experts_per_device=4, tokens=256, output_height_shard_dim=HS)
    assert MoeComputeCustom.unsupported_reason(_Dev(ttnn.Arch.BLACKHOLE), **kw) is None
    assert MoeComputeCustom.unsupported_reason(_Dev(ttnn.Arch.WORMHOLE_B0), **kw).startswith("Blackhole-only")
    assert "exceed" in MoeComputeCustom.unsupported_reason(_Dev(ttnn.Arch.BLACKHOLE), **dict(kw, tokens=2048))


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_routing_shards_off_the_drain_core_refused(mesh_device, device_params, expect_error):
    """The tilize CBs bind the routing shards in place on the drain core; a routing tensor sharded anywhere else is
    refused by the op instead of being read from the wrong core."""
    s = Setup(mesh_device, E=4, T=256)
    op = s.custom(mesh_device, "standalone", skip_zero_scores=True)
    drain = ttnn.CoreCoord(*op.cores["tilize"][0])
    other = ttnn.CoreCoord(0, 0) if (drain.x, drain.y) != (0, 0) else ttnn.CoreCoord(1, 0)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(other, other)]), [s.T, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    idx_elsewhere = ttnn.from_torch(
        s.idx.reshape(1, s.T, K), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16, memory_config=mc
    )
    with expect_error(AssertionError, "tilize drain core"):
        op(s.tt_tokens, idx_elsewhere, s.tt_sc, s.tt_mapping, s.tt_w01, s.tt_w2)
