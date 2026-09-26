# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE dispatch op in isolation, at MiMo-V2 shapes, with controlled routing: for profiling how the op behaves.

The op (ttnn.experimental.deepseek_prefill.dispatch) is the all-to-all inside one dispatch group (a mesh column,
cluster_axis=0): every (token, expert) pair's 4096-wide row is copied from the token's chip into the flat dispatch
buffer of the chip that holds the expert. Per fabric link it runs one sender core (fabric only) and
``num_workers_per_sender`` worker cores (read + untilize tokens, pick destinations, write local pairs over NoC and
hand remote pairs to the sender). Routing setup (bincount + all-gather + offset_cumsum) runs first and is not timed.

Routing modes (tokens are TP-replicated, so both columns see the same indices; each column keeps its own half):
  uniform      8 distinct experts drawn uniformly from 256
  local        4 experts per column, all on the token's own chip -> NoC only, no fabric
  remote       4 experts per column, all on the other chip of the column -> every pair crosses the fabric
  measured_L1  sampled from the per-expert counts of the real L1 router (51 of 256 experts active, very skewed)
  measured_L5  same for L5 (196 active)

Each case: 1 warm-up + ``MIMO_DISPATCH_ITERS`` signposted runs tagged
``dispatch_{mode}_S{seq}_{layout}_links{L}_w{W}``, and a JSON line of host-side routing stats in
``generated/mimo_dispatch/cases.jsonl``. The warm-up run is checked against the inputs (every received row matches
the token its metadata names). Summarise with analyze_dispatch.py <ops_perf_results.csv>.

    scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_dispatch_perf.py
    MIMO_DISPATCH_MODES=remote MIMO_DISPATCH_LINKS=1,2,3 MIMO_DISPATCH_WORKERS=1,2,3 ...   # sweeps
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.ffn import moe_capacity_factor

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


MODES = _env_list("MIMO_DISPATCH_MODES", "uniform,local,remote,measured_L1,measured_L5")
SEQS = _env_list("MIMO_DISPATCH_SEQ", "640", int)
LAYOUTS = _env_list("MIMO_DISPATCH_LAYOUT", "tile,row_major")
LINKS = _env_list("MIMO_DISPATCH_LINKS", "1,3", int)
WORKERS = _env_list("MIMO_DISPATCH_WORKERS", "2", int)
ITERS = int(os.environ.get("MIMO_DISPATCH_ITERS", "3"))
CHECK = os.environ.get("MIMO_DISPATCH_CHECK", "1") == "1"
STATS_PATH = Path(os.environ.get("MIMO_DISPATCH_STATS", "generated/mimo_dispatch/cases.jsonl"))
COUNTS_PATH = Path(__file__).with_name("routing_counts_1280tok.json")


def make_indices(mode, sp, tp, S, E, K, gen):
    """Global expert ids [sp * S, K] (row r's tokens first). Expert e lives in column e // (E / tp), on row
    (e % (E / tp)) // (E / tp / sp) of that column (ExpertMapping.create_dispatch_table)."""
    per_col = E // tp
    epc = per_col // sp
    rows = []
    for r in range(sp):
        if mode == "uniform":
            idx = torch.rand(S, E, generator=gen).argsort(-1)[:, :K]
        elif mode in ("local", "remote"):
            src = r if mode == "local" else (r + 1) % sp
            parts = [
                torch.rand(S, epc, generator=gen).argsort(-1)[:, : K // tp] + c * per_col + src * epc for c in range(tp)
            ]
            idx = torch.cat(parts, -1)
        elif mode.startswith("measured_"):
            p = torch.tensor(json.loads(COUNTS_PATH.read_text())[mode.split("_", 1)[1]], dtype=torch.float)
            idx = torch.multinomial(p.expand(S, E), K, replacement=False, generator=gen)
        else:
            raise ValueError(mode)
        rows.append(idx)
    return torch.cat(rows, 0)


def routing_stats(indices, sp, tp, S, E):
    """Per destination chip (row, col): (token, expert) pairs received, split local (same row) / remote."""
    per_col, epc = E // tp, E // tp // sp
    src_row = torch.arange(sp).repeat_interleave(S)[:, None].expand_as(indices)
    col, dst_row = indices // per_col, (indices % per_col) // epc
    chips = {}
    for c in range(tp):
        for r in range(sp):
            here = (col == c) & (dst_row == r)
            chips[f"{r},{c}"] = {"pairs": int(here.sum()), "remote": int((here & (src_row != r)).sum())}
    active = int((torch.bincount(indices.flatten(), minlength=E) > 0).sum())
    return chips, active


def check_dispatch(mesh_device, x_host, buf, meta, counts, regions, sp, tp, S, E):
    """Every row the op wrote must be the token its metadata points at; every chip gets exactly its pairs."""
    per_col, epc = E // tp, E // tp // sp
    bufs = [ttnn.to_torch(t).reshape(-1, x_host.shape[-1]).float() for t in ttnn.get_device_tensors(buf)]
    metas = [ttnn.to_torch(t).reshape(bufs[0].shape[0], -1) for t in ttnn.get_device_tensors(meta)]
    cnts = [ttnn.to_torch(t).flatten().long() for t in ttnn.get_device_tensors(counts)]
    regs = [ttnn.to_torch(t).flatten().long() for t in ttnn.get_device_tensors(regions)]
    for dev in range(sp * tp):
        r, c = divmod(dev, tp)
        for e in range(c * per_col + r * epc, c * per_col + (r + 1) * epc):
            n, start = int(cnts[dev][e]), int(regs[dev][e])
            if n == 0:
                continue
            m = metas[dev][start : start + n]
            src_rows = m[:, 0].long() // tp  # linearized source mesh coord -> source row
            want = x_host[src_rows, m[:, 1].long()]
            got = bufs[dev][start : start + n]
            assert torch.equal(got, want), f"chip ({r},{c}) expert {e}: dispatched rows do not match their metadata"


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("workers", WORKERS, ids=lambda w: f"w{w}")
@pytest.mark.parametrize("num_links", LINKS, ids=lambda n: f"links{n}")
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("seq", SEQS, ids=lambda s: f"S{s}")
@pytest.mark.parametrize("mode", MODES)
def test_dispatch_perf(mesh_device, device_params, mode, seq, layout, num_links, workers):
    # The op lays [sender, worker x W] per link along one core row; BH exposes 11 worker cores in that row.
    if num_links * (1 + workers) > mesh_device.compute_with_storage_grid_size().x:
        pytest.skip(f"{num_links} links x (1 sender + {workers} workers) does not fit one core row")
    cfg = MiMoTextConfig.from_json()
    E, K, H = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size
    sp, tp = tuple(mesh_device.shape)
    n_dev = sp * tp
    mc = extract_mesh_config(mesh_device)
    dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
    cap = moe_capacity_factor(K, E, n_dev)
    experts_per_chip, metadata_len, max_buf, _ = compute_constants(seq, E, K, n_dev, dgs, cap)
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])

    gen = torch.Generator().manual_seed(0)
    indices = make_indices(mode, sp, tp, seq, E, K, gen)
    chips, active = routing_stats(indices, sp, tp, seq, E)
    tag = f"dispatch_{mode}_S{seq}_{layout}_links{num_links}_w{workers}"
    stats = {
        "tag": tag,
        "mode": mode,
        "seq": seq,
        "layout": layout,
        "num_links": num_links,
        "workers": workers,
        "emb_dim": H,
        "row_bytes": 2 * H,
        "buffer_rows": max_buf,
        "active_experts": active,
        "chips": chips,
    }
    logger.info(
        f"{tag}: active experts {active}/{E}, received pairs per chip (remote) "
        + ", ".join(f"({k}) {v['pairs']} ({v['remote']})" for k, v in chips.items())
    )
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(json.dumps(stats) + "\n")

    table = ExpertMapping.create_dispatch_table(E, dgs, ndg)
    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, dispatch_axis=0)
    routing = TtMoERoutingSetup(mesh_device, table, num_links=num_links, experts_per_chip=experts_per_chip)
    dispatch = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dgs,
        experts_per_chip=experts_per_chip,
        num_routed_experts=E,
        num_experts_per_tok=K,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_buf,
        seq_len_per_chip=seq,
        emb_dim=H,
        cluster_axis=0,
        num_links=num_links,
        topology=sp_topo,
        subdevice_id=None,
        num_workers_per_sender=workers,
    )

    rows = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None))  # SP-sharded, TP-replicated
    x_host = torch.randn(sp, seq, H).bfloat16()
    x = ttnn.from_torch(
        x_host,
        device=mesh_device,
        mesh_mapper=rows,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT,
    )
    idx_2d = ttnn.from_torch(
        indices.to(torch.int32), device=mesh_device, mesh_mapper=rows, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    idx = ttnn.reshape(idx_2d, (1, seq, K))
    w = ttnn.from_torch(
        torch.full((sp, seq, K), 1.0 / K),
        device=mesh_device,
        mesh_mapper=rows,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )  # unused by the op; kept for the call signature

    for it in range(1 + ITERS):
        offsets, counts, regions, _ = routing(
            ttnn_top_k_experts_indices=idx_2d, num_routed_experts=E, num_experts_per_tok=K
        )
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        buf, meta = dispatch(x, w, idx, offsets, tt_table)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
        elif CHECK:
            check_dispatch(mesh_device, x_host.float(), buf, meta, counts, regions, sp, tp, seq, E)
        for t in (buf, meta, offsets, counts, regions):
            t.deallocate(True)
    logger.info(f"ran {tag}")


AG_LINKS = _env_list("MIMO_AG_LINKS", "1,2,3,4", int)
AG_WORKERS = _env_list("MIMO_AG_WORKERS", "1,2,4", int)
AG_DTYPES = _env_list("MIMO_AG_DTYPES", "bf16,bf8")


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("workers_per_link", AG_WORKERS, ids=lambda w: f"wpl{w}")
@pytest.mark.parametrize("num_links", AG_LINKS, ids=lambda n: f"links{n}")
@pytest.mark.parametrize("dtype", AG_DTYPES)
@pytest.mark.parametrize("seq", SEQS, ids=lambda s: f"S{s}")
def test_all_gather_perf(mesh_device, device_params, seq, dtype, num_links, workers_per_link):
    """The alternative to dispatch: all-gather the column's tokens (cluster_axis=0) so every chip holds the whole
    chunk and picks its experts' rows locally. Each chip sends its [S, 4096] slab once to every other chip of the
    column, regardless of routing. Tag ``allgather_S{seq}_{dtype}_links{L}_wpl{W}``; summarise with analyze_tags.py."""
    H = MiMoTextConfig.from_json().hidden_size
    sp, tp = tuple(mesh_device.shape)
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    tt_dtype = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[dtype]
    x = ttnn.from_torch(
        torch.randn(sp, 1, seq, H),
        device=mesh_device,
        dtype=tt_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tag = f"allgather_S{seq}_{dtype}_links{num_links}_wpl{workers_per_link}"
    for it in range(1 + ITERS):
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        y = ttnn.all_gather(
            x,
            dim=2,
            cluster_axis=0,
            num_links=num_links,
            topology=sp_topo,
            num_workers_per_link=workers_per_link,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
        assert tuple(y.shape) == (1, 1, sp * seq, H)
        y.deallocate(True)
    logger.info(f"ran {tag}")


HBW_LINKS = _env_list("MIMO_HBW_LINKS", "0,1,2,3", int)  # 0 = op default (every usable link on the axis)
HBW_LAYOUTS = _env_list("MIMO_HBW_LAYOUT", "tile,row_major")


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("num_links", HBW_LINKS, ids=lambda n: f"links{n or 'auto'}")
@pytest.mark.parametrize("layout", HBW_LAYOUTS)
@pytest.mark.parametrize("dtype", AG_DTYPES)
@pytest.mark.parametrize("seq", SEQS, ids=lambda s: f"S{s}")
def test_high_bw_all_gather_perf(mesh_device, device_params, seq, dtype, layout, num_links):
    """Same gather as test_all_gather_perf with ttnn.experimental.high_bw_all_gather (persistent output, native
    one-hop store-and-forward transport). Tag ``hbwag_S{seq}_{dtype}_{layout}_links{L|auto}``."""
    if dtype == "bf8" and layout == "row_major":
        pytest.skip("bfloat8_b is tile-only")
    H = MiMoTextConfig.from_json().hidden_size
    sp, tp = tuple(mesh_device.shape)
    tt_dtype = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[dtype]
    tt_layout = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    x = ttnn.from_torch(
        torch.randn(sp, 1, seq, H),
        device=mesh_device,
        dtype=tt_dtype,
        layout=tt_layout,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros(1, 1, sp * seq, H),
        device=mesh_device,
        dtype=tt_dtype,
        layout=tt_layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tag = f"hbwag_S{seq}_{dtype}_{layout}_links{num_links or 'auto'}"
    for it in range(1 + ITERS):
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        ttnn.experimental.high_bw_all_gather(x, dim=2, output_tensor=out, cluster_axis=0, num_links=num_links or None)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
    if CHECK:  # every chip must hold its column's full chunk, in row order
        want = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, 3)))
        for dev, got in enumerate(ttnn.get_device_tensors(out)):
            c = dev % tp
            ref = want[:, :, :, c * H : (c + 1) * H].reshape(1, 1, sp * seq, H)
            assert torch.equal(ttnn.to_torch(got).float(), ref.float()), f"device {dev}: gathered data mismatch"
    logger.info(f"ran {tag}")
