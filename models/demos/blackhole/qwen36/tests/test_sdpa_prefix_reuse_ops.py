# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device tests (8-chip TP row, e.g. one Galaxy row with QWEN36_FABRIC_RING=1) of the chunked prefill SDPA prefix
reuse at op level: ttnn.transformer.chunked_scaled_dot_product_attention exactly as TPAttention.forward_prefill_paged
calls it (3 local Q heads, 1 KV head, head dim 256, bf8 Q and paged bf8 K/V with 64-token blocks, HiFi2 with fp32 dest,
approximate exp, K chunk 128, full grid), the chunk start from a device tensor (the traced path) or a scalar.

* test_prefix_reuse_bitwise: SDPAProgramConfig.prefix_reuse bits (1 read batches, 2 prefix chains, 4 unpaired, chain
  lengths) and the smaller Q chunks of QWEN36_SDPA_PREFILL_QSPLIT give outputs bit-identical to today's call
  (prefix_reuse 0, Q chunk = K chunk) over chunk sizes 128..8192 and prefixes 0..30K; today's call is checked against
  torch at the small cases.
* test_prefix_reuse_bench (SPR_BENCH=1): device time per call (one trace of SPR_OPS calls, min of SPR_REPS replays) of
  every arm in SPR_BENCH_ARMS at SPR_BENCH_ROWS x SPR_BENCH_DEPTHS; jsonl lines to SPR_OUT.

  source <8-chip env>; python -m pytest models/demos/blackhole/qwen36/tests/test_sdpa_prefix_reuse_ops.py -s
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

_ROUTER = tpc.fabric_router_config()
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "trace_region_size": 1 << 27,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        **({"fabric_router_config": _ROUTER} if _ROUTER is not None else {}),
    }
]
NH, NKV, HD, BLOCK, K_CHUNK = 3, 1, 256, 64, 128
OUT = os.environ.get("SPR_OUT", "/tmp/spr/bench.jsonl")
pytestmark = pytest.mark.skipif(
    not tpc.SDPA_PREFIX_REUSE_SUPPORTED, reason="this ttnn build has no SDPAProgramConfig.prefix_reuse"
)


def emit(rec):
    rec = dict(rec, ts=round(time.time(), 2), load=[round(x, 2) for x in os.getloadavg()])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"[SPR] {json.dumps(rec)}")


class Paged:
    """A replicated paged bf8 K/V cache of max_tokens tokens behind a random block permutation (page table padded to a
    multiple of 32 blocks, as the model pads it). The host copies keep the logical order for the torch reference."""

    def __init__(self, mesh, max_tokens, seed=0, table_blocks=None):
        nblk = -(-max_tokens // BLOCK)
        g = torch.Generator().manual_seed(seed)
        self.k_host = torch.randn(nblk, NKV, BLOCK, HD, generator=g)
        self.v_host = torch.randn(nblk, NKV, BLOCK, HD, generator=g)
        self.perm = torch.randperm(nblk, generator=g).to(torch.int32)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        # physical block perm[i] holds logical block i
        k_phys = torch.empty_like(self.k_host)
        v_phys = torch.empty_like(self.v_host)
        k_phys[self.perm.long()] = self.k_host
        v_phys[self.perm.long()] = self.v_host
        mk = dict(dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.k = ttnn.from_torch(k_phys, mesh_mapper=rep, **mk)
        self.v = ttnn.from_torch(v_phys, mesh_mapper=rep, **mk)
        # the bf8 round trip the reference must see
        self.k_host = ttnn.to_torch(ttnn.get_device_tensors(self.k)[0]).float()[self.perm.long()]
        self.v_host = ttnn.to_torch(ttnn.get_device_tensors(self.v)[0]).float()[self.perm.long()]
        width = max(-(-nblk // 32) * 32, table_blocks or 0)
        pt = torch.zeros(1, width, dtype=torch.int32)
        pt[0, :nblk] = self.perm
        self.pt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        self.max_tokens = nblk * BLOCK


def _q(mesh, rows, seed, nh=NH):
    g = torch.Generator().manual_seed(1000 + seed)
    q = torch.randn(1, nh, rows, HD, generator=g)
    t = ttnn.from_torch(
        q,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return t, ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _start_tensor(mesh, start):
    return ttnn.from_torch(
        torch.tensor([start], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _sdpa(mesh, q, paged, start, start_t, q_chunk, bits):
    grid = mesh.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        exp_approx_mode=True,
        q_chunk_size=q_chunk,
        k_chunk_size=K_CHUNK,
        **({"prefix_reuse": bits} if bits else {}),
    )
    kw = dict(chunk_start_idx_tensor=start_t) if start_t is not None else dict(chunk_start_idx=start)
    return ttnn.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=q,
        input_tensor_k=paged.k,
        input_tensor_v=paged.v,
        page_table_tensor=paged.pt,
        compute_kernel_config=tpc.COMPUTE_HIFI2,
        program_config=cfg,
        **kw,
    )


def _all_devices(t):
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _reference(q_host, paged, start):
    rows = q_host.shape[2]
    end = start + rows
    nblk = -(-end // BLOCK)
    k = paged.k_host[:nblk].permute(1, 0, 2, 3).reshape(1, NKV, nblk * BLOCK, HD)[:, :, :end]
    v = paged.v_host[:nblk].permute(1, 0, 2, 3).reshape(1, NKV, nblk * BLOCK, HD)[:, :, :end]
    k = k.repeat_interleave(q_host.shape[1] // NKV, dim=1)
    v = v.repeat_interleave(q_host.shape[1] // NKV, dim=1)
    pos_q = torch.arange(start, end).unsqueeze(1)
    pos_k = torch.arange(end).unsqueeze(0)
    mask = torch.where(pos_k <= pos_q, 0.0, float("-inf"))
    scores = q_host @ k.transpose(-2, -1) / (HD**0.5) + mask
    return scores.softmax(dim=-1) @ v


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


# (rows, prefix): the model's chunk sizes (8192 traced chunks, 2048 chunks and the 128..1024 short traces)
ID_CASES = [
    (8192, 0),
    (8192, 8192),
    (8192, 24576),
    (2048, 0),
    (2048, 4096),
    (2048, 30720),
    (1024, 0),
    (1024, 9216),
    (512, 2048),
    (256, 0),
    (256, 6144),
    (128, 0),
    (128, 20480),
    # (r7 resume) deep follow-up shapes of QWEN36_SDPA_PREFIX_REUSE=auto (32-row Q chunks + read batches): a resume
    # snapshot 128-aligned but not 256-aligned, and a short trace at 96K
    (128, 39936),
    (256, 98304),
]
# (q_chunk, prefix_reuse) variants checked against today's (K_CHUNK, 0)
VARIANTS = [
    (128, 1),
    (128, 2),
    (128, 3),
    (128, 4),
    (128, 7),
    (128, 0x407),  # chains of at most 4 cores
    (128, 0x1007),  # chains of at most 16 cores
    (64, 0),
    (64, 7),
    (32, 0),
    (32, 4),
    (32, 5),  # auto: split calls on few cores
    (32, 7),
    (64, 5),
]


def _parse_pairs(raw):
    """'a:b,c:d' -> [(a, b), (c, d)] (ints, 0x... allowed)"""
    return [tuple(int(x, 0) for x in item.split(":")) for item in filter(None, raw.split(","))]


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefix_reuse_bitwise(mesh_device):
    """Every (rows, prefix) of SPR_CASES (default ID_CASES), chunk start from a tensor and (8192 / 128 rows) from a
    scalar: each (q_chunk, prefix_reuse) of SPR_VARIANTS (default VARIANTS) equals today's call bit for bit, on all 8
    devices, on a program-cache miss and on the hit after it."""
    mesh = mesh_device
    mesh.enable_program_cache()
    cases = _parse_pairs(os.environ["SPR_CASES"]) if os.environ.get("SPR_CASES") else ID_CASES
    variants = _parse_pairs(os.environ["SPR_VARIANTS"]) if os.environ.get("SPR_VARIANTS") else VARIANTS
    paged = Paged(mesh, max(s + r for r, s in cases) + BLOCK, seed=3)
    bad = []
    for rows, start in cases:
        for flexible in (True, False):
            if not flexible and rows not in (8192, 128):
                continue  # scalar chunk start: 8192 and 128 rows cover it
            q, q_host = _q(mesh, rows, seed=rows + start)
            start_t = _start_tensor(mesh, start) if flexible else None
            base_t = _sdpa(mesh, q, paged, start, start_t, K_CHUNK, 0)
            base = _all_devices(base_t)
            ttnn.deallocate(base_t)
            assert all(torch.equal(d, base[0]) for d in base[1:]), "devices disagree on today's output"
            pcc = None
            if rows * (start + rows) <= 2048 * 34816:
                pcc = _pcc(base[0], _reference(q_host, paged, start))
                assert pcc > 0.99, f"today's call vs torch at rows {rows} start {start}: pcc {pcc}"
            emit(dict(kind="baseline", rows=rows, start=start, flexible=flexible, pcc_vs_torch=pcc))
            for q_chunk, bits in variants:
                if q_chunk > rows:
                    continue
                got, again = None, None
                try:
                    out = _sdpa(mesh, q, paged, start, start_t, q_chunk, bits)
                    got = _all_devices(out)
                    ttnn.deallocate(out)
                    out = _sdpa(mesh, q, paged, start, start_t, q_chunk, bits)  # program cache hit
                    again = _all_devices(out)
                    ttnn.deallocate(out)
                except Exception as e:  # noqa: BLE001
                    err = dict(rows=rows, start=start, flexible=flexible, q_chunk=q_chunk, bits=bits)
                    emit(dict(kind="bitwise", error=str(e)[:400], **err))
                    bad.append((rows, start, flexible, q_chunk, hex(bits), "error"))
                    continue
                same = all(torch.equal(g, base[0]) for g in got)
                same_again = all(torch.equal(g, base[0]) for g in again)
                emit(
                    dict(
                        kind="bitwise",
                        rows=rows,
                        start=start,
                        flexible=flexible,
                        q_chunk=q_chunk,
                        bits=bits,
                        identical=same,
                        identical_cached=same_again,
                        max_abs=float((got[0] - base[0]).abs().max()),
                    )
                )
                if not (same and same_again):
                    bad.append((rows, start, flexible, q_chunk, hex(bits)))
            ttnn.deallocate(q)
            if start_t is not None:
                ttnn.deallocate(start_t)
    assert not bad, f"not bit-identical to today's call: {bad}"


# --------------------------------------------------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------------------------------------------------
def _parse_arms(raw):
    """'name:q_chunk:bits[:nh],...' -> [(name, q_chunk, bits, nh)]"""
    arms = []
    for item in filter(None, raw.split(",")):
        parts = item.split(":")
        arms.append((parts[0], int(parts[1]), int(parts[2], 0), int(parts[3]) if len(parts) > 3 else NH))
    return arms


DEFAULT_ARMS = "base:128:0,reads:128:1,chains:128:2,reads_chains:128:3,all:128:7,base_nh1:128:0:1"


def _trace_ms(mesh, fn, n_ops, reps):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(n_ops)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ts.append(1e3 * (time.perf_counter() - t0) / n_ops)
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return ts


@pytest.mark.timeout(7200)
@pytest.mark.skipif(os.environ.get("SPR_BENCH") != "1", reason="set SPR_BENCH=1")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefix_reuse_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    rows_list = [int(x) for x in os.environ.get("SPR_BENCH_ROWS", "8192").split(",")]
    depths = [int(x) for x in os.environ.get("SPR_BENCH_DEPTHS", "0,8192,24576,49152,98304").split(",")]
    arms = _parse_arms(os.environ.get("SPR_BENCH_ARMS", DEFAULT_ARMS))
    n_ops = int(os.environ.get("SPR_OPS", "4"))
    reps = int(os.environ.get("SPR_REPS", "3"))
    rounds = int(os.environ.get("SPR_ROUNDS", "2"))
    # the model's page table spans max_seq_len (262144 tokens / 64 = 4096 blocks)
    paged_b = Paged(mesh, max(depths) + max(rows_list) + BLOCK, seed=11, table_blocks=4096)
    emit(dict(kind="bench_env", rows=rows_list, depths=depths, arms=arms, n_ops=n_ops, reps=reps, rounds=rounds))
    for rows in rows_list:
        qs = {nh: _q(mesh, rows, seed=rows, nh=nh)[0] for nh in sorted({a[3] for a in arms})}
        for depth in depths:
            start_t = _start_tensor(mesh, depth)
            for rnd in range(rounds):  # interleaved arms, `rounds` times
                for name, q_chunk, bits, nh in arms:
                    if q_chunk > rows:
                        continue
                    try:
                        ts = _trace_ms(
                            mesh, lambda: _sdpa(mesh, qs[nh], paged_b, depth, start_t, q_chunk, bits), n_ops, reps
                        )
                        emit(
                            dict(
                                kind="bench",
                                arm=name,
                                rows=rows,
                                depth=depth,
                                q_chunk=q_chunk,
                                bits=bits,
                                nh=nh,
                                round=rnd,
                                ms=[round(t, 4) for t in ts],
                                ms_min=round(min(ts), 4),
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        emit(dict(kind="bench", arm=name, rows=rows, depth=depth, error=str(e)[:400]))
            ttnn.deallocate(start_t)
        for t in qs.values():
            ttnn.deallocate(t)
