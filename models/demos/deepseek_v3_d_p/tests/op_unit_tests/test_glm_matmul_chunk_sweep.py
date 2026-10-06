# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Program-config sweep of every GLM-5.3 layer matmul at a given chunk size, batch 1 (TP=4) and batch 4
(batch-axis, TP=1), so a new chunk size gets tuned entries instead of TTNN's default tiling.

Per chunk C the per-chip row count is m = C / SP (SP = 8). Shapes, dtypes, compute configs and output
memory are the model's own (mla.py / indexer.py / tt_moe_gate_prefill.py / tt_shared_expert.py); only the
program config is swept, plus DRAM instead of an L1 output where the model's 5k entry puts it in L1:

  layout b1 (TP=4)                                 layout b4 (TP=1, batch-axis)
    q_a_proj            m   x 1536  x 2048            m  x 6144  x 2048
    q_b_proj            m   x 2048  x 4096            m  x 2048  x 16384
    kv_a_proj_with_mqa  m   x 1536  x 576             m  x 6144  x 576
    wkv_b1        Z=16  m   x 192   x 512       Z=64  m  x 192   x 512
    wkv_b2        Z=16  m   x 512   x 256       Z=64  m  x 512   x 256
    o_proj              m   x 4096  x 6144            m  x 16384 x 6144
    indexer.wq_b        m/4 x 2048  x 4096            m  x 2048  x 4096
    indexer.wk          m   x 1536  x 128             m  x 6144  x 128
    indexer.weights_proj m  x 1536  x 32              m  x 6144  x 32
    indexer.q_hadamard Z=32 m/4 x 128 x 128     Z=32  m  x 128   x 128  (weight broadcast over heads)
    indexer.k_hadamard  m   x 128   x 128             m  x 128   x 128
    moe.gate            m   x 1536  x 256  (HiFi4)    4m x 6144  x 256
    moe.shared_gate/up  m   x 6144  x 512  (12x9)     4m x 6144  x 512
    moe.shared_down     m   x 512   x 6144 (12x9)     4m x 512   x 6144

Every candidate runs ITERS times behind a `SWEEP|<layout>|<chunk>|<name>|<idx>` signpost and is PCC-checked
(a Reuse config with more blocks than cores runs and returns garbage). Rejected and wrong candidates are
logged and skipped. The candidate catalog (signpost -> config) is written to $SWEEP_OUT_DIR for the parser
(models/demos/deepseek_v3_d_p/utils/parse_matmul_chunk_sweep.py), which picks the fastest per matmul from
the tracy CSV. Chip-local, so one chip measures it: the test opens a 1x1 mesh.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked_batch4 import CHUNK_SIZES, chunk_id
from models.demos.deepseek_v3_d_p.tt.glm_chunk_matmul_configs import program_config_from_desc
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_BATCH_AXIS_MATMUL_CONFIG, MLA_MATMUL_CONFIG

PCC_REQUIRED = 0.99
ITERS = 3  # the parser keeps the fastest
SP, TP = 8, 4
MLA_GRID = (11, 10)
SHARED_GRID = (12, 9)  # the shared expert's overlapped sub-device: grid rows 1..9

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
BF16, BF8 = ttnn.bfloat16, ttnn.bfloat8_b
HIFI2 = dict(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
HIFI4_FP32 = dict(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _spec(
    name,
    z,
    m,
    k,
    n,
    in0,
    in1,
    out_dtype,
    out_mem,
    *,
    table=None,
    key_m=None,
    act_mem=DRAM,
    ckc=HIFI2,
    grid=MLA_GRID,
    op="linear",
    broadcast=False,
    batched=False,
    l1_out_alt=True,
):
    return dict(
        name=name,
        z=z,
        m=m,
        k=k,
        n=n,
        in0=in0,
        in1=in1,
        out_dtype=out_dtype,
        out_mem=out_mem,
        table=table,
        key_m=key_m if key_m is not None else m,
        act_mem=act_mem,
        ckc=ckc,
        grid=grid,
        op=op,
        broadcast=broadcast,
        batched=batched,
        l1_out_alt=l1_out_alt,
    )


def _mla_entry(table, name, key):
    """The model's 640-row entry for this weight: the out dtype / memory a new chunk size inherits."""
    t = MLA_BATCH_AXIS_MATMUL_CONFIG if table == "bax" else MLA_MATMUL_CONFIG
    e = t.get(name, {}).get(key)
    cands = e if isinstance(e, list) else [e]
    for c in cands:
        if c and c.get("num_heads") in (None, 64) and c.get("q_lora_rank") in (None, 2048):
            return c
    return None


def specs(layout: str, chunk: int):
    m = chunk // SP
    b4 = layout == "b4"
    tp = 1 if b4 else TP
    table = "bax" if b4 else "tp"
    hidden_k = 6144 // tp
    rows_q = m if b4 else m // TP  # indexer query rows: TP-split over the sequence in batch 1
    heads = 64 // tp
    out = []

    def mla(
        name,
        z,
        mm,
        k,
        n,
        in0,
        in1,
        default_out_dtype,
        key_m=None,
        op="linear",
        broadcast=False,
        batched=False,
        fixed_out=None,
    ):
        ref = _mla_entry(table, name, 640 if not (name in ("indexer.wq_b", "indexer.q_hadamard") and not b4) else 160)
        out_dtype = (ref or {}).get("out_dtype", default_out_dtype)
        out_mem = fixed_out or (ref or {}).get("out_mem_config", DRAM)
        act_mem = (ref or {}).get("act_mem_config", DRAM)
        out.append(
            _spec(
                name,
                z,
                mm,
                k,
                n,
                in0,
                in1,
                out_dtype,
                out_mem,
                table=table,
                key_m=key_m,
                act_mem=act_mem,
                op=op,
                broadcast=broadcast,
                batched=batched,
                l1_out_alt=fixed_out is None,
            )
        )

    mla("q_a_proj", 1, m, hidden_k, 2048, BF16, BF8, BF16)
    mla("q_b_proj", 1, m, 2048, heads * 256, BF16, BF8, BF16)
    mla("kv_a_proj_with_mqa", 1, m, hidden_k, 576, BF16, BF8, BF16)
    mla("wkv_b1", heads, m, 192, 512, BF16, BF8, BF16, batched=True)
    mla("wkv_b2", heads, m, 512, 256, BF16, BF8, BF8, batched=True)
    mla("o_proj", 1, m, heads * 256, 6144, BF8, BF8, BF16)
    mla("indexer.wq_b", 1, rows_q, 2048, 4096, BF16, BF8, BF16, key_m=rows_q)
    mla("indexer.wk", 1, m, hidden_k, 128, BF16, BF8, BF16)
    mla("indexer.weights_proj", 1, m, hidden_k, 32, BF16, BF16, BF16)
    mla(
        "indexer.q_hadamard",
        32,
        rows_q,
        128,
        128,
        BF16,
        BF16,
        BF8,
        key_m=rows_q,
        op="matmul",
        broadcast=True,
        fixed_out=DRAM,
    )
    mla("indexer.k_hadamard", 1, m, 128, 128, BF16, BF16, BF16, op="matmul", broadcast=True, fixed_out=DRAM)
    moe_m = 4 * m if b4 else m
    gate_k = 6144 if b4 else 6144 // TP
    out.append(
        _spec("moe.gate", 1, moe_m, gate_k, 256, BF16, BF16, BF16, L1, table="gate", ckc=HIFI4_FP32, l1_out_alt=False)
    )
    for nm, k, n in (("moe.shared_gate_up", 6144, 512), ("moe.shared_down", 512, 6144)):
        out.append(_spec(nm, 1, moe_m, k, n, BF16, BF8, BF16, DRAM, table="shared", grid=SHARED_GRID, l1_out_alt=False))
    return out


# --------------------------------------------------------------------------------------- candidates
def _divisors(x):
    return [d for d in range(1, x + 1) if x % d == 0]


def _subblock(pcm, pcn, max_dst):
    """Largest h x w <= max_dst with h | pcm, w | pcn; wider first."""
    best = (1, 1)
    for w in _divisors(pcn):
        for h in _divisors(pcm):
            if h * w <= max_dst and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _ibs(kt, n=4):
    """Largest K blocks first: deep blocks won at 256 rows (in0_block_w 24/32/48 for the K=6144 stems)."""
    return [d for d in (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1) if kt % d == 0][:n]


def candidates(s):
    """[(desc dict)] -- desc carries everything needed to rebuild the program config."""
    mt, kt, nt = math.ceil(s["m"] / 32), s["k"] // 32, s["n"] // 32
    gx, gy = s["grid"]
    max_dst = 4 if s["ckc"]["fp32_dest_acc_en"] else 8
    out = [dict(kind="default")]
    if s["batched"]:
        z = s["z"]
        pairs = []
        for pcm in _divisors(mt):
            for pcn in _divisors(nt):
                cores = z * (mt // pcm) * (nt // pcn)
                if cores <= gx * gy:
                    pairs.append((cores, -pcm * pcn, pcm, pcn))
        for cores, _, pcm, pcn in sorted(pairs, reverse=True)[:3]:
            for ib in _ibs(kt):
                sh, sw = _subblock(pcm, pcn, max_dst)
                out.append(dict(kind="reuse", ib=ib, sh=sh, sw=sw, pcm=pcm, pcn=pcn, cores=cores))
        return out
    if s["broadcast"]:
        # Batch folded into M (fuse_batch): split the Z * M_t rows over the cores, all of N per core. The three
        # smallest per-core heights that fit the grid = the three highest core counts.
        meff = s["z"] * mt
        for pcm in sorted({math.ceil(meff / c) for c in range(1, gx * gy + 1)})[:3]:
            cores = math.ceil(meff / pcm)
            for ib in _ibs(kt, 2):
                sh, sw = _subblock(pcm, nt, max_dst)
                out.append(dict(kind="mc1d_in1", ib=ib, sh=sh, sw=sw, pcm=pcm, pcn=nt, cores=cores))
        if s["z"] == 1:
            out += _mc2d_cands(mt, kt, nt, gx, gy, max_dst)
        return out
    out += _mc2d_cands(mt, kt, nt, gx, gy, max_dst)
    # 1D: in0-multicast splits N (narrow M), in1-multicast splits M (tall M, narrow N).
    for pcn in sorted({math.ceil(nt / c) for c in (nt, max(1, nt // 2))}):
        cores = math.ceil(nt / pcn)
        if cores <= gx * gy:
            for ib in _ibs(kt, 2):
                sh, sw = _subblock(mt, pcn, max_dst)
                out.append(dict(kind="mc1d_in0", ib=ib, sh=sh, sw=sw, pcm=mt, pcn=pcn, cores=cores))
    if nt <= 16:
        for c in (gx * gy, mt):
            pcm = max(1, math.ceil(mt / c))
            cores = math.ceil(mt / pcm)
            for ib in _ibs(kt, 2):
                sh, sw = _subblock(pcm, nt, max_dst)
                out.append(dict(kind="mc1d_in1", ib=ib, sh=sh, sw=sw, pcm=pcm, pcn=nt, cores=cores))
    # dedupe
    uniq, seen = [], set()
    for d in out:
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def _mc2d_cands(mt, kt, nt, gx, gy, max_dst):
    pcms = sorted({math.ceil(mt / r) for r in range(1, gy + 1)})
    pcns = sorted({math.ceil(nt / c) for c in range(1, gx + 1)})
    combos = [(math.ceil(mt / a) * math.ceil(nt / b), a, b) for a in pcms for b in pcns]
    maxc = max(c for c, _, _ in combos)
    keep = sorted([x for x in combos if x[0] >= 0.75 * maxc], key=lambda x: (-x[0], x[1] * x[2]))[:4]
    out = []
    for cores, pcm, pcn in keep:
        for ib in _ibs(kt):
            sh, sw = _subblock(pcm, pcn, max_dst)
            out.append(dict(kind="mc2d", ib=ib, sh=sh, sw=sw, pcm=pcm, pcn=pcn, cores=cores))
    return out


def build_program_config(d, grid):
    return program_config_from_desc({**d, "grid": list(grid)})


# --------------------------------------------------------------------------------------------- run
def _run(mesh_device, s, prog, out_mem, tag):
    torch.manual_seed(0)
    z_a = s["z"]
    a = torch.randn(1, z_a, s["m"], s["k"], dtype=torch.bfloat16)
    w = torch.randn(1, 1 if (s["broadcast"] or not s["batched"]) else s["z"], s["k"], s["n"], dtype=torch.bfloat16)
    w = w * 0.02
    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_a = ttnn.from_torch(
        a, device=mesh_device, dtype=s["in0"], layout=ttnn.TILE_LAYOUT, memory_config=s["act_mem"], mesh_mapper=repl
    )
    tt_w = ttnn.from_torch(
        w, device=mesh_device, dtype=s["in1"], layout=ttnn.TILE_LAYOUT, memory_config=DRAM, mesh_mapper=repl
    )
    ckc = ttnn.init_device_compute_kernel_config(mesh_device.arch(), **s["ckc"])
    kwargs = {"program_config": prog} if prog is not None else {}
    op = ttnn.matmul if s["op"] == "matmul" else ttnn.linear
    ttnn.tracy_message(f"`TT_SIGNPOST: {tag}`")
    got = None
    try:
        for _ in range(ITERS):
            if got is not None:
                ttnn.deallocate(got)
            got = op(tt_a, tt_w, memory_config=out_mem, dtype=s["out_dtype"], compute_kernel_config=ckc, **kwargs)
            ttnn.synchronize_device(mesh_device)
        dev = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
        ok, pcc = comp_pcc(torch.matmul(a.float(), w.float()), dev, PCC_REQUIRED)
        return ok, pcc
    finally:
        for t in (tt_a, tt_w, got):
            if t is not None:
                ttnn.deallocate(t)


# 5k / 2k too: their hand-tuned table entries always win (the merge only fills empty slots), so the sweep only adds
# the slots they lack -- batch-1 MLA at 256 rows, the batch-4 full-width gate and shared expert at 4x the rows.
@pytest.mark.parametrize("chunk", list(CHUNK_SIZES), ids=lambda c: chunk_id(c))
@pytest.mark.parametrize("layout", ["b1", "b4"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.timeout(0)
def test_glm_matmul_chunk_sweep(mesh_device, layout, chunk):
    out_dir = Path(os.environ.get("SWEEP_OUT_DIR", "generated/matmul_chunk_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog, rejected, wrong = {}, [], []
    for s in specs(layout, chunk):
        cands = candidates(s)
        out_mems = [s["out_mem"]] + (
            [DRAM] if s["l1_out_alt"] and s["out_mem"].buffer_type == ttnn.BufferType.L1 else []
        )
        logger.info(
            f"[sweep] {layout} {chunk} {s['name']}: Z={s['z']} {s['m']}x{s['k']}x{s['n']}, "
            f"{len(cands)} configs x {len(out_mems)} out mem"
        )
        idx = 0
        for d in cands:
            for om in out_mems:
                tag = f"SWEEP|{layout}|{chunk}|{s['name']}|{idx}"
                idx += 1
                entry = dict(
                    spec={k: v for k, v in s.items() if k in ("name", "z", "m", "k", "n", "table", "key_m", "grid")},
                    cfg=d,
                    out_mem="L1" if om.buffer_type == ttnn.BufferType.L1 else "DRAM",
                    out_dtype=str(s["out_dtype"]),
                    act_mem="L1" if s["act_mem"].buffer_type == ttnn.BufferType.L1 else "DRAM",
                )
                try:
                    ok, pcc = _run(mesh_device, s, build_program_config(d, s["grid"]), om, tag)
                except Exception as e:  # rejected by the op (grid / L1 / validation)
                    rejected.append(tag)
                    entry["status"] = f"rejected: {str(e).splitlines()[0][:160]}"
                else:
                    entry["status"] = "ok" if ok else f"wrong pcc {pcc}"
                    entry["pcc"] = float(pcc) if not isinstance(pcc, str) else None
                    if not ok:
                        wrong.append(tag)
                catalog[tag] = entry
    ttnn.ReadDeviceProfiler(mesh_device)
    path = out_dir / f"catalog_{layout}_{chunk}.json"
    path.write_text(json.dumps(catalog, indent=1, default=str))
    logger.info(
        f"[sweep] {layout} {chunk}: {len(catalog)} candidates, {len(rejected)} rejected, {len(wrong)} wrong "
        f"-> {path}"
    )
