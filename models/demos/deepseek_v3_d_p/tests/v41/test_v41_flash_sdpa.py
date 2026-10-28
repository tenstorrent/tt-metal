# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode step's flash-decode SDPA (``tt/v41/flash_sdpa.py``, ``V41_DECODE_SDPA=flash``)
against ``sparse_sdpa`` on the chip's own heads.

No weights, the decode tile: ``V41_SL_TRIALS`` random bf16 ``[1, 1, 32, 16 * 512]`` query tiles a chip (every chip its
own), random RoPE tables, a random KV cache in both served formats (BF16 rows and the pool's packed scaled FP8),
index rows in the c1 pattern (one live row a mesh row, the others sentinels), the ``k2`` / ``k3`` patterns (the live
row and K draft rows of the same user, each with its own index row: the speculative verify step's layout) and the
32-row pattern (every row live, varying counts). The two SDPAs' ``[1, 32, 32, 512]`` outputs are compared on the live
rows (max abs difference, max relative to the largest output, PCC; the accumulation order differs, so bit equality
is not expected), the dead rows of the flash output checked zero; then each SDPA alone under trace (``V41_SL_REPLAYS``
replays of ``V41_SL_CALLS`` calls) per pattern and format, and the whole local chain (``q_heads``, SDPA, ``o_heads``)
for the c1 pattern. Results: ``V41_FS_RESULT``, ``V41_SL_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.attention import SDPA_MIN_HEADS
from models.demos.deepseek_v3_d_p.tt.v41.flash_sdpa import FlashSdpa
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

SENTINEL = 0xFFFFFFFF
ROWS = 32
PCC_FLOOR = float(os.environ.get("V41_FS_PCC", "0.999"))


def _trace_us(mesh_device, fn, calls: int, replays: int) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [fn() for _ in range(calls)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    best = None
    for _ in range(replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / calls
        best = dt if best is None else min(best, dt)
    ttnn.release_trace(mesh_device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return round(best * 1e6, 1)


def _pcc(a, b) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    if a.std() == 0 or b.std() == 0:
        return float(torch.equal(a, b))
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_flash_sdpa(mesh_device, device_params):
    trials = int(os.environ.get("V41_SL_TRIALS", 3))
    replays = int(os.environ.get("V41_SL_REPLAYS", 20))
    calls = int(os.environ.get("V41_SL_CALLS", 16))
    cache_rows = int(os.environ.get("V41_SL_CACHE", 4096))
    width = int(os.environ.get("V41_SL_WIDTH", C.SLIDING_WINDOW + C.INDEX_TOPK))  # the index rows' width
    patterns = os.environ.get("V41_FS_PATTERNS", "c1,k2,k3,f8,f16,full").split(",")
    sp, tp = tuple(mesh_device.shape)
    heads, head_dim, rope_dim, groups = C.NUM_ATTENTION_HEADS, C.HEAD_DIM, C.QK_ROPE_HEAD_DIM, C.O_GROUPS
    local_heads, sdpa_heads = heads // tp, max(heads // tp, SDPA_MIN_HEADS)
    v_dim = head_dim  # the model's V width is the head dim: the scaled rows have no bf16 RoPE tail
    scale = head_dim**-0.5
    shard = lambda dims: ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=dims)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    torch.manual_seed(143)
    fs = FlashSdpa(mesh_device, heads=sdpa_heads, rows=ROWS, width=width, v_dim=v_dim, real_heads=local_heads)

    def tiled(t, dims, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard(dims) if dims else replicate
        )

    def rm(t, dims, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=shard(dims) if dims else replicate,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    trans = tiled(get_rot_transformation_mat(), None)
    sink = (torch.randn(1, 1, 1, heads) * 2.0 / scale).to(torch.bfloat16)
    local = sink.view(tp, local_heads)
    padded = torch.cat([local, torch.zeros(tp, sdpa_heads - local_heads, dtype=local.dtype)], 1)
    sink_local = rm(padded.reshape(1, 1, 1, tp * sdpa_heads), (None, 3))

    host = {}  # the host copies for the diagnostic: the bf16 cache per mesh row, the index rows, the sinks

    def kv_cache(fmt: MlaKvCacheFormat, seed: int):
        gen = torch.Generator().manual_seed(seed)
        rows = torch.randn(1, 1, sp * cache_rows, head_dim, generator=gen)
        if fmt == MlaKvCacheFormat.BF16_RM:
            host["cache"] = rows.to(torch.bfloat16).float().view(sp, cache_rows, head_dim)
            return rm(rows.to(torch.bfloat16), (2, None))
        latent = ttnn.from_torch(
            rows.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=shard((2, None)),
        )
        fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(latent, round_scale_to_power_of_two=True)
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(fp8, scales)
        for t in (latent, fp8, scales):
            ttnn.deallocate(t)
        return packed

    def index_rows(pattern: str, seed: int):
        """``[1, 1, 32, width]`` uint32 per mesh row and the live rows per mesh row: ``c1`` one live row (a random
        one, every key), ``kK`` the live row and K draft rows (rows 28.. as the verify step's c1 layout: each draft
        row sees the user's keys and the drafts before it, at a random valid count), ``fK`` the first K rows live
        and ``full`` every row live, with random valid counts (a sentinel tail)."""
        gen = torch.Generator().manual_seed(seed)
        rows = torch.full((1, sp, ROWS, width), SENTINEL, dtype=torch.int64)
        live_rows = []
        u = int(torch.randint(ROWS - 4, (1,), generator=gen))  # the same rows on every mesh row (one live page)
        for mesh_row in range(sp):
            if pattern == "c1":
                live = [u]
            elif pattern.startswith("k"):
                live = [u] + list(range(ROWS - 4, ROWS - 4 + int(pattern[1:])))
            elif pattern.startswith("f"):
                live = list(range(int(pattern[1:])))  # the first K rows live, random valid counts
            else:
                live = list(range(ROWS))
            keys = torch.randperm(cache_rows, generator=gen)
            for j, r in enumerate(live):
                if pattern == "c1":
                    n = width
                elif pattern.startswith("k"):
                    n = int(torch.randint(width // 2, width + 1 - 4, (1,), generator=gen)) + j
                    rows[0, mesh_row, r, :n] = keys[:n]
                    continue
                else:
                    n = int(torch.randint(1, width + 1, (1,), generator=gen))
                rows[0, mesh_row, r, :n] = torch.randperm(cache_rows, generator=gen)[:n]
            live_rows.append(live)
        host["rows"] = rows[0]
        return rm(rows.to(torch.int32), (1, None), ttnn.uint32), live_rows, fs.live_page(live_rows[0])

    def diagnose(qh, out_flash, out_sparse, u: int, chip: int = 0):
        """The c1 BF16 case on one chip against a torch reference: the flash output, the sparse output, and the
        partials read back from the scratch (each unit's max, sum tile and acc against the same arithmetic in
        torch, then the merge in torch) -> a dict of PCCs."""
        q = ttnn.to_torch(ttnn.get_device_tensors(qh)[chip]).float()[0, :, u, :]  # [32 heads, 512]
        idx = host["rows"][chip // tp, u]
        nv = int((idx != SENTINEL).sum())
        keys = host["cache"][chip // tp][idx[:nv].long()]  # [nv, 512]
        sink_t = padded[chip % tp].float()  # [32]
        scores = q @ keys.T  # [32, nv]
        full = torch.cat([scores * scale, sink_t[:, None]], 1)
        p = torch.softmax(full, dim=1)[:, :nv]
        ref = p @ keys  # [32, 512]
        fl = ttnn.to_torch(ttnn.get_device_tensors(out_flash)[chip]).float()[0, :, u, :]
        sr = ttnn.to_torch(ttnn.get_device_tensors(out_sparse)[chip]).float()[0, :, u, :]
        d = {
            "torch_vs_flash": _pcc(ref[:local_heads], fl[:local_heads]),
            "torch_vs_sparse": _pcc(ref[:local_heads], sr[:local_heads]),
        }
        d["flash_head_pcc"] = [round(_pcc(ref[h], fl[h]), 4) for h in range(local_heads)]
        # the partials of live rank 0 (slot j at tiles j * slot_tiles ..)
        scratch = ttnn.to_torch(ttnn.get_device_tensors(fs.scratch)[chip]).float()[0, 0]  # [tiles * 32, 32]
        U = -(-nv // fs.kpu)
        pm, ps, pa, pr = [], [], [], []
        M = None
        merged_num = torch.zeros(32, v_dim)
        merged_den = torch.zeros(32)
        ms = []
        for j in range(U):
            base = j * fs.slot_tiles
            tile = lambda t: scratch[(base + t) * 32 : (base + t + 1) * 32]
            s_j = scores[:, j * fs.kpu : (j + 1) * fs.kpu]
            m_ref = s_j.max(dim=1).values
            p_ref = torch.exp((s_j - m_ref[:, None]) * scale)
            a_ref = p_ref @ keys[j * fs.kpu : (j + 1) * fs.kpu]
            pm.append(_pcc(m_ref[:local_heads], tile(0)[:local_heads, 0]))
            n = s_j.shape[1]
            ps.append(_pcc(p_ref[:local_heads], tile(1)[:local_heads, :n]))
            acc = torch.cat([tile(2 + v) for v in range(fs.vt)], 1)
            pa.append(_pcc(a_ref[:local_heads], acc[:local_heads]))
            ms.append(tile(0)[:, 0])
        d["partial_max_pcc_min"] = min(pm)
        d["partial_sum_pcc_min"] = min(ps)
        d["partial_acc_pcc_min"] = min(pa)
        # the merge in torch from the device partials
        ms = torch.stack(ms)  # [U, 32]
        M = ms.max(dim=0).values
        w = torch.exp((ms - M[None]) * scale)  # [U, 32]
        num = torch.zeros(32, v_dim)
        den = torch.exp((sink_t - M) * scale)
        for j in range(U):
            base = j * fs.slot_tiles
            acc = torch.cat([scratch[(base + 2 + v) * 32 : (base + 3 + v) * 32] for v in range(fs.vt)], 1)
            num += w[j][:, None] * acc
            den += w[j] * scratch[(base + 1) * 32 : (base + 2) * 32].sum(dim=1)
        hm = num / den[:, None]
        d["torch_merge_of_partials_vs_flash"] = _pcc(hm[:local_heads], fl[:local_heads])
        d["torch_merge_of_partials_vs_torch"] = _pcc(hm[:local_heads], ref[:local_heads])
        return d

    def queries(seed: int):
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 1, sp * ROWS, heads * head_dim, generator=gen) * 2.0
        angle = torch.rand(sp * ROWS, rope_dim // 2, generator=gen) * 6.2832
        angle = torch.cat([angle, angle], -1)
        cos = tiled(angle.cos().view(1, 1, sp * ROWS, rope_dim).to(torch.bfloat16), (2, None))
        sin = tiled(angle.sin().view(1, 1, sp * ROWS, rope_dim).to(torch.bfloat16), (2, None))
        return tiled(q.to(torch.bfloat16), (2, 3)), cos, sin

    def sparse(qh, kv, rows, fmt, w):
        return ttnn.transformer.sparse_sdpa(
            qh,
            kv,
            rows,
            head_dim,
            kv_format=fmt.sparse_sdpa_format,
            scale=scale,
            k_chunk_size=next(c for c in (128, 64, 32) if w % c == 0),
            attention_sink=sink_local,
        )

    def flash(qh, kv, rows, fmt, w, live=None):
        return fs(qh, kv, rows, fmt == MlaKvCacheFormat.SCALED_FP8, scale, sink_local, live=live)

    result = {"mesh": [sp, tp], "width": width, "cache_rows": cache_rows, "kpu": fs.kpu, "trials": []}
    caches = {fmt.name: kv_cache(fmt, 7) for fmt in (MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8)}
    failures = []
    for trial in range(trials):
        q, cos, sin = queries(100 + trial)
        qh = q_heads(q, cos, sin, trans, local_heads, rope_dim, out_heads=sdpa_heads)
        for fmt in (MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8):
            for pattern in patterns:
                rows, live_rows, live = index_rows(pattern, 200 + trial)
                ref = sparse(qh, caches[fmt.name], rows, fmt, width)
                out = flash(qh, caches[fmt.name], rows, fmt, width, live)
                expect = (1, sdpa_heads, ROWS, v_dim)
                assert tuple(out.shape) == tuple(ref.shape) == expect, (tuple(out.shape), tuple(ref.shape))
                words, max_abs, ref_max, pcc_min, dead_nonzero = 0, 0.0, 0.0, 1.0, 0
                for chip, (a, b) in enumerate(zip(ttnn.get_device_tensors(ref), ttnn.get_device_tensors(out))):
                    a, b = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
                    live_r = live_rows[chip // tp]
                    dead = [r for r in range(ROWS) if r not in live_r]
                    al, bl = a[:, :local_heads, live_r, :], b[:, :local_heads, live_r, :]
                    words += int((al != bl).sum())
                    max_abs = max(max_abs, float((al - bl).abs().max()))
                    ref_max = max(ref_max, float(al.abs().max()))
                    pcc_min = min(pcc_min, _pcc(al, bl))
                    if dead:
                        dead_nonzero += int((b[:, :, dead, :] != 0).sum())
                entry = {
                    "trial": trial,
                    "kv_format": fmt.name,
                    "pattern": pattern,
                    "live_rows": len(live_rows[0]),
                    "words_differ": words,
                    "max_abs": max_abs,
                    "rel_max": max_abs / max(ref_max, 1e-30),
                    "pcc_min": pcc_min,
                    "dead_nonzero": dead_nonzero,
                }
                if pattern == "c1" and fmt == MlaKvCacheFormat.BF16_RM and trial == 0:
                    entry["diag"] = diagnose(qh, out, ref, live_rows[0][0])
                result["trials"].append(entry)
                logger.info(f"flash sdpa trial {json.dumps(entry)}")
                if pcc_min < PCC_FLOOR or dead_nonzero:
                    failures.append(entry)
                for t in (rows, ref, out, live):
                    ttnn.deallocate(t)
        for t in (q, cos, sin, qh):
            ttnn.deallocate(t)
    # the traced per-layer time of each SDPA alone per pattern and format, and the local chain for c1
    q, cos, sin = queries(999)
    qh = q_heads(q, cos, sin, trans, local_heads, rope_dim, out_heads=sdpa_heads)
    timing = {}
    for fmt in (MlaKvCacheFormat.SCALED_FP8, MlaKvCacheFormat.BF16_RM):
        for pattern in patterns:
            rows, _, live = index_rows(pattern, 999)
            for name, fn in (("sparse", sparse), ("flash", flash)):
                key = f"{name}_{fmt.name}_{pattern}"
                timing[key] = _trace_us(
                    mesh_device, lambda: fn(qh, caches[fmt.name], rows, fmt, width, live), calls, replays
                )
                logger.info(f"flash sdpa traced {key}: {timing[key]} us a layer")
            if pattern == "c1" and fmt == MlaKvCacheFormat.SCALED_FP8:
                for name, fn in (("sparse", sparse), ("flash", flash)):
                    chain = lambda: o_heads(
                        fn(
                            q_heads(q, cos, sin, trans, local_heads, rope_dim, out_heads=sdpa_heads),
                            caches[fmt.name],
                            rows,
                            fmt,
                            width,
                            live,
                        ),
                        cos,
                        ttnn.neg(sin),
                        trans,
                        groups // tp,
                        rope_dim,
                        heads=local_heads,
                    )
                    timing[f"chain_{name}_c1"] = _trace_us(mesh_device, chain, calls, replays)
                    logger.info(f"flash sdpa traced chain {name} c1: {timing[f'chain_{name}_c1']} us a layer")
            for t in (rows, live):
                ttnn.deallocate(t)
    # the window layers' geometry: bf16 rows, 128 keys in every row (the pool's window rows are constants: every
    # row of the tile reads its user's ring), one live row
    if width > 128:
        rows_w = torch.zeros((1, sp, ROWS, 128), dtype=torch.int64)
        for mesh_row in range(sp):
            for r in range(ROWS):
                rows_w[0, mesh_row, r, :] = torch.randperm(cache_rows)[:128]
        rows = rm(rows_w.to(torch.int32), (1, None), ttnn.uint32)
        live = fs.live_page([5])
        fmt = MlaKvCacheFormat.BF16_RM
        for name, fn in (("sparse", sparse), ("flash", flash)):
            key = f"{name}_{fmt.name}_c1_w128"
            timing[key] = _trace_us(mesh_device, lambda: fn(qh, caches[fmt.name], rows, fmt, 128, live), calls, replays)
            logger.info(f"flash sdpa traced {key}: {timing[key]} us a layer")
        for t in (rows, live):
            ttnn.deallocate(t)
    result["layer_us"] = timing
    logger.info(f"V41_FS_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_SL_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
    assert not failures, failures
