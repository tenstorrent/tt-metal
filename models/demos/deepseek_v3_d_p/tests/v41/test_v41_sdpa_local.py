# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode step's SDPA on the chip's own heads (``V41_DECODE_SDPA=local``,
``tt/v41/batch_decode.py::_attention``) against the served path (the queries all-gathered over TP,
``sparse_sdpa`` over all 64 heads on
every chip, the chip's groups of the output kept with ``mesh_partition``).

No weights, the decode tile: ``V41_SL_TRIALS`` random bf16 ``[1, 1, 32, 16 * 512]`` query tiles a chip (every
chip its
own), random RoPE tables, a random KV cache in both served formats (BF16 and the pool's scaled FP8), index
rows in the
c1 pattern (one live row a chip, the others sentinels) and the 32-row pattern (every row live, varying
counts), the
local path's ``[1, 2, 32, 4096]`` output compared bit for bit with the gather path's on every chip, then each
path
under trace (``V41_SL_REPLAYS`` replays of ``V41_SL_CALLS`` calls) for the per-layer device time. Results:
``V41_SL_RESULT``, ``V41_SL_OUT``."""

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
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

SENTINEL = 0xFFFFFFFF
ROWS = 32


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


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_sdpa_local(mesh_device, device_params):
    trials = int(os.environ.get("V41_SL_TRIALS", 3))
    replays = int(os.environ.get("V41_SL_REPLAYS", 20))
    calls = int(os.environ.get("V41_SL_CALLS", 16))
    cache_rows = int(os.environ.get("V41_SL_CACHE", 4096))
    width = int(os.environ.get("V41_SL_WIDTH", C.SLIDING_WINDOW + C.INDEX_TOPK))  # the index rows' width
    sp, tp = tuple(mesh_device.shape)
    heads, head_dim, rope_dim, groups = C.NUM_ATTENTION_HEADS, C.HEAD_DIM, C.QK_ROPE_HEAD_DIM, C.O_GROUPS
    local_heads, sdpa_heads = heads // tp, max(heads // tp, SDPA_MIN_HEADS)
    scale = head_dim**-0.5
    ccl = V41Collectives(mesh_device)
    shard = lambda dims: ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=dims)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    torch.manual_seed(118)

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
    sink_full = rm(sink, None)
    local = sink.view(tp, local_heads)
    padded = torch.cat([local, torch.zeros(tp, sdpa_heads - local_heads, dtype=local.dtype)], 1)
    sink_local = rm(padded.reshape(1, 1, 1, tp * sdpa_heads), (None, 3))

    def kv_cache(fmt: MlaKvCacheFormat, seed: int):
        """A random cache of ``cache_rows`` rows in ``fmt`` per mesh row (replicated over TP, as the pool's)."""
        gen = torch.Generator().manual_seed(seed)
        rows = torch.randn(1, 1, sp * cache_rows, head_dim, generator=gen)  # a cache per mesh row, as the pool
        if fmt == MlaKvCacheFormat.BF16_RM:
            return rm(rows.to(torch.bfloat16), (2, None))
        # the pool's packed scaled-FP8 rows (``V41BatchDecoder._encode``)
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
        """``[1, 1, 32, width]`` uint32 per mesh row (the TP chips share it, as the pool's): ``c1`` one live
        row (a
        random one), the rest sentinels; ``full`` every row live with a random valid count (a sentinel
        tail)."""
        gen = torch.Generator().manual_seed(seed)
        rows = torch.full((1, sp, ROWS, width), SENTINEL, dtype=torch.int64)
        for mesh_row in range(sp):
            live = [int(torch.randint(ROWS, (1,), generator=gen))] if pattern == "c1" else list(range(ROWS))
            for r in live:
                n = width if pattern == "c1" else int(torch.randint(1, width + 1, (1,), generator=gen))
                rows[0, mesh_row, r, :n] = torch.randperm(cache_rows, generator=gen)[:n]
        return rm(rows.to(torch.int32), (1, None), ttnn.uint32)

    def queries(seed: int):
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 1, sp * ROWS, heads * head_dim, generator=gen) * 2.0
        angle = torch.rand(sp * ROWS, rope_dim // 2, generator=gen) * 6.2832
        angle = torch.cat([angle, angle], -1)
        cos = tiled(angle.cos().view(1, 1, sp * ROWS, rope_dim).to(torch.bfloat16), (2, None))
        sin = tiled(angle.sin().view(1, 1, sp * ROWS, rope_dim).to(torch.bfloat16), (2, None))
        return tiled(q.to(torch.bfloat16), (2, 3)), cos, sin

    def gather_path(q, cos, sin, kv, rows, fmt):
        qa = ccl.tp_all_gather(q, dim=3)
        out = ttnn.transformer.sparse_sdpa(
            q_heads(qa, cos, sin, trans, heads, rope_dim),
            kv,
            rows,
            head_dim,
            kv_format=fmt.sparse_sdpa_format,
            scale=scale,
            k_chunk_size=next(c for c in (128, 64, 32) if width % c == 0),
            attention_sink=sink_full,
        )
        out = o_heads(out, cos, ttnn.neg(sin), trans, groups, rope_dim)
        return ttnn.mesh_partition(out, dim=1, cluster_axis=TP_AXIS)

    def local_path(q, cos, sin, kv, rows, fmt):
        out = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, trans, local_heads, rope_dim, out_heads=sdpa_heads),
            kv,
            rows,
            head_dim,
            kv_format=fmt.sparse_sdpa_format,
            scale=scale,
            k_chunk_size=next(c for c in (128, 64, 32) if width % c == 0),
            attention_sink=sink_local,
        )
        return o_heads(out, cos, ttnn.neg(sin), trans, groups // tp, rope_dim, heads=local_heads)

    result = {"mesh": [sp, tp], "width": width, "cache_rows": cache_rows, "trials": []}
    caches = {fmt.name: kv_cache(fmt, 7) for fmt in (MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8)}
    for trial in range(trials):
        q, cos, sin = queries(100 + trial)
        for fmt in (MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8):
            for pattern in ("c1", "full"):
                rows = index_rows(pattern, 200 + trial)
                ref = gather_path(q, cos, sin, caches[fmt.name], rows, fmt)
                out = local_path(q, cos, sin, caches[fmt.name], rows, fmt)
                expect = (1, groups // tp, ROWS, heads // groups * head_dim)
                assert tuple(out.shape) == tuple(ref.shape) == expect, (tuple(out.shape), tuple(ref.shape))
                diffs, rel = [], 0.0
                for a, b in zip(ttnn.get_device_tensors(ref), ttnn.get_device_tensors(out)):
                    a, b = ttnn.to_torch(a), ttnn.to_torch(b)
                    diffs.append(int((a != b).sum()))
                    rel = max(rel, float((a.double() - b.double()).abs().max() / max(a.double().abs().max(), 1e-30)))
                entry = {
                    "trial": trial,
                    "kv_format": fmt.name,
                    "pattern": pattern,
                    "words_differ": sum(diffs),
                    "chips_differ": sum(1 for d in diffs if d),
                    "rel_max": rel,
                }
                result["trials"].append(entry)
                logger.info(f"sdpa local trial {json.dumps(entry)}")
                assert sum(diffs) == 0, entry
                for t in (rows, ref, out):
                    ttnn.deallocate(t)
        for t in (q, cos, sin):
            ttnn.deallocate(t)
    # the traced per-layer time of each path at the served format, both row patterns
    q, cos, sin = queries(999)
    timing = {}
    fmt = MlaKvCacheFormat.SCALED_FP8
    for pattern in ("c1", "full"):
        rows = index_rows(pattern, 999)
        for name, fn in (("gather", gather_path), ("local", local_path)):
            timing[f"{name}_{pattern}"] = _trace_us(
                mesh_device, lambda: fn(q, cos, sin, caches[fmt.name], rows, fmt), calls, replays
            )
            logger.info(f"sdpa local traced {name} {pattern}: {timing[f'{name}_{pattern}']} us a layer")
        if pattern == "c1" and os.environ.get("V41_SL_SPLIT", "0") == "1":
            # the local chain's pieces alone: the SDPA kernel on the chip's 16 + 16 zero
            # heads over the live row's rows, the two head ops, each under its own trace
            qh = q_heads(q, cos, sin, trans, local_heads, rope_dim, out_heads=sdpa_heads)
            sdpa = lambda: ttnn.transformer.sparse_sdpa(
                qh,
                caches[fmt.name],
                rows,
                head_dim,
                kv_format=fmt.sparse_sdpa_format,
                scale=scale,
                k_chunk_size=next(c for c in (128, 64, 32) if width % c == 0),
                attention_sink=sink_local,
            )
            timing["local_c1_sdpa"] = _trace_us(mesh_device, sdpa, calls, replays)
            timing["local_c1_q_heads"] = _trace_us(
                mesh_device,
                lambda: q_heads(q, cos, sin, trans, local_heads, rope_dim, out_heads=sdpa_heads),
                calls,
                replays,
            )
            o = sdpa()
            timing["local_c1_o_heads"] = _trace_us(
                mesh_device,
                lambda: o_heads(o, cos, ttnn.neg(sin), trans, groups // tp, rope_dim, heads=local_heads),
                calls,
                replays,
            )
            timing["local_c1_neg_sin"] = _trace_us(mesh_device, lambda: ttnn.neg(sin), calls, replays)
            for k in ("local_c1_sdpa", "local_c1_q_heads", "local_c1_o_heads", "local_c1_neg_sin"):
                logger.info(f"sdpa local traced {k}: {timing[k]} us a layer")
            for t in (qh, o):
                ttnn.deallocate(t)
        ttnn.deallocate(rows)
    result["layer_us"] = timing
    logger.info(f"V41_SL_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_SL_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
