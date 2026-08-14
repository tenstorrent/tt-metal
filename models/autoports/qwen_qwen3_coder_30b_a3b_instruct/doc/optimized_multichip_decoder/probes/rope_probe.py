# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04 micro-probe: ``rotary_embedding_llama`` against the shipped HF rotary.

``README.md`` limitation 4 names 28.478 us of single-core attention ops (rows
158-167 and 169 of the stage-04 decode profile) and names
``rotary_embedding_llama``'s sharded decode factory as the next step -- named,
not measured. Two of those rows are the rotary embeddings themselves:

    row 163  RotaryEmbedding  4.699 us  on **1 core**
    row 164  RotaryEmbedding  4.659 us  on **1 core**

so the lever is bounded at 9.358 us, 2.6% of the 362.828 us layer. This prices
it instead of arguing about it.

The two ops are *not* interchangeable at fixed weights: the llama variant wants
Meta channel ordering (head channels interleaved, pairs ``(2i, 2i+1)``) where
HF pairs ``(i, i+64)``. That is a checkpoint-side permutation of the q/k rows of
``wqkv`` **and** of Qwen3's per-head QK-norm vectors, which is why
``weight_mapping.py`` treats the two conventions as a pair. This probe applies
the permutation on the host so the comparison is like-for-like, and reports the
PCC of the llama result -- permuted back to HF order -- against the shipped op's.

Shapes are the shipped per-die decode shapes: batch 1, 8 local Q heads and 1
local KV head, both padded to 32 rows by ``nlp_create_qkv_heads_decode``, head
dim 128.

Timing is by trace slope (median-of-30 blocking replay of a 33-op trace minus a
1-op trace, over 32), the same harness ``norm_router_probe.py`` uses, so the
host-dispatch floor is removed.

    python rope_probe.py

Prints ``P|`` lines only.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import load_config, rotary_embeddings

REPS = 32
HEAD_DIM = 128
ROWS = 32  # nlp_create_qkv_heads_decode pads the head count to a tile
BATCH = 1
POS = 127

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=60_000_000, l1_small_size=32768)


def slope(fn):
    """us per op, trace slope over REPS."""
    out = []

    def build(n):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            out.append(fn())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        s = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            s.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(mesh, tid)
        return statistics.median(s)

    fn()
    ttnn.synchronize_device(mesh)
    long = build(REPS + 1)
    short = build(1)
    return (long - short) / REPS


def rep(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


# HF channel i pairs with i + 64; Meta channel 2i pairs with 2i + 1. The same
# permutation maps an HF-ordered activation *and* an HF-ordered cos/sin row into
# Meta order, because HF's cos row is [c0..c63, c0..c63] and Meta's is
# [c0, c0, c1, c1, ...].
HF_TO_META = torch.stack([torch.arange(HEAD_DIM // 2), torch.arange(HEAD_DIM // 2) + HEAD_DIM // 2], dim=1).reshape(-1)
META_TO_HF = torch.argsort(HF_TO_META)


def height_sharded(rows, cols, cores=BATCH):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))}),
            [rows, cols],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def trans_mat_torch(d=32):
    m = torch.zeros(1, 1, d, d)
    m[..., torch.arange(0, d, 2), torch.arange(1, d, 2)] = 1
    m[..., torch.arange(1, d, 2), torch.arange(0, d, 2)] = -1
    return m


results = {}
try:
    torch.manual_seed(0)
    hf = load_config()
    cos_t, sin_t = rotary_embeddings(hf, 1024)  # [1, S, head_dim], HF order
    cos_t, sin_t = cos_t.unsqueeze(0).float(), sin_t.unsqueeze(0).float()

    x_t = torch.randn(1, BATCH, ROWS, HEAD_DIM) * 0.5

    # --- leg A: the shipped op, HF ordering, DRAM interleaved ------------------
    cos_hf, sin_hf = rep(cos_t), rep(sin_t)
    x_hf = rep(x_t)
    fn_hf = lambda: ttnn.experimental.rotary_embedding(x_hf, cos_hf, sin_hf, POS)
    results["hf"] = slope(fn_hf)
    out_hf = ttnn.to_torch(fn_hf(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()
    print(f"P|rotary_embedding (shipped, HF order, DRAM)   {results['hf']:6.2f} us", flush=True)

    # --- leg B: rotary_embedding_llama, decode mode, Meta ordering -------------
    # Input, cos and sin height-sharded over ``batch`` cores -- which is what the
    # decode factory shards on. At batch 1 that is one core.
    smc = height_sharded(ROWS, HEAD_DIM)
    x_meta = rep(x_t[..., HF_TO_META], mc=smc)
    cos_row = cos_t[:, :, POS : POS + 1, :].expand(1, 1, ROWS, HEAD_DIM).contiguous()
    sin_row = sin_t[:, :, POS : POS + 1, :].expand(1, 1, ROWS, HEAD_DIM).contiguous()
    cos_m = rep(cos_row[..., HF_TO_META], mc=smc)
    sin_m = rep(sin_row[..., HF_TO_META], mc=smc)
    tm = rep(trans_mat_torch(32).repeat(1, 1, BATCH, 1), mc=height_sharded(32, 32))

    def fn_llama():
        return ttnn.experimental.rotary_embedding_llama(x_meta, cos_m, sin_m, tm, is_decode_mode=True)

    try:
        results["llama"] = slope(fn_llama)
        got = ttnn.to_torch(fn_llama(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()
        back = got[..., META_TO_HF]
        d = (back - out_hf).abs().max().item()
        a, b = back.flatten(), out_hf.flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(
            f"P|rotary_embedding_llama (decode, Meta, L1)   {results['llama']:6.2f} us"
            f"   max|diff| {d:.3e}  PCC {pcc:.7f}",
            flush=True,
        )
    except Exception as exc:
        print(f"P|rotary_embedding_llama FAILED {str(exc)[:300]}", flush=True)

    # --- what the shipped path would additionally pay --------------------------
    # The shipped decode path hands rope a DRAM-interleaved tensor (rms_norm
    # wants interleaved) and hands the result to paged_update_cache, which wants
    # it sharded. The llama op needs the input sharded, so price the conversion.
    i2s = slope(lambda: ttnn.to_memory_config(x_hf, smc))
    print(f"P|  interleaved->height-sharded [32,128]       {i2s:6.2f} us", flush=True)

    # Head counts on this die, for the record: the decode factory shards on
    # batch, not on heads.
    print(f"P|  cores: shipped op 1 (profile rows 163/164);  llama decode factory = batch = {BATCH}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
print("P|done")
