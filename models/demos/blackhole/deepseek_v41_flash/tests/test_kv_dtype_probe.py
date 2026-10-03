"""KV cache dtype probe (bf16 / bfp8 / bfp4): (1) paged_update_cache single-row writes into a block-float cache (value error on the written row,
drift of untouched rows); (2) SDPA decode over [T,1,L,512] caches at growing L: PCC vs bf16 and traced time. Prints KVD lines."""
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, PAD_HEADS

T = 4
DTS = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}
MESH = pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)


def up(md, t, dtype, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def h0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


def bench(md, fn, reps=10):
    out = fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ts = []
    for _ in range(4):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return out, min(ts)


@MESH
@PARAMS
@pytest.mark.parametrize("name", ["bf16", "bfp8", "bfp4"])
@torch.no_grad()
def test_update_rows(mesh_device, name):
    md, L = mesh_device, 256
    g = torch.Generator().manual_seed(1)
    init = torch.randn(T, 1, L, HEAD_DIM, generator=g)
    cache = up(md, init, DTS[name])
    base = h0(cache)  # quantised initial content
    ucfg = ttnn.create_sharded_memory_config(
        shape=(PAD_HEADS, HEAD_DIM),
        core_grid=ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pos = torch.tensor([5, 77, 130, 255], dtype=torch.int32)
    row = torch.randn(1, T, PAD_HEADS, HEAD_DIM, generator=g)
    tt_row = up(md, row.to(torch.bfloat16), ttnn.bfloat16, mc=ucfg)
    idx = up(md, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.experimental.paged_update_cache(cache, tt_row, update_idxs_tensor=idx, page_table=None)
    got = h0(cache)
    errs, drift = [], 0.0
    for u in range(T):
        p = int(pos[u])
        want = row[0, u, 0]
        errs.append(pcc(got[u, 0, p], want))
        mask = torch.ones(L, dtype=torch.bool)
        mask[p] = False
        d = (got[u, 0][mask] - base[u, 0][mask]).abs().max().item()
        drift = max(drift, d)
    print(
        f"KVD update {name}: written-row PCC vs fp32 {[round(e, 5) for e in errs]}  max drift of untouched rows vs pre-write {drift:.4e}  (base absmax {base.abs().max():.2f})",
        flush=True,
    )


@MESH
@PARAMS
@pytest.mark.parametrize("L", [256, 4224, 16512, 65664])
@torch.no_grad()
def test_sdpa_dtypes(mesh_device, L):
    md = mesh_device
    g = torch.Generator().manual_seed(2)
    # latent-like data: per-channel scale spread (few outlier channels) like real normed KV
    ch = torch.exp(torch.randn(HEAD_DIM, generator=g) * 0.5)
    kv = torch.randn(T, 1, L, HEAD_DIM, generator=g) * ch
    q = (torch.randn(1, T, 32, HEAD_DIM, generator=g) * 0.7 * ch).to(torch.bfloat16)
    mask = torch.zeros(T, 1, 32, L)
    sinks = up(md, torch.zeros(32, 32), ttnn.bfloat16)
    kc = next(c for c in (256, 128, 64, 32) if L % c == 0)
    n = T
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(min(n, 8), (n + 7) // 8),
        q_chunk_size=0,
        k_chunk_size=kc,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tq, tm = up(md, q, ttnn.bfloat16), up(md, mask.to(torch.bfloat16), ttnn.bfloat16)
    ref = None
    for name in ("bf16", "bfp8", "bfp4"):
        cache = up(md, kv, DTS[name])
        f = lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            tq,
            cache,
            cache,
            is_causal=False,
            attn_mask=tm,
            attention_sink=sinks,
            scale=HEAD_DIM**-0.5,
            program_config=cfg,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            out, us = bench(md, f)
        except Exception as e:  # noqa: BLE001
            print(f"KVD sdpa L={L} {name}: FAILED {str(e)[:300]}", flush=True)
            continue
        o = h0(out)
        if ref is None:
            ref = o
        print(f"KVD sdpa L={L} {name}: {us:8.1f} us  PCC vs bf16-cache {pcc(o, ref):.5f}", flush=True)
        ttnn.deallocate(cache)
