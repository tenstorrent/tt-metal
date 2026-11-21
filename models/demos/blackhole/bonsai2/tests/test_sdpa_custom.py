# The custom decode SDPA (tt/sdpa_dec) vs a torch reference at Bonsai's full-attention shape (B=1, 24 q heads, 4 kv heads,
# head_dim 256, bf8 paged KV), and traced device time vs the stock paged SDPA decode.
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.sdpa_dec.sdpa_dec import DecodeSDPA

BLOCK, NKV, NH, HD = 64, 4, 24, 256


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("pos", [4096, 4131, 100, 31])
def test_sdpa_custom(device, pos):
    torch.manual_seed(0)
    nblk = 4608 // BLOCK
    perm = torch.randperm(nblk)
    kt = torch.randn(nblk, NKV, BLOCK, HD) * 2
    vt = torch.randn(nblk, NKV, BLOCK, HD)
    k = ttnn.from_torch(kt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(vt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    kq, vq = ttnn.to_torch(k).float(), ttnn.to_torch(v).float()
    pt = ttnn.from_torch(perm.reshape(1, -1).to(torch.int32), dtype=ttnn.int32, device=device)
    cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=device)
    qt = torch.randn(1, 1, NH, HD)
    q = ttnn.from_torch(
        qt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    g = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(112)]
    op = DecodeSDPA(device, cores)
    o = ttnn.to_torch(op(q, k, v, cur, pt)).float().reshape(NH, HD)
    # reference: positions 0 .. pos through the page table
    n = pos + 1
    K = torch.cat([kq[perm[b]] for b in range(-(-n // BLOCK))], 1)[:, :n]  # [NKV, n, HD]
    V = torch.cat([vq[perm[b]] for b in range(-(-n // BLOCK))], 1)[:, :n]
    qb = qt.bfloat16().float().reshape(NH, HD)
    ref = torch.empty(NH, HD)
    for h in range(NH):
        s = (qb[h] @ K[h // 6].T) * HD**-0.5
        ref[h] = torch.softmax(s, -1) @ V[h // 6]
    pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
    err = ((o - ref).norm() / ref.norm()).item()
    print(f"SDPA_CUSTOM pos={pos} pcc={pcc:.6f} rel={err:.5f}", flush=True)
    assert pcc > 0.999, pcc
    if pos != 4096:
        return
    out = op(q, k, v, cur, pt)
    qd = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
    stock_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g.x, g.y), q_chunk_size=32, k_chunk_size=64)
    fns = {
        "custom": lambda: op(q, k, v, cur, pt, out=out),
        "stock": lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qd, k, v, page_table_tensor=pt, cur_pos_tensor=cur, scale=HD**-0.5, program_config=stock_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ),  # fmt: skip
    }
    for name, fn in fns.items():
        fn()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(20):
            fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        print(f"SDPA_TIME {name} {(time.perf_counter() - t0) / 20 * 1e6:.1f} us", flush=True)
        ttnn.release_trace(device, tid)
