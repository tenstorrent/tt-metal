# Decode SDPA alone at Bonsai's full-attention shape (B=1, 24 q heads, 4 kv heads, head_dim 256, paged KV at pos 4096):
# device time per config (traced wall per call).
import time

import pytest
import torch

import ttnn

BLOCK, POS, NKV, NH, HD = 64, 4096, 4, 24, 256


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("kv_dtype", ["bf16", "bf8"])
@pytest.mark.parametrize("cfg", ["k0", "k128", "k256", "k512", "mc32", "approx"])
def test_sdpa_dec(device, kv_dtype, cfg):
    nblk = (POS + 256) // BLOCK
    dt = ttnn.bfloat16 if kv_dtype == "bf16" else ttnn.bfloat8_b
    k = ttnn.from_torch(torch.randn(nblk, NKV, BLOCK, HD), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(torch.randn(nblk, NKV, BLOCK, HD), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    pt = ttnn.from_torch(torch.arange(nblk, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32, device=device)
    cur = ttnn.from_torch(torch.tensor([POS], dtype=torch.int32), dtype=ttnn.int32, device=device)
    q = ttnn.from_torch(torch.randn(1, 1, NH, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = device.compute_with_storage_grid_size()
    kw = dict(compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=cfg == "approx", q_chunk_size=0)
    kw["k_chunk_size"] = int(cfg[1:]) if cfg.startswith("k") else 0
    if cfg == "mc32":
        kw["max_cores_per_head_batch"] = 32
    pc = ttnn.SDPAProgramConfig(**kw)
    fn = lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k,
        v,
        page_table_tensor=pt,
        cur_pos_tensor=cur,
        scale=HD**-0.5,
        program_config=pc,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    o = fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(50):
        o = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / 50 * 1e6
    ttnn.release_trace(device, tid)
    print(f"SDPA kv={kv_dtype} cfg={cfg} us={us:.1f}")
