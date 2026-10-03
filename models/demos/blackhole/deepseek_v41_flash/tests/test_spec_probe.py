# SPDX-License-Identifier: Apache-2.0
"""M1 probe: paged cache shared by the (1+k) virtual-user rows of one user (same page ids, different positions):
paged_update_cache with several positions of one page in ONE call (read-modify-write race?) + paged SDPA decode at d=512 with sink,
sliding window, per-row cur_pos (causality inside the block). Single device."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc

H, D, WIN, SCALE = 8, 512, 128, 512**-0.5
NH = 32


def _ref(q, cache_rows, sink, pos):
    # q [H,D], cache_rows [S,D] (K==V), attends [max(0,pos+1-WIN), pos]
    lo = max(0, pos + 1 - WIN)
    kk = cache_rows[lo : pos + 1].float()
    s = (q.float() @ kk.T) * SCALE
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    return (p @ kk) / (p.sum(-1, keepdim=True) + torch.exp(sink.view(-1, 1) / 1.0 - m))


@pytest.mark.parametrize("n", [2, 4, 6])
@pytest.mark.parametrize("base", [130, 62, 255 - 5])
@pytest.mark.parametrize("page", [256, 64])
def test_paged_block(device, n, base, page):
    torch.manual_seed(1)
    U, S = 4, 256
    T = U * n  # token rows, user-major (u*n + j)
    ppu = S // page
    # shuffled pages so addressing is tested
    perm = torch.randperm(U * ppu)
    pt_user = perm.view(U, ppu).to(torch.int32)
    pt_tok = pt_user.repeat_interleave(n, dim=0)  # [T, ppu]
    hist = torch.randn(U, S, D) * 0.5
    cache = torch.zeros(U * ppu, 1, page, D)
    for u in range(U):
        for j in range(ppu):
            cache[perm[u * ppu + j], 0] = hist[u, j * page : (j + 1) * page]
    pos_user = [base + 3 * u % 2 for u in range(U)]
    pos_tok = torch.tensor([pos_user[u] + j for u in range(U) for j in range(n)], dtype=torch.int32)
    newkv = torch.randn(U, n, D) * 0.5
    q = torch.randn(U, n, H, D) * 0.5
    sink = torch.randn(H) * 0.5

    ck = ttnn.from_torch(cache, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_pt = ttnn.from_torch(pt_tok, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_pos = ttnn.from_torch(pos_tok, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ucfg = ttnn.create_sharded_memory_config(
        shape=(NH, D),
        core_grid=ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    kvt = torch.zeros(1, T, NH, D)
    kvt[0, :, 0] = newkv.reshape(T, D)
    tt_kv = ttnn.from_torch(kvt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ucfg)
    mode = os.environ.get("PROBE_UPDATE", "perj")
    if mode == "one":
        ttnn.experimental.paged_update_cache(ck, tt_kv, update_idxs_tensor=tt_pos, page_table=tt_pt)
    else:  # one call per block index j, rows of other block indices skipped with idx -1
        for j in range(n):
            idx_j = torch.tensor(
                [pos_user[u] + j if jj == j else -1 for u in range(U) for jj in range(n)], dtype=torch.int32
            )
            ttnn.experimental.paged_update_cache(
                ck,
                tt_kv,
                update_idxs_tensor=ttnn.from_torch(
                    idx_j, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                page_table=tt_pt,
            )
    # readback
    got = ttnn.to_torch(ck).float()
    full = torch.zeros(U, S, D)
    for u in range(U):
        for j in range(ppu):
            full[u, j * page : (j + 1) * page] = got[perm[u * ppu + j], 0]
    exp = hist.clone()
    for u in range(U):
        for j in range(n):
            exp[u, pos_user[u] + j] = newkv[u, j]
    bad = [
        (u, j)
        for u in range(U)
        for j in range(n)
        if pcc(full[u, pos_user[u] + j], exp[u, pos_user[u] + j].bfloat16().float()) < 0.999
    ]
    others = (full - exp).abs().max(-1).values  # per position
    untouched = torch.ones(U, S, dtype=torch.bool)
    for u in range(U):
        untouched[u, pos_user[u] : pos_user[u] + n] = False
    clobbered = int(((others > 0.02) & untouched).sum())
    print(f"n={n} base={base} page={page}: bad written rows {bad} clobbered other rows {clobbered}")
    # SDPA
    qt = torch.zeros(1, T, NH, D)
    qt[0, :, :H] = q.reshape(T, H, D)
    tt_q = ttnn.from_torch(qt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sinks = torch.zeros(NH, 32)
    sinks[:H, 0] = sink / SCALE
    tt_sink = ttnn.from_torch(sinks, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    grid = ttnn.CoreCoord(min(T, 8), (T + 7) // 8)
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q,
        ck,
        ck,
        page_table_tensor=tt_pt,
        cur_pos_tensor=tt_pos,
        sliding_window_size=WIN,
        attention_sink=tt_sink,
        scale=SCALE,
        program_config=prog,
        compute_kernel_config=ckc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o = ttnn.to_torch(out).float().reshape(T, -1, D)[:, :H]
    ps = []
    for u in range(U):
        for j in range(n):
            p = pos_user[u] + j
            r = _ref(q[u, j], exp[u].bfloat16().float(), sink, p)
            ps.append(pcc(o[u * n + j], r))
    print(f"  SDPA PCC min {min(ps):.5f} mean {sum(ps)/len(ps):.5f}")
    assert not bad and clobbered == 0, "paged_update_cache race / corruption"
    assert min(ps) > 0.99
