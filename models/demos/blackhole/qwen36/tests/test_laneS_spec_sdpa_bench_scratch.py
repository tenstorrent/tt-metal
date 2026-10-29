# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lane S scratch: device cost of the TP=2 speculative-verify SDPA, per layer, on ONE chip.

At TP=2 each device holds NH=12 Q heads and NKV=2 KV heads of Qwen3.8-27B (head_dim 256, bf8 paged
KV, block 64). The verify runs n_users x T candidate rows. Two forms:

* ``legacy``: one paged SDPA-decode call over B = n_users*T rows (today's TP=2 path; each row
  re-reads its user's KV; default program config = the model's ``sdpa_dec_cfg``).
* ``fold``: the same rows folded with ``spec_multi_pos_tiles=Tg`` into G groups (needs the NKV>1
  extension; ``spec_q_heads`` = 12 real heads per candidate tile).

Every device does identical SDPA work at TP=2 (its own 2 KV heads), so one chip's per-call device
time x 16 full-attention layers = the per-verify-step SDPA device time. Run under tracy
(``python -m tracy -r -v -m pytest``) and aggregate the SdpaDecodeDeviceOperation rows per signpost.

Env: LANES_MODES (legacy,fold) -- REQUIRED, the module is skipped without it so a plain pytest sweep of
qwen36/tests never runs this device bench; LANES_ITERS (default 5), LANES_CASES (comma list of ids, default
all), LANES_BLOCKS (page-table width per user), LANES_TORCH_CHECK=1 (per-row fp32 check of every mode).
``hold`` = rows of users that are not active in the served 4x8 bucket (cur_pos = -1).
``u1_T8_p32768`` / ``u4_T8_p32768`` verify positions >= 32768 (the page-table-width repro: LANES_BLOCKS=513 vs 1024).
"""

import os

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except Exception:  # pragma: no cover

    def signpost(*a, **k):
        pass


pytestmark = pytest.mark.skipif(
    not os.environ.get("LANES_MODES"), reason="lane S device bench: set LANES_MODES=legacy,fold to run"
)

NH, NKV, HD, BLOCK = 12, 2, 256, 64
TILE = 32
SCALE = HD**-0.5

# (id, n_users, T, isl, active_users)
CASES = [
    ("u1_T8_2k", 1, 8, 2048, 1),
    ("u1_T8_32k", 1, 8, 32768, 1),
    ("u4_T8_2k", 4, 8, 2048, 4),
    ("u4_T8_32k", 4, 8, 32768, 4),
    ("b4_active1_2k", 4, 8, 2048, 1),  # served 4x8 bucket, 1 active user
    ("b4_active1_32k", 4, 8, 32768, 1),
    ("u1_T16_32k", 1, 16, 32768, 1),
    ("u2_T8_32k", 2, 8, 32768, 2),
    ("u1_T8_128", 1, 8, 128, 1),
    ("b4_active1_128", 4, 8, 128, 1),
    ("u4_T8_128", 4, 8, 128, 4),
    ("u1_T8_8k", 1, 8, 8192, 1),
    ("b4_active1_8k", 4, 8, 8192, 1),
    ("u4_T8_8k", 4, 8, 8192, 4),
    ("u1_T16_2k", 1, 16, 2048, 1),
    ("u1_T16_128", 1, 16, 128, 1),
    ("u1_T8_p32768", 1, 8, 32776, 1),  # positions 32768..32775
    ("u4_T8_p32768", 4, 8, 32776, 4),  # user u: 32768+3u..32775+3u
]

# fold plan per T: (groups per user, max_cores_per_head_batch cap) -- env override LANES_FOLD_PLAN="8:2:55,16:4:36"
FOLD_PLAN = {8: (2, 55), 16: (4, 27), 4: (1, 64), 12: (3, 36)}
if os.environ.get("LANES_FOLD_PLAN"):
    for tok in os.environ["LANES_FOLD_PLAN"].split(","):
        t, g, c = (int(x) for x in tok.split(":"))
        FOLD_PLAN[t] = (g, c)


def _selected():
    want = os.environ.get("LANES_CASES")
    if not want:
        return CASES
    w = set(want.split(","))
    return [c for c in CASES if c[0] in w]


def _modes():
    return os.environ.get("LANES_MODES", "legacy").split(",")


def _build(device, n_users, T, isl, active, seed=0):
    g = torch.Generator().manual_seed(seed)
    # page-table width like the served model (a power of 2, room for the prompt plus generation)
    blocks_per_user = int(os.environ.get("LANES_BLOCKS", str(2 * isl // BLOCK)))
    nb = n_users * blocks_per_user
    perm = torch.randperm(nb, generator=g).to(torch.int32)
    user_rows = perm.reshape(n_users, blocks_per_user)
    kc = torch.randn(nb, NKV, BLOCK, HD, generator=g).bfloat16()
    vc = torch.randn(nb, NKV, BLOCK, HD, generator=g).bfloat16()
    k_tt = ttnn.from_torch(kc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(vc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    B = n_users * T
    pos = []
    for u in range(n_users):
        base = isl - 1 - T + 1 + 3 * u  # user u verifies positions [base, base+T)
        for j in range(T):
            pos.append(base + j if u < active else -1)
    cur_pos = torch.tensor(pos, dtype=torch.int32)
    q = torch.randn(1, B, NH, HD, generator=g).bfloat16()
    return dict(k=k_tt, v=v_tt, user_rows=user_rows, cur_pos=cur_pos, q=q, B=B)


def _legacy(device, inp, n_users, T):
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
    )
    pt = inp["user_rows"].repeat_interleave(T, dim=0).contiguous()
    q_tt = ttnn.from_torch(
        inp["q"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    pt_tt = ttnn.from_torch(pt, dtype=ttnn.int32, device=device)
    cp_tt = ttnn.from_torch(inp["cur_pos"], dtype=ttnn.int32, device=device)

    def run():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            inp["k"],
            inp["v"],
            page_table_tensor=pt_tt,
            cur_pos_tensor=cp_tt,
            scale=SCALE,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return run


def _fold(device, inp, n_users, T):
    gpu, cap = FOLD_PLAN[T]
    G = gpu * n_users
    Tg = T // gpu
    grid = device.compute_with_storage_grid_size()
    max_cores = min(cap, max(1, (grid.x * grid.y) // G))
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
        max_cores_per_head_batch=max_cores,
    )
    pt = inp["user_rows"].repeat_interleave(gpu, dim=0).contiguous()
    qpad = torch.zeros(1, inp["B"], TILE, HD, dtype=torch.bfloat16)
    qpad[:, :, :NH] = inp["q"]
    q_tt = ttnn.from_torch(
        qpad.reshape(1, G, Tg * TILE, HD),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pt_tt = ttnn.from_torch(pt, dtype=ttnn.int32, device=device)
    cp_tt = ttnn.from_torch(inp["cur_pos"], dtype=ttnn.int32, device=device)

    def run():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            inp["k"],
            inp["v"],
            page_table_tensor=pt_tt,
            cur_pos_tensor=cp_tt,
            scale=SCALE,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            spec_multi_pos_tiles=Tg,
            spec_q_heads=NH,
        )

    return run


@pytest.mark.parametrize("case", _selected(), ids=[c[0] for c in _selected()])
def test_spec_sdpa_bench(device, case):
    cid, n_users, T, isl, active = case
    iters = int(os.environ.get("LANES_ITERS", "5"))
    inp = _build(device, n_users, T, isl, active)
    outs = {}
    for mode in _modes():
        run = (_legacy if mode == "legacy" else _fold)(device, inp, n_users, T)
        o = run()  # compile + warm
        ttnn.synchronize_device(device)
        signpost(f"start_{cid}_{mode}")
        for _ in range(iters):
            ttnn.deallocate(o)
            o = run()
        ttnn.synchronize_device(device)
        signpost(f"stop_{cid}_{mode}")
        t = ttnn.to_torch(o).float()
        outs[mode] = t.reshape(1, inp["B"], TILE, HD)[:, :, :NH] if mode == "fold" else t[:, :, :NH]
        ttnn.deallocate(o)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    if os.environ.get("LANES_TORCH_CHECK") == "1":
        kc = ttnn.to_torch(inp["k"]).float()
        vc = ttnn.to_torch(inp["v"]).float()
        for r in range(inp["B"]):
            pos = int(inp["cur_pos"][r])
            if pos < 0:
                continue
            rows = inp["user_rows"][r // T]
            k = torch.cat([kc[int(blk)] for blk in rows], dim=1)[:, : pos + 1]
            v = torch.cat([vc[int(blk)] for blk in rows], dim=1)[:, : pos + 1]
            q = inp["q"][0, r].float()
            ref = torch.zeros(NH, HD)
            g = NH // NKV
            for h in range(NKV):
                ref[h * g : (h + 1) * g] = torch.softmax((q[h * g : (h + 1) * g] @ k[h].T) * SCALE, -1) @ v[h]
            errs = {m: round((o[0, r] - ref).abs().max().item(), 4) for m, o in outs.items()}
            print(f"LANES_TORCH {cid} row {r} pos {pos} {errs}")
    if "legacy" in outs and "fold" in outs:
        a, b = outs["legacy"], outs["fold"]
        act = inp["cur_pos"] >= 0
        a, b = a[:, act], b[:, act]
        pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        print(f"LANES_CMP {cid} pcc={pcc:.6f} maxabs={(a - b).abs().max().item():.4g} equal={torch.equal(a, b)}")
        assert pcc > 0.998  # the 32-row legacy call runs 1 core per (row, head) at 32k and is the less accurate side
