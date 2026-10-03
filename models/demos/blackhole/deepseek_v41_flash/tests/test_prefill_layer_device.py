# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: one full decoder layer PREFILL on the device (S tokens x 16 users) vs the CPU reference dump, plus the decode state it leaves.

1. hidden-state PCC of the layer output streams / ffn pre vs ``prefill.h_out`` / ``pm_out`` (first S positions);
2. the decode cache (window ring, compressed latents) and ``prev_cs`` PCC vs the reference state (the cache is ZEROED before the prefill);
3. the existing DECODE layer run on that state reproduces the reference decode output (``dec_out``).
Env: DSV41_PREFILL_DIR (default /mnt/tt-data/ssinghal/dsv4-prefill-s{S})."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt.attention import WINDOW
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, pad_len
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE

T = 32  # tokens per chunk of the token-wise blocks


def rows_to_chunks(x, rows, Sp, S, dims_tail):
    """x [16,S,...] -> list of host chunks [rows*T, 1, ...] (per row: 4 users x Sp tokens, user-major, padded with the last real token)."""
    B = x.shape[0]
    xp = torch.cat([x, x[:, -1:].expand(B, Sp - S, *x.shape[2:])], dim=1)  # [16,Sp,...]
    xr = xp.reshape(rows, 4 * Sp, *x.shape[2:])
    return [xr[:, c * T : (c + 1) * T].reshape(rows * T, 1, *dims_tail) for c in range(4 * Sp // T)]


def chunks_to_users(ch, rows, Sp, S, tail):
    """list of [rows*T,...] host tensors -> [16, S, ...]."""
    xr = torch.stack([c.reshape(rows, T, *tail) for c in ch], dim=1).reshape(
        rows, 4 * Sp, *tail
    )  # [rows, chunks*T,...]
    return xr.reshape(rows * 4, Sp, *tail)[:, :S]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_id,S", [(0, 9), (2, 9), (20, 9), (0, 128), (2, 128), (20, 128)])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_prefill_layer(mesh_device, layer_id, S):
    md = mesh_device
    rows, cols = tuple(md.shape)
    B = rows * 4
    d = os.environ.get("DSV41_PREFILL_DIR", f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}")
    ref = torch.load(os.path.join(d, f"layer_{layer_id}.pt"))
    ref["S"] = S
    pf = ref["prefill"]
    chain = DSV41DecodeChain(md, users_per_row=4, max_comp=256)
    w = load_layer(layer_id, max_seq_len=256)
    layer, attn = chain.build_layer(layer_id, ref, w)
    ratio = w["meta"]["ratio"]
    # zero the decode state the build seeded from the reference: the prefill has to produce it
    ttnn.copy(ttnn.zeros_like(attn.cache), attn.cache)
    if getattr(attn, "prev_cs", None) is not None:
        ttnn.copy(ttnn.zeros_like(attn.prev_cs), attn.prev_cs)
    attn.prefill = DSV41PrefillAttention(attn, w["attn"]["attn_sink"])
    Sp = pad_len(S)

    def attn_check(tag):  # attention-only prefill on the reference input: where does the bad row come from?
        mv = ttnn.get_memory_view(md, ttnn.BufferType.L1)
        xin = torch.zeros(rows * 4, Sp, 5120)
        xin[:, :S] = pf["attn_in"].float()
        hh_ = ttnn.from_torch(
            xin.reshape(rows, 1, 4 * Sp, 5120),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
        )
        o_ = attn.prefill.forward(hh_, S)
        ttnn.synchronize_device(md)
        g_ = torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(o_)[r * cols]).float().reshape(-1, 5120) for r in range(rows)]
        )
        r_ = torch.stack(
            [c_.reshape(rows, T, -1) for c_ in rows_to_chunks(pf["attn_out"].float(), rows, Sp, S, (5120,))], 1
        ).reshape(rows, 4 * Sp, -1)
        bad_ = [t0_ for t0_ in range(0, 4 * Sp, 32) if not pcc(g_[:, t0_ : t0_ + 32], r_[:, t0_ : t0_ + 32]) > 0.99]
        print(
            f"ATTN CHECK {tag}: L1 largest contiguous free/bank {mv.largest_contiguous_bytes_free_per_bank} allocated {mv.total_bytes_allocated_per_bank}; bad row blocks {bad_}",
            flush=True,
        )

    attn_check("before PrefillMoE")
    pmoe = DSV41PrefillMoE(layer.moe, T=T)
    attn_check("after PrefillMoE creation")
    pl = DSV41PrefillLayer(layer, attn.prefill, pmoe, T=T)

    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t: ttnn.from_torch(
        t.float().contiguous(),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    xs_h = rows_to_chunks(pf["h_in"].float(), rows, Sp, S, (4, 5120))
    pre_h = rows_to_chunks(pf["pm_in"].float(), rows, Sp, S, (1, 4))
    xs, pres = [up(c) for c in xs_h], [up(c) for c in pre_h]
    pl.forward(xs, pres, S)  # compile + first MoE use
    ttnn.synchronize_device(md)
    attn_check("after a prefill layer run")
    times = []
    for rep in range(2):  # first call compiles
        if rep == 1 and os.environ.get("DSV41_PF_DEBUG") == "1":
            pl.debug = {}
        t0 = time.perf_counter()
        outs, pouts = pl.forward(xs, pres, S)
        ttnn.synchronize_device(md)
        times.append(time.perf_counter() - t0)
    rd = lambda t: torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float() for r in range(rows)])
    if pl.debug:
        rdc = lambda t: torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float().reshape(T, -1) for r in range(rows)]
        )
        rdc2 = lambda t: torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float().reshape(-1, 5120) for r in range(rows)]
        )
        refs = {
            "a": rows_to_chunks(pf["attn_out"].float(), rows, Sp, S, (5120,)),
            "hh": rows_to_chunks(pf["ffn_in"].float(), rows, Sp, S, (5120,)),
            "ffn": rows_to_chunks(pf["ffn_out"].float(), rows, Sp, S, (5120,)),
        }
        a_ref = torch.stack(
            [c_.reshape(rows, T, -1) for c_ in rows_to_chunks(pf["attn_out"].float(), rows, Sp, S, (5120,))], 1
        ).reshape(rows, 4 * Sp, -1)
        a_late = torch.stack(
            [
                ttnn.to_torch(ttnn.get_device_tensors(pl.debug["a_full"])[r * cols]).float().reshape(-1, 5120)
                for r in range(rows)
            ]
        )
        a_early = torch.stack(pl.debug["a_early"])
        for t0_ in range(0, 4 * Sp, 32):
            print(
                f"DEBUG a rows {t0_:3d}: early {pcc(a_early[:, t0_:t0_+32], a_ref[:, t0_:t0_+32]):.4f} late {pcc(a_late[:, t0_:t0_+32], a_ref[:, t0_:t0_+32]):.4f}",
                flush=True,
            )
        for c in range(len(xs)):
            ref_in = rows_to_chunks(pf["attn_in"].float(), rows, Sp, S, (5120,))[c].reshape(rows * T, -1)
            p_hs = pcc(rdc(pl.debug["hs"][c]), ref_in)
            hfull = rdc2(pl.debug["h"])[:, c * T : (c + 1) * T].reshape(rows * T, -1)
            p_h = pcc(hfull, ref_in)
            print(f"DEBUG chunk {c:2d}: collapse_norm out {p_hs:.4f} concat h rows {p_h:.4f}", flush=True)
            pa, ph = pcc(rdc(pl.debug["a"][c]), refs["a"][c].reshape(rows * T, -1)), pcc(
                rdc(pl.debug["hh"][c]), refs["hh"][c].reshape(rows * T, -1)
            )
            pm = pcc(rdc(pl.debug["m"][c]) + rdc(pl.debug["sh"][c]), refs["ffn"][c].reshape(rows * T, -1))
            print(f"DEBUG chunk {c:2d}: attn_out {pa:.4f} ffn_in {ph:.4f} ffn_out(m+sh) {pm:.4f}", flush=True)
    got = chunks_to_users([rd(o) for o in outs], rows, Sp, S, (4, 5120))
    got_pre = chunks_to_users([rd(o) for o in pouts], rows, Sp, S, (4,))
    ref_h, ref_pm = pf["h_out"].float(), pf["pm_out"].float()
    p_h = pcc(got, ref_h)
    p_streams = [round(pcc(got[:, :, i], ref_h[:, :, i]), 5) for i in range(4)]
    p_pre = pcc(got_pre, ref_pm)
    p_last = pcc(got[:, -1], ref_h[:, -1])
    rel = (got - ref_h).flatten(2).norm(dim=-1) / ref_h.flatten(2).norm(dim=-1).clamp(
        min=1e-6
    )  # [B,S] per-token relative error
    pos_p = rel.max(0).values
    bad = [t for t in range(S) if not float(pos_p[t]) < 0.15]
    print(
        f"PREFILL LAYER {layer_id} S={S}: per-token rel err max {float(rel.max()):.4f} mean {float(rel.mean()):.4f}",
        flush=True,
    )
    nan_u = [int(torch.isnan(got[u]).any()) for u in range(B)]
    print(
        f"PREFILL LAYER {layer_id} S={S}: bad positions {bad[:40]} (n={len(bad)}), users with NaN {nan_u}", flush=True
    )
    if bad:
        u_p = [[round(pcc(got[u, t], ref_h[u, t]), 3) for u in range(B)] for t in bad[:3]]
        print(f"PREFILL LAYER {layer_id}: per-user PCC at first bad positions {u_p}", flush=True)
    cache = torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(attn.cache)[r * cols]).float().reshape(4, -1, 512) for r in range(rows)]
    )
    st = ref["state"]
    msg = (
        f"PREFILL LAYER {layer_id} S={S}: hidden PCC {p_h:.5f} (streams {p_streams}) last-token {p_last:.5f} ffn-pre PCC {p_pre:.5f} "
        f"ring {pcc(cache[:, :S], st['window'][:, :S]):.5f}"
    )
    if ratio:
        nc = S // ratio
        msg += f" comp {pcc(cache[:, WINDOW:WINDOW + nc], st['comp'][:, :nc]):.5f}"
    print(msg + f" | eager prefill {times[0]:.2f}s first, {times[1]:.2f}s second", flush=True)

    # decode continuation on the prefill-written state
    from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc as _p

    x_dec, pre_dec = ref["dec_in"], ref["pre_in"]
    stp = attn.step_inputs(torch.full((B,), S))
    tx, tp = chain.to_dev(x_dec.reshape(B, -1), 4 * 5120), chain.to_dev(pre_dec.reshape(B, -1), 4)
    out, nxt = layer.forward(tx, tp, stp)
    ttnn.synchronize_device(md)
    got_d = chain.to_host(out, 4 * 5120).reshape(B, 1, 4, 5120)
    p_dec = _p(got_d, ref["dec_out"].float())
    print(f"PREFILL LAYER {layer_id} S={S}: decode-after-prefill streams PCC {p_dec:.5f}", flush=True)
    assert not bad and p_h > 0.999 and p_dec > 0.99
