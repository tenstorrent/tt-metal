# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flat spatially pipelined streamed routed expert (SwiGLU), one Blackhole chip: the harness for the op in
``models/demos/mimo_v2_d_p/tt/flat_expert.py`` (FlatExpert).

    y = (act(x @ Wg, x @ Wu)) @ Wd        x [M, H], Wg / Wu [H, I], Wd [I, H]

MIMO_FL_M (tokens per expert the program is built for), MIMO_FL_EXPERTS, MIMO_FL_H / _I, MIMO_FL_WDTYPE, MIMO_FL_ACT;
MIMO_FL_DYN=1 (dynamic counts, the model's mode: MIMO_FL_COUNTS "a,b,..;c,d,.." / MIMO_FL_COUNTS_FILE count sets run
back to back on one program) or MIMO_FL_E2E=1 (static e2e: row-major bf16 x in, bfp8 y out); neither: pre-tiled x.
The builder's MIMO_FL_* tuning knobs are in flat_expert.py / FLAT_EXPERT_WORKLOG.md. Tag ``streamfl_H{H}_M{M}_E{E}_w{dtype}...``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.tt.flat_expert import KBLK, W_DTYPES, FlatExpert, _bank_sharded, _env_list, act_ref

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

MS = _env_list("MIMO_FL_M", "128,256,512", int)
EXPERTS = int(os.environ.get("MIMO_FL_EXPERTS", "4"))
ITERS = int(os.environ.get("MIMO_FL_ITERS", "3"))
H = int(os.environ.get("MIMO_FL_H", "7168"))
I = int(os.environ.get("MIMO_FL_I", "2048"))  # routed-expert intermediate size (per device: I / TP)
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
WDTYPES = _env_list("MIMO_FL_WDTYPE", "bf4")
DYN = int(os.environ.get("MIMO_FL_DYN", "0"))  # dynamic token counts read on device (implies E2E)
E2E = int(os.environ.get("MIMO_FL_E2E", "0")) or DYN  # row-major bf16 dispatch buffer in, bfp8 tile buffer out
PIN = int(os.environ.get("MIMO_FL_PIN", "0"))  # (the harness default; the model pins)
ACT = os.environ.get("MIMO_FL_ACT", "silu")
W_STD = float(os.environ.get("MIMO_FL_WSTD", "0.02"))  # weight init std (larger: the clamping activations clamp)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert_flat(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    assert m % 32 == 0
    E = EXPERTS
    tok_pad = -(-m // 32) * 32  # e2e static: expert e's region starts at row e * tok_pad of the dispatch buffer
    # per-expert token counts (dynamic mode: read on device; the program is built for up to m per expert)
    # dynamic mode: one or more count vectors (MIMO_FL_COUNTS "a,b,..;c,d,.." or MIMO_FL_COUNTS_FILE, a json list of
    # [label, counts]) run back to back on the one program, each with its own tag; regions packed like a real
    # dispatch buffer (expert a's rows follow expert a-1's, 32-row aligned)
    if DYN and os.environ.get("MIMO_FL_COUNTS_FILE"):
        count_sets = [(str(lb), [int(c) for c in cs]) for lb, cs in json.load(open(os.environ["MIMO_FL_COUNTS_FILE"]))]
    elif DYN and os.environ.get("MIMO_FL_COUNTS"):
        count_sets = [
            ("", [int(c) for c in cs.split(",")]) for cs in os.environ["MIMO_FL_COUNTS"].split(";") if cs.strip()
        ]
    else:
        count_sets = [("", [m] * E)]
    for _, cs in count_sets:
        assert len(cs) == E and max(cs) <= m, cs
    cnts = count_sets[0][1]
    band = [int(v) for v in os.environ.get("MIMO_FL_BAND", "1,1000000000").split(",")]
    pack_offs = lambda cs: [sum(-(-c // 32) * 32 for c in cs[:e]) for e in range(E)]
    cap = max(32, max(sum(-(-c // 32) * 32 for c in cs) for _, cs in count_sets)) if DYN else E * tok_pad

    torch.manual_seed(0)
    # independent weights per expert (MIMO_FL_DISTINCT_W=0: one set repeated): a wrong expert id / region / stale
    # pinned block then shows up in the per-expert check
    distinct = bool(int(os.environ.get("MIMO_FL_DISTINCT_W", "1")))
    W_l = []
    for e in range(E if distinct else 1):
        W_l.append((torch.randn(H, I) * W_STD, torch.randn(H, I) * W_STD, torch.randn(I, H) * 0.02))
    W_l = W_l if distinct else W_l * E
    NG = 2 * E + 2  # the routing's rows: local experts at odd global ids
    fe = FlatExpert(
        device, [W_l], m=m, H=H, I=I, n_global=NG, wdtype=wdtype, act=ACT, pin=PIN, dyn=DYN, e2e=E2E, band=band
    )
    c_ = fe.tag_cfg
    NP, G, MT, NH, HBUF, S, V = c_["NP"], c_["G"], c_["MT"], c_["NH"], c_["HBUF"], c_["S"], c_["V"]
    m_pad = S * MT * 32
    Wg, Wu, Wd = W_l[E - 1]  # (the static-mode reference checks the last expert)
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
    act_r = lambda g, u: act_ref(g, u, ACT)
    if not DYN:  # (dynamic mode builds its x per count set, below)
        xs = torch.randn(E, m_pad, H)
        xs[:, m:] = 0
        xs = xs.view(V, MT * 32, H)
        x_last = xs[(E - 1) * S :].reshape(m_pad, H)[:m]
        ref = (act_r(x_last @ Wg, x_last @ Wu)) @ Wd
        ref_q = (act_r(x_last @ q(Wg), x_last @ q(Wu))) @ q(Wd)
    if not E2E:  # pre-tiled x: blocks (v, K-block) of [MT * 32 x KBLK * 32], block k in region k % nreg
        xl = fe.x_layout
        xblocks = [
            xs[v][:, b * KBLK * 32 : (b + 1) * KBLK * 32]
            .reshape(MT, 32, KBLK, 32)
            .permute(0, 2, 1, 3)
            .reshape(-1, 32, 32)
            for v in range(V)
            for b in range(xl["nk_gu"])
        ]
        x_dev = _bank_sharded(
            [[torch.cat(xblocks[r :: xl["nreg"]]) for r in range(xl["nreg"])]], xl["banks"], ttnn.bfloat8_b, device
        )

    def dyn_host(cs):  # one count set: its dispatch buffer, per-expert x, counts / regions rows
        offs = pack_offs(cs)
        disp_ = torch.zeros(cap, H)
        xe = []
        for e in range(E):
            xe.append(torch.randn(cs[e], H))
            disp_[offs[e] : offs[e] + cs[e]] = xe[e]
        c_h = torch.full((1, NG), 7777, dtype=torch.int32)
        r_h = torch.full((1, NG), 999999, dtype=torch.int32)
        for e in range(E):
            c_h[0, 2 * e + 1] = cs[e]
            r_h[0, 2 * e + 1] = offs[e]
        return disp_, xe, c_h, r_h, offs

    counts_dev = regions_dev = None
    rm = lambda t, dt: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if DYN:
        disp, dyn_xe, c_host, r_host, dyn_offs = dyn_host(cnts)
        counts_dev, regions_dev = rm(c_host, ttnn.uint32), rm(r_host, ttnn.uint32)
    if E2E and not DYN:  # the model's dispatch buffer: row-major bf16 [cap, H], expert e's m tokens at row e * tok_pad
        xs_e = xs.view(E, m_pad, H)
        disp = torch.zeros(cap, H)
        for e in range(E):
            disp[e * tok_pad : e * tok_pad + cnts[e]] = xs_e[e, : cnts[e]]
    if E2E:
        x_dev = rm(disp, ttnn.bfloat16)
    y_dram = fe.alloc_output(x_dev)
    if os.environ.get("MIMO_FL_SHOW_KERNELS"):
        fe.show_kernels(fe.program(x_dev, y_dram, counts_dev, regions_dev))
    run = lambda: fe(x_dev, counts_dev, regions_dev, y=y_dram)
    qW = {}  # expert -> quantized (gate, up, down) for the reference

    for set_i, (label, cs) in enumerate(count_sets):
        if DYN and set_i:  # same buffers (the program's addresses), new contents
            cnts = cs
            disp, dyn_xe, c_host, r_host, dyn_offs = dyn_host(cnts)
            for h_, d_, dt_ in (
                (disp, x_dev, ttnn.bfloat16),
                (c_host, counts_dev, ttnn.uint32),
                (r_host, regions_dev, ttnn.uint32),
            ):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(h_, dtype=dt_, layout=ttnn.ROW_MAJOR_LAYOUT), d_)
        GU_ACC, L1ACC_GRP = c_["GU_ACC"], c_["L1ACC_GRP"]
        tag = (
            f"streamfl_H{H}_M{m}_E{E}_w{wdtype}"
            + (f"_{ACT}" if ACT != "silu" else "")
            + (f"_I{I}" if I != 2048 else "")
            + (f"_np{NP}g{G}" if (NP, G) != (1, 1) else "")
            + (f"_pin{c_['PIN']}" if c_["PIN"] else "")
            + (f"_xh{NH}" if NH != 1 else "")
            + (f"_hb{HBUF}" if HBUF != 3 else "")
            + ((f"_acc{GU_ACC}" + (str(L1ACC_GRP) if GU_ACC == "l1acc" else "")) if GU_ACC != "fp32" else "")
            + ("_dstfull" if c_["GU_FULL"] else "")
            + ("_rp" if c_["GU_RP"] else "")
            + ("_grect" if c_["GROUP_RECT"] else "")
            + (f"_mt{MT}" if os.environ.get("MIMO_FL_MT") else "")
            + ("_e2e" if E2E else "")
        )
        tok_act = (
            sum(c for c in cnts if c and band[0] <= c <= band[1]) if DYN else E * m
        )  # tokens this program processes
        if DYN:
            tag += (
                "_dyn"
                + (label or "-".join(map(str, cnts)))
                + (f"_b{band[0]}-{band[1]}" if os.environ.get("MIMO_FL_BAND") else "")
            )
        # logical weight bytes of the experts this launch streams (active ones; each gate/up / down set once)
        n_w = len([c for c in cnts if c and band[0] <= c <= band[1]]) if DYN else E
        w_bytes = n_w * 3 * H * I * w_tile / 1024
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATS_PATH.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "tag": tag,
                        "M": m,
                        "E": E,
                        "wdtype": wdtype,
                        "weight_bytes": w_bytes,
                        "flops": 6 * tok_act * H * I,
                        "tokens": tok_act,
                        "counts": cnts,
                        # logical activation bytes (e2e: bf16 row-major x in, bfp8 y out; pre-tiled: bfp8 x, bf16 y);
                        # the relays read x once per rectangle team (2x) on top of this
                        "x_bytes": tok_act * H * (2 if E2E else 1.0625),
                        "y_bytes": tok_act * H * (1.0625 if E2E else 2),
                        "config": {
                            "I": I,
                            "NP": NP,
                            "G": G,
                            "MT": MT,
                            "PIN": c_["PIN"],
                            "NH": NH,
                            "HBUF": HBUF,
                            "X_SLOTS": c_["X_SLOTS"],
                            "DRING": c_["DRING"],
                            "act": ACT,
                        },
                    }
                )
                + "\n"
            )
        for it in range(1 + ITERS):
            ttnn.synchronize_device(device)
            if it:
                signpost(f"{tag}_start")
            run()
            ttnn.synchronize_device(device)
            if it:
                signpost(f"{tag}_end")
            if os.environ.get("MIMO_FL_BETWEEN") == "matmul":  # a large multi-core matmul between launches (the flat
                # kernels must leave clean NoC state: see test_flat_expert_mesh.py)
                mm_a = ttnn.from_torch(
                    torch.randn(1, 1, 1024, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                mm_b = ttnn.from_torch(
                    torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                for t_ in (mm_a, mm_b, ttnn.matmul(mm_a, mm_b)):
                    ttnn.deallocate(t_)
                ttnn.synchronize_device(device)
                logger.info(f"between-op done after launch {it}")
            if it == 0:
                if E2E:  # every expert's rows at its region of the model-shaped output
                    yh = ttnn.to_torch(y_dram).float()
                    act = [
                        e for e in range(E) if cnts[e] and band[0] <= cnts[e] <= band[1]
                    ]  # experts this program serves
                    for e in act:  # quantized-weight reference per expert (cached across count sets)
                        if e not in qW:
                            qW[e] = tuple(q(w_) for w_ in W_l[e])
                    if DYN:  # rows checked: all up to CHK_ALL per expert, else every other row tile (so every
                        # sub-block and every pinned chunk) plus the last 32 rows (host reference cost)
                        CHK_ALL = int(os.environ.get("MIMO_FL_CHECK_ALL", "512"))
                        rows_ = {
                            e: (
                                list(range(cnts[e]))
                                if cnts[e] <= CHK_ALL
                                else sorted(
                                    {r_ for r_ in range(cnts[e]) if (r_ // 32) % 2 == 0}
                                    | set(range(max(0, cnts[e] - 32), cnts[e]))
                                )
                            )
                            for e in act
                        }
                        xin = {e: dyn_xe[e][rows_[e]] for e in act}
                        yout = {e: yh[dyn_offs[e] : dyn_offs[e] + cnts[e]][rows_[e]] for e in act}
                    else:
                        xs_e = xs.view(E, m_pad, H)
                        xin = {e: xs_e[e, : cnts[e]] for e in act}
                        yout = {e: yh[e * tok_pad : e * tok_pad + cnts[e]] for e in act}
                    refs_q = {e: (act_r(xin[e] @ qW[e][0], xin[e] @ qW[e][1])) @ qW[e][2] for e in act}
                    pccs = [comp_pcc(refs_q[e], yout[e], 0.99) for e in act]
                    # magnitude too (PCC is scale-blind): norm ratio and relative error per expert
                    for e in act:
                        nr = float(yout[e].norm() / refs_q[e].norm())
                        rel = float((yout[e] - refs_q[e]).norm() / refs_q[e].norm())
                        logger.info(f"expert {e} ({cnts[e]} tok): norm ratio {nr:.4f}, rel err {rel:.4f}")
                        assert os.environ.get("MIMO_FL_NO_NORM_CHECK") or (
                            0.9 < nr < 1.1 and rel < 0.2
                        ), f"expert {e}: norm ratio {nr:.3f}, rel err {rel:.3f}"
                    silu_ref = lambda e: (torch.nn.functional.silu(xin[e] @ qW[e][0]) * (xin[e] @ qW[e][1])) @ qW[e][2]
                    if ACT != "silu" and act:  # the device must match the chosen activation better than SiLU-GLU
                        alt = [comp_pcc(silu_ref(e), yout[e], 0)[1] for e in act]
                        logger.info(
                            f"{ACT}: min PCC {min(p_[1] for p_ in pccs):.5f} vs silu-glu reference {min(alt):.5f}"
                        )
                        # (only when the two references differ: e.g. the clamps never engage at small weights)
                        sep = min(comp_pcc(silu_ref(e), refs_q[e], 0)[1] for e in act)
                        if sep < 0.999:
                            assert min(p_[1] for p_ in pccs) > max(alt), "activation not distinguishable / wrong"
                    logger.info(f"e2e per-expert PCC {[round(float(p_[1]), 5) for p_ in pccs]}")
                    assert all(p_[0] for p_ in pccs) or os.environ.get("MIMO_FL_XRD_SKIP")
                    if DYN:
                        logger.info(f"dyn: counts {cnts} band {band} -> active {act}")
                        ok = True
                        pcc_q = min(float(p_[1]) for p_ in pccs) if pccs else 1.0
                    if not DYN:
                        got = yh[(E - 1) * tok_pad : (E - 1) * tok_pad + m]
                else:
                    got = ttnn.to_torch(y_dram).float()[(E - 1) * S * MT * 32 :][:m]
                if not DYN:
                    ok, pcc_q = comp_pcc(ref_q, got, 0.99)
                    logger.info(
                        f"static: norm ratio {float(got.norm() / ref_q.norm()):.4f} vs fp32 ref {float(got.norm() / ref.norm()):.4f}"
                    )
                pcc = comp_pcc(ref, got, 0.0)[1] if not DYN else "n/a"
                logger.info(f"{tag}: PCC {pcc_q} vs quantized-weight reference, {pcc} vs fp32")
                assert ok or os.environ.get("MIMO_FL_XRD_SKIP"), pcc_q
        if int(os.environ.get("MIMO_FL_B2B", "0")):  # dispatch probe: N launches back to back, no host sync between
            ttnn.synchronize_device(device)
            signpost(f"{tag}_b2b_start")
            for _ in range(int(os.environ["MIMO_FL_B2B"])):
                run()
            ttnn.synchronize_device(device)
            signpost(f"{tag}_b2b_end")
        if E2E and ITERS and not os.environ.get("MIMO_FL_XRD_SKIP"):  # the measured launches' output too:
            # every active expert's rows bit-identical to the checked warm-up launch
            y_last = ttnn.to_torch(y_dram).float()
            for e in act:
                o_ = dyn_offs[e] if DYN else e * tok_pad
                assert torch.equal(
                    y_last[o_ : o_ + cnts[e]], yh[o_ : o_ + cnts[e]]
                ), f"launch {ITERS}: expert {e} differs"
        logger.info(f"ran {tag}")
