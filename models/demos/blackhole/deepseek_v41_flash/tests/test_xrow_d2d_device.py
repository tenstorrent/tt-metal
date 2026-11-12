# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""(derived from test_xrow_overlap_device / T24) DSV41_XROW_D2D device test: host-staged copy (T24 batched) vs the device-to-device copy (DSV41_XROW_D2D=1), bit-identity of logits vs cold and host; then DSV41_D2D_REPS repeated d2d copies interleaved with decode replays.

(original docstring:)  Device test of the cross-row hit of the paged APC (DSV41_APC_PAGED_XROW=1): the production B=8 geometry (4 rows x 2 users, single chunk trace, fp8 pool, bounded replay), layers 0-23.

A donor prompt D runs cold on a user of mesh row 0 and its blocks are cached there. Both users of row 0 are then occupied (index reservations), and a prompt P that shares the first ``s`` tokens
of D is planned: the best hit is on row 0, which has no free user, so the index plans a CROSS-ROW hit on another row (fresh pages, donor blocks pinned); the model copies the blocks (host-staged)
and continues P from ``s``. Compared with a cold run of P (and cold-vs-cold in another row as the noise baseline): last-token logits and 8 greedy decode tokens. A third request that finds the same
prefix must now hit LOCALLY on the receiving row.

    DSV41_APC_PAGED=1 DSV41_APC_PAGED_XROW=1 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_paged_xrow_device.py -s
"""

import json
import os
import time

import pytest
import torch

import ttnn

U = int(os.environ.get("DSV41_D2D_U", "2"))  # users per mesh row (B = 4 * U; production B=16 is U=4)
CHUNK = 512
LAYERS = os.environ.get("DSV41_PH_LAYERS", "0-23")
DECODE = int(os.environ.get("DSV41_PH_DECODE", "8"))
MAXCTX = int(os.environ.get("DSV41_PH_MAXCTX", "16384"))
HIT = int(os.environ.get("DSV41_OV_HIT", "65536"))
CQS = int(os.environ.get("DSV41_OV_CQS", "2"))
REPS = int(os.environ.get("DSV41_D2D_REPS", "20"))  # repeated d2d copies interleaved with decode replays
DECODE_STEPS = int(os.environ.get("DSV41_OV_STEPS", "60"))
CASES = [
    tuple(int(x) for x in c.split(":"))
    for c in os.environ.get("DSV41_XR_CASES", "6000:3072:6000,8000:6144:8400").split(",")
]


def set_env():
    os.environ.update(
        {
            "DSV41_APC_PAGED": "1",
            "DSV41_APC_PAGED_XROW": "1",
            "DSV41_APC_PAGED_XROW_MIN": "1024",
            "DSV41_PF_PAGED": "read",
            "DSV41_PF_SINGLE": "1",
            "DSV41_PF_UMASK": "1",
            "DSV41_PREFILL_UP": str(U),
            "DSV41_PREFILL_UP_BATCH": "1",
            "DSV41_POOL_DTYPE": "fp8",
            "DSV41_FP8_SERVING": "1",
            "DSV41_PF_PAGED_FP8": "1",
            "DSV41_MO_OVERLAP": "0",
        }
    )


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return dict(
        max=float(d.max()),
        mean=float(d.mean()),
        argmax_equal=int(a.argmax()) == int(b.argmax()),
        bits=bool(torch.equal(a.float(), b.float())),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": int(os.environ.get("DSV41_L1_SMALL", "16384")),
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
                "num_command_queues": CQS,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(9000)
@torch.no_grad()
def test_xrow_d2d(mesh_device):
    """arms (DSV41_D2D_ARMS, default host,d2d,d2d,d2d: the first of each kind includes compile) then DSV41_D2D_REPS d2d copies each followed by decode replays of a decoder on another row."""
    set_env()
    os.environ["DSV41_XROW_D2D"] = "1"
    os.environ["DSV41_XROW_SYNC_EVERY"] = "8"
    os.environ["DSV41_XROW_BATCH"] = "1"
    from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_model as APM
    from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
    from models.tt_transformers.tt.common import PagedAttentionConfig

    md = mesh_device
    log = lambda m: print(m, flush=True)
    res = {"hit": HIT}
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    ctx = int(os.environ.get("DSV41_PH_MAXCTX", str(HIT + 8192)))
    pp = default_page_params(ctx, U)
    pp["page_max_num_blocks_per_dp"] += APM.extra_pool_pages(U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    pa = m.enable_apc_paged()
    assert pa is not None and pa.idx.xrow, "paged APC / cross-row switch off"
    S_pad = m.single_spad(CHUNK)
    g = torch.Generator().manual_seed(77)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)
    L = HIT + 4096
    P = rnd(L)
    D = torch.cat([P[:HIT], rnd(L - HIT)])

    def run(u, Pp, st, rid):
        t1 = time.perf_counter()
        r = m.prefill_interleaved(
            [(u, Pp, st, Pp.numel())],
            CHUNK,
            s_pad_max=S_pad,
            want_logits=True,
            totals={u: Pp.numel()},
            final={u: True},
            rids={u: rid},
        )
        m.flush_key_exports(why="test")
        return int(r[u][0]), r[u][1].float().clone(), time.perf_counter() - t1

    def clean(keep=()):
        for x in range(B):
            if x not in keep:
                m.release_user(x)
        pa.reset()

    def dec_step(u, tok_, pos_):
        tok = torch.zeros(B, dtype=torch.long)
        pos = torch.zeros(B, dtype=torch.long)
        tok[u], pos[u] = tok_, pos_
        return int(m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True).clone()[u])

    def one_copy(tag, use_d2d):
        pa.d2d = d2d_obj if use_d2d else None
        run(0, D, 0, f"donor_{tag}")
        m.release_user(0)
        for i_ in range(U):
            pa.idx.bind(f"occ{i_}_{tag}", 0, user=i_)
        plan = pa.idx.plan_hit(f"x_{tag}", P.numpy(), L)
        assert plan.xrow_src == 0 and plan.row != 0 and plan.hit_tokens > 0, plan
        s0, f0 = pa.stats["xrow_s"], pa.stats["xrow_failed"]
        t_x, l_x, dt_x = run(plan.user, P, plan.hit_tokens, f"x_{tag}")
        log(pa.line())
        for i_ in range(U):
            pa.idx.release(f"occ{i_}_{tag}")
        return l_x, pa.stats["xrow_s"] - s0, pa.stats["xrow_failed"] - f0, dt_x, plan

    def mem():
        ttnn.synchronize_device(md)
        mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
        return dict(
            alloc_mib=round(mv.total_bytes_allocated_per_bank / 2**20, 2),
            free_mib=round(mv.total_bytes_free_per_bank / 2**20, 2),
            largest_free_mib=round(mv.largest_contiguous_bytes_free_per_bank / 2**20, 2),
            l1_alloc_b=ttnn.get_memory_view(md, ttnn.BufferType.L1).total_bytes_allocated_per_bank,
        )

    clean()
    t_cold, l_cold, dt_cold = run(0, P, 0, "cold")
    res["cold_prefill_s"] = round(dt_cold, 2)
    d2d_obj = (
        pa.d2d
    )  # (built by init_d2d (apc_paged_build) in the compile of the first prefill, before the trace captures)
    assert d2d_obj is not None, "DSV41_XROW_D2D did not build the d2d buffers (init_d2d not run?)"
    last_host = None
    arms = []
    for k, arm in enumerate(os.environ.get("DSV41_D2D_ARMS", "host,d2d,d2d,d2d").split(",")):
        clean()
        l_x, xs, failed, dt_x, plan = one_copy(f"{arm}{k}", arm == "d2d")
        if arm == "host":
            last_host = l_x
        r = dict(
            arm=arm,
            xrow_s=round(xs, 3),
            failed=failed,
            call_s=round(dt_x, 2),
            hit=plan.hit_tokens,
            bits_vs_cold=bool(torch.equal(l_x, l_cold)),
            maxdiff_vs_cold=float((l_x - l_cold).abs().max()),
            argmax_ok=int(l_x.argmax()) == int(l_cold.argmax()),
            bits_vs_host=None if last_host is None else bool(torch.equal(l_x, last_host)),
        )
        arms.append(r)
        log("ARM " + json.dumps(r))
        pa.stats["xrow_s"] = 0.0
        assert failed == 0, "the copy fell back to recompute"
    res["arms"] = arms
    # repeated d2d copies interleaved with decode replays of a decoder on row 2
    clean()
    d_user = 2 * U  # a decoder on row 2
    t0_, _, _ = run(d_user, rnd(2048), 0, "dec")
    cur, pos0 = t0_, 2048
    for k in range(5):
        cur = dec_step(d_user, cur, pos0 + k)
    pos0 += 5
    reps = []
    res["mem_before_reps"] = mem()
    log("MEM before reps " + json.dumps(res["mem_before_reps"]))
    for rep in range(REPS):
        for x in range(B):
            if x // U != 2:
                m.release_user(x)
        pa.reset()
        for _ in range(4):
            cur = dec_step(d_user, cur, pos0)
            pos0 += 1
        l_x, xs, failed, dt_x, plan = one_copy(f"rep{rep}", True)
        for _ in range(4):
            cur = dec_step(d_user, cur, pos0)
            pos0 += 1
        r = dict(
            rep=rep,
            xrow_s=round(xs, 3),
            failed=failed,
            bits_vs_cold=bool(torch.equal(l_x, l_cold)),
            maxdiff=float((l_x - l_cold).abs().max()),
            mem=mem(),
        )
        reps.append(r)
        log("REP " + json.dumps(r))
        assert failed == 0
        pa.idx.check()
    res["reps"] = reps
    out_f = os.environ.get("DSV41_OV_OUT")
    if out_f:
        json.dump(res, open(out_f, "w"), indent=1)
    log("RESULT " + json.dumps(res))
