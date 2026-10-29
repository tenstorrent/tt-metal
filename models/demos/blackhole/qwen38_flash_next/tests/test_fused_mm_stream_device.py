# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x52: ``mm_stream`` against the stock DRAM-sharded decode linear (same weights, activation, HiFi2/HiFi4 fp32-DEST compute
config), BIT FOR BIT on one die (1x1 mesh), many seeds, and traced timing (slope of a 4- and a 12-op trace)."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mm_stream as ms
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import placement as block_placement

BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16
RESULTS = Path(os.environ.get("X52_RESULTS", "/tmp/x52_stream.jsonl"))
VALID_ROWS = 1  # the decode step's one live row (mm_stream's compact activation carries row 0 only)
HUBS = int(os.environ.get("X52_HUBS", "1"))
SEEDS = [int(s) for s in os.environ.get("X52_SEEDS", "0,1,2,3,4,5").split(",")]
# name: K, N, dtype, readers, activation cores
SHAPES = {
    "gdn_in": dict(k=2560, n=2112, dtype=BF8, readers=2, cores=8),
    "router": dict(k=2560, n=512, dtype=BF16, readers=1, cores=5),
    "gate_up": dict(k=2560, n=352, dtype=BF8, readers=1, cores=5),
    "out_proj": dict(k=768, n=2560, dtype=BF8, readers=2, cores=8),
    "qsa_proj": dict(k=2560, n=2304, dtype=BF8, readers=1, cores=8),
}
ONLY = [s for s in os.environ.get("X52_OPS", "gdn_in").split(",") if s]


def stock_config(device, spec):
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    two = (k, n) in dm.TWO_WORKER_PROJECTIONS and spec["readers"] == 2
    act_cfg, prog = dm.dram_sharded_matmul_configs(
        device, k, n, num_cores=spec["cores"], num_workers_per_dram_bank=2 if two else 1
    )
    ccfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, dm.dense_math_fidelity_name(dtype)),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    wcfg = dm.dram_sharded_weight_memory_config(device, k, n, num_workers_per_dram_bank=2 if two else 1)
    return act_cfg, prog, ccfg, wcfg


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", ONLY)
def test_mm_stream_vs_stock(mesh_device, op):
    device = mesh_device
    spec = SHAPES[op]
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    act_cfg, prog, ccfg, wcfg = stock_config(device, spec)
    cols_list = [int(c) for c in os.environ.get("X52_COLS", "1,2,3,4").split(",")]
    blk = int(os.environ.get("X52_BLK", "10" if k % 10 == 0 and (k // 32) % 10 == 0 else "8"))
    exclude = block_placement.reserved_cores(device)
    all_equal = True
    for cols, seed in [(c, sd) for c in cols_list for sd in SEEDS]:
        g = torch.Generator().manual_seed(500 + seed)
        w = (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16)
        a = torch.randn(1, 1, 32, k, generator=g) * 0.5
        if seed % 2 == 0:
            a[:, :, 1:, :] = 0
        a = a.to(torch.bfloat16)
        weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=wcfg)
        if os.environ.get("X52_ACT1"):
            a = a[:, :, :1, :].contiguous()  # the model's activation: logical [1, 1, 1, K]
        logger.info(
            f"BANKS new {ms.shard_banks(weight)} old {[int(r.start.x) for r in weight.memory_config().shard_spec.grid.ranges()]} ranges {len(weight.memory_config().shard_spec.grid.ranges())}"
        )
        act_dram = ttnn.from_torch(
            a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        act = ttnn.to_memory_config(act_dram, act_cfg)
        ref = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        ref_t = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, n)[:VALID_ROWS, :n]
        mine = ms.mm_stream(act, weight, k=k, n=n, blk=blk, cols=cols, exclude=exclude)
        mine_t = ttnn.to_torch(mine).reshape(-1, mine.shape[-1])[:VALID_ROWS, :n]
        eq = torch.equal(ref_t.view(torch.int16), mine_t.view(torch.int16))
        bad = int((ref_t.view(torch.int16) != mine_t.view(torch.int16)).sum())
        logger.info(
            f"{op} seed {seed}: bitwise equal {eq} ({bad} differing of {ref_t.numel()}; max abs {(ref_t.float() - mine_t.float()).abs().max():.3e})"
        )
        if not eq:
            cols_bad = sorted(
                {int(c) // 32 for c in torch.nonzero(ref_t.view(torch.int16) != mine_t.view(torch.int16))[:, 1]}
            )
            logger.info(
                f"ref[64:72]={r[0,64:72].tolist()} got[64:72]={o[0,64:72].tolist()} ref[0:4]={r[0,0:4].tolist()} got[0:4]={o[0,0:4].tolist()}"
            )
            logger.info(f"differing tile columns {cols_bad}")
        all_equal &= eq
        if seed == SEEDS[0] and cols == cols_list[0] and os.environ.get("X52_TIME", "1") == "1":

            def stock_call():
                o = ttnn.linear(
                    act,
                    weight,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=prog,
                    compute_kernel_config=ccfg,
                )
                ttnn.deallocate(o)

            def mine_call():
                ttnn.deallocate(ms.mm_stream(act, weight, k=k, n=n, blk=blk, cols=cols, exclude=exclude))

            row = dict(
                op=op,
                cols=cols,
                blk=blk,
                stock_us=round(traced_wall_us(device, stock_call, replays=200)[0], 2),
                stream_us=round(traced_wall_us(device, mine_call, replays=200)[0], 2),
            )
            logger.info(json.dumps(row))
            with RESULTS.open("a") as f:
                f.write(json.dumps(row) + "\n")
        for t in (weight, act_dram, act, ref, mine):
            ttnn.deallocate(t)
    assert all_equal


REAL = {  # op -> glob of a real cached tensorbin (layer given in the pattern)
    "router": "*/moe-small/**/mesh-1x8-physical-1-5-4-0-2-3-7-6/backbone/layer-09/router_replicated_dram_sharded_dtype_BFLOAT16_layout_TILE.tensorbin",
    "gate_up": "*/moe-small/**/mesh-1x8-physical-1-0-2-3-7-6-4-5/backbone/layer-00/shared_gate_up_scalar_dram_sharded.bf8b_dtype_BFLOAT8_B_layout_TILE.tensorbin",
}


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", list(REAL))
def test_mm_stream_real_weights(mesh_device, op):
    """A real cached layer's weights (router layer 9, shared gate|up|scalar layer 0; die 0's shard) against the stock op."""
    import glob

    device = mesh_device
    spec = SHAPES[op]
    k, n = spec["k"], spec["n"]
    root = "/home/ttuser/models/qwen38_cache/caches/c32768-vllm/components/"
    path = glob.glob(root + REAL[op], recursive=True)[0]
    host = ttnn.load_tensor(path)
    part = ttnn.get_device_tensors(host)[0]
    weight = ttnn.to_device(part, device, memory_config=part.memory_config())
    act_cfg, prog, ccfg, _wcfg = stock_config(device, spec)
    exclude = block_placement.reserved_cores(device)
    for seed in range(8):
        g = torch.Generator().manual_seed(900 + seed)
        a = torch.zeros(1, 1, 32, k)
        a[:, :, 0, :] = torch.randn(k, generator=g) * (0.3 + 0.2 * seed)
        act = ttnn.to_memory_config(
            ttnn.from_torch(
                a.to(torch.bfloat16),
                dtype=BF16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            act_cfg,
        )
        ref = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        ref_t = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(32, -1)[:1, :n]
        for cols in (1, 2):
            mine_t = ttnn.to_torch(ms.mm_stream(act, weight, k=k, n=n, blk=10, cols=cols, exclude=exclude)).reshape(
                32, -1
            )[:1, :n]
            assert torch.equal(ref_t.view(torch.int16), mine_t.view(torch.int16)), (op, seed, cols)
        logger.info(f"{op} real weights seed {seed}: bitwise equal (cols 1, 2)")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", ["router", "gate_up", "gdn_in", "qsa_proj"])
def test_linear_entry_sharded_output(mesh_device, op):
    """``mm_stream.linear`` (the model's entry) writes the STOCK op's L1 width-sharded output layout: same shape, memory config, row 0."""
    os.environ["QWEN38_MM_STREAM"] = "1"
    device = mesh_device
    spec = SHAPES[op]
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    act_cfg, prog, ccfg, wcfg = stock_config(device, spec)
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16)
    weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=wcfg)
    exclude0 = block_placement.reserved_cores(device)
    SHAPES_COLS = {"router": 1, "gate_up": 2, "gdn_in": 2, "qsa_proj": 2}
    for call in range(4):
        a = torch.zeros(1, 1, 1, k)
        a[..., :] = torch.randn(k, generator=g) * 0.5
        act = ttnn.to_memory_config(
            ttnn.from_torch(
                a.to(torch.bfloat16),
                dtype=BF16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            act_cfg,
        )
        ref = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        if True:
            from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp_

            sentinel = ttnn.from_torch(
                torch.full((1, 1, 32, n), 7.0).to(torch.bfloat16),
                dtype=BF16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            fp_.run_program(
                [act, weight, sentinel],
                ms.mm_stream_program(act, weight, sentinel, k=k, n=n, blk=10, cols=SHAPES_COLS[op], exclude=exclude0),
            )
            sv = ttnn.to_torch(sentinel).reshape(32, -1)[:1, :n].float()
            rr0 = (
                ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG))
                .reshape(-1, ref.shape[-1])[:1]
                .float()
            )
            for t in range(n // 32):
                seg = sv[0, 32 * t : 32 * t + 32]
                logger.info(
                    f"sentinel tile {t}: unwritten {bool((seg == 7).all())} equals ref {bool(torch.equal(seg, rr0[0, 32 * t : 32 * t + 32]))} first {seg[:3].tolist()} ref {rr0[0, 32*t:32*t+3].tolist()}"
                )
            ttnn.deallocate(sentinel)
            for rep in range(3):
                d_ = ttnn.to_torch(
                    ms.mm_stream(act, weight, k=k, n=n, blk=10, cols=SHAPES_COLS[op], exclude=exclude0)
                ).reshape(32, -1)[:1, :n]
                rr = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1]
                tile_ok = [
                    bool(torch.equal(rr.view(torch.int16).reshape(-1, 32)[t], d_.view(torch.int16).reshape(-1, 32)[t]))
                    for t in range(rr.shape[-1] // 32)
                ]
                logger.info(
                    f"consecutive DRAM-out run {rep} equals ref: {torch.equal(rr.view(torch.int16), d_.view(torch.int16))} tiles ok {tile_ok}"
                )
        if call > 0:  # the DRAM-out program first (the same placement as the linear's)
            r0 = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1]
            dram0 = ttnn.to_torch(
                ms.mm_stream(act, weight, k=k, n=n, blk=10, cols=SHAPES_COLS[op], exclude=exclude0)
            ).reshape(32, -1)[:1, :n]
            logger.info(
                f"DRAM-out variant (before the linear) equals ref: {torch.equal(r0.view(torch.int16), dram0.view(torch.int16))}"
            )
        got = ms.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        act_back = ttnn.to_torch(ttnn.to_memory_config(act, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, k)[:1]
        logger.info(
            f"act intact after the streamed linear: {torch.equal(act_back.view(torch.int16), a.to(torch.bfloat16).reshape(1, k).view(torch.int16))}"
        )
        ref2 = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        logger.info(
            f"stock rerun equals first stock: {torch.equal(ttnn.to_torch(ttnn.to_memory_config(ref2, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1].view(torch.int16), ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1].view(torch.int16))}"
        )
        assert tuple(got.shape) == tuple(ref.shape) and got.memory_config() == ref.memory_config(), (
            got.memory_config(),
            ref.memory_config(),
        )
        r = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1]
        o = ttnn.to_torch(ttnn.to_memory_config(got, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, got.shape[-1])[:1]
        if call > 0:
            dram = ttnn.to_torch(
                ms.mm_stream(act, weight, k=k, n=n, blk=10, cols=SHAPES_COLS[op], exclude=exclude0)
            ).reshape(32, -1)[:1, :n]
            logger.info(f"DRAM-out variant equals ref: {torch.equal(r.view(torch.int16), dram.view(torch.int16))}")
        eq = torch.equal(r.view(torch.int16), o.view(torch.int16))
        logger.info(
            f"{op} call {call} (streamed {call > 0}): row 0 equal {eq}; shape {tuple(got.shape)} cfg {got.memory_config()}"
        )
        if not eq:
            d = torch.nonzero(r.view(torch.int16) != o.view(torch.int16))[:, 1]
            logger.info(
                f"ref[64:68]={r[0,64:68].tolist()} got[64:68]={o[0,64:68].tolist()} ref[0:3]={r[0,0:3].tolist()} got[0:3]={o[0,0:3].tolist()} ref[96:99]={r[0,96:99].tolist()} got[96:99]={o[0,96:99].tolist()}"
            )
            rt = r.view(torch.int16).reshape(-1, 32)
            ot = o.view(torch.int16).reshape(-1, 32)
            match = []
            for t in range(ot.shape[0]):
                m = [u for u in range(rt.shape[0]) if torch.equal(rt[u], ot[t])]
                match.append(m[0] if m else -1)
            logger.info(f"got tile t equals ref tile: {match}")
            logger.info(
                f"differing tile columns {sorted({int(c) // 32 for c in d})} nonzero_out {int((o != 0).sum())} of {o.numel()}"
            )
        for t_ in (act, ref, ref2, got):
            ttnn.deallocate(t_)
        assert eq, (op, call)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fresh_program_launch(mesh_device):
    """A freshly built program (program-cache miss: a new slots value) against a re-launched one."""
    device = mesh_device
    spec = SHAPES["router"]
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    act_cfg, prog, ccfg, wcfg = stock_config(device, spec)
    g = torch.Generator().manual_seed(5)
    weight = ttnn.from_torch(
        (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=wcfg,
    )
    a = torch.randn(1, 1, 1, k, generator=g).to(torch.bfloat16)
    act = ttnn.to_memory_config(
        ttnn.from_torch(a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        act_cfg,
    )
    ref = ttnn.to_torch(
        ttnn.to_memory_config(
            ttnn.linear(
                act,
                weight,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=prog,
                compute_kernel_config=ccfg,
            ),
            ttnn.DRAM_MEMORY_CONFIG,
        )
    ).reshape(-1, n)[:1]
    excl = block_placement.reserved_cores(device)
    for slots in (6, 6, 6, 7, 7, 8, 8, 5, 5):
        out = ms.mm_stream(act, weight, k=k, n=n, blk=10, cols=1, slots=slots, exclude=excl)
        got = ttnn.to_torch(out).reshape(32, -1)[:1, :n]
        logger.info(f"slots {slots}: equals ref {torch.equal(ref.view(torch.int16), got.view(torch.int16))}")
        ttnn.deallocate(out)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", ["router", "gate_up", "gdn_in", "qsa_proj"])
def test_fresh_linear_l1(mesh_device, op):
    """The model entry after a plain DRAM-out run: the L1 width-sharded output of ``mm_stream.linear`` against the stock op."""
    os.environ["QWEN38_MM_STREAM"] = "1"
    device = mesh_device
    spec = SHAPES[op]
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    act_cfg, prog, ccfg, wcfg = stock_config(device, spec)
    g = torch.Generator().manual_seed(5)
    weight = ttnn.from_torch(
        (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=wcfg,
    )
    for call in range(4):
        a = torch.randn(1, 1, 1, k, generator=g).to(torch.bfloat16)
        act = ttnn.to_memory_config(
            ttnn.from_torch(
                a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            act_cfg,
        )
        ref = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        got = ms.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        r = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, n)[:1]
        o = ttnn.to_torch(ttnn.to_memory_config(got, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, n)[:1]
        logger.info(f"{op} call {call}: equals ref {torch.equal(r.view(torch.int16), o.view(torch.int16))}")
        for t_ in (act, ref, got):
            ttnn.deallocate(t_)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_l1_sentinel(mesh_device):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp_

    device = mesh_device
    spec = SHAPES["router"]
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    act_cfg, prog, ccfg, wcfg = stock_config(device, spec)
    g = torch.Generator().manual_seed(5)
    weight = ttnn.from_torch(
        (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=wcfg,
    )
    a = torch.randn(1, 1, 1, k, generator=g).to(torch.bfloat16)
    act = ttnn.to_memory_config(
        ttnn.from_torch(a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        act_cfg,
    )
    ref = ttnn.linear(
        act, weight, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, program_config=prog, compute_kernel_config=ccfg
    )
    cfg = ref.memory_config()
    r = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, n)[:1].float()
    sent = ttnn.from_torch(
        torch.full((1, 1, 1, n), 7.0).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    logger.info(f"sentinel cfg equal stock {sent.memory_config() == cfg} shape {tuple(sent.shape)}")
    fp_.run_program(
        [act, weight, sent],
        ms.mm_stream_program(
            act, weight, sent, k=k, n=n, blk=10, cols=1, exclude=block_placement.reserved_cores(device)
        ),
    )
    for key_, (rect_, rc_, hub_, asg_) in ms._PLANS.items():
        logger.info(
            f"PLAN rect {rect_} hub {(hub_.x, hub_.y)} assign {[(i, tuple(c)) for i, c in sorted(asg_.items())]} outcores {key_[-1]} act {key_[-2]}"
        )
    s = ttnn.to_torch(ttnn.to_memory_config(sent, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, n)[:1].float()
    for t in range(n // 32):
        seg = s[0, 32 * t : 32 * t + 32]
        match = [u for u in range(n // 32) if torch.equal(seg, r[0, 32 * u : 32 * u + 32])]
        logger.info(f"L1 sentinel tile {t}: unwritten {bool((seg == 7).all())} equals ref tile {match}")
