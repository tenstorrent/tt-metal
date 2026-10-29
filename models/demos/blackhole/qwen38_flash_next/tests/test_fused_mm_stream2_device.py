# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x60: ``mm_stream2`` (re-laid weights, one contiguous region per core) against the stock DRAM-sharded decode linear, BIT FOR BIT on
one die: sentinel-filled outputs, a fresh activation per seed, real cached weights where available, traced timing.
Run through x60_mm_stream2/run_dev.sh (``timeout 600``)."""

import glob
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mm_stream2 as m2
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import placement
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16
RESULTS = Path(os.environ.get("X60_RESULTS", "/tmp/x60_stream2.jsonl"))
SEEDS = [int(s) for s in os.environ.get("X60_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
ROOT = "/home/ttuser/models/qwen38_cache/caches/c32768-vllm/components/"
# name: K, N, dtype, stock readers, activation cores, columns per core, K rows per block
SHAPES = {
    "router": dict(k=2560, n=512, dtype=BF16, readers=1, cores=5, cols=1, blk=10),
    "gate_up": dict(k=2560, n=352, dtype=BF8, readers=1, cores=5, cols=1, blk=16),
    "qsa_proj": dict(k=2560, n=2304, dtype=BF8, readers=1, cores=8, cols=3, blk=10),
    "gdn_in": dict(k=2560, n=2112, dtype=BF8, readers=2, cores=8, cols=4, blk=10),
    "out_proj": dict(k=768, n=2560, dtype=BF8, readers=2, cores=8, cols=2, blk=8),
}
REAL = {
    "router": "*/moe-small/**/mesh-1x8-physical-1-5-4-0-2-3-7-6/backbone/layer-09/router_replicated_dram_sharded_dtype_BFLOAT16_layout_TILE.tensorbin",
    "gate_up": "*/moe-small/**/mesh-1x8-physical-1-0-2-3-7-6-4-5/backbone/layer-00/shared_gate_up_scalar_dram_sharded.bf8b_dtype_BFLOAT8_B_layout_TILE.tensorbin",
}
ONLY = [s for s in os.environ.get("X60_OPS", "router").split(",") if s]


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
    nb = dm.bank_tiles(device, k, n, 2 if two else 1)
    return act_cfg, prog, ccfg, wcfg, nb


def fresh_act(device, act_cfg, k, seed, scale=0.5, rows=32):
    g = torch.Generator().manual_seed(900 + seed)
    a = torch.zeros(1, 1, rows, k)
    a[:, :, 0, :] = torch.randn(k, generator=g) * scale * (1 + 0.3 * seed)
    return ttnn.to_memory_config(
        ttnn.from_torch(
            a.to(BF16 and torch.bfloat16),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        act_cfg,
    )


def sentinel_run(device, act, weight2, spec, nb, exclude):
    """DRAM-out run into a sentinel-filled output (fresh allocation of a dirty buffer): unwritten tiles cannot pass."""
    k, n = spec["k"], spec["n"]
    out = fp.allocate((1, 1, 32, -(-n // 32) * 32), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    fill = ttnn.from_torch(
        torch.full((1, 1, 32, -(-n // 32) * 32), 7.0).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn.copy(fill, out)
    fp.run_program(
        [act, weight2, out],
        m2.mm_stream2_program(act, weight2, out, k=k, n=n, nb=nb, blk=spec["blk"], cols=spec["cols"], exclude=exclude),
    )
    got = ttnn.to_torch(out).reshape(32, -1)[:1, :n]
    ttnn.deallocate(out)
    ttnn.deallocate(fill)
    return got


def check(device, op, weight, spec, seeds, act_cfg, prog, ccfg, nb, tag):
    k, n = spec["k"], spec["n"]
    exclude = placement.reserved_cores(device)
    w2 = m2.from_stock(device, weight, k, n, nb, spec["cols"])
    # host + device round trip: the re-laid tensor decodes to the stock tensor's values
    back = m2.unlayout(ttnn.to_torch(w2).reshape(-1, 8 * 32).to(torch.bfloat16), k, n, nb, spec["cols"])
    orig = ttnn.to_torch(weight).reshape(-1, weight.shape[-1])[:k, :n].to(torch.bfloat16)
    assert torch.equal(
        back.view(torch.int16), orig.view(torch.int16)
    ), "re-laid weight does not decode to the stock weight"
    ok = True
    for seed in seeds:
        act = fresh_act(device, act_cfg, k, seed)
        ref = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=ccfg,
        )
        ref_t = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1, :n]
        got = sentinel_run(device, act, w2, spec, nb, exclude)
        eq = torch.equal(ref_t.view(torch.int16), got.view(torch.int16))
        bad = int((ref_t.view(torch.int16) != got.view(torch.int16)).sum())
        logger.info(f"{op} {tag} seed {seed}: bitwise equal {eq} ({bad} of {ref_t.numel()} differ)")
        if not eq:
            logger.info(
                f"differing tile columns {sorted({int(c) // 32 for c in torch.nonzero(ref_t.view(torch.int16) != got.view(torch.int16))[:, 1]})}"
            )
        ok &= eq
        ttnn.deallocate(act)
        ttnn.deallocate(ref)
    if os.environ.get("X60_TIME", "1") == "1":
        act = fresh_act(device, act_cfg, k, 0)

        def stock_call():
            ttnn.deallocate(
                ttnn.linear(
                    act,
                    weight,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=prog,
                    compute_kernel_config=ccfg,
                )
            )

        def mine_call():
            ttnn.deallocate(
                m2.mm_stream2(act, w2, k=k, n=n, nb=nb, blk=spec["blk"], cols=spec["cols"], exclude=exclude)
            )

        row = dict(
            op=op,
            tag=tag,
            cols=spec["cols"],
            blk=spec["blk"],
            stock_us=round(traced_wall_us(device, stock_call, replays=200)[0], 2),
            stream2_us=round(traced_wall_us(device, mine_call, replays=200)[0], 2),
            equal=ok,
        )
        logger.info(json.dumps(row))
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")
    ttnn.deallocate(w2)
    return ok


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", ONLY)
def test_random_weights(mesh_device, op):
    device = mesh_device
    spec = dict(SHAPES[op])
    spec["cols"] = int(os.environ.get("X60_COLS", spec["cols"]))
    spec["blk"] = int(os.environ.get("X60_BLK", spec["blk"]))
    act_cfg, prog, ccfg, wcfg, nb = stock_config(device, spec)
    g = torch.Generator().manual_seed(500)
    w = (torch.randn(1, 1, spec["k"], spec["n"], generator=g) * 0.02).to(torch.bfloat16)
    weight = ttnn.from_torch(w, dtype=spec["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=wcfg)
    assert check(device, op, weight, spec, SEEDS, act_cfg, prog, ccfg, nb, "random")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", [o for o in ONLY if o in REAL])
def test_real_weights(mesh_device, op):
    device = mesh_device
    spec = SHAPES[op]
    act_cfg, prog, ccfg, _wcfg, nb = stock_config(device, spec)
    path = glob.glob(ROOT + REAL[op], recursive=True)[0]
    part = ttnn.get_device_tensors(ttnn.load_tensor(path))[0]
    weight = ttnn.to_device(part, device, memory_config=part.memory_config())
    assert check(device, op, weight, spec, SEEDS, act_cfg, prog, ccfg, nb, "real")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dispatch_ablation(mesh_device):
    """Traced us of the router program with EMPTY kernels (dispatch cost only) and with kernel groups dropped (X60_ABL variants, set per call)."""
    device = mesh_device
    spec = SHAPES["router"]
    act_cfg, prog, ccfg, wcfg, nb = stock_config(device, spec)
    g = torch.Generator().manual_seed(500)
    weight = ttnn.from_torch(
        (torch.randn(1, 1, spec["k"], spec["n"], generator=g) * 0.02).to(torch.bfloat16),
        dtype=spec["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=wcfg,
    )
    w2 = m2.from_stock(device, weight, spec["k"], spec["n"], nb, spec["cols"])
    act = fresh_act(device, act_cfg, spec["k"], 0)
    exclude = placement.reserved_cores(device)
    for abl in ("empty", "empty,nosem", "empty,cbs="):
        os.environ["X60_ABL"] = abl

        def call():
            ttnn.deallocate(
                m2.mm_stream2(
                    act, w2, k=spec["k"], n=spec["n"], nb=nb, blk=spec["blk"], cols=spec["cols"], exclude=exclude
                )
            )

        try:
            us = traced_wall_us(device, call, replays=200)[0]
        except Exception as e:
            us = str(e)[:80]
        logger.info(f"ABLATION {abl}: {us}")
    os.environ.pop("X60_ABL", None)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", ["router", "gate_up", "gdn_in", "qsa_proj"])
def test_linear_entry_l1_output(mesh_device, op):
    """``mm_stream2.linear`` (the model's entry): the stock op's shape / L1 width-sharded memory config, row 0 bitwise equal, on a
    sentinel-filled L1 output (a dirty buffer at the same addresses), fresh activation per call, then under trace capture.
    """
    os.environ["QWEN38_MM_STREAM2"] = "1"
    device = mesh_device
    spec = SHAPES[op]
    k, n = spec["k"], spec["n"]
    act_cfg, prog, ccfg, wcfg, nb = stock_config(device, spec)
    g = torch.Generator().manual_seed(11)
    weight = ttnn.from_torch(
        (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16),
        dtype=spec["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=wcfg,
    )
    kw = dict(memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, program_config=prog, compute_kernel_config=ccfg)
    streamed_before = m2.CALLS.get((k, n), 0)
    for call in range(5):
        act = fresh_act(device, act_cfg, k, call, rows=1)
        ref = ttnn.linear(act, weight, **kw)
        ref_t = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, ref.shape[-1])[:1, :n]
        if call >= 2:  # dirty the L1 buffer the output will land on
            dirty = ttnn.to_memory_config(
                ttnn.from_torch(
                    torch.full((1, 1, 32, n), 7.0).to(torch.bfloat16),
                    dtype=BF16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                ),
                ref.memory_config(),
            )
            ttnn.deallocate(dirty)
        got = m2.linear(act, weight, **kw)
        got_t = ttnn.to_torch(ttnn.to_memory_config(got, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, got.shape[-1])[:1, :n]
        assert tuple(got.shape) == tuple(ref.shape) and got.memory_config() == ref.memory_config()
        eq = torch.equal(ref_t.view(torch.int16), got_t.view(torch.int16))
        logger.info(f"{op} call {call}: row 0 equal {eq}")
        assert eq, (op, call)
        for t in (act, ref, got):
            ttnn.deallocate(t)
    assert m2.CALLS.get((k, n), 0) - streamed_before >= 2, "the streamed program did not run"
    act = fresh_act(device, act_cfg, k, 9, rows=1)
    ref_t = ttnn.to_torch(ttnn.to_memory_config(ttnn.linear(act, weight, **kw), ttnn.DRAM_MEMORY_CONFIG)).reshape(
        -1, n
    )[:1]
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        out = m2.linear(act, weight, **kw)
    finally:
        ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        got_t = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, out.shape[-1])[:1, :n]
        assert torch.equal(ref_t.view(torch.int16), got_t.view(torch.int16)), "trace replay differs"
    finally:
        ttnn.release_trace(device, trace)
    os.environ.pop("QWEN38_MM_STREAM2", None)
