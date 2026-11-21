# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""custom_ops/qwen_cp_sdpa against the stock chunked SDPA calls of the context-parallel prefill (qwen_cp_prefill.py).

test_chunked_sdpa_port: chunked_sdpa_port (Python port of the stock SDPA program factory, stock or local kernel
    files) against ttnn.transformer.chunked_scaled_dot_product_attention, both called as chunked_sdpa_pair calls them:
    Q one [1, 8, S/8, 128] buffer per owned slice, K / V scratch-shaped [S/64, 1, 64, 128] through the identity page
    table, a per-device chunk_start_idx_tensor. torch.equal on all 32 devices, both slices, plus a second call on fresh
    buffers (descriptor cache hit). Prints the host-wall median of the stock call and the port.

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_cp_sdpa_op.py -s
"""

import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill as cp
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_cp_sdpa import chunked_sdpa_port

N_LOCAL_HEADS = 8
HEAD_DIM = 128
ROWS, COLS = 8, 4
DEVICE_PARAMS = [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000}]


def _replicated(mesh_device, t, dtype=ttnn.bfloat8_b):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _model_sdpa_config(mesh_device, seq_len, lay):
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

    margs = TtQwenModelArgs(mesh_device, max_batch_size=32, max_seq_len=seq_len)
    mc = margs.get_model_config()
    base = margs.compute_kernel_config_ring_sdpa_long
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=base.math_fidelity,
        math_approx_mode=base.math_approx_mode,
        fp32_dest_acc_en=False,
        packer_l1_acc=base.packer_l1_acc,
    )
    ring_pc = mc.get("SDPA_RING_PROGCFG", mc["SDPA_PROGCFG"])(seq_len, chunk_start_idx=0)
    return ckc, cp.chunked_sdpa_config(ring_pc, lay.sl)


def _host(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _median_ms(mesh_device, fn, iters=5):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(mesh_device)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    return statistics.median(ts)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("kernels", ["stock", "local"])
@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_chunked_sdpa_port(mesh_device, seq_len, kernels):
    ctx = cp.CPContext(mesh_device, seq_len)
    lay = ctx.lay
    ckc, pc = _model_sdpa_config(mesh_device, seq_len, lay)
    scale = float(torch.tensor(HEAD_DIM**-0.5, dtype=torch.float32))
    logger.info(f"QCS S={seq_len} q{pc.q_chunk_size} k{pc.k_chunk_size} grid {pc.compute_with_storage_grid_size}")
    torch.manual_seed(seq_len)
    kv = [_replicated(mesh_device, torch.randn(seq_len // cp.KV_BLOCK, 1, cp.KV_BLOCK, HEAD_DIM)) for _ in range(2)]
    pt = ctx.identity_page_table()
    cs = [ctx.per_device_i32(lambda r, c, j=j: [lay.slices(c)[j] * lay.sl]) for j in range(2)]
    bad = []
    for rep in range(2):  # rep 1: fresh Q buffers (descriptor cache hit on new addresses for q / out)
        qs = [_replicated(mesh_device, torch.randn(1, N_LOCAL_HEADS, lay.sl, HEAD_DIM)) for _ in range(2)]
        for j in range(2):
            ref = ttnn.transformer.chunked_scaled_dot_product_attention(
                qs[j],
                kv[0],
                kv[1],
                pt,
                chunk_start_idx_tensor=cs[j],
                scale=scale,
                compute_kernel_config=ckc,
                program_config=pc,
            )
            new = chunked_sdpa_port(
                qs[j],
                kv[0],
                kv[1],
                pt,
                cs[j],
                scale=scale,
                compute_kernel_config=ckc,
                program_config=pc,
                kernels=kernels,
            )
            rd, nd = _host(ref), _host(new)
            eq = sum(int(torch.equal(a, b)) for a, b in zip(rd, nd))
            diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(rd, nd))
            logger.info(f"QCS S={seq_len} kernels={kernels} rep {rep} call {j}: equal {eq}/32, max abs diff {diff}")
            if eq != ROWS * COLS:
                bad.append(f"rep {rep} call {j}: equal {eq}/32 (max abs {diff})")
            ttnn.deallocate(ref)
            ttnn.deallocate(new)
    for j in range(2):
        t_ref = _median_ms(
            mesh_device,
            lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                qs[j],
                kv[0],
                kv[1],
                pt,
                chunk_start_idx_tensor=cs[j],
                scale=scale,
                compute_kernel_config=ckc,
                program_config=pc,
            ),
        )
        t_new = _median_ms(
            mesh_device,
            lambda: chunked_sdpa_port(
                qs[j],
                kv[0],
                kv[1],
                pt,
                cs[j],
                scale=scale,
                compute_kernel_config=ckc,
                program_config=pc,
                kernels=kernels,
            ),
        )
        logger.info(f"QCS_TIME S={seq_len} kernels={kernels} call {j}: stock {t_ref:.3f} ms, port {t_new:.3f} ms")
    assert not bad, bad


def _stock_pair(ctx, qs, kv, pt, scale, ckc, pc, concat_heads):
    """keep-34 attention core: cp.chunked_sdpa_pair (two stock chunked calls + concat), then nlp_concat_heads."""
    out = cp.chunked_sdpa_pair(ctx, qs, kv[0], kv[1], pt, scale, ckc, pc)
    if not concat_heads:
        return out
    cat = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)
    return cat


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [16384, 24576, 32768, 40960])
def test_qwen_cp_sdpa(mesh_device, seq_len):
    """qwen_cp_sdpa (both slices of a column in one program, 70 cores) against the stock pair it replaces: torch.equal on
    all 32 devices for the concat-heads output (vs pair + nlp_concat_heads) and the [1, H, S/4, D] output (vs the
    pair), plus a second call on fresh Q buffers. Prints host-wall medians (10 calls) of both."""
    from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_cp_sdpa import qwen_cp_sdpa

    ctx = cp.CPContext(mesh_device, seq_len)
    lay = ctx.lay
    ckc, pc = _model_sdpa_config(mesh_device, seq_len, lay)
    scale = float(torch.tensor(HEAD_DIM**-0.5, dtype=torch.float32))
    torch.manual_seed(seq_len + 1)
    kv = [_replicated(mesh_device, torch.randn(seq_len // cp.KV_BLOCK, 1, cp.KV_BLOCK, HEAD_DIM)) for _ in range(2)]
    pt = ctx.identity_page_table()
    bad = []
    for rep in range(2):
        qs = [_replicated(mesh_device, torch.randn(1, N_LOCAL_HEADS, lay.sl, HEAD_DIM)) for _ in range(2)]
        for concat_heads in (True, False):
            ref = _stock_pair(ctx, qs, kv, pt, scale, ckc, pc, concat_heads)
            new = qwen_cp_sdpa(
                qs[0],
                qs[1],
                kv[0],
                kv[1],
                pt,
                scale=scale,
                compute_kernel_config=ckc,
                program_config=pc,
                concat_heads=concat_heads,
            )
            rd, nd = _host(ref), _host(new)
            eq = sum(int(torch.equal(a, b)) for a, b in zip(rd, nd))
            diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(rd, nd))
            logger.info(
                f"QCP S={seq_len} q{pc.q_chunk_size} k{pc.k_chunk_size} rep {rep} concat_heads={concat_heads}: "
                f"shape {tuple(new.shape)} equal {eq}/32, max abs diff {diff}"
            )
            if eq != ROWS * COLS:
                bad.append(f"rep {rep} concat_heads={concat_heads}: equal {eq}/32 (max abs {diff})")
            ttnn.deallocate(ref)
            ttnn.deallocate(new)
    t_ref = _median_ms(mesh_device, lambda: _stock_pair(ctx, qs, kv, pt, scale, ckc, pc, True), iters=10)
    t_new = _median_ms(
        mesh_device,
        lambda: qwen_cp_sdpa(qs[0], qs[1], kv[0], kv[1], pt, scale=scale, compute_kernel_config=ckc, program_config=pc),
        iters=10,
    )
    t_pair = _median_ms(mesh_device, lambda: _stock_pair(ctx, qs, kv, pt, scale, ckc, pc, False), iters=10)
    logger.info(
        f"QCP_TIME S={seq_len}: stock pair {t_pair:.3f} ms, pair + nlp_concat_heads {t_ref:.3f} ms, "
        f"qwen_cp_sdpa (concat heads) {t_new:.3f} ms"
    )
    assert not bad, bad


def _pipelined_ms(mesh_device, fn, iters=20):
    """Mean ms per call of iters back-to-back calls with one sync at the end (host dispatch overlaps the device, as
    in the model's prefill)."""
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1e3 / iters


@torch.no_grad()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [40960])
def test_qwen_cp_sdpa_time(mesh_device, seq_len):
    """Pipelined timing (20 back-to-back calls, one sync): the stock chunked SDPA pair (+ nlp_concat_heads) that the
    model ran before qwen_cp_sdpa, against qwen_cp_sdpa, and one stock chunked call against its port with the stock /
    local kernel files. Fails if qwen_cp_sdpa is slower than the stock pair + nlp_concat_heads it replaces."""
    from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_cp_sdpa import qwen_cp_sdpa

    ctx = cp.CPContext(mesh_device, seq_len)
    lay = ctx.lay
    ckc, pc = _model_sdpa_config(mesh_device, seq_len, lay)
    scale = float(torch.tensor(HEAD_DIM**-0.5, dtype=torch.float32))
    torch.manual_seed(seq_len + 2)
    kv = [_replicated(mesh_device, torch.randn(seq_len // cp.KV_BLOCK, 1, cp.KV_BLOCK, HEAD_DIM)) for _ in range(2)]
    pt = ctx.identity_page_table()
    qs = [_replicated(mesh_device, torch.randn(1, N_LOCAL_HEADS, lay.sl, HEAD_DIM)) for _ in range(2)]
    cs = ctx.per_device_i32(lambda r, c: [lay.slices(c)[1] * lay.sl])
    res = {
        "pair": _pipelined_ms(mesh_device, lambda: _stock_pair(ctx, qs, kv, pt, scale, ckc, pc, False)),
        "pair+nlp_concat_heads": _pipelined_ms(mesh_device, lambda: _stock_pair(ctx, qs, kv, pt, scale, ckc, pc, True)),
        "qwen_cp_sdpa": _pipelined_ms(
            mesh_device,
            lambda: qwen_cp_sdpa(
                qs[0], qs[1], kv[0], kv[1], pt, scale=scale, compute_kernel_config=ckc, program_config=pc
            ),
        ),
        "call1 stock": _pipelined_ms(
            mesh_device,
            lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                qs[1],
                kv[0],
                kv[1],
                pt,
                chunk_start_idx_tensor=cs,
                scale=scale,
                compute_kernel_config=ckc,
                program_config=pc,
            ),
        ),
    }
    for kern in ("stock", "local"):
        res[f"call1 port/{kern}"] = _pipelined_ms(
            mesh_device,
            lambda: chunked_sdpa_port(
                qs[1], kv[0], kv[1], pt, cs, scale=scale, compute_kernel_config=ckc, program_config=pc, kernels=kern
            ),
        )
    for name, ms in res.items():
        logger.info(f"QCP_PIPE S={seq_len} {name}: {ms:.3f} ms")
    # 10% allowance for run-to-run timing noise on a shared host; the op replaces the pair + concat in the model.
    assert res["qwen_cp_sdpa"] <= 1.10 * res["pair+nlp_concat_heads"], (
        f"qwen_cp_sdpa {res['qwen_cp_sdpa']:.3f} ms is slower than the stock pair + nlp_concat_heads "
        f"{res['pair+nlp_concat_heads']:.3f} ms"
    )
