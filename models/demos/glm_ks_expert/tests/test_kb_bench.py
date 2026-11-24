# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 1: GLM-5.3's gate/up sub-block on one core, operands resident (kernels/kb_*.cpp).

MODE 0 = today's flat expert (one pair over all 192 K tiles); MODE 1 / 2 = K split in pairs (both pairs over this
core's 96 K tiles, the partner's partials loaded into DEST by copy (1) or by an identity matmul (2)). Reports cycles a
tile product (LoFi peak 16) and checks h (and the partials) against torch.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm_ks_expert.bench_util import (
    ACT_ARGS,
    DT,
    GLM,
    TILE_BYTES,
    aiclk_mhz,
    cb,
    crs,
    kpath,
    pcc,
    quantize,
    read_stamps,
    rel_err,
    stamps_tensor,
    to_dev,
)

RT_SB, PR = 4, 2  # 128-row sub-blocks, passes of 2 row tiles (fp32 DEST half-sync: 4 tiles)


def run(device, mode, nsb=100, part="fp32"):
    core = (0, 0)
    grid = crs([core])
    KH = GLM["H"] // 32
    KT = KH if mode == 0 else KH // 2
    WC = 2 if mode == 0 else 4
    part_dt = DT[part] if mode == 1 else ttnn.bfloat16
    torch.manual_seed(0)
    x = torch.randn(RT_SB * 32, KH * 32)
    w = torch.randn(KH * 32, 4 * 32) / (KH * 32) ** 0.5  # (cols: own gate, own up, partner gate, partner up)
    xq, wq = quantize(x, ttnn.bfloat8_b), quantize(w, ttnn.bfloat4_b)
    og, ou, pg, pu = (wq[:, 32 * i : 32 * i + 32] for i in range(4))
    if mode == 0:
        x_dev, w_dev = x, torch.cat([w[:, 0:32], w[:, 32:64]], 1)
        ref_h = torch.nn.functional.silu(xq @ og) * (xq @ ou)
        pin = torch.zeros(32, 32)
        ref_p = None
    else:
        h0, h1 = slice(0, KT * 32), slice(KT * 32, KH * 32)  # this core: K half 0; the partner: half 1
        x_dev = x[:, h0]
        w_dev = torch.cat([w[h0, 64:96], w[h0, 96:128], w[h0, 0:32], w[h0, 32:64]], 1)  # partner pair, own pair
        # the partner's partials for this core's pair (its K half), as the partner would pack them
        pin = torch.cat([xq[:, h1] @ og[h1], xq[:, h1] @ ou[h1]], 1)
        pin_q = quantize(pin, part_dt)
        ref_h = torch.nn.functional.silu(xq[:, h0] @ og[h0] + pin_q[:, :32]) * (xq[:, h0] @ ou[h0] + pin_q[:, 32:])
        ref_p = torch.cat([xq[:, h0] @ pg[h0], xq[:, h0] @ pu[h0]], 1)
    full_h = torch.nn.functional.silu(xq @ og) * (xq @ ou)
    ident = torch.zeros(64, 64)
    ident[:32, :32] = torch.eye(32)
    ident[32:, 32:] = torch.eye(32)

    x_t = to_dev(x_dev, ttnn.bfloat8_b, device)
    w_t = to_dev(w_dev, ttnn.bfloat4_b, device)
    p_t = to_dev(pin, part_dt, device)
    i_t = to_dev(ident, ttnn.bfloat16, device)
    h_t = to_dev(torch.zeros(RT_SB * 32, 32), ttnn.bfloat8_b, device)
    po_dt = part_dt
    po_t = to_dev(torch.zeros(RT_SB * 32, 64), po_dt, device)
    st = stamps_tensor(device, [core])
    n = [RT_SB * KT, KT * WC, 0 if mode == 0 else RT_SB * 2, 4 if mode == 2 else 0]

    rd_rt = ttnn.RuntimeArgs()
    rd_rt[core[0]][core[1]] = [x_t.buffer_address(), w_t.buffer_address(), p_t.buffer_address(), i_t.buffer_address()] + n
    wr_rt = ttnn.RuntimeArgs()
    wr_rt[core[0]][core[1]] = [h_t.buffer_address(), po_t.buffer_address(), nsb, RT_SB, 0 if mode == 0 else RT_SB * 2]
    cp_rt = ttnn.RuntimeArgs()
    cp_rt[core[0]][core[1]] = [st.buffer_address()]
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rd = ttnn.KernelDescriptor(
        kernel_source=kpath("kb_reader.cpp"),
        core_ranges=grid,
        compile_time_args=ta(x_t) + ta(w_t) + ta(p_t) + ta(i_t),
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    wr = ttnn.KernelDescriptor(
        kernel_source=kpath("kb_writer.cpp"),
        core_ranges=grid,
        compile_time_args=ta(h_t) + ta(po_t),
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=False)
    if mode == 1 and part == "fp32":
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    cp = ttnn.KernelDescriptor(
        kernel_source=kpath("kb_compute.cpp"),
        core_ranges=grid,
        compile_time_args=[mode, KT, RT_SB, PR, nsb, WC],
        named_compile_time_args=ACT_ARGS,
        runtime_args=cp_rt,
        config=cfg,
    )
    cbs = [
        cb(0, ttnn.bfloat8_b, RT_SB * KT, grid),
        cb(1, ttnn.bfloat4_b, KT * WC, grid),
        cb(2, part_dt, max(1, n[2]), grid),
        cb(3, ttnn.bfloat16, 4, grid),
        cb(16, ttnn.bfloat8_b, 2 * RT_SB, grid),
        cb(17, po_dt, 2 * RT_SB * 2, grid),
    ]
    l1 = sum(c.total_size for c in cbs)
    assert l1 < 1400 * 1024, l1
    ttnn.generic_op([x_t, w_t, p_t, i_t, h_t, po_t, st], ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    s = read_stamps(st, 1)[0]
    cyc = int((s[5] - s[0]) & 0xFFFFFFFF)
    products = nsb * RT_SB * 2 * (2 * KT if mode else KT)
    h = ttnn.to_torch(h_t).float()
    out = dict(
        cpt=cyc / products,
        cyc_sb=cyc / nsb,
        pcc_h=pcc(h, ref_h),
        err_h=rel_err(h, ref_h),
        err_full=rel_err(h, full_h),
        aiclk=aiclk_mhz(0),
    )
    if ref_p is not None:
        out["err_p"] = rel_err(ttnn.to_torch(po_t).float(), ref_p)
    return out


@pytest.mark.parametrize("mode, part", [(0, "fp32"), (1, "fp32"), (1, "bf16"), (2, "bf16")], ids=["today", "ks_copy_fp32", "ks_copy_bf16", "ks_ident"])
def test_kb_bench(device, mode, part):
    r = run(device, mode, part=part)
    logger.info(
        f"KBENCH mode {mode} partials {part}: {r['cpt']:.2f} cyc / tile product ({1600 / r['cpt']:.1f}% LoFi), "
        f"{r['cyc_sb']:.0f} cyc a 128-row sub-block ({r['cyc_sb'] / 1350:.2f} us at 1.35 GHz); h pcc {r['pcc_h']:.6f} "
        f"err {r['err_h']:.2e} (vs unsplit fp32 {r['err_full']:.2e}) partials err {r.get('err_p', 0):.2e}; aiclk {r['aiclk']}"
    )
    assert r["pcc_h"] > 0.999
