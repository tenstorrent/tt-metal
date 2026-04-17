# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn
import ttl

# Metal tile layout: one tile spans TILE_WIDTH elements along each logical axis in TILE_LAYOUT.
TILE_WIDTH = 32

# Dataflow buffer block pool depth (matches typical Metal CB double-buffering).
DFB_BLOCK_COUNT = 2

# Torch reference: epsilon for RMS denominator (wide vs single-tile-W uses different stability choice).
EPS_RMS_WIDE = 1e-5
EPS_RMS_SINGLE_TILE_W = 0.0078125  # 1 / 128; used when elem_c == TILE_WIDTH in _run_case

# Reproducibility and device defaults for the self-test harness.
TORCH_MANUAL_SEED = 0
DEFAULT_DEVICE_ID = 0

# bf16 TTL vs torch reference comparison (accumulation / kernel numerics).
ASSERT_CLOSE_RTOL = 0.12
ASSERT_CLOSE_ATOL = 0.12

def get_block_size(num_inner: int, max_block_size: int = 4) -> int:
    """
    Same logic as tt-train/.../metal/common/program_utils.hpp get_block_size:
    """
    for block_size in range(max_block_size, 1, -1):
        if num_inner % block_size == 0:
            return block_size
    return 1


def make_rmsnorm_bw_device_kernels_ttl():
    @ttl.operation(grid=(1, 1))
    def rmsnorm_bw_device_kernels_ttl(
        input_t,
        gamma_t,
        rms_t,
        dL_dout_t,
        dL_dinput_out,
        dL_dgamma_comp_out,
    ):
        ht = input_t.shape[0] // TILE_WIDTH
        wt = input_t.shape[1] // TILE_WIDTH
        block_size = get_block_size(wt, 2)
        bs = block_size
        # DFB tile shape for one reader/writer "push": one tile-row × block_size column tiles
        tr, tc = 1, bs
        # Inner width in elements for one full tile-row strip
        elem_c = wt * TILE_WIDTH
        # 1/C for RMS mean denominator;
        inv_c_scalar = 1.0 / elem_c
        num_iters = ht * (wt // bs)

        inp_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        gamma_dfb = ttl.make_dataflow_buffer_like(gamma_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        rms_dfb = ttl.make_dataflow_buffer_like(rms_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        dL_dfb = ttl.make_dataflow_buffer_like(dL_dout_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)

        # Full tile-row strips for scale = sum_c (a * gained)
        wide_row_shape = (1, wt)
        wide_inp = ttl.make_dataflow_buffer_like(input_t, shape=wide_row_shape, block_count=DFB_BLOCK_COUNT)
        wide_gam = ttl.make_dataflow_buffer_like(gamma_t, shape=wide_row_shape, block_count=DFB_BLOCK_COUNT)
        wide_rms = ttl.make_dataflow_buffer_like(rms_t, shape=wide_row_shape, block_count=DFB_BLOCK_COUNT)
        wide_dL = ttl.make_dataflow_buffer_like(dL_dout_t, shape=wide_row_shape, block_count=DFB_BLOCK_COUNT)

        # --- Scale path (compute_scale): reduce_sum wants a scaler tile; contrib must be materialized first. ---
        # Unity scaler for ttl.math.reduce_sum
        single_tile_shape = (1, 1)
        scaler_dfb = ttl.make_dataflow_buffer_like(input_t, shape=single_tile_shape, block_count=DFB_BLOCK_COUNT)
        # a * gained over the full (1, Wt) strip
        contrib_wide_dfb = ttl.make_dataflow_buffer_like(input_t, shape=wide_row_shape, block_count=DFB_BLOCK_COUNT)
        # Reduced scale along horizontal tiles (dim C)
        scale_red_dfb = ttl.make_dataflow_buffer_like(input_t, shape=single_tile_shape, block_count=DFB_BLOCK_COUNT)
        # Broadcast scale to current block for mul_bcast_cols-style use in dL_da rhs
        scale_bc_blk_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        # Constant 1/C tile for rhs
        inv_c_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)

        # --- dL_da / dL_dgamma intermediates  ---
        # gained_dL_dout staging;
        gained_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        # (scale * a * recip^2 * 1/c) before subtract from gained
        rhs_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        # a * recip before * dL_out for dL_dgamma_component
        norm_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        out_da_dfb = ttl.make_dataflow_buffer_like(dL_dinput_out, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)
        out_dg_dfb = ttl.make_dataflow_buffer_like(dL_dgamma_comp_out, shape=(tr, tc), block_count=DFB_BLOCK_COUNT)

        @ttl.compute()
        def compute():
            for _ in range(num_iters):
                # Scaler tile for reduce_sum
                with scaler_dfb.reserve() as sc:
                    sc.store(ttl.math.fill(sc, 1.0))

                with (
                    wide_inp.wait() as wix,
                    wide_gam.wait() as wgam,
                    wide_rms.wait() as wrms,
                    wide_dL.wait() as wdl,
                ):
                    recip_w = ttl.math.recip(wrms)
                    gained_w = recip_w * wgam * wdl
                    with contrib_wide_dfb.reserve() as cw:
                        cw.store(wix * gained_w)

                # Computes scale factor: scale = sum(a * gained_dL_dout, dim=C)
                # where gained_dL_dout = (gamma / rms_a) * dL_dout.
                # Result is reduced along the inner (C / horizontal tile) dimension, then broadcast for dL_da.
                # contrib must live in a DFB before reduce (ttlang: elementwise into reduce is not fused yet).
                with contrib_wide_dfb.wait() as cwv, scaler_dfb.wait() as scw:
                    with scale_red_dfb.reserve() as sr:
                        sr.store(ttl.math.reduce_sum(cwv, scw, dims=[1]))
                with scale_red_dfb.wait() as srt, scale_bc_blk_dfb.reserve() as sbb:
                    sbb.store(ttl.math.broadcast(srt, sbb, dims=[1]))

                with (
                    inp_dfb.wait() as iv,
                    gamma_dfb.wait() as gv,
                    rms_dfb.wait() as rmv,
                    dL_dfb.wait() as dlv,
                    scale_bc_blk_dfb.wait() as sbv,
                ):
                    with inv_c_dfb.reserve() as ic:
                        ic.store(ttl.math.fill(ic, inv_c_scalar))
                    with inv_c_dfb.wait() as icv:
                        # Per-tile 1/rms for this block (Metal already has recip in CB for scale path).
                        recip_b = ttl.math.recip(rmv)
                        # compute_gained_dL_dout: gained_dL_dout = (gamma * 1/rms) * dL_out
                        with gained_dfb.reserve() as gd:
                            gd.store(recip_b * gv * dlv)
                        with gained_dfb.wait() as gdv:
                            # compute_dL_da rhs: (scale * a) / (c * rms^2) == scale * a * recip^2 * (1/c)
                            with rhs_dfb.reserve() as rh:
                                rh.store(sbv * iv * recip_b * recip_b * icv)
                            with rhs_dfb.wait() as rhv, out_da_dfb.reserve() as oa:
                                # dL_da = gained_dL_dout - rhs
                                oa.store(gdv - rhv)
                        # compute_dL_dgamma_components: normalized_a = a * recip, then * dL_out
                        with norm_dfb.reserve() as nd:
                            nd.store(iv * recip_b)
                        with norm_dfb.wait() as ndv, out_dg_dfb.reserve() as og:
                            # dL_dgamma_components = normalized_a * dL_out (reduction to dL_dgamma is outside the kernel)
                            og.store(ndv * dlv)

        @ttl.datamovement()
        def dm_read():
            for r in range(ht):
                for c in range(0, wt, bs):
                    with wide_inp.reserve() as blk:
                        tx = ttl.copy(input_t[r : r + 1, 0:wt], blk)
                        tx.wait()
                    with wide_gam.reserve() as blk:
                        tx = ttl.copy(gamma_t[r : r + 1, 0:wt], blk)
                        tx.wait()
                    with wide_rms.reserve() as blk:
                        tx = ttl.copy(rms_t[r : r + 1, 0:wt], blk)
                        tx.wait()
                    with wide_dL.reserve() as blk:
                        tx = ttl.copy(dL_dout_t[r : r + 1, 0:wt], blk)
                        tx.wait()
                    with inp_dfb.reserve() as blk:
                        tx = ttl.copy(input_t[r : r + 1, c : c + bs], blk)
                        tx.wait()
                    with gamma_dfb.reserve() as blk:
                        tx = ttl.copy(gamma_t[r : r + 1, c : c + bs], blk)
                        tx.wait()
                    with rms_dfb.reserve() as blk:
                        tx = ttl.copy(rms_t[r : r + 1, c : c + bs], blk)
                        tx.wait()
                    with dL_dfb.reserve() as blk:
                        tx = ttl.copy(dL_dout_t[r : r + 1, c : c + bs], blk)
                        tx.wait()

        @ttl.datamovement()
        def dm_write():
            for r in range(ht):
                for c in range(0, wt, bs):
                    with out_da_dfb.wait() as blk:
                        tx = ttl.copy(blk, dL_dinput_out[r : r + 1, c : c + bs])
                        tx.wait()
                    with out_dg_dfb.wait() as blk:
                        tx = ttl.copy(blk, dL_dgamma_comp_out[r : r + 1, c : c + bs])
                        tx.wait()

    return rmsnorm_bw_device_kernels_ttl


def torch_rmsnorm_bw_reference(x_bf16, gamma_bf16, eps, dL_dy_bf16):
    x = x_bf16.float().clone().requires_grad_(True)
    gamma = gamma_bf16.float().clone().requires_grad_(True)
    dL_dy = dL_dy_bf16.float()
    var = x.pow(2).mean(-1, keepdim=True)
    rms = (var + eps).sqrt()
    y = x / rms * gamma
    y.backward(dL_dy)
    return x.grad.to(torch.bfloat16), gamma.grad.to(torch.bfloat16), rms.detach().to(torch.bfloat16)


def _mse_upstream_grad_y(y_bf16: torch.Tensor) -> torch.Tensor:
    y = y_bf16.float()
    n = y.numel()
    return ((2.0 / n) * y).to(torch.bfloat16)


def from_torch(torch_2d: torch.Tensor, device):
    return ttnn.from_torch(
        torch_2d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_case(device, elem_r: int, elem_c: int, label: str) -> None:
    wt = elem_c // TILE_WIDTH
    eps = EPS_RMS_WIDE if wt > 1 else EPS_RMS_SINGLE_TILE_W
    torch.manual_seed(TORCH_MANUAL_SEED)
    x = torch.randn(elem_r, elem_c, dtype=torch.bfloat16)
    gamma_1d = torch.randn(elem_c, dtype=torch.bfloat16)
    gamma_2d = gamma_1d.unsqueeze(0).expand(elem_r, elem_c).contiguous()
    with torch.no_grad():
        var = x.float().pow(2).mean(-1, keepdim=True)
        rms = (var + eps).sqrt()
        y = x.float() / rms * gamma_2d.float()
    dL = _mse_upstream_grad_y(y.to(torch.bfloat16))

    dL_dx_exp, dL_dgamma_exp, _rms = torch_rmsnorm_bw_reference(x, gamma_1d, eps, dL)
    rms_2d = rms.to(torch.bfloat16).expand(-1, elem_c).contiguous()

    k = make_rmsnorm_bw_device_kernels_ttl()
    out_da = from_torch(torch.zeros_like(x), device)
    out_dg = from_torch(torch.zeros_like(x), device)
    k(
        from_torch(x, device),
        from_torch(gamma_2d, device),
        from_torch(rms_2d, device),
        from_torch(dL, device),
        out_da,
        out_dg,
    )

    da = ttnn.to_torch(out_da)
    dg = ttnn.to_torch(out_dg)
    torch.testing.assert_close(da, dL_dx_exp, rtol=ASSERT_CLOSE_RTOL, atol=ASSERT_CLOSE_ATOL)
    torch.testing.assert_close(dg.sum(dim=0), dL_dgamma_exp, rtol=ASSERT_CLOSE_RTOL, atol=ASSERT_CLOSE_ATOL)
    print(f"OK: {label}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=DEFAULT_DEVICE_ID)
    try:
        _run_case(device, 32, 64, "device-kernels TTL [32,64]")
        _run_case(device, 32, 32, "device-kernels TTL [32,32]")
        _run_case(device, 64, 32, "device-kernels TTL [64,32]")
        _run_case(device, 32, 128, "device-kernels TTL [32,128]")
        _run_case(device, 128, 64, "device-kernels TTL [128,64]")
        _run_case(device, 128, 128, "device-kernels TTL [128,128]")
        _run_case(device, 256, 96, "device-kernels TTL [256,96]")
        _run_case(device, 256, 128, "device-kernels TTL [256,128]")
        print("rmsnorm_bw_device_kernels_ttl: all checks passed.")
    finally:
        ttnn.close_device(device)
