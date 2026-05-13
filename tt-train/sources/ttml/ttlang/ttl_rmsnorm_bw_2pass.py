# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
RMSNorm backward — 2-pass tile streaming (vs. ``remsnorm_whole_row.py`` whole-row strip).

**Pass 1:** per tile-column, accumulate row scalar
``scale = Σ_c (x · (γ/rms) · dL_dout)`` via ``(contrib @ ones)`` matmul reduction
(same workaround as ``polynorm3_on_tiles_2pass_working.py``).

**Pass 2:** re-stream tiles and emit
``dL_dinput = (γ/rms)·dL_dout − scale·x·(1/rms)²·(1/C)``,
``dL_dgamma = x·(1/rms)·dL_dout``.
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE_WIDTH = 32


def get_block_size(whole_row_width: int = 32, max_block_size: int = 4) -> int:
    for block_size in range(max_block_size, 1, -1):
        if whole_row_width % block_size == 0:
            return block_size
    return 1


def make_kernel():
    bc = 2

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def rmsnorm_bw_2pass(
        input_t,
        gamma_t,
        rms_t,
        dL_dout_t,
        dL_dinput_out,
        dL_dgamma_comp_out,
    ):
        ht = input_t.shape[0] // TILE_WIDTH
        wt = input_t.shape[1] // TILE_WIDTH
        one = (1, 1)
        elem_c = wt * TILE_WIDTH
        inv_c_scalar = 1.0 / elem_c

        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_cols * grid_rows
        rows_per_core = -(-ht // total_cores)

        inp_tile = ttl.make_dataflow_buffer_like(input_t, shape=one, block_count=bc)
        gamma_tile = ttl.make_dataflow_buffer_like(gamma_t, shape=one, block_count=bc)
        rms_tile = ttl.make_dataflow_buffer_like(rms_t, shape=one, block_count=bc)
        dL_tile = ttl.make_dataflow_buffer_like(dL_dout_t, shape=one, block_count=bc)
        out_da_tile = ttl.make_dataflow_buffer_like(dL_dinput_out, shape=one, block_count=bc)
        out_dg_tile = ttl.make_dataflow_buffer_like(dL_dgamma_comp_out, shape=one, block_count=bc)

        scale_acc = ttl.make_dataflow_buffer_like(input_t, shape=one, block_count=bc)
        inv_c_dfb = ttl.make_dataflow_buffer_like(input_t, shape=one, block_count=1)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            with inv_c_dfb.reserve() as ic:
                ic.store(ttl.math.fill(ic, inv_c_scalar))
            inv_c = inv_c_dfb.wait()

            for local_row in range(rows_per_core):
                row = core_linear * rows_per_core + local_row
                if row < ht:
                    with scale_acc.reserve() as z:
                        z.store(ttl.math.fill(z, 0.0))

                    for _local_col in range(wt):
                        with (
                            inp_tile.wait() as iv,
                            gamma_tile.wait() as gv,
                            rms_tile.wait() as rmv,
                            dL_tile.wait() as dlv,
                            scale_acc.wait() as s,
                        ):
                            recip = ttl.math.recip(rmv)
                            gained = recip * gv * dlv
                            contrib = iv * gained
                            reduce_tile = ttl.math.fill(contrib, 1.0)
                            with scale_acc.reserve() as o:
                                o.store(s + (contrib @ reduce_tile))

                    scale = scale_acc.wait()

                    for _local_col in range(wt):
                        with (
                            inp_tile.wait() as iv,
                            gamma_tile.wait() as gv,
                            rms_tile.wait() as rmv,
                            dL_tile.wait() as dlv,
                        ):
                            recip = ttl.math.recip(rmv)
                            with out_da_tile.reserve() as oa:
                                oa.store(recip * gv * dlv - scale * iv * recip * recip * inv_c)
                            with out_dg_tile.reserve() as og:
                                og.store(iv * recip * dlv)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            for local_row in range(rows_per_core):
                r = core_linear * rows_per_core + local_row
                if r < ht:
                    for local_col in range(wt):
                        with inp_tile.reserve() as b:
                            ttl.copy(input_t[r, local_col], b).wait()
                        with gamma_tile.reserve() as b:
                            ttl.copy(gamma_t[r, local_col], b).wait()
                        with rms_tile.reserve() as b:
                            ttl.copy(rms_t[r, local_col], b).wait()
                        with dL_tile.reserve() as b:
                            ttl.copy(dL_dout_t[r, local_col], b).wait()
                    for local_col in range(wt):
                        with inp_tile.reserve() as b:
                            ttl.copy(input_t[r, local_col], b).wait()
                        with gamma_tile.reserve() as b:
                            ttl.copy(gamma_t[r, local_col], b).wait()
                        with rms_tile.reserve() as b:
                            ttl.copy(rms_t[r, local_col], b).wait()
                        with dL_tile.reserve() as b:
                            ttl.copy(dL_dout_t[r, local_col], b).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            for local_row in range(rows_per_core):
                r = core_linear * rows_per_core + local_row
                if r < ht:
                    for local_col in range(wt):
                        with out_da_tile.wait() as b:
                            ttl.copy(b, dL_dinput_out[r, local_col]).wait()
                        with out_dg_tile.wait() as b:
                            ttl.copy(b, dL_dgamma_comp_out[r, local_col]).wait()

    return rmsnorm_bw_2pass


def _torch_ref(x, gamma_1d, eps, dL):
    x = x.float().clone().requires_grad_(True)
    g = gamma_1d.float().clone().requires_grad_(True)
    dL = dL.float()
    var = x.pow(2).mean(-1, keepdim=True)
    rms = (var + eps).sqrt()
    y = x / rms * g
    y.backward(dL)
    return x.grad.to(torch.bfloat16), g.grad.to(torch.bfloat16), rms.detach().to(torch.bfloat16)


def _to_dev(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def rmsnorm_2pass_smoke_test(device, height: int, width: int) -> None:
    assert height % TILE_WIDTH == 0 and width % TILE_WIDTH == 0, (height, width)
    eps = 1e-5 if width // TILE_WIDTH > 1 else 0.0078125
    torch.manual_seed(0)
    x = torch.randn(height, width, dtype=torch.bfloat16)
    g1d = torch.randn(width, dtype=torch.bfloat16)
    g2d = g1d.unsqueeze(0).expand(height, width).contiguous()
    with torch.no_grad():
        var = x.float().pow(2).mean(-1, keepdim=True)
        rms = (var + eps).sqrt()
        y = x.float() / rms * g2d.float()
    dL = ((2.0 / y.numel()) * y).to(torch.bfloat16)

    dL_dx_ref, dL_dg_ref, _ = _torch_ref(x, g1d, eps, dL)
    rms_2d = rms.to(torch.bfloat16).expand(-1, width).contiguous()

    k = make_kernel()
    out_da = _to_dev(torch.zeros(height, width, dtype=torch.bfloat16), device)
    out_dg = _to_dev(torch.zeros(height, width, dtype=torch.bfloat16), device)
    k(
        _to_dev(x, device),
        _to_dev(g2d, device),
        _to_dev(rms_2d, device),
        _to_dev(dL, device),
        out_da,
        out_dg,
    )
    ttnn.synchronize_device(device)

    da = ttnn.to_torch(out_da)
    dg = ttnn.to_torch(out_dg)
    torch.testing.assert_close(da, dL_dx_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dg.sum(dim=0), dL_dg_ref, rtol=5e-2, atol=5e-2)
    print(f"OK: rmsnorm_2pass {height}x{width}")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        rmsnorm_2pass_smoke_test(device, 256, 384)
        rmsnorm_2pass_smoke_test(device, 2048, 2048)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
