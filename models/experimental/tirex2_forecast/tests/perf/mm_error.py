# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-2f mechanism check: the error of each dense role's matmul alone against an fp32 host reference.

    python -m pytest .../tests/perf/mm_error.py -s        (under ttq --kind bench)

Per role (the ``MM_BLOCKS`` shapes at M = 8192 rows): ``ttnn.linear`` default; ``minimal_matmul`` with the swept
blocks; the same with K in one block (no partial-sum spill); and with an fp32 output (partials never rounded to
bf16). Relative max / RMS error vs ``x.float() @ w.float()`` on the bf16-rounded inputs.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mm_error(device):
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import kernel_config
    from models.experimental.tirex2_forecast.tt.layers import MM_BLOCKS

    cfg = kernel_config(ttnn.MathFidelity.HiFi4)
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    for (K, N), (mb, kb, nb, sh, sw) in MM_BLOCKS.items():
        x = torch.randn(1, 1, 8192, K).bfloat16()
        w = (torch.randn(K, N) / K**0.5).bfloat16()
        ref = x.double() @ w.double()
        tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        tw = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device)

        def mm(kblock, dtype):
            c = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kblock,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            )
            return ttnn.experimental.minimal_matmul(tx, tw, config=c, compute_kernel_config=cfg, dtype=dtype)

        kt = K // 32
        kb = max(d for d in range(1, min(kb, kt) + 1) if kt % d == 0)
        variants = {
            "linear default": lambda: ttnn.linear(tx, tw, compute_kernel_config=cfg),
            f"minimal K{kb}": lambda: mm(kb, ttnn.bfloat16),
            f"minimal K{kt} (one block)": lambda: mm(kt, ttnn.bfloat16),
            f"minimal K{kb} fp32 out": lambda: mm(kb, ttnn.float32),
        }
        for name, fn in variants.items():
            try:
                got = ttnn.to_torch(fn()).double()
            except Exception as e:  # noqa: BLE001
                print(f"[MMERR] K{K} N{N} {name}: error {str(e).splitlines()[0][:80]}")
                continue
            d = got - ref
            print(
                f"[MMERR] K{K} N{N} {name}: max rel {float(d.abs().max() / ref.abs().max()):.3e}  rms rel {float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()):.3e}"
            )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mm_epilogue_error(device):
    """The fused epilogues alone: ``W_in`` with its fp32 bias and fp32 output, and bias + SiLU (embed / head)."""
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import kernel_config
    from models.experimental.tirex2_forecast.tt.layers import MM_BLOCKS

    cfg = kernel_config(ttnn.MathFidelity.HiFi4)
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    cases = [((512, 2048), ttnn.float32, None), ((64, 512), ttnn.bfloat16, "silu"), ((512, 512), ttnn.bfloat16, None)]
    for (K, N), odt, act in cases:
        mb, kb, nb, sh, sw = MM_BLOCKS[(K, N)]
        kb = min(kb, K // 32)
        x = torch.randn(1, 1, 8192, K).bfloat16()
        w = (torch.randn(K, N) / K**0.5).bfloat16()
        b = torch.randn(1, 1, 1, N) * 3.0
        bdt = ttnn.float32 if odt == ttnn.float32 else ttnn.bfloat16
        bh = b.float() if bdt == ttnn.float32 else b.bfloat16().float()
        ref = x.double() @ w.double() + bh.double()
        if act:
            ref = torch.nn.functional.silu(ref)
        tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        tw = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device)
        tb = ttnn.from_torch(b, dtype=bdt, layout=ttnn.TILE_LAYOUT, device=device)
        c = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        )
        variants = {
            "linear": lambda: ttnn.linear(tx, tw, bias=tb, activation=act, dtype=odt, compute_kernel_config=cfg),
            "minimal": lambda: ttnn.experimental.minimal_matmul(
                tx,
                tw,
                bias_tensor=tb,
                config=c,
                compute_kernel_config=cfg,
                dtype=odt,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if act else None,
            ),
        }
        for name, fn in variants.items():
            got = ttnn.to_torch(fn()).double()
            d = got - ref
            print(
                f"[MMEPI] K{K} N{N} {odt} act={act} {name}: max abs {float(d.abs().max()):.3e}  rms {float(d.pow(2).mean().sqrt()):.3e}"
            )


# D5 (e): per role, in both precisions, minimal_matmul (with its fused bias / SiLU) must be no less accurate than
# ttnn.linear on the same inputs: rms error vs fp64 within MM_TOL x linear's. The control (LoFi, no fp32 dest
# accumulation, SiLU as a separate bf16 unary) must fail that same check, or the check cannot fail.
MM_TOL = 1.10
EPI_CASES = [
    # (K, N), role, output dtype, bias, activation
    ((512, 2048), "w_in", "float32", True, None),  # sLSTM W_in: fp32 bias and output
    ((64, 512), "embed", "bfloat16", True, "silu"),  # embed / head residual-block hidden
    ((512, 288), "head", "bfloat16", True, None),
    ((512, 2048), "variate", "bfloat16", False, "silu"),  # variate w_wi: SiLU without bias
    ((512, 512), "mlstm", "bfloat16", False, None),
    ((2048, 512), "ffn", "bfloat16", False, None),
]


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mm_vs_linear(device, precision):
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import Precision, kernel_config
    from models.experimental.tirex2_forecast.tt.layers import MM_BLOCKS

    p = Precision.by_name(precision)
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    failed, control_failed = [], []
    for (K, N), role, odt_name, has_bias, act in EPI_CASES:
        odt = getattr(ttnn, odt_name)
        cfg = p.gate_config() if role == "w_in" else p.matmul_config(role)
        wdt = p.weight_dtype(role)
        mb, kb, nb, sh, sw = MM_BLOCKS[(K, N)]
        kt = K // 32
        kb = max(d for d in range(1, min(kb, kt) + 1) if kt % d == 0)
        x = torch.randn(1, 1, 8192, K).bfloat16()
        w = (torch.randn(K, N) / K**0.5).bfloat16()
        tw = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
        w_dev = ttnn.to_torch(tw).double()  # the weights as the device holds them (bfp8-rounded in performance)
        tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        ref = x.double() @ w_dev
        tb = None
        if has_bias:
            bdt = ttnn.float32 if odt == ttnn.float32 else ttnn.bfloat16
            b = torch.randn(1, 1, 1, N) * 3.0
            tb = ttnn.from_torch(b, dtype=bdt, layout=ttnn.TILE_LAYOUT, device=device)
            ref = ref + ttnn.to_torch(tb).double()
        if act:
            ref = torch.nn.functional.silu(ref)
        mmc = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        )
        silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if act else None

        def minimal(kcfg, fused=True):
            y = ttnn.experimental.minimal_matmul(
                tx,
                tw,
                bias_tensor=tb,
                config=mmc,
                compute_kernel_config=kcfg,
                dtype=odt,
                fused_activation=silu if fused else None,
            )
            return ttnn.silu(y) if act and not fused else y

        lin = ttnn.to_torch(ttnn.linear(tx, tw, bias=tb, activation=act, dtype=odt, compute_kernel_config=cfg))
        got = ttnn.to_torch(minimal(cfg)).double()
        ctl = ttnn.to_torch(minimal(kernel_config(ttnn.MathFidelity.LoFi, fp32_acc=False), fused=False)).double()
        rms = lambda t: float((t - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        e_lin, e_mm, e_ctl = rms(lin.double()), rms(got), rms(ctl)
        dmax = float((got - lin.double()).abs().max())
        ok, ctl_ok = e_mm <= MM_TOL * e_lin, e_ctl <= MM_TOL * e_lin
        print(
            f"[MMVS] {precision} {role} K{K} N{N} {odt_name} bias={has_bias} act={act}: rms rel linear {e_lin:.3e} "
            f"minimal {e_mm:.3e} ({'ok' if ok else 'FAIL'})  max|minimal-linear| {dmax:.3e}  "
            f"control {e_ctl:.3e} ({'passes: BAD' if ctl_ok else 'fails: good'})"
        )
        if not ok:
            failed.append(role)
        if ctl_ok:
            control_failed.append(role)
    assert not failed, f"minimal_matmul less accurate than ttnn.linear on {failed}"
    assert not control_failed, f"the positive control passed on {control_failed}: the check cannot fail there"
