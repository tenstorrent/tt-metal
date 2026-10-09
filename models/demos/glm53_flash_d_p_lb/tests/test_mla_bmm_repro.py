# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro: MLA per-head absorb matmuls with explicit MatmulMultiCoreReuseProgramConfig vs the auto config, on
one chip at the model's chunk-5120 shapes (640 query rows per chip, 64 heads), with the inputs built by the model's own
producer ops:
  q w_uk: qh = nlp_create_qkv_heads(q [1, 1, 640, 64 * 256]) -> [1, 64, 640, 256] x w_uk [1, 64, 256, 512]
  o w_uv: ot = to_layout(o [1, 64, 640, 512] ROW_MAJOR, TILE) x w_uv [1, 64, 512, 256]
and, for comparison, the same values uploaded directly with from_torch. Each config's output is scored against fp32
torch (rel L2) and against the auto config's output on the same input. GLM_BMMR_ROWS (default 640)."""

import os

import pytest
import torch

import ttnn

ROWS = int(os.environ.get("GLM_BMMR_ROWS", "640"))
H, DQK, R, DV = 64, 256, 512, 256


def _cfg(grid, pm, Nt, Kt):
    bw = max(d for d in (1, 2, 4) if Kt % d == 0)
    sw = max(d for d in (1, 2, 4) if Nt % d == 0)
    sh = max(d for d in range(1, 5) if pm % d == 0 and d * sw <= 4)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=Nt,
    )


def _host(t):  # chip 0 of a mesh tensor, or the tensor itself
    devs = ttnn.get_device_tensors(t) if hasattr(t, "device") and t.device().get_num_devices() > 1 else [t]
    return ttnn.to_torch(devs[0])


def _rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm())


def _run(device):
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    mc = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
        t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc
    )
    Mt = ROWS // 32

    # q w_uk
    q = torch.randn(1, 1, ROWS, H * DQK)
    qh_prod, _, _ = ttnn.experimental.nlp_create_qkv_heads(
        up(q), num_heads=H, num_kv_heads=0, transpose_k_heads=False, memory_config=mc
    )
    qh_host = q.reshape(1, ROWS, H, DQK).permute(0, 2, 1, 3).contiguous()
    w_uk = torch.randn(1, H, DQK, R) / DQK**0.5
    # o w_uv
    o = torch.randn(1, H, ROWS, R)
    ot_prod = ttnn.to_layout(up(o, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT, memory_config=mc)
    w_uv = torch.randn(1, H, R, DV) / R**0.5

    cases = {
        "q w_uk": (qh_prod, up(qh_host), qh_host, up(w_uk), w_uk, DQK, R),
        "o w_uv": (ot_prod, up(o), o, up(w_uv), w_uv, R, DV),
    }
    print(
        f"[bmmr] rows {ROWS}, grid {grid.x}x{grid.y}; producer tensors: qh {tuple(qh_prod.shape)} "
        f"{qh_prod.memory_config()} | ot {tuple(ot_prod.shape)}",
        flush=True,
    )
    for name, (a_prod, a_host, a_t, w_dev, w_t, K, N) in cases.items():
        truth = a_t.to(torch.bfloat16).double() @ w_t.to(torch.bfloat16).double()
        Kt, Nt = K // 32, N // 32
        for src, a in (("producer", a_prod), ("from_torch", a_host)):
            auto = _host(ttnn.matmul(a, w_dev, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=mc))
            print(f"[bmmr] {name:7s} {src:10s} auto        rel vs fp32 {_rel(auto, truth):.3e}", flush=True)
            for pm in [d for d in range(1, 21) if Mt % d == 0]:
                try:
                    out = ttnn.matmul(
                        a,
                        w_dev,
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=ckc,
                        memory_config=mc,
                        program_config=_cfg(grid, pm, Nt, Kt),
                    )
                    o_t = _host(out)
                    blocks = H * Mt // pm
                    print(
                        f"[bmmr] {name:7s} {src:10s} pm {pm:2d} ({blocks:4d} blocks) rel vs fp32 {_rel(o_t, truth):.3e}"
                        f"  vs auto {_rel(o_t, auto):.3e}",
                        flush=True,
                    )
                    ttnn.deallocate(out)
                except Exception as ex:
                    print(f"[bmmr] {name:7s} {src:10s} pm {pm:2d} FAILED {str(ex).splitlines()[0][:80]}", flush=True)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_mla_bmm_repro(device):
    _run(device)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_mla_bmm_repro_mesh(mesh_device):
    """The same on the model's 2x4 mesh with 2D fabric (the device setup the model opens)."""
    print(
        f"[bmmr] mesh {tuple(mesh_device.shape)} worker grid {mesh_device.compute_with_storage_grid_size()}", flush=True
    )
    _run(mesh_device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_mla_bmm_replay(device):
    """Replay chip 0's live in-model inputs (GLM_MLA_CHECK_DUMP dir, written by mla_attention._check) on one chip."""
    d = os.environ["GLM_BMMR_DUMP"]
    grid = device.compute_with_storage_grid_size()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    mc = ttnn.DRAM_MEMORY_CONFIG
    for tag, pm in (("q_w_uk", 5), ("o_w_uv", 10)):
        t = torch.load(f"{d}/{tag}.pt")
        a = ttnn.from_torch(t["a"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        w = ttnn.from_torch(t["w"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        Kt, Nt = t["w"].shape[-2] // 32, t["w"].shape[-1] // 32
        auto = ttnn.to_torch(ttnn.matmul(a, w, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=mc))
        cfg = ttnn.to_torch(
            ttnn.matmul(
                a,
                w,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=mc,
                program_config=_cfg(grid, pm, Nt, Kt),
            )
        )
        fin = lambda x: (~torch.isfinite(x.float())).sum().item()  # noqa: E731
        print(
            f"[bmmr-replay] {tag}: a {tuple(t['a'].shape)} non-finite {fin(t['a'])} max|a| {t['a'].float().abs().max():.3e}"
            f" | replay cfg vs auto {_rel(cfg, auto):.3e} | in-model cfg vs auto {_rel(t['cfg_out'], t['auto_out']):.3e}"
            f" | replay auto vs in-model auto {_rel(auto, t['auto_out']):.3e} | replay cfg vs in-model cfg {_rel(cfg, t['cfg_out']):.3e}",
            flush=True,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("pred", ["none", "matmul", "minimal_matmul", "tilize", "untilize"])
def test_mla_bmm_after(device, pred):
    """The configured batched matmul run DIRECTLY after a predecessor op (no auto matmul in between): device state a
    previous kernel leaves behind (unpacker / packer config) that the multi-core-reuse bmm kernel does not reset shows
    as a wrong result. Inputs are fixed; only the op run just before differs."""
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    mc = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
        t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc
    )
    a_t = torch.randn(1, H, ROWS, DQK)
    w_t = torch.randn(1, H, DQK, R) / DQK**0.5
    a, w = up(a_t), up(w_t)
    truth = a_t.to(torch.bfloat16).double() @ w_t.to(torch.bfloat16).double()
    x2 = up(torch.randn(1, 1, ROWS, 1536))
    w2 = up(torch.randn(1, 1, 1536, 4096) / 40)
    rm = up(torch.randn(1, 1, ROWS, 2048), ttnn.ROW_MAJOR_LAYOUT)
    preds = {
        "none": lambda: None,
        "matmul": lambda: ttnn.matmul(x2, w2, compute_kernel_config=ckc, memory_config=mc),
        "minimal_matmul": lambda: ttnn.experimental.minimal_matmul(x2, w2, compute_kernel_config=ckc, memory_config=mc),
        "tilize": lambda: ttnn.to_layout(rm, ttnn.TILE_LAYOUT, memory_config=mc),
        "untilize": lambda: ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc),
    }
    for pm in (5, 10):
        t = preds[pred]()
        out = ttnn.matmul(
            a,
            w,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ckc,
            memory_config=mc,
            program_config=_cfg(grid, pm, R // 32, DQK // 32),
        )
        rel = _rel(ttnn.to_torch(out), truth)
        print(f"[bmmr-after] pred {pred:15s} pm {pm:2d}: rel vs fp32 {rel:.3e}", flush=True)
        if t is not None:
            ttnn.deallocate(t)
        ttnn.deallocate(out)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("shape", ["uk", "uv"])
def test_mla_bmm_blocks(device, shape):
    """Every per_core_M for the batched per-head matmul, each run right after an unrelated matmul (so no stale L1 from a
    same-shape run can mask a wrong read): rel vs fp32 and the block count vs the core count."""
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    mc = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t: ttnn.from_torch(  # noqa: E731
        t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    K, N = (DQK, R) if shape == "uk" else (R, DV)
    a_t, w_t = torch.randn(1, H, ROWS, K), torch.randn(1, H, K, N) / K**0.5
    a, w = up(a_t), up(w_t)
    truth = a_t.to(torch.bfloat16).double() @ w_t.to(torch.bfloat16).double()
    x2, w2 = up(torch.randn(1, 1, 2048, 2048)), up(torch.randn(1, 1, 2048, 2048))
    Mt = ROWS // 32
    for pm in [d for d in range(1, Mt + 1) if Mt % d == 0]:
        ttnn.deallocate(ttnn.matmul(x2, w2, compute_kernel_config=ckc, memory_config=mc))  # scrub L1
        try:
            out = ttnn.matmul(
                a,
                w,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=mc,
                program_config=_cfg(grid, pm, N // 32, K // 32),
            )
            rel = _rel(ttnn.to_torch(out), truth)
            ttnn.deallocate(out)
            print(
                f"[bmmr-blocks] {shape} pm {pm:2d}: {H * Mt // pm:5d} blocks / {grid.x * grid.y} cores, rel vs fp32 {rel:.3e}",
                flush=True,
            )
        except Exception as ex:
            print(f"[bmmr-blocks] {shape} pm {pm:2d}: FAILED {str(ex).splitlines()[0][:70]}", flush=True)
