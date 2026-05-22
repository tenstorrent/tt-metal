# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression-demo for the dots.ocr bottom-up test suite.

Two demonstrations, both standalone — they open and close their own mesh
device. These are NOT part of the pytest suite; they exist to prove that
when the suite's assertions DO fire, they fire for real reasons.

How to run::

    cd /home/ttuser/salnahari/tt-metal
    source python_env/bin/activate
    MESH_DEVICE=T3K \
    python models/experimental/tt_symbiote/tests/unit/dots_ocr/scripts/regression_demo.py

Demo 1 — math_fidelity downgrade
--------------------------------
Two sub-demos in series:

  Demo 1a (rms_norm):  Picks the first captured ``ttnn.rms_norm`` row
    (call_id=8, shape ``(1, 1, 14, 1536)``, BFLOAT16, captured
    ``math_fidelity=HiFi4``). Runs the op twice on the same inputs —
    once with the captured HiFi4 config and once with a forced
    ``LoFi + math_approx_mode=True`` config — and prints both PCCs
    alongside the suite's per-row threshold (0.999). Mostly normative:
    RMSNorm is fairly fidelity-insensitive so the delta is small but
    measurable (~2e-4).

  Demo 1b (matmul):  ``ttnn.matmul`` with BFP8 weights, sizes large
    enough to expose math-fidelity error budget. Runs HiFi4 then LoFi
    and prints both PCCs alongside the matmul threshold derived from
    the input dtypes. With BFP8 weight + LoFi the PCC is expected to
    drop noticeably below the BFP8 dtype-min threshold (0.985).

Demo 2 — wrong reference
------------------------
Runs ``ttnn.add(a, b)`` on a small random tensor and compares against
``torch.sub(a, b)`` instead of ``torch.add(a, b)``. ``assert_op_pcc`` is
expected to raise. The exception is caught and the assertion message is
printed verbatim so the failure path is exercised end-to-end.

No files outside ``scripts/`` are touched.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import ttnn

# Make the in-tree test utilities importable when run as a plain script.
_REPO_ROOT = "/home/ttuser/salnahari/tt-metal"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.common.utility_functions import comp_pcc as _upstream_comp_pcc  # noqa: E402
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (  # noqa: E402
    rms_norm as ref_rms_norm,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import (  # noqa: E402
    assert_op_pcc,
    op_pcc_threshold,
)


def _open_t3k_dp_mesh():
    """Open the same (8, 1) DP-on-T3K mesh the unit tests use."""
    mesh_shape = ttnn.MeshShape(8, 1)
    device_params = {
        "trace_region_size": 1_000_000,
        "num_command_queues": 1,
    }
    # Multi-device mesh needs fabric.
    from conftest import set_fabric, reset_fabric  # type: ignore

    fabric_cfg = ttnn.FabricConfig.FABRIC_1D_RING
    set_fabric(fabric_cfg, None, None, None, None)
    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh, fabric_cfg, reset_fabric


def _close_mesh(mesh, fabric_cfg, reset_fabric_fn):
    for sub in mesh.get_submeshes():
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(mesh)
    reset_fabric_fn(fabric_cfg)


# ---------------------------------------------------------------------------
# Demo 1
# ---------------------------------------------------------------------------


def _run_rms_norm_once(
    device,
    x_torch: torch.Tensor,
    w_torch: torch.Tensor,
    eps: float,
    compute_kernel_config,
) -> torch.Tensor:
    hidden = x_torch.shape[-1]
    w_tile = w_torch.reshape(1, 1, hidden // 32, 32)

    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    w_tt = ttnn.from_torch(
        w_tile,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    out_tt = ttnn.rms_norm(
        x_tt,
        weight=w_tt,
        epsilon=eps,
        compute_kernel_config=compute_kernel_config,
    )

    # Replicated gather: take device-0's copy via ConcatMeshToTensor + slice.
    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    full = ttnn.to_torch(out_tt, mesh_composer=composer)
    num_devices = device.shape[0] * device.shape[1]
    per_lead = full.shape[0] // num_devices
    out_torch = full[:per_lead]

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(w_tt)
    return out_torch


def demo1_math_fidelity_downgrade(device) -> None:
    print("=" * 70)
    print("Demo 1a — math_fidelity downgrade (rms_norm cid=8, HiFi4 vs LoFi)")
    print("=" * 70)

    # Captured row 0 ("call_id": 8, "ttnn.rms_norm", shape (1,1,14,1536)).
    torch.manual_seed(0)
    x = torch.randn(1, 1, 14, 1536, dtype=torch.bfloat16) * 0.5
    w = torch.randn(1536, dtype=torch.bfloat16) * 0.1 + 1.0
    eps = 1e-6
    ref = ref_rms_norm(x, w, eps=eps).to(torch.float32)

    # Threshold the actual suite uses for this row (HiFi4, bfloat16):
    threshold = op_pcc_threshold("ttnn.rms_norm", [ttnn.bfloat16], "HiFi4")
    print(f"  threshold (suite, op=ttnn.rms_norm, in=bf16) = {threshold:.4f}")

    # --- HiFi4 (captured) ---
    ckc_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    out_hifi4 = _run_rms_norm_once(device, x, w, eps, ckc_hifi4).to(torch.float32)
    while out_hifi4.dim() > ref.dim() and out_hifi4.shape[0] == 1:
        out_hifi4 = out_hifi4.squeeze(0)
    if out_hifi4.dim() > ref.dim():
        try:
            out_hifi4 = out_hifi4.reshape(ref.shape)
        except RuntimeError:
            pass
    ok_h, info_h = _upstream_comp_pcc(ref, out_hifi4, pcc=threshold)
    pcc_h = _extract_pcc(info_h)
    print(f"  HiFi4 PCC      = {pcc_h:.6f}  (ok={ok_h})")

    # --- LoFi + approx (forced downgrade) ---
    ckc_lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    out_lofi = _run_rms_norm_once(device, x, w, eps, ckc_lofi).to(torch.float32)
    while out_lofi.dim() > ref.dim() and out_lofi.shape[0] == 1:
        out_lofi = out_lofi.squeeze(0)
    if out_lofi.dim() > ref.dim():
        try:
            out_lofi = out_lofi.reshape(ref.shape)
        except RuntimeError:
            pass
    ok_l, info_l = _upstream_comp_pcc(ref, out_lofi, pcc=threshold)
    pcc_l = _extract_pcc(info_l)
    print(f"  LoFi  PCC      = {pcc_l:.6f}  (ok={ok_l})")

    delta = pcc_h - pcc_l
    print(f"  DELTA (HiFi4 - LoFi) = {delta:.6f}")
    print(f"  Expectation: HiFi4 PCC >= threshold ({threshold:.4f}) and LoFi PCC drops below it.")
    print()


# ---------------------------------------------------------------------------
# Demo 2
# ---------------------------------------------------------------------------


def demo1b_matmul_math_fidelity(device) -> None:
    """Larger demonstration: BFP8-weight matmul + LoFi shows a clear PCC drop.

    Shapes are inspired by the captured text rows (K=1536, N=2048, BFLOAT16
    activation, BFP8_B weight) — that combination is in the suite as one of
    the smaller linear rows.
    """
    print("=" * 70)
    print("Demo 1b — math_fidelity downgrade (matmul bf16 x bfp4, HiFi4 vs LoFi)")
    print("=" * 70)

    torch.manual_seed(0)
    # Large K accumulation in bfp4 weights: small per-element math_fidelity
    # losses pile up and the LoFi PCC degrades visibly.
    M, K, N = 256, 8192, 2048
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.5
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.5
    ref = a.to(torch.float32) @ b.to(torch.float32)

    a_tt = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    # Pre-quantize b to bfp4_b on device — most fidelity-sensitive captured weight dtype.
    b_tt = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    threshold = op_pcc_threshold("ttnn.matmul", [ttnn.bfloat16, ttnn.bfloat4_b], "HiFi4")
    print(f"  threshold (suite, op=ttnn.matmul, in=[bf16,bfp4]) = {threshold:.4f}")

    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    num_devices = device.shape[0] * device.shape[1]

    def _run(ckc):
        out = ttnn.matmul(a_tt, b_tt, compute_kernel_config=ckc)
        full = ttnn.to_torch(out, mesh_composer=composer)
        per_lead = full.shape[0] // num_devices
        result = full[:per_lead].to(torch.float32)
        ttnn.deallocate(out)
        return result

    ckc_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out_hifi4 = _run(ckc_hifi4)
    while out_hifi4.dim() > ref.dim() and out_hifi4.shape[0] == 1:
        out_hifi4 = out_hifi4.squeeze(0)
    if out_hifi4.dim() > ref.dim():
        try:
            out_hifi4 = out_hifi4.reshape(ref.shape)
        except RuntimeError:
            pass
    ok_h, info_h = _upstream_comp_pcc(ref, out_hifi4, pcc=threshold)
    pcc_h = _extract_pcc(info_h)
    print(f"  HiFi4 PCC      = {pcc_h:.6f}  (ok={ok_h})")

    ckc_lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    out_lofi = _run(ckc_lofi)
    while out_lofi.dim() > ref.dim() and out_lofi.shape[0] == 1:
        out_lofi = out_lofi.squeeze(0)
    if out_lofi.dim() > ref.dim():
        try:
            out_lofi = out_lofi.reshape(ref.shape)
        except RuntimeError:
            pass
    ok_l, info_l = _upstream_comp_pcc(ref, out_lofi, pcc=threshold)
    pcc_l = _extract_pcc(info_l)
    print(f"  LoFi  PCC      = {pcc_l:.6f}  (ok={ok_l})")

    delta = pcc_h - pcc_l
    print(f"  DELTA (HiFi4 - LoFi) = {delta:.6f}")
    print(f"  Expectation: HiFi4 PCC >= threshold ({threshold:.4f}) and LoFi PCC drops below it.")

    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)
    print()


def demo2_wrong_reference(device) -> None:
    print("=" * 70)
    print("Demo 2 — wrong reference (ttnn.add vs torch.sub)")
    print("=" * 70)

    torch.manual_seed(0)
    a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    b_tt = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    out_tt = ttnn.add(a_tt, b_tt)

    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    full = ttnn.to_torch(out_tt, mesh_composer=composer)
    num_devices = device.shape[0] * device.shape[1]
    per_lead = full.shape[0] // num_devices
    actual = full[:per_lead].to(torch.float32)

    ttnn.deallocate(out_tt)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)

    wrong_ref = torch.sub(a, b).to(torch.float32)  # intentionally wrong
    threshold = op_pcc_threshold("ttnn.add", [ttnn.bfloat16], "HiFi2")
    print(f"  threshold (suite, op=ttnn.add, in=bf16) = {threshold:.4f}")
    print(f"  reference op: torch.sub (intentionally wrong; correct is torch.add)")

    try:
        pcc = assert_op_pcc(
            wrong_ref,
            actual,
            threshold=threshold,
            op_name="ttnn.add",
            row_id="demo2_wrong_reference",
        )
        print(f"  UNEXPECTED PASS — measured PCC={pcc:.6f}; assertion did not fire.")
    except AssertionError as exc:
        print("  AssertionError fired (expected). Message follows:")
        for line in str(exc).splitlines():
            print(f"    {line}")
    print()


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def _extract_pcc(info) -> float:
    if isinstance(info, (int, float)):
        return float(info)
    if isinstance(info, str):
        import re

        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", info)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
    return float("nan")


def main() -> int:
    if os.environ.get("MESH_DEVICE", "").upper() != "T3K":
        print("Warning: MESH_DEVICE is not T3K; this script assumes an 8-device DP mesh.")
    mesh, fabric_cfg, reset_fabric_fn = _open_t3k_dp_mesh()
    try:
        demo1_math_fidelity_downgrade(mesh)
        demo1b_matmul_math_fidelity(mesh)
        demo2_wrong_reference(mesh)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        _close_mesh(mesh, fabric_cfg, reset_fabric_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
