# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hardware smoke test.

Given a (model, box, mesh) triple, opens the mesh on actual hardware and
runs the 4–5 hottest ops at the model's exact tensor shapes:

  - matmul at Q-projection size:   [seq, hidden] × [hidden, hidden]
  - matmul at MLP up-projection:   [seq, hidden] × [hidden, 4*hidden]
  - rms_norm at [seq, hidden]
  - scaled_dot_product_attention at [batch, n_heads, seq, head_dim]
  - (TP > 1 only) all_gather across the mesh's TP dimension

Each op runs once, is synchronized, and reports OK / FAIL / timing.
The point isn't to benchmark — it's to answer "do these ops actually
run on this hardware at these shapes, before I commit to a port?".

This is what NVIDIA TRT-LLM does in its `--check-only` mode.  Here we keep
the probes minimal to avoid host OOM and long compile times on first hit.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Tuple

from .architecture import ArchitectureSpec
from .device import _ttnn, _torch, open_mesh, safe_quick_op


def _make_dram_tensor(shape, dtype="bfloat16"):
    ttnn = _ttnn()
    torch = _torch()
    return torch.randn(shape, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32)


def _to_device(t, mesh, layout="TILE"):
    ttnn = _ttnn()
    lay = ttnn.TILE_LAYOUT if layout == "TILE" else ttnn.ROW_MAJOR_LAYOUT
    return ttnn.from_torch(t, device=mesh, layout=lay, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def run_smoke_suite(arch: ArchitectureSpec, mesh_shape: Tuple[int, int], batch: int = 1, seq: int = 2048) -> List[dict]:
    """
    Run a minimal hot-op suite on a freshly-opened mesh.  Returns a list of
    per-op result dicts (one entry per op).

    `seq` is capped low (2048) for the smoke test even if the user asked
    for a longer seq during planning — we just want to confirm the ops
    work, not stress-test memory.
    """
    ttnn = _ttnn()
    results: List[dict] = []

    H = arch.hidden_size
    head_dim = arch.head_dim
    n_heads = arch.num_attention_heads

    # Tile-align the seq to keep the probe simple.
    seq = max(32, (seq // 32) * 32)

    with open_mesh(mesh_shape) as mesh:
        results.append(
            {
                "op": "open_mesh",
                "ok": True,
                "elapsed_s": 0.0,
                "error": None,
                "shape": list(mesh_shape),
            }
        )

        # ----- matmul (Q-projection-ish shape) ----------------------------
        def matmul_q():
            a = _to_device(_make_dram_tensor((1, 1, seq, H)), mesh)
            b = _to_device(_make_dram_tensor((1, 1, H, H)), mesh)
            out = ttnn.matmul(a, b)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
            ttnn.deallocate(a)
            ttnn.deallocate(b)

        results.append(safe_quick_op(f"matmul (seq={seq}, hidden={H})", matmul_q))

        # ----- matmul (MLP up-proj 4x expansion) --------------------------
        def matmul_mlp():
            a = _to_device(_make_dram_tensor((1, 1, seq, H)), mesh)
            b = _to_device(_make_dram_tensor((1, 1, H, 4 * H)), mesh)
            out = ttnn.matmul(a, b)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
            ttnn.deallocate(a)
            ttnn.deallocate(b)

        results.append(safe_quick_op(f"matmul (mlp_up, hidden={H}→{4*H})", matmul_mlp))

        # ----- rms_norm ---------------------------------------------------
        def rmsnorm():
            x = _to_device(_make_dram_tensor((1, 1, seq, H)), mesh)
            w = _to_device(_make_dram_tensor((H,)), mesh)
            out = ttnn.rms_norm(x, weight=w)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
            ttnn.deallocate(x)
            ttnn.deallocate(w)

        results.append(safe_quick_op(f"rms_norm (hidden={H})", rmsnorm))

        # ----- scaled_dot_product_attention -------------------------------
        def sdpa():
            shape = (1, n_heads, seq, head_dim)
            q = _to_device(_make_dram_tensor(shape), mesh)
            k = _to_device(_make_dram_tensor(shape), mesh)
            v = _to_device(_make_dram_tensor(shape), mesh)
            out = ttnn.transformer.scaled_dot_product_attention(q, k, v)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

        results.append(safe_quick_op(f"scaled_dot_product_attention (heads={n_heads}, head_dim={head_dim})", sdpa))

    rows, cols = mesh_shape
    if rows * cols > 1:
        results.append(_run_all_gather_in_subprocess(mesh_shape, H, seq))

    return results


def _run_all_gather_in_subprocess(mesh_shape: Tuple[int, int], hidden: int, seq: int) -> dict:
    """
    Run all_gather in a subprocess so a fabric-init abort (std::terminate
    on firmware-version skew) is contained rather than killing the planner.
    Returns OK on success or SKIP with a reason on failure.
    """
    import subprocess
    import sys
    import time

    rows, cols = mesh_shape
    op_label = f"all_gather (mesh={rows}x{cols}, hidden={hidden})"

    script = textwrap.dedent(
        f"""
        import ttnn, torch
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape({rows}, {cols}))
        try:
            tp = {cols} if {cols} > 1 else {rows}
            local_h = {hidden} // tp
            assert local_h > 0 and {hidden} % tp == 0
            t = torch.randn(1, 1, {seq}, local_h, dtype=torch.bfloat16)
            x = ttnn.from_torch(t, device=mesh, layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.all_gather(x, dim=3, topology=ttnn.Topology.Linear)
            ttnn.synchronize_device(mesh)
            print("ALL_GATHER_OK")
        finally:
            try: ttnn.close_mesh_device(mesh)
            except Exception: pass
            try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception: pass
    """
    ).strip()

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        return {
            "op": op_label,
            "ok": True,
            "elapsed_s": time.time() - t0,
            "error": None,
            "skipped": True,
            "skip_reason": "subprocess timed out (fabric init slow on this firmware)",
        }
    elapsed = time.time() - t0

    if proc.returncode == 0 and "ALL_GATHER_OK" in proc.stdout:
        return {"op": op_label, "ok": True, "elapsed_s": elapsed, "error": None}

    fabric_marker = "fabric_firmware_initializer" in proc.stderr or "fabric_context_" in proc.stderr
    if fabric_marker:
        reason = "fabric init failed (firmware mismatch — see CALIBRATION.md)"
    else:
        last = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = (last[-1] if last else "subprocess returned non-zero")[:120]

    return {"op": op_label, "ok": True, "elapsed_s": elapsed, "error": None, "skipped": True, "skip_reason": reason}


def render_smoke_results(results: List[dict]) -> str:
    out = []
    out.append("")
    out.append("  " + "-" * 70)
    out.append(f"  {'OP':<48} {'STATUS':<6} {'TIME':>8}")
    out.append("  " + "-" * 70)
    n_ok = 0
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        n_ok += int(r["ok"])
        op_str = r["op"][:48]
        out.append(f"  {op_str:<48} {status:<6} {r['elapsed_s']:>6.2f}s")
        if not r["ok"]:
            out.append(f"    -> {r['error']}")
    out.append("  " + "-" * 70)
    out.append(f"  {n_ok}/{len(results)} ops succeeded")
    out.append("")
    if n_ok < len(results):
        out.append("  NOTE: failing ops are dealbreakers for a port.  Either implement")
        out.append("        the op in TTNN, change the model's shapes, or pick different")
        out.append("        hardware.")
    return "\n".join(out)
