# SPDX-License-Identifier: Apache-2.0
"""Matmul config sweep (MODEL-AGNOSTIC) — a warm-start pre-pass for the optimize loop.

Before the profile-driven loop starts, enumerate the model's DISTINCT matmul ops (via the generic
op-sig probe — no per-model knowledge) and micro-benchmark a small, bounded grid of the two biggest
matmul knobs — math fidelity (LoFi/HiFi2/HiFi4) and operand dtype (bf16/bf8_b) — per shape, each run
PCC-gated against a full-precision (HiFi4/bf16) reference. The result is a best-config-per-shape table
the loop seeds its matmul buckets from, so it starts near the matmul floor instead of rediscovering
these two rungs one op at a time.

The knobs mirror the FIRST two device rungs of the per-op ladder (knob:fidelity -> knob:dtype), which
is where most matmul headroom lives; grid/shard/structural rungs stay in the loop where they depend on
the surrounding pipeline. PCC gating means a config is only ever recommended if it keeps accuracy.

Pure helpers (parse/candidate/pick/summarize) take no ttnn and are unit-tested; the device sweep
imports ttnn lazily and reuses the proven tp_fracture idioms (from_torch/matmul/_time_ms/_pcc)."""

from __future__ import annotations

import argparse
import ast
import json
import os

from agent.layer_depth import set_depth as _set_depth
from agent.probes import adaptive_op_timeout as _adaptive_op_timeout
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# the sweep grid: the two top matmul knobs, in ladder order. Kept small (3 x 2 = 6 configs/shape) so a
# whole-model sweep stays tractable — a big model has O(10) distinct matmul shapes, not O(1000).
_FIDELITIES = ("LoFi", "HiFi2", "HiFi4")
_DTYPES = ("bfloat16", "bfloat8_b")
_DEFAULT_PCC = 0.99


def _split_operand(op):
    """An op-sig operand is either ``((dims...), 'DTYPE')`` or a bare ``(dims...)`` (dtype unknown).
    Return ``(dims_tuple, dtype_str)`` with dtype '' when absent; ``(None, '')`` if unparseable."""
    if (
        isinstance(op, tuple)
        and len(op) == 2
        and isinstance(op[0], tuple)
        and op[1].__class__ is str
        and all(isinstance(d, int) for d in op[0])
    ):
        return op[0], op[1]
    if isinstance(op, tuple) and op and all(isinstance(d, int) for d in op):
        return op, ""
    return None, ""


def parse_matmul_sigs(sigs) -> List[dict]:
    """Extract distinct matmul ops from generic op-sig strings (``PERF_OP_SIGS``).

    A sig is ``<op.name><str(shape_sig_tuple)>`` where shape_sig is a tuple of per-operand
    ``((dims), 'DTYPE')`` entries (see ``_op_sig_probe._shape_sig``). We keep matmul/linear ops,
    read (M, K) from operand 0 and N from operand 1 (handling a (K, N) or a transposed (N, K)
    weight), and DEDUP by (m, k, n, in_dtype, w_dtype). Returns a list of shape dicts; malformed or
    non-matmul sigs are skipped, never raised."""
    out: List[dict] = []
    seen = set()
    for sig in sigs or []:
        if not isinstance(sig, str):
            continue
        p = sig.find("(")
        if p < 0:
            continue
        name = sig[:p]
        low = name.lower()
        if "matmul" not in low and "linear" not in low:
            continue
        try:
            args = ast.literal_eval(sig[p:])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(args, tuple):
            continue
        shapes, dtypes = [], []
        for op in args:
            dims, dt = _split_operand(op)
            if dims is None or len(dims) < 2:
                continue
            shapes.append(dims)
            dtypes.append(dt)
        if len(shapes) < 2:
            continue
        a, b = shapes[0], shapes[1]
        m, k = a[-2], a[-1]
        # in1 is normally (K, N); a stored/transposed weight can be (N, K). Match the shared K dim.
        if b[-2] == k:
            n = b[-1]
        elif b[-1] == k:
            n = b[-2]
        else:
            continue
        in_dt = dtypes[0] or "BFLOAT16"
        w_dt = (dtypes[1] if len(dtypes) > 1 else "") or "BFLOAT16"
        key = (m, k, n, in_dt, w_dt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"m": int(m), "k": int(k), "n": int(n), "in_dtype": in_dt, "w_dtype": w_dt})
    return out


def candidate_configs(m: int, k: int, n: int) -> List[dict]:
    """The bounded sweep grid for one matmul shape: fidelity x dtype (6 configs). Model-agnostic —
    the same grid for every shape; the sweep MEASURES which wins rather than guessing per shape."""
    return [{"fidelity": fid, "dtype": dt} for fid in _FIDELITIES for dt in _DTYPES]


def pick_best(results: List[dict], pcc_threshold: float = _DEFAULT_PCC) -> Optional[dict]:
    """From timed candidate results, the fastest config that still PASSES the PCC gate. A result
    counts only if it has a real timing (``ms``) AND ``pcc >= threshold`` — a fast but inaccurate
    config (e.g. bf4/LoFi that garbles a sensitive matmul) is never recommended. None if none pass."""
    ok = [r for r in results if r.get("ms") and r.get("pcc", 0.0) >= pcc_threshold]
    if not ok:
        return None
    return min(ok, key=lambda r: r["ms"])


def summarize(table: List[dict], pcc_threshold: float = _DEFAULT_PCC) -> dict:
    """Roll a per-shape sweep table into a compact summary for the MCP tool / loop: how many shapes
    got a non-default best (an fp16/HiFi4-beating config), and the recommended seed config per shape
    (fidelity+dtype+speedup vs the full-precision baseline)."""
    seeds, improved = [], 0
    for row in table or []:
        best = row.get("best")
        if not best:
            continue
        base = _baseline_ms(row)
        speedup = round(base / best["ms"], 3) if (base and best.get("ms")) else None
        if speedup and speedup > 1.01:
            improved += 1
        seeds.append(
            {
                "shape": {"m": row["m"], "k": row["k"], "n": row["n"]},
                "fidelity": best["fidelity"],
                "dtype": best["dtype"],
                "best_ms": best.get("ms"),
                "baseline_ms": base,
                "speedup": speedup,
                "pcc": best.get("pcc"),
            }
        )
    return {"shapes": len(table or []), "seeded": len(seeds), "improved": improved, "seeds": seeds}


def _baseline_ms(row: dict) -> Optional[float]:
    """The full-precision (HiFi4 + bf16) timing for a shape — the config the loop would use with no
    sweep — so speedup is measured against the honest default, not against the sweep's own worst."""
    for r in row.get("candidates", []):
        if r.get("fidelity") == "HiFi4" and r.get("dtype") == "bfloat16" and r.get("ms"):
            return r["ms"]
    return None


# --------------------------------------------------------------------------------------------------
# device sweep — imports ttnn lazily; reuses tp_fracture's timing + pcc primitives.
# --------------------------------------------------------------------------------------------------
def _compute_kernel_config(mesh_device, fidelity: str):
    import ttnn

    fid = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}[fidelity]
    try:
        return ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=fid)
    except Exception:  # noqa: BLE001 — older/arch-specific builds
        return ttnn.WormholeComputeKernelConfig(math_fidelity=fid)


def _tt_dtype(name: str):
    import ttnn

    return {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[name]


def _bench_config(mesh_device, x, w, ref, cfg: dict, iters: int) -> dict:
    import ttnn

    from cc_optimize.tp_fracture import _pcc, _time_ms

    dt = _tt_dtype(cfg["dtype"])
    ck = _compute_kernel_config(mesh_device, cfg["fidelity"])
    xt = ttnn.from_torch(
        x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    wt = ttnn.from_torch(
        w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    def run():
        y = ttnn.matmul(xt, wt, compute_kernel_config=ck)
        ttnn.synchronize_device(mesh_device)
        return y

    y = run()
    ms = _time_ms(run, iters)
    got = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: x.shape[0]]
    pcc = _pcc(ref, got.reshape(ref.shape))
    for t in (xt, wt, y):
        try:
            ttnn.deallocate(t)
        except Exception:  # noqa: BLE001
            pass
    return {**cfg, "ms": round(float(ms), 4), "pcc": round(float(pcc), 5)}


def sweep_one(mesh_device, m: int, k: int, n: int, pcc_threshold: float = _DEFAULT_PCC, iters: int = 5) -> dict:
    """Sweep every candidate config for one matmul shape on-device; return the full per-config table
    plus the PCC-gated best. A config that raises (OOM, unsupported dtype) is recorded ms=None so the
    sweep is robust to a single bad point."""
    import torch

    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(k, n)
    ref = x @ w
    results = []
    for cfg in candidate_configs(m, k, n):
        try:
            results.append(_bench_config(mesh_device, x, w, ref, cfg, iters))
        except Exception as exc:  # noqa: BLE001
            results.append({**cfg, "ms": None, "pcc": 0.0, "error": str(exc)[-200:]})
    return {"m": m, "k": k, "n": n, "candidates": results, "best": pick_best(results, pcc_threshold)}


def sweep_matmuls(mesh_device, matmuls: List[dict], pcc_threshold: float = _DEFAULT_PCC, iters: int = 5) -> List[dict]:
    """Sweep a list of distinct matmul shapes (from ``parse_matmul_sigs``). Returns one table row per
    shape. Deduped upstream, so each shape is benched once."""
    return [
        sweep_one(mesh_device, mm["m"], mm["k"], mm["n"], pcc_threshold=pcc_threshold, iters=iters) for mm in matmuls
    ]


# --------------------------------------------------------------------------------------------------
# standalone pre-pass — a SEPARATE step run BEFORE the optimize loop. Does NOT touch perf_mcp / the
# loop; it only enumerates matmuls (via the generic op-sig probe) and writes matmul_sweep.json, which
# the operator can then hand to / seed the optimize run with. Kept fully self-contained on purpose.
# --------------------------------------------------------------------------------------------------
_CC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CC_DIR.parent.parent.parent.parent


def enumerate_matmul_sigs(node: str, case: Optional[str] = None, repo_root: Optional[Path] = None) -> List[str]:
    """Run the generic all-layers op-sig probe (TT_PERF_LAYERS=0, no tracy) for a perf-test node and
    return the raw op-signature strings. Model-agnostic — reuses _op_sig_probe.py verbatim, so it
    needs no knowledge of the model. Returns [] on failure (never raises)."""
    repo = Path(repo_root or _REPO_ROOT)
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(repo)
    env["PYTHONPATH"] = str(repo)
    _set_depth(env, None)  # ALL layers: cap REMOVED, never sent as 0 (see agent/layer_depth.py)
    env["TT_PERF_OSL_TOKENS"] = "1"
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    cmd = [sys.executable, str(_CC_DIR / "_op_sig_probe.py"), node]
    if case:
        cmd.append(case)
    try:
        r = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            # scales with the shapes/fidelities/dtypes swept, so it is workload-dependent too
            timeout=int(os.environ.get("PERF_MCP_MEASURE_STALL_SEC") or _adaptive_op_timeout("profile")),
        )
    except Exception as exc:  # noqa: BLE001
        print("[matmul-sweep] op-sig probe failed: %s" % str(exc)[-300:], file=sys.stderr)
        return []
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    for prefix in ("PERF_OP_SIG_COUNTS=", "PERF_OP_SIGS=", "PERF_OP_SIG_SEQUENCE="):
        for line in out.splitlines():
            if line.startswith(prefix):
                try:
                    payload = json.loads(line.split("=", 1)[1])
                except (ValueError, TypeError):
                    continue
                return list(payload.keys()) if isinstance(payload, dict) else list(payload)
    print("[matmul-sweep] no op signatures emitted by the probe", file=sys.stderr)
    return []


def run_prepass(
    node: str,
    case: Optional[str] = None,
    out_path: Optional[str] = None,
    pcc_threshold: float = _DEFAULT_PCC,
    iters: int = 5,
    max_shapes: int = 0,
    repo_root: Optional[Path] = None,
) -> dict:
    """The full pre-pass: enumerate -> parse distinct matmuls -> open the run's planned mesh -> sweep
    fidelity x dtype per shape (PCC-gated) -> write out_path + return the summary. Opens the SAME
    topology the optimize loop uses (resolve_mesh_shape reads TT_PERF_MESH_ROWS/COLS; FABRIC_1D when
    multi-chip), falling back to 1x1 no-fabric when the env is unset so it is still safe standalone.
    Matching the loop's format means no 1x1<->NxM fabric transition to deadlock the next mesh open.
    Prints a dropped-shape count when max_shapes caps coverage (never silently truncates)."""
    sigs = enumerate_matmul_sigs(node, case, repo_root)
    matmuls = parse_matmul_sigs(sigs)
    if not matmuls:
        # An empty enumeration must never clobber a good table: a probe that crashes or a node that
        # fails to resolve returns zero, and overwriting a previous 76-shape table with that erases
        # every seed. Keep any existing non-empty table and report it instead of writing zeros over it.
        prior = None
        if out_path:
            try:
                prior = json.loads(Path(out_path).read_text())
            except Exception:  # noqa: BLE001
                prior = None
        if isinstance(prior, dict) and prior.get("shapes"):
            prior["note"] = "no matmul ops enumerated this pass; kept prior table"
            return prior
        result = {"ok": True, "shapes": 0, "seeded": 0, "note": "no matmul ops enumerated"}
        _write(out_path, result)
        return result
    if max_shapes and len(matmuls) > max_shapes:
        dropped = len(matmuls) - max_shapes
        print(
            "[matmul-sweep] capping at %d of %d distinct matmul shapes (%d dropped)"
            % (max_shapes, len(matmuls), dropped),
            file=sys.stderr,
        )
        matmuls = matmuls[:max_shapes]
    import ttnn  # lazy — only needed for the device sweep

    from agent.perf_adapter import resolve_mesh_shape

    _rows, _cols = resolve_mesh_shape(default_rows=1, default_cols=1)
    rows, cols = int(_rows), int(_cols)
    multichip = rows * cols > 1
    if multichip:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        table = sweep_matmuls(mesh, matmuls, pcc_threshold=pcc_threshold, iters=iters)
    finally:
        ttnn.close_mesh_device(mesh)
        if multichip:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
    summary = summarize(table, pcc_threshold)
    summary["ok"] = True
    summary["table"] = table
    _write(out_path, summary)
    return summary


def _write(out_path: Optional[str], payload: dict) -> None:
    path = Path(out_path) if out_path else (_CC_DIR / "matmul_sweep.json")
    try:
        path.write_text(json.dumps(payload, indent=2))
        print("[matmul-sweep] wrote %s" % path, file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print("[matmul-sweep] could not write %s: %s" % (path, exc), file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="matmul_sweep",
        description="Matmul fidelity x dtype sweep pre-pass — run BEFORE the optimize loop to build a "
        "best-config-per-shape warm-start table. Standalone; does not modify the optimize tool.",
    )
    ap.add_argument("node", help="perf-test pytest node id (same one the optimize run uses)")
    ap.add_argument("--case", default=None, help="optional -k case filter for the perf test")
    ap.add_argument("--out", default=None, help="output json path (default: cc_optimize/matmul_sweep.json)")
    ap.add_argument("--pcc", type=float, default=_DEFAULT_PCC, help="min PCC to accept a config (default 0.99)")
    ap.add_argument("--iters", type=int, default=5, help="timed reps per config (default 5)")
    ap.add_argument("--max-shapes", type=int, default=0, help="cap distinct shapes swept (0 = all)")
    ap.add_argument("--repo-root", default=None, help="repo root the perf-test node is relative to (default: derived)")
    a = ap.parse_args(argv)
    summary = run_prepass(
        a.node,
        case=a.case,
        out_path=a.out,
        pcc_threshold=a.pcc,
        iters=a.iters,
        max_shapes=a.max_shapes,
        repo_root=a.repo_root,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
