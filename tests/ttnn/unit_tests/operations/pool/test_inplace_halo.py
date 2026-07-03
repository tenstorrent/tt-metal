# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness harness for the SILENT in-place halo (see IN_PLACE_HALO_REDO.md section 11).

Principle
---------
In-place halo and normal halo must produce the BITWISE-IDENTICAL haloed tensor -- halo is
pure data movement (gather), there is no math. Small races can corrupt a few sticks, pass a
PCC threshold, and be masked by max-pooling. We cannot call halo directly from Python, so we
test THROUGH the pool op, using a pool that does not mask corruption:

  * AVERAGE pool sums every stick in each window -> corruption in ANY haloed stick changes
    the output. This is the PRIMARY detector.
  * MAX pool is a secondary (it can mask non-max sticks) -- included but weaker.

Because in-place vs normal feed the pool the identical layout, a correct in-place path yields
a bitwise-identical pool output. So we run the same pool+shape+input twice -- once with
in-place halo ACTIVE, once DISABLED -- and assert EXACT elementwise equality (torch.equal).
Any mismatch is an in-place halo bug; we print the first mismatch for diagnosis.

Why separate processes (guard against a FALSE PASS)
---------------------------------------------------
The in-place decision reads the env var TT_METAL_DISABLE_INPLACE_HALO at op-call time, and
that env var is NOT part of the program-cache key. A naive same-process toggle would return
the STALE cached program on the second setting (making golden == inplace trivially -> false
PASS). We therefore run each setting in a SEPARATE PROCESS (fresh device + fresh program
cache), dump each output tensor to a file, and compare in the parent. Each worker process
routes the C++ logger to its own TT_LOGGER_FILE so we can prove -- per process -- that the
"in-place halo active" line APPEARS when active and is ABSENT when disabled.

dtype note (why input stays bf16 row-major)
--------------------------------------------
In-place halo requires ROW-MAJOR (non-tiled) height-sharded input -- the gate in
should_halo_be_in_place() returns false for tiled inputs. bfloat8_b tensors are ALWAYS
TILE-layout, so a bf8_b *input* would never activate in-place and the test would be
meaningless. To keep in-place ACTIVE while still covering bf8_b, the pool INPUT is always
bf16 row-major height-sharded and the `out_dtype` parameter varies the pool OUTPUT dtype
(bf16 -> ROW_MAJOR output, bf8_b -> TILE output). Halo is pool-type- and output-dtype-
agnostic on the input side, so in-place activates identically for every case here.
"""

import os
import sys
import json
import subprocess
import tempfile

import torch
import pytest


# ---------------------------------------------------------------------------
# Shapes: the height-sharded, row-major-input SAVE shapes that auto-activate in-place
# (IN_PLACE_HALO_REDO.md section 9e -- the confirmed net-L1-saving region).
# ---------------------------------------------------------------------------
SHAPES = [
    # name, NCHW, kernel(h,w), stride(h,w), padding(t,b,l,r)
    ("resnet_150x150_k2s2p0", [1, 128, 150, 150], (2, 2), (2, 2), (0, 0, 0, 0)),
    ("massive_400x544_k3s2p1", [1, 64, 400, 544], (3, 3), (2, 2), (1, 1, 1, 1)),
    ("n8_112x112_k3s2p1", [8, 64, 112, 112], (3, 3), (2, 2), (1, 1, 1, 1)),
    ("n32_264x40_k5s2p2", [32, 32, 264, 40], (5, 5), (2, 2), (2, 2, 2, 2)),
]

POOL_TYPES = ["avg", "max"]
OUT_DTYPES = ["bfloat16", "bfloat8_b"]

L1_SMALL_SIZE = 24576
ACTIVATION_LINE = "in-place halo active"


def _build_structured_input(n, c, h, w):
    """
    Deterministic STRUCTURED input where each stick's value encodes its (n,h,w) position and
    each channel a distinct offset, so a mismatch localizes to an exact stick/channel.

    value(n,h,w,c) = 1.0 + (((n*7 + h*37 + w*13 + c) % 127) + 1) / 128.0

    Every value is a multiple of 1/128 in [1.0078, 1.9922], which is EXACTLY representable in
    bfloat16 (7 mantissa bits in [1,2)) -- so from_torch introduces no rounding and the value
    round-trips exactly. The multipliers (7,37,13) are coprime-ish so consecutive rows/cols
    always shift the value (127 is prime; 37*d for small d never hits a multiple of 127), which
    means any two sticks a small halo-depth apart get DISTINCT values -> a mis-copied stick is
    always visible in the (sum-based) avg-pool output. Values live in [1,2), the region of best
    bf16 resolution, so single-stick corruption survives the pool accumulation.
    """
    n_idx = torch.arange(n).view(n, 1, 1).expand(n, h, w)
    h_idx = torch.arange(h).view(1, h, 1).expand(n, h, w)
    w_idx = torch.arange(w).view(1, 1, w).expand(n, h, w)
    base = (n_idx * 7 + h_idx * 37 + w_idx * 13).reshape(-1, 1)  # (N*H*W, 1)
    c_idx = torch.arange(c).view(1, c)  # (1, C)
    combined = base + c_idx  # (N*H*W, C)
    val = 1.0 + (((combined % 127) + 1).to(torch.float32) / 128.0)
    return val.reshape(1, 1, n * h * w, c)


# ===========================================================================
# WORKER: runs one (shape, out_dtype) x {avg, max} batch in an isolated process.
# The in-place setting is controlled purely by whether TT_METAL_DISABLE_INPLACE_HALO
# is set in this process's environment (read C++-side at op-call time).
# ===========================================================================
def _worker(spec):
    import ttnn

    shape = spec["shape"]
    n, c, h, w = shape
    kernel = tuple(spec["kernel"])
    stride = tuple(spec["stride"])
    pad = tuple(spec["pad"])  # (t, b, l, r)
    out_dtype_str = spec["out_dtype"]
    outdir = spec["outdir"]
    mode = spec["mode"]  # "inplace" or "normal"

    out_dtype = ttnn.bfloat16 if out_dtype_str == "bfloat16" else ttnn.bfloat8_b
    # bf8_b output is only supported with TILE layout; bf16 uses ROW_MAJOR.
    output_layout = ttnn.ROW_MAJOR_LAYOUT if out_dtype_str == "bfloat16" else ttnn.TILE_LAYOUT

    torch_input = _build_structured_input(n, c, h, w)

    results = {}
    device = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL_SIZE)
    try:
        for pool_type in POOL_TYPES:
            key = f"{pool_type}_{out_dtype_str}"
            # Print a marker to stdout for human-readable segmentation (the authoritative
            # activation evidence is the TT_LOGGER_FILE, read by the parent).
            print(f"=== BEGIN {mode} {key} ===", flush=True)
            try:
                # Fresh input EVERY op: in-place halo deallocates/aliases the input buffer, so
                # it must not be reused across ops.
                ttnn_input = ttnn.from_torch(
                    torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                )
                if pool_type == "avg":
                    ttnn_out = ttnn.avg_pool2d(
                        input_tensor=ttnn_input,
                        batch_size=n,
                        input_h=h,
                        input_w=w,
                        channels=c,
                        kernel_size=kernel,
                        stride=stride,
                        padding=list(pad),
                        ceil_mode=False,
                        count_include_pad=True,
                        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        dtype=out_dtype,
                        output_layout=output_layout,
                    )
                else:
                    ttnn_out = ttnn.max_pool2d(
                        input_tensor=ttnn_input,
                        batch_size=n,
                        input_h=h,
                        input_w=w,
                        channels=c,
                        kernel_size=kernel,
                        stride=stride,
                        padding=list(pad),
                        dilation=(1, 1),
                        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        dtype=out_dtype,
                        output_layout=output_layout,
                    )
                torch_out = ttnn.to_torch(ttnn_out).to(torch.float32)
                out_path = os.path.join(outdir, f"{mode}_{key}.pt")
                torch.save(torch_out, out_path)
                results[key] = {"status": "ok", "path": out_path, "shape": list(torch_out.shape)}
                ttnn.deallocate(ttnn_out, force=True)
            except Exception as e:  # noqa: BLE001
                import traceback

                results[key] = {"status": "error", "error": f"{e}", "trace": traceback.format_exc()}
                print(f"CASE ERROR {key}: {e}", flush=True)
            print(f"=== END {mode} {key} ===", flush=True)
    finally:
        ttnn.close_device(device)

    with open(os.path.join(outdir, f"{mode}_results.json"), "w") as f:
        json.dump(results, f)
    print(f"=== WORKER {mode} DONE ===", flush=True)


# ===========================================================================
# PARENT-SIDE helpers
# ===========================================================================
def _run_worker_subprocess(shape_name, shape, kernel, stride, pad, out_dtype, outdir, disable_inplace):
    mode = "normal" if disable_inplace else "inplace"
    spec = {
        "shape": shape,
        "kernel": list(kernel),
        "stride": list(stride),
        "pad": list(pad),
        "out_dtype": out_dtype,
        "outdir": outdir,
        "mode": mode,
    }
    log_file = os.path.join(outdir, f"{mode}_ttlogger.log")

    env = dict(os.environ)
    env.pop("TT_METAL_DISABLE_INPLACE_HALO", None)
    if disable_inplace:
        env["TT_METAL_DISABLE_INPLACE_HALO"] = "1"
    # Route the C++ logger to a per-process file so activation evidence is unambiguous, and
    # ensure INFO level so the log_info() line is emitted.
    env["TT_LOGGER_FILE"] = log_file
    env["TT_LOGGER_LEVEL"] = "info"

    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log_text = ""
    if os.path.exists(log_file):
        with open(log_file, "r", errors="replace") as f:
            log_text = f.read()
    return proc, log_text, mode


def _nan_safe_exact_equal(a, b):
    """True iff a and b are exactly equal, treating NaN-in-same-position as equal."""
    if a.shape != b.shape:
        return False, None, None
    eq = torch.eq(a, b)
    both_nan = torch.isnan(a) & torch.isnan(b)
    mismatch = ~(eq | both_nan)
    if not bool(mismatch.any()):
        return True, None, None
    idx = mismatch.nonzero()
    first = tuple(idx[0].tolist())
    return False, first, int(mismatch.sum())


# ===========================================================================
# TEST
# ===========================================================================
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape_entry", SHAPES, ids=[s[0] for s in SHAPES])
def test_inplace_halo_matches_normal(shape_entry, out_dtype):
    shape_name, shape, kernel, stride, pad = shape_entry
    outdir = tempfile.mkdtemp(prefix=f"inplace_halo_{shape_name}_{out_dtype}_")

    # ACTIVE (in-place auto-activates) and DISABLED (forced normal) in isolated processes.
    proc_active, log_active, _ = _run_worker_subprocess(
        shape_name, shape, kernel, stride, pad, out_dtype, outdir, disable_inplace=False
    )
    proc_normal, log_normal, _ = _run_worker_subprocess(
        shape_name, shape, kernel, stride, pad, out_dtype, outdir, disable_inplace=True
    )

    # Surface worker crashes explicitly (never a silent pass).
    assert proc_active.returncode == 0, (
        f"[{shape_name}/{out_dtype}] ACTIVE worker crashed (rc={proc_active.returncode})\n"
        f"STDOUT:\n{proc_active.stdout[-4000:]}\nSTDERR:\n{proc_active.stderr[-4000:]}"
    )
    assert proc_normal.returncode == 0, (
        f"[{shape_name}/{out_dtype}] NORMAL worker crashed (rc={proc_normal.returncode})\n"
        f"STDOUT:\n{proc_normal.stdout[-4000:]}\nSTDERR:\n{proc_normal.stderr[-4000:]}"
    )

    # --- Log evidence: in-place actually engaged in ACTIVE and NOT in DISABLED. ---
    active_hits = log_active.count(ACTIVATION_LINE)
    normal_hits = log_normal.count(ACTIVATION_LINE)
    print(
        f"[{shape_name}/{out_dtype}] activation-log '{ACTIVATION_LINE}': "
        f"ACTIVE={active_hits}  DISABLED={normal_hits}"
    )
    assert active_hits >= 1, (
        f"[{shape_name}/{out_dtype}] in-place halo did NOT activate in the ACTIVE run "
        f"(0 '{ACTIVATION_LINE}' lines) -> the test would be MEANINGLESS. "
        f"Active log tail:\n{log_active[-3000:]}"
    )
    assert normal_hits == 0, (
        f"[{shape_name}/{out_dtype}] in-place halo unexpectedly ACTIVE in the DISABLED run "
        f"({normal_hits} lines) -> env hook not effective. Disabled log tail:\n{log_normal[-3000:]}"
    )

    # --- Load per-pool results and compare EXACTLY. ---
    with open(os.path.join(outdir, "inplace_results.json")) as f:
        res_active = json.load(f)
    with open(os.path.join(outdir, "normal_results.json")) as f:
        res_normal = json.load(f)

    failures = []
    for pool_type in POOL_TYPES:
        key = f"{pool_type}_{out_dtype}"
        ra = res_active.get(key, {})
        rn = res_normal.get(key, {})
        if ra.get("status") != "ok" or rn.get("status") != "ok":
            failures.append(
                f"  [{key}] ERROR (active={ra.get('status')}, normal={rn.get('status')}): "
                f"{ra.get('error', '')} | {rn.get('error', '')}"
            )
            continue
        a = torch.load(ra["path"])
        b = torch.load(rn["path"])
        equal, first, n_mismatch = _nan_safe_exact_equal(a, b)
        if equal:
            print(f"[{shape_name}/{out_dtype}] {pool_type}: PASS (exact match, shape={list(a.shape)})")
        else:
            va = a[first].item() if first is not None else "N/A"
            vb = b[first].item() if first is not None else "N/A"
            msg = (
                f"  [{key}] FAIL: {n_mismatch} mismatched elements; "
                f"first at {first}: inplace={va} normal={vb} (shapes {list(a.shape)} vs {list(b.shape)})"
            )
            print(f"[{shape_name}/{out_dtype}] {pool_type}: " + msg)
            failures.append(msg)

    assert not failures, (
        f"[{shape_name}/{out_dtype}] in-place halo produced OUTPUT DIVERGENT from normal halo "
        f"(a corruption bug):\n" + "\n".join(failures)
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(json.loads(sys.argv[2]))
        sys.exit(0)
    raise SystemExit("This module is a pytest test; run it under pytest. (--worker is internal.)")
