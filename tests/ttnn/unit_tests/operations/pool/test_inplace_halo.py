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

dtype / layout note
-------------------
In-place halo supports height-sharded input in BOTH layouts:

  * ROW-MAJOR input (skip-untilize path): the classic slice. Covered by
    test_inplace_halo_matches_normal, which keeps the pool INPUT bf16 row-major and varies
    the pool OUTPUT dtype (bf16 -> ROW_MAJOR output, bf8_b -> TILE output).

  * TILED input (untilize-in-place path): halo untilizes the tiled input directly into the
    (overlapping) output buffer. Covered by test_inplace_halo_tiled_matches_normal, which
    feeds TILE-layout input in bf16 and bf8_b (bf8_b exercises class-12: after untilize
    bf8_b becomes bf16, so the remote-temp / untilize-temp CBs are sized for the post-untilize
    bf16 width). Structured values live in [1,2) and are multiples of 1/128, which are EXACTLY
    representable in bf8_b (block exponent 0, 7 mantissa bits) so the tiled comparison stays
    bitwise. The tiled matrix includes a NARROW case (ntiles_per_block <= 8 -> pack_untilize)
    and a WIDE case (ntiles_per_block > 8 -> the untilize.cpp + one-tile-row-at-a-time temp CB).

Both tests assert BITWISE equality (torch.eq) between in-place-active and in-place-disabled
runs of the identical shape/pool/input, in isolated processes, with activation-log proof.
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
    # --- original validated SAVE shapes ---
    ("resnet_150x150_k2s2p0", [1, 128, 150, 150], (2, 2), (2, 2), (0, 0, 0, 0)),
    ("massive_400x544_k3s2p1", [1, 64, 400, 544], (3, 3), (2, 2), (1, 1, 1, 1)),
    ("n8_112x112_k3s2p1", [8, 64, 112, 112], (3, 3), (2, 2), (1, 1, 1, 1)),
    ("n32_264x40_k5s2p2", [32, 32, 264, 40], (5, 5), (2, 2), (2, 2, 2, 2)),
    # --- broadened coverage: large-feature-map + small-kernel candidates that were verified
    #     to auto-activate in-place (the "in-place halo active" log appears in the ACTIVE run).
    #     Two proposed candidates were DROPPED because in-place did NOT activate for them
    #     (gate verdict LOSE -- too few sticks/core relative to halo depth), so recording them
    #     would be meaningless:
    #       * [1, 512, 56, 56] k3s2p1  -> did NOT activate (dropped)
    #       * [1, 96, 180, 320] k3s2p1 -> did NOT activate (dropped)
    ("c256_112x112_k2s2p0", [1, 256, 112, 112], (2, 2), (2, 2), (0, 0, 0, 0)),
    ("c64_300x300_k3s2p1", [1, 64, 300, 300], (3, 3), (2, 2), (1, 1, 1, 1)),
    ("n2_c96_160x160_k3s1p1", [2, 96, 160, 160], (3, 3), (1, 1), (1, 1, 1, 1)),
    ("c128_224x224_k3s2p1", [1, 128, 224, 224], (3, 3), (2, 2), (1, 1, 1, 1)),
    ("n4_c32_128x128_k2s2p0", [4, 32, 128, 128], (2, 2), (2, 2), (0, 0, 0, 0)),
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
    # Pool INPUT dtype/layout. Defaults keep the classic row-major bf16 slice; the tiled test
    # overrides these to drive the untilize-in-place path.
    in_dtype_str = spec.get("in_dtype", "bfloat16")
    in_layout_str = spec.get("in_layout", "row_major")
    # Sharding scheme applied to the (dense) input tensor: height / width / block. Width-sharded
    # halo is all-local (max_ref_size==0, no remote-temp CB); block-sharded uses the 2D grid +
    # column-major NOC orientation. All three are validated corruption-safe.
    shard_scheme_str = spec.get("shard_scheme", "height")
    shard_scheme = {
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }[shard_scheme_str]

    out_dtype = ttnn.bfloat16 if out_dtype_str == "bfloat16" else ttnn.bfloat8_b
    # bf8_b output is only supported with TILE layout; bf16 uses ROW_MAJOR.
    output_layout = ttnn.ROW_MAJOR_LAYOUT if out_dtype_str == "bfloat16" else ttnn.TILE_LAYOUT

    in_dtype = ttnn.bfloat16 if in_dtype_str == "bfloat16" else ttnn.bfloat8_b
    in_layout = ttnn.TILE_LAYOUT if in_layout_str == "tile" else ttnn.ROW_MAJOR_LAYOUT

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
                ttnn_input = ttnn.from_torch(torch_input, dtype=in_dtype, layout=in_layout, device=device)
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
                        applied_shard_scheme=shard_scheme,
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
                        applied_shard_scheme=shard_scheme,
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
def _run_worker_subprocess(
    shape_name,
    shape,
    kernel,
    stride,
    pad,
    out_dtype,
    outdir,
    disable_inplace,
    in_dtype="bfloat16",
    in_layout="row_major",
    shard_scheme="height",
):
    mode = "normal" if disable_inplace else "inplace"
    spec = {
        "shape": shape,
        "kernel": list(kernel),
        "stride": list(stride),
        "pad": list(pad),
        "out_dtype": out_dtype,
        "outdir": outdir,
        "mode": mode,
        "in_dtype": in_dtype,
        "in_layout": in_layout,
        "shard_scheme": shard_scheme,
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
# Shared core: run ACTIVE vs DISABLED in isolated processes and assert bitwise equality.
# `require_untilize` (None | "pack_untilize" | "untilize-temp-CB") additionally asserts the
# tiled untilize-in-place path taken (grep of the factory's diagnostic log line).
# ===========================================================================
def _run_and_compare(
    tag, shape, kernel, stride, pad, out_dtype, in_dtype, in_layout, require_untilize=None, shard_scheme="height"
):
    outdir = tempfile.mkdtemp(prefix=f"inplace_halo_{tag.replace('/', '_')}_")

    # ACTIVE (in-place auto-activates) and DISABLED (forced normal) in isolated processes.
    proc_active, log_active, _ = _run_worker_subprocess(
        tag,
        shape,
        kernel,
        stride,
        pad,
        out_dtype,
        outdir,
        disable_inplace=False,
        in_dtype=in_dtype,
        in_layout=in_layout,
        shard_scheme=shard_scheme,
    )
    proc_normal, log_normal, _ = _run_worker_subprocess(
        tag,
        shape,
        kernel,
        stride,
        pad,
        out_dtype,
        outdir,
        disable_inplace=True,
        in_dtype=in_dtype,
        in_layout=in_layout,
        shard_scheme=shard_scheme,
    )

    # Surface worker crashes explicitly (never a silent pass).
    assert proc_active.returncode == 0, (
        f"[{tag}] ACTIVE worker crashed (rc={proc_active.returncode})\n"
        f"STDOUT:\n{proc_active.stdout[-4000:]}\nSTDERR:\n{proc_active.stderr[-4000:]}"
    )
    assert proc_normal.returncode == 0, (
        f"[{tag}] NORMAL worker crashed (rc={proc_normal.returncode})\n"
        f"STDOUT:\n{proc_normal.stdout[-4000:]}\nSTDERR:\n{proc_normal.stderr[-4000:]}"
    )

    # --- Log evidence: in-place actually engaged in ACTIVE and NOT in DISABLED. ---
    active_hits = log_active.count(ACTIVATION_LINE)
    normal_hits = log_normal.count(ACTIVATION_LINE)
    print(f"[{tag}] activation-log '{ACTIVATION_LINE}': ACTIVE={active_hits}  DISABLED={normal_hits}")
    assert active_hits >= 1, (
        f"[{tag}] in-place halo did NOT activate in the ACTIVE run "
        f"(0 '{ACTIVATION_LINE}' lines) -> the test would be MEANINGLESS. "
        f"Active log tail:\n{log_active[-3000:]}"
    )
    assert normal_hits == 0, (
        f"[{tag}] in-place halo unexpectedly ACTIVE in the DISABLED run "
        f"({normal_hits} lines) -> env hook not effective. Disabled log tail:\n{log_normal[-3000:]}"
    )

    # --- Grid-geometry evidence: does this shape exercise NOOP cores / a partial grid? ---
    grid_line = ""
    for ln in log_active.splitlines():
        if "in-place halo grid:" in ln:
            grid_line = ln[ln.index("in-place halo grid:") :]
            break
    print(f"[{tag}] {grid_line or 'in-place halo grid: <not found>'}")

    # --- Untilize-path evidence (tiled input only): prove pack_untilize vs the wide temp-CB path. ---
    if require_untilize is not None:
        untilize_line = ""
        for ln in log_active.splitlines():
            if "in-place halo untilize:" in ln:
                untilize_line = ln[ln.index("in-place halo untilize:") :]
                break
        print(f"[{tag}] {untilize_line or 'in-place halo untilize: <not found>'}")
        assert f"path={require_untilize}" in untilize_line, (
            f"[{tag}] expected untilize path '{require_untilize}' but got: "
            f"'{untilize_line or '<not found>'}' -> the intended narrow/wide path was NOT exercised."
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
            print(f"[{tag}] {pool_type}: PASS (exact match, shape={list(a.shape)})")
        else:
            va = a[first].item() if first is not None else "N/A"
            vb = b[first].item() if first is not None else "N/A"
            msg = (
                f"  [{key}] FAIL: {n_mismatch} mismatched elements; "
                f"first at {first}: inplace={va} normal={vb} (shapes {list(a.shape)} vs {list(b.shape)})"
            )
            print(f"[{tag}] {pool_type}: " + msg)
            failures.append(msg)

    assert (
        not failures
    ), f"[{tag}] in-place halo produced OUTPUT DIVERGENT from normal halo (a corruption bug):\n" + "\n".join(failures)


# ===========================================================================
# TEST: row-major input (skip-untilize path) -- the original validated slice.
# ===========================================================================
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape_entry", SHAPES, ids=[s[0] for s in SHAPES])
def test_inplace_halo_matches_normal(shape_entry, out_dtype):
    shape_name, shape, kernel, stride, pad = shape_entry
    _run_and_compare(
        f"{shape_name}/{out_dtype}", shape, kernel, stride, pad, out_dtype, in_dtype="bfloat16", in_layout="row_major"
    )


# ---------------------------------------------------------------------------
# TILED-input SAVE shapes (untilize-in-place path). Each entry declares the expected untilize
# path so the test proves the intended narrow (pack_untilize) vs WIDE (untilize-temp-CB) branch.
# All are height-sharded and net-L1 SAVE (widening channels is width-independent for the gate, so
# a widened known-SAVE shape stays SAVE). ntiles_per_block = ceil(channels/32); WIDE iff > 8.
# ---------------------------------------------------------------------------
TILED_SHAPES = [
    # name, NCHW, kernel, stride, pad(t,b,l,r), expected untilize path.
    # SAVE verdict confirmed via the gate (in/core vs max_ref). The 150x150 k2s2p0 geometry
    # (max_ref=176 << in/core=352) is a robust SAVE region; channels are widened to reach the
    # narrow(<=8 tiles)/WIDE(>8 tiles) untilize branches without changing the SAVE verdict
    # (the gate is channel-width-independent).
    ("resnet_150x150_k2s2p0", [1, 128, 150, 150], (2, 2), (2, 2), (0, 0, 0, 0), "pack_untilize"),  # 4 tiles, narrow
    ("n8_112x112_k3s2p1", [8, 64, 112, 112], (3, 3), (2, 2), (1, 1, 1, 1), "pack_untilize"),  # 2 tiles, padded, fwd/rev
    ("c256_150x150_k2s2p0", [1, 256, 150, 150], (2, 2), (2, 2), (0, 0, 0, 0), "pack_untilize"),  # 8 tiles (boundary)
    ("wide320_150x150_k2s2p0", [1, 320, 150, 150], (2, 2), (2, 2), (0, 0, 0, 0), "untilize-temp-CB"),  # 10 tiles, WIDE
]

# Tiled-input dtypes: bf16 (untilize keeps bf16) and bf8_b (class-12: bf8_b -> bf16 after untilize).
TILED_IN_DTYPES = ["bfloat16", "bfloat8_b"]


# ===========================================================================
# TEST: tiled input (untilize-in-place path) -- Part B. Output kept bf16 ROW_MAJOR for the most
# sensitive (no output-quantization) bitwise comparison; the tiled-ness is on the INPUT side.
# ===========================================================================
@pytest.mark.parametrize("in_dtype", TILED_IN_DTYPES)
@pytest.mark.parametrize("shape_entry", TILED_SHAPES, ids=[s[0] for s in TILED_SHAPES])
def test_inplace_halo_tiled_matches_normal(shape_entry, in_dtype):
    shape_name, shape, kernel, stride, pad, expected_untilize = shape_entry
    _run_and_compare(
        f"{shape_name}/tiled_{in_dtype}",
        shape,
        kernel,
        stride,
        pad,
        out_dtype="bfloat16",
        in_dtype=in_dtype,
        in_layout="tile",
        require_untilize=expected_untilize,
    )


# ===========================================================================
# WIDTH-SHARDED SAVE shapes (full-pool-support widening).
#
# Width sharding splits CHANNELS across cores, so each core holds ALL of NHW spatial. The halo
# is therefore ENTIRELY LOCAL (no cross-core remote transfers): max_ref_size == 0, so the
# remote-temp CB is skipped and the net-L1 gate always SAVEs (0 < 0.75*in_nsticks). The
# in-place overlap-ordering still applies WITHIN each core's own shard. Both row-major
# (skip-untilize) and tiled (untilize-in-place / pack_untilize) inputs are covered.
# WH-probed: all activate (activate=1), full 8x8 grid, no crash/hang. The all-local / max_ref==0
# path is what these exercise (no remote-temp CB).
# ---------------------------------------------------------------------------
WIDTH_SHARD_SHAPES = [
    # name, NCHW, kernel, stride, pad(t,b,l,r), in_layout
    ("ws_c8192_12x12_k3s1p1", [1, 8192, 12, 12], (3, 3), (1, 1), (1, 1, 1, 1), "row_major"),  # RM skip-untilize, padded
    ("ws_c8192_12x12_k3s1p1", [1, 8192, 12, 12], (3, 3), (1, 1), (1, 1, 1, 1), "tile"),  # tiled untilize-in-place
    ("ws_c16384_8x8_k2s1p0", [1, 16384, 8, 8], (2, 2), (1, 1), (0, 0, 0, 0), "tile"),  # tiled, 8-tile boundary
]


@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape_entry", WIDTH_SHARD_SHAPES, ids=[f"{s[0]}_{s[5]}" for s in WIDTH_SHARD_SHAPES])
def test_inplace_halo_width_sharded_matches_normal(shape_entry, out_dtype):
    shape_name, shape, kernel, stride, pad, in_layout = shape_entry
    _run_and_compare(
        f"{shape_name}/ws_{in_layout}/{out_dtype}",
        shape,
        kernel,
        stride,
        pad,
        out_dtype,
        in_dtype="bfloat16",
        in_layout=in_layout,
        shard_scheme="width",
    )


# ===========================================================================
# BLOCK-SHARDED SAVE shapes (full-pool-support widening).
#
# Block sharding uses a 2D core grid (channels split across one axis, NHW across the other) and
# the column-major / transpose_mcast NOC orientation machinery. The overlap-aliasing + delta and
# the stage-to-temp -> barrier -> distribute ordering must still hold per core. Small kernels on
# modest spatial maps are net-L1 SAVE (halo depth < per-core input sticks); large kernels LOSE.
#
# The net-L1 gate decision depends on the exact per-core geometry, which shifts with the OUTPUT
# layout: bf8_b forces a TILE output whose per-core NHW is padded to 32-stick multiples, shrinking
# the per-core input-stick count relative to the (layout-invariant) halo depth. So small-spatial
# 16x16 shapes SAVE with bf16 (row-major) output but LOSE with bf8_b (tiled) output. Each entry
# therefore lists exactly the OUTPUT dtypes it was WH-probed to ACTIVATE for; combos that did NOT
# activate (net-L1 LOSE -- gate correctly declined, NOT a bug) are intentionally excluded:
#     * [1, 2048, 16, 16] k5s2p2 ceil (any dtype) -> LOSE (halo depth >> per-core sticks).
#     * [1, 2048, 16, 16] k2s1p0 + bf8_b, [1, 4096, 16, 16] k2s1p0 + bf8_b -> LOSE (tile-pad shrinks sticks).
# NOTE: the pool caller shards block inputs ROW_MAJOR (orientation is forced), so transpose_mcast
# (column-major block NOC) is not reachable through pool here; it stays structurally supported
# (gate does not exclude it; kernel has the is_col_major noc_orient path) but is not bitwise-
# exercised by this pool-only harness.
# ---------------------------------------------------------------------------
BLOCK_SHARD_SHAPES = [
    # name, NCHW, kernel, stride, pad(t,b,l,r), in_layout, [activating out_dtypes]
    # larger-spatial shapes SAVE for BOTH output dtypes:
    ("bs_c1024_32x32_k3s1p1", [1, 1024, 32, 32], (3, 3), (1, 1), (1, 1, 1, 1), "row_major", ["bfloat16", "bfloat8_b"]),
    ("bs_c1024_32x32_k3s1p1", [1, 1024, 32, 32], (3, 3), (1, 1), (1, 1, 1, 1), "tile", ["bfloat16", "bfloat8_b"]),
    ("bs_c512_64x64_k3s2p1", [1, 512, 64, 64], (3, 3), (2, 2), (1, 1, 1, 1), "tile", ["bfloat16", "bfloat8_b"]),
    # 48x48 bf8_b lands on a PARTIAL block grid (8x6, 48 cores) -> exercises block grid geometry:
    ("bs_c1024_48x48_k3s2p1", [1, 1024, 48, 48], (3, 3), (2, 2), (1, 1, 1, 1), "tile", ["bfloat16", "bfloat8_b"]),
    # WIDE untilize-in-place path (>8 tiles -> untilize-temp-CB); SAVE only with bf16 output:
    ("bs_c4096_16x16_k2s1p0", [1, 4096, 16, 16], (2, 2), (1, 1), (0, 0, 0, 0), "tile", ["bfloat16"]),
]

# Expand (shape_entry, out_dtype) into a flat param list so only ACTIVATING combos are exercised.
_BLOCK_PARAMS = [
    (name, shape, kernel, stride, pad, in_layout, od)
    for (name, shape, kernel, stride, pad, in_layout, ods) in BLOCK_SHARD_SHAPES
    for od in ods
]


@pytest.mark.parametrize(
    "block_param",
    _BLOCK_PARAMS,
    ids=[f"{p[0]}_{p[5]}_{p[6]}" for p in _BLOCK_PARAMS],
)
def test_inplace_halo_block_sharded_matches_normal(block_param):
    shape_name, shape, kernel, stride, pad, in_layout, out_dtype = block_param
    _run_and_compare(
        f"{shape_name}/bs_{in_layout}/{out_dtype}",
        shape,
        kernel,
        stride,
        pad,
        out_dtype,
        in_dtype="bfloat16",
        in_layout=in_layout,
        shard_scheme="block",
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(json.loads(sys.argv[2]))
        sys.exit(0)
    raise SystemExit("This module is a pytest test; run it under pytest. (--worker is internal.)")
