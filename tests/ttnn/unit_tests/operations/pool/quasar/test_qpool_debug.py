# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool DEBUG harness — runs exactly ONE trace per invocation, controlled by the CONFIG
block below. Edit the variables, rerun, repeat.

Purpose: a lightweight, repeatable correctness check for ttnn.experimental.quasar.max_pool2d /
avg_pool2d while developing the multi-Tensix (num_threads > 1) pool implementation. Unlike the
sweep suites, this runs a single hand-picked shape with a hand-picked input pattern so a failure
is immediately attributable, and dumps the worst mismatching output sticks instead of just a PCC.

Run it via the sibling run_qpool_sim.sh (sets up the craq-sim env), or directly on any backend
(emulator / WH part) with plain pytest and no TT_METAL_SIMULATOR.

NOTES:
  * N*H*W must be a multiple of 32 (height sharding needs tile-aligned stick counts) and C must
    be a multiple of 32 (shard width == physical tile width).
  * avg + nonzero padding: golden uses torch's default count_include_pad=True, which may not
    match the op's semantics — prefer PADDING = (0, 0) for avg (a warning is printed otherwise).
  * The leak invariant (out.max() <= in.max()) is checked on every run — it is the hard detector
    for stale-L1 / partial-face leaks regardless of pattern.
  * Known craq-sim limitation (2026-08-25): shards >= 256 sticks stall or corrupt the second
    channel tile IN THE SIM ONLY (both pass exactly on WH silicon). Iterate in sim on <= 128-stick
    shards; check bigger shapes on silicon.
"""

import os

import pytest
import torch

import ttnn

# =============================== CONFIG — edit me ===============================
OP = "max"  # "max" | "avg"
BATCH = 1
IN_H, IN_W, CHANNELS = 16, 8, 64  # input spatial dims + channels
# ^ SIM WARNING: keep N*H*W <= 128 sticks for craq-sim runs — 256-stick shards (e.g. 16,16,64)
#   hit the open craq-sim DFB bug and STALL UNTIL THE RUNNER TIMEOUT (fine on WH silicon).
KERNEL = (3, 3)
STRIDE = (2, 2)
PADDING = (1, 1)

# Input pattern: "random" | "ones" | "zeros" | "const:<v>" | "sticks" | "channels"
#   sticks   = every stick (spatial position) is UNIFORM across channels with value = its global
#              stick index (monotonically increasing) — a window's max is exactly its highest-index
#              stick, so wrong-stick selection / off-by-one indexing shows as an exact wrong integer.
#   channels = value = channel index, constant across space (row-invariant oracle: any spatial
#              mixing bug leaves the output unchanged only if channel routing is intact).
PATTERN = "random"
SEED = 0  # RNG seed for PATTERN = "random"
MOD = 0  # wrap deterministic pattern values at this modulus (0 = off). bf16 represents
#          integers exactly only up to 256; set MOD = 256 for bit-exact big inputs.

CORES = 0  # height-shard core count; 0 = largest count that fits the grid and divides the
#            height tiles evenly; 1 pins everything to a single cluster.

PCC_THRESHOLD = 0.99  # applied when the golden has variance (undefined for constant patterns)
RTOL = None  # allclose tolerances; None = per-op default (0.01 for max, 0.02 for avg)
ATOL = None
DUMP = 8  # how many worst mismatching sticks to print on failure
# =================================================================================


def _build_input(pattern, batch, in_h, in_w, channels, seed, mod):
    """Returns an NHWC float32 tensor; caller quantizes to bf16."""
    shape = (batch, in_h, in_w, channels)
    if pattern == "random":
        torch.manual_seed(seed)
        return torch.rand(shape)
    if pattern == "ones":
        return torch.ones(shape)
    if pattern == "zeros":
        return torch.zeros(shape)
    if pattern.startswith("const:"):
        return torch.full(shape, float(pattern.split(":", 1)[1]))
    if pattern == "sticks":
        s = torch.arange(batch * in_h * in_w, dtype=torch.float32).reshape(batch, in_h, in_w, 1)
        if mod:
            s = s % mod
        return s.expand(shape).contiguous()
    if pattern == "channels":
        c = torch.arange(channels, dtype=torch.float32).reshape(1, 1, 1, channels)
        if mod:
            c = c % mod
        return c.expand(shape).contiguous()
    raise ValueError(f"unknown PATTERN={pattern!r}")


def _dump_mismatches(got, golden, out_h, out_w, channels, n_dump):
    """got/golden: (sticks, C) float tensors. Prints the n_dump worst sticks."""
    diff = (got - golden).abs()
    per_stick = diff.max(dim=1).values
    bad = int((per_stick > 0).sum().item())
    n = min(n_dump, bad)
    if n == 0:
        return
    worst = torch.topk(per_stick, n).indices
    print(f"\nQPOOL: {bad}/{got.shape[0]} sticks mismatch; worst {n}:")
    for s in worst.tolist():
        b, r = divmod(s, out_h * out_w)
        oh, ow = divmod(r, out_w)
        ch = int(diff[s].argmax().item())
        k = min(8, channels)
        print(
            f"  stick {s} (b={b}, oh={oh}, ow={ow}) worst ch={ch} "
            f"exp={golden[s, ch].item():.4f} got={got[s, ch].item():.4f} | "
            f"ch0..{k - 1} exp={[round(v, 3) for v in golden[s, :k].tolist()]} "
            f"got={[round(v, 3) for v in got[s, :k].tolist()]}"
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_debug(mesh_device):
    device = mesh_device
    is_max = OP == "max"
    in_h, in_w, channels = IN_H, IN_W, CHANNELS
    batch = BATCH
    kernel = list(KERNEL)
    stride = list(STRIDE)
    padding = list(PADDING)
    rtol = RTOL if RTOL is not None else (0.01 if is_max else 0.02)
    atol = ATOL if ATOL is not None else (0.01 if is_max else 0.02)

    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1
    tensor_height = batch * in_h * in_w
    assert tensor_height % 32 == 0, f"N*H*W={tensor_height} must be a multiple of 32 (height sharding)"
    assert channels % 32 == 0, f"C={channels} must be a multiple of 32 (shard width = tile width)"

    # Guard against the open craq-sim DFB bug: on the SIMULATOR, tensors >= 256 total sticks either
    # stall the halo program until the runner timeout or silently corrupt the second channel tile
    # (verified 2026-08-25: 16x16x64 k3p1 stalls, 16x16x64 k2p0 zeroes ch32+, 16x16x32 k2p0 stalls;
    # every one of these passes EXACTLY on WH silicon). Fail fast here instead of wasting a timeout.
    SIM_MAX_STICKS = 128
    if os.environ.get("TT_METAL_SIMULATOR") and tensor_height > SIM_MAX_STICKS:
        pytest.fail(
            f"CONFIG hits the open craq-sim DFB bug: N*H*W={tensor_height} sticks > {SIM_MAX_STICKS} "
            f"(this shape would stall or corrupt IN THE SIM ONLY). Shrink the shape for sim runs, "
            f"or run this exact config on WH silicon instead: "
            f"TT_METAL_SLOW_DISPATCH_MODE=1 pytest -q -s {__file__}"
        )
    if not is_max and (padding[0] or padding[1]):
        print("QPOOL WARNING: avg with padding — torch golden uses count_include_pad=True; prefer PADDING=(0,0)")

    # Input: build float pattern, quantize to bf16 FIRST, then compute the golden from the
    # quantized values so bf16 rounding can never masquerade as a device bug.
    x_nhwc = _build_input(PATTERN, batch, in_h, in_w, channels, SEED, MOD).to(torch.bfloat16)
    input_max = x_nhwc.float().max().item()
    x_nchw_f = x_nhwc.permute(0, 3, 1, 2).float()
    pool_fn = torch.nn.functional.max_pool2d if is_max else torch.nn.functional.avg_pool2d
    golden_nchw = pool_fn(x_nchw_f, kernel_size=kernel, stride=stride, padding=padding)
    golden = golden_nchw.permute(0, 2, 3, 1).reshape(batch * out_h * out_w, channels).contiguous()

    # Height sharding: forced core count via CORES, else grid-adaptive max.
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    height_tiles = tensor_height // 32
    if CORES:
        assert CORES <= max_cores, f"CORES={CORES} > grid {grid.x}x{grid.y}"
        assert height_tiles % CORES == 0, f"CORES={CORES} must divide height tiles ({height_tiles})"
        num_cores = CORES
    else:
        num_cores = max(c for c in range(1, max_cores + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    print(
        f"\nQPOOL: op={OP} in={batch}x{in_h}x{in_w}x{channels} "
        f"k={kernel} s={stride} p={padding} out={out_h}x{out_w} pattern={PATTERN} "
        f"(seed={SEED}, mod={MOD}) cores={num_cores} shard={shard_height}x{channels}"
    )

    x = ttnn.from_torch(x_nhwc.reshape(1, 1, tensor_height, channels), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    if is_max:
        out = ttnn.experimental.quasar.max_pool2d(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=[1, 1],
        )
    else:
        out = ttnn.experimental.quasar.avg_pool2d(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float().reshape(batch * out_h * out_w, channels)

    # (1) HARD leak invariant: a correct max/avg pool can never exceed the input max.
    got_max = got.max().item()
    assert got_max <= input_max + 1e-2, (
        f"stale-L1 leak: out.max={got_max:.4f} > in.max={input_max:.4f} " f"(pattern={PATTERN}, cores={num_cores})"
    )

    # (2) Value check: allclose always; PCC additionally when the golden has variance
    # (PCC is undefined for constant patterns like ones/zeros/const).
    max_diff = (got - golden).abs().max().item()
    close = torch.allclose(got, golden, rtol=rtol, atol=atol)
    pcc = None
    if golden.std() > 0 and got.std() > 0:
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    print(f"QPOOL: max_abs_diff={max_diff:.6f} allclose={close}" + (f" pcc={pcc:.6f}" if pcc is not None else ""))

    ok = close and (pcc is None or pcc >= PCC_THRESHOLD)
    if not ok:
        _dump_mismatches(got, golden, out_h, out_w, channels, DUMP)
    assert ok, f"mismatch: max_abs_diff={max_diff:.6f} (rtol={rtol}, atol={atol})" + (
        f", pcc={pcc:.6f} < {PCC_THRESHOLD}" if pcc is not None else ""
    )
