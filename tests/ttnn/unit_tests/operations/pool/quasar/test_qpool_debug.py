# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool DEBUG harness — runs exactly ONE trace per invocation, fully controlled by env vars.

Purpose: a lightweight, repeatable correctness check for ttnn.experimental.quasar.max_pool2d /
avg_pool2d while developing the multi-Tensix (num_threads > 1) pool implementation. Unlike the
sweep suites, this runs a single hand-picked shape with a hand-picked input pattern so a failure
is immediately attributable, and dumps the first mismatching output sticks instead of just a PCC.

Run it via the sibling run_qpool_sim.sh (sets up the craq-sim env), or directly on any backend
(emulator / WH part) with plain pytest and no TT_METAL_SIMULATOR.

ENV VARS (all optional):
    QPOOL_OP       max | avg                                   (default max)
    QPOOL_SHAPE    "H,W,C"   input spatial dims + channels     (default "16,16,64")
    QPOOL_BATCH    batch size                                  (default 1)
    QPOOL_KERNEL   "kh,kw"                                     (default "3,3")
    QPOOL_STRIDE   "sh,sw"                                     (default "2,2")
    QPOOL_PAD      "ph,pw"                                     (default "1,1")
    QPOOL_PATTERN  random | ones | zeros | const:<v> | sticks | channels   (default random)
                   sticks   = every stick (spatial position) is UNIFORM across channels with
                              value = its global stick index (monotonically increasing) — a
                              window's max is exactly its highest-index stick, so wrong-stick
                              selection / off-by-one indexing shows up as an exact wrong integer.
                   channels = value = channel index, constant across space (row-invariant
                              oracle: any spatial mixing bug leaves the output unchanged only
                              if channel routing is intact).
    QPOOL_MOD      wrap deterministic pattern values at this modulus (0 = off, default 0).
                   bf16 represents integers exactly only up to 256; set QPOOL_MOD=256 for
                   bit-exact expectations on big inputs.
    QPOOL_SEED     RNG seed for pattern=random                 (default 0)
    QPOOL_CORES    force the height-shard core count (must divide the height-tile count);
                   default = largest core count that fits the grid and divides evenly.
                   QPOOL_CORES=1 pins everything to a single cluster.
    QPOOL_PCC      PCC threshold (used when golden has variance) (default 0.99)
    QPOOL_RTOL / QPOOL_ATOL   allclose tolerances (default 0.01/0.01 for max, 0.02/0.02 for avg)
    QPOOL_DUMP     how many worst mismatching sticks to print on failure (default 8)

NOTES:
  * N*H*W must be a multiple of 32 (height sharding needs tile-aligned stick counts) and C must
    be a multiple of 32 (shard width == physical tile width).
  * avg + nonzero padding: golden uses torch's default count_include_pad=True, which may not
    match the op's semantics — prefer QPOOL_PAD=0,0 for avg (a warning is printed otherwise).
  * The leak invariant (out.max() <= in.max()) is checked on every run — it is the hard detector
    for stale-L1 / partial-face leaks regardless of pattern.

EXAMPLES:
    # resnet stem trace, monotonic sticks, single cluster:
    QPOOL_SHAPE=112,112,64 QPOOL_PATTERN=sticks QPOOL_CORES=1 ./run_qpool_sim.sh
    # all-ones control, 2x2 window no padding:
    QPOOL_PATTERN=ones QPOOL_KERNEL=2,2 QPOOL_PAD=0,0 ./run_qpool_sim.sh
"""

import os

import pytest
import torch

import ttnn


def _env(name, default):
    return os.environ.get(name, default)


def _env_pair(name, default):
    v = [int(t) for t in _env(name, default).split(",")]
    assert len(v) == 2, f"{name} must be 'a,b'"
    return v


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
    raise ValueError(f"unknown QPOOL_PATTERN={pattern!r}")


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
    is_max = _env("QPOOL_OP", "max") == "max"
    in_h, in_w, channels = [int(t) for t in _env("QPOOL_SHAPE", "16,16,64").split(",")]
    batch = int(_env("QPOOL_BATCH", "1"))
    kernel = _env_pair("QPOOL_KERNEL", "3,3")
    stride = _env_pair("QPOOL_STRIDE", "2,2")
    padding = _env_pair("QPOOL_PAD", "1,1")
    pattern = _env("QPOOL_PATTERN", "random")
    seed = int(_env("QPOOL_SEED", "0"))
    mod = int(_env("QPOOL_MOD", "0"))
    pcc_thresh = float(_env("QPOOL_PCC", "0.99"))
    rtol = float(_env("QPOOL_RTOL", "0.01" if is_max else "0.02"))
    atol = float(_env("QPOOL_ATOL", "0.01" if is_max else "0.02"))
    n_dump = int(_env("QPOOL_DUMP", "8"))

    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1
    tensor_height = batch * in_h * in_w
    assert tensor_height % 32 == 0, f"N*H*W={tensor_height} must be a multiple of 32 (height sharding)"
    assert channels % 32 == 0, f"C={channels} must be a multiple of 32 (shard width = tile width)"
    if not is_max and (padding[0] or padding[1]):
        print("QPOOL WARNING: avg with padding — torch golden uses count_include_pad=True; prefer QPOOL_PAD=0,0")

    # Input: build float pattern, quantize to bf16 FIRST, then compute the golden from the
    # quantized values so bf16 rounding can never masquerade as a device bug.
    x_nhwc = _build_input(pattern, batch, in_h, in_w, channels, seed, mod).to(torch.bfloat16)
    input_max = x_nhwc.float().max().item()
    x_nchw_f = x_nhwc.permute(0, 3, 1, 2).float()
    pool_fn = torch.nn.functional.max_pool2d if is_max else torch.nn.functional.avg_pool2d
    golden_nchw = pool_fn(x_nchw_f, kernel_size=kernel, stride=stride, padding=padding)
    golden = golden_nchw.permute(0, 2, 3, 1).reshape(batch * out_h * out_w, channels).contiguous()

    # Height sharding: forced core count via QPOOL_CORES, else grid-adaptive max.
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    height_tiles = tensor_height // 32
    forced = int(_env("QPOOL_CORES", "0"))
    if forced:
        assert forced <= max_cores, f"QPOOL_CORES={forced} > grid {grid.x}x{grid.y}"
        assert height_tiles % forced == 0, f"QPOOL_CORES={forced} must divide height tiles ({height_tiles})"
        num_cores = forced
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
        f"\nQPOOL: op={'max' if is_max else 'avg'} in={batch}x{in_h}x{in_w}x{channels} "
        f"k={kernel} s={stride} p={padding} out={out_h}x{out_w} pattern={pattern} "
        f"(seed={seed}, mod={mod}) cores={num_cores} shard={shard_height}x{channels}"
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
        f"stale-L1 leak: out.max={got_max:.4f} > in.max={input_max:.4f} " f"(pattern={pattern}, cores={num_cores})"
    )

    # (2) Value check: allclose always; PCC additionally when the golden has variance
    # (PCC is undefined for constant patterns like ones/zeros/const).
    max_diff = (got - golden).abs().max().item()
    close = torch.allclose(got, golden, rtol=rtol, atol=atol)
    pcc = None
    if golden.std() > 0 and got.std() > 0:
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    print(f"QPOOL: max_abs_diff={max_diff:.6f} allclose={close}" + (f" pcc={pcc:.6f}" if pcc is not None else ""))

    ok = close and (pcc is None or pcc >= pcc_thresh)
    if not ok:
        _dump_mismatches(got, golden, out_h, out_w, channels, n_dump)
    assert ok, f"mismatch: max_abs_diff={max_diff:.6f} (rtol={rtol}, atol={atol})" + (
        f", pcc={pcc:.6f} < {pcc_thresh}" if pcc is not None else ""
    )
