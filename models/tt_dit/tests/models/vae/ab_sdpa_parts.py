"""Split ONE fused-GNA-SDPA call into gather / mask-window / mask-fill / compute, by ablation.

The op is called DIRECTLY on synthetic Q/K/V -- no decoder, no checkpoint, no CCL, no permutes. The
grid is chosen so a single device sees exactly the per-device work of the sharded production path
(Q rows, Q-chunk count and neighborhood box all match), which makes the *ratios* here the same ones
that matter in the real decode while removing everything that is not the op.

Why ablation rather than the device profiler: the per-RISC columns Tracy reports are kernel SPANS, so
a RISC parked in ``cb_wait_front`` reads as fully busy and the split cannot be recovered from them.
Removing a stage and re-timing measures its cost directly. The three kernel probes each keep every
reserve/push in lockstep, so the CB contract -- and therefore the schedule -- is unchanged when a
stage is switched off.

Each arm patches the probe ``#define``s in the kernel sources and relies on the JIT dephash to
rebuild (headers are tracked per-object, see tt_metal/impl/kernels/kernel.cpp). The sources are
ALWAYS restored, including on exception: a probe left at 1 silently produces garbage pixels for
every later run in the workspace.

What this does NOT produce is a breakdown that sums to the total. Reader, writer and compute run
concurrently on one core, so the op is the critical path through a producer/consumer pipeline: taking
a stage out does not subtract its work, it promotes whatever was second to bottleneck. Each arm
therefore answers "what would this op cost if stage X were free" -- the right question for deciding
what to optimize, and a LOWER bound on X's own cost, reading 0 for any stage already hidden behind
another. The remaining floor is not "compute": it holds Q reads, output writes, both matmuls,
softmax and all CB/dispatch overhead, none of which has a probe yet.

Env: ``ITERS`` (default 10), ``GRID_T``/``GRID_H``/``GRID_W`` (default the production per-device
shape), ``BLOCK`` and ``STRIDE`` as physical "t,h,w".
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import torch

import ttnn

KERNELS = Path("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow")
MASK_SRC = KERNELS / "windowed_mask_gen.hpp"
READER_SRC = KERNELS / "reader_interleaved.cpp"
PROBE_FILE = {
    "SDPA_MASK_FLOOR_PROBE": MASK_SRC,
    "SDPA_MASK_FILL_PROBE": MASK_SRC,
    "SDPA_MASK_NEGINF_PROBE": MASK_SRC,
    "SDPA_GATHER_FLOOR_PROBE": READER_SRC,
}

ITERS = int(os.environ.get("ITERS", 10))
# Default is stage5_kernel from the checkpoint metadata. A wider kernel is a different FUNCTION, not a
# schedule knob -- but it is the only way to make the box a whole number of tiles, which is what lets
# the single-window fill drop to a pure NOC zero-fill with no packed tail to -inf.
KERNEL = tuple(int(v) for v in os.environ.get("KERNEL", "11,11,11").split(","))
#: K chunk in tokens. Sk_chunk_t = KCHUNK/32 sets how many keys one gather+mask chunk covers, so it
#: divides the per-chunk fixed cost -- and a box that is a whole multiple of it has no partial tail.
KCHUNK = int(os.environ.get("KCHUNK", 32))
HEAD_DIM = 64

# Physical (t, h, w). w is the per-device W shard: the production path all-gathers K/V to full W, but
# an unsharded run with w == w_local gives the same Q chunks and the same interior box, and interior
# blocks are what the ratios are about.
GRID = (int(os.environ.get("GRID_T", 145)), int(os.environ.get("GRID_H", 272)), int(os.environ.get("GRID_W", 60)))
BLOCK = tuple(int(v) for v in os.environ.get("BLOCK", "5,8,12").split(","))
STRIDE = tuple(int(v) for v in os.environ.get("STRIDE", "5,8,4").split(","))

# (label, probes to set at 1). Removal is cumulative down the list.
ARMS = [
    ("full", ()),
    ("no mask fill", ("SDPA_MASK_FILL_PROBE",)),
    ("no neginf blanket", ("SDPA_MASK_NEGINF_PROBE",)),
    ("no mask at all", ("SDPA_MASK_FLOOR_PROBE",)),
    ("no gather", ("SDPA_GATHER_FLOOR_PROBE",)),
    ("no mask + no gather", ("SDPA_MASK_FLOOR_PROBE", "SDPA_GATHER_FLOOR_PROBE")),
]


def _read(path: Path) -> str:
    return path.read_text()


def _set_probes(active: tuple[str, ...]) -> None:
    """Rewrite every probe define to 0, then raise the requested ones to 1."""
    for path in {MASK_SRC, READER_SRC}:
        text = _read(path)
        for name in PROBE_FILE:
            if PROBE_FILE[name] != path:
                continue
            want = 1 if name in active else 0
            text, n = re.subn(rf"^#define {name} \d+$", f"#define {name} {want}", text, flags=re.M)
            assert n == 1, f"expected exactly one '#define {name}' in {path}, found {n}"
        path.write_text(text)


def _assert_probes_clean() -> None:
    for name, path in PROBE_FILE.items():
        assert re.search(rf"^#define {name} 0$", _read(path), flags=re.M), f"{name} not restored in {path}"


def _ext(b: int, s: int, k: int) -> int:
    """One axis of neighborhood_box_block's extent for an interior block (windowed_loop_geometry.hpp)."""
    return ((b - 1) // s) * s + k


def _check_legal() -> None:
    t, h, w = GRID
    for axis, name, b, s, k in zip((t, h, w), "thw", BLOCK, STRIDE, KERNEL):
        assert axis % b == 0, f"block {name}={b} does not divide {name}={axis}"
        assert axis % s == 0, f"stride {name}={s} does not divide {name}={axis}"
        assert s <= min(k, axis), f"stride {name}={s} exceeds effective kernel {min(k, axis)}"
    vol = BLOCK[0] * BLOCK[1] * BLOCK[2]
    assert vol % 32 == 0 and 32 <= vol <= 512, f"block vol {vol} must be a multiple of 32 in [32, 512]"


def run_arm(device, q, k, v, off, prog_config, label: str, probes: tuple[str, ...]) -> float:
    _set_probes(probes)
    t_full, h_full, w_full = GRID
    # Op-axis order is (w, h, t) -- to_seq flattens W-outer with T innermost -- so the physical grid,
    # kernel, block and stride all permute the same way na3d permutes them.
    nbr = (w_full, h_full, t_full, KERNEL[2], KERNEL[1], KERNEL[0])
    op_block = (BLOCK[2], BLOCK[1], BLOCK[0])
    op_stride = (STRIDE[2], STRIDE[1], STRIDE[0])

    def call():
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            neighborhood_3d=nbr,
            neighborhood_gather=True,
            neighborhood_block=op_block,
            neighborhood_stride=op_stride,
            scale=HEAD_DIM**-0.5,
            windowed_q_token_offset=0,
            windowed_q_token_offset_tensor=off,
            program_config=prog_config,
        )

    for _ in range(2):  # program cache: probe set and shapes are both part of its key
        ttnn.deallocate(call())
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.deallocate(call())
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"[{label:22s}] {ms:9.2f} ms/call", flush=True)
    return ms


def main() -> None:
    _check_legal()
    t_full, h_full, w_full = GRID
    sites = t_full * h_full * w_full
    vol = BLOCK[0] * BLOCK[1] * BLOCK[2]
    box = (
        _ext(BLOCK[0], STRIDE[0], KERNEL[0])
        * _ext(BLOCK[1], STRIDE[1], KERNEL[1])
        * _ext(BLOCK[2], STRIDE[2], KERNEL[2])
    )
    print(
        f"\n=== fused-SDPA parts · grid {GRID} · block {BLOCK} · stride {STRIDE} · kernel {KERNEL} ===\n"
        f"    {sites} sites · vol {vol} ({sites // vol} q-chunks) · box {box} · box/vol {box / vol:.2f}\n"
        f"    kchunk {KCHUNK} -> {(box + KCHUNK - 1) // KCHUNK} packed chunks/q-chunk, tail {box % KCHUNK} · {ITERS} iters",
        flush=True,
    )

    # open_device inside the try: a faulted device throws here, and if that happened outside the try the
    # finally below would not run -- leaving a probe set at 1 for every later run in the workspace.
    device = None
    try:
        device = ttnn.open_device(device_id=0)
        g = torch.Generator().manual_seed(0)
        t_full, h_full, w_full = GRID

        def dev(x: torch.Tensor, layout) -> ttnn.Tensor:
            return ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)

        # Q is TILE (B, NH, S, D) with one head -- production runs TP=4 over 4 heads, so a device sees
        # one. K/V must be ROW_MAJOR paged by the innermost axis: (w,h) pages of one T-row each, which is
        # what the in-kernel gather slices (sdpa_device_operation.cpp:45 rejects TILE K/V here).
        q = dev(torch.randn(1, 1, sites, HEAD_DIM, generator=g), ttnn.TILE_LAYOUT)
        kv = [
            dev(torch.randn(1, 1, w_full * h_full, t_full * HEAD_DIM, generator=g), ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(2)
        ]
        off = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        # q_chunk_size MUST be the block volume: under neighborhood_block a Q chunk IS one block, and the
        # geometry helpers derive the block index from the chunk index.
        grid = device.compute_with_storage_grid_size()
        prog_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False,
            q_chunk_size=vol,
            k_chunk_size=KCHUNK,
        )

        # ONE arm per process. ttnn's program cache is keyed on op attributes and tensor specs, not on
        # kernel source, so a second arm in the same process gets handed arm 1's already-built program
        # and the probe edit is never compiled -- every arm then times the same binary. A fresh process
        # is the only way to be sure the source that was patched is the source that ran.
        want = os.environ["ARM"]
        label, probes = next((l, p) for l, p in ARMS if l == want)
        ms = run_arm(device, q, kv[0], kv[1], off, prog_config, label, probes)
        print(f"RESULT\t{label}\t{ms:.2f}", flush=True)
    finally:
        _set_probes(())
        _assert_probes_clean()
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
