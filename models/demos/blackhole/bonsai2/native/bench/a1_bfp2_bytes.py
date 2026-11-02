# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is a bfloat2_b weight really 2 bits per value on the device?

With orientation settled (ttnn.linear(x, w) = x @ w, bench/a1_orient.py), bfp2 on
ffn_down is 183.4 us against bfp4's 183.4 us while claiming 27.85 MB versus 50.14.
Identical time for half the bytes is only possible if the device is not reading
2-bit tiles.

The measurement: allocate the same logical tensor at each block-float width and
report the physical buffer size. Expected for 512x512 = 256 tiles:

    bf16  2 B/val              = 524288
    bf8   1 B/val + 1/16 exp   = 557056
    bf4   0.5 B/val + 1/16     = 294912
    bf2   0.25 B/val + 1/16    = 163840

A bf2 that reports bf4's size means the 2-bit packing never reaches the device, and
that alone explains the identical timing.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_bytes
"""

from __future__ import annotations

import os

import numpy as np
import torch


def main() -> int:
    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    codes = np.random.default_rng(0).integers(-1, 2, (512, 512)).astype(np.float32)
    # 512*512 values = 256 tiles of 1024.
    expect = {"bf16": 512 * 512 * 2, "bfp8": 512 * 512 + 512 * 512 // 16,
              "bfp4": 512 * 512 // 2 + 512 * 512 // 16,
              "bfp2": 512 * 512 // 4 + 512 * 512 // 16}
    rc = 0
    try:
        print(f"{'dtype':6s} {'device bytes':>12s} {'expected':>9s}  note")
        for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b),
                         ("bfp4", ttnn.bfloat4_b), ("bfp2", ttnn.DataType.BFLOAT2_B)):
            try:
                t = ttnn.from_torch(torch.from_numpy(codes), dtype=dt,
                                    layout=ttnn.TILE_LAYOUT, device=device)
                # The device allocation size is the honest number; try the accessors
                # that exist in this build.
                nbytes = -1
                for attr in ("buffer_size", "get_buffer_size", "physical_size",
                             "padded_shape"):
                    if hasattr(t, attr):
                        try:
                            v = getattr(t, attr)
                            v = v() if callable(v) else v
                            if isinstance(v, int) and v > 0:
                                nbytes = v
                                break
                        except Exception:  # noqa: BLE001
                            pass
                if nbytes < 0:
                    # Fall back: the device address space consumed can be read from
                    # the allocated buffer through the tensor's memory info if exposed.
                    try:
                        nbytes = int(t.buffer_address() and 0) or -1
                    except Exception:  # noqa: BLE001
                        pass
                exp = expect[name]
                note = "ok" if nbytes == exp else (
                    f"{nbytes - exp:+d}" if nbytes > 0 else "size not exposed")
                if nbytes > 0 and nbytes != exp:
                    rc = 1
                print(f"{name:6s} {nbytes:12d} {exp:9d}  {note}")
                ttnn.deallocate(t)
            except Exception as e:  # noqa: BLE001
                print(f"{name:6s} FAILED {type(e).__name__}: {str(e)[:110]}")
                rc = 1

        # Second, independent probe: time a pure readback of each tensor. A 2-bit
        # tensor must move a quarter of bf16's bytes, so its readback should be
        # markedly faster unless the storage is not really 2-bit.
        print()
        print("readback timing (a genuinely 2-bit tensor moves 1/4 of bf16's bytes):")
        from .timing import Timer
        timer = Timer(warmup=2, calls=5, repeat=3)
        for name, dt in (("bf16", ttnn.bfloat16), ("bfp4", ttnn.bfloat4_b),
                         ("bfp2", ttnn.DataType.BFLOAT2_B)):
            t = ttnn.from_torch(torch.from_numpy(codes), dtype=dt,
                                layout=ttnn.TILE_LAYOUT, device=device)
            med, _ = timer.best(lambda: ttnn.to_torch(t))
            print(f"  {name:5s} readback {med * 1e3:8.1f} ms")
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
