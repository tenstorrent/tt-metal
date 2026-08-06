"""Two questions that gate an L1-resident band, answered before any kernel is written.

Q1. The L1_FULL conv1d path returns a wrong answer at C=16 / B=2 (max abs diff 1.456) while C=8 and
    C=512 are bit-exact. 16 is the one width that is neither a full 32-lane tile nor a divisor that
    tiles evenly into one the way 8 does. If padding C to 32 makes it exact, L1_FULL becomes usable
    everywhere and the FIR stops round-tripping DRAM.

Q2. Can the band's intermediates actually live in L1 at all? Probes each op the band uses against an
    L1 height-sharded tensor and reports which accept it. This is what decides whether K5 can be done
    by memory-config plumbing or needs a genuinely new op.
"""

import os
import traceback

import torch

import ttnn

from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter


def q1(device):
    print("=== Q1: does padding C=16 -> 32 make the L1_FULL path exact? ===")
    T_pad, K, stride = 82806, 7, 1
    taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
    torch.manual_seed(0)

    for C in (16, 8, 32):
        x = torch.randn(2, T_pad, C) * 0.3
        outs = {}
        for mode in ("off", "aggressive"):
            os.environ["MINIMAX_H3_AUDIO_CONV1D_L1"] = mode
            os.environ["MINIMAX_H3_AUDIO_DEPTHWISE_MAC"] = "0"
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            outs[mode] = ttnn.to_torch(
                depthwise_tap_filter(xd, taps, stride, mesh_device=device, dtype=ttnn.float32, cache={})
            ).float()
        d = float((outs["off"] - outs["aggressive"]).abs().max())
        print(f"  C={C:<4} bare          maxdiff={d:.3e}  exact={torch.equal(outs['off'], outs['aggressive'])}")

        if C < 32:
            # Same data, channel axis zero-padded to a full tile; compare the real channels only.
            xp = torch.nn.functional.pad(x, (0, 32 - C))
            outs_p = {}
            for mode in ("off", "aggressive"):
                os.environ["MINIMAX_H3_AUDIO_CONV1D_L1"] = mode
                xd = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                outs_p[mode] = ttnn.to_torch(
                    depthwise_tap_filter(xd, taps, stride, mesh_device=device, dtype=ttnn.float32, cache={})
                ).float()[:, :, :C]
            dp = float((outs_p["off"] - outs_p["aggressive"]).abs().max())
            ref = float((outs_p["off"] - outs["off"]).abs().max())
            print(
                f"  C={C:<4} padded to 32  maxdiff={dp:.3e}  exact={torch.equal(outs_p['off'], outs_p['aggressive'])}"
                f"  (pad changes DRAM result by {ref:.3e})"
            )
    print()


def q2(device):
    print("=== Q2: which band ops accept an L1 height-sharded tensor? ===")
    B, T, C = 2, 8192, 32
    x = torch.randn(B, T, C) * 0.3
    xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    grid = device.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    rows = B * T
    per_core = (rows + ncores - 1) // ncores
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
        (per_core, C),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    try:
        xs = ttnn.to_memory_config(ttnn.reshape(xd, (1, 1, rows, C)), mem)
        print(f"  shard to L1 ({ncores} cores, {per_core} rows/core): OK  {xs.memory_config().buffer_type}")
    except Exception as exc:  # noqa: BLE001
        print(f"  shard to L1: FAILED {str(exc).splitlines()[0][:90]}")
        return

    a = ttnn.from_torch(torch.rand(1, 1, C) + 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.rand(1, 1, C) + 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    probes = {
        "to_layout(TILE) sharded": lambda: ttnn.to_layout(xs, ttnn.TILE_LAYOUT),
        "snake_beta on sharded TILE": lambda: ttnn.snake_beta(ttnn.to_layout(xs, ttnn.TILE_LAYOUT), a, b),
        "snake_beta on sharded ROW_MAJOR": lambda: ttnn.snake_beta(xs, a, b),
        "add sharded+sharded": lambda: ttnn.add(xs, xs),
        "concat sharded dim0": lambda: ttnn.concat([xs, xs], dim=2),
        "slice sharded": lambda: ttnn.slice(xs, [0, 0, 0, 0], [1, 1, rows // 2, C]),
    }
    for name, fn in probes.items():
        try:
            out = fn()
            loc = out.memory_config().buffer_type
            print(f"  {name:<34} OK   -> {loc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<34} FAIL {str(exc).splitlines()[0][:80]}")
    print()


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        q1(device)
        q2(device)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
