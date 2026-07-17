# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar max_pool2d / avg_pool2d CORRECTNESS suite (LLK handoff).

Unlike test_max_pool2d_dprint_debug.py (a DPRINT-coupled single-config repro), this is a clean,
self-contained correctness test: it runs the PRODUCTION quasar kernels (no DEBUG_PRINT needed) and
asserts on PCC vs a torch golden. It is backend-agnostic — grid-adaptive height sharding makes it run
unchanged on the 2-core Quasar emulator, craq-sim, and a full Quasar/WH part.

WHAT IT GUARDS AGAINST (the value-inflation bug):
    The quasar pool reduce over a PARTIAL FACE (face_r_dim < 16, e.g. a 3x3 window -> 9 rows) reads the
    full 16-row face while the reader fills only the populated rows; the un-written rows leak stale L1
    into the reduce. For MAX this can produce an output value ABOVE the input max (impossible for a
    correct max-pool) — a hard, unambiguous detector regardless of magnitude. The leak is config- and
    residue-dependent, so it is masked by a constant input or a single-core/single-buffer in_cb — hence
    this suite sweeps windows / channels / sizes so both the single-core and the multi-core (multi-
    buffered in_cb) paths are exercised, with a random input so leaks are visible.

CHECKS (per case):
    1. HARD leak invariant: got.max() <= input.max() + eps   (a correct max/avg pool never exceeds the
       input max; avg of any window <= max of window <= input max, so this holds for both pool types).
    2. PCC vs torch.nn.functional.{max,avg}_pool2d >= 0.99.

HOW TO RUN (kernel asserts OFF so execution reaches the reduce):
    Quasar craq-sim:
      unset TT_METAL_LLK_ASSERTS; unset TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
      TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
      pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_correctness.py
    Quasar emulator / WH: same pytest line without TT_METAL_SIMULATOR.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (in_h, in_w, channels, kernel, stride, padding) — all sizes chosen so N*H*W is tile-aligned (multiple
# of 32) for clean height sharding. Batch is fixed at 1. Larger H/W shard across >1 core on any real part
# (exercising the multi-buffered in_cb path where the leak appears); 16x16 stays single-core (control).
POOL_CONFIGS = [
    # (in_h, in_w, channels, kernel, stride, padding, id)
    (112, 112, 64, (3, 3), (2, 2), (1, 1), "stem_3x3_64c"),  # resnet50 stem — the known repro
    (56, 56, 64, (3, 3), (2, 2), (1, 1), "56_3x3_64c"),  # multi-core, 64c (2 tiles)
    (16, 16, 64, (3, 3), (2, 2), (1, 1), "16_3x3_64c_1core"),  # single-core control
    (56, 56, 32, (3, 3), (2, 2), (1, 1), "56_3x3_32c"),  # 32c = exactly 1 tile
    (32, 32, 128, (3, 3), (2, 2), (1, 1), "32_3x3_128c"),  # 128c = 4 tiles (wide)
    # NOTE: sub-32-channel configs (e.g. 16c) are intentionally excluded — height sharding
    # requires shard width == physical tile width (32); 16 < 32 fails tensor construction.
    # resnet50 maxpool is always 64ch, so this is not a real path.
    (64, 64, 64, (2, 2), (2, 2), (0, 0), "64_2x2_64c_nopad"),  # 2x2 window (face_r_dim=4), no padding
]


def _run_pool(mesh_device, is_max, in_h, in_w, channels, kernel, stride, padding):
    device = mesh_device
    torch.manual_seed(0)
    batch = 1
    dilation = (1, 1)

    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1

    # Random input in [0,1): a correct pool output is also in [0,1); any out-of-range value is a leak.
    x_nchw = torch.rand((batch, channels, in_h, in_w), dtype=torch.bfloat16)
    input_max = x_nchw.float().max().item()

    if is_max:
        golden_nchw = torch.nn.functional.max_pool2d(
            x_nchw.float(), kernel_size=list(kernel), stride=list(stride), padding=list(padding)
        )
    else:
        # avg with padding=0 so torch's count_include_pad ambiguity can't cause a golden mismatch.
        golden_nchw = torch.nn.functional.avg_pool2d(
            x_nchw.float(), kernel_size=list(kernel), stride=list(stride), padding=list(padding)
        )

    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * in_h * in_w, channels).contiguous()
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * out_h * out_w, channels).contiguous()

    tensor_height = batch * in_h * in_w
    tensor_width = channels
    assert tensor_height % 32 == 0, "test sizes must be tile-aligned in N*H*W"

    # Grid-adaptive HEIGHT sharding: largest core count that fits the device AND evenly tile-divides the
    # height. On a full part this shards across many cores (multi-buffered in_cb); on the 2-core emulator
    # it uses <=2 cores; a size that only divides by 1 stays single-core.
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    height_tiles = tensor_height // 32
    num_cores = max(c for c in range(1, max_cores + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, tensor_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    # max_pool2d and avg_pool2d have different signatures (avg takes output_layout/dtype/compute_kernel_config
    # and NOT dilation), so build the call per pool type.
    if is_max:
        out = ttnn.experimental.quasar.max_pool2d(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=list(kernel),
            stride=list(stride),
            padding=list(padding),
            dilation=list(dilation),
        )
    else:
        out = ttnn.experimental.quasar.avg_pool2d(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=list(kernel),
            stride=list(stride),
            padding=list(padding),
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float().reshape(1, 1, batch * out_h * out_w, channels)

    got_max = got.max().item()
    # (1) HARD leak invariant.
    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(cores={num_cores}, kernel={kernel}, ch={channels}, {in_h}x{in_w})"
    )
    # (2) PCC vs torch golden.
    assert_with_pcc(golden_flat, got, pcc=0.99)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "in_h,in_w,channels,kernel,stride,padding",
    [c[:6] for c in POOL_CONFIGS],
    ids=[c[6] for c in POOL_CONFIGS],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_max_pool2d(mesh_device, in_h, in_w, channels, kernel, stride, padding):
    _run_pool(mesh_device, True, in_h, in_w, channels, kernel, stride, padding)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "in_h,in_w,channels,kernel,stride,padding",
    # AVG regression: unpadded windows only (avoids torch count_include_pad ambiguity). Guards that the
    # forced-MAX-clear change did not disturb the AVG clear-in-loop path.
    [
        (64, 64, 64, (2, 2), (2, 2), (0, 0)),
        (56, 56, 32, (2, 2), (2, 2), (0, 0)),
        (32, 32, 128, (2, 2), (2, 2), (0, 0)),
    ],
    ids=["avg_64_2x2_64c", "avg_56_2x2_32c", "avg_32_2x2_128c"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_avg_pool2d(mesh_device, in_h, in_w, channels, kernel, stride, padding):
    _run_pool(mesh_device, False, in_h, in_w, channels, kernel, stride, padding)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("batch", [1, 16], ids=["b1", "b16"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_avg_pool2d_global(mesh_device, batch):
    """Resnet50 final GLOBAL avg_pool2d: [batch, 2048, 7, 7] -> 1x1.

    This is the ONLY genuinely non-tile-aligned pool shape resnet50 actually runs: the spatial stick
    count is batch*7*7 = 49 (or 784), NOT a multiple of 32, so the op pads it (nearest_32(49)=64). It is
    WIDTH-sharded over the 2048 channels (49 sticks is too few to height-shard) — a different setup than
    the height-sharded cases above. Mirrors the model's fit_width_sharded_cores + nearest_32 (see
    ttnn_functional_resnet50.py avg_pool2d call). Same leak invariant + PCC (looser: bf16 LoFi averaging).
    """
    device = mesh_device
    torch.manual_seed(0)
    channels = 2048
    input_h = input_w = 7
    kernel_size = [input_h, input_w]
    stride = [1, 1]
    padding = [0, 0, 0, 0]
    out_h = out_w = 1

    x_nchw = torch.rand((batch, channels, input_h, input_w), dtype=torch.bfloat16)
    input_max = x_nchw.float().max().item()
    golden_nchw = torch.nn.functional.avg_pool2d(x_nchw.float(), kernel_size=kernel_size, stride=stride, padding=0)
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * out_h * out_w, channels).contiguous()

    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * input_h * input_w, channels).contiguous()
    tensor_height = batch * input_h * input_w  # 49 or 784 -- NOT tile-aligned (padded to nearest_32)
    tensor_width = channels

    # WIDTH_SHARDED, grid-adaptive: largest core count that fits the device AND evenly divides the 64 width
    # tiles. WIDTH_SHARDED keeps the FULL height per core, so the shard height is the tile-padded full
    # height (nearest_32(stick_count)) for the TILE_LAYOUT input to place.
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    width_tiles = tensor_width // 32  # 64
    num_cores = max(c for c in range(1, max_cores + 1) if width_tiles % c == 0)
    shard_height = ((tensor_height + 31) // 32) * 32  # nearest_32
    shard_width = (width_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    out = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
        ),
    )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float().reshape(1, 1, batch * out_h * out_w, channels)
    got_max = got.max().item()
    assert got_max <= input_max + 1e-2, (
        f"global avg_pool2d leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(batch={batch}, cores={num_cores})"
    )
    assert_with_pcc(golden_flat, got, pcc=0.98)  # looser: bf16 LoFi averaging of 49 elements
