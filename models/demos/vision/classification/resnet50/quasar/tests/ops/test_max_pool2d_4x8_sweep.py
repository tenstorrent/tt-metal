# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC (throwaway): SWEEP of the quasar max_pool2d with a 4x8 kernel and a (2,4) stride.

WINDOW / STRIDE: kernel=(4,8) => window_size_hw = 32 = a FULL tile (num_faces=4), so the partial-face
reduce path (face_r_dim<16, tail-fill, clear) is NOT exercised here. stride=(2,4) < kernel => the
windows OVERLAP (like a real resnet maxpool), so the halo uop actually gathers cross-row/col overlap
(unlike the old kernel==stride version, where each window was one contiguous tile with no halo).

ONE test, driven by two switchable lists:
  * SWEEP        -- (in_h, in_w, channels, num_cores) geometry cases.
  * INPUT_MODES  -- how the input is generated (comment a mode out to switch it off):
      - "chan_inc": input channel c = value (c+1) everywhere => every output stick must be [1..C]
                    regardless of the spatial gather. The original deterministic oracle.
      - "randn":    torch.randn (seed 0). Random data makes wrong/zero sticks visible even in range.
      - "pos_inc":  each spatial position holds ONE value, identical across all C channels, increasing
                    with position (pos = h*in_w + w + 1). Every output stick is constant across channels
                    and equals the max window position -> isolates a wrong SPATIAL gather (vs channel mix).

The golden is always torch.nn.functional.max_pool2d, and max-pool is exact value-SELECTION (no arithmetic),
so a correct op reproduces the golden per stick EXACTLY for every mode. The per-stick check classifies each
stick correct/zero/wrong and tags reader0 (even output col) vs reader1 (odd output col) -- this is the
diagnostic that surfaced the "odd sticks are zero" split-reader bug. Plus a HARD leak invariant
(got.max() <= input.max()) that catches stale-L1 value inflation.

SWEPT AXES: (in_h, in_w, channels, num_cores) x input_mode. num_cores is EXPLICIT (not grid-adaptive) so
the same geometry runs single-core vs multi-core; it must satisfy num_cores <= device cores (else skipped)
and evenly tile-divide the sharded height (else a hard config error).

The split reader is HARDCODED on in the factory (split_reader=true): out_w==1 => reader1 idle;
out_w>=2 => reader1 handles the odd output columns.

RUN (kernel asserts + watcher OFF; use the wrapper):
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_4x8_sweep.py
    ./qsr_sim_run ".../test_max_pool2d_4x8_sweep.py" -k "randn"
"""
import pytest
import torch

import ttnn

KERNEL = (4, 8)
STRIDE = (2, 4)
PADDING = (0, 0)
DILATION = (1, 1)

# (in_h, in_w, channels, num_cores, id). Sizes chosen so in_h*in_w is tile-aligned (mult of 32),
# (in_w-8)%4==0 and (in_h-4)%2==0 (clean out_h/out_w), channels a mult of 32, and num_cores divides the
# height tiles (in_h*in_w/32). out_h=(in_h-4)//2+1, out_w=(in_w-8)//4+1.
SWEEP = [
    # --- no split (out_w==1): reader1 idle ---
    (4, 20, 64, 1, "in4x8_C64_k1__out1x1_nosplit"),  # 1 tile -> 1 core only
    # (8, 8, 64, 1, "in8x8_C64_k1__out3x1_nosplit"),  # 2 tiles, single core
    # (8, 8, 64, 2, "in8x8_C64_k2__out3x1_nosplit"),  # 2 tiles across 2 cores
    # # --- split active (out_w>=2): the reader1 path ---
    # (8, 12, 64, 1, "in8x12_C64_k1__out3x2_split"),  # 3 tiles, single core
    # (8, 12, 64, 3, "in8x12_C64_k3__out3x2_split"),  # 3 tiles across 3 cores
    # (8, 16, 64, 1, "in8x16_C64_k1__out3x3_split"),  # 4 tiles, single core
    # (8, 16, 64, 4, "in8x16_C64_k4__out3x3_split"),  # 4 tiles across 4 cores
    # (16, 12, 64, 6, "in16x12_C64_k6__out7x2_split"),  # 6 tiles across 6 cores
    # (16, 16, 64, 8, "in16x16_C64_k8__out7x3_split"),  # 8 tiles across 8 cores
    # # --- larger / multi-core ---
    # (32, 32, 64, 8, "in32x32_C64_k8__out15x7_split"),  # 32 tiles across 8 cores
    # # --- channel-count variants of the split repro ---
    # (8, 16, 32, 4, "in8x16_C32_k4__out3x3_split_1tile"),
    # (8, 16, 128, 4, "in8x16_C128_k4__out3x3_split_4tile"),
]

# Input generators -- comment a line out to switch that mode off.
INPUT_MODES = [
    # "chan_inc",  # channel c -> value (c+1); every output stick must be [1..C]
    "randn",  # torch.randn (seed 0)
    # "pos_inc",  # each position -> one increasing value, identical across all channels
]

# On-core shard layout -- comment a line out to switch it off.
SHARD_LAYOUTS = [
    "row_major",  # shard height = ANY divisor of tensor_height (e.g. 48 -> 1 core=48, 2 cores=24, ...)
    # "tiled",  # shard height must be a multiple of 32 (TILE_HEIGHT); infeasible geometries are skipped
]


def _dims(in_h, in_w):
    out_h = (in_h - KERNEL[0] + 2 * PADDING[0]) // STRIDE[0] + 1
    out_w = (in_w - KERNEL[1] + 2 * PADDING[1]) // STRIDE[1] + 1
    return out_h, out_w


def _reader_of(global_stick, out_w):
    """The split reader assigns even output COLUMNS to reader0, odd to reader1. For a row-major flattened
    stick index the column is (stick % out_w), so reader id == column parity."""
    return (global_stick % out_w) & 0x1


def _gen_input(mode, batch, C, in_h, in_w):
    """Build the NCHW bf16 input tensor for the requested generator mode."""
    if mode == "chan_inc":
        x = torch.zeros((batch, C, in_h, in_w), dtype=torch.bfloat16)
        for c in range(C):
            x[:, c, :, :] = float(c + 1)
        return x
    if mode == "randn":
        torch.manual_seed(0)
        return torch.randn((batch, C, in_h, in_w), dtype=torch.bfloat16)
    if mode == "pos_inc":
        pos = (torch.arange(in_h * in_w, dtype=torch.float32) + 1.0).reshape(in_h, in_w)
        return pos.reshape(1, 1, in_h, in_w).expand(batch, C, in_h, in_w).to(torch.bfloat16).contiguous()
    raise ValueError(f"unknown input_mode {mode!r}")


def _shard_and_upload(device, x_nhwc_flat, channels, num_cores, shard_layout):
    """HEIGHT-shard the input across EXACTLY num_cores cores and upload it pre-sharded (pool needs sharded
    input on quasar; the interleaved->sharded op is not yet Gen2-clean).

    shard_layout selects the on-core tensor layout:
      "row_major": shard height = tensor_height // num_cores, with NO 32-alignment requirement (e.g. a
                   48-row tensor -> 1 core=48 rows, 2 cores=24 rows, 3 cores=16 rows, ...).
      "tiled":     TILE_LAYOUT; the shard height must be a multiple of 32 (TILE_HEIGHT), so geometries
                   whose per-core height isn't tile-aligned are skipped rather than failed.
    num_cores > device cores -> skip; num_cores not dividing tensor_height -> hard config error."""
    tensor_height = x_nhwc_flat.shape[2]
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores > max_cores:
        pytest.skip(f"need {num_cores} cores, device has {max_cores}")
    assert tensor_height % num_cores == 0, (
        f"num_cores={num_cores} must evenly divide tensor_height={tensor_height}"
    )
    shard_height = tensor_height // num_cores

    if shard_layout == "tiled":
        if tensor_height % 32 != 0 or shard_height % 32 != 0:
            pytest.skip(
                f"tiled shard needs tensor_height ({tensor_height}) and shard_height ({shard_height}) "
                f"to be multiples of 32"
            )
        layout = ttnn.TILE_LAYOUT
    elif shard_layout == "row_major":
        layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        raise ValueError(f"unknown shard_layout {shard_layout!r}")

    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=layout).to(device, mem_config)
    return x


def _run_pool(device, x_nchw, in_h, in_w, channels, num_cores, shard_layout):
    """Upload x_nchw (bf16), run the quasar max_pool2d, return got as [n_out, C] float."""
    batch = x_nchw.shape[0]
    out_h, out_w = _dims(in_h, in_w)
    print(f"out_h={out_h}, out_w={out_w}")
    n_out = batch * out_h * out_w
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * in_h * in_w, channels).contiguous()
    x = _shard_and_upload(device, x_nhwc_flat, channels, num_cores, shard_layout)
    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=in_h,
        input_w=in_w,
        channels=channels,
        kernel_size=list(KERNEL),
        stride=list(STRIDE),
        padding=list(PADDING),
        dilation=list(DILATION),
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float().reshape(-1, channels)[:n_out]
    return got, out_h, out_w, n_out


@pytest.mark.timeout(600)
@pytest.mark.parametrize("shard_layout", SHARD_LAYOUTS)
@pytest.mark.parametrize("input_mode", INPUT_MODES)
@pytest.mark.parametrize("in_h,in_w,C,num_cores", [c[:4] for c in SWEEP], ids=[c[4] for c in SWEEP])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pool(mesh_device, in_h, in_w, C, num_cores, input_mode, shard_layout):
    """Run the quasar max_pool2d on the generated input and check it against a torch golden per stick."""
    device = mesh_device
    batch = 1
    x_nchw = _gen_input(input_mode, batch, C, in_h, in_w)
    input_max = x_nchw.float().max().item()
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE), padding=list(PADDING)
    )

    got, out_h, out_w, n_out = _run_pool(device, x_nchw, in_h, in_w, C, num_cores, shard_layout)
    golden = golden_nchw.permute(0, 2, 3, 1).reshape(-1, C)[:n_out]

    # Classify every stick: zero (all-0 but golden isn't) / wrong / correct. max-pool is exact selection,
    # so a correct op matches the golden exactly; the tolerance only absorbs bf16 readback noise.
    zero, wrong = [], []
    for s in range(n_out):
        if torch.count_nonzero(got[s]) == 0 and torch.count_nonzero(golden[s]) != 0:
            zero.append(s)
        elif not torch.allclose(got[s], golden[s], atol=2e-1, rtol=2e-1):
            wrong.append(s)

    def tag(sticks):
        r0 = [s for s in sticks if _reader_of(s, out_w) == 0]
        r1 = [s for s in sticks if _reader_of(s, out_w) == 1]
        return f"r0={r0} r1={r1}"

    got_max = got.max().item()
    n_correct = n_out - len(zero) - len(wrong)
    print(
        f"\n[{input_mode} {in_h}x{in_w} C{C} k{num_cores} {shard_layout}] out={n_out} sticks (out_h={out_h} out_w={out_w}), "
        f"split_active={out_w >= 2}\n"
        f"   CORRECT {n_correct}/{n_out}  |  ZERO {len(zero)}: {tag(zero)}  |  WRONG {len(wrong)}: {tag(wrong)}  "
        f"|  got.max={got_max:.4f} input.max={input_max:.4f}"
    )
    if wrong:
        s = wrong[0]
        print(f"   first WRONG stick {s}: got[:8]={got[s][:8].tolist()}  golden[:8]={golden[s][:8].tolist()}")
    if zero:
        print(f"   (zero sticks read all-0 -> reduce/feed produced nothing for those)")

    # (1) HARD leak invariant: a correct max-pool output never exceeds the input max.
    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(cores={num_cores}, {in_h}x{in_w}, C{C}, {input_mode})"
    )
    # (2) Per-stick exact match vs golden.
    n_bad = len(zero) + len(wrong)
    assert n_bad == 0, (
        f"{n_bad}/{n_out} sticks bad (zero={len(zero)} wrong={len(wrong)}); zero[{tag(zero)}] wrong[{tag(wrong)}]"
    )
