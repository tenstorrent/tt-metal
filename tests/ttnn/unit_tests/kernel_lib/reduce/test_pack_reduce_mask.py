# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Check the complete packed tile, including reduction padding, independently of the math kernel."""

import pytest
import torch
import ttnn

KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/pack_reduce_mask.cpp"
REPEATS = 5
OUTPUTS_PER_SECTION = 4 + 3 * REPEATS
TILE_SHAPES = [(32, 32), (32, 16), (16, 32), (16, 16), (8, 32), (4, 16), (2, 32), (1, 32), (1, 16)]
BFP8_TILE_SHAPES = [(32, 32), (32, 16), (8, 32), (4, 16), (2, 32), (1, 32), (1, 16)]
CASES = [
    (tile_shape, dtype, fp32_dest)
    for tile_shape in TILE_SHAPES
    for dtype, fp32_dest in [(ttnn.float32, True), (ttnn.bfloat16, False), (ttnn.bfloat16, True)]
] + [(tile_shape, ttnn.bfloat8_b, False) for tile_shape in BFP8_TILE_SHAPES]


def _single_core():
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _sharded(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("tile_shape,dtype,fp32_dest", CASES)
@pytest.mark.parametrize("full_sync", [False, True])
def test_pack_reduce_mask(device, tile_shape, dtype, fp32_dest, full_sync, runtime_output=False):
    if device.arch() not in (ttnn.device.Arch.BLACKHOLE, ttnn.device.Arch.WORMHOLE_B0):
        pytest.skip("Blackhole/Wormhole reduction-mask regression")
    height, width = tile_shape
    tile = ttnn.Tile(tile_shape)
    # Different positive values in the two source tiles expose wrong-face selection and wrong DEST addressing.
    source = (torch.arange(height * 2 * width, dtype=torch.float32) + 1).reshape(1, 1, height, 2 * width)
    if dtype == ttnn.bfloat16:
        source = source.bfloat16().float()
    elif dtype == ttnn.bfloat8_b:
        # Every value shares the [64, 128) exponent, so the 7-bit BFP mantissa holds it exactly.
        source = 64 + source % 64
    inp = ttnn.from_torch(
        source,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded((height, 2 * width)),
        tile=tile,
    )
    output_width = 2 * OUTPUTS_PER_SECTION * width
    # Prefill output with nonzero data so a missing write cannot masquerade as correct zero padding.
    out = ttnn.from_torch(
        torch.full((1, 1, height, output_width), -7.0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded((height, output_width)),
        tile=tile,
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, inp), ttnn.cb_descriptor_from_sharded_tensor(16, out)]
    if runtime_output:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(17, out))
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=_single_core(),
                compile_time_args=[0, 16, REPEATS, int(runtime_output)],
                runtime_args=[(ttnn.CoreCoord(0, 0), [17])],
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest, dst_full_sync_en=full_sync),
            )
        ],
        semaphores=[],
        cbs=cbs,
    )
    actual = ttnn.to_torch(ttnn.generic_op([inp, out], program))[0, 0].float()
    inputs = [source[0, 0, :, i * width : (i + 1) * width] for i in range(2)]
    expected_section = inputs.copy()
    for dim in ("row", "col", "scalar"):
        for i in range(REPEATS):
            original = inputs[i % 2]
            masked = torch.zeros_like(original)
            if dim == "row":
                masked[:, 0] = original[:, 0]
            elif dim == "col":
                masked[0, :] = original[0, :]
            else:
                masked[0, 0] = original[0, 0]
            expected_section.append(masked)
    expected_section.extend(inputs)
    expected = torch.cat(expected_section * 2, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tile_shape,dtype,fp32_dest", CASES)
@pytest.mark.parametrize("full_sync", [False, True])
def test_pack_reduce_mask_runtime_output(device, tile_shape, dtype, fp32_dest, full_sync):
    test_pack_reduce_mask(device, tile_shape, dtype, fp32_dest, full_sync, runtime_output=True)
