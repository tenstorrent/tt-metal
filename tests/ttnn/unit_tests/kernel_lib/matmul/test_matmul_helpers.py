# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exercise compute helpers directly, without production matmul kernels or factories."""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device
KERNEL = "tests/ttnn/unit_tests/kernel_lib/matmul/kernels/matmul.cpp"


def _tiles(matrix):
    rows, cols = matrix.shape
    return matrix.reshape(rows // 32, 32, cols // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)


@pytest.mark.parametrize("k_blocks", [1, 3], ids=["single-k", "spill-reload"])
@pytest.mark.parametrize("l1_acc", [False, True], ids=["software", "l1-acc"])
@pytest.mark.parametrize(
    "post_op",
    [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13],
    ids=[
        "plain",
        "bias",
        "relu",
        "relu6",
        "bias-relu6",
        "bias-relu",
        "gelu-tanh",
        "column-indexed-bias",
        "full-block-bias",
        "mish",
        "bias-mish",
        "sqrt",
    ],
)
@pytest.mark.parametrize("static_shape", [False, True], ids=["runtime-shape", "static-shape"])
def test_matmul_helpers(device, k_blocks, l1_acc, post_op, static_shape):
    _run_matmul_helpers(device, k_blocks, l1_acc, post_op, static_shape)


def _run_matmul_helpers(
    device,
    k_blocks,
    l1_acc,
    post_op,
    static_shape,
    batches=1,
    same_cb=False,
    output_blocks=1,
    bias_dtype=ttnn.bfloat16,
    activation_on_math=False,
    fast_approx=False,
    fp32_dest_acc_en=False,
    math_approx_mode=False,
):
    # Signed, exactly representable inputs distinguish tile order and ensure ReLU
    # must happen after the K reduction. Different K slices also catch lost partials.
    generator = torch.Generator().manual_seed(42)
    a = torch.randint(-2, 3, (128 * batches, 32 * k_blocks), generator=generator).float()
    width = 128 * output_blocks
    b = torch.zeros(32 * k_blocks, width)
    for k in range(k_blocks):
        for n in range(4 * output_blocks):
            b[k * 32 : (k + 1) * 32, n * 32 : (n + 1) * 32] = torch.eye(32) * (1 if (k + n) % 2 else -1)
    if post_op == 13:
        a, b = a.abs(), b.abs()
    elif post_op == 16:
        a = a / 8  # Keep exp inputs small enough to check BF16 outputs accurately.
    elif post_op == 17:
        a, b = a.abs() + 1, b.abs()  # Avoid poles in the reciprocal reference.
    expected = a @ b
    if post_op == 10:
        assert output_blocks == 1  # Full-block bias uses the matmul block's N stride.
        bias = (torch.arange(128 * 128).reshape(128, 128) % 7 - 3).float()
    elif post_op == 9:
        bias = (torch.arange(32 * width).reshape(32, width) % 7 - 3).float()
    else:
        bias = torch.zeros(32, width)
        bias[0] = torch.arange(width) % 5 - 2
    if post_op in (1, 4, 7, 12):
        expected += bias[0]
    elif post_op == 9:
        expected += bias.repeat(4, 1)
    elif post_op == 10:
        expected += bias
    if post_op in (2, 7):
        expected = expected.relu()
    elif post_op in (3, 4):
        expected = expected.clamp(0, 6)
    elif post_op == 8:
        expected = torch.nn.functional.gelu(expected, approximate="tanh")
    elif post_op in (11, 12):
        expected = torch.nn.functional.mish(expected)
    elif post_op == 13:
        expected = expected.sqrt()
    elif post_op == 14:
        expected = torch.nn.functional.leaky_relu(expected, negative_slope=0.125)
    elif post_op == 15:
        expected = torch.nn.functional.elu(expected)
    elif post_op == 16:
        expected = expected.exp()
    elif post_op == 17:
        expected = expected.reciprocal()

    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    def memory(shape):
        return ttnn.create_sharded_memory_config(
            shape,
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def tensor(tiles, dtype=ttnn.bfloat16):
        physical = tiles.reshape(-1, 32)
        return ttnn.from_torch(
            physical,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory(tuple(physical.shape)),
        )

    # The helper consumes M x block_K and block_K x N tiles for each K block.
    ta = tensor(
        torch.cat(
            [
                _tiles(a[batch * 128 : (batch + 1) * 128, k * 32 : (k + 1) * 32])
                for block in range(output_blocks)
                for batch in range(batches)
                for k in range(k_blocks)
            ]
        )
    )
    tb = tensor(
        torch.cat(
            [_tiles(b[:, block * 128 : (block + 1) * 128]).repeat(batches, 1, 1) for block in range(output_blocks)]
        )
    )
    t_bias = tensor(_tiles(bias), dtype=bias_dtype)
    output_tiles = 16 * batches * output_blocks
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape((output_tiles * 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory((output_tiles * 32, 32))
    )
    page = ttnn.tile_size(ttnn.bfloat16)
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(i, t) for i, t in [(0, ta), (1, tb), (3, t_bias), (16, out)]]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=16 * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=page)],
        )
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[
            k_blocks,
            int(l1_acc),
            post_op,
            int(static_shape),
            batches,
            int(same_cb),
            output_blocks,
            int(activation_on_math),
            int(fast_approx),
        ],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=math_approx_mode,
            dst_full_sync_en=False,
        ),
    )
    result = ttnn.generic_op([ta, tb, t_bias, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))
    actual_tiles = ttnn.to_torch(result).reshape(output_tiles, 32, 32).float()
    expected_tiles = _tiles(expected)
    order = [
        (batch * 4 + r) * (4 * output_blocks) + block * 4 + c
        for block in range(output_blocks)
        for batch in range(batches)
        for br in (0, 2)
        for bc in (0, 2)
        for r in range(br, br + 2)
        for c in range(bc, bc + 2)
    ]
    expected_tiles = expected_tiles[order]
    if post_op in (16, 17):
        torch.testing.assert_close(actual_tiles, expected_tiles, rtol=0.03, atol=0.005)
    elif post_op in (8, 11, 12, 13, 15):
        torch.testing.assert_close(actual_tiles, expected_tiles, rtol=0.03, atol=0.125)
    else:
        torch.testing.assert_close(actual_tiles, expected_tiles, rtol=0, atol=0)
    return actual_tiles


@pytest.mark.parametrize("l1_acc", [False, True])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
@pytest.mark.parametrize("math_approx_mode", [False, True])
@pytest.mark.parametrize(
    "post_op,fast_approx",
    [(11, False), (11, True), (13, False), (13, True), (14, False), (15, False), (16, False), (16, True), (17, False)],
    ids=["mish", "mish-fast", "sqrt", "sqrt-fast", "leaky-relu", "elu", "exp", "exp-fast", "recip"],
)
def test_matmul_activation_threads(device, l1_acc, fp32_dest_acc_en, math_approx_mode, post_op, fast_approx):
    outputs = [
        _run_matmul_helpers(
            device,
            3,
            l1_acc,
            post_op,
            False,
            batches=2,
            output_blocks=2,
            activation_on_math=on_math,
            fast_approx=fast_approx,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=math_approx_mode,
        )
        for on_math in (True, False)
    ]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("l1_acc", [False, True])
def test_matmul_activation_batches(device, l1_acc):
    _run_matmul_helpers(device, 3, l1_acc, 3, False, batches=2)


@pytest.mark.parametrize("k_blocks", [1, 3])
@pytest.mark.parametrize("l1_acc", [False, True])
@pytest.mark.parametrize(
    "post_op",
    [0, 1, 2, 3, 4, 7, 9, 10],
    ids=["plain", "bias", "relu", "relu6", "bias-relu6", "bias-relu", "column-indexed-bias", "full-block-bias"],
)
def test_matmul_same_output_and_partials(device, k_blocks, l1_acc, post_op):
    _run_matmul_helpers(device, k_blocks, l1_acc, post_op, False, same_cb=True)


@pytest.mark.parametrize("l1_acc", [False, True])
@pytest.mark.parametrize("post_op", [0, 1, 4, 7], ids=["plain", "bias", "bias-relu6", "bias-relu"])
@pytest.mark.parametrize("bias_dtype", [ttnn.bfloat16, ttnn.float32])
def test_matmul_output_block_transitions(device, l1_acc, post_op, bias_dtype):
    # Retain both bias slices, then advance the offset and restore input formats
    # for the next width block. Float32 bias distinguishes SrcB from BF16 in0.
    _run_matmul_helpers(device, 3, l1_acc, post_op, False, output_blocks=2, bias_dtype=bias_dtype)
