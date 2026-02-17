import torch
import ttnn
import pytest

from models.common.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
    skip_for_wormhole_b0,
    skip_for_blackhole,
)

torch.manual_seed(0)


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        return torch.rand(shape, dtype=torch.bfloat16)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize("shape", [(768, 768, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("in_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_slice_arguments(device, shape, layout, in_memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    H, H, W = torch_tensor.shape
    x = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=in_memory_config)

    x = x[:686, :686, :160]


def slice_test(
    input_layout,
    input_tensor_shape,
    output_tensor_start,
    output_tensor_end,
    device,
    in_mem_config,
    out_mem_config,
    dtype,
    slice_step=(1, 1, 1, 1),
):
    if dtype == ttnn.float32:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.float)
    else:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_layout, device=device, memory_config=in_mem_config
    )

    tt_output_tensor = ttnn.slice(
        tt_input_tensor,
        slice_start=output_tensor_start,
        slice_end=output_tensor_end,
        slice_step=slice_step,
        memory_config=out_mem_config,
    )

    a_pt = ttnn.to_torch(tt_output_tensor)

    # Pytorch reference
    a_ref = torch_input_tensor[
        output_tensor_start[0] : output_tensor_end[0] : slice_step[0],
        output_tensor_start[1] : output_tensor_end[1] : slice_step[1],
        output_tensor_start[2] : output_tensor_end[2] : slice_step[2],
        output_tensor_start[3] : output_tensor_end[3] : slice_step[3],
    ]

    return a_pt, a_ref, device.num_program_cache_entries()


@pytest.mark.parametrize("shape", [(1, 686, 686, 512), (1, 686, 686, 256), (1, 686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("in_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_slice_boltz(device, shape, layout, in_memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)

    x = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=in_memory_config)
    B, H, H, W = x.shape

    # Case 1: Slice on Width into 2 halves
    g_in = x[:, :, :, : W // 2]
    p_in = x[:, :, :, W // 2 :]

    # Case 2: Slice on W into chunks of 32
    TRIANGLE_MULT_CHUNK_SIZE = 32
    for chunk_start in range(0, W // 2, TRIANGLE_MULT_CHUNK_SIZE):
        a_chunk = ttnn.slice(
            x,
            [0, 0, 0, chunk_start],
            [B, H, H, chunk_start + TRIANGLE_MULT_CHUNK_SIZE],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # Case 3: Slice on Sequence Length
    TRANSITION_CHUNK_SIZE = 64
    for chunk_start in range(0, torch_tensor.shape[1], TRANSITION_CHUNK_SIZE):
        x_chunk = x[
            :,
            chunk_start : min(chunk_start + TRANSITION_CHUNK_SIZE, torch_tensor.shape[1]),
            :,
            :,
        ]


@pytest.mark.parametrize("shape", [(768, 768, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("in_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_slice_v2_boltz(device, shape, layout, in_memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    H, H, W = torch_tensor.shape
    x = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=in_memory_config)

    x = x[:686, :686, :]


# @pytest.mark.parametrize(
#    "dtype",
#    (ttnn.bfloat16, ttnn.float32),
#    ids=["bfloat16", "float"],
# )
# @pytest.mark.parametrize(
#    "out_mem_config",
#    (ttnn.DRAM_MEMORY_CONFIG,),
#    ids=["out_DRAM"],
# )
# @pytest.mark.parametrize(
#    "in_mem_config",
#    (ttnn.DRAM_MEMORY_CONFIG,),
#    ids=["in0_DRAM"],
# )
# @pytest.mark.parametrize(
#    "input_tensor_shape_0, output_tensor_start_0, output_tensor_end_0",
#    (
#        ((4, 3, 64, 64), (0, 0, 0, 0), (4, 3, 32, 32)),
##        ((1, 1, 64, 64), (0, 0, 0, 0), (1, 1, 32, 64)),
##        ((1, 1, 128, 96), (0, 0, 64, 32), (1, 1, 96, 96)),
##        ((1, 1, 128, 96), (0, 0, 64, 32), (1, 1, 96, 96)),
##        ((1, 3, 32, 32), (0, 1, 0, 0), (1, 2, 32, 32)),
##        ((1, 6, 32, 32), (0, 2, 0, 0), (1, 4, 32, 32)),
##        ((1, 6, 128, 64), (0, 2, 64, 32), (1, 4, 96, 64)),
##        ((4, 6, 128, 64), (1, 2, 64, 32), (2, 4, 96, 64)),
#    ),
# )
##@pytest.mark.parametrize(
##    "input_tensor_shape_1, output_tensor_start_1, output_tensor_end_1",
##    (((9, 8, 128, 128), (0, 0, 0, 0), (9, 8, 32, 32)),),
##)
# @pytest.mark.parametrize(
#    "slice_step",
#    ((1, 1, 1, 1),),
# )
# def test_slice_run_test(
#    input_tensor_shape_0,
#    output_tensor_start_0,
#    output_tensor_end_0,
##    input_tensor_shape_1,
##    output_tensor_start_1,
##    output_tensor_end_1,
#    device,
#    in_mem_config,
#    out_mem_config,
#    dtype,
#    slice_step,
# ):
##    a_pt, a_ref, num_cache_entries = slice_test(
##        ttnn.ROW_MAJOR_LAYOUT,
##        input_tensor_shape_0,
##        output_tensor_start_0,
##        output_tensor_end_0,
##        device,
##        in_mem_config,
##        out_mem_config,
##        dtype,
##        slice_step,
##    )
##    assert a_pt.shape == a_ref.shape
##    eq = torch.equal(a_pt, a_ref)
##    assert eq
##    assert num_cache_entries == 1
##
##    a_pt, a_ref, num_cache_entries = slice_test(
##        ttnn.ROW_MAJOR_LAYOUT,
##        input_tensor_shape_1,
##        output_tensor_start_1,
##        output_tensor_end_1,
##        device,
##        in_mem_config,
##        out_mem_config,
##        dtype,
##        slice_step,
##    )
##    assert a_pt.shape == a_ref.shape
##    eq = torch.equal(a_pt, a_ref)
##    assert eq
##    # different width for row major
##    assert num_cache_entries == 2
#
#    a_pt, a_ref, num_cache_entries = slice_test(
#        ttnn.TILE_LAYOUT,
#        input_tensor_shape_0,
#        output_tensor_start_0,
#        output_tensor_end_0,
#        device,
#        in_mem_config,
#        out_mem_config,
#        dtype,
#        slice_step,
#    )
#    # change from RM to TILE
##    assert num_cache_entries == 3
#    assert a_pt.shape == a_ref.shape
#    eq = torch.equal(a_pt, a_ref)
#    assert eq
#
##    a_pt, a_ref, num_cache_entries = slice_test(
##        ttnn.TILE_LAYOUT,
##        input_tensor_shape_1,
##        output_tensor_start_1,
##        output_tensor_end_1,
##        device,
##        in_mem_config,
##        out_mem_config,
##        dtype,
##        slice_step,
##    )
##    # CACHE HIT
##    assert num_cache_entries == 4
##    assert a_pt.shape == a_ref.shape
##    eq = torch.equal(a_pt, a_ref)
##    assert eq


@pytest.mark.parametrize("shape", [(1, 686, 686, 512)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("in_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
)
def test_sample(device, shape, layout, in_memory_config, dtype):
    if is_blackhole():
        num_banks = device.dram_grid_size().x
        print(f"num_banks: {num_banks}")

    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)

    x = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=in_memory_config)
    B, H, H, W = x.shape

    # Case 1: Slice on Width into 2 halves
    g_in = x[:, :, :, : W // 2]
    p_in = x[:, :, :, W // 2 :]
