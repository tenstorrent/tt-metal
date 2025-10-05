import pytest
import torch
import ttnn
import itertools
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from models.common.utility_functions import is_wormhole_b0, is_blackhole


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("shape", [(686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_permute_3D(device, shape, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=memory_config)
    output_tensor = ttnn.permute(input_tensor, (1, 0, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (1, 0, 2))
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [(686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_pad_3D(device, shape, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    x = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=memory_config)

    x = ttnn.to_layout(x, layout)
    seq_len = x.shape[0]
    padding = -seq_len % 256
    x = ttnn.pad(x, [(0, padding), (0, padding), (0, 0)], 0)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("shape", [(1, 686, 686, 768)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_nlp_create_qkv_h(device, shape, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=memory_config)

    g_in, p_in, g_out = ttnn.experimental.nlp_create_qkv_heads_boltz(
        input_tensor,
        num_heads=1,
        num_kv_heads=1,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    one_slice = input_tensor.shape[-1] // 3
    g_in1 = input_tensor[:, :, :, :one_slice]
    p_in1 = input_tensor[:, :, :, one_slice : 2 * one_slice]
    g_out1 = input_tensor[:, :, :, 2 * one_slice :]


@pytest.mark.parametrize("shape", [(1, 686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_mamtul_gpg(device, shape, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    input_tensor = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w = input_tensor.shape[-1]
    torch_tensor_2 = random_torch_tensor(ttnn.bfloat16, (w, 6 * w))
    gpg_weight = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    gpg_in = ttnn.linear(
        input_tensor,
        gpg_weight,
        # compute_kernel_config=self.compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
    )


@pytest.mark.parametrize("shape", [(1, 686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_mamtul_gpg_v2(device, shape, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    input_tensor = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w = input_tensor.shape[-1]
    torch_tensor_2 = random_torch_tensor(ttnn.bfloat16, (w, 4 * w))
    gpg_weight = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    gpg_in = ttnn.linear(
        input_tensor,
        gpg_weight,
        # compute_kernel_config=self.compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
    )


@pytest.mark.parametrize("shape", [(1, 768, 768, 32)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("chunk_size_q", [128, 256])
@pytest.mark.parametrize("chunk_size_k", [128, 256])
@pytest.mark.parametrize(
    "dtype",
    [
        # ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_sdpa_boltz(device, shape, layout, memory_config, dtype, chunk_size_q, chunk_size_k):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    head_dim = shape[-1]
    seq_len = shape[-2]
    torch_tensor_2 = random_torch_tensor(ttnn.bfloat16, (1, 1, seq_len, seq_len))
    head_q = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_k = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_v = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_triangle_bias = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if chunk_size_q == 768 and chunk_size_k == 768:
        return

    head_o = ttnn.transformer.scaled_dot_product_attention(
        head_q,
        head_k,
        head_v,
        attn_mask=head_triangle_bias,
        is_causal=False,
        scale=head_dim**-0.5,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=((13, 10) if is_blackhole() else (8, 8)),
            exp_approx_mode=False,
            q_chunk_size=chunk_size_q,
            k_chunk_size=chunk_size_k,
        ),
        memory_config=memory_config,
    )


@pytest.mark.parametrize("shape", [(1, 704, 704, 32)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("chunk_size_q", [64, 352])
@pytest.mark.parametrize("chunk_size_k", [64, 352])
@pytest.mark.parametrize(
    "dtype",
    [
        # ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_sdpa_boltz_v2(device, shape, layout, memory_config, dtype, chunk_size_q, chunk_size_k):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    head_dim = shape[-1]
    seq_len = shape[-2]
    torch_tensor_2 = random_torch_tensor(ttnn.bfloat16, (1, 1, seq_len, seq_len))
    head_q = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_k = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_v = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_triangle_bias = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if chunk_size_q == 768 and chunk_size_k == 768:
        return

    head_o = ttnn.transformer.scaled_dot_product_attention(
        head_q,
        head_k,
        head_v,
        attn_mask=head_triangle_bias,
        is_causal=False,
        scale=head_dim**-0.5,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=((13, 10) if is_blackhole() else (8, 8)),
            exp_approx_mode=False,
            q_chunk_size=chunk_size_q,
            k_chunk_size=chunk_size_k,
        ),
        memory_config=memory_config,
    )


@pytest.mark.parametrize("shape", [(686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("in_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("out_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_layer_norm_boltz(device, shape, layout, in_memory_config, out_memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_tensor_2 = random_torch_tensor(dtype, (1, 128))
    x = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=in_memory_config)
    in_norm_weight = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in_norm_bias = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_norm_in = ttnn.layer_norm(
        x,
        weight=in_norm_weight,
        bias=in_norm_bias,
        epsilon=1e-5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", [(686, 686, 128)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("out_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat8_b,
    ],
)
def test_sigmoid_boltz(device, shape, layout, out_memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    x = ttnn.from_torch(
        torch_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gp_weight_torch = random_torch_tensor(ttnn.bfloat16, (128, 256))
    gp_weight = ttnn.from_torch(
        gp_weight_torch, layout=layout, device=device, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g_in = ttnn.linear(
        x,
        gp_weight,
        # compute_kernel_config=ttnn.DeviceComputeKernelConfig,
        memory_config=out_memory_config,
        dtype=dtype,
        core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
    )
    g_in = ttnn.sigmoid_accurate(g_in, output_tensor=g_in)
    g_in_accurate = ttnn.sigmoid_accurate(g_in, output_tensor=g_in)


@pytest.mark.parametrize("shape", [(686, 686, 256)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("in_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("out_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_multiply_boltz(device, shape, layout, in_memory_config, out_memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    torch_tensor_2 = random_torch_tensor(ttnn.bfloat16, shape)

    if in_memory_config == ttnn.L1_MEMORY_CONFIG and out_memory_config == ttnn.L1_MEMORY_CONFIG:
        return

    p_in = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat8_b, memory_config=in_memory_config
    )
    g_in = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_pg_in = ttnn.multiply(p_in, g_in, dtype=dtype, memory_config=out_memory_config)


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


@pytest.mark.parametrize("shape", [(1, 12, 3072, 192)])  # 32, 64, 96, 192(12)
# @pytest.mark.parametrize("shape", [(1, 16, 2048, 96)]) # 32, 64, 96, 192(12)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])  # , ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("chunk_size_q", [128, 256, 512])
@pytest.mark.parametrize("chunk_size_k", [128, 256, 512, 1024])
# @pytest.mark.parametrize(
#     "dtype",
#     [
#         # ttnn.bfloat16,
#         ttnn.bfloat8_b,
#     ],
# )
def test_sdpa_vit(device, shape, memory_config, chunk_size_q, chunk_size_k):
    dtype = ttnn.bfloat8_b
    layout = ttnn.TILE_LAYOUT
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    head_dim = shape[-1]
    seq_len = shape[-2]
    torch_tensor_2 = random_torch_tensor(ttnn.bfloat16, (1, 1, seq_len, seq_len))
    head_q = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_k = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_v = ttnn.from_torch(
        torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    head_triangle_bias = ttnn.from_torch(
        torch_tensor_2, layout=layout, device=device, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if chunk_size_q == 768 and chunk_size_k == 768:
        return

    head_o = ttnn.transformer.scaled_dot_product_attention(
        head_q,
        head_k,
        head_v,
        is_causal=False,
        scale=head_dim**-0.5,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=((13, 10) if is_blackhole() else (8, 8)),
            exp_approx_mode=True,
            q_chunk_size=chunk_size_q,
            k_chunk_size=chunk_size_k,
        ),
        memory_config=memory_config,
    )
