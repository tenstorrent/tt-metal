# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics, tt_dtype_to_torch_dtype


TILE_SIZE = ttnn.TILE_SIZE


def chunk_and_shard_tensor(tensor, num_simulated_devices, device, dim, dtype, layout=ttnn.TILE_LAYOUT):
    """Chunk a tensor and convert each chunk to a ttnn tensor."""
    chunked = torch.chunk(tensor, num_simulated_devices, dim)
    result = [
        ttnn.from_torch(chunk, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunked
    ]
    return result


def create_recip_tensor_for_welford(device, width):
    """Create reciprocal tensor for Welford algorithm in pre-allgather op."""
    grid = device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_layer_norm_reciprocals(device, core_range_set, width)


def prepare_row_major_weight_bias(torch_weight_1d, num_simulated_devices):
    """
    Prepare weight/bias tensor for ROW_MAJOR layout, matching DistributedLayerNorm._prepare_torch_state.

    This reshapes the 1D weight/bias tensor to interleave across devices in the specific
    pattern expected by the distributed layernorm operation.

    Args:
        torch_weight_1d: 1D tensor of shape (embedding_dim,)
        num_simulated_devices: Number of simulated TP devices

    Returns:
        Reshaped tensor of shape (embedding_dim // (TILE_SIZE * num_simulated_devices), TILE_SIZE * num_simulated_devices)
    """
    # Reshape to interleave: [mesh_width, -1, TILE_SIZE] -> permute -> reshape
    # This matches: weight.reshape(mesh_width, -1, TILE_SIZE).permute(1, 0, 2).reshape(-1, TILE_SIZE * mesh_width)
    reshaped = (
        torch_weight_1d.reshape(num_simulated_devices, -1, TILE_SIZE)
        .permute(1, 0, 2)
        .reshape(-1, TILE_SIZE * num_simulated_devices)
    )
    return reshaped


def run_distributed_dit_layernorm(
    device,
    num_simulated_devices,
    inp_shape,
    dtype,
    stats_dtype,
    use_affine=True,
    weight_bias_layout=ttnn.TILE_LAYOUT,
):
    """
    Test distributed DIT layernorm by simulating multi-device operation on a single device.

    The test:
    1. Chunks input across simulated devices
    2. Runs pre_allgather on each chunk to get per-device stats
    3. Concatenates stats (simulating all-gather)
    4. Runs post_allgather on each chunk with gathered stats
    5. Concatenates output and compares to torch reference

    Args:
        device: The ttnn device
        num_simulated_devices: Number of TP devices to simulate
        inp_shape: Input tensor shape (batch, channel, seq_len, embedding_dim)
        dtype: Input data type
        stats_dtype: Stats data type
        use_affine: Whether to use weight and bias
        weight_bias_layout: Layout for weight/bias tensors (TILE_LAYOUT or ROW_MAJOR_LAYOUT)
    """
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)

    embedding_dim = inp_shape[-1]
    torch_input = torch.randn(inp_shape) * 4 - 1

    torch_weight = None
    torch_bias = None
    if use_affine:
        # Create 1D weight and bias tensors
        torch_weight = torch.rand(embedding_dim)
        torch_bias = torch.rand(embedding_dim)

    epsilon = 1e-5

    # Chunk input across simulated devices on the last dimension
    tt_inp = chunk_and_shard_tensor(torch_input, num_simulated_devices, device, -1, dtype)

    tt_weight = None
    tt_bias = None
    if use_affine:
        if weight_bias_layout == ttnn.ROW_MAJOR_LAYOUT:
            # For ROW_MAJOR, reshape weight/bias using the interleaved pattern
            # then chunk across devices
            weight_reshaped = prepare_row_major_weight_bias(torch_weight, num_simulated_devices)
            bias_reshaped = prepare_row_major_weight_bias(torch_bias, num_simulated_devices)
            tt_weight = chunk_and_shard_tensor(
                weight_reshaped, num_simulated_devices, device, -1, dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            tt_bias = chunk_and_shard_tensor(
                bias_reshaped, num_simulated_devices, device, -1, dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        else:
            # For TILE_LAYOUT, use simple 2D shape [1, embedding_dim] and chunk
            weight_2d = torch_weight.unsqueeze(0)
            bias_2d = torch_bias.unsqueeze(0)
            tt_weight = chunk_and_shard_tensor(weight_2d, num_simulated_devices, device, -1, dtype)
            tt_bias = chunk_and_shard_tensor(bias_2d, num_simulated_devices, device, -1, dtype)

    # Create reciprocal tensor for Welford algorithm
    # Width per chunk is embedding_dim / num_simulated_devices
    chunk_width = embedding_dim // num_simulated_devices
    recip_tensor = create_recip_tensor_for_welford(device, chunk_width)

    tt_stats = []
    for tt_inp_chunk in tt_inp:
        tt_stats.append(
            ttnn.experimental.dit_layernorm_pre_allgather(
                tt_inp_chunk, recip_tensor, compute_kernel_config=compute_kernel_config, dtype=stats_dtype
            )
        )

    # Simulate all-gather by concatenating stats from all devices
    tt_stats_gathered = ttnn.concat(tt_stats, -1)

    # Run post_allgather on each chunk with gathered stats
    tt_out = []
    for idx in range(len(tt_inp)):
        tt_out.append(
            ttnn.experimental.dit_layernorm_post_allgather(
                tt_inp[idx],
                tt_stats_gathered,
                epsilon=epsilon,
                weight=tt_weight[idx] if tt_weight is not None else None,
                bias=tt_bias[idx] if tt_bias is not None else None,
                compute_kernel_config=compute_kernel_config,
            )
        )

    # Concat the output on the sharded dimension
    tt_out = ttnn.concat(tt_out, -1)
    tt_out = ttnn.to_torch(tt_out)

    # Reference implementation using torch layer_norm
    out_torch = torch.nn.functional.layer_norm(
        torch_input, normalized_shape=(embedding_dim,), weight=None, bias=None, eps=epsilon
    )
    if use_affine:
        out_torch = out_torch * torch_weight
        out_torch = out_torch + torch_bias

    if dtype == ttnn.float32:
        # Post-allgather layernorm output noise floor decomposition:
        #
        # - The final (x - mu) / sqrt(var + eps) * gamma + beta cascade is built on FPU
        #   bcast ops (sub_tiles_bcast_cols, mul_tiles_bcast_cols, mul_tiles_bcast_rows,
        #   add_tiles_bcast_rows). Their SrcA/SrcB reads are 19-bit (TF32, 10 mantissa),
        #   so any FP32 routed through them is structurally truncated to ~1e-3 relative.
        # - The test always uses bf16 stats (and the affine path uses bf16 gamma/beta),
        #   so mean / sqrt(var + eps) carry bf16 quantization (~0.78% relative). For
        #   per-element output of magnitude up to ~3 that gives an absolute floor of
        #   3 * 0.0078 ~= 0.024 per element. bf16 quantization of gamma/beta gives the
        #   same magnitude, dominating the TF32 FPU floor.
        #
        # atol leaves ~1.6x headroom over the structural element floor across shapes.
        # PCC and frobenius are global metrics that come in much tighter than the
        # element-wise atol; thresholds set close to observed (~5x headroom in 1-PCC).
        rtol = 0.005
        atol = 0.04
        pcc = 0.999994
        frobenius_threshold = 0.005
    else:
        rtol = 1e-1
        atol = 1e-1
        pcc = 0.99996
        frobenius_threshold = 0.005
    assert_numeric_metrics(
        out_torch,
        tt_out,
        rtol=rtol,
        atol=atol,
        pcc_threshold=pcc,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "embedding_dim",
    [2048, 2432, 3072, 5120],
    ids=["dim2048", "dim2432", "dim3072", "dim5120"],
)
@pytest.mark.parametrize(
    "seq_len",
    [512, 2048, 4096, 9472],
    ids=["len512", "len2048", "len4096", "len9472"],
)
@pytest.mark.parametrize("use_affine", [False, True], ids=["no_affine", "with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [1, 2, 4], ids=["tp1", "tp2", "tp4"])
def test_distributed_dit_layernorm_use_cases(
    device,
    num_simulated_devices,
    embedding_dim,
    seq_len,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    Covers all DistributedLayerNorm instantiations seen in tt_dit models.
    Tests the DIT distributed layernorm op on single device, emulating TP devices.
    """
    # Skip if embedding_dim is not divisible by num_simulated_devices * TILE_SIZE
    if embedding_dim % (num_simulated_devices * TILE_SIZE) != 0:
        pytest.skip(
            f"embedding_dim ({embedding_dim}) must be divisible by num_simulated_devices * TILE_SIZE ({num_simulated_devices * TILE_SIZE})"
        )

    # Skip layout test when no affine (layout doesn't matter without weight/bias)
    if not use_affine and weight_bias_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Layout doesn't matter when affine is disabled")

    inp_shape = (1, 1, seq_len, embedding_dim)
    run_distributed_dit_layernorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, use_affine, weight_bias_layout
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [4096],
)
@pytest.mark.parametrize(
    "hidden_dim",
    [64, 128, 256, 320, 384, 512, 2432],
    ids=[
        "hidden_dim64",
        "hidden_dim128",
        "hidden_dim256",
        "hidden_dim320",
        "hidden_dim384",
        "hidden_dim512",
        "hidden_dim2432",
    ],
)
@pytest.mark.parametrize("use_affine", [True], ids=["with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [1], ids=["num_simulated_devices1"])
def test_distributed_dit_layernorm_odd_hidden_dim(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    This test case ensures that the hidden dim tiles is a number not divisible by DST size,
    testing that CB sizes are correct in order to prevent hangs.
    """
    # Skip if hidden_dim is not divisible by num_simulated_devices * TILE_SIZE
    if hidden_dim % (num_simulated_devices * TILE_SIZE) != 0:
        pytest.skip(
            f"hidden_dim ({hidden_dim}) must be divisible by num_simulated_devices * TILE_SIZE ({num_simulated_devices * TILE_SIZE})"
        )

    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_dit_layernorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, use_affine, weight_bias_layout
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [9472, 18944],
    ids=["seqlen9472", "seqlen18944"],
)
@pytest.mark.parametrize("hidden_dim", [5120], ids=["hidden_dim5120"])
@pytest.mark.parametrize("use_affine", [True], ids=["with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [2, 4], ids=["num_simulated_devices2", "num_simulated_devices4"])
def test_distributed_dit_layernorm_wan_configs(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    Test configurations matching WAN model requirements.
    """
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_dit_layernorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, use_affine, weight_bias_layout
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
def test_distributed_dit_layernorm_affine_layouts_and_dtypes(
    device,
    dtype,
    stats_dtype,
    weight_bias_layout,
    reset_seeds,
):
    """
    Test weight/bias with different layouts (tile vs row-major) and dtypes (bf16 vs fp32).
    Uses a single reasonable configuration: dim=2048, seqlen=512, num_devices=2.
    """
    num_simulated_devices = 2
    hidden_dim = 2048
    seqlen = 512
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_dit_layernorm(
        device,
        num_simulated_devices,
        inp_shape,
        dtype,
        stats_dtype,
        use_affine=True,
        weight_bias_layout=weight_bias_layout,
    )


def run_distributed_dit_layernorm_batched_affine(
    device,
    num_simulated_devices,
    batch_size,
    seq_len,
    embedding_dim,
    dtype,
    stats_dtype,
):
    """
    Test distributed DIT layernorm with batched weight/bias (different affine per batch).

    This tests the Motif-style usage where weight/bias have shape [batch, 1, embedding_dim],
    applying different affine transformations to each batch element.

    Args:
        device: The ttnn device
        num_simulated_devices: Number of TP devices to simulate
        batch_size: Number of batch elements (each gets different weight/bias)
        seq_len: Sequence length
        embedding_dim: Hidden dimension
        dtype: Input data type
        stats_dtype: Stats data type
    """
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)

    inp_shape = (1, batch_size, seq_len, embedding_dim)
    torch_input = torch.randn(inp_shape) * 4 - 1

    # Create batched weight and bias: shape [batch_size, 1, embedding_dim]
    # Each batch element gets different affine parameters
    torch_weight = torch.rand(batch_size, 1, embedding_dim)
    torch_bias = torch.rand(batch_size, 1, embedding_dim)

    epsilon = 1e-5

    # Chunk input across simulated devices on the last dimension
    tt_inp = chunk_and_shard_tensor(torch_input, num_simulated_devices, device, -1, dtype)

    # For TILE_LAYOUT with batched affine, we need shape [batch_size, TILE_HEIGHT, embedding_dim]
    # But the op expects [batch, 1, dim] style broadcasting, so we use [batch_size, 32, embedding_dim/num_devices]
    # Actually, for tile layout the shape needs to be [batch, 32, dim] where 32 is TILE_HEIGHT
    # Let's just use the simpler approach: shape [batch_size, 32, embedding_dim] and chunk
    weight_tiled = torch_weight.expand(batch_size, TILE_SIZE, embedding_dim)
    bias_tiled = torch_bias.expand(batch_size, TILE_SIZE, embedding_dim)

    tt_weight = chunk_and_shard_tensor(weight_tiled, num_simulated_devices, device, -1, dtype)
    tt_bias = chunk_and_shard_tensor(bias_tiled, num_simulated_devices, device, -1, dtype)

    # Create reciprocal tensor for Welford algorithm
    # Width per chunk is embedding_dim / num_simulated_devices
    chunk_width = embedding_dim // num_simulated_devices
    recip_tensor = create_recip_tensor_for_welford(device, chunk_width)

    tt_stats = []
    for tt_inp_chunk in tt_inp:
        tt_stats.append(
            ttnn.experimental.dit_layernorm_pre_allgather(
                tt_inp_chunk, recip_tensor, compute_kernel_config=compute_kernel_config, dtype=stats_dtype
            )
        )

    # Simulate all-gather by concatenating stats from all devices
    tt_stats_gathered = ttnn.concat(tt_stats, -1)

    # Run post_allgather on each chunk with gathered stats and batched weight/bias
    tt_out = []
    for idx in range(len(tt_inp)):
        tt_out.append(
            ttnn.experimental.dit_layernorm_post_allgather(
                tt_inp[idx],
                tt_stats_gathered,
                epsilon=epsilon,
                weight=tt_weight[idx],
                bias=tt_bias[idx],
                compute_kernel_config=compute_kernel_config,
            )
        )

    # Concat the output on the sharded dimension
    tt_out = ttnn.concat(tt_out, -1)
    tt_out = ttnn.to_torch(tt_out)

    # Reference implementation: layer_norm then apply batched affine
    out_torch = torch.nn.functional.layer_norm(
        torch_input, normalized_shape=(embedding_dim,), weight=None, bias=None, eps=epsilon
    )
    # Apply batched affine: weight/bias shape [batch, 1, dim] broadcasts over [1, batch, seq, dim]
    out_torch = out_torch * torch_weight  # broadcasts [batch, 1, dim] over [1, batch, seq, dim]
    out_torch = out_torch + torch_bias

    # Post-allgather layernorm output has a structural TF32 floor: the final
    # (x - mu) / sqrt(var + eps) * gamma + beta cascade routes through FPU bcast ops
    # which read operands through SrcA/SrcB (19-bit, 10 mantissa). For bf16 stats the
    # bf16 quantization of mean / sqrt(var + eps) dominates; for fp32 stats the TF32
    # FPU floor takes over.
    if dtype == ttnn.float32:
        rtol = 0.01
        atol = 0.05
        pcc = 0.9999
        frobenius_threshold = 0.005
    else:
        rtol = 0.01
        atol = 0.06
        pcc = 0.999
        frobenius_threshold = 0.01
    assert_numeric_metrics(
        out_torch,
        tt_out,
        rtol=rtol,
        atol=atol,
        pcc_threshold=pcc,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize("batch_size", [2, 4], ids=["batch2", "batch4"])
@pytest.mark.parametrize("seq_len", [512, 4096], ids=["seq512", "seq4096"])
@pytest.mark.parametrize("embedding_dim", [1920, 2048], ids=["dim1920_motif", "dim2048"])
@pytest.mark.parametrize("num_simulated_devices", [1, 2], ids=["tp1", "tp2"])
def test_distributed_dit_layernorm_batched_affine(
    device,
    num_simulated_devices,
    batch_size,
    seq_len,
    embedding_dim,
    dtype,
    stats_dtype,
    reset_seeds,
):
    """
    Test batched weight/bias broadcasting for Motif-style dynamic affine.

    In Motif, the dynamic weight/bias have shape [batch, 1, embedding_dim], meaning
    each batch element gets different affine parameters. This tests that the op
    correctly applies different weight/bias per batch.
    """
    # Skip if embedding_dim is not divisible by num_simulated_devices * TILE_SIZE
    if embedding_dim % (num_simulated_devices * TILE_SIZE) != 0:
        pytest.skip(
            f"embedding_dim ({embedding_dim}) must be divisible by num_simulated_devices * TILE_SIZE ({num_simulated_devices * TILE_SIZE})"
        )

    run_distributed_dit_layernorm_batched_affine(
        device,
        num_simulated_devices,
        batch_size,
        seq_len,
        embedding_dim,
        dtype,
        stats_dtype,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize("seqlen", [2048])
@pytest.mark.parametrize("hidden_dim", [8192], ids=["hidden_dim8192"])
@pytest.mark.parametrize("use_affine", [True], ids=["with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [8], ids=["num_simulated_devices8"])
def test_distributed_dit_layernorm_program_cache(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    Test program cache behavior by running the same configuration multiple times.
    """
    inp_shape = (1, 1, seqlen, hidden_dim)
    dummy_tensors = []
    for i in range(2):
        run_distributed_dit_layernorm(
            device,
            num_simulated_devices,
            inp_shape,
            dtype,
            stats_dtype,
            use_affine,
            weight_bias_layout,
        )
        # Allocate dummy tensor to potentially invalidate cache
        dummy_tensors.append(
            ttnn.from_torch(
                torch.zeros(32, 32),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )


@pytest.mark.parametrize(
    "inp_shape",
    [(1, 1, 32, 128), (1, 1, 32, 1024)],
)
@pytest.mark.parametrize(
    "inp_dtype, stats_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
    ],
    ids=["bf16_inp_bf16_stats", "fp32_inp_fp32_stats"],
)
def test_distributed_dit_pre_allgather_welford_precision(device, inp_shape, inp_dtype, stats_dtype, reset_seeds):
    """DiT pre_allgather (welford) per-row mean and var vs torch reference.

    For fp32 stats: scratch CB c_1 is fp32 with UnpackToDestFp32, so the welford recurrence
    and the final transpose round-trip both preserve full fp32 precision. The tolerance is at
    the analytical Welford noise floor (~sqrt(W)*eps_fp32) so any precision regression -- e.g.
    holding a fp32 intermediate in bf16 or losing UnpackToDestFp32 -- is caught.

    For bf16 stats: scratch CB and output are bf16, so the floor is bf16 quantization (~8e-3).
    """
    torch.manual_seed(0)
    w = inp_shape[-1]
    torch_dtype = tt_dtype_to_torch_dtype[inp_dtype]
    torch_inp = torch.randn(inp_shape, dtype=torch_dtype)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_inp = ttnn.from_torch(
        torch_inp,
        dtype=inp_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    recip_tensor = create_recip_tensor_for_welford(device, w)

    tt_stats = ttnn.experimental.dit_layernorm_pre_allgather(
        tt_inp, recip_tensor, compute_kernel_config=compute_kernel_config, dtype=stats_dtype
    )
    out = ttnn.to_torch(tt_stats)

    # DiT welford output layout: per-row mean lives in tile 0 column 0,
    # per-row variance in tile 1 column 0 (= overall column 32).
    torch_inp_fp32 = torch_inp.float()
    torch_mean = torch_inp_fp32.mean(dim=-1, keepdim=False)
    torch_var = torch_inp_fp32.var(dim=-1, keepdim=False, unbiased=False)
    tt_mean = out[..., 0]
    tt_var = out[..., 32]

    # fp32 stats: For W=1024 (the larger parametrization) Welford mean/var noise is bounded by
    # sqrt(W) * eps_fp32 ~ 3.8e-6; atol/rtol 1e-5 leaves ~2x headroom.
    if stats_dtype == ttnn.float32:
        atol = 1e-5
        rtol = 1e-5
        pcc = 0.99999
        frobenius_threshold = 1e-5
    else:
        atol = 0.01
        rtol = 0.01
        pcc = 0.999
        frobenius_threshold = 0.004
    assert_numeric_metrics(
        torch_mean, tt_mean, rtol=rtol, atol=atol, pcc_threshold=pcc, frobenius_threshold=frobenius_threshold
    )
    assert_numeric_metrics(
        torch_var, tt_var, rtol=rtol, atol=atol, pcc_threshold=pcc, frobenius_threshold=frobenius_threshold
    )


@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_dit_layernorm_pre_allgather_fp32_precision(device, offset):
    """dit_layernorm_pre_allgather Welford stats are accurate for Float32 input regardless of mean offset.

    The Welford kernel requires fp32 precision end-to-end: the input CB and the intermediate
    scratch CB must both use Float32 format, and the unpacker must be configured with
    unpack_to_dest_mode=UnpackToDestFp32 so that fp32 values are not silently downcast to
    TF32 (10 mantissa bits) when routed through SrcA. When either of these conditions is
    violated, the Welford (x - M) subtraction catastrophically loses precision at large offsets
    because the subtracted values share a large common exponent.
    """
    torch.manual_seed(0)
    embedding_dim = 128
    shape = (1, 1, 32, embedding_dim)
    torch_input = torch.randn(shape, dtype=torch.float32) + offset

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    recip_tensor = create_recip_tensor_for_welford(device, embedding_dim)
    tt_inp = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_stats = ttnn.experimental.dit_layernorm_pre_allgather(
        tt_inp, recip_tensor, compute_kernel_config=compute_kernel_config, dtype=ttnn.float32
    )
    actual = ttnn.to_torch(tt_stats)

    # dit_pre returns Welford (mean, variance), unlike the legacy distributed pre_all_gather
    # which returns (sum(x^2), sum(x)). The Welford kernel uses welford_finalize_to_row to
    # convert M2 -> variance and writes (mean, var) into the row-broadcast slots.
    # Output layout: column 0 of tile 0 holds the mean, column 0 of tile 1 (offset 32) holds
    # variance.  Reference is computed in fp64 so it isn't itself contaminated by fp32 noise.
    torch_mean = torch_input.to(torch.float64).mean(dim=-1)
    torch_var = torch_input.to(torch.float64).var(dim=-1, correction=0)

    tt_mean = actual[..., 0].to(torch.float64).squeeze(-1)
    tt_var = actual[..., 32].to(torch.float64).squeeze(-1)

    mean_pcc_threshold = 0.99999
    mean_frob = 1e-5
    if offset == 0.0:
        # No catastrophic cancellation: Welford is at the fp32 noise floor for W=128 (mean/var
        # error ~sqrt(W)*eps_fp32 ~ 1.4e-6), so tolerances can be tight to catch any precision
        # regression.
        mean_check_pcc = True
        mean_rtol = 1e-7
        mean_atol = 1e-7
        var_pcc_threshold = 0.99999
        var_frob = 1e-5
        var_rtol = 1e-5
        var_atol = 1e-5
    else:
        # At large offset, Welford mean stagnates once delta/k < ULP(offset)/2; the final mean
        # reflects only the first few samples, giving low theoretical PCC.  Intrinsic to
        # stagnation, so PCC check is disabled.
        mean_check_pcc = False
        mean_rtol = 6e-7
        mean_atol = 1e-5
        # Variance error per row has a long tail. Typical error stays small so PCC stays high.
        # Relative Frobenius is larger here than for the mean: even though the variance's
        # absolute error is smaller than the mean's, it's divided by ~1.0 (variance is
        # translation-invariant), while the mean's larger absolute error is dwarfed when
        # divided by ~1e6.
        var_pcc_threshold = 0.95
        var_frob = 0.05
        var_rtol = 0.001
        var_atol = 0.25

    mean_passed, mean_msg = assert_numeric_metrics(
        torch_mean,
        tt_mean,
        rtol=mean_rtol,
        atol=mean_atol,
        frobenius_threshold=mean_frob,
        pcc_threshold=mean_pcc_threshold,
        check_pcc=mean_check_pcc,
        assert_on_fail=False,
    )
    var_passed, var_msg = assert_numeric_metrics(
        torch_var,
        tt_var,
        rtol=var_rtol,
        atol=var_atol,
        frobenius_threshold=var_frob,
        pcc_threshold=var_pcc_threshold,
        assert_on_fail=False,
    )
    assert mean_passed and var_passed, (
        f"offset={offset}\n"
        f"--- MEAN: {'PASSED' if mean_passed else 'FAILED'} ---\n{mean_msg}\n"
        f"--- VARIANCE: {'PASSED' if var_passed else 'FAILED'} ---\n{var_msg}"
    )
