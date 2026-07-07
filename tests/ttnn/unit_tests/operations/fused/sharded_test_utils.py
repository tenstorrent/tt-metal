# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.test_utils import TILE_HEIGHT, TILE_WIDTH


def generate_input_tensor(h, w, type, dtype):
    """
    Generate various torch tensors
    Returns:
        A torch tensor of shape (h, w) of the given type and dtype.
    """
    if type == "random":
        return torch.rand((h, w), dtype=dtype)
    elif type == "random_normal":
        return torch.randn((h, w), dtype=dtype)
    elif type == "ascending_values_repeated_rows":
        return torch.arange(w).repeat(h, 1).to(dtype)
    elif type == "monotonically_ascending_values":
        return torch.arange(h * w).reshape(h, w).to(dtype)
    else:
        raise ValueError(f"Invalid tensor type: {type}")


def create_shard_spec(shard_height, shard_width, num_cores_h, num_cores_w):
    """
    Create a shard spec for a given height, width, number of cores, and number of cores per row.
    Args:
        shard_height: The height of the shard.
        shard_width: The width of the shard.
        num_cores_h: The number of cores in y
        num_cores_w: The number of cores in x.
    Returns:
        Shard spec corresponding to given input shapes
    """
    return ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )


def create_single_stage_shard_spec(h, w, num_cores_h, num_cores_w):
    """
    Create a shard spec that will trigger the single-stage reduction
    algorithm
    Args:
        h: The height of the input tensor.
        w: The width of the input tensor.
        num_cores_h: The number of cores in y.
        num_cores_w: The number of cores in x.
    Returns:
        Shard spec corresponding to given input shapes
    """
    shard_height = h // num_cores_h
    shard_width = w // num_cores_w
    return create_shard_spec(shard_height, shard_width, num_cores_h, num_cores_w)


def create_two_stage_shard_spec(h, w, num_cores_h, num_cores_w):
    """
    Create a shard spec that will trigger the two-stage reduction
    algorithm
    Args:
        h: The height of the input tensor.
        w: The width of the input tensor.
        num_cores_h: The number of cores in y.
        num_cores_w: The number of cores in x.
    Returns:
        Shard spec corresponding to given input shapes
    """
    shard_height = h
    shard_width = w // (num_cores_w * num_cores_h)
    return create_shard_spec(shard_height, shard_width, num_cores_h, num_cores_w)


def create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage):
    """
    Create a sharded memory config for a given height, width, number of cores, and number of cores per row..
    Sets up the config to trigger the single-stage or two-stage reduction algorithm based
    on the input flag
    Args:
        h: The height of the input tensor.
        w: The width of the input tensor.
        num_cores_h: The number of cores in y.
        num_cores_w: The number of cores in x.
        two_stage: Whether to use the two-stage reduction algorithm.
    Returns:
        Sharded memory config corresponding to given input shapes
    """
    mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED if two_stage else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    if two_stage:
        shard_spec = create_two_stage_shard_spec(h, w, num_cores_h, num_cores_w)
    else:
        shard_spec = create_single_stage_shard_spec(h, w, num_cores_h, num_cores_w)
    return ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)


def make_sharded_norm_mem_config(num_cores_w, h, shard_w, num_cores_h=1):
    """Block-sharded SRAM config over a num_cores_w x num_cores_h grid, each core owning an [h, shard_w] shard."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}),
        [h, shard_w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )


def to_poisoned_sharded(device, torch_tensor, mem_config, pad_value):
    """Tilize onto device with the given sharded config and poison the implicit tile padding with pad_value.

    Poisoning makes any read of the padded columns observable: a kernel that normalizes over the logical
    width is unaffected, while one that folds the padded columns into its statistics is grossly wrong.
    """
    tt = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    return ttnn.fill_implicit_tile_padding(tt, pad_value)


def ttnn_layer_norm_sharded(
    device, tt_input_tensor, use_welford, block_ht, block_wt, subblock_w=1, residual=None, weight=None, bias=None
):
    """
    Run layer norm sharded on a TTNN tensor.
    Args:
        device: The device to run the layer norm on.
        tt_input_tensor: The TTNN tensor to run the layer norm on.
        use_welford: Whether to use Welford's algorithm.
        block_ht: The height of the block in tiles.
        block_wt: The width of the block in tiles.
        subblock_w: The width of the subblock in tiles.
        residual: The residual tensor to add to the input tensor.
        weight: The weight tensor to use for the layer norm.
        bias: The bias tensor to use for the layer norm.
    Returns:
        The output tensor as a torch tensor.
    """
    # Create output memory config (same sharding as input)
    output_memory_config = ttnn.get_memory_config(tt_input_tensor)

    # Create reciprocal tensor for Welford algorithm if needed
    recip_tensor = None
    if use_welford:
        shard_spec = tt_input_tensor.memory_config().shard_spec
        recip_tensor = ttnn.create_layer_norm_reciprocals(device, shard_spec.grid, shard_spec.shape[1])

    # Run layernorm
    output_ttnn = ttnn.layer_norm(
        tt_input_tensor,
        residual_input_tensor=residual,
        weight=weight,
        bias=bias,
        memory_config=output_memory_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=subblock_w,
            use_welford=use_welford,
            inplace=False,
        ),
        recip_tensor=recip_tensor,
    )

    output_ttnn = ttnn.from_device(output_ttnn)
    return ttnn.to_torch(output_ttnn)


def ttnn_rms_norm_sharded(
    device, tt_input_tensor, block_ht, block_wt, subblock_w=1, residual=None, weight=None, bias=None
):
    """
    Run rms norm sharded on a TTNN tensor.
    Args:
        device: The device to run the rms norm on.
        tt_input_tensor: The TTNN tensor to run the rms norm on.
        block_ht: The height of the block in tiles.
        block_wt: The width of the block in tiles.
        subblock_w: The width of the subblock in tiles.
        residual: The residual tensor to add to the input tensor.
        weight: The weight tensor to use for the rms norm.
        bias: The bias tensor to use for the rms norm.
    Returns:
        The output tensor as a torch tensor.
    """
    # Create output memory config (same sharding as input)
    output_memory_config = ttnn.get_memory_config(tt_input_tensor)

    # Run rms norm
    output_ttnn = ttnn.rms_norm(
        tt_input_tensor,
        residual_input_tensor=residual,
        weight=weight,
        bias=bias,
        memory_config=output_memory_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=subblock_w,
            use_welford=False,
            inplace=False,
        ),
    )

    output_ttnn = ttnn.from_device(output_ttnn)
    return ttnn.to_torch(output_ttnn)


def torch_layer_norm(torch_input_tensor, residual=None, weight=None, bias=None):
    """
    Run layer norm in torch
    Args:
        torch_input_tensor: The input tensor to run the layer norm on.
        residual: The residual tensor to add to the input tensor.
        weight: The weight tensor to use for the layer norm.
        bias: The bias tensor to use for the layer norm.
    Returns:
        The output tensor as a torch tensor.
    """
    return torch.nn.functional.layer_norm(
        torch_input_tensor if residual is None else torch_input_tensor + residual,
        normalized_shape=[torch_input_tensor.shape[1]],
        weight=weight,
        bias=bias,
    )


def rms_norm_golden(input_tensor, weight=None, bias=None, epsilon=1e-5):
    """
    Compute golden RMS output for an input tensor with optional weight and bias.
    Args:
        input_tensor: The input tensor to run the rms norm on.
        weight: The weight tensor (gamma) to use for the rms norm, or None.
        bias: The bias tensor (beta) to use for the rms norm, or None.
        epsilon: Epsilon for rsqrt stability.
    Returns:
        The output tensor as a torch tensor.
    """
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    golden_rms = golden_function(input_tensor, weight)
    if bias is not None:
        golden_rms = golden_rms + bias

    return golden_rms


def simple_size_params(two_stage):
    """
    Generate a valid single set of shape inputs for either
    the single-stage or two-stage reduction algorithm
    """
    h = 32 * 8
    w = 32 * 10
    num_cores_h = 2
    num_cores_w = 5
    if two_stage:
        block_ht = 8
        block_wt = 1
        subblock_wt = 1
    else:
        block_ht = 4
        block_wt = 2
        subblock_wt = 1

    return h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt


def single_stage_param_sets():
    """
    Generate valid single-stage reduction tensor,block and shard shapes
    for input h,w
    Returns:
        List[Tuple] of (h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt) sets
    """
    hs = [32 * 2, 32 * 8]
    ws = [32 * 2, 32 * 16]
    num_cores_hs = [4, 8]
    num_cores_ws = [4, 8]
    block_ht_mults = [1, 2, 4, 8]
    block_wt_mults = [1, 2, 4, 8]
    possible_subblock_wts = [1, 2]
    param_sets = []
    for h in hs:
        for w in ws:
            for num_cores_h in num_cores_hs:
                for num_cores_w in num_cores_ws:
                    h_per_core = h // num_cores_h
                    w_per_core = w // num_cores_w
                    ht_per_core = h_per_core // TILE_HEIGHT
                    wt_per_core = w_per_core // TILE_WIDTH
                    possible_block_hts = [
                        v for v in [h_per_core // (TILE_HEIGHT * m) for m in block_ht_mults] if v >= 1
                    ]
                    possible_block_wts = [v for v in [w_per_core // (TILE_WIDTH * m) for m in block_wt_mults] if v >= 1]
                    for block_ht in possible_block_hts:
                        for block_wt in possible_block_wts:
                            for subblock_wt in possible_subblock_wts:
                                block_ht_valid = block_ht >= 1 and block_ht == ht_per_core
                                block_wt_valid = block_wt >= 1 and block_wt == wt_per_core
                                subblock_wt_valid = subblock_wt >= 1 and subblock_wt <= block_wt
                                if block_ht_valid and block_wt_valid and subblock_wt_valid:
                                    param_sets.append((h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt))
    return param_sets


def do_test_main(
    device,
    h,
    w,
    num_cores_h,
    num_cores_w,
    block_ht,
    block_wt,
    subblock_wt,
    use_welford,
    two_stage,
    tensor_type,
    dtype,
    residual=None,
    weight=None,
    bias=None,
    weight_bias_layout=ttnn.TILE_LAYOUT,
    op_name="layer_norm",
):
    """
    Helper function to run the layer norm or rms norm tests.
    Runs the ttnn op and compares to a reference tensor (torch or golden).
    Asserts PCC agreement
    """

    if op_name not in ["layer_norm", "rms_norm"]:
        raise ValueError(f"Invalid operation name: {op_name}")

    torch.manual_seed(12345)

    # Run torch layernorm to get the reference tensor
    torch_input_tensor = generate_input_tensor(h, w, tensor_type, dtype)
    if op_name == "layer_norm":
        ref_output_tensor = torch_layer_norm(torch_input_tensor, residual=residual, weight=weight, bias=bias)
    elif op_name == "rms_norm":
        ref_output_tensor = rms_norm_golden(
            torch_input_tensor + residual if residual is not None else torch_input_tensor,
            weight=weight,
            bias=bias,
        )

    # Generate the tt tensor based on the inputs
    sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=sharded_mem_config,
    )

    if residual is not None:
        residual = ttnn.from_torch(residual, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config)
    if weight is not None:
        weight = ttnn.from_torch(weight, layout=weight_bias_layout, device=device)
    if bias is not None:
        bias = ttnn.from_torch(bias, layout=weight_bias_layout, device=device)

    # Run the ttnn op
    if op_name == "layer_norm":
        output_ttnn = ttnn_layer_norm_sharded(
            device,
            tt_input_tensor,
            use_welford,
            block_ht,
            block_wt,
            subblock_wt,
            residual=residual,
            weight=weight,
            bias=bias,
        )
    elif op_name == "rms_norm":
        output_ttnn = ttnn_rms_norm_sharded(
            device,
            tt_input_tensor,
            block_ht,
            block_wt,
            subblock_wt,
            residual=residual,
            weight=weight,
            bias=bias,
        )

    if op_name == "layer_norm":
        if use_welford:
            pcc_threshold = 0.99975
            rtol = 0.14
            atol = 0.085
            frobenius_threshold = 0.02
        else:
            pcc_threshold = 0.9999
            rtol = 0.065
            atol = 0.065
            frobenius_threshold = 0.014
    else:
        if dtype == torch.bfloat16:
            pcc_threshold = 0.999
            rtol = 0.031
            atol = 0.052
            frobenius_threshold = 0.010
        else:
            pcc_threshold = 0.999
            rtol = 0.060
            atol = 0.049
            frobenius_threshold = 0.011
    assert_numeric_metrics(
        ref_output_tensor,
        output_ttnn,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


def layernorm_test_main(
    device,
    h,
    w,
    num_cores_h,
    num_cores_w,
    block_ht,
    block_wt,
    subblock_wt,
    use_welford,
    two_stage,
    tensor_type,
    dtype,
    residual=None,
    weight=None,
    bias=None,
    weight_bias_layout=ttnn.TILE_LAYOUT,
):
    """
    Run the layer norm test for inputs
    """
    do_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=weight,
        bias=bias,
        weight_bias_layout=weight_bias_layout,
        op_name="layer_norm",
    )


def rms_norm_test_main(
    device,
    h,
    w,
    num_cores_h,
    num_cores_w,
    block_ht,
    block_wt,
    subblock_wt,
    two_stage,
    tensor_type,
    dtype,
    residual=None,
    weight=None,
    bias=None,
    weight_layout=ttnn.TILE_LAYOUT,
):
    """
    Run the rms norm test for inputs
    """
    do_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        False,  # use_welford
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=weight,
        bias=bias,
        weight_bias_layout=weight_layout,
        op_name="rms_norm",
    )


# Tolerances for the "normalize over the logical width" correctness check below, derived from the bf16
# error budget.
# bf16 unit roundoff (round-to-nearest, 7 stored mantissa bits): u = 2^-8 ~= 0.0039. The input and unit
# gamma / zero beta are quantized to bf16 and the same quantized values feed both the device op and the
# torch reference, so input quantization cancels. With fp32 dest accumulation the reduction is exact to
# fp32; the residual is the 1/sqrt(var+eps) SFPU step (a few bf16 ulps, bounded generously by 8u ~=
# 0.031, acting as a per-row scale) plus output bf16 quantization (<= u). So a correct result has
# relative-Frobenius <= ~0.035 and per-element relative error <= ~0.035; the tolerances sit a margin
# above that. The geometries are chosen so a padded-width normalization gives a >= 13% scale error
# (w96/c2 -> sqrt(128/96) = 1.155, w224/c3 -> sqrt(288/224) = 1.134), several times the tolerance.
# Per-dtype numeric budget for the logical-width masking check, given as assert_numeric_metrics kwargs.
# This test detects a padded-vs-logical normalization: a near-pure scale error of magnitude
# |sqrt(padded / logical) - 1|, which is at least ~0.13 for the geometries covered here. Every budget
# below stays well under that (so the scale error is still caught) while leaving room for each format's
# own quantization noise: fp32 tightest, bfloat8_b loosest. atol is the floor for near-zero normalized
# outputs, where relative tolerance is meaningless.
_LOGICAL_WIDTH_NORM_BUDGET = {
    ttnn.float32: dict(frobenius_threshold=0.02, rtol=0.02, atol=0.02),
    ttnn.bfloat16: dict(frobenius_threshold=0.06, rtol=0.05, atol=0.05),
    ttnn.bfloat8_b: dict(frobenius_threshold=0.10, rtol=0.08, atol=0.08),
}
_LOGICAL_WIDTH_NORM_PAD_POISON = 1000.0  # poison the implicit tile padding so any leak is caught


# Block sharding mandates that every core must get the same-sized tile-aligned shard:
# shard_w = ceil(w / cores / 32) * 32, so the padded width is cores * shard_w.
# This means the final core's shard may have padding.
# Two categories are covered:
#   - tile-aligned widths whose tiles do not divide evenly across the cores (96 over 2, 224 over 3):
#     every tile on the final core is either fully valid or fully padding, never partial. E.g. 96 over
#     2 gives the final core one fully-valid tile and one fully-padding tile.
#   - non-tile-aligned widths (72 over 2, 200 over 3): the logical columns run out mid-tile, so the
#     final core holds a partially-valid tile followed by a fully-padding tile. E.g. 72 over 2 gives
#     the final core one partially-valid tile (8 of its 32 columns valid) and one fully-padding tile.
# In both, the op must normalize over the logical width, not the padded per-core width.
UNEVEN_MULTICORE_LOGICAL_WIDTH_CASES = [(96, 2), (224, 3), (72, 2), (200, 3)]
UNEVEN_MULTICORE_LOGICAL_WIDTH_IDS = ["w96_c2", "w224_c3", "w72_c2_nonaligned", "w200_c3_nonaligned"]


def run_sharded_norm_logical_width_multicore(
    device,
    is_rmsnorm,
    w,
    num_cores_w,
    num_cores_h=1,
    dtype=ttnn.bfloat16,
    eps=1e-5,
    use_welford=False,
    weight_layout=ttnn.TILE_LAYOUT,
):
    """Verify a width-sharded layer/RMS norm normalizes over the LOGICAL width when the width is split
    across cores so the final core owns fewer real tiles (and, for a non-tile-aligned width, a partially
    valid final tile plus pure-padding tiles).

    Covers two related cases with one path: a tile-aligned width whose tiles do not divide evenly across
    cores (e.g. 96 over 2), and a non-tile-aligned width split across cores (e.g. 200 over 2).
    The statistics should be reduced over the logical element count only (excluding padding).
    Dividing by the padded count (num_blocks * block_w) instead causes the output to be too large by
    a factor of ~sqrt(padded/logical), which is a near-pure scale. The check therefore gates on the
    relative-Frobenius and allclose metrics (both scale-sensitive) and DISABLES PCC, which is
    scale-invariant and so blind to this class of error. Tolerances come from the per-dtype budget
    defined above.

    The width is spread over all num_cores_w * num_cores_h cores. A single-row grid (num_cores_h == 1)
    uses the single-stage cross-core reduce; a 2D grid (both dimensions > 1) selects the two-stage reduce
    (first stage combines the shards within a row, second combines the per-row results), with the
    partially-valid boundary block on the final core in row-major order.

    use_welford applies to LayerNorm only, since RMSNorm does not support Welford.

    weight_layout selects the gamma/beta layout: TILE_LAYOUT or ROW_MAJOR_LAYOUT (which selects the
    row-major gamma/beta writer kernel).
    """
    assert not (use_welford and is_rmsnorm), "RMSNorm does not use the Welford reduction"
    total_cores = num_cores_w * num_cores_h
    kt = -(-w // TILE_WIDTH)  # ceil: a non-tile-aligned width rounds up to a whole number of tiles
    shard_wt = -(-kt // total_cores)  # tiles per core (ceil) -> last core owns fewer real tiles
    shard_w = shard_wt * TILE_WIDTH
    assert (total_cores - 1) * shard_wt < kt, "geometry must leave the last core at least one real tile"

    torch.manual_seed(0)
    x = torch.randn(1, 1, TILE_HEIGHT, w)  # fp32 base; quantized to `dtype` on device below

    core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}
    )
    sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(TILE_HEIGHT, shard_w),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    tt_x = ttnn.to_memory_config(tt_x, memory_config=sharded_cfg)
    tt_x = ttnn.fill_implicit_tile_padding(tt_x, _LOGICAL_WIDTH_NORM_PAD_POISON)

    # Reference is computed from exactly the values the op consumes: read the sharded, padding-poisoned
    # input back and take the logical width. For a lossy format (bfloat8_b) the block-shared exponent
    # means the padding value can perturb the boundary tile, so deriving the reference from the actual
    # input keeps the comparison fair; any residual error then reflects a masking failure rather than
    # input quantization.
    xf = ttnn.to_torch(ttnn.to_memory_config(tt_x, ttnn.L1_MEMORY_CONFIG)).float()[..., :w]
    if is_rmsnorm:
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    else:
        ref = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(xf.var(-1, unbiased=False, keepdim=True) + eps)

    # gamma = 1, beta = 0 isolate the normalization (the affine is trivially correct and would only
    # hide a scale error behind PCC). gamma padded width must equal the input padded width (= w).
    if weight_layout == ttnn.ROW_MAJOR_LAYOUT:
        # A row-major gamma/beta presents its last padded dim as a whole tile width, with the tile count
        # carried in an earlier dim, so it is shaped [kt, TILE_WIDTH]. Its volume aligns with the input's
        # padded width (kt whole tiles), not the wider physical shard span. The elements past the logical
        # width w are gamma's tile padding and must be zero, matching how tile-layout gamma is zero-padded:
        # otherwise the padding columns' normalized (poisoned) values reach the output, and for a lossy
        # sharded output their magnitude corrupts the boundary tile's shared exponent.
        gamma_flat = torch.zeros(kt * TILE_WIDTH, dtype=torch.bfloat16)
        gamma_flat[:w] = 1.0
        gamma_torch = gamma_flat.reshape(kt, TILE_WIDTH)
        beta_torch = torch.zeros(kt, TILE_WIDTH, dtype=torch.bfloat16)
    else:
        gamma_torch = torch.ones(1, 1, 1, w, dtype=torch.bfloat16)
        beta_torch = torch.zeros(1, 1, 1, w, dtype=torch.bfloat16)
    gamma = ttnn.from_torch(gamma_torch, layout=weight_layout, device=device)

    # fp32 dest caps subblock_w at 4; keep it a divisor of the shard width in tiles.
    subblock_w = next(d for d in range(min(shard_wt, 4), 0, -1) if shard_wt % d == 0)
    prgm = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[num_cores_w, num_cores_h],
        subblock_w=subblock_w,
        block_h=1,
        block_w=shard_wt,
        use_welford=use_welford,
        inplace=False,
    )
    # fp32 dest + HiFi4 isolate the normalization from accumulation/fidelity error (see budget above).
    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    common = dict(
        epsilon=eps, weight=gamma, program_config=prgm, compute_kernel_config=compute_cfg, memory_config=sharded_cfg
    )
    if use_welford:
        # Welford normalizes via a reciprocal LUT rather than a divide, so it needs the per-core LUT.
        common["recip_tensor"] = ttnn.create_layer_norm_reciprocals(
            device, sharded_cfg.shard_spec.grid, sharded_cfg.shard_spec.shape[1]
        )
    if is_rmsnorm:
        tt_out = ttnn.rms_norm(tt_x, **common)
    else:
        beta = ttnn.from_torch(beta_torch, layout=weight_layout, device=device)
        tt_out = ttnn.layer_norm(tt_x, bias=beta, **common)
    out = ttnn.to_torch(ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)).float()[..., :w]

    assert_numeric_metrics(
        ref,
        out,
        **_LOGICAL_WIDTH_NORM_BUDGET[dtype],
        check_pcc=False,  # PCC is scale-invariant; a padded-width normalization is a (near-)pure scale
    )
