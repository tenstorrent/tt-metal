import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
import math


def partitioned_vol2col_conv3d_slicing(
    input: torch.Tensor,
    conv3d_module: torch.nn.Conv3d,
    depth_block: int = 1,
    hw_block: int = 1,
    out_chan_block: int = 1,
):
    """
    Emulates a 3D convolution by partitioning the *output volume* in (D, H/W, out_channels),
    then using a vol2col + matrix-multiply (GEMM) on each sub-volume.

    - Stride is forced to (1,1,1)
    - Dilation is forced to (1,1,1)
    - Groups are forced to 1
    - We gather patches via Python slicing (no advanced index tensors).
    - This is a teaching example, not optimized for speed.

    Args:
        input: [N, C_in, D_in, H_in, W_in]
        conv3d_module: an nn.Conv3d with:
            - stride=(1,1,1)
            - dilation=(1,1,1)
            - groups=1
        depth_parallel: how many depth slices of the output to process at once
        hw_parallel: how many height/width slices of the output to process at once
        out_chan_parallel: how many output channels to process at once

    Returns:
        output: [N, out_channels, D_out, H_out, W_out]
    """

    # -----------------
    # 1) Extract Params
    # -----------------
    assert conv3d_module.stride == (1, 1, 1), "This example only supports stride=1"
    assert conv3d_module.dilation == (1, 1, 1), "This example only supports dilation=1"
    assert conv3d_module.groups == 1, "This example assumes groups=1"

    weight = conv3d_module.weight  # [out_channels, C_in, kD, kH, kW]
    bias = conv3d_module.bias
    pad_d, pad_h, pad_w = conv3d_module.padding
    kD, kH, kW = conv3d_module.kernel_size
    out_channels = conv3d_module.out_channels

    N, C_in, D_in, H_in, W_in = input.shape

    # ----------------------
    # 2) Compute Output Size
    # ----------------------
    # For stride=1, dilation=1, groups=1:
    # out_dim = in_dim + 2*pad - (kernel - 1)
    def _out_size(in_size, pad, k):
        return in_size + 2 * pad - (k - 1)

    D_out = _out_size(D_in, pad_d, kD)
    H_out = _out_size(H_in, pad_h, kH)
    W_out = _out_size(W_in, pad_w, kW)

    # -------------------------------------------
    # 3) Pad input along D, H, W (if needed)
    # -------------------------------------------
    if conv3d_module.padding_mode == "zeros":
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="constant", value=0)
    elif conv3d_module.padding_mode == "replicate":
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding_mode {conv3d_module.padding_mode}")

    # input_padded now has shape [N, C_in, D_in + 2*pad_d, H_in + 2*pad_h, W_in + 2*pad_w]
    D_pad, H_pad, W_pad = input_padded.shape[2:]

    # -----------------------------
    # 4) Allocate final output
    # -----------------------------
    output = torch.zeros((N, out_channels, D_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # ------------------------------------------------------------------
    # 5) Partition the output volume in D, (H/W), and out_channels.
    #
    #   For each sub-volume:
    #     (A) Slice a sub-region of input_padded (slightly bigger
    #         so we can gather the kD,kH,kW patches).
    #     (B) Gather patches via Python slicing (3 nested loops).
    #     (C) Flatten patches => col matrix.
    #     (D) Multiply by flattened weights => partial out.
    #     (E) Reshape => place into 'output'.
    # ------------------------------------------------------------------

    for d_start in range(0, D_out, depth_block):
        d_end = min(d_start + depth_block, D_out)
        d_blk_size = d_end - d_start

        for h_start in range(0, H_out, hw_block):
            h_end = min(h_start + hw_block, H_out)
            h_blk_size = h_end - h_start

            for w_start in range(0, W_out, hw_block):
                w_end = min(w_start + hw_block, W_out)
                w_blk_size = w_end - w_start

                for oc_start in range(0, out_channels, out_chan_block):
                    oc_end = min(oc_start + out_chan_block, out_channels)
                    out_chan_blk_size = oc_end - oc_start

                    # -------------
                    # (A) Sub-slice
                    # -------------
                    # For outputs [d_start:d_end, h_start:h_end, w_start:w_end],
                    # we read input_padded from:
                    #   depth range  [d_start, d_end) + up to (kD-1)
                    #   height range [h_start, h_end) + up to (kH-1)
                    #   width range  [w_start, w_end) + up to (kW-1)
                    d_slice_end = min(d_end + (kD - 1), D_pad)
                    h_slice_end = min(h_end + (kH - 1), H_pad)
                    w_slice_end = min(w_end + (kW - 1), W_pad)

                    sub_input = input_padded[:, :, d_start:d_slice_end, h_start:h_slice_end, w_start:w_slice_end]
                    # shape => [N, C_in,
                    #           (d_blk_size + kD - 1), (h_blk_size + kH - 1), (w_blk_size + kW - 1)]

                    num_patches = N * d_blk_size * h_blk_size * w_blk_size
                    patch_size = C_in * kD * kH * kW  # (groups=1)

                    # We'll gather all patches in a list, then stack once at the end
                    patches = []

                    # Triple nested loop over kernel offsets kd, kh, kw
                    #   sub_input[:, :, kd: kd + d_blk_size, kh: kh + h_blk_size, kw_: kw_ + w_blk_size]
                    # => [N, C_in, d_blk_size, h_blk_size, w_blk_size]

                    # Iterate over all possible output points in this block
                    for dd in range(d_end - d_start):
                        for hh in range(h_end - h_start):
                            for ww in range(w_end - w_start):
                                slice_5d = sub_input[:, :, dd : dd + kD, hh : hh + kH, ww : ww + kW]
                                assert slice_5d.shape == (N, C_in, kD, kH, kW)
                                # slice_5d = slice_5d.permute(0, 2, 3, 4, 1).contiguous()
                                # (N, kD*kH*kW*C_in)
                                slice_2d = slice_5d.reshape(N, -1)
                                patches.append(slice_2d)

                    # Stack on dim=1 => [num_patches, C_in*kD*kH*kW]
                    col = torch.cat(patches, dim=0)

                    # ----------------------------------------
                    # (B) Flattened weight slice + matmul
                    # ----------------------------------------
                    # weight => [out_channels, C_in, kD, kH, kW]
                    # Flatten => [out_channels, patch_size]
                    w_chunk = weight[oc_start:oc_end].reshape(out_chan_blk_size, patch_size)
                    # col @ w_chunk^T => [num_patches, out_chan_blk_size]
                    out_mat = col @ w_chunk.transpose(0, 1)

                    # ---------------------------------------
                    # (C) Reshape => sub-volume & store
                    # ---------------------------------------
                    # out_mat: [N*d_blk_size*h_blk_size*w_blk_size, out_chan_blk_size]
                    # => [N, d_blk_size, h_blk_size, w_blk_size, out_chan_blk_size]
                    # out_mat_5d = out_mat.view(
                    #     N, d_blk_size, h_blk_size, w_blk_size, out_chan_blk_size
                    # )
                    out_mat_5d = out_mat.reshape(N, d_blk_size, h_blk_size, w_blk_size, out_chan_blk_size)
                    # permute => [N, out_chan_blk_size, d_blk_size, h_blk_size, w_blk_size]
                    out_mat_5d = out_mat_5d.permute(0, 4, 1, 2, 3).contiguous()

                    # Place into final output
                    output[:, oc_start:oc_end, d_start:d_end, h_start:h_end, w_start:w_end] = out_mat_5d

    # ----------------------------------
    # 6) Add bias if needed (once overall)
    # ----------------------------------
    if bias is not None:
        # Broadcast bias [out_channels] over all dims
        output += bias.view(1, -1, 1, 1, 1)

    return output


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 5, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["test0", "variant0", "variant1", "variant2", "variant3", "variant4"],
)
@pytest.mark.parametrize("T_parallel_factor", [8])
def test_vol2col_conv3d_torch(input_shape, out_channels, kernel_size, stride, padding, padding_mode, T_parallel_factor):
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)

    # Define input dimensions.
    N, C, D, H, W = input_shape
    D = math.ceil(D / T_parallel_factor)

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    # Create a Conv3d module with chosen parameters.
    in_channels = C
    dilation = (1, 1, 1)
    conv3d_module = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
        padding_mode=padding_mode,
    )

    # Compute the output using PyTorch's built-in conv3d.
    output_builtin = conv3d_module(input_tensor)

    # Compute the output using the decomposed conv3d (based on conv2d).
    output_decomposed = partitioned_vol2col_conv3d_slicing(
        input_tensor,
        conv3d_module,
        depth_block=2,
        hw_block=2,
        out_chan_block=2,
    )

    pcc, mse, mae = compute_metrics(output_builtin, output_decomposed)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"


def naive_estimation(
    N,
    C_in,
    D_out,
    H_out,
    W_out,
    out_channels,
    kD,
    kH,
    kW,
    input_padded,
    weight,
    depth_blocks,
    hw_blocks,
    out_chan_blocks,
    BYTES_PER_DATUM,
):
    results = {}
    import itertools
    import tqdm

    for depth_block, hw_block, out_chan_block in tqdm.tqdm(itertools.product(depth_blocks, hw_blocks, out_chan_blocks)):
        total_activation_datums = 0
        total_weight_datums = 0
        max_block_l1_bytes = 0

        """
        Calculate the amount of L1 used for a single block
        Calculate the total amount of data read from memory
        """

        num_d_blocks = math.ceil(D_out / depth_block)
        num_h_blocks = math.ceil(H_out / hw_block)
        num_w_blocks = math.ceil(W_out / hw_block)
        num_oc_blocks = math.ceil(out_channels / out_chan_block)

        total_num_blocks = num_d_blocks * num_h_blocks * num_w_blocks * num_oc_blocks

        # Input patch to read from DRAM
        input_buffer_datums = (
            input_padded.shape[0]
            * input_padded.shape[1]
            * (depth_block + kD - 1)
            * (hw_block + kH - 1)
            * (hw_block + kW - 1)
        )
        # Across all blocks, this is the total amount of data read from DRAM
        total_activation_datums = input_buffer_datums * total_num_blocks

        num_patches = N * depth_block * hw_block * hw_block
        patch_size = C_in * kD * kH * kW

        # Buffer sizes for vol2col and weight
        vol2col_buffer_datums = num_patches * patch_size
        weight_buffer_datums = out_chan_block * patch_size
        # Across all blocks, this is the total amount of data read from DRAM
        total_weight_datums = weight_buffer_datums * total_num_blocks

        # Buffer sizes for output
        out_col_buffer_datums = num_patches * out_chan_block
        out_vol_buffer_datums = out_col_buffer_datums

        # For a single block, this is the total amount of L1 used
        max_block_l1_bytes = BYTES_PER_DATUM * (
            input_buffer_datums
            + vol2col_buffer_datums
            + weight_buffer_datums
            + out_col_buffer_datums
            + out_vol_buffer_datums
        )

        results[depth_block, hw_block, out_chan_block] = {
            "total_activation_bytes": total_activation_datums * BYTES_PER_DATUM,
            "total_weight_bytes": total_weight_datums * BYTES_PER_DATUM,
            "max_block_l1_bytes": max_block_l1_bytes,
        }
    return results


def weight_reuse_estimation(
    N,
    C_in,
    D_out,
    H_out,
    W_out,
    out_channels,
    kD,
    kH,
    kW,
    input_padded,
    weight,
    depth_blocks,
    hw_blocks,
    out_chan_blocks,
    BYTES_PER_DATUM,
):
    N_CORES = 64
    results = {}
    import itertools
    import tqdm

    for depth_block, hw_block, out_chan_block in tqdm.tqdm(itertools.product(depth_blocks, hw_blocks, out_chan_blocks)):
        total_activation_datums = 0
        total_weight_datums = 0
        max_block_l1_bytes = 0

        """
        Calculate the amount of L1 used for a single block
        Calculate the total amount of data read from memory
        """

        num_d_blocks = math.ceil(D_out / depth_block)
        num_h_blocks = math.ceil(H_out / hw_block)
        num_w_blocks = math.ceil(W_out / hw_block)
        num_oc_blocks = math.ceil(out_channels / out_chan_block)

        total_num_blocks = num_d_blocks * num_h_blocks * num_w_blocks * num_oc_blocks

        # Input patch to read from DRAM
        input_buffer_datums = (
            input_padded.shape[0]
            * input_padded.shape[1]
            * (depth_block + kD - 1)
            * (hw_block + kH - 1)
            * (hw_block + kW - 1)
        )
        # Across all blocks, this is the total amount of data read from DRAM
        total_activation_datums = input_buffer_datums * total_num_blocks

        num_patches = N * depth_block * hw_block * hw_block
        patch_size = C_in * kD * kH * kW

        # Buffer sizes for vol2col and weight
        vol2col_buffer_datums = num_patches * patch_size
        weight_buffer_datums = out_chan_block * patch_size

        # Assuming that the out_chan_blocks are distributed across cores, cores can reuse that weight buffer across
        # the other blocking dimensions
        # For each out_chan_block on a core, the core reads it once and reuses it across other block dimensions
        num_chan_blocks_per_core = math.ceil(num_oc_blocks / N_CORES)
        num_weight_reads = num_chan_blocks_per_core * N_CORES
        # Across all blocks, this is the total amount of data read from DRAM
        total_weight_datums = weight_buffer_datums * num_weight_reads

        # Buffer sizes for output
        out_col_buffer_datums = num_patches * out_chan_block
        out_vol_buffer_datums = out_col_buffer_datums

        # For a single block, this is the total amount of L1 used
        max_block_l1_bytes = BYTES_PER_DATUM * (
            input_buffer_datums
            + vol2col_buffer_datums
            + weight_buffer_datums
            + out_col_buffer_datums
            + out_vol_buffer_datums
        )

        results[depth_block, hw_block, out_chan_block] = {
            "total_activation_bytes": total_activation_datums * BYTES_PER_DATUM,
            "total_weight_bytes": total_weight_datums * BYTES_PER_DATUM,
            "max_block_l1_bytes": max_block_l1_bytes,
        }
    return results


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 5, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["test0", "variant0", "variant1", "variant2", "variant3", "variant4"],
)
@pytest.mark.parametrize("estimation_type", ["naive", "weight_reuse"])
@pytest.mark.parametrize("T_parallel_factor", [8])
def test_gather_vol2col_usage(
    input_shape, out_channels, kernel_size, stride, padding, padding_mode, T_parallel_factor, estimation_type
):
    """
    For a given input shape, calculate
    - for each possible combintation of blocking factors
        - the amount of L1 required to hold inputs/intermediates/output
        - the total amount of data read from memory

    Assuming:
    - kernel is streamed as needed
    - ignoring bias
    - input is streamed as needed


    TODO: Assume weight is cached?
    """
    torch.manual_seed(42)

    # Define input dimensions.
    N, C, D, H, W = input_shape
    D = math.ceil(D / T_parallel_factor)

    BYTES_PER_DATUM = 2

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    # Create a Conv3d module with chosen parameters.
    in_channels = C
    dilation = (1, 1, 1)
    conv3d_module = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
        padding_mode=padding_mode,
    )

    assert conv3d_module.stride == (1, 1, 1), "This example only supports stride=1"
    assert conv3d_module.dilation == (1, 1, 1), "This example only supports dilation=1"
    assert conv3d_module.groups == 1, "This example assumes groups=1"

    weight = conv3d_module.weight  # [out_channels, C_in, kD, kH, kW]
    bias = conv3d_module.bias
    pad_d, pad_h, pad_w = conv3d_module.padding
    kD, kH, kW = conv3d_module.kernel_size
    out_channels = conv3d_module.out_channels

    N, C_in, D_in, H_in, W_in = input_tensor.shape

    # ----------------------
    # 2) Compute Output Size
    # ----------------------
    # For stride=1, dilation=1, groups=1:
    # out_dim = in_dim + 2*pad - (kernel - 1)
    def _out_size(in_size, pad, k):
        return in_size + 2 * pad - (k - 1)

    D_out = _out_size(D_in, pad_d, kD)
    H_out = _out_size(H_in, pad_h, kH)
    W_out = _out_size(W_in, pad_w, kW)

    # -------------------------------------------
    # 3) Pad input along D, H, W (if needed)
    # -------------------------------------------
    if conv3d_module.padding_mode == "zeros":
        input_padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="constant", value=0)
    elif conv3d_module.padding_mode == "replicate":
        input_padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding_mode {conv3d_module.padding_mode}")

    # input_padded now has shape [N, C_in, D_in + 2*pad_d, H_in + 2*pad_h, W_in + 2*pad_w]
    D_pad, H_pad, W_pad = input_padded.shape[2:]

    # -----------------------------
    # 4) Allocate final output
    # -----------------------------
    output = torch.zeros((N, out_channels, D_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)

    # Blocking factors which divide output dimensions - lead to fewer options, worse overall in memory usage
    # depth_blocks = [i for i in range(1, D_out+1) if D_out % i == 0]
    # hw_blocks = [i for i in range(1, min(H_out, W_out)+1) if H_out % i == 0 and W_out % i == 0]
    # out_chan_blocks = [i for i in range(1, out_channels+1) if out_channels % i == 0]

    # ------------------------------------------------------------------
    # Try all possible blocking factors
    # ------------------------------------------------------------------
    depth_blocks = range(1, D_out + 1)
    hw_blocks = range(1, min(H_out, W_out) + 1)
    out_chan_blocks = range(1, out_channels + 1)

    if estimation_type == "naive":
        results = naive_estimation(
            N=N,
            C_in=C_in,
            D_out=D_out,
            H_out=H_out,
            W_out=W_out,
            out_channels=out_channels,
            kD=kD,
            kH=kH,
            kW=kW,
            input_padded=input_padded,
            weight=weight,
            depth_blocks=depth_blocks,
            hw_blocks=hw_blocks,
            out_chan_blocks=out_chan_blocks,
            BYTES_PER_DATUM=BYTES_PER_DATUM,
        )
    elif estimation_type == "weight_reuse":
        results = weight_reuse_estimation(
            N=N,
            C_in=C_in,
            D_out=D_out,
            H_out=H_out,
            W_out=W_out,
            out_channels=out_channels,
            kD=kD,
            kH=kH,
            kW=kW,
            input_padded=input_padded,
            weight=weight,
            depth_blocks=depth_blocks,
            hw_blocks=hw_blocks,
            out_chan_blocks=out_chan_blocks,
            BYTES_PER_DATUM=BYTES_PER_DATUM,
        )

    input_tensor_memory = input_padded.numel() * BYTES_PER_DATUM
    kernel_tensor_memory = weight.numel() * BYTES_PER_DATUM

    print("With perfect reuse:")
    print(f"\tTotal mem MB: {(input_tensor_memory + kernel_tensor_memory) / 1024**2:.2f} MB")
    print(f"\tActivation MB: {input_tensor_memory / 1024**2:.2f} MB")
    print(f"\tWeight MB: {kernel_tensor_memory / 1024**2:.2f} MB")
    print()
    # Process results
    L1_CAPACITY = 2**20
    oom_configs = {k: v for k, v in results.items() if v["max_block_l1_bytes"] > L1_CAPACITY}
    # print(f"OOM configs:")
    # for k, v in oom_configs.items():
    #     print(f'db: {k[0]}, hb: {k[1]}, ob: {k[2]}')
    #     print(v)

    non_oom_configs = {k: v for k, v in results.items() if v["max_block_l1_bytes"] <= L1_CAPACITY}
    mem_sorted = sorted(
        non_oom_configs.items(),
        key=lambda x: x[1]["total_activation_bytes"] + x[1]["total_weight_bytes"],
        reverse=False,
    )
    print(f"Memory sorted top 5:")
    for k, v in mem_sorted[:5]:
        print(f"db: {k[0]}, hb: {k[1]}, ob: {k[2]}")
        print(f'\tTotal mem MB: {(v["total_activation_bytes"] + v["total_weight_bytes"]) / 1024**2:.2f}')
        print(f'\tActivation MB: {v["total_activation_bytes"] / 1024**2:.2f}')
        print(f'\tWeight MB: {v["total_weight_bytes"] / 1024**2:.2f}')
        print(f'\tMax block L1 KB: {v["max_block_l1_bytes"] / 1024:.2f}')
        print()

    # Print ratio of best config to perfect reuse
    best_config = mem_sorted[0]
    print(f"Best config: {best_config[0]}")
    print(
        f'Ratio to perfect reuse: {(best_config[1]["total_activation_bytes"] + best_config[1]["total_weight_bytes"]) / (input_tensor_memory + kernel_tensor_memory):.2f}'
    )
