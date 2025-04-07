import os
import math
import torch
from pathlib import Path
from models.tt_transformers.tt.common import get_out_subblock_w
from typing import Tuple, Optional
import ttnn


def get_mochi_dir():
    mochi_dir = os.environ.get("MOCHI_DIR")
    if not mochi_dir:
        raise ValueError("MOCHI_DIR environment variable must be set")
    return mochi_dir


def get_cache_path(device_name):
    mochi_dir = get_mochi_dir()
    cache_path = Path(mochi_dir) / device_name
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def compute_metrics(reference_output, test_output):
    # Compute PCC
    pcc = torch.corrcoef(torch.stack([reference_output.flatten(), test_output.flatten()]))[0, 1].item()

    # Compute MSE
    mse = torch.nn.functional.mse_loss(test_output, reference_output).item()

    # Compute MAE
    mae = torch.nn.functional.l1_loss(test_output, reference_output).item()

    return pcc, mse, mae


def stack_cos_sin(cos, sin):
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    return cos, sin


SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]


def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, out_sharded=False, fp32_dest_acc_en=False):
    for subblock_hw in SUBBLOCK_HW_CHOICES:
        out_subblock_h = subblock_hw[0]
        out_subblock_w = subblock_hw[1]

        if fp32_dest_acc_en:
            if (out_subblock_h * out_subblock_w) > 4:
                continue

        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
            return (out_subblock_h, out_subblock_w)

    return (1, 1)


def matmul_config(
    m: int,
    k: int,
    n: int,
    grid_size: Tuple[int, int],
    in0_block_w: int = None,
    out_block_h: int = None,
    out_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
    fp32_dest_acc_en: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    TILE_SIZE = 32
    per_core_M = math.ceil(m / (TILE_SIZE * grid_size[1]))
    per_core_N = math.ceil(n / (TILE_SIZE * grid_size[0]))

    MAX_OUT_BLOCK_SIZE = TILE_SIZE * 96  # 3072
    num_out_blocks_h = math.ceil(m / MAX_OUT_BLOCK_SIZE)
    num_out_blocks_w = math.ceil(n / MAX_OUT_BLOCK_SIZE)

    # Special case find divisible num_out_blocks_h
    if per_core_M % num_out_blocks_h != 0:
        num_out_blocks_h = min(i for i in range(num_out_blocks_h, m // TILE_SIZE) if per_core_M % i == 0)

    assert (
        per_core_M % num_out_blocks_h == 0
    ), f"num_out_blocks_h {num_out_blocks_h} must evenly divide per_core_M {per_core_M}"
    assert (
        per_core_N % num_out_blocks_w == 0
    ), f"num_out_blocks_w {num_out_blocks_w} must evenly divide per_core_N {per_core_N}"
    out_block_h = per_core_M / num_out_blocks_h
    out_block_w = per_core_N / num_out_blocks_w
    out_subblock_h, out_subblock_w = get_subblock_sizes(
        out_block_h, out_block_w, fp32_dest_acc_en=fp32_dest_acc_en
    )  # want these to be 1 and 6

    if in0_block_w is None:
        # in0_block_w = min(4, max(1, k // (TILE_SIZE * grid_size[0])))
        in0_block_w = 6

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M // num_out_blocks_h,
        out_block_w=per_core_N // num_out_blocks_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


def ff_matmul_config(
    m,
    k,
    n,
    in0_block_w,
    num_out_blocks_h,
    num_out_blocks_w,
    grid_size,
    fp32_dest_acc_en=False,
):
    TILE_SIZE = 32
    per_core_M = math.ceil(m / (TILE_SIZE * grid_size[1]))
    per_core_N = math.ceil(n / (TILE_SIZE * grid_size[0]))
    k_tiles = math.ceil(k / TILE_SIZE)

    assert (
        per_core_M % num_out_blocks_h == 0
    ), f"num_out_blocks_h {num_out_blocks_h} must evenly divide per_core_M {per_core_M}"
    assert (
        per_core_N % num_out_blocks_w == 0
    ), f"num_out_blocks_w {num_out_blocks_w} must evenly divide per_core_N {per_core_N}"
    out_block_h = per_core_M // num_out_blocks_h
    out_block_w = per_core_N // num_out_blocks_w
    out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, False, fp32_dest_acc_en)

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def matmul_2d_config(
    m,
    k,
    n,
    grid_size,
):
    TILE_SIZE = 32
    per_core_M = math.ceil(m / (TILE_SIZE * grid_size[1]))
    per_core_N = math.ceil(n / (TILE_SIZE * grid_size[0]))
    k_tiles = math.ceil(k / TILE_SIZE)

    # Compute in0_block_w as largest divisor of k_tiles that's <= 12
    # 12 is first guess at largest in0_block_w
    in0_block_w = 1
    for i in range(min(12, k_tiles), 0, -1):
        if k_tiles % i == 0:
            in0_block_w = i
            break

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)
    # TODO: sweep below values for optimal config
    out_block_h = out_subblock_h
    out_block_w = out_subblock_w

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def to_tt_tensor(tensor, mesh_device, dtype=ttnn.bfloat16, shard_dim=None):
    """Convert torch tensor to TT tensor."""
    if shard_dim is None:
        return ttnn.from_torch(
            tensor,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        return ttnn.as_tensor(
            tensor,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )


def to_torch_tensor(tensor, mesh_device, dtype=ttnn.bfloat16, dim=-1):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim), dtype=dtype)


def unsqueeze_to_4d(x):
    while x.ndim < 4:
        x = x.unsqueeze(0)
    return x


def as_sharded_tensor(tensor, mesh_device, dim, cache_file_name=None):
    return ttnn.as_tensor(
        tensor,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file_name,
    )


def as_replicated_tensor(tensor, mesh_device, cache_file_name=None):
    return ttnn.as_tensor(
        tensor,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file_name,
    )


def col_parallel_linear(name, bias, weight_cache_path, state_dict, state_dict_prefix, mesh_device):
    w = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
    b = state_dict[f"{state_dict_prefix}.{name}.bias"] if bias else None
    w = as_sharded_tensor(
        w,
        mesh_device,
        dim=-1,
        cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
    )
    if b is not None:
        b = as_sharded_tensor(
            b.reshape(1, -1),
            mesh_device,
            dim=-1,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
        )
    return w, b


def load_linear(name, bias, weight_cache_path, state_dict, state_dict_prefix, mesh_device):
    w = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
    b = state_dict[f"{state_dict_prefix}.{name}.bias"] if bias else None
    w = as_sharded_tensor(
        w,
        mesh_device,
        dim=-1,
        cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
    )
    if b is not None:
        b = as_replicated_tensor(
            b.reshape(1, -1),
            mesh_device,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
        )
    return w, b


def create_linear_layer(
    name: str,
    weight_cache_path: Path,
    state_dict: dict,
    state_dict_prefix: str,
    mesh_device,
    bias: bool = True,
) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
    """Create a linear layer with both weights and biases.

    Args:
        name: Name of the layer (e.g. 'mod_x', 'mod_y')
        in_features: Input dimension
        out_features: Output dimension
        weight_cache_path: Path to cache weights
        state_dict: State dict containing weights and biases
        state_dict_prefix: Prefix for state dict keys
        mesh_device: TensorTorch mesh device
        bias: Whether to include bias tensor

    Returns:
        Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]: Weight and bias tensors
    """
    # Get weight and transpose it
    weight_key = f"{state_dict_prefix}.{name}.weight"
    weight = torch.transpose(state_dict[weight_key], -2, -1)

    # Get bias if it exists
    bias_key = f"{state_dict_prefix}.{name}.bias"
    bias_pt = state_dict.get(bias_key)  # Returns None if key doesn't exist

    # Check that bias exists if bias=True is specified
    if bias and bias_pt is None:
        raise ValueError(f"Bias was specified but not found in state dict for {name} layer")
    # Create weight tensor
    weight_tensor = as_replicated_tensor(
        weight,
        mesh_device,
        cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
    )

    # Create bias tensor if it exists
    bias_tensor = None
    if bias:
        bias_tensor = as_replicated_tensor(
            bias_pt.reshape(1, -1),
            mesh_device,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
        )

    return weight_tensor, bias_tensor


def get_padded_vision_seq_len(N, num_devices):
    TILE_SIZE = 32
    divisor = TILE_SIZE * num_devices

    # Calculate padding needed to make seq_len divisible by both tile size and num_devices
    padded_seq_len = math.ceil(N / divisor) * divisor
    return padded_seq_len


def pad_vision_seq_parallel(tensor, num_devices):
    """
    Sequence parallelism shards the vision tensor in dim2.
    dim2 must be divisible by tile size and num_devices.
    """
    seq_len = tensor.shape[2]
    padded_seq_len = get_padded_vision_seq_len(seq_len, num_devices)
    pad_len = padded_seq_len - seq_len

    # Pad the sequence length dimension (dim2) on the right
    if pad_len > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))

    return tensor
