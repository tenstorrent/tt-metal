import os
import math
from pathlib import Path
from models.demos.llama3.tt.llama_common import get_out_subblock_w
from typing import Tuple
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


import torch


def compute_metrics(reference_output, test_output):
    # Compute PCC
    pcc = torch.corrcoef(torch.stack([reference_output.flatten(), test_output.flatten()]))[0, 1].item()

    # Compute MSE
    mse = torch.nn.functional.mse_loss(test_output, reference_output).item()

    # Compute MAE
    mae = torch.nn.functional.l1_loss(test_output, reference_output).item()

    return pcc, mse, mae


def matmul_config(
    m: int,
    k: int,
    n: int,
    grid_size: Tuple[int, int],
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    TILE_SIZE = 32
    per_core_M = math.ceil(m / (TILE_SIZE * grid_size[1]))
    per_core_N = math.ceil(n / (TILE_SIZE * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)

    if in0_block_w is None:
        in0_block_w = min(4, max(1, k // (TILE_SIZE * grid_size[0])))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )
