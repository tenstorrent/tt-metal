# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger
import re
from typing import Tuple
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import tt2torch_tensor, torch2tt_tensor

MAX_SEQ_LEN = 4096
BASE_URL = "layers"
UNIT_TEST_N_LAYER = 1
UNIT_TEST_LAYER_NUM = 0
UNIT_TEST_START_POS = 1  # 0 is low for test_decoder: 0.9986
UNIT_TEST_GENERATION_LENGTH = 20


def get_llama_path(devices, model_config, n_devices, emulated):
    ckpt_dir = model_config["DEFAULT_CKPT_DIR"]
    tokenizer_path = model_config["DEFAULT_TOKENIZER_PATH"]
    cache_path = model_config["DEFAULT_CACHE_PATH"]

    assert os.path.exists(
        ckpt_dir
    ), f"Checkpoint directory {ckpt_dir} does not exist, please use export LLAMA_CKPT_DIR=..."
    assert os.path.exists(
        tokenizer_path
    ), f"Tokenizer file {tokenizer_path} does not exist, please use export LLAMA_TOKENIZER_PATH=..."
    assert os.path.exists(
        cache_path
    ), f"Cache directory {cache_path} does not exist, please use export LLAMA_CACHE_PATH=..."

    logger.info(f"Checkpoint directory: {ckpt_dir}")
    logger.info(f"Tokenizer file: {tokenizer_path}")
    logger.info(f"Cache directory: {cache_path}")

    if emulated:
        logger.info(f"Running emulated, replicating on {n_devices} devices")
        devices = [devices[0] for _ in range(n_devices)]  # Emulate fracturing on N chips
    else:
        logger.info(f"Running on {n_devices} devices on T3000 chips")

    return devices, ckpt_dir, tokenizer_path, cache_path


def extract_pcc_from_log(log):
    pattern = r"PCC: ([\d.]+)"
    extracted_pcc = re.search(pattern, log)
    extracted_pcc = float(extracted_pcc.group(1))
    return extracted_pcc


def get_weight_cache_path(base_cache_path, tensor_str, device_idx, num_devices, cache_id=None):
    return base_cache_path / f"{tensor_str}{'' if cache_id is None else cache_id}_{device_idx}_{num_devices}.bin"


def get_weight_cache_path_galaxy(base_cache_path, tensor_str, device_idx, num_devices, x, y):
    return base_cache_path / f"{tensor_str}_{device_idx}_{num_devices}_{x}_{y}.bin"


def rms_decomp(x, norm_weight, eps):
    squared = tt_lib.tensor.pow(x, 2)
    # mean_squared = tt_lib.tensor.mean(squared, )
    sum_squared = tt_lib.tensor.reduce(squared, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.W, scaler=1.0)
    # Tensor is 1,1,32,1+31 now
    mean_squared = tt_lib.tensor.div_unary(sum_squared, x.shape[-1])
    mean_squared_eps = tt_lib.tensor.add_unary(mean_squared, eps)
    rms = tt_lib.tensor.pow(mean_squared_eps, 0.5)
    rms_recip = tt_lib.tensor.recip(rms)
    normed_x = tt_lib.tensor.bcast(x, rms_recip, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W)
    norm_out = tt_lib.tensor.mul(normed_x, norm_weight)
    return norm_out


def tt_all_reduce(tensors):
    """
    reduction on a list of tensors
    """
    if len(tensors) == 1:
        return tensors[0]

    assert [tensor.shape for tensor in tensors] == [tensors[0].shape for _ in range(len(tensors))]
    dev = tensors[0].device()
    tensors_torch = [tt2torch_tensor(tensor) for tensor in tensors]
    base_tensor_torch = tensors_torch[0]

    for tensor_torch in tensors_torch[1:]:
        base_tensor_torch += tensor_torch
    # Emulate replication on all chips
    res = [torch2tt_tensor(base_tensor_torch.clone(), dev) for _ in range(len(tensors))]
    return res


def tt_all_gather_torch(tensors, dim=-1):
    tensors_torch = [tt2torch_tensor(tensor) for tensor in tensors]
    all_gathered_output_torch = torch.cat(tensors_torch, dim=dim)
    # Emulate replication on all chips
    dev = tensors[0].device()
    res = [torch2tt_tensor(all_gathered_output_torch.clone(), dev) for _ in range(len(tensors))]

    return res


def generate_rot_emb(dhead, end):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    return rot_mat


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """
    Transform cos/sin frequencies to a rotation matrix.
    """
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
    return rot_emb_matrix


def gather_rotary_emb(rot_emb_matrix, position_ids):
    """
    Gather the rotary embeddings for a given position_ids
    """
    batch_size, seqlen = position_ids.shape
    emb_size, _, dhead = rot_emb_matrix.shape
    position_ids = position_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, dhead, dhead)
    rot_emb = rot_emb_matrix.gather(0, position_ids).view(batch_size, seqlen, dhead, dhead)
    return rot_emb


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rotation_mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given cosine and sine frequency tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        rotation_mat (torch.Tensor): Precomputed rotation matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    xq_out = xq @ rotation_mat
    xk_out = xk @ rotation_mat
    return xq_out, xk_out
