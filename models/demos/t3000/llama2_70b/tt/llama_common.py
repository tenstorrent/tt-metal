# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
from loguru import logger
import re
from typing import Tuple
import numpy as np
import torch
from torch import nn
import ttnn
from models.utility_functions import tt2torch_tensor, torch2tt_tensor
from loguru import logger
from pathlib import Path
from models.demos.t3000.llama2_70b.reference.llama.llama.generation import (
    load_chunked_checkpoints,
    load_sharded_checkpoints,
)
import pytest
from models.demos.t3000.llama2_70b.tt.model_config import get_model_config

MAX_SEQ_LEN = 4096
MAX_SEQ_LEN_LLAMA3 = 8192
MAX_SEQ_LEN_LLAMA3_1 = 128 * 1024
BASE_URL = "layers"
UNIT_TEST_N_LAYER = 1
UNIT_TEST_LAYER_NUM = 0
UNIT_TEST_START_POS = 0
UNIT_TEST_GENERATION_LENGTH = 20
from ttnn import (
    TensorToMesh,
    MeshToTensor,
)


class ShardTensor2dMesh(TensorToMesh):
    def __init__(self, mesh_device, dims, cluster_shape):
        super().__init__(mesh_device)
        self.dims = dims
        self.cluster_shape = cluster_shape

    def map(self, tensor: torch.tensor):
        # Returns list of tensors to map to row-major ordering of chips in cluster
        tensors_grid_y = None
        if self.dims[1] == None:
            tensors_grid_y = [tensor.clone() for _ in range(self.cluster_shape[1])]
        else:
            tensors_grid_y = torch.chunk(tensor, self.cluster_shape[1], dim=self.dims[1])

        tensors_grid_all = None
        if self.dims[0] == None:
            tensors_grid_all = [t.clone() for t in tensors_grid_y for _ in range(self.cluster_shape[0])]
        else:
            tensors_grid_all = [
                tt for t in tensors_grid_y for tt in torch.chunk(t, self.cluster_shape[0], dim=self.dims[0])
            ]

        return list(tensors_grid_all)

    def config(self):
        return {
            "strategy": "shard",
            "shard_dim": f"{self.dims[0] if self.dims[0] else self.dims[1]}",
        }


class ConcatMesh2DToTensor(MeshToTensor):
    def __init__(self, mesh_device, dims, cluster_shape):
        self.dims = dims
        self.cluster_shape = cluster_shape
        self.mesh_device = mesh_device

    def compose(self, tensor: ttnn.Tensor) -> torch.Tensor:
        tt_shards = [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)]

        row_concat = []
        for cluster_row in range(self.cluster_shape[1]):
            start = cluster_row * self.cluster_shape[0]
            end = start + self.cluster_shape[0]
            row_concat.append(torch.cat(tt_shards[start:end], dim=self.dims[0]))
        all_concat = torch.cat(row_concat, dim=self.dims[1])
        return all_concat


def load_llama_state_dict(ckpt_dir, n_layers, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = "layers_" in str(checkpoints[0])
    if is_chunked:
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints(checkpoints, n_layers)

    return checkpoint


# string similarity score in percentage of two strings based on words in the same order
def string_similarity_score(ground_truths, predictions):
    scores = []
    for ground_truth, prediction in zip(ground_truths, predictions):
        ground_truth = ground_truth.split()
        prediction = prediction.split()
        if len(ground_truth) == 0:
            return 0
        score = 0
        for i in range(len(ground_truth)):
            if i < len(prediction) and ground_truth[i] == prediction[i]:
                score += 1
        scores.append(score / len(ground_truth))

    return scores


def should_skip_model_load():
    skip_model_load = bool(os.environ.get("LLAMA_SKIP_MODEL_LOAD", 0))
    if skip_model_load:
        logger.warning("LLAMA_SKIP_MODEL_LOAD is set. Skipping model load")
    return skip_model_load


def setup_llama_env(llama_version="llama3", batch=32, seq_len=1, n_devices=8, max_batch_size=32, max_context_len=4096):
    if os.getenv("CI") == "true":
        if llama_version == "llama3":
            ckpt_dir = "/mnt/MLPerf/tt_dnn-models/llama-3/llama-3-70b-repacked/"
            tokenizer_path = "/mnt/MLPerf/tt_dnn-models/llama-3/tokenizer.model"
            cache_path = Path("/mnt/MLPerf/tt_dnn-models/llama-3/llama-data-cache/weights-cache-3")
        elif llama_version == "llama3-tg":
            ckpt_dir = "/mnt/MLPerf/tt_dnn-models/llama-3/llama-3-70b-repacked/"
            tokenizer_path = "/mnt/MLPerf/tt_dnn-models/llama-3/tokenizer.model"
            cache_path = Path("/mnt/MLPerf/tt_dnn-models/llama-3/llama-data-cache/weights-cache-tg")
        elif llama_version == "llama3-405b":
            ckpt_dir = "/mnt/MLPerf/tt_dnn-models/llama-3-405b/llama-3-405b-repacked/"
            tokenizer_path = "/mnt/MLPerf/tt_dnn-models/llama-3-405b/tokenizer.model"
            cache_path = Path("/mnt/MLPerf/tt_dnn-models/llama-3-405b/llama-data-cache/weights-cache-3-405b")
        else:
            ckpt_dir = "/mnt/MLPerf/tt_dnn-models/llama-2/llama-2-70b-repacked/"
            tokenizer_path = "/mnt/MLPerf/tt_dnn-models/llama-2/tokenizer.model"
            cache_path = Path("/mnt/MLPerf/tt_dnn-models/llama-2/llama-data-cache/weights-cache-2")
    else:
        if llama_version == "llama3":
            ckpt_dir = os.getenv("LLAMA3_CKPT_DIR", "/proj_sw/llama3-data-repacked/llama-3-70b/")
            tokenizer_path = os.getenv("LLAMA3_TOKENIZER_PATH", "/proj_sw/llama3-data-repacked/tokenizer.model")
            cache_path = Path(os.getenv("LLAMA3_CACHE_PATH", "/proj_sw/llama-cache/llama-3-70b"))
        elif llama_version == "llama3-tg":
            ckpt_dir = os.getenv("LLAMA3_CKPT_DIR", "/proj_sw/user_dev/llama3-data-repacked/llama-3-70b/")
            tokenizer_path = os.getenv(
                "LLAMA3_TOKENIZER_PATH", "/proj_sw/user_dev/llama3-data-repacked/tokenizer.model"
            )
            cache_path = Path(os.getenv("LLAMA3_CACHE_PATH", "/proj_sw/user_dev/llama3-data-cache/weights-cache-2"))
        elif llama_version == "llama3-405b":
            ckpt_dir = os.getenv("LLAMA3_405B_CKPT_DIR", "/proj_sw/user_dev/llama3-405B-data-repacked/llama-3-405b/")
            tokenizer_path = os.getenv(
                "LLAMA3_405B_TOKENIZER_PATH", "/proj_sw/user_dev/llama3-405B-data-repacked/tokenizer.model"
            )
            cache_path = Path(os.getenv("LLAMA3_405B_CACHE_PATH", "/proj_sw/user_dev/llama3-405B-cache/llama-3-405b"))
        else:
            ckpt_dir = os.getenv("LLAMA2_CKPT_DIR", "/proj_sw/llama2-data-repacked/llama-2-70b/")
            tokenizer_path = os.getenv("LLAMA2_TOKENIZER_PATH", "/proj_sw/llama2-data-repacked/tokenizer.model")
            cache_path = Path(os.getenv("LLAMA2_CACHE_PATH", "/proj_sw/llama-cache/llama-2-70b"))

        assert os.path.exists(
            ckpt_dir
        ), f"Checkpoint directory {ckpt_dir} does not exist, please use export {llama_version.upper()}_CKPT_DIR=..."
        assert os.path.exists(
            tokenizer_path
        ), f"Tokenizer file {tokenizer_path} does not exist, please use export {llama_version.upper()}_TOKENIZER_PATH=..."
        assert os.path.exists(
            cache_path
        ), f"Cache directory {cache_path} does not exist, please use export {llama_version.upper()}_CACHE_PATH=..."

    logger.info(f"Checkpoint directory: {ckpt_dir}")
    logger.info(f"Tokenizer file: {tokenizer_path}")
    logger.info(f"Cache directory: {cache_path}")

    model_config = get_model_config(
        llama_version=llama_version,
        seq_len=seq_len,
        num_devices=n_devices,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    return model_config, ckpt_dir, tokenizer_path, cache_path


def check_mesh_device(t3k_mesh_device, model_config):
    assert t3k_mesh_device.get_num_devices() >= model_config["NUM_DEVICES"], (
        "Requires at least %d devices to run",
        model_config["NUM_DEVICES"],
    )

    compute_grid_size = t3k_mesh_device.get_device(0).compute_with_storage_grid_size()
    assert not (
        compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]
    ), ("Requires grid size of at least %d to run", model_config["MAX_GRID_SIZE"])


def extract_pcc_from_log(log):
    pattern = r"PCC: ([\d.]+)"
    extracted_pcc = re.search(pattern, log)
    extracted_pcc = float(extracted_pcc.group(1))
    return extracted_pcc


def get_weight_cache_path(base_cache_path, tensor_str, device_idx, num_devices, cache_id=None):
    return base_cache_path / f"{tensor_str}{'' if cache_id is None else cache_id}_{device_idx}_{num_devices}.bin"


def get_weight_cache_path_ttnn(
    base_cache_path, tensor_str, device_idx, num_devices, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
):
    return base_cache_path / f"{tensor_str}_{device_idx}_{num_devices}_dtype_{dtype.name}_layout_{layout.name}.bin"


def get_weight_cache_path_galaxy(base_cache_path, tensor_str, device_idx, num_devices, x, y):
    return base_cache_path / f"{tensor_str}_{device_idx}_{num_devices}_{x}_{y}.bin"


def rms_decomp(x, norm_weight, eps):
    squared = ttnn.pow(x, 2)
    # mean_squared = tt_lib.tensor.mean(squared, )
    sum_squared = ttnn.sum(squared, 3)
    # Tensor is 1,1,32,1+31 now
    mean_squared = ttnn.multiply(sum_squared, (1 / x.shape[-1]))
    mean_squared_eps = ttnn.add(mean_squared, eps)
    rms = ttnn.pow(mean_squared_eps, 0.5)
    rms_recip = ttnn.reciprocal(rms)
    normed_x = ttnn.multiply(x, rms_recip)
    norm_out = ttnn.mul(normed_x, norm_weight)
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


def generate_rot_emb(dhead, end, theta: float = 10000.0, use_scaled: bool = False):
    cos, sin = precompute_freqs(dhead, end, theta, use_scaled)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    return rot_mat


def apply_scaling(freqs: torch.Tensor):
    # Llama-3.1 specific scaling
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
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
    if use_scaled:
        freqs = apply_scaling(freqs)
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


def get_rotation_mat_prefill(rot_mat, start_pos, seqlen, batch):
    position_ids = torch.ones(batch, seqlen, dtype=torch.long) * torch.arange(start_pos, start_pos + seqlen).unsqueeze(
        0
    )
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


def get_rotation_mat(rot_mat, start_pos, seqlen, batch):
    # position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    position_ids = start_pos.view(seqlen, batch)
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


#  Add-Multiply method of rotary embeddings for prefill
def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = cal_pcc >= pcc
    if not passing:
        output_str += ", PCC check failed"
    return passing, output_str


def get_atol_rtol_pcc(golden, calculated):
    if golden.is_complex() and calculated.is_complex():
        golden = torch.view_as_real(golden.clone())
        calculated = torch.view_as_real(calculated.clone())

    if not (golden.is_floating_point() or calculated.is_floating_point()):
        golden = golden.to(torch.float)
        calculated = calculated.to(torch.float)

    # Calculate atol and rtol
    # cal_atol = torch.max(torch.abs(golden - calculated)).item()
    # cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()
    cal_atol = 0
    cal_rtol = 0

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logger.warning("Both tensors are 'nan'")
            return 1.0

        # One tensor is all nan, the other is not
        if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logger.error("One tensor is all nan, the other is not.")
            return 0.0

        # One tensor is all zero, the other is not
        if torch.any(golden.bool()) != torch.any(calculated.bool()):
            logger.warning("One tensor is all zero")
            return 0.0

        # if torch.any(torch.isinf(golden)) or torch.any(torch.isinf(calculated)):
        #    raise RuntimeError(f"Tensor overflow to infinity: \n{golden}\n{calculated}")

        # if torch.any(torch.isneginf(golden)) or torch.any(torch.isneginf(calculated)):
        #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{golden}\n{calculated}")

        else:
            # For now, mask all infs and nans so that we check the rest... TODO
            golden = golden.clone()
            golden[
                torch.logical_or(
                    torch.isnan(golden),
                    torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
                )
            ] = 0
            calculated = calculated.clone()
            calculated[
                torch.logical_or(
                    torch.isnan(calculated),
                    torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
                )
            ] = 0

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
                calculated = calculated.type(torch.float32)

            # Single element case
            if golden.numel() == 1:
                return float(torch.equal(golden, calculated))

            # one tensor is constant
            if torch.max(golden) == torch.min(golden) or torch.max(calculated) == torch.min(calculated):
                return float(torch.equal(golden, calculated))

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
            )
            # Remove correlation coefficient with self (typically always 1.0)
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def check_kv_cache(pt_cache_all, tt_cache_all, generation_start_pos, generation_length, seq_len, is_prefill, pcc):
    test_passed = True
    for cache_pt, cache_tt in zip(pt_cache_all, tt_cache_all):
        cache_length_to_check = generation_start_pos + generation_length
        if is_prefill:
            cache_pt = cache_pt[:, :, :seq_len, :]
            cache_tt = cache_tt[:, :, :seq_len, :]
        else:
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
        does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
        logger.info(f"Output: {output_pcc}")

        if does_pass:
            logger.info(f"KV Cache Passed!")
        else:
            logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
            test_passed = False
    return test_passed


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )
