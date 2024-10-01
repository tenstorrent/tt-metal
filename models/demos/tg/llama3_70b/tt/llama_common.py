# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import os
from pathlib import Path
from loguru import logger

from models.demos.tg.llama3_70b.tt.model_config import get_model_config


class PytorchLlamaModel(torch.nn.Module):
    def __init__(self, hf_reference_model):
        super().__init__()
        self.model = hf_reference_model

        # Disable dropout
        self.model.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def forward(self, x, start_pos):
        """
        x: (batch, seq)
        start_pos: int

        return: (batch, seq, hidden_dim)
        """
        with torch.no_grad():
            return self.model(x, start_pos)


def tt_all_reduce(input_tensor, mesh_device, cluster_axis, dim=0, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    gathered_tensor = ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
    )
    reduced_tensors = ttnn.experimental.fast_reduce_nc(
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None
    )

    return reduced_tensors


def tt_all_gather(input_tensor, mesh_device, cluster_axis, dim, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    return ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
    )


def tt_sharded_all_reduce(input_tensor, mesh_device, cluster_axis, dim=0, num_links=2, memory_config=None):
    gathered_tensor = ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        memory_config=memory_config,
        topology=ttnn.Topology.Linear,
    )
    # Fast_reduce_nc does not support sharded memory configuration, convert to interleaved
    gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)
    reduced_tensors = ttnn.experimental.fast_reduce_nc(
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None
    )
    return reduced_tensors


def tt_sharded_all_gather(input_tensor, mesh_device, cluster_axis, dim, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration

    return ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        memory_config=memory_config,
        topology=ttnn.Topology.Linear,
    )


def upper_pad_sequence_length(length, padding_size):
    if length % padding_size == 0:
        return length  # No padding needed
    return ((length + padding_size - 1) // padding_size) * padding_size


def setup_llama_env(llama_version="llama3", max_batch_size=32, max_context_len=4096):
    if os.getenv("CI") == "true":
        if llama_version == "llama3-tg":
            ckpt_dir = "/mnt/MLPerf/tt_dnn-models/llama-3/llama-3-70b-repacked/"
            tokenizer_path = "/mnt/MLPerf/tt_dnn-models/llama-3/tokenizer.model"
            cache_path = Path("/mnt/MLPerf/tt_dnn-models/llama-3/llama-data-cache/weights-cache-tg")
        else:
            raise ValueError(f"Unknown llama version: {llama_version}")
    else:
        if llama_version == "llama3-tg":
            ckpt_dir = os.getenv("LLAMA3_CKPT_DIR", "/proj_sw/user_dev/llama3-data-repacked/llama-3-70b/")
            tokenizer_path = os.getenv(
                "LLAMA3_TOKENIZER_PATH", "/proj_sw/user_dev/llama3-data-repacked/tokenizer.model"
            )
            cache_path = Path(os.getenv("LLAMA3_CACHE_PATH", "/proj_sw/user_dev/llama3-data-cache/weights-cache-2"))
        else:
            raise ValueError(f"Unknown llama version: {llama_version}")

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
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    return model_config, ckpt_dir, tokenizer_path, cache_path
