# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
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
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    return reduced_tensors


def tt_composite_sharded_all_reduce(
    input_tensor, mesh_device, cluster_axis, dim=3, num_links=2, reduce_scatter_mem_cfg=None
):
    input_mem_cfg = input_tensor.memory_config()
    reduce_scattered_tensor = ttnn.reduce_scatter(
        input_tensor,
        dim=dim,
        math_op=ttnn.ReduceType.Sum,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        memory_config=reduce_scatter_mem_cfg,
        topology=ttnn.Topology.Linear,
    )
    reduced_tensor = ttnn.all_gather(
        reduce_scattered_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        memory_config=input_mem_cfg,
        topology=ttnn.Topology.Linear,
    )
    return reduced_tensor


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


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, compute_kernel_config):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)

    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))  # TODO: Figure out why we need this
    tt_stats = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        dim=3,
        cluster_axis=1,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    tt_stats.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp, epsilon, gamma, mesh_device, ln_sharded_input_memcfg, ln_sharded_progcfg, ln_sharded_stats_memcfg
):
    inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=ln_sharded_progcfg)

    # All gather stats
    tt_stats = ttnn.all_gather(
        tt_stats,
        3,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh_device,
        memory_config=ln_sharded_stats_memcfg,
        topology=ttnn.Topology.Linear,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        epsilon=epsilon,
        weight=gamma,
        program_config=ln_sharded_progcfg,
        stats=tt_stats,
    )
    tt_stats.deallocate(True)

    return tt_out
