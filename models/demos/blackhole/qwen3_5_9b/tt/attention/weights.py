import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc
from models.demos.blackhole.qwen3_5_9b.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class Qwen35AttentionWeights:
    wq: ttnn.Tensor  # q_proj
    wg: ttnn.Tensor  # gate_proj
    wk: ttnn.Tensor  # k_proj
    wv: ttnn.Tensor  # v_proj
    wo: ttnn.Tensor  # o_proj
    w_q_norm: ttnn.Tensor  # Replicated across devices
    w_k_norm: ttnn.Tensor  # Replicated across devices


def load_attention_weights(mesh_device, state_dict, args, tensor_cache_path=None) -> Qwen35AttentionWeights:
    if tensor_cache_path is not None:
        os.makedirs(tensor_cache_path, exist_ok=True)

    def split_q_and_gate(w):
        """
        HF checkpoint / state_dict ships query and gate projections fused into a single tensor.
        This function splits them into two separate weight tensors wq and wg, each with shape [hidden_size, num_heads * head_dim], that the TPAttention expects.
            w: [hidden_size, 2 * num_heads * head_dim]

        Returns:
            wq: [hidden_size, num_heads * head_dim]
            wg: [hidden_size, num_heads * head_dim]
        """

        NH, HD = args.n_heads, args.head_dim
        w_q_and_gate = w.reshape(NH, 2 * HD, -1)
        wq = w_q_and_gate[:, :HD, :].reshape(NH * HD, -1)
        wg = w_q_and_gate[:, HD:, :].reshape(NH * HD, -1)
        return wq, wg

    wq, wg = split_q_and_gate(state_dict["q_proj.weight"])

    return Qwen35AttentionWeights(
        wq=tpc.shard_w(
            wq,
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wq"),
            dtype=ttnn.bfloat8_b,
        ),
        wg=tpc.shard_w(
            wg,
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wg"),
            dtype=ttnn.bfloat8_b,
        ),
        wk=tpc.shard_w(
            state_dict["k_proj.weight"],
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wk"),
            dtype=ttnn.bfloat8_b,
        ),
        wv=tpc.shard_w(
            state_dict["v_proj.weight"],
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wv"),
            dtype=ttnn.bfloat8_b,
        ),
        # Row-parallel: shard input dim → reduce-scatter after
        wo=tpc.shard_w(
            state_dict["o_proj.weight"],
            mesh_device,
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wo"),
            dtype=ttnn.bfloat8_b,
        ),
        w_q_norm=tpc.replicate(state_dict["q_norm.weight"].to(torch.float32), mesh_device, None),
        w_k_norm=tpc.replicate(state_dict["k_norm.weight"].to(torch.float32), mesh_device, None),
    )
