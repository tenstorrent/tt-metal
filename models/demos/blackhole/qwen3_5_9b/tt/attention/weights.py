import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc
from models.demos.blackhole.qwen3_5_9b.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class Qwen35AttentionWeights:
    wq: ttnn.Tensor  # q_proj
    wk: ttnn.Tensor  # k_proj
    wv: ttnn.Tensor  # v_proj
    wo: ttnn.Tensor  # o_proj
    w_q_norm: ttnn.Tensor  # Replicated across devices
    w_k_norm: ttnn.Tensor  # Replicated across devices


def load_attention_weights(mesh_device, state_dict, tensor_cache_path=None) -> Qwen35AttentionWeights:
    if tensor_cache_path is not None:
        os.makedirs(tensor_cache_path, exist_ok=True)

    # Cache key per weight. get_cache_file_name returns a proper "{dir}/{name}" STRING when a
    # cache dir is given, else None (caching disabled). Passing the raw (tensor_cache_path, name)
    # TUPLE to shard_w/as_tensor instead was a bug: with the default tensor_cache_path=None the
    # tuple (None, "wq") is truthy, so caching turned ON with a stringified "(None, 'wq')" file in
    # the cwd — and that mesh-sharded cache reloads CORRUPT in a fresh process (garbage per-device
    # shards → PCC≈0 attention output). The generating run looked fine, so it only bit on re-runs.
    return Qwen35AttentionWeights(
        wq=tpc.shard_w(
            state_dict["q_proj.weight"],
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wq"),
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
