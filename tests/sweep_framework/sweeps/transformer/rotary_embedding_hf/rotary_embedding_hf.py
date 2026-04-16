# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import is_blackhole, nearest_32, torch_random
from ttnn.types import BlackholeComputeKernelConfig

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


def _attention_rotary_embedding_hf_compute_kernel_config():
    """Match ``ModelArgs.compute_kernel_config_hifi4`` / ``Attention._hf_rope_new_{decode,prefill}``."""
    cls = BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# Shared dtype/layout/mem columns (decode path ignores sweep ``input_memory_config`` for Q/K — uses
# explicit height-sharded L1 like ``test_hf_rope_decode_reassembly_parity._height_sharded_decode_heads_mem_cfg``;
# ``ModelArgs`` WH uses layout-only ``L1_HEIGHT_SHARDED_MEMORY_CONFIG``, which is invalid for ``from_torch``.)
_COMMON = {
    "input_dtype": [ttnn.bfloat16],
    "input_layout": [ttnn.TILE_LAYOUT],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    # Retained for vector serialization; ``Attention`` does not pass ``memory_config`` to ``rotary_embedding_hf``.
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}

# Parameters provided to the test vector generator are defined here.
#
# Prefill and decode must live in **separate** suites: a Cartesian product of two non-None dict lists
# would always take the prefill branch (``if prefill_spec:``), so decode shapes would never run.
#
# **Llama-3.1-8B-Instruct decode capture** (e.g. ``TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR=tmp/hf_rope_cap/run3``):
# log parity via ``TT_HF_ROPE_SWEEP_PARITY_LOG=1 pytest … test_hf_rope_attention_capture_replay_vs_captured_cos_sin_golden``:
#   Q ``[1,1,32,128]``, K ``[1,1,8,128]`` (logical heads; device padded head tile matches NLP layout),
#   cos/sin logical ``[1,1,1,128]``, prefill hint ``[1,32,256,128]`` / ``cache_size=256`` from manifest ``max_seq_len``.
parameters = {
    # Minimal vectors that match a typical ``run3``-style HF decode capture (see module docstring above).
    "hf_rope_capture_run3_prefill": {
        "prefill_spec": [
            {"input_shape": [1, 32, 256, 128], "cache_size": 256, "is_decode": False},
        ],
        "decode_spec": [None],
        **_COMMON,
    },
    "hf_rope_capture_run3_decode": {
        "prefill_spec": [None],
        "decode_spec": [
            {"input_shape": [1, 1, 32, 128], "is_decode": True},
            {"input_shape": [1, 1, 8, 128], "is_decode": True},
        ],
        **_COMMON,
    },
    "nightly_prefill": {
        "prefill_spec": [
            # [1, num_heads, seq_len, head_dim]
            {"input_shape": [1, 8, 32, 64], "cache_size": 128, "is_decode": False},
            {"input_shape": [1, 16, 64, 64], "cache_size": 256, "is_decode": False},
            {"input_shape": [1, 32, 128, 128], "cache_size": 512, "is_decode": False},
            {"input_shape": [1, 12, 256, 64], "cache_size": 512, "is_decode": False},
            {"input_shape": [1, 32, 256, 128], "cache_size": 256, "is_decode": False},
        ],
        "decode_spec": [None],
        **_COMMON,
    },
    "nightly_decode": {
        "prefill_spec": [None],
        "decode_spec": [
            # [1, batch, num_heads, head_dim] — first two match run3 capture log (Q then GQA K).
            {"input_shape": [1, 1, 32, 128], "is_decode": True},
            {"input_shape": [1, 1, 8, 128], "is_decode": True},
            {"input_shape": [1, 8, 8, 64], "is_decode": True},
            {"input_shape": [1, 16, 16, 64], "is_decode": True},
            {"input_shape": [1, 32, 32, 128], "is_decode": True},
            {"input_shape": [1, 32, 8, 128], "is_decode": True},
        ],
        **_COMMON,
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b requires TILE_LAYOUT!"

    ps = test_vector.get("prefill_spec")
    ds = test_vector.get("decode_spec")
    if ps and ds:
        return (
            True,
            "invalid vector: set only one of prefill_spec or decode_spec (use nightly_prefill / nightly_decode)",
        )

    if ps:
        spec = ps
        if spec["input_shape"][-1] % 64 != 0:
            return True, f"Input head_dim ({spec['input_shape'][-1]}) must be divisible by 64 for tiling"
    elif ds:
        spec = ds
        if spec["input_shape"][-1] % 64 != 0:
            return True, f"Input head_dim ({spec['input_shape'][-1]}) must be divisible by 64 for tiling"
    else:
        return True, "vector must set exactly one of prefill_spec or decode_spec"

    return False, None


def rotate_half(x):
    """Rotates half the hidden dims of the input (HF style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_hf(x, cos, sin):
    """Golden function for HF-style rotary embedding."""
    return (x * cos) + (rotate_half(x) * sin)


def _decode_qk_heads_mem_config(device, batch: int, num_heads: int, head_dim: int) -> ttnn.MemoryConfig:
    """Height-sharded L1 mem for decode Q/K.

    ``ModelArgs`` uses ``L1_HEIGHT_SHARDED_MEMORY_CONFIG`` on Wormhole for op-allocated tensors; host
    ``ttnn.from_torch`` requires a non-None ``shard_spec`` (see ``test_hf_rope_decode_reassembly_parity``).
    """
    padded_heads = nearest_32(num_heads)
    if is_blackhole():
        return ttnn.create_sharded_memory_config(
            shape=(padded_heads, head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(batch, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_hf_cos_sin_sharded(device, batch: int, head_dim: int, cos_torch, sin_torch, *, dtype):
    """Match ``HfRotarySetupNew.get_rot_mats`` sharding: HEIGHT (TILE_SIZE, head_dim) on a batch core grid."""
    core_grid = device.compute_with_storage_grid_size()
    num_cores = min(batch, core_grid.x * core_grid.y)
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # TILE row padding on the singleton head row (same as transpose path in ``HfRotarySetupNew``).
    pad_h = ttnn.TILE_SIZE - cos_torch.shape[2]
    if pad_h > 0:
        z = torch.zeros(1, batch, pad_h, head_dim, dtype=cos_torch.dtype, device=cos_torch.device)
        cos_torch = torch.cat([cos_torch, z], dim=2)
        sin_torch = torch.cat([sin_torch, z], dim=2)
    cos_interleaved = ttnn.from_torch(
        cos_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_interleaved = ttnn.from_torch(
        sin_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if batch % ttnn.TILE_SIZE != 0:
        cos_interleaved = cos_interleaved[:, :batch, :, :]
        sin_interleaved = sin_interleaved[:, :batch, :, :]
    cos_tensor = ttnn.interleaved_to_sharded(cos_interleaved, mem_config)
    sin_tensor = ttnn.interleaved_to_sharded(sin_interleaved, mem_config)
    return cos_tensor, sin_tensor


def run(
    prefill_spec,
    decode_spec,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,  # noqa: ARG001 — kept for sweep vector schema; Attention omits output mem on the op.
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    # Determine which mode we're testing (suites pass None for the unused spec — see ``parameters``).
    if prefill_spec:
        spec = prefill_spec
        input_shape = spec["input_shape"]
        cache_size = spec["cache_size"]
        is_decode = False

        # Prefill: cos/sin shape is [1, 1, cache_size, head_dim]
        cos_sin_shape = [1, 1, cache_size, input_shape[-1]]
    elif decode_spec:
        spec = decode_spec
        input_shape = spec["input_shape"]
        is_decode = True

        # Decode: cos/sin shape is [1, batch, 1, head_dim]
        batch_size = input_shape[1]
        cos_sin_shape = [1, batch_size, 1, input_shape[-1]]
    else:
        raise RuntimeError("rotary_embedding_hf sweep vector needs prefill_spec or decode_spec")

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)
    torch_cos_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_dtype
    )(cos_sin_shape)
    torch_sin_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_dtype
    )(cos_sin_shape)

    if is_decode:
        # In decode mode, input is [1, batch, num_heads, head_dim]
        # cos/sin are [1, batch, 1, head_dim]
        # For golden, we need to broadcast cos/sin properly
        cos_expanded = torch_cos_cache_tensor.expand(-1, -1, input_shape[2], -1)
        sin_expanded = torch_sin_cache_tensor.expand(-1, -1, input_shape[2], -1)
        torch_output_tensor = apply_rotary_pos_emb_hf(torch_input_tensor, cos_expanded, sin_expanded)
    else:
        # In prefill mode, input is [1, num_heads, seq_len, head_dim]
        # cos/sin are [1, 1, cache_size, head_dim]
        # Extract the needed sequence length
        seq_len = input_shape[2]
        cos_sliced = torch_cos_cache_tensor[:, :, :seq_len, :]
        sin_sliced = torch_sin_cache_tensor[:, :, :seq_len, :]
        torch_output_tensor = apply_rotary_pos_emb_hf(torch_input_tensor, cos_sliced, sin_sliced)

    # Decode: match ``test_attention`` / ``Attention.forward_decode`` — Q/K from
    # ``nlp_create_qkv_heads_decode`` use ``get_attn_create_head_output_mem_config``; cos/sin from
    # ``HfRotarySetupNew.get_rot_mats`` (DRAM interleaved → HEIGHT shard on batch grid).
    if is_decode:
        batch = input_shape[1]
        num_heads = input_shape[2]
        head_dim = input_shape[3]
        padded_heads = nearest_32(num_heads)
        qk_mem_config = _decode_qk_heads_mem_config(device, batch, num_heads, head_dim)

        torch_input_for_device = torch_input_tensor
        if padded_heads != num_heads:
            pad_h = padded_heads - num_heads
            z = torch.zeros(
                1,
                batch,
                pad_h,
                head_dim,
                dtype=torch_input_tensor.dtype,
                device=torch_input_tensor.device,
            )
            torch_input_for_device = torch.cat([torch_input_tensor, z], dim=2)

        input_tensor = ttnn.from_torch(
            torch_input_for_device,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=qk_mem_config,
        )
        cos_cache_tensor, sin_cache_tensor = _decode_hf_cos_sin_sharded(
            device,
            batch,
            head_dim,
            torch_cos_cache_tensor,
            torch_sin_cache_tensor,
            dtype=input_dtype,
        )
    else:
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_memory_config,
        )
        cos_cache_tensor = ttnn.from_torch(
            torch_cos_cache_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_memory_config,
        )
        sin_cache_tensor = ttnn.from_torch(
            torch_sin_cache_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_memory_config,
        )

    rope_cfg = _attention_rotary_embedding_hf_compute_kernel_config()
    start_time = start_measuring_time()
    # Match ``Attention._hf_rope_new_decode`` / ``_hf_rope_new_prefill``: HiFi4 compute config, no ``memory_config``.
    output_tensor = ttnn.experimental.rotary_embedding_hf(
        input_tensor,
        cos_cache_tensor,
        sin_cache_tensor,
        is_decode=is_decode,
        compute_kernel_config=rope_cfg,
    )
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)
    if is_decode and nearest_32(input_shape[2]) != input_shape[2]:
        output_tensor = output_tensor[:, :, : input_shape[2], :]

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
