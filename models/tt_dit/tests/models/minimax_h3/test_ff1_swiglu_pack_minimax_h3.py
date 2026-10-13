# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One transformer block at the FastH3 Turbo fl2va 10 s shape: ff1's SwiGLU epilogue vs SwiGLU on the pack thread.

TT_AGMM_SWIGLU_PACK=1 moves all_gather_minimal_matmul_async's fused SwiGLU into the last K block. The knob is
not in the program hash, so each arm clears the program cache and recompiles. Arms run old, new, old: the two
old arms show whether the block is deterministic, which an exactness check of new vs old needs. Random weights.
"""

import os
import time

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import padded_sequence_length
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import (
    GALAXY_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    upload_rope,
)
from .common_av import CALIBRATED_FOX_PROMPT_NUM_TOKENS
from .test_transformer_minimax_h3 import _packed_sizes, logical_length_tensor

SWIGLU_PACK_ENV = "TT_AGMM_SWIGLU_PACK"
ARMS = os.environ.get("T477_ARMS", "old,new,old").split(",")
ROUNDS = int(os.environ.get("T477_ROUNDS", "3"))


def _synced_ms(mesh_device, fn):
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(mesh_device)
    return out, 1e3 * (time.perf_counter() - t0)


def _device0(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _compare(name: str, a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    for i, (x, y) in enumerate(zip(a, b)):
        equal = (x == y).float().mean().item()
        max_diff = (x - y).abs().max().item()
        logger.info(
            f"T477 {name}[{i}]: bit-identical={torch.equal(x, y)} equal-frac={equal:.6f} max|diff|={max_diff:.3g}"
        )
        assert_quality(x, y, pcc=0.0)


@GALAXY_RING
def test_ff1_swiglu_pack_block_10s(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, monkeypatch, reset_seeds
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    if tuple(mesh_device.shape)[sp_axis] != 8:
        pytest.skip("timed at the production 4x8 sharding")
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    sizes = _packed_sizes(10.0, CALIBRATED_FOX_PROMPT_NUM_TOKENS)
    seq_len = sizes["seq_len"]
    padded_len = padded_sequence_length(seq_len, sp_factor)
    position_ids, tags, timestep_indices = packed_layout(
        sizes["num_text"], sizes["num_audio"], sizes["num_video"], (sizes["grid_h"], sizes["grid_w"]), padded_len
    )
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)
    logger.info(f"T477 seq_len {seq_len} padded {padded_len} ({padded_len // sp_factor} rows/device)")

    torch_block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, REAL_BLOCK_CONFIG["attention_head_dim"])

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    tt_block = MiniMaxH3TransformerBlock(
        **TT_BLOCK_CONFIG,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())
    del torch_block

    tt_spatial = bf16_tensor_2dshard(
        torch.randn(1, 1, padded_len, REAL_BLOCK_CONFIG["hidden_size"]),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
    )
    tt_temb = from_torch(
        torch.randn(1, 1, 2, REAL_BLOCK_CONFIG["time_embed_dim"]), device=mesh_device, dtype=ttnn.float32
    )
    tt_adaln = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_logical = logical_length_tensor(mesh_device, seq_len)

    def run_block() -> ttnn.Tensor:
        return tt_block(
            tt_spatial, tt_logical, temb=tt_temb, adaln_indices=tt_adaln, rope_cos=tt_rope_cos, rope_sin=tt_rope_sin
        )

    agmm = ttnn.experimental.all_gather_minimal_matmul_async
    ff1_outs: list[torch.Tensor] = []
    ff1_ms: list[float] = []
    mode = {"capture": False, "time": False}

    def wrapped_agmm(*args, **kwargs):
        if not kwargs.get("fuse_swiglu", False):
            return agmm(*args, **kwargs)
        if mode["time"]:
            out, ms = _synced_ms(mesh_device, lambda: agmm(*args, **kwargs))
            ff1_ms.append(ms)
        else:
            out = agmm(*args, **kwargs)
        if mode["capture"]:
            ff1_outs.append(_device0(out[0] if isinstance(out, (list, tuple)) else out))
        return out

    monkeypatch.setattr(ttnn.experimental, "all_gather_minimal_matmul_async", wrapped_agmm)

    results = []
    for arm in ARMS:
        monkeypatch.setenv(SWIGLU_PACK_ENV, "1" if arm == "new" else "0")
        mesh_device.clear_program_cache()
        _, compile_ms = _synced_ms(mesh_device, run_block)
        ff1_outs.clear()
        mode["capture"] = True
        out, _ = _synced_ms(mesh_device, run_block)
        mode["capture"] = False
        block_out = [_device0(out)]
        captured = list(ff1_outs)
        assert captured, "ff1 did not go through all_gather_minimal_matmul_async with fuse_swiglu"

        block_ms, ff1_round_ms = [], []
        for _ in range(ROUNDS):
            _, ms = _synced_ms(mesh_device, run_block)
            block_ms.append(ms)
            ff1_ms.clear()
            mode["time"] = True
            run_block()
            mode["time"] = False
            ff1_round_ms.append(sum(ff1_ms))
        logger.info(
            f"T477 arm {arm}: compile {compile_ms:.0f} ms; block best {min(block_ms):.2f} ms "
            f"mean {sum(block_ms) / ROUNDS:.2f}; ff1 best {min(ff1_round_ms):.2f} ms "
            f"mean {sum(ff1_round_ms) / ROUNDS:.2f} ({len(captured)} ff1 calls, {ROUNDS} rounds)"
        )
        results.append((arm, captured, block_out, min(block_ms), min(ff1_round_ms)))

    old = [r for r in results if r[0] == "old"]
    new = [r for r in results if r[0] == "new"]
    if len(old) >= 2:
        _compare("old-vs-old ff1", old[0][1], old[1][1])
        _compare("old-vs-old block", old[0][2], old[1][2])
    if old and new:
        _compare("new-vs-old ff1", new[0][1], old[0][1])
        _compare("new-vs-old block", new[0][2], old[0][2])
        old_block = min(r[3] for r in old)
        old_ff1 = min(r[4] for r in old)
        logger.info(
            f"T477 summary: ff1 new/old {new[0][4] / old_ff1:.4f}, block new/old {new[0][3] / old_block:.4f} "
            f"(best of {ROUNDS} rounds per arm)"
        )
