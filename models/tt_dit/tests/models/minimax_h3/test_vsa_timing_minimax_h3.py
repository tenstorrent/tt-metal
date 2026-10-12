# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One attention layer at the FastH3 Turbo fl2va 10 s shape: dense ring SDPA vs VSA, timed on host.

Not a correctness test. Each arm runs one warm-up forward, then timed forwards with a device sync
around the whole call (`total`), then forwards with a sync around each wrapped op (`ops`), which
splits VSA's time into the coarse stage, the K/V all-gathers and the fine stage. Random weights and
inputs: the selection pattern differs from production, the per-tile work (k, exempt rows) does not.
"""

import math
import os
import time
from collections import defaultdict

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSACoarseStage, MiniMaxH3VSAConfig
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import GALAXY_RING, upload_rope

# Self-contained (no diffusers MiniMax H3 reference): the device boxes' venvs ship stock diffusers.
HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16

# fl2va 10 s at 768p as the pipeline packs it: text, conditioning image, audio rows, then the video grid.
PREFIX_10S = (1460, 1008, 405)
GRID_10S = (72, 24, 42)
DENSE_BUCKET_10S = 76800  # the dense arm's padded length in job 990's reference (#408 out_t10)
TIMED_ITERS = int(os.environ.get("T453_ITERS", "3"))


def _attention_state() -> dict[str, torch.Tensor]:
    inner = NUM_HEADS * HEAD_DIM
    std = HIDDEN_SIZE**-0.5
    state = {f"to_{p}.weight": std * torch.randn(inner, HIDDEN_SIZE) for p in "qkv"}
    state["to_out.0.weight"] = inner**-0.5 * torch.randn(HIDDEN_SIZE, inner)
    for name in ("norm_q", "norm_k"):
        state[f"{name}.weight"] = 1.0 + 0.5 * torch.randn(HEAD_DIM)
    return state


def _logical_length(mesh_device, value):
    return from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, None],
    )


def _setup_tt_model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry, vsa_config, state_dict):
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    tt_model = MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        vsa_config=vsa_config,
    )
    tt_model.load_torch_state_dict(dict(state_dict))
    if vsa_config is not None:
        tt_model.set_vsa_stage(
            MiniMaxH3VSACoarseStage(
                geometry,
                sparsity=vsa_config.sparsity,
                padded_pooling=vsa_config.padded_pooling,
                head_dim=HEAD_DIM,
                mesh_device=mesh_device,
                sp_axis=sp_axis,
                ccl_manager=ccl_manager,
            )
        )
    return tt_model, ccl_manager


def _timed(fn, mesh_device, acc, label):
    def wrapper(*args, **kwargs):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ttnn.synchronize_device(mesh_device)
        key = label(*args, **kwargs) if callable(label) else label
        acc[key].append(time.perf_counter() - t0)
        return out

    return wrapper


def _rope(n_rows):
    angles = torch.rand(n_rows, 2 * 3 * ROPE_FREQ_DIM) * 2 * math.pi
    return prepare_rope_tables(torch.cos(angles), torch.sin(angles), HEAD_DIM)


def _forward_ms(tt_model, mesh_device, tt_x, logical_n, tt_cos, tt_sin, iters):
    times = []
    for _ in range(iters):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = tt_model(tt_x, logical_n=logical_n, rope_cos=tt_cos, rope_sin=tt_sin)
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
        ttnn.deallocate(out)
    return 1e3 * min(times), 1e3 * sum(times) / len(times)


def _time_arm(name, tt_model, ccl_manager, mesh_device, tt_x, logical_n, tt_cos, tt_sin, monkeypatch):
    out = tt_model(tt_x, logical_n=logical_n, rope_cos=tt_cos, rope_sin=tt_sin)  # compile
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    best, mean = _forward_ms(tt_model, mesh_device, tt_x, logical_n, tt_cos, tt_sin, TIMED_ITERS)
    logger.info(f"T453 {name} total: best {best:.2f} ms, mean {mean:.2f} ms over {TIMED_ITERS}")

    acc = defaultdict(list)
    with monkeypatch.context() as m:
        for op in ("vsa_sdpa", "vsa_ring_sdpa", "ring_joint_scaled_dot_product_attention"):
            m.setattr(ttnn.transformer, op, _timed(getattr(ttnn.transformer, op), mesh_device, acc, op))
        m.setattr(
            MiniMaxH3VSACoarseStage,
            "__call__",
            _timed(MiniMaxH3VSACoarseStage.__call__, mesh_device, acc, "vsa_coarse_stage"),
        )
        m.setattr(
            ccl_manager,
            "all_gather_persistent_buffer",
            _timed(
                ccl_manager.all_gather_persistent_buffer,
                mesh_device,
                acc,
                lambda *a, **k: f"all_gather(axis={k.get('mesh_axis')}, dim={k.get('dim')})",
            ),
        )
        synced, _ = _forward_ms(tt_model, mesh_device, tt_x, logical_n, tt_cos, tt_sin, TIMED_ITERS)
    parts = {key: 1e3 * min(v) for key, v in acc.items()}
    for key, ms in sorted(parts.items(), key=lambda kv: -kv[1]):
        logger.info(f"T453 {name} op {key}: best {ms:.2f} ms x{len(acc[key]) // TIMED_ITERS}/fwd")
    logger.info(f"T453 {name} synced-op forward: best {synced:.2f} ms")
    return best


@GALAXY_RING
@pytest.mark.parametrize("vsa_ring", [False, True], ids=["two_op", "vsa_ring"])
@pytest.mark.parametrize("sparsity", [0.9])
def test_vsa_vs_dense_attention_timing_10s(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, sparsity, vsa_ring, monkeypatch, reset_seeds
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    if sp_factor != 8:
        pytest.skip("timed at the production 4x8 sharding")
    monkeypatch.setenv("VSA_RING", "1" if vsa_ring else "0")

    geometry = build_vsa_geometry(PREFIX_10S, GRID_10S, sp_factor=sp_factor)
    vsa_config = MiniMaxH3VSAConfig(sparsity=sparsity)
    seq_len = geometry.seq_len
    logger.info(
        f"T453 geometry: seq_len={seq_len} padded_len={geometry.padded_len} tiles={geometry.n_tiles} "
        f"({geometry.n_pad_tiles} pad)"
    )
    state = _attention_state()
    x = torch.randn(1, seq_len, HIDDEN_SIZE)
    cos, sin = _rope(seq_len)
    logical_n = _logical_length(mesh_device, seq_len)
    shard = {sp_axis: 2, tp_axis: 3}

    results = {}
    if not vsa_ring:
        # dense arm: packed order, pad tail to the pipeline's bucket and masked by logical_n
        pad = DENSE_BUCKET_10S - seq_len
        x_d = torch.cat([x[0], x.new_zeros(pad, HIDDEN_SIZE)]).reshape(1, 1, DENSE_BUCKET_10S, HIDDEN_SIZE)
        cos_d = torch.cat([cos, cos[:pad]])
        sin_d = torch.cat([sin, sin[:pad]])
        dense_model, dense_ccl = _setup_tt_model(
            mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry, None, state
        )
        tt_cos, tt_sin = upload_rope(cos_d, sin_d, mesh_device=mesh_device, sp_axis=sp_axis)
        tt_x = bf16_tensor_2dshard(x_d, device=mesh_device, shard_mapping=shard)
        results["dense"] = _time_arm(
            "dense", dense_model, dense_ccl, mesh_device, tt_x, logical_n, tt_cos, tt_sin, monkeypatch
        )
        ttnn.deallocate(tt_x)
        del dense_model

    vsa_model, vsa_ccl = _setup_tt_model(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry, vsa_config, state
    )
    stage = vsa_model.vsa_stage
    logger.info(f"T453 vsa: k={stage.k} exempt={stage.n_exempt} dense_row_hint={stage.dense_row_hint}")
    x_t = geometry.pack_rows(x[0], dim=0).reshape(1, 1, geometry.padded_len, HIDDEN_SIZE)
    tt_cos, tt_sin = upload_rope(
        geometry.permute_metadata(cos, dim=0),
        geometry.permute_metadata(sin, dim=0),
        mesh_device=mesh_device,
        sp_axis=sp_axis,
    )
    tt_x = bf16_tensor_2dshard(x_t, device=mesh_device, shard_mapping=shard)
    name = f"vsa{sparsity}_{'ring' if vsa_ring else 'two_op'}"
    results[name] = _time_arm(name, vsa_model, vsa_ccl, mesh_device, tt_x, logical_n, tt_cos, tt_sin, monkeypatch)
    logger.info(f"T453 summary (best total ms): {results}")
