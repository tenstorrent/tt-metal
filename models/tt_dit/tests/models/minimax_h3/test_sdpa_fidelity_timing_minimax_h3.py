# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One dense attention layer at the FastH3 Turbo fl2va 10 s shape: ring SDPA matmuls at HiFi2 vs LoFi.

Not a correctness test. Both arms run in one process on one mesh, interleaved, each timed as a whole
forward and as the ring SDPA op alone, with a device sync around each. Random weights and inputs; the
output PCC between the arms is a first look at the LoFi error, not a quality verdict.
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3 import attention_minimax_h3 as am
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import GALAXY_RING, upload_rope

HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16

# fl2va 10 s at 768p as the pipeline packs it: text, conditioning image and audio rows, then the video grid,
# padded to the pipeline's bucket and masked by logical_n.
SEQ_LEN_10S = 1460 + 1008 + 405 + 72 * 24 * 42
BUCKET_10S = 76800
ROUNDS = int(os.environ.get("T466_ROUNDS", "3"))


def _attention_state() -> dict[str, torch.Tensor]:
    inner = NUM_HEADS * HEAD_DIM
    std = HIDDEN_SIZE**-0.5
    state = {f"to_{p}.weight": std * torch.randn(inner, HIDDEN_SIZE) for p in "qkv"}
    state["to_out.0.weight"] = inner**-0.5 * torch.randn(HIDDEN_SIZE, inner)
    for name in ("norm_q", "norm_k"):
        state[f"{name}.weight"] = 1.0 + 0.5 * torch.randn(HEAD_DIM)
    return state


def _model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, state):
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    model = am.MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    model.load_torch_state_dict(dict(state))
    return model


def _synced_ms(mesh_device, fn):
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(mesh_device)
    return out, 1e3 * (time.perf_counter() - t0)


@GALAXY_RING
def test_ring_sdpa_matmul_fidelity_timing_10s(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, monkeypatch, reset_seeds
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    if tuple(mesh_device.shape)[sp_axis] != 8:
        pytest.skip("timed at the production 4x8 sharding")

    state = _attention_state()
    pad = BUCKET_10S - SEQ_LEN_10S
    x = torch.randn(SEQ_LEN_10S, HIDDEN_SIZE)
    x = torch.cat([x, x.new_zeros(pad, HIDDEN_SIZE)]).reshape(1, 1, BUCKET_10S, HIDDEN_SIZE)
    angles = torch.rand(SEQ_LEN_10S, 2 * 3 * ROPE_FREQ_DIM) * 2 * math.pi
    cos, sin = am.prepare_rope_tables(torch.cos(angles), torch.sin(angles), HEAD_DIM)
    cos, sin = torch.cat([cos, cos[:pad]]), torch.cat([sin, sin[:pad]])
    tt_cos, tt_sin = upload_rope(cos, sin, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_x = bf16_tensor_2dshard(x, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    logical_n = from_torch(
        torch.tensor([SEQ_LEN_10S], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, None],
    )

    arms = {}
    for name, env in (("hifi2", None), ("lofi", "lofi")):
        if env is None:
            monkeypatch.delenv(am.SDPA_MM_FIDELITY_ENV, raising=False)
        else:
            monkeypatch.setenv(am.SDPA_MM_FIDELITY_ENV, env)
        arms[name] = _model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, state)
        logger.info(f"T466 {name}: sdpa_matmul_fidelity={arms[name].sdpa_matmul_fidelity}")

    def forward(model):
        return model(tt_x, logical_n=logical_n, rope_cos=tt_cos, rope_sin=tt_sin)

    outputs = {}
    for name, model in arms.items():  # compile, and keep device 0's shard for the PCC
        out = forward(model)
        outputs[name] = ttnn.to_torch(ttnn.get_device_tensors(out)[0])[..., : SEQ_LEN_10S // 8, :]
        ttnn.deallocate(out)

    sdpa_ms = []
    ring_sdpa = ttnn.transformer.ring_joint_scaled_dot_product_attention

    def timed_sdpa(*args, **kwargs):
        out, ms = _synced_ms(mesh_device, lambda: ring_sdpa(*args, **kwargs))
        sdpa_ms.append(ms)
        return out

    totals = {name: [] for name in arms}
    sdpas = {name: [] for name in arms}
    for _ in range(ROUNDS):
        for name, model in arms.items():
            out, ms = _synced_ms(mesh_device, lambda: forward(model))
            ttnn.deallocate(out)
            totals[name].append(ms)
            with monkeypatch.context() as m:
                m.setattr(ttnn.transformer, "ring_joint_scaled_dot_product_attention", timed_sdpa)
                sdpa_ms.clear()
                ttnn.deallocate(forward(model))
            sdpas[name].append(sum(sdpa_ms))

    for name in arms:
        logger.info(
            f"T466 {name}: layer best {min(totals[name]):.2f} ms mean {sum(totals[name]) / ROUNDS:.2f}; "
            f"ring_joint_sdpa best {min(sdpas[name]):.2f} ms mean {sum(sdpas[name]) / ROUNDS:.2f} ({ROUNDS} rounds)"
        )
    gain = 1 - min(sdpas["lofi"]) / min(sdpas["hifi2"])
    layer_gain = 1 - min(totals["lofi"]) / min(totals["hifi2"])
    logger.info(f"T466 summary: ring_joint_sdpa gain {100 * gain:.1f}%, layer gain {100 * layer_gain:.1f}%")
    assert_quality(outputs["hifi2"], outputs["lofi"], pcc=0.0)
