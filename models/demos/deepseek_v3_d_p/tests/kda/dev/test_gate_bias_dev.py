# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the device Kimi-K3 ``layers.0`` decay gate's error against FP64 on real input (dev measurement, not a gate).

For the first 5,120-token production call of the golden's tokens, it evaluates the production gate
(``ttKDA._compute_decay``) and the BF16-bias chain it replaced (folded BF16 projection and BF16 offset A * dt_bias
passed as the matmul ``bias=`` with a fused sigmoid, as on main cf74f1b5ce8), and compares both with the FP64 gate
sigmoid(A (f_b r + dt_bias)) evaluated from each device's own decay rank r. It writes the metrics as JSON.

Environment: KIMI_K3_CKPT, KDA_TRACE_DIR, KDA_GATE_BIAS_OUT (output JSON path; the test skips without them).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FirstLayerInput
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _L1_SMALL_SIZE
from models.demos.deepseek_v3_d_p.tests.kda.utils import KimiK3TestCase, make_kimi_k3_device_case
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]


def _shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _bf16_chain_gate(layer, state_dict, config, projected, offset, mesh_device, tensor_parallel_axis):
    """The replaced chain: BF16 A (f_b), BF16 A dt_bias as the matmul bias, sigmoid fused into the matmul."""
    scale = state_dict["A_log"].float().exp().expand(-1, -1, -1, config.head_k_dim).reshape(1, config.q_dim)
    scale = scale.to(torch.bfloat16).float()
    projection = (state_dict["f_b_proj.weight"].T.to(torch.bfloat16).float() * scale).to(torch.bfloat16)
    bias = (state_dict["dt_bias"].reshape(1, config.q_dim).to(torch.bfloat16).float() * scale).to(torch.bfloat16)
    mesh_dims = [None, None]
    mesh_dims[tensor_parallel_axis] = -1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(mesh_dims), mesh_shape=tuple(mesh_device.shape))
    upload = lambda tensor: ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    base = layer.decay_program_config
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=base.compute_with_storage_grid_size,
        in0_block_w=base.in0_block_w,
        out_subblock_h=base.out_subblock_h,
        out_subblock_w=base.out_subblock_w,
        out_block_h=base.out_block_h,
        out_block_w=base.out_block_w,
        per_core_M=base.per_core_M,
        per_core_N=base.per_core_N,
        transpose_mcast=base.transpose_mcast,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
        fuse_batch=True,
    )
    return ttnn.linear(
        projected,
        upload(projection),
        bias=upload(bias),
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=layer.compute_config,
        in0_column_offset=offset,
    )


def _metrics(gate: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    relative = (gate - expected) / expected
    channel = relative.reshape(-1, relative.shape[-1]).mean(dim=0)
    return {
        "mean_dG_over_G": relative.mean().item(),
        "mean_abs_channel_bias": channel.abs().mean().item(),
        "max_abs_channel_bias": channel.abs().max().item(),
        "rms_dG_over_G": relative.square().mean().sqrt().item(),
        "max_abs_dG_over_G": relative.abs().max().item(),
    }


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        # Local verification only: Fabric2D on a Galaxy without the x wrap cabled.
        pytest.param(
            (8, 4),
            1,
            fabric2d_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-fabric2d",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_k3_layer0_gate_bias(mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params) -> None:
    env = {name: os.environ.get(name) for name in ("KIMI_K3_CKPT", "KDA_TRACE_DIR", "KDA_GATE_BIAS_OUT")}
    if not all(env.values()):
        pytest.skip("set KIMI_K3_CKPT, KDA_TRACE_DIR and KDA_GATE_BIAS_OUT")
    checkpoint = Path(env["KIMI_K3_CKPT"])
    config = kimi_k3_kda_config()
    state_dict = load_kda_layer_state_dict(checkpoint, 0, config)
    token_ids = load_trace_token_ids(resolve_trace_dir(env["KDA_TRACE_DIR"]))[:PREFILL_CHUNK_TOKENS]
    case = KimiK3TestCase(config, state_dict, FirstLayerInput(checkpoint)(token_ids), kda_state_dict_sha256(state_dict))
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device, case, tensor_parallel_axis=tensor_parallel_axis, cache_weights=False
    )
    inputs = layer._project_inputs(hidden_tt)
    offset = inputs.decay_rank_offset
    gates = {
        "fp32_offset_add": _shards(layer._compute_decay(inputs.projected, offset)),
        "bf16_matmul_bias": _shards(
            _bf16_chain_gate(layer, state_dict, config, inputs.projected, offset, mesh_device, tensor_parallel_axis)
        ),
    }
    projected = _shards(inputs.projected)

    tp_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    local_q = config.q_dim // tp_size
    scale = state_dict["A_log"].double().exp().expand(-1, -1, -1, config.head_k_dim).reshape(config.q_dim)
    f_b = state_dict["f_b_proj.weight"].double()
    dt_bias = state_dict["dt_bias"].double().reshape(config.q_dim)
    expected, actual = [], {name: [] for name in gates}
    columns = tuple(mesh_device.shape)[1]
    for index, shard in enumerate(projected):
        tp_rank = divmod(index, columns)[tensor_parallel_axis]
        heads = slice(tp_rank * local_q, (tp_rank + 1) * local_q)
        rank = shard.double().reshape(-1, shard.shape[-1])[:, offset : offset + config.head_k_dim]
        expected.append(torch.sigmoid(scale[heads] * (rank @ f_b[heads].T + dt_bias[heads])))
        for name, shards in gates.items():
            actual[name].append(shards[index].double().reshape(-1, local_q))
    # Group by TP rank so the per-channel bias averages over every token of that channel.
    by_channel = lambda tensors: torch.cat(
        [
            torch.cat([t for i, t in enumerate(tensors) if divmod(i, columns)[tensor_parallel_axis] == r], dim=0)
            for r in range(tp_size)
        ],
        dim=1,
    )
    expected_all = by_channel(expected)
    result = {
        "tokens": PREFILL_CHUNK_TOKENS,
        "dtypes": {name: str(state_dict[name].dtype) for name in ("A_log", "dt_bias", "f_b_proj.weight")},
        "max_abs_logit": torch.logit(expected_all.clamp(1e-300, 1 - 1e-16)).abs().max().item(),
        **{name: _metrics(by_channel(tensors), expected_all) for name, tensors in actual.items()},
    }
    out = Path(env["KDA_GATE_BIAS_OUT"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
