# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy checks for DFlash context K/V, slot isolation, and trace replay."""

import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.dflash import DFlashPrefill, allocate_dflash_kv_cache
from models.demos.gemma4_d_p.tt.dflash_config import DFlashConfig, load_dflash_weights
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata


def reference_kv(config, state, features, positions):
    context = F.linear(torch.cat(features, dim=-1).float(), state["fc.weight"].float())
    context = F.rms_norm(context, (config.hidden_size,), state["hidden_norm.weight"].float(), config.rms_norm_eps)
    inv_freq = config.rope_theta ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    angles = positions.float().unsqueeze(-1) * inv_freq
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    result = []
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}.self_attn"
        k, v = [
            F.linear(context, state[f"{prefix}.{kind}_proj.weight"].float()).reshape(
                -1, config.num_key_value_heads, config.head_dim
            )
            for kind in ("k", "v")
        ]
        k = F.rms_norm(k, (config.head_dim,), state[f"{prefix}.k_norm.weight"].float(), config.rms_norm_eps)
        left, right = k.chunk(2, dim=-1)
        k = k * angles.cos() + torch.cat((-right, left), dim=-1) * angles.sin()
        result.append((k, v))
    return result


@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 32 * 1024 * 1024})
@pytest.mark.parametrize(
    "checkpoint,chunk_size", [(False, 8192), (True, 8192), (False, 6656)], ids=["synthetic", "checkpoint", "chunk6656"]
)
def test_dflash_slots_and_trace(mesh_device, checkpoint, chunk_size, monkeypatch, tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(8)
    mesh_config = MeshConfig(mesh_device)
    if checkpoint:
        config, state = load_dflash_weights()
    else:
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        config = DFlashConfig(128, 5, 8, 128, (1, 3), 1e-6, 1e6, 16384)
        state = {
            key: (torch.ones(shape) if "norm" in key else torch.randn(shape) * 0.02).to(torch.bfloat16)
            for key, shape in config.weight_shapes().items()
        }
    local_chunk = chunk_size // mesh_config.cp_degree
    caches = allocate_dflash_kv_cache(mesh_config, config, num_users=2, max_seq_len=2 * chunk_size)
    draft = DFlashPrefill(
        mesh_config, config, state, CCLManager(mesh_config), caches, max_seq_len=2 * chunk_size, chunk_size=chunk_size
    )
    metadata = PrefillMetadata(mesh_config)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_config.mesh_shape, dims=(2, None))
    features = [torch.randn(1, 1, chunk_size, config.hidden_size).to(torch.bfloat16) for _ in config.target_layer_ids]
    device_features = [
        ttnn.from_torch(x, device=mesh_device, mesh_mapper=mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        for x in features
    ]

    def forward():
        accumulator = None
        for layer_id, feature in zip(config.target_layer_ids, device_features):
            accumulator = draft.tap(feature, layer_id, accumulator)
        draft.write_kv(accumulator, metadata)

    def read_caches():
        return {
            (kind, cp, tp): ttnn.to_torch(ttnn.get_device_tensors(tensor)[cp * mesh_config.tp_degree + tp]).float()
            for kind, tensor in (("k", caches.k), ("v", caches.v))
            for cp, tp in ((0, 0), (0, 3), (7, 0), (7, 3))
        }

    def check_values(actual, slot, start):
        min_pcc = 1.0
        for cp in (0, 7):
            rows = torch.arange(cp * local_chunk, cp * local_chunk + 64)
            expected = reference_kv(config, state, [x[0, 0, rows] for x in features], rows + start)
            for tp in (0, 3):
                for layer, pair in enumerate(expected):
                    for kind, golden in zip(("k", "v"), pair):
                        observed = actual[kind, cp, tp][
                            slot * config.num_hidden_layers + layer,
                            :,
                            start // mesh_config.cp_degree : start // mesh_config.cp_degree + 64,
                        ]
                        golden = golden[:, tp * 2 : (tp + 1) * 2].permute(1, 0, 2).contiguous()
                        pcc = torch.corrcoef(torch.stack((observed.flatten(), golden.flatten())))[0, 1].item()
                        min_pcc = min(min_pcc, pcc)
                        assert pcc > 0.99, (kind, layer, cp, tp, pcc)
                        relative_rmse = ((observed - golden).square().mean() / golden.square().mean()).sqrt().item()
                        assert relative_rmse < 0.08, (kind, layer, cp, tp, relative_rmse)
        logger.info(
            f"DFlash {'checkpoint' if checkpoint else 'synthetic'} slot={slot} start={start} min_pcc={min_pcc:.6f}"
        )

    metadata.update(slot_idx=0, kv_actual_global=0)
    forward()
    ttnn.synchronize_device(mesh_device)
    baseline = read_caches()
    check_values(baseline, 0, 0)
    for tensor in baseline.values():
        assert torch.count_nonzero(tensor[config.num_hidden_layers :]) == 0

    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for turn, (slot, start) in enumerate(((1, 0), (0, chunk_size), (1, chunk_size), (0, 0))):
            for host, device in zip(features, device_features):
                host.mul_(0.7).add_(0.2 * (turn + 1))
                staged = ttnn.from_torch(host, mesh_mapper=mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(staged, device)
            draft.stage_positions(start)
            metadata.update(slot_idx=slot, kv_actual_global=start)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            actual = read_caches()
            check_values(actual, slot, start)
            other = 1 - slot
            for key in actual:
                assert torch.equal(
                    actual[key][other * config.num_hidden_layers : (other + 1) * config.num_hidden_layers],
                    baseline[key][other * config.num_hidden_layers : (other + 1) * config.num_hidden_layers],
                )
                if start:
                    assert torch.equal(
                        actual[key][
                            slot * config.num_hidden_layers : (slot + 1) * config.num_hidden_layers, :, :local_chunk
                        ],
                        baseline[key][
                            slot * config.num_hidden_layers : (slot + 1) * config.num_hidden_layers, :, :local_chunk
                        ],
                    )
            baseline = actual
        start_time = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"DFlash KV-only trace: {(time.perf_counter() - start_time) * 1000 / 5:.3f} ms per {chunk_size}-token chunk"
        )
    finally:
        ttnn.release_trace(mesh_device, trace)
