# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC/top-k coverage for the V4 CSA indexer on the GLM block-cyclic cache path."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtCsaIndexer
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache


def _config(max_seq_len):
    return DeepseekV4Config(
        hidden_size=256,
        q_lora_rank=128,
        head_dim=512,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=16,
        max_position_embeddings=max_seq_len,
        num_hidden_layers=1,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
    )


def _compute_config(mesh_device, *, fp32):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if fp32 else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def _to_hidden(mesh_device, tensor):
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _to_qr(mesh_device, tensor):
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _assert_topk_overlap(actual, expected, minimum=0.9):
    valid = expected >= 0
    if not valid.any():
        return
    overlaps = []
    for row_actual, row_expected, row_valid in zip(
        actual.reshape(-1, actual.shape[-1]),
        expected.reshape(-1, expected.shape[-1]),
        valid.reshape(-1, valid.shape[-1]),
    ):
        expected_set = set(row_expected[row_valid].tolist())
        if not expected_set:
            continue
        actual_set = set(row_actual[row_actual >= 0].tolist())
        overlaps.append(len(expected_set & actual_set) / len(expected_set))
    assert overlaps and sum(overlaps) / len(overlaps) >= minimum


# (slot_num, layer_num, cache_user_id, cache_layer_idx). The multi-slot case is the whole point of the
# user-major layer-stacked layout: write_k and the ring scorer each compute the flat slot themselves, so
# a disagreement makes the scorer read a zeroed slot and the top-k overlap collapses.
_CACHE_SLOTS = [
    pytest.param(1, 1, 0, 0, id="single-slot"),
    pytest.param(2, 2, 1, 1, id="user1-layer1"),
]


@pytest.mark.parametrize("slot_num, layer_num, cache_user_id, cache_layer_idx", _CACHE_SLOTS)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="2x2",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_csa_indexer_block_cyclic_two_chunks(
    mesh_device, device_params, slot_num, layer_num, cache_user_id, cache_layer_idx, tmp_path
):
    torch.manual_seed(17)
    sp_axis, tp_axis = 0, 1
    sp_factor = mesh_device.shape[sp_axis]
    chunk_tokens = 256
    max_seq_len = 2 * chunk_tokens
    local_chunk = chunk_tokens // sp_factor
    config = _config(max_seq_len)
    reference = DeepseekV4Indexer(config).to(torch.bfloat16).eval()
    torch.nn.init.normal_(reference.position_bias, std=0.02)

    tt_ccl = get_tt_ccl(mesh_device)
    indexer = TtCsaIndexer.from_reference(
        reference,
        config=config,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        default_compute_kernel_config=_compute_config(mesh_device, fp32=False),
        hifi4_fp32_compute_kernel_config=_compute_config(mesh_device, fp32=True),
        weight_cache_path=tmp_path,
        layer_idx=0,
        tt_ccl=tt_ccl,
        ccl_num_links=2 if is_blackhole() else 1,
        sp_ccl_topology=ttnn.Topology.Linear,
        tp_ccl_topology=ttnn.Topology.Linear,
        seq_len=max_seq_len,
        active_seq_len=chunk_tokens,
        slot_num=slot_num,
        layer_num=layer_num,
    )
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=max_seq_len // 4,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=layer_num,
        num_users=slot_num,
    )

    hidden_all = torch.randn(1, max_seq_len, config.hidden_size, dtype=torch.bfloat16)
    qr_all = torch.randn(1, max_seq_len, config.q_lora_rank, dtype=torch.bfloat16)
    expected_all = reference(
        hidden_all,
        qr_all,
        torch.arange(max_seq_len).unsqueeze(0),
        past_key_values=None,
        layer_idx=0,
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1))
    for chunk_idx in range(2):
        start = chunk_idx * chunk_tokens
        stop = start + chunk_tokens
        hidden = hidden_all[:, start:stop].unsqueeze(1)
        qr = qr_all[:, start:stop].unsqueeze(1)
        actual_tt = indexer.forward(
            _to_hidden(mesh_device, hidden),
            _to_qr(mesh_device, qr),
            seq_len=local_chunk,
            start_pos=start,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            index_kv_cache=index_cache,
        )
        actual_mesh = ttnn.to_torch(actual_tt, mesh_composer=composer).to(torch.int64)
        expected = expected_all[:, start:stop]
        for tp_rank in range(mesh_device.shape[tp_axis]):
            actual = actual_mesh[:, tp_rank]
            actual = torch.where(actual == 0xFFFFFFFF, -1, actual)
            assert actual.shape == expected.shape
            _assert_topk_overlap(actual, expected)

        if start == 0:
            assert torch.all(actual_mesh[:, 0, 0] == 0xFFFFFFFF)
