# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""V4's KV chunk table against caches holding known data: every chunk it lists, read off the device, is
the chunk the caches hold there.

Flash's first five layers (SWA, SWA, CSA, HCA, CSA) hold one of each migrated kind and two CSA layers, so
both the per-kind row stride and the per-user one are exercised. The caches are laid out exactly as
``allocate_v4_kv_caches`` lays them out, filled from host, and described the way the runner describes
them: ``kv_migration_stages`` -> all-gathered layouts -> ``build_v4_kv_chunk_table``.
"""

import dataclasses

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.migration import allgather_kv_stage_layouts
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import V4_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import _Rows, allocate_v4_kv_caches, build_v4_kv_chunk_table

CONFIG = v4_hf_config(DeepSeekV4FlashConfig)
LAYERS = range(5)
MAX_SEQ_LEN, CHUNK = 10240, 5120
PAGE = 32


def _untilize(raw, width):
    """One 32-row TILE page of bf16 back to rows: tiles left to right, each four 16x16 faces."""
    faces = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(width // 32, 2, 2, 16, 16)
    return faces.permute(1, 3, 0, 2, 4).reshape(32, width)


def _upload(mesh_device, host, like, mapper):
    return ttnn.from_torch(
        host,
        dtype=like.dtype,
        layout=like.layout,
        device=mesh_device,
        memory_config=like.memory_config(),
        mesh_mapper=mapper,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, topology", V4_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("num_users", [1, 2], ids=["users1", "users2"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
def test_table_reads_back_every_chunk(mesh_device, device_params, topology, num_users, tmp_path):
    mesh_shape, sp = tuple(mesh_device.shape), mesh_device.shape[0]
    rows = _Rows.of(CONFIG, MAX_SEQ_LEN, CHUNK, sp)
    caches = allocate_v4_kv_caches(
        mesh_device, CONFIG, LAYERS, slot_num=num_users, max_seq_len=MAX_SEQ_LEN, chunk=CHUNK, sp_axis=0
    )
    torch.manual_seed(0)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    hca = torch.randn(num_users * len(caches.hca_layers), 1, rows.hca, CONFIG.head_dim).bfloat16()
    csa = torch.randn(num_users * len(caches.csa_layers), 1, rows.csa, CONFIG.head_dim).bfloat16()
    # The index cache is held block-cyclic: SP-split the physical order, and expect the natural one back.
    index = torch.randn(num_users * len(caches.csa_layers), 1, rows.index, CONFIG.index_head_dim)
    index = ttnn.to_torch(ttnn.from_torch(index, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).bfloat16()
    physical = index[:, :, blockcyclic_positions(sp, rows.period, rows.index)]
    sp_split = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    filled = dataclasses.replace(
        caches,
        hca=_upload(mesh_device, hca, caches.hca, replicate),
        csa=_upload(mesh_device, csa, caches.csa, replicate),
        index=_upload(mesh_device, physical, caches.index, sp_split),
    )

    layouts = allgather_kv_stage_layouts(mesh_device, filled.migration_stages(CONFIG), mesh_shape)
    path = build_v4_kv_chunk_table(
        CONFIG,
        stage_layouts=layouts,
        mesh_shape=mesh_shape,
        sp_axis=0,
        num_users=num_users,
        max_seq_len=MAX_SEQ_LEN,
        chunk=CHUNK,
        path=str(tmp_path / "v4.pb"),
    )
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(path)

    def decode(config_id, raw):
        if config_id == 0:
            return _untilize(raw, CONFIG.head_dim)
        if config_id == 1:
            return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(1, CONFIG.head_dim)
        page = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw, [1, 1, PAGE, CONFIG.index_head_dim])
        return ttnn.to_torch(page).bfloat16()[0, 0]

    # (config, layers, host rows, published rows, rows a page)
    kinds = (
        (0, caches.hca_layers, hca, rows.hca, PAGE),
        (1, caches.csa_layers, csa, rows.csa_entries, 1),
        (2, caches.csa_layers, index, rows.index, PAGE),
    )
    for config_id, layers, host, published, step in kinds:
        for slot in range(num_users):
            for pos_in_kind, layer in enumerate(layers):
                want_rows = host[slot * len(layers) + pos_in_kind, 0]
                for position in range(0, published, step):
                    got = decode(config_id, table.read_device_chunk(layer, position, slot, config_id))
                    want = want_rows[position : position + step]
                    assert torch.equal(got, want), f"config {config_id} layer {layer} slot {slot} row {position}"
