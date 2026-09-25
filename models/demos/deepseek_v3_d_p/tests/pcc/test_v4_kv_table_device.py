# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate 1a, in-process: prefill two chunks through TtV4PrefillRuntime into the engine-owned caches, build the
contract's multi-config KV chunk table + device map exactly as the mock-migration runner does, then read every
config back DEVICE-LESSLY through the table (ttnn.experimental.disaggregation.read_dram_umd + the producer's
chunk decoder) and compare with the device tensors read through ttnn. Proves the linear address math on real
hardware for all four migrated kinds (bfp8 tiles x512 / x128, bf16 ROW_MAJOR x512)."""

import json
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.runners.migration import serialize_device_map
from models.demos.common.prefill.runners.prefill_producer import _decode_kv_chunk, _resolve_unique_id
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_block import reference_layer_weights
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_transformer import _init_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.v4 import kv_contract as kc
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import layers_of_kind
from models.demos.deepseek_v3_d_p.tt.v4.runtime import TtV4PrefillRuntime, TtV4PrefillRuntimeConfig

_EXPERTS = 64
_CHUNK = 1024
_CHUNKS = [1024, 1024]
_USERS = 1

_MESH_CONFIGS = [
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        id="fabric2d-mesh-2x4",
    ),
]


def _one_chip(mesh_device, t):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1)))
    )


@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_kv_chunk_table_reads_back_every_config(mesh_device, device_params, tmp_path):
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)  # SWA SWA CSA HCA
    cfg.n_routed_experts = _EXPERTS
    model = _init_model(cfg)
    total = sum(_CHUNKS)
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, total))
    sp, tp = mesh_device.shape
    top = {
        "model.embed_tokens.weight": model.embed_tokens.weight.detach().clone(),
        "model.hc_head.hc_fn": model.hc_head.hc_fn.detach().clone(),
        "model.hc_head.hc_base": model.hc_head.hc_base.detach().clone(),
        "model.hc_head.hc_scale": model.hc_head.hc_scale.detach().clone(),
        "model.norm.weight": model.norm.weight.detach().clone(),
    }
    rc = TtV4PrefillRuntimeConfig(
        chunk_size=_CHUNK,
        max_seq_len=total,
        first_layer_idx=0,
        num_layers=4,
        is_first_rank=True,
        is_last_rank=True,
        num_users=_USERS,
        mesh_shape=(sp, tp),
        kv_only_last_layer=True,
    )
    params = SimpleNamespace(
        max_seq_len=total,
        sp_factor=sp,
        first_layer_idx=0,
        num_layers=4,
        mesh_shape=(sp, tp),
        sp_axis=0,
        num_users=_USERS,
    )
    caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=cfg, params=params)
    rt = TtV4PrefillRuntime(
        mesh_device,
        cfg,
        rc,
        layer_weights=lambda i: reference_layer_weights(model.layers[i]),
        top_level_weights=top,
        num_routed_experts=_EXPERTS,
    )
    rt.compile(caches)
    start = 0
    for k, n in enumerate(_CHUNKS):
        chunk_ids = ids[0, start : start + n].tolist()
        if k % 2 == 0:
            x = rt.make_chunk_input(chunk_ids)
        else:
            # request mode (DS4F-0245): the chunk arrives over the H2D socket as a device tensor and no host stash exists,
            # so the runtime must read the ids back for the hash-routed MoE layers (0..2) -- exercised here without the engine
            x = prepare_prefill_input_tensor(chunk_ids, mesh_device, sp, False, (sp, tp), 0)
            assert rt._needs_token_ids
            assert torch.equal(rt._token_ids_from_device(x), torch.tensor(chunk_ids, dtype=torch.int64))
            # the device view the gate consumes: global [S/32, 32] SP-sharded, TP-replicated -> the same ids in order
            view = ttnn.to_torch(
                rt._token_ids_view(x),
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, 1)),
            )
            assert torch.equal(view[:, :32].reshape(-1).to(torch.int64), torch.tensor(chunk_ids, dtype=torch.int64))
        out = rt.prefill_chunk(x, caches, slot_id=0, actual_start=start, actual_end=start + n, request_id=0)
        assert out is None  # last (and only) rank: the caches are the output
        start += n
    ttnn.synchronize_device(mesh_device)

    # the mock-migration runner path: table + device map
    table_path = rt.build_kv_chunk_table(caches, path=str(tmp_path / "kv_table.pb"))
    map_path = serialize_device_map(mesh_device, str(tmp_path / "device_map.json"))
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(table_path)
    with open(map_path) as f:
        device_map = {tuple(int(x) for x in k.split(":")): int(v) for k, v in json.load(f).items()}
    geom = caches.geometry
    plan = [g for g in kc.CONTRACT if layers_of_kind(cfg, g.kind)]
    assert table.num_configs() == len(plan), (table.num_configs(), [g.name for g in plan])
    logger.info(
        f"table: {table.num_configs()} configs, {table.total_entries()} entries; device map {len(device_map)} chips"
    )

    worst = 1.0
    for config_id, g in enumerate(plan):
        tensor = caches.group_tensors()[g.name]
        host = _one_chip(mesh_device, tensor).float()  # [users*layers, 1, rows, W]
        layers = geom.layers(g.name)
        extent = geom.extent(g.name)
        for slot in range(_USERS):
            for local, gidx in enumerate(layers):
                kind_rank = layers_of_kind(cfg, g.kind).index(gidx)
                batch = slot * len(layers) + local
                rows = []
                for pos in range(0, extent, kc.CHUNK_N_TOKENS):
                    loc = table.lookup(kind_rank, pos, slot, config_id)
                    uid = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
                    raw = ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes)
                    assert len(raw) == g.chunk_size_bytes, (g.name, len(raw), g.chunk_size_bytes)
                    rows.append(_decode_kv_chunk(bytes(raw), g.width))
                via_table = torch.cat(rows, 0).float()  # [extent, W]
                ref = host[batch, 0, :extent]
                nz = ref.abs().sum(-1) > 0
                _, pcc = comp_pcc(ref, via_table) if nz.any() else (True, 1.0)
                maxdiff = (ref - via_table).abs().max().item()
                logger.info(
                    f"{g.name:12s} config {config_id} layer {gidx} slot {slot}: {int(nz.sum())}/{extent} nonzero rows, PCC {pcc:.6f}, max|diff| {maxdiff:.3e}"
                )
                if nz.any():
                    worst = min(worst, pcc)
                    assert (
                        maxdiff < 1e-6
                    ), f"{g.name} layer {gidx}: bytes read through the table differ from the device tensor"
    assert worst > 0.99999, worst
