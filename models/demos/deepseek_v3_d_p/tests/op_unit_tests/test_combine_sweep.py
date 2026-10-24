# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-launch cost of combine v1 and combine_fabric2d, over models, traffic shapes and input layouts."""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from conftest import reset_fabric, set_fabric
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from tests.scripts.common import get_updated_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_prefill_combine import COMBINE_MODELS, _model_scaledown
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    combine_overlap_payload_size,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_gate_outputs,
    moe_fabric_payload_size,
)

_REPS = int(os.environ.get("CMB_SWEEP_REPS", "10"))
_MESH = (8, 4)
_SEQ_LEN = 640
_CAPACITY_FACTOR = 2
_NUM_LINKS = 2
_TRACE_REGION = 330000
_TRAFFIC = {"balanced": None, "hot_expert": 0}
_LAYOUTS = {"row_major": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}


def _axis(name):
    return os.environ[f"CMB_SWEEP_{name}"].split(",")


def _dispatch_table(num_routed_experts, H, G):
    experts_per_group = num_routed_experts // G
    table = torch.full((G, num_routed_experts + 1), -1, dtype=torch.int32)
    for g in range(G):
        for e in range(experts_per_group):
            table[g, g * experts_per_group + e] = e // (experts_per_group // H)
    return table


def _indices(G, H, topk, num_routed_experts, hot_expert):
    torch.manual_seed(42)
    if hot_expert is None:
        return torch.randint(0, num_routed_experts, (G, H, _SEQ_LEN, topk), dtype=torch.int64)
    else:
        others = torch.randint(0, num_routed_experts - 1, (G, H, _SEQ_LEN, topk), dtype=torch.int64)
        indices = others + (others >= hot_expert).to(torch.int64)
        indices[..., 0] = hot_expert
        return indices


def _routing(indices, H, G, num_routed_experts, experts_per_chip, topk, capacity):
    """Per chip: where each pick is held, which page it sits on, and the control tables that say so."""
    table = _dispatch_table(num_routed_experts, H, G)
    offs = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    counts = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    region = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g], H, num_routed_experts, experts_per_chip, _SEQ_LEN, topk, expert_dispatch_table=table[g : g + 1]
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)

    holder = torch.full((G, H, _SEQ_LEN, topk), -1, dtype=torch.int64)
    page = torch.full((G, H, _SEQ_LEN, topk), -1, dtype=torch.int64)
    for g in range(G):
        for s in range(H):
            flat = indices[g, s].reshape(-1)
            rank = (torch.nn.functional.one_hot(flat, num_routed_experts).cumsum(0) - 1).gather(1, flat.unsqueeze(1))
            pos = table[g, flat].to(torch.int64)
            holder[g, s] = torch.where(pos != -1, pos, -1).reshape(_SEQ_LEN, topk)
            page[g, s] = torch.where(pos != -1, offs[g, s, flat].to(torch.int64) + rank.squeeze(1), -1).reshape(
                _SEQ_LEN, topk
            )
    kept = (holder >= 0) & (page < capacity)

    meta = torch.full((H, G, capacity, 3), -1, dtype=torch.int32)
    g_idx, s_idx, t_idx, k_idx = kept.nonzero(as_tuple=True)
    meta[holder[kept], g_idx, page[kept]] = torch.stack([s_idx * G + g_idx, t_idx, k_idx], dim=-1).to(torch.int32)
    return holder, page, kept, meta, counts, region, offs


def _shard(tensor, dims, dtype, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        tensor,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        layout=layout,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _check(output, payload, holder, page, kept, H, G, topk, emb_dim, label):
    shards = ttnn.get_device_tensors(output)
    assert len(shards) == H * G, f"{label}: {len(shards)} shards for {H * G} chips"
    checked = 0
    for dev, shard in enumerate(shards):
        s, g = dev // G, dev % G
        mask = kept[g, s]
        if not mask.any():
            continue
        got = ttnn.to_torch(shard).reshape(_SEQ_LEN, topk, emb_dim)[mask]
        want = payload[holder[g, s][mask], g, page[g, s][mask]]
        assert torch.equal(
            got.view(torch.int16), want.view(torch.int16)
        ), f"{label}: chip {dev} (pos {s}, group {g}) differs from the combine reference"
        checked += int(mask.sum())
    assert checked > 0, f"{label}: no kept picks; the routing is wrong"
    logger.info(f"{label}: {checked} slots byte-exact")


def _launch(version, buffer, meta, counts, region, offs, geom, sp_axis):
    if version == "v1":
        return ttnn.experimental.deepseek_prefill.combine(
            buffer,
            meta,
            counts,
            region,
            dispatch_group_size=geom["H"],
            experts_per_chip=geom["experts_per_chip"],
            num_experts_per_tok=geom["topk"],
            seq_len_per_chip=_SEQ_LEN,
            cluster_axis=sp_axis,
            num_links=_NUM_LINKS,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=False,
        )
    else:
        return ttnn.experimental.deepseek_prefill.combine_fabric2d(
            buffer,
            meta,
            counts,
            region,
            offs,
            experts_per_chip=geom["experts_per_chip"],
            num_experts_per_tok=geom["topk"],
            seq_len_per_chip=_SEQ_LEN,
            cluster_axis=sp_axis,
            num_links=_NUM_LINKS,
            fwd_reduction=geom["fwd_reduction"],
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def _record(name, outcome, **fields):
    detail = " ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items())
    with open(os.path.join(os.environ["CMB_SWEEP_DIR"], "state.tsv"), "a", encoding="utf-8") as handle:
        handle.write(f"{outcome}\t{name}\t{detail}\n")


def _open_mesh(device_params, payload_size):
    params = get_updated_device_params(dict(device_params))
    params.pop("fabric_router_config", None)
    fabric_config = params.pop("fabric_config")
    set_fabric(
        fabric_config,
        params.pop("reliability_mode", None),
        params.pop("fabric_tensix_config", None),
        params.pop("fabric_manager", None),
        create_fabric_router_config(max_payload_size=payload_size),
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*_MESH), **params), fabric_config


@pytest.mark.timeout(21600)
@pytest.mark.requires_mesh_topology(mesh_shape=_MESH, topology="mesh-8x4")
@pytest.mark.parametrize(
    "device_params", [torus_xy_device_params(trace_region_size=_TRACE_REGION)], indirect=True, ids=["torus-xy-8x4"]
)
def test_ttnn_combine_sweep(device_params):
    failed = []
    for model_name, config_class, test_meshes in (m for m in COMBINE_MODELS if m[0] in _axis("MODELS")):
        config = _model_scaledown(config_class(), test_meshes.full_model_mesh, _MESH, False)
        mesh_device, fabric_config = _open_mesh(
            device_params, max(moe_fabric_payload_size(config), combine_overlap_payload_size(config.EMB_SIZE))
        )
        try:
            cfg = extract_mesh_config(mesh_device)
            H, G = cfg.dispatch_group_size, cfg.num_dispatch_groups
            experts_per_chip, _, capacity, _ = compute_constants(
                _SEQ_LEN,
                config.NUM_ROUTED_EXPERTS,
                config.NUM_EXPERTS_PER_TOKEN,
                mesh_device.get_num_devices(),
                H,
                _CAPACITY_FACTOR,
            )
            geom = {
                "H": H,
                "experts_per_chip": experts_per_chip,
                "topk": config.NUM_EXPERTS_PER_TOKEN,
                "fwd_reduction": config.COMBINE_FWD_REDUCTION,
            }
            payload = torch.randn(H, G, capacity, config.EMB_SIZE, generator=torch.Generator().manual_seed(0)).to(
                torch.bfloat16
            )
            routing = {
                t: _routing(
                    _indices(G, H, config.NUM_EXPERTS_PER_TOKEN, config.NUM_ROUTED_EXPERTS, _TRAFFIC[t]),
                    H,
                    G,
                    config.NUM_ROUTED_EXPERTS,
                    experts_per_chip,
                    config.NUM_EXPERTS_PER_TOKEN,
                    capacity,
                )
                for t in _axis("TRAFFIC")
            }

            for layout_id in _axis("LAYOUTS"):
                tt_buffer = _shard(payload, (0, 1), ttnn.bfloat16, mesh_device, layout=_LAYOUTS[layout_id])
                for traffic_id in _axis("TRAFFIC"):
                    holder, page, kept, meta, counts, region, offs = routing[traffic_id]
                    tt_meta = _shard(meta, (0, 1), ttnn.int32, mesh_device)
                    tt_counts = _shard(counts[:, 0:1, :], (None, 0), ttnn.int32, mesh_device)
                    tt_region = _shard(region[:, 0:1, :], (None, 0), ttnn.int32, mesh_device)
                    tt_offs = _shard(offs, (None, 0), ttnn.int32, mesh_device)
                    for version in _axis("VERSIONS"):
                        name = f"{model_name}-{traffic_id}-{layout_id}-cmb_{version}"
                        logger.info(f"[sweep] === {name} ===")
                        signpost(name)
                        try:
                            dram_before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
                            output = _launch(
                                version, tt_buffer, tt_meta, tt_counts, tt_region, tt_offs, geom, cfg.sp_axis
                            )
                            ttnn.synchronize_device(mesh_device)
                            _check(
                                output,
                                payload,
                                holder,
                                page,
                                kept,
                                H,
                                G,
                                config.NUM_EXPERTS_PER_TOKEN,
                                config.EMB_SIZE,
                                name,
                            )
                            ttnn.deallocate(output)
                            dram_after = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
                            op_alloc_bytes = (
                                dram_after.total_bytes_allocated_per_bank - dram_before.total_bytes_allocated_per_bank
                            ) * dram_after.num_banks

                            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                            traced_output = _launch(
                                version, tt_buffer, tt_meta, tt_counts, tt_region, tt_offs, geom, cfg.sp_axis
                            )
                            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

                            ttnn.synchronize_device(mesh_device)
                            started = time.perf_counter()
                            for _ in range(_REPS):
                                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh_device)
                            traced_us = (time.perf_counter() - started) * 1e6 / _REPS

                            outputs = []
                            ttnn.synchronize_device(mesh_device)
                            started = time.perf_counter()
                            for _ in range(_REPS):
                                outputs.append(
                                    _launch(
                                        version, tt_buffer, tt_meta, tt_counts, tt_region, tt_offs, geom, cfg.sp_axis
                                    )
                                )
                            ttnn.synchronize_device(mesh_device)
                            untraced_us = (time.perf_counter() - started) * 1e6 / _REPS

                            ttnn.release_trace(mesh_device, trace_id)
                            for tensor in [traced_output, *outputs]:
                                ttnn.deallocate(tensor)
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"[sweep] {name} FAILED: {e}")
                            _record(name, "FAIL", error=str(e))
                            failed.append(name)
                            continue
                        logger.info(f"[perf] {name}: traced {traced_us:.0f} untraced {untraced_us:.0f} us/launch")
                        _record(
                            name,
                            "pass",
                            op_alloc_bytes=op_alloc_bytes,
                            untraced_us=untraced_us,
                            traced_us=traced_us,
                            reps=_REPS,
                            seq_len=_SEQ_LEN,
                            fwd_reduction=config.COMBINE_FWD_REDUCTION,
                        )
                    for tensor in (tt_meta, tt_counts, tt_region, tt_offs):
                        ttnn.deallocate(tensor)
                ttnn.deallocate(tt_buffer)
            ttnn.ReadDeviceProfiler(mesh_device)
        finally:
            ttnn.close_mesh_device(mesh_device)
            reset_fabric(fabric_config)
    assert not failed, f"sweep phases failed: {failed}"
