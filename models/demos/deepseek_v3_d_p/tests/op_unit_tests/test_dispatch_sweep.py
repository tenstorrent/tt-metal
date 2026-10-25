# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-launch cost of dispatch_fabric2d, over models, ring sizes, chunk sizes and traffic shapes."""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from conftest import reset_fabric, set_fabric
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_prefill_combine import (
    COMBINE_MODELS,
    SINGLE_GLX_AND_PROXY_MESHES,
    _model_scaledown,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    create_fabric_router_config,
    get_gate_outputs,
    moe_fabric_payload_size,
)
from tests.scripts.common import get_updated_device_params

_REPS = int(os.environ.get("DSP_SWEEP_REPS", "10"))
_MESH = (8, 4)
_CAPACITY_FACTOR = 2
_NUM_LINKS = 2
_TRACE_REGION = 330000
_ROUTING_TAIL = 64
_MODELS = COMBINE_MODELS + [("kimi_k3", KimiK3Config, SINGLE_GLX_AND_PROXY_MESHES)]
_TRAFFIC = {"balanced": None, "hot_expert": 0}
_CHUNKS = {"2.5k": 2560, "5k": 5120, "10k": 10240}
# Mesh rows, mesh columns, and which mesh axis the dispatch group rides. This torus closes on both
# axes -- 8 along X, 4 along Y -- and those are the only two rings it has. A 4-chip dispatch group
# is therefore the Y ring, not half of the X ring: carving X in two leaves a line, which is what the
# repo's own 16-chip descriptor records as dim_types [LINE, RING]. So ring4 keeps the full 4 columns
# and takes 4 of the 8 rows, putting the ring on the columns and the dispatch groups on the rows.
_RINGS = {"ring8": (8, 4, 0), "ring4": (4, 4, 1)}


def _axis(name):
    return os.environ[f"DSP_SWEEP_{name}"].split(",")


def _pos_group(dev, geom):
    """A chip's (position in its dispatch group, dispatch group).

    Devices are numbered along mesh rows. When the ring rides the columns the two halves of that
    division swap meaning, so the decomposition follows the ring axis rather than assuming rows.
    """
    a, b = divmod(dev, geom["cols"])
    return (a, b) if geom["ring_axis"] == 0 else (b, a)


def _dispatch_table(num_routed_experts, H, G):
    experts_per_group = num_routed_experts // G
    table = torch.full((G, num_routed_experts + 1), -1, dtype=torch.int32)
    for g in range(G):
        for e in range(experts_per_group):
            table[g, g * experts_per_group + e] = e // (experts_per_group // H)
    return table


def _indices(G, H, seq, topk, num_routed_experts, hot_expert):
    torch.manual_seed(42)
    if hot_expert is None:
        return torch.randint(0, num_routed_experts, (G, H, seq, topk), dtype=torch.int64)
    others = torch.randint(0, num_routed_experts - 1, (G, H, seq, topk), dtype=torch.int64)
    indices = others + (others >= hot_expert).to(torch.int64)
    indices[..., 0] = hot_expert
    return indices


def _routing(indices, H, G, num_routed_experts, experts_per_chip, seq, topk, capacity):
    table = _dispatch_table(num_routed_experts, H, G)
    offs = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    counts = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    region = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g], H, num_routed_experts, experts_per_chip, seq, topk, expert_dispatch_table=table[g : g + 1]
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)

    holder = torch.full((G, H, seq, topk), -1, dtype=torch.int64)
    page = torch.full((G, H, seq, topk), -1, dtype=torch.int64)
    for g in range(G):
        for s in range(H):
            flat = indices[g, s].reshape(-1)
            rank = (torch.nn.functional.one_hot(flat, num_routed_experts).cumsum(0) - 1).gather(1, flat.unsqueeze(1))
            pos = table[g, flat].to(torch.int64)
            holder[g, s] = torch.where(pos != -1, pos, -1).reshape(seq, topk)
            page[g, s] = torch.where(pos != -1, offs[g, s, flat].to(torch.int64) + rank.squeeze(1), -1).reshape(
                seq, topk
            )
    kept = (holder >= 0) & (page < capacity)
    return table, offs, counts, region, holder, page, kept


def _shard(tensor, dims, dtype, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        tensor,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        layout=layout,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _check(payload, tokens, holder, page, kept, geom, token_dim, label):
    """A chip's payload holds the picks routed TO it, so each device is selected by holder."""
    checked = 0
    for dev, shard in enumerate(ttnn.get_device_tensors(payload)):
        h, g = _pos_group(dev, geom)
        sel = (holder[g] == h) & kept[g]
        if not sel.any():
            continue
        got = ttnn.to_torch(shard).reshape(-1, token_dim)
        s_idx, t_idx, _ = sel.nonzero(as_tuple=True)
        assert torch.equal(
            got[page[g][sel]].view(torch.int16), tokens[s_idx, g, t_idx].view(torch.int16)
        ), f"{label}: chip {dev} (pos {h}, group {g}) differs from the dispatch reference"
        checked += int(sel.sum())
    assert checked > 0, f"{label}: no kept picks; the routing is wrong"
    logger.info(f"{label}: {checked} pages byte-exact")


def _record(name, outcome, **fields):
    detail = " ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items())
    with open(os.path.join(os.environ["DSP_SWEEP_DIR"], "state.tsv"), "a", encoding="utf-8") as handle:
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
def test_dispatch_sweep(device_params):
    failed = []
    # Every phase the axes ask for, recorded before any of them runs. A later outcome overwrites its
    # row, so whatever the sweep never reached stays visible as unrun instead of scoring a clean run.
    for planned in (
        f"{m}-{r}-{c}-{t}"
        for m in _axis("MODELS")
        for r in _axis("RINGS")
        for c in _axis("CHUNKS")
        for t in _axis("TRAFFIC")
    ):
        _record(planned, "unrun")
    for model_name, config_class, test_meshes in (m for m in _MODELS if m[0] in _axis("MODELS")):
        config = _model_scaledown(config_class(), test_meshes.full_model_mesh, _MESH, False)
        token_dim = getattr(config, "ROUTED_EXPERT_HIDDEN_SIZE", config.EMB_SIZE)
        topk = config.NUM_EXPERTS_PER_TOKEN
        # dispatch sends a whole token in one packet, so the mesh needs room for it plus the tail.
        payload = max(moe_fabric_payload_size(config), 2 * token_dim + _ROUTING_TAIL)
        for ring_id in _axis("RINGS"):
            rows, cols, ring_axis = _RINGS[ring_id]
            # Opened fresh per ring, and always the full torus. Always full because TorusXY wraps
            # across all 8 rows, so a bare 4x4 open leaves the boundary routers waiting on partners
            # outside the mesh and bring-up times out on the ethernet handshake. Fresh per ring
            # because a mesh that has already run one dispatch-group geometry corrupts the next one:
            # ring8 then ring4 returns wrong data, ring4 then ring8 hangs. Each is correct alone.
            full_mesh, fabric_config = _open_mesh(device_params, payload)
            mesh_device = full_mesh
            try:
                if (rows, cols) != _MESH:
                    mesh_device = full_mesh.create_submesh(ttnn.MeshShape(rows, cols), ttnn.MeshCoordinate(0, 0))
                H, G = (rows, cols) if ring_axis == 0 else (cols, rows)
                for chunk_id in _axis("CHUNKS"):
                    seq = _CHUNKS[chunk_id] // H
                    experts_per_chip, _, capacity, _ = compute_constants(
                        seq, config.NUM_ROUTED_EXPERTS, topk, mesh_device.get_num_devices(), H, _CAPACITY_FACTOR
                    )
                    # An (H, G, ...) tensor onto the mesh, and a (G, ...) one replicated along the
                    # ring. Both follow the ring axis, so they transpose together with it.
                    geom = {
                        "cols": cols,
                        "ring_axis": ring_axis,
                        "hg_dims": (0, 1) if ring_axis == 0 else (1, 0),
                        "g_dims": (None, 0) if ring_axis == 0 else (0, None),
                    }
                    tokens = torch.randn(
                        H, G, seq, token_dim, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
                    )
                    tt_tokens = _shard(tokens, geom["hg_dims"], ttnn.bfloat16, mesh_device, layout=ttnn.TILE_LAYOUT)
                    for traffic_id in _axis("TRAFFIC"):
                        name = f"{model_name}-{ring_id}-{chunk_id}-{traffic_id}"
                        logger.info(f"[sweep] === {name} ===")
                        signpost(name)
                        indices = _indices(G, H, seq, topk, config.NUM_ROUTED_EXPERTS, _TRAFFIC[traffic_id])
                        table, offs, counts, region, holder, page, kept = _routing(
                            indices, H, G, config.NUM_ROUTED_EXPERTS, experts_per_chip, seq, topk, capacity
                        )
                        tt = [
                            _shard(
                                indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16),
                                geom["hg_dims"],
                                ttnn.uint16,
                                mesh_device,
                            ),
                            _shard(offs, geom["g_dims"], ttnn.int32, mesh_device),
                            _shard(table.unsqueeze(1), geom["g_dims"], ttnn.int32, mesh_device),
                            _shard(counts[:, 0:1, :], geom["g_dims"], ttnn.int32, mesh_device),
                            _shard(region[:, 0:1, :], geom["g_dims"], ttnn.int32, mesh_device),
                        ]

                        def launch():
                            return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
                                tt_tokens,
                                *tt,
                                experts_per_chip=experts_per_chip,
                                num_routed_experts=config.NUM_ROUTED_EXPERTS,
                                num_experts_per_tok=topk,
                                metadata_len=3,
                                max_dispatch_buffer_token_size=capacity,
                                seq_len_per_chip=seq,
                                cluster_axis=ring_axis,
                                num_links=_NUM_LINKS,
                                topology=ttnn.Topology.Ring,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )

                        try:
                            dram_before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
                            first = launch()
                            ttnn.synchronize_device(mesh_device)
                            _check(first[0], tokens, holder, page, kept, geom, token_dim, name)
                            for tensor in first:
                                ttnn.deallocate(tensor)
                            dram_after = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)

                            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                            traced = launch()
                            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

                            ttnn.synchronize_device(mesh_device)
                            started = time.perf_counter()
                            for _ in range(_REPS):
                                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh_device)
                            traced_us = (time.perf_counter() - started) * 1e6 / _REPS

                            held = []
                            ttnn.synchronize_device(mesh_device)
                            started = time.perf_counter()
                            for _ in range(_REPS):
                                held.append(launch())
                            ttnn.synchronize_device(mesh_device)
                            untraced_us = (time.perf_counter() - started) * 1e6 / _REPS

                            ttnn.release_trace(mesh_device, trace_id)
                            for out in [traced, *held]:
                                for tensor in out:
                                    ttnn.deallocate(tensor)
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"[sweep] {name} FAILED: {e}")
                            _record(name, "FAIL", error=str(e))
                            failed.append(name)
                            continue
                        finally:
                            for tensor in tt:
                                ttnn.deallocate(tensor)
                        logger.info(f"[perf] {name}: traced {traced_us:.0f} untraced {untraced_us:.0f} us/launch")
                        _record(
                            name,
                            "pass",
                            op_alloc_bytes=(
                                dram_after.total_bytes_allocated_per_bank - dram_before.total_bytes_allocated_per_bank
                            )
                            * dram_after.num_banks,
                            traced_us=traced_us,
                            untraced_us=untraced_us,
                            reps=_REPS,
                            seq_len=seq,
                            chunk=_CHUNKS[chunk_id],
                            chips=mesh_device.get_num_devices(),
                            experts_per_chip=experts_per_chip,
                            payload=payload,
                        )
                    ttnn.deallocate(tt_tokens)
                    # Per chunk, not per model: a model's worth of markers overflows the host
                    # profiler buffer, and the dropped start markers abort the drain at mesh close.
                    ttnn.ReadDeviceProfiler(mesh_device)
            finally:
                # The child closes first: a submesh and its parent must not both hold a command
                # queue on the same physical device at close, and the parent is the one holding it.
                if mesh_device is not full_mesh:
                    ttnn.close_mesh_device(mesh_device)
                ttnn.close_mesh_device(full_mesh)
                reset_fabric(fabric_config)
    assert not failed, f"sweep phases failed: {failed}"
