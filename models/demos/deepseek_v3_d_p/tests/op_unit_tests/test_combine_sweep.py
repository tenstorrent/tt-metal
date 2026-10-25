# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-launch cost of combine v1 and combine_fabric2d.

Swept over models, ring sizes, chunk sizes, traffic shapes and input layouts.
"""

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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_prefill_combine import (
    COMBINE_MODELS,
    SINGLE_GLX_AND_PROXY_MESHES,
    _model_scaledown,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    combine_overlap_payload_size,
    compute_constants,
    create_fabric_router_config,
    get_gate_outputs,
    moe_fabric_payload_size,
)

_REPS = int(os.environ.get("CMB_SWEEP_REPS", "10"))
_MESH = (8, 4)
_BASE_SEQ = 640
_CAPACITY_FACTOR = 2
_NUM_LINKS = 2
_TRACE_REGION = 330000
# kimi_k3 is not in COMBINE_MODELS: that list also parametrizes test_prefill_combine, and widening it
# there is a change to that test rather than to this sweep.
_MODELS = COMBINE_MODELS + [("kimi_k3", KimiK3Config, SINGLE_GLX_AND_PROXY_MESHES)]
_TRAFFIC = {"balanced": None, "hot_expert": 0}
_LAYOUTS = {"row_major": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}
_CHUNKS = {"2.5k": 2560, "5k": 5120, "10k": 10240}
# Mesh rows, mesh columns, and which mesh axis the dispatch group rides. This torus closes on both
# axes -- 8 along X, 4 along Y -- and those are the only two rings it has. A 4-chip dispatch group
# is therefore the Y ring, not half of the X ring: carving X in two leaves a line, which is what the
# repo's own 16-chip descriptor records as dim_types [LINE, RING]. So ring4 keeps the full 4 columns
# and takes 4 of the 8 rows, putting the ring on the columns and the dispatch groups on the rows.
_RINGS = {"ring8": (8, 4, 0), "ring4": (4, 4, 1)}


def _axis(name):
    return os.environ[f"CMB_SWEEP_{name}"].split(",")


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
    else:
        others = torch.randint(0, num_routed_experts - 1, (G, H, seq, topk), dtype=torch.int64)
        indices = others + (others >= hot_expert).to(torch.int64)
        indices[..., 0] = hot_expert
        return indices


def _routing(indices, H, G, num_routed_experts, experts_per_chip, seq, topk, capacity, geom):
    """Per chip: where each pick is held, which page it sits on, and the control tables that say so."""
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

    meta = torch.full((H, G, capacity, 3), -1, dtype=torch.int32)
    g_idx, s_idx, t_idx, k_idx = kept.nonzero(as_tuple=True)
    # Field 0 is the source device as a linearized mesh coordinate, so it is row * cols + col on the
    # mesh the op sees. Which of (position, group) is the row follows the ring axis.
    row, col = (s_idx, g_idx) if geom["ring_axis"] == 0 else (g_idx, s_idx)
    coord = row * geom["cols"] + col
    meta[holder[kept], g_idx, page[kept]] = torch.stack([coord, t_idx, k_idx], dim=-1).to(torch.int32)
    return holder, page, kept, meta, counts, region, offs


def _pos_group(dev, geom):
    """A chip's (position in its dispatch group, dispatch group).

    Devices are numbered along mesh rows. When the ring rides the columns the two halves of that
    division swap meaning, so the decomposition follows the ring axis rather than assuming rows.
    """
    a, b = divmod(dev, geom["cols"])
    return (a, b) if geom["ring_axis"] == 0 else (b, a)


def _shard(tensor, dims, dtype, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        tensor,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        layout=layout,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _check(output, payload, holder, page, kept, geom, token_dim, label):
    H, G = geom["H"], geom["G"]
    shards = ttnn.get_device_tensors(output)
    assert len(shards) == H * G, f"{label}: {len(shards)} shards for {H * G} chips"
    checked = 0
    for dev, shard in enumerate(shards):
        s, g = _pos_group(dev, geom)
        mask = kept[g, s]
        if not mask.any():
            continue
        got = ttnn.to_torch(shard).reshape(geom["seq"], geom["topk"], token_dim)[mask]
        want = payload[holder[g, s][mask], g, page[g, s][mask]]
        assert torch.equal(
            got.view(torch.int16), want.view(torch.int16)
        ), f"{label}: chip {dev} (pos {s}, group {g}) differs from the combine reference"
        checked += int(mask.sum())
    assert checked > 0, f"{label}: no kept picks; the routing is wrong"
    logger.info(f"{label}: {checked} slots byte-exact")


def _launch(version, buffer, meta, counts, region, offs, geom):
    if version == "v1":
        return ttnn.experimental.deepseek_prefill.combine(
            buffer,
            meta,
            counts,
            region,
            dispatch_group_size=geom["H"],
            experts_per_chip=geom["experts_per_chip"],
            num_experts_per_tok=geom["topk"],
            seq_len_per_chip=geom["seq"],
            cluster_axis=geom["ring_axis"],
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
            seq_len_per_chip=geom["seq"],
            cluster_axis=geom["ring_axis"],
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
    # Every phase the axes ask for, recorded before any of them runs. A later outcome overwrites its
    # row, so whatever the sweep never reached stays visible as unrun instead of scoring a clean run.
    for planned in (
        f"{m}-{r}-{c}-{t}-{layout}-cmb_{v}"
        for m in _axis("MODELS")
        for r in _axis("RINGS")
        for c in _axis("CHUNKS")
        for t in _axis("TRAFFIC")
        for layout in _axis("LAYOUTS")
        for v in _axis("VERSIONS")
    ):
        _record(planned, "unrun")
    for model_name, config_class, test_meshes in (m for m in _MODELS if m[0] in _axis("MODELS")):
        # Scaled to the full galaxy for every ring size. ring4 is the same model on half the chips,
        # which is twice the experts per chip, and a scaledown would make it a smaller model instead.
        config = _model_scaledown(config_class(), test_meshes.full_model_mesh, _MESH, False)
        # The routed side is what combine carries. K3 runs it in a latent space half the embedding
        # width; tt_moe reads the same attribute the same way.
        token_dim = getattr(config, "ROUTED_EXPERT_HIDDEN_SIZE", config.EMB_SIZE)
        topk = config.NUM_EXPERTS_PER_TOKEN
        payload_size = max(moe_fabric_payload_size(config), combine_overlap_payload_size(token_dim))
        for ring_id in _axis("RINGS"):
            rows, cols, ring_axis = _RINGS[ring_id]
            # Opened fresh per ring, and always the full torus. Always full because TorusXY wraps
            # across all 8 rows, so a bare 4x4 open leaves the boundary routers waiting on partners
            # outside the mesh and bring-up times out on the ethernet handshake. Fresh per ring
            # because a mesh that has already run one dispatch-group geometry corrupts the next one:
            # ring8 then ring4 returns wrong data, ring4 then ring8 hangs. Each is correct alone.
            full_mesh, fabric_config = _open_mesh(device_params, payload_size)
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
                    geom = {
                        "H": H,
                        "G": G,
                        "cols": cols,
                        "ring_axis": ring_axis,
                        # An (H, G, ...) tensor onto the mesh, and a (G, ...) one replicated along
                        # the ring. Both follow the ring axis, so they transpose together with it.
                        "hg_dims": (0, 1) if ring_axis == 0 else (1, 0),
                        "g_dims": (None, 0) if ring_axis == 0 else (0, None),
                        "seq": seq,
                        "topk": topk,
                        "experts_per_chip": experts_per_chip,
                        # The forwarding buffer is the dense size over this factor, and the dense
                        # size is proportional to seq_len_per_chip. Scaling the factor with the
                        # sequence holds the buffer itself fixed, so what the chunk axis varies is
                        # the traffic and not the space the op gets to stage it in.
                        "fwd_reduction": max(1, config.COMBINE_FWD_REDUCTION * seq // _BASE_SEQ),
                    }
                    payload = torch.randn(H, G, capacity, token_dim, generator=torch.Generator().manual_seed(0)).to(
                        torch.bfloat16
                    )
                    routing = {
                        t: _routing(
                            _indices(G, H, seq, topk, config.NUM_ROUTED_EXPERTS, _TRAFFIC[t]),
                            H,
                            G,
                            config.NUM_ROUTED_EXPERTS,
                            experts_per_chip,
                            seq,
                            topk,
                            capacity,
                            geom,
                        )
                        for t in _axis("TRAFFIC")
                    }

                    for layout_id in _axis("LAYOUTS"):
                        tt_buffer = _shard(
                            payload, geom["hg_dims"], ttnn.bfloat16, mesh_device, layout=_LAYOUTS[layout_id]
                        )
                        for traffic_id in _axis("TRAFFIC"):
                            holder, page, kept, meta, counts, region, offs = routing[traffic_id]
                            tt_meta = _shard(meta, geom["hg_dims"], ttnn.int32, mesh_device)
                            tt_counts = _shard(counts[:, 0:1, :], geom["g_dims"], ttnn.int32, mesh_device)
                            tt_region = _shard(region[:, 0:1, :], geom["g_dims"], ttnn.int32, mesh_device)
                            tt_offs = _shard(offs, geom["g_dims"], ttnn.int32, mesh_device)
                            for version in _axis("VERSIONS"):
                                name = f"{model_name}-{ring_id}-{chunk_id}-{traffic_id}-{layout_id}-cmb_{version}"
                                logger.info(f"[sweep] === {name} ===")
                                signpost(name)
                                try:
                                    dram_before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
                                    output = _launch(version, tt_buffer, tt_meta, tt_counts, tt_region, tt_offs, geom)
                                    ttnn.synchronize_device(mesh_device)
                                    _check(output, payload, holder, page, kept, geom, token_dim, name)
                                    ttnn.deallocate(output)
                                    dram_after = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
                                    op_alloc_bytes = (
                                        dram_after.total_bytes_allocated_per_bank
                                        - dram_before.total_bytes_allocated_per_bank
                                    ) * dram_after.num_banks

                                    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                                    traced_output = _launch(
                                        version, tt_buffer, tt_meta, tt_counts, tt_region, tt_offs, geom
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
                                            _launch(version, tt_buffer, tt_meta, tt_counts, tt_region, tt_offs, geom)
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
                                logger.info(
                                    f"[perf] {name}: traced {traced_us:.0f} untraced {untraced_us:.0f} us/launch"
                                )
                                _record(
                                    name,
                                    "pass",
                                    op_alloc_bytes=op_alloc_bytes,
                                    untraced_us=untraced_us,
                                    traced_us=traced_us,
                                    reps=_REPS,
                                    seq_len=seq,
                                    chunk=_CHUNKS[chunk_id],
                                    chips=mesh_device.get_num_devices(),
                                    experts_per_chip=experts_per_chip,
                                    fwd_reduction=geom["fwd_reduction"],
                                )
                            for tensor in (tt_meta, tt_counts, tt_region, tt_offs):
                                ttnn.deallocate(tensor)
                        ttnn.deallocate(tt_buffer)
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
