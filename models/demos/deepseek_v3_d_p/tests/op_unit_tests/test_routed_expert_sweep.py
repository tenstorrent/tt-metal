# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-launch cost of the hybrid routed expert, over models, traffic shapes and chunk sizes."""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from conftest import reset_fabric, set_fabric
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_prefill_combine import (
    COMBINE_MODELS,
    SINGLE_GLX_AND_PROXY_MESHES,
    _model_scaledown,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    get_gate_outputs,
    moe_fabric_payload_size,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME, TtRoutedExpert
from tests.scripts.common import get_updated_device_params

_REPS = int(os.environ.get("RE_SWEEP_REPS", "10"))
_MESH = (8, 4)
_CAPACITY_FACTOR = 2
_TRACE_REGION = 330000
_PCC_ROWS = 16  # rows graded per chip; the FFN reference is fp32 on host and the full set is hours
# test_ttnn_moe grades routed output at 0.96: the weights are bfloat4_b against an fp32 reference.
_PCC_FLOOR = 0.96
_MODELS = COMBINE_MODELS + [("kimi_k3", KimiK3Config, SINGLE_GLX_AND_PROXY_MESHES)]
_TRAFFIC = {"balanced": None, "hot_expert": 0}
_CHUNKS = {"2.5k": 2560, "5k": 5120, "10k": 10240}
# Mesh rows, mesh columns, and which mesh axis the dispatch group rides. This op takes no cluster
# axis of its own -- the ring only reaches it through the shape of the work. Half the chips means
# twice the experts each and twice the tokens each, which is the whole point of measuring it.
# ring4 is the Y ring rather than half the X ring, for the reason the combine sweep records.
_RINGS = {"ring8": (8, 4, 0), "ring4": (4, 4, 1)}


def _axis(name):
    return os.environ[f"RE_SWEEP_{name}"].split(",")


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


def _routing(indices, H, G, num_routed_experts, experts_per_chip, seq, topk):
    table = _dispatch_table(num_routed_experts, H, G)
    counts = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    region = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    for g in range(G):
        _, c, r, _ = get_gate_outputs(
            indices[g], H, num_routed_experts, experts_per_chip, seq, topk, expert_dispatch_table=table[g : g + 1]
        )
        counts[g], region[g] = c[0].to(torch.int32), r[0].to(torch.int32)
    return counts, region


def _shard(tensor, dims, dtype, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        tensor,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        layout=layout,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _experts(mesh_device, config, experts_per_chip, H, G, token_dim, max_per_expert, geom):
    """One random weight set every expert shares: the reference below uses the same one, and a set
    per expert would be tens of GB of host conversion."""
    hidden = config.MOE_INTERMEDIATE_SIZE
    generator = torch.Generator().manual_seed(2)
    weights = {
        "gate_proj": torch.randn(hidden, token_dim, generator=generator) * 0.02,
        "up_proj": torch.randn(hidden, token_dim, generator=generator) * 0.02,
        "down_proj": torch.randn(token_dim, hidden, generator=generator) * 0.02,
    }
    idx_table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=experts_per_chip, dispatch_group_size=H, num_dispatch_groups=G
    )
    idx_slice = ttnn.squeeze(ttnn.squeeze(_shard(idx_table, geom["gh_dims"], ttnn.uint32, mesh_device), 0), 0)
    activation = getattr(config, "ROUTED_EXPERT_ACTIVATION", "silu")
    module = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=idx_slice,
        emb_dim=token_dim,
        hidden_dim=hidden,
        max_tokens=max_per_expert,
        torch_weights=[weights] * (H * G * experts_per_chip),
        activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[activation],
    )
    return module, idx_slice, weights, activation


def _reference(tt_pages, counts, region, weights, activation, config, experts_per_chip, H, G, token_dim, geom):
    """The rows the op should produce, taken before it runs -- it writes over its input."""
    expert = TorchExpert(
        token_dim,
        config.MOE_INTERMEDIATE_SIZE,
        torch_weights=weights,
        activation=activation,
        situ_beta=getattr(config, "ACTIVATION_SITU_BETA", 1.0),
        situ_linear_beta=getattr(config, "ACTIVATION_SITU_LINEAR_BETA", None),
    )
    wanted = []
    for dev, shard in enumerate(ttnn.get_device_tensors(tt_pages)):
        h, g = _pos_group(dev, geom)
        src = ttnn.to_torch(shard).reshape(-1, token_dim)
        for local in range(experts_per_chip):
            ge = ExpertMapping.get_global_expert_idx(
                group=g,
                chip=h,
                local_expert=local,
                experts_per_chip=experts_per_chip,
                dispatch_group_size=H,
                num_dispatch_groups=G,
                is_col_major=True,
            )
            rows = min(int(counts[g, h, ge]), _PCC_ROWS)
            if rows <= 0:
                continue
            start = int(region[g, h, ge])
            with torch.no_grad():
                wanted.append((dev, start, rows, expert(src[start : start + rows].float())))
            break
    return wanted


def _check(out, wanted, G, token_dim, label):
    worst, graded = 1.0, 0
    shards = ttnn.get_device_tensors(out)
    for dev, start, rows, want in wanted:
        got = ttnn.to_torch(shards[dev]).reshape(-1, token_dim)[start : start + rows]
        pcc = torch.corrcoef(torch.stack([got.float().flatten(), want.flatten()]))[0, 1].item()
        # min() would keep 1.0 against a NaN, so a NaN output would report a perfect score
        assert pcc == pcc, f"{label}: chip {dev} produced NaN"
        worst = min(worst, pcc)
        graded += rows
    assert graded > 0, f"{label}: no dispatched rows to grade; the routing is wrong"
    assert worst > _PCC_FLOOR, f"{label}: PCC {worst:.5f} below {_PCC_FLOOR}"
    logger.info(f"{label}: {graded} rows graded, worst PCC {worst:.5f}")


def _record(name, outcome, **fields):
    detail = " ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items())
    with open(os.path.join(os.environ["RE_SWEEP_DIR"], "state.tsv"), "a", encoding="utf-8") as handle:
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
def test_routed_expert_sweep(device_params):
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
        # Scaled to the full galaxy for every ring size. ring4 is the same model on half the chips,
        # which is twice the experts per chip, and a scaledown would make it a smaller model instead.
        config = _model_scaledown(config_class(), test_meshes.full_model_mesh, _MESH, False)
        token_dim = getattr(config, "ROUTED_EXPERT_HIDDEN_SIZE", config.EMB_SIZE)
        topk = config.NUM_EXPERTS_PER_TOKEN
        threshold = getattr(config, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", 0)
        for ring_id in _axis("RINGS"):
            rows, cols, ring_axis = _RINGS[ring_id]
            # Opened fresh per ring, and always the full torus. Always full because TorusXY wraps
            # across all 8 rows, so a bare 4x4 open leaves the boundary routers waiting on partners
            # outside the mesh and bring-up times out. Fresh per ring because a mesh that has
            # already run one dispatch-group geometry corrupts the next one, as the combine sweep
            # records: wrong data one way round, a hang the other, each correct alone.
            full_mesh, fabric_config = _open_mesh(device_params, moe_fabric_payload_size(config))
            mesh_device = full_mesh
            try:
                if (rows, cols) != _MESH:
                    mesh_device = full_mesh.create_submesh(ttnn.MeshShape(rows, cols), ttnn.MeshCoordinate(0, 0))
                H, G = (rows, cols) if ring_axis == 0 else (cols, rows)
                # An (H, G, ...) tensor onto the mesh, and a (G, H, ...) one, which is how the
                # counts, region and expert-id tables are laid out. Both follow the ring axis.
                geom = {
                    "cols": cols,
                    "ring_axis": ring_axis,
                    "hg_dims": (0, 1) if ring_axis == 0 else (1, 0),
                    "gh_dims": (1, 0) if ring_axis == 0 else (0, 1),
                }
                for chunk_id in _axis("CHUNKS"):
                    seq = _CHUNKS[chunk_id] // H
                    experts_per_chip, _, capacity, max_per_expert = compute_constants(
                        seq, config.NUM_ROUTED_EXPERTS, topk, mesh_device.get_num_devices(), H, _CAPACITY_FACTOR
                    )
                    module, idx_slice, weights, activation = _experts(
                        mesh_device, config, experts_per_chip, H, G, token_dim, max_per_expert, geom
                    )
                    pages = torch.randn(
                        H, G, capacity, token_dim, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1)
                    )
                    for traffic_id in _axis("TRAFFIC"):
                        # Uploaded per phase: the op writes over its input, so a phase that inherited
                        # the previous one's buffer would be measured on 22 rounds of its own output.
                        tt_pages = _shard(pages, geom["hg_dims"], ttnn.bfloat8_b, mesh_device, layout=ttnn.TILE_LAYOUT)
                        name = f"{model_name}-{ring_id}-{chunk_id}-{traffic_id}"
                        logger.info(f"[sweep] === {name} ===")
                        signpost(name)
                        counts, region = _routing(
                            _indices(G, H, seq, topk, config.NUM_ROUTED_EXPERTS, _TRAFFIC[traffic_id]),
                            H,
                            G,
                            config.NUM_ROUTED_EXPERTS,
                            experts_per_chip,
                            seq,
                            topk,
                        )
                        # Per chip, not replicated: a chip's region offsets are its own. Sharded as
                        # get_expert_token_counts_mesh_mapper does, then squeezed to the 2D the op takes.
                        tt_counts = ttnn.squeeze(_shard(counts, geom["gh_dims"], ttnn.uint32, mesh_device), 0)
                        tt_region = ttnn.squeeze(_shard(region, geom["gh_dims"], ttnn.uint32, mesh_device), 0)

                        def launch():
                            return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
                                tt_pages,
                                tt_region,
                                tt_counts,
                                idx_slice,
                                module.gate_projs,
                                module.up_projs,
                                module.down_projs,
                                max_per_expert,
                                hybrid_token_threshold=threshold,
                                compute_kernel_config=module.compute_kernel_config,
                                activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[activation],
                            )

                        try:
                            wanted = _reference(
                                tt_pages,
                                counts,
                                region,
                                weights,
                                activation,
                                config,
                                experts_per_chip,
                                H,
                                G,
                                token_dim,
                                geom,
                            )
                            dram_before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
                            first = launch()
                            ttnn.synchronize_device(mesh_device)
                            _check(first, wanted, G, token_dim, name)
                            # The op writes into its input unless the hybrid split gave it its own buffer;
                            # freeing an aliased output would take tt_pages with it.
                            aliased = first.buffer_address() == tt_pages.buffer_address()
                            if not aliased:
                                ttnn.deallocate(first)
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
                            if not aliased:
                                for tensor in [traced, *held]:
                                    ttnn.deallocate(tensor)
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"[sweep] {name} FAILED: {e}")
                            _record(name, "FAIL", error=str(e))
                            failed.append(name)
                            continue
                        finally:
                            for tensor in (tt_counts, tt_region, tt_pages):
                                ttnn.deallocate(tensor)
                            # Per phase, not per model: six phases of markers overflow the host profiler
                            # buffer, and the dropped start markers abort the drain at mesh close.
                            ttnn.ReadDeviceProfiler(mesh_device)
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
                            threshold=threshold,
                        )
            finally:
                # The child closes first: a submesh and its parent must not both hold a command
                # queue on the same physical device at close, and the parent is the one holding it.
                if mesh_device is not full_mesh:
                    ttnn.close_mesh_device(mesh_device)
                ttnn.close_mesh_device(full_mesh)
                reset_fabric(fabric_config)
    assert not failed, f"sweep phases failed: {failed}"
