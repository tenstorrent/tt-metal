# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The flat routed expert overlapped with combine_fabric2d (ttnn.bringup.flat_combine_overlap, initial version), on
the same seq-640 layer cases as test_hybrid_routed_expert_combine.py (8 x 1 ring; GLM 5.3 / Kimi K2.7 scaled to it).

By default (FLAT_CMB_YRM=1) the flat expert writes row-major bf16 y, which combine's readers read directly (no
untilizer cores): the flat expert is planned below combine's sender row 0 (MIMO_FL_ROWS=1,9, set here before any plan
is built). FLAT_CMB_YRM=0: bfp8 y tiles, combine untilizes them on rows 0-1, the flat expert on rows 2-9. The
back-to-back baseline runs the same flat expert. Accuracy: the overlap's combine output against flat then
combine_fabric2d back to back (bit-exact: combine reads the same rows either way) and against the PyTorch reference
(per-slot PCC, as the hybrid's test). Perf: every program replayed from a trace.

Galaxy (BH 8 x 4, torus both ways): the `8x1-galaxy` cases open the whole mesh on FABRIC_2D_TORUS_XY and run on one
(8, 1) column, comparable to the LoudBox 8 x 1 cases. Do NOT set TT_MESH_GRAPH_DESC_PATH (the LoudBox ring descriptor):
  scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_flat_combine_overlap.py \
      -k "8x1-galaxy and not perf"                                 # accuracy (random weights, poisoned y)
  scripts/run_safe_pytest.sh --run-all <same> -k "8x1-galaxy and perf"            # serial + overlap, fake weights
  FLAT_CMB_NO_OVERLAP=1 <same> -k "8x1-galaxy and perf"   # serial baseline only (flat on its full-grid default plan)
  FLAT_CMB_ONLY=overlap <same> -k "8x1-galaxy and perf"   # overlap only (sweeps)
and the PR's hybrid on the same column: test_hybrid_routed_expert_combine.py -k "8x1-galaxy"."""

import os

_YRM = os.environ.get("FLAT_CMB_YRM", "1") == "1"
# FLAT_CMB_GRID=12x9: dispatch on a row (fabric Tensix mux), a 12 x 9 compute grid: the trace-capable LoudBox proxy for
# a Galaxy chip's 12-column grid (default 11 x 10, dispatch on a column)
_GRID = os.environ.get("FLAT_CMB_GRID", "11x10")
assert _GRID in ("11x10", "12x9"), _GRID
_LAST_ROW = 8 if _GRID == "12x9" else 9
if not os.environ.get("FLAT_CMB_NO_OVERLAP"):  # (serial: the flat expert's own default plan on the whole grid)
    os.environ.setdefault("MIMO_FL_ROWS", f"{1 if _YRM else 2},{_LAST_ROW}")

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tests.op_unit_tests import test_hybrid_routed_expert_combine as hyb
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import validate_combine_output
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

_MESH = (8, 1)
_FABRIC = ttnn.FabricConfig.FABRIC_2D_TORUS_Y
_PERF_ITERS = int(os.environ.get("FLAT_CMB_ITERS", "20"))


def _device_params():
    params = hyb._device_params(_FABRIC)
    if _GRID == "12x9":
        params = dict(params)
        params["fabric_tensix_config"] = ttnn.FabricTensixConfig.MUX
        params["dispatch_core_config"] = ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW, fabric_tensix_config=ttnn.FabricTensixConfig.MUX
        )
    return params


_GALAXY = (8, 4)


def _params():
    out = []
    for model_id in ("kimi-k27", "glm-53"):
        for threshold_id in ("balanced", "hot-expert"):
            out.append(
                pytest.param(
                    _MESH,
                    _device_params(),
                    threshold_id,
                    model_id,
                    "8x1",
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=_MESH, topology="ring"),
                    id=f"{model_id}-8x1-{threshold_id}",
                )
            )
    # The (8, 1) ring on a Galaxy (as test_hybrid_routed_expert_combine.py's 8x1-galaxy): an (8, 1) fabric cannot come
    # up alone (every router handshakes with a live partner), so the whole (8, 4) mesh opens on its native torus and
    # the ops run on one column: 8 chips, one dispatch group, the same model scaling and routing as the LoudBox 8 x 1
    # (no mesh graph descriptor needed). Grid 12 x 10: the flat expert's default rows 1-9 plan is np 1, 64 gu + 24 down.
    for model_id in ("kimi-k27", "glm-53"):
        for threshold_id in ("balanced", "hot-expert"):
            out.append(
                pytest.param(
                    _GALAXY,
                    hyb._device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
                    threshold_id,
                    model_id,
                    "8x1-galaxy",
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=_GALAXY, topology="mesh-8x4"),
                    id=f"{model_id}-8x1-galaxy-{threshold_id}",
                )
            )
    return out


def _mesh_for(mesh_device, variant):
    return mesh_device.create_submesh(ttnn.MeshShape(8, 1)) if variant == "8x1-galaxy" else mesh_device


def _build_case(mesh_device, device_params, threshold_id, model_id, fake_weights=False):
    torch.manual_seed(42)
    model = hyb._scaled_model(tuple(mesh_device.shape), model_id)
    H, I = model.EMB_SIZE, model.MOE_INTERMEDIATE_SIZE
    num_routed_experts, num_experts_per_tok, num_links = model.NUM_ROUTED_EXPERTS, model.NUM_EXPERTS_PER_TOKEN, 2
    seq = hyb._SEQ_LEN_PER_CHIP
    mc = extract_mesh_config(mesh_device)
    dg_size, n_dg, n_dev = mc.dispatch_group_size, mc.num_dispatch_groups, mesh_device.get_num_devices()
    experts_per_chip, metadata_len, max_buf, max_per_expert = compute_constants(
        seq, num_routed_experts, num_experts_per_tok, n_dev, dg_size, hyb._CAPACITY_FACTOR
    )
    x, weights, indices = initialize_test_inputs(
        dg_size, seq, H, num_routed_experts, num_experts_per_tok, max_per_expert, num_dispatch_groups=n_dg
    )
    idx_table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=experts_per_chip, dispatch_group_size=dg_size, num_dispatch_groups=n_dg
    )
    group_counts = hyb._CAPTURED_COUNTS[model_id][threshold_id]
    indices = hyb._indices_from_counts(group_counts, dg_size, seq, num_experts_per_tok)
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts, dispatch_group_size=dg_size, num_dispatch_groups=n_dg
    )
    expert_offsets, counts, region_offsets, _ = get_gate_outputs(
        indices,
        dg_size,
        num_routed_experts,
        experts_per_chip,
        seq,
        num_experts_per_tok,
        expert_dispatch_table=dispatch_table,
    )
    dispatched, metadata = TorchDispatchModule(
        dispatch_group_size=dg_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_per_expert,
        max_dispatch_buffer_token_size=max_buf,
        seq_len_per_chip=seq,
        emb_dim=H,
        num_dispatch_groups=n_dg,
        expert_dispatch_table=dispatch_table,
    )(x, weights, indices, expert_offsets)

    ep = get_ep_mesh_mapper(mesh_device)
    cm = get_expert_token_counts_mesh_mapper(mesh_device)
    tt_x = ttnn.from_torch(
        dispatched, mesh_mapper=ep, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_meta = hyb._int_tensor(metadata, mesh_device, ep)
    tt_counts = ttnn.squeeze(hyb._int_tensor(counts, mesh_device, cm, dtype=ttnn.uint32), 0)
    tt_regions = ttnn.squeeze(hyb._int_tensor(region_offsets, mesh_device, cm, dtype=ttnn.uint32), 0)
    full2d = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0))
    tt_offsets = hyb._int_tensor(expert_offsets, mesh_device, full2d, dtype=ttnn.uint32)
    tt_idx_slice = ttnn.squeeze(ttnn.squeeze(hyb._int_tensor(idx_table, mesh_device, ep, dtype=ttnn.uint32), 0), 0)
    tt_idx_full = hyb._int_tensor(idx_table, mesh_device, full2d, dtype=ttnn.uint32)
    gids = [ttnn.to_torch(t).flatten().tolist() for t in ttnn.get_device_tensors(tt_idx_slice)]

    ew = {
        "gate_proj": torch.randn(I, H) * 0.02,
        "up_proj": torch.randn(I, H) * 0.02,
        "down_proj": torch.randn(H, I) * 0.02,
    }
    w = (ew["gate_proj"].T.contiguous(), ew["up_proj"].T.contiguous(), ew["down_proj"].T.contiguous())
    flat = FlatRoutedExpert(
        mesh_device,
        "fake" if fake_weights else [[w] * experts_per_chip for _ in range(n_dev)],
        m=max_per_expert,
        H=H,
        I=I,
        gids=gids,
        n_global=num_routed_experts,
        act="silu",
        pin=int(os.environ.get("FLAT_CMB_PIN", "1")),
    )
    topology = per_axis_topology(device_params["fabric_config"])[0]

    def flat_solo():
        return flat(tt_x, tt_counts, tt_regions, y_row_major=_YRM)

    def combine(y_bf16):
        return ttnn.experimental.deepseek_prefill.combine_fabric2d(
            y_bf16,
            tt_meta,
            tt_counts,
            tt_regions,
            tt_offsets,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq,
            cluster_axis=mc.sp_axis,
            num_links=num_links,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    sems = []

    def poisoned_y():
        # a fresh y full of a value no row of the expert produces: a combine read that runs ahead of the flat expert's
        # write then shows up as a mismatch instead of returning a previous call's (correct) rows
        rows, h = list(tt_x.shape)[-2], list(tt_x.shape)[-1]
        if _YRM:
            return ttnn.from_torch(
                torch.full((rows, h), 7777.0),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        return ttnn.from_torch(
            torch.full((rows, h), 7777.0),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def overlapped(y_out=None):
        if not sems:
            grid = mesh_device.compute_with_storage_grid_size()
            allc = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            sems.extend(ttnn.create_global_semaphore(mesh_device, allc, 0) for _ in range(3))
            ttnn.synchronize_device(mesh_device)
        return ttnn.bringup.flat_combine_overlap(
            tt_x,
            tt_counts,
            tt_regions,
            flat.gidx,
            flat.w_gu,
            flat.w_d,
            flat.w_rd,
            flat.done,
            I,
            max_per_expert,
            tt_meta,
            tt_offsets,
            tt_idx_full,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq,
            combine_axis=mc.sp_axis,
            combine_num_links=num_links,
            fwd_arrived_semaphore=sems[0],
            final_arrived_semaphore=sems[1],
            expert_go_semaphore=sems[2],
            activation=flat.act,
            pin=flat.pin,
            y_row_major=_YRM,
            y_out=y_out,
        )

    ref = []

    def torch_reference():
        if ref:
            return ref[0]
        expert = TorchExpert(H, I, torch_weights=ew)
        with torch.no_grad():
            for g in range(n_dg):
                for chip in range(dg_size):
                    for le in range(experts_per_chip):
                        ge = ExpertMapping.get_global_expert_idx(
                            group=g,
                            chip=chip,
                            local_expert=le,
                            experts_per_chip=experts_per_chip,
                            dispatch_group_size=dg_size,
                            num_dispatch_groups=n_dg,
                            is_col_major=True,
                        )
                        start, rows = int(region_offsets[g, chip, ge]), int(counts[g, 0, ge])
                        if rows:
                            region = dispatched[g, chip, start : start + rows]
                            region.copy_(expert(region.float()).to(region.dtype))
        ref.append(
            TorchCombineModule(
                dispatch_group_size=dg_size,
                experts_per_chip=experts_per_chip,
                num_experts_per_tok=num_experts_per_tok,
                seq_len_per_chip=seq,
                num_dispatch_groups=n_dg,
            )(dispatched, metadata, counts, region_offsets)
        )
        return ref[0]

    def validate(actual, what):
        r = validate_combine_output(
            torch_reference(),
            actual,
            indices,
            n_dg,
            num_routed_experts,
            use_pcc=True,
            verbose=False,
            expert_dispatch_table=dispatch_table,
            expert_token_counts=counts,
            experts_per_chip=experts_per_chip,
        )
        worst = min((m_[-1] for m_ in r.mismatches), default=None)
        logger.info(
            f"{what}: {r.matches}/{r.total} combine slots match the PyTorch reference"
            + (f", worst slot PCC {worst:.6f}" if worst is not None else "")
        )
        r.assert_passed(what)

    return dict(
        flat_solo=flat_solo,
        combine=combine,
        overlapped=overlapped,
        poisoned_y=poisoned_y,
        validate=validate,
        composer=get_ep_mesh_composer(mesh_device),
        experts_per_chip=experts_per_chip,
    )


def _combine_input(y):
    """What standalone combine reads of the flat expert's y: its bf16 rows as they are, or the bfp8 tiles as bf16."""
    return y if _YRM else ttnn.typecast(y, ttnn.bfloat16)


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id, model_id, variant",
    _params(),
    indirect=["mesh_device", "device_params"],
)
def test_flat_combine_overlap(mesh_device, device_params, threshold_id, model_id, variant):
    mesh_device = _mesh_for(mesh_device, variant)
    c = _build_case(mesh_device, device_params, threshold_id, model_id)
    # back to back first (the reference rewrites the host dispatch buffer in place, so the device runs first)
    y = c["flat_solo"]()
    seq_out = ttnn.to_torch(c["combine"](_combine_input(y)), mesh_composer=c["composer"])
    # every call on a freshly poisoned y (2nd: a cache hit): a read ahead of the write cannot hide behind earlier rows
    ov = [ttnn.to_torch(c["overlapped"](c["poisoned_y"]()), mesh_composer=c["composer"]) for _ in range(2)]
    for i, o in enumerate(ov):
        same = torch.equal(o, seq_out)
        logger.info(
            f"overlap run {i} vs flat then combine: bit-identical {same}, max|diff| "
            f"{(o.float() - seq_out.float()).abs().max().item():.4g}"
        )
    c["validate"](seq_out, "flat then combine")
    for i, o in enumerate(ov):
        c["validate"](o, f"overlap run {i}")
        assert torch.equal(o, seq_out), f"overlap run {i} differs from flat then combine"


@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id, model_id, variant",
    _params(),
    indirect=["mesh_device", "device_params"],
)
def test_flat_combine_overlap_perf(mesh_device, device_params, threshold_id, model_id, variant):
    mesh_device = _mesh_for(mesh_device, variant)
    # perf only needs the weights' layout, not their values: fake (uninitialised) weights skip the host packing
    # (FLAT_CMB_REAL_WEIGHTS=1: the real path). FLAT_CMB_ONLY=overlap: time the overlap only (sweeps).
    c = _build_case(
        mesh_device, device_params, threshold_id, model_id, fake_weights=not os.environ.get("FLAT_CMB_REAL_WEIGHTS")
    )
    if os.environ.get("FLAT_CMB_ONLY") == "overlap":
        c["overlapped"]()
        tr, out = hyb._capture(mesh_device, c["overlapped"])
        try:
            ov_us = hyb._replay_us(mesh_device, tr, _PERF_ITERS)
        finally:
            ttnn.release_trace(mesh_device, tr)
            del out
        logger.info(f"flat+combine {model_id} {threshold_id} traced: overlap {ov_us:.1f} us")
        return
    c["flat_solo"]()
    tr, y = hyb._capture(mesh_device, c["flat_solo"])
    try:
        flat_us = hyb._replay_us(mesh_device, tr, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, tr)
    yw = _combine_input(y)
    c["combine"](yw)
    tr, out = hyb._capture(mesh_device, lambda: c["combine"](yw))
    try:
        combine_us = hyb._replay_us(mesh_device, tr, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, tr)
        del y, yw, out
    if os.environ.get("FLAT_CMB_NO_OVERLAP"):  # (the flat expert on other rows, e.g. MIMO_FL_ROWS=0,9: no overlap)
        logger.info(
            f"flat+combine {model_id} {threshold_id} traced: flat {flat_us:.1f} + combine {combine_us:.1f} = "
            f"{flat_us + combine_us:.1f} us (MIMO_FL_ROWS={os.environ.get('MIMO_FL_ROWS', 'default')})"
        )
        return
    c["overlapped"]()
    tr, out = hyb._capture(mesh_device, c["overlapped"])
    try:
        ov_us = hyb._replay_us(mesh_device, tr, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, tr)
        del out
    seq_us = flat_us + combine_us
    saved = seq_us - ov_us
    logger.info(
        f"flat+combine {model_id} {threshold_id} traced: sequential {seq_us:.1f} us (flat {flat_us:.1f} + combine "
        f"{combine_us:.1f}), overlap {ov_us:.1f} us -> {seq_us / ov_us:.3f}x, saved {saved:.1f} us "
        f"({saved / seq_us:.1%}); combine hidden {min(max(saved, 0), combine_us) / combine_us:.0%}"
    )
