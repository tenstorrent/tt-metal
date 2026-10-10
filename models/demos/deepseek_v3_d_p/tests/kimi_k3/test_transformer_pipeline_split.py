# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy gate for splitting Kimi-K3 across pipeline ranks.

`test_transformer_depth.py`'s L24 rung says what 24 layers of this model produce. This says the
same 24 layers produce it when the stack is cut in half at layer 12 and the halves only communicate
through the payload a rank boundary carries. Two `TtKimiK3Transformer`s, built exactly as the runner
builds them (`first_layer_idx` / `is_first_rank` / `is_last_rank`), on ONE mesh and in one process —
so what is measured is the handoff, not the socket.

Why a split can be wrong while every layer looks right: AttnRes reads score the live stream against
every sealed snapshot, and the snapshots for layers before the boundary are produced on the other
rank. A rank that starts with an empty sealed set still runs, still writes plausible KV, and still
produces a smooth per-layer curve — it is simply a different model. The signature is in the SHAPE:

  L24 measured, undivided: step at the layer-12 seal, minimum 0.9864 at layer 19, back to 0.9980
  by layer 23. The climb is the live sum regrowing until it dominates the softmax mixture again.

An inherited-set failure has no such climb — the second block never sees the first block's
snapshot, so error accumulates monotonically to the end of the rank. So the assertion here is not
only "PCC above a bar" but "recovers by layer 23", which a floor-only check would miss.

Layers 12-23 are the load-bearing half. Layers 0-11 run on the first rank and would pass unchanged
with the handoff entirely broken.
"""

import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_x_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_1M, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, gather_cache_tp0, unrotate_cache_layer

SP_AXIS, TP_AXIS = 0, 1
# `KIMI_K3_TEST_SEQ_LEN`: see test_transformer_depth.py (2560 keeps a 4x4 stage at 640 tokens per chip).
SEQ_LEN = int(os.getenv("KIMI_K3_TEST_SEQ_LEN", "5120"))
NUM_LAYERS = 24
BOUNDARY = 12

# Same bars as the depth ladder's L24 rung, for the same reasons documented there.
LAYER_PCC = 0.98
KV_CACHE_PCC = 0.96
# The undivided L24 run recovers to ~0.998 by the last layer. Requiring recovery is what separates
# "error accumulates, as it does undivided" from "the sealed set never arrived".
RECOVERY_PCC = 0.99
RECOVERY_FROM = 22

PLACEMENTS = [
    pytest.param(
        (8, 4),
        # 4096: see test_transformer_depth.py. 1152 fails once the sealed set has two blocks, 24576
        # breaks MLA's chunked attention. Both regimes are live here.
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": KimiK3Config.L1_SMALL_SIZE},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]

# Two stages on one galaxy: open the whole 8x4 and carve its top and bottom 4x4 halves, one per rank, so
# each rank holds only its own 12 layers (all 24 on one 4x4 would be twice a real stage's weights).
# TORUS_X: a half wraps only its 4-wide in-tray axis (mesh axis 1); torus_xy would ask Ring on the
# half's unwrapped 4-row axis.
SUBMESH_PLACEMENTS = [
    pytest.param(
        (8, 4),
        torus_x_device_params(l1_small_size=KimiK3Config.L1_SMALL_SIZE),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-x-8x4-as-2x4x4",
    )
]
STAGE_SHAPE = (4, 4)


def _compose(mesh_device, tensor):
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).reshape(-1, KimiK3Config.EMB_SIZE)[:SEQ_LEN]


def _build_rank(mesh_device, config, cache, first_layer_idx, num_layers, *, is_first, is_last):
    """One rank's slice, built the way the runner builds it.

    `state_dict={}` is the production path: every weight comes from the TTNN cache, which is keyed
    by GLOBAL layer index. That is deliberate here rather than incidental — a rank asking the cache
    for the wrong window is one of the failure modes under test, and passing a checkpoint-derived
    state_dict would paper over it.
    """
    return TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        {},
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first,
        is_last_rank=is_last,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )


def _host_handoff(packed, src_mesh, dst_mesh):
    """Move rank 0's packed boundary tensor onto rank 1's submesh, sharded the way the runner's D2D
    socket carries it (`D2D_MAPPER_CONFIG`: rows on the SP axis, hidden on the TP axis). Lossless: the
    tensor is bf16 on both sides, so any difference from the single-mesh result is a placement bug."""
    shape = tuple(src_mesh.shape)
    # A mesh tensor's `.shape` is the per-device shard: rows / SP and hidden / TP when it is sharded the
    # way the socket carries it.
    per_device = tuple(packed.shape)
    assert (
        per_device[2] * shape[SP_AXIS] == SEQ_LEN and per_device[3] * shape[TP_AXIS] == KimiK3Config.EMB_SIZE
    ), f"packed tensor is not [rows on SP, hidden on TP] sharded: per device {per_device} on a {shape} mesh"
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    host = ttnn.to_torch(packed, mesh_composer=ttnn.ConcatMesh2dToTensor(src_mesh, dims=tuple(dims), mesh_shape=shape))
    return ttnn.from_torch(
        host,
        dtype=packed.dtype,
        layout=packed.layout,
        device=dst_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(dst_mesh, dims=tuple(dims), mesh_shape=tuple(dst_mesh.shape)),
    )


def _run_split(mesh_0, mesh_1, handoff):
    """Rank 0 (layers [0, 12)) on `mesh_0`, rank 1 (layers [12, 24)) on `mesh_1`, joined by `handoff`."""
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_1M)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 1M golden trace")
    assert tuple(mesh_0.shape) == tuple(mesh_1.shape), "both ranks must run the same stage shape"

    checkpoint = Path(checkpoint)
    resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    cache = cache_root(checkpoint, tuple(mesh_0.shape), TP_AXIS)

    rank0 = _build_rank(mesh_0, config, cache, 0, BOUNDARY, is_first=True, is_last=False)
    rank1 = _build_rank(mesh_1, config, cache, BOUNDARY, NUM_LAYERS - BOUNDARY, is_first=False, is_last=True)
    meshes = {"rank0": mesh_0, "rank1": mesh_1}

    # The plane counts the runner would size the socket with. Asserted here so a mismatch is a
    # readable failure in-process rather than a rendezvous TT_FATAL on two machines.
    assert (
        rank0.outbound_planes == rank1.inbound_planes
    ), f"rank 0 sends {rank0.outbound_planes} planes, rank 1 expects {rank1.inbound_planes}"
    assert rank1.inbound_planes == 1 + BOUNDARY // KimiK3Config.ATTN_RES_BLOCK_SIZE

    # A KV cache per rank: `schedule.kv_slot` is rank-local, so rank 1's MLA layers (15, 19, 23)
    # occupy slots 0..2 of its own cache, not slots 3..5 of a shared one.
    caches = {}
    for name, model in (("rank0", rank0), ("rank1", rank1)):
        mesh = meshes[name]
        caches[name] = (
            allocate_mla_kvpe_cache(
                mesh_device=mesh,
                hf_config=config,
                max_seq_len=SEQ_LEN,
                mesh_shape=tuple(mesh.shape),
                sp_axis=SP_AXIS,
                num_layers=model.schedule.num_mla_layers,
                num_users=1,
            )
            if model.schedule.num_mla_layers
            else None
        )

    tokens_tt = prepare_prefill_input_tensor(
        trace.token_ids(SEQ_LEN)[0].tolist(),
        mesh_0,
        tuple(mesh_0.shape)[SP_AXIS],
        False,
        tuple(mesh_0.shape),
        SP_AXIS,
    )

    per_layer = {}

    def tap_for(first_layer_idx, mesh):
        def tap(local_idx, hidden):
            per_layer[first_layer_idx + local_idx] = _compose(mesh, hidden)

        return tap

    try:
        # The handoff: rank 0's packed [live | sealed] goes into rank 1. Over a real pipeline the same
        # tensor crosses a D2D socket, which is a lossless bf16 transfer of a tensor that is already
        # bf16 — so any runner-level difference from this result is a transport bug, not an AttnRes one.
        packed = rank0.forward(tokens_tt, kvpe_cache=caches["rank0"], layer_tap=tap_for(0, mesh_0))
        assert (
            packed.shape[1] == rank0.outbound_planes
        ), f"rank 0 handed off {packed.shape[1]} planes, declared {rank0.outbound_planes}"
        rank1.forward(handoff(packed), kvpe_cache=caches["rank1"], layer_tap=tap_for(BOUNDARY, mesh_1))
    finally:
        for model in (rank0, rank1):
            if model.kda_states is not None:
                model.kda_states.deallocate()

    # KV first: it is the one oracle a residual-shaped self-consistency cannot fake. Rank 1's slabs
    # are written from reads that exist only if the inherited sealed set arrived.
    for name, model in (("rank0", rank0), ("rank1", rank1)):
        kvpe = caches[name]
        if kvpe is None:
            continue
        mesh = meshes[name]
        # slot -> GLOBAL layer, via the schedule rather than `mla_layer_ids[:num_mla_layers]`.
        # `mla_layer_ids` is the whole MODEL's list, so slicing its head gives 3/7/11 for every
        # rank — right only for a rank starting at layer 0, and silently wrong for any other
        # (it would score rank 1's layer-15 slab against layer 3's golden). Same local/global
        # confusion as #54843.
        slot_to_global = {
            slot: model.schedule.global_index(local)
            for local, slot in enumerate(model.schedule.kv_slot_of_local)
            if slot is not None
        }
        gathered = gather_cache_tp0(kvpe.storage, mesh)
        positions = blockcyclic_positions(tuple(mesh.shape)[SP_AXIS], SEQ_LEN, SEQ_LEN)
        for slot, model_layer in sorted(slot_to_global.items()):
            if not trace.has_kv_cache(model_layer):
                continue
            device_rows = unrotate_cache_layer(gathered[slot], positions, SEQ_LEN)
            golden_rows = trace.kv_cache(model_layer, 0, SEQ_LEN)
            pcc_nope, pcc_pe = cache_half_pccs(golden_rows, device_rows, KimiK3Config.KV_LORA_RANK, pe_interleave=False)
            logger.info(
                f"split KV {name} slot {slot} (model layer {model_layer}): lora={pcc_nope:.6f} rope={pcc_pe:.6f}"
            )
            assert min(pcc_nope, pcc_pe) >= KV_CACHE_PCC, (
                f"KV cache {name} slot {slot} (model layer {model_layer}) diverged: "
                f"lora={pcc_nope:.6f} rope={pcc_pe:.6f}"
            )

    # Score everything before asserting: the shape of the curve is the diagnosis, and stopping at
    # the first shortfall hides it.
    scores = {}
    for idx in range(NUM_LAYERS):
        want = trace.decoder_output(idx, 0, SEQ_LEN)
        _, message = comp_pcc(want, per_layer[idx], LAYER_PCC)
        scores[idx] = float(str(message).split()[-1])
        marks = []
        if idx % KimiK3Config.ATTN_RES_BLOCK_SIZE == 0:
            marks.append("seal")
        if idx == BOUNDARY:
            marks.append("RANK BOUNDARY")
        suffix = f"  <- {', '.join(marks)}" if marks else ""
        logger.info(f"split layer {idx} vs decoder_output_layer_{idx}: {scores[idx]:.7f}{suffix}")

    worst = min(scores, key=scores.get)
    logger.info(f"split worst layer {worst}: {scores[worst]:.7f} (bar {LAYER_PCC})")
    below = {i: s for i, s in scores.items() if s < LAYER_PCC}
    assert not below, f"layers below {LAYER_PCC}: " + ", ".join(f"{i}:{s:.6f}" for i, s in sorted(below.items()))

    # The shape assertion. An empty inherited set degrades monotonically through the second block
    # instead of recovering, and would clear a floor-only check on the strength of its early layers.
    tail = min(scores[i] for i in range(RECOVERY_FROM, NUM_LAYERS))
    assert tail >= RECOVERY_PCC, (
        f"no recovery by layer {RECOVERY_FROM}+ (min {tail:.6f} < {RECOVERY_PCC}); the second block "
        f"is not reading the first block's sealed snapshot"
    )


# Two 12-layer stacks get built and run before anything is scored, which is past the suite's
# 300s default. The measurement itself is quick; the model build is not.
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_pipeline_split_matches_golden(mesh_device, device_params):
    _run_split(mesh_device, mesh_device, lambda packed: packed)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", SUBMESH_PLACEMENTS, indirect=True)
def test_pipeline_split_on_stage_submeshes(mesh_device, device_params):
    """The same split with each rank on its own 4x4 half of the galaxy: real per-stage memory, and a
    boundary tensor that has to move between meshes."""
    top, bottom = mesh_device.create_submeshes(ttnn.MeshShape(*STAGE_SHAPE))
    _run_split(top, bottom, lambda packed: _host_handoff(packed, top, bottom))
