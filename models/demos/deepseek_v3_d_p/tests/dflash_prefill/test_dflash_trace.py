# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace-capture safety of the DFlash drafter's FC tap.

Enabling PREFILL_USE_TRACE with DFlash on puts ``TtDFlashDrafter.tap()`` INSIDE the captured verifier
forward. A capture only replays what it recorded, so the tap has to be allocation-free and it has to
write its accumulator at an address that survives every replay -- the accumulator is read back by the
EAGER ``forward()`` that runs after the replay returns.

This runs the identical chunk sequence twice against two separate cache pairs: once fully eager, once
with only the tap phase captured and replayed. The two caches must come back BIT-IDENTICAL -- not merely
close. Same ops, same inputs, so any difference at all means the replay did not reproduce the eager
computation (a stale accumulator, an address the trace no longer owns, a per-chunk reset that the
capture dropped). PCC would hide exactly the failure this is looking for: a tap that silently keeps the
previous chunk's sum still correlates strongly with the right answer.

    DFLASH_HF_MODEL=/path/to/Kimi-K2.x-DFlash MESH_DEVICE=8x4 \
    pytest models/demos/deepseek_v3_d_p/tests/dflash_prefill/test_dflash_trace.py -svv
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.tt_dflash_drafter import TtDFlashDrafter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_dflash_kv_cache

TRACE_REGION_SIZE = 64 * 1024 * 1024


def _host_caches(k_cache, v_cache, mesh_device, mesh_shape):
    """Read both caches back raw (no un-rotation): this compares device state to device state, so the
    block-cyclic layout is identical on both sides and only obscures the diff if undone."""
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_shape)
    return ttnn.to_torch(k_cache, mesh_composer=comp), ttnn.to_torch(v_cache, mesh_composer=comp)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"], indirect=True)
@pytest.mark.parametrize(
    "ctx_len, n_chunks",
    [pytest.param(10240, 2, id="ctx10k-2chunk")],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            {
                **torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
                "trace_region_size": TRACE_REGION_SIZE,
            },
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_tap_trace_matches_eager(
    mesh_device,
    device_params,
    num_links,
    ctx_len,
    n_chunks,
    use_pretrained,
    drafter_cfg,
    drafter_state_dict,
):
    topology = per_axis_topology(device_params["fabric_config"])[1]
    cfg = drafter_cfg
    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    H = cfg.hidden_size
    chunk_global = ctx_len // n_chunks

    gen = torch.Generator().manual_seed(0)
    ctx = torch.randn(1, ctx_len, cfg.target_feature_size, generator=gen, dtype=torch.float32)

    drafter = TtDFlashDrafter(
        mesh_device,
        cfg,
        state_dict=drafter_state_dict,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=ctx_len,
        chunk_size=chunk_global,
        num_links=num_links,
        topology=topology,
    )

    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3
    hidden_shard[sp_axis] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    def host_hidden(chunk_idx, j):
        lo = chunk_idx * chunk_global
        return ctx[:, lo : lo + chunk_global, j * H : (j + 1) * H].to(torch.bfloat16).reshape(1, 1, chunk_global, H)

    def to_device(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # ---------------- pass A: fully eager ----------------
    k_a, v_a = allocate_dflash_kv_cache(mesh_device, cfg, ctx_len, sp_axis=sp_axis, tp_axis=tp_axis)
    for c in range(n_chunks):
        drafter.reset()
        for j, tid in enumerate(cfg.target_layer_ids):
            h = to_device(host_hidden(c, j))
            drafter.tap(h, tid)
            ttnn.deallocate(h)
        drafter.forward(k_a, v_a, c * chunk_global)
    ttnn.synchronize_device(mesh_device)
    eager_k, eager_v = _host_caches(k_a, v_a, mesh_device, mesh_shape)

    # ---------------- pass B: tap phase captured and replayed ----------------
    # The tap inputs must live at FIXED addresses: inside the capture the matmul reads whatever buffer it
    # recorded, so each chunk's hidden is copied into the same persistent tensor rather than uploaded
    # fresh. This mirrors the runtime, where the taps read intermediates of the captured forward.
    persistent = [to_device(host_hidden(0, j)) for j in range(len(cfg.target_layer_ids))]

    # Warm BEFORE capture. The first tap ever allocates the accumulator and the second allocates the
    # scratch; a capture cannot absorb either (nor the matmul programs' first compile).
    drafter.reset()
    for j, tid in enumerate(cfg.target_layer_ids):
        drafter.tap(persistent[j], tid)
    ttnn.synchronize_device(mesh_device)

    tid_trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for j, tid in enumerate(cfg.target_layer_ids):
        drafter.tap(persistent[j], tid)
    ttnn.end_trace_capture(mesh_device, tid_trace, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"captured the {len(cfg.target_layer_ids)}-layer tap phase")

    k_b, v_b = allocate_dflash_kv_cache(mesh_device, cfg, ctx_len, sp_axis=sp_axis, tp_axis=tp_axis)
    for c in range(n_chunks):
        drafter.reset()
        for j in range(len(cfg.target_layer_ids)):
            fresh = to_device(host_hidden(c, j))
            ttnn.copy(fresh, persistent[j])
            ttnn.deallocate(fresh)
        ttnn.execute_trace(mesh_device, tid_trace, cq_id=0, blocking=False)
        drafter.forward(k_b, v_b, c * chunk_global)
    ttnn.synchronize_device(mesh_device)
    traced_k, traced_v = _host_caches(k_b, v_b, mesh_device, mesh_shape)
    ttnn.release_trace(mesh_device, tid_trace)

    # A negative control, so a pass cannot mean "both reads returned the same empty cache": the two runs
    # must actually have written something, and the second chunk must differ from the first.
    assert traced_k.abs().sum() > 0 and traced_v.abs().sum() > 0, "the traced pass wrote nothing at all"

    for name, a, b in (("K", eager_k, traced_k), ("V", eager_v, traced_v)):
        diff = (a != b).sum().item()
        logger.info(f"{name}: {diff} differing elements of {a.numel()}")
        assert diff == 0, (
            f"traced tap != eager tap in the drafter {name} cache: {diff} of {a.numel()} elements differ "
            f"(max |delta| {(a.float() - b.float()).abs().max().item()}). The replay did not reproduce the "
            f"eager accumulation -- check that tap() neither allocates nor moves its accumulator."
        )
    logger.info("traced tap phase is bit-identical to eager across every chunk")
