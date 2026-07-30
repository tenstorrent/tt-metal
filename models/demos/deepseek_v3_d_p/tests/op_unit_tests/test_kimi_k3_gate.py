# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi-K3 output-gate epilogue as a unit: all-gather -> g_proj(+sigmoid) -> concat_heads -> multiply
-> o_proj -> reduce-scatter.

The gate's risk is **layout agreement**, not any single op. Every op involved already has its own
coverage; what is new in K3 is the claim that

    device d's g_proj output columns  ==  device d's attention head range after nlp_concat_heads

so the multiply needs no reshape. That holds because ``nlp_concat_heads`` emits head-major over the
last dim, and ``mapper_tp1`` gives device d the columns ``[d*N_loc, (d+1)*N_loc)`` of a 12288-wide
weight, which is exactly the ``[d*H_loc, (d+1)*H_loc)`` head range at ``v_head_dim`` each. If that
reasoning is wrong the module PCC test would degrade without saying why, so it is pinned here
directly -- including a negative case that permutes the head order and must fail.

Also compares the two viable collective layouts (see docs/KIMI_K3_MLA.md §4):

  * **B (implemented)** -- all-gather ``hidden_states`` to the full hidden size, N-shard ``g_proj``
    (``mapper_tp1``), sigmoid fuses into the matmul.
  * **A** -- keep ``g_proj`` K-sharded (``mapper_tp0``), reduce-scatter the 12288-wide partial;
    sigmoid must follow the cross-device reduce.

Both are asserted numerically equivalent, so the choice stays a performance question rather than a
correctness one.

Per-device geometry matches the op audit: S_loc = 640, tp = 4, H_loc = 24.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

PCC_REQUIRED = 0.99
# Head-order corruption must be unmistakable, not marginal.
PERMUTED_PCC_CEILING = 0.95

HIDDEN = 7168
NUM_HEADS = 96
V_HEAD_DIM = 128
GATE_WIDTH = NUM_HEADS * V_HEAD_DIM  # 12288
S_LOC = 640

SP_AXIS, TP_AXIS = 0, 1


def _reference(hidden_full, w_g, attn_full, w_o):
    """Torch reference for the whole epilogue, in the upstream order."""
    g = torch.sigmoid(hidden_full.float() @ w_g.float())  # [1,1,S,12288]
    bsz, _, seq, _ = hidden_full.shape
    # nlp_concat_heads: [1, H, S, v] -> [1, 1, S, H*v], head-major over the last dim.
    concat = attn_full.float().permute(0, 2, 1, 3).reshape(bsz, 1, seq, GATE_WIDTH)
    return (concat * g) @ w_o.float()


def _shard(mesh_device, t, *, sp_dim=None, tp_dim=None, dtype=ttnn.bfloat16, mem=ttnn.DRAM_MEMORY_CONFIG):
    if sp_dim is None and tp_dim is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        dims = [None, None]
        dims[SP_AXIS] = sp_dim
        dims[TP_AXIS] = tp_dim
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    return ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper
    )


def _gather(mesh_device, tt, *, sp_dim=None, tp_dim=None):
    dims = [None, None]
    dims[SP_AXIS] = sp_dim
    dims[TP_AXIS] = tp_dim
    return ttnn.to_torch(
        tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    )


def _make_inputs(mesh_device, seed=42):
    """Global (unsharded) tensors plus the derived per-device geometry."""
    sp = mesh_device.shape[SP_AXIS]
    tp = mesh_device.shape[TP_AXIS]
    assert NUM_HEADS % tp == 0, f"{NUM_HEADS} heads must divide tp={tp}"
    seq = S_LOC * sp

    torch.manual_seed(seed)
    hidden_full = (torch.randn(1, 1, seq, HIDDEN) * 0.5).to(torch.bfloat16)
    attn_full = (torch.randn(1, NUM_HEADS, seq, V_HEAD_DIM) * 0.5).to(torch.bfloat16)
    w_g = (torch.randn(HIDDEN, GATE_WIDTH) * 0.02).to(torch.bfloat16)
    w_o = (torch.randn(GATE_WIDTH, HIDDEN) * 0.02).to(torch.bfloat16)
    return hidden_full, attn_full, w_g, w_o, sp, tp, seq


def _compute_kernel_config(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _run_epilogue(mesh_device, hidden_full, attn_full, w_g, w_o, *, layout, fuse_sigmoid, gate_weight_override=None):
    """Run the gate epilogue on device and return the gathered [1,1,S,HIDDEN] result."""
    tt_ccl = get_tt_ccl(mesh_device)
    ckc = _compute_kernel_config(mesh_device)
    tp = mesh_device.shape[TP_AXIS]

    # Residual stream: SP-sharded on seq, TP-fractured on features.
    tt_hidden = _shard(mesh_device, hidden_full, sp_dim=2, tp_dim=3)
    # Per-head attention output: SP on seq, TP on heads.
    tt_attn = _shard(mesh_device, attn_full, sp_dim=2, tp_dim=1)
    # o_proj is K-sharded over the head ranges (mapper_tp0 in ttMLA).
    tt_w_o = _shard(mesh_device, w_o, tp_dim=0)

    w_g_host = w_g if gate_weight_override is None else gate_weight_override

    if layout == "B":
        # All-gather hidden to full HIDDEN, N-shard g_proj -> g is complete per device.
        tt_w_g = _shard(mesh_device, w_g_host, tp_dim=1, dtype=ttnn.bfloat8_b)
        hidden_gathered = ttnn.experimental.all_gather_async(
            tt_hidden,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=TP_AXIS),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=TP_AXIS),
            num_links=tt_ccl_num_links(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=TP_AXIS,
        )
        # No program_config here (this test is about layout, not tiling), so the fused form goes
        # through `activation=`. A tuned config in mla_config.py carries the equivalent
        # UnaryWithParam(SIGMOID, 4.0, 0.0) as fused_activation instead -- the two are the same
        # kernel path, which test_kimi_k3_mla_matmuls.py pins directly.
        g = ttnn.linear(
            hidden_gathered,
            tt_w_g,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat16,
            activation="sigmoid" if fuse_sigmoid else None,
        )
        if not fuse_sigmoid:
            g = ttnn.sigmoid(g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_gathered)
    elif layout == "A":
        # K-shard g_proj -> each device holds a 12288-wide partial; reduce-scatter, then sigmoid.
        tt_w_g = _shard(mesh_device, w_g_host, tp_dim=0, dtype=ttnn.bfloat8_b)
        partial = ttnn.linear(
            tt_hidden,
            tt_w_g,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat16,
        )
        g = ttnn.experimental.reduce_scatter_minimal_async(
            partial,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=TP_AXIS),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=TP_AXIS),
            num_links=tt_ccl_num_links(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=TP_AXIS,
        )
        ttnn.deallocate(partial)
        # sigmoid MUST follow the cross-device reduce -- it cannot fuse into the matmul here.
        g = ttnn.sigmoid(g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        raise AssertionError(f"unknown layout {layout!r}")

    v_out = ttnn.experimental.nlp_concat_heads(tt_attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_out = ttnn.multiply(v_out, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(g)

    out = ttnn.linear(
        v_out,
        tt_w_o,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat16,
    )
    out = ttnn.experimental.reduce_scatter_minimal_async(
        out,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=TP_AXIS),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=TP_AXIS),
        num_links=tt_ccl_num_links(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        cluster_axis=TP_AXIS,
    )
    ttnn.synchronize_device(mesh_device)
    # Output is SP-sharded on seq and TP-sharded on features again.
    return _gather(mesh_device, out, sp_dim=2, tp_dim=3).float()


def tt_ccl_num_links(mesh_device):
    """Blackhole trains 2 fabric routing planes, others 1 -- mirrors ttMLA.ccl_num_links."""
    return 2 if is_blackhole() else 1


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("fuse_sigmoid", [True, False], ids=["fused_sigmoid", "standalone_sigmoid"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 gate epilogue is validated on Blackhole")
def test_k3_gate_epilogue(mesh_device, fuse_sigmoid):
    """Layout B (the implemented one) against a torch reference."""
    hidden_full, attn_full, w_g, w_o, sp, tp, seq = _make_inputs(mesh_device)
    logger.info(f"gate epilogue: sp={sp} tp={tp} seq={seq} S_loc={seq // sp} H_loc={NUM_HEADS // tp}")

    actual = _run_epilogue(mesh_device, hidden_full, attn_full, w_g, w_o, layout="B", fuse_sigmoid=fuse_sigmoid)
    expected = _reference(hidden_full, w_g, attn_full, w_o)

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    passing, pcc = comp_pcc(expected, actual, PCC_REQUIRED)
    logger.info(f"K3 gate epilogue (layout B, fuse_sigmoid={fuse_sigmoid}): PCC={pcc}")
    assert passing, f"gate epilogue PCC {pcc} < {PCC_REQUIRED}"


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 gate epilogue is validated on Blackhole")
def test_k3_gate_layouts_agree(mesh_device):
    """Layout A and layout B must produce the same numbers.

    Keeps the A-vs-B decision a performance question. If this ever fails, the collective layouts are
    not equivalent and the choice recorded in docs/KIMI_K3_MLA.md §4 needs revisiting.
    """
    hidden_full, attn_full, w_g, w_o, *_ = _make_inputs(mesh_device)

    out_b = _run_epilogue(mesh_device, hidden_full, attn_full, w_g, w_o, layout="B", fuse_sigmoid=True)
    out_a = _run_epilogue(mesh_device, hidden_full, attn_full, w_g, w_o, layout="A", fuse_sigmoid=False)

    passing, pcc = comp_pcc(out_b, out_a, PCC_REQUIRED)
    logger.info(f"K3 gate layout A vs B: PCC={pcc}")
    assert passing, f"gate layouts A and B disagree: PCC {pcc} < {PCC_REQUIRED}"


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 gate epilogue is validated on Blackhole")
def test_k3_gate_head_order_is_load_bearing(mesh_device):
    """Permuting g_proj's output columns across head ranges must break the result.

    This is the negative half of the layout claim. The permutation rotates the weight's columns by
    exactly one device's share, so every device multiplies its heads by another device's gate. If
    this still passed, it would mean the gate is not actually aligned per head and the correct-looking
    PCC above is coincidental.
    """
    hidden_full, attn_full, w_g, w_o, sp, tp, seq = _make_inputs(mesh_device)
    n_loc = GATE_WIDTH // tp
    rolled = torch.roll(w_g, shifts=n_loc, dims=1)

    actual = _run_epilogue(
        mesh_device, hidden_full, attn_full, w_g, w_o, layout="B", fuse_sigmoid=True, gate_weight_override=rolled
    )
    expected = _reference(hidden_full, w_g, attn_full, w_o)

    _, pcc = comp_pcc(expected, actual, PCC_REQUIRED)
    logger.info(f"K3 gate with head order rolled by one device: PCC={pcc} (must be low)")
    assert pcc < PERMUTED_PCC_CEILING, (
        f"rolling g_proj's head ranges left PCC at {pcc} (>= {PERMUTED_PCC_CEILING}); the gate is "
        "evidently not head-aligned, so the positive test is not proving what it claims"
    )
