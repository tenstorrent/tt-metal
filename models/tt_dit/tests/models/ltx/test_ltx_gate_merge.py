# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Equivalence guard for the gate merge: folding the per-head gate into the Q/QKV projection must
leave Q, K, V and the gate itself bit-for-bit what the standalone-gate path produces.

The fold is a silent corruptor — a mis-ordered column block yields a well-shaped tensor of the wrong
data — and no other test can see it: the block harness disables its PCC path whenever audio (hence
the gate) is on. So this compares the merged attention against the unmerged one directly, from the
same weights, rather than against a reference derived from the fold itself."""

import pytest
import torch

import models.tt_dit.models.transformers.ltx.attention_ltx as attention_ltx
import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor_2dshard, from_torch, to_torch
from models.tt_dit.utils.test import ring_params

# (id, is_self, dim, num_heads, query_input_dim, M) — every gated projection the AV model runs, at
# both the stage-1 (M=1216) and stage-2 (M=4864) per-device video sequence lengths.
CASES = [
    ("video_self_s1", True, 4096, 32, 4096, 1216),
    ("video_self_s2", True, 4096, 32, 4096, 4864),
    ("video_cross_s1", False, 4096, 32, 4096, 1216),
    ("video_cross_s2", False, 4096, 32, 4096, 4864),
    ("a2v_q_s1", False, 2048, 32, 4096, 1216),
    ("a2v_q_s2", False, 2048, 32, 4096, 4864),
    ("audio_self", True, 2048, 32, 2048, 32),
    ("audio_cross", False, 2048, 32, 2048, 32),
]


def _build(merge, **kwargs):
    """LTXAttention with the gate merge forced on or off (it is otherwise env-scoped)."""
    original = attention_ltx.gate_merge_enabled
    attention_ltx.gate_merge_enabled = lambda: merge
    try:
        return LTXAttention(**kwargs)
    finally:
        attention_ltx.gate_merge_enabled = original


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("case", "is_self", "dim", "num_heads", "query_input_dim", "M"), CASES, ids=[c[0] for c in CASES]
)
def test_gate_merge_matches_standalone_gate(
    mesh_device, sp_axis, tp_axis, num_links, topology, case, is_self, dim, num_heads, query_input_dim, M
):
    torch.manual_seed(0)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    kwargs = dict(
        dim=dim,
        num_heads=num_heads,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_self=is_self,
        query_input_dim=None if is_self else query_input_dim,
        context_dim=None if is_self else dim,
        apply_gated_attention=True,
    )
    merged = _build(True, **kwargs)
    plain = _build(False, **kwargs)
    assert merged.merge_gate and not plain.merge_gate

    state = {
        "q_norm.weight": torch.randn(dim),
        "k_norm.weight": torch.randn(dim),
        "to_q.weight": torch.randn(dim, query_input_dim) * 0.05,
        "to_q.bias": torch.randn(dim) * 0.05,
        "to_k.weight": torch.randn(dim, dim) * 0.05,
        "to_k.bias": torch.randn(dim) * 0.05,
        "to_v.weight": torch.randn(dim, dim) * 0.05,
        "to_v.bias": torch.randn(dim) * 0.05,
        "to_out.0.weight": torch.randn(dim, dim) * 0.05,
        "to_out.0.bias": torch.randn(dim) * 0.05,
        "to_gate_logits.weight": torch.randn(num_heads, query_input_dim) * 0.05,
        "to_gate_logits.bias": torch.randn(num_heads) * 0.05,
    }
    merged.load_torch_state_dict({k: v.clone() for k, v in state.items()})
    plain.load_torch_state_dict({k: v.clone() for k, v in state.items()})

    x = torch.randn(1, 1, M, query_input_dim)
    x_tt = from_torch(
        x, device=mesh_device, layout=ttnn.Layout.TILE, dtype=ttnn.bfloat16, mesh_axes=[None, None, None, tp_axis]
    )

    def project(attn):
        proj = attn.to_qkv if is_self else attn.to_q
        out = proj(x_tt, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=parallel_config)
        if attn.merge_gate:
            *qkv, gate_logits = out if isinstance(out, list) else [out]
            return qkv, attn._gate_from_logits(gate_logits)
        qkv = out if isinstance(out, list) else [out]
        return qkv, attn._compute_gate(x_tt, parallel_config)

    merged_qkv, merged_gate = project(merged)
    plain_qkv, plain_gate = project(plain)

    assert len(merged_qkv) == len(plain_qkv) == merged.base_chunks
    for m, p in zip(merged_qkv, plain_qkv):
        assert_quality(
            to_torch(p, mesh_axes=[None, None, None, tp_axis]),
            to_torch(m, mesh_axes=[None, None, None, tp_axis]),
            pcc=0.9999,
            relative_rmse=0.01,
        )

    # The merged gate lands in 1BND (concatenate_heads) layout — one column per head CHANNEL — while
    # the standalone gate is (B, H, N, 1) and broadcasts over those channels. Expanding the latter is
    # exactly what the merged fold precomputes, so they must agree value for value.
    plain_1bnd = to_torch(plain_gate, mesh_axes=[None, tp_axis, None, None])  # (B, num_heads, M, 1)
    plain_1bnd = plain_1bnd.permute(3, 0, 2, 1).repeat_interleave(merged.head_dim, dim=-1)  # (1, B, M, dim)
    assert_quality(
        plain_1bnd,
        to_torch(merged_gate, mesh_axes=[None, None, None, tp_axis]),
        pcc=0.9999,
        relative_rmse=0.01,
    )


# (id, is_self, dim, num_heads, N) — the projection test above proves Q/K/V/gate come out of the
# merged weight correctly; this covers what it cannot: the whole forward, i.e. that the gate chunk
# still means the same thing once SDPA has run and `concatenate_heads` has laid the heads back out.
FORWARD_CASES = [
    ("video_self", True, 4096, 32, 9728),
    ("video_cross", False, 4096, 32, 9728),
    ("audio_self", True, 2048, 32, 256),
    ("audio_cross", False, 2048, 32, 256),
]


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("case", "is_self", "dim", "num_heads", "N"), FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES])
def test_gate_merge_forward_matches_standalone_gate(
    mesh_device, sp_axis, tp_axis, num_links, topology, case, is_self, dim, num_heads, N
):
    """Full LTXAttention.forward(), merged gate vs standalone gate, from identical weights."""
    torch.manual_seed(0)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    kwargs = dict(
        dim=dim,
        num_heads=num_heads,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_self=is_self,
        context_dim=None if is_self else dim,
        apply_gated_attention=True,
    )
    merged = _build(True, **kwargs)
    plain = _build(False, **kwargs)
    assert merged.merge_gate and not plain.merge_gate

    state = {
        "q_norm.weight": torch.randn(dim),
        "k_norm.weight": torch.randn(dim),
        "to_q.weight": torch.randn(dim, dim) * 0.05,
        "to_q.bias": torch.randn(dim) * 0.05,
        "to_k.weight": torch.randn(dim, dim) * 0.05,
        "to_k.bias": torch.randn(dim) * 0.05,
        "to_v.weight": torch.randn(dim, dim) * 0.05,
        "to_v.bias": torch.randn(dim) * 0.05,
        "to_out.0.weight": torch.randn(dim, dim) * 0.05,
        "to_out.0.bias": torch.randn(dim) * 0.05,
        "to_gate_logits.weight": torch.randn(num_heads, dim) * 0.05,
        "to_gate_logits.bias": torch.randn(num_heads) * 0.05,
    }
    merged.load_torch_state_dict({k: v.clone() for k, v in state.items()})
    plain.load_torch_state_dict({k: v.clone() for k, v in state.items()})

    # spatial is SP- and TP-fractured, exactly as the block feeds it.
    spatial = torch.randn(1, 1, N, dim)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    fwd = dict(N=N)
    if not is_self:
        # Text cross-attn: replicated prompt over SP, TP-fractured on the feature dim (kv_replicated).
        prompt = torch.randn(1, 1, 32, dim)
        fwd["prompt_1BLP"] = from_torch(
            prompt,
            device=mesh_device,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, None, tp_axis],
        )
        fwd["kv_replicated"] = True

    out_m = merged(spatial_1BND=tt_spatial, **fwd)
    out_p = plain(spatial_1BND=tt_spatial, **fwd)

    concat = [None, None]
    concat[sp_axis] = 2
    concat[tp_axis] = 3
    to_t = lambda t: ttnn.to_torch(  # noqa: E731
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat, mesh_shape=tuple(mesh_device.shape))
    )
    assert_quality(to_t(out_p), to_t(out_m), pcc=0.999, relative_rmse=0.02)
