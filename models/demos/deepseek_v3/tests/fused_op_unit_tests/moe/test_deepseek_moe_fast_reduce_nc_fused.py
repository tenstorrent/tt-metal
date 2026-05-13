# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for deepseek_moe_fast_reduce_nc_fused — a single fused kernel that
replaces the four-op chain:
    permute(scores, (3,1,0,2)) → to_layout(TILE) → mul(activation, scores)
    → deepseek_moe_fast_reduce_nc

Runs on a single Tenstorrent device. Emulates the 16×8 mesh (128 devices)
by 128 sequential iterations (same pattern as test_deepseek_moe_fast_reduce_nc_single.py).
"""

import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# REFERENCE_MESH_SHAPE = (1, 1)
REFERENCE_MESH_SHAPE = (16, 8)
NUM_REFERENCE_MESH_DEVICES = REFERENCE_MESH_SHAPE[0] * REFERENCE_MESH_SHAPE[1]


def _bf16_to_float(bf16):
    """Convert a torch.bfloat16 scalar or tensor to float32 using numpy reinterpretation"""
    import numpy as np

    if isinstance(bf16, float):
        return float(bf16)
    # Convert torch tensor to numpy uint16, upcast to uint32 << 16
    arr = bf16.detach().cpu().numpy().astype(np.uint16)
    arr32 = arr.astype(np.uint32) << 16
    return arr32.view(np.float32)


def _torch_golden_scale_and_fast_reduce_nc(
    torch_unsqueezed_global,
    torch_expert_scores_global,
    *,
    num_replicated_devices,
):
    """Reference: permute + broadcast mul + split-width sum along expert dim (dim 0)."""
    topk_experts_weights = torch_expert_scores_global.permute(3, 1, 0, 2)
    scaled = torch_unsqueezed_global * topk_experts_weights
    hidden_size = scaled.shape[-1]
    split_size = hidden_size // num_replicated_devices
    assert hidden_size % split_size == 0
    num_chunks = hidden_size // split_size
    goldens = []
    for i in range(num_chunks):
        chunk = scaled[:, :, :, i * split_size : (i + 1) * split_size]
        goldens.append(chunk.sum(dim=0, keepdim=True))
    return goldens


def _gen_expert_mapping_linearized(experts, num_devices):
    """Convention-B mapping: shape [1, experts], cell value = linearized device id owning that expert.

    Distributes experts evenly across devices in contiguous blocks (device d owns
    [d*E, (d+1)*E) where E = experts/num_devices). Matches the format
    `MoEOptimized.create_shared_state` constructs (pre-`map_shared_experts`).
    The one-hot variant from `tests/nightly/t3000/ccl/test_all_to_all_dispatch.py:gen_expert_mapping`
    is intentionally not used — this op expects the linearized-device-id encoding.
    """
    assert experts % num_devices == 0
    experts_per_device = experts // num_devices
    mapping = torch.empty((1, experts), dtype=torch.int32)
    for e in range(experts):
        mapping[0, e] = e // experts_per_device
    return mapping


def _get_expert_indices(batch, experts, selected_experts_k, seq_len):
    """Per-token random distinct expert ids. Shape [batch, 1, seq_len, k], int32.

    Trimmed from the `random` scheme of
    `tests/nightly/t3000/ccl/test_all_to_all_dispatch.py:get_expert_indices`
    (dropped the avg_perf / worst_perf / congestion variants that aren't relevant here).
    """
    indices = torch.empty((batch, 1, seq_len, selected_experts_k), dtype=torch.int32)
    for b in range(batch):
        for s in range(seq_len):
            picks = random.sample(range(experts), selected_experts_k)
            for k, e in enumerate(picks):
                indices[b, 0, s, k] = e
    return indices


def _on_axis_mask(indices_local, mapping, mesh_shape, mesh_coord, cluster_axis):
    """Per (token, k) bool mask: True iff the expert routed for that slot lives on this device's
    cluster axis. Matches the kernel's compute_on_axis(t, k) predicate.

    Args:
        indices_local: [tokens, 1, seq, k] int — this device's slice of expert_indices.
        mapping:       [1, experts] int — global expert→owning-device map (linearized).
        mesh_shape:    (rows, cols) of the emulated mesh.
        mesh_coord:    (row, col) of the device we're emulating.
        cluster_axis:  0 ⇒ cluster is a column; 1 ⇒ cluster is a row.
    """
    owner_device = mapping[0, indices_local.long()]  # [tokens, 1, seq, k]
    owner_row = owner_device // mesh_shape[1]
    owner_col = owner_device % mesh_shape[1]
    if cluster_axis == 0:
        return owner_col == mesh_coord[1]
    return owner_row == mesh_coord[0]


def _torch_golden_gated(
    u_local,
    s_local,
    ind_local,
    mapping,
    mesh_shape,
    mesh_coord,
    cluster_axis,
    num_replicated_devices,
    num_shared_experts,
    shared_expert_scale_bf16,
):
    """Per-device golden with shared-expert support.

    Activation layout: ``u_local`` has effective_k = select_experts_k + num_shared_experts entries
    along the reduction dim (dim 0). The first select_experts_k are routed experts (multiplied by
    on-axis-gated scores); the trailing num_shared_experts are shared experts (multiplied by a
    constant BF16 scale, no per-token gating — matches the kernel which writes
    shared_expert_scale_bf16 unconditionally into those score-tile slots).
    """
    select_experts_k = s_local.shape[-1]
    effective_k = u_local.shape[0]
    assert effective_k == select_experts_k + num_shared_experts

    # Routed contribution: gate off-axis scores to zero, then permute/mul/split-sum.
    on_axis = _on_axis_mask(ind_local, mapping, mesh_shape, mesh_coord, cluster_axis)
    gated_scores = torch.where(on_axis, s_local, torch.zeros_like(s_local))
    routed_u = u_local[:select_experts_k]
    routed_weights = gated_scores.permute(3, 1, 0, 2)
    routed_scaled = routed_u * routed_weights

    # Shared-expert contribution: constant BF16 scale, no per-token gating.
    shared_u = u_local[select_experts_k:]
    shared_scaled = shared_u * shared_expert_scale_bf16

    scaled = torch.cat([routed_scaled, shared_scaled], dim=0)
    hidden_size = scaled.shape[-1]
    split_size = hidden_size // num_replicated_devices
    assert hidden_size % split_size == 0
    num_chunks = hidden_size // split_size
    return [scaled[:, :, :, i * split_size : (i + 1) * split_size].sum(dim=0, keepdim=True) for i in range(num_chunks)]


PCC_THRESHOLD = 0.999


@pytest.mark.parametrize(
    "num_shared_experts, shared_expert_scale",
    [
        pytest.param(0, 1.0, id="no_shared"),
        pytest.param(1, 0.5, id="1_shared_scale_0p5"),
        pytest.param(2, 0.25, id="2_shared_scale_0p25"),
    ],
)
@pytest.mark.parametrize("batch_per_device", [32, 8, 3, 1])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((4, 8), (4, 8), id="16x8_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_deepseek_moe_fast_reduce_nc_fused(
    mesh_device,
    mesh_shape,
    batch_per_device,
    select_experts_k,
    seq,
    hidden_size,
    experts_per_device,
    cluster_axis,
    num_shared_experts,
    shared_expert_scale,
):
    torch.manual_seed(2005)
    random.seed(2005)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    num_replicated_devices = mesh_shape[1 - cluster_axis]
    batch = batch_per_device * num_dispatch_devices
    experts = experts_per_device * num_devices

    # Activation's reduction dim covers routed experts + trailing shared-expert slots.
    # The kernel treats indices [select_experts_k, effective_select_experts_k) as shared experts,
    # multiplying those activation slices by ``shared_expert_scale_bf16`` (no per-token gating).
    effective_select_experts_k = select_experts_k + num_shared_experts
    # Pre-quantize the scale through BF16 so the golden mirrors the kernel's BF16 storage.
    shared_expert_scale_bf16 = torch.tensor(shared_expert_scale, dtype=torch.bfloat16)

    # Memory configs (identical to test_deepseek_moe_fast_reduce_nc_single.py)
    activation_memory_config = ttnn.L1_MEMORY_CONFIG

    fast_reduce_output_memory_config = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            ttnn.Shape([1, 32, 128]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 5), ttnn.CoreCoord(3, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    replicate_mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)

    # Global tensors. Activation reduction dim now spans routed + shared expert slots.
    torch_unsqueezed_global = (
        torch.rand((effective_select_experts_k, 1, batch, hidden_size), dtype=torch.bfloat16) - 0.5
    )

    # Scores tensor only carries the routed scores; the kernel synthesizes shared-expert "scores"
    torch_expert_scores_global = torch.rand((batch, 1, seq, select_experts_k), dtype=torch.bfloat16)

    torch_expert_scores_global = torch_expert_scores_global / torch_expert_scores_global.sum(dim=-1, keepdim=True)

    # New: per-token expert routing + global expert→owning-device map.
    torch_expert_indices_global = _get_expert_indices(batch, experts, select_experts_k, seq)
    torch_expert_mapping = _gen_expert_mapping_linearized(experts, num_devices)

    per_device_goldens = []
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            mesh_coord = (m0, m1)

            t0 = (m1 if cluster_axis == 1 else m0) * batch_per_device
            t1 = t0 + batch_per_device

            u_slice = torch_unsqueezed_global[:, :, t0:t1, :].contiguous()
            s_slice = torch_expert_scores_global[t0:t1, :, :, :].contiguous()
            ind_slice = torch_expert_indices_global[t0:t1, :, :, :].contiguous()

            # Per-device golden — gated routed contribution + constant-scale shared contribution.
            per_device_goldens.append(
                _torch_golden_gated(
                    u_slice,
                    s_slice,
                    ind_slice,
                    torch_expert_mapping,
                    mesh_shape,
                    mesh_coord,
                    cluster_axis,
                    num_replicated_devices,
                    num_shared_experts,
                    shared_expert_scale_bf16,
                )
            )

    # Pad u_slice along dim=2 (tokens) to be a multiple of 32 if necessary
    current_tokens = u_slice.shape[2]
    target_tokens = ((current_tokens + 31) // 32) * 32  # Next multiple of 32
    if current_tokens < target_tokens:
        pad_amt = target_tokens - current_tokens
        pad_shape = list(u_slice.shape)
        pad_shape[2] = pad_amt
        pad_tensor = torch.zeros(pad_shape, dtype=u_slice.dtype, device=u_slice.device)
        u_slice = torch.cat([u_slice, pad_tensor], dim=2)

    def _shard_dims(dim):
        return (dim, None) if cluster_axis == 0 else (None, dim)

    # Activation: TILE layout, L1 — same as unsqueezed_output in unfused path
    tt_activation = ttnn.from_torch(
        torch_unsqueezed_global,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=activation_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, dims=_shard_dims(2)),
    )

    # Scores: ROW_MAJOR layout, DRAM — passed directly (no permute/tilize in Python)
    tt_scores_dram = ttnn.from_torch(
        torch_expert_scores_global,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, dims=_shard_dims(0)),
    )

    # New: expert_indices and expert_mapping — uint16, ROW_MAJOR, DRAM, replicated.
    tt_expert_indices = ttnn.from_torch(
        torch_expert_indices_global,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, dims=_shard_dims(0)),
    )
    tt_expert_mapping = ttnn.from_torch(
        torch_expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_mapper,
    )

    # Single fused call replacing permute + to_layout + mul + deepseek_moe_fast_reduce_nc.
    # Performs per-(t, k) on-axis gating using the expert_indices / expert_mapping tensors and
    # cluster_axis (off-axis scores are zeroed). The trailing ``num_shared_experts`` slots along
    # the activation's reduction dim are multiplied by ``shared_expert_scale`` (no gating) and
    # summed into the output.
    tt_fused_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        tt_activation,
        tt_expert_indices,
        tt_expert_mapping,
        reduce_dim=0,
        split_size=int(hidden_size // num_replicated_devices),
        cluster_axis=cluster_axis,
        output_memory_config=fast_reduce_output_memory_config,
        scores_tensor=tt_scores_dram,
        num_shared_experts=num_shared_experts,
        shared_expert_scale=shared_expert_scale,
    )

    for cidx, tt_out_list in enumerate(tt_fused_outputs):
        for didx, tt_out in enumerate(ttnn.get_device_tensors(tt_out_list)):
            tt_host = ttnn.to_torch(tt_out, dtype=torch.bfloat16)
            tt_host_slice = tt_host[:, :, 0:current_tokens, :]

            ref = per_device_goldens[didx][cidx]

            ok, msg = comp_pcc(ref, tt_host_slice, pcc=PCC_THRESHOLD)
            logger.info(f"virtual_dev={didx} mesh_coord={mesh_coord} chunk={cidx}: {msg}")
            assert ok, f"virtual_dev={didx} mesh_coord={mesh_coord} chunk={cidx} failed: {msg}"

    ttnn.deallocate(tt_activation)
    ttnn.deallocate(tt_scores_dram)
    ttnn.deallocate(tt_expert_indices)
    ttnn.deallocate(tt_expert_mapping)
    for t in tt_fused_outputs:
        ttnn.deallocate(t)
