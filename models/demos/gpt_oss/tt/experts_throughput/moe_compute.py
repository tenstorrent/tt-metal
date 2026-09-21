# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole-capable fused MoE decode built on ``ttnn.experimental.moe_compute``.

``moe_gpt`` (the fused path Wormhole uses) shards K across the DRAM-bank-aligned matmul
cores with a hardcoded 12-bank layout -- ``tiles_per_core_table[12]`` sums to 90 = 2880/32
only at 12 banks, and ``COMBINE_WIDTH_SHARD_DIM=3`` must divide the bank count. Wormhole has
12 DRAM banks, Blackhole 8, so it cannot run there. ``moe_compute`` is the arch-agnostic
successor: it derives the ring from the live DRAM-bank count and generates the shard tables
at compile time, and it ships its own host-side weight packers so none of the 12-bank layout
code in ``weights.py`` is needed.

It also subsumes more of the pipeline than ``moe_gpt`` did -- it performs the combine itself
(hence the fabric arguments) and applies the routing scores to the expert outputs -- so the
flow here is ``all_to_all_dispatch_metadata -> moe_compute -> sum over k -> all_reduce``,
with no separate ``selective_reduce_combine`` and no score multiply.
"""

import os
from dataclasses import dataclass

import torch
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

# gen_expert_mapping lives in the moe_compute test suite, which is where the op's expert->device
# mapping contract is maintained; experts_throughput/config.py already imports it the same way.
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import gen_expert_mapping

from .config import ThroughputExpertConfig


@dataclass
class MoeComputeConfig:
    """Pre-built weights, buffers and core placement for the moe_compute decode path."""

    tt_w0_w1: ttnn.Tensor
    tt_w2: ttnn.Tensor
    tt_expert_mapping: ttnn.Tensor
    dispatch_mapping: ttnn.Tensor
    dispatch_sparse: ttnn.Tensor
    dispatch_indices: ttnn.Tensor
    dispatch_scores: ttnn.Tensor
    dispatch_semaphore: object  # ttnn global semaphore
    combine_output: ttnn.Tensor
    combine_semaphore: object  # ttnn global semaphore
    mux_core_range_set: ttnn.CoreRangeSet
    cluster_axis: int
    num_links: int
    intermediate_size: int
    output_height_shard_dim: int
    topology: ttnn.Topology
    tokens_per_device: int
    total_tokens: int


def _per_device_expert_blocks(tensor_all, ring_devices, mesh_cols, experts_per_cluster, experts_per_device):
    """Lay experts out so ShardTensor2dMesh(dims=(0, 1)) hands each device its own slice.

    Device (row r, col c) owns experts [c*experts_per_cluster + r*E, +E), matching
    gen_expert_mapping's cluster_axis=0 assignment. Rows are concatenated on dim 0 and
    columns on dim 1, so a (rows, cols) 2D shard gives every device [L=1, E, ...].
    """
    rows = []
    for r in range(ring_devices):
        cols = []
        for c in range(mesh_cols):
            start = c * experts_per_cluster + r * experts_per_device
            cols.append(tensor_all[start : start + experts_per_device].unsqueeze(0))
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows, dim=0)


def _split_gate_up(state_dict, num_experts, K, N):
    """Return (w0, w1, w2, b0, b1, b2) as float torch tensors in PyTorch layout."""
    if "gate_up_proj" in state_dict:
        gate_up = state_dict["gate_up_proj"]
        w0 = gate_up[..., ::2].contiguous().float()
        w1 = gate_up[..., 1::2].contiguous().float()
    else:
        w0 = state_dict["gate_proj"].contiguous().float()
        w1 = state_dict["up_proj"].contiguous().float()
    w2 = state_dict["down_proj"].contiguous().float()

    if "gate_up_proj_bias" in state_dict:
        gub = state_dict["gate_up_proj_bias"]
        b0 = gub[..., ::2].contiguous().float()
        b1 = gub[..., 1::2].contiguous().float()
    else:
        b0 = state_dict.get("gate_proj_bias", torch.zeros(num_experts, N)).contiguous().float()
        b1 = state_dict.get("up_proj_bias", torch.zeros(num_experts, N)).contiguous().float()
    b2 = state_dict.get("down_proj_bias", torch.zeros(num_experts, K)).contiguous().float()
    return w0, w1, w2, b0, b1, b2


def _build_packed_weights(mesh_device, config, state_dict, K, N, E, ring_devices, mesh_cols, experts_per_cluster):
    """Upload the per-device raw experts, run the op's on-device packers, quantize to bfloat4_b.

    Returns host tensors so they can be cached and re-landed under the kernel's DRAM-sharded
    memory config on later runs.
    """
    w0_all, w1_all, w2_all, b0_all, b1_all, b2_all = _split_gate_up(state_dict, config.num_experts, K, N)

    def _shard_raw(t_all):
        blocks = _per_device_expert_blocks(t_all, ring_devices, mesh_cols, experts_per_cluster, E)
        return ttnn.from_torch(
            blocks.to(torch.bfloat16),
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(ring_devices, mesh_cols)),
        )

    tt_w0_raw, tt_w1_raw, tt_w2_raw = _shard_raw(w0_all), _shard_raw(w1_all), _shard_raw(w2_all)
    tt_b0_raw, tt_b1_raw, tt_b2_raw = _shard_raw(b0_all), _shard_raw(b1_all), _shard_raw(b2_all)

    tt_w0_w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_with_bias(
        tt_w0_raw, tt_w1_raw, tt_b0_raw, tt_b1_raw, L=1, E=E, K=K, N=N
    )
    tt_w2_prepped = ttnn.experimental.prepare_w2_tensor_with_bias(tt_w2_raw, tt_b2_raw, L=1, E=E, N=N, K=K)
    for t in (tt_w0_raw, tt_w1_raw, tt_w2_raw, tt_b0_raw, tt_b1_raw, tt_b2_raw):
        ttnn.deallocate(t)

    # memory_config=None keeps the quantized result on host.
    w0_w1_host = ttnn.experimental.quantize_weights_via_host(tt_w0_w1_prepped, dtype=ttnn.bfloat4_b, memory_config=None)
    w2_host = ttnn.experimental.quantize_weights_via_host(tt_w2_prepped, dtype=ttnn.bfloat4_b, memory_config=None)
    ttnn.deallocate(tt_w0_w1_prepped)
    ttnn.deallocate(tt_w2_prepped)
    return w0_w1_host, w2_host


def create_moe_compute_config(
    mesh_device,
    config: ThroughputExpertConfig,
    state_dict,
    tokens_per_device: int,
    num_links: int,
    cluster_axis: int = 0,
    topology: ttnn.Topology = ttnn.Topology.Linear,
    tensor_cache_path: str = None,
) -> MoeComputeConfig:
    """Build the weights, buffers and core placement moe_compute needs for one decoder layer."""

    K = config.hidden_size
    N = config.intermediate_size
    E = config.num_experts_per_device
    k_sel = config.num_experts_per_tok
    ring_devices = mesh_device.shape[cluster_axis]
    total_devices = mesh_device.get_num_devices()
    mesh_cols = total_devices // ring_devices
    experts_per_cluster = config.num_experts // mesh_cols
    total_tokens = tokens_per_device * ring_devices
    compute_grid = mesh_device.compute_with_storage_grid_size()

    # --- weights -----------------------------------------------------------------
    # moe_compute ships its own packers, so we only have to get the raw per-device experts
    # onto the right device; prepare_* does the interleave/pad/reorder on device and
    # quantize_weights_via_host does the bfloat4_b conversion.
    # The packed weights are expensive to build (upload -> on-device prepare -> host
    # quantize, per layer), and a warm run has no state_dict at all when the demo is invoked
    # with --skip-model-load. Cache the packed host tensors and reload them when present.
    w0_w1_cache = get_cache_file_name(tensor_cache_path, "moe_compute_w0_w1.tensorbin")
    w2_cache = get_cache_file_name(tensor_cache_path, "moe_compute_w2.tensorbin")
    have_cache = bool(w0_w1_cache) and os.path.exists(w0_w1_cache) and os.path.exists(w2_cache)

    if have_cache:
        w0_w1_host = ttnn.load_tensor(w0_w1_cache)
        w2_host = ttnn.load_tensor(w2_cache)
    else:
        if not state_dict:
            raise RuntimeError(
                f"moe_compute weights are not cached at {w0_w1_cache} and no state_dict was "
                "provided. Re-run once with weights loaded (drop --skip-model-load, or set "
                "GPT_OSS_FORCE_MODEL_LOAD=1) to build the cache."
            )
        w0_w1_host, w2_host = _build_packed_weights(
            mesh_device, config, state_dict, K, N, E, ring_devices, mesh_cols, experts_per_cluster
        )
        if w0_w1_cache:
            os.makedirs(os.path.dirname(w0_w1_cache), exist_ok=True)
            ttnn.dump_tensor(w0_w1_cache, w0_w1_host)
            ttnn.dump_tensor(w2_cache, w2_host)

    weight_mem_configs = ttnn.experimental.get_weight_mem_configs(
        mesh_device,
        num_layers=1,
        experts_per_device=E,
        hidden_size=K,
        intermediate_size=N,
        has_bias=True,
    )
    tt_w0_w1 = ttnn.to_device(w0_w1_host, mesh_device, memory_config=weight_mem_configs.w0_w1)
    tt_w2 = ttnn.to_device(w2_host, mesh_device, memory_config=weight_mem_configs.w2)

    # --- expert -> device mapping --------------------------------------------------
    # num_replicated_devices is the extent of the NON-dispatch axis (the columns when
    # cluster_axis=0), not the whole mesh: get_linearized_mesh_coord computes
    # device = (expert_within_cluster // E) * num_replicated_devices + cluster_id, which only
    # lands inside the mesh when that stride is the row length. This matches
    # _per_device_expert_blocks: expert e lives at row (e % experts_per_cluster) // E,
    # column e // experts_per_cluster.
    mapping = gen_expert_mapping(total_devices, mesh_cols, cluster_axis, config.num_experts, experts_per_cluster, E)
    tt_expert_mapping = ttnn.from_torch(
        mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_dispatch_mapping = ttnn.from_torch(
        mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- dispatch buffers (same contract as the moe_gpt path) ------------------------
    drain_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    mesh_shape = tuple(mesh_device.shape)
    dispatch_sparse = ttnn.from_torch(
        torch.zeros(ring_devices, total_tokens, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape),
    )
    metadata_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(drain_core, drain_core)}),
            [total_tokens, k_sel],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    dispatch_indices = ttnn.from_torch(
        torch.zeros(ring_devices, total_tokens, k_sel, dtype=torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=metadata_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape),
    )
    dispatch_scores = ttnn.from_torch(
        torch.zeros(ring_devices, total_tokens, k_sel, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=metadata_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape),
    )

    # --- combine placement -----------------------------------------------------------
    # width parallelism must divide both hidden/32 and the matmul ring; on Blackhole the
    # ring is 8 so this picks 2 where Wormhole's 12-core ring picks 3 (moe_gpt's fixed 3 is
    # exactly what makes it Wormhole-only).
    output_height_shard_dim = 4
    output_width_shard_dim = auto_output_width_shard_dim(K, matmul_ring_size=effective_matmul_ring_size(mesh_device))
    # The fused combine needs num_links * neighbours mux cores (2 on a 2-link Blackhole ring).
    # Placement is unconstrained -- the op positions the matmul/tilize/combine groups to avoid
    # whatever cells mux takes -- so use the same small block the moe_compute suite uses.
    mux_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])
    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim, K, mux_core_range_set=mux_core_range_set
    )
    combine_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in output_shard_cores])
    combine_semaphore = ttnn.create_global_semaphore(mesh_device, combine_core_range_set, 0)

    all_worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    dispatch_semaphore = ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)

    combine_output = ttnn.from_torch(
        torch.zeros(k_sel, total_tokens, K, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        # Tokens are split across the dispatch ring (the rows), so dim 1 shards on the row
        # axis and replicates across the tensor-parallel columns: [k, tokens_per_device, K].
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=mesh_shape),
    )

    return MoeComputeConfig(
        tt_w0_w1=tt_w0_w1,
        tt_w2=tt_w2,
        tt_expert_mapping=tt_expert_mapping,
        dispatch_mapping=tt_dispatch_mapping,
        dispatch_sparse=dispatch_sparse,
        dispatch_indices=dispatch_indices,
        dispatch_scores=dispatch_scores,
        dispatch_semaphore=dispatch_semaphore,
        combine_output=combine_output,
        combine_semaphore=combine_semaphore,
        mux_core_range_set=mux_core_range_set,
        cluster_axis=cluster_axis,
        num_links=num_links,
        intermediate_size=N,
        output_height_shard_dim=output_height_shard_dim,
        topology=topology,
        tokens_per_device=tokens_per_device,
        total_tokens=total_tokens,
    )


def moe_compute_decode_forward(
    hidden_states,
    topk_expert_indices,
    topk_expert_scores,
    config: ThroughputExpertConfig,
    mc_config: MoeComputeConfig,
    mesh_device,
    ccl_manager,
):
    """Fused MoE decode: all_to_all_dispatch_metadata -> moe_compute -> sum(k) -> all_reduce.

    moe_compute performs the combine itself and applies the routing scores, so unlike the
    moe_gpt flow there is no selective_reduce_combine and no post-combine score multiply.

    Returns [1, 1, tokens_per_device, hidden_size].
    """
    cluster_axis = mc_config.cluster_axis
    k_sel = config.num_experts_per_tok
    # Derive the token count from the actual input rather than the config: callers hand this
    # [1, 1, tokens, H] or [1, tokens, 1, H], and the unit tests exercise token counts that do
    # not always equal max_local_batch_size. (fused_decode derives it the same way.)
    input_shape = hidden_states.shape
    tokens_per_device = input_shape[0] * input_shape[2]
    total_tokens = tokens_per_device * mesh_device.shape[cluster_axis]

    # all_to_all_dispatch_metadata needs ROW_MAJOR L1 for all three inputs.
    def _rm_l1(t):
        if t.layout != ttnn.ROW_MAJOR_LAYOUT:
            t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        if t.memory_config().buffer_type != ttnn.BufferType.L1:
            t = ttnn.clone(t, memory_config=ttnn.L1_MEMORY_CONFIG)
        return t

    # all_to_all_dispatch_metadata consumes BFLOAT16 activations. The decoder hands the MLP
    # bfloat8_b, and because the first call in a process may have populated the program cache
    # with a bfloat16 program, a later bfloat8_b call can slip past validation and be read as
    # bfloat16 -- silently, as wrong values rather than an error. Measured: bf16 input gives
    # experts PCC 0.983, bfloat8_b gives 0.726. Convert explicitly.
    if hidden_states.dtype != ttnn.bfloat16:
        hidden_states = ttnn.typecast(hidden_states, dtype=ttnn.bfloat16)
    hidden_states = _rm_l1(hidden_states)
    hidden_states = ttnn.reshape(hidden_states, (tokens_per_device, 1, 1, config.hidden_size))
    # all_to_all_dispatch_metadata requires UINT16 indices. The fused Wormhole router emits
    # those natively, but Blackhole runs the generic linear+topk router, whose ttnn.topk
    # indices are a wider integer type -- feeding those straight through reads as garbage
    # expert ids. The dense path normalises the same way (experts/decode.py).
    if topk_expert_indices.dtype != ttnn.uint16:
        topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint32)
        topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint16)
    tt_indices = ttnn.reshape(_rm_l1(topk_expert_indices), (tokens_per_device, 1, 1, k_sel))
    tt_scores = ttnn.reshape(_rm_l1(topk_expert_scores), (tokens_per_device, 1, 1, k_sel))
    # Keep a copy of the pre-dispatch scores: the combine output is unweighted.
    scores_for_weighting = ttnn.clone(tt_scores, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    (tt_sparse, tt_disp_indices, tt_disp_scores) = ttnn.experimental.all_to_all_dispatch_metadata(
        hidden_states,
        tt_indices,
        tt_scores,
        mc_config.dispatch_mapping,
        cluster_axis=cluster_axis,
        num_links=mc_config.num_links,
        output_tensors=(mc_config.dispatch_sparse, mc_config.dispatch_indices, mc_config.dispatch_scores),
        cross_device_semaphore=mc_config.dispatch_semaphore,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )
    ttnn.deallocate(hidden_states)

    # The combine accumulates into its output tensor, so it must start zeroed on every call --
    # reusing a persistent buffer leaves the previous step's partial sums in place. (The
    # moe_gpt path zeroes the same way.)
    combine_out = ttnn.moreh_full(
        shape=list(mc_config.combine_output.shape),
        fill_value=0,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    outputs = ttnn.experimental.moe_compute(
        tt_sparse,
        tt_disp_indices,
        tt_disp_scores,
        mc_config.tt_expert_mapping,
        mc_config.tt_w0_w1,
        mc_config.tt_w2,
        layer_id=0,
        output_height_shard_dim=mc_config.output_height_shard_dim,
        intermediate_size=mc_config.intermediate_size,
        has_bias=True,
        cluster_axis=cluster_axis,
        topology=mc_config.topology,
        num_links=mc_config.num_links,
        mux_core_range_set=mc_config.mux_core_range_set,
        optional_output_tensor=combine_out,
        optional_cross_device_semaphore=mc_config.combine_semaphore,
        activation_type=MoEActivationFunction.SWIGLU,
    )
    # moe_compute returns (per_expert, activation, e_t, _, matmul, combine); everything but
    # the combine result is an L1 intermediate we do not need.
    l1_per_expert, l1_activation, l1_e_t, _unused, l1_matmul, combined = outputs
    for t in (l1_per_expert, l1_activation, l1_e_t, l1_matmul):
        ttnn.deallocate(t)

    # [k, tokens_per_device, hidden] -> sum over the k expert slots. Scores are already
    # applied inside moe_compute, so this is a plain reduction.
    combined = ttnn.to_layout(combined, ttnn.TILE_LAYOUT)
    combined = ttnn.reshape(combined, (k_sel, 1, tokens_per_device, config.hidden_size))

    # Despite the op docstring describing expert_scores as "applied to the expert outputs",
    # moe_compute uses them for routing metadata only and returns UNWEIGHTED per-slot outputs
    # -- same as moe_gpt, which is why the moe_gpt flow also multiplies after the combine.
    # Measured: without this multiply the experts PCC is 0.859, with it 0.983.
    scores = ttnn.to_layout(scores_for_weighting, ttnn.TILE_LAYOUT)
    scores = ttnn.reshape(scores, (tokens_per_device, 1, 1, k_sel))
    scores = ttnn.transpose(scores, 0, 3)  # -> [k, 1, 1, tokens]
    scores = ttnn.transpose(scores, 2, 3)  # -> [k, 1, tokens, 1]
    combined = ttnn.mul(combined, scores)
    ttnn.deallocate(scores)

    summed = ttnn.sum(combined, dim=0, keepdim=True)
    ttnn.deallocate(combined)
    summed = ttnn.typecast(summed, ttnn.bfloat8_b)

    # Tensor-parallel reduction across the other mesh axis.
    return ttnn.all_reduce(
        summed,
        num_links=ccl_manager.num_links,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
