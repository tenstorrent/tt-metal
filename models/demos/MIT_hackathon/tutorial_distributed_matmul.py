"""
Tutorial: Distributed Matmul on a Tenstorrent Mesh — Column-Parallel & Row-Parallel
====================================================================================

This file demonstrates how to perform tensor-parallel (TP) matmul on a (1, 4)
Tenstorrent device mesh.  The two patterns shown here — column-parallel and
row-parallel — are the same primitives that PyTorch's FSDP and tensor-parallel
libraries build on top of.

Background: Collectives in PyTorch vs TT-NN
--------------------------------------------

PyTorch (torch.distributed)          TT-NN equivalent
─────────────────────────────        ─────────────────
torch.distributed.all_gather   →     ttnn.all_gather
torch.distributed.all_reduce   →     ttnn.all_reduce
torch.distributed.reduce_scatter →   ttnn.reduce_scatter
Shard a tensor to ranks        →     ttnn.ShardTensorToMesh(mesh, dim=…)
Replicate to all ranks         →     ttnn.ReplicateTensorToMesh(mesh)
Concat shards back             →     ttnn.ConcatMeshToTensor(mesh, dim=…)

Connection to PyTorch FSDP
--------------------------

FSDP shards *all* model parameters across data-parallel ranks and uses
collectives to reconstruct them on the fly:

    ┌─────────────────────────────────────────────────────────────┐
    │  FSDP forward pass (per layer)                              │
    │                                                             │
    │  1. all_gather  – reconstruct full weight from shards       │
    │  2. matmul      – compute with full weight                  │
    │  3. discard     – free the unsharded weight to save memory  │
    │                                                             │
    │  FSDP backward pass (per layer)                             │
    │                                                             │
    │  1. all_gather  – reconstruct full weight again             │
    │  2. local grad  – compute gradient w.r.t. local input      │
    │  3. reduce_scatter – reduce grads, each rank keeps a shard  │
    └─────────────────────────────────────────────────────────────┘

The two tests below show the *compute* half of this story using TT-NN on
real Tenstorrent hardware:

• **Column-parallel** (test 1): weights sharded on the *output* dimension.
  This is what happens when FSDP shards an nn.Linear's weight along dim=0
  (output features).  After the local matmul, each device holds a *slice*
  of the output; an `all_gather` stitches the slices together.

• **Row-parallel** (test 2): weights sharded on the *input* dimension.
  Each device computes a partial dot-product; an `all_reduce` sums the
  partials.  FSDP uses this pattern when sharding along dim=1 (input
  features) of a weight matrix.


PyTorch pseudo-code (single GPU simulation of what FSDP does)
-------------------------------------------------------------

    # Column-parallel (shard output dim, then all_gather)
    weight_shards = weight.chunk(world_size, dim=0)          # shard N
    local_out     = input @ weight_shards[rank].T             # partial output
    full_out      = torch.distributed.all_gather(local_out)   # [M, N]

    # Row-parallel (shard input dim, then all_reduce)
    weight_shards = weight.chunk(world_size, dim=1)          # shard K
    input_shards  = input.chunk(world_size, dim=-1)          # shard K
    local_out     = input_shards[rank] @ weight_shards[rank].T  # partial sum
    full_out      = torch.distributed.all_reduce(local_out)     # [M, N]


TT-NN equivalents (runnable tests below)
-----------------------------------------

    # Column-parallel
    tt_input   = ttnn.from_torch(input,   mesh_mapper=ReplicateTensorToMesh(mesh))
    tt_weights = ttnn.from_torch(weights, mesh_mapper=ShardTensorToMesh(mesh, dim=3))
    tt_out     = ttnn.matmul(tt_input, tt_weights)     # each device: [M, N/4]
    tt_out     = ttnn.all_gather(tt_out, dim=3)        # every device: [M, N]

    # Row-parallel
    tt_input   = ttnn.from_torch(input,   mesh_mapper=ShardTensorToMesh(mesh, dim=3))
    tt_weights = ttnn.from_torch(weights, mesh_mapper=ShardTensorToMesh(mesh, dim=2))
    tt_out     = ttnn.matmul(tt_input, tt_weights)     # each device: [M, N] partial
    tt_out     = ttnn.all_reduce(tt_out)               # every device: [M, N] full

Running
-------
    pytest models/demos/MIT_hackathon/tests/test_matmul_1d_sharded.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: PyTorch reference that simulates the distributed matmul on CPU
# ═══════════════════════════════════════════════════════════════════════════════


def torch_column_parallel_matmul(input_tensor, weight_tensor, world_size):
    """
    Simulate column-parallel matmul across `world_size` ranks.

    Equivalent to what FSDP does when nn.Linear.weight is sharded on dim=0
    (the output-features dimension).

    Each rank holds weight[:, n_start:n_end] and the full input.
    After local matmul the outputs are all_gathered.
    """
    weight_shards = weight_tensor.chunk(world_size, dim=-1)  # split N
    local_outputs = [input_tensor @ ws for ws in weight_shards]
    full_output = torch.cat(local_outputs, dim=-1)  # all_gather on N
    return full_output


def torch_row_parallel_matmul(input_tensor, weight_tensor, world_size):
    """
    Simulate row-parallel matmul across `world_size` ranks.

    Equivalent to sharding nn.Linear.weight on dim=1 (input features).
    Each rank holds input[:, k_start:k_end] and weight[k_start:k_end, :].
    After local matmul the partial sums are all_reduced.
    """
    input_shards = input_tensor.chunk(world_size, dim=-1)  # split K
    weight_shards = weight_tensor.chunk(world_size, dim=-2)  # split K
    partial_products = [xs @ ws for xs, ws in zip(input_shards, weight_shards)]
    full_output = sum(partial_products)  # all_reduce (sum)
    return full_output


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_mesh_output(tt_result, torch_reference, test_name, pcc_threshold=0.99):
    """Check that every device in the mesh holds the correct result."""
    for i, device_tensor in enumerate(ttnn.get_device_tensors(tt_result)):
        tt_torch = device_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        pcc = torch.corrcoef(torch.stack([tt_torch.flatten().float(), torch_reference.flatten().float()]))[0, 1].item()
        logger.info(f"[{test_name}] Device {i}: PCC = {pcc:.6f}")
        assert pcc > pcc_threshold, f"PCC too low on device {i}: {pcc:.6f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Column-Parallel Matmul  (FSDP: shard output dim → all_gather)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  ┌────────┐   ┌────────┐        ┌────────┐       ┌────────────────┐
#  │ Dev 0  │   │ Dev 1  │        │ Dev 2  │       │    Dev 3       │
#  │        │   │        │        │        │       │                │
#  │ X full │   │ X full │        │ X full │       │   X full       │
#  │ W[:,0] │   │ W[:,1] │        │ W[:,2] │       │   W[:,3]       │
#  │        │   │        │        │        │       │                │
#  │ Y[:,0] │   │ Y[:,1] │        │ Y[:,2] │       │   Y[:,3]       │
#  └───┬────┘   └───┬────┘        └───┬────┘       └────┬───────────┘
#      │            │                 │                  │
#      └────────────┴─────────────────┴──────────────────┘
#                          all_gather (dim=N)
#                               ↓
#                    Y_full on every device
#


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_grid")], indirect=True)
def test_matmul_1d_sharded_outer_dim(mesh_device):
    """
    Column-parallel matmul on a (1, 4) mesh.

    FSDP analogy: nn.Linear.weight is sharded along output features (dim=0).
    Before compute FSDP would all_gather the weight; here we instead keep the
    weight sharded and all_gather the *output* — mathematically equivalent,
    and more communication-efficient when N >> M.

    Shapes (4D for TT-NN tile layout):
        input  : [1, 1, M, K]   — replicated on all 4 devices
        weights: [1, 1, K, N]   — sharded on N (dim=3), each device holds [1,1,K,N/4]
        local  : [1, 1, M, N/4] — per-device matmul result
        output : [1, 1, M, N]   — after all_gather on dim=3
    """
    torch.manual_seed(0)
    num_devices = 4
    M, K, N = 32, 512, 256
    assert N % num_devices == 0

    # ── Step 0: PyTorch golden reference ─────────────────────────────────
    torch_input = torch.randn(1, 1, M, K).bfloat16()
    torch_weights = torch.randn(1, 1, K, N).bfloat16()

    torch_output = torch_input @ torch_weights
    torch_output_tp = torch_column_parallel_matmul(torch_input, torch_weights, num_devices)
    assert torch.allclose(
        torch_output, torch_output_tp, atol=1.0, rtol=0.05
    ), "Sanity: column-parallel matches full matmul"

    # ── Step 1: Distribute tensors to the TT mesh ───────────────────────
    #
    # ReplicateTensorToMesh  →  same as torch.distributed.broadcast
    # ShardTensorToMesh      →  same as torch.distributed.scatter / chunk
    #
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    # ── Step 2: Local matmul on each device ──────────────────────────────
    # Each device: [1,1,M,K] @ [1,1,K,N/4] → [1,1,M,N/4]
    tt_output = ttnn.matmul(tt_input, tt_weights)

    # ── Step 3: all_gather to reconstruct full output ────────────────────
    # Equivalent to: torch.distributed.all_gather along the N dimension.
    # After this, every device holds the complete [1,1,M,N] tensor.
    tt_output_gathered = ttnn.all_gather(tt_output, dim=3, num_links=1)

    # ── Step 4: Validate against PyTorch reference ───────────────────────
    validate_mesh_output(tt_output_gathered, torch_output, "column_parallel")
    logger.info("test_matmul_1d_sharded_outer_dim (column-parallel) passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Row-Parallel Matmul  (FSDP: shard input dim → all_reduce)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  ┌────────┐   ┌────────┐        ┌────────┐       ┌────────────────┐
#  │ Dev 0  │   │ Dev 1  │        │ Dev 2  │       │    Dev 3       │
#  │        │   │        │        │        │       │                │
#  │ X[:,0] │   │ X[:,1] │        │ X[:,2] │       │   X[:,3]       │
#  │ W[0,:] │   │ W[1,:] │        │ W[2,:] │       │   W[3,:]       │
#  │        │   │        │        │        │       │                │
#  │ P_0    │   │ P_1    │        │ P_2    │       │   P_3          │
#  └───┬────┘   └───┬────┘        └───┬────┘       └────┬───────────┘
#      │            │                 │                  │
#      └────────────┴─────────────────┴──────────────────┘
#                       all_reduce (sum)
#                            ↓
#               Y = P_0 + P_1 + P_2 + P_3
#                   on every device
#


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_grid")], indirect=True)
def test_matmul_1d_sharded_inner_dim(mesh_device):
    """
    Row-parallel matmul on a (1, 4) mesh.

    FSDP analogy: nn.Linear.weight is sharded along input features (dim=1).
    Each rank holds a slice of the input *and* the corresponding rows of the
    weight.  The local matmul produces a *partial sum*; an all_reduce (sum)
    across ranks yields the full result.

    This is also the pattern used by Megatron-LM's RowParallelLinear and by
    FSDP when it needs to overlap communication with the backward pass.

    Shapes:
        input  : [1, 1, M, K]   — sharded on K (dim=3), each device holds [1,1,M,K/4]
        weights: [1, 1, K, N]   — sharded on K (dim=2), each device holds [1,1,K/4,N]
        local  : [1, 1, M, N]   — partial dot-product on each device
        output : [1, 1, M, N]   — after all_reduce (sum)
    """
    torch.manual_seed(0)
    num_devices = 4
    M, K, N = 32, 512, 256
    assert K % num_devices == 0

    # ── Step 0: PyTorch golden reference ─────────────────────────────────
    torch_input = torch.randn(1, 1, M, K).bfloat16()
    torch_weights = torch.randn(1, 1, K, N).bfloat16()

    torch_output = torch_input @ torch_weights
    torch_output_tp = torch_row_parallel_matmul(torch_input, torch_weights, num_devices)
    assert torch.allclose(
        torch_output, torch_output_tp, atol=1.0, rtol=0.05
    ), "Sanity: row-parallel matches full matmul"

    # ── Step 1: Distribute tensors to the TT mesh ───────────────────────
    #
    # Both input and weights are sharded on the K (contraction) dimension.
    # ShardTensorToMesh(dim=3) on input  → splits columns of the activation
    # ShardTensorToMesh(dim=2) on weight → splits rows of the weight matrix
    #
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    tt_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    # ── Step 2: Local matmul on each device ──────────────────────────────
    # Each device: [1,1,M,K/4] @ [1,1,K/4,N] → [1,1,M,N]  (partial product)
    # Note: the output shape is the *same* on every device, but each holds
    # only a partial sum of the full dot product.
    tt_output = ttnn.matmul(tt_input, tt_weights)

    # ── Step 3: all_reduce to sum partial products ───────────────────────
    # Equivalent to: torch.distributed.all_reduce(local_out, op=ReduceOp.SUM)
    # After this, every device holds the complete [1,1,M,N] result.
    tt_output_reduced = ttnn.all_reduce(tt_output)

    # ── Step 4: Validate against PyTorch reference ───────────────────────
    validate_mesh_output(tt_output_reduced, torch_output, "row_parallel")
    logger.info("test_matmul_1d_sharded_inner_dim (row-parallel) passed!")
