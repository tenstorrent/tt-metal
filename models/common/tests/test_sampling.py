# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.sampling import LogProbsCalculator
from models.common.utility_functions import comp_pcc

# ---------------------------------------------------------------------------
# Legacy single-token logprobs tests (argmax path, multi-device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_calculation(shape, mesh_device):
    """Legacy test: single-token logprobs for argmax indices on multi-device."""
    seed = 1234
    torch.manual_seed(seed)

    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )
    # Push inputs to device
    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_num_logprobs(1)
    tt_log_probs = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]

    # Calculate log-probs for each user on each chip using torch
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = torch.reshape(log_probs_torch_argmax, (1, 1, 1, 32))

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_returns_none_when_disabled(shape, mesh_device):
    """Test that calculate_log_probs returns None when num_logprobs is 0."""
    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Log probs disabled (default) - should return None
    log_probs_calculator.set_num_logprobs(0)
    result = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    assert result is None, f"Expected None when log_probs disabled, got {type(result)}"

    # Log probs enabled - should return a tensor (not None)
    log_probs_calculator.set_num_logprobs(5)
    num_devices = mesh_device.get_num_devices()
    result = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    if num_devices in (8, 32) and log_probs_calculator.num_devices_for_sharding >= 2:
        assert result is not None, "Expected tensor when log_probs enabled on supported device"
    else:
        assert result is None, "Expected None on unsupported device count"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 16032],  # llama on TG with 8 chips sharded vocab
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            }
        ),
    ],
    indirect=True,
    ids=["fabric_linear"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_log_probs_with_sub_core_grids_on_galaxy(shape, mesh_device):
    """Legacy test: single-token logprobs on Galaxy mesh."""
    seed = 1234
    torch.manual_seed(seed)

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    log_probs_calculator = LogProbsCalculator(mesh_device, sub_core_grids)

    torch_tensor = torch.randn(shape)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    if mesh_device.get_num_devices() == 8:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    elif mesh_device.get_num_devices() == 32:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape))
    else:
        raise ValueError(f"Unsupported number of devices: {mesh_device.get_num_devices()}")

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_num_logprobs(1)
    tt_log_probs = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # slice from (1,1,32,256) -> (1,1,1,32)
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]

    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = torch.reshape(log_probs_torch_argmax, (1, 1, 1, 32))

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


# ---------------------------------------------------------------------------
# New top-k logprobs tests (calculate_top_k_log_probs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_logprobs", [1, 5, 10, 20])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_top_k_log_probs_t3k(num_logprobs, shape, mesh_device):
    """
    Test top-k logprobs on T3K: computes logprobs for top-32 tokens per user
    on device, then verifies against torch log_softmax on host.

    This tests the primary logprobs path used during non-argmax sampling.
    """
    seed = 1234
    torch.manual_seed(seed)
    batch_size = shape[2]
    vocab_size = shape[3]
    num_devices = mesh_device.get_num_devices()
    vocab_per_device = vocab_size // num_devices
    top_k = 32

    log_probs_calculator = LogProbsCalculator(mesh_device, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(vocab_size)]

    # Compute reference logprobs using torch
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)

    # Get the global top-K tokens and their logit values (simulating what TTSampling does)
    # In practice, TTSampling does per-device topk then all-gather.
    # For testing, we compute global topk on the torch tensor.
    topk_values_torch, topk_indices_torch = torch.topk(torch_tensor, top_k, dim=-1)
    # topk_values_torch shape: (1, 1, batch_size, top_k)

    # Get reference logprobs for the top-k tokens
    ref_topk_logprobs = torch.gather(log_probs_torch, dim=-1, index=topk_indices_torch.long())

    # Push full logits to device (sharded across devices)
    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Push top-k values to device (replicated, as they would be after all-gather)
    topk_values_tt = ttnn.from_torch(
        topk_values_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Push top-k indices to device (replicated)
    topk_indices_tt = ttnn.from_torch(
        topk_indices_torch.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_num_logprobs(num_logprobs)
    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected non-None result when logprobs enabled"

    lp_values_tt, lp_indices_tt = result

    # Convert back to torch
    lp_values_host = ttnn.to_torch(lp_values_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    lp_values_host = lp_values_host[:1, :, :, :]  # take first device's copy

    lp_indices_host = ttnn.to_torch(lp_indices_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    lp_indices_host = lp_indices_host[:1, :, :, :]

    # Compare logprobs values (all top_k columns)
    passing, pcc = comp_pcc(ref_topk_logprobs, lp_values_host, pcc=0.99)
    print(f"Top-k logprobs pcc={pcc}, num_logprobs={num_logprobs}")
    assert passing, f"Top-k logprobs PCC failed: {pcc}"

    # Verify that selecting the first num_logprobs still matches
    selected_ref = ref_topk_logprobs[:, :, :, :num_logprobs]
    selected_tt = lp_values_host[:, :, :, :num_logprobs]
    passing, pcc = comp_pcc(selected_ref, selected_tt, pcc=0.99)
    print(f"Selected {num_logprobs} logprobs pcc={pcc}")
    assert passing, f"Selected logprobs PCC failed: {pcc}"


@pytest.mark.parametrize("num_logprobs", [1, 5, 10, 20])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 16032],  # llama on TG with 8 chips sharded vocab
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            }
        ),
    ],
    indirect=True,
    ids=["fabric_linear"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_top_k_log_probs_galaxy(num_logprobs, shape, mesh_device):
    """
    Test top-k logprobs on Galaxy mesh: computes logprobs for top-32 tokens
    per user on device, then verifies against torch log_softmax on host.
    """
    seed = 1234
    torch.manual_seed(seed)
    batch_size = shape[2]
    vocab_size = shape[3]
    top_k = 32

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    log_probs_calculator = LogProbsCalculator(mesh_device, sub_core_grids, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(vocab_size)]

    # Reference logprobs
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    topk_values_torch, topk_indices_torch = torch.topk(torch_tensor, top_k, dim=-1)
    ref_topk_logprobs = torch.gather(log_probs_torch, dim=-1, index=topk_indices_torch.long())

    if mesh_device.get_num_devices() == 8:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    elif mesh_device.get_num_devices() == 32:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape))
    else:
        raise ValueError(f"Unsupported number of devices: {mesh_device.get_num_devices()}")

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        topk_values_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        topk_indices_torch.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_num_logprobs(num_logprobs)
    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected non-None result when logprobs enabled"

    lp_values_tt, lp_indices_tt = result
    lp_values_host = ttnn.to_torch(lp_values_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    lp_values_host = lp_values_host[:1, :, :, :]

    # Compare all top-k logprobs
    passing, pcc = comp_pcc(ref_topk_logprobs, lp_values_host, pcc=0.99)
    print(f"Galaxy top-k logprobs pcc={pcc}, num_logprobs={num_logprobs}")
    assert passing, f"Galaxy top-k logprobs PCC failed: {pcc}"


# ---------------------------------------------------------------------------
# num_logprobs validation tests
# ---------------------------------------------------------------------------


def test_set_num_logprobs_validation():
    """Test that set_num_logprobs validates inputs correctly (no device needed)."""

    class FakeMeshDevice:
        def __init__(self):
            self.shape = [1, 1]

        def get_num_devices(self):
            return 1

    # We can't fully instantiate LogProbsCalculator without a real mesh_device,
    # so we test the validation logic directly.

    # Test: values above 20 should be capped
    calc = LogProbsCalculator.__new__(LogProbsCalculator)
    calc.num_logprobs = 0
    calc.num_devices_for_sharding = 1
    calc.common_args = {}

    # Direct num_logprobs setting
    calc.set_num_logprobs(5)
    assert calc.num_logprobs == 5

    calc.set_num_logprobs(20)
    assert calc.num_logprobs == 20

    # Cap at 20
    calc.set_num_logprobs(25)
    assert calc.num_logprobs == 20

    calc.set_num_logprobs(100)
    assert calc.num_logprobs == 20

    # Disable
    calc.set_num_logprobs(0)
    assert calc.num_logprobs == 0

    calc.set_num_logprobs(None)
    assert calc.num_logprobs == 0

    # List input: use max
    calc.set_num_logprobs([0, 5, 3])
    assert calc.num_logprobs == 5

    calc.set_num_logprobs([0, 0, 0])
    assert calc.num_logprobs == 0

    calc.set_num_logprobs([25, 10])
    assert calc.num_logprobs == 20  # capped

    # Negative values should assert
    with pytest.raises(AssertionError):
        calc.set_num_logprobs(-1)

    with pytest.raises(AssertionError):
        calc.set_num_logprobs([-5, 3])


def test_top_k_log_probs_returns_none_when_disabled():
    """Test that calculate_top_k_log_probs returns None when num_logprobs is 0."""
    calc = LogProbsCalculator.__new__(LogProbsCalculator)
    calc.num_logprobs = 0
    calc.num_devices_for_sharding = 1
    calc.common_args = {}

    result = calc.calculate_top_k_log_probs(None, None, None)
    assert result is None
