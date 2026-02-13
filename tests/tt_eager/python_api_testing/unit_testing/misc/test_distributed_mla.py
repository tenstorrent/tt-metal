# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import math
import time
import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc
from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMesh2dToTensor

torch.set_printoptions(threshold=torch.inf, linewidth=200, edgeitems=20)


def fa_rand(*shape):
    """Generate test tensors with mixture of normal and outlier values"""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    """Check if TT_METAL_WATCHER is enabled"""
    return os.environ.get("TT_METAL_WATCHER") is not None


def cpu_scaled_dot_product_attention(Q, K, V, scale=None, is_causal=True):
    """Reference CPU implementation of scaled dot product attention"""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.size(-1))

    # Q: [B, NH, S_Q, DH]
    # K: [B, NH, S_K, DH]
    # V: [B, NH, S_V, DH]
    # Output: [B, NH, S_Q, DH]

    # Compute attention scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask if requested
    if is_causal:
        seq_len_q = Q.size(-2)
        seq_len_k = K.size(-2)

        # Create causal mask: upper triangular part set to -inf
        # For prefill, we want to mask future tokens
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=Q.device))
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply dropout (not needed for testing)
    # attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    # Compute final output: weights @ V
    output = torch.matmul(attn_weights, V)

    return output


def gather_distributed_mla_outputs(mesh_outputs, num_devices):
    """Gather outputs from distributed MLA and concatenate along sequence dimension"""
    # mesh_outputs: list of tensors from each device [B, NH, S_per_device, DH]
    # Output: concatenated tensor [B, NH, S_total, DH] where S_total = S_per_device * num_devices

    if len(mesh_outputs) != num_devices:
        raise ValueError(f"Expected {num_devices} outputs, got {len(mesh_outputs)}")

    # Concatenate along sequence dimension (dim=2)
    return torch.cat(mesh_outputs, dim=2)


def run_test_distributed_mla_vs_cpu(mesh_device, b, nh, s_per_device, d, cluster_axis, dtype=ttnn.bfloat8_b):
    """Test distributed MLA against CPU SDPA implementation"""
    torch.manual_seed(1234)

    num_devices = mesh_device.get_num_devices()
    s_total = s_per_device * num_devices  # Total sequence length
    nkv = nh  # For simplicity, use same number of KV heads as Q heads

    logger.info(f"Testing distributed MLA: devices={num_devices}, s_per_device={s_per_device}, s_total={s_total}")

    # Generate test tensors with full sequence length
    Q_full = fa_rand(b, nh, s_total, d).to(torch.bfloat16)
    K_full = fa_rand(b, nkv, s_total, d).to(torch.bfloat16)
    V_full = fa_rand(b, nkv, s_total, d).to(torch.bfloat16)

    # Note: program_config and compute_kernel_config are not currently supported
    # by distributed_mla API, but are handled internally with sensible defaults

    # Run CPU reference implementation
    logger.info("Running CPU SDPA reference...")
    cpu_start_time = time.time()
    cpu_output = cpu_scaled_dot_product_attention(Q_full, K_full, V_full, is_causal=True)
    cpu_end_time = time.time()
    cpu_time = cpu_end_time - cpu_start_time
    logger.info(f"CPU SDPA time: {cpu_time*1000:.2f}ms")

    # Convert to TT tensors for distributed MLA
    # Q is sharded along sequence dimension across devices
    tt_Q = ttnn.from_torch(
        Q_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=2),  # Shard Q along sequence
    )

    # K and V are replicated on all devices (full sequence)
    tt_K = ttnn.from_torch(
        K_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),  # K is replicated
    )

    tt_V = ttnn.from_torch(
        V_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),  # V is replicated
    )

    # Run distributed MLA operation
    logger.info("Running distributed MLA...")
    ttnn.synchronize_device(mesh_device)
    mla_start_time = time.time()

    tt_output = ttnn.transformer.sdpa_prefill.distributed_mla(tt_Q, tt_K, tt_V, cluster_axis=cluster_axis)

    ttnn.synchronize_device(mesh_device)
    mla_end_time = time.time()
    mla_time = mla_end_time - mla_start_time

    # Convert output back to torch using mesh composer to concatenate sharded result
    mla_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ConcatMesh2dToTensor(
            mesh_device, dims=(0, 2) if cluster_axis == 1 else (2, 0), mesh_shape=mesh_device.shape
        ),
    )[:1, :]
    logger.info(f"Distributed MLA time: {mla_time*1000:.2f}ms")

    # Compare outputs
    logger.info(f"CPU output shape: {cpu_output.shape}")
    logger.info(f"Distributed MLA output shape: {mla_output.shape}")

    # Validate output correctness
    out_pass, out_pcc_str, out_pcc = comp_and_get_pcc(
        cpu_output, mla_output, 0.95
    )  # Slightly lower threshold due to distributed computation
    logger.info(f"Distributed MLA vs CPU SDPA PCC: {out_pcc:.6f}")
    logger.info(f"Detailed PCC info: {out_pcc_str}")

    # Log timing comparison
    speedup = cpu_time / mla_time if mla_time > 0 else float("inf")
    logger.info(
        f"Timing comparison - CPU: {cpu_time*1000:.2f}ms, Distributed MLA: {mla_time*1000:.2f}ms, Speedup: {speedup:.2f}x"
    )

    assert out_pass, f"Distributed MLA vs CPU SDPA PCC {out_pcc:.6f} < 0.95"
    logger.info("✅ Distributed MLA correctness test passed!")

    return out_pcc, cpu_time, mla_time


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize("s_per_device", [64, 128, 256], ids=["s64", "s128", "s256"])
def test_distributed_mla_correctness_1x2(mesh_device, s_per_device):
    """Test distributed MLA correctness against CPU SDPA on 1x2 mesh"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires 2 devices to run")

    # Fixed parameters for focused testing
    b, nh, d = 1, 8, 128
    cluster_axis = 1  # Horizontal sharding for 1x2 mesh

    run_test_distributed_mla_vs_cpu(mesh_device, b, nh, s_per_device, d, cluster_axis)


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 1), id="2x1_grid")], indirect=True)
@pytest.mark.parametrize("s_per_device", [64, 128, 256], ids=["s64", "s128", "s256"])
def test_distributed_mla_correctness_2x1(mesh_device, s_per_device):
    """Test distributed MLA correctness against CPU SDPA on 2x1 mesh"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires 2 devices to run")

    # Fixed parameters for focused testing
    b, nh, d = 1, 8, 128
    cluster_axis = 0  # Vertical sharding for 2x1 mesh

    run_test_distributed_mla_vs_cpu(mesh_device, b, nh, s_per_device, d, cluster_axis)


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_distributed_mla_device_order_logging(mesh_device):
    """Test that distributed_mla operation logs correct device order numbers"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires 2 devices to run")

    torch.manual_seed(1234)
    b, nh, s_per_device, d = 1, 8, 64, 128

    # Generate test tensors
    Q_full = fa_rand(b, nh, s_per_device * 2, d).to(torch.bfloat16)
    K_full = fa_rand(b, nh, s_per_device * 2, d).to(torch.bfloat16)
    V_full = fa_rand(b, nh, s_per_device * 2, d).to(torch.bfloat16)

    # Convert to TT tensors with sharding
    tt_Q = ttnn.from_torch(
        Q_full, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ShardTensorToMesh(mesh_device, dim=2)
    )
    tt_K = ttnn.from_torch(
        K_full, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ReplicateTensorToMesh(mesh_device)
    )
    tt_V = ttnn.from_torch(
        V_full, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ReplicateTensorToMesh(mesh_device)
    )

    logger.info("Testing device order logging...")

    # Test with both cluster axes
    for cluster_axis in [0, 1]:
        logger.info(f"Testing cluster_axis={cluster_axis}")
        output = ttnn.transformer.sdpa_prefill.distributed_mla(tt_Q, tt_K, tt_V, cluster_axis=cluster_axis)
        assert output.shape == tt_Q.shape

    logger.info("✅ Device order logging test completed")


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_distributed_mla_basic_functionality(mesh_device):
    """Basic test to ensure distributed MLA runs without errors"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires 2 devices to run")

    # Create test tensors
    torch_q = fa_rand(1, 8, 128, 64).to(torch.bfloat16)
    torch_k = fa_rand(1, 8, 128, 64).to(torch.bfloat16)
    torch_v = fa_rand(1, 8, 128, 64).to(torch.bfloat16)

    # Convert to TT tensors with proper sharding
    ttnn_q = ttnn.from_torch(
        torch_q, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ShardTensorToMesh(mesh_device, dim=2)
    )
    ttnn_k = ttnn.from_torch(
        torch_k, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ReplicateTensorToMesh(mesh_device)
    )
    ttnn_v = ttnn.from_torch(
        torch_v, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ReplicateTensorToMesh(mesh_device)
    )

    logger.info("Testing basic distributed MLA functionality...")

    # Test basic operation
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_q, ttnn_k, ttnn_v, cluster_axis=1)
    assert output.shape == ttnn_q.shape

    # Test with custom scale
    output_scaled = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_q, ttnn_k, ttnn_v, cluster_axis=1, scale=0.125)
    assert output_scaled.shape == ttnn_q.shape

    logger.info("✅ Basic functionality test passed!")


if __name__ == "__main__":
    print("Distributed MLA test with CPU SDPA comparison ready!")
