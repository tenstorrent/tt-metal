# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtFFN module (TP=4).

Compares TorchExpert (reference) against TtFFN (multi-chip TTNN)
to verify correctness with DeepSeek 671B FFN dimensions.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_x_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import ACTIVATION_SILU, ACTIVATION_SITU
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_ffn import EMB_DIM, HIDDEN_DIM, TtFfn
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.tt_transformers.tt.ccl import get_num_links
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_seq_len, hidden_dim, activation",
    [
        (PREFILL_CHUNK_TOKENS_PER_CHIP, HIDDEN_DIM, ACTIVATION_SILU),
        # Kimi-K3's layer-0 dense FFN: 33792 wide, on the checkpoint's SiTU-GLU (#53625). Both the
        # width and the activation are new here — 33792 is 1.8x DSv3's 18432, and it is far past
        # ttnn.situ_glu's 3072 L1 cutoff, so every intermediate is a full-size DRAM tensor.
        (PREFILL_CHUNK_TOKENS_PER_CHIP, KimiK3Config.INTERMEDIATE_SIZE, ACTIVATION_SITU),
    ],
    ids=["isl_5k", "isl_5k-k3-33792-situ"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (1, 4),
            torus_x_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="torus-x-1x4",
        ),
        # BH Galaxy. A Blackhole box accepts no mesh smaller than all 32 devices, so this is the
        # only param where the SiTU case can run at all -- SiTU needs ttnn.softcap, Blackhole-only.
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ffn_pcc(
    mesh_device,
    device_params,
    batch_seq_len: int,
    hidden_dim: int,
    activation: str,
    num_links: int,
):
    """
    Test TtFfn PCC against TorchExpert reference.

    Uses DeepSeek 671B dimensions:
        - emb_dim: 7168
        - hidden_dim: 18432
        - activations: bfloat16
        - weights bfloat8_b (explore bfp4 in future)

    ``hidden_dim`` / ``activation`` are parametrized so Kimi-K3's dense layer 0 (33792 wide, SiTU)
    is covered alongside the DeepSeek default.
    """
    if activation == ACTIVATION_SITU and not is_blackhole():
        pytest.skip("SiTU-GLU needs ttnn.softcap, which is Blackhole-only")

    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b
    # Only read on the SiTU path; reference and device must share one pair of betas.
    situ_betas = dict(
        situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
    )

    topology = per_axis_topology(device_params["fabric_config"])[1]
    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    logger.debug(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.debug(f"batch_seq_len={batch_seq_len}, emb_dim={EMB_DIM}, hidden_dim={hidden_dim}, {activation=}")

    signpost(f"FFN PCC test - {mesh_shape=} {batch_seq_len=} {hidden_dim=} {activation=} {num_links=} {topology=}")

    actual_num_links = get_num_links(mesh_device, cluster_axis=1)
    logger.debug(f"Available ethernet links along mesh columns: {actual_num_links}")
    logger.debug(f"Using num_links={num_links}, topology={topology}")

    # Create PyTorch reference model with FFN dimensions
    logger.debug("Creating TorchExpert reference with FFN dimensions")
    torch_model = TorchExpert(EMB_DIM, hidden_dim, activation=activation, **situ_betas)

    torch_weights = {
        "gate_proj": torch_model.gate_proj.data,
        "up_proj": torch_model.up_proj.data,
        "down_proj": torch_model.down_proj.data,
    }

    # Create TTNN FFN model
    logger.debug("Creating TtFfn with same weights")
    tt_model = TtFfn(
        mesh_device=mesh_device,
        torch_weights=torch_weights,
        hidden_dim=hidden_dim,
        num_links=num_links,
        topology=topology,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        activation=activation,
        **situ_betas,
    )

    # Create input tensor (replicated across all devices)
    torch_input = torch.randn(batch_seq_len, EMB_DIM, dtype=torch.float32)
    logger.debug(f"Created torch input: {torch_input.shape}")

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=activations_dtype,
    )
    logger.debug(f"Created ttnn input (replicated): {tt_input.shape}")

    # Run forward passes
    logger.debug("Running torch forward pass")
    torch_output = torch_model(torch_input)
    logger.debug(f"Torch output shape: {torch_output.shape}")

    logger.debug("Running ttnn forward pass")
    tt_output = tt_model(tt_input)
    logger.debug(f"TTNN output shape (sharded): {tt_output.shape}")

    # Convert and compare
    logger.debug("Converting TTNN output to torch for comparison")
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
    )
    logger.debug(f"TTNN output converted to torch: {tt_output_torch.shape}")

    # The input is replicated across mesh ROWS (only the TP axis, dim 1, is sharded), so composing
    # dim 0 stacks mesh_shape[0] copies of the same result. Tile the reference to match rather than
    # slicing off one copy: a row that diverged would then still show up in the PCC. No-op on a
    # single-row mesh, which is what every non-Galaxy param here is.
    #
    # Tile once and only once -- at K3's 3200x7168 each fp32 copy is ~92 MB, ~734 MB tiled across a
    # Galaxy's 8 rows, in a test that is already multi-gigabyte on the host.
    torch_reference = torch_output.repeat(mesh_shape[0], 1)

    logger.debug("Comparing outputs with PCC")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_reference.to(torch.float32),
        tt_output_torch.to(torch.float32),
        pcc=0.97,
    )

    logger.debug(f"PCC comparison: {pcc_message}")
    assert pcc_passed, f"PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")
