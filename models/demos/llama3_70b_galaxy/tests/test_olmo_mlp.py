# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test OLMo SwiGLU MLP TTNN implementation against PyTorch reference.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_mlp.py -v
"""

import os
import pytest
import torch
import ttnn

from models.demos.llama3_70b_galaxy.reference.functional import swiglu_mlp_forward
from models.common.utility_functions import comp_pcc


def get_olmo_weights():
    """Load OLMo weights from HF_MODEL."""
    hf_model = os.getenv("HF_MODEL")
    if not hf_model:
        pytest.skip("HF_MODEL not set")

    import glob
    import json
    from safetensors.torch import load_file

    # Find model files
    base_path = os.path.expanduser(hf_model)
    if os.path.exists(os.path.join(base_path, "snapshots")):
        snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
        if snapshot_dirs:
            base_path = snapshot_dirs[0]

    # Load config
    config_path = os.path.join(base_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load first safetensor file for layer 0 weights
    safetensor_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
    if not safetensor_files:
        pytest.skip("No safetensor files found")

    # Load weights from first file (contains layer 0)
    state_dict = load_file(safetensor_files[0])

    return config, state_dict


@pytest.fixture
def olmo_config_and_weights():
    """Fixture to load OLMo config and weights."""
    return get_olmo_weights()


class TestOlmoSwiGLUMLP:
    """Test OLMo SwiGLU MLP against PyTorch reference."""

    def test_swiglu_mlp_reference(self, olmo_config_and_weights):
        """Test SwiGLU MLP reference implementation with real weights."""
        config, state_dict = olmo_config_and_weights

        dim = config["hidden_size"]  # 5120
        intermediate_size = config["intermediate_size"]  # 27648

        # OLMo MLP weight keys
        w1_key = "model.layers.0.mlp.gate_proj.weight"
        w2_key = "model.layers.0.mlp.down_proj.weight"
        w3_key = "model.layers.0.mlp.up_proj.weight"

        for key in [w1_key, w2_key, w3_key]:
            if key not in state_dict:
                pytest.skip(f"Weight {key} not found")

        w1 = state_dict[w1_key].float()  # [intermediate_size, dim]
        w2 = state_dict[w2_key].float()  # [dim, intermediate_size]
        w3 = state_dict[w3_key].float()  # [intermediate_size, dim]

        # Verify shapes
        assert w1.shape == (intermediate_size, dim), f"w1 shape: {w1.shape}"
        assert w2.shape == (dim, intermediate_size), f"w2 shape: {w2.shape}"
        assert w3.shape == (intermediate_size, dim), f"w3 shape: {w3.shape}"

        # Create test input
        torch.manual_seed(42)
        batch, seq_len = 1, 128
        x = torch.randn(batch, seq_len, dim, dtype=torch.float32)

        # Reference output
        ref_output = swiglu_mlp_forward(x, w1, w2, w3)

        # Verify output
        assert ref_output.shape == x.shape, f"Output shape mismatch: {ref_output.shape} vs {x.shape}"
        assert not torch.isnan(ref_output).any(), "Output contains NaN"
        assert not torch.isinf(ref_output).any(), "Output contains Inf"

        print(f"SwiGLU MLP reference test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  w1 (gate_proj): {w1.shape}")
        print(f"  w2 (down_proj): {w2.shape}")
        print(f"  w3 (up_proj): {w3.shape}")
        print(f"  Output shape: {ref_output.shape}")
        print(f"  Intermediate size: {intermediate_size}")


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
class TestOlmoMLPTTNN:
    """Test OLMo MLP on TTNN device using TtLlamaMLP."""

    def test_olmo_mlp_prefill(self, mesh_device, olmo_config_and_weights):
        """Test OLMo MLP TTNN vs PyTorch reference PCC."""
        from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
        from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
        from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
        from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
        from loguru import logger

        config, _ = olmo_config_and_weights
        dim = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        dtype = ttnn.bfloat8_b
        batch_size = 1
        seq_len = 128

        # Load OLMo model config
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128)
        model_args.n_layers = 1
        model_args.use_prefetcher = False
        state_dict = model_args.load_state_dict()

        logger.info(f"OLMo Model Config Loaded")

        # Setup prefetcher and CCL
        prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
        tt_ccl = TT_CCL(
            mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill", is_qwen=False, is_olmo=True
        )

        # Create TTNN MLP model
        model_args.WEIGHTS_DTYPE = dtype
        tt_model = TtLlamaMLP(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=0,
            dtype=dtype,
            model_config=model_args.get_model_config(),
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )

        # Create test input
        torch.manual_seed(42)
        torch_input = torch.randn(1, 1, seq_len, dim)

        # TTNN forward
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3),
                mesh_shape=model_args.cluster_shape,
            ),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = tt_model.forward_prefill(tt_input, batch_size)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_output_torch[:, :1, :, :dim]

        # Reference forward (convert weights to float32 for reference computation)
        w1 = state_dict["layers.0.feed_forward.w1.weight"].float()
        w2 = state_dict["layers.0.feed_forward.w2.weight"].float()
        w3 = state_dict["layers.0.feed_forward.w3.weight"].float()
        ref_input = torch_input[:, :1, :, :dim].float()
        ref_output = swiglu_mlp_forward(ref_input, w1, w2, w3)

        # Compare PCC
        pcc_required = 0.99
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required)

        logger.info(f"OLMo MLP PCC: {pcc_message}")

        tt_ccl.close()

        assert passing, f"OLMo MLP prefill PCC {pcc_message} < {pcc_required}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
