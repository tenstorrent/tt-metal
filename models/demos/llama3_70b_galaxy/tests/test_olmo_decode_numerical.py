# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Decode Numerical Overflow Debug Tests.

Tests each operation in the decode pipeline with controlled inputs to identify
where numerical overflow occurs.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decode_numerical.py -v -s
"""

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.reference.functional import (
    rmsnorm_forward,
    swiglu_mlp_forward,
)
from models.common.utility_functions import comp_pcc


def check_tensor_stats(name: str, tensor: torch.Tensor, threshold: float = 1e6):
    """Check tensor for overflow and report stats."""
    if torch.isnan(tensor).any():
        logger.error(f"{name}: Contains NaN!")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"{name}: Contains Inf!")
        return False

    abs_max = tensor.abs().max().item()
    mean = tensor.mean().item()
    std = tensor.std().item()

    status = "OK" if abs_max < threshold else "OVERFLOW"
    logger.info(f"{name}: max={abs_max:.4e}, mean={mean:.4e}, std={std:.4e} [{status}]")

    return abs_max < threshold


class TestOlmoDecodeNumerical:
    """Test individual operations for numerical stability."""

    @pytest.fixture
    def model_config(self):
        """Return OLMo model configuration values."""
        return {
            "dim": 5120,
            "n_heads": 40,
            "n_kv_heads": 8,
            "head_dim": 128,
            "intermediate_size": 27648,
            "eps": 1e-5,
        }

    @pytest.fixture
    def weights(self, model_config):
        """Load real OLMo weights for layer 0."""
        import os

        hf_model = os.environ.get("HF_MODEL", "~/models/models--allenai--Olmo-3.1-32B-Think")
        hf_model = os.path.expanduser(hf_model)

        # Load state dict
        from safetensors import safe_open
        import glob

        state_dict = {}
        safetensor_files = glob.glob(f"{hf_model}/snapshots/*/model*.safetensors")
        if not safetensor_files:
            safetensor_files = glob.glob(f"{hf_model}/model*.safetensors")

        for f in safetensor_files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if "layers.0." in key or "model.norm" in key or "embed" in key:
                        state_dict[key] = sf.get_tensor(key)

        logger.info(f"Loaded {len(state_dict)} weights for layer 0")
        logger.info(f"Keys: {list(state_dict.keys())}")
        return state_dict

    def test_input_range(self, model_config):
        """Test that typical input values are reasonable."""
        batch_size = 32
        dim = model_config["dim"]

        # Test with random inputs similar to embedding outputs
        x = torch.randn(batch_size, 1, dim) * 0.02  # Typical embedding scale

        assert check_tensor_stats("Input (scaled randn)", x)

        # Test with larger values
        x_large = torch.randn(batch_size, 1, dim) * 10.0
        check_tensor_stats("Input (10x scale)", x_large)

    def test_rmsnorm_numerical(self, model_config, weights):
        """Test RMSNorm with real weights."""
        batch_size = 32
        dim = model_config["dim"]
        eps = model_config["eps"]

        # Get attention norm weight (OLMo uses post-norm: post_attention_layernorm)
        weight_key = "model.layers.0.post_attention_layernorm.weight"
        if weight_key not in weights:
            pytest.skip(f"Weight {weight_key} not found")

        weight = weights[weight_key].float()

        # Test inputs at different scales
        for scale in [0.02, 0.1, 1.0, 10.0]:
            x = torch.randn(batch_size, 1, dim) * scale

            # Reference RMSNorm
            out = rmsnorm_forward(x.float(), weight, eps)

            ok = check_tensor_stats(f"RMSNorm output (input scale={scale})", out)
            if not ok:
                logger.warning(f"RMSNorm overflows at input scale {scale}")

    def test_qkv_projection_numerical(self, model_config, weights):
        """Test QKV projection with real weights."""
        batch_size = 32
        dim = model_config["dim"]
        n_heads = model_config["n_heads"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        # Get QKV weights
        q_key = "model.layers.0.self_attn.q_proj.weight"
        k_key = "model.layers.0.self_attn.k_proj.weight"
        v_key = "model.layers.0.self_attn.v_proj.weight"

        if q_key not in weights:
            pytest.skip(f"Weight {q_key} not found")

        wq = weights[q_key].float()
        wk = weights[k_key].float()
        wv = weights[v_key].float()

        check_tensor_stats("Q weight", wq)
        check_tensor_stats("K weight", wk)
        check_tensor_stats("V weight", wv)

        # Test with typical RMSNorm output scale
        x = torch.randn(batch_size, 1, dim) * 1.0  # Post-RMSNorm is usually ~1.0 scale

        q = torch.matmul(x, wq.T)
        k = torch.matmul(x, wk.T)
        v = torch.matmul(x, wv.T)

        check_tensor_stats("Q projection", q)
        check_tensor_stats("K projection", k)
        check_tensor_stats("V projection", v)

    def test_wo_projection_numerical(self, model_config, weights):
        """Test WO projection with real weights."""
        batch_size = 32
        n_heads = model_config["n_heads"]
        head_dim = model_config["head_dim"]
        dim = model_config["dim"]

        wo_key = "model.layers.0.self_attn.o_proj.weight"
        if wo_key not in weights:
            pytest.skip(f"Weight {wo_key} not found")

        wo = weights[wo_key].float()
        check_tensor_stats("WO weight", wo)

        # Attention output is typically softmax(QK^T)V, values ~1.0
        attn_out = torch.randn(batch_size, 1, n_heads * head_dim) * 1.0

        out = torch.matmul(attn_out, wo.T)
        check_tensor_stats("WO projection", out)

    def test_mlp_numerical(self, model_config, weights):
        """Test MLP (SwiGLU) with real weights."""
        batch_size = 32
        dim = model_config["dim"]
        intermediate = model_config["intermediate_size"]

        w1_key = "model.layers.0.mlp.gate_proj.weight"
        w2_key = "model.layers.0.mlp.down_proj.weight"
        w3_key = "model.layers.0.mlp.up_proj.weight"

        if w1_key not in weights:
            pytest.skip(f"Weight {w1_key} not found")

        w1 = weights[w1_key].float()  # gate
        w2 = weights[w2_key].float()  # down
        w3 = weights[w3_key].float()  # up

        check_tensor_stats("W1 (gate) weight", w1)
        check_tensor_stats("W2 (down) weight", w2)
        check_tensor_stats("W3 (up) weight", w3)

        # Test with typical post-RMSNorm input
        x = torch.randn(batch_size, 1, dim) * 1.0

        # Step by step MLP
        gate = torch.matmul(x, w1.T)
        check_tensor_stats("Gate projection (pre-silu)", gate)

        gate_silu = torch.nn.functional.silu(gate)
        check_tensor_stats("Gate (post-silu)", gate_silu)

        up = torch.matmul(x, w3.T)
        check_tensor_stats("Up projection", up)

        hidden = gate_silu * up
        check_tensor_stats("Hidden (gate * up)", hidden)

        out = torch.matmul(hidden, w2.T)
        check_tensor_stats("MLP output", out)

    def test_full_decode_step_numerical(self, model_config, weights):
        """Test full decode step numerically."""
        batch_size = 32
        dim = model_config["dim"]
        eps = model_config["eps"]

        # Get all layer 0 weights (OLMo uses post-norm architecture)
        attn_norm_w = weights.get("model.layers.0.post_attention_layernorm.weight")
        ff_norm_w = weights.get("model.layers.0.post_feedforward_layernorm.weight")
        wq = weights.get("model.layers.0.self_attn.q_proj.weight")
        wk = weights.get("model.layers.0.self_attn.k_proj.weight")
        wv = weights.get("model.layers.0.self_attn.v_proj.weight")
        wo = weights.get("model.layers.0.self_attn.o_proj.weight")
        w1 = weights.get("model.layers.0.mlp.gate_proj.weight")
        w2 = weights.get("model.layers.0.mlp.down_proj.weight")
        w3 = weights.get("model.layers.0.mlp.up_proj.weight")

        if attn_norm_w is None:
            pytest.skip("Weights not loaded")

        # Initial input (embedding scale)
        x = torch.randn(batch_size, 1, dim) * 0.02
        check_tensor_stats("Step 0: Input", x)

        # Attention norm
        attn_in = rmsnorm_forward(x.float(), attn_norm_w.float(), eps)
        check_tensor_stats("Step 1: Attention norm output", attn_in)

        # QKV projection (simplified - no RoPE, no KV cache)
        q = torch.matmul(attn_in, wq.float().T)
        k = torch.matmul(attn_in, wk.float().T)
        v = torch.matmul(attn_in, wv.float().T)
        check_tensor_stats("Step 2: Q", q)
        check_tensor_stats("Step 2: K", k)
        check_tensor_stats("Step 2: V", v)

        # Skip attention (requires proper GQA head reshaping for Q/K/V shape mismatch)
        # For numerical stability testing, simulate attention output as normalized Q values
        # Real attention output would be post-softmax weighted V, typically unit-scale
        attn_out_simulated = torch.randn(batch_size, 1, dim) * 1.0  # Simulate post-attention
        check_tensor_stats("Step 3: Simulated attention output", attn_out_simulated)

        # WO projection
        dense_out = torch.matmul(attn_out_simulated.float(), wo.float().T)
        check_tensor_stats("Step 4: WO projection", dense_out)

        # Residual
        h = x + dense_out
        check_tensor_stats("Step 5: Post-attention residual", h)

        # FF norm
        ff_in = rmsnorm_forward(h.float(), ff_norm_w.float(), eps)
        check_tensor_stats("Step 6: FF norm output", ff_in)

        # MLP
        mlp_out = swiglu_mlp_forward(ff_in, w1.float(), w2.float(), w3.float())
        check_tensor_stats("Step 7: MLP output", mlp_out)

        # Final residual
        out = h + mlp_out
        check_tensor_stats("Step 8: Final output", out)

        logger.info("Full decode step completed - check above for overflow points")


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
class TestOlmoDecodeTTNNNumerical:
    """Test TTNN decode operations for numerical stability."""

    def test_ttnn_matmul_numerical(self, mesh_device):
        """Test TTNN matmul doesn't overflow with typical values."""
        # Simple matmul test
        a = torch.randn(1, 1, 32, 1280) * 1.0
        b = torch.randn(1, 1, 1280, 1280) * 0.02  # Weight scale

        # Reference
        ref = a @ b
        check_tensor_stats("Reference matmul", ref)

        # TTNN
        tt_a = ttnn.from_torch(
            a,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_b = ttnn.from_torch(
            b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_out = ttnn.matmul(tt_a, tt_b, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(
            tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(8, 4))
        )[0, 0]

        check_tensor_stats("TTNN matmul bfloat16", out)

        # Also test bfloat8_b
        tt_out_bf8 = ttnn.matmul(tt_a, tt_b, dtype=ttnn.bfloat8_b)
        out_bf8 = ttnn.to_torch(
            tt_out_bf8, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(8, 4))
        )[0, 0]

        check_tensor_stats("TTNN matmul bfloat8_b", out_bf8)

    def test_ttnn_rmsnorm_numerical(self, mesh_device):
        """Test TTNN RMSNorm numerical stability."""
        dim = 5120
        batch = 32
        eps = 1e-5

        x = torch.randn(1, 1, batch, dim) * 1.0
        weight = torch.ones(1, 1, 1, dim)  # Unit weight for testing

        # Reference
        variance = x.pow(2).mean(-1, keepdim=True)
        ref = x * torch.rsqrt(variance + eps)
        check_tensor_stats("Reference RMSNorm", ref)

        # TTNN
        tt_x = ttnn.from_torch(
            x,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_w = ttnn.from_torch(
            weight,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_out = ttnn.rms_norm(tt_x, epsilon=eps, weight=tt_w)
        out = ttnn.to_torch(
            tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(8, 4))
        )[0, 0]

        check_tensor_stats("TTNN RMSNorm", out)

        # Compare PCC
        passing, pcc = comp_pcc(ref[0, 0], out, 0.99)
        logger.info(f"RMSNorm PCC: {pcc:.6f}")

    def test_decode_single_layer_numerical(self, mesh_device, reset_seeds, ensure_gc):
        """Test single decode layer with TTNN and track numerical values."""
        from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
        from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
        from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
        from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL

        batch_size = 32
        max_seq_len = 256
        dtype = ttnn.bfloat8_b

        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
        model_args.n_layers = 1
        state_dict = model_args.load_state_dict()

        # Setup without prefetcher
        all_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
        all_sub_device = ttnn.SubDevice([all_core_range_set])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_manager = mesh_device.create_sub_device_manager([all_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])

        tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, mode="decode", is_olmo=True)

        # Setup RoPE
        rope_setup = TtLlamaRotarySetup(
            mesh_device,
            batch_size,
            model_args.head_dim,
            max_seq_len,
            model_args.rope_theta,
            model_args.use_scaled_rope,
            model_args.rope_scaling_factor,
        )
        transformation_mats = rope_setup.get_both_trans_mats()

        # Create decoder
        tt_model = TtTransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=0,
            n_layers=1,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            tt_ccl=tt_ccl,
        )

        # Test with different input scales
        for scale in [0.02, 0.1, 1.0]:
            logger.info(f"\n=== Testing with input scale {scale} ===")

            pt_input = torch.randn(batch_size, 1, model_args.dim) * scale
            check_tensor_stats(f"PyTorch input (scale={scale})", pt_input)

            tt_input = model_args.prepare_residual_tensor_decode(
                pt_input.clone(), model_args.model_config["DECODE_RESIDUAL_MEMCFG"]
            )

            current_pos = torch.tensor([127] * batch_size)
            current_pos_tt = ttnn.from_torch(
                current_pos,
                device=mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=(8, 4)),
            )

            rot_mats = rope_setup.get_rm_rot_mats(current_pos)

            tt_out, _ = tt_model(tt_input, None, current_pos_tt, rot_mats=rot_mats, mode="decode")

            out_torch = ttnn.to_torch(
                tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=(8, 4))
            )[:, 0:1, :batch_size, : model_args.dim].view(batch_size, 1, model_args.dim)

            check_tensor_stats(f"TTNN output (scale={scale})", out_torch)

        tt_ccl.close()
        logger.info("\nNumerical test complete - check above for overflow points")
