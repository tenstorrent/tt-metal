# models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py
"""Single-layer PCC validation: TTNN vs torch reference.

Requires a Blackhole P150 device.
Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v
"""
import glob

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import Qwen35TransformerBlock
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

CHECKPOINT_DIR = "/local/ttuser/atupe/Qwen9b"
pytestmark = run_for_blackhole()
PCC_THRESHOLD = 0.98


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    denom = (a_centered.norm() * b_centered.norm()) + 1e-8
    return (num / denom).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model_fixtures(device):
    """Load config and weights once for all tests."""
    args = Qwen35ModelArgs(mesh_device=device, checkpoint_dir=CHECKPOINT_DIR)

    from safetensors import safe_open

    raw_sd = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw_sd[key] = f.get_tensor(key)

    state_dict = remap_qwen35_state_dict(raw_sd)
    return args, state_dict


class TestGatedAttentionLayer:
    """Test layer 3 (first full attention layer)."""

    def test_gated_attention_prefill(self, device, model_fixtures):
        args, state_dict = model_fixtures
        layer = Qwen35TransformerBlock(args, state_dict, layer_num=3, device=device)

        B, T = 1, 128
        x_torch = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup

        rope = Qwen35RoPESetup(device, args)
        pos_ids = torch.arange(T).unsqueeze(0)
        cos, sin = rope.get_rot_mats(pos_ids)

        output = layer.forward(x_ttnn, cos=cos, sin=sin, mode="prefill")
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (B, T, args.dim), f"Wrong shape: {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN"
        assert not torch.isinf(output_torch).any(), "Output contains Inf"


class TestDeltaNetLayer:
    """Test layer 0 (first DeltaNet layer)."""

    def test_deltanet_recurrent(self, device, model_fixtures):
        args, state_dict = model_fixtures
        layer = Qwen35TransformerBlock(args, state_dict, layer_num=0, device=device)

        B, T = 1, 1
        x_torch = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        output = layer.forward(x_ttnn, mode="decode")
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (B, T, args.dim), f"Wrong shape: {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN"

    def test_deltanet_chunked(self, device, model_fixtures):
        args, state_dict = model_fixtures
        layer = Qwen35TransformerBlock(args, state_dict, layer_num=0, device=device)

        B, T = 1, 128
        x_torch = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        output = layer.forward(x_ttnn, mode="prefill", chunk_size=64)
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (B, T, args.dim), f"Wrong shape: {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN"
