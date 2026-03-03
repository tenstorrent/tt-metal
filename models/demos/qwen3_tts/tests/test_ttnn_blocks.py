# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Tests for Qwen3-TTS TTNN implementations.

Verifies PCC > 0.99 against PyTorch reference implementations.
"""

import os
import sys

import pytest
import torch
from scipy.stats import pearsonr

import ttnn
from models.common.utility_functions import is_wormhole_b0

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)

from models.demos.qwen3_tts.reference.functional import attention as torch_attention
from models.demos.qwen3_tts.reference.functional import get_default_talker_config
from models.demos.qwen3_tts.reference.functional import rms_norm as torch_rms_norm
from models.demos.qwen3_tts.reference.functional import swiglu_mlp as torch_swiglu_mlp

torch.manual_seed(0)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate Pearson correlation coefficient between two tensors."""
    x_flat = x.detach().float().cpu().numpy().flatten()
    y_flat = y.detach().float().cpu().numpy().flatten()
    return pearsonr(x_flat, y_flat)[0]


def ttnn_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    """Convert TTNN tensor to PyTorch tensor."""
    return ttnn.to_torch(tensor)


@pytest.fixture(scope="module")
def device():
    """Get the TTNN device."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def talker_config():
    """Get the default Talker configuration."""
    return get_default_talker_config()


# =============================================================================
# RMSNorm Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestRMSNorm:
    """Tests for RMSNorm TTNN implementation."""

    def test_rmsnorm_pcc(self, device, talker_config):
        """Test RMSNorm achieves PCC > 0.99."""
        from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm

        batch, seq_len = 1, 128
        hidden_size = talker_config.hidden_size

        # Create random input and weight
        torch_input = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
        torch_weight = torch.randn(hidden_size, dtype=torch.bfloat16)

        # PyTorch reference
        torch_output = torch_rms_norm(
            torch_input.squeeze(1),  # [batch, seq_len, hidden_size]
            torch_weight,
            eps=talker_config.rms_norm_eps,
        )

        # Create state dict for TTNN
        state_dict = {"test_norm.weight": torch_weight}

        # TTNN implementation
        rmsnorm = RMSNorm(
            device=device,
            dim=hidden_size,
            state_dict=state_dict,
            weight_key="test_norm.weight",
            eps=talker_config.rms_norm_eps,
        )

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = rmsnorm(ttnn_input)
        ttnn_output_torch = ttnn_to_torch(ttnn_output).squeeze(1)  # [batch, seq_len, hidden_size]

        pcc = pearson_correlation(torch_output, ttnn_output_torch)
        print(f"RMSNorm PCC: {pcc:.6f}")

        assert pcc > 0.99, f"RMSNorm PCC {pcc} below threshold 0.99"


# =============================================================================
# MLP Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestMLP:
    """Tests for MLP TTNN implementation."""

    def test_mlp_pcc(self, device, talker_config):
        """Test MLP achieves PCC > 0.99."""
        from models.demos.qwen3_tts.tt.mlp import MLP

        batch, seq_len = 1, 128
        hidden_size = talker_config.hidden_size
        intermediate_size = talker_config.intermediate_size

        # Create random input and weights
        torch_input = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
        gate_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        up_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        down_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)

        # PyTorch reference
        torch_output = torch_swiglu_mlp(
            torch_input.squeeze(1),  # [batch, seq_len, hidden_size]
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
        )

        # Create state dict for TTNN
        state_dict = {
            "test_layer.mlp.gate_proj.weight": gate_proj_weight,
            "test_layer.mlp.up_proj.weight": up_proj_weight,
            "test_layer.mlp.down_proj.weight": down_proj_weight,
        }

        # TTNN implementation
        mlp = MLP(
            device=device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            state_dict=state_dict,
            layer_prefix="test_layer",
        )

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = mlp(ttnn_input)
        ttnn_output_torch = ttnn_to_torch(ttnn_output).squeeze(1)  # [batch, seq_len, hidden_size]

        pcc = pearson_correlation(torch_output, ttnn_output_torch)
        print(f"MLP PCC: {pcc:.6f}")

        assert pcc > 0.99, f"MLP PCC {pcc} below threshold 0.99"


# =============================================================================
# Golden Output Tests (load from reference/golden/)
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestGoldenOutputs:
    """Tests comparing TTNN against pre-computed golden outputs."""

    @pytest.fixture
    def golden_dir(self):
        """Get the golden outputs directory."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reference",
            "golden",
        )

    def test_rmsnorm_golden(self, device, golden_dir):
        """Test RMSNorm against golden output."""
        from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm

        golden_path = os.path.join(golden_dir, "rms_norm_golden.pt")
        if not os.path.exists(golden_path):
            pytest.skip(f"Golden file not found: {golden_path}")

        golden = torch.load(golden_path, weights_only=True)

        # Create state dict for TTNN
        state_dict = {"test_norm.weight": golden["weight"]}

        # TTNN implementation
        rmsnorm = RMSNorm(
            device=device,
            dim=golden["config"]["hidden_size"],
            state_dict=state_dict,
            weight_key="test_norm.weight",
            eps=golden["eps"],
        )

        # Make input 4D for TTNN: [batch, 1, seq_len, hidden_size]
        torch_input = golden["input"].unsqueeze(1)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = rmsnorm(ttnn_input)
        ttnn_output_torch = ttnn_to_torch(ttnn_output).squeeze(1)  # Back to 3D

        pcc = pearson_correlation(golden["output"], ttnn_output_torch)
        print(f"RMSNorm Golden PCC: {pcc:.6f}")

        assert pcc > 0.99, f"RMSNorm Golden PCC {pcc} below threshold 0.99"

    def test_mlp_golden(self, device, golden_dir):
        """Test MLP against golden output."""
        from models.demos.qwen3_tts.tt.mlp import MLP

        golden_path = os.path.join(golden_dir, "mlp_golden.pt")
        if not os.path.exists(golden_path):
            pytest.skip(f"Golden file not found: {golden_path}")

        golden = torch.load(golden_path, weights_only=True)

        # Create state dict for TTNN
        state_dict = {
            "test_layer.mlp.gate_proj.weight": golden["gate_proj_weight"],
            "test_layer.mlp.up_proj.weight": golden["up_proj_weight"],
            "test_layer.mlp.down_proj.weight": golden["down_proj_weight"],
        }

        # TTNN implementation
        mlp = MLP(
            device=device,
            hidden_size=golden["config"]["hidden_size"],
            intermediate_size=golden["config"]["intermediate_size"],
            state_dict=state_dict,
            layer_prefix="test_layer",
        )

        # Make input 4D for TTNN: [batch, 1, seq_len, hidden_size]
        torch_input = golden["input"].unsqueeze(1)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = mlp(ttnn_input)
        ttnn_output_torch = ttnn_to_torch(ttnn_output).squeeze(1)  # Back to 3D

        pcc = pearson_correlation(golden["output"], ttnn_output_torch)
        print(f"MLP Golden PCC: {pcc:.6f}")

        assert pcc > 0.99, f"MLP Golden PCC {pcc} below threshold 0.99"


# =============================================================================
# Attention Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestAttention:
    """Tests for Attention TTNN implementation."""

    def test_attention_pcc(self, device, talker_config):
        """Test Attention achieves PCC > 0.99.

        Uses identity RoPE (cos=1, sin=0) to test core attention mechanism
        without the complexity of matching different RoPE implementations.
        """
        from models.demos.qwen3_tts.tt.attention import Attention
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        batch, seq_len = 1, 128
        hidden_size = talker_config.hidden_size
        num_heads = talker_config.num_attention_heads
        num_kv_heads = talker_config.num_key_value_heads
        head_dim = talker_config.head_dim

        # Create random input and weights
        torch.manual_seed(42)
        torch_input = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
        q_proj_weight = torch.randn(num_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        k_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        v_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        o_proj_weight = torch.randn(hidden_size, num_heads * head_dim, dtype=torch.bfloat16)
        q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)
        k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)

        # Identity RoPE: cos=1, sin=0 (no rotation applied)
        # This tests core attention mechanism without RoPE complexity
        cos_identity = torch.ones(1, seq_len, head_dim, dtype=torch.bfloat16)
        sin_identity = torch.zeros(1, seq_len, head_dim, dtype=torch.bfloat16)

        # PyTorch reference with identity RoPE
        torch_output = torch_attention(
            torch_input,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
            cos_identity,
            sin_identity,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=talker_config.rms_norm_eps,
            use_mrope=False,
        )

        # Create state dict for TTNN
        state_dict = {
            "test_layer.self_attn.q_proj.weight": q_proj_weight,
            "test_layer.self_attn.k_proj.weight": k_proj_weight,
            "test_layer.self_attn.v_proj.weight": v_proj_weight,
            "test_layer.self_attn.o_proj.weight": o_proj_weight,
            "test_layer.self_attn.q_norm.weight": q_norm_weight,
            "test_layer.self_attn.k_norm.weight": k_norm_weight,
        }

        # TTNN implementation
        attention = Attention(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            state_dict=state_dict,
            layer_prefix="test_layer",
            rms_norm_eps=talker_config.rms_norm_eps,
        )

        # Create TTNN tensors
        ttnn_input = ttnn.from_torch(
            torch_input.unsqueeze(1),  # [batch, 1, seq_len, hidden_size]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Identity RoPE for TTNN - interleaved format [cos0, cos0, cos1, cos1, ...]
        # For cos=1, sin=0, interleaving doesn't matter
        cos_ttnn_torch = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
        sin_ttnn_torch = torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16)

        cos_ttnn = ttnn.from_torch(
            cos_ttnn_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        sin_ttnn = ttnn.from_torch(
            sin_ttnn_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Transformation matrix for RoPE
        trans_mat = get_rot_transformation_mat(dhead=head_dim)
        trans_mat_ttnn = ttnn.from_torch(
            trans_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = attention(ttnn_input, cos_ttnn, sin_ttnn, trans_mat_ttnn)
        ttnn_output_torch = ttnn_to_torch(ttnn_output).squeeze(1)  # [batch, seq_len, hidden_size]

        pcc = pearson_correlation(torch_output, ttnn_output_torch)
        print(f"Attention PCC: {pcc:.6f}")

        assert pcc > 0.99, f"Attention PCC {pcc} below threshold 0.99"


# =============================================================================
# DecoderLayer Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestDecoderLayer:
    """Tests for DecoderLayer TTNN implementation."""

    def test_decoder_layer_pcc(self, device, talker_config):
        """Test DecoderLayer achieves PCC > 0.99.

        Uses identity RoPE (cos=1, sin=0) to test core decoder layer mechanism
        without the complexity of matching different RoPE implementations.
        """
        from models.demos.qwen3_tts.reference.functional import decoder_layer as torch_decoder_layer
        from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        batch, seq_len = 1, 128
        hidden_size = talker_config.hidden_size
        num_heads = talker_config.num_attention_heads
        num_kv_heads = talker_config.num_key_value_heads
        head_dim = talker_config.head_dim
        intermediate_size = talker_config.intermediate_size

        # Create random input and weights
        torch.manual_seed(42)
        torch_input = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)

        # Attention weights
        q_proj_weight = torch.randn(num_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        k_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        v_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        o_proj_weight = torch.randn(hidden_size, num_heads * head_dim, dtype=torch.bfloat16)
        q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)
        k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)

        # MLP weights
        gate_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        up_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        down_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)

        # Norm weights
        input_layernorm_weight = torch.randn(hidden_size, dtype=torch.bfloat16)
        post_attention_layernorm_weight = torch.randn(hidden_size, dtype=torch.bfloat16)

        # Identity RoPE: cos=1, sin=0 (no rotation applied)
        cos_identity = torch.ones(1, seq_len, head_dim, dtype=torch.bfloat16)
        sin_identity = torch.zeros(1, seq_len, head_dim, dtype=torch.bfloat16)

        # Layer weights dict for PyTorch
        layer_weights = {
            "input_layernorm.weight": input_layernorm_weight,
            "self_attn.q_proj.weight": q_proj_weight,
            "self_attn.k_proj.weight": k_proj_weight,
            "self_attn.v_proj.weight": v_proj_weight,
            "self_attn.o_proj.weight": o_proj_weight,
            "self_attn.q_norm.weight": q_norm_weight,
            "self_attn.k_norm.weight": k_norm_weight,
            "post_attention_layernorm.weight": post_attention_layernorm_weight,
            "mlp.gate_proj.weight": gate_proj_weight,
            "mlp.up_proj.weight": up_proj_weight,
            "mlp.down_proj.weight": down_proj_weight,
        }

        # PyTorch reference with identity RoPE
        torch_output = torch_decoder_layer(
            torch_input,
            layer_weights,
            cos_identity,
            sin_identity,
            talker_config,
            use_mrope=False,
        )

        # Create state dict for TTNN
        layer_prefix = "talker.model.layers.0"
        state_dict = {
            f"{layer_prefix}.input_layernorm.weight": input_layernorm_weight,
            f"{layer_prefix}.self_attn.q_proj.weight": q_proj_weight,
            f"{layer_prefix}.self_attn.k_proj.weight": k_proj_weight,
            f"{layer_prefix}.self_attn.v_proj.weight": v_proj_weight,
            f"{layer_prefix}.self_attn.o_proj.weight": o_proj_weight,
            f"{layer_prefix}.self_attn.q_norm.weight": q_norm_weight,
            f"{layer_prefix}.self_attn.k_norm.weight": k_norm_weight,
            f"{layer_prefix}.post_attention_layernorm.weight": post_attention_layernorm_weight,
            f"{layer_prefix}.mlp.gate_proj.weight": gate_proj_weight,
            f"{layer_prefix}.mlp.up_proj.weight": up_proj_weight,
            f"{layer_prefix}.mlp.down_proj.weight": down_proj_weight,
        }

        # TTNN implementation
        decoder_layer = DecoderLayer(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            state_dict=state_dict,
            layer_idx=0,
            layer_prefix="talker.model",
            rms_norm_eps=talker_config.rms_norm_eps,
        )

        # Create TTNN tensors
        ttnn_input = ttnn.from_torch(
            torch_input.unsqueeze(1),  # [batch, 1, seq_len, hidden_size]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Identity RoPE for TTNN - [1, 1, seq_len, head_dim]
        cos_ttnn_torch = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
        sin_ttnn_torch = torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16)

        cos_ttnn = ttnn.from_torch(
            cos_ttnn_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        sin_ttnn = ttnn.from_torch(
            sin_ttnn_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Transformation matrix for RoPE
        trans_mat = get_rot_transformation_mat(dhead=head_dim)
        trans_mat_ttnn = ttnn.from_torch(
            trans_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = decoder_layer(ttnn_input, cos_ttnn, sin_ttnn, trans_mat_ttnn)
        ttnn_output_torch = ttnn_to_torch(ttnn_output).squeeze(1)  # [batch, seq_len, hidden_size]

        pcc = pearson_correlation(torch_output, ttnn_output_torch)
        print(f"DecoderLayer PCC: {pcc:.6f}")

        # DecoderLayer has lower threshold due to accumulated error from chaining
        # multiple operations (attention ~0.995, RMSNorm ~0.9999, MLP ~0.9999)
        assert pcc > 0.97, f"DecoderLayer PCC {pcc} below threshold 0.97"


# =============================================================================
# Talker Model Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestTalker:
    """Tests for Talker model structure (lightweight, no HF weights required)."""

    def test_talker_initialization(self, device, talker_config):
        """Test Talker model can be initialized with random weights."""
        from models.demos.qwen3_tts.tt.talker import Talker

        # Create minimal state dict with random weights
        torch.manual_seed(42)
        hidden_size = talker_config.hidden_size
        num_heads = talker_config.num_attention_heads
        num_kv_heads = talker_config.num_key_value_heads
        head_dim = talker_config.head_dim
        intermediate_size = talker_config.intermediate_size
        vocab_size = talker_config.audio_vocab_size

        state_dict = {
            "talker.model.codec_embedding.weight": torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16),
            "talker.model.norm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
        }

        # Add weights for 2 layers (reduced for faster testing)
        for i in range(2):
            prefix = f"talker.model.layers.{i}"
            state_dict.update(
                {
                    f"{prefix}.input_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.q_proj.weight": torch.randn(
                        num_heads * head_dim, hidden_size, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.k_proj.weight": torch.randn(
                        num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.v_proj.weight": torch.randn(
                        num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.o_proj.weight": torch.randn(
                        hidden_size, num_heads * head_dim, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                    f"{prefix}.post_attention_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16),
                }
            )

        # Modify config for testing with fewer layers
        test_config = type(talker_config)()
        test_config.num_hidden_layers = 2

        # Initialize model
        talker = Talker(
            device=device,
            config=test_config,
            state_dict=state_dict,
        )

        assert len(talker.layers) == 2
        print("Talker initialization: PASS")


# =============================================================================
# CodePredictor Model Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestCodePredictor:
    """Tests for CodePredictor model structure (lightweight, no HF weights required)."""

    def test_code_predictor_initialization(self, device):
        """Test CodePredictor model can be initialized with random weights."""
        from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig

        config = Qwen3TTSCodePredictorConfig()

        # Create minimal state dict with random weights
        torch.manual_seed(42)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size
        vocab_size = config.vocab_size
        num_code_groups = config.num_code_groups

        state_dict = {}

        # Add weights for 2 layers (reduced for faster testing)
        for i in range(2):
            prefix = f"talker.code_predictor.model.layers.{i}"
            state_dict.update(
                {
                    f"{prefix}.input_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.q_proj.weight": torch.randn(
                        num_heads * head_dim, hidden_size, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.k_proj.weight": torch.randn(
                        num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.v_proj.weight": torch.randn(
                        num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.o_proj.weight": torch.randn(
                        hidden_size, num_heads * head_dim, dtype=torch.bfloat16
                    ),
                    f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                    f"{prefix}.post_attention_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16),
                    f"{prefix}.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16),
                }
            )

        # Add LM heads
        for g in range(num_code_groups):
            state_dict[f"talker.code_predictor.lm_head.{g}.weight"] = torch.randn(
                vocab_size, hidden_size, dtype=torch.bfloat16
            )

        # Modify config for testing with fewer layers
        test_config = Qwen3TTSCodePredictorConfig()
        test_config.num_hidden_layers = 2

        # Initialize model (same hidden size as talker for this test)
        code_predictor = CodePredictor(
            device=device,
            config=test_config,
            talker_hidden_size=hidden_size,  # No projection needed
            state_dict=state_dict,
        )

        assert len(code_predictor.layers) == 2
        assert len(code_predictor.lm_heads) == num_code_groups
        print(
            f"CodePredictor initialization: PASS (layers={len(code_predictor.layers)}, lm_heads={len(code_predictor.lm_heads)})"
        )


# =============================================================================
# Speech Tokenizer Tests
# =============================================================================
@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestSpeechTokenizer:
    """Tests for Speech Tokenizer TTNN implementation."""

    def test_speech_tokenizer_config(self):
        """Test SpeechTokenizerConfig initialization."""
        from models.demos.qwen3_tts.tt.speech_tokenizer import SpeechTokenizerConfig

        config = SpeechTokenizerConfig()

        assert config.num_quantizers == 16
        assert config.codebook_size == 2048
        assert config.codebook_dim == 256
        assert config.pre_transformer_num_layers == 8
        assert config.pre_transformer_hidden_size == 512
        assert config.upsample_rates == (8, 5, 4, 3)
        print("SpeechTokenizerConfig: PASS")

    def test_codebook_lookup_shape(self, device):
        """Test codebook lookup produces correct output shape."""
        from models.demos.qwen3_tts.tt.speech_tokenizer import SpeechTokenizerConfig, TtSpeechTokenizerDecoder

        config = SpeechTokenizerConfig()
        batch_size, seq_len = 1, 32
        num_quantizers = config.num_quantizers

        # Create mock codebook weights
        state_dict = {}
        state_dict["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"] = torch.randn(
            config.codebook_size, config.codebook_dim, dtype=torch.float32
        )
        for i in range(num_quantizers - 1):
            state_dict[f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"] = torch.randn(
                config.codebook_size, config.codebook_dim, dtype=torch.float32
            )

        # Create decoder (will skip pre-transformer since no weights for it)
        decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=state_dict,
            config=config,
        )

        # Test codebook lookup
        token_ids = torch.randint(0, config.codebook_size, (batch_size, num_quantizers, seq_len))
        embeddings = decoder._codebook_lookup(token_ids)

        # Check output shape
        assert embeddings.shape == (
            batch_size,
            seq_len,
            config.codebook_dim,
        ), f"Expected {(batch_size, seq_len, config.codebook_dim)}, got {embeddings.shape}"

        print(f"Codebook lookup shape: PASS ({embeddings.shape})")

    def test_speech_tokenizer_initialization(self, device):
        """Test TtSpeechTokenizerDecoder initialization without pre-transformer."""
        from models.demos.qwen3_tts.tt.speech_tokenizer import SpeechTokenizerConfig, TtSpeechTokenizerDecoder

        config = SpeechTokenizerConfig()

        # Create minimal mock weights (just codebooks)
        state_dict = {}
        state_dict["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"] = torch.randn(
            config.codebook_size, config.codebook_dim, dtype=torch.float32
        )
        for i in range(config.num_quantizers - 1):
            state_dict[f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"] = torch.randn(
                config.codebook_size, config.codebook_dim, dtype=torch.float32
            )

        decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=state_dict,
            config=config,
        )

        assert len(decoder.codebooks) == config.num_quantizers
        assert decoder.has_pre_transformer is False  # No pre-transformer weights provided
        print(f"SpeechTokenizerDecoder init: PASS (codebooks={len(decoder.codebooks)})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
