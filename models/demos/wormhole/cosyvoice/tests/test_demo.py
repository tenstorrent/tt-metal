# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the CosyVoice model bring-up.

These tests validate the model architecture and pipeline structure.
End-to-end tests require TT hardware and are gated by the
is_tt_device_available fixture.
"""

import pytest
import torch

from models.common.utility_functions import is_wormhole_b0


@pytest.mark.parametrize(
    "mode, text",
    [
        ("sft", "Hello, this is a test."),
        ("zero_shot", "Voice cloning test."),
        ("cross_lingual", "你好，测试。"),
        ("instruct", "Test with instruction."),
    ],
    ids=["sft_mode", "zero_shot_mode", "cross_lingual_mode", "instruct_mode"],
)
@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_cosyvoice_pipeline(mode, text, mesh_device, is_ci_env):
    """Test the full CosyVoice pipeline end-to-end.

    This test validates that the pipeline can be instantiated and
    produces output of the expected shape.
    """
    if is_ci_env and not is_wormhole_b0():
        pytest.skip("Skipping CosyVoice test in CI without WH B0 hardware")

    from models.demos.wormhole.cosyvoice.tt.pipeline import TtCosyVoicePipeline
    from models.demos.wormhole.cosyvoice.tt.model_config import CosyVoiceModelConfig

    config = CosyVoiceModelConfig()

    # Create placeholder state dict for testing
    hidden_size = config.llm_hidden_size
    state_dict = {}

    # Minimal state dict for pipeline initialization
    state_dict["llm.model.model.embed_tokens.weight"] = torch.randn(100, hidden_size)
    state_dict["llm.speech_embedding.weight"] = torch.randn(config.speech_token_size + 3, hidden_size)
    state_dict["llm.llm_embedding.weight"] = torch.randn(2, hidden_size)
    state_dict["llm.llm_decoder.weight"] = torch.randn(hidden_size, config.speech_token_size + 3)
    state_dict["llm.llm_decoder.bias"] = torch.zeros(config.speech_token_size + 3)
    state_dict["llm.text_encoder_affine_layer.weight"] = torch.randn(hidden_size, hidden_size)
    state_dict["llm.spk_embed_affine_layer.weight"] = torch.randn(192, hidden_size)

    # Transformer layers
    for i in range(min(2, config.llm_num_layers)):
        prefix = f"llm.llm.model.model.layers.{i}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(config.llm_intermediate_size, hidden_size)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(config.llm_intermediate_size, hidden_size)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, config.llm_intermediate_size)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)

    state_dict["llm.llm.model.model.norm.weight"] = torch.ones(hidden_size)
    state_dict["flow.input_embedding.weight"] = torch.randn(config.flow_vocab_size, config.flow_input_size)
    state_dict["flow.spk_embed_affine_layer.weight"] = torch.randn(192, config.flow_output_size)
    state_dict["flow.encoder_proj.weight"] = torch.randn(config.flow_input_size, config.flow_output_size)

    device = mesh_device
    pipeline = TtCosyVoicePipeline(device, config, state_dict)

    # Run inference
    audio = pipeline.tts(text=text, mode=mode)

    # Check output shape
    assert audio is not None, "Pipeline returned None"
    assert audio.dim() == 2, f"Expected 2D tensor, got {audio.dim()}D"
    assert audio.shape[-1] > 0, "Output audio is empty"

    print(f"Test passed: mode={mode}, audio shape={audio.shape}")


@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_cosyvoice_imports(mesh_device):
    """Test that all CosyVoice modules can be imported correctly."""
    from models.demos.wormhole.cosyvoice.tt.model_config import (
        CosyVoiceModelConfig,
        create_model_config,
    )
    from models.demos.wormhole.cosyvoice.tt.pipeline import TtCosyVoicePipeline
    from models.demos.wormhole.cosyvoice.tt.llm.cosyvoice_llm import TtCosyVoiceLLM
    from models.demos.wormhole.cosyvoice.tt.flow.flow_matching import (
        TtMaskedDiffWithXvec,
        TtCFMDecoder,
    )
    from models.demos.wormhole.cosyvoice.tt.vocoder.hifigan import TtHiFiGAN

    # Instantiate config
    config = CosyVoiceModelConfig()
    assert config.llm_hidden_size == 896
    assert config.llm_num_layers == 24
    assert config.flow_vocab_size == 4096
    print("All imports successful")


def test_model_config():
    """Test model configuration values."""
    from models.demos.wormhole.cosyvoice.tt.model_config import CosyVoiceModelConfig

    config = CosyVoiceModelConfig()

    # Verify architecture dimensions
    assert config.llm_input_size == 896
    assert config.llm_num_layers == 24
    assert config.llm_num_heads == 14
    assert config.llm_hidden_size == 896
    assert config.speech_token_size == 4096
    assert config.flow_output_size == 80
    assert config.flow_vocab_size == 4096
    assert config.flow_input_frame_rate == 50


def test_cfm_decoder():
    """Test the CFM decoder configuration."""
    from models.demos.wormhole.cosyvoice.tt.flow.flow_matching import TtCFMDecoder
    from models.demos.wormhole.cosyvoice.tt.model_config import CosyVoiceModelConfig

    config = CosyVoiceModelConfig()
    decoder = TtCFMDecoder(config)

    assert decoder.solver == "euler"
    assert decoder.t_scheduler == "cosine"
    assert decoder.inference_cfg_rate == 0.7
