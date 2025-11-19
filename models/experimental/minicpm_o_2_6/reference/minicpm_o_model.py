#!/usr/bin/env python3
"""
MiniCPM-o-2_6 PyTorch Reference Implementation

This module provides the complete PyTorch reference implementation
combining multimodal Qwen2.5, modality projectors, and ChatTTS decoder.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from pathlib import Path
import json

# Import our components
from multimodal_qwen import MultimodalQwen2ForCausalLM, MultimodalQwen2Config
from modality_projectors import MiniCPMModalityProjectors
from chattts_decoder import create_chatts_decoder_from_config


class MiniCPMOModelConfig:
    """Configuration for complete MiniCPM-o model"""

    def __init__(
        self,
        # Base model config
        vocab_size: int = 151700,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        # Multimodal config
        cross_attention_layers: List[int] = None,
        # Vision config (SigLip)
        vision_hidden_size: int = 1152,
        vision_seq_len: int = 256,  # Typical number of vision tokens
        # Audio config (Whisper)
        audio_hidden_size: int = 1024,
        audio_seq_len: int = 128,  # Typical number of audio tokens
        # Generation config
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        **kwargs,
    ):
        # Base model parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Multimodal parameters
        self.cross_attention_layers = cross_attention_layers or [8, 16, 24]  # Default layers

        # Modality parameters
        self.vision_hidden_size = vision_hidden_size
        self.vision_seq_len = vision_seq_len
        self.audio_hidden_size = audio_hidden_size
        self.audio_seq_len = audio_seq_len

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MiniCPMOForConditionalGeneration(nn.Module):
    """
    Complete MiniCPM-o model for conditional generation.

    Supports text generation, vision-language tasks, audio-language tasks,
    and text-to-speech synthesis.
    """

    def __init__(self, config: MiniCPMOModelConfig):
        super().__init__()
        self.config = config

        # Create multimodal Qwen2.5 model
        qwen_config = MultimodalQwen2Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            cross_attention_layers=config.cross_attention_layers,
            vision_hidden_size=config.vision_hidden_size,
            audio_hidden_size=config.audio_hidden_size,
        )
        self.language_model = MultimodalQwen2ForCausalLM(qwen_config)

        # Create modality projectors
        self.modality_projectors = MiniCPMModalityProjectors(
            vision_config={"hidden_size": config.vision_hidden_size},
            audio_config={"hidden_size": config.audio_hidden_size},
            llm_hidden_size=config.hidden_size,
        )

        # Create ChatTTS decoder
        tts_config = {
            "tts_config": {"model_type": "conditional_chattts", "llm_dim": config.hidden_size},
            "hidden_size": config.hidden_size,
        }
        self.tts_decoder = create_chatts_decoder_from_config(tts_config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tts_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass through the complete MiniCPM-o model.

        Args:
            input_ids: [batch_size, seq_len] text token ids
            attention_mask: [batch_size, seq_len] attention mask
            vision_embeds: [batch_size, vision_seq_len, vision_hidden] vision embeddings
            audio_embeds: [batch_size, audio_seq_len, audio_hidden] audio embeddings
            labels: [batch_size, seq_len] labels for language modeling loss
            tts_targets: [batch_size, tts_seq_len] targets for TTS loss

        Returns:
            Dict with logits, loss, and optional TTS outputs
        """
        # Project modalities to LLM space
        projected_vision, projected_audio = self.modality_projectors(
            vision_embeds=vision_embeds, audio_embeds=audio_embeds
        )

        # Forward through language model
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_embeds=projected_vision,
            audio_embeds=projected_audio,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        # Extract LLM hidden states for TTS
        lm_hidden_states = lm_outputs["last_hidden_state"] if isinstance(lm_outputs, dict) else lm_outputs[0]

        # Get the last token's representation for TTS generation
        if lm_hidden_states.dim() == 3:
            # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
            tts_input = lm_hidden_states[:, -1, :]  # Last token
            # Expand to sequence for decoder
            tts_input = tts_input.unsqueeze(1)  # [batch_size, 1, hidden_size]
        else:
            tts_input = lm_hidden_states.unsqueeze(1)

        # Forward through TTS decoder
        tts_logits, tts_loss = self.tts_decoder(llm_outputs=tts_input, target_audio_tokens=tts_targets)

        # Combine outputs
        total_loss = None
        losses = []

        # Language modeling loss
        if isinstance(lm_outputs, dict) and "loss" in lm_outputs:
            losses.append(lm_outputs["loss"])
        elif isinstance(lm_outputs, tuple) and len(lm_outputs) > 0:
            # Assume first element might be loss if labels provided
            if labels is not None and len(lm_outputs) > 1:
                losses.append(lm_outputs[0])

        # TTS loss
        if tts_loss is not None:
            losses.append(tts_loss)

        # Filter out None values and compute total loss
        valid_losses = [l for l in losses if l is not None]
        if valid_losses:
            total_loss = sum(valid_losses) / len(valid_losses)  # Average losses

        return {
            "lm_logits": lm_outputs["logits"] if isinstance(lm_outputs, dict) else lm_outputs[0],
            "tts_logits": tts_logits,
            "loss": total_loss,
            "lm_loss": lm_outputs.get("loss") if isinstance(lm_outputs, dict) else None,
            "tts_loss": tts_loss,
            "past_key_values": lm_outputs.get("past_key_values") if isinstance(lm_outputs, dict) else None,
            "hidden_states": lm_outputs.get("hidden_states") if isinstance(lm_outputs, dict) else None,
            "attentions": lm_outputs.get("attentions") if isinstance(lm_outputs, dict) else None,
        }

    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        max_length: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] input token ids
            attention_mask: [batch_size, seq_len] attention mask
            vision_embeds: [batch_size, vision_seq_len, vision_hidden] vision embeddings
            audio_embeds: [batch_size, audio_seq_len, audio_hidden] audio embeddings
            max_length: Maximum generation length
            **kwargs: Additional generation arguments

        Returns:
            [batch_size, max_length] generated token ids
        """
        # Project modalities
        projected_vision, projected_audio = self.modality_projectors(
            vision_embeds=vision_embeds, audio_embeds=audio_embeds
        )

        # Generate using language model
        return self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_embeds=projected_vision,
            audio_embeds=projected_audio,
            max_length=max_length,
            **kwargs,
        )

    def generate_speech(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        max_length: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate speech tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] input token ids
            attention_mask: [batch_size, seq_len] attention mask
            vision_embeds: [batch_size, vision_seq_len, vision_hidden] vision embeddings
            audio_embeds: [batch_size, audio_seq_len, audio_hidden] audio embeddings
            max_length: Maximum speech token generation length

        Returns:
            [batch_size, max_length] generated speech token ids
        """
        # Get LLM outputs for TTS input
        with torch.no_grad():
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                audio_embeds=audio_embeds,
                output_hidden_states=True,
            )

            # Use last hidden state as TTS input
            if isinstance(lm_outputs, dict):
                hidden_states = lm_outputs["hidden_states"][-1]  # Last layer
            else:
                hidden_states = lm_outputs[-1]  # Last element

            tts_input = hidden_states[:, -1, :].unsqueeze(1)  # Last token, add seq dim

        # Generate speech tokens
        return self.tts_decoder.generate(tts_input, max_length=max_length, **kwargs)

    @classmethod
    def from_pretrained_minicpm(cls, model_path: str, **kwargs):
        """Create model from MiniCPM-o checkpoint (when available)"""
        # Load config from downloaded files
        config_path = Path.home() / ".cache/huggingface/hub/openbmb___MiniCPM-o-2_6/config.json"

        if config_path.exists():
            with open(config_path, "r") as f:
                minicpm_config = json.load(f)
        else:
            # Fallback to default config
            minicpm_config = {
                "vocab_size": 151700,
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "vision_config": {"hidden_size": 1152},
                "audio_config": {"hidden_size": 1024},
                "tts_config": {"model_type": "conditional_chattts", "llm_dim": 3584},
            }

        # Create our config
        config = MiniCPMOModelConfig(
            vocab_size=minicpm_config["vocab_size"],
            hidden_size=minicpm_config["hidden_size"],
            num_hidden_layers=minicpm_config["num_hidden_layers"],
            num_attention_heads=minicpm_config["num_attention_heads"],
            num_key_value_heads=minicpm_config.get("num_key_value_heads", 4),
            vision_hidden_size=minicpm_config["vision_config"]["hidden_size"],
            audio_hidden_size=minicpm_config["audio_config"]["hidden_size"],
            **kwargs,
        )

        return cls(config)

    @classmethod
    def from_qwen_checkpoint(cls, qwen_path: str, **kwargs):
        """Create model from Qwen2.5 checkpoint with added multimodal capabilities"""
        # Load Qwen2.5 as base
        multimodal_qwen = MultimodalQwen2ForCausalLM.from_qwen2_checkpoint(qwen_path)

        # Create our config
        config = MiniCPMOModelConfig(**kwargs)

        # Create full model
        model = cls(config)

        # Copy the multimodal Qwen2.5 weights
        model.language_model.load_state_dict(multimodal_qwen.state_dict())

        return model


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors"""
    tensor1_flat = tensor1.flatten().float()
    tensor2_flat = tensor2.flatten().float()

    mean1 = torch.mean(tensor1_flat)
    mean2 = torch.mean(tensor2_flat)
    std1 = torch.std(tensor1_flat)
    std2 = torch.std(tensor2_flat)

    covariance = torch.mean((tensor1_flat - mean1) * (tensor2_flat - mean2))
    pcc = covariance / (std1 * std2 + 1e-8)

    return pcc.item()


# Test functions
def test_minicpm_o_model():
    """Test the complete MiniCPM-o model"""

    # Create model
    config = MiniCPMOModelConfig()
    model = MiniCPMOForConditionalGeneration(config)

    batch_size, seq_len = 2, 10

    # Test text-only forward
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids=input_ids)
    print(f"Text-only forward: logits shape = {outputs['lm_logits'].shape}")
    assert outputs["lm_logits"].shape == (batch_size, seq_len, config.vocab_size)

    # Test with vision
    vision_embeds = torch.randn(batch_size, config.vision_seq_len, config.vision_hidden_size)
    outputs_with_vision = model(input_ids=input_ids, vision_embeds=vision_embeds)
    print(f"Vision forward: logits shape = {outputs_with_vision['lm_logits'].shape}")
    assert outputs_with_vision["lm_logits"].shape == (batch_size, seq_len, config.vocab_size)

    # Test with audio
    audio_embeds = torch.randn(batch_size, config.audio_seq_len, config.audio_hidden_size)
    outputs_with_audio = model(input_ids=input_ids, audio_embeds=audio_embeds)
    print(f"Audio forward: logits shape = {outputs_with_audio['lm_logits'].shape}")

    # Test with both modalities
    outputs_multimodal = model(input_ids=input_ids, vision_embeds=vision_embeds, audio_embeds=audio_embeds)
    print(f"Multimodal forward: logits shape = {outputs_multimodal['lm_logits'].shape}")

    # Test TTS generation
    speech_tokens = model.generate_speech(input_ids=input_ids)
    print(f"TTS generation: {speech_tokens.shape}")

    print("✓ MiniCPM-o model test passed!")


if __name__ == "__main__":
    test_minicpm_o_model()
