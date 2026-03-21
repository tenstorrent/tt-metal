# SPDX-License-Identifier: MIT

import torch
import ttnn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from models.experimental.speecht5.tt_speecht5_encoder import TtSpeechT5Encoder
from models.experimental.speecht5.tt_speecht5_decoder import TtSpeechT5Decoder
from models.experimental.speecht5.tt_speecht5_speech_encoder_prenet import TtSpeechT5SpeechEncoderPreNet
from models.experimental.speecht5.tt_speecht5_speech_decoder_prenet import TtSpeechT5SpeechDecoderPreNet
from models.experimental.speecht5.tt_speecht5_speech_decoder_postnet import TtSpeechT5SpeechDecoderPostNet


@dataclass
class SpeechT5VCConfig:
    hidden_size: int = 768
    encoder_layers: int = 12
    encoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072
    decoder_layers: int = 6
    decoder_attention_heads: int = 12
    decoder_ffn_dim: int = 3072
    max_speech_positions: int = 4096
    max_text_positions: int = 450
    vocab_size: int = 81
    speech_decoder_prenet_layers: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_units: int = 512
    speech_decoder_postnet_kernel: int = 5
    num_mel_bins: int = 80
    reduction_factor: int = 2
    speaker_embedding_dim: int = 512
    dropout: float = 0.1
    use_guided_attention_loss: bool = False
    use_cache: bool = True


class TtSpeechT5ForVoiceConversion:
    def __init__(self, config: SpeechT5VCConfig, device, mesh_device=None):
        self.config = config
        self.device = device
        self.mesh_device = mesh_device
        self.use_multi_device = mesh_device is not None
        
        self.speech_encoder_prenet = TtSpeechT5SpeechEncoderPreNet(config, device)
        self.encoder = TtSpeechT5Encoder(config, device, mesh_device)
        
        self.speech_decoder_prenet = TtSpeechT5SpeechDecoderPreNet(config, device)
        self.decoder = TtSpeechT5Decoder(config, device, mesh_device)
        self.speech_decoder_postnet = TtSpeechT5SpeechDecoderPostNet(config, device)
        
        # Speaker embedding projection
        self.speaker_projection = ttnn.Linear(
            config.speaker_embedding_dim,
            config.hidden_size,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # Output projection for mel spectrograms
        self.speech_decoder_prenet_proj = ttnn.Linear(
            config.hidden_size,
            config.num_mel_bins * config.reduction_factor,
            device=device,
            dtype=ttnn.bfloat16
        )

    def preprocess_inputs(self, input_values: torch.Tensor) -> ttnn.Tensor:
        """Convert input speech features to TTNN tensor"""
        batch_size, seq_len, num_mels = input_values.shape
        
        # Pad to make dimensions compatible with TTNN
        padded_seq_len = ((seq_len + 31) // 32) * 32
        padded_num_mels = ((num_mels + 31) // 32) * 32
        
        if seq_len != padded_seq_len or num_mels != padded_num_mels:
            padded_input = torch.zeros(batch_size, padded_seq_len, padded_num_mels, dtype=input_values.dtype)
            padded_input[:, :seq_len, :num_mels] = input_values
            input_values = padded_input
        
        input_tensor = ttnn.from_torch(
            input_values,
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        return input_tensor

    def preprocess_speaker_embeddings(self, speaker_embeddings: torch.Tensor) -> ttnn.Tensor:
        """Convert speaker x-vector embeddings to TTNN tensor"""
        batch_size, embedding_dim = speaker_embeddings.shape
        
        # Pad embedding dimension if needed
        padded_dim = ((embedding_dim + 31) // 32) * 32
        if embedding_dim != padded_dim:
            padded_embeddings = torch.zeros(batch_size, padded_dim, dtype=speaker_embeddings.dtype)
            padded_embeddings[:, :embedding_dim] = speaker_embeddings
            speaker_embeddings = padded_embeddings
        
        speaker_tensor = ttnn.from_torch(
            speaker_embeddings,
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        return speaker_tensor

    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass for voice conversion
        
        Args:
            input_values: Input mel spectrograms [batch_size, seq_len, num_mel_bins]
            speaker_embeddings: Target speaker x-vector embeddings [batch_size, speaker_embedding_dim]
            labels: Target mel spectrograms for training [batch_size, target_seq_len, num_mel_bins]
            return_dict: Whether to return a dictionary
        
        Returns:
            Dictionary containing:
            - spectrogram: Generated mel spectrogram
            - spectrogram_postnet: Post-processed mel spectrogram
            - loss: Training loss (if labels provided)
        """
        batch_size = input_values.shape[0]
        
        # Preprocess inputs
        input_tensor = self.preprocess_inputs(input_values)
        speaker_tensor = self.preprocess_speaker_embeddings(speaker_embeddings)
        
        # Encode input speech
        encoder_inputs = self.speech_encoder_prenet(input_tensor)
        encoder_outputs = self.encoder(encoder_inputs)
        
        # Project speaker embeddings
        speaker_proj = self.speaker_projection(speaker_tensor)
        speaker_proj = ttnn.unsqueeze(speaker_proj, 1)  # Add sequence dimension
        
        # Expand speaker embeddings to match encoder sequence length
        encoder_seq_len = encoder_outputs.shape[1]
        speaker_expanded = ttnn.repeat(speaker_proj, [1, encoder_seq_len, 1])
        
        # Add speaker conditioning to encoder outputs
        conditioned_encoder_outputs = ttnn.add(encoder_outputs, speaker_expanded)
        
        # Prepare decoder inputs
        if labels is not None:
            # Training mode - use teacher forcing
            decoder_input_tensor = self.preprocess_inputs(labels)
            decoder_inputs = self.speech_decoder_prenet(decoder_input_tensor)
            
            # Decoder forward pass
            decoder_outputs = self.decoder(
                decoder_inputs,
                encoder_hidden_states=conditioned_encoder_outputs
            )
            
            # Generate mel spectrogram predictions
            spectrogram = self.speech_decoder_prenet_proj(decoder_outputs)
            spectrogram = ttnn.reshape(spectrogram, [batch_size, -1, self.config.num_mel_bins])
            
            # Apply postnet for refinement
            spectrogram_postnet = self.speech_decoder_postnet(spectrogram)
            spectrogram_final = ttnn.add(spectrogram, spectrogram_postnet)
            
            # Calculate loss
            loss = self._compute_loss(spectrogram_final, decoder_input_tensor)
            
        else:
            # Inference mode - autoregressive generation
            spectrogram, spectrogram_final = self._generate_autoregressive(
                conditioned_encoder_outputs, batch_size
            )
            loss = None
        
        # Convert back to torch tensors
        spectrogram_torch = ttnn.to_torch(spectrogram)
        spectrogram_final_torch = ttnn.to_torch(spectrogram_final)
        
        if not return_dict:
            return (spectrogram_final_torch, spectrogram_torch, loss)
        
        return {
            "spectrogram": spectrogram_torch,
            "spectrogram_postnet": spectrogram_final_torch,
            "loss": loss
        }

    def _generate_autoregressive(self, encoder_outputs, batch_size, max_length=1000):
        """Generate mel spectrogram autoregressively during inference"""
        generated_mels = []
        
        # Initialize with zero frame
        prev_mel = ttnn.zeros(
            [batch_size, 1, self.config.num_mel_bins],
            dtype=ttnn.bfloat16,
            device=self.device
        )
        
        for step in range(max_length):
            # Process previous mel through decoder prenet
            decoder_input = self.speech_decoder_prenet(prev_mel)
            
            # Decoder forward pass
            decoder_output = self.decoder(
                decoder_input,
                encoder_hidden_states=encoder_outputs
            )
            
            # Generate next mel frame
            next_mel_logits = self.speech_decoder_prenet_proj(decoder_output)
            next_mel = ttnn.reshape(next_mel_logits, [batch_size, 1, self.config.num_mel_bins])
            
            generated_mels.append(next_mel)
            prev_mel = next_mel
            
            # Simple stopping condition - in practice would use attention weights
            if step > 0 and step % 50 == 0:
                # Check for convergence or silence
                break
        
        # Concatenate all generated frames
        spectrogram = ttnn.concat(generated_mels, dim=1)
        
        # Apply postnet
        spectrogram_postnet = self.speech_decoder_postnet(spectrogram)
        spectrogram_final = ttnn.add(spectrogram, spectrogram_postnet)
        
        return spectrogram, spectrogram_final

    def _compute_loss(self, predictions, targets):
        """Compute L1 loss for mel spectrogram prediction"""
        loss_tensor = ttnn.subtract(predictions, targets)
        loss_tensor = ttnn.abs(loss_tensor)
        loss = ttnn.mean(loss_tensor)
        return ttnn.to_torch(loss).item()

    def generate_speech(
        self,
        input_values: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        vocoder=None
    ) -> torch.Tensor:
        """
        Generate speech waveform from input speech and target speaker embedding
        
        Args:
            input_values: Input mel spectrograms
            speaker_embeddings: Target speaker x-vector embeddings  
            vocoder: Optional vocoder for waveform generation
            
        Returns:
            Generated mel spectrogram or waveform if vocoder provided
        """
        with torch.no_grad():
            outputs = self.forward(
                input_values=input_values,
                speaker_embeddings=speaker_embeddings,
                return_dict=True
            )
            
            generated_mel = outputs["spectrogram_postnet"]
            
            if vocoder is not None:
                # Convert mel to waveform using vocoder
                waveform = vocoder(generated_mel)
                return waveform
            else:
                return generated_mel

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load model weights from PyTorch state dict"""
        # Convert PyTorch weights to TTNN format and load
        for name, param in state_dict.items():
            if "encoder" in name:
                self.encoder.load_parameter(name, param)
            elif "decoder" in name:
                self.decoder.load_parameter(name, param)
            elif "speech_encoder_prenet" in name:
                self.speech_encoder_prenet.load_parameter(name, param)
            elif "speech_decoder_prenet" in name:
                self.speech_decoder_prenet.load_parameter(name, param)
            elif "speech_decoder_postnet" in name:
                self.speech_decoder_postnet.load_parameter(name, param)