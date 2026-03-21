# SPDX-License-Identifier: MIT

import ttnn
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import math


class SpeechT5PreNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.linear(x, self.input_dim, self.hidden_dim, bias=True)
        x = ttnn.relu(x)
        x = ttnn.dropout(x, self.dropout)
        
        x = ttnn.linear(x, self.hidden_dim, self.output_dim, bias=True)
        x = ttnn.relu(x)
        x = ttnn.dropout(x, self.dropout)
        
        return x


class SpeechT5FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        
    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = ttnn.linear(hidden_states, self.hidden_size, self.intermediate_size, bias=True)
        hidden_states = ttnn.gelu(hidden_states)
        hidden_states = ttnn.dropout(hidden_states, self.dropout)
        
        hidden_states = ttnn.linear(hidden_states, self.intermediate_size, self.hidden_size, bias=True)
        hidden_states = ttnn.dropout(hidden_states, self.dropout)
        
        return hidden_states


class SpeechT5Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0, is_decoder: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = dropout
        self.is_decoder = is_decoder
        
        if self.head_size * num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
            
    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_bias: Optional[ttnn.Tensor] = None,
        key_value_states: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor]] = None,
        output_attentions: bool = False
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[Tuple[ttnn.Tensor]]]:
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project to query, key, value
        query_states = ttnn.linear(hidden_states, self.hidden_size, self.hidden_size, bias=False)
        
        if key_value_states is not None:
            key_states = ttnn.linear(key_value_states, self.hidden_size, self.hidden_size, bias=False)
            value_states = ttnn.linear(key_value_states, self.hidden_size, self.hidden_size, bias=False)
        else:
            key_states = ttnn.linear(hidden_states, self.hidden_size, self.hidden_size, bias=False)
            value_states = ttnn.linear(hidden_states, self.hidden_size, self.hidden_size, bias=False)
            
        # Reshape for multi-head attention
        query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_size))
        key_states = ttnn.reshape(key_states, (batch_size, -1, self.num_heads, self.head_size))
        value_states = ttnn.reshape(value_states, (batch_size, -1, self.num_heads, self.head_size))
        
        # Transpose to [batch_size, num_heads, seq_length, head_size]
        query_states = ttnn.transpose(query_states, -3, -2)
        key_states = ttnn.transpose(key_states, -3, -2)
        value_states = ttnn.transpose(value_states, -3, -2)
        
        # Compute attention scores
        attention_scores = ttnn.matmul(query_states, ttnn.transpose(key_states, -2, -1))
        attention_scores = ttnn.multiply(attention_scores, 1.0 / math.sqrt(self.head_size))
        
        if position_bias is not None:
            attention_scores = ttnn.add(attention_scores, position_bias)
            
        if attention_mask is not None:
            attention_scores = ttnn.add(attention_scores, attention_mask)
            
        attention_probs = ttnn.softmax(attention_scores, dim=-1)
        attention_probs = ttnn.dropout(attention_probs, self.dropout)
        
        # Apply attention to values
        context_layer = ttnn.matmul(attention_probs, value_states)
        
        # Transpose back and reshape
        context_layer = ttnn.transpose(context_layer, -3, -2)
        context_layer = ttnn.reshape(context_layer, (batch_size, seq_length, self.hidden_size))
        
        # Final projection
        attention_output = ttnn.linear(context_layer, self.hidden_size, self.hidden_size, bias=False)
        
        outputs = (attention_output, attention_probs if output_attentions else None, past_key_value)
        return outputs


class SpeechT5Layer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1, layer_norm_eps: float = 1e-5, is_decoder: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_decoder = is_decoder
        self.layer_norm_eps = layer_norm_eps
        
        self.attention = SpeechT5Attention(hidden_size, num_heads, dropout, is_decoder)
        if is_decoder:
            self.crossattention = SpeechT5Attention(hidden_size, num_heads, dropout, is_decoder=False)
        self.feed_forward = SpeechT5FeedForward(hidden_size, intermediate_size, dropout)
        
    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_bias: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[ttnn.Tensor, ...]:
        
        # Self-attention
        residual = hidden_states
        hidden_states = ttnn.layer_norm(hidden_states, eps=self.layer_norm_eps)
        
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions
        )
        hidden_states = attention_outputs[0]
        hidden_states = ttnn.add(hidden_states, residual)
        
        # Cross-attention for decoder
        if self.is_decoder and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = ttnn.layer_norm(hidden_states, eps=self.layer_norm_eps)
            
            cross_attention_outputs = self.crossattention(
                hidden_states,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                output_attentions=output_attentions
            )
            hidden_states = cross_attention_outputs[0]
            hidden_states = ttnn.add(hidden_states, residual)
        
        # Feed forward
        residual = hidden_states
        hidden_states = ttnn.layer_norm(hidden_states, eps=self.layer_norm_eps)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = ttnn.add(hidden_states, residual)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_outputs[1],)
            if self.is_decoder and encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
                
        return outputs


class SpeechT5Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = [
            SpeechT5Layer(hidden_size, num_heads, intermediate_size, dropout, is_decoder=False)
            for _ in range(num_layers)
        ]
        
    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[ttnn.Tensor, ...]:
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class SpeechT5Decoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = [
            SpeechT5Layer(hidden_size, num_heads, intermediate_size, dropout, is_decoder=True)
            for _ in range(num_layers)
        ]
        
    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[ttnn.Tensor, ...]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)


class SpeechT5PostNet(nn.Module):
    def __init__(self, mel_dim: int, postnet_dim: int, num_layers: int = 5, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.mel_dim = mel_dim
        self.postnet_dim = postnet_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        
    def __call__(self, mel_spectrogram: ttnn.Tensor) -> ttnn.Tensor:
        x = mel_spectrogram
        
        for i in range(self.num_layers):
            if i == 0:
                x = ttnn.conv1d(x, self.mel_dim, self.postnet_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
            elif i == self.num_layers - 1:
                x = ttnn.conv1d(x, self.postnet_dim, self.mel_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
            else:
                x = ttnn.conv1d(x, self.postnet_dim, self.postnet_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
                
            if i < self.num_layers - 1:
                x = ttnn.tanh(x)
                
            x = ttnn.dropout(x, self.dropout)
            
        return ttnn.add(mel_spectrogram, x)


class SpeechT5SpeakerEmbedding(nn.Module):
    def __init__(self, speaker_embedding_dim: int, hidden_size: int):
        super().__init__()
        self.speaker_embedding_dim = speaker_embedding_dim
        self.hidden_size = hidden_size
        
    def __call__(self, speaker_embeddings: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(speaker_embeddings, self.speaker_embedding_dim, self.hidden_size, bias=True)


class SpeechT5VoiceConversionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        encoder_layers: int = 12,
        decoder_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        mel_dim: int = 80,
        reduction_factor: int = 2,
        speaker_embedding_dim: int = 512,
        postnet_dim: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mel_dim = mel_dim
        self.reduction_factor = reduction_factor
        self.speaker_embedding_dim = speaker_embedding_dim
        
        # Speech pre-net
        self.speech_prenet = SpeechT5PreNet(mel_dim, 256, hidden_size, dropout)
        
        # Shared encoder
        self.encoder = SpeechT5Encoder(hidden_size, encoder_layers, num_heads, intermediate_size, dropout)
        
        # Decoder
        self.decoder = SpeechT5Decoder(hidden_size, decoder_layers, num_heads, intermediate_size, dropout)
        
        # Speaker embedding projection
        self.speaker_projection = SpeechT5SpeakerEmbedding(speaker_embedding_dim, hidden_size)
        
        # Decoder pre-net
        self.decoder_prenet = SpeechT5PreNet(mel_dim, 256, hidden_size, dropout)
        
        # Post-net
        self.postnet = SpeechT5PostNet(mel_dim, postnet_dim, num_layers=5, dropout=dropout)
        
        # Output projection
        self.mel_output_projection = ttnn.linear
        
    def __call__(
        self,
        input_values: ttnn.Tensor,
        speaker_embeddings: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        decoder_input_values: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        
        batch_size, seq_length, _ = input_values.shape
        
        # Speech pre-net
        encoder_hidden_states = self.speech_prenet(input_values)
        
        # Encoder
        encoder_outputs = self.encoder(
            encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        encoder_hidden_states = encoder_outputs[0]
        
        # Speaker embedding
        speaker_embeddings_projected = self.speaker_projection(speaker_embeddings)
        speaker_embeddings_expanded = ttnn.expand_dims(speaker_embeddings_projected, 1)
        
        # Initialize decoder input
        if decoder_input_values is None:
            decoder_input_values = ttnn.zeros((batch_size, 1, self.mel_dim))
            
        # Decoder pre-net
        decoder_hidden_states = self.decoder_prenet(decoder_input_values)
        
        # Add speaker embedding to decoder input
        decoder_hidden_states = ttnn.add(decoder_hidden_states, speaker_embeddings_expanded)
        
        # Decoder
        decoder_outputs = self.decoder(
            decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        decoder_hidden_states = decoder_outputs[0]
        
        # Mel spectrogram output projection
        mel_outputs_before_postnet = ttnn.linear(
            decoder_hidden_states, 
            self.hidden_size, 
            self.mel_dim * self.reduction_factor, 
            bias=True
        )
        mel_outputs_before_postnet = ttnn.reshape(
            mel_outputs_before_postnet,
            (batch_size, -1, self.mel_dim)
        )
        
        # Post-net
        mel_outputs = self.postnet(mel_outputs_before_postnet)
        
        # Stop token prediction
        stop_tokens = ttnn.linear(decoder_hidden_states, self.hidden_size, 1, bias=True)
        stop_tokens = ttnn.sigmoid(stop_tokens)
        
        return mel_outputs, mel_outputs_before_postnet, stop_tokens
    
    def generate_speech(
        self,
        input_values: ttnn.Tensor,
        speaker_embeddings: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        max_length: int = 1000,
        threshold: float = 0.5
    ) -> ttnn.Tensor:
        """Generate speech mel-spectrogram using autoregressive decoding"""
        
        batch_size = input_values.shape[0]
        
        # Encode input speech
        encoder_hidden_states = self.speech_prenet(input_values)
        encoder_outputs = self.encoder(encoder_hidden_states, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs[0]
        
        # Speaker embedding
        speaker_embeddings_projected = self.speaker_projection(speaker_embeddings)
        
        # Initialize output sequence
        decoder_input = ttnn.zeros((batch_size, 1, self.mel_dim))
        mel_outputs = []
        
        for step in range(max_length):
            # Decoder pre-net
            decoder_hidden_states = self.decoder_prenet(decoder_input)
            
            # Add speaker embedding
            speaker_embeddings_expanded = ttnn.expand_dims(speaker_embeddings_projected, 1)
            decoder_hidden_states = ttnn.add(decoder_hidden_states, speaker_embeddings_expanded)
            
            # Decoder
            decoder_outputs = self.decoder(
                decoder_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask
            )
            decoder_hidden_states = decoder_outputs[0]
            
            # Generate mel frame
            mel_frame = ttnn.linear(decoder_hidden_states, self.hidden_size, self.mel_dim, bias=True)
            
            # Stop token prediction
            stop_token = ttnn.linear(decoder_hidden_states, self.hidden_size, 1, bias=True)
            stop_token = ttnn.sigmoid(stop_token)
            
            mel_outputs.append(mel_frame)
            
            # Check for stop condition
            if ttnn.mean(stop_token).item() > threshold:
                break
                
            # Update decoder input for next step
            decoder_input = mel_frame
            
        # Concatenate mel outputs
        mel_sequence = ttnn.concat(mel_outputs, dim=1)
        
        # Apply post-net
        mel_sequence = self.postnet(mel_sequence)
        
        return mel_sequence