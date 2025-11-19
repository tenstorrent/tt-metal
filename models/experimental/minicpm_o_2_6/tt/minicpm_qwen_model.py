# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniCPM Qwen Model wrapper that uses tt_transformers library with MiniCPMTransformer.
"""

import os
import torch
import ttnn
from typing import Optional, Dict
from loguru import logger

from models.tt_transformers.tt.common import create_tt_model
from models.tt_transformers.tt.model_config import DecodersPrecision
from minicpm_transformer import MiniCPMTransformer


class MiniCPMQwenModel:
    """
    TTNN implementation of MiniCPM Qwen2.5 model.

    Uses tt_transformers library with a custom MiniCPMTransformer that includes
    cross-attention layers at positions 8, 16, 24 for multimodal fusion.

    Args:
        mesh_device: TTNN mesh device
        optimizations: Model optimizations (performance/accuracy)
        cross_attention_layers: List of layer indices where cross-attention is added
        max_seq_len: Maximum sequence length
        max_batch_size: Maximum batch size
        qwen_model_name: Which Qwen model to use as base (default: Qwen2.5-7B)
    """

    def __init__(
        self,
        mesh_device,
        optimizations=None,
        cross_attention_layers=[8, 16, 24],
        max_seq_len=2048,
        max_batch_size=1,
        qwen_model_name="Qwen/Qwen2.5-7B",
    ):
        self.mesh_device = mesh_device
        self.cross_attention_layers = cross_attention_layers
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.qwen_model_name = qwen_model_name

        # Set default optimizations if not provided
        if optimizations is None:
            optimizations = DecodersPrecision.performance(num_decoders=28, model_name="Qwen2.5-7B")

        logger.info(
            f"Creating MiniCPM Qwen model with {qwen_model_name} base and cross-attention layers: {cross_attention_layers}"
        )

        # Set HF_MODEL environment variable for tt_transformers
        original_hf_model = os.environ.get("HF_MODEL")
        os.environ["HF_MODEL"] = qwen_model_name

        try:
            # Create base Qwen model components using tt_transformers
            self.model_args, base_model, self.kv_cache, self.state_dict = create_tt_model(
                mesh_device,
                instruct=False,
                max_batch_size=max_batch_size,
                optimizations=optimizations,
                max_seq_len=max_seq_len,
                paged_attention_config=None,
                dtype=ttnn.bfloat16,
                state_dict=None,  # Will load weights manually
            )

            logger.info("✅ Base Qwen model created successfully")

            # Replace the Transformer with our MiniCPMTransformer
            self.model = MiniCPMTransformer(
                args=self.model_args,
                dtype=ttnn.bfloat16,
                mesh_device=mesh_device,
                state_dict=self.state_dict,
                weight_cache_path=self.model_args.weight_cache_path(ttnn.bfloat16),
                cross_attention_layers=cross_attention_layers,
            )

            logger.info("✅ MiniCPMTransformer created successfully")

        except Exception as e:
            logger.error(f"❌ Failed to create MiniCPM Qwen model: {e}")
            logger.error("   This is a production model - no fallback implementations allowed.")
            raise RuntimeError(f"MiniCPM Qwen model initialization failed: {e}") from e
        finally:
            # Restore original HF_MODEL if it was set
            if original_hf_model is not None:
                os.environ["HF_MODEL"] = original_hf_model
            elif "HF_MODEL" in os.environ:
                del os.environ["HF_MODEL"]

        # Initialize generator for model execution
        self.generator = None
        self.weights_loaded = False
        logger.info("✅ MiniCPM Qwen model initialization complete")

    def load_weights(self, weights_dict: Dict[str, torch.Tensor]):
        """
        Load weights into the model.

        Args:
            weights_dict: Dictionary mapping weight keys to tensors
                         Keys should follow model_key_mapping.txt structure with 'llm.' prefix
        """
        logger.info("Loading weights into MiniCPM Qwen model...")

        try:
            # Load base Qwen weights (these go to the MiniCPMTransformer)
            if self.model is not None:
                # The MiniCPMTransformer inherits weight loading from the base Transformer
                # We need to map MiniCPM keys to Qwen2.5 keys
                base_weights = {}
                cross_attn_weights = {}

                for k, v in weights_dict.items():
                    if k.startswith("llm."):
                        if any(f".layers.{idx}.cross_attn" in k for idx in self.cross_attention_layers):
                            # This is a cross-attention weight
                            cross_attn_weights[k] = v
                        else:
                            # This is a base Qwen weight - map to Qwen2.5 format
                            qwen_key = k.replace("llm.", "")
                            base_weights[qwen_key] = v
                    elif not any(f".layers.{idx}.cross_attn" in k for idx in self.cross_attention_layers):
                        # Include keys that are already in Qwen format
                        base_weights[k] = v
                    else:
                        # Cross-attention weight in different format
                        cross_attn_weights[k] = v

                # Update the state_dict with base weights
                self.state_dict.update(base_weights)
                logger.info(f"✅ Loaded {len(base_weights)} base Qwen weights")

                # Load cross-attention weights into the cross-attention modules
                for layer_idx in self.cross_attention_layers:
                    layer_weights = {}
                    for k, v in cross_attn_weights.items():
                        if f".layers.{layer_idx}.cross_attn" in k:
                            # Extract the cross-attention specific key
                            ca_key = k.split(f".layers.{layer_idx}.cross_attn.")[-1]
                            layer_weights[ca_key] = v

                    if layer_weights:
                        self.model.cross_attn_modules[layer_idx].load_weights(layer_weights)
                        logger.info(f"✅ Loaded cross-attention weights for layer {layer_idx}")

            self.weights_loaded = True
            logger.info("✅ All weights loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load weights: {e}")
            raise

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the MiniCPM Qwen model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            encoder_hidden_states: Multimodal embeddings [batch_size, seq_len_enc, hidden_size]
            attention_mask: Attention mask for cross-attention

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        if not self.weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        logger.info(f"Forward pass - input shape: {input_ids.shape}")
        if encoder_hidden_states is not None:
            logger.info(f"Forward pass - encoder shape: {encoder_hidden_states.shape}")

        # Convert inputs to TTNN format
        input_ids_ttnn = ttnn.from_torch(
            input_ids.unsqueeze(0),  # Add batch dimension
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
        )

        # Convert encoder_hidden_states to TTNN format if provided
        encoder_hidden_states_ttnn = None
        if encoder_hidden_states is not None:
            encoder_hidden_states_ttnn = ttnn.from_torch(
                encoder_hidden_states.unsqueeze(0),  # Add batch dimension
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

        # Run forward pass directly on the model
        # The MiniCPMTransformer expects encoder_hidden_states parameter
        logits = self.model(
            input_ids_ttnn,
            current_pos=0,
            encoder_hidden_states=encoder_hidden_states_ttnn,
            mode="prefill",
        )

        # Convert back to PyTorch (handle mesh device)
        try:
            if hasattr(self.mesh_device, "get_num_devices") and self.mesh_device.get_num_devices() > 1:
                # For mesh devices, try different concatenation strategies
                try:
                    # Try concatenating along vocab dimension (dim=-1)
                    mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
                    pytorch_logits_full = ttnn.to_torch(logits, mesh_composer=mesh_composer)
                    # If it's replicated, take the first half (original vocab size)
                    vocab_size = pytorch_logits_full.shape[-1] // 2
                    logits = pytorch_logits_full[..., :vocab_size]
                except Exception:
                    try:
                        # Try concatenating along sequence dimension (dim=1)
                        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=1)
                        logits = ttnn.to_torch(logits, mesh_composer=mesh_composer)
                    except Exception:
                        # Fall back to from_device approach
                        logits = ttnn.to_torch(ttnn.from_device(logits))
            else:
                # For single device, use the standard approach
                logits = ttnn.to_torch(ttnn.from_device(logits))
        except Exception as e:
            logger.warning(f"Tensor conversion failed: {e}, trying fallback")
            if hasattr(logits, "cpu"):
                logits = logits.cpu()
            if hasattr(logits, "numpy"):
                logits = torch.from_numpy(logits.numpy())

        # Ensure float32
        logits = logits.float()

        logger.info(f"✅ TTNN forward pass complete - output shape: {logits.shape}")
        return logits

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "model") and self.model is not None:
                # Cleanup will be handled by the MiniCPMTransformer's parent class
                pass
        except:
            pass
