# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phi-1 Model Demo for Tenstorrent Wormhole

This demo runs inference on the microsoft/phi-1 model using tt-transformers framework.
Usage:
    python demo_phi1.py
    
Environment Variables:
    PHI1_CKPT_DIR: Path to Phi-1 checkpoint directory (default: None, uses random weights)
    PHI1_TOKENIZER_PATH: Path to Phi-1 tokenizer (default: microsoft/phi-1)
"""

import os
import sys
import torch
from loguru import logger
from pathlib import Path

# Add tt-metal to path
tt_metal_path = Path(__file__).parent.parent.parent.parent.parent / "tt-metal-main"
if tt_metal_path.exists():
    sys.path.insert(0, str(tt_metal_path))

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.tt_transformers.tt.common import create_tt_model, preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator, SamplingParams
from models.tt_transformers.tt.model_config import ModelOptimizations


def run_phi1_demo(
    device,
    prompt: str = "Write a Python function to calculate factorial:",
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    """
    Run Phi-1 inference demo on Tenstorrent hardware.
    
    Args:
        device: TT device to run on
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    """
    logger.info("=" * 60)
    logger.info("Phi-1 Model Demo on Tenstorrent Wormhole")
    logger.info("=" * 60)
    
    model_name = "phi-1"
    
    # Model configuration
    logger.info(f"Loading Phi-1 model configuration...")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Max tokens to generate: {max_tokens}")
    
    # Create model args
    model_args = create_tt_model(
        device=device,
        model_name=model_name,
        optimizations=ModelOptimizations.accuracy(model_name),
        max_batch_size=1,
        max_seq_len=2048,
    )
    
    # Initialize generator
    logger.info("Initializing generator...")
    generator = Generator(model_args)
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # Tokenize input
    logger.info("Tokenizing input...")
    input_ids = model_args.tokenizer.encode(prompt, return_tensors="pt")
    logger.info(f"Input tokens: {input_ids.shape[-1]}")
    
    # Run inference
    logger.info("Running inference...")
    logger.info("-" * 60)
    
    outputs = generator.generate(
        prompts=[prompt],
        sampling_params=sampling_params,
    )
    
    # Display output
    generated_text = outputs[0]
    logger.info("-" * 60)
    logger.info("Generated Output:")
    logger.info(generated_text)
    logger.info("=" * 60)
    
    return generated_text


def main():
    """Main entry point for Phi-1 demo."""
    
    # Check device availability
    if not is_wormhole_b0():
        logger.warning("This demo is optimized for Wormhole hardware")
    
    # Get device
    device = ttnn.open_device(device_id=0)
    logger.info(f"Using device: {device}")
    
    try:
        # Run demo
        prompt = os.getenv("PHI1_PROMPT", "Write a Python function to calculate factorial:")
        max_tokens = int(os.getenv("PHI1_MAX_TOKENS", "128"))
        
        result = run_phi1_demo(
            device=device,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        
        logger.info("Demo completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
