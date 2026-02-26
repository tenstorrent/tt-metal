# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B Demo for Tenstorrent Hardware.

This demo showcases multimodal visual question answering using Molmo2-8B
running on Tenstorrent devices (N150/N300/T3K).

Features:
- Vision-language multimodal inference
- KV cache for efficient autoregressive generation
- Optional tracing for improved performance

Usage:
    # Run with default image and prompt
    python -m models.demos.molmo2.demo.demo

    # Run with custom image
    python -m models.demos.molmo2.demo.demo --image path/to/image.jpg

    # Run with tracing enabled
    python -m models.demos.molmo2.demo.demo --use-trace
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image

import ttnn

# Default paths
DEMO_DIR = Path(__file__).parent
DEFAULT_IMAGE = DEMO_DIR / "dog.jpg"
MODEL_ID = "allenai/Molmo2-8B"


def load_processor():
    """Load the Molmo2 tokenizer from HuggingFace."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    return tokenizer


def load_model_weights():
    """Load all model weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info(f"Loading model weights from {MODEL_ID}")
    state_dict = load_state_dict_from_safetensors(MODEL_ID)
    logger.info(f"Loaded {len(state_dict)} weight tensors")
    return state_dict


def preprocess_image(image_path: str, target_size: int = 378):
    """
    Preprocess image for Molmo2.

    Args:
        image_path: Path to image file
        target_size: Target image size (378 for Molmo2)

    Returns:
        Tuple of (PIL Image, pixel_values tensor)
    """
    import numpy as np

    logger.info(f"Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Resize to model's expected size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    logger.info(f"Image resized to {target_size}x{target_size}")

    # Convert to tensor and normalize (ImageNet normalization)
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Convert to [1, 3, H, W] tensor
    pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()

    return image, pixel_values


def create_model(mesh_device, state_dict, num_layers: Optional[int] = None):
    """
    Create the Molmo2 TTNN model.

    Args:
        mesh_device: TTNN device or mesh device
        state_dict: Model state dict
        num_layers: Optional number of text layers (default: 36)

    Returns:
        Molmo2Model instance
    """
    from models.demos.molmo2.tt.molmo2_model import Molmo2Model

    logger.info("Creating Molmo2 TTNN model")

    text_num_layers = num_layers if num_layers is not None else 36

    model = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        # Vision config
        vit_num_layers=25,
        vit_hidden_dim=1152,
        vit_intermediate_dim=4304,
        vit_num_heads=16,
        vit_head_dim=72,
        patch_size=14,
        image_size=378,
        feature_layers=(18, 24),
        # Adapter config
        adapter_hidden_dim=1152,
        adapter_intermediate_dim=12288,
        adapter_num_heads=16,
        adapter_head_dim=72,
        # Text config
        text_num_layers=text_num_layers,
        text_hidden_dim=4096,
        text_intermediate_dim=12288,
        text_num_heads=32,
        text_num_kv_heads=8,
        text_head_dim=128,
        vocab_size=152064,
        max_seq_len=8192,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        dtype=ttnn.bfloat8_b,
    )

    logger.info("Model created successfully")
    return model


class Molmo2Generator:
    """
    Molmo2 generator with separate tracing for prefill and decode.

    Tracing captures the computation graph and replays it for improved performance.
    - Prefill trace: processes the full input sequence
    - Decode trace: processes one token at a time with KV cache
    """

    def __init__(
        self,
        mesh_device,
        model,
        tokenizer,
        num_layers: int,
        batch_size: int = 1,
        max_seq_len: int = 2048,
    ):
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Separate trace state for prefill and decode
        self.prefill_traces = {}  # {seq_len: (trace_id, trace_inputs, trace_output)}
        self.decode_trace_id = None
        self.decode_trace_inputs = None
        self.decode_trace_output = None

        # KV cache (initialized on first run)
        self.kv_caches = None
        self.current_pos = None

        # Mesh mapper
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

    def init_kv_cache(self):
        """Initialize KV cache for generation."""
        from models.demos.molmo2.tt.text_model import init_decode_position, init_kv_cache

        if self.kv_caches is None:
            self.kv_caches = init_kv_cache(
                mesh_device=self.mesh_device,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                num_kv_heads=8,
                max_seq_len=self.max_seq_len,
                head_dim=128,
                dtype=ttnn.bfloat8_b,
            )
            self.current_pos = init_decode_position(
                mesh_device=self.mesh_device,
                batch_size=self.batch_size,
                initial_pos=0,
            )

    def reset_kv_cache(self, start_pos: int = 0):
        """Reset KV cache position for new generation."""
        if self.current_pos is not None:
            pos_tensor = torch.full((self.batch_size,), start_pos, dtype=torch.int32)
            pos_ttnn = ttnn.from_torch(
                pos_tensor,
                dtype=ttnn.int32,
                device=self.mesh_device,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(pos_ttnn, self.current_pos)
            ttnn.deallocate(pos_ttnn)

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Prepare text model inputs by processing vision and fusing embeddings.

        This must be done BEFORE trace capture since it involves host-device transfers.

        Args:
            input_ids: Input token IDs
            pixel_values: Preprocessed image tensor
            pooled_patches_idx: Indices for vision pooling

        Returns:
            Tuple of (hidden_states_ttnn, hidden_states_torch)
        """
        # Process vision and prepare fused embeddings (this has host-device transfers)
        if pixel_values is not None and pooled_patches_idx is not None:
            visual_embeddings = self.model.embed_image(pixel_values, pooled_patches_idx)
            hidden_states = self.model.prepare_inputs_for_multimodal(input_ids, visual_embeddings)
        else:
            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            hidden_states = self.model.text_model.embed_tokens(input_ids_ttnn)
            hidden_states = ttnn.to_torch(hidden_states).squeeze(0).squeeze(0)

        # Keep torch version for later use
        hidden_states_torch = hidden_states.unsqueeze(0).unsqueeze(0).clone()

        # Convert to TTNN tensor
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return hidden_states_ttnn, hidden_states_torch

    def _allocate_prefill_trace_tensors(
        self,
        seq_len: int,
        hidden_dim: int = 4096,
    ) -> dict:
        """
        Pre-allocate all tensors needed for traced prefill.

        Args:
            seq_len: Sequence length
            hidden_dim: Hidden dimension

        Returns:
            Dict with pre-allocated tensors
        """
        # Allocate hidden states input tensor
        hidden_states_shape = [1, 1, seq_len, hidden_dim]
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape(hidden_states_shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-compute rotation matrices (these will be used during trace)
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

        # Allocate rot_mats tensors (we'll copy into these)
        trace_cos = ttnn.allocate_tensor_on_device(
            rot_mats[0].shape,
            rot_mats[0].dtype,
            rot_mats[0].layout,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_sin = ttnn.allocate_tensor_on_device(
            rot_mats[1].shape,
            rot_mats[1].dtype,
            rot_mats[1].layout,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Copy initial values
        ttnn.copy(rot_mats[0], trace_cos)
        ttnn.copy(rot_mats[1], trace_sin)

        # Clean up temporary rot_mats
        ttnn.deallocate(rot_mats[0])
        ttnn.deallocate(rot_mats[1])

        return {
            "hidden_states": trace_hidden_states,
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
        }

    def _capture_prefill_trace(
        self,
        trace_tensors: dict,
    ) -> Tuple[int, ttnn.Tensor]:
        """
        Capture trace for text model prefill phase.

        Uses pre-allocated tensors for all inputs.

        Args:
            trace_tensors: Dict with pre-allocated tensors

        Returns:
            Tuple of (trace_id, trace_output)
        """
        logger.info("Compiling text model prefill (first run)...")

        # First run to compile (using pre-allocated tensors)
        rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
        logits, _ = self.model.text_model.forward(
            hidden_states=trace_tensors["hidden_states"],
            start_pos=0,
            attn_mask=None,
            kv_caches=None,
            rot_mats=rot_mats,
        )
        logger.info("Text model prefill compiled")

        # Capture trace
        logger.info("Capturing text model prefill trace...")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        logits_trace, _ = self.model.text_model.forward(
            hidden_states=trace_tensors["hidden_states"],
            start_pos=0,
            attn_mask=None,
            kv_caches=None,
            rot_mats=rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Text model prefill trace captured")

        return trace_id, logits_trace

    def _execute_prefill_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states_torch: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Execute captured prefill trace with new inputs.

        Args:
            trace_id: Captured trace ID
            trace_tensors: Pre-allocated trace tensors
            trace_output: Trace output tensor
            hidden_states_torch: New hidden states (torch tensor)

        Returns:
            Output logits tensor
        """
        # Copy new hidden states to trace input location
        new_hidden = ttnn.from_torch(
            hidden_states_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        ttnn.copy(new_hidden, trace_tensors["hidden_states"])
        ttnn.deallocate(new_hidden)

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def _capture_decode_trace(self) -> Tuple[int, ttnn.Tensor, List[ttnn.Tensor]]:
        """
        Capture trace for decode phase (single token generation).

        Returns:
            Tuple of (trace_id, trace_output, trace_inputs)
        """
        logger.info("Compiling decode (first run)...")

        # Create dummy single token input
        dummy_token = torch.zeros((1, 1), dtype=torch.long)

        # Get embeddings
        input_ids_ttnn = ttnn.from_torch(
            dummy_token,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        hidden_states = self.model.text_model.embed_tokens(input_ids_ttnn)

        # First run to compile decode
        logits = self.model.text_model.forward_decode(
            hidden_states=hidden_states,
            kv_caches=self.kv_caches,
            current_pos=self.current_pos,
        )
        logger.info("Decode compiled")

        # Prepare inputs for trace
        hidden_states_trace = self.model.text_model.embed_tokens(input_ids_ttnn)

        # Capture trace
        logger.info("Capturing decode trace...")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        logits_trace = self.model.text_model.forward_decode(
            hidden_states=hidden_states_trace,
            kv_caches=self.kv_caches,
            current_pos=self.current_pos,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Decode trace captured")

        return trace_id, logits_trace, [hidden_states_trace]

    def run_prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        use_trace: bool = False,
    ) -> Tuple[ttnn.Tensor, float]:
        """
        Run prefill phase (process prompt + image).

        Args:
            input_ids: Input token IDs
            pixel_values: Preprocessed image tensor
            pooled_patches_idx: Indices for vision pooling
            use_trace: Whether to use tracing

        Returns:
            Tuple of (logits, prefill_time_ms)
        """
        # Initialize KV cache if needed
        self.init_kv_cache()

        seq_len = input_ids.shape[1]

        if use_trace:
            # Prepare inputs (vision processing) - this happens OUTSIDE trace
            logger.info("Preparing inputs (vision processing)...")
            vision_start = time.perf_counter()
            hidden_states_ttnn, hidden_states_torch = self._prepare_text_inputs(
                input_ids, pixel_values, pooled_patches_idx
            )
            vision_time = (time.perf_counter() - vision_start) * 1000
            logger.info(f"Vision processing completed in {vision_time:.2f}ms")

            # Text model forward with tracing
            start_time = time.perf_counter()

            # Check if we have a trace for this sequence length
            if seq_len not in self.prefill_traces:
                # Allocate trace tensors
                logger.info("Allocating trace tensors...")
                trace_tensors = self._allocate_prefill_trace_tensors(seq_len, hidden_dim=4096)

                # Copy initial hidden states to trace tensor
                ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])

                # Capture trace
                trace_id, trace_output = self._capture_prefill_trace(trace_tensors)
                self.prefill_traces[seq_len] = (trace_id, trace_tensors, trace_output)
                logits = trace_output
            else:
                trace_id, trace_tensors, trace_output = self.prefill_traces[seq_len]
                logits = self._execute_prefill_trace(trace_id, trace_tensors, trace_output, hidden_states_torch)

            # Clean up intermediate tensor
            ttnn.deallocate(hidden_states_ttnn)

            end_time = time.perf_counter()
            text_time = (end_time - start_time) * 1000
            prefill_time = vision_time + text_time

            logger.info(f"Text model prefill completed in {text_time:.2f}ms")
            logger.info(f"Total prefill completed in {prefill_time:.2f}ms")
        else:
            # Run without tracing
            start_time = time.perf_counter()

            logits, _ = self.model.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pooled_patches_idx=pooled_patches_idx,
            )

            end_time = time.perf_counter()
            prefill_time = (end_time - start_time) * 1000

            logger.info(f"Prefill completed in {prefill_time:.2f}ms")

        # Update position for decode
        self.reset_kv_cache(seq_len)

        return logits, prefill_time

    def run_decode_step(
        self,
        token_id: int,
        use_trace: bool = False,
    ) -> Tuple[ttnn.Tensor, float]:
        """
        Run single decode step.

        Args:
            token_id: Token ID to decode
            use_trace: Whether to use tracing

        Returns:
            Tuple of (logits, decode_time_ms)
        """
        start_time = time.perf_counter()

        # Create token tensor
        token_tensor = torch.tensor([[token_id]], dtype=torch.long)

        if use_trace:
            if self.decode_trace_id is None:
                trace_id, trace_output, trace_inputs = self._capture_decode_trace()
                self.decode_trace_id = trace_id
                self.decode_trace_inputs = trace_inputs
                self.decode_trace_output = trace_output

            # Update inputs and execute trace
            input_ids_ttnn = ttnn.from_torch(
                token_tensor,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            hidden_states = self.model.text_model.embed_tokens(input_ids_ttnn)
            ttnn.copy(hidden_states, self.decode_trace_inputs[0])

            ttnn.execute_trace(self.mesh_device, self.decode_trace_id, cq_id=0, blocking=False)
            logits = self.decode_trace_output
        else:
            # Run without tracing
            input_ids_ttnn = ttnn.from_torch(
                token_tensor,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            hidden_states = self.model.text_model.embed_tokens(input_ids_ttnn)

            logits = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
            )

        end_time = time.perf_counter()
        decode_time = (end_time - start_time) * 1000

        # Increment position
        pos_torch = ttnn.to_torch(self.current_pos)
        new_pos = pos_torch + 1
        new_pos_ttnn = ttnn.from_torch(
            new_pos,
            dtype=ttnn.int32,
            device=self.mesh_device,
            mesh_mapper=self.mesh_mapper,
        )
        ttnn.copy(new_pos_ttnn, self.current_pos)
        ttnn.deallocate(new_pos_ttnn)

        return logits, decode_time

    def run_inference(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 100,
        use_trace: bool = False,
    ) -> Tuple[str, dict]:
        """
        Run full inference (prefill-only for now, decode not yet fully optimized).

        Args:
            pixel_values: Preprocessed image tensor
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing

        Returns:
            Tuple of (output_text, perf_metrics)
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Create pooled_patches_idx
        batch_size = 1
        num_output_tokens = 64
        pool_kernel = 11
        pooled_patches_idx = torch.arange(num_output_tokens * pool_kernel).reshape(
            batch_size, num_output_tokens, pool_kernel
        )

        # Run prefill
        logits, prefill_time = self.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=use_trace,
        )

        # Get first prediction (prefill-only mode)
        logits_torch = ttnn.to_torch(logits).squeeze()
        if logits_torch.dim() == 2:
            next_token_logits = logits_torch[-1, :]
        else:
            next_token_logits = logits_torch[0, -1, :]

        top_token = torch.argmax(next_token_logits).item()
        top_word = self.tokenizer.decode([top_token])

        perf_metrics = {
            "prefill_time_ms": prefill_time,
            "avg_decode_time_ms": 0.0,
            "total_decode_time_ms": 0.0,
            "input_tokens": input_ids.shape[1],
            "generated_tokens": 1,
            "tokens_per_sec": 1000.0 / prefill_time if prefill_time > 0 else 0,
            "output_text": top_word,
        }

        logger.info(f"Input tokens: {input_ids.shape[1]}")
        logger.info(f"Top prediction: '{top_word}' (token {top_token})")

        return top_word, perf_metrics


def run_demo(
    image_path: Optional[str] = None,
    prompt: str = "What is in this image?",
    max_new_tokens: int = 100,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    use_trace: bool = False,
    iterations: int = 1,
):
    """
    Run the Molmo2 demo.

    Args:
        image_path: Path to input image (uses default if None)
        prompt: Text prompt for the model
        max_new_tokens: Maximum tokens to generate
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        use_trace: Whether to use tracing for performance
        iterations: Number of inference iterations (to measure trace performance)
    """
    if image_path is None:
        image_path = str(DEFAULT_IMAGE)

    logger.info("=" * 60)
    logger.info("Molmo2-8B Demo")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess image
    image, pixel_values = preprocess_image(image_path)

    # Load weights
    state_dict = load_model_weights()

    # Open device
    logger.info(f"Opening TTNN device {device_id}")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Create model
        model = create_model(device, state_dict, num_layers)
        text_num_layers = num_layers if num_layers is not None else 36

        # Create generator
        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=2048,
        )

        # Run inference
        logger.info("\n" + "=" * 60)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Running {iterations} iteration(s)")
        logger.info("=" * 60)

        all_prefill_times = []
        for i in range(iterations):
            logger.info(f"\n--- Iteration {i + 1}/{iterations} ---")
            response, perf_metrics = generator.run_inference(
                pixel_values=pixel_values,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_trace=use_trace,
            )
            all_prefill_times.append(perf_metrics["prefill_time_ms"])

        logger.info("\n" + "=" * 60)
        logger.info("Performance Metrics:")
        if iterations > 1:
            logger.info(f"  Iteration 1 (compile + trace capture): {all_prefill_times[0]:.2f}ms")
            if len(all_prefill_times) > 1:
                avg_traced = sum(all_prefill_times[1:]) / len(all_prefill_times[1:])
                logger.info(f"  Avg traced execution (iter 2-{iterations}): {avg_traced:.2f}ms")
                logger.info(f"  Speedup: {all_prefill_times[0] / avg_traced:.2f}x")
        else:
            logger.info(f"  Prefill time: {perf_metrics['prefill_time_ms']:.2f}ms")
        logger.info(f"  Avg decode time: {perf_metrics.get('avg_decode_time_ms', 0):.2f}ms")
        logger.info(f"  Total decode time: {perf_metrics.get('total_decode_time_ms', 0):.2f}ms")
        logger.info(f"  Input tokens: {perf_metrics['input_tokens']}")
        logger.info(f"  Generated tokens: {perf_metrics.get('generated_tokens', 0)}")
        logger.info(f"  Tokens/sec: {perf_metrics.get('tokens_per_sec', 0):.2f}")
        logger.info(f"  Output: '{perf_metrics.get('output_text', '')}'")
        logger.info("=" * 60)

        return perf_metrics

    finally:
        ttnn.close_device(device)
        logger.info("Device closed")


def main():
    parser = argparse.ArgumentParser(description="Molmo2-8B Demo")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (uses default dog.jpg if not specified)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="TTNN device ID",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of text layers (default: 36, use fewer for faster testing)",
    )
    parser.add_argument(
        "--use-trace",
        action="store_true",
        help="Enable tracing for improved performance",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of inference iterations (to measure trace performance)",
    )

    args = parser.parse_args()

    run_demo(
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        device_id=args.device,
        num_layers=args.num_layers,
        use_trace=args.use_trace,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
