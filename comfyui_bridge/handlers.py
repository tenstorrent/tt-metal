# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Operation handlers for ComfyUI Bridge.

Handles operations from ComfyUI frontend and executes them using SDXLRunner.
Provides shared memory tensor transfer via TensorBridge.

CRITICAL FIX (CRITICAL-2): Fixed shared memory race condition.
    - Bridge side no longer unlinks shared memory after reading
    - Client is responsible for unlinking after receiving response

SIGNIFICANT FIX (SIGNIFICANT-5): Implemented handle_full_denoise properly.
    - Now accepts text prompts and runs full inference
    - Returns images via shared memory
"""

import logging
import torch
import numpy as np
from PIL import Image
from multiprocessing import shared_memory
from typing import Dict, Any, Optional
from sdxl_runner import SDXLRunner
from sdxl_config import SDXLConfig

logger = logging.getLogger(__name__)


def _detect_and_convert_tt_to_standard_format(tensor: torch.Tensor, expected_channels: int = 4) -> torch.Tensor:
    """
    Detect if tensor is in TT format [B, 1, H*W, C] and convert to standard format [B, C, H, W].

    TT-Metal's ttnn.to_torch() returns tensors in TT format: [B, 1, H*W, C]
    Standard PyTorch format is: [B, C, H, W]

    This function:
    1. Detects the format by checking dimensions
    2. Converts TT format to standard format if needed
    3. Validates the final shape

    Args:
        tensor: Input tensor (either TT format or standard format)
        expected_channels: Expected number of channels (default: 4 for SDXL latents)

    Returns:
        Tensor in standard format [B, C, H, W]

    Raises:
        ValueError: If tensor format is invalid or conversion fails
    """
    logger.debug(f"Format detection - input shape: {tensor.shape}, expected_channels: {expected_channels}")

    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D: {tensor.shape}")

    # Check if already in standard format [B, C, H, W]
    B, dim1, dim2, dim3 = tensor.shape

    # Standard format: [B, C, H, W] where C = expected_channels
    if dim1 == expected_channels and dim2 > expected_channels and dim3 > expected_channels:
        logger.debug(f"Tensor already in standard format [B={B}, C={dim1}, H={dim2}, W={dim3}]")
        return tensor

    # TT format: [B, 1, H*W, C] where dim1=1 and dim3=expected_channels
    if dim1 == 1 and dim3 == expected_channels:
        logger.info(f"Detected TT format [B={B}, 1, H*W={dim2}, C={dim3}], converting to standard format")

        # Calculate H and W from H*W
        # For SDXL: H*W = 16384 = 128 * 128
        HW = dim2
        H = int(HW**0.5)
        W = HW // H

        if H * W != HW:
            raise ValueError(f"Cannot compute square dimensions from H*W={HW}")

        logger.debug(f"Computed dimensions: H={H}, W={W}")

        # Convert: [B, 1, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        tensor = tensor.reshape(B, H, W, dim3)  # [B, 1, H*W, C] -> [B, H, W, C]
        tensor = tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        logger.info(f"Converted to standard format: {tensor.shape}")
        return tensor

    # Unknown format
    raise ValueError(
        f"Unrecognized tensor format: {tensor.shape}. "
        f"Expected either standard [B, {expected_channels}, H, W] or TT [B, 1, H*W, {expected_channels}]"
    )


class TensorBridge:
    """
    Manages shared memory tensor transfer between ComfyUI and bridge server.

    Compatible with ComfyUI's TensorBridge implementation in tenstorrent_backend.py.

    CRITICAL-2 Fix Protocol:
    - When reading tensors (client-created): Just close, don't unlink
    - When writing tensors (server-created): Client will unlink after reading
    """

    def __init__(self):
        self._active_segments = {}  # shm_name -> SharedMemory object

    def tensor_from_shm(self, handle: Dict[str, Any]) -> torch.Tensor:
        """
        Reconstruct a PyTorch tensor from shared memory.

        CRITICAL-2 Fix: Does NOT unlink - client is responsible for cleanup.

        Args:
            handle: Dictionary with shm_name, shape, dtype, size_bytes

        Returns:
            PyTorch tensor (copied from shared memory)
        """
        shm_name = handle["shm_name"]
        shape = tuple(handle["shape"])
        dtype_str = handle["dtype"]

        logger.debug(f"Reading tensor from shm: {shm_name}, shape={shape}, dtype={dtype_str}")

        try:
            # Attach to existing shared memory
            shm = shared_memory.SharedMemory(name=shm_name)

            # Parse dtype
            np_dtype = _parse_dtype(dtype_str)

            # Create numpy array view
            np_array = np.ndarray(shape=shape, dtype=np_dtype, buffer=shm.buf)

            # Copy to new tensor (to avoid shared memory lifetime issues)
            tensor = torch.from_numpy(np_array.copy())

            # CRITICAL-2 FIX: Just close, don't unlink
            # Client is responsible for unlinking after receiving our response
            shm.close()
            # REMOVED: shm.unlink()

            logger.debug(f"Successfully read tensor: {tensor.shape}")
            return tensor

        except Exception as e:
            logger.error(f"Failed to read from shared memory {shm_name}: {e}")
            raise RuntimeError(f"Shared memory read failed: {e}")

    def tensor_to_shm(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Transfer a PyTorch tensor to shared memory.

        Args:
            tensor: PyTorch tensor to share

        Returns:
            Dictionary with metadata for reconstructing the tensor
        """
        # Ensure tensor is contiguous and on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.contiguous()

        # Get tensor metadata
        shape = tensor.shape
        dtype_str = str(tensor.dtype)

        # Convert to numpy for shared memory
        np_array = tensor.numpy()
        size_bytes = np_array.nbytes

        # Create unique name for this shared memory segment
        import uuid

        shm_name = f"tt_bridge_{uuid.uuid4().hex[:16]}"

        logger.debug(f"Creating shm: {shm_name}, shape={shape}, dtype={dtype_str}, size={size_bytes}")

        try:
            # Create shared memory
            shm = shared_memory.SharedMemory(create=True, size=size_bytes, name=shm_name)

            # Copy data to shared memory
            shm_array = np.ndarray(shape=np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
            shm_array[:] = np_array[:]

            # Store reference (client will unlink after reading)
            self._active_segments[shm_name] = shm

            # Return handle
            return {"shm_name": shm_name, "shape": list(shape), "dtype": dtype_str, "size_bytes": size_bytes}

        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise RuntimeError(f"Shared memory creation failed: {e}")

    def cleanup_segment(self, shm_name: str):
        """Clean up a specific shared memory segment."""
        if shm_name in self._active_segments:
            try:
                shm = self._active_segments[shm_name]
                shm.close()
                # Don't unlink - client will unlink after reading
            except Exception as e:
                logger.warning(f"Failed to clean up shared memory {shm_name}: {e}")
            finally:
                del self._active_segments[shm_name]

    def cleanup_all(self):
        """Clean up all active shared memory segments."""
        for shm_name in list(self._active_segments.keys()):
            self.cleanup_segment(shm_name)


class OperationHandler:
    """
    Handles operations from ComfyUI frontend.

    Maps ComfyUI operations to SDXLRunner API calls.
    """

    def __init__(self, config: SDXLConfig):
        """
        Initialize operation handler.

        Args:
            config: SDXL configuration
        """
        self.config = config
        self.sdxl_runner: Optional[SDXLRunner] = None
        self.tensor_bridge = TensorBridge()
        self.model_id: Optional[str] = None
        logger.info("OperationHandler initialized")

    def handle_init_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize SDXL model.

        Input data:
            - model_type: str (e.g., "sdxl")
            - config: dict (optional config overrides)
            - device_id: str (device ID to use)

        Returns:
            - model_id: str
            - status: str
        """
        model_type = data.get("model_type", "sdxl")
        logger.info(f"Initializing model: {model_type}")

        if model_type != "sdxl":
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            # Create SDXLRunner with worker_id=0 (single bridge server)
            self.sdxl_runner = SDXLRunner(worker_id=0, config=self.config)

            # Initialize device
            logger.info("Initializing device...")
            self.sdxl_runner.initialize_device()

            # Load model (includes warmup)
            logger.info("Loading model and warming up...")
            self.sdxl_runner.load_model()

            # Generate model ID
            import uuid

            self.model_id = f"sdxl_{uuid.uuid4().hex[:8]}"

            logger.info(f"Model initialized successfully: {self.model_id}")

            return {"model_id": self.model_id, "status": "ready"}

        except Exception as e:
            logger.error(f"Model initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Model initialization failed: {e}")

    def handle_full_denoise(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete denoising loop with text prompts.

        SIGNIFICANT-5 FIX: Properly implements prompt-based inference.

        Input data:
            - model_id: str
            - prompt: str - Positive prompt text
            - negative_prompt: str - Negative prompt text
            - prompt_2: str (optional) - Second prompt for SDXL
            - negative_prompt_2: str (optional) - Second negative prompt for SDXL
            - num_inference_steps: int (default: 20)
            - guidance_scale: float (default: 5.0)
            - guidance_rescale: float (optional, default: 0.0)
            - width: int (default: 1024)
            - height: int (default: 1024)
            - seed: int (optional)

        Returns:
            - images_shm: shm_handle for generated images
            - num_images: int
        """
        model_id = data.get("model_id")
        if model_id != self.model_id:
            raise ValueError(f"Invalid model_id: {model_id}")

        if self.sdxl_runner is None:
            raise RuntimeError("Model not initialized. Call init_model first.")

        logger.info("Running full denoise operation...")

        try:
            # Extract parameters
            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")
            prompt_2 = data.get("prompt_2", prompt)  # Default to prompt if not specified
            negative_prompt_2 = data.get("negative_prompt_2", negative_prompt)

            num_steps = data.get("num_inference_steps", 20)
            guidance_scale = data.get("guidance_scale", 5.0)
            guidance_rescale = data.get("guidance_rescale", 0.0)
            width = data.get("width", 1024)
            height = data.get("height", 1024)
            seed = data.get("seed")

            logger.info(
                f"Inference params: prompt='{prompt[:50]}...', steps={num_steps}, "
                f"guidance={guidance_scale}, size={width}x{height}, seed={seed}"
            )

            # Build inference request
            request = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "prompt_2": prompt_2,
                "negative_prompt_2": negative_prompt_2,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "guidance_rescale": guidance_rescale,
                "width": width,
                "height": height,
                "seed": seed,
            }

            # Run inference via SDXLRunner
            logger.info("Calling SDXLRunner.run_inference...")
            images = self.sdxl_runner.run_inference([request])

            if not images:
                raise RuntimeError("No images generated")

            logger.info(f"Generated {len(images)} image(s)")

            # Convert PIL image to tensor and transfer via shared memory
            # ComfyUI expects [B, H, W, C] format with values in [0, 1]
            image = images[0]  # First image

            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            # Ensure we have RGB (not RGBA)
            if image_np.ndim == 3 and image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]

            # Convert to float32 in range [0, 1]
            image_tensor = torch.from_numpy(image_np).float() / 255.0

            # Shape: [H, W, C] -> add batch dimension -> [1, H, W, C]
            image_tensor = image_tensor.unsqueeze(0)

            logger.info(
                f"Image tensor shape: {image_tensor.shape}, "
                f"range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]"
            )

            # Transfer via shared memory
            images_shm = self.tensor_bridge.tensor_to_shm(image_tensor)

            return {"images_shm": images_shm, "num_images": 1}

        except Exception as e:
            logger.error(f"Full denoise failed: {e}", exc_info=True)
            raise RuntimeError(f"Full denoise failed: {e}")

    def handle_denoise_only(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run denoising loop only, returning latents without VAE decode.

        Supports both txt2img (random latents) and img2img (input latents).
        Can accept either text prompts OR pre-encoded embeddings.

        Input data:
            - model_id: str
            - prompt: str (optional) - Positive prompt text
            - negative_prompt: str (optional) - Negative prompt text
            - prompt_2: str (optional) - Second prompt for SDXL
            - negative_prompt_2: str (optional) - Second negative prompt
            - prompt_embeds_shm: dict (optional) - Pre-encoded prompt embeddings handle
            - pooled_prompt_embeds_shm: dict (optional) - Pre-encoded pooled embeddings
            - input_latents_shm: dict (optional) - Input latents for img2img
            - denoise_strength: float (optional) - For img2img, default 1.0
            - num_inference_steps: int (default: 20)
            - guidance_scale: float (default: 5.0)
            - guidance_rescale: float (default: 0.0)
            - width: int (default: 1024)
            - height: int (default: 1024)
            - seed: int (optional)

        Returns:
            - latents: shm_handle for latent tensor [B, 4, H//8, W//8]
            - latent_metadata: dict with shape info
            - timing: dict with timing breakdown
        """
        import time

        timing_start = time.perf_counter()

        # Validate request
        model_id = data.get("model_id")
        if model_id != self.model_id:
            raise ValueError(f"Invalid model_id: {model_id}")

        if self.sdxl_runner is None:
            raise RuntimeError("Model not initialized. Call init_model first.")

        # Check that we have either text prompts OR embeddings
        has_text_prompts = data.get("prompt") is not None
        has_embeddings = data.get("prompt_embeds_shm") is not None

        if not has_text_prompts and not has_embeddings:
            raise ValueError("Must provide either text prompts (prompt) or pre-encoded embeddings (prompt_embeds_shm)")

        logger.info("Running denoise_only operation...")

        try:
            # Extract parameters
            num_steps = data.get("num_inference_steps", 20)
            guidance_scale = data.get("guidance_scale", 5.0)
            guidance_rescale = data.get("guidance_rescale", 0.0)
            width = data.get("width", 1024)
            height = data.get("height", 1024)
            seed = data.get("seed")
            denoise_strength = data.get("denoise_strength", 1.0)
            has_input_latents = data.get("input_latents_shm") is not None

            logger.info(
                f"Denoise params: steps={num_steps}, guidance={guidance_scale}, "
                f"guidance_rescale={guidance_rescale}, size={width}x{height}, "
                f"seed={seed}, denoise_strength={denoise_strength}, "
                f"has_input_latents={has_input_latents}"
            )

            # Update pipeline parameters
            if num_steps != self.sdxl_runner.tt_sdxl.pipeline_config.num_inference_steps:
                self.sdxl_runner.tt_sdxl.set_num_inference_steps(num_steps)

            if guidance_scale != self.sdxl_runner.tt_sdxl.pipeline_config.guidance_scale:
                self.sdxl_runner.tt_sdxl.set_guidance_scale(guidance_scale)

            if guidance_rescale != self.sdxl_runner.tt_sdxl.pipeline_config.guidance_rescale:
                self.sdxl_runner.tt_sdxl.set_guidance_rescale(guidance_rescale)

            # Step 1: Handle CLIP encoding
            encode_start = time.perf_counter()

            if has_text_prompts:
                # Encode text prompts
                prompt = data.get("prompt", "")
                negative_prompt = data.get("negative_prompt", "")
                prompt_2 = data.get("prompt_2", prompt)
                negative_prompt_2 = data.get("negative_prompt_2", negative_prompt)

                logger.info(f"Encoding prompts: '{prompt[:50]}...'")
                prompt_embeds, pooled_prompt_embeds = self.sdxl_runner.tt_sdxl.encode_prompts(
                    [prompt], [negative_prompt], prompt_2=[prompt_2], negative_prompt_2=[negative_prompt_2]
                )
            else:
                # Read pre-encoded embeddings from shared memory
                logger.info("Reading pre-encoded embeddings from shared memory")
                prompt_embeds_handle = data.get("prompt_embeds_shm")
                pooled_embeds_handle = data.get("pooled_prompt_embeds_shm")

                if prompt_embeds_handle is None or pooled_embeds_handle is None:
                    raise ValueError(
                        "Both prompt_embeds_shm and pooled_prompt_embeds_shm required when using pre-encoded embeddings"
                    )

                prompt_embeds = self.tensor_bridge.tensor_from_shm(prompt_embeds_handle)
                pooled_prompt_embeds = self.tensor_bridge.tensor_from_shm(pooled_embeds_handle)

            encode_time = (time.perf_counter() - encode_start) * 1000

            # Step 2: Prepare input latents
            latents_start = time.perf_counter()

            if has_input_latents:
                # Img2img mode: read latents from shared memory
                logger.info("Reading input latents from shared memory (img2img mode)")
                input_latents_handle = data.get("input_latents_shm")
                input_latents = self.tensor_bridge.tensor_from_shm(input_latents_handle)

                # Validate latent shape
                expected_shape = [1, 4, height // 8, width // 8]
                if list(input_latents.shape) != expected_shape:
                    raise ValueError(
                        f"Input latents shape {list(input_latents.shape)} doesn't match expected {expected_shape}"
                    )

                # For img2img, adjust num_steps based on denoise_strength
                # This is a simplified approach - proper img2img would need scheduler state management
                effective_steps = max(1, int(num_steps * denoise_strength))
                if effective_steps != num_steps:
                    logger.info(
                        f"Adjusted steps for img2img: {num_steps} -> {effective_steps} (strength={denoise_strength})"
                    )
                    self.sdxl_runner.tt_sdxl.set_num_inference_steps(effective_steps)
            else:
                # Txt2img mode: will generate random latents in generate_input_tensors
                input_latents = None
                logger.info("Txt2img mode: will generate random latents")

            latents_prep_time = (time.perf_counter() - latents_start) * 1000

            # Step 3: Generate input tensors (includes latent generation for txt2img)
            tensors_start = time.perf_counter()

            # Reset scheduler state for new inference run
            self.sdxl_runner.tt_sdxl.tt_scheduler.set_begin_index(0)

            # Generate tensors
            # For img2img, pass start_latents instead of start_latent_seed
            if has_input_latents:
                tt_latents, tt_prompts, tt_texts = self.sdxl_runner.tt_sdxl.generate_input_tensors(
                    prompt_embeds, pooled_prompt_embeds, start_latents=input_latents
                )
            else:
                tt_latents, tt_prompts, tt_texts = self.sdxl_runner.tt_sdxl.generate_input_tensors(
                    prompt_embeds, pooled_prompt_embeds, start_latent_seed=seed
                )

            tensors_time = (time.perf_counter() - tensors_start) * 1000

            # Step 4: Prepare input tensors on device
            prepare_start = time.perf_counter()
            self.sdxl_runner.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompts, tt_texts])
            prepare_time = (time.perf_counter() - prepare_start) * 1000

            # Step 5: Run denoising loop (return latents, skip VAE decode)
            denoise_start = time.perf_counter()
            logger.info("Running denoising loop...")
            tt_latents_output = self.sdxl_runner.tt_sdxl.generate_images(return_latents=True)
            denoise_time = (time.perf_counter() - denoise_start) * 1000

            # Step 6: Convert latents to torch format and transfer to shared memory
            convert_start = time.perf_counter()

            # Convert ttnn tensor to torch
            import ttnn

            latents_torch = ttnn.to_torch(
                tt_latents_output, mesh_composer=ttnn.ConcatMeshToTensor(self.sdxl_runner.ttnn_device, dim=0)
            )[: self.sdxl_runner.tt_sdxl.batch_size, ...]

            logger.info(f"Raw latents from ttnn.to_torch(): shape={latents_torch.shape}, dtype={latents_torch.dtype}")

            # Convert BFloat16 to Float32 for shared memory transfer
            # (TT-Metal uses BFloat16 internally, ComfyUI expects Float32)
            if latents_torch.dtype == torch.bfloat16:
                latents_torch = latents_torch.float()

            # Convert from TT format [B, 1, H*W, C] to standard format [B, C, H, W]
            latents_torch = _detect_and_convert_tt_to_standard_format(latents_torch, expected_channels=4)

            # Validate final standard format
            if latents_torch.dim() != 4:
                raise ValueError(
                    f"Expected 4D tensor after conversion, got {latents_torch.dim()}D: {latents_torch.shape}"
                )

            B, C, H, W = latents_torch.shape
            if C != 4:
                raise ValueError(f"Expected 4 channels for SDXL after conversion, got {C}")

            logger.info(
                f"Latents converted to standard format: shape={latents_torch.shape}, dtype={latents_torch.dtype}"
            )

            # Transfer to shared memory
            latents_handle = self.tensor_bridge.tensor_to_shm(latents_torch)
            convert_time = (time.perf_counter() - convert_start) * 1000

            total_time = (time.perf_counter() - timing_start) * 1000

            logger.info(f"Denoise completed successfully in {total_time:.1f}ms")

            return {
                "latents_shm": latents_handle,
                "latent_metadata": {
                    "batch_size": B,
                    "channels": C,
                    "height": H,
                    "width": W,
                    "scaling_factor": 0.13025,
                    "model_type": "sdxl",
                },
                "timing": {
                    "encode_ms": encode_time,
                    "latents_prep_ms": latents_prep_time,
                    "tensors_gen_ms": tensors_time,
                    "tensors_prep_ms": prepare_time,
                    "denoise_ms": denoise_time,
                    "convert_ms": convert_time,
                    "total_ms": total_time,
                },
            }

        except Exception as e:
            logger.error(f"Denoise only operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Denoise only operation failed: {e}")

    def handle_ping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Health check / ping operation.

        Returns:
            - status: str
            - model_loaded: bool
            - model_id: str (if loaded)
        """
        logger.debug("Handling ping request")

        return {
            "status": "ok",
            "model_loaded": self.sdxl_runner is not None,
            "model_id": self.model_id if self.model_id else None,
        }

    def handle_vae_decode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode latents to images using VAE.

        Input data:
            - model_id: str
            - latents_shm: dict - Shared memory handle for latents [B, 4, H, W]

        Returns:
            - images: shm_handle for images [B, H, W, 3]
            - image_metadata: dict with shape info
            - timing: dict with timing breakdown
        """
        import time
        import ttnn

        timing_start = time.perf_counter()

        # Validate request
        model_id = data.get("model_id")
        if model_id != self.model_id:
            raise ValueError(f"Invalid model_id: {model_id}")

        if self.sdxl_runner is None:
            raise RuntimeError("Model not initialized. Call init_model first.")

        latents_handle = data.get("latents_shm")
        if latents_handle is None:
            raise ValueError("latents_shm is required")

        logger.info("Running VAE decode operation...")

        try:
            # Step 1: Read latents from shared memory
            read_start = time.perf_counter()
            latents_torch = self.tensor_bridge.tensor_from_shm(latents_handle)
            read_time = (time.perf_counter() - read_start) * 1000

            logger.info(f"Latents from shared memory - shape: {latents_torch.shape}, dtype: {latents_torch.dtype}")

            # Step 1.5: Reshape from standard format [B, C, H, W] to TT pipeline format [B, 1, H*W, C]
            if latents_torch.dim() != 4:
                raise ValueError(f"Latents must be 4D tensor [B, C, H, W], got shape {latents_torch.shape}")

            B, C, H, W = latents_torch.shape

            # Validate standard format before reshape
            if C != 4:
                raise ValueError(f"Latents must have 4 channels for SDXL, got {C}")

            logger.info(
                f"Reshaping latents from standard format [B={B}, C={C}, H={H}, W={W}] to TT format [B, 1, H*W, C]"
            )

            # Reshape: [B, C, H, W] -> [B, H, W, C] -> [B, 1, H*W, C]
            latents_torch = latents_torch.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            latents_torch = latents_torch.reshape(B, 1, H * W, C)  # [B, H, W, C] -> [B, 1, H*W, C]

            # Validate TT format after reshape
            if latents_torch.shape[3] != 4:
                raise ValueError(
                    f"Latents must have 4 channels (at position 3) for SDXL, got shape {latents_torch.shape}"
                )

            logger.info(f"Latents reshaped to TT format: {latents_torch.shape}, dtype: {latents_torch.dtype}")

            # Step 2: Run VAE decode
            decode_start = time.perf_counter()

            vae_on_device = self.sdxl_runner.config.vae_on_device

            if vae_on_device:
                # Use TT VAE on device
                logger.info("Using TT VAE on device")

                # Convert to ttnn tensor FIRST (before scaling)
                tt_latents = ttnn.from_torch(
                    latents_torch,
                    dtype=ttnn.bfloat16,
                    device=self.sdxl_runner.ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.sdxl_runner.ttnn_device),
                )

                # Apply scaling factor ON DEVICE (same as run_tt_image_gen)
                scaling_factor = 0.13025
                tt_latents = ttnn.div(tt_latents, scaling_factor)

                # Decode using TT VAE
                output_tensor, [C_out, H_out, W_out] = self.sdxl_runner.tt_sdxl.tt_vae.decode(tt_latents, [B, 4, H, W])

                # Convert back to torch
                imgs = ttnn.to_torch(
                    output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.sdxl_runner.ttnn_device, dim=0)
                ).float()[:B, ...]

                ttnn.synchronize_device(self.sdxl_runner.ttnn_device)

                # Reshape to [B, C, H, W]
                imgs = imgs.reshape(B, C_out, H_out, W_out)

                # Deallocate device tensors
                ttnn.deallocate(tt_latents)
                ttnn.deallocate(output_tensor)

            else:
                # Use CPU VAE
                logger.info("Using CPU VAE")

                # Convert TT format [B, 1, H*W, C] back to standard format [B, C, H, W] for torch VAE
                logger.info(f"Converting latents from TT format {latents_torch.shape} to standard format for CPU VAE")
                latents_standard = latents_torch.reshape(B, H, W, C)  # [B, 1, H*W, C] -> [B, H, W, C]
                latents_standard = latents_standard.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                logger.info(f"Latents converted to standard format: {latents_standard.shape}")

                # Apply scaling factor for torch VAE
                scaling_factor = 0.13025
                latents_scaled = latents_standard / scaling_factor

                # Ensure on CPU
                if latents_scaled.is_cuda:
                    latents_scaled = latents_scaled.cpu()

                # Decode using torch VAE
                with torch.no_grad():
                    imgs = self.sdxl_runner.pipeline.vae.decode(latents_scaled, return_dict=False)[0]

            decode_time = (time.perf_counter() - decode_start) * 1000

            # Step 4: Convert to ComfyUI format
            convert_start = time.perf_counter()

            # VAE output is [B, C, H, W] in range [-1, 1]
            # Need to convert to [B, H, W, C] in range [0, 1]

            # Rescale from [-1, 1] to [0, 1]
            imgs = (imgs / 2 + 0.5).clamp(0, 1)

            # Permute to [B, H, W, C]
            imgs = imgs.permute(0, 2, 3, 1)

            # Ensure contiguous
            imgs = imgs.contiguous()

            B_out, H_out, W_out, C_out = imgs.shape

            logger.info(
                f"Images shape: {imgs.shape}, dtype: {imgs.dtype}, " f"range: [{imgs.min():.3f}, {imgs.max():.3f}]"
            )

            # Transfer to shared memory
            images_handle = self.tensor_bridge.tensor_to_shm(imgs)
            convert_time = (time.perf_counter() - convert_start) * 1000

            total_time = (time.perf_counter() - timing_start) * 1000

            logger.info(f"VAE decode completed successfully in {total_time:.1f}ms")

            return {
                "images_shm": images_handle,
                "image_metadata": {"batch_size": B_out, "height": H_out, "width": W_out, "channels": C_out},
                "timing": {
                    "read_ms": read_time,
                    "decode_ms": decode_time,
                    "convert_ms": convert_time,
                    "total_ms": total_time,
                },
            }

        except Exception as e:
            logger.error(f"VAE decode failed: {e}", exc_info=True)
            raise RuntimeError(f"VAE decode failed: {e}")

    def handle_vae_encode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode images to latents using VAE.

        Input data:
            - model_id: str
            - images_shm: dict - Shared memory handle for images [B, H, W, 3] in [0, 1]

        Returns:
            - latents: shm_handle for latents [B, 4, H//8, W//8]
            - latent_metadata: dict with shape info
            - timing: dict with timing breakdown
        """
        import time
        import ttnn

        timing_start = time.perf_counter()

        # Validate request
        model_id = data.get("model_id")
        if model_id != self.model_id:
            raise ValueError(f"Invalid model_id: {model_id}")

        if self.sdxl_runner is None:
            raise RuntimeError("Model not initialized. Call init_model first.")

        images_handle = data.get("images_shm")
        if images_handle is None:
            raise ValueError("images_shm is required")

        logger.info("Running VAE encode operation...")

        try:
            # Step 1: Read images from shared memory
            read_start = time.perf_counter()
            images_torch = self.tensor_bridge.tensor_from_shm(images_handle)
            read_time = (time.perf_counter() - read_start) * 1000

            # Validate image shape
            if images_torch.dim() != 4:
                raise ValueError(f"Images must be 4D tensor [B, H, W, C], got shape {images_torch.shape}")

            B, H, W, C = images_torch.shape
            if C != 3:
                raise ValueError(f"Images must have 3 channels (RGB), got {C}")

            logger.info(
                f"Images shape: {images_torch.shape}, dtype: {images_torch.dtype}, "
                f"range: [{images_torch.min():.3f}, {images_torch.max():.3f}]"
            )

            # Step 2: Convert to VAE input format
            convert_start = time.perf_counter()

            # Permute from [B, H, W, C] to [B, C, H, W]
            images_vae = images_torch.permute(0, 3, 1, 2)

            # Rescale from [0, 1] to [-1, 1]
            images_vae = 2.0 * images_vae - 1.0

            # Ensure contiguous
            images_vae = images_vae.contiguous()

            convert_time = (time.perf_counter() - convert_start) * 1000

            # Step 3: Run VAE encode
            encode_start = time.perf_counter()

            vae_on_device = self.sdxl_runner.config.vae_on_device

            if vae_on_device:
                # Use TT VAE on device
                logger.info("Using TT VAE on device")

                # Convert to ttnn tensor
                tt_images = ttnn.from_torch(
                    images_vae,
                    dtype=ttnn.bfloat16,
                    device=self.sdxl_runner.ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.sdxl_runner.ttnn_device),
                )

                # Encode using TT VAE
                tt_latents, [C_out, H_out, W_out] = self.sdxl_runner.tt_sdxl.tt_vae.encode(tt_images, [B, 3, H, W])

                # Convert back to torch
                latents_torch = ttnn.to_torch(
                    tt_latents, mesh_composer=ttnn.ConcatMeshToTensor(self.sdxl_runner.ttnn_device, dim=0)
                ).float()[:B, ...]

                ttnn.synchronize_device(self.sdxl_runner.ttnn_device)

                # Reshape to [B, C, H, W]
                latents_torch = latents_torch.reshape(B, C_out, H_out, W_out)

                # Deallocate device tensors
                ttnn.deallocate(tt_images)
                ttnn.deallocate(tt_latents)

            else:
                # Use CPU VAE
                logger.info("Using CPU VAE")

                # Ensure on CPU
                if images_vae.is_cuda:
                    images_vae = images_vae.cpu()

                # Encode using torch VAE
                with torch.no_grad():
                    latents_dist = self.sdxl_runner.pipeline.vae.encode(images_vae)
                    latents_torch = latents_dist.latent_dist.sample()

            # Step 4: Apply scaling factor
            scaling_factor = 0.13025
            latents_torch = latents_torch * scaling_factor

            encode_time = (time.perf_counter() - encode_start) * 1000

            # Step 5: Transfer to shared memory
            transfer_start = time.perf_counter()

            B_out, C_out, H_out, W_out = latents_torch.shape

            logger.info(f"Latents shape: {latents_torch.shape}, dtype: {latents_torch.dtype}")

            latents_handle = self.tensor_bridge.tensor_to_shm(latents_torch)
            transfer_time = (time.perf_counter() - transfer_start) * 1000

            total_time = (time.perf_counter() - timing_start) * 1000

            logger.info(f"VAE encode completed successfully in {total_time:.1f}ms")

            return {
                "latents_shm": latents_handle,
                "latent_metadata": {
                    "batch_size": B_out,
                    "channels": C_out,
                    "height": H_out,
                    "width": W_out,
                    "scaling_factor": scaling_factor,
                    "model_type": "sdxl",
                },
                "timing": {
                    "read_ms": read_time,
                    "convert_ms": convert_time,
                    "encode_ms": encode_time,
                    "transfer_ms": transfer_time,
                    "total_ms": total_time,
                },
            }

        except Exception as e:
            logger.error(f"VAE encode failed: {e}", exc_info=True)
            raise RuntimeError(f"VAE encode failed: {e}")

    def handle_unload_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unload model and release resources.

        Input data:
            - model_id: str

        Returns:
            - status: str
        """
        model_id = data.get("model_id")
        logger.info(f"Unloading model: {model_id}")

        if model_id != self.model_id:
            raise ValueError(f"Invalid model_id: {model_id}")

        if self.sdxl_runner is not None:
            try:
                # Close device and cleanup
                self.sdxl_runner.close_device()
                self.sdxl_runner = None
                self.model_id = None

                # Cleanup shared memory
                self.tensor_bridge.cleanup_all()

                logger.info("Model unloaded successfully")

                return {"status": "unloaded"}

            except Exception as e:
                logger.error(f"Model unload failed: {e}", exc_info=True)
                raise RuntimeError(f"Model unload failed: {e}")
        else:
            logger.warning("No model loaded, ignoring unload request")
            return {"status": "no_model_loaded"}

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up handler resources...")

        if self.sdxl_runner is not None:
            try:
                self.sdxl_runner.close_device()
            except Exception as e:
                logger.error(f"Error closing device: {e}")

        self.tensor_bridge.cleanup_all()


def _parse_dtype(dtype_str: str) -> np.dtype:
    """
    Parse PyTorch dtype string to numpy dtype.

    Args:
        dtype_str: String like "torch.float32", "torch.float16", etc.

    Returns:
        Numpy dtype
    """
    if "float32" in dtype_str:
        return np.float32
    elif "float16" in dtype_str:
        return np.float16
    elif "float64" in dtype_str:
        return np.float64
    elif "int64" in dtype_str:
        return np.int64
    elif "int32" in dtype_str:
        return np.int32
    elif "int16" in dtype_str:
        return np.int16
    elif "int8" in dtype_str:
        return np.int8
    elif "uint8" in dtype_str:
        return np.uint8
    else:
        logger.warning(f"Unknown dtype {dtype_str}, defaulting to float32")
        return np.float32
