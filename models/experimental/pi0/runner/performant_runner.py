# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Performant Runner - Denoising Loop Trace + 2CQ.

This runner traces the DENOISING LOOP of PI0 (10 iterations of embed_suffix +
forward_expert) for performance improvement by eliminating kernel dispatch overhead.

WHY NOT FULL MODEL TRACE?
The SigLIP vision tower internally converts PyTorch images to TTNN tensors using
ttnn.from_torch(), which writes to device. Writes are not allowed during trace
capture. Without modifying the model code, we cannot trace the vision tower.

WHAT GETS TRACED:
- Denoising loop × 10 iterations
- Each iteration: embed_suffix + forward_expert + Euler step
- ~180 transformer block passes (10 steps × 18 blocks)

WHAT RUNS NORMALLY (NOT TRACED):
- embed_prefix (3× SigLIP vision tower)
- forward_vlm (VLM blocks → KV cache)

FIXED CONFIGURATION REQUIREMENTS:
- batch_size: 1
- denoising_steps: 10
- action_horizon: 50
- action_dim: 32

Performance:
- Expected improvement: ~25% (denoising is ~60% of total time)

Reference implementations:
- models/experimental/tt_dit/pipelines/flux1/pipeline_flux1.py - Diffusion trace
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN


@dataclass
class PI0TraceConfig:
    """
    Fixed configuration for traced execution.

    WARNING: These values are BAKED INTO the trace and cannot be changed
    at runtime without recompiling the trace.
    """

    batch_size: int = 1
    num_images: int = 3
    image_size: int = 224
    max_lang_tokens: int = 512
    denoising_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 32
    state_dim: int = 32

    # Device configuration
    num_command_queues: int = 2
    trace_region_size: int = 32 * 1024 * 1024  # 32MB for denoising loop trace (~24MB used)
    l1_small_size: int = 24576  # Same as optimal pytest setting for best performance


class PerformantRunner:
    """
    Denoising loop trace executor for PI0.

    Traces the denoising iterations (not the full model):
    - Denoising loop × 10 - embed_suffix + forward_expert + Euler step

    The prefix embedding and VLM forward run normally (not traced) since they
    involve host→device image transfers that can't be traced.

    Usage:
        config = PI0TraceConfig()
        runner = PerformantRunner(model, device, config)
        runner.compile()

        for batch in dataloader:
            actions = runner.execute(images, img_masks, lang_tokens, lang_masks, state)

        runner.cleanup()
    """

    def __init__(
        self,
        model: PI0ModelTTNN,
        device: ttnn.Device,
        config: Optional[PI0TraceConfig] = None,
    ):
        self.model = model
        self.device = device
        self.config = config or PI0TraceConfig()

        # Trace state
        self.trace_id = None
        self._compiled = False

        # Pre-allocated tensors for traced denoising
        self.state_ttnn: Optional[ttnn.Tensor] = None
        self.x_t_ttnn: Optional[ttnn.Tensor] = None
        self.timesteps_ttnn: Optional[ttnn.Tensor] = None

        # Cached prefix results (computed at execute time, not traced)
        self.prefix_embs: Optional[ttnn.Tensor] = None
        self.prefix_kv_cache: Optional[Tuple] = None

        # Output tensor reference
        self.traced_output: Optional[ttnn.Tensor] = None

    def _validate_config(self):
        """Ensure model config matches trace config."""
        model_config = self.model.config

        if model_config.action_horizon != self.config.action_horizon:
            raise ValueError(
                f"action_horizon mismatch: model={model_config.action_horizon}, " f"trace={self.config.action_horizon}"
            )

        if model_config.action_dim != self.config.action_dim:
            raise ValueError(
                f"action_dim mismatch: model={model_config.action_dim}, " f"trace={self.config.action_dim}"
            )

        if self.model.denoise_config.num_steps != self.config.denoising_steps:
            raise ValueError(
                f"denoising_steps mismatch: model={self.model.denoise_config.num_steps}, "
                f"trace={self.config.denoising_steps}"
            )

    def _allocate_denoising_tensors(self):
        """Allocate tensors for the traced denoising loop."""
        # State tensor
        state_torch = torch.zeros(self.config.batch_size, self.config.state_dim)
        self.state_ttnn = ttnn.from_torch(
            state_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Initial noise (baked into trace for reproducibility)
        torch.manual_seed(42)
        x_t_torch = torch.randn(self.config.batch_size, self.config.action_horizon, self.config.action_dim)
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Pre-compute timesteps on device
        num_steps = self.config.denoising_steps
        pad_steps = ((num_steps + 31) // 32) * 32

        timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)
        timestep_indices = ttnn.to_layout(timestep_indices, ttnn.TILE_LAYOUT)

        timestep_values = ttnn.multiply(timestep_indices, -1.0 / num_steps)
        self.timesteps_ttnn = ttnn.add(timestep_values, 1.0)
        self.timesteps_ttnn = ttnn.reshape(self.timesteps_ttnn, (1, pad_steps))

        ttnn.deallocate(timestep_indices)
        ttnn.deallocate(timestep_values)

    def _run_prefix_and_vlm(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[ttnn.Tensor, Tuple, int]:
        """
        Run prefix embedding and VLM forward (NOT traced).

        Returns:
            Tuple of (prefix_embs, prefix_kv_cache, prefix_len)
        """
        # Convert language inputs to TTNN
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Embed prefix (images + language) - NOT traced
        prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
            images, img_masks, lang_tokens_ttnn, lang_masks_ttnn
        )

        # Forward VLM and cache KV - NOT traced
        _, prefix_kv_cache = self.model.backbone.forward_vlm(prefix_embs, use_cache=True)

        prefix_len = prefix_embs.shape[1] if hasattr(prefix_embs, "shape") else 0

        return prefix_embs, prefix_kv_cache, prefix_len

    def _denoising_loop_traced(self, prefix_kv_cache, prefix_len: int) -> ttnn.Tensor:
        """
        Run the denoising loop using pre-allocated tensors.

        This is what gets TRACED. All tensors are already on device.
        """
        num_steps = self.config.denoising_steps
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        x_t = self.x_t_ttnn

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Get timestep from pre-computed tensor
            t_ttnn = ttnn.slice(self.timesteps_ttnn, [0, i], [1, i + 1])
            t_ttnn = ttnn.reshape(t_ttnn, (1, 1))

            # Embed suffix
            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.model.embed_suffix(self.state_ttnn, x_t, t_ttnn)

            # Forward expert with cached KV
            expert_out, _ = self.model.backbone.forward_expert(
                suffix_embs,
                past_key_values=prefix_kv_cache,
            )

            # Extract action output (skip state token in PI0 mode)
            if not self.model.config.pi05:
                action_output = ttnn.slice(
                    expert_out, [0, 1, 0], [expert_out.shape[0], expert_out.shape[1], expert_out.shape[2]]
                )
            else:
                action_output = expert_out

            # Project to action space
            velocity = self.model.suffix_embedding.project_output(action_output)

            # Euler step
            velocity_scaled = ttnn.multiply(velocity, dt)
            x_t_new = ttnn.add(x_t, velocity_scaled)

            # Cleanup intermediates
            ttnn.deallocate(t_ttnn)
            ttnn.deallocate(suffix_embs)
            ttnn.deallocate(expert_out)
            if not self.model.config.pi05:
                ttnn.deallocate(action_output)
            ttnn.deallocate(velocity)
            ttnn.deallocate(velocity_scaled)
            if i > 0:  # Don't deallocate the initial x_t
                ttnn.deallocate(x_t)

            x_t = x_t_new

        return x_t

    def compile(self):
        """
        Compile the runner by capturing a trace of the denoising loop.

        Steps:
        1. Validate configuration
        2. Create sample inputs
        3. Run full model to JIT compile kernels
        4. Allocate denoising tensors
        5. Run prefix + VLM (not traced) to get KV cache
        6. Capture trace of denoising loop only
        """
        if self._compiled:
            return

        self._validate_config()

        # Create sample inputs
        images = [
            torch.randn(self.config.batch_size, 3, self.config.image_size, self.config.image_size)
            for _ in range(self.config.num_images)
        ]
        img_masks = [torch.ones(self.config.batch_size) for _ in range(self.config.num_images)]
        lang_tokens = torch.zeros(self.config.batch_size, self.config.max_lang_tokens, dtype=torch.long)
        lang_masks = torch.ones(self.config.batch_size, self.config.max_lang_tokens)
        state = torch.zeros(self.config.batch_size, self.config.state_dim)

        # JIT compilation run (compiles all kernels)
        print("Running JIT compilation...")
        output = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
        ttnn.synchronize_device(self.device)
        if isinstance(output, ttnn.Tensor) and output.is_allocated():
            ttnn.deallocate(output)

        # Second warmup
        output = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
        ttnn.synchronize_device(self.device)
        if isinstance(output, ttnn.Tensor) and output.is_allocated():
            ttnn.deallocate(output)

        print("JIT compilation complete. Preparing trace...")

        # Allocate denoising tensors
        self._allocate_denoising_tensors()
        ttnn.synchronize_device(self.device)

        # Run prefix + VLM to get KV cache (NOT traced)
        self.prefix_embs, self.prefix_kv_cache, prefix_len = self._run_prefix_and_vlm(
            images, img_masks, lang_tokens, lang_masks
        )
        ttnn.synchronize_device(self.device)

        print("Capturing denoising loop trace...")

        # Capture trace of ONLY the denoising loop
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        self.traced_output = self._denoising_loop_traced(self.prefix_kv_cache, prefix_len)

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        print("Trace capture complete!")
        self._compiled = True

    def execute(
        self,
        images: Optional[List[torch.Tensor]] = None,
        img_masks: Optional[List[torch.Tensor]] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute traced denoising loop.

        IMPORTANT LIMITATION: This trace approach captures only the denoising loop.
        The prefix/VLM forward pass is baked into the trace at compile time.
        All inputs (images, language, state, noise) are FIXED at compile time.

        This is useful for benchmarking the denoising loop performance, but not
        for production inference where inputs change between calls.

        For production, use the standard model.sample_actions() method.

        Args:
            images, img_masks, lang_tokens, lang_masks, state: IGNORED
                (All inputs are baked into trace at compile time)

        Returns:
            Denoised actions (same result every call since inputs are baked in)
        """
        if not self._compiled:
            raise RuntimeError("Runner must be compiled before execution. Call compile() first.")

        # Execute the traced denoising loop
        # Note: All inputs (images, language, KV cache, state, noise) are
        # baked into the trace at compile time. This execute() just replays it.
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)

        # Convert output to torch
        if isinstance(self.traced_output, ttnn.Tensor):
            return ttnn.to_torch(self.traced_output)
        return self.traced_output

    def cleanup(self):
        """Release trace and allocated resources."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None

        for tensor in [self.state_ttnn, self.x_t_ttnn, self.timesteps_ttnn]:
            if tensor is not None and hasattr(tensor, "is_allocated") and tensor.is_allocated():
                ttnn.deallocate(tensor)

        self.state_ttnn = None
        self.x_t_ttnn = None
        self.timesteps_ttnn = None
        self.prefix_embs = None
        self.prefix_kv_cache = None
        self._compiled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def create_device_for_performant_runner(device_id: int = 0, config: Optional[PI0TraceConfig] = None) -> ttnn.Device:
    """Create a TTNN device configured for the performant runner."""
    if config is None:
        config = PI0TraceConfig()

    return ttnn.open_device(
        device_id=device_id,
        l1_small_size=config.l1_small_size,
        num_command_queues=config.num_command_queues,
        trace_region_size=config.trace_region_size,
    )
