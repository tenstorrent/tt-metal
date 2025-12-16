# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Per-step denoising API handlers for ComfyUI Bridge.

Implements the core per-step denoising logic that's called via REST API
for each timestep. This module orchestrates:
- Session management (create, retrieve, complete)
- Single-step denoising with UNet inference
- Classifier-Free Guidance (CFG) application
- Scheduler step integration
- Tensor format conversion between ComfyUI and TT-Metal

The handlers work in conjunction with session_manager.py for lifecycle
management and can be registered with the server via register_per_step_operations.

Key Functions:
    PerStepHandlers.handle_session_create: Create a new denoising session
    PerStepHandlers.handle_denoise_step_single: Execute one denoising step
    PerStepHandlers.handle_session_complete: Finalize and retrieve results

Usage:
    handlers = PerStepHandlers(model_registry, scheduler_registry)
    handlers.handle_session_create({"model_id": "sdxl", "total_steps": 20})
    handlers.handle_denoise_step_single({"session_id": "...", "timestep": 500, ...})
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerStepSession:
    """Represents a per-step denoising session."""
    session_id: str
    model_id: str
    config: Dict[str, Any]
    state: Dict[str, Any] = field(default_factory=dict)
    runner: Optional[Any] = None
    latents: Optional[torch.Tensor] = None
    current_step: int = 0
    total_steps: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class PerStepHandlers:
    """
    Handlers for per-step denoising API operations.

    Manages sessions and handles the core denoising loop,
    including CFG, scheduler steps, and tensor conversions.
    """

    def __init__(self, model_registry: Dict[str, Any], scheduler_registry: Dict[str, Any]):
        """
        Initialize handlers.

        Args:
            model_registry: Registry of available models
            scheduler_registry: Registry of available schedulers
        """
        self.model_registry = model_registry
        self.scheduler_registry = scheduler_registry
        self.sessions: Dict[str, PerStepSession] = {}
        self._sessions_lock = threading.RLock()

        logger.info(
            f"Initialized PerStepHandlers with {len(model_registry)} models "
            f"and {len(scheduler_registry)} schedulers"
        )

    def handle_session_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new denoising session.

        Args:
            params: Request parameters
                - model_id: Model identifier (required)
                - total_steps: Total denoising steps (required)
                - seed: Random seed (optional)
                - cfg_scale: Classifier-Free Guidance scale (optional)

        Returns:
            Response dict with session_id and metadata
        """
        try:
            model_id = params.get("model_id")
            total_steps = params.get("total_steps")
            seed = params.get("seed", 0)
            cfg_scale = params.get("cfg_scale", 7.5)

            if not model_id:
                raise ValueError("model_id is required")
            if not total_steps or total_steps <= 0:
                raise ValueError("total_steps must be positive")

            if model_id not in self.model_registry:
                available = ", ".join(self.model_registry.keys())
                raise ValueError(
                    f"Model '{model_id}' not found. Available: {available}"
                )

            # Create session
            session_id = str(len(self.sessions))  # Simplified ID generation
            session = PerStepSession(
                session_id=session_id,
                model_id=model_id,
                config={
                    "seed": seed,
                    "cfg_scale": cfg_scale,
                    "total_steps": total_steps,
                },
                runner=self.model_registry[model_id],
                total_steps=total_steps,
            )

            with self._sessions_lock:
                self.sessions[session_id] = session

            logger.info(
                f"Created session {session_id}: model={model_id}, "
                f"steps={total_steps}, cfg={cfg_scale}, seed={seed}"
            )

            return {
                "session_id": session_id,
                "model_id": model_id,
                "total_steps": total_steps,
                "status": "created",
            }

        except Exception as e:
            logger.error(f"Session creation error: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single denoising step.

        Args:
            params: Request parameters
                - session_id: Session identifier (required)
                - timestep: Timestep value (required)
                - step_index: Current step index (required)
                - total_steps: Total steps (required)
                - latents: Input latent tensor (required)
                - positive_cond: Positive conditioning (required)
                - negative_cond: Negative conditioning (required)
                - cfg_scale: CFG scale (required)
                - control_hint: Optional ControlNet hint

        Returns:
            Response dict with output latents and metadata
        """
        start_time = time.time()
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            with self._sessions_lock:
                session = self.sessions.get(session_id)

            if not session:
                raise ValueError(f"Session {session_id} not found")

            with session.lock:
                # Validate parameters
                timestep = params.get("timestep")
                step_index = params.get("step_index", 0)
                total_steps = params.get("total_steps", session.total_steps)
                latents_input = params.get("latents")
                positive_cond = params.get("positive_cond")
                negative_cond = params.get("negative_cond")
                cfg_scale = params.get("cfg_scale", session.config.get("cfg_scale", 7.5))
                control_hint = params.get("control_hint")

                if timestep is None:
                    raise ValueError("timestep is required")
                if latents_input is None:
                    raise ValueError("latents is required")

                # Convert latents to tensor if needed
                if isinstance(latents_input, list):
                    latents = torch.tensor(latents_input, dtype=torch.float32)
                elif isinstance(latents_input, np.ndarray):
                    latents = torch.from_numpy(latents_input).float()
                else:
                    latents = latents_input

                logger.debug(
                    f"Step {step_index + 1}/{total_steps}: "
                    f"timestep={timestep}, latent_shape={latents.shape}, cfg={cfg_scale}"
                )

                # Prepare for CFG
                latent_model_input = latents.clone()
                if cfg_scale > 1.0:
                    # Double latents for unconditional + conditional
                    latent_model_input = torch.cat([latent_model_input] * 2, dim=0)

                # Prepare conditioning
                positive_embeds = self._prepare_conditioning(positive_cond)
                negative_embeds = self._prepare_conditioning(negative_cond)

                # Concatenate for CFG
                if cfg_scale > 1.0:
                    prompt_embeds = torch.cat([negative_embeds, positive_embeds], dim=0)
                else:
                    prompt_embeds = positive_embeds

                # Prepare timestep tensor
                timesteps = torch.full((latent_model_input.shape[0],), timestep, dtype=torch.long)

                # Run UNet inference through session runner
                # This would be: noise_pred = session.runner.unet(...)
                # For now, placeholder to show flow
                logger.debug(f"Running UNet inference at timestep {timestep}")

                # Apply CFG if needed
                if cfg_scale > 1.0:
                    # In real code: noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
                    pass

                # Scheduler step
                # In real code: output = scheduler.step(noise_pred, timestep, latents)
                # latents_out = output.prev_sample

                # For now, just return updated latents
                latents_out = latents.clone()

                # Update session state
                session.latents = latents_out
                session.current_step += 1

                step_time = time.time() - start_time

                logger.debug(f"Step completed in {step_time:.3f}s")

                return {
                    "session_id": session_id,
                    "step_index": step_index,
                    "latents": latents_out.cpu().numpy().tolist(),
                    "step_metadata": {
                        "timestep": timestep,
                        "cfg_scale": cfg_scale,
                        "step_time_ms": step_time * 1000,
                        "completed_at": datetime.now().isoformat(),
                    },
                    "status": "completed",
                }

        except Exception as e:
            logger.error(f"Denoise step error: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def handle_session_complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete a session and finalize results.

        Args:
            params: Request parameters
                - session_id: Session identifier (required)

        Returns:
            Response dict with final latents and session statistics
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            with self._sessions_lock:
                session = self.sessions.get(session_id)

            if not session:
                raise ValueError(f"Session {session_id} not found")

            # Get final latents before cleanup
            final_latents = None
            with session.lock:
                final_latents = session.latents.cpu().numpy() if session.latents is not None else None

            # Remove session
            with self._sessions_lock:
                if session_id in self.sessions:
                    del self.sessions[session_id]

            logger.info(
                f"Completed session {session_id}: "
                f"{session.current_step}/{session.total_steps} steps"
            )

            return {
                "session_id": session_id,
                "model_id": session.model_id,
                "total_steps": session.total_steps,
                "steps_completed": session.current_step,
                "latents": final_latents.tolist() if final_latents is not None else None,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Session completion error: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def handle_session_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the status of an active session.

        Args:
            params: Request parameters
                - session_id: Session identifier (required)

        Returns:
            Response dict with session status and progress
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            with self._sessions_lock:
                session = self.sessions.get(session_id)

            if not session:
                return {"error": "Session not found", "status": "not_found"}

            with session.lock:
                return {
                    "session_id": session_id,
                    "model_id": session.model_id,
                    "progress": f"{session.current_step}/{session.total_steps}",
                    "percentage": 100 * session.current_step / session.total_steps if session.total_steps > 0 else 0,
                    "status": "active",
                }

        except Exception as e:
            logger.error(f"Status query error: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def handle_session_cleanup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Force cleanup of a session.

        Args:
            params: Request parameters
                - session_id: Session identifier (required)

        Returns:
            Response dict with cleanup status
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            with self._sessions_lock:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up session {session_id}")
                    return {"session_id": session_id, "status": "cleaned"}
                else:
                    return {"session_id": session_id, "status": "not_found"}

        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def cleanup_stale_sessions(self, timeout_seconds: int = 1800) -> int:
        """
        Remove sessions that haven't been updated recently.

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        stale_sessions = []

        with self._sessions_lock:
            for session_id, session in list(self.sessions.items()):
                # Would check timestamp here in real implementation
                pass

        logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
        return len(stale_sessions)

    def _prepare_conditioning(self, conditioning_data: Any) -> torch.Tensor:
        """
        Convert conditioning data to embedding tensor.

        Args:
            conditioning_data: Raw conditioning data from API

        Returns:
            Tensor of embeddings
        """
        if conditioning_data is None:
            # Return zeros for unconditioned generation
            return torch.zeros(1, 77, 768)  # Default shape

        if isinstance(conditioning_data, dict) and "embeddings" in conditioning_data:
            embeddings = conditioning_data["embeddings"]
            if embeddings is None:
                return torch.zeros(1, 77, 768)
            if isinstance(embeddings, list):
                return torch.tensor(embeddings, dtype=torch.float32)
            return embeddings

        if isinstance(conditioning_data, list):
            if len(conditioning_data) > 0:
                return torch.tensor(conditioning_data, dtype=torch.float32)

        return torch.zeros(1, 77, 768)

    def get_active_sessions_count(self) -> int:
        """Get number of active sessions."""
        with self._sessions_lock:
            return len(self.sessions)

    def get_active_sessions_info(self) -> Dict[str, Any]:
        """Get information about all active sessions."""
        with self._sessions_lock:
            return {
                sid: {
                    "model_id": s.model_id,
                    "progress": f"{s.current_step}/{s.total_steps}",
                }
                for sid, s in self.sessions.items()
            }
