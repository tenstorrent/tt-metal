# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime setup helpers for initialization orchestration."""

from typing import Any, Optional, Tuple

import torch
from loguru import logger


class InitServiceSetupMixin:
    """Device/runtime normalization and status helpers."""

    def _resolve_initialize_device(self, requested_device: str) -> str:
        """TTNN host_preprocess always runs HF stubs on CPU."""
        if requested_device in ("auto", "cpu"):
            return "cpu"
        logger.warning(
            "[initialize_service] device={!r} ignored in TTNN host_preprocess; using CPU.",
            requested_device,
        )
        return "cpu"

    def _configure_initialize_runtime(
        self,
        *,
        device: str,
        compile_model: bool,
        quantization: Optional[str],
    ) -> Tuple[bool, Optional[str], bool]:
        """Apply backend constraints and return normalized compile/quantization settings."""
        mlx_compile_requested = False
        normalized_compile = compile_model
        normalized_quantization = quantization

        if device == "mps":
            if normalized_compile:
                logger.info(
                    "[initialize_service] MPS detected: torch.compile is not "
                    "supported - redirecting to mx.compile for MLX components."
                )
                mlx_compile_requested = True
                normalized_compile = False
            if normalized_quantization is not None:
                logger.warning("[initialize_service] Quantization (torchao) is not supported on MPS; disabling.")
                normalized_quantization = None

        return normalized_compile, normalized_quantization, mlx_compile_requested

    @staticmethod
    def _ensure_len_for_compile(model: Any, method_name: str) -> None:
        """Inject a fallback ``__len__`` implementation for torch.compile introspection.

        Args:
            model: Model instance whose class may need a ``__len__`` shim.
            method_name: Label used in debug logs to identify the target model.
        """
        if hasattr(model.__class__, "__len__"):
            return

        def _len_impl(_model_self):
            """Return a neutral length for torch.compile compatibility."""
            return 0

        model.__class__.__len__ = _len_impl
        logger.debug(f"[initialize_service] Injected __len__ into {method_name} class for torch.compile")

    def _validate_quantization_setup(self, *, quantization: Optional[str], compile_model: bool) -> None:
        """Validate quantization prerequisites before model loading."""
        if quantization is None:
            return
        try:
            import torchao  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "torchao is required for quantization but is unavailable or incompatible "
                "with this PyTorch build. Please install a compatible torchao version."
            ) from exc

    def _initialize_mlx_backends(
        self,
        *,
        device: str,
        use_mlx_dit: bool,
        mlx_compile_requested: bool,
    ) -> Tuple[str, str]:
        """MLX backends are not used in TTNN host_preprocess (no-op)."""
        _ = device, use_mlx_dit, mlx_compile_requested
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_vae = None
        self.use_mlx_vae = False
        return "N/A (TTNN preprocess)", "N/A (TTNN preprocess)"

    @staticmethod
    def _build_initialize_status_message(
        *,
        device: str,
        model_path: str,
        vae_path: str,
        text_encoder_path: str,
        dtype: torch.dtype,
        attention: str,
        compile_model: bool,
        mlx_compile_requested: bool,
        offload_to_cpu: bool,
        offload_dit_to_cpu: bool,
        quantization: Optional[str],
        mlx_dit_status: str,
        mlx_vae_status: str,
        preprocess_only: bool = False,
    ) -> str:
        """Format initialize_service status output for UI/API consumers."""
        status_msg = f"[OK] Model initialized successfully on {device}\n"
        status_msg += f"Main model: {model_path}\n"
        status_msg += f"VAE: {vae_path}\n"
        status_msg += f"Text encoder: {text_encoder_path}\n"
        status_msg += f"Dtype: {dtype}\n"
        status_msg += f"Attention: {attention}\n"
        if preprocess_only:
            status_msg += "Mode: TTNN preprocess only (CPU batching; DiT/VAE on device)\n"
            status_msg += f"Quantization: {quantization or 'Disabled'}"
            return status_msg
        compiled_label = "mx.compile (MLX)" if mlx_compile_requested else str(compile_model)
        status_msg += f"Compiled: {compiled_label}\n"
        status_msg += f"Quantization: {quantization or 'Disabled'}\n"
        status_msg += f"Offload to CPU: {offload_to_cpu}\n"
        status_msg += f"Offload DiT to CPU: {offload_dit_to_cpu}\n"
        status_msg += f"MLX DiT: {mlx_dit_status}\n"
        status_msg += f"MLX VAE: {mlx_vae_status}"
        return status_msg
