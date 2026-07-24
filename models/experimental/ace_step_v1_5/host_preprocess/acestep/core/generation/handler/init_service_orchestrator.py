# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Top-level initialization orchestration for the handler."""

import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger


class InitServiceOrchestratorMixin:
    """Public ``initialize_service`` orchestration entrypoint."""

    def initialize_service(
        self,
        project_root: str,
        config_path: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        use_mlx_dit: bool = True,
        preprocess_only: bool = False,
        vae_checkpoint: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Initialize model artifacts and runtime backends for generation.

        This method intentionally supports repeated calls to reinitialize models
        with new settings; it does not short-circuit when components are already loaded.
        """
        try:
            if config_path is None:
                config_path = "acestep-v15-turbo"
                logger.warning("[initialize_service] config_path not set; defaulting to 'acestep-v15-turbo'.")

            self.preprocess_only = bool(preprocess_only)
            if self.preprocess_only:
                resolved_device = "cpu"
                use_mlx_dit = False
            else:
                resolved_device = self._resolve_initialize_device(device)
            self.device = resolved_device
            self.offload_to_cpu = offload_to_cpu
            self.offload_dit_to_cpu = offload_dit_to_cpu

            normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
                device=resolved_device,
                compile_model=compile_model,
                quantization=quantization,
            )
            self.compiled = normalized_compile
            self.dtype = torch.float32
            self.quantization = normalized_quantization
            try:
                self._validate_quantization_setup(
                    quantization=self.quantization,
                    compile_model=normalized_compile,
                )
            except ImportError as exc:
                if self.quantization is not None:
                    logger.warning(
                        "[initialize_service] Quantization disabled: {}",
                        exc,
                    )
                    self.quantization = None
                else:
                    raise

            from acestep.model_downloader import DEFAULT_VAE_VARIANT, get_checkpoints_dir

            env_ckpt = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
            if env_ckpt:
                checkpoint_dir = str(get_checkpoints_dir())
            elif project_root:
                # Normalize the caller-supplied project_root to a canonical
                # absolute path before joining, collapsing any ".." components
                # so the resulting checkpoint directory is deterministic and
                # self-contained.
                safe_root = os.path.realpath(os.path.abspath(project_root))
                checkpoint_dir = os.path.join(safe_root, "checkpoints")
            else:
                checkpoint_dir = str(get_checkpoints_dir())
            checkpoint_dir = os.path.realpath(checkpoint_dir)
            checkpoint_path = Path(checkpoint_dir)

            # ``config_path`` is expected to be a simple checkpoint name (e.g.
            # "acestep-v15-turbo") that is joined onto ``checkpoint_dir``.
            # Reject any value that would resolve outside the checkpoint
            # directory (absolute paths or ".." traversal) before it is used to
            # touch the filesystem.
            resolved_ckpt_root = os.path.realpath(checkpoint_dir)
            resolved_model_path = os.path.realpath(os.path.join(checkpoint_dir, config_path))
            try:
                within_root = os.path.commonpath([resolved_ckpt_root, resolved_model_path]) == resolved_ckpt_root
            except (ValueError, TypeError):
                within_root = False
            if not within_root:
                error_msg = f"Invalid config_path '{config_path}': resolves outside checkpoint directory"
                logger.error("[initialize_service] {}", error_msg)
                return error_msg, False

            # Resolve VAE selection: explicit param > env var > default.
            resolved_vae_variant = vae_checkpoint or os.environ.get("ACESTEP_VAE_CHECKPOINT") or DEFAULT_VAE_VARIANT

            precheck_failure = self._ensure_models_present(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                prefer_source=prefer_source,
                vae_variant=resolved_vae_variant,
            )
            if precheck_failure is not None:
                self.model = None
                self.vae = None
                self.text_encoder = None
                self.text_tokenizer = None
                self.config = None
                self.silence_latent = None
                return precheck_failure

            self._sync_model_code_if_needed(config_path, checkpoint_path)

            model_path = os.path.join(checkpoint_dir, config_path)
            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=resolved_device,
                use_flash_attention=use_flash_attention,
                compile_model=normalized_compile,
                quantization=self.quantization,
            )
            vae_path = self._load_vae_model(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
                compile_model=normalized_compile,
                vae_variant=resolved_vae_variant,
            )
            text_encoder_path = self._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
            )

            mlx_dit_status, mlx_vae_status = self._initialize_mlx_backends(
                device=resolved_device,
                use_mlx_dit=use_mlx_dit,
                mlx_compile_requested=mlx_compile_requested,
            )

            status_msg = self._build_initialize_status_message(
                device=resolved_device,
                model_path=model_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                dtype=self.dtype,
                attention=getattr(self.config, "_attn_implementation", "eager"),
                compile_model=normalized_compile,
                mlx_compile_requested=mlx_compile_requested,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                quantization=self.quantization,
                mlx_dit_status=mlx_dit_status,
                mlx_vae_status=mlx_vae_status,
                preprocess_only=self.preprocess_only,
            )

            self.last_init_params = {
                "project_root": project_root,
                "config_path": config_path,
                "device": resolved_device,
                "use_flash_attention": use_flash_attention,
                "compile_model": normalized_compile,
                "offload_to_cpu": offload_to_cpu,
                "offload_dit_to_cpu": offload_dit_to_cpu,
                "quantization": self.quantization,
                "use_mlx_dit": use_mlx_dit,
                "preprocess_only": self.preprocess_only,
                "prefer_source": prefer_source,
                "vae_checkpoint": resolved_vae_variant,
            }

            return status_msg, True
        except Exception as exc:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return error_msg, False
