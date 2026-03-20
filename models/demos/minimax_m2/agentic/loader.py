# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
loader.py — one-time model loading for the N300 agentic workflow.

All models share the same mesh_device (opened once by the caller).
After load_all_models() returns, no further model loading occurs.

ARCHITECTURE: All models run on full (1,2) mesh with fabric_config enabled.
Fabric enables multi-chip tensor handling via mesh_composer for all models.

Estimated DRAM budget (BF8/BF16 mix) across both chips:

  Llama 3.1 8B (BF8)  ~8.0 GB (sharded across 2 chips)
  Whisper distil-v3   ~1.5 GB
  SpeechT5            ~0.6 GB
  OWL-ViT             ~0.3 GB
  BERT Large          ~0.7 GB
  KV cache + traces   ~1.0 GB
  ─────────────────────────────
  Total               ~12.1 GB / 24 GB  (across 2 × 12 GB chips)
"""

from dataclasses import dataclass

from loguru import logger

import ttnn


@dataclass
class ModelBundle:
    """Container for all loaded models."""

    llm: object = None
    whisper: object = None
    speecht5: object = None
    owlvit: object = None
    bert: object = None
    sd: object = None  # Stable Diffusion
    yunet: object = None  # Face detection
    t5: object = None  # Translation


def open_n300_device(enable_fabric: bool = True) -> ttnn.MeshDevice:
    """
    Open the N300 mesh device (1 row × 2 cols = 2 × Wormhole B0 chips).

    Device params satisfy the most demanding requirements across all models:
      l1_small_size=24_576      : SpeechT5 needs >1024 for decoder ops.
                                  300_000 clashes with Whisper's L1 buffers at
                                  offset 1018400 (CB region would end at 1185120).
                                  24576 ends at ~909696 — safely below 1018400.
      trace_region_size=100MB   : Whisper decode trace (largest requirement)
      num_command_queues=2      : SpeechT5 2CQ async decode uses ttnn.wait_for_event
                                  on CQ1; Whisper only uses CQ0 — CQ1 is unused by it.
      enable_fabric             : Required for LLM on full (1,2) mesh. Calls
                                  ttnn.set_fabric_config() before opening mesh.
    """
    # Enable fabric before opening mesh device (required for LLM on full mesh)
    if enable_fabric:
        logger.info("Enabling fabric config for multi-chip LLM support...")
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,  # tensix_config placeholder
            ttnn.FabricTensixConfig.DISABLED,
        )

    device_params = {
        "l1_small_size": 24_576,
        "trace_region_size": 100_000_000,
        "num_command_queues": 2,
    }
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        **device_params,
    )
    mesh_device.enable_program_cache()
    logger.info(f"Opened N300 mesh device: {mesh_device.get_num_devices()} chips")
    return mesh_device


def load_all_models(
    mesh_device,
    load_llm: bool = True,
    load_whisper: bool = True,
    load_speecht5: bool = True,
    load_owlvit: bool = True,
    load_bert: bool = True,
    load_sd: bool = False,  # Stable Diffusion (disabled by default - large model)
    load_yunet: bool = False,  # Face detection
    load_t5: bool = False,  # Translation
) -> ModelBundle:
    """
    Load all specialist models into device DRAM.

    All models run on full (1,2) mesh with fabric enabled.

    Args:
        mesh_device:  Opened N300 MeshDevice (or single-chip device for testing).
        load_*:       Flags to selectively skip loading specific models.

    Returns:
        ModelBundle with all requested models loaded and ready.
    """
    bundle = ModelBundle()

    # All models run on full (1,2) mesh with fabric enabled for multi-chip parallelism.

    # --- LLM (Llama 3.1 8B Instruct — orchestrator) ----------------------
    if load_llm:
        logger.info("[1/8] Loading LLM orchestrator (Llama 3.1 8B Instruct on TTNN)...")
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

        bundle.llm = LLMTool(mesh_device=mesh_device)

    # --- Whisper (STT + translate) --------------------------------------
    if load_whisper:
        logger.info("[2/8] Loading Whisper STT (distil-large-v3)...")
        from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

        bundle.whisper = WhisperTool(mesh_device=mesh_device)

    # --- SpeechT5 (TTS) -------------------------------------------------
    if load_speecht5:
        logger.info("[3/8] Loading SpeechT5 TTS...")
        from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

        bundle.speecht5 = SpeechT5Tool(mesh_device=mesh_device)

    # --- OWL-ViT (object detection) -------------------------------------
    if load_owlvit:
        logger.info("[4/8] Loading OWL-ViT object detection...")
        from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

        bundle.owlvit = OWLViTTool(mesh_device=mesh_device)

    # --- BERT Large (extractive QA) -------------------------------------
    if load_bert:
        logger.info("[5/8] Loading BERT Large QA...")
        from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

        bundle.bert = BERTTool(mesh_device=mesh_device)

    # --- Stable Diffusion (text-to-image) -------------------------------
    if load_sd:
        logger.info("[6/8] Loading Stable Diffusion...")
        from models.demos.minimax_m2.agentic.tool_wrappers.sd_tool import StableDiffusionTool

        bundle.sd = StableDiffusionTool(mesh_device=mesh_device)

    # --- YUNet (face detection) -----------------------------------------
    if load_yunet:
        logger.info("[7/8] Loading YUNet face detection...")
        from models.demos.minimax_m2.agentic.tool_wrappers.yunet_tool import YUNetTool

        bundle.yunet = YUNetTool(mesh_device=mesh_device)

    # --- T5 (translation) -----------------------------------------------
    if load_t5:
        logger.info("[8/8] Loading T5 translation...")
        from models.demos.minimax_m2.agentic.tool_wrappers.t5_tool import T5Tool

        bundle.t5 = T5Tool(mesh_device=mesh_device)

    logger.info("All models loaded. Agentic system ready.")
    return bundle


def cleanup_models(bundle: ModelBundle) -> None:
    """
    Release all model traces before closing the device.

    MUST be called before ttnn.close_mesh_device() to prevent segfault.
    The issue is that Python's garbage collection runs __del__ methods
    after the device is closed, causing trace release to fail.
    """
    logger.info("Cleaning up models (releasing traces)...")

    # LLM - release decode traces
    if bundle.llm is not None and hasattr(bundle.llm, "close"):
        try:
            bundle.llm.close()
        except Exception as e:
            logger.warning(f"LLM cleanup failed: {e}")

    # Whisper - release decoder trace
    if bundle.whisper is not None and hasattr(bundle.whisper, "close"):
        try:
            bundle.whisper.close()
        except Exception as e:
            logger.warning(f"Whisper cleanup failed: {e}")

    # SpeechT5 - release any traces
    if bundle.speecht5 is not None and hasattr(bundle.speecht5, "close"):
        try:
            bundle.speecht5.close()
        except Exception as e:
            logger.warning(f"SpeechT5 cleanup failed: {e}")

    # OWL-ViT and BERT don't use traces, but call close if available
    if bundle.owlvit is not None and hasattr(bundle.owlvit, "close"):
        try:
            bundle.owlvit.close()
        except Exception as e:
            logger.warning(f"OWL-ViT cleanup failed: {e}")

    if bundle.bert is not None and hasattr(bundle.bert, "close"):
        try:
            bundle.bert.close()
        except Exception as e:
            logger.warning(f"BERT cleanup failed: {e}")

    # New models - SD, YUNet, T5
    if bundle.sd is not None and hasattr(bundle.sd, "close"):
        try:
            bundle.sd.close()
        except Exception as e:
            logger.warning(f"Stable Diffusion cleanup failed: {e}")

    if bundle.yunet is not None and hasattr(bundle.yunet, "close"):
        try:
            bundle.yunet.close()
        except Exception as e:
            logger.warning(f"YUNet cleanup failed: {e}")

    if bundle.t5 is not None and hasattr(bundle.t5, "close"):
        try:
            bundle.t5.close()
        except Exception as e:
            logger.warning(f"T5 cleanup failed: {e}")

    logger.info("Model cleanup complete.")
