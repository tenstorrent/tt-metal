# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-8B vLLM generator for tt-inference-server.

Integrates TtMolmo2Model with the vLLM plugin system following the Qwen3-VL pattern.
The server is started via tt-inference-server/run.py and the vLLM plugin
(TTMolmo2ForConditionalGeneration) is loaded by TTModelLoader.

Video processing: vLLM's Molmo2MultiModalProcessor calls the HF Molmo2VideoProcessor
internally, so pixel_values_videos arriving at prefill_forward are already-processed
Molmo2 patches — identical to demo.py inputs.
"""

from pathlib import Path
from typing import Mapping, Optional

import torch
from loguru import logger

import ttnn
from models.common.warmup import WarmupForwardMixin
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.tt_transformers.tt.ccl import TT_CCL
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.molmo2 import Molmo2DummyInputsBuilder, Molmo2MultiModalProcessor, Molmo2ProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY

WEIGHT_CACHE_PATH = Path("/tmp/molmo2_weight_cache")


def allocate_molmo2_kv_cache(kv_cache_shape, dtype, num_layers, model, cfg):
    """Replace each layer's KV cache with new TTNN tensors.

    vLLM requests a paged KV shape but our model uses sequential KV cache.
    We allocate using our model's native shape and return them for vLLM to
    pass back in prefill/decode calls.
    """
    import torch

    # Use our model's native KV shape: [max_batch, n_local_kv_heads, max_seq_len, head_dim]
    cache_shape = (cfg.max_batch_size, cfg.n_local_kv_heads, cfg.max_seq_len, cfg.head_dim)
    for layer_idx in range(num_layers):
        cache_kv = torch.zeros(cache_shape, dtype=torch.bfloat16)
        model.layers[layer_idx].attention.layer_past = [
            ttnn.as_tensor(
                cache_kv,
                device=model.mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                cache_file_name=None,
            )
            for _ in range(2)  # K and V
        ]
    return [layer.attention.layer_past for layer in model.layers]


class TT_Molmo2ProcessingInfo(Molmo2ProcessingInfo):
    """Enable both image and video modality for TT Molmo2."""

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1, "video": 1}


@MULTIMODAL_REGISTRY.register_processor(
    Molmo2MultiModalProcessor, info=TT_Molmo2ProcessingInfo, dummy_inputs=Molmo2DummyInputsBuilder
)
class Molmo2ForConditionalGeneration(WarmupForwardMixin, SupportsMultiModal):
    """TT Molmo2 vLLM generator.

    Wraps TtMolmo2Model to expose the vLLM generator interface expected by
    TTModelLoader and WarmupForwardMixin.
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, model, cfg, mesh_device, processor):
        self.model = model  # TtMolmo2Model
        self.cfg = cfg  # Molmo2Config
        self.mesh_device = mesh_device
        self.processor = processor
        self._decode_trace_captured = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
    ):
        """Called by TTModelLoader after vLLM resolves TTMolmo2ForConditionalGeneration."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from models.demos.molmo2.tt.model import TtMolmo2Model

        hf_model_id = getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", "")
        logger.info(f"Initializing Molmo2 TT model from {hf_model_id}")

        logger.info("Loading HF state dict (bfloat16)...")
        hf = AutoModelForImageTextToText.from_pretrained(
            hf_model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        state_dict = hf.state_dict()
        del hf

        processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)

        cfg = Molmo2Config(mesh_device=mesh_device)
        cfg.max_batch_size = max_batch_size
        cfg.max_seq_len = max_seq_len

        ccl = TT_CCL(mesh_device)
        WEIGHT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

        model = TtMolmo2Model(
            mesh_device=mesh_device,
            tt_ccl=ccl,
            state_dict=state_dict,
            weight_cache_path=WEIGHT_CACHE_PATH,
            dtype=ttnn.bfloat16,
            configuration=cfg,
        )
        del state_dict
        logger.info("TtMolmo2Model ready")

        return cls(model=model, cfg=cfg, mesh_device=mesh_device, processor=processor)

    @property
    def cache_path(self):
        return WEIGHT_CACHE_PATH

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_molmo2_kv_cache(*args, **kwargs, model=self.model, cfg=self.cfg)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, enable_trace, **kwargs):
        """Run prefill for a single user (batch_size=1).

        tokens: [batch, padded_seq_len] int32
        prompt_lens: [batch] int — actual length before padding
        kwargs: multimodal inputs from Molmo2MultiModalProcessor
          - pixel_values_videos: [n_frames, 729, 588] float — already-processed Molmo2 patches
          - video_token_pooling: [N_pooled, pool_window] int
          - pixel_values: [n_crops, 729, 588] float — for image inputs
          - image_token_pooling: [N_pooled, pool_window] int
          - token_type_ids: [1, S] int
        """
        # Extract multimodal inputs
        pv = None
        pool_idx = None
        token_type_ids = None

        if "token_type_ids" in kwargs and kwargs["token_type_ids"] is not None:
            token_type_ids = kwargs["token_type_ids"]

        # Prefer video over image
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        video_token_pooling = kwargs.get("video_token_pooling", None)
        pixel_values = kwargs.get("pixel_values", None)
        image_token_pooling = kwargs.get("image_token_pooling", None)

        if pixel_values_videos is not None:
            # Video input: [n_frames, 729, 588] → [1, n_frames, 729, 588]
            if isinstance(pixel_values_videos, list):
                pixel_values_videos = pixel_values_videos[0] if len(pixel_values_videos) > 0 else None
            if pixel_values_videos is not None:
                pv = pixel_values_videos.float().unsqueeze(0)
            if video_token_pooling is not None:
                if isinstance(video_token_pooling, list):
                    video_token_pooling = video_token_pooling[0]
                pool_idx = video_token_pooling.unsqueeze(0)
        elif pixel_values is not None:
            # Image input: [n_crops, 729, 588] → [1, n_crops, 729, 588]
            if isinstance(pixel_values, list):
                pixel_values = pixel_values[0] if len(pixel_values) > 0 else None
            if pixel_values is not None:
                pv = pixel_values.float().unsqueeze(0)
            if image_token_pooling is not None:
                if isinstance(image_token_pooling, list):
                    image_token_pooling = image_token_pooling[0]
                pool_idx = image_token_pooling.unsqueeze(0)

        # Trim tokens to actual prompt length (remove padding)
        seq_len = int(prompt_lens[0].item()) if hasattr(prompt_lens[0], "item") else int(prompt_lens[0])
        input_ids = tokens[:1, :seq_len]

        # Reset KV cache for this request
        self.model.reset_kv_cache(user_id=0)

        logger.info(
            f"Prefill: S={seq_len}, vision={'video' if pixel_values_videos is not None else 'image' if pixel_values is not None else 'none'}"
        )

        # Run TT prefill
        logits = self.model.forward_prefill(
            input_ids=input_ids,
            pixel_values=pv,
            pooled_patches_idx=pool_idx,
            token_type_ids=token_type_ids,
            user_id=0,
        )

        # Capture decode trace after first prefill (if enabled)
        if enable_trace and not self._decode_trace_captured:
            logger.info("Capturing decode trace...")
            self._capture_decode_trace(seq_len)

        # Return [1, 1, vocab_size] — vLLM expects logits for last token
        return logits, None  # (logits, rope_deltas=None)

    def _capture_decode_trace(self, prefill_seq_len: int):
        """Capture decode trace via TtMolmo2Model infrastructure."""
        self.model._decode_trace_tensors = self.model._allocate_decode_trace_tensors()
        self.model._decode_trace_id, self.model._decode_trace_output = self.model._capture_decode_trace(
            self.model._decode_trace_tensors, prefill_seq_len
        )
        self._decode_trace_captured = True
        logger.info("Decode trace captured")

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=False,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        """Run single-token decode.

        tokens: [batch, 1] int32 — current token
        start_pos: [batch] int32 — position in sequence
        enable_trace: if True and trace not captured, captures the trace first
        """
        token_id = int(tokens[0, 0].item())
        position = int(start_pos[0].item()) if hasattr(start_pos[0], "item") else int(start_pos[0])

        # Capture trace on first decode with enable_trace=True (warm-up phase)
        if enable_trace and not self._decode_trace_captured:
            logger.info("Capturing decode trace during warm-up...")
            self._capture_decode_trace(position)

        if self._decode_trace_captured:
            logits = self.model._execute_decode_trace(token_id, position)
        else:
            logits = self.model.forward_decode_step(token_id, position)

        return logits.unsqueeze(0)  # [1, vocab_size]
