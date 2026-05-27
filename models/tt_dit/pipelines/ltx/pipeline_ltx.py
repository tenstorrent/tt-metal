# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Video Generation Pipeline for tt_dit.

Implements the text-to-video inference pipeline:
1. Text encoding (Gemma, torch-only)
2. Sigma schedule computation (LTX2Scheduler)
3. Denoising loop (Euler first-order steps with CFG)
4. VAE decoding (future)

Reference: LTX-2/packages/ltx-pipelines/ + Wan pipeline_wan.py
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...models.transformers.ltx.ltx_transformer import LTXTransformerModel
from ...parallel.config import DiTParallelConfig, ParallelFactor, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard

if TYPE_CHECKING:
    pass


# =============================================================================
# Scheduler
# =============================================================================

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def compute_sigmas(
    steps: int,
    num_tokens: int = MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.FloatTensor:
    """
    Compute the LTX-2 sigma schedule.

    Generates a sequence of noise levels (sigmas) from high noise (~1.0)
    to low noise (~terminal) with token-count-dependent shifting.

    Args:
        steps: Number of denoising steps
        num_tokens: Number of spatial tokens (affects sigma shift)
        max_shift: Maximum shift factor
        base_shift: Base shift factor
        stretch: Whether to stretch schedule to terminal value
        terminal: Final sigma value

    Returns:
        Tensor of shape (steps + 1,) with sigma values
    """
    sigmas = torch.linspace(1.0, 0.0, steps + 1)

    # Adaptive shift based on token count
    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = num_tokens * mm + b

    # Exponential shift
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
        0,
    )

    # Stretch to terminal
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return sigmas.to(torch.float32)


# =============================================================================
# Diffusion step
# =============================================================================


def euler_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    """
    First-order Euler diffusion step.

    x_{t+1} = x_t + velocity * dt
    where velocity = (x_t - denoised) / sigma, dt = sigma_next - sigma

    Args:
        sample: Current noisy latent
        denoised: Model's denoised prediction
        sigma: Current noise level
        sigma_next: Next noise level

    Returns:
        Updated latent at next noise level
    """
    dt = sigma_next - sigma
    velocity = (sample.float() - denoised.float()) / sigma
    return (sample.float() + velocity * dt).to(sample.dtype)


# =============================================================================
# Pipeline
# =============================================================================


class LTXPipeline:
    """
    LTX-2 text-to-video generation pipeline.

    Usage:
        pipeline = LTXPipeline(mesh_device, parallel_config, ccl_manager)
        pipeline.load_transformer(checkpoint_or_state_dict)
        pipeline.load_text_encoder(gemma_checkpoint)

        output = pipeline(
            prompt="A cat playing piano",
            num_frames=33,
            height=480,
            width=832,
            num_inference_steps=30,
        )
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        ccl_manager: CCLManager,
        *,
        vae_parallel_config: VaeHWParallelConfig | None = None,
        # Transformer config
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        # Mode: "video" or "av"
        mode: str = "video",
        # RoPE config
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: float = 1000.0,
    ):
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        # VAE H/W shard config. Defaults to mirroring transformer's tp/sp axes:
        # H gets the tp_axis, W gets the sp_axis (matches Wan's role assignment).
        if vae_parallel_config is None:
            vae_parallel_config = VaeHWParallelConfig(
                height_parallel=parallel_config.tensor_parallel,
                width_parallel=parallel_config.sequence_parallel,
            )
        self.vae_parallel_config = vae_parallel_config
        self.mode = mode

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.cross_attention_dim = cross_attention_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [20, 2048, 2048]
        self.timestep_scale_multiplier = timestep_scale_multiplier

        self.is_fsdp: bool = False
        self.dynamic_load: bool = False
        self.transformer: LTXTransformerModel | None = None
        self.text_encoder: GemmaTokenizerEncoderPair | None = None
        self.gemma_encoder = None  # TTNN Gemma encoder (device)
        self.gemma_tokenizer = None
        self.video_connector = None  # EmbeddingsConnector for video
        self.audio_connector = None  # EmbeddingsConnector for audio
        self.vae_decoder = None  # LTXVideoDecoder or LTXVideoDecoderTorch

        # Cached device tensors (computed once, reused across steps)
        self._cached_rope_cos: ttnn.Tensor | None = None
        self._cached_rope_sin: ttnn.Tensor | None = None
        # trans_mat not needed: using SPLIT RoPE (not interleaved)
        self._cached_prompt: ttnn.Tensor | None = None
        self._cached_negative_prompt: ttnn.Tensor | None = None

        # Checkpoint identifiers (Wan-style). Set by create_pipeline; consumed by the
        # _prepare_* methods (transformer, vae, connectors) and the reference encode
        # helpers. Either a local path or a HuggingFace repo string; auto-resolved on use.
        self.checkpoint_name: str | None = None
        self.gemma_path: str | None = None

        # Optional LoRA specs fused into the transformer state dict on next
        # ``_prepare_transformer`` call. Empty by default — distilled-LoRA stage-2
        # pipelines (see ``LTXAVTwoStagesPipeline``) toggle these between stages.
        # The cache key is suffixed when non-empty so LoRA-fused and base weights
        # do not alias in ``TT_DIT_CACHE_DIR``.
        from ...utils.lora import LoraSpec  # local import to avoid circulars

        self._lora_specs: list[LoraSpec] = []

    @staticmethod
    def _resolve_checkpoint_file(checkpoint: str, default_filename: str = "ltx-2.3-22b-dev.safetensors") -> str:
        """Resolve a checkpoint reference to a local file path.ß"""
        if os.path.exists(checkpoint):
            return checkpoint
        from huggingface_hub import hf_hub_download

        if ":" in checkpoint:
            repo_id, filename = checkpoint.split(":", 1)
        else:
            repo_id, filename = checkpoint, default_filename
        logger.info(f"Resolving HuggingFace checkpoint {repo_id}:{filename} (auto-download if missing)")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    @staticmethod
    def _resolve_gemma_dir(gemma: str) -> str:
        """Resolve a Gemma reference to a local directory path.

        Accepts a local directory or a HuggingFace repo ID. Auto-snapshot-downloads if needed.
        """
        if os.path.isdir(gemma):
            return gemma
        from huggingface_hub import snapshot_download

        logger.info(f"Resolving HuggingFace Gemma repo {gemma} (auto-download if missing)")
        return snapshot_download(repo_id=gemma)

    @staticmethod
    def create_pipeline(
        mesh_device: ttnn.MeshDevice,
        *,
        checkpoint_name: str | None = None,
        gemma_path: str | None = None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
        mode: str = "av",
        pipeline_class: type["LTXPipeline"] | None = None,
    ) -> "LTXPipeline":
        """Factory method matching Wan's create_pipeline pattern.

        Auto-configures parallel settings from mesh shape. `checkpoint_name` and `gemma_path`
        accept either local paths or HuggingFace repo strings (auto-downloaded on first use).
        """
        mesh_shape = tuple(mesh_device.shape)
        device_configs: dict[tuple[int, int], dict] = {}
        if ttnn.device.is_blackhole():
            device_configs[(2, 4)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                # DIAGNOSTIC: was True in merge 87c799a7c54 to match Wan BH 2x4 pattern, but
                # produced 11s/step + noise output (race symptom). Reverted to False to
                # isolate whether dynamic_load is the cause.
                "dynamic_load": True,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
            }
            defaults = device_configs.get(mesh_shape, device_configs[(2, 4)])
        else:
            device_configs[(1, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 1,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(2, 4)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 4,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
            }
            defaults = device_configs.get(mesh_shape, device_configs[(2, 4)])
        sp_axis = sp_axis if sp_axis is not None else defaults["sp_axis"]
        tp_axis = tp_axis if tp_axis is not None else defaults["tp_axis"]
        num_links = num_links if num_links is not None else defaults["num_links"]
        dynamic_load = dynamic_load if dynamic_load is not None else defaults["dynamic_load"]
        topology = topology if topology is not None else defaults["topology"]
        is_fsdp = is_fsdp if is_fsdp is not None else defaults["is_fsdp"]

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
            tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
        )
        # VAE shards H on the tp axis, W on the sp axis — same role mapping as Wan.
        # For bh_2x4sp1tp0 (mesh 2x4, tp=axis 0, sp=axis 1): h_factor=2, w_factor=4.
        vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
            width_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        )
        ccl_manager = CCLManager(mesh_device, topology=topology)

        pipeline_cls = pipeline_class or LTXPipeline
        pipeline = pipeline_cls(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            vae_parallel_config=vae_parallel_config,
            mode=mode,
        )
        pipeline.is_fsdp = is_fsdp
        pipeline.dynamic_load = dynamic_load

        if checkpoint_name is not None:
            pipeline.checkpoint_name = LTXPipeline._resolve_checkpoint_file(checkpoint_name)
            pipeline._load_config_from_checkpoint()
        if gemma_path is not None:
            pipeline.gemma_path = LTXPipeline._resolve_gemma_dir(gemma_path)

        return pipeline

    def _load_config_from_checkpoint(self) -> None:
        """Parse the safetensors header to detect model variant + VAE config. No tensor loads."""
        import json

        from safetensors import safe_open

        assert self.checkpoint_name is not None
        checkpoint_path = self.checkpoint_name

        # Transformer + connector detection (key scan + one tensor-shape read).
        with safe_open(checkpoint_path, framework="pt") as f:
            keys = list(f.keys())
            adaln_key = "model.diffusion_model.adaln_single.linear.weight"
            if adaln_key in keys:
                self._cross_attention_adaln = f.get_tensor(adaln_key).shape[0] > 6 * self.inner_dim
            else:
                self._cross_attention_adaln = True
            self._has_gate = any("to_gate_logits" in k for k in keys)
        logger.info(f"Detected: has_gate={self._has_gate}, cross_attention_adaln={self._cross_attention_adaln}")

        has_connectors = any(
            k.startswith("text_embedding_projection.")
            or k.startswith("model.diffusion_model.video_embeddings_connector.")
            for k in keys
        )
        if has_connectors:
            self._connector_checkpoint_path = checkpoint_path

        # VAE config from JSON metadata header.
        with open(checkpoint_path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size))
        vae_cfg = json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})
        self._vae_checkpoint_path = checkpoint_path
        self._vae_decoder_blocks = vae_cfg.get("decoder_blocks", [])
        self._vae_causal = vae_cfg.get("causal_decoder", False)
        self._vae_base_channels = vae_cfg.get("decoder_base_channels", 128)
        if self._vae_decoder_blocks:
            logger.info(f"VAE config: {len(self._vae_decoder_blocks)} blocks, causal={self._vae_causal}")

    def _transformer_state_dict(self) -> dict[str, torch.Tensor]:
        """Lazy state dict provider for the transformer. Invoked only on cache miss.

        Mirrors Wan's `lambda: state.torch_model.state_dict()` but for safetensors —
        Wan keeps the torch model in host RAM; LTX reads the safetensors on demand.
        When ``self._lora_specs`` is non-empty the resulting state dict is fused
        with those LoRAs before being returned (mirrors the reference
        ``DiffusionStage(loras=...)`` builder path).
        """
        from safetensors.torch import load_file

        logger.info(f"Transformer cache miss — loading safetensors: {self.checkpoint_name}")
        raw = load_file(self.checkpoint_name)
        prefix = "model.diffusion_model."
        sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        if self._lora_specs:
            from ...utils.lora import fuse_loras_into

            sd = fuse_loras_into(sd, self._lora_specs)
        return sd

    def _transformer_cache_name(self) -> str:
        """Cache key for ``cache_module.load_model``. Append a LoRA tag so the
        base and LoRA-fused weights do not collide in ``TT_DIT_CACHE_DIR``.
        """
        base = os.path.basename(self.checkpoint_name).removesuffix(".safetensors")
        if not self._lora_specs:
            return base
        tag = "+".join(
            f"{os.path.basename(s.path).removesuffix('.safetensors')}@{s.strength}" for s in self._lora_specs
        )
        return f"{base}.lora-{tag}"

    def _prepare_transformer(self) -> None:
        """Push transformer weights onto the mesh. Cached when `TT_DIT_CACHE_DIR` is set.

        Mirrors Wan's `_prepare_transformer`. Construction flags + Module instantiation
        happen once in `create_pipeline` via `_load_config_from_checkpoint`; this method
        only handles weight transfer.
        """
        if self.transformer is None:
            self.transformer = LTXTransformerModel(
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                cross_attention_dim=self.cross_attention_dim,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                has_audio=self.mode == "av",
                apply_gated_attention=self._has_gate,
                cross_attention_adaln=self._cross_attention_adaln,
            )
        cache_module.load_model(
            self.transformer,
            model_name=self._transformer_cache_name(),
            subfolder="transformer",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=self._transformer_state_dict,
        )
        logger.info(f"Loaded LTX transformer ({self.mode} mode) with {self.num_layers} layers")

    def load_transformer(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Direct state-dict load (used by tests with random weights, no caching).

        For the cache-aware production path, use `_prepare_transformer` instead.
        Auto-detects model variant flags from the state dict.
        """
        has_gate = any("to_gate_logits" in k for k in state_dict)
        # 9-output (22B): adaln_single.linear.weight has shape (9*dim, dim).
        # 6-output (19B distilled): shape (6*dim, dim).
        adaln_weight = state_dict.get("adaln_single.linear.weight")
        cross_attention_adaln = True if adaln_weight is None else adaln_weight.shape[0] > 6 * self.inner_dim

        self.transformer = LTXTransformerModel(
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            cross_attention_dim=self.cross_attention_dim,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            has_audio=self.mode == "av",
            apply_gated_attention=has_gate,
            cross_attention_adaln=cross_attention_adaln,
        )
        self.transformer.load_torch_state_dict(state_dict)
        logger.info(f"Loaded LTX transformer ({self.mode} mode) with {self.num_layers} layers")

    def load_text_encoder(
        self,
        checkpoint: str = "google/gemma-3-12b-it",
        *,
        sequence_length: int = 1024,
        hidden_layer_index: int = -1,
    ) -> None:
        """Load Gemma text encoder (torch-only). Fallback when reference encode_prompts is not available."""
        self.text_encoder = GemmaTokenizerEncoderPair(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            embedding_dim=self.cross_attention_dim,
            hidden_layer_index=hidden_layer_index,
        )
        logger.info(f"Loaded Gemma text encoder from {checkpoint}")

    def load_gemma_encoder(
        self,
        gemma_path: str,
        *,
        num_layers: int = 48,
        hidden_layer_index: int = -1,
        sequence_length: int = 1024,
    ) -> None:
        """Load TTNN Gemma-3 text encoder on device. 13x faster than CPU torch.

        Args:
            gemma_path: HuggingFace model path or local directory
            num_layers: Number of Gemma layers (48 for 12B)
            hidden_layer_index: Which layer's hidden states to extract (-1 = last)
            sequence_length: Max token sequence length
        """
        import glob

        from transformers import AutoTokenizer

        from ...encoders.gemma.model_gemma import GemmaConfig, GemmaEncoder
        from ...parallel.config import EncoderParallelConfig

        config = GemmaConfig(
            num_hidden_layers=num_layers,
            hidden_layer_index=hidden_layer_index,
            max_position_embeddings=sequence_length,
        )

        # Use TP on the same axis as the DiT transformer
        tp_factor = self.parallel_config.tensor_parallel.factor
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        enc_parallel = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis))

        enc_ccl = CCLManager(self.mesh_device, topology=ttnn.Topology.Linear)

        self.gemma_encoder = GemmaEncoder(config, self.mesh_device, enc_ccl, enc_parallel)

        # Load weights
        from safetensors.torch import load_file

        weight_files = sorted(glob.glob(f"{gemma_path}/model-*.safetensors"))
        if not weight_files:
            weight_files = sorted(glob.glob(f"{gemma_path}/*.safetensors"))
        state_dict = {}
        for f in weight_files:
            state_dict.update(load_file(f))

        t0 = __import__("time").time()
        self.gemma_encoder.load_torch_state_dict(state_dict)
        del state_dict
        logger.info(f"Loaded TTNN Gemma encoder ({num_layers}L) in {__import__('time').time()-t0:.0f}s")

        self.gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_path)
        # Use left-padding matching the reference FeatureExtractorV2 pipeline.
        # Left-padding: [PAD, ..., PAD, BOS, real tokens]. With causal SDPA,
        # real tokens at the end attend to everything including padding positions.
        # The reference handles this the same way (padding hidden states are zeroed
        # out after encoding via attention_mask).
        self._gemma_hidden_layer_index = hidden_layer_index
        self._gemma_sequence_length = sequence_length

    def load_embeddings_connectors(
        self,
        checkpoint_state: dict[str, torch.Tensor],
        *,
        gemma_hidden_size: int = 3840,
        gemma_num_layers: int = 49,  # embedding layer + 48 decoder layers
        video_num_blocks: int = 8,
        audio_num_blocks: int = 2,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        num_heads: int = 32,
    ) -> None:
        """Load video and audio embeddings connectors from the LTX-2 checkpoint.

        Checkpoint keys:
        - text_embedding_projection.video_aggregate_embed.{weight,bias} → video connector aggregate_embed
        - text_embedding_projection.audio_aggregate_embed.{weight,bias} → audio connector aggregate_embed
        - model.diffusion_model.video_embeddings_connector.* → video connector blocks + norm
        - model.diffusion_model.audio_embeddings_connector.* → audio connector blocks + norm
        """
        from ...encoders.gemma.embeddings_connector import EmbeddingsConnector
        from ...parallel.config import EncoderParallelConfig

        input_dim = gemma_hidden_size * gemma_num_layers

        # Use same TP as DiT transformer
        tp_factor = self.parallel_config.tensor_parallel.factor
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        enc_parallel = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis))
        enc_ccl = CCLManager(self.mesh_device, topology=ttnn.Topology.Linear)

        # --- Video connector ---
        self.video_connector = EmbeddingsConnector(
            input_dim=input_dim,
            output_dim=video_dim,
            num_blocks=video_num_blocks,
            num_heads=num_heads,
            mesh_device=self.mesh_device,
            ccl_manager=enc_ccl,
            parallel_config=enc_parallel,
        )

        # Build state dict for video connector
        video_sd = {}
        # aggregate_embed from text_embedding_projection prefix
        agg_prefix = "text_embedding_projection.video_aggregate_embed."
        for k, v in checkpoint_state.items():
            if k.startswith(agg_prefix):
                video_sd["aggregate_embed." + k[len(agg_prefix) :]] = v

        # Connector blocks from model.diffusion_model prefix
        conn_prefix = "model.diffusion_model.video_embeddings_connector."
        for k, v in checkpoint_state.items():
            if k.startswith(conn_prefix):
                video_sd[k[len(conn_prefix) :]] = v

        result = self.video_connector.load_torch_state_dict(video_sd, strict=False)
        if result.missing_keys:
            logger.warning(f"Video connector missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Video connector unexpected keys: {result.unexpected_keys}")
        logger.info(f"Loaded video embeddings connector ({video_num_blocks} blocks, dim={video_dim})")

        # --- Audio connector ---
        if self.mode == "av":
            self.audio_connector = EmbeddingsConnector(
                input_dim=input_dim,
                output_dim=audio_dim,
                num_blocks=audio_num_blocks,
                num_heads=num_heads,
                mesh_device=self.mesh_device,
                ccl_manager=enc_ccl,
                parallel_config=enc_parallel,
            )

            audio_sd = {}
            agg_prefix = "text_embedding_projection.audio_aggregate_embed."
            for k, v in checkpoint_state.items():
                if k.startswith(agg_prefix):
                    audio_sd["aggregate_embed." + k[len(agg_prefix) :]] = v

            conn_prefix = "model.diffusion_model.audio_embeddings_connector."
            for k, v in checkpoint_state.items():
                if k.startswith(conn_prefix):
                    sub = k[len(conn_prefix) :]
                    # Filter out blocks beyond audio_num_blocks
                    if sub.startswith("transformer_1d_blocks."):
                        block_idx = int(sub.split(".")[1])
                        if block_idx >= audio_num_blocks:
                            continue
                    audio_sd[sub] = v

            result = self.audio_connector.load_torch_state_dict(audio_sd, strict=False)
            if result.missing_keys:
                logger.warning(f"Audio connector missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                logger.warning(f"Audio connector unexpected keys: {result.unexpected_keys}")
            logger.info(f"Loaded audio embeddings connector ({audio_num_blocks} blocks, dim={audio_dim})")

    @staticmethod
    def _norm_and_concat_per_token_rms(
        hidden_states: list[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token RMS normalization matching FeatureExtractorV2.

        Args:
            hidden_states: List of L tensors, each (B, T, D)
            attention_mask: (B, T) binary mask

        Returns:
            (B, T, D*L) normalized and flattened tensor with padding zeroed.
        """
        # Stack: [B, T, D, L]
        encoded = torch.stack(hidden_states, dim=-1)
        B, T, D, L = encoded.shape
        # Per-token RMS norm over D dimension per layer
        variance = torch.mean(encoded**2, dim=2, keepdim=True)  # [B,T,1,L]
        normed = encoded * torch.rsqrt(variance + 1e-6)
        normed = normed.reshape(B, T, D * L)
        # Zero out padding positions
        mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
        return torch.where(mask_3d, normed, torch.zeros_like(normed))

    @staticmethod
    def _replace_padded_with_registers(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        learnable_registers: torch.Tensor,
        num_registers: int,
    ) -> torch.Tensor:
        """Replace padded tokens with tiled learnable registers.

        Matching reference Embeddings1DConnector._replace_padded_with_learnable_registers:
        - Non-padded tokens are kept and left-packed
        - Remaining positions filled with tiled learnable registers
        """
        seq_len = hidden_states.shape[1]
        num_duplications = seq_len // num_registers
        registers = learnable_registers.repeat(num_duplications, 1)  # (seq_len, dim)

        # Binary mask: 1 = real token, 0 = padding
        mask_binary = attention_mask.bool()  # (B, T)

        result = hidden_states.clone()
        for b in range(hidden_states.shape[0]):
            real_tokens = hidden_states[b, mask_binary[b], :]  # (n_real, dim)
            n_real = real_tokens.shape[0]
            pad_length = seq_len - n_real
            # Pack real tokens first, then registers
            padded = torch.nn.functional.pad(real_tokens, (0, 0, 0, pad_length))
            # Flip: registers at the beginning (where attention_mask was 0 = left-padded)
            flipped_mask = torch.flip(mask_binary[b : b + 1], dims=[1]).squeeze(0).unsqueeze(-1).int()
            result[b] = flipped_mask.float() * padded + (1 - flipped_mask.float()) * registers.to(padded)

        return result

    @staticmethod
    def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
        """Rescale normalization: x * sqrt(target_dim / source_dim)."""
        return x * math.sqrt(target_dim / source_dim)

    def encode_prompts_device(self, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Encode prompts using TTNN Gemma encoder + embeddings connectors.

        Matches the reference FeatureExtractorV2 pipeline:
        1. Gemma forward → collect all 49 hidden states
        2. Stack as [B, T, D, L] → per-token RMS norm → flatten to [B, T, D*L]
        3. Rescale → aggregate_embed → connector blocks → final norm

        Returns list of (video_embeds, audio_embeds) tuples per prompt.
        """
        assert self.gemma_encoder is not None, "Call load_gemma_encoder() first"

        results = []
        for prompt in prompts:
            tokens = self.gemma_tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self._gemma_sequence_length,
                truncation=True,
            )

            tt_ids = ttnn.from_torch(
                tokens.input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            all_hidden_states = self.gemma_encoder(tt_ids, attention_mask=tokens.attention_mask)

            if self.video_connector is not None:
                # Collect 49 hidden states (embedding + 48 layers, skip final norm)
                # Concat on device to avoid 49 separate D2H transfers
                hs_list = all_hidden_states[:-1]
                tt_stacked = ttnn.concat(hs_list, dim=-1)  # (B, seq, D*L) on device
                stacked_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_stacked)[0]).float()
                # Free device memory
                for hs in all_hidden_states:
                    ttnn.deallocate(hs)
                ttnn.deallocate(tt_stacked)

                # FeatureExtractorV2: per-token RMS norm + flatten
                # Reshape to [B, T, D, L] for per-token per-layer normalization
                B_enc, T_enc = stacked_torch.shape[0], stacked_torch.shape[1]
                D_gemma = 3840  # Gemma hidden size
                L_layers = len(hs_list)  # 49
                encoded = stacked_torch.reshape(B_enc, T_enc, D_gemma, L_layers)
                variance = torch.mean(encoded**2, dim=2, keepdim=True)
                normed = encoded * torch.rsqrt(variance + 1e-6)
                normed = normed.reshape(B_enc, T_enc, D_gemma * L_layers)
                mask_3d = tokens.attention_mask.bool().unsqueeze(-1)
                normed = torch.where(mask_3d, normed, torch.zeros_like(normed)).bfloat16()

                # Rescale and run through connectors
                # Reference FeatureExtractorV2 uses Gemma hidden_size (3840) as embedding_dim
                embedding_dim = D_gemma  # 3840 (NOT D*L=188160)

                def _run_connector(connector, normed_features, attn_mask):
                    """Run rescale → aggregate_embed → register replacement → RoPE blocks → norm."""
                    dim = connector.output_dim
                    rescaled = self._rescale_norm(normed_features.float(), dim, embedding_dim).bfloat16()

                    # Run aggregate_embed on device
                    tt_input = ttnn.from_torch(
                        rescaled,
                        device=self.mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                    )
                    tt_projected = connector.aggregate_embed(tt_input)
                    projected = ttnn.to_torch(ttnn.get_device_tensors(tt_projected)[0])

                    # Replace padded tokens with learnable registers (on host, matching reference)
                    if connector.num_learnable_registers > 0:
                        registers = ttnn.to_torch(ttnn.get_device_tensors(connector.learnable_registers.data)[0])
                        projected = self._replace_padded_with_registers(
                            projected,
                            attn_mask,
                            registers,
                            connector.num_learnable_registers,
                        )

                    # Compute 1D RoPE for connector blocks (matching reference Embeddings1DConnector).
                    # Uses reference precompute_freqs_cis directly since 1D grid format differs from
                    # our video/audio 4D grid convention.
                    import sys

                    sys.path.insert(0, "LTX-2/packages/ltx-core/src")
                    from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType
                    from ltx_core.model.transformer.rope import generate_freq_grid_pytorch
                    from ltx_core.model.transformer.rope import precompute_freqs_cis as ref_precompute

                    seq_len = projected.shape[1]
                    num_heads = connector.transformer_1d_blocks[0].num_heads
                    indices_grid = torch.arange(seq_len, dtype=torch.float32)[None, None, :]
                    rope_cos, rope_sin = ref_precompute(
                        indices_grid,
                        dim=dim,
                        out_dtype=torch.bfloat16,
                        theta=10000.0,
                        max_pos=[1],
                        num_attention_heads=num_heads,
                        rope_type=RefRopeType.INTERLEAVED,
                        freq_grid_generator=generate_freq_grid_pytorch,
                    )  # INTERLEAVED: (1, seq_len, dim) — applied to flat Q/K before head split

                    # Push back to device and run transformer blocks with RoPE
                    tt_x = ttnn.from_torch(
                        projected.bfloat16(),
                        device=self.mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                    )
                    for block in connector.transformer_1d_blocks:
                        tt_x = block(tt_x, rope_cos=rope_cos, rope_sin=rope_sin)

                    # Final parameter-free RMS norm
                    from ...encoders.gemma.embeddings_connector import _rms_norm

                    tt_x = _rms_norm(tt_x)
                    result = ttnn.to_torch(ttnn.get_device_tensors(tt_x)[0]).float()

                    # NOTE: Do NOT zero out register positions here. The reference
                    # FeatureExtractorV2 replaces padding with learnable registers
                    # and then sets attention_mask to all-zeros (= no masking), so
                    # all 1024 tokens (real + register) carry information after the
                    # connector blocks.
                    return result

                video_embeds = _run_connector(self.video_connector, normed, tokens.attention_mask)

                audio_embeds = None
                if self.audio_connector is not None:
                    audio_embeds = _run_connector(self.audio_connector, normed, tokens.attention_mask)

                results.append((video_embeds, audio_embeds))
            else:
                # Fallback: return raw hidden states from specified layer
                hs = all_hidden_states[self._gemma_hidden_layer_index]
                hs_torch = ttnn.to_torch(ttnn.get_device_tensors(hs)[0]).float()
                mask = tokens.attention_mask.unsqueeze(-1).float()
                hs_torch = hs_torch * mask
                results.append((hs_torch, None))

        return results

    def encode_prompts_reference(self, prompts: list[str]) -> list:
        """Encode prompts using the official LTX-2 reference pipeline (recommended for AV mode)."""
        assert self.checkpoint_name is not None, "checkpoint_name must be set before encode_prompts_reference"
        assert self.gemma_path is not None, "gemma_path must be set before encode_prompts_reference"
        try:
            import sys

            sys.path.insert(0, "LTX-2/packages/ltx-core/src")
            sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
            torch.cuda.synchronize = lambda *a, **kw: None  # No CUDA on TT host
            from ltx_pipelines.utils.blocks import PromptEncoder
        except ImportError as e:
            raise ImportError(
                "encode_prompts_reference() requires the LTX-2 reference package. "
                "Use load_text_encoder() + __call__() for standalone text encoding."
            ) from e

        # Check embedding cache to skip expensive Gemma encoding
        import hashlib

        cache_dir = os.environ.get("TT_DIT_CACHE_DIR") or os.path.expanduser("~/.cache/tt-dit")
        embed_cache_dir = os.path.join(cache_dir, "ltx-embeddings")
        os.makedirs(embed_cache_dir, exist_ok=True)

        cache_key = hashlib.md5("||".join(prompts).encode()).hexdigest()
        cache_path = os.path.join(embed_cache_dir, f"{cache_key}.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}")
            return torch.load(cache_path, weights_only=False)

        # PromptEncoder owns Gemma text encoder + embeddings processor lifecycle:
        # builds Gemma, encodes, frees, then builds the embeddings processor.
        prompt_encoder = PromptEncoder(
            checkpoint_path=self.checkpoint_name,
            gemma_root=self.gemma_path,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        results = prompt_encoder(prompts)
        del prompt_encoder

        torch.save(results, cache_path)
        logger.info(f"Cached embeddings to {cache_path}")
        return results

    def load_vae_decoder(
        self,
        state_dict: dict[str, torch.Tensor],
        decoder_blocks: list[tuple[str, dict]],
        *,
        use_ttnn: bool = True,
        patch_size: int = 4,
        base_channels: int = 128,
    ) -> None:
        """Load VAE decoder weights.

        Args:
            state_dict: PyTorch state dict for the decoder
            decoder_blocks: Block configuration list
            use_ttnn: If True, use TTNN decoder; if False, use torch-only wrapper
        """
        if use_ttnn:
            from ...models.vae.vae_ltx import LTXVideoDecoder

            self.vae_decoder = LTXVideoDecoder(
                decoder_blocks=decoder_blocks,
                in_channels=self.in_channels,
                out_channels=3,
                patch_size=patch_size,
                base_channels=base_channels,
                mesh_device=self.mesh_device,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.ccl_manager,
            )
            self.vae_decoder.load_torch_state_dict(state_dict)
            logger.info("Loaded TTNN VAE decoder")
        else:
            from ...models.vae.vae_ltx import LTXVideoDecoderTorch

            self.vae_decoder = LTXVideoDecoderTorch.from_config(
                decoder_blocks, in_channels=self.in_channels, patch_size=patch_size, base_channels=base_channels
            )
            self.vae_decoder.load_state_dict(state_dict)
            logger.info("Loaded torch-only VAE decoder")

    def _prepare_vae(
        self,
        *,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Push VAE decoder onto the mesh. Cached when `TT_DIT_CACHE_DIR` is set.

        Pass ``num_frames``, ``height``, ``width`` so the decoder can pre-compute
        per-layer ConvDims and pick exact-match blockings from ``_BLOCKINGS``.
        Omit them to fall back to channel-only blocking lookup.
        """
        if not self._vae_decoder_blocks:
            return

        from ...models.vae.vae_ltx import LTXVideoDecoder
        from ...utils.conv3d import conv3d_blocking_hash

        self.vae_decoder = LTXVideoDecoder(
            decoder_blocks=self._vae_decoder_blocks,
            causal=self._vae_causal,
            base_channels=self._vae_base_channels,
            mesh_device=self.mesh_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.ccl_manager,
            num_frames=num_frames,
            height=height,
            width=width,
        )

        def _vae_state_provider() -> dict[str, torch.Tensor]:
            from safetensors.torch import load_file

            logger.info(f"VAE cache miss — loading safetensors: {self._vae_checkpoint_path}")
            raw = load_file(self._vae_checkpoint_path)
            vae_state = {}
            for k, v in raw.items():
                if k.startswith("vae.decoder."):
                    vae_state[k[len("vae.decoder.") :]] = v
                elif k.startswith("vae.per_channel_statistics."):
                    short_key = k[len("vae.") :]
                    if short_key in ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"):
                        vae_state[short_key] = v
            return vae_state

        if self.checkpoint_name is not None:
            # Blocking-hash subfolder mirrors Wan: when conv3d C_in_block changes
            # (e.g. blockings refined from a sweep), prepare_conv3d_weights re-runs
            # automatically rather than silently loading mis-shaped weights.
            blocking_key = conv3d_blocking_hash(self.vae_decoder)
            subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
            cache_module.load_model(
                self.vae_decoder,
                model_name=os.path.basename(self.checkpoint_name).removesuffix(".safetensors"),
                subfolder=subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                get_torch_state_dict=_vae_state_provider,
            )
        else:
            self.vae_decoder.load_torch_state_dict(_vae_state_provider())

        logger.info(f"Loaded TTNN VAE decoder ({len(self._vae_decoder_blocks)} blocks)")

    def _prepare_connectors(self) -> None:
        """Load embeddings connectors. Call after Gemma encoder is loaded."""
        if not hasattr(self, "_connector_checkpoint_path"):
            logger.warning("No connector checkpoint path saved — skipping connector loading")
            return
        from safetensors.torch import load_file

        raw = load_file(self._connector_checkpoint_path)
        self.load_embeddings_connectors(raw)
        del raw

    def decode_latents(self, latent: torch.Tensor, latent_frames: int, latent_h: int, latent_w: int) -> torch.Tensor:
        """Decode latent tensor to video pixels.

        Args:
            latent: (B, num_tokens, C) flat latent from denoising loop
            latent_frames, latent_h, latent_w: Spatial dimensions

        Returns:
            (B, 3, F, H, W) decoded video
        """
        if self.vae_decoder is None:
            logger.warning("No VAE decoder loaded, returning raw latent")
            return latent

        B = latent.shape[0]
        # Reshape flat tokens to spatial: (B, num_tokens, C) -> (B, C, F', H', W')
        latent_spatial = latent.reshape(B, latent_frames, latent_h, latent_w, self.in_channels)
        latent_spatial = latent_spatial.permute(0, 4, 1, 2, 3)  # BCTHW

        from ...models.vae.vae_ltx import LTXVideoDecoder

        if isinstance(self.vae_decoder, LTXVideoDecoder):
            return self.vae_decoder(latent_spatial)
        else:
            return self.vae_decoder.decode(latent_spatial)

    def _prepare_rope(
        self, num_frames: int, latent_height: int, latent_width: int, fps: float = 24.0
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute video RoPE using reference SPLIT rotation with pixel-space coordinates.

        Uses the official LTX-2 VideoLatentPatchifier for positions, matching the reference
        pipeline's precompute_freqs_cis with SPLIT rotation. No trans_mat needed.
        """
        from ...models.transformers.ltx.rope_ltx import precompute_freqs_cis
        from ...utils.ltx import VideoLatentShape, get_pixel_coords, video_get_patch_grid_bounds

        v_shape = VideoLatentShape(batch=1, channels=128, frames=num_frames, height=latent_height, width=latent_width)
        v_coords = video_get_patch_grid_bounds(v_shape)
        v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
        v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps

        from ...models.transformers.ltx.rope_ltx import LTXRopeType

        # NOTE: positions MUST stay fp32 (was .bfloat16() - introduced catastrophic phase error in
        # high-frequency RoPE channels: 1700x worse than fp32, completely randomizing cos/sin in the
        # top half of head_dim).
        cos_freq, sin_freq = precompute_freqs_cis(
            v_positions,
            dim=self.inner_dim,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.num_attention_heads,
            rope_type=LTXRopeType.SPLIT,
        )  # (1, num_heads, N, D_half)

        # SP padding: pad sequence dim to ttnn.TILE_SIZE * sp_factor so ring SDPA's
        # N_local % TILE_HEIGHT == 0 and N_global == N_local * ring_size checks pass.
        # Padded slots use cos=1, sin=0 (identity rotation) — matches the audio path
        # convention so padded Q/K stay well-defined; SDPA still masks them via logical_n.
        cos_freq, sin_freq = self._pad_video_rope_sp(cos_freq, sin_freq)

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        tt_cos = bf16_tensor_2dshard(cos_freq, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_sin = bf16_tensor_2dshard(sin_freq, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

        self._cached_rope_cos = tt_cos
        self._cached_rope_sin = tt_sin

        return tt_cos, tt_sin

    def _prepare_prompt(self, prompt_embeds: torch.Tensor) -> ttnn.Tensor:
        """Push prompt embeddings to device, padding to cross_attention_dim if needed.

        When the transformer has a caption_projection (19B distilled), skip padding —
        the projection handles the dimension mapping (3840→4096/2048) inside inner_step.
        """
        # (B, L, D) -> (1, B, L, D)
        prompt = prompt_embeds.unsqueeze(0)
        # Only pad/truncate if no caption projection (22B model outputs at cross_attention_dim directly)
        has_caption_proj = (
            self.transformer is not None
            and hasattr(self.transformer, "_caption_proj_state")
            and self.transformer._caption_proj_state
        )
        if not has_caption_proj:
            if prompt.shape[-1] < self.cross_attention_dim:
                pad_size = self.cross_attention_dim - prompt.shape[-1]
                prompt = torch.nn.functional.pad(prompt, (0, pad_size))
            elif prompt.shape[-1] > self.cross_attention_dim:
                prompt = prompt[..., : self.cross_attention_dim]
        tt_prompt = bf16_tensor(prompt, device=self.mesh_device)
        return tt_prompt

    def __call__(
        self,
        prompt: str | list[str],
        *,
        negative_prompt: str | list[str] | None = None,
        num_frames: int = 33,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        # Scheduler params
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        # Latent space params (LTX uses 8x temporal, 32x spatial compression)
        temporal_compression: int = 8,
        spatial_compression: int = 32,
    ) -> torch.Tensor:
        """
        Run the full text-to-video generation pipeline.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            num_frames: Number of output video frames
            height: Output video height in pixels
            width: Output video width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance strength (1.0 = no guidance)
            seed: Random seed for reproducibility

        Returns:
            Denoised latent tensor of shape (B, num_tokens, out_channels)
        """
        assert self.transformer is not None, "Call load_transformer() first"

        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)

        # Compute latent dimensions
        latent_frames = (num_frames - 1) // temporal_compression + 1
        latent_height = height // spatial_compression
        latent_width = width // spatial_compression
        video_N_real = latent_frames * latent_height * latent_width
        # SP padding: round seq dim up to TILE_SIZE * sp_factor so ring SDPA's
        # N_local % TILE_HEIGHT == 0 and N_global == N_local * ring_size checks pass.
        video_N = self._video_sp_pad_len(video_N_real)
        # Used by compute_sigmas / scheduler shift — keep based on logical (real) token count.
        num_tokens = video_N_real

        logger.info(
            f"Generating: {num_frames} frames @ {height}x{width}, "
            f"latent: {latent_frames}x{latent_height}x{latent_width} = {video_N_real} tokens"
            + (f" (SP-padded to {video_N})" if video_N > video_N_real else "")
        )

        # 1. Encode text
        do_cfg = guidance_scale > 1.0
        if self.text_encoder is not None:
            prompt_embeds = self.text_encoder.encode(prompt)
        else:
            # Fallback: zero embeddings
            prompt_embeds = torch.zeros(B, 256, self.cross_attention_dim)

        if do_cfg:
            if self.text_encoder is not None and negative_prompt is not None:
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * B
                negative_embeds = self.text_encoder.encode(negative_prompt)
            else:
                negative_embeds = torch.zeros_like(prompt_embeds)

        # Push prompts to device
        tt_prompt = self._prepare_prompt(prompt_embeds)
        tt_negative_prompt = self._prepare_prompt(negative_embeds) if do_cfg else None

        # 2. Prepare RoPE
        rope_cos, rope_sin = self._prepare_rope(latent_frames, latent_height, latent_width)

        # 3. Compute sigma schedule
        sigmas = compute_sigmas(
            steps=num_inference_steps,
            num_tokens=num_tokens,
            max_shift=max_shift,
            base_shift=base_shift,
        )
        logger.info(f"Sigmas: {sigmas[0]:.4f} -> {sigmas[-1]:.4f} ({len(sigmas)} values)")

        # 4. Prepare initial noise
        if seed is not None:
            torch.manual_seed(seed)
        # Generate noise at logical seq len, then zero-pad to video_N on dim=1 for SP.
        latent_real = torch.randn(B, video_N_real, self.in_channels, dtype=torch.float32)
        latent_real = latent_real * sigmas[0]
        if video_N > video_N_real:
            latent = torch.zeros(B, video_N, self.in_channels, dtype=torch.float32)
            latent[:, :video_N_real, :] = latent_real
        else:
            latent = latent_real

        # 5. Denoising loop
        for step_idx in range(num_inference_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            # Prepare spatial input: (1, B, video_N, in_channels) — already SP-padded.
            spatial_torch = latent.unsqueeze(0)
            timestep_torch = torch.tensor([sigma])

            # Forward pass (conditioned). video_N is the LOGICAL (unpadded) count and
            # is forwarded as ``logical_n`` to ring SDPA so padded K positions get masked.
            tt_denoised = self.transformer.inner_step(
                video_1BNI_torch=spatial_torch,
                video_prompt_1BLP=tt_prompt,
                video_rope_cos=rope_cos,
                video_rope_sin=rope_sin,
                trans_mat=None,
                video_N=video_N_real,
                timestep_torch=timestep_torch,
            )
            # Model output is velocity (shape (B, video_N, C)). Zero padded slots so the
            # latent stays clean and Euler updates don't drift in the padded region.
            velocity = LTXTransformerModel.device_to_host(tt_denoised).squeeze(0)
            velocity = self._zero_video_padding(velocity, video_N_real)
            denoised = latent.float() - velocity.float() * sigma

            # CFG
            if do_cfg:
                tt_uncond = self.transformer.inner_step(
                    video_1BNI_torch=spatial_torch,
                    video_prompt_1BLP=tt_negative_prompt,
                    video_rope_cos=rope_cos,
                    video_rope_sin=rope_sin,
                    trans_mat=None,
                    video_N=video_N_real,
                    timestep_torch=timestep_torch,
                )
                uncond_velocity = LTXTransformerModel.device_to_host(tt_uncond).squeeze(0)
                uncond_velocity = self._zero_video_padding(uncond_velocity, video_N_real)
                uncond = latent.float() - uncond_velocity.float() * sigma

                # guidance = uncond + scale * (cond - uncond)
                denoised = uncond + guidance_scale * (denoised - uncond)

            # Euler step
            latent = euler_step(latent, denoised, sigma, sigma_next)
            # Re-zero padded slots (cheap insurance — Euler with zeroed velocity should
            # leave the zeros in place, but bf16 round-trips can introduce tiny drift).
            latent = self._zero_video_padding(latent, video_N_real)

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                logger.info(
                    f"Step {step_idx + 1}/{num_inference_steps}: "
                    f"sigma {sigma:.4f} -> {sigma_next:.4f}, "
                    f"latent range [{latent.min():.3f}, {latent.max():.3f}]"
                )

        # Slice out SP padding before VAE decode.
        latent = latent[:, :video_N_real, :]
        logger.info(f"Denoising complete. Output latent shape: {latent.shape}")

        # Optionally decode latents to video
        if self.vae_decoder is not None:
            video = self.decode_latents(latent, latent_frames, latent_height, latent_width)
            logger.info(f"Decoded video shape: {video.shape}")
            return video

        return latent

    def _prepare_audio_rope(self, audio_N: int, audio_N_real: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute audio RoPE using AudioPatchifier time-in-seconds positions with SPLIT rotation."""

        from ...models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
        from ...utils.ltx import AudioLatentShape, audio_get_patch_grid_bounds

        a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
        a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, N, 2)

        # NOTE: positions MUST stay fp32 (was .bfloat16() - introduced catastrophic phase error in
        # high-frequency RoPE channels: 1700x worse than fp32 for audio, completely randomizing
        # cos/sin in the top half of head_dim. HF reference uses fp32/fp64 throughout.
        a_cos, a_sin = precompute_freqs_cis(
            a_positions,
            dim=2048,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=[20],  # 1D temporal only
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
        )  # (1, 32, audio_N_real, D_half)

        # Pad to audio_N if needed
        if audio_N > audio_N_real:
            d_half = a_cos.shape[-1]
            a_cos_padded = torch.ones(1, 32, audio_N, d_half)
            a_cos_padded[:, :, :audio_N_real, :] = a_cos
            a_sin_padded = torch.zeros(1, 32, audio_N, d_half)
            a_sin_padded[:, :, :audio_N_real, :] = a_sin
            a_cos, a_sin = a_cos_padded, a_sin_padded

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        return (
            bf16_tensor_2dshard(a_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
            bf16_tensor_2dshard(a_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
        )

    def _prepare_av_cross_pe(
        self,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        audio_N: int,
        audio_N_real: int,
        fps: float = 24.0,
        cross_pe_max_pos: int = 20,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Compute temporal-only cross positional embeddings for A↔V cross-attention.

        Reference: ``MultiModalTransformerArgsPreprocessor.prepare`` builds ``cross_pe`` from
        ``modality.positions[:, 0:1, :]`` (temporal slice only) at ``dim=audio_cross_attention_dim``
        with ``max_pos=[cross_pe_max_pos]``. Both video and audio share this scheme so that audio
        token at time t and video tokens at time t share the same rotary phase — this is what
        ties footstep onsets, lip movement, and other AV alignment cues.

        Returns 8 device tensors used by inner_step:
            (v_q_cos, v_q_sin)         — video Q in A→V cross-attn (SP×TP sharded).
            (a_q_cos, a_q_sin)         — audio Q in V→A cross-attn (SP×TP sharded).
            (v_k_cos, v_k_sin)         — video K in V→A cross-attn (TP-only; K side after AllGather).
            (a_k_cos, a_k_sin)         — audio K in A→V cross-attn (TP-only; K side after AllGather).
        """
        from ...models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
        from ...utils.ltx import (
            AudioLatentShape,
            VideoLatentShape,
            audio_get_patch_grid_bounds,
            get_pixel_coords,
            video_get_patch_grid_bounds,
        )

        v_shape = VideoLatentShape(
            batch=1, channels=128, frames=latent_frames, height=latent_height, width=latent_width
        )
        v_coords = video_get_patch_grid_bounds(v_shape)
        v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
        v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps  # temporal axis → seconds
        v_temporal = v_positions[:, 0:1, :]  # (1, 1, video_N, 2)

        a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
        a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, audio_N_real, 2)

        rope_kwargs = dict(
            dim=2048,  # audio_cross_attention_dim — both sides share this
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=[cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
        )

        # NOTE: positions MUST stay fp32 (was .bfloat16() - introduced catastrophic phase error in
        # high-frequency RoPE channels: 1700x worse than fp32, completely randomizing cos/sin in the
        # top half of head_dim).
        v_cos, v_sin = precompute_freqs_cis(v_temporal, **rope_kwargs)  # (1, 32, video_N, 32)
        a_cos, a_sin = precompute_freqs_cis(a_positions, **rope_kwargs)  # (1, 32, audio_N_real, 32)

        # Pad video cross-PE to the same SP boundary used by the main video RoPE
        # (cos=1, sin=0 in padded slots — identity rotation).
        v_cos, v_sin = self._pad_video_rope_sp(v_cos, v_sin)

        # Pad audio cross-PE to audio_N (matching the audio RoPE padding scheme: cos=1, sin=0).
        if audio_N > audio_N_real:
            d_half = a_cos.shape[-1]
            a_cos_padded = torch.ones(1, 32, audio_N, d_half)
            a_cos_padded[:, :, :audio_N_real, :] = a_cos
            a_sin_padded = torch.zeros(1, 32, audio_N, d_half)
            a_sin_padded[:, :, :audio_N_real, :] = a_sin
            a_cos, a_sin = a_cos_padded, a_sin_padded

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # Q-side: SP×TP sharded (matches the Q tensor layout post-attention QKV split).
        v_q_cos = bf16_tensor_2dshard(v_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        v_q_sin = bf16_tensor_2dshard(v_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        a_q_cos = bf16_tensor_2dshard(a_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        a_q_sin = bf16_tensor_2dshard(a_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

        # K-side: TP-only on heads (sequence is replicated after AllGather on K).
        v_k_cos = bf16_tensor(v_cos, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)
        v_k_sin = bf16_tensor(v_sin, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)
        a_k_cos = bf16_tensor(a_cos, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)
        a_k_sin = bf16_tensor(a_sin, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)

        return (v_q_cos, v_q_sin, a_q_cos, a_q_sin, v_k_cos, v_k_sin, a_k_cos, a_k_sin)

    @staticmethod
    def _zero_audio_padding(t: torch.Tensor, audio_N_real: int) -> torch.Tensor:
        """Zero SP-padded audio token slots so they do not affect guidance or GE."""
        if t.shape[1] <= audio_N_real:
            return t
        out = t.clone()
        out[:, audio_N_real:, :] = 0.0
        return out

    @staticmethod
    def _zero_video_padding(t: torch.Tensor, video_N_real: int) -> torch.Tensor:
        """Zero SP-padded video token slots (mirrors _zero_audio_padding)."""
        if t.shape[1] <= video_N_real:
            return t
        out = t.clone()
        out[:, video_N_real:, :] = 0.0
        return out

    def _video_sp_pad_len(self, video_N_real: int) -> int:
        """Round video seq len up to ttnn.TILE_SIZE * sp_factor (ring-SDPA / SP requirement)."""
        sp_factor = self.parallel_config.sequence_parallel.factor
        divisor = ttnn.TILE_SIZE * sp_factor
        return ((video_N_real + divisor - 1) // divisor) * divisor

    def _pad_video_rope_sp(self, cos_freq: torch.Tensor, sin_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad video RoPE cos/sin on dim=2 to the SP boundary.

        Padded slots use cos=1, sin=0 (identity rotation). Same convention as the audio
        RoPE padding in ``_prepare_audio_rope`` / ``_prepare_av_cross_pe``.
        """
        video_N_real = cos_freq.shape[2]
        video_N = self._video_sp_pad_len(video_N_real)
        if video_N == video_N_real:
            return cos_freq, sin_freq
        pad = video_N - video_N_real
        H = cos_freq.shape[1]
        d_half = cos_freq.shape[-1]
        cos_pad = torch.ones(1, H, pad, d_half, dtype=cos_freq.dtype)
        sin_pad = torch.zeros(1, H, pad, d_half, dtype=sin_freq.dtype)
        cos_freq = torch.cat([cos_freq, cos_pad], dim=2)
        sin_freq = torch.cat([sin_freq, sin_pad], dim=2)
        return cos_freq, sin_freq

    @staticmethod
    def _apply_modal_guidance(
        den: torch.Tensor,
        unc,
        ptb,
        iso,
        *,
        cfg_scale: float,
        stg_scale: float,
        modality_scale: float,
        rescale_scale: float,
        do_cfg: bool,
        do_stg: bool,
        do_mod: bool,
        real_token_count: int | None = None,
    ) -> torch.Tensor:
        """CFG + STG + modality guidance with optional per-modality token slice (audio pad)."""
        if real_token_count is not None:
            den_s = den[:, :real_token_count, :]
        else:
            den_s = den

        pred = den_s.float()
        c = den_s.float()
        if do_cfg and isinstance(unc, torch.Tensor):
            unc_s = unc[:, :real_token_count, :] if real_token_count is not None else unc
            pred = pred + (cfg_scale - 1) * (c - unc_s.float())
        if do_stg and isinstance(ptb, torch.Tensor):
            ptb_s = ptb[:, :real_token_count, :] if real_token_count is not None else ptb
            pred = pred + stg_scale * (c - ptb_s.float())
        if do_mod and isinstance(iso, torch.Tensor):
            iso_s = iso[:, :real_token_count, :] if real_token_count is not None else iso
            pred = pred + (modality_scale - 1) * (c - iso_s.float())
        if rescale_scale != 0:
            pred = pred * (rescale_scale * (c.std() / pred.std()) + (1 - rescale_scale))

        if real_token_count is not None:
            out = den.clone()
            out[:, :real_token_count, :] = pred.bfloat16()
            return out
        return pred.bfloat16()

    def _prepare_audio_masks(self, audio_N: int, audio_N_real: int) -> tuple:
        """Create SDPA attn mask and padding masks for SP-sharded vs gathered audio.

        Returns (attn_mask, pad_mask_sp, pad_mask_full). pad_mask_sp is sharded on the
        sequence dimension for multiply with local audio activations; pad_mask_full is
        replicated for multiply after all_gather on A-to-V keys.
        """
        if audio_N <= audio_N_real:
            return None, None, None

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        # Column mask only: real/padded queries are barred from attending TO padded keys.
        # Do NOT mask padded-query rows to -inf — that makes all attention scores in those
        # rows -inf → softmax NaN → NaN propagates via padded-token outputs (which we then
        # multiply by 0; IEEE 0*NaN = NaN, not 0). audio_padding_mask already zeros the
        # padded-query outputs after attention, so column-only masking is sufficient and
        # numerically safer at high σ where activations have largest magnitude.
        mask = torch.zeros(1, 1, audio_N, audio_N)
        mask[:, :, :, audio_N_real:] = float("-inf")
        mask = mask.to(torch.bfloat16)
        tt_attn_mask = bf16_tensor(
            mask,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )

        pad_mask = torch.ones(1, 1, audio_N, 1, dtype=torch.bfloat16)
        pad_mask[:, :, audio_N_real:, :] = 0.0
        tt_pad_mask_sp = bf16_tensor(
            pad_mask,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )
        tt_pad_mask_full = bf16_tensor(pad_mask, device=self.mesh_device)
        return tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full

    def _prepare_video_masks(self, video_N: int, video_N_real: int) -> tuple:
        """Create SP-sharded + replicated padding masks for video.

        Returns ``(pad_mask_sp, pad_mask_full)``:
            * ``pad_mask_sp``  — shape (1, 1, video_N, 1), SP-sharded on dim 2. Multiply
              the local (sharded) video activations by this to zero padded slots before
              they propagate downstream (self-attn residual / cross-attn K / FF).
            * ``pad_mask_full`` — replicated copy, used after the V→A AllGather to
              re-mask gathered K so other chips' padded slots also get zeroed.

        Returns ``(None, None)`` when no padding is needed.

        No SDPA attn_mask is returned (unlike audio) — video self-attention uses
        ring SDPA which masks padded keys via the ``logical_n=video_N_real`` arg.
        """
        if video_N <= video_N_real:
            return None, None

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        pad_mask = torch.ones(1, 1, video_N, 1, dtype=torch.bfloat16)
        pad_mask[:, :, video_N_real:, :] = 0.0
        tt_pad_mask_sp = bf16_tensor(
            pad_mask,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )
        tt_pad_mask_full = bf16_tensor(pad_mask, device=self.mesh_device)
        return tt_pad_mask_sp, tt_pad_mask_full

    @torch.no_grad()
    def call_av(
        self,
        video_prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        neg_video_prompt_embeds: torch.Tensor | None = None,
        neg_audio_prompt_embeds: torch.Tensor | None = None,
        num_frames: int = 33,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        seed: int | None = None,
        ge_gamma: float = 2.0,
        profiler=None,
        profiler_iteration: int = 0,
        sigmas: torch.Tensor | None = None,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run AV denoising with full MultiModalGuider guidance. Returns (video_latent, audio_latent).

        Stage-2 / refine usage: pass ``sigmas`` (the explicit schedule), the
        upsampled ``initial_video_latent`` plus the stage-1 ``initial_audio_latent``,
        and ``noise_scale = sigmas[0]`` to renoise. Set all guidance scales to
        their neutral values (``cfg=1.0, stg=0.0, mod=1.0, ge_gamma=0.0``).
        """
        from ...utils.ltx import AudioLatentShape, VideoPixelShape

        B = 1
        latent_frames = (num_frames - 1) // 8 + 1
        latent_h, latent_w = height // 32, width // 32
        video_N_real = latent_frames * latent_h * latent_w
        # SP padding: round seq dim up to TILE_SIZE * sp_factor so ring SDPA's
        # N_local % TILE_HEIGHT == 0 and N_global == N_local * ring_size checks pass.
        video_N = self._video_sp_pad_len(video_N_real)

        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        sp_factor = self.parallel_config.sequence_parallel.factor
        audio_N = ((audio_N_real + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)

        logger.info(
            f"AV: {num_frames}f@{height}x{width}, "
            f"vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real})"
        )

        v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
        a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
        # Cross-modal positional embeddings for A↔V cross-attention. Without these, audio queries
        # attend to video keys with a uniform (no-RoPE) phase — destroying temporal sync. Reference
        # MultiModalTransformerArgsPreprocessor builds these from the temporal-only column at
        # dim=audio_cross_attention_dim with max_pos=[cross_pe_max_pos]. See test_av_model_pcc_vs_reference.
        (
            v_xpe_cos,
            v_xpe_sin,
            a_xpe_cos,
            a_xpe_sin,
            v_xpe_cos_full,
            v_xpe_sin_full,
            a_xpe_cos_full,
            a_xpe_sin_full,
        ) = self._prepare_av_cross_pe(latent_frames, latent_h, latent_w, audio_N, audio_N_real)
        tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = self._prepare_audio_masks(audio_N, audio_N_real)
        tt_v_pad_mask_sp, tt_v_pad_mask_full = self._prepare_video_masks(video_N, video_N_real)

        tt_vp = self._prepare_prompt(video_prompt_embeds)
        tt_ap = bf16_tensor(audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
        tt_nv = self._prepare_prompt(neg_video_prompt_embeds) if neg_video_prompt_embeds is not None else None
        tt_na = (
            bf16_tensor(neg_audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
            if neg_audio_prompt_embeds is not None
            else None
        )

        if sigmas is None:
            # Scheduler shift is keyed on the logical token count, not padded.
            sigmas = compute_sigmas(steps=num_inference_steps, num_tokens=video_N_real + audio_N_real)
        else:
            assert (
                len(sigmas) == num_inference_steps + 1
            ), f"sigmas length {len(sigmas)} must equal num_inference_steps+1 ({num_inference_steps+1})"

        if initial_video_latent is not None:
            assert noise_scale is not None, "noise_scale required when initial_video_latent is provided"
            if seed is not None:
                torch.manual_seed(seed)
            init_v = initial_video_latent.float()
            if init_v.dim() == 2:
                init_v = init_v.unsqueeze(0)
            noise_v = torch.randn_like(init_v)
            video_lat_real = init_v * (1.0 - noise_scale) + noise_v * noise_scale
        else:
            if seed is not None:
                torch.manual_seed(seed)
            video_lat_real = torch.randn(B, video_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]

        # Zero-pad video latent on dim=1 to video_N for SP sharding.
        if video_N > video_N_real:
            video_lat = torch.zeros(B, video_N, self.in_channels)
            video_lat[:, :video_N_real, :] = video_lat_real
        else:
            video_lat = video_lat_real

        if initial_audio_latent is not None:
            assert noise_scale is not None, "noise_scale required when initial_audio_latent is provided"
            if seed is not None:
                torch.manual_seed(seed + 1)
            init_a = initial_audio_latent.float()
            if init_a.dim() == 2:
                init_a = init_a.unsqueeze(0)
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = init_a[:, :audio_N_real, :]
            noise_a = torch.randn_like(audio_lat)
            audio_lat = audio_lat * (1.0 - noise_scale) + noise_a * noise_scale
        else:
            audio_lat_real = torch.randn(B, audio_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = audio_lat_real

        do_cfg = video_cfg_scale > 1.0 or audio_cfg_scale > 1.0
        do_stg = video_stg_scale != 0.0 or audio_stg_scale != 0.0
        do_mod = video_modality_scale != 1.0 or audio_modality_scale != 1.0

        # Gradient estimation state (tracks velocity for velocity correction)
        prev_v_vel = None
        prev_a_vel = None

        for step_idx in range(num_inference_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            def _run(vp, ap, skip_ca=False, skip_sa_blocks=None):
                # video_N / audio_N here are LOGICAL (unpadded) — passed as ``logical_n``
                # to ring SDPA so padded K positions get masked. Tensor shapes themselves
                # carry the padded counts.
                v, a = self.transformer.inner_step(
                    video_1BNI_torch=video_lat.unsqueeze(0),
                    video_prompt_1BLP=vp,
                    video_rope_cos=v_cos,
                    video_rope_sin=v_sin,
                    video_N=video_N_real,
                    audio_1BNI_torch=audio_lat.unsqueeze(0),
                    audio_prompt_1BLP=ap,
                    audio_rope_cos=a_cos,
                    audio_rope_sin=a_sin,
                    audio_N=audio_N,
                    trans_mat=None,
                    timestep_torch=torch.tensor([sigma]),
                    video_cross_pe_cos=v_xpe_cos,
                    video_cross_pe_sin=v_xpe_sin,
                    audio_cross_pe_cos=a_xpe_cos,
                    audio_cross_pe_sin=a_xpe_sin,
                    video_cross_pe_cos_full=v_xpe_cos_full,
                    video_cross_pe_sin_full=v_xpe_sin_full,
                    audio_cross_pe_cos_full=a_xpe_cos_full,
                    audio_cross_pe_sin_full=a_xpe_sin_full,
                    skip_cross_attn=skip_ca,
                    skip_self_attn_blocks=skip_sa_blocks,
                    audio_attn_mask=tt_attn_mask,
                    audio_padding_mask=tt_pad_mask_sp,
                    audio_padding_mask_full=tt_pad_mask_full,
                    video_padding_mask=tt_v_pad_mask_sp,
                    video_padding_mask_full=tt_v_pad_mask_full,
                )
                vv = LTXTransformerModel.device_to_host(
                    v,
                    ccl_manager=self.ccl_manager,
                    parallel_config=self.parallel_config,
                    sp_already_gathered=True,
                    tp_already_gathered=True,
                ).squeeze(0)
                av = LTXTransformerModel.device_to_host(
                    a,
                    ccl_manager=self.ccl_manager,
                    parallel_config=self.parallel_config,
                    sp_already_gathered=True,
                    tp_already_gathered=True,
                ).squeeze(0)
                vd = (video_lat.bfloat16().float() - vv.float() * sigma).bfloat16()
                ad = (audio_lat.bfloat16().float() - av.float() * sigma).bfloat16()
                return self._zero_video_padding(vd, video_N_real), self._zero_audio_padding(ad, audio_N_real)

            v_den, a_den = _run(tt_vp, tt_ap)

            v_unc = a_unc = v_ptb = a_ptb = v_iso = a_iso = 0.0
            if do_cfg:
                v_unc, a_unc = _run(tt_nv, tt_na)
            if do_stg:
                v_ptb, a_ptb = _run(tt_vp, tt_ap, skip_sa_blocks=[stg_block])
            if do_mod:
                v_iso, a_iso = _run(tt_vp, tt_ap, skip_ca=True)

            if do_cfg or do_stg or do_mod:
                v_den = self._apply_modal_guidance(
                    v_den,
                    v_unc,
                    v_ptb,
                    v_iso,
                    cfg_scale=video_cfg_scale,
                    stg_scale=video_stg_scale,
                    modality_scale=video_modality_scale,
                    rescale_scale=rescale_scale,
                    do_cfg=do_cfg,
                    do_stg=do_stg,
                    do_mod=do_mod,
                    real_token_count=video_N_real if video_N > video_N_real else None,
                )
                a_den = self._apply_modal_guidance(
                    a_den,
                    a_unc,
                    a_ptb,
                    a_iso,
                    cfg_scale=audio_cfg_scale,
                    stg_scale=audio_stg_scale,
                    modality_scale=audio_modality_scale,
                    rescale_scale=rescale_scale,
                    do_cfg=do_cfg,
                    do_stg=do_stg,
                    do_mod=do_mod,
                    real_token_count=audio_N_real,
                )

            # Gradient estimation: correct velocity using previous step's velocity.
            # GE math operates on real (unpadded) slices only — padded slots are noise-free
            # placeholders, and including them here would leak garbage into the GE state.
            if ge_gamma != 0.0 and sigma_next != 0.0:
                v_lat_real = video_lat[:, :video_N_real, :]
                v_den_real = v_den[:, :video_N_real, :]
                v_velocity = (v_lat_real.float() - v_den_real.float()) / sigma
                a_lat_real = audio_lat[:, :audio_N_real, :]
                a_den_real = a_den[:, :audio_N_real, :]
                a_velocity = (a_lat_real.float() - a_den_real.float()) / sigma
                if prev_v_vel is not None:
                    v_total = ge_gamma * (v_velocity - prev_v_vel) + prev_v_vel
                    a_total = ge_gamma * (a_velocity - prev_a_vel) + prev_a_vel
                    v_den_real = (v_lat_real.float() - v_total * sigma).bfloat16()
                    a_den_real = (a_lat_real.float() - a_total * sigma).bfloat16()
                    # v_den/a_den came pre-zeroed at padded slots from `_run` — only
                    # the real slice needs updating.
                    v_den[:, :video_N_real, :] = v_den_real
                    a_den[:, :audio_N_real, :] = a_den_real
                prev_v_vel, prev_a_vel = v_velocity, a_velocity

            # Last step: return denoised directly (sigma_next == 0)
            if sigma_next == 0.0:
                video_lat_new = v_den.float()
                audio_lat_new = a_den.float()
            else:
                video_lat_new = euler_step(video_lat, v_den.float(), sigma, sigma_next).bfloat16().float()
                audio_lat_new = euler_step(audio_lat, a_den.float(), sigma, sigma_next).bfloat16().float()
            # Re-zero padded slots after each step to keep the latent clean.
            video_lat = self._zero_video_padding(video_lat_new, video_N_real)
            audio_lat = self._zero_audio_padding(audio_lat_new, audio_N_real)

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                logger.info(f"Step {step_idx+1}/{num_inference_steps}: σ {sigma:.4f}→{sigma_next:.4f}")

        logger.info(
            f"AV done. video: ({B},{video_N_real},{self.in_channels}), "
            f"audio: ({B},{audio_N_real},{self.in_channels})"
        )
        return video_lat[:, :video_N_real, :], audio_lat[:, :audio_N_real, :]

    def _prepare_audio_decoder(self) -> None:
        """Lazily build the on-device audio decoder chain (mel-VAE + vocoder + BWE).

        Reads weights from ``self.checkpoint_name`` (the LTX-2.3 safetensors).
        Caches the chain on ``self.tt_audio_decoder`` / ``self.tt_audio_vocoder_bwe``.
        """
        if getattr(self, "tt_audio_vocoder_bwe", None) is not None:
            return

        from safetensors import safe_open

        from ...models.audio_vae.audio_decoder_ltx import LTXAudioDecoder
        from ...models.audio_vae.bwe_ltx import LTXMelSTFT, LTXVocoderWithBWE
        from ...models.audio_vae.vocoder_ltx import LTXVocoder

        logger.info(f"Loading on-device audio decoder weights from {self.checkpoint_name}")

        # Pull only audio-related keys to avoid materializing the full 22B transformer.
        def _extract_prefix(prefix: str) -> dict:
            sub = {}
            with safe_open(self.checkpoint_name, framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        sub[k[len(prefix) :]] = f.get_tensor(k)
            return sub

        # ---- LTXAudioDecoder (mel-VAE) ----
        audio_dec_state = _extract_prefix("audio_vae.decoder.")
        # per_channel_statistics lives under audio_vae.per_channel_statistics
        with safe_open(self.checkpoint_name, framework="pt") as f:
            audio_dec_state["per_channel_statistics.mean-of-means"] = f.get_tensor(
                "audio_vae.per_channel_statistics.mean-of-means"
            )
            audio_dec_state["per_channel_statistics.std-of-means"] = f.get_tensor(
                "audio_vae.per_channel_statistics.std-of-means"
            )

        self.tt_audio_decoder = LTXAudioDecoder(
            ch=128,
            out_ch=2,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(),
            resolution=256,
            z_channels=8,
            mid_block_add_attention=False,
            sample_rate=16000,
            mel_hop_length=160,
            is_causal=True,
            mel_bins=64,
            mesh_device=self.mesh_device,
            dtype=ttnn.bfloat16,
        )
        self.tt_audio_decoder.load_torch_state_dict(audio_dec_state, strict=False)

        # ---- main LTXVocoder (16 kHz, total upsample 160) ----
        voc_state = _extract_prefix("vocoder.vocoder.")
        main_vocoder = LTXVocoder(
            upsample_initial_channel=1536,
            resblock="AMP1",
            upsample_rates=[5, 2, 2, 2, 2, 2],
            upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation="snakebeta",
            use_tanh_at_final=False,
            apply_final_activation=True,
            use_bias_at_final=False,
            in_channels=128,
            out_channels=2,
            mesh_device=self.mesh_device,
        )
        main_vocoder.load_torch_state_dict(voc_state, strict=False)

        # ---- BWE generator (48 kHz, total upsample 240, apply_final_activation=False) ----
        bwe_state = _extract_prefix("vocoder.bwe_generator.")
        bwe_vocoder = LTXVocoder(
            upsample_initial_channel=512,
            resblock="AMP1",
            upsample_rates=[6, 5, 2, 2, 2],
            upsample_kernel_sizes=[12, 11, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation="snakebeta",
            use_tanh_at_final=False,
            apply_final_activation=False,  # BWE generator outputs unclamped residual
            use_bias_at_final=False,
            in_channels=128,
            out_channels=2,
            mesh_device=self.mesh_device,
        )
        bwe_vocoder.load_torch_state_dict(bwe_state, strict=False)

        # ---- MelSTFT ----
        melstft_state = _extract_prefix("vocoder.mel_stft.")
        melstft = LTXMelSTFT(
            filter_length=512,
            hop_length=80,
            win_length=512,
            n_mel_channels=64,
            mesh_device=self.mesh_device,
        )
        melstft.load_torch_state_dict(melstft_state, strict=False)

        # ---- VocoderWithBWE wrapper ----
        self.tt_audio_vocoder_bwe = LTXVocoderWithBWE(
            vocoder=main_vocoder,
            bwe_generator=bwe_vocoder,
            mel_stft=melstft,
            input_sampling_rate=16000,
            output_sampling_rate=48000,
            hop_length=80,
            mesh_device=self.mesh_device,
        )
        logger.info("On-device audio decoder ready")

    def decode_audio_device(self, audio_latent: torch.Tensor, num_frames: int, fps: float = 24.0):
        """On-device counterpart to ``decode_audio_reference``.

        Replaces the host-torch ``AudioDecoder`` + ``VocoderWithBWE`` chain with
        ``LTXAudioDecoder`` (mel-VAE) → ``LTXVocoderWithBWE`` running on the mesh.
        Falls back to the reference decoder on any exception so the pipeline
        never silently produces garbage.
        """
        try:
            from ltx_core.types import Audio
        except ImportError:
            import sys

            sys.path.insert(0, "LTX-2/packages/ltx-core/src")
            sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
            from ltx_core.types import Audio

        try:
            self._prepare_audio_decoder()
            audio_N = audio_latent.shape[1]
            audio_spatial = audio_latent.reshape(1, audio_N, 8, 16).permute(0, 2, 1, 3).float().contiguous()
            mel_features = self.tt_audio_decoder(audio_spatial)
            waveform = self.tt_audio_vocoder_bwe(mel_features)
            sampling_rate = 48000

            video_duration = num_frames / fps
            target_samples = int(video_duration * sampling_rate)
            if waveform.shape[-1] > target_samples:
                waveform = waveform[..., :target_samples]

            audio_obj = Audio(waveform=waveform.squeeze(0).float(), sampling_rate=sampling_rate)
            logger.info(
                f"Audio decoded (on-device): {audio_obj.waveform.shape} "
                f"({audio_obj.waveform.shape[-1] / audio_obj.sampling_rate:.2f}s @ {sampling_rate}Hz)"
            )
            return audio_obj
        except Exception as e:
            logger.warning(f"On-device audio decode failed: {e}; falling back to reference")
            return self.decode_audio_reference(audio_latent, num_frames, fps=fps)

    def decode_audio_reference(self, audio_latent: torch.Tensor, num_frames: int, fps: float = 24.0):
        """Decode audio latent using reference audio VAE + vocoder (CPU torch).

        Args:
            audio_latent: (1, audio_N, 128) raw audio latent from call_av()
            num_frames: video frame count (for duration trimming)
            fps: video frame rate

        Returns:
            Audio object with .waveform and .sampling_rate, or None on failure
        """
        assert self.checkpoint_name is not None, "checkpoint_name must be set before decode_audio_reference"
        try:
            import sys

            sys.path.insert(0, "LTX-2/packages/ltx-core/src")
            sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
            torch.cuda.synchronize = lambda *a, **kw: None
            from ltx_core.types import Audio
            from ltx_pipelines.utils.blocks import AudioDecoder

            # AudioDecoder block owns both the audio VAE decoder and the
            # vocoder lifecycle (build → decode → free), replacing the old
            # `ModelLedger.audio_decoder()` + `ledger.vocoder()` +
            # `vae_decode_audio(...)` triplet from LTX-2 pre-1.1.
            #
            # NOTE: must build in fp32 on CPU.  LTX-2 main's `VocoderWithBWE.forward`
            # wraps itself in `torch.autocast(dtype=torch.float32)` and feeds the
            # vocoder `mel_spec.float()`.  On GPU autocast handles the bf16-weight ↔
            # fp32-input mismatch per-op; on CPU `autocast(dtype=fp32)` is silently
            # disabled (a UserWarning is logged at import time), so the fp32 input
            # collides with bf16 conv biases and the decode aborts with
            # "Input type (float) and bias type (c10::BFloat16) should be the same".
            # Loading both audio_decoder and vocoder in fp32 sidesteps this; the
            # audio VAE + vocoder are small relative to the 22B transformer.
            audio_block = AudioDecoder(
                checkpoint_path=self.checkpoint_name,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

            # Unpatchify: (1, N, 128) → (1, 8, N, 16).  Match the fp32 audio
            # decoder dtype to avoid the same bias-dtype mismatch at its first conv.
            audio_N = audio_latent.shape[1]
            audio_spatial = audio_latent.reshape(1, audio_N, 8, 16).permute(0, 2, 1, 3).float()

            with torch.no_grad():
                audio_obj = audio_block(audio_spatial)

            # Trim to video duration
            video_duration = num_frames / fps
            target_samples = int(video_duration * audio_obj.sampling_rate)
            if audio_obj.waveform.shape[-1] > target_samples:
                audio_obj = Audio(
                    waveform=audio_obj.waveform[..., :target_samples], sampling_rate=audio_obj.sampling_rate
                )

            logger.info(
                f"Audio decoded: {audio_obj.waveform.shape} "
                f"({audio_obj.waveform.shape[-1]/audio_obj.sampling_rate:.2f}s @ {audio_obj.sampling_rate}Hz)"
            )
            return audio_obj
        except Exception as e:
            logger.warning(f"Audio decode failed: {e}")
            return None

    def export_video(self, video_pixels: torch.Tensor, output_path: str, fps: int = 24, audio=None) -> None:
        """Export decoded video (and optionally audio) to MP4.

        Matches reference ltx_pipelines.utils.media_io.encode_video exactly:
        - H.264 video with yuv420p pixel format
        - AAC audio stream (if audio provided)
        - Correct [-1,1] → uint8 conversion

        Args:
            video_pixels: (B, C, F, H, W) from decode_latents(), range [-1, 1]
            output_path: output .mp4 path
            fps: frame rate
            audio: object with .waveform (torch.Tensor) and .sampling_rate (int), or None
        """
        from fractions import Fraction

        import av

        # Convert to (F, H, W, C) uint8
        frames = (((video_pixels[0] + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C)

        _, height, width, _ = frames.shape

        container = av.open(output_path, mode="w")
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        # Prepare audio stream if provided
        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=audio.sampling_rate)
            audio_stream.codec_context.sample_rate = audio.sampling_rate
            audio_stream.codec_context.layout = "stereo"
            audio_stream.codec_context.time_base = Fraction(1, audio.sampling_rate)

        # Write video frames
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush video encoder
        for packet in stream.encode():
            container.mux(packet)

        # Write audio if provided
        if audio is not None and audio_stream is not None:
            samples = audio.waveform
            if samples.ndim == 1:
                samples = samples[:, None]
            if samples.shape[1] != 2 and samples.shape[0] == 2:
                samples = samples.T
            if samples.shape[1] != 2:
                logger.warning(f"Audio has {samples.shape[1]} channels, expected 2 — duplicating mono")
                samples = samples[:, :1].repeat(1, 2)

            if samples.dtype != torch.int16:
                samples = torch.clip(samples, -1.0, 1.0)
                samples = (samples * 32767.0).to(torch.int16)

            frame_in = av.AudioFrame.from_ndarray(
                samples.contiguous().reshape(1, -1).cpu().numpy(),
                format="s16",
                layout="stereo",
            )
            frame_in.sample_rate = audio.sampling_rate

            # Resample to encoder format and write
            cc = audio_stream.codec_context
            resampler = av.audio.resampler.AudioResampler(
                format=cc.format or "fltp",
                layout=cc.layout or "stereo",
                rate=cc.sample_rate or audio.sampling_rate,
            )
            for resampled in resampler.resample(frame_in):
                for packet in audio_stream.encode(resampled):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)

        container.close()
        logger.info(f"Saved: {output_path} ({frames.shape[0]}f @ {fps}fps)")
