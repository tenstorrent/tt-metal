# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device Gemma-3 text encoder + embeddings connectors for LTX-2, following the
tt_dit encoder_pair convention (cf. ``encoders/t5/encoder_pair.py``)."""

from __future__ import annotations

import glob
import os
import time
from typing import Callable

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

import ttnn

from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.mochi import get_rot_transformation_mat
from ...utils.tensor import bf16_tensor
from .embeddings_connector import EmbeddingsConnector
from .feature_extractor import GemmaFeatureExtractor
from .model_gemma import GemmaConfig, GemmaEncoder

GEMMA_SEQUENCE_LENGTH = 1024
VIDEO_EMBED_DIM = 4096
AUDIO_EMBED_DIM = 2048

_CONNECTOR_PREFIXES = (
    "text_embedding_projection.video_aggregate_embed.",
    "text_embedding_projection.audio_aggregate_embed.",
    "model.diffusion_model.video_embeddings_connector.",
    "model.diffusion_model.audio_embeddings_connector.",
)


def _resolve_gemma_dir(gemma: str) -> str:
    """Resolve a Gemma reference to a local directory: a local dir is returned as-is, a
    HuggingFace repo id is snapshot-downloaded (cached). The original reference — not this
    resolved snapshot dir, whose basename is an opaque revision hash — names the cache."""
    if os.path.isdir(gemma):
        return gemma
    logger.info(f"Resolving HuggingFace Gemma repo {gemma} (auto-download if missing)")
    return snapshot_download(repo_id=gemma)


# --- source state dicts (lazy: only read on a cache miss) -------------------
def _gemma_state_dict(gemma_path: str) -> dict[str, torch.Tensor]:
    """All Gemma encoder weights, from the HF safetensors shards."""
    weight_files = sorted(glob.glob(f"{gemma_path}/model-*.safetensors")) or sorted(
        glob.glob(f"{gemma_path}/*.safetensors")
    )
    sd: dict[str, torch.Tensor] = {}
    for f in weight_files:
        sd.update(load_file(f))
    return sd


def _read_connector_checkpoint(checkpoint_name: str) -> dict[str, torch.Tensor]:
    """The connector + aggregate_embed tensors from the LTX checkpoint."""
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_name, "pt") as f:
        for k in f.keys():
            if k.startswith(_CONNECTOR_PREFIXES):
                state[k] = f.get_tensor(k)
    return state


def _feature_extractor_state_dict(ckpt, *, mode: str, gemma_hidden_size: int, gemma_num_layers: int) -> dict:
    """video/audio aggregate_embed weights, permuted D-major→layer-major to match the
    on-device layer-major concat (see GemmaFeatureExtractor._weight_to_layer_major)."""
    sd = {}
    for axis in ("video", "audio") if mode == "av" else ("video",):
        prefix = f"text_embedding_projection.{axis}_aggregate_embed."
        for k, v in ckpt.items():
            if k.startswith(prefix):
                sub = k[len(prefix) :]
                if sub == "weight":
                    v = GemmaFeatureExtractor._weight_to_layer_major(v, gemma_hidden_size, gemma_num_layers)
                sd[f"{axis}_aggregate_embed.{sub}"] = v
    return sd


def _connector_state_dict(ckpt, axis: str, num_blocks: int) -> dict:
    """One connector's transformer blocks + norm, dropping blocks beyond ``num_blocks``."""
    prefix = f"model.diffusion_model.{axis}_embeddings_connector."
    sd = {}
    for k, v in ckpt.items():
        if not k.startswith(prefix):
            continue
        sub = k[len(prefix) :]
        if sub.startswith("transformer_1d_blocks.") and int(sub.split(".")[1]) >= num_blocks:
            continue
        sd[sub] = v
    return sd


class GemmaTokenizerEncoderPair:
    """Tokenizer + on-device Gemma encoder + embeddings connectors for LTX-2.

    ``encode(prompts)`` returns ``[(video_embeds, audio_embeds), ...]``. Gemma weights come
    from ``gemma_path`` safetensors; connector/feature-extractor weights come from the LTX
    ``checkpoint_name``. The prompt-embedding disk cache lives in the pipeline, not here —
    this class is pure load + compute.
    """

    def __init__(
        self,
        gemma_path: str | None,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config,
        checkpoint_name: str | None = None,
        mode: str = "av",
        dynamic_load: bool = False,
        sequence_length: int = GEMMA_SEQUENCE_LENGTH,
        video_dim: int = VIDEO_EMBED_DIM,
        audio_dim: int = AUDIO_EMBED_DIM,
        num_layers: int = 48,
        hidden_layer_index: int = -1,
    ) -> None:
        # Name the cache by the caller's reference (a repo id / dir basenames cleanly),
        # mirroring T5; resolve it to a local dir for tokenizer + weight loading.
        self.gemma_ref = gemma_path
        self.gemma_path = _resolve_gemma_dir(gemma_path) if gemma_path else None
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.checkpoint_name = checkpoint_name
        self.mode = mode
        self.dynamic_load = dynamic_load
        self._num_layers = num_layers
        self._hidden_layer_index = hidden_layer_index
        self._sequence_length = sequence_length
        self._video_dim = video_dim
        self._audio_dim = audio_dim

        self.gemma_encoder = None
        self.tokenizer = None
        self.feature_extractor = None
        self.video_connector = None
        self.audio_connector = None
        self._coresident_peers: list = []
        self._cached_trans_mat = None

    # Dims the pipeline warmup needs before the encoder is built.
    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def video_dim(self) -> int:
        return self._video_dim

    @property
    def audio_dim(self) -> int:
        return self._audio_dim

    def register_coresident_peers(self, peers: list) -> None:
        """Store the DiT/VAE peers the encoder modules must not be L1-coresident with.
        Exclusions are wired at each module's first build, so peers must be registered before
        the first ``ensure_loaded``/``load_*``."""
        self._coresident_peers = list(peers)

    def _register_exclusions(self, module) -> None:
        """Bidirectionally exclude a freshly-built encoder module against the stored peers,
        before its first load — so the encoder load auto-evicts the DiT (and the later DiT
        reload auto-evicts the encoder). No-op unless dynamic_load."""
        if not self.dynamic_load or not self._coresident_peers:
            return
        module.register_coresident_exclusions(*self._coresident_peers)
        for p in self._coresident_peers:
            p.register_coresident_exclusions(module)

    def is_loaded(self) -> bool:
        return self.gemma_encoder is not None and self.gemma_encoder.is_loaded()

    def ensure_loaded(self, connector_state: dict | Callable[[], dict] | None = None) -> None:
        """Build the modules once and (re)load their weights if a prior DiT/VAE load evicted
        them (the common case under dynamic_load)."""
        if self.is_loaded():
            return
        self.load_gemma_encoder()
        self.load_embeddings_connectors(
            connector_state
            if connector_state is not None
            else (lambda: _read_connector_checkpoint(self.checkpoint_name))
        )

    def load_gemma_encoder(self, gemma_path: str | None = None) -> None:
        """Load the TTNN Gemma-3 encoder. Built once and reused across reloads; weights reload
        from the shared cache (no re-tilizing 12B params). The CCLManager is supplied by the
        pipeline (shared with the connectors), so reloads never rebuild it."""
        gemma_path = gemma_path or self.gemma_path
        if self.gemma_encoder is None:
            config = GemmaConfig(
                num_hidden_layers=self._num_layers,
                hidden_layer_index=self._hidden_layer_index,
                max_position_embeddings=self._sequence_length,
            )
            self.gemma_encoder = GemmaEncoder(config, self.mesh_device, self.ccl_manager, self.parallel_config)
            self._register_exclusions(self.gemma_encoder)
            # Left-padding matches the reference FeatureExtractorV2: [PAD..PAD, BOS, real];
            # padded hidden states are zeroed via attention_mask on both sides.
            self.tokenizer = AutoTokenizer.from_pretrained(gemma_path)

        t0 = time.time()
        cache_module.load_model(
            self.gemma_encoder,
            model_name=os.path.basename(os.path.normpath(self.gemma_ref)),
            subfolder="text_encoder",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: _gemma_state_dict(gemma_path),
        )
        logger.info(f"Loaded TTNN Gemma encoder ({self._num_layers}L) in {time.time()-t0:.0f}s")

    def load_embeddings_connectors(
        self,
        checkpoint_state: dict[str, torch.Tensor] | Callable[[], dict[str, torch.Tensor]],
        *,
        gemma_hidden_size: int = 3840,
        gemma_num_layers: int = 49,  # embedding layer + 48 decoder layers
        video_num_blocks: int = 8,
        audio_num_blocks: int = 8,
        num_heads: int = 32,
    ) -> None:
        """Load the feature extractor + video/audio connectors from the LTX checkpoint through
        the shared cache. Modules are built once and reused across reloads; the CCLManager is
        supplied by the pipeline (shared with the Gemma encoder).

        ``checkpoint_state`` is a dict or a zero-arg callable; resolved once and only on a cache
        miss, so an all-cache-hit reload reads neither the checkpoint nor the tilizer.
        """
        ckpt = _memoize(checkpoint_state)
        ckpt_name = (
            os.path.basename(self.checkpoint_name).removesuffix(".safetensors")
            if self.checkpoint_name
            else "ltx-connectors"
        )

        if self.feature_extractor is None:
            self.feature_extractor = GemmaFeatureExtractor(
                input_dim=gemma_hidden_size * gemma_num_layers,
                embedding_dim=gemma_hidden_size,
                video_dim=self._video_dim,
                audio_dim=self._audio_dim if self.mode == "av" else None,
                mesh_device=self.mesh_device,
            )
            self._register_exclusions(self.feature_extractor)
        cache_module.load_model(
            self.feature_extractor,
            model_name=ckpt_name,
            subfolder="feature_extractor",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: _feature_extractor_state_dict(
                ckpt(), mode=self.mode, gemma_hidden_size=gemma_hidden_size, gemma_num_layers=gemma_num_layers
            ),
        )

        self.video_connector = self._load_connector(
            "video", self._video_dim, video_num_blocks, num_heads, ckpt_name, ckpt
        )
        self.audio_connector = (
            self._load_connector("audio", self._audio_dim, audio_num_blocks, num_heads, ckpt_name, ckpt)
            if self.mode == "av"
            else None
        )

    def _load_connector(self, axis, output_dim, num_blocks, num_heads, ckpt_name, ckpt) -> EmbeddingsConnector:
        connector = getattr(self, f"{axis}_connector")
        if connector is None:
            connector = EmbeddingsConnector(
                output_dim=output_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
            )
            self._register_exclusions(connector)
        cache_module.load_model(
            connector,
            model_name=ckpt_name,
            subfolder=f"{axis}_connector",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: _connector_state_dict(ckpt(), axis, num_blocks),
        )
        logger.info(f"Loaded {axis} embeddings connector ({num_blocks} blocks, dim={output_dim})")
        return connector

    def encode(self, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Tokenize → Gemma encoder → feature extractor → connectors, one
        ``(video_embeds, audio_embeds)`` per prompt. Pure compute; the disk cache lives in the
        pipeline."""
        assert self.gemma_encoder is not None, "Call ensure_loaded() first"
        trans_mat = self._prepare_trans_mat()

        results = []
        for prompt in prompts:
            tokens = self.tokenizer(
                prompt, return_tensors="pt", padding="max_length", max_length=self._sequence_length, truncation=True
            )
            tt_ids = ttnn.from_torch(
                tokens.input_ids, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            all_hidden_states = self.gemma_encoder(tt_ids, attention_mask=tokens.attention_mask)

            # The 49 states FeatureExtractorV2 consumes match HF output_hidden_states:
            # [embed, L0..L46, final_norm] — the last entry is post-final-norm, not the raw last
            # layer. The encoder emits [embed, L0..L47, final_norm], so drop index -2.
            hs_list = list(all_hidden_states[:-2]) + [all_hidden_states[-1]]
            video_feats, audio_feats = self.feature_extractor(hs_list, tokens.attention_mask)
            for hs in all_hidden_states:
                ttnn.deallocate(hs)

            video_embeds = self.video_connector(video_feats, tokens.attention_mask, trans_mat=trans_mat)
            audio_embeds = (
                self.audio_connector(audio_feats, tokens.attention_mask, trans_mat=trans_mat)
                if self.audio_connector is not None
                else None
            )
            results.append((video_embeds, audio_embeds))
        return results

    def _prepare_trans_mat(self) -> ttnn.Tensor:
        """Cached per-tile rotation matrix for rotary_embedding_llama."""
        if self._cached_trans_mat is None:
            self._cached_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return self._cached_trans_mat


def _memoize(state: dict | Callable[[], dict]) -> Callable[[], dict]:
    """Wrap a dict-or-callable as a zero-arg callable that resolves at most once."""
    cached: dict = {}

    def get() -> dict:
        if "sd" not in cached:
            cached["sd"] = state() if callable(state) else state
        return cached["sd"]

    return get
