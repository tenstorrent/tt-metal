# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device Gemma-4 text encoder + embeddings connectors for LTX-2.5.

Mirrors the 2.3 pair, and reuses its feature extractor and connectors unchanged — both are
identical across the two releases. What differs is where the weights come from. 2.3 read
Gemma from an HF directory and everything else from the monolithic LTX checkpoint; 2.5
packs Gemma, the aggregate projection, the config and the tokenizer into one text-encoder
file, and leaves the connectors in the transformer checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

import numpy as np
import torch
from loguru import logger
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import ttnn

from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.mochi import get_rot_transformation_mat
from ...utils.tensor import bf16_tensor
from ...utils.tracing import traced_function
from ..gemma3.embeddings_connector import EmbeddingsConnector
from ..gemma3.feature_extractor import GemmaFeatureExtractor
from .model_gemma import Gemma4Config, Gemma4Encoder

GEMMA_SEQUENCE_LENGTH = 1024
VIDEO_EMBED_DIM = 4096
AUDIO_EMBED_DIM = 2048

GEMMA_CONFIG_METADATA_KEY = "gemma_config"
TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
HF_ASSET_TENSOR_PREFIX = "hf_asset__"

# `added_tokens_decoder` is for from_pretrained reconstruction and TypeErrors as a kwarg.
# `extra_special_tokens` and `model_specific_special_tokens` are transformers-5 spellings
# that the pinned 4.53 rejects; their tokens are in the vocab regardless, so dropping them
# costs only an attribute alias.
_TOKENIZER_CONFIG_SKIP = frozenset(
    {
        "added_tokens_decoder",
        "auto_map",
        "backend",
        "extra_special_tokens",
        "is_local",
        "local_files_only",
        "model_max_length",
        "model_specific_special_tokens",
        "processor_class",
        "tokenizer_class",
    }
)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().numpy().astype(np.uint8).tobytes()


class Gemma4Assets:
    """Config, tokenizer and weights read out of the packed text-encoder file."""

    def __init__(self, path: str) -> None:
        self.path = path
        with safe_open(path, "pt") as f:
            metadata = f.metadata() or {}
            if GEMMA_CONFIG_METADATA_KEY not in metadata:
                msg = f"{path} has no {GEMMA_CONFIG_METADATA_KEY!r} metadata; not a packed LTX-2.5 text encoder"
                raise ValueError(msg)
            self.config_dict = json.loads(metadata[GEMMA_CONFIG_METADATA_KEY])
            self.tokenizer_json = _tensor_to_bytes(f.get_tensor(TOKENIZER_JSON_TENSOR_KEY))
            self.tokenizer_config = json.loads(
                _tensor_to_bytes(f.get_tensor(f"{HF_ASSET_TENSOR_PREFIX}tokenizer_config.json"))
            )

    def text_config(self) -> dict:
        return self.config_dict["text_config"]

    def build_tokenizer(self, max_length: int) -> PreTrainedTokenizerFast:
        kwargs = {k: v for k, v in self.tokenizer_config.items() if k not in _TOKENIZER_CONFIG_SKIP}
        return PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(self.tokenizer_json),
            model_max_length=max_length,
            **kwargs,
        )

    def read(self, prefix: str) -> dict[str, torch.Tensor]:
        with safe_open(self.path, "pt") as f:
            return {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}


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


def _read_connectors(transformer_checkpoint: str) -> dict[str, torch.Tensor]:
    prefix = "model.diffusion_model."
    with safe_open(transformer_checkpoint, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix) and "embeddings_connector." in k}


class Gemma4TokenizerEncoderPair:
    """Tokenizer + on-device Gemma-4 encoder + embeddings connectors for LTX-2.5.

    ``encode(prompts)`` returns ``[(video_embeds, audio_embeds), ...]``. Gemma, the tokenizer
    and the aggregate projection come from ``text_encoder_path``; the connectors from
    ``transformer_checkpoint``.
    """

    def __init__(
        self,
        text_encoder_path: str,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config,
        transformer_checkpoint: str | None = None,
        mode: str = "av",
        dynamic_load: bool = False,
        sequence_length: int = GEMMA_SEQUENCE_LENGTH,
        video_dim: int = VIDEO_EMBED_DIM,
        audio_dim: int = AUDIO_EMBED_DIM,
    ) -> None:
        self.text_encoder_path = text_encoder_path
        self.transformer_checkpoint = transformer_checkpoint
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.mode = mode
        self.dynamic_load = dynamic_load
        self._sequence_length = sequence_length
        self._video_dim = video_dim
        self._audio_dim = audio_dim

        self.assets = Gemma4Assets(text_encoder_path)
        self.config = Gemma4Config.from_hf_text_config(self.assets.text_config())

        self.gemma_encoder = None
        self.tokenizer = None
        self.feature_extractor = None
        self.video_connector = None
        self.audio_connector = None
        self._coresident_peers: list = []
        self._cached_trans_mat = None
        # Tracing only pays off while the encoder stays resident; under dynamic_load it is
        # evicted after every encode, so each one would be a cold capture.
        self._encoder_trace = not dynamic_load

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
        Exclusions are wired at each module's first build, so peers must be registered
        before the first ``ensure_loaded``."""
        self._coresident_peers = list(peers)

    def _register_exclusions(self, module) -> None:
        if not self.dynamic_load or not self._coresident_peers:
            return
        module.register_coresident_exclusions(*self._coresident_peers)
        for peer in self._coresident_peers:
            peer.register_coresident_exclusions(module)

    def is_loaded(self) -> bool:
        return self.gemma_encoder is not None and self.gemma_encoder.is_loaded()

    def ensure_loaded(self, connector_state: dict | Callable[[], dict] | None = None) -> None:
        if self.is_loaded():
            return
        self.load_gemma_encoder()
        self.load_embeddings_connectors(
            connector_state if connector_state is not None else (lambda: _read_connectors(self.transformer_checkpoint))
        )

    def load_gemma_encoder(self) -> None:
        """Load the TTNN Gemma-4 encoder. Built once and reused across reloads; weights come
        back from the shared cache rather than re-tilizing 12B parameters."""
        if self.gemma_encoder is None:
            self.gemma_encoder = Gemma4Encoder(
                self.config,
                self.mesh_device,
                self.ccl_manager,
                self.parallel_config,
                max_seq_len=self._sequence_length,
            )
            self._register_exclusions(self.gemma_encoder)
            self.tokenizer = self.assets.build_tokenizer(self._sequence_length)

        t0 = time.time()
        cache_module.load_model(
            self.gemma_encoder,
            model_name=self._cache_name(self.text_encoder_path),
            subfolder="text_encoder",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: self.assets.read("model."),
        )
        logger.info(f"Loaded TTNN Gemma-4 encoder ({self.config.num_hidden_layers}L) in {time.time() - t0:.0f}s")

    def load_embeddings_connectors(
        self,
        connector_state: dict[str, torch.Tensor] | Callable[[], dict[str, torch.Tensor]],
        *,
        video_num_blocks: int = 8,
        audio_num_blocks: int = 8,
        num_heads: int = 32,
    ) -> None:
        """Load the feature extractor from the packed text encoder and the connectors from the
        transformer checkpoint, both through the shared cache. ``connector_state`` is a dict or
        a zero-arg callable, resolved once and only on a cache miss."""
        ckpt = _memoize(connector_state)
        connector_name = self._cache_name(self.transformer_checkpoint or "ltx-connectors")
        # The projection consumes the embedding output plus every decoder layer.
        num_aggregated = self.config.num_hidden_layers + 1

        if self.feature_extractor is None:
            self.feature_extractor = GemmaFeatureExtractor(
                input_dim=self.config.hidden_size * num_aggregated,
                embedding_dim=self.config.hidden_size,
                video_dim=self._video_dim,
                audio_dim=self._audio_dim if self.mode == "av" else None,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
            )
            self._register_exclusions(self.feature_extractor)
        cache_module.load_model(
            self.feature_extractor,
            model_name=self._cache_name(self.text_encoder_path),
            subfolder="feature_extractor",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: _feature_extractor_state_dict(
                self.assets.read("text_embedding_projection."),
                mode=self.mode,
                gemma_hidden_size=self.config.hidden_size,
                gemma_num_layers=num_aggregated,
            ),
        )

        self.video_connector = self._load_connector(
            "video", self._video_dim, video_num_blocks, num_heads, connector_name, ckpt
        )
        self.audio_connector = (
            self._load_connector("audio", self._audio_dim, audio_num_blocks, num_heads, connector_name, ckpt)
            if self.mode == "av"
            else None
        )

    def _load_connector(self, axis, output_dim, num_blocks, num_heads, connector_name, ckpt) -> EmbeddingsConnector:
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
            model_name=connector_name,
            subfolder=f"{axis}_connector",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            dtype="float32",
            get_torch_state_dict=lambda: _connector_state_dict(ckpt(), axis, num_blocks),
        )
        logger.info(f"Loaded {axis} embeddings connector ({num_blocks} blocks, dim={output_dim})")
        return connector

    def tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-padded ids and mask. Gemma-4's tokenizer has no BOS post-processor — unlike
        Gemma-3's — so the leading BOS is added here; without it every hidden state shifts."""
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            msg = "tokenizer has no bos_token_id; the encode path requires a leading BOS"
            raise ValueError(msg)

        encoded = self.tokenizer(
            prompt.strip(), padding=False, truncation=True, max_length=self._sequence_length, return_tensors="pt"
        )
        ids = encoded.input_ids[0].tolist()
        if not ids or ids[0] != bos_id:
            ids = [bos_id, *ids][: self._sequence_length]

        padded = self.tokenizer.pad(
            {"input_ids": [ids]},
            padding="max_length",
            max_length=self._sequence_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return padded.input_ids, padded.attention_mask

    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=True)
    def _encode_device(self, tt_ids, tt_gemma_mask, fe_mask, src_idx, keep_mask):
        """Whole-encode device graph (gemma → feature extractor → connectors), captured as one
        ttnn trace and replayed per prompt. Returns DEVICE embeds; the tokenizer and the final
        to_torch stay in encode."""
        trans_mat = self._prepare_trans_mat()
        all_hidden = self.gemma_encoder(tt_ids, tt_attn_mask=tt_gemma_mask)
        # The projection consumes [embed, L0..L46, final_norm]; the encoder emits
        # [embed, L0..L47, final_norm], so drop index -2.
        hs_list = list(all_hidden[:-2]) + [all_hidden[-1]]
        video_feats, audio_feats = self.feature_extractor(hs_list, fe_mask)
        video = self.video_connector(video_feats, src_idx, keep_mask, trans_mat=trans_mat)
        audio = (
            self.audio_connector(audio_feats, src_idx, keep_mask, trans_mat=trans_mat)
            if self.audio_connector is not None
            else None
        )
        return video, audio

    def encode(self, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Tokenize → traced whole-encode device graph → host embeds, one
        ``(video_embeds, audio_embeds)`` per prompt."""
        assert self.gemma_encoder is not None, "Call ensure_loaded() first"

        results = []
        for prompt in prompts:
            input_ids, attention_mask = self.tokenize(prompt)
            tt_ids = ttnn.from_torch(
                input_ids, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            seq = tt_ids.shape[-1]
            tt_gemma_mask = self.gemma_encoder.build_attn_mask(attention_mask, seq)
            fe_mask = self.feature_extractor.build_mask(attention_mask)
            # src_idx/keep_mask are dim-independent → shared by both connectors.
            src_idx, keep_mask = self.video_connector.build_indices(attention_mask, seq)

            video_dev, audio_dev = self._encode_device(
                tt_ids, tt_gemma_mask, fe_mask, src_idx, keep_mask, traced=self._encoder_trace
            )
            video_embeds = ttnn.to_torch(ttnn.get_device_tensors(video_dev)[0]).float()
            audio_embeds = (
                ttnn.to_torch(ttnn.get_device_tensors(audio_dev)[0]).float() if audio_dev is not None else None
            )
            results.append((video_embeds, audio_embeds))
        return results

    def _prepare_trans_mat(self) -> ttnn.Tensor:
        """Cached per-tile rotation matrix for rotary_embedding_llama."""
        if self._cached_trans_mat is None:
            self._cached_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return self._cached_trans_mat

    @staticmethod
    def _cache_name(path: str) -> str:
        return os.path.basename(os.path.normpath(path)).removesuffix(".safetensors")


def _memoize(state: dict | Callable[[], dict]) -> Callable[[], dict]:
    """Wrap a dict-or-callable as a zero-arg callable that resolves at most once."""
    cached: dict = {}

    def get() -> dict:
        if "sd" not in cached:
            cached["sd"] = state() if callable(state) else state
        return cached["sd"]

    return get
