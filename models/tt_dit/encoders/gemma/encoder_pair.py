# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
On-device Gemma-3 text encoder for LTX-2, following the tt_dit encoder_pair
convention (cf. ``encoders/t5/encoder_pair.py``): the pair owns module
construction, weight loading through the shared cache, and encoding.

Unlike T5 (encoder only), the LTX text-conditioning the DiT consumes is the
*connector* output, so this pair owns the full device assembly — the Gemma
encoder, the FeatureExtractorV2 (per-token RMS + dual aggregate_embed), and the
video/audio EmbeddingsConnectors. All four route through ``cache.load_model``.

Under ``dynamic_load`` the encoder modules are coresident-excluded with the DiT
and VAE (they can't all fit on BH-LB L1), so a DiT load evicts the encoder and
vice-versa. The pipeline registers the peer modules once via
``register_coresident_peers`` and reloads via ``ensure_loaded`` after each
evicting DiT load.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Callable

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

import ttnn

from ...models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis, reshape_interleaved_to_bhnd
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.tensor import bf16_tensor, prepare_rot_transformation_mat
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


class GemmaTokenizerEncoderPair:
    """Tokenizer + on-device Gemma encoder + embeddings connectors for LTX-2.

    ``encode(prompts)`` returns ``[(video_embeds, audio_embeds), ...]``. Gemma weights
    come from ``gemma_path`` safetensors; connector/feature-extractor weights come from
    the LTX ``checkpoint_name``. The prompt-embedding disk cache lives in the pipeline,
    not here — this class is pure load + compute.
    """

    def __init__(
        self,
        gemma_path: str | None,
        *,
        mesh_device: ttnn.MeshDevice,
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
        self.gemma_path = gemma_path
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.checkpoint_name = checkpoint_name
        self.mode = mode
        self.dynamic_load = dynamic_load
        self._sequence_length = sequence_length
        self._video_dim = video_dim
        self._audio_dim = audio_dim
        self._num_layers = num_layers
        self._hidden_layer_index = hidden_layer_index

        self.gemma_encoder = None
        self.tokenizer = None
        self.feature_extractor = None
        self.video_connector = None
        self.audio_connector = None
        self._enc_ccl = None
        self._coresident_peers: list = []
        self._cached_trans_mat = None

    # --- dims the pipeline warmup needs before the encoder is built ---
    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def video_dim(self) -> int:
        return self._video_dim

    @property
    def audio_dim(self) -> int:
        return self._audio_dim

    # --- coresident-exclusion handshake (peer modules are pipeline-owned) ---
    def register_coresident_peers(self, peers: list) -> None:
        """Store the DiT/VAE peers the encoder modules must not be L1-coresident with.
        Exclusions are wired (bidirectionally) at each module's first build, so the peer
        list must be registered before the first ``ensure_loaded``/``load_*``."""
        self._coresident_peers = list(peers)

    def _register_exclusions(self, module) -> None:
        """Bidirectionally exclude a freshly-built encoder module against the stored peers,
        BEFORE its first load — so the encoder load auto-evicts the DiT (and the later DiT
        reload auto-evicts the encoder). No-op unless dynamic_load."""
        if not self.dynamic_load or not self._coresident_peers:
            return
        peers = self._coresident_peers
        module.register_coresident_exclusions(*peers)
        for p in peers:
            p.register_coresident_exclusions(module)

    # --- lifecycle ---
    def is_loaded(self) -> bool:
        return self.gemma_encoder is not None and self.gemma_encoder.is_loaded()

    def ensure_loaded(self, connector_state: dict | Callable[[], dict] | None = None) -> None:
        """Build the modules once and (re)load their weights if a prior DiT/VAE load evicted
        them. Reloads are the common case under dynamic_load."""
        if self.is_loaded():
            return
        self.load_gemma_encoder()
        self.load_embeddings_connectors(connector_state or self._read_connector_state)

    def _read_connector_state(self) -> dict[str, torch.Tensor]:
        """Read the connector + aggregate_embed tensors from the LTX checkpoint. Lazy —
        only invoked on a connector cache miss."""
        conn_state = {}
        with safe_open(self.checkpoint_name, "pt") as f:
            for k in f.keys():
                if k.startswith(_CONNECTOR_PREFIXES):
                    conn_state[k] = f.get_tensor(k)
        return conn_state

    def load_gemma_encoder(
        self,
        gemma_path: str | None = None,
        *,
        num_layers: int | None = None,
        hidden_layer_index: int | None = None,
        sequence_length: int | None = None,
    ) -> None:
        """Load TTNN Gemma-3 text encoder on device. 13x faster than CPU torch."""
        gemma_path = gemma_path or self.gemma_path
        num_layers = num_layers if num_layers is not None else self._num_layers
        hidden_layer_index = hidden_layer_index if hidden_layer_index is not None else self._hidden_layer_index
        sequence_length = sequence_length if sequence_length is not None else self._sequence_length

        # Build the encoder + its CCLManager once. Under dynamic_load a later DiT/VAE load
        # evicts the encoder weights (coresident exclusion), so this is called again per
        # cache-missing prompt to reload them — the module and CCLManager are reused, since a
        # fresh CCLManager would leak its global semaphores and orphan the exclusion refs.
        if self.gemma_encoder is None:
            config = GemmaConfig(
                num_hidden_layers=num_layers,
                hidden_layer_index=hidden_layer_index,
                max_position_embeddings=sequence_length,
            )
            enc_ccl = CCLManager(self.mesh_device, topology=ttnn.Topology.Linear)
            self.gemma_encoder = GemmaEncoder(config, self.mesh_device, enc_ccl, self.parallel_config)

            # Register coresident exclusions BEFORE loading so the encoder load auto-evicts
            # the DiT (and the later DiT reload auto-evicts the encoder).
            self._register_exclusions(self.gemma_encoder)

            # Left-padding matches the reference FeatureExtractorV2 pipeline:
            # [PAD, ..., PAD, BOS, real tokens]. With causal SDPA the real tokens attend to
            # the padding; both sides zero padded hidden states out via attention_mask.
            self.tokenizer = AutoTokenizer.from_pretrained(gemma_path)
            self._sequence_length = sequence_length

        # Load through the shared weight cache (same path as the DiT/VAE): the first run
        # tilizes from safetensors and writes a pre-tilized cache; reloads read that cache
        # instead of re-tilizing 12B params. The state-dict read is lazy — only a cache miss
        # touches safetensors.
        def _gemma_state_dict() -> dict[str, torch.Tensor]:
            weight_files = sorted(glob.glob(f"{gemma_path}/model-*.safetensors")) or sorted(
                glob.glob(f"{gemma_path}/*.safetensors")
            )
            sd = {}
            for f in weight_files:
                sd.update(load_file(f))
            return sd

        t0 = time.time()
        cache_module.load_model(
            self.gemma_encoder,
            model_name=os.path.basename(os.path.normpath(gemma_path)),
            subfolder="text_encoder",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=_gemma_state_dict,
        )
        logger.info(f"Loaded TTNN Gemma encoder ({num_layers}L) in {time.time()-t0:.0f}s")

    def load_embeddings_connectors(
        self,
        checkpoint_state: dict[str, torch.Tensor] | Callable[[], dict[str, torch.Tensor]],
        *,
        gemma_hidden_size: int = 3840,
        gemma_num_layers: int = 49,  # embedding layer + 48 decoder layers
        video_num_blocks: int = 8,
        audio_num_blocks: int = 8,
        video_dim: int | None = None,
        audio_dim: int | None = None,
        num_heads: int = 32,
    ) -> None:
        """Load the feature extractor + video/audio embeddings connectors from the LTX-2
        checkpoint, through the shared cache."""
        video_dim = video_dim if video_dim is not None else self._video_dim
        audio_dim = audio_dim if audio_dim is not None else self._audio_dim
        input_dim = gemma_hidden_size * gemma_num_layers
        # Connector weights come from the LTX checkpoint, so cache them under its name
        # alongside the transformer/VAE caches.
        ckpt_name = (
            os.path.basename(self.checkpoint_name).removesuffix(".safetensors")
            if self.checkpoint_name
            else "ltx-connectors"
        )

        # checkpoint_state may be a dict or a zero-arg callable returning it. Resolved once and
        # only on a cache miss, so an all-cache-hit reload reads neither the checkpoint nor the
        # tilizer.
        _resolved: dict = {}

        def _ckpt() -> dict[str, torch.Tensor]:
            if "sd" not in _resolved:
                _resolved["sd"] = checkpoint_state() if callable(checkpoint_state) else checkpoint_state
            return _resolved["sd"]

        # Build the modules + their shared CCLManager once; weights load through the shared
        # cache. The modules/CCLManager are reused across reloads (a fresh CCLManager would
        # leak its global semaphores), and the cache skips re-tilizing on every reload.
        if self._enc_ccl is None:
            self._enc_ccl = CCLManager(self.mesh_device, topology=ttnn.Topology.Linear)

        # --- Feature extractor (per-token RMS + rescale + dual aggregate_embed) ---
        # Mirrors FeatureExtractorV2: owns the aggregate_embed weights; connectors consume
        # its projected output. The aggregate weight is permuted D-major→layer-major to match
        # the on-device layer-major concat (see GemmaFeatureExtractor._weight_to_layer_major).
        if self.feature_extractor is None:
            self.feature_extractor = GemmaFeatureExtractor(
                input_dim=input_dim,
                embedding_dim=gemma_hidden_size,
                video_dim=video_dim,
                audio_dim=audio_dim if self.mode == "av" else None,
                mesh_device=self.mesh_device,
            )
            self._register_exclusions(self.feature_extractor)

        def _fe_state_dict() -> dict[str, torch.Tensor]:
            sd = {}
            for axis in ("video", "audio") if self.mode == "av" else ("video",):
                agg_prefix = f"text_embedding_projection.{axis}_aggregate_embed."
                for k, v in _ckpt().items():
                    if k.startswith(agg_prefix):
                        sub = k[len(agg_prefix) :]
                        if sub == "weight":
                            v = GemmaFeatureExtractor._weight_to_layer_major(v, gemma_hidden_size, gemma_num_layers)
                        sd[f"{axis}_aggregate_embed.{sub}"] = v
            return sd

        cache_module.load_model(
            self.feature_extractor,
            model_name=ckpt_name,
            subfolder="feature_extractor",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=_fe_state_dict,
        )

        def _load_connector(axis: str, output_dim: int, num_blocks: int, connector) -> EmbeddingsConnector:
            if connector is None:
                connector = EmbeddingsConnector(
                    output_dim=output_dim,
                    num_blocks=num_blocks,
                    num_heads=num_heads,
                    mesh_device=self.mesh_device,
                    ccl_manager=self._enc_ccl,
                    parallel_config=self.parallel_config,
                )
                self._register_exclusions(connector)

            def _conn_state_dict() -> dict[str, torch.Tensor]:
                conn_prefix = f"model.diffusion_model.{axis}_embeddings_connector."
                sd = {}
                for k, v in _ckpt().items():
                    if k.startswith(conn_prefix):
                        sub = k[len(conn_prefix) :]
                        if sub.startswith("transformer_1d_blocks."):
                            if int(sub.split(".")[1]) >= num_blocks:  # drop blocks beyond num_blocks
                                continue
                        sd[sub] = v
                return sd

            cache_module.load_model(
                connector,
                model_name=ckpt_name,
                subfolder=f"{axis}_connector",
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                get_torch_state_dict=_conn_state_dict,
            )
            logger.info(f"Loaded {axis} embeddings connector ({num_blocks} blocks, dim={output_dim})")
            return connector

        self.video_connector = _load_connector("video", video_dim, video_num_blocks, self.video_connector)
        self.audio_connector = (
            _load_connector("audio", audio_dim, audio_num_blocks, self.audio_connector) if self.mode == "av" else None
        )

    def encode(self, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Tokenize → Gemma encoder → feature extractor → connectors. Returns one
        ``(video_embeds, audio_embeds)`` per prompt. Pure compute; the disk cache lives
        in the pipeline."""
        assert self.gemma_encoder is not None, "Call ensure_loaded() first"

        results = []
        for prompt in prompts:
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self._sequence_length,
                truncation=True,
            )

            tt_ids = ttnn.from_torch(
                tokens.input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            all_hidden_states = self.gemma_encoder(tt_ids, attention_mask=tokens.attention_mask)

            # The 49 states FeatureExtractorV2 consumes match HF output_hidden_states:
            # [embed, L0..L46, final_norm] — the last entry is post-final-norm, not the raw
            # last layer. The encoder emits [embed, L0..L47, final_norm], so drop index -2.
            hs_list = list(all_hidden_states[:-2]) + [all_hidden_states[-1]]

            # FeatureExtractorV2 (on device): per-token RMS + rescale + dual aggregate_embed
            # → video/audio features at the connector input dims.
            video_feats, audio_feats = self.feature_extractor(hs_list, tokens.attention_mask)
            for hs in all_hidden_states:
                ttnn.deallocate(hs)

            video_embeds = self._run_connector(self.video_connector, video_feats, tokens.attention_mask)
            audio_embeds = None
            if self.audio_connector is not None:
                audio_embeds = self._run_connector(self.audio_connector, audio_feats, tokens.attention_mask)

            results.append((video_embeds, audio_embeds))

        return results

    def _run_connector(self, connector, features, attn_mask) -> torch.Tensor:
        """Register replacement → on-device RoPE transformer blocks → final norm, on the
        aggregate_embed features from the feature extractor."""
        dim = connector.output_dim
        projected = ttnn.to_torch(ttnn.get_device_tensors(features)[0])
        ttnn.deallocate(features)

        # Replace padded tokens with learnable registers (on host, matching reference)
        if connector.num_learnable_registers > 0:
            registers = ttnn.to_torch(ttnn.get_device_tensors(connector.learnable_registers.data)[0])
            projected = self._replace_padded_with_registers(
                projected,
                attn_mask,
                registers,
                connector.num_learnable_registers,
            )

        # Connector RoPE on device. Checkpoint is rope_type=SPLIT, but the block's Q/K (and
        # q_norm/k_norm) weights were permuted at load (SPLIT→INTERLEAVED), so the fast
        # on-device rotary_embedding_llama interleaved kernel is equivalent. cos/sin use the
        # same fp32 freq grid as the reference. Computed once per connector, replicated.
        seq_len = projected.shape[1]
        num_heads = connector.transformer_1d_blocks[0].num_heads
        indices_grid = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)
        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid,
            dim=dim,
            out_dtype=torch.float32,
            theta=10000.0,
            max_pos=[4096],
            num_attention_heads=num_heads,
            rope_type=LTXRopeType.INTERLEAVED,
        )  # (1, seq, dim)
        cos_freq = reshape_interleaved_to_bhnd(cos_freq, num_heads)
        sin_freq = reshape_interleaved_to_bhnd(sin_freq, num_heads)
        # Shard the head dim on the connector's TP axis so cos/sin match the per-device
        # local-head count rotary_embedding_llama sees (the rope is per-head-varying, so it
        # can't be broadcast as num_heads=1). TP=1 → no-op.
        conn_tp = connector.transformer_1d_blocks[0].parallel_config.tensor_parallel
        shard_kw = {"mesh_axis": conn_tp.mesh_axis, "shard_dim": 1} if conn_tp.factor > 1 else {}
        rope_cos = bf16_tensor(cos_freq, device=self.mesh_device, **shard_kw)
        rope_sin = bf16_tensor(sin_freq, device=self.mesh_device, **shard_kw)
        trans_mat = self._prepare_trans_mat()

        tt_x = ttnn.from_torch(
            projected.bfloat16(),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        for block in connector.transformer_1d_blocks:
            tt_x = block(tt_x, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat)

        tt_x = ttnn.experimental.dit_rms_norm_unary_fused(
            tt_x, weight=None, epsilon=1e-6, compute_kernel_config=connector.rmsnorm_cc
        )
        result = ttnn.to_torch(ttnn.get_device_tensors(tt_x)[0]).float()

        # NOTE: Do NOT zero out register positions here. The reference FeatureExtractorV2
        # replaces padding with learnable registers and then sets attention_mask to
        # all-zeros (= no masking), so all 1024 tokens (real + register) carry information
        # after the connector blocks.
        return result

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
            padded = torch.nn.functional.pad(real_tokens, (0, 0, 0, pad_length))
            # Flip: registers at the beginning (where attention_mask was 0 = left-padded)
            flipped_mask = torch.flip(mask_binary[b : b + 1], dims=[1]).squeeze(0).unsqueeze(-1).int()
            result[b] = flipped_mask.float() * padded + (1 - flipped_mask.float()) * registers.to(padded)

        return result

    def _prepare_trans_mat(self) -> ttnn.Tensor:
        """Cached per-tile rotation matrix for rotary_embedding_llama (shared builder)."""
        if self._cached_trans_mat is None:
            self._cached_trans_mat = prepare_rot_transformation_mat(self.mesh_device)
        return self._cached_trans_mat
