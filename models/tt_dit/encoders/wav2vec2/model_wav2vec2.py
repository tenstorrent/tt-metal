# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import conv_pad_in_channels
from ...utils.tensor import from_torch, to_torch
from .encoder_wav2vec2 import Wav2Vec2EncoderStack, Wav2Vec2FeatureProjection
from .feature_extractor_wav2vec2 import Wav2Vec2FeatureExtractor


class Wav2Vec2Config:
    """
    Configuration for Wav2Vec2Encoder, mirroring the HuggingFace
    `facebook/wav2vec2-base-960h` model architecture used by the WAN 2.2 S2V
    pipeline's reference AudioEncoder.

    Args:
        hidden_size: Transformer-encoder hidden dimension (768 for base, 1024 for large).
        num_hidden_layers: Number of transformer encoder layers (12 for base, 24 for large).
        num_attention_heads: Number of self-attention heads.
        intermediate_size: FFN inner dimension.
        conv_dim: Per-conv-layer output channel sizes for the feature extractor.
        conv_stride: Strides for each conv layer.
        conv_kernel: Kernel sizes for each conv layer.
        conv_bias: Whether the conv layers use bias.
        feat_extract_norm: "group" (only first layer has GroupNorm) or "layer"
            (every layer has LayerNorm). "group" is the base default.
        num_conv_pos_embeddings: Kernel size of the positional convolution applied
            in the encoder before the transformer stack.
        num_conv_pos_embedding_groups: Number of groups in that positional conv.
        layer_norm_eps: Eps for all LayerNorms.
        hidden_act: Activation for the FFN ("gelu").
        feat_proj_layer_norm: Whether feature_projection applies LayerNorm before
            the Linear projection. True for both base and large.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: tuple = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2),
        conv_bias: bool = False,
        feat_extract_norm: str = "group",
        num_conv_pos_embeddings: int = 128,
        num_conv_pos_embedding_groups: int = 16,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "gelu",
        feat_proj_layer_norm: bool = True,
        do_stable_layer_norm: bool = False,
    ):
        assert len(conv_dim) == len(conv_stride) == len(conv_kernel)
        assert feat_extract_norm in ("group", "layer")
        assert hidden_act == "gelu"

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = tuple(conv_dim)
        self.conv_stride = tuple(conv_stride)
        self.conv_kernel = tuple(conv_kernel)
        self.conv_bias = conv_bias
        self.feat_extract_norm = feat_extract_norm
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.feat_proj_layer_norm = feat_proj_layer_norm
        # `wav2vec2-base-960h`: False (post-LN).  `wav2vec2-large-xlsr-53`: True (pre-LN).
        self.do_stable_layer_norm = do_stable_layer_norm

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, hf_config) -> "Wav2Vec2Config":
        return cls(
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            conv_dim=tuple(hf_config.conv_dim),
            conv_stride=tuple(hf_config.conv_stride),
            conv_kernel=tuple(hf_config.conv_kernel),
            conv_bias=bool(hf_config.conv_bias),
            feat_extract_norm=hf_config.feat_extract_norm,
            num_conv_pos_embeddings=hf_config.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=hf_config.num_conv_pos_embedding_groups,
            layer_norm_eps=hf_config.layer_norm_eps,
            hidden_act=hf_config.hidden_act,
            feat_proj_layer_norm=getattr(hf_config, "feat_proj_layer_norm", True),
            do_stable_layer_norm=getattr(hf_config, "do_stable_layer_norm", False),
        )


class Wav2Vec2Encoder(Module):
    """TTNN port of HuggingFace `Wav2Vec2Model` for the WAN 2.2 S2V pipeline.

    On device:
        feature_extractor: 7 Conv1d layers (``ttnn.experimental.conv3d`` with
            ``kernel_size=(k, 1, 1)``; GroupNorm-as-InstanceNorm on layer 0).
        feature_projection: LayerNorm + Linear(512 -> 768).
        encoder.layers: 12 transformer encoder layers.

    On CPU (one short device→host→device hop per audio clip):
        encoder.pos_conv_embed: grouped Conv1d (groups=16, kernel=128).
        encoder.layer_norm: initial LayerNorm before the transformer stack.

    Why pos-conv is on CPU: ``ttnn.experimental.conv3d``'s grouped path
    requires the per-group in_channels to equal ``C_in_block``, which itself
    must be a multiple of TILE_WIDTH (32). With ``in_per_group = 768 / 16 = 48``
    no valid ``C_in_block`` satisfies both constraints. The initial LayerNorm
    runs on CPU too because it directly consumes the pos-conv output. Both ops
    are tiny (~3M FLOPs total) and run once per audio clip outside the denoise
    loop, so the round-trip cost is negligible.

    Weight loading: call ``load_torch_state_dict(hf_model.state_dict())`` and
    ``bind_cpu_modules(hf_model)``.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.feature_extractor = Wav2Vec2FeatureExtractor(config, mesh_device=mesh_device)
        self.feature_projection = Wav2Vec2FeatureProjection(config, mesh_device=mesh_device)
        self.encoder = Wav2Vec2EncoderStack(
            config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # CPU-only modules — populated by `bind_cpu_modules`.
        self._cpu_pos_conv_embed: torch.nn.Module | None = None
        self._cpu_pre_layer_norm: torch.nn.Module | None = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # `masked_spec_embed` is a training-time buffer for SpecAugment.
        # ``encoder.pos_conv_embed.*`` is loaded via ``bind_cpu_modules``.
        # ``encoder.layer_norm.*`` is loaded via ``bind_cpu_modules`` in
        # post-LN mode, but in stable-LN mode it lives in the device stack
        # under ``encoder.layer_norm.*`` — let those keys flow through.
        state.pop("masked_spec_embed", None)
        for k in list(state):
            if k.startswith("encoder.pos_conv_embed."):
                state.pop(k)
            elif k.startswith("encoder.layer_norm.") and not self.config.do_stable_layer_norm:
                state.pop(k)

    def bind_cpu_modules(self, hf_model: torch.nn.Module) -> None:
        """Snapshot the HF modules used on host.

        In post-LN mode (wav2vec2-base), both ``encoder.pos_conv_embed`` and
        ``encoder.layer_norm`` are needed on host. In stable mode
        (wav2vec2-large-xlsr), only ``encoder.pos_conv_embed`` is used on host;
        the final ``encoder.layer_norm`` runs on device inside the stack.
        """
        self._cpu_pos_conv_embed = hf_model.encoder.pos_conv_embed.eval()
        if not self.config.do_stable_layer_norm:
            self._cpu_pre_layer_norm = hf_model.encoder.layer_norm.eval()

    # Chunking parameters for the feature extractor. The 7-conv stack is
    # stateless and has a small receptive field (~720 input samples / ~2.3
    # output features), so overlap-and-trim chunks produce output that's
    # numerically equivalent to the single-call path within the trim margin.
    # Tile-aligned overlap (32 output features) gives a clean trim slice.
    _AUDIO_SR = 16000
    _FE_STRIDE = 320  # cumulative stride of the 7 conv layers
    _CHUNK_BODY_SEC = 5.0  # ~5s body per chunk
    _CHUNK_OVERLAP_FEATURES = 32  # tile-aligned overlap each side (~0.64s of input)

    def forward(
        self,
        input_values_torch: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        """Run the encoder end-to-end on device.

        Args:
            input_values_torch: CPU `torch.Tensor` shaped `[B, T_raw]` or
                `[B, T_raw, 1]`, already normalized by `Wav2Vec2Processor`.
            output_hidden_states: If `True`, return a list of
                `num_hidden_layers + 1` tensors. Matches HF's
                `output_hidden_states=True`.
        """
        x = input_values_torch.squeeze(-1) if input_values_torch.dim() == 3 else input_values_torch
        B, T_raw = x.shape

        # The feature extractor's intermediate LayerNorm allocations grow
        # linearly with T_raw and run in fp32 for the layer-norm variant
        # (wav2vec2-large-xlsr); at ~22 s audio the peak DRAM blows past the
        # per-chip bank limit. Chunk the feature extractor in time for long
        # audio so the peak only depends on chunk size, not total duration.
        # Subsequent steps (pos_conv_embed, transformer) run on the
        # concatenated 50 Hz features so global self-attention is preserved.
        chunk_body_samples = int(self._CHUNK_BODY_SEC * self._AUDIO_SR)
        overlap_samples = self._CHUNK_OVERLAP_FEATURES * self._FE_STRIDE
        chunk_threshold = chunk_body_samples + 2 * overlap_samples

        if T_raw <= chunk_threshold:
            feats = self._run_feature_extractor_single(x, B, T_raw)
        else:
            feats = self._run_feature_extractor_chunked(x, B, T_raw, chunk_body_samples, overlap_samples)

        # Boundary cast: the feature_projection's Linear+LayerNorm carries
        # bf16 weights, so the fp32 feature extractor's output is cast down
        # before entering the bf16 transformer trunk.
        if feats.dtype != ttnn.bfloat16:
            feats = ttnn.typecast(feats, ttnn.bfloat16)

        # Feature projection (LayerNorm + Linear(512 -> 768)) — on device.
        projected = self.feature_projection(feats)

        # Pos-conv — always on CPU. Initial LayerNorm placement depends on
        # the variant:
        #   * post-LN (base): applied here on CPU before the stack.
        #   * pre-LN / stable (large-xlsr): NOT applied here — owned by
        #     `Wav2Vec2EncoderStack.layer_norm` and run on device after the stack.
        if self._cpu_pos_conv_embed is None:
            raise RuntimeError("bind_cpu_modules(hf_model) was not called.")
        projected_torch = to_torch(projected).float().reshape(B, -1, self.config.hidden_size)
        with torch.no_grad():
            pos = self._cpu_pos_conv_embed(projected_torch)
            hidden_torch = projected_torch + pos
            if not self.config.do_stable_layer_norm:
                if self._cpu_pre_layer_norm is None:
                    raise RuntimeError("post-LN variant requires bound _cpu_pre_layer_norm")
                hidden_torch = self._cpu_pre_layer_norm(hidden_torch)
        hidden = from_torch(
            hidden_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Transformer encoder layers on device. In stable mode the stack also
        # applies the final layer_norm on device.
        return self.encoder(hidden, output_hidden_states=output_hidden_states)

    def _run_feature_extractor_single(self, x_BT: torch.Tensor, B: int, T_raw: int) -> ttnn.Tensor:
        """Single-call feature extractor for short audio.

        Returns `[B, T_out, 512]` in TILE layout, dtype matches feature_extractor.dtype.
        """
        x_BTHWC = x_BT.reshape(B, T_raw, 1, 1, 1)
        x_BTHWC = conv_pad_in_channels(x_BTHWC)
        audio_dev = from_torch(
            x_BTHWC,
            device=self.mesh_device,
            dtype=self.feature_extractor.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        feats = self.feature_extractor(audio_dev)  # [B, T_out, 1, 1, 512] TILE
        return ttnn.reshape(feats, (B, feats.shape[1], self.config.conv_dim[-1]))

    def _run_feature_extractor_chunked(
        self,
        x_BT: torch.Tensor,
        B: int,
        T_raw: int,
        chunk_body_samples: int,
        overlap_samples: int,
    ) -> ttnn.Tensor:
        """Run the feature extractor in overlapping chunks; concat on device.

        Each chunk reads `body + overlap` samples on each side (no left overlap
        for the first chunk, no right overlap for the last) and the corresponding
        output overlap features (`overlap_samples / FE_STRIDE`) are trimmed
        before concatenation. Slice + concat happen on device — chunks never
        round-trip to host until the full sequence is assembled and handed to
        the next stage. Overlap is tile-aligned (32 output features) so the
        slice stays cheap.

        Returns `[B, T_out, 512]` in TILE layout, dtype matches feature_extractor
        (fp32 for layer-norm variants, bf16 otherwise). The total T_out may
        differ from the single-call output_length by a few features due to
        per-chunk conv boundary loss; downstream linear_interpolation absorbs
        this rounding.
        """
        overlap_features = overlap_samples // self._FE_STRIDE
        chunk_feats_list: list[ttnn.Tensor] = []
        last_dim = self.config.conv_dim[-1]

        pos = 0
        while pos < T_raw:
            body_end = min(pos + chunk_body_samples, T_raw)
            left_ovl = overlap_samples if pos > 0 else 0
            right_ovl = overlap_samples if body_end < T_raw else 0
            chunk = x_BT[:, pos - left_ovl : body_end + right_ovl]

            chunk_BTHWC = chunk.reshape(B, chunk.shape[1], 1, 1, 1)
            chunk_BTHWC = conv_pad_in_channels(chunk_BTHWC)
            chunk_dev = from_torch(
                chunk_BTHWC,
                device=self.mesh_device,
                dtype=self.feature_extractor.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            chunk_feats_tt = self.feature_extractor(chunk_dev)  # [B, T_chunk_out, 1, 1, 512] TILE
            # Collapse degenerate (1, 1) dims so T is the height axis; downstream
            # ops and the single-call return shape both use [B, T, 512].
            t_chunk_out = chunk_feats_tt.shape[1]
            chunk_feats_BTC = ttnn.reshape(chunk_feats_tt, (B, t_chunk_out, last_dim))

            # Trim overlap features on device. Slicing along T (the height axis)
            # is tile-free for outer dims and we kept the overlap a multiple of
            # 32 features so tile alignment isn't an issue either.
            left_trim = overlap_features if pos > 0 else 0
            right_trim = overlap_features if body_end < T_raw else 0
            if left_trim > 0 or right_trim > 0:
                t_end = t_chunk_out - right_trim
                chunk_feats_BTC = chunk_feats_BTC[:, left_trim:t_end, :]
            chunk_feats_list.append(chunk_feats_BTC)

            ttnn.deallocate(chunk_dev)
            pos = body_end

        # On-device concat along T. ttnn.concat preserves the TILE layout and
        # dtype of the inputs; the typecast in forward() handles fp32 → bf16
        # for the feature_projection.
        feats = ttnn.concat(chunk_feats_list, dim=1)
        for c in chunk_feats_list:
            ttnn.deallocate(c)
        return feats
