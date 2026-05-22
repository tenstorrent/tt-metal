# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import conv_pad_in_channels
from ...utils.tensor import from_torch
from .config_wav2vec2 import Wav2Vec2Config
from .encoder_wav2vec2 import Wav2Vec2EncoderStack, Wav2Vec2FeatureProjection
from .feature_extractor_wav2vec2 import Wav2Vec2FeatureExtractor


class Wav2Vec2Encoder(Module):
    """TTNN port of HuggingFace ``Wav2Vec2Model`` for the WAN 2.2 S2V pipeline.

    Fully on-device: feature_extractor (7 Conv1d), feature_projection, then the
    encoder stack which owns the grouped-Conv1d pos_conv_embed and the N
    transformer layers. Only ``do_stable_layer_norm=True`` (large-xlsr-53) is
    supported today; post-LN variants need the encoder pre-LN moved on device.
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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # masked_spec_embed is a training-time buffer for SpecAugment.
        state.pop("masked_spec_embed", None)

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

        # Feature projection (LayerNorm + Linear(512 -> 768)) on device, then
        # the encoder stack — pos_conv + residual + transformer layers + final LN.
        projected = self.feature_projection(feats)
        return self.encoder(projected, output_hidden_states=output_hidden_states)

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
