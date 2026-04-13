# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Embedding layer implementations for TTNN."""

import os

from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch, tree_map
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import (
    DistributedTensorConfig,
    trace_enabled,
)
from models.experimental.tt_symbiote.modules.decoder_layer import _next_power_of_2

# Bailing rotary: long padded sequences make ``matmul @ position_ids`` and the following ``transpose`` large;
# chunk along sequence to cap peak DRAM (see :meth:`TTNNRotaryEmbeddingCompute.forward`).
_BAILING_ROPE_MATMUL_SEQ_CHUNK = max(32, int(os.environ.get("TT_SYMBIOTE_BAILING_ROPE_MATMUL_SEQ_CHUNK", "1024")))


def _aligned_up(x: int, alignment: int) -> int:
    return ((int(x) + alignment - 1) // alignment) * alignment


def _pad_last_dim_row_major(tensor, pad_amount: int, *, value):
    """Right-pad the last dimension (row-major)."""
    rank = len(tensor.shape)
    last = rank - 1
    padding = tuple((0, pad_amount if i == last else 0) for i in range(rank))
    return ttnn.pad(tensor, padding=padding, value=value)


def _slice_dim(tensor, dim: int, length: int):
    starts = [0] * len(tensor.shape)
    ends = list(tensor.shape)
    ends[dim] = length
    return ttnn.slice(tensor, starts, ends)


def _prepare_embedding_indices(tt_indices):
    """Return ``(indices_uint32_or_unchanged, orig_seq_len)`` for ``ttnn.embedding``.

    ``ttnn.embedding`` requires indices in **UINT32** or BFLOAT16; symbiote maps ``torch.long`` → INT32.
    ``ttnn.typecast`` to UINT32 on row-major INT32 requires the **last dim** to be a multiple of 32;
    we pad with 0, then the caller must slice the **embedding output** on the sequence dim when
    ``orig_seq_len`` is not ``None``.
    """
    if not isinstance(tt_indices, ttnn.Tensor):
        return tt_indices, None
    dt = tt_indices.dtype
    if dt == ttnn.uint32 or dt == ttnn.bfloat16:
        return tt_indices, None

    rank = len(tt_indices.shape)
    seq_dim = rank - 1
    seq_len = int(tt_indices.shape[seq_dim])
    padded_len = _aligned_up(seq_len, 32)
    orig_seq_len = None
    if padded_len != seq_len:
        tt_indices = _pad_last_dim_row_major(tt_indices, padded_len - seq_len, value=0)
        orig_seq_len = seq_len

    tt_indices = ttnn.typecast(tt_indices, ttnn.uint32)
    return tt_indices, orig_seq_len


def _maybe_slice_embedding_output(out, orig_seq_len):
    """Undo sequence padding: embedding output is ``[..., seq, hidden]``."""
    if orig_seq_len is None:
        return out
    rank = len(out.shape)
    seq_dim = rank - 2
    return _slice_dim(out, seq_dim, orig_seq_len)


@trace_enabled
class TTNNEmbedding(TTNNModule):
    """TTNN-accelerated embedding lookup for Ling/Bailing-style models.

    Replaces ``nn.Embedding``. On a multi-device mesh, **weights are sharded along the hidden
    dimension** (``ShardTensor2dMesh``), so each device holds a slice of the embedding table—appropriate
    when downstream layers consume column-sharded activations.

    :class:`TTNNQwen3OmniMoeCodecPredictorEmbedding` extends this with UINT32 index handling and
    optional ``padding_idx`` for ``codec_embedding``; weights stay **hidden-sharded** like here so
    downstream column-parallel layers match.
    """

    @classmethod
    def from_torch(cls, embedding: nn.Embedding):
        new_layer = cls()
        new_layer._fallback_torch_layer = embedding
        return new_layer

    def preprocess_weights_impl(self):
        self.tt_weight_host = ttnn.from_torch(
            self.torch_layer.weight.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 1), mesh_shape=list(self.device.shape))
        self.tt_weight = ttnn.to_device(
            ttnn.from_torch(
                self.torch_layer.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            ),
            self.device,
        )

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        super().deallocate_weights_impl()

    def forward(self, tt_indices):
        out = ttnn.embedding(
            tt_indices,
            self.tt_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out


@trace_enabled
class TTNNQwen3OmniMoeCodecPredictorEmbedding(TTNNEmbedding):
    """``nn.Embedding`` slots in ``Qwen3OmniMoeTalkerCodePredictorModel.codec_embedding`` (``ModuleList``).

    HF builds ``nn.ModuleList([nn.Embedding(vocab_size, hidden_size), ...])`` for
    ``num_code_groups - 1`` groups. Weights use the same **hidden-dim sharding** as
    :class:`TTNNEmbedding` so activations align with :class:`~models.experimental.tt_symbiote.modules.normalization.TTNNDistributedRMSNorm`
    and the rest of the talker decoder on mesh. This class adds ``ttnn.embedding``-compatible index
    prep (UINT32 / sequence padding) and optional ``padding_idx``.

    On a mesh, outputs must declare **column-sharded** last-dim metadata (``ShardTensorToMesh`` /
    ``ConcatMeshToTensor(dim=-1)``) like decoder norms and attention. The default
    ``DistributedConfig`` uses 2-D mesh compose; without this override, host readback can mis-compose
    shards and corrupt codec hidden states (audible TTS glitches that worsen on longer generations).
    """

    @property
    def weight(self):
        return self.torch_layer.weight

    @property
    def padding_idx(self):
        return self.torch_layer.padding_idx

    def set_output_tensors_config_impl(self, output_tensors):
        """Match :class:`~models.experimental.tt_symbiote.modules.normalization.TTNNDistributedRMSNorm` col-shard layout."""

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None and self.device is not None:
                if self.device.get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        shape_list[-1] = shape_list[-1] * self.device.get_num_devices()
                        return tuple(shape_list)

                    e.set_distributed_tensor_config(
                        DistributedTensorConfig(
                            mesh_mapper=mesh_mapper,
                            mesh_composer=mesh_composer,
                            logical_shape_fn=logical_shape_for_col_sharded,
                        )
                    )
            return e

        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

    def forward(self, tt_indices):
        tt_indices, orig_seq_len = _prepare_embedding_indices(tt_indices)
        pad = self.torch_layer.padding_idx
        pad_token = int(pad) if pad is not None and int(pad) >= 0 else None
        if pad_token is None:
            out = ttnn.embedding(
                tt_indices,
                self.tt_weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            out = ttnn.embedding(
                tt_indices,
                self.tt_weight,
                padding_idx=pad_token,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _maybe_slice_embedding_output(out, orig_seq_len)


class TTNNBailingPaddedEmbedding(TTNNModule):
    """Padded embedding wrapper that pads sequence to power-of-2 before lookup.

    Weight is sharded along hidden dim across mesh devices. Indices are
    replicated. Each device produces a slice of the hidden dimension.
    """

    @staticmethod
    def _pad_dim(tensor, dim, pad_amount, value=0.0):
        """Pad a single dimension of a tensor by ``pad_amount``."""
        rank = len(tensor.shape)
        padding = tuple((0, pad_amount if i == dim else 0) for i in range(rank))
        return ttnn.pad(tensor, padding=padding, value=value)

    @staticmethod
    def _slice_dim(tensor, dim, length):
        """Slice a tensor along ``dim`` to ``length``."""
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        ends[dim] = length
        return ttnn.slice(tensor, starts, ends)

    @classmethod
    def from_torch(cls, embedding: nn.Embedding):
        new_layer = cls()
        new_layer.embedder = TTNNEmbedding.from_torch(embedding)
        return new_layer

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_ids):
        rank = len(input_ids.shape)
        seq_dim = rank - 1  # sequence length is always second-to-last
        seq_len = input_ids.shape[seq_dim]
        padded_seq_len = _next_power_of_2(seq_len)
        pad_amount = padded_seq_len - seq_len
        if pad_amount > 0:
            input_ids = self._pad_dim(input_ids, seq_dim, pad_amount, value=0)

        input_ids = self.embedder(input_ids)
        if pad_amount > 0:
            input_ids = self._slice_dim(input_ids, seq_dim, seq_len)

        return input_ids


@trace_enabled
class TTNNRotaryEmbeddingCompute(TTNNModule):
    """Computes cos/sin from inv_freq and position_ids."""

    @staticmethod
    def from_torch(rotary_emb):
        new_layer = TTNNRotaryEmbeddingCompute()
        new_layer._fallback_torch_layer = rotary_emb
        return new_layer

    def forward(self, tt_inv_freq, position_ids):
        # Typecast int32 -> bfloat16 (requires last dim multiple of 32)
        position_ids = ttnn.typecast(position_ids, ttnn.bfloat16)
        position_ids = ttnn.to_layout(position_ids, ttnn.TILE_LAYOUT)

        # Reshape [batch, padded_seq] -> [batch, 1, padded_seq]
        position_ids = ttnn.reshape(position_ids, (position_ids.shape[0], 1, -1))
        b = int(position_ids.shape[0])
        seq = int(position_ids.shape[2])
        chunk = _BAILING_ROPE_MATMUL_SEQ_CHUNK

        def _cos_sin_from_emb(emb):
            c = ttnn.cos(emb)
            s = ttnn.sin(emb)
            if self.torch_layer.attention_scaling != 1.0:
                c = ttnn.multiply(c, self.torch_layer.attention_scaling)
                s = ttnn.multiply(s, self.torch_layer.attention_scaling)
            return c, s

        if seq <= chunk:
            # freqs = inv_freq @ position_ids -> [batch, rotary_dim/2, seq] -> transpose -> [batch, seq, rotary_dim/2]
            freqs = ttnn.matmul(tt_inv_freq, position_ids)
            freqs = ttnn.transpose(freqs, -2, -1)
            emb = ttnn.concat([freqs, freqs], dim=-1)
            return _cos_sin_from_emb(emb)

        cos_parts = []
        sin_parts = []
        for s0 in range(0, seq, chunk):
            s1 = min(s0 + chunk, seq)
            pos_sl = ttnn.slice(position_ids, [0, 0, s0], [b, 1, s1])
            freqs = ttnn.matmul(tt_inv_freq, pos_sl)
            freqs = ttnn.transpose(freqs, -2, -1)
            emb = ttnn.concat([freqs, freqs], dim=-1)
            c, s = _cos_sin_from_emb(emb)
            cos_parts.append(c)
            sin_parts.append(s)

        cos = cos_parts[0]
        sin = sin_parts[0]
        for i in range(1, len(cos_parts)):
            cos = ttnn.concat([cos, cos_parts[i]], dim=1)
            sin = ttnn.concat([sin, sin_parts[i]], dim=1)
        return cos, sin


class TTNNBailingRotaryEmbedding(TTNNModule):
    """TTNN-accelerated rotary position embedding for Ling/Bailing models.

    Replaces BailingMoeV2RotaryEmbedding. Pre-computes cos/sin caches on device
    via BailingRotarySetup. Returns torch cos/sin in HF doubled-half format —
    the downstream TTNNBailingMoEAttention handles format conversion internally.
    """

    @classmethod
    def from_torch(cls, rotary_emb):
        new_layer = cls()
        new_layer._fallback_torch_layer = rotary_emb
        config = rotary_emb.config
        new_layer._head_dim = config.head_dim
        new_layer._rope_theta = config.rope_theta
        new_layer._partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        new_layer._max_seq_len = config.max_position_embeddings
        new_layer._attention_scaling = rotary_emb.attention_scaling
        new_layer._inv_freq_torch = rotary_emb.inv_freq.float()  # to be populated in preprocess_weights
        new_layer.rotary_compute = TTNNRotaryEmbeddingCompute.from_torch(rotary_emb)
        return new_layer

    @staticmethod
    def _pad_dim(tensor, dim, pad_amount, value=0.0):
        """Pad a single dimension of a tensor by ``pad_amount``."""
        rank = len(tensor.shape)
        padding = tuple((0, pad_amount if i == dim else 0) for i in range(rank))
        return ttnn.pad(tensor, padding=padding, value=value)

    @staticmethod
    def _slice_dim(tensor, dim, length):
        """Slice a tensor along ``dim`` to ``length``."""
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        ends[dim] = length
        return ttnn.slice(tensor, starts, ends)

    def set_output_tensors_config_impl(self, output_tensors):
        """Override to use replicated config — cos/sin are replicated across all devices."""
        from models.experimental.tt_symbiote.core.module import set_distributed_tensor_config
        from models.experimental.tt_symbiote.core.utils import tree_map

        replicated_config = DistributedTensorConfig(
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        return tree_map(set_distributed_tensor_config(replicated_config), output_tensors)

    def move_weights_to_device_impl(self):
        inv_freq_expanded = self._inv_freq_torch[None, :, None]
        self.tt_inv_freq = ttnn.from_torch(
            inv_freq_expanded,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states: ttnn.Tensor, position_ids: ttnn.Tensor):
        # Replicate HF BailingMoeV2RotaryEmbedding.forward in TTNN:
        #   freqs = inv_freq_expanded @ position_ids_expanded  -> [batch, seq, rotary_dim/2]
        #   emb = cat(freqs, freqs)  -> [batch, seq, rotary_dim]
        #   cos = emb.cos() * attention_scaling
        #   sin = emb.sin() * attention_scaling
        # Now position_ids is torch [batch, seq]
        batch = position_ids.shape[0]
        seq_len = position_ids.shape[-1]
        padded_seq_len = max(_next_power_of_2(seq_len), 32)
        pad_amount = padded_seq_len - seq_len
        if pad_amount > 0:
            position_ids = self._pad_dim(position_ids, 1, pad_amount, value=0)

        cos, sin = self.rotary_compute.forward(self.tt_inv_freq, position_ids)

        # Slice back to original seq_len: [batch, 1, seq_len]
        if pad_amount > 0:
            cos = self._slice_dim(cos, 1, seq_len)
            sin = self._slice_dim(sin, 1, seq_len)
        return cos, sin
