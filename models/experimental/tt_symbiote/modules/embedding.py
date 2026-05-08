# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Embedding layer implementations for TTNN."""

import torch
from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    run_on_devices,
    DeviceArch,
    set_distributed_tensor_config,
)
from models.experimental.tt_symbiote.core.run_config import (
    DistributedTensorConfig,
    trace_enabled,
)
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.decoder_layer import _next_power_of_2


@trace_enabled
class TTNNEmbedding(TTNNModule):
    """TTNN-accelerated embedding lookup for Ling/Bailing/Qwen models.

    Replaces ``nn.Embedding`` (typically the model's ``embed_tokens``). The
    weight is sharded along the hidden dimension across the mesh's column
    axis (``ShardTensor2dMesh(dims=(None, 1))``), so each device owns a
    ``[V, H/num_devices]`` slice. With **replicated** indices (every device
    sees the full ``[B, S]`` int tensor) every device looks up the same rows
    in its own slice and the per-device output is ``[B, S, H/num_devices]`` --
    exactly the col-sharded layout that the downstream Qwen ``input_layernorm``
    / ``self_attn`` / ``mlp`` chain expects.

    Two correctness gates that have to hold:

    1. **Indices must reach this op replicated.** The dispatcher's
       ``get_tensor_config_for_tensor`` heuristic uses the default
       ``ShardTensor2dMesh((0, -1))`` config for any 2D tensor whose last dim
       is divisible by ``mesh.shape[-1]`` -- so ``input_ids`` of shape
       ``[1, T]`` with ``T`` divisible by ``num_devices`` ends up sharded
       along the **seq** dim, and each device sees only ``T/num_devices``
       tokens of the prompt. The embedding lookup then produces a
       ``[1, T/num_devices, H]`` tensor that's far too short, and the model
       generates fluent but completely off-topic text (e.g. a paper on
       climate change in response to a question about condiments). We
       intercept in ``call`` (overridden below) and pre-stage every
       integer-dtype torch input as a ``ReplicateTensorToMesh`` ttnn.Tensor
       *before* the dispatcher's wrap/transform pipeline runs, sidestepping
       the heuristic entirely.

    2. **The output config must describe what the data actually is.** The
       default ``set_output_tensors_config_impl`` falls back to the device's
       generic ``ShardTensor2dMesh((0, -1))`` config, which works in this case
       but disagrees with the 1D-API ``ShardTensorToMesh(dim=-1)`` config that
       ``TTNNQwen3FullAttention`` / ``TTNNQwen3MoE`` set on their own outputs.
       We override below to set the same 1D-API config so a downstream
       layout-aware op (e.g. ``TTNNQwen3MoeDecoderLayer._residual_add``) sees
       a single consistent layout for every operand in the residual stream.

    Index dtype handling: ``ttnn.embedding`` only accepts ``UINT32`` /
    ``BFLOAT16``. Most callers pre-convert via
    ``ttnn.from_torch(..., dtype=ttnn.uint32)`` before calling this op, but
    when the wrapper is registered directly (as for Qwen3.6) the HF text
    model hands us raw ``input_ids`` whose default TTNN dtype is ``INT32``.
    ``_ensure_uint32`` covers that case with a cached pad-typecast-slice
    (``ttnn.typecast`` requires ``padded_shape[-1] % 32 == 0``); the path is
    a no-op when the input is already ``UINT32`` / ``BFLOAT16``, so existing
    Bailing / Gemma4 callers are unaffected.
    """

    @classmethod
    def from_torch(cls, embedding: nn.Embedding, scale_factor=None):
        new_layer = cls()
        new_layer._fallback_torch_layer = embedding
        new_layer._scale_factor = scale_factor
        # Cache: (leading-dim shape, pad_amount) -> pre-allocated zero buffer
        # used to pad the index tensor to a multiple of 32 for `ttnn.typecast`.
        # Populated lazily on the first non-UINT32 forward (warm-up runs
        # outside trace capture, so the allocation is safe; subsequent capture
        # / replay re-uses the cached buffer with no allocation).
        new_layer._typecast_pad_buffers = {}
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

    def call(self, *args, **kwds):
        """Pre-stage integer index inputs as REPLICATED before the dispatcher sees them.

        This is the actual fix for the off-topic-generation symptom that bit
        Qwen3.6: the dispatcher's ``get_tensor_config_for_tensor`` checks
        ``tensor.shape[-1] % mesh_device.shape[-1]`` -- when ``input_ids``
        happens to be ``[1, T]`` with ``T`` divisible by ``num_devices`` (the
        common case once a chat-template prompt + a few generated tokens add
        up), it returns the default ``ShardTensor2dMesh((0, -1))`` config
        which then **shards the prompt along the seq dim**: each device sees
        only its ``T/num_devices`` slice of tokens. The embedding lookup then
        produces a per-device output of length ``T/num_devices`` and the
        composer assembles a [B, T/num_devices, H] tensor -- the model sees
        only 1/8 of the real prompt and confidently generates fluent but
        completely off-topic text.

        We can't change the dispatcher's heuristic without touching every
        other test in the repo, so we shortcut it here: re-stage every
        integer-dtype torch input as a ``ReplicateTensorToMesh`` ttnn.Tensor
        before calling the standard module pipeline. Once the input is
        already a ttnn.Tensor with an explicit replicated mesh layout,
        ``wrap_to_torch_ttnn_tensor`` preserves it and ``to_ttnn_wrap`` is a
        no-op (it just hands back the existing ttnn.Tensor). Float inputs
        (the embedding lookup never sees these in practice but be defensive)
        and any non-tensor args fall through unchanged.
        """
        args = tuple(self._maybe_pre_stage_replicated(a) for a in args)
        kwds = {k: self._maybe_pre_stage_replicated(v) for k, v in kwds.items()}
        return super().call(*args, **kwds)

    def _maybe_pre_stage_replicated(self, t):
        """Stage an integer-index torch tensor as a replicated on-device ttnn.Tensor.

        Only runs for genuine torch inputs whose dtype is integer (the index
        case). Already-staged ttnn / TorchTTNN tensors and float tensors are
        passed through unchanged so we don't disturb other call sites.
        """
        if self.device is None or self.device.get_num_devices() <= 1:
            return t

        # Late import to avoid a circular import via core/module.
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        # Skip anything already on device or wrapped (preserve caller intent).
        if isinstance(t, (ttnn.Tensor, TorchTTNNTensor)):
            return t
        if not isinstance(t, torch.Tensor):
            return t
        if torch.is_floating_point(t):
            return t
        if t.device.type == "meta":
            return t

        # Stage as int32 replicated -- ``_ensure_uint32`` later promotes to
        # uint32 inside ``forward`` since the kernel only accepts uint32 / bf16.
        return ttnn.from_torch(
            t.detach().to(torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded (along hidden) output.

        The per-device shape is ``[B, S, H/num_devices]`` and the logical
        shape is ``[B, S, H]`` (last dim multiplied by num_devices). Mirrors
        the config that ``TTNNQwen3FullAttention`` / ``TTNNQwen3MoE`` set on
        their own outputs, so a layout-aware downstream op sees a single
        consistent col-sharded layout for every tensor in the residual stream.
        Falls back to the base implementation on a single-device mesh.
        """
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        num_devices = self.device.get_num_devices()

        def _logical_shape_for_col_sharded(shape):
            shape_list = list(shape)
            shape_list[-1] = shape_list[-1] * num_devices
            return tuple(shape_list)

        config = DistributedTensorConfig(
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1),
            logical_shape_fn=_logical_shape_for_col_sharded,
        )
        return tree_map(set_distributed_tensor_config(config), output_tensors)

    def _ensure_uint32(self, tt_indices):
        """Convert ``tt_indices`` to UINT32 if it isn't already a kernel-compatible dtype.

        ``ttnn.embedding`` only accepts ``UINT32`` or ``BFLOAT16``. ``ttnn.typecast``
        requires the last dim padded to a multiple of 32, so we pad with a cached
        zero buffer, typecast on device, then slice back to the original length.
        Returns ``tt_indices`` unchanged when it's already a compatible dtype or
        not even a ``ttnn.Tensor`` (the kernel will then surface its own error).
        """
        if not isinstance(tt_indices, ttnn.Tensor):
            return tt_indices
        if tt_indices.dtype in (ttnn.uint32, ttnn.bfloat16):
            return tt_indices

        # ``ttnn._ttnn.types.Shape`` only supports integer indexing (no slicing
        # and no ``list(...)``), so materialize the dims as a Python tuple up
        # front and operate on that.
        rank = len(tt_indices.shape)
        shape_dims = tuple(int(tt_indices.shape[i]) for i in range(rank))
        orig_size = shape_dims[-1] if rank > 0 else 1
        pad_amount = (32 - orig_size % 32) % 32

        if pad_amount > 0:
            # Cache pad buffers keyed on (leading-dims, pad_amount) so trace
            # capture / replay never re-allocates -- prefill (variable seq_len)
            # gets one buffer per length and decode (seq_len=1) shares one.
            leading_dims = shape_dims[:-1] if rank > 1 else (1,)
            cache_key = leading_dims + (pad_amount,)
            if cache_key not in self._typecast_pad_buffers:
                pad_torch = torch.zeros(*cache_key, dtype=torch.int32)
                mesh_mapper = (
                    ttnn.ReplicateTensorToMesh(self.device)
                    if self.device is not None and self.device.get_num_devices() > 1
                    else None
                )
                self._typecast_pad_buffers[cache_key] = ttnn.from_torch(
                    pad_torch,
                    device=self.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
            tt_indices = ttnn.concat([tt_indices, self._typecast_pad_buffers[cache_key]], dim=-1)

        tt_indices = ttnn.typecast(tt_indices, ttnn.uint32)

        if pad_amount > 0:
            # Re-materialize the post-typecast shape as a Python list for
            # ``ttnn.slice`` (which expects plain ints, not a Shape object).
            post_rank = len(tt_indices.shape)
            ends = [int(tt_indices.shape[i]) for i in range(post_rank)]
            starts = [0] * post_rank
            ends[-1] = orig_size
            tt_indices = ttnn.slice(tt_indices, starts, ends)
        return tt_indices

    def forward(self, tt_indices):
        # ``call`` (overridden above) has already re-staged the indices as a
        # replicated on-device int32 tensor; ``_ensure_uint32`` then promotes
        # int32 -> uint32 for the kernel via the cached pad/typecast/slice.
        tt_indices = self._ensure_uint32(tt_indices)
        out = ttnn.embedding(
            tt_indices,
            self.tt_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._scale_factor is not None:
            out = ttnn.multiply(out, self._scale_factor)
        return out


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

    @run_on_devices(DeviceArch.T3K, DeviceArch.QB2)
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

        # freqs = inv_freq @ position_ids -> [batch, rotary_dim/2, seq] -> transpose -> [batch, seq, rotary_dim/2]
        freqs = ttnn.matmul(tt_inv_freq, position_ids)
        freqs = ttnn.transpose(freqs, -2, -1)

        # emb = cat(freqs, freqs) -> [batch, seq, rotary_dim]
        emb = ttnn.concat([freqs, freqs], dim=-1)

        # cos/sin
        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)

        if self.torch_layer.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.torch_layer.attention_scaling)
            sin = ttnn.multiply(sin, self.torch_layer.attention_scaling)

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

    @run_on_devices(DeviceArch.T3K, DeviceArch.QB2)
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
