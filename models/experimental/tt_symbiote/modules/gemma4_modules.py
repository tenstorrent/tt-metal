# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 specific module implementations for TTNN.

Provides TTNN-accelerated versions of Gemma 4 architectural components:
- TTNNGemma4ScaledEmbedding: Scaled word embedding (multiplies by sqrt(hidden_size))
- TTNNGemma4FFN: GeGLU feed-forward network with gate/up/down projections
- TTNNGemma4DecoderLayer: Decoder layer wrapper for on-device residual adds and tracing
"""

import math

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIReplicatedWColSharded,
    TTNNLinearIColShardedWRowSharded,
    TTNNLinearIColShardedWAllReduced,
)
from models.experimental.tt_symbiote.modules.gemma4_attention import TTNNGemma4Attention
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


class TTNNGemma4ScaledEmbedding(TTNNModule):
    """TTNN-accelerated scaled word embedding for Gemma 4 models.

    Wraps TTNNEmbedding and multiplies the output by sqrt(hidden_size),
    replicating the behaviour of Gemma4TextScaledWordEmbedding.

    NOT trace-enabled: forward() converts torch input_ids via ttnn.from_torch
    which allocates new device memory (forbidden during trace capture).
    """

    @classmethod
    def from_torch(cls, embedding):
        """Create from a Gemma4TextScaledWordEmbedding instance.

        Args:
            embedding: HF Gemma4TextScaledWordEmbedding with an ``embed_scale``
                attribute and an underlying ``nn.Embedding`` weight.
        """
        new_layer = cls()
        new_layer._fallback_torch_layer = embedding
        new_layer.embedder = TTNNEmbedding.from_torch(embedding)
        new_layer.embed_scale = float(getattr(embedding, "embed_scale", math.sqrt(embedding.embedding_dim)))
        return new_layer

    def forward(self, tt_indices):
        # Ensure indices are UINT32 for the TTNN embedding op.
        # The framework may pass torch.Tensor or TTNN Tensor (INT32).
        # TTNN typecast requires last dim % 32 == 0, so we convert via host if needed.
        if isinstance(tt_indices, torch.Tensor):
            tt_indices = ttnn.from_torch(
                tt_indices.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            tt_indices = ttnn.to_device(tt_indices, self.device)
        elif isinstance(tt_indices, ttnn.Tensor) and tt_indices.dtype != ttnn.uint32:
            # Roundtrip through host to cast INT32 -> UINT32 without padding constraints
            torch_indices = ttnn.to_torch(tt_indices, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            # Take first device shard (all shards are identical for replicated input)
            if torch_indices.shape[0] > tt_indices.shape[0]:
                torch_indices = torch_indices[: tt_indices.shape[0]]
            tt_indices = ttnn.from_torch(
                torch_indices.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            tt_indices = ttnn.to_device(tt_indices, self.device)
        out = self.embedder(tt_indices)
        out = ttnn.multiply(out, self.embed_scale)
        return out


@trace_enabled
class TTNNGemma4FFN(TTNNModule):
    """TTNN-accelerated GeGLU feed-forward network for Gemma 4 models.

    Replaces Gemma4TextMLP which uses a gated architecture with separate
    gate_proj, up_proj, and down_proj linear layers with GELU activation
    on the gate path.
    """

    @classmethod
    def from_torch(cls, torch_mlp):
        """Create from a Gemma4TextMLP instance.

        Args:
            torch_mlp: HF Gemma4TextMLP with gate_proj, up_proj, down_proj
                (all nn.Linear) and act_fn attributes.
        """
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_mlp
        new_layer.gate_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.gate_proj)
        new_layer.up_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.up_proj)
        new_layer.down_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_mlp.down_proj)
        return new_layer

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        # Input arrives col-sharded from the decoder layer; gate/up expect replicated.
        # Use Ring topology for trace compatibility (Linear may allocate unpinned intermediates).
        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        gate = self.gate_proj(hidden_states)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        up = self.up_proj(hidden_states)
        intermediate = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        output = self.down_proj(intermediate)
        ttnn.deallocate(intermediate)
        return output


@trace_enabled
class TTNNGemma4DecoderLayer(TTNNModule):
    """Replaces Gemma4TextDecoderLayer to keep residual adds on-device.

    Eliminates host round-trips by using ttnn.add for residual connections.
    This is the primary trace boundary for Gemma4 — @trace_enabled captures
    the full op sequence (norms + attention + FFN) in a single trace.

    HF Gemma4TextDecoderLayer forward flow (31B, no MoE):
        input_layernorm -> self_attn -> post_attention_layernorm -> residual_add
        pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> residual_add
        layer_scalar multiplication
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None
        self.self_attn = None
        self.mlp = None
        self.layer_scalar = 1.0

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from Gemma4TextDecoderLayer."""
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer

        new_layer.input_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.pre_feedforward_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.pre_feedforward_layernorm)
        new_layer.post_feedforward_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_feedforward_layernorm)
        new_layer.self_attn = TTNNGemma4Attention.from_torch(torch_layer.self_attn)
        new_layer.mlp = TTNNGemma4FFN.from_torch(torch_layer.mlp)

        # layer_scalar is a learnable scalar (nn.Parameter) initialised to 1.0
        scalar = getattr(torch_layer, "layer_scalar", None)
        if scalar is not None:
            new_layer.layer_scalar = float(scalar.item() if hasattr(scalar, "item") else scalar)

        return new_layer

    def update_trace_stable_buffers(self, func_kwargs):
        """Copy trace-sensitive kwargs to module-owned persistent buffers.

        Called BEFORE execute_trace by TracedRun to avoid buffer aliasing:
        the trace kwarg buffer for cache_position can be overwritten by another
        layer's trace intermediates (addresses reused by TTNN's trace allocator).
        Copying to _decode_cur_pos here (outside the trace) ensures the value
        survives any intermediate overwrites during trace replay.

        IMPORTANT: The source tensor (func_kwargs['cache_position']) MUST be at a
        device address that is NOT aliased by any trace's intermediates. This is
        guaranteed when the caller pre-allocates the cache_position buffer BEFORE
        trace capture (so the trace allocator knows the address is in use and won't
        assign it to intermediates). If the caller creates a NEW tensor each step,
        the allocator may assign an address that overlaps with trace intermediates,
        causing the source to be overwritten before this copy reads it.
        """
        cache_position = func_kwargs.get("cache_position")
        if cache_position is None:
            return
        attn = self.self_attn
        if not hasattr(attn, "_decode_cur_pos") or attn._decode_cur_pos is None:
            return
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        cp = cache_position
        if isinstance(cp, TorchTTNNTensor):
            cp = cp.ttnn_tensor
        # Only copy during decode (shape [1]). Prefill (shape [N]) doesn't need
        # this fix — buffer aliasing only affects decode trace replays.
        if cp.shape != attn._decode_cur_pos.shape:
            return
        ttnn.copy(cp, attn._decode_cur_pos)

    def forward(
        self,
        hidden_states,
        per_layer_input=None,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        hs = hidden_states

        # Ensure TILE layout and bfloat16 for TTNN ops
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # ----- Attention block -----
        residual = hs

        hs = self.input_layernorm(hs)

        # Pass cache_position as explicit kwarg so TracedRun can
        # pre-allocate a device buffer for it (matches Ling pattern).
        seq_len = hs.shape[-2]
        is_decode = seq_len == 1
        attn_out, _ = self.self_attn(
            hidden_states=hs,
            position_embeddings=None,
            attention_mask=None if is_decode else attention_mask,
            past_key_values=past_key_values,
            cache_position=kwargs.get("cache_position"),
        )

        attn_out = self.post_attention_layernorm(attn_out)

        # Residual add ON DEVICE
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        # NOTE: Do NOT deallocate residual here — it is the pre-allocated trace
        # input buffer.  ttnn.deallocate inside a traced forward would be
        # replayed by execute_trace, freeing the buffer that
        # _copy_inputs_to_trace_buffer needs on the next replay iteration.

        # ----- FFN block -----
        residual = hs

        hs = self.pre_feedforward_layernorm(hs)
        mlp_out = self.mlp(hs)
        mlp_out = self.post_feedforward_layernorm(mlp_out)

        # Residual add ON DEVICE
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        # Layer scalar (1.0 in 31B — effectively a no-op)
        if self.layer_scalar != 1.0:
            hs = ttnn.multiply(hs, self.layer_scalar)

        # HF Gemma4TextDecoderLayer returns a plain tensor (not a tuple).
        return hs


class TTNNGemma4LMHead(TTNNModule):
    """TTNN-accelerated lm_head for Gemma 4 models.

    Replaces the nn.Linear lm_head (5120 -> 262144) which runs on CPU.
    Input arrives col-sharded from the final RMSNorm. Uses matmul + all_reduce
    to produce replicated output compatible with HF generate() sampling.

    Gemma4 uses tied embeddings (lm_head.weight == embed_tokens.weight).
    The from_torch method handles the weight regardless of tying.
    """

    @classmethod
    def from_torch(cls, linear):
        new_layer = cls()
        new_layer._fallback_torch_layer = linear
        new_layer.linear = TTNNLinearIColShardedWAllReduced.from_torch(linear)
        return new_layer

    def forward(self, hidden_states):
        return self.linear(hidden_states)
