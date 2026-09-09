# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One Llama-3.1-8B decoder layer: pre-norm attention + pre-norm MLP, each with a residual add.

Composition borrowed from `minimax_m3/tt/layer.py` (measured at hidden 6144, sp8×tp4, this mesh),
with the hybrid schedule removed: **all 32 Llama layers are identical**. There is no
`moe_layer_freq` lookup, no dense/sparse branch, no per-layer type dispatch anywhere in this
package — every layer is dense GQA attention plus a dense SwiGLU MLP.

## The residual stream and where the collectives sit

Under the default sharded residual (`tt/residual.py`) the residual carries `emb/tp` per TP column,
and full width is reconstituted only where a column-parallel projection needs it: **one all-gather
per norm output**, shared by every consumer below that norm. If each consumer gathered for itself
the layout would buy nothing. Attention and the MLP each close with a reduce-scatter back to
`emb/tp`, so the layer's adds are `emb/tp` + `emb/tp`.

Under the replicated residual the residual is full emb everywhere and both blocks close with an
all-reduce instead.
"""

import ttnn
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama_3_1_8b_d_p.utils.substate import substate

from .attention import Attention, AttentionConfig, LlamaAttentionProgramConfig
from .dense_mlp import DenseMLP
from .residual import gather_before_norm, use_sharded_residual
from .rms_norm import RMSNorm


def build_attention_config(hf_config, *, max_seq_len, chunk_size, max_local_batch_size=1, users_row_sharded=False):
    """`AttentionConfig` from the HF config plus the two spec shapes.

    `head_dim` is derived rather than read: this checkpoint's `config.json` does not carry it.
    """
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    return AttentionConfig(
        hidden_size=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        max_local_batch_size=max_local_batch_size,
        users_row_sharded=users_row_sharded,
        sequence_parallel=True,
    )


class DecoderLayer:
    """One decoder layer. Identical for every layer index — only the weights differ."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        mesh_config,
        *,
        max_seq_len,
        chunk_size,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        transformation_mats=None,
        max_local_batch_size=1,
        users_row_sharded=False,
        cache_layer_idx=None,
    ):
        """
        Args:
            layer_idx: GLOBAL index — selects this layer's checkpoint weights.
            cache_layer_idx: LOCAL index into the KV cache's layer packing. None => same as
                `layer_idx`. They differ only under pipeline parallelism.
            state_dict: this layer's substate (`self_attn.*`, `mlp.*`, the two norms), or `{}` for
                cache-only loading.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.layer_idx = layer_idx

        # Residual-stream layout. Decided once, here, and read by every block below.
        self.sharded_residual = use_sharded_residual() and mesh_config is not None and mesh_config.tp > 1
        self.gather_before_norm = self.sharded_residual and gather_before_norm()

        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )
        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=build_attention_config(
                hf_config,
                max_seq_len=max_seq_len,
                chunk_size=chunk_size,
                max_local_batch_size=max_local_batch_size,
                users_row_sharded=users_row_sharded,
            ),
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=LlamaAttentionProgramConfig(),
            global_layer_idx=layer_idx,
            local_layer_idx=cache_layer_idx,
            transformation_mats=transformation_mats,
            weight_dtype=weight_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
        )
        self.mlp = DenseMLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            weight_dtype=weight_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
        )

    def _gather_emb(self, tensor):
        """TP all-gather `emb/tp` -> full emb, through the MANAGED all_gather (ping-pong + barrier
        semaphores and the CCL manager's topology) rather than the raw prim."""
        return self.mesh_config.allgather(tensor, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)

    def _norm_to_full_emb(self, norm, hidden_states):
        """Normalize the residual and hand back FULL emb, gathering exactly once.

        Replicated residual: the input is already full emb, so this is just the norm. Sharded: one
        TP all-gather, sitting either before a single-pass norm (`gather_first`) or after a
        distributed one — the same single gather either way.
        """
        if not self.sharded_residual:
            return norm(hidden_states)
        if self.gather_before_norm:
            gathered = self._gather_emb(hidden_states)
            out = norm(gathered)
            gathered.deallocate(True)
            return out
        normed = norm(hidden_states)
        out = self._gather_emb(normed)
        normed.deallocate(True)
        return out

    def __call__(
        self,
        hidden_states,
        *,
        rope_mats,
        kv_cache=None,
        slot_idx=0,
        cached_len=0,
        logical_n=None,
        indexed_rope=False,
        write_chunk=True,
    ):
        """hidden_states `[1, 1, tokens_local, emb_or_emb_over_tp]` -> same shape.

        `cached_len` / `logical_n` / `indexed_rope` are the chunked-prefill contract and pass
        straight through to attention; a one-shot run leaves them at their defaults.
        """
        if hidden_states.shape[-2] > 32 * 1024:
            # Reallocate to keep long-context prefill from fragmenting DRAM.
            hidden_states = ttnn.move(hidden_states)

        residual = hidden_states
        normed = self._norm_to_full_emb(self.input_layernorm, hidden_states)
        attn_out = self.self_attn(
            normed,
            rope_mats=rope_mats,
            kv_cache=kv_cache,
            slot_idx=slot_idx,
            cached_len=cached_len,
            logical_n=logical_n,
            indexed_rope=indexed_rope,
            write_chunk=write_chunk,
        )
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, attn_out, output_tensor=attn_out)

        residual = hidden_states
        normed = self._norm_to_full_emb(self.post_attention_layernorm, hidden_states)
        mlp_out = self.mlp(normed)
        normed.deallocate(True)
        return ttnn.add(residual, mlp_out, output_tensor=mlp_out)
