# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Auto-sharded TransformerBlock for Gemma-3.

decoder_auto_shard's block is the Llama wiring -- residual add, then ff_norm, then MLP. Gemma-3 norms
the attention output BEFORE the residual add and wraps the MLP in two more norms, so the block is

    attention_norm -> attention -> ff_norm -> +residual -> pre_ff_norm -> MLP -> post_ff_norm -> +residual

which is decoder.py's pre_ff_norm branch (decoder.py:276-336). Note ff_norm here is Gemma's
post_attention_layernorm: load_checkpoints maps that HF key onto the name "ffn_norm", and what makes
it Gemma rather than Llama is only where it sits relative to the residual add.

Only the replicated-residual default path is implemented. _fused_residual_axis() returns None, which
turns off this block's fuse_residual path and also keeps Transformer._stack_fractured_axis() at None
for the whole stack, so the model never calls enable_stack_fractured(). Because the stream is
replicated at every norm point, all four norms are plain local RMSNorms (axis=None) and none of
decoder.py's mesh_partition juggling is needed -- that code exists only to re-align a fractured
residual, which this path does not have.

pre/post_ff_norm are built only when the checkpoint has them, so a non-Gemma model routed through
this block degrades to the stock ordering rather than breaking.
"""

import ttnn
from models.tt_transformers.tt.auto_shard.ccl_auto_shard import all_gather, partition
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.auto_shard.decoder_auto_shard import TransformerBlock
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.auto_shard.rmsnorm_auto_shard import RMSNorm


class GemmaTransformerBlock(TransformerBlock):
    def __init__(self, **kwargs):
        # Transformer.__init__ builds layers with keyword arguments only (model.py:105-118).
        super().__init__(**kwargs)

        args = kwargs["args"]
        state_dict = kwargs["state_dict"]
        prefix = args.get_state_dict_prefix("", kwargs["layer_num"])

        def norm(weight_key):
            if f"{prefix}{weight_key}.weight" not in state_dict:
                return None
            return RMSNorm(
                device=kwargs["mesh_device"],
                dim=args.dim,
                state_dict=state_dict,
                weight_key=weight_key,
                axis=None,
                state_dict_prefix=prefix,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
                # Gemma-3's hidden dim is wide enough that fp32 norm accumulation overflows L1;
                # see the note in rmsnorm_auto_shard.
                fp32_dest_acc_en=False,
            )

        # The parent built attention_norm/ff_norm with fp32 accumulation on. Rebuild them here so
        # every norm in a Gemma block uses the same L1 budget; the parent's versions never run.
        self.attention_norm = norm("attention_norm")
        self.ff_norm = norm("ffn_norm")
        self.pre_ff_norm = norm("pre_feedforward_layernorm")
        self.post_ff_norm = norm("post_feedforward_layernorm")

    def _fused_residual_axis(self):
        """Replicated residual only -- see the module docstring."""
        return None

    def forward(
        self,
        x,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        residual = x  # replicated: full hidden dim on every chip

        rot_mats = rot_mats_local if self.attention.is_sliding else rot_mats_global

        attn_in = self.attention_norm(x)
        attn_in = partition(attn_in, self.mesh_device, self.attention.sharding.reduce_col_over)

        # Reshape to [B, 1, S_per_user, H] so attention infers batch_size from shape[0]
        if batch_size > 1:
            attn_in = ttnn.reshape(attn_in, [batch_size, 1, attn_in.shape[-2] // batch_size, -1])

        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        # Match the batch-related reshape inside attention (prefill with batched prefill).
        if mode == Mode.PREFILL and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])

        attn_out = all_gather(
            attn_out,
            self.mesh_device,
            self._output_axis(self.attention.sharding),
            topology=self.ccl_topology,
            label=f"gemma decoder attn_out all_gather [{mode}]",
        )

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        # Gemma: norm the attention output, THEN add the residual.
        if self.pre_ff_norm is not None:
            hidden_states = self.ff_norm(attn_out)
            ttnn.deallocate(attn_out)  # only safe once ff_norm has produced a separate tensor
        else:
            hidden_states = attn_out  # stock ordering; aliases attn_out, so do not deallocate it
        hidden_states = ttnn.add(residual, hidden_states)
        residual = hidden_states
        if mode == "prefill":
            x.deallocate(True)

        # MLP: pre-norm, fracture, run, gather back to replicated, post-norm.
        ff_in = self.pre_ff_norm(hidden_states) if self.pre_ff_norm is not None else self.ff_norm(hidden_states)
        ff_in = partition(ff_in, self.mesh_device, self.feed_forward.sharding.reduce_col_over)
        ff_out = self.feed_forward.forward(ff_in, mode)
        ff_out = all_gather(
            ff_out,
            self.mesh_device,
            self._output_axis(self.feed_forward.sharding),
            topology=self.ccl_topology,
            label=f"gemma decoder ff_out all_gather [{mode}]",
        )

        if self.post_ff_norm is not None:
            ff_out = self.post_ff_norm(ff_out)

        return ttnn.add(residual, ff_out, dtype=activation_dtype or ttnn.bfloat16)
