# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Auto-sharded TransformerBlock: same wiring as decoder.py (attention_norm -> attention -> residual
-> ff_norm -> MLP -> residual) with the auto-sharded 2D-mesh Attention and MLP submodules.

Single clean path: no galaxy (TG) branches, no MoE, and no Gemma pre/post feed-forward norms.

Residual stream is kept fully replicated (full hidden dim on every chip), so the norm is a plain
local RMSNorm and the residual adds always line up regardless of how attention and MLP each shard.
The decoder adapts at each module boundary: `partition` the replicated input onto the module's
contraction axis if it splits the dim, then `all_gather` the module's fractured output back to
replicated. This works for any placement on any mesh (line or 2D); the cost is one full-width gather
after each module.

Stack-fractured fast path (enabled by the model via enable_stack_fractured): when every block keeps
its residual fractured on the same mesh axis, the model leaves the residual fractured across the
WHOLE stack and gathers once at the very end. A block in this mode takes a fractured input and
returns a fractured output, does both norms distributed over that axis, and skips the per-block
partition/gather entirely.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.auto_shard.attention_auto_shard import Attention
from models.tt_transformers.tt.auto_shard.ccl_auto_shard import all_gather, partition
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.auto_shard.mlp_auto_shard import MLP
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.auto_shard.rmsnorm_auto_shard import RMSNorm


class TransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,  # accepted for API parity with decoder.py; auto-shard always uses its own Attention
        prefetcher=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.num_devices = args.num_devices
        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()
        self.layer_num = layer_num
        # Residual-stream gathers run on the same fabric topology as the in-module collectives
        # (Ring on a ring fabric, Linear otherwise). Cached so the per-layer gathers don't re-call
        # ccl_topology() (which logs) on every forward.
        self.ccl_topology = args.ccl_topology()

        self.attention = Attention(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher=prefetcher,
        )

        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher=prefetcher,
        )

        # Fast path: when attention fractures its output onto exactly the axis MLP wants as its input
        # (they picked matching placements and that axis is really split), we can keep the residual
        # stream sharded on that axis between the two modules instead of re-replicating after each.
        # That drops a full-width all_gather; the price is a distributed ff_norm over the same axis.
        self._fuse_axis = self._fused_residual_axis()
        self.fuse_residual = self._fuse_axis is not None

        # The norm sees whatever layout the residual is in: attention always takes the replicated
        # block input (axis=None -> local norm); ff_norm sees a residual fractured on _fuse_axis on
        # the fast path (axis=_fuse_axis -> stats gathered over that axis), else replicated.
        def norm(weight_key, axis):
            return RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_key=weight_key,
                axis=axis,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
            )

        self._make_norm = norm  # kept so enable_stack_fractured can rebuild attention_norm distributed

        # attention_norm sees a replicated block input by default (local norm); ff_norm sees a residual
        # fractured on _fuse_axis on the within-block fast path. Stack-fractured mode (see below) later
        # rebuilds attention_norm distributed too.
        self.attention_norm = norm("attention_norm", axis=None)
        self.ff_norm = norm("ffn_norm", axis=self._fuse_axis)

        # Off unless the model turns it on for the whole stack (enable_stack_fractured).
        self.stack_fractured = False

    def _output_axis(self, sharding):
        """Mesh axis a module's output dim ends up sharded on (None = already replicated).

        The row matmul places the output on model_axis, and on a line the intermediate-axis
        reduce-scatter leaves the dim split there too (on a 2D mesh that axis is gathered inside the
        module).
        """
        p = sharding.placement
        mesh_shape = tuple(self.mesh_device.shape)
        if p.model_axis is not None and mesh_shape[p.model_axis] > 1:
            return p.model_axis
        if 1 in mesh_shape and p.intermediate_axis is not None and mesh_shape[p.intermediate_axis] > 1:
            return p.intermediate_axis
        return None

    def _fused_residual_axis(self):
        """Mesh axis to keep the residual sharded on across the block, or None to re-replicate.

        Valid only when attention's fractured output already sits on the axis MLP wants to contract
        over (so no gather+partition is needed between them) and MLP's output returns to that same
        axis (so the final residual add lines up). Placement equality is the by-chance trigger; the
        axis checks are what actually make it safe. None everywhere disables the fast path.
        """
        attn_sharding = self.attention.sharding
        mlp_sharding = self.feed_forward.sharding
        if attn_sharding.placement != mlp_sharding.placement:
            return None
        axis = self._output_axis(attn_sharding)
        if axis is None or axis != mlp_sharding.reduce_col_over or axis != self._output_axis(mlp_sharding):
            return None
        return axis

    def enable_stack_fractured(self):
        """Switch this block into the stack-fractured residual scheme.

        The model keeps the residual fractured on _fuse_axis across the WHOLE stack (one gather at the
        very end) instead of re-replicating after every block. That only changes this block's entry
        contract: the residual now arrives fractured on _fuse_axis, so attention_norm becomes a
        distributed norm over that axis (ff_norm already is), and forward() neither partitions the
        input (already fractured onto the attention contraction axis) nor gathers the output.
        """
        assert self._fuse_axis is not None, "stack-fractured requires a fused residual axis"
        self.stack_fractured = True
        self.attention_norm = self._make_norm("attention_norm", axis=self._fuse_axis)

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
        # replicated (full hidden dim on every chip) by default; fractured on _fuse_axis in
        # stack-fractured mode. DRAM-interleaved throughout either way.
        residual = x

        # Choose the correct rotation matrices based on the mode
        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        # attention: norm -> (fracture onto its contraction axis) -> run. In stack-fractured mode the
        # input is already fractured on _fuse_axis (== attention's contraction axis) and attention_norm
        # is distributed over it, so there is nothing to partition.

        attn_in = self.attention_norm(x)
        if not self.stack_fractured:
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

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.stack_fractured:
            print("in stack-fractured path\n")
            # Residual stays fractured on _fuse_axis for the whole stack: no gather after attention,
            # no partition before MLP, no gather at the block end. attn_out is already fractured on
            # _fuse_axis (attention's output axis), so it lines up with the fractured residual; the
            # model gathers exactly once after the last block.
            hidden_states = ttnn.add(residual, attn_out)
            residual = hidden_states
            if mode == "prefill":
                x.deallocate(True)
            ttnn.deallocate(attn_out)

            # ff_norm gathers its RMS stats over _fuse_axis; its output is the fractured input MLP wants.
            ff_in = self.ff_norm(hidden_states)
            ff_out = self.feed_forward.forward(ff_in, mode)
            return ttnn.add(residual, ff_out, dtype=activation_dtype or ttnn.bfloat16)  # stays fractured

        if self.fuse_residual:
            print("in fused path")
            # Keep the residual sharded on _fuse_axis across the block: skip the full-width gather
            # after attention and the partition before MLP; one gather re-replicates at the end.
            residual = partition(residual, self.mesh_device, self._fuse_axis)  # free -- data already on-chip
            hidden_states = ttnn.add(residual, attn_out)
            residual = hidden_states
            if mode == "prefill":
                x.deallocate(True)
            ttnn.deallocate(attn_out)

            # ff_norm gathers its RMS stats over _fuse_axis; its output is already the fractured input MLP wants.
            ff_in = self.ff_norm(hidden_states)
            ff_out = self.feed_forward.forward(ff_in, mode)
            out = ttnn.add(residual, ff_out, dtype=activation_dtype or ttnn.bfloat16)
            return all_gather(out, self.mesh_device, self._fuse_axis, topology=self.ccl_topology)  # replicate for the next block

        # Default path: re-replicate the residual after each module (works for any placement).
        attn_out = all_gather(
            attn_out,
            self.mesh_device,
            self._output_axis(self.attention.sharding),
            topology=self.ccl_topology,
            label=f"decoder attn_out all_gather [{mode}]",
        )
        hidden_states = ttnn.add(residual, attn_out)
        residual = hidden_states
        if mode == "prefill":
            x.deallocate(True)
        ttnn.deallocate(attn_out)

        # MLP: local norm, fracture, run, gather back to replicated.
        ff_in = self.ff_norm(hidden_states)
        ff_in = partition(ff_in, self.mesh_device, self.feed_forward.sharding.reduce_col_over)
        ff_out = self.feed_forward.forward(ff_in, mode)
        ff_out = all_gather(
            ff_out,
            self.mesh_device,
            self._output_axis(self.feed_forward.sharding),
            topology=self.ccl_topology,
            label=f"decoder ff_out all_gather [{mode}]",
        )

        out = ttnn.add(residual, ff_out, dtype=activation_dtype or ttnn.bfloat16)

        return out  # replicated across devices
