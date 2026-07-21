# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Auto-shard Transformer: the stock Transformer, but with a *replicated* residual stream.

The auto-shard decoder block (decoder_auto_shard.py) and RMSNorm (rmsnorm_auto_shard.py) keep the
hidden activation replicated -- full hidden dim on every chip -- across the whole stack; each module
fractures/gathers internally. Stock model.py instead fractures the hidden dim across the mesh and
leans on DistributedNorm to regather it, which is what breaks when the auto-shard block (expecting
replicated) is dropped in.

This subclass keeps everything stock (embedding, rope, lm_head, prepare_inputs, tracing) except:
  * decoder layers are the auto-shard block (model.TransformerBlock is rebound below),
  * the final norm is an auto-shard RMSNorm with axis=None -- a plain local norm on the replicated
    full hidden dim -- replacing DistributedNorm, and
  * forward() gathers the fractured embedding output to replicated once up front, then runs the
    stack + final norm + lm_head on the replicated stream (no per-layer residual resharding).

Stack-fractured fast path: when every block agrees on one fused residual axis (matching placements
across attention and MLP) and that axis is the embedding's fracture axis, forward() skips the up-front
gather, keeps the residual fractured on that axis through the whole stack, and gathers exactly once
before the final norm -- one all_gather per forward instead of two per layer. See _stack_fractured_axis
and decoder_auto_shard.enable_stack_fractured. It falls back to the replicated path automatically when
the placements don't line up.

The lm_head is untouched: its weight is k=args.dim (full hidden), vocab-sharded, so it already wants
the replicated full-hidden input the auto-shard final norm hands it -- byte-for-byte the layout
DistributedNorm feeds it in the stock model.

Scope: the single-axis-fractured demo path (1xN mesh, e.g. 1x4), prefill (get_last_token) + decode,
no prefetcher, non-Galaxy. The stock batched-prefill/trace helpers (_apply_norm_and_lm_head,
process_*) call self.norm(x, mode=, norm_config=) and are not part of this path.

Importing this module wires everything; a consumer only needs to `import` it before create_tt_model.
"""

import os

import torch

import ttnn

import models.tt_transformers.tt.model as _model
from models.tt_transformers.tt.auto_shard.ccl_auto_shard import all_gather
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.inline_profile import section
from models.tt_transformers.tt.auto_shard.decoder_auto_shard import TransformerBlock as _AutoShardTransformerBlock
from models.tt_transformers.tt.auto_shard.lm_head_auto_shard import LMHeadAutoShard as _AutoShardLMHead
from models.tt_transformers.tt.auto_shard.rmsnorm_auto_shard import RMSNorm as _AutoShardRMSNorm

# Decoder layers are built from the auto-shard block (Transformer.__init__ reads this name).
_model.TransformerBlock = _AutoShardTransformerBlock

# LM head drops the stock tail all-reduce (Transformer.__init__ reads this name to build self.lm_head).
# Its vocab shards are assembled downstream by ttnn_decode_forward / concat_host_output on any mesh.
_model.LMHead = _AutoShardLMHead


def _hidden_shard_axis(mesh_device):
    """Mesh axis the stock embedding fractures the hidden dim on (last axis with size > 1)."""
    shape = tuple(mesh_device.shape)
    for axis in reversed(range(len(shape))):
        if shape[axis] > 1:
            return axis
    return None


class Transformer(_model.Transformer):
    def __init__(self, *args, **kwargs):
        state_dict = kwargs["state_dict"]  # create_tt_model passes everything by keyword
        super().__init__(*args, **kwargs)
        self._hidden_axis = _hidden_shard_axis(self.mesh_device)
        # Fabric topology for the stack-level residual/logit gathers (Ring on a ring fabric, Linear
        # otherwise). Cached so per-forward gathers don't re-call ccl_topology() (which logs).
        self._ccl_topology = self.args.ccl_topology()

        # Fully-fractured stack: if every block keeps its residual fractured on the SAME mesh axis --
        # and that's the axis the embedding already fractures the hidden dim on -- leave the residual
        # fractured across the whole stack and gather exactly once at the end, instead of
        # re-replicating after every block. Falls back to the per-block replicated path otherwise
        # (mixed placements, replicated hidden, etc.).
        self._fractured_axis = self._stack_fractured_axis()
        if self._fractured_axis is not None:
            for layer in self.layers:
                layer.enable_stack_fractured()
            print(f"auto-shard: stack-fractured residual on mesh axis {self._fractured_axis} "
                  f"(one all_gather per forward)\n")

        # On-device sampling shards the vocab along a single mesh axis (tt_penalties: num_devices =
        # max(rows, cols)). On a 1xN line that matches the lm_head's flat vocab shard, so leave it
        # on. On a 2D mesh they disagree (2x2: 64128 vs 32064 per device) and the penalty ops fail
        # to broadcast, so fall back to host sampling only there -- host sampling consumes the full
        # logits assembled by ttnn_decode_forward's all_gather / process_output_prefill.
        rows, cols = tuple(self.mesh_device.shape)
        if rows > 1 and cols > 1:
            self._supports_on_device_sampling = False

        # EXPERIMENT (temporary -- revert with `git checkout`): FORCE_HOST_SAMPLING=1 takes the host
        # path regardless of mesh. The stock gate (model.py) is vocab_size // num_devices <= 64K, and
        # Qwen2.5-7B's 152064 straddles it -- host at 1x2 (76032), on-device at 1x4 (38016) -- so the
        # two meshes are not running the same code after the last block. This pins them to one path.
        # Unlike the other knobs here, host sampling is a supported path: output stays valid.
        if os.environ.get("FORCE_HOST_SAMPLING") == "1":
            self._supports_on_device_sampling = False

        # Replicated-stream final norm: full hidden dim on every chip -> plain local rms_norm.
        self.norm = _AutoShardRMSNorm(
            device=self.mesh_device,
            dim=self.args.dim,
            state_dict=state_dict,
            weight_key="norm",
            axis=None,
            state_dict_prefix=self.args.get_state_dict_prefix("", None),
            eps=self.args.norm_eps,
            add_unit_offset=self.args.rms_norm_add_unit_offset,
        )

    def _stack_fractured_axis(self):
        """The mesh axis to keep the residual fractured on across the whole stack, or None.

        Requires every block to agree on one non-None fused axis (identical placements everywhere, so
        each block's attention output, MLP input, and MLP output all sit on it) and that axis to be
        the one the embedding already fractures the hidden dim on (so the stack's fractured input is
        the embedding output as-is -- no up-front gather). None disables the fast path.
        """
        axes = {getattr(layer, "_fuse_axis", None) for layer in self.layers}
        if len(axes) != 1:
            return None
        axis = axes.pop()
        if axis is None or axis != self._hidden_axis:
            return None
        return axis

    def forward(
        self,
        x,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        page_tables_per_layer=None,
    ):
        # Embedding fractures the hidden dim across the mesh. Default: gather to replicated once up
        # front, then each block re-replicates internally. Stack-fractured: leave x fractured on
        # _fractured_axis (== the embedding's fracture axis) and keep it fractured to the end.
        if self._fractured_axis is None:
            x = all_gather(x, self.mesh_device, self._hidden_axis, topology=self._ccl_topology)

        if mode == Mode.PREFILL:
            rot_mats_global = self._slice_prefill_rot_mats(rot_mats_global, chunk_start_idx)
            if rot_mats_local is not None:
                rot_mats_local = self._slice_prefill_rot_mats(rot_mats_local, chunk_start_idx)

        for i, layer in enumerate(self.layers):
            layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table
            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=layer_page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
            )

        if mode == Mode.PREFILL and get_last_token == -1:
            # Chunked-prefill hidden-state return: hand back the replicated layout stock expects.
            if self._fractured_axis is not None:
                x = all_gather(x, self.mesh_device, self._fractured_axis, topology=self._ccl_topology)
            return x

        # The one gather: reassemble the full hidden dim the final norm + lm_head need. In the default
        # path x is already replicated, so this is skipped (fractured_axis is None).
        if self._fractured_axis is not None:
            x = all_gather(x, self.mesh_device, self._fractured_axis, topology=self._ccl_topology)

        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Final norm on the replicated full hidden dim, then the stock lm_head (unchanged).
        x = self.norm(x)

        # lm_head expects a width-sharded input (same sharded config for prefill and decode when
        # no prefetcher), so reshard to match.
        #
        # x is USUALLY interleaved here, because the last thing the block stack did was an all_gather
        # and that returns interleaved DRAM. It is not interleaved on a single-chip mesh: every
        # collective short-circuits on a size-1 axis, so the gather never runs and the block's own
        # width-sharded output (which the norm preserves) arrives here untouched. Go through
        # interleaved in that case -- interleaved_to_sharded rejects an already-sharded input.
        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(mode, None)
        if lm_head_input_mem_cfg.is_sharded():
            if x.memory_config().is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)

        x = self.lm_head(x)
        if mode == Mode.PREFILL:
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def process_output_prefill(self, tt_out, last_token_idx):
        """Assemble full logits from the flat vocab-sharded lm_head output.

        The lm_head shards the vocab across ALL mesh devices (ShardTensorToMesh(dim=-1)), so the
        shards sit in linear device order. Stock concat_host_output assumes vocab-on-columns with
        rows as a separate (DP/replicate) axis: on a 2D mesh it concatenates each row's shards on
        the vocab dim but stacks the rows on dim 1, so [0, 0, ...] returns only the first row's half
        of the vocab. Concatenate every device's shard on the vocab dim in device order instead --
        correct on a 1xN line and a 2D mesh alike.
        """
        assert tt_out.storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
        shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tt_out)]
        full = torch.cat(shards, dim=-1)
        return full[0, 0, last_token_idx, : self.vocab_size]

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        on_device_logits=False,
        page_tables_per_layer=None,
    ):
        """Same contract as stock ttnn_decode_forward, but assembles the vocab-sharded logits with
        the axis-aware ccl_auto_shard.all_gather over each real mesh axis instead of the stock
        experimental all_gather_async(cluster_axis=None).

        The lm_head shards the vocab flat across the whole mesh, so full logits need a whole-mesh
        gather. The stock single collective hops D0->D1->D2->D3 as one chain, which a 2D fabric can't
        route (D1->D2 is diagonal on a 2x2). Gathering one axis at a time is an adjacent-hop
        collective that routes on any mesh: gather the inner (column) axis first, then the outer
        (row) axis, so the row-major ShardTensorToMesh shards reassemble in order. all_gather no-ops
        on a size-1 axis, so a 1xN line runs exactly one gather.
        """
        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = self.rope_local_setup.get_rot_mats(rot_mat_idxs) if hasattr(self, "rope_local_setup") else None

        x_embed = self._transform_decode_inputs_device(x)

        if page_tables_per_layer is None:
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)

        with section("TOTAL decode step", self.mesh_device):
            tt_logits = self.forward(
                x_embed,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                mode=Mode.DECODE,
                page_table=page_table,
                kv_cache=kv_cache,
                page_tables_per_layer=page_tables_per_layer,
            )

        if on_device_logits:
            assert self.sampling is not None, (
                "ttnn_decode_forward got on_device_logits=True but no on-device sampling "
                "module exists (self.sampling is None)."
            )
            self._increment_decode_positions_device(current_pos, rot_mat_idxs)
            return tt_logits

        # Assemble full logits from the flat vocab shard: inner (column, axis 1) then outer (row,
        # axis 0). Axis-local hops route on any mesh; a size-1 axis is a no-op.
        if self.args.num_devices > 1:
            tt_logits = all_gather(tt_logits, self.mesh_device, 1, topology=self._ccl_topology)
            tt_logits = all_gather(tt_logits, self.mesh_device, 0, topology=self._ccl_topology)

        tt_logits = ttnn.untilize(
            tt_logits,
            use_multicore=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if self.prefetcher is not None else None,
        )

        return tt_logits, None


# Rebind so create_tt_model's function-local `from ...model import Transformer` builds this one.
# The subclass captured the stock base at definition time, so this doesn't affect inheritance.
_model.Transformer = Transformer
