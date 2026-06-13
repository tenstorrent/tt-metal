# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT-NN model wrapper for NVIDIA LocateAnything-3B's LLM backbone.

In autoregressive (AR) mode the LocateAnything LLM is a standard causal
Qwen2.5-3B. The only difference from a vanilla text run is that prefill is fed
*pre-merged* image+text embeddings (host float `[1, S, hidden]`) rather than
token ids. This subclass therefore keeps the entire stock
:class:`models.tt_transformers.tt.model.Transformer` behaviour and only adds an
embeds-driven prefill input preparation method.

RoPE is *standard 1D* (rope_theta=1e6), so we reuse the stock prefill rope
slicing from the parent class -- no mrope (unlike models/demos/qwen25_vl).
"""


import ttnn
from models.tt_transformers.tt.model import Transformer


class LATransformer(Transformer):
    """Qwen2.5-3B backbone for LocateAnything with embeds-driven prefill."""

    def prepare_inputs_prefill_embeds(
        self,
        embeds,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        last_token_idx=None,
    ):
        """Prepare prefill inputs from pre-merged host embeddings.

        Mirrors the stock :meth:`Transformer.prepare_inputs_prefill` rope-slicing
        logic, but takes already-embedded inputs (image+text merged on host)
        instead of token ids.

        Args:
            embeds: torch float tensor of shape [B=1, S, hidden_dim].
            start_pos: position offset for the rope slice (default 0).
            page_table: optional torch int tensor for paged attention.
            chunk_page_table: optional torch int tensor for chunked prefill.
            last_token_idx: index of the last meaningful token; used to validate
                the requested sequence length fits in the precomputed rope mats.

        Returns:
            (tokens_embd, [cos_slice, sin_slice], tt_page_table, tt_chunk_page_table)
        """
        assert embeds.dim() == 3, "embeds must be a 3D tensor [B=1, S, hidden]"
        assert embeds.shape[0] == 1, "LocateAnything only supports batch_size=1"
        S = embeds.shape[1]

        # [1, S, hidden] -> [1, 1, S, hidden]; shard the hidden dim across the mesh
        tokens_embd = ttnn.from_torch(
            embeds.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device,
                dims=(None, 3),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        # --- Stock prefill RoPE slicing (copied from Transformer.prepare_inputs_prefill) ---
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)

        # Slice the precomputed (on-device) cos/sin prefill matrices.
        slice_end = min(mat_len, required_end)
        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, start_pos:slice_end, :]

        if pad_len > 0:
            # Pad at end of 3rd dim (dim=2) by pad_len.
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill_global = [cos_slice, sin_slice]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, tt_page_table, tt_chunk_page_table
