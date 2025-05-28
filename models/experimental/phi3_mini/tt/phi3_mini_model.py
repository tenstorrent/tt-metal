# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.model import Transformer
from models.experimental.phi3_mini.tt.phi3_mini_rope import Phi3MiniRotarySetup


class Phi3Transformer(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        self.rope_setup = Phi3MiniRotarySetup(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            scale_factor=args.rope_scaling_factor,
            ext_scale_tensors=args.rope_scaling,
            orig_context_len=args.orig_context_len,
            datatype=ttnn.bfloat16,
        )

    def prepare_inputs_prefill(self, tokens, seq_len, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        """

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the prefill seqlen
        assert self.rope_setup.cos_matrix["long_scaled"].shape[2] >= (
            start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix['long_scaled'].shape[2]}"
        if seq_len > self.rope_setup.orig_context_len:
            tt_rot_mats_prefill = [
                self.rope_setup.cos_matrix["long_scaled"][:, :, start_pos : start_pos + S, :],
                self.rope_setup.sin_matrix["long_scaled"][:, :, start_pos : start_pos + S, :],
            ]
        else:
            tt_rot_mats_prefill = [
                self.rope_setup.cos_matrix["short_scaled"][:, :, start_pos : start_pos + S, :],
                self.rope_setup.sin_matrix["short_scaled"][:, :, start_pos : start_pos + S, :],
            ]

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

        return tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table
