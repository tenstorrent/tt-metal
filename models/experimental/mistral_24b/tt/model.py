# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
This is the end-to-end pipeline for the Mistral-Small-3.1-24B-Instruct-2503 model.

The `MistralTransformer` class inherits from the `Transformer` class in tt_transformers.
It overrides `prepare_inputs_prefill` to run inference on the vision model and
pass the resulting visual tokens to the text model along with text tokens.
"""


import ttnn
import torch

from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.model import Transformer
from ttnn import ConcatMeshToTensor


class MistralTransformer(Transformer):
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
        # On Blackhole, greedy on-device sampling otherwise falls through to the full top-k
        # path, which issues two all-gathers per decode step. Those alias the depth-2 CCL
        # semaphore pool across the back-to-back non-blocking decode + sampling traces and
        # make greedy decode non-deterministic / garbled. Enabling force-argmax collapses
        # greedy sampling to a single all-gather + argmax, removing the aliasing. Scoped here
        # (not in shared tt_transformers config) so only this model is affected; Wormhole keeps
        # the default. Must run before super().__init__, which builds the sampling module that
        # reads this flag.
        if is_blackhole():
            sampling_ag_config = args.model_config.get("SAMPLING_AG_CONFIG")
            if sampling_ag_config is not None:
                sampling_ag_config["allow_force_argmax"] = True

        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

    # Explicit overrides ensure MistralGenerator.super() calls dispatch through this subclass.
    def ttnn_prefill_forward(self, *args, **kwargs):
        return super().ttnn_prefill_forward(*args, **kwargs)

    def ttnn_decode_forward(self, *args, **kwargs):
        return super().ttnn_decode_forward(*args, **kwargs)

    def prepare_prefill_inputs_trace(
        self, tokens, page_table=None, chunk_page_table=None, batch_size=1, user_id=0, **kwargs
    ):
        # Entry point for trace-mode prefill; always forces trace_enabled=True for fixed-shape input tensors.
        ret = self.prepare_inputs_prefill(
            tokens,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=True,
            batch_size=batch_size,
            user_id=user_id,
            **kwargs,
        )
        return ret

    def _prepare_fused_prefill_embeddings(
        self, text_input_ids, processed_inputs=None, vision_model=None, return_host=False
    ):
        # Run vision tower, then scatter vision features into text embeddings at image-token positions.
        tt_tokens = ttnn.from_torch(
            text_input_ids.reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if not return_host else None,
        )
        tokens_embd = self.embd(tt_tokens)

        if processed_inputs is None or processed_inputs.get("pixel_values", None) is None:
            if return_host:
                tokens_embd_torch = ttnn.to_torch(
                    tokens_embd, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1)
                )
                tokens_embd = ttnn.from_torch(
                    tokens_embd_torch,
                    dtype=ttnn.bfloat16,
                    device=None,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(None, 2), mesh_shape=list(self.mesh_device.shape)
                    ),
                )
            return ttnn.unsqueeze_to_4D(tokens_embd)

        pixel_values = processed_inputs["pixel_values"]
        image_sizes = processed_inputs["image_sizes"]
        image_token_index = getattr(self.args, "image_token_index", 10)
        assert text_input_ids is not None, "text_input_ids must be provided for multimodal fusion"

        # Run vision tower — result is replicated on all devices: [num_image_tokens, H_full]
        vision_output = vision_model(pixel_values, image_sizes)

        # Pull vision features to host for re-sharding (small tensor: num_image_tokens × H_full).
        # vision_output is replicated, so ConcatMeshToTensor(dim=-1) gives [num_tokens, H_full*n].
        # Slice back to [num_tokens, H_full] (vision_output.shape[-1] is the per-device = full H
        # since the tensor is replicated, not sharded).
        vision_features = ttnn.to_torch(vision_output, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1))[
            :, : vision_output.shape[-1]
        ]  # [num_image_tokens, H_full]
        num_image_tokens, H_full = vision_features.shape

        # tokens_embd global shape: [1, S_padded, H_full], sharded along H via ShardTensor2dMesh.
        # S_padded is the tile-aligned sequence length (may differ from text_input_ids.shape[1]).
        S_padded = tokens_embd.shape[-2]  # tile-aligned S from TTNN logical shape
        H_full_embd = tokens_embd.shape[-1]  # H_full from TTNN logical shape (global)

        # --- Compute scatter indices on CPU (data-dependent; cannot run on device) ---
        # torch.nn.functional.pad is NOT needed here: ttnn.scatter uses explicit row indices
        # so we locate image tokens directly in the unpadded sequence.  Padded positions
        # (S..S_padded-1) are filled with 0 ≠ image_token_index, so they never match.
        mask_indices = torch.where(text_input_ids.view(-1) == image_token_index)[
            0
        ]  # [num_image_tokens] — absolute positions in the S_padded dimension
        assert len(mask_indices) == num_image_tokens, (
            f"Image-token count mismatch: input_ids has {len(mask_indices)} [IMG] tokens, "
            f"vision output has {num_image_tokens} feature vectors."
        )

        # Build scatter index: [num_image_tokens, H_full_embd].
        # Row i holds the target sequence position for the i-th image token;
        # it is broadcast across the H dimension so every hidden unit of token i
        # lands at the correct row after scatter.
        mask_indices_tt = ttnn.from_torch(
            mask_indices.view(-1, 1).expand(num_image_tokens, H_full_embd).contiguous(),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Shard vision features along H to match the tokens_embd per-device layout.
        # vision_features: [num_image_tokens, H_full] on host
        # → each device gets [num_image_tokens, H_full/n_devices] (ShardTensor2dMesh dim 1)
        vision_tt = ttnn.from_torch(
            vision_features,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(None, 1), mesh_shape=list(self.mesh_device.shape)
            ),
        )

        # --- ttnn.scatter replaces the to_torch → masked_scatter → from_torch round-trip ---
        # tokens_embd [1, S_padded, H_full] is kept on device throughout; no large CPU transfer.
        # Reshape to 2D for dim-0 scatter, scatter vision features at image-token positions,
        # then restore the original shape.
        tokens_embd = ttnn.reshape(tokens_embd, (S_padded, H_full_embd))
        tokens_embd = ttnn.scatter(tokens_embd, 0, mask_indices_tt, vision_tt)
        tokens_embd = ttnn.reshape(tokens_embd, (1, S_padded, H_full_embd))

        ttnn.deallocate(mask_indices_tt)
        ttnn.deallocate(vision_tt)

        if return_host:
            # Trace mode: fused embeddings must be a host tensor for fixed-shape trace capture.
            tokens_embd_torch = ttnn.to_torch(
                tokens_embd, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1)
            )  # [1, S_padded, H_full]
            tokens_embd = ttnn.from_torch(
                tokens_embd_torch[..., :H_full],
                dtype=ttnn.bfloat16,
                device=None,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 2), mesh_shape=list(self.mesh_device.shape)
                ),
            )

        return ttnn.unsqueeze_to_4D(tokens_embd)

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        # Trace prefill supports two input kinds:
        # - text-only: uint32 token ids -> embed here
        # - multimodal: pre-fused bfloat16 embeddings -> pass through
        if tokens.dtype == ttnn.uint32:
            tt_tokens = self.embd(tokens)
            tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        else:
            tt_tokens = tokens
        return tt_tokens, tt_page_table, tt_chunk_page_table

    def prepare_inputs_prefill(
        self, tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False, **kwargs
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """
        # When trace_enabled, use None for device to keep tensors on host
        device = None if trace_enabled else self.mesh_device

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        text_input_ids = tokens.reshape(1, -1)  # Preserve flat token ids for multimodal fusion before ttnn conversion.

        tokens = ttnn.from_torch(
            tokens,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if not trace_enabled else None,
        )

        # Extract multimodal kwargs to select the embedding path (vision fusion vs text-only).
        processed_inputs = kwargs.get("processed_inputs", None)
        vision_model = kwargs.get("vision_model", None)
        has_multimodal_inputs = processed_inputs is not None and processed_inputs.get("pixel_values", None) is not None

        tokens_embd = None
        # Multimodal: run vision forward + fusion; text-only (no trace): embed tokens directly; text trace: skip embed.
        if has_multimodal_inputs:
            tokens_embd = self._prepare_fused_prefill_embeddings(
                text_input_ids=text_input_ids,
                processed_inputs=processed_inputs,
                vision_model=vision_model,
                return_host=trace_enabled and has_multimodal_inputs,
            )
        elif not trace_enabled:
            tokens_embd = self.embd(tokens)
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Guard against sequences exceeding the pre-allocated RoPE matrix (uses prefill-specific matrix).
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = kwargs.get("last_token_idx", None) + 1 if kwargs.get("last_token_idx", None) is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        # Trace mode always starts at position 0 and slices full max_seq_len for fixed-shape trace capture.
        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)
        prefill_start_pos = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(mat_len, required_end)

        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :]

        # Pad RoPE slice to full shape when the sequence end exceeds the pre-allocated matrix length.
        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill_global = [cos_slice, sin_slice]

        if hasattr(self, "rope_local_setup"):
            # Same slice/pad logic applied to the local RoPE matrix for models with dual RoPE setups.
            local_mat_len = self.rope_local_setup.cos_matrix_prefill.shape[2]
            local_required_end = start_pos + S
            local_pad_len = max(0, local_required_end - local_mat_len)
            local_slice_end = self.args.max_seq_len if trace_enabled else min(local_mat_len, local_required_end)

            local_cos_slice = self.rope_local_setup.cos_matrix_prefill[:, :, prefill_start_pos:local_slice_end, :]
            local_sin_slice = self.rope_local_setup.sin_matrix_prefill[:, :, prefill_start_pos:local_slice_end, :]

            if local_pad_len > 0:
                local_padding = [(0, 0)] * 4
                local_padding[2] = (0, local_pad_len)
                local_cos_slice = ttnn.pad(local_cos_slice, padding=local_padding, value=0.0)
                local_sin_slice = ttnn.pad(local_sin_slice, padding=local_padding, value=0.0)

            tt_rot_mats_prefill_local = [local_cos_slice, local_sin_slice]
        else:
            tt_rot_mats_prefill_local = None

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if not trace_enabled else None,
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if not trace_enabled else None,
            )
        else:
            tt_chunk_page_table = None

        # Trace text path: token ids as static inputs; multimodal path: pre-fused bfloat16 embeddings.
        if trace_enabled and not has_multimodal_inputs:
            return tokens, tt_rot_mats_prefill_global, tt_rot_mats_prefill_local, tt_page_table, tt_chunk_page_table
        else:
            return (
                tokens_embd,
                tt_rot_mats_prefill_global,
                tt_rot_mats_prefill_local,
                tt_page_table,
                tt_chunk_page_table,
            )
