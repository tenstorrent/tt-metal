# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This is the end-to-end pipeline for the Mistral-Small-3.1-24B-Instruct-2503 model.

The `MistralTransformer` class inherits from the `Transformer` class in tt_transformers.
It overrides `prepare_inputs_prefill` to run inference on the vision model and
pass the resulting visual tokens to the text model along with text tokens.
"""


import ttnn
import torch

from models.tt_transformers.tt.model import Transformer
from ttnn import ConcatMeshToTensor

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


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
        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

    def ttnn_prefill_forward(self, *args, **kwargs):
        signpost("Mistral24B::TextModel::PrefillForward::Start", f"kv_cache={kwargs.get('kv_cache') is not None}")
        try:
            return super().ttnn_prefill_forward(*args, **kwargs)
        finally:
            signpost("Mistral24B::TextModel::PrefillForward::End")

    def ttnn_decode_forward(self, *args, **kwargs):
        signpost("Mistral24B::TextModel::DecodeForward::Start", f"kv_cache={kwargs.get('kv_cache') is not None}")
        if kwargs.get("kv_cache") is not None:
            signpost("Mistral24B::KVCacheUpdates::Start", "decode forward")
        try:
            return super().ttnn_decode_forward(*args, **kwargs)
        finally:
            if kwargs.get("kv_cache") is not None:
                signpost("Mistral24B::KVCacheUpdates::End", "decode forward")
            signpost("Mistral24B::TextModel::DecodeForward::End")

    def prepare_prefill_inputs_trace(
        self, tokens, page_table=None, chunk_page_table=None, batch_size=1, user_id=0, **kwargs
    ):
        signpost("Mistral24B::TracePrefillInputs::Start", f"batch_size={batch_size}")
        ret = self.prepare_inputs_prefill(
            tokens,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=True,
            batch_size=batch_size,
            user_id=user_id,
            **kwargs,
        )
        signpost("Mistral24B::TracePrefillInputs::End", f"batch_size={batch_size}")
        return ret

    def _prepare_fused_prefill_embeddings(
        self, text_input_ids, processed_inputs=None, vision_model=None, return_host=False
    ):
        signpost("Mistral24B::FusedPrefillEmbeddings::Start", f"return_host={return_host}")
        tt_tokens = ttnn.from_torch(
            text_input_ids.reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        signpost("Mistral24B::EmbeddingLookup::Start", "prefill text tokens")
        tokens_embd = self.embd(tt_tokens)
        signpost("Mistral24B::EmbeddingLookup::End", "prefill text tokens")

        if processed_inputs is None or processed_inputs.get("pixel_values", None) is None:
            if return_host:
                signpost("Mistral24B::DeviceTransfer::DeviceToHost::Start", "text embeddings")
                tokens_embd_torch = ttnn.to_torch(
                    tokens_embd, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1)
                )
                signpost("Mistral24B::DeviceTransfer::DeviceToHost::End", "text embeddings")
                signpost("Mistral24B::DeviceTransfer::HostTensorCreate::Start", "text embeddings")
                tokens_embd = ttnn.from_torch(
                    tokens_embd_torch,
                    dtype=ttnn.bfloat16,
                    device=None,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(None, 2), mesh_shape=list(self.mesh_device.shape)
                    ),
                )
                signpost("Mistral24B::DeviceTransfer::HostTensorCreate::End", "text embeddings")
            signpost("Mistral24B::FusedPrefillEmbeddings::End", "text-only")
            return ttnn.unsqueeze_to_4D(tokens_embd)

        pixel_values = processed_inputs["pixel_values"]
        image_sizes = processed_inputs["image_sizes"]
        image_token_index = getattr(self.args, "image_token_index", 10)

        signpost("Mistral24B::VisionEncoder::Start", "prefill multimodal")
        vision_output = vision_model(pixel_values, image_sizes)
        signpost("Mistral24B::VisionEncoder::End", "prefill multimodal")
        signpost("Mistral24B::DeviceTransfer::DeviceToHost::Start", "vision output")
        vision_output_torch = ttnn.to_torch(vision_output, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1))[
            :, : vision_output.shape[-1]
        ]
        signpost("Mistral24B::DeviceTransfer::DeviceToHost::End", "vision output")

        signpost("Mistral24B::DeviceTransfer::DeviceToHost::Start", "text embeddings")
        tokens_embd_torch = ttnn.to_torch(tokens_embd, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1))
        signpost("Mistral24B::DeviceTransfer::DeviceToHost::End", "text embeddings")
        assert text_input_ids is not None, "text_input_ids must be provided for multimodal fusion"
        input_ids = torch.nn.functional.pad(
            text_input_ids, (0, tokens_embd_torch.shape[1] - text_input_ids.shape[1]), "constant", 0
        )

        signpost("Mistral24B::MultimodalFusion::Start", "masked_scatter image tokens")
        special_image_mask = (input_ids == image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(tokens_embd_torch)
        image_features = vision_output_torch.to(tokens_embd_torch.device, tokens_embd_torch.dtype)
        tokens_embd_torch = tokens_embd_torch.masked_scatter(special_image_mask, image_features)
        signpost("Mistral24B::MultimodalFusion::End", "masked_scatter image tokens")

        target_device = None if return_host else self.mesh_device
        signpost("Mistral24B::DeviceTransfer::HostToDevice::Start", "fused embeddings")
        tokens_embd = ttnn.from_torch(
            tokens_embd_torch,
            dtype=ttnn.bfloat16,
            device=target_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(None, 2), mesh_shape=list(self.mesh_device.shape)
            ),
        )
        signpost("Mistral24B::DeviceTransfer::HostToDevice::End", "fused embeddings")
        signpost("Mistral24B::FusedPrefillEmbeddings::End", "multimodal")
        return ttnn.unsqueeze_to_4D(tokens_embd)

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        # Trace prefill supports two input kinds:
        # - text-only: uint32 token ids -> embed here
        # - multimodal: pre-fused bfloat16 embeddings -> pass through
        if tokens.dtype == ttnn.uint32:
            signpost("Mistral24B::EmbeddingLookup::Start", "trace device token ids")
            tt_tokens = self.embd(tokens)
            tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
            signpost("Mistral24B::EmbeddingLookup::End", "trace device token ids")
        else:
            signpost("Mistral24B::EmbeddingLookup::Skip", "trace uses pre-fused multimodal embeddings")
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

        assert tokens.dim() == 2, "tokens must be a 2D tensor"
        signpost("Mistral24B::PrefillInputPrep::Start", f"trace_enabled={trace_enabled}")
        if kwargs.get("batch_size", 1) > 1:
            S = tokens.shape[-1]
            tokens = tokens.reshape(1, 1, 1, -1)
        else:
            tokens = tokens.reshape(1, 1, 1, -1)
            S = tokens.shape[-1]

        text_input_ids = tokens.reshape(1, -1)
        signpost("Mistral24B::DeviceTransfer::FromTorchTokens::Start", f"trace_enabled={trace_enabled}")
        tokens = ttnn.from_torch(
            tokens,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        signpost("Mistral24B::DeviceTransfer::FromTorchTokens::End", f"trace_enabled={trace_enabled}")

        processed_inputs = kwargs.get("processed_inputs", None)
        vision_model = kwargs.get("vision_model", None)
        has_multimodal_inputs = processed_inputs is not None and processed_inputs.get("pixel_values", None) is not None

        tokens_embd = None
        if has_multimodal_inputs:
            tokens_embd = self._prepare_fused_prefill_embeddings(
                text_input_ids=text_input_ids,
                processed_inputs=processed_inputs,
                vision_model=vision_model,
                return_host=trace_enabled and has_multimodal_inputs,
            )
        elif not trace_enabled:
            signpost("Mistral24B::EmbeddingLookup::Start", "non-trace prefill")
            tokens_embd = self.embd(tokens)
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
            signpost("Mistral24B::EmbeddingLookup::End", "non-trace prefill")

        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = kwargs.get("last_token_idx", None) + 1 if kwargs.get("last_token_idx", None) is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)
        prefill_start_pos = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(mat_len, required_end)

        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :]

        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill_global = [cos_slice, sin_slice]

        if hasattr(self, "rope_local_setup"):
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
            signpost("Mistral24B::DeviceTransfer::FromTorchPageTable::Start", f"trace_enabled={trace_enabled}")
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            signpost("Mistral24B::DeviceTransfer::FromTorchPageTable::End", f"trace_enabled={trace_enabled}")
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            signpost("Mistral24B::DeviceTransfer::FromTorchChunkPageTable::Start", f"trace_enabled={trace_enabled}")
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            signpost("Mistral24B::DeviceTransfer::FromTorchChunkPageTable::End", f"trace_enabled={trace_enabled}")
        else:
            tt_chunk_page_table = None

        # Trace text-only path keeps token ids as static trace inputs.
        # Trace multimodal path feeds pre-fused embeddings to the trace.
        if trace_enabled and not has_multimodal_inputs:
            signpost("Mistral24B::PrefillInputPrep::End", "trace text-only")
            return tokens, tt_rot_mats_prefill_global, tt_rot_mats_prefill_local, tt_page_table, tt_chunk_page_table
        else:
            signpost(
                "Mistral24B::PrefillInputPrep::End",
                "trace multimodal" if trace_enabled else "non-trace",
            )
            return (
                tokens_embd,
                tt_rot_mats_prefill_global,
                tt_rot_mats_prefill_local,
                tt_page_table,
                tt_chunk_page_table,
            )
