# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Voxtral-Mini-3B: the tt_transformers text decoder (Transformer) + the Voxtral audio
tower, wired by scattering audio embeddings into the text embeddings at audio_token_id. Mirrors
MistralTransformer (mistral_24b VLM e2e) with the Pixtral vision tower replaced by the Whisper-style
audio tower and the image-token scatter replaced by the audio-token scatter."""
import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.multimodal.voxtral.voxtral_audio_tower import TtVoxtralAudioTower


class VoxtralTransformer(Transformer):
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
        input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
        self.audio_token_id = args.hf_config.audio_token_id
        self.audio_tower = TtVoxtralAudioTower(
            mesh_device=mesh_device,
            state_dict=args.voxtral_audio_state_dict,
            hf_config=args.hf_config,
            weights_mesh_mapper=weights_mesh_mapper,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

    def compute_audio_token(self, input_features=None, **kwargs):
        if input_features is not None:
            return self.audio_tower(input_features)
        return None

    def prepare_inputs_prefill(
        self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False, **kwargs
    ):
        """Embed tokens, scatter audio features at audio_token_id, and return the prefill inputs.
        Mirrors MistralTransformer.prepare_inputs_prefill (audio instead of vision)."""
        device = None if trace_enabled else self.mesh_device

        S = pt_tokens.shape[-1]
        tokens = ttnn.from_torch(
            pt_tokens.reshape(1, 1, 1, -1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        if not trace_enabled:
            tokens_embd = self.embd(tokens)
            audio_output = self.compute_audio_token(**kwargs)

            if audio_output is not None:
                tokens_embd = ttnn.to_torch(
                    tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
                )
                comp_audio_output = ttnn.to_torch(
                    audio_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[: audio_output.shape[0], :]

                audio_features = comp_audio_output.squeeze(0)
                special_audio_mask = (pt_tokens == self.audio_token_id).unsqueeze(-1)
                special_audio_mask = special_audio_mask.expand_as(tokens_embd)
                audio_features = audio_features.to(tokens_embd.device, tokens_embd.dtype)
                tokens_embd = tokens_embd.masked_scatter(special_audio_mask, audio_features)

                tokens_embd = self.args.prepare_residual_tensor_prefill(tokens_embd)

            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
        else:
            tokens_embd = tokens

        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        assert mat_len >= start_pos + S, f"Prefill end idx {start_pos + S} exceeds max seq len {mat_len}"
        prefill_start_pos = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(mat_len, start_pos + S)
        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :],
            self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, None, tt_page_table, tt_chunk_page_table
