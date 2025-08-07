"""
This is the end-to-end pipeline for the Qwen-VL 2.5 model.

The `Qwen25VLTransformer` class inherits from the `Transformer` class in tt_transformers.
It overrides `prepare_inputs_prefill` to run inference on the vision model and
pass the resulting visual tokens to the text model along with text tokens.
"""


import ttnn
import torch

from models.tt_transformers.tt.model import Transformer


class Qwen25VLTransformer(Transformer):
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

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
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
        # self.embed_scale = args.dim**0.5
        tokens_embd = self.embd(tokens)
        # tokens_embd = ttnn.multiply(tokens_embd, self.embed_scale)

        pixel_values = kwargs["processed_inputs"]["pixel_values"]
        input_ids = kwargs["processed_inputs"]["input_ids"]
        image_grid_thw = kwargs["processed_inputs"]["image_grid_thw"]

        vision_model = kwargs["vision_model"]
        pixel_values = self.args.prepare_residual_tensor_prefill(pixel_values.unsqueeze(0), force_replicated=True)

        vision_output = vision_model(pixel_values, image_grid_thw)

        tokens_embd = ttnn.to_torch(tokens_embd)
        comp_vision_output = ttnn.to_torch(ttnn.from_device(vision_output))

        input_ids = torch.nn.functional.pad(input_ids, (0, tokens_embd.shape[1] - input_ids.shape[1]), "constant", 0)
        image_features = comp_vision_output.squeeze(0)
        special_image_mask = (input_ids == 151655).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(tokens_embd)
        image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
        tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)

        tokens_embd = ttnn.from_torch(
            tokens_embd,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
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

        return tokens_embd, tt_rot_mats_prefill_global, tt_page_table, tt_chunk_page_table
