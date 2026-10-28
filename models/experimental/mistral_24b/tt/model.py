# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This is the end-to-end pipeline for the Mistral-Small-3.1-24B-Instruct-2503 model.

The `MistralTransformer` class inherits from the `Transformer` class in tt_transformers.
It overrides `prepare_inputs_prefill` to run inference on the vision model and
pass the resulting visual tokens to the text model along with text tokens.
"""


import torch

import ttnn
from models.experimental.mistral_24b.tt.pipeline.vision_model import TtMistralVisionTransformer
from models.tt_transformers.tt.model import Transformer

_MISTRAL_VISION_MAX_SEQ_LEN_FLOOR = 4096


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
        own_vision=True,
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

        # own_vision=False: the caller builds the vision tower and passes it to prefill as vision_model.
        self.vision_model = None
        if own_vision:
            self.vision_model = TtMistralVisionTransformer(
                mesh_device=mesh_device,
                state_dict=state_dict,
                state_dict_prefix="vision_tower.",
                dtype=dtype,
                model_args=args,
                tt_ccl=self.tt_ccl,
            )

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        trace_enabled=False,
        **kwargs,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device (trace_enabled=False) or on host (trace_enabled=True).

        Images come in as pixel_values / image_sizes (vLLM and the generator warmup), or as
        processed_inputs plus a separately built vision_model (pipeline_tests/test_end2end.py).
        Everything but the image splice comes from Transformer.prepare_inputs_prefill. Traced
        prefill relies on that: it must get the whole RoPE table, because a sliced copy is a new
        buffer that the generator drops after trace capture.
        """
        inputs = super().prepare_inputs_prefill(
            tokens,
            start_pos=start_pos,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            trace_enabled=trace_enabled,
            **kwargs,
        )
        # Traced prefill embeds the token ids inside the trace, so it cannot take image features.
        if trace_enabled:
            return inputs

        processed_inputs = kwargs.get("processed_inputs")
        if processed_inputs is not None:
            image_token_ids = processed_inputs["input_ids"]
            vision_output = None
            if processed_inputs["pixel_values"] is not None:
                vision_output = kwargs["vision_model"](
                    processed_inputs["pixel_values"], processed_inputs["image_sizes"]
                )
        else:
            image_token_ids = tokens
            vision_output = self.compute_vision_token(**kwargs)

        if vision_output is None:
            return inputs

        tokens_embd = ttnn.to_torch(inputs[0], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        tokens_embd = tokens_embd.squeeze(0)  # [1, 1, S, H] -> [1, S, H]
        comp_vision_output = ttnn.to_torch(
            vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        )[: vision_output.shape[0], :]

        image_features = comp_vision_output.squeeze(0)
        image_token_ids = torch.nn.functional.pad(
            image_token_ids, (0, tokens_embd.shape[1] - image_token_ids.shape[1]), "constant", 0
        )
        special_image_mask = (image_token_ids == self.args.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(tokens_embd)
        image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
        tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)

        return (self.args.prepare_residual_tensor_prefill(tokens_embd), *inputs[1:])

    def compute_vision_token(self, pixel_values=None, image_sizes=None, **kwargs):
        # With own_vision=False there is no tower here, so the generator's vision warmup runs text-only.
        if pixel_values is None or self.vision_model is None:
            return None
        if image_sizes is not None and not isinstance(image_sizes[0], (list, tuple)):
            image_sizes = [image_sizes]
        return self.vision_model(pixel_values, image_sizes)


def create_mistral_24b_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    use_paged_kv_cache=False,
    checkpoint=None,
):
    from models.tt_transformers.tt.model_config import ModelArgs

    max_seq_len = max(max_seq_len, _MISTRAL_VISION_MAX_SEQ_LEN_FLOOR)
    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    assert tt_model_args.is_multimodal, "This model is multimodal"

    # checkpoint is None => load here; {} => explicit/DP-reuse skip; populated => reuse.
    if checkpoint is None:
        checkpoint = tt_model_args.load_state_dict()

    model = MistralTransformer(
        mesh_device=mesh_device,
        state_dict=checkpoint,
        weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
        dtype=ttnn.bfloat8_b,
        args=tt_model_args,
        use_paged_kv_cache=use_paged_kv_cache,
    )

    return tt_model_args, model, checkpoint
