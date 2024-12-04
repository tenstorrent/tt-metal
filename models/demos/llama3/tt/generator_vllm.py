# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import torch
import PIL
from llama_models.llama3.api.chat_format import create_vision_mask

from models.demos.llama3.tt.generator import LlamaGenerator
from models.demos.llama3.demo.simple_vision_demo import create_multimodal_model

from vllm.inputs import INPUT_REGISTRY, DecoderOnlyInputs, EncoderDecoderInputs, InputContext


def input_processor_for_mllama(ctx: InputContext, inputs: Union[DecoderOnlyInputs, EncoderDecoderInputs]):
    """
    Based on vllm.model_executor.models.mllama.py::input_processor_for_mllama().
    Note that vLLM's input_processor_for_mllama performs additional processing to handle chunking which we do not yet support.
    """

    # Move encoder_prompt to prompt. If the user does not explicitly provide separate
    # encoder and decoder prompts, vLLM by default will treat the prompt as the encoder prompt.
    # For the block manager to allocate enough blocks and add them to the block table, the decoder prompt
    # must contain the full text prompt.
    if inputs.get("prompt") is None:
        inputs["prompt"] = inputs["encoder_prompt"]
        inputs["prompt_token_ids"] = inputs["encoder_prompt_token_ids"]

    return inputs


@INPUT_REGISTRY.register_input_processor(input_processor_for_mllama)
class TtMllamaForConditionalGeneration(LlamaGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MLLAMA_IMAGE_TOKEN_ID = 128256
        self.max_gen_len = self.model_args.max_seq_len - 1  # TODO: double check what this should be

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size):
        max_seq_len = 512  # TODO: Increase to 131072 once it's verified to work
        model_args, model = create_multimodal_model(mesh_device, max_batch_size, max_seq_len, use_paged_kv_cache=True)
        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        images: List[PIL.Image.Image],
        xattn_caches,
        start_pos,
        page_table: torch.Tensor = None,
        kv_cache=None,
        prompt_lens=None,
    ):
        """
        Replaces prefill_forward from LlamaGenerator with a version that supports mask creation.
        """
        batch = tokens.shape[0]

        vision_images = []
        vision_masks = []
        total_lens = []
        for user_id in range(batch):
            vision_images.append([images[user_id]])
            prompt_tokens = [int(tokens[user_id, i]) for i in range(prompt_lens[user_id])]
            vision_masks.append(create_vision_mask(prompt_tokens, self.MLLAMA_IMAGE_TOKEN_ID))
            total_lens.append(prompt_lens[user_id] + self.max_gen_len)

        return super().prefill_forward(
            vision_images, vision_masks, tokens, xattn_caches, total_lens, prompt_lens, page_table, kv_cache
        )
