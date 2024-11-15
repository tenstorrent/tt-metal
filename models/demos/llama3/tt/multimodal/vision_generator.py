# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch

from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    StopReason,
)

from llama_models.llama3.reference_impl.generation import (
    ChatPrediction,
    CompletionPrediction,
    TokenResult,
    sample_top_p,
)


class LlamaVision:
    def __init__(self, model, model_args, mesh_device, vllm=False, tokenizer=None, formatter=None):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.vllm = vllm
        self.tokenizer = tokenizer
        self.formatter = formatter

    def prefill_forward_single_user(
        self,
        vision_images,
        vision_mask,
        tokens,
        xattn_caches,
        user_id,
        total_len,
        prefill_len,
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        vision_tokens, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
            batch_images=[vision_images],
            batch_masks=[vision_mask],
            total_len=total_len,
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            rot_mats,
            transformation_mats,
        ) = self.model.prepare_inputs_prefill(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, prefill_len=prefill_len
        )

        tt_logits = self.model.ttnn_prefill_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            xattn_caches,
            tt_position_id,
            rot_mats,
            transformation_mats,
            user_id,
            vision_tokens,
        )

        logits = self.model.process_output_prefill(tt_logits, B, prefill_len)

        return xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits

    def decode_forward(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Performs text decode step.
        Returns logits
        """

        # forward_decode should be traced callable
        # decorator does compilation, capture, execute
        # B = 1 # TODO: Only supports batch=1 right now! Might make tokens input a tensor.
        B, S = tokens.shape

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            _,
            tt_position_id,
            rot_mats,
            _,
        ) = self.model.prepare_inputs_decode(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=position_id
        )

        tt_logits = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            rot_mats,
        )

        logits = self.model.process_output_decode(tt_logits, B, S)
        return logits

    def capture_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            _,
            tt_position_id,
            rot_mats,
            _,
        ) = self.model.prepare_inputs_decode(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=position_id
        )

        # Compile run
        tt_logits_rm = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            rot_mats,
        )

        # Get inputs ready for trace run
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            _,
            tt_position_id,
            rot_mats,
            _,
        ) = self.model.prepare_decode_inputs_host(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            rot_mats,
        ) = self.model.copy_host_to_device(
            (tt_h, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_position_id, rot_mats)
        )

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        B = tokens.shape[0]
        # Do on-device transformations of inputs before forward
        tt_xattn_mask_transform, tt_full_text_mask_expand_1NSH_transform = self.model.transform_decode_inputs_device(
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            B=B,
        )

        tt_logits_rm = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask_transform,
            tt_full_text_mask_expand_1NSH_transform,
            xattn_caches,
            tt_position_id,
            rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        return trace_id, tt_logits_rm, tt_h, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_position_id, rot_mats

    def decode_forward_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,  # TODO: unused since captured in trace?
        trace_id,
        trace_logits_rm,
        trace_h,
        trace_xattn_mask,
        trace_full_text_mask_expand_1NSH,
        trace_position_id,
        trace_rot_mats,
    ):
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            _,
            tt_position_id,
            rot_mats,
            _,
        ) = self.model.prepare_decode_inputs_host(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=position_id
        )

        self.model.copy_host_to_device(
            host_tensors=(tt_h, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_position_id, rot_mats),
            device_tensors=(
                trace_h,
                trace_xattn_mask,
                trace_full_text_mask_expand_1NSH,
                trace_position_id,
                trace_rot_mats,
            ),
        )

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        B, S = tokens.shape
        logits = self.model.process_output_decode(trace_logits_rm, B=B, S=S)

        return logits

    def easy_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_id"):
            (
                trace_id,
                tt_logits_rm,
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_position_id,
                rot_mats,
            ) = self.capture_trace(
                position_id,
                tokens,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                xattn_caches,
            )
            self.trace_id = trace_id
            self.trace_inputs = {
                "tt_h": tt_h,
                "tt_xattn_mask": tt_xattn_mask,
                "tt_full_text_mask_expand_1NSH": tt_full_text_mask_expand_1NSH,
                "tt_position_id": tt_position_id,
                "rot_mats": rot_mats,
            }
            self.trace_outputs = {
                "tt_logits_rm": tt_logits_rm,
            }

        return self.decode_forward_trace(
            position_id,
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            xattn_caches,
            self.trace_id,
            self.trace_outputs["tt_logits_rm"],
            self.trace_inputs["tt_h"],
            self.trace_inputs["tt_xattn_mask"],
            self.trace_inputs["tt_full_text_mask_expand_1NSH"],
            self.trace_inputs["tt_position_id"],
            self.trace_inputs["rot_mats"],
        )

    def generate(
        self,
        model_input,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        # Do initial prefill
        vision_images = model_input.vision.images
        vision_mask = model_input.vision.mask
        prompt_tokens = model_input.tokens
        prefill_len = len(prompt_tokens)
        total_len = prefill_len + max_gen_len  # Prepares mask for full length of output

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
        # Suboptimal to allocate caches every time
        xattn_caches = self.model.setup_cache(self.model_args.max_batch_size)
        (
            xattn_caches,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            logits,
        ) = self.prefill_forward_single_user(
            vision_images,
            vision_mask,
            prompt_tokens_tensor,
            xattn_caches,
            user_id=0,
            total_len=total_len,
            prefill_len=prefill_len,
        )

        def sample(logits):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            return next_token, self.tokenizer.decode(next_token.tolist())

        next_token, text = sample(logits)

        yield TokenResult(
            token=next_token[0].item(),
            text=text,
        )

        for gen_idx in range(max_gen_len - 1):
            position_id = prefill_len + gen_idx
            next_token_tensor = next_token.reshape(1, 1)  # B, S

            logits = self.decode_forward(
                position_id,
                next_token_tensor,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                xattn_caches,
            )

            next_token, text = sample(logits)
            yield TokenResult(
                token=next_token[0].item(),
                text=text,
            )

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.configuration.max_seq_len:
            max_gen_len = self.model.configuration.max_seq_len - 1

        tokens = []

        stop_reason = None
        for result in self.generate(
            model_input=self.formatter.encode_dialog_prompt(messages, tool_prompt_format=False),
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        message = self.formatter.decode_assistant_message(tokens, stop_reason)

        return ChatPrediction(generation=message)

    def text_completion(
        self,
        content: InterleavedTextMedia,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.configuration.max_seq_len:
            max_gen_len = self.model.configuration.max_seq_len - 1

        model_input = self.formatter.encode_content(content)

        tokens = []

        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)

        generation = self.tokenizer.decode(tokens)

        return CompletionPrediction(generation=generation)
