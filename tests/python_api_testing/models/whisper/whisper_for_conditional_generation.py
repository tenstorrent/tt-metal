import tt_lib
import torch
import torch.nn as nn
from dataclasses import dataclass
from loguru import logger

from typing import Optional, Tuple, Union
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperConfig,
)
from datasets import load_dataset

from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
    create_unpadded_tensor,
)
from python_api_testing.models.whisper.whisper_model import TtWhisperModel
from python_api_testing.models.whisper.whisper_linear_layer import WhisperPaddedLinear


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@dataclass
class TtWhisperLMOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: tt_lib.tensor.Tensor = None
    past_key_values: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    decoder_attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    cross_attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    encoder_last_hidden_state: Optional[tt_lib.tensor.Tensor] = None
    encoder_hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    encoder_attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


class TtWhisperForConditionalGeneration(nn.Module):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.version",
        r"decoder.version",
        r"proj_out.weight",
    ]
    _keys_to_ignore_on_save = [
        r"proj_out.weight",
    ]

    def __init__(self, state_dict, device, config: WhisperConfig):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.config = config

        self.model = TtWhisperModel(
            base_address="model",
            state_dict=self.state_dict,
            device=self.device,
            config=self.config,
        )

        self.proj_out = WhisperPaddedLinear(
            config.d_model,
            config.vocab_size,
            state_dict[f"proj_out.weight"],
            None,
            self.device,
        )

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        # class has to inherit from transformers PreTrainedModel to support this method
        raise NotImplementedError

        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[tt_lib.tensor.Tensor], TtWhisperLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        """TODO: Used in training mode"""
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logger.info(
            f"Tt Whisper Model output shape {outputs.last_hidden_state.shape()}"
        )

        lm_logits = self.proj_out(outputs.last_hidden_state)

        # Unpad
        lm_logits = create_unpadded_tensor(
            lm_logits, [1, 1, lm_logits.shape()[-2], self.config.vocab_size]
        )

        # Convert to Torch
        logits_to_torch = torch.Tensor(lm_logits.data()).reshape(lm_logits.shape())
        logits_to_torch = torch.squeeze(logits_to_torch, 0)

        """TODO: Not supporting Training in TTM for the moment"""
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits_to_torch.view(-1, self.config.vocab_size), labels.reshape(-1)
            )

        if not return_dict:
            output = (logits_to_torch,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TtWhisperLMOutput(
            loss=loss,
            logits=logits_to_torch,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
