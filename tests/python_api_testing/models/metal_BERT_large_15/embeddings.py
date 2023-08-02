import torch
from torch import nn
import packaging
import tt_lib
from functools import partial
from models.utility_functions import pad_by_zero
from typing import Optional


class PytorchEmbeddings(nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.bert.embeddings

        # Disable dropout
        self.eval()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)


class TtBertEmbeddings(nn.Module):
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    def __init__(
        self,
        config,
        hugging_face_reference_model,
        state_dict,
        base_address,
        device,
        host,
        layer_norm_on_dev=True,
    ):
        super().__init__()
        self.device = device
        self.host = host

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.word_embeddings.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.word_embeddings.weight"]
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.position_embeddings.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.position_embeddings.weight"]
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.token_type_embeddings.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.token_type_embeddings.weight"]
        )

        self.layer_norm_on_dev = layer_norm_on_dev

        assert f"{base_address}.word_embeddings.weight" in state_dict
        assert f"{base_address}.word_embeddings.weight" in state_dict
        assert f"{base_address}.LayerNorm.weight" in state_dict
        assert f"{base_address}.LayerNorm.bias" in state_dict

        if layer_norm_on_dev:
            gamma = pad_by_zero(
                state_dict[f"{base_address}.LayerNorm.weight"], self.device
            )[0]
            beta = pad_by_zero(
                state_dict[f"{base_address}.LayerNorm.bias"], self.device
            )[0]
            self.TTLayerNorm = self.LayerNorm = partial(
                tt_lib.tensor.layernorm,
                eps=config.layer_norm_eps,
                gamma=gamma,
                beta=beta,
            )
        else:
            self.PyTorchLayerNorm = torch.nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
            self.PyTorchLayerNorm.weight = torch.nn.Parameter(
                state_dict[f"{base_address}.LayerNorm.weight"]
            )
            self.PyTorchLayerNorm.bias = torch.nn.Parameter(
                state_dict[f"{base_address}.LayerNorm.bias"]
            )

        # disabling dropout
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        # Disable dropout
        self.eval()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.layer_norm_on_dev:
            embeddings = embeddings.bfloat16()
            # convert to ttl
            oldShape = embeddings.size()
            newShape = [
                1,
            ] + list(oldShape)
            # pack to 4d tensor
            embeddings = torch.reshape(embeddings, newShape)

            tt_embeddings = tt_lib.tensor.Tensor(
                tt_lib.utils.tilize_to_list(embeddings),
                list(embeddings.size()),
                tt_lib.tensor.DataType.BFLOAT16,
                tt_lib.tensor.Layout.TILE,
                self.device,
            )
            embeddings = self.TTLayerNorm(tt_embeddings)

            # convert back to pytorch
            embeddings_data = embeddings.to(self.host).data()
            # back to 4d to untilize
            embeddings = torch.Tensor(embeddings_data).reshape(newShape)
            embeddings = tt_lib.utils.untilize(embeddings)

            # to original 3d shape
            embeddings = torch.Tensor(embeddings).reshape(oldShape)

        else:
            embeddings = self.PyTorchLayerNorm(embeddings)
            # to compare to TT version
            embeddings = embeddings.bfloat16()

        return embeddings
