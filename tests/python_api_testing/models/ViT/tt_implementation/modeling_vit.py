from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Union, Optional, Tuple, Dict, Set, List
import math
import collections

import torch
from torch import nn

import tt_lib
import tt_lib.fallback_ops as fallback_ops
from tt_lib.fused_ops.linear import linear as linear
from utility_functions import torch_to_tt_tensor
from tt_utils import find_pruneable_heads_and_indices, prune_linear_layer
from configuration_vit import ViTConfig
from activations import ACT2FN

tt_tensor = tt_lib.tensor.Tensor


def make_address(base_address, op_name):
    return op_name if base_address == "" else f"{base_address}.{op_name}"

def make_linear(in_feature, out_feature, op_name, state_dict, base_address, device):
            q_weight = state_dict[make_address(base_address, f"{op_name}.weight")]
            q_weight = torch_to_tt_tensor(q_weight, device)
            if make_address(base_address, f"{op_name}.bias") in state_dict:
                q_bias = state_dict[make_address(base_address, f"{op_name}.bias")]
                q_bias = torch_to_tt_tensor(q_bias, device)
            else:
                q_bias = None
            return linear(in_feature, out_feature, weight=q_weight, bias=q_bias)

class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig, base_address, state_dict) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense = make_linear(config.intermediate_size, config.hidden_size, "dense", state_dict, base_address)


    def forward(self, hidden_states: tt_lib.tensor.Tensor, input_tensor: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states, input_tensor)
        return hidden_states


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.device = device
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.recip_sqrt_attention_head_size_tensor = torch.full((1, 1, 32, 32), 1/math.sqrt(self.attention_head_size))
        self.recip_sqrt_attention_head_size_tensor = torch_to_tt_tensor(self.recip_sqrt_attention_head_size_tensor)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = make_linear(config.hidden_size, self.all_head_size, "query", state_dict, base_address)
        self.key = make_linear(config.hidden_size, self.all_head_size, "key", state_dict, base_address)
        self.value = make_linear(config.hidden_size, self.all_head_size, "value", state_dict, base_address)

    def transpose_for_scores(self, x: tt_tensor) -> tt_tensor:
        new_x_shape = x.shape()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = tt_lib.tensor.reshape(x, *new_x_shape)
        return tt_lib.tensor.permute(x, 0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[tt_tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[tt_tensor, tt_tensor], Tuple[tt_tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer_T = tt_lib.tensor.tranpose(key_layer)
        attention_scores = tt_lib.tensor.matmul(query_layer, key_layer_T)

        attention_scores = tt_lib.tensor.bcast(attention_scores, self.recip_sqrt_attention_head_size_tensor,
                                                tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.HW)

        # Normalize the attention scores to probabilities.
        attention_probs = tt_lib.fallback_ops.softmax(attention_scores, dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tt_lib.tensor.mul(attention_probs, head_mask)

        context_layer = tt_lib.tensor.matmul(attention_probs, value_layer)

        context_layer = tt_lib.tensor.permute(context_layer, 0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape()[:-2] + (self.all_head_size,)
        context_layer = tt_lib.tensor.reshape(context_layer, *new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict) -> None:
        super().__init__()
        self.dense = make_linear(config.hidden_size, config.hidden_size, "dense", state_dict, base_address)


    def forward(self, hidden_states: tt_tensor, input_tensor: tt_tensor) -> tt_tensor:
        hidden_states = self.dense(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config, base_address, state_dict, device)
        self.output = ViTSelfOutput(config, base_address, state_dict)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: tt_tensor,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_tensor, tt_tensor], Tuple[tt_tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.dense = make_linear(config.hidden_size, config.intermediate_size, "dense", state_dict, base_address)
        # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: tt_tensor) -> tt_tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config, base_address, state_dict, device)
        self.intermediate = ViTIntermediate(config, base_address, state_dict, device)
        self.output = ViTOutput(config, base_address, state_dict, device)

        lbw = state_dict[make_address(base_address, "layernorm_before.weight")]
        lbb = state_dict[make_address(base_address, "layernorm_before.bias")]
        self.layernorm_before = fallback_ops.layernorm_before(normalized_shape=config.hidden_size, weights=lbw, biases=lbb, eps=config.layer_norm_eps)
        # self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        law = state_dict[make_address(base_address, "layernorm_after.weight")]
        lab = state_dict[make_address(base_address, "layernorm_after.bias")]
        self.layernorm_after = fallback_ops.layernorm_before(normalized_shape=config.hidden_size, weights=law, biases=lab, eps=config.layer_norm_eps)
        # self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: tt_tensor,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_tensor, tt_tensor], Tuple[tt_tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config, base_address, state_dict, device) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: tt_tensor,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict or True:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        weights = state_dict[make_address(base_address, "projection.weight")]
        biases = state_dict[make_address(base_address, "projection.bias")]

        self.projection = fallback_ops.Conv2d(
                                weights=weights,
                                biases=biases,
                                in_channels=num_channels,
                                out_channels=hidden_size,
                                kernel_size=patch_size,
                                stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

# candidate for CPU
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.dense = make_linear(config.hidden_size, config.hidden_size, "dense", state_dict, base_address, device)
        self.activation = tt_lib.tensor.tanh

    def forward(self, hidden_states: tt_tensor) -> tt_tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViTModel(nn.Module):
    def __init__(self,
                config: ViTConfig,
                base_address: str,
                state_dict: Dict,
                device,
                add_pooling_layer: bool = True,
                use_mask_token: bool = False) -> None:
        super().__init__()
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config, base_address=base_address, state_dict=state_dict, device=device)

        wln = state_dict[make_address(base_address, "layernorm.weight")]
        bln = state_dict[make_address(base_address, "layernorm.bias")]
        self.layernorm = fallback_ops.LayerNorm(
                                    normalized_shape=config.hidden_size,
                                    eps=config.layer_norm_eps,
                                    weights=wln,
                                    biases=bln)
        self.pooler = ViTPooler(config, base_address=base_address, state_dict=state_dict, device=device) if add_pooling_layer else None

        # Initialize weights and apply final processing

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[tt_tensor] = None,
        bool_masked_pos: Optional[tt_tensor] = None, # torch.booltensor
        head_mask: Optional[tt_tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None


        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        return head_outputs + encoder_outputs[1:]


class ViTForImageClassification(nn.Module):
    def __init__(self,
                config: ViTConfig,
                base_address: str,
                state_dict: Dict,
                device,) -> None:
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.vit = ViTModel(
                    config,
                    base_address=base_address,
                    state_dict=state_dict,
                    device=device,
                    add_pooling_layer=False
                )

        # Classifier head
        self.classifier = make_linear(config.hidden_size, config.num_labels, "classifier", state_dict, base_address, device) if config.num_labels > 0 else None
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing

    def forward(
        self,
        pixel_values: Optional[tt_tensor] = None,
        head_mask: Optional[tt_tensor] = None,
        labels: Optional[tt_tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        assert labels == None, "we do not support training, hence labels should be None"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = sequence_output[:, 0, :]
        if self.classifier is not None:
            logits = self.classifier(sequence_output[:, 0, :])

        loss = None

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output
