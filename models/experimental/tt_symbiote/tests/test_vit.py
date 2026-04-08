# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for ViT model with TTNN backend."""

import torch
from torch import nn
from transformers import AutoModelForImageClassification
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTIntermediate, ViTLayer, ViTOutput, ViTSelfAttention

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.attention import TTNNViTSelfAttention
from models.experimental.tt_symbiote.modules.conv import TTNNViTEmbeddings
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNViTIntermediate
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.modules.tensor import TTNNAdd
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


class RewrittenViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, original_layer) -> None:
        super().__init__()
        self.cls_token = original_layer.cls_token
        self.mask_token = original_layer.mask_token
        self.position_embeddings = original_layer.position_embeddings
        self.embeddings = TTNNViTEmbeddings.from_torch(
            original_layer.patch_embeddings, original_layer.cls_token, original_layer.position_embeddings
        )
        self.patch_embeddings = self.embeddings.patch_embeddings
        self.dropout = original_layer.dropout
        self.patch_size = original_layer.patch_size
        self.config = original_layer.config

    @classmethod
    def from_torch(cls, embeddings: ViTEmbeddings):
        """Create TTNNViTEmbeddings from PyTorch ViTEmbeddings layer."""
        new_embeddings = RewrittenViTEmbeddings(embeddings)
        return new_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos=None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        assert bool_masked_pos is None, "Masked positions are not supported in this implementation."
        assert not interpolate_pos_encoding, "Position embedding interpolation is not supported in this implementation."
        embeddings = self.embeddings(pixel_values)
        return embeddings


class RewrittenViTOutput(nn.Module):
    def __init__(self, old_layer) -> None:
        super().__init__()
        self.dense = old_layer.dense
        self.add = TTNNAdd()

    @classmethod
    def from_torch(cls, old_layer: ViTOutput):
        """Create TTNNViTEmbeddings from PyTorch ViTEmbeddings layer."""
        new_layer = RewrittenViTOutput(old_layer)
        return new_layer

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.add(hidden_states, input_tensor)

        return hidden_states


class RewrittenViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, old_layer) -> None:
        super().__init__()
        self.chunk_size_feed_forward = old_layer.chunk_size_feed_forward
        self.seq_len_dim = old_layer.seq_len_dim
        self.attention = old_layer.attention
        self.intermediate = old_layer.intermediate
        self.output = RewrittenViTOutput.from_torch(old_layer.output)
        self.layernorm_before = old_layer.layernorm_before
        self.layernorm_after = old_layer.layernorm_after
        self.add = TTNNAdd()

    @classmethod
    def from_torch(cls, old_layer: ViTLayer):
        """Create TTNNViTEmbeddings from PyTorch ViTEmbeddings layer."""
        new_layer = RewrittenViTLayer(old_layer)
        return new_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask=None,
        output_attentions: bool = False,
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # first residual connection
        hidden_states = self.add(attention_output, hidden_states)

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


def test_vit(device):
    """Test ViT model with TTNN acceleration."""
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model = model.to(dtype=torch.bfloat16)
    nn_to_nn = {
        ViTEmbeddings: RewrittenViTEmbeddings,
        ViTLayer: RewrittenViTLayer,
        ViTOutput: RewrittenViTOutput,
    }
    modules1 = register_module_replacement_dict(model, nn_to_nn, model_config={"program_config_ffn": {}})
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.LayerNorm: TTNNLayerNorm,
        ViTSelfAttention: TTNNViTSelfAttention,
        ViTIntermediate: TTNNViTIntermediate,
    }
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config={"program_config_ffn": {}})
    set_device(model, device)
    modules = {**modules1, **modules2}
    for k, v in modules.items():
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    input_tensor = torch.randn(1, 224, 224, 4)
    model(input_tensor)
    DispatchManager.clear_timings()
    result = model(input_tensor)
    DispatchManager.save_stats_to_file("vit_timing_stats.csv")
    print(result.logits)
