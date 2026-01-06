"""Test for Whisper 3 model with TTNN backend."""

import pytest
import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.modeling_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer as OriginalWhisperEncoderLayer
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import create_custom_mesh_preprocessor, encoder_layer
from models.tt_symbiote.core.module import TTNNModule
from models.tt_symbiote.core.run_config import DispatchManager
from models.tt_symbiote.modules.activation import TTNNSilu
from models.tt_symbiote.modules.attention import TTNNSDPAAttention
from models.tt_symbiote.modules.linear import TTNNLinear
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


class WhisperEncoderLayer(TTNNModule):
    def preprocess_weights_impl(self):
        super().preprocess_weights_impl()
        input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(self.device)
        self.ttnn_parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_layer,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=self.device,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        assert output_attentions is False, "output_attentions=True not supported yet"
        assert layer_head_mask is None, "layer_head_mask not supported yet"
        assert attention_mask is None, "attention_mask not supported yet"
        config = WhisperConfig.from_pretrained("distil-whisper/distil-large-v3")
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        layer_output = encoder_layer(config, hidden_states, parameters=self.ttnn_parameters)
        layer_output = ttnn.to_memory_config(layer_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return (layer_output,)


def get_attention_mappings():
    try:
        from transformers.models.whisper.modeling_whisper import WhisperAttention as OriginalWhisperAttention
    except:
        return {}
    return {}

    class WhisperAttention(nn.Module):
        """Multi-headed attention from 'Attention Is All You Need' paper"""

        def __init__(self, original_layer):
            super().__init__()
            self.embed_dim = original_layer.embed_dim
            self.num_heads = original_layer.num_heads
            self.dropout = original_layer.dropout
            self.head_dim = original_layer.head_dim
            self.config = original_layer.config
            self.scaling = self.head_dim**-0.5
            self.is_decoder = original_layer.is_decoder
            self.is_causal = original_layer.is_causal
            self.layer_idx = original_layer.layer_idx

            self.k_proj = original_layer.k_proj
            self.v_proj = original_layer.v_proj
            self.q_proj = original_layer.q_proj
            self.out_proj = original_layer.out_proj
            self.ttnn_attention_module = TTNNSDPAAttention()
            # self.ttnn_permute = TTNNPermute()
            # self.ttnn_reshape = TTNNReshape()

        @classmethod
        def from_torch(cls, attention_module: OriginalWhisperAttention) -> "TTNNWhisperAttention":
            """Create TTNNBottleneck from PyTorch Bottleneck layer."""
            new_attention = WhisperAttention(attention_module)
            return new_attention

        def forward(
            self,
            hidden_states,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
            cache_position=None,
            **kwargs,
        ):
            """Input shape: Batch x Time x Channel"""

            # if key_value_states are provided this layer is used as a cross-attention layer
            # for the decoder
            is_cross_attention = key_value_states is not None

            # determine input shapes
            bsz, tgt_len = hidden_states.shape[:-1]
            q_input_shape = (bsz, tgt_len, -1, self.head_dim)

            # Scaling is susceptible to floating point arithmetics' inprecisions
            # which can lead to different results (this is dependent from model
            # to model, e.g. whisper is one such case). We therefore keep the
            # original order of scaling to follow the original implementation
            # and enforce no scaling (1.0) in the attention call below.
            query_states = self.q_proj(hidden_states) * self.scaling
            query_states = query_states.view(*q_input_shape)
            query_states = query_states.transpose(1, 2).contiguous()

            if past_key_value is not None:
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    past_key_value.is_updated[self.layer_idx] = True
                    past_key_value = past_key_value.cross_attention_cache
                else:
                    past_key_value = past_key_value.self_attention_cache

            # use key_value_states if cross attention
            current_states = key_value_states if key_value_states is not None else hidden_states
            if is_cross_attention and past_key_value and is_updated:
                # reuse k,v, cross_attentions
                key_states = past_key_value.key_cache[self.layer_idx]
                value_states = past_key_value.value_cache[self.layer_idx]
            else:
                key_states = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
                value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
                key_states = key_states.transpose(1, 2).contiguous()
                value_states = value_states.transpose(1, 2).contiguous()
                if past_key_value is not None:
                    # save all key/value_states to cache to be re-used for fast auto-regressive generation
                    cache_position = cache_position if not is_cross_attention else None
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                    )

            attn_output = self.ttnn_attention_module(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.dropout,
                scaling=1.0,
                output_attentions=output_attentions,
                head_mask=layer_head_mask,
                **kwargs,
            )

            attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
            attn_output = self.out_proj(attn_output)

            return attn_output, None, past_key_value

    return {OriginalWhisperAttention: WhisperAttention}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1024}], indirect=True)
def test_whisper3(device):
    """Test Whisper 3 model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        OriginalWhisperEncoderLayer: WhisperEncoderLayer
        #   , OriginalWhisperDecoderLayer: WhisperDecoderLayer
    }
    nn_to_ttnn.update(get_attention_mappings())
    torch_dtype = torch.bfloat16

    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(dtype=torch_dtype)
    all_modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]

    result = pipe(sample, return_timestamps=True)
    print(result["text"])
    DispatchManager.save_stats_to_file("whisper_timing_stats.csv")
