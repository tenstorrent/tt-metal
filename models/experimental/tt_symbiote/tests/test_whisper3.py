# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Whisper 3 model with TTNN backend."""

import pytest
import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.modeling_whisper import WhisperAttention as OriginalWhisperAttention
from transformers.models.whisper.modeling_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer as OriginalWhisperEncoderLayer
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import create_custom_mesh_preprocessor, encoder_layer
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNGelu, TTNNSilu
from models.experimental.tt_symbiote.modules.attention import TTNNWhisperAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


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
        layer_output = ttnn.squeeze(layer_output, 1)
        return (layer_output,)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1024}], indirect=True)
def test_whisper3(device):
    """Test Whisper 3 model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        OriginalWhisperEncoderLayer: WhisperEncoderLayer,
        OriginalWhisperAttention: TTNNWhisperAttention,
        nn.LayerNorm: TTNNLayerNorm,
        nn.GELU: TTNNGelu,
    }
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
    DispatchManager.clear_timings()
    result = pipe(sample, return_timestamps=True)
    print(result["text"])
    DispatchManager.save_stats_to_file("whisper_timing_stats.csv")
