import torch
import torch.nn as nn
from transformers import ModernBertConfig
from transformers.models.modern_bert.modeling_modern_bert import ModernBertModel
from tt_train.models.modern_bert.configuration_modern_bert import ModernBertTTConfig
from tt_train.models.modern_bert.tokenization_modern_bert import ModernBertTokenizer
from tt_train.models.modern_bert.modeling_modern_bert import ModernBertTTModel

class ModernBertTT(nn.Module):
    def __init__(self, config: ModernBertTTConfig):
        super().__init__()
        self.config = config
        self.hf_config = ModernBertConfig.from_pretrained(config.model_name)
        self.hf_model = ModernBertModel(self.hf_config)
        self.tt_model = ModernBertTTModel(config)

    def forward(self, input_ids, attention_mask=None):
        # Convert input to TTNN tensors
        input_ids_tt = self.tt_model.convert_to_ttnn(input_ids)
        attention_mask_tt = self.tt_model.convert_to_ttnn(attention_mask) if attention_mask is not None else None

        # Forward pass through TT model
        outputs = self.tt_model(input_ids_tt, attention_mask_tt)

        # Convert outputs back to PyTorch tensors
        last_hidden_state = self.tt_model.convert_to_torch(outputs.last_hidden_state)

        return last_hidden_state