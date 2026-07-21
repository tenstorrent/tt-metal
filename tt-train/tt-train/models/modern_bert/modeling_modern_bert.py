import torch
import torch.nn as nn
import ttnn
from transformers.models.modern_bert.modeling_modern_bert import ModernBertModel
from tt_train.models.modern_bert.configuration_modern_bert import ModernBertTTConfig

class ModernBertTTModel(nn.Module):
    def __init__(self, config: ModernBertTTConfig):
        super().__init__()
        self.config = config
        self.hf_model = ModernBertModel(config)
        self.device = ttnn.open_device(device_type=config.device)

    def forward(self, input_ids, attention_mask=None):
        # Convert input to TTNN tensors
        input_ids_tt = ttnn.from_torch(input_ids, device=self.device)
        attention_mask_tt = ttnn.from_torch(attention_mask, device=self.device) if attention_mask is not None else None

        # Forward pass through TT model
        outputs = self.hf_model(
            input_ids=input_ids_tt,
            attention_mask=attention_mask_tt
        )

        return outputs

    def convert_to_ttnn(self, tensor):
        return ttnn.from_torch(tensor, device=self.device)

    def convert_to_torch(self, tensor):
        return tensor.to_torch()