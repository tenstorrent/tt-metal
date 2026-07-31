from transformers import ModernBertConfig
from tt_train.models.modern_bert.modeling_modern_bert import ModernBertTTModel

class ModernBertTTConfig(ModernBertConfig):
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        device: str = "n300",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device
        self.tt_model = ModernBertTTModel(self)