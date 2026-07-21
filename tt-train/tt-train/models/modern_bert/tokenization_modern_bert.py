from transformers import ModernBertTokenizer

class ModernBertTokenizer(ModernBertTokenizer):
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_name = model_name

    def encode(self, text, **kwargs):
        return super().encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return super().decode(token_ids, **kwargs)