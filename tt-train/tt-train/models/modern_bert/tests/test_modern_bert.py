import unittest
import torch
from transformers import ModernBertConfig
from tt_train.models.modern_bert.modern_bert import ModernBertTT
from tt_train.models.modern_bert.configuration_modern_bert import ModernBertTTConfig

class TestModernBert(unittest.TestCase):
    def setUp(self):
        self.config = ModernBertTTConfig(
            model_name="answerdotai/ModernBERT-base",
            device="n300"
        )
        self.model = ModernBertTT(self.config)

    def test_forward_pass(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

        outputs = self.model(input_ids, attention_mask)
        self.assertEqual(outputs.shape, (1, 5, self.config.hidden_size))

    def test_tokenization(self):
        tokenizer = self.model.tokenizer
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(text, decoded)