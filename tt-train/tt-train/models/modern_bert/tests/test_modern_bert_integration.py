import unittest
import torch
from transformers import ModernBertConfig
from tt_train.models.modern_bert.modern_bert import ModernBertTT
from tt_train.models.modern_bert.configuration_modern_bert import ModernBertTTConfig

class TestModernBertIntegration(unittest.TestCase):
    def setUp(self):
        self.config = ModernBertTTConfig(
            model_name="answerdotai/ModernBERT-base",
            device="n300"
        )
        self.model = ModernBertTT(self.config)

    def test_end_to_end(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

        outputs = self.model(input_ids, attention_mask)
        self.assertEqual(outputs.shape, (1, 5, self.config.hidden_size))

        # Verify against HF reference
        hf_outputs = self.model.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # Calculate PCC (Pearson Correlation Coefficient)
        pcc = torch.nn.functional.cosine_similarity(
            outputs.flatten(),
            hf_outputs.flatten(),
            dim=0
        )
        self.assertGreater(pcc, 0.99)  # High similarity threshold