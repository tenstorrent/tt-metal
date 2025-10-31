# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Padding Mask Validation

Tests BERT behavior with variable-length sequences and attention masks:
- Variable-length sequences with padding
- Different attention mask patterns
- Verify masked tokens don't affect output
- Compare HuggingFace and TTML masking behavior

This ensures that padding and attention masking work correctly.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")
from transformers import BertModel  # noqa: E402


class BERTPaddingMaskValidator:
    """Validates BERT padding and attention mask behavior."""

    def __init__(self, model_name: str, batch_size: int = 2, seq_len: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Load HuggingFace model
        print(f"\nLoading HuggingFace model: {model_name}")
        self.hf_model = BertModel.from_pretrained(model_name)
        self.hf_model.eval()
        self.config = self.hf_model.config

        print(
            f"Model config: {self.config.num_hidden_layers} layers, "
            f"{self.config.hidden_size} hidden dim, {self.config.num_attention_heads} heads"
        )

        # Save HuggingFace model to safetensors
        safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
        if not safetensors_path.exists():
            from safetensors.torch import save_file

            save_file(self.hf_model.state_dict(), str(safetensors_path))

        # Create TTML model
        ttml_config = ttml.models.bert.BertConfig()
        ttml_config.vocab_size = self.config.vocab_size
        ttml_config.max_sequence_length = seq_len
        ttml_config.embedding_dim = self.config.hidden_size
        ttml_config.intermediate_size = self.config.intermediate_size
        ttml_config.num_heads = self.config.num_attention_heads
        ttml_config.num_blocks = self.config.num_hidden_layers
        ttml_config.dropout_prob = 0.0
        ttml_config.layer_norm_eps = self.config.layer_norm_eps
        ttml_config.use_token_type_embeddings = True
        ttml_config.use_pooler = False

        self.ttml_model = ttml.models.bert.create(ttml_config)
        self.ttml_model.load_model_from_safetensors(str(safetensors_path))

    def compute_pcc(self, tensor1, tensor2):
        """Compute Pearson Correlation Coefficient."""
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()

        mean1 = np.mean(tensor1_flat)
        mean2 = np.mean(tensor2_flat)

        numerator = np.sum((tensor1_flat - mean1) * (tensor2_flat - mean2))
        denominator = np.sqrt(np.sum((tensor1_flat - mean1) ** 2) * np.sum((tensor2_flat - mean2) ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def validate_single_case(self, input_ids_np, token_type_ids_np, attention_mask_np, case_name: str):
        """Run single validation case."""

        # HuggingFace forward pass
        with torch.no_grad():
            input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)
            token_type_ids_torch = torch.tensor(token_type_ids_np, dtype=torch.long)
            attention_mask_torch = torch.tensor(attention_mask_np, dtype=torch.long)

            hf_outputs = self.hf_model(
                input_ids=input_ids_torch,
                token_type_ids=token_type_ids_torch,
                attention_mask=attention_mask_torch,
            )
            hf_output = hf_outputs.last_hidden_state.numpy()

        # TTML forward pass
        input_ids_ttml = ttml.autograd.Tensor.from_numpy(
            input_ids_np.astype(np.uint32).reshape(self.batch_size, 1, 1, self.seq_len)
        )
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
            token_type_ids_np.astype(np.uint32).reshape(self.batch_size, 1, 1, self.seq_len)
        )
        attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
            attention_mask_np.astype(np.float32).reshape(self.batch_size, 1, 1, self.seq_len)
        )

        ttml_output = self.ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
        ttml_output_np = ttml_output.to_numpy()

        # Compute metrics
        pcc = self.compute_pcc(hf_output, ttml_output_np)
        mean_diff = np.mean(np.abs(hf_output - ttml_output_np))
        max_diff = np.max(np.abs(hf_output - ttml_output_np))

        return {
            "case": case_name,
            "pcc": pcc,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "passed": pcc >= 0.95,
        }

    def validate(self):
        """Run padding mask validation."""
        print("\n" + "=" * 80)
        print("PADDING MASK VALIDATION")
        print("=" * 80)
        print(f"\nModel: {self.model_name}")
        print(f"Batch size: {self.batch_size}, Sequence length: {self.seq_len}")
        print("\nTesting variable-length sequences with attention masks.\n")

        results = []

        # Test 1: No padding (baseline - all masks = 1)
        print("Test 1: No padding (all tokens active)...")
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

        result = self.validate_single_case(input_ids, token_type_ids, attention_mask, "No padding")
        results.append(result)
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Test 2: Variable lengths - first sequence full, second sequence half
        print("\nTest 2: Variable lengths (seq1=full, seq2=half)...")
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
        # Second sequence only uses first half, rest is padding
        attention_mask[1, self.seq_len // 2 :] = 0
        input_ids[1, self.seq_len // 2 :] = 0  # PAD token

        result = self.validate_single_case(input_ids, token_type_ids, attention_mask, "Variable lengths")
        results.append(result)
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Test 3: Both sequences with different amounts of padding
        print("\nTest 3: Both sequences with different padding...")
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
        # First sequence: 3/4 length
        attention_mask[0, (self.seq_len * 3) // 4 :] = 0
        input_ids[0, (self.seq_len * 3) // 4 :] = 0
        # Second sequence: 1/2 length
        attention_mask[1, self.seq_len // 2 :] = 0
        input_ids[1, self.seq_len // 2 :] = 0

        result = self.validate_single_case(input_ids, token_type_ids, attention_mask, "Different padding")
        results.append(result)
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Test 4: Short sequences (1/4 length)
        print("\nTest 4: Short sequences (1/4 length)...")
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
        # Both sequences: only first 1/4 active
        attention_mask[:, self.seq_len // 4 :] = 0
        input_ids[:, self.seq_len // 4 :] = 0

        result = self.validate_single_case(input_ids, token_type_ids, attention_mask, "Short sequences")
        results.append(result)
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passed"])

        print(f"\nTotal tests: {total_tests}")
        print(f"Passed (PCC ≥ 0.95): {passed_tests}")
        print(f"Failed (PCC < 0.95): {total_tests - passed_tests}")

        # PCC statistics
        pccs = [r["pcc"] for r in results]
        print(f"\nPCC Statistics:")
        print(f"  Minimum: {min(pccs):.6f}")
        print(f"  Maximum: {max(pccs):.6f}")
        print(f"  Average: {np.mean(pccs):.6f}")
        print(f"  Median:  {np.median(pccs):.6f}")

        # Detailed results table
        print(f"\n" + "=" * 80)
        print("DETAILED TEST RESULTS")
        print("=" * 80)
        print()
        print(f"{'Test Case':<30} {'Status':<8} {'PCC':>10} {'Mean Diff':>15} {'Max Diff':>15}")
        print("-" * 80)

        for result in results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(
                f"{result['case']:<30} {status:<8} {result['pcc']:>10.6f} "
                f"{result['mean_diff']:>15.6e} {result['max_diff']:>15.6e}"
            )

        print("=" * 80)

        return passed_tests == total_tests


@pytest.mark.parametrize(
    "batch_size,seq_len,model_name",
    [
        (2, 32, "prajjwal1/bert-tiny"),
        (2, 32, "prajjwal1/bert-small"),
        (2, 32, "bert-base-uncased"),
    ],
)
def test_bert_padding_mask_validation(batch_size, seq_len, model_name):
    """Test BERT with variable-length sequences and padding."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name} (batch={batch_size}, seq_len={seq_len})")
    print(f"{'=' * 80}")

    validator = BERTPaddingMaskValidator(model_name, batch_size, seq_len)
    all_passed = validator.validate()

    if not all_passed:
        print(f"\n⚠️  RESULT: Some tests failed (PCC < 0.95)")
    else:
        print(f"\n✅ RESULT: All tests passed (PCC ≥ 0.95)")


if __name__ == "__main__":
    # Run for all models
    models = [
        "prajjwal1/bert-tiny",
        "prajjwal1/bert-small",
        "bert-base-uncased",
    ]

    for model_name in models:
        validator = BERTPaddingMaskValidator(model_name, batch_size=2, seq_len=32)
        validator.validate()
        print("\n\n")
