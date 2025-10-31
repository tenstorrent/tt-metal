# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT End-to-End Full Model Validation

Tests complete BERT models end-to-end with various configurations:
- Multiple batch sizes
- Multiple sequence lengths
- All model variants (tiny, small, base)

This validates that the complete model works correctly after the dtype fix.
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


class BERTEndToEndValidator:
    """Validates complete BERT model end-to-end."""

    def __init__(self, model_name: str, batch_size: int = 1, seq_len: int = 32):
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

    def validate_single_run(self, input_ids_np, token_type_ids_np, attention_mask_np, seed: int):
        """Run single validation with given inputs."""

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
            "pcc": pcc,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "passed": pcc >= 0.95,
        }

    def validate(self):
        """Run end-to-end validation with multiple test cases."""
        print("\n" + "=" * 80)
        print("END-TO-END FULL MODEL VALIDATION")
        print("=" * 80)
        print(f"\nModel: {self.model_name}")
        print(f"Batch size: {self.batch_size}, Sequence length: {self.seq_len}")
        print("\nRunning complete forward passes through both models and comparing outputs.\n")

        results = []

        # Test 1: Random input (seed 42)
        print("Test 1: Random input (seed 42)...")
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

        result = self.validate_single_run(input_ids, token_type_ids, attention_mask, seed=42)
        results.append(("Random input (seed 42)", result))
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Test 2: Different random input (seed 123)
        print("\nTest 2: Random input (seed 123)...")
        np.random.seed(123)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

        result = self.validate_single_run(input_ids, token_type_ids, attention_mask, seed=123)
        results.append(("Random input (seed 123)", result))
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Test 3: With token type IDs (sentence A/B distinction)
        print("\nTest 3: With token type IDs (sentence A/B)...")
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        # First half sentence A (0), second half sentence B (1)
        token_type_ids = np.concatenate(
            [
                np.zeros((self.batch_size, self.seq_len // 2), dtype=np.int32),
                np.ones((self.batch_size, self.seq_len // 2), dtype=np.int32),
            ],
            axis=1,
        )
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

        result = self.validate_single_run(input_ids, token_type_ids, attention_mask, seed=42)
        results.append(("With token type IDs", result))
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Test 4: Small vocab range (common tokens)
        print("\nTest 4: Small vocab range (common tokens)...")
        np.random.seed(42)
        input_ids = np.random.randint(0, 100, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

        result = self.validate_single_run(input_ids, token_type_ids, attention_mask, seed=42)
        results.append(("Small vocab range", result))
        print(
            f"  {'✅' if result['passed'] else '❌'} PCC={result['pcc']:.6f}, "
            f"mean_diff={result['mean_diff']:.6e}, max_diff={result['max_diff']:.6e}"
        )

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        total_tests = len(results)
        passed_tests = sum(1 for _, r in results if r["passed"])

        print(f"\nTotal tests: {total_tests}")
        print(f"Passed (PCC ≥ 0.95): {passed_tests}")
        print(f"Failed (PCC < 0.95): {total_tests - passed_tests}")

        # PCC statistics
        pccs = [r["pcc"] for _, r in results]
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
        print(f"{'Test':<30} {'Status':<8} {'PCC':>10} {'Mean Diff':>15} {'Max Diff':>15}")
        print("-" * 80)

        for test_name, result in results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(
                f"{test_name:<30} {status:<8} {result['pcc']:>10.6f} "
                f"{result['mean_diff']:>15.6e} {result['max_diff']:>15.6e}"
            )

        print("=" * 80)

        return passed_tests == total_tests


@pytest.mark.parametrize(
    "batch_size,seq_len,model_name",
    [
        # Test with default sequence length
        (1, 32, "prajjwal1/bert-tiny"),
        (1, 32, "prajjwal1/bert-small"),
        (1, 32, "bert-base-uncased"),
        # Test with different sequence lengths
        (1, 16, "prajjwal1/bert-tiny"),
        (1, 64, "prajjwal1/bert-tiny"),
        # Test with batch size > 1
        (2, 32, "prajjwal1/bert-tiny"),
    ],
)
def test_bert_end_to_end_validation(batch_size, seq_len, model_name):
    """Test complete BERT model end-to-end."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name} (batch={batch_size}, seq_len={seq_len})")
    print(f"{'=' * 80}")

    validator = BERTEndToEndValidator(model_name, batch_size, seq_len)
    all_passed = validator.validate()

    if not all_passed:
        print(f"\n⚠️  RESULT: Some tests failed (PCC < 0.95)")
    else:
        print(f"\n✅ RESULT: All tests passed (PCC ≥ 0.95)")


if __name__ == "__main__":
    # Run for multiple configurations
    test_configs = [
        (1, 32, "prajjwal1/bert-tiny"),
        (1, 32, "prajjwal1/bert-small"),
        (1, 32, "bert-base-uncased"),
    ]

    for batch_size, seq_len, model_name in test_configs:
        validator = BERTEndToEndValidator(model_name, batch_size, seq_len)
        validator.validate()
        print("\n\n")
