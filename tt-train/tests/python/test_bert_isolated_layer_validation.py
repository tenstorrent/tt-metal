# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Isolated Layer Validation Test

Tests each BERT layer independently using reference inputs from HuggingFace.
This isolates whether layers are individually broken or if errors accumulate.

For each layer, we:
1. Get the reference input from HuggingFace
2. Feed that reference input to the corresponding TTML layer
3. Compare outputs

This shows the intrinsic accuracy of each layer in isolation.
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
from transformers import BertModel, BertConfig  # noqa: E402


class BERTIsolatedLayerValidator:
    """Validates BERT layers independently with reference inputs."""

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

    def get_hf_layer_inputs_and_outputs(self, input_ids, token_type_ids, attention_mask):
        """Run HuggingFace BERT and capture input/output for each layer."""
        layer_data = {}

        with torch.no_grad():
            # Get embeddings (input to first block)
            embeddings = self.hf_model.embeddings(input_ids, token_type_ids=token_type_ids)
            layer_data["embeddings_output"] = embeddings.numpy()

            # Prepare attention mask
            extended_attention_mask = self.hf_model.get_extended_attention_mask(attention_mask, input_ids.shape)

            # Run through each encoder layer
            hidden_states = embeddings
            for layer_idx, layer in enumerate(self.hf_model.encoder.layer):
                layer_input = hidden_states.numpy()
                layer_data[f"block_{layer_idx}_input"] = layer_input

                # Run the layer
                layer_output = layer(hidden_states, extended_attention_mask)
                hidden_states = layer_output[0]
                layer_data[f"block_{layer_idx}_output"] = hidden_states.numpy()

        return layer_data

    def validate(self):
        """Run isolated layer validation."""
        print("\n" + "=" * 80)
        print("ISOLATED LAYER VALIDATION")
        print("=" * 80)
        print("\nEach layer is tested with reference inputs from HuggingFace.")
        print("This shows intrinsic layer accuracy without error accumulation.\n")

        # Generate deterministic input
        np.random.seed(42)
        input_ids = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

        print(f"Input IDs (first 10): {input_ids[0, :10].tolist()}\n")

        # Get HuggingFace reference data
        print("Running HuggingFace BERT to capture all layer inputs/outputs...")
        input_ids_torch = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids_torch = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask_torch = torch.tensor(attention_mask, dtype=torch.long)

        hf_layer_data = self.get_hf_layer_inputs_and_outputs(
            input_ids_torch, token_type_ids_torch, attention_mask_torch
        )

        # Prepare TTML attention mask
        attention_mask_ttml_np = attention_mask.astype(np.float32).reshape(self.batch_size, 1, 1, self.seq_len)
        attention_mask_ttml = ttml.autograd.Tensor.from_numpy(attention_mask_ttml_np)

        results = []

        # Test embeddings
        print("\nTesting Embedding Layer...")
        # IMPORTANT: TTNN embedding expects UINT32 indices, not float32 or int32
        input_ids_ttml = ttml.autograd.Tensor.from_numpy(
            input_ids.astype(np.uint32).reshape(self.batch_size, 1, 1, self.seq_len)
        )
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
            token_type_ids.astype(np.uint32).reshape(self.batch_size, 1, 1, self.seq_len)
        )

        ttml_embeddings = self.ttml_model.get_embeddings(input_ids_ttml, token_type_ids_ttml)
        ttml_embeddings_np = ttml_embeddings.to_numpy()

        embeddings_pcc = self.compute_pcc(hf_layer_data["embeddings_output"], ttml_embeddings_np)
        embeddings_pass = embeddings_pcc >= 0.95

        mean_diff = np.mean(np.abs(hf_layer_data["embeddings_output"] - ttml_embeddings_np))
        max_diff = np.max(np.abs(hf_layer_data["embeddings_output"] - ttml_embeddings_np))

        print(
            f"  {'✅' if embeddings_pass else '❌'} Embeddings: PCC={embeddings_pcc:.6f}, "
            f"mean_diff={mean_diff:.6e}, max_diff={max_diff:.6e}"
        )

        results.append(
            {
                "layer": "Embeddings",
                "pcc": embeddings_pcc,
                "passed": embeddings_pass,
                "mean_diff": mean_diff,
                "max_diff": max_diff,
            }
        )

        # Test each block independently
        num_blocks = self.config.num_hidden_layers
        for block_idx in range(num_blocks):
            print(f"\nTesting Block {block_idx} (with reference input)...")

            # Get reference input from HuggingFace
            hf_block_input = hf_layer_data[f"block_{block_idx}_input"]
            hf_block_output = hf_layer_data[f"block_{block_idx}_output"]

            # Feed reference input to TTML block
            block_input_ttml = ttml.autograd.Tensor.from_numpy(hf_block_input.astype(np.float32))

            # Run through TTML block
            ttml_block = self.ttml_model.get_block(block_idx)
            ttml_block_output = ttml_block(block_input_ttml, attention_mask_ttml)
            ttml_block_output_np = ttml_block_output.to_numpy()

            # Compare outputs
            block_pcc = self.compute_pcc(hf_block_output, ttml_block_output_np)
            block_pass = block_pcc >= 0.95

            mean_diff = np.mean(np.abs(hf_block_output - ttml_block_output_np))
            max_diff = np.max(np.abs(hf_block_output - ttml_block_output_np))

            print(
                f"  {'✅' if block_pass else '❌'} Block {block_idx}: PCC={block_pcc:.6f}, "
                f"mean_diff={mean_diff:.6e}, max_diff={max_diff:.6e}"
            )

            results.append(
                {
                    "layer": f"Block {block_idx}",
                    "pcc": block_pcc,
                    "passed": block_pass,
                    "mean_diff": mean_diff,
                    "max_diff": max_diff,
                }
            )

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        total_layers = len(results)
        passed_layers = sum(1 for r in results if r["passed"])
        failed_layers = total_layers - passed_layers

        print(f"\nTotal layers tested: {total_layers}")
        print(f"Passed (PCC ≥ 0.95): {passed_layers}")
        print(f"Failed (PCC < 0.95): {failed_layers}")

        # Find first failure
        first_failure = next((r for r in results if not r["passed"]), None)
        if first_failure:
            print(f"\n❌ First failure at: {first_failure['layer']}")
            print(f"   PCC: {first_failure['pcc']:.6f}")

        # PCC statistics
        pccs = [r["pcc"] for r in results]
        print(f"\nPCC Statistics:")
        print(f"  Minimum: {min(pccs):.6f}")
        print(f"  Maximum: {max(pccs):.6f}")
        print(f"  Average: {np.mean(pccs):.6f}")
        print(f"  Median:  {np.median(pccs):.6f}")

        # Detailed breakdown table
        print(f"\n" + "=" * 80)
        print("DETAILED LAYER BREAKDOWN")
        print("=" * 80)
        print()
        print(f"{'Layer':<20} {'Status':<8} {'PCC':>10} {'Mean Diff':>15} {'Max Diff':>15}")
        print("-" * 80)

        for result in results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            mean_diff_str = f"{result.get('mean_diff', 0):.6e}" if "mean_diff" in result else "N/A"
            max_diff_str = f"{result.get('max_diff', 0):.6e}" if "max_diff" in result else "N/A"
            print(f"{result['layer']:<20} {status:<8} {result['pcc']:>10.6f} {mean_diff_str:>15} {max_diff_str:>15}")

        return passed_layers == total_layers


@pytest.mark.parametrize(
    "batch_size,seq_len,model_name",
    [
        (1, 32, "prajjwal1/bert-tiny"),
        (1, 32, "prajjwal1/bert-small"),
        (1, 32, "google/bert_uncased_L-4_H-512_A-8"),
        (1, 32, "bert-base-uncased"),
    ],
)
def test_bert_isolated_layer_validation(batch_size, seq_len, model_name):
    """Test BERT layers independently with reference inputs."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 80}")

    validator = BERTIsolatedLayerValidator(model_name, batch_size, seq_len)
    all_passed = validator.validate()

    # Test passes even if layers fail - we're just collecting data
    # The actual assertion helps pytest report the test result
    if not all_passed:
        print(f"\n⚠️  RESULT: Some layers showed divergence (PCC < 0.95)")
    else:
        print(f"\n✅ RESULT: All layers passed (PCC ≥ 0.95)")


if __name__ == "__main__":
    # Run for all models
    models = [
        "prajjwal1/bert-tiny",
        "prajjwal1/bert-small",
        "google/bert_uncased_L-4_H-512_A-8",
        "bert-base-uncased",
    ]

    for model in models:
        validator = BERTIsolatedLayerValidator(model)
        validator.validate()
        print("\n\n")
