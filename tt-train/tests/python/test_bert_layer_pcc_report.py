# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Layer PCC Report - Clean reporting format

Generates a clean PCC report for each layer across multiple BERT models.
Shows layer-by-layer PCC in a table format for easy comparison.

Note: This runs through the model cumulatively (each layer receives the
previous layer's output from TTML). For true isolation (feeding reference
inputs to each layer), we would need to expose individual blocks in Python bindings.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(golden: np.ndarray, actual: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    if len(golden_flat) != len(actual_flat):
        return 0.0

    mean_golden = np.mean(golden_flat)
    mean_actual = np.mean(actual_flat)
    numerator = np.sum((golden_flat - mean_golden) * (actual_flat - mean_actual))
    denominator = np.sqrt(np.sum((golden_flat - mean_golden) ** 2) * np.sum((actual_flat - mean_actual) ** 2))
    return numerator / denominator if denominator > 0 else (1.0 if numerator == 0 else 0.0)


class BERTLayerPCCReporter:
    """Generate PCC report for all BERT layers."""

    def __init__(self, model_name: str, batch_size: int = 1, seq_len: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Load HuggingFace model
        self.hf_model = transformers.BertModel.from_pretrained(model_name)
        self.hf_model.eval()
        self.hf_config = self.hf_model.config

        # Save to safetensors
        safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
        if not safetensors_path.exists():
            from safetensors.torch import save_file

            save_file(self.hf_model.state_dict(), str(safetensors_path))

        # Create TTML model
        ttml_config = ttml.models.bert.BertConfig()
        ttml_config.vocab_size = self.hf_config.vocab_size
        ttml_config.max_sequence_length = seq_len
        ttml_config.embedding_dim = self.hf_config.hidden_size
        ttml_config.intermediate_size = self.hf_config.intermediate_size
        ttml_config.num_heads = self.hf_config.num_attention_heads
        ttml_config.num_blocks = self.hf_config.num_hidden_layers
        ttml_config.dropout_prob = 0.0
        ttml_config.layer_norm_eps = self.hf_config.layer_norm_eps
        ttml_config.use_token_type_embeddings = True
        ttml_config.use_pooler = False

        self.ttml_model = ttml.models.bert.create(ttml_config)
        self.ttml_model.load_model_from_safetensors(str(safetensors_path))

    def get_hf_intermediate_outputs(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Run HuggingFace BERT and capture all intermediate outputs."""
        outputs = {}

        with torch.no_grad():
            # Get embeddings
            embeddings = self.hf_model.embeddings(input_ids, token_type_ids=token_type_ids)
            outputs["embeddings"] = embeddings.numpy()

            # Run through each encoder layer
            hidden_states = embeddings
            for layer_idx, layer in enumerate(self.hf_model.encoder.layer):
                # Run attention
                attention_output = layer.attention(hidden_states)[0]
                attention_residual = attention_output + hidden_states
                attention_norm = layer.attention.output.LayerNorm(attention_residual)
                outputs[f"block_{layer_idx}_attention"] = attention_norm.numpy()

                # Run FFN
                ffn_output = layer.intermediate(attention_norm)
                ffn_output = layer.output.dense(ffn_output)
                ffn_residual = ffn_output + attention_norm
                block_output = layer.output.LayerNorm(ffn_residual)
                outputs[f"block_{layer_idx}_output"] = block_output.numpy()

                hidden_states = block_output

            outputs["final"] = hidden_states.numpy()

        return outputs

    def generate_report(self):
        """Generate clean PCC report for all layers."""
        # Generate deterministic input
        np.random.seed(42)
        input_ids_np = np.random.randint(100, 1000, size=(self.batch_size, self.seq_len), dtype=np.int32)
        token_type_ids_np = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        attention_mask_np = np.ones((self.batch_size, self.seq_len), dtype=np.float32)

        # Get HuggingFace outputs
        input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)
        token_type_ids_torch = torch.tensor(token_type_ids_np, dtype=torch.long)

        hf_outputs = self.get_hf_intermediate_outputs(input_ids_torch, token_type_ids_torch)

        # Get TTML outputs (convert to uint32 for TTML)
        # IMPORTANT: TTNN embedding expects UINT32 indices, not float32 or int32
        input_ids_ttml = ttml.autograd.Tensor.from_numpy(
            input_ids_np.astype(np.uint32).reshape(self.batch_size, 1, 1, self.seq_len)
        )
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
            token_type_ids_np.astype(np.uint32).reshape(self.batch_size, 1, 1, self.seq_len)
        )
        attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
            attention_mask_np.reshape(self.batch_size, 1, 1, self.seq_len)
        )

        ttml_intermediates = self.ttml_model.forward_with_intermediates(
            input_ids_ttml, attention_mask_ttml, token_type_ids_ttml
        )

        # Build PCC report
        report = []

        # Embeddings
        emb_pcc = compute_pcc(hf_outputs["embeddings"], ttml_intermediates.embeddings.to_numpy())
        report.append(("Embeddings", emb_pcc))

        # Each block
        for block_idx in range(self.hf_config.num_hidden_layers):
            attn_pcc = compute_pcc(
                hf_outputs[f"block_{block_idx}_attention"],
                ttml_intermediates.block_attention_outputs[block_idx].to_numpy(),
            )
            report.append((f"Block {block_idx} Attn", attn_pcc))

            output_pcc = compute_pcc(
                hf_outputs[f"block_{block_idx}_output"],
                ttml_intermediates.block_outputs[block_idx].to_numpy(),
            )
            report.append((f"Block {block_idx} Out", output_pcc))

        # Final
        final_pcc = compute_pcc(hf_outputs["final"], ttml_intermediates.final_output.to_numpy())
        report.append(("Final", final_pcc))

        return report


def print_multi_model_report(models: List[str]):
    """Print comprehensive PCC report for multiple models."""
    print("\n" + "=" * 100)
    print("BERT LAYER-BY-LAYER PCC REPORT")
    print("=" * 100)
    print("\nComparing HuggingFace reference vs TTML implementation")
    print("Note: Each layer receives output from previous TTML layer (cumulative errors)")
    print("=" * 100)

    # Collect reports for all models
    model_reports = {}
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        reporter = BERTLayerPCCReporter(model_name)
        model_reports[model_name] = reporter.generate_report()

    # Find max layers across all models
    max_layers = max(len(report) for report in model_reports.values())

    # Print table header
    print("\n" + "=" * 100)
    header = f"{'Layer':<20}"
    for model_name in models:
        short_name = model_name.split("/")[-1][:15]
        header += f"{short_name:>15}"
    print(header)
    print("-" * 100)

    # Print each layer row
    all_layer_names = []
    for i in range(max_layers):
        # Get layer name from first model that has this layer
        layer_name = None
        for model_name in models:
            if i < len(model_reports[model_name]):
                layer_name = model_reports[model_name][i][0]
                break

        if layer_name is None:
            continue

        if layer_name not in all_layer_names:
            all_layer_names.append(layer_name)

        row = f"{layer_name:<20}"
        for model_name in models:
            if i < len(model_reports[model_name]):
                pcc = model_reports[model_name][i][1]
                status = "✅" if pcc >= 0.95 else "❌"
                row += f"  {status} {pcc:6.4f}    "
            else:
                row += f"{'':>15}"
        print(row)

    print("=" * 100)

    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 100)
    print(f"{'Model':<40} {'Layers':<10} {'Pass/Total':<15} {'Min PCC':<12} {'Max PCC':<12} {'Avg PCC':<12}")
    print("-" * 100)

    for model_name in models:
        report = model_reports[model_name]
        pccs = [pcc for _, pcc in report]
        passed = sum(1 for pcc in pccs if pcc >= 0.95)
        total = len(pccs)

        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        print(
            f"{short_name:<40} {total:<10} {f'{passed}/{total}':<15} "
            f"{min(pccs):<12.6f} {max(pccs):<12.6f} {np.mean(pccs):<12.6f}"
        )

    print("=" * 100)
    print()


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 32),
    ],
)
def test_bert_layer_pcc_report(batch_size, seq_len):
    """Generate comprehensive PCC report for all BERT models."""
    models = [
        "prajjwal1/bert-tiny",
        "prajjwal1/bert-small",
        "google/bert_uncased_L-4_H-512_A-8",
        "bert-base-uncased",
    ]

    print_multi_model_report(models)


if __name__ == "__main__":
    models = [
        "prajjwal1/bert-tiny",
        "prajjwal1/bert-small",
        "google/bert_uncased_L-4_H-512_A-8",
        "bert-base-uncased",
    ]

    print_multi_model_report(models)
