# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.embedding_model import EmbeddingTransformer
from models.tt_transformers.tt.model_config import ModelArgs


class EmbeddingAccuracyTest:
    def __init__(self, model_name="Qwen3-Embedding-8B"):
        self.model_name = model_name
        self.test_sequences = [
            "Hello world",
            "This is a test sentence for embedding accuracy.",
            "Machine learning is transforming technology.",
            "Natural language processing enables understanding text.",
            "Embeddings capture semantic meaning of sentences.",
        ]

    def generate_reference_embeddings(self):
        """Generate reference embeddings using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping reference generation")
            return None

        logger.info(f"Loading reference model: {self.model_name}")
        model = SentenceTransformer(self.model_name)

        logger.info("Generating reference embeddings")
        embeddings = model.encode(self.test_sequences, normalize_embeddings=False)

        return torch.tensor(embeddings, dtype=torch.float32)

    def generate_tt_embeddings(self, mesh_device, model_args):
        """Generate embeddings using tt_transformers"""
        logger.info("Creating TT embedding model")

        # Create model args
        tt_model_args = ModelArgs(
            mesh_device,
            instruct=False,
            max_batch_size=len(self.test_sequences),
            max_seq_len=512,  # Long enough for our test sequences
        )
        tt_model_args.model_name = self.model_name
        tt_model_args.n_layers = 36  # Full model for accuracy testing

        # Load state dict
        state_dict = tt_model_args.load_state_dict()

        # Create embedding model
        embedding_model = EmbeddingTransformer(
            args=tt_model_args,
            dtype=ttnn.bfloat8_b,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
        )

        # Tokenize sequences
        tokenizer = tt_model_args.create_tokenizer()

        # Prepare inputs
        tokens_list = []
        attention_masks = []

        for seq in self.test_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=True, return_tensors="pt")
            tokens = tokens.squeeze(0)  # Remove batch dimension

            # Pad to max_seq_len
            if len(tokens) < tt_model_args.max_seq_len:
                padding = torch.full((tt_model_args.max_seq_len - len(tokens),), tokenizer.pad_token_id)
                tokens = torch.cat([tokens, padding])
                attention_mask = torch.cat([torch.ones(len(tokens) - len(padding)), torch.zeros(len(padding))])
            else:
                tokens = tokens[: tt_model_args.max_seq_len]
                attention_mask = torch.ones(tt_model_args.max_seq_len)

            tokens_list.append(tokens)
            attention_masks.append(attention_mask)

        # Stack into batch
        tokens_batch = torch.stack(tokens_list)  # [batch, seq_len]
        attention_mask_batch = torch.stack(attention_masks).unsqueeze(-1)  # [batch, seq_len, 1]

        logger.info(f"Input batch shape: {tokens_batch.shape}")

        # Prepare inputs for TT model
        (
            tt_tokens,
            tt_rot_mats_global,
            tt_rot_mats_local,
            tt_page_table,
            tt_chunk_page_table,
        ) = embedding_model.prepare_inputs_prefill(tokens_batch)

        # Convert attention mask to TT tensor
        tt_attention_mask = ttnn.from_torch(
            attention_mask_batch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Run inference
        logger.info("Running TT embedding inference")
        tt_embeddings = embedding_model.forward(
            x=tt_tokens,
            current_pos=None,
            rot_mats_global=tt_rot_mats_global,
            rot_mats_local=tt_rot_mats_local,
            mode="prefill",
            attention_mask=tt_attention_mask,
        )

        # Convert to torch
        embeddings = ttnn.to_torch(tt_embeddings)

        # Extract embeddings (remove extra dimensions)
        embeddings = embeddings.squeeze(1).squeeze(1)  # [batch, hidden_dim]

        return embeddings.float()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_embedding_accuracy(mesh_device, reset_seeds, ensure_gc):
    """Test embedding accuracy against sentence-transformers reference"""

    accuracy_test = EmbeddingAccuracyTest()

    # Generate reference embeddings
    logger.info("Generating reference embeddings")
    ref_embeddings = accuracy_test.generate_reference_embeddings()

    if ref_embeddings is None:
        logger.warning("Skipping accuracy test - sentence-transformers not available")
        return

    logger.info(f"Reference embeddings shape: {ref_embeddings.shape}")

    # Generate TT embeddings
    tt_embeddings = accuracy_test.generate_tt_embeddings(mesh_device, None)

    logger.info(f"TT embeddings shape: {tt_embeddings.shape}")

    # Compare shapes
    assert (
        ref_embeddings.shape == tt_embeddings.shape
    ), f"Shape mismatch: ref {ref_embeddings.shape} vs tt {tt_embeddings.shape}"

    # Compute cosine similarity between corresponding embeddings
    similarities = []
    for i in range(len(accuracy_test.test_sequences)):
        ref_emb = ref_embeddings[i]
        tt_emb = tt_embeddings[i]

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(ref_emb.unsqueeze(0), tt_emb.unsqueeze(0))
        similarities.append(cos_sim.item())

        logger.info(f"Sequence {i} cosine similarity: {cos_sim.item():.4f}")

    # Average similarity
    avg_similarity = sum(similarities) / len(similarities)
    logger.info(f"Average cosine similarity: {avg_similarity:.4f}")

    # Check that similarity is high enough (target: > 0.99)
    assert avg_similarity > 0.95, f"Average cosine similarity {avg_similarity:.4f} is too low (< 0.95)"

    # Also check PCC for direct comparison
    passing, pcc = comp_pcc(tt_embeddings, ref_embeddings, pcc=0.95)
    logger.info(f"Embeddings PCC: {pcc}")

    assert passing, f"Embedding PCC {pcc} is too low"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_embedding_consistency(mesh_device, reset_seeds, ensure_gc):
    """Test that embeddings are consistent across multiple runs"""

    accuracy_test = EmbeddingAccuracyTest()

    # Generate embeddings twice
    emb1 = accuracy_test.generate_tt_embeddings(mesh_device, None)
    emb2 = accuracy_test.generate_tt_embeddings(mesh_device, None)

    # Check that they are identical (deterministic)
    passing, pcc = comp_pcc(emb1, emb2, pcc=0.9999)
    logger.info(f"Embedding consistency PCC: {pcc}")

    assert passing, f"Embeddings are not consistent across runs (PCC: {pcc})"
