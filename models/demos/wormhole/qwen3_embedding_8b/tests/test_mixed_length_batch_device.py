# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device check that a right-padded batch embeds each request like the same text sent alone."""

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import Qwen3ForEmbedding

MODEL = "Qwen/Qwen3-Embedding-8B"
SHORT = "What is the capital of France?"
MEDIUM = " ".join(["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."] * 4)
LONG = " ".join(["Tenstorrent builds AI accelerators from a mesh of Tensix cores and RISC-V processors."] * 20)
MIN_COSINE = 0.99


@pytest.mark.parametrize("device_params", [{"fabric_config": True, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_mixed_length_batch_matches_single_request(mesh_device, device_params):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
    embedder = Qwen3ForEmbedding(device=mesh_device, max_batch_size=2, max_seq_len=2048, model_name=MODEL)

    def embed(texts):
        enc = tokenizer(texts, padding=True, return_tensors="pt")
        return embedder.forward(enc["input_ids"], attention_mask=enc["attention_mask"]).float()

    # SHORT and LONG land in different prefill buckets (per-user path); SHORT and MEDIUM share one (batched path).
    worst = 1.0
    for pair in ([SHORT, LONG], [SHORT, MEDIUM]):
        lengths = tokenizer(pair, padding=True, return_tensors="pt")["attention_mask"].sum(dim=1).tolist()
        batched = embed(pair)
        for row, text in enumerate(pair):
            alone = embed([text])[0]
            cosine = torch.nn.functional.cosine_similarity(batched[row], alone, dim=0).item()
            worst = min(worst, cosine)
            logger.info(f"real lengths {lengths}, row {row}: cosine to the text alone {cosine:.6f}")
            assert cosine >= MIN_COSINE, f"row {row} of a {lengths} batch: cosine {cosine:.6f} < {MIN_COSINE}"
    logger.info(f"Worst cosine to the text alone: {worst:.6f}")
