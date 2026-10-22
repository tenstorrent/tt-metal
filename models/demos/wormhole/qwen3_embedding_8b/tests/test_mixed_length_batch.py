# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-device check that a request gets the same embedding in a mixed-length batch as when it is sent alone.

The two requests are 40 and 300 tokens long, so they land in different prefill buckets (128 and 1024) and take
the per-user prefill path. Before the fix, the 40-token row was read at the batch length, in its padding.

Runs on an N150, N300 or T3K (both buckets are traced on each), with HF_MODEL=Qwen/Qwen3-Embedding-8B and vLLM
(the model is built by tt_transformers' generator_vllm, which imports vllm). It is not part of any CI leg:

    HF_MODEL=Qwen/Qwen3-Embedding-8B pytest models/demos/wormhole/qwen3_embedding_8b/tests/test_mixed_length_batch.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

pytest.importorskip("vllm")

from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import Qwen3ForEmbedding  # noqa: E402

PAD = 151643  # <|endoftext|>, which the server pads a batch with
LENGTHS = [40, 300]  # under and over 128 tokens


@pytest.mark.skipif(
    "Qwen3-Embedding-8B" not in os.environ.get("HF_MODEL", ""), reason="needs HF_MODEL=Qwen/Qwen3-Embedding-8B"
)
# Fabric needs at least two chips: a single N150 runs without it.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": len(ttnn.get_device_ids()) > 1, "num_command_queues": 1}], indirect=True
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_mixed_length_batch_matches_each_request_alone(mesh_device):
    embedder = Qwen3ForEmbedding(device=mesh_device, max_batch_size=2)
    generator = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1000, 150000, (n,), generator=generator) for n in LENGTHS]

    # Right-padded batch, as the server tokenizes it
    input_ids = torch.full((len(prompts), max(LENGTHS)), PAD, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        input_ids[row, : len(prompt)] = prompt
        attention_mask[row, : len(prompt)] = 1

    batched = embedder.forward(input_ids, attention_mask=attention_mask).float()
    alone = torch.cat(
        [embedder.forward(p[None], attention_mask=torch.ones(1, len(p), dtype=torch.long)) for p in prompts]
    ).float()

    cosine = torch.nn.functional.cosine_similarity(batched, alone, dim=-1)
    logger.info(f"cosine of each row of the batch to the same request alone: {cosine.tolist()}")
    assert (cosine > 0.99).all(), f"a row of the {LENGTHS}-token batch differs from its request alone: {cosine}"
