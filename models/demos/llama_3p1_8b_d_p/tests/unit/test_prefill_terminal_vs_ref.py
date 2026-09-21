# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""All-chip embedding and terminal projection gates before the full prefill wrapper."""

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.model.reference import chat_tokens, rms
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics
from models.demos.llama_3p1_8b_d_p.tests.utils import positions as _positions
from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.input import upload_token_chunk
from models.demos.llama_3p1_8b_d_p.tt.model import FinalNormHead, TokenEmbedding

CHECKPOINT = os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct")


def _assert_metric(expected, actual, label):
    pcc, nl2 = metrics(expected, actual)
    logger.info(f"{label}: PCC={pcc:.9f}, NL2={nl2:.9f}")
    assert pcc >= 0.999 and nl2 <= 0.025, (label, pcc, nl2)


# Exact embedding lookup is checked on all 32 chips, including rotated SP starts and padding.
# Distinct token allocations remain live together, exposing stale cached addresses during replay.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prefill_embedding_real_tokens_and_rotated_starts(mesh_device):
    assert mesh_device.get_num_devices() == 32
    raw = read_raw_weights(CHECKPOINT, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    embedding = TokenEmbedding(mesh_device, raw)
    stream, _ = chat_tokens(CHECKPOINT, slot=0, length=2048)
    held, token_allocations = [], []
    try:
        for start, end in [(0, 1), (0, 33), (0, 1024), (224, 257), (1024, 1033), (2016, 2048), (0, 33)]:
            ids = upload_token_chunk(mesh_device, stream[start:end], actual_start=start, actual_end=end)
            held.append(ids)
            token_allocations.append(ids)
            # A runtime input is 3-D; a host upload is 4-D. The 3-D view shares IDs, so only the
            # original allocation is explicitly freed after both entry forms finish.
            for supplied in (ids, ttnn.reshape(ids, (1, 1, 256))):
                output = embedding(supplied)
                held.append(output)
                assert tuple(output.shape) == (1, 1, 256, 4096)
                for chip, (token_shard, output_shard) in enumerate(
                    zip(ttnn.get_device_tensors(supplied), ttnn.get_device_tensors(output))
                ):
                    positions = _positions(start, chip // 8)
                    expected_ids = torch.tensor([stream[p].item() if p < end else 0 for p in positions])
                    assert torch.equal(ttnn.to_torch(token_shard).flatten().long(), expected_ids)
                    assert torch.equal(ttnn.to_torch(output_shard)[0, 0], F.embedding(expected_ids, raw).bfloat16())
        addresses = [
            tuple(int(shard.buffer_address()) for shard in ttnn.get_device_tensors(t)) for t in token_allocations
        ]
        for earlier, later in zip(addresses, addresses[1:]):
            assert all(a != b for a, b in zip(earlier, later))
    finally:
        for tensor in held:
            tensor.deallocate(True)
        embedding.close()


# Real untied norm/head weights are compared against raw HF mathematics per TP vocabulary slice.
# Boundary tokens16031/16032 and128255 prove exact shard offsets without padded vocabulary IDs.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prefill_final_norm_and_exact_vocabulary(mesh_device):
    raw = read_raw_weights(CHECKPOINT, ["model.norm.weight", "lm_head.weight"])
    module = FinalNormHead(mesh_device, MeshConfig((4, 8), 8), raw["model.norm.weight"], raw["lm_head.weight"])
    generator = torch.Generator().manual_seed(80031)
    host = (torch.randn(1, 1, 1024, 4096, generator=generator) * 0.2).bfloat16()
    value = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(4, 8), dims=(2, None)),
    )
    outputs = []
    try:
        for _ in range(2):
            output = module(value)
            outputs.append(output)
            assert tuple(output.shape) == (1, 1, 256, 16032)
            for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                sp, tp = divmod(chip, 8)
                # Score 32 distributed rows to bound CPU projection cost. Every vocabulary entry
                # participates, including each chip's first and last entry.
                rows = torch.arange(0, 256, 8)
                hidden = host[0, 0, sp * 256 + rows].float()
                expected = F.linear(
                    rms(hidden, raw["model.norm.weight"], 1e-5),
                    raw["lm_head.weight"][tp * 16032 : (tp + 1) * 16032].float(),
                )
                actual = ttnn.to_torch(shard)[0, 0, rows].float()
                _assert_metric(expected, actual, f"final head chip={chip}")
            for chip, shard in enumerate(ttnn.get_device_tensors(value)):
                assert torch.equal(ttnn.to_torch(shard), host[:, :, (chip // 8) * 256 : (chip // 8 + 1) * 256])
        for first, second in zip(ttnn.get_device_tensors(outputs[0]), ttnn.get_device_tensors(outputs[1])):
            assert first.buffer_address() != second.buffer_address()
            assert torch.equal(ttnn.to_torch(first), ttnn.to_torch(second))
    finally:
        for tensor in outputs:
            tensor.deallocate(True)
        value.deallocate(True)
        module.close()
