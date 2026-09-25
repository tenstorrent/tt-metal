# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in pretrained attention-block qualification, not a full video evaluation."""

import json
import os
import statistics
import time

import pytest
import torch
import ttnn

from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, metrics
from .test_sdpa_recipe_ring import recipe_ring_device

pytestmark = pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_WAN") != "1", reason="Opt-in pretrained Wan block")
REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


@pytest.fixture(scope="module")
def wan_block(recipe_ring_device, tmp_path_factory):
    from diffusers.models.transformers.transformer_wan import WanAttention as TorchAttention
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from models.tt_dit.models.transformers.wan2_2.attention_wan import WanAttention
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager

    mesh = recipe_ring_device[0]

    def download(name):
        return hf_hub_download(REPO, f"transformer/{name}", revision=REVISION)

    with open(download("diffusion_pytorch_model.safetensors.index.json")) as handle:
        index = json.load(handle)["weight_map"]
    prefix = "blocks.0.attn1."
    state = {}
    for shard in sorted({value for key, value in index.items() if key.startswith(prefix)}):
        with safe_open(download(shard), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key).float()
    reference = TorchAttention(dim=5120, heads=40, dim_head=128, eps=1e-6, processor=WanAttnProcessor2_0()).eval()
    reference.load_state_dict(state, strict=True)
    manager = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    config = DiTParallelConfig(
        cfg_parallel=None,
        tensor_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(2, 1),
    )
    cache = tmp_path_factory.mktemp("wan-recipe-weights")
    initialized = False

    def construct(variant):
        nonlocal initialized
        precision = (
            None if variant == "legacy" else getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION"))
        )
        dtype = {"E_bfp4": ttnn.bfloat4_b, "E_bfp8": ttnn.bfloat8_b}.get(variant, ttnn.bfloat16)
        model = WanAttention(
            dim=5120,
            num_heads=40,
            eps=1e-6,
            mesh_device=mesh,
            ccl_manager=manager,
            parallel_config=config,
            sdpa_precision=precision,
            sdpa_kv_dtype=dtype,
        )
        if initialized:
            model.load(cache)
        else:
            model.load_torch_state_dict(state)
            model.save(cache)
            initialized = True
        return model

    return reference, construct


@pytest.mark.parametrize("variant", ["legacy", *VARIANTS])
@pytest.mark.parametrize("sequence", [1024, 2048])
def test_wan_pretrained_attention(recipe_ring_device, wan_block, variant, sequence, record_property):
    mesh = recipe_ring_device[0]
    reference, construct = wan_block
    generator = torch.Generator().manual_seed(sequence)
    host = torch.randn(1, sequence, 5120, generator=generator).bfloat16()
    with torch.no_grad():
        expected = reference(host.float())
    inputs = ttnn.from_torch(
        host.unsqueeze(0),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )
    model = construct(variant)

    def collect(tensor):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)).squeeze(0)

    try:
        actual = collect(model(inputs, N=sequence))
        result = metrics(actual, expected)
        for key, value in result.items():
            record_property(key, value)
        record_property("model_revision", REVISION)
        assert result["pcc"] >= 0.988
        assert result["l2_pct"] < (20 if variant.startswith("E_") else 10)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = model(inputs, N=sequence)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            samples = []
            for iteration in range(7):
                start = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if iteration >= 2:
                    samples.append((time.perf_counter() - start) * 1000)
            assert digest(collect(traced)) == digest(actual)
            record_property("block_trace_wall_ms_median", statistics.median(samples))
            record_property("preparation_in_timing", True)
        finally:
            ttnn.release_trace(mesh, trace)
    finally:
        model.deallocate_weights()
