# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter on QB2 (1x4 Blackhole), reduced to real layers 0 (sliding) and 3 (global):
with an externally allocated KV cache, a resident-input decode (inputs zeroed,
reload_inputs=False) must reproduce the authoritative greedy sequence exactly, and the
synchronous path (process_decode_output_host on an unread device output) must agree with
the async read. Needs the checkpoint (MODEL_WEIGHTS_DIR, HF_MODEL or the HF cache)."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.muse_glimmer_30b_qb2.tt.generator_vllm import MuseGlimmerForConditionalGeneration

LAYERS = 52
SLOTS = 4


@pytest.fixture
def adapter():
    os.environ["MUSE_VLLM_LAYER_INDICES"] = "0,3"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    obj = None
    try:
        obj = MuseGlimmerForConditionalGeneration.initialize_vllm_model(None, mesh, SLOTS)
        yield obj
    finally:
        if obj is not None and obj.generator is not None:
            obj.generator.close()
        os.environ.pop("MUSE_VLLM_LAYER_INDICES", None)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_adapter_resident_decode_matches_reload(adapter):
    specs = [((4096, 1, 32, 128), torch.bfloat16, i) for i in range(LAYERS)]
    cache = adapter.allocate_kv_cache_per_layer(specs)
    adapter.warmup_model_prefill(cache, True, True)
    gen = adapter.generator
    params = SimpleNamespace(temperature=[0.0] * SLOTS, top_k=[1] * SLOTS, top_p=[1.0] * SLOTS, seed=[11, 12, 13, 14])
    tables = []
    for i in range(LAYERS):
        table = torch.zeros((SLOTS, 4096), dtype=torch.int32)
        for slot in range(SLOTS):
            # Layer 3 (global) gets pages disjoint from the sliding layers.
            table[slot, :64] = torch.arange(64) + slot * 64 + (512 if i == 3 else 0)
        tables.append(table)
    # Non-aligned prompt lengths around a 128-token boundary.
    prompts = torch.tensor([[200000] + [100 + slot] * 130 for slot in range(SLOTS)])
    lengths = [129, 130, 131, 129]

    sequences = []
    for trial in range(2):
        prefill, _ = adapter.prefill_forward(
            prompts, tables[0], cache, lengths, page_tables_per_layer=tables, sampling_params=params
        )
        seq = [prefill.flatten().tolist()]
        pos = torch.tensor(lengths, dtype=torch.int32)
        token = prefill
        for step in range(4):
            authoritative = trial == 0 or step == 0
            device = adapter.decode_forward(
                token if authoritative else torch.zeros_like(token),
                pos if authoritative else torch.zeros_like(pos),
                tables[0],
                cache,
                sampling_params=params,
                page_tables_per_layer=tables,
                read_from_device=False,
                reload_inputs=authoritative,
                reload_sampling_params=step == 0,
                reset_sampling_state=step == 0,
            )
            if trial == 0:
                # Synchronous path: the plugin hands over the unread device output.
                token, _ = adapter.process_decode_output_host(device, is_tokens=True)
            else:
                host, events = adapter.read_decode_output(device, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                token, _ = adapter.process_decode_output_host(host, is_tokens=True)
            seq.append(token.flatten().tolist())
            pos += 1
            actual = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).flatten()[:SLOTS]
            assert torch.equal(actual.to(torch.int32), pos), (actual, pos)
        sequences.append(seq)
    assert sequences[0] == sequences[1], sequences


def test_adapter_host_sampled_prefill_returns_logits(adapter):
    specs = [((4096, 1, 32, 128), torch.bfloat16, i) for i in range(LAYERS)]
    cache = adapter.allocate_kv_cache_per_layer(specs)
    adapter.warmup_model_prefill(cache, True, True)
    tables = [torch.arange(4 * 64, dtype=torch.int32).reshape(SLOTS, 64) for _ in range(LAYERS)]
    tables[3] = tables[3] + 512
    prompts = torch.tensor([[200000] + [100 + slot] * 130 for slot in range(SLOTS)])
    # A step that carries an intermediate chunk arrives without sampling params.
    logits, _ = adapter.prefill_forward(prompts, tables[0], cache, [131] * SLOTS, page_tables_per_layer=tables)
    assert logits.shape[0] == SLOTS and logits.shape[-1] >= adapter.model.vocab_size, logits.shape
    assert torch.isfinite(logits.float()).all()
