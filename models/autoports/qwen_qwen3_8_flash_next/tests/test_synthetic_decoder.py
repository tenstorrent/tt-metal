# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline CI at actual Qwen3.8-Flash-Next layer shapes, with synthetic weights.

Run under a bounded timeout with an operator-selected TT_VISIBLE_DEVICES. This
file opens only a1x1 mesh and never downloads model weights. Layers0/3 tests
exercise complete decoders. PLE tests exercise arithmetic with explicit prepared
embedding features; they do NOT claim complete PLE decoder/hash/lookup coverage.
Advertised context remains262144; these tests are short CI cases, not capacity
acceptance evidence. Run the real-weight/context harnesses for stage acceptance.
"""

import gc

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextPLELayer,
    Qwen4ExpTextRotaryEmbedding,
)
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.synthetic_weights import (
    embedding_fixture,
    layer_weights,
    target_config,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.ple import PLE


def assert_pcc(expected, actual, *, label):
    actual = actual.reshape(expected.shape)
    assert torch.isfinite(actual).all(), f"{label}: nonfinite device output"
    value = torch.corrcoef(torch.stack((expected.float().flatten(), actual.float().flatten())))[0, 1].item()
    assert value >= 0.995, f"{label}: PCC={value}, required>=0.995"
    return value


def host(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)


def device(value, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.to_device(host(value, dtype=dtype, layout=layout), mesh)


def snapshot(buffers):
    return {
        name: host(ttnn.to_torch(tensor), dtype=tensor.dtype, layout=tensor.layout) for name, tensor in buffers.items()
    }


def restore(buffers, saved):
    for name, tensor in buffers.items():
        ttnn.copy_host_to_device_tensor(saved[name], tensor)


@pytest.fixture(scope="module")
def single_chip_mesh():
    """Do not request repository device/mesh fixtures, which may enumerate chips."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module", params=(0, 3), ids=("linear_layer0", "indexed_layer3"))
def decoder_bundle(request, single_chip_mesh):
    config = target_config()
    weights = layer_weights(request.param)
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, request.param)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    model = FunctionalDecoder.from_state_dict(
        weights, hf_config=config, layer_idx=request.param, mesh_device=single_chip_mesh
    )
    yield config, request.param, reference, model
    del model, reference, weights
    gc.collect()


@pytest.mark.parametrize("sequence", (1, 33))
def test_full_decoder_prefill_and_traced_decode(decoder_bundle, single_chip_mesh, sequence, record_property):
    config, layer_idx, reference, model = decoder_bundle
    generator = torch.Generator().manual_seed(1900 + 100 * layer_idx + sequence)
    hidden = torch.empty(1, sequence + 3, 10240, dtype=torch.bfloat16).normal_(mean=0, std=0.02, generator=generator)
    indexed = layer_idx == 3
    cache = DynamicCache(config=config)
    with torch.inference_mode():
        if indexed:
            rope = Qwen4ExpTextRotaryEmbedding(config)
            cos, sin = rope(hidden, torch.arange(sequence + 3).reshape(1, -1))
            mask = torch.full((sequence, sequence), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16).triu(1)
            expected_prefill = reference(
                hidden[:, :sequence],
                position_embeddings=(cos[:, :sequence], sin[:, :sequence]),
                attention_mask=mask[None, None],
                past_key_values=cache,
            )
        else:
            expected_prefill = reference(hidden[:, :sequence], position_embeddings=(None, None), past_key_values=cache)
        expected_decode = []
        for position in range(sequence, sequence + 3):
            expected_decode.append(
                reference(
                    hidden[:, position : position + 1],
                    position_embeddings=(cos[:, : position + 1], sin[:, : position + 1]) if indexed else (None, None),
                    attention_mask=torch.zeros(1, 1, 1, position + 1, dtype=torch.bfloat16) if indexed else None,
                    past_key_values=cache,
                )
            )
    # 128 is a cache fixture covering these tests; config/context are untouched.
    page_table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    state = model.create_state(capacity=128, page_table=page_table) if indexed else model.create_state()
    if indexed:
        names = ("position", "k_cache", "v_cache", "raw_history", "compressed_keys")
        buffers = {name: getattr(state.attention, name) for name in names}
    else:
        buffers = {"conv": state.conv, "recurrent": state.recurrent}
    initial = snapshot(buffers)
    prompt = device(hidden[:, :sequence].unsqueeze(0), single_chip_mesh)
    current = device(hidden[:, sequence : sequence + 1].unsqueeze(0), single_chip_mesh)
    refresh = [host(hidden[:, position : position + 1].unsqueeze(0)) for position in range(sequence, sequence + 3)]
    output = model.prefill_forward(prompt, state=state)
    actual_prefill = ttnn.to_torch(output)
    record_property("prefill_pcc", assert_pcc(expected_prefill, actual_prefill, label="synthetic prefill"))
    ttnn.deallocate(output)
    saved = snapshot(buffers)
    warm = model.decode_forward(current, state=state)
    ttnn.synchronize_device(single_chip_mesh)
    ttnn.deallocate(warm)
    restore(buffers, saved)
    trace = ttnn.begin_trace_capture(single_chip_mesh, cq_id=0)
    output = model.decode_forward(current, state=state)
    ttnn.end_trace_capture(single_chip_mesh, trace, cq_id=0)
    # Exact output only: consumed by synchronous readback before any replay.
    trace_allocation_tracker.acknowledge_corruptible(output)
    decoded = []
    try:
        for repetition in range(2):
            restore(buffers, saved)  # Capture also advances device cache/position.
            values = []
            for step, expected in enumerate(expected_decode):
                ttnn.copy_host_to_device_tensor(refresh[step], current)
                ttnn.execute_trace(single_chip_mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(output).reshape(expected.shape)
                value = assert_pcc(expected, actual, label=f"synthetic decode step{step}, repeat{repetition}")
                if repetition == 0:
                    record_property(f"decode_{step}_pcc", value)
                values.append(actual.clone())
            decoded.append(values)
        assert all(torch.equal(a, b) for a, b in zip(*decoded)), "reset trace replay is not deterministic"
    finally:
        ttnn.release_trace(single_chip_mesh, trace)
    restore(buffers, initial)
    reset_output = model.prefill_forward(prompt, state=state)
    assert torch.equal(actual_prefill, ttnn.to_torch(reset_output)), "fresh state did not reproduce prefill"
    record_property("weights", "deterministic synthetic from pinned per-tensor mean/std; actual target shapes")


class PreparedEmbedding(torch.nn.Module):
    """Explicit diagnostic feature boundary, never a claimed decoder implementation."""

    def __init__(self):
        super().__init__()
        self.current = None

    def forward(self, input_ids, past_key_values):
        assert self.current is not None
        return self.current


@pytest.mark.parametrize("sequence", (1, 33))
def test_ple_arithmetic_only(single_chip_mesh, sequence, record_property):
    """Does not exercise n-gram hashing or the full resident embedding table."""
    config = target_config()
    weights = layer_weights(1, component="ple.")
    with torch.device("meta"):
        reference = Qwen4ExpTextPLELayer(config, 1, 0)
    reference.ple_embedding = PreparedEmbedding()
    reference.load_state_dict({key.removeprefix("ple."): value for key, value in weights.items()}, assign=True)
    reference.eval()
    hidden = torch.empty(1, sequence + 1, 10240, dtype=torch.bfloat16).normal_(
        mean=0, std=0.02, generator=torch.Generator().manual_seed(451 + sequence)
    )
    embeddings = embedding_fixture(sequence + 1, seed=347 + sequence)
    cache = DynamicCache(config=config)
    with torch.inference_mode():
        reference.ple_embedding.current = embeddings[:, :sequence]
        expected_prefill = reference(hidden[:, :sequence], None, cache)
        reference.ple_embedding.current = embeddings[:, sequence:]
        expected_decode = reference(hidden[:, sequence:], None, cache)
    module = PLE(WeightLoader(weights, single_chip_mesh), config)
    conv = device(torch.zeros(1, 1, 9, 10240, dtype=torch.bfloat16), single_chip_mesh)
    prefill_x = device(hidden[:, :sequence].unsqueeze(0), single_chip_mesh)
    prefill_e = device(embeddings[:, :sequence].unsqueeze(0), single_chip_mesh)
    decode_x = device(hidden[:, sequence:].unsqueeze(0), single_chip_mesh)
    decode_e = device(embeddings[:, sequence:].unsqueeze(0), single_chip_mesh)
    output = module(prefill_x, prefill_e, conv_state=conv)
    record_property(
        "ple_arithmetic_prefill_pcc",
        assert_pcc(expected_prefill, ttnn.to_torch(output), label="PLE arithmetic prefill"),
    )
    ttnn.deallocate(output)
    saved = snapshot({"conv": conv})
    warm = module(decode_x, decode_e, conv_state=conv)
    ttnn.synchronize_device(single_chip_mesh)
    ttnn.deallocate(warm)
    restore({"conv": conv}, saved)
    trace = ttnn.begin_trace_capture(single_chip_mesh, cq_id=0)
    output = module(decode_x, decode_e, conv_state=conv)
    ttnn.end_trace_capture(single_chip_mesh, trace, cq_id=0)
    trace_allocation_tracker.acknowledge_corruptible(output)
    try:
        restore({"conv": conv}, saved)
        ttnn.execute_trace(single_chip_mesh, trace, cq_id=0, blocking=True)
        record_property(
            "ple_arithmetic_decode_pcc",
            assert_pcc(expected_decode, ttnn.to_torch(output), label="PLE arithmetic decode"),
        )
    finally:
        ttnn.release_trace(single_chip_mesh, trace)
    record_property("scope", "PLE arithmetic only; prepared synthetic embedding features; no hash/full table claim")
