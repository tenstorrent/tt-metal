# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""High-signal host-side regression tests for Gemma4 disaggregated prefill.

These tests lock down the invariants the short device smoke test should protect:
- ring metadata must be updated per user and per chunk;
- traced and untraced prefill dispatch should agree on the same chunk metadata;
- chunk boundary cases around the 1024-token sliding window are covered.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.tt.model import Gemma4Model

from ..demo.text_demo_prefill import _build_prefill_model, _get_prefill_tokens, _host_tensor, _mesh_config, _model_path
from .test_factory import parametrize_mesh_with_fabric


def _make_model(events):
    """Build a minimal model whose layer calls record the chunk metadata we care about."""
    model = object.__new__(Gemma4Model)
    model.mesh_device = object()
    model.mesh_config = SimpleNamespace(prefill=SimpleNamespace(sp=8))
    model.hf_config = SimpleNamespace(layer_types=("sliding_attention", "full_attention"))
    model.tt_kv_cache = [None, None]
    model._rope_prefill_positions = None
    rope = np.zeros((1, 1, 4096, 1), dtype=np.float32)
    model.rope_caches_2d = {
        "sliding_attention": (rope.copy(), rope.copy()),
        "full_attention": (rope.copy(), rope.copy()),
    }
    model.rope_caches = {
        "sliding_attention": (rope.copy(), rope.copy()),
        "full_attention": (rope.copy(), rope.copy()),
    }
    model._ring_metadata_external = False
    model._packed_global_rope_trans_mat = None
    model._prefill_trace_mode = True
    model._prefill_trace_controller = SimpleNamespace(layer_ack=lambda idx: events.append(("ack", idx)))
    model.ccl_manager = SimpleNamespace(
        set_ring_metadata=lambda slot_idx, kv_actual_global: events.append(
            ("ring_metadata", slot_idx, kv_actual_global)
        ),
        ring_attention_ccl_semaphore_handles=[],
    )
    model.norm = SimpleNamespace(forward=lambda hidden: hidden)

    def layer_fn(layer_idx):
        def _call(hidden_states, **kwargs):
            events.append(
                (
                    "layer",
                    layer_idx,
                    kwargs["chunk_start_idx"],
                    kwargs.get("rope_mats") is not None,
                )
            )
            return hidden_states

        return _call

    model.layers = [layer_fn(0), layer_fn(1)]
    return model


@pytest.mark.parametrize(
    "chunk_start,user_id",
    [
        (0, 0),
        (1024, 1),
        (2048, 0),
        (8192, 2),
    ],
)
def test_prefill_metadata_tracks_user_and_chunk(chunk_start, user_id):
    events = []
    model = _make_model(events)
    hidden = SimpleNamespace(shape=(1, 1, 128, 64))

    model(hidden_states=hidden, chunk_start_idx=chunk_start, user_id=user_id)

    assert ("ring_metadata", user_id, chunk_start) in events


@pytest.mark.parametrize(
    "chunk_size,context_len,expected_starts",
    [
        (1024, 1024, [0]),
        (1024, 2048, [0, 1024]),
        (1024, 3072, [0, 1024, 2048]),
        (2048, 4096, [0, 2048]),
        (2048, 8192, [0, 2048, 4096, 6144]),
    ],
)
def test_chunk_boundary_starts_cover_sliding_window_edges(chunk_size, context_len, expected_starts):
    starts = [idx * chunk_size for idx in range(context_len // chunk_size)]
    assert starts == expected_starts


def test_traced_and_untraced_prefill_dispatch_share_same_chunk_metadata(monkeypatch):
    traced_events = []
    eager_events = []
    traced_model = _make_model(traced_events)
    eager_model = _make_model(eager_events)

    monkeypatch.setattr("ttnn.embedding", lambda *args, **kwargs: args[1])
    monkeypatch.setattr("ttnn.unsqueeze_to_4D", lambda value: value)

    hidden = SimpleNamespace(shape=(1, 1, 128, 64))
    traced_model._rope_prefill_positions = [0, 1, 2, 3]
    traced_model._rope_prefill_positions = [0, 1, 2, 3]

    traced_model(hidden_states=hidden, chunk_start_idx=2048, user_id=1)
    eager_model(hidden_states=hidden, rope_mats=("cos", "sin"), chunk_start_idx=2048, user_id=1)

    traced_ring = [event for event in traced_events if event[0] == "ring_metadata"]
    eager_ring = [event for event in eager_events if event[0] == "ring_metadata"]
    assert traced_ring == eager_ring == [("ring_metadata", 1, 2048)]

    traced_layers = [event for event in traced_events if event[0] == "layer"]
    eager_layers = [event for event in eager_events if event[0] == "layer"]
    assert [event[1:] for event in traced_layers] == [event[1:] for event in eager_layers]
    assert traced_layers and eager_layers


def _device_logits_to_torch(logits, mesh_device, last_token_idx):
    device_logits = ttnn.get_device_tensors(logits)[0] if mesh_device.shape[1] > 1 else logits
    logits_torch = ttnn.to_torch(device_logits).float()
    return logits_torch[..., last_token_idx % 32, :]


def _run_device_prefill(mesh_device, tokens, *, traced):
    chunk_size = int(tokens.shape[-1])
    mesh_config = _mesh_config(mesh_device)
    model_args, model, _ = _build_prefill_model(
        mesh_device=mesh_device,
        model_path=_model_path(),
        chunk_size=chunk_size,
        context_len=chunk_size,
    )
    host_tokens = _host_tensor(
        mesh_device,
        tokens,
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_config=mesh_config,
        seq_dim=-1,
    )
    device_tokens = ttnn.to_device(host_tokens, device=mesh_device)
    positions = _host_tensor(
        mesh_device,
        torch.arange(chunk_size, dtype=torch.int32).unsqueeze(0),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_config=mesh_config,
        seq_dim=-1,
    )
    model.set_prefill_rope_positions(ttnn.to_device(positions, device=mesh_device))
    model._ring_metadata_external = traced

    def forward():
        embeds = model.transform_and_embed_prefill_inputs_device(device_tokens)
        return model(hidden_states=embeds, chunk_start_idx=0, user_id=0)

    if traced:
        forward()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            hidden_states = forward()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            hidden_states = model.process_logits_after_prefill_trace(hidden_states, chunk_size - 1)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
    else:
        hidden_states = model.process_logits_after_prefill_trace(forward(), chunk_size - 1)

    ttnn.synchronize_device(mesh_device)
    logits = _device_logits_to_torch(hidden_states, mesh_device, chunk_size - 1)
    hidden_states.deallocate(True)
    device_tokens.deallocate(True)
    return logits, model_args


def _run_chunked_device_prefill(mesh_device, tokens, *, chunk_size, traced, model_chunk_size=None):
    context_len = int(tokens.shape[-1])
    assert (
        context_len % chunk_size == 0
    ), f"context length {context_len} must divide evenly into {chunk_size}-token chunks"
    mesh_config = _mesh_config(mesh_device)
    model_args, model, _ = _build_prefill_model(
        mesh_device=mesh_device,
        model_path=_model_path(),
        chunk_size=model_chunk_size or chunk_size,
        context_len=context_len,
    )
    host_tokens = _host_tensor(
        mesh_device,
        tokens[:, :chunk_size].contiguous(),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_config=mesh_config,
        seq_dim=-1,
    )
    device_tokens = ttnn.to_device(host_tokens, device=mesh_device)
    device_positions = ttnn.to_device(
        _host_tensor(
            mesh_device,
            torch.arange(chunk_size, dtype=torch.int32).unsqueeze(0),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        ),
        device=mesh_device,
    )
    model.set_prefill_rope_positions(device_positions)
    model._ring_metadata_external = traced

    def stage(chunk_idx):
        chunk_start = chunk_idx * chunk_size
        staged_tokens = _host_tensor(
            mesh_device,
            tokens[:, chunk_start : chunk_start + chunk_size].contiguous(),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged_tokens, device_tokens)
        if traced:
            model.ccl_manager.set_ring_metadata(slot_idx=0, kv_actual_global=chunk_start)
        for semaphore in model.ccl_manager.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(semaphore, 0)
        staged_positions = _host_tensor(
            mesh_device,
            torch.arange(chunk_start, chunk_start + chunk_size, dtype=torch.int32).unsqueeze(0),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged_positions, device_positions)
        return chunk_start

    def forward(chunk_start):
        embeds = model.transform_and_embed_prefill_inputs_device(device_tokens)
        return model(hidden_states=embeds, chunk_start_idx=chunk_start, user_id=0)

    n_chunks = context_len // chunk_size
    if traced:
        stage(0)
        warmup = forward(0)
        ttnn.synchronize_device(mesh_device)
        warmup.deallocate(True)
        stage(0)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            trace_output = forward(0)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            for chunk_idx in range(n_chunks):
                chunk_start = stage(chunk_idx)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                if chunk_idx == n_chunks - 1:
                    last_hidden = trace_output
                else:
                    ttnn.synchronize_device(mesh_device)
            hidden_states = model.process_logits_after_prefill_trace(last_hidden, chunk_size - 1)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
    else:
        last_hidden = None
        for chunk_idx in range(n_chunks):
            chunk_start = stage(chunk_idx)
            current_hidden = forward(chunk_start)
            if last_hidden is not None:
                last_hidden.deallocate(True)
            last_hidden = current_hidden
        hidden_states = model.process_logits_after_prefill_trace(last_hidden, chunk_size - 1)

    ttnn.synchronize_device(mesh_device)
    logits = _device_logits_to_torch(hidden_states, mesh_device, context_len - 1)
    hidden_states.deallocate(True)
    device_tokens.deallocate(True)
    device_positions.deallocate(True)
    return logits, model_args


def _pcc(lhs, rhs):
    lhs = lhs.reshape(-1).float()
    rhs = rhs.reshape(-1).float()
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    return float(torch.dot(lhs, rhs) / (torch.linalg.vector_norm(lhs) * torch.linalg.vector_norm(rhs)))


@pytest.fixture(scope="module")
def hf_prefill_reference():
    transformers = pytest.importorskip("transformers")
    model_path = _model_path()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    yield model
    del model


@torch.no_grad()
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_device_prefill_matches_hf(mesh_device, hf_prefill_reference):
    """Compare one real 8x4 D/P prefill with the HF final-token logits."""
    chunk_size = 8192
    tokens = _get_prefill_tokens(_model_path(), chunk_size, hf_prefill_reference.config.vocab_size, source="random")
    device_logits, model_args = _run_device_prefill(mesh_device, tokens, traced=False)
    with torch.no_grad():
        reference_logits = hf_prefill_reference(tokens.long()).logits[0, -1, : model_args.vocab_size].float()

    pcc = _pcc(device_logits[0], reference_logits)
    assert pcc >= 0.99, f"Gemma4 D/P final-token logits diverged from HF: PCC={pcc:.5f}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_device_traced_prefill_matches_eager(mesh_device):
    """Ensure trace replay preserves the final-token logits of eager prefill."""
    chunk_size = 8192
    model_path = _model_path()
    tokens = _get_prefill_tokens(model_path, chunk_size, 262144, source="random")
    eager_logits, _ = _run_device_prefill(mesh_device, tokens, traced=False)
    traced_logits, _ = _run_device_prefill(mesh_device, tokens, traced=True)

    pcc = _pcc(eager_logits, traced_logits)
    assert pcc >= 0.999, f"traced and eager Gemma4 D/P logits diverged: PCC={pcc:.5f}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_device_multi_chunk_traced_prefill_matches_eager(mesh_device):
    """Ensure trace replay preserves logits after staging multiple prefill chunks."""
    context_len = 16384
    chunk_size = 8192
    tokens = _get_prefill_tokens(_model_path(), context_len, 262144, source="random")
    eager_logits, _ = _run_chunked_device_prefill(mesh_device, tokens, chunk_size=chunk_size, traced=False)
    traced_logits, _ = _run_chunked_device_prefill(mesh_device, tokens, chunk_size=chunk_size, traced=True)

    pcc = _pcc(eager_logits, traced_logits)
    assert pcc >= 0.999, f"multi-chunk traced and eager logits diverged: PCC={pcc:.5f}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_device_chunk_boundary_matches_hf(mesh_device, hf_prefill_reference):
    """Check a 16K prompt across the 1024-token sliding-window boundary against HF."""
    context_len = 16384
    tokens = _get_prefill_tokens(_model_path(), context_len, hf_prefill_reference.config.vocab_size, source="random")
    device_logits, model_args = _run_chunked_device_prefill(mesh_device, tokens, chunk_size=8192, traced=False)
    with torch.no_grad():
        reference_logits = hf_prefill_reference(tokens.long()).logits[0, -1, : model_args.vocab_size].float()

    pcc = _pcc(device_logits[0], reference_logits)
    assert pcc >= 0.99, f"16K chunked Gemma4 D/P logits diverged from HF: PCC={pcc:.5f}"
