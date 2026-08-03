# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for mutable inputs bound into Gemma4 decode traces.

The trace is captured once over persistent device tensors.  Each replay then
overwrites every request-owned input at the same device address: hidden state,
RoPE cos/sin, current position, page table, and both physical cache tensors.
The A/B/A sequence catches stale captured addresses as well as accidental
dependence on cache contents left by the preceding replay.
"""

import json

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder import (
    ARTIFACT_DIR,
    _as_tt,
    _load_layer_state,
    _load_text_config,
    _to_torch,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MODEL_ID,
    SLIDING_BLOCK_SIZE,
    SLIDING_HEAD_DIM,
    SLIDING_NUM_KV_HEADS,
    FunctionalDecoder,
)
from models.common.utility_functions import comp_pcc

BATCH = 32
BLOCKS_PER_USER = 4
CONTROL_PCC = 0.9999
REPEAT_PCC = 0.9999


def _host_tt(mesh_device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Create a 1x1-mesh host tensor accepted by copy_host_to_device_tensor."""
    return ttnn.from_torch(
        tensor,
        device=None,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _copy_payload_to_stable(mesh_device, payload, stable):
    """Overwrite captured device buffers in place, outside trace capture."""
    specs = {
        "hidden_states": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "position_cos": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "position_sin": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "current_pos": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        "page_table": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        "key_cache": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "value_cache": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    }
    host_tensors = []
    for name, (dtype, layout) in specs.items():
        host = _host_tt(mesh_device, payload[name], dtype=dtype, layout=layout)
        host_tensors.append(host)
        target = stable["kv_cache"][0 if name == "key_cache" else 1] if name.endswith("_cache") else stable[name]
        ttnn.copy_host_to_device_tensor(host, target)
    return host_tensors


def _independently_permuted_page_table(*, variant: int) -> torch.Tensor:
    """Give every user a private block pool and its own deterministic order."""
    rows = []
    for user in range(BATCH):
        local = torch.arange(
            user * BLOCKS_PER_USER,
            (user + 1) * BLOCKS_PER_USER,
            dtype=torch.int32,
        )
        generator = torch.Generator().manual_seed(0x5047E + 97 * user)
        independently_shuffled = local[torch.randperm(BLOCKS_PER_USER, generator=generator)]
        rows.append(torch.roll(independently_shuffled, shifts=variant))
    return torch.stack(rows)


def _payload(cfg, layer_type: str, *, variant: int):
    """Build distinct decode inputs and nonzero shared-HMA physical history."""
    seed = 0xA4B0 + 101 * variant + (0 if layer_type == "sliding_attention" else 17)
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(BATCH, 1, HIDDEN_SIZE, dtype=torch.bfloat16, generator=generator)
    if variant == 0:
        positions = 32 + torch.arange(BATCH, dtype=torch.long)
    else:
        positions = 95 - torch.arange(BATCH, dtype=torch.long)

    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions.view(BATCH, 1), layer_type=layer_type)
    if layer_type == "sliding_attention":
        tt_cos = cos.unsqueeze(0)
        tt_sin = sin.unsqueeze(0)
    else:
        tt_cos = cos.transpose(0, 1).unsqueeze(0)
        tt_sin = sin.transpose(0, 1).unsqueeze(0)

    # This is the common physical HMA cache allocation.  Full attention
    # reinterprets it as [blocks, 2, 128, 512] through decoder view overrides.
    cache_shape = (
        BATCH * BLOCKS_PER_USER,
        SLIDING_NUM_KV_HEADS,
        SLIDING_BLOCK_SIZE,
        SLIDING_HEAD_DIM,
    )
    key_cache = torch.randn(cache_shape, dtype=torch.bfloat16, generator=generator).mul_(0.125)
    value_cache = torch.randn(cache_shape, dtype=torch.bfloat16, generator=generator).mul_(0.125)
    # Make physical pages and K/V roles visibly distinguishable without
    # overwhelming the random history used by attention.
    block_ids = torch.arange(cache_shape[0], dtype=torch.float32).view(-1, 1, 1, 1)
    key_cache.add_((block_ids * 0.0005).to(torch.bfloat16))
    value_cache.sub_((block_ids * 0.0005 + 0.03125).to(torch.bfloat16))
    key_nonzero_fraction = int(torch.count_nonzero(key_cache)) / key_cache.numel()
    value_nonzero_fraction = int(torch.count_nonzero(value_cache)) / value_cache.numel()
    assert key_nonzero_fraction > 0.99, f"sparse key-cache seed: {key_nonzero_fraction:.6f} nonzero"
    assert value_nonzero_fraction > 0.99, f"sparse value-cache seed: {value_nonzero_fraction:.6f} nonzero"
    assert not torch.equal(key_cache, value_cache), "K/V cache histories must be distinguishable"

    return {
        "hidden_states": hidden.transpose(0, 1).unsqueeze(0),
        "position_cos": tt_cos,
        "position_sin": tt_sin,
        "current_pos": positions.to(torch.int32),
        "page_table": _independently_permuted_page_table(variant=variant),
        "key_cache": key_cache,
        "value_cache": value_cache,
    }


def _device_args(mesh_device, payload):
    return {
        "hidden_states": _as_tt(mesh_device, payload["hidden_states"]),
        "position_cos": _as_tt(mesh_device, payload["position_cos"]),
        "position_sin": _as_tt(mesh_device, payload["position_sin"]),
        "current_pos": _as_tt(
            mesh_device,
            payload["current_pos"],
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": _as_tt(
            mesh_device,
            payload["page_table"],
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "kv_cache": (
            _as_tt(mesh_device, payload["key_cache"]),
            _as_tt(mesh_device, payload["value_cache"]),
        ),
    }


def _read_decode_output(mesh_device, output):
    return _to_torch(mesh_device, output).reshape(1, BATCH, HIDDEN_SIZE).transpose(0, 1).to(torch.bfloat16)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention_shared_hma"])
def test_trace_mutable_stable_buffers(mesh_device, device_params, layer_idx):
    """Capture once, overwrite all stable inputs, and replay payloads A/B/A."""
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    payloads = [_payload(cfg, layer_type, variant=variant) for variant in range(2)]
    assert all(
        not torch.equal(a, b) for a, b in zip(payloads[0]["page_table"], payloads[1]["page_table"])
    ), "every page-table row must change between A and B"

    # Fresh eager executions are the control for the trace mechanism.  The
    # broader real-weight tests independently establish this decoder's HF PCC.
    controls = []
    for payload in payloads:
        control_args = _device_args(mesh_device, payload)
        controls.append(_read_decode_output(mesh_device, decoder.decode_forward(**control_args)))

    stable = _device_args(mesh_device, payloads[0])
    # Exact-shape warm compile before capture.
    decoder.decode_forward(**stable)
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**stable)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    replay_outputs = []
    try:
        for variant in (0, 1, 0):
            # Keep host staging tensors alive until the blocking replay has
            # consumed the queued copies on command queue zero.
            host_copies = _copy_payload_to_stable(mesh_device, payloads[variant], stable)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replay_outputs.append(_read_decode_output(mesh_device, traced_output))
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    control_results = [
        comp_pcc(controls[variant], replay_outputs[index], CONTROL_PCC) for index, variant in enumerate((0, 1, 0))
    ]
    repeat_ok, repeat_pcc = comp_pcc(replay_outputs[0], replay_outputs[2], REPEAT_PCC)
    ab_max_abs_diff = float((replay_outputs[0].float() - replay_outputs[1].float()).abs().max())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"trace_mutable_buffers_{layer_type}_batch{BATCH}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "batch": BATCH,
                "physical_cache_view": [
                    BATCH * BLOCKS_PER_USER,
                    SLIDING_NUM_KV_HEADS,
                    SLIDING_BLOCK_SIZE,
                    SLIDING_HEAD_DIM,
                ],
                "payload_order": ["A", "B", "A"],
                "current_positions_a": payloads[0]["current_pos"].tolist(),
                "current_positions_b": payloads[1]["current_pos"].tolist(),
                "page_table_a": payloads[0]["page_table"].tolist(),
                "page_table_b": payloads[1]["page_table"].tolist(),
                "eager_control_vs_replay_pcc": [float(result[1]) for result in control_results],
                "eager_control_vs_replay_threshold": CONTROL_PCC,
                "a_repeat_pcc": float(repeat_pcc),
                "a_repeat_threshold": REPEAT_PCC,
                "a_vs_b_max_abs_diff": ab_max_abs_diff,
                "stable_buffers_overwritten": [
                    "hidden_states",
                    "position_cos",
                    "position_sin",
                    "current_pos",
                    "page_table",
                    "key_cache",
                    "value_cache",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    for index, (ok, pcc) in enumerate(control_results):
        assert ok, f"payload {(0, 1, 0)[index]} eager-control PCC {pcc}"
    assert ab_max_abs_diff > 1.0e-3, "A and B replay outputs were unexpectedly identical"
    assert repeat_ok, repeat_pcc
