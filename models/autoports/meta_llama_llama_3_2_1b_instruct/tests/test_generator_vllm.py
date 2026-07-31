# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.generator_vllm import Llama32ForCausalLM


MODEL_DIR = Path("models/autoports/meta_llama_llama_3_2_1b_instruct")
ARTIFACT_DIR = MODEL_DIR / "readiness_vllm"


@contextmanager
def _open_t3k_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8))
    try:
        yield mesh_device
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in after}


def _build_adapter(mesh_device) -> Llama32ForCausalLM:
    return Llama32ForCausalLM(
        model_dir=MODEL_DIR,
        mesh_device=mesh_device,
        max_seq_len=128,
        max_batch_size=32,
        page_block_size=64,
        num_layers=1,
        cache_path=MODEL_DIR / ".ttnn_cache",
        use_vllm_paged_kv_cache=True,
    )


def test_vllm_adapter_trace_input_refresh_contract():
    with _open_t3k_mesh() as mesh_device:
        adapter = _build_adapter(mesh_device)
        kv_cache = adapter.allocate_kv_cache((2, 1, 64, 64), torch.bfloat16, adapter.model.n_layers)
        sampling_params = adapter._make_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        page_a = torch.tensor([[0, 1]], dtype=torch.int32)
        page_b = torch.tensor([[1, 0]], dtype=torch.int32)
        try:
            adapter.decode_forward(
                torch.tensor([[11]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                page_table=page_a,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=sampling_params,
                read_from_device=False,
                reset_batch=True,
            )
            ttnn.synchronize_device(mesh_device)
            after_capture = dict(adapter.trace_audit()["counters"])

            adapter.decode_forward(
                torch.tensor([[12345]], dtype=torch.long),
                torch.tensor([17], dtype=torch.long),
                page_table=page_a.clone(),
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=sampling_params,
                read_from_device=False,
                reset_batch=False,
            )
            ttnn.synchronize_device(mesh_device)
            after_steady = dict(adapter.trace_audit()["counters"])
            steady_delta = _counter_delta(after_steady, after_capture)

            adapter.decode_forward(
                torch.tensor([[6789]], dtype=torch.long),
                torch.tensor([18], dtype=torch.long),
                page_table=page_b,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=sampling_params,
                read_from_device=False,
                reset_batch=False,
            )
            ttnn.synchronize_device(mesh_device)
            after_page_change = dict(adapter.trace_audit()["counters"])
            page_change_delta = _counter_delta(after_page_change, after_steady)

            adapter.decode_forward(
                torch.tensor([[42]], dtype=torch.long),
                torch.tensor([3], dtype=torch.long),
                page_table=page_b,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=sampling_params,
                read_from_device=False,
                reset_batch=True,
            )
            ttnn.synchronize_device(mesh_device)
            after_reset = dict(adapter.trace_audit()["counters"])
            reset_delta = _counter_delta(after_reset, after_page_change)
        finally:
            adapter.teardown()

    assert steady_delta["token_refreshes"] == 0
    assert steady_delta["current_position_refreshes"] == 0
    assert steady_delta["rope_index_refreshes"] == 0
    assert steady_delta["page_table_refreshes"] == 0
    assert steady_delta["model_trace_replays"] == 1
    assert steady_delta["device_token_feedback_steps"] == 1

    assert page_change_delta["token_refreshes"] == 0
    assert page_change_delta["current_position_refreshes"] == 0
    assert page_change_delta["rope_index_refreshes"] == 0
    assert page_change_delta["page_table_refreshes"] == 1
    assert page_change_delta["model_trace_replays"] == 1

    assert reset_delta["token_refreshes"] == 1
    assert reset_delta["current_position_refreshes"] == 1
    assert reset_delta["rope_index_refreshes"] == 1
    assert reset_delta["page_table_refreshes"] == 1
    assert reset_delta["model_trace_replays"] == 1

    _write_json(
        ARTIFACT_DIR / "adapter_trace_input_refresh_contract.json",
        {
            "supports_async_decode": Llama32ForCausalLM.model_capabilities["supports_async_decode"],
            "tt_async_decode_allows_overlap": Llama32ForCausalLM.model_capabilities[
                "tt_async_decode_allows_overlap"
            ],
            "decode_trace_enabled": True,
            "kv_cache_owner": "vLLM adapter allocate_kv_cache",
            "steady_unchanged_page_table_delta": steady_delta,
            "changed_page_table_delta": page_change_delta,
            "reset_batch_delta": reset_delta,
            "status": "passed",
        },
    )
