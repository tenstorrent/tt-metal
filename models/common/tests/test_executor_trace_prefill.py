# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.common.models import executor as executor_module


def test_easy_trace_prefill_replay_copies_only_mutable_inputs(monkeypatch):
    engine = executor_module.TracedLLMExecutor.__new__(executor_module.TracedLLMExecutor)
    engine.mesh_device = "mesh"
    engine.trace_id_prefill = {128: 7}
    engine.trace_inputs_prefill = {
        128: ("device_tokens", "device_cos", "device_sin", "device_page_table", None),
    }
    engine.trace_output_prefill = {128: "trace_output"}

    monkeypatch.setattr(
        engine,
        "_prepare_prefill_trace_inputs_host",
        lambda tokens, page_table, last_token_idx: (
            "host_tokens",
            "device_cos_slice",
            "device_sin_slice",
            "host_page_table",
            None,
        ),
    )

    copy_calls = []
    monkeypatch.setattr(
        executor_module,
        "copy_host_to_device",
        lambda host_tensors, device_tensors: copy_calls.append((host_tensors, device_tensors)),
    )

    execute_calls = []
    monkeypatch.setattr(
        executor_module.ttnn,
        "execute_trace",
        lambda mesh_device, trace_id, cq_id, blocking: execute_calls.append((mesh_device, trace_id, cq_id, blocking)),
    )

    result = engine._easy_trace_prefill(
        tokens="tokens",
        page_table="page_table",
        user_id=0,
        last_token_idx=127,
        prefill_seq_len=128,
    )

    assert result == "trace_output"
    assert copy_calls == [
        (
            ("host_tokens", "host_page_table", None),
            ("device_tokens", "device_page_table", None),
        )
    ]
    assert execute_calls == [("mesh", 7, 0, False)]
