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


def test_traced_sampled_decode_uses_device_feedback_after_reset(monkeypatch):
    class Model:
        vocab_size = 128
        num_devices = 1
        sampling = object()

        def increment_positions(self, current_pos, rot_mat_idxs):
            pass

    engine = executor_module.TracedLLMExecutor.__new__(executor_module.TracedLLMExecutor)
    engine.model = Model()
    engine.mesh_device = "mesh"
    engine.mode = None
    engine.device_decode_feedback_enabled = True
    engine.trace_ids_decode = {True: 7}
    engine.trace_inputs_decode = {True: ("device_tokens", "device_pos", "device_rot", "device_page_table")}
    engine.trace_output_decode = {True: ("device_toks", None)}
    engine._prev_decode_page_table = None

    class Eager:
        def _assert_kv_cache_identity(self, kv_cache):
            pass

        def prepare_decode_inputs_host(self, tokens, start_pos, page_table):
            return ("host_tokens", "host_pos", "host_rot", "host_page_table")

    engine._eager = Eager()

    copy_calls = []
    monkeypatch.setattr(
        executor_module,
        "copy_host_to_device",
        lambda host_tensors, device_tensors: copy_calls.append((host_tensors, device_tensors)),
    )

    page_table_copies = []
    monkeypatch.setattr(
        executor_module.ttnn,
        "copy_host_to_device_tensor",
        lambda host_tensor, device_tensor: page_table_copies.append((host_tensor, device_tensor)),
    )
    monkeypatch.setattr(executor_module.ttnn, "execute_trace", lambda *args, **kwargs: None)

    tokens = executor_module.torch.tensor([11])
    start_pos = executor_module.torch.tensor([5])
    page_table = executor_module.torch.tensor([[0]], dtype=executor_module.torch.int32)

    engine.decode_forward(
        tokens,
        start_pos,
        page_table=page_table,
        sampling_params=object(),
        read_from_device=False,
        reset_batch=True,
    )
    engine.decode_forward(
        tokens,
        start_pos + 1,
        page_table=page_table,
        sampling_params=object(),
        read_from_device=False,
    )
    new_page_table = executor_module.torch.tensor([[1]], dtype=executor_module.torch.int32)
    engine.decode_forward(
        tokens,
        start_pos + 2,
        page_table=new_page_table,
        sampling_params=object(),
        read_from_device=False,
    )

    assert copy_calls == [
        (
            ("host_tokens", "host_pos", "host_rot", "host_page_table"),
            ("device_tokens", "device_pos", "device_rot", "device_page_table"),
        )
    ]
    assert page_table_copies == [("host_page_table", "device_page_table")]
