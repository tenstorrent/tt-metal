# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

import ttnn
from models.common.sampling import SamplingParams
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from ttnn.tools import trace_allocation_tracker


@pytest.fixture
def fresh_prefetcher_cache():
    # The legacy address-table cache is process-global, whereas parametrized
    # mesh fixtures create a new device object for each case. Never reuse the
    # previous case's address table on the new mesh.
    from models.demos.llama3_70b_galaxy.tt import prefetcher_common

    prefetcher_common.global_tt_tensor_address = None
    yield
    prefetcher_common.global_tt_tensor_address = None
    gc.collect()


def _tensors(value):
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors(item)]
    return [] if value is None else [value]


def _host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def _release(generator):
    generator.model.switch_mode("prefill")
    for trace_id in generator.trace_id_prefill.values():
        if trace_id is not None:
            ttnn.release_trace(generator.mesh_device, trace_id)
    generator.trace_id_prefill.clear()
    generator.model.switch_mode("decode")
    generator.model.sampling.reset_trace()
    for trace_id in generator.trace_ids_decode.values():
        if trace_id is not None:
            ttnn.release_trace(generator.mesh_device, trace_id)
    generator.trace_ids_decode.clear()
    gc.collect()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "worker_l1_size": 1345000,
            "trace_region_size": 64000000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("first_capture", ["prefill", "decode"])
def test_prepared_galaxy_trace_io(
    mesh_device, reset_seeds, monkeypatch, tmp_path, first_capture, expect_error, fresh_prefetcher_cache
):
    if "wormhole" not in str(mesh_device.arch()).lower():
        pytest.skip("Exercises Wormhole Galaxy's prefetcher and sub-device managers")
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    monkeypatch.setenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    args, model, page_table, kv_cache = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=32,
        optimizations=LlamaOptimizations.accuracy,
        max_seq_len=2048,
        num_layers=1,
        dummy_weights=True,
        page_params={"page_block_size": 64, "page_max_num_blocks": 1024},
        use_paged_kv_cache=True,
    )
    generator = Generator(model, args, mesh_device)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    tokens = torch.ones((32, 1), dtype=torch.int32)
    positions = torch.full((32,), -1, dtype=torch.int32)
    positions[0] = 97
    prompts = [torch.ones((1, length), dtype=torch.int32) for length in (97, 777)]

    def prefill(prompt, trace=True):
        return generator.prefill_forward_text(
            prompt,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([prompt.shape[1]]),
            enable_trace=trace,
        )

    def decode(mode, layout, **kwargs):
        return generator.decode_forward(
            tokens,
            positions,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=params if mode else None,
            is_cur_pos_sharded=layout[0],
            is_page_table_sharded=layout[1],
            **kwargs,
        )

    try:
        generator._prepare_decode_before_prefill(page_table, kv_cache, True)
        assert len(generator._prepared_decode_traces) == 8
        prepared_decode = generator._prepared_decode_traces
        assert len({state["device_inputs"][0].buffer_unique_id() for state in prepared_decode.values()}) == 1
        for mode in (False, True):
            assert (
                len({state["output"][0].buffer_unique_id() for key, state in prepared_decode.items() if key[0] == mode})
                == 1
            )
        generator.already_warmed_up_prefill = True
        expected_prefill = [prefill(prompt, trace=False) for prompt in prompts]
        generator.warming_up_prefill = True
        generator._defer_trace_recording = True
        for prompt in prompts:
            prefill(prompt)
        generator.warming_up_prefill = False
        generator._defer_trace_recording = False
        short_inputs = generator._pending_prefill_traces["128_1_sp0"]["device_inputs"]
        long_inputs = generator._pending_prefill_traces["1024_1_sp0"]["device_inputs"]
        for index in (1, 2, 4, 5):
            assert short_inputs[index].buffer_unique_id() == long_inputs[index].buffer_unique_id()

        # Eager sampling omits the optional feedback output and therefore has
        # its own program-cache entry. Build the independent references now.
        expected_decode = {}
        prefill(prompts[0], trace=False)
        for layout in ((False, False), (True, False), (False, True), (True, True)):
            for mode in (False, True):
                expected_decode[(mode, layout)] = decode(
                    mode, layout, enable_trace=False, reset_inputs=True, reset_batch=True
                )[0][0].clone()
        model.switch_mode("prefill")

        persistent = [entry[-1] for entry in generator._prepared_trace_io._buffers]
        assert all(t.memory_config().buffer_type == ttnn.BufferType.DRAM for t in persistent)
        addresses = [t.buffer_address() for t in persistent]
        prepared_prefill = next(iter(generator._pending_prefill_traces.values()))
        prefill_output = prepared_prefill["compile_out"]
        retained = ttnn.empty_like(prefill_output)
        ttnn.copy(
            prefill_output, retained, sub_core_grids=args.sub_core_grids
        )  # Warm the retention copy before capture.
        layout_before = model.global_cb_trace_state._layout

        if first_capture == "decode":
            decode(True, (True, True), reset_inputs=True, reset_batch=True)
            assert not any(generator.trace_id_prefill.values())
        else:
            prefill(prompts[0])
            assert not any(generator.trace_ids_decode.values())

        saved_ids = None
        cache_entries = None
        for repetition in range(2):
            for prompt, expected in zip(prompts, expected_prefill):
                torch.testing.assert_close(prefill(prompt), expected, rtol=0, atol=0)
            prefill(prompts[0])
            ttnn.copy(prefill_output, retained, sub_core_grids=args.sub_core_grids)
            queued = prefill_output.cpu(blocking=False)
            event = ttnn.record_event(mesh_device, 0)
            before_decode = _host(prefill_output)
            for layout in ((False, False), (True, False), (False, True), (True, True)):
                for mode in (False, True):
                    result = decode(mode, layout, reset_inputs=True, reset_batch=True)[0][0]
                    assert torch.isfinite(result).all()
                    key = (mode, layout)
                    torch.testing.assert_close(result, expected_decode[key], rtol=0, atol=0)
            ttnn.event_synchronize(event)
            torch.testing.assert_close(_host(queued), before_decode, rtol=0, atol=0)
            torch.testing.assert_close(_host(retained), before_decode, rtol=0, atol=0)
            torch.testing.assert_close(_host(prefill_output), before_decode, rtol=0, atol=0)
            assert [t.buffer_address() for t in persistent] == addresses
            assert model.global_cb_trace_state._layout == layout_before
            ids = (dict(generator.trace_id_prefill), dict(generator.trace_ids_decode), tuple(model.sampling.trace_ids))
            if saved_ids is not None:
                assert ids == saved_ids
            saved_ids = ids
            if cache_entries is not None:
                assert mesh_device.num_program_cache_entries() == cache_entries
            cache_entries = mesh_device.num_program_cache_entries()

        # Device-resident sharded positions must advance through DRAM backing,
        # even when the host tokens/positions are deliberately left stale.
        decode(True, (True, True), reset_inputs=True, reset_batch=True)
        inputs = generator.trace_inputs_decode[generator._active_decode_key]
        before = _host(inputs[1])
        decode(True, (True, True), reset_inputs=False)
        after = _host(inputs[1])
        torch.testing.assert_close(after, torch.where(before >= 0, before + 1, before), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="preparation is closed"):  # allow-pytest.raises: Python-only guard
            generator._prepare_trace_decode(tokens, positions, page_table, kv_cache[0])

        # A model bug must remain visible after removing the broad capture
        # exemption: deliberately retain a capture-local result and reject replay.
        model.switch_mode("prefill")
        survivors = []
        forward = model.ttnn_prefill_forward

        def leak(*args, **kwargs):
            output = forward(*args, **kwargs)
            survivors.append(output)
            return output

        monkeypatch.setattr(model, "ttnn_prefill_forward", leak)
        newer_id = None
        try:
            newer_id, _ = generator._record_trace_prefill(prepared_prefill)
            older_id = generator.trace_id_prefill["128_1_sp0"]
            unsafe = trace_allocation_tracker.get_unsafe_tracked_ids(mesh_device, older_id)
            assert any(t.buffer_unique_id() in unsafe for t in _tensors(survivors))
            with expect_error(RuntimeError, "still alive before trace replay"):
                ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
        finally:
            if newer_id is not None:
                ttnn.release_trace(mesh_device, newer_id)
            survivors.clear()
    finally:
        generator.warming_up_prefill = False
        generator._defer_trace_recording = False
        _release(generator)
