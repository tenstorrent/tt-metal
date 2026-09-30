# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc
import time
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from ttnn.tools import trace_allocation_tracker


@pytest.mark.timeout(900)
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
def test_global_cb_survives_trace_mode_switches(mesh_device, expect_error, reset_seeds, monkeypatch, tmp_path):
    """Keep both decode variants and a sampler trace resident across repeated prefill."""
    if "wormhole" not in str(mesh_device.arch()).lower():
        pytest.skip("Exercises the Wormhole Galaxy tensor prefetcher")

    # Dummy weights use the bundled architecture config; no checkpoint is required.
    monkeypatch.setenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))

    args, model, page_table, kv_cache = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=32,
        optimizations=LlamaOptimizations.accuracy,
        max_seq_len=512,
        num_layers=1,
        dummy_weights=True,
        page_params={"page_block_size": 32, "page_max_num_blocks": 512},
        use_paged_kv_cache=True,
    )
    assert model.use_prefetcher
    generator = Generator(model, args, mesh_device)
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    prompt = torch.ones((1, 128), dtype=torch.int32)
    tokens = torch.ones((32, 1), dtype=torch.int32)
    positions = torch.full((32,), -1, dtype=torch.int32)
    positions[0] = 32

    try:
        # Compile both decode layouts and the sampling configurations before prefill
        # traces exist. The actual trace capture must still happen in decode mode.
        for on_device_logits in (False, True):
            generator._prepare_decode_before_prefill(page_table, kv_cache, on_device_logits)

        # Stage and compile the exact prefill shape used below before recording it.
        # This follows the generator's deferred-capture warmup sequence without
        # sweeping unrelated prompt lengths, batch sizes, and cached prefixes.
        generator.already_warmed_up_prefill = True
        # Eager prefill has a separate persistent input cache. Warm it before
        # any trace is captured, just as for the traced input layout below.
        generator.prefill_forward_text(
            prompt,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([32]),
            enable_trace=False,
        )
        generator.warming_up_prefill = True
        generator._defer_trace_recording = True
        try:
            generator.prefill_forward_text(
                prompt,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=torch.tensor([32]),
                enable_trace=True,
            )
            generator._defer_trace_recording = False
            generator._record_pending_traces()
        finally:
            generator.warming_up_prefill = False
            generator._defer_trace_recording = False
            generator._pending_prefill_traces.clear()

        expected_layout = None
        captured_ids = None
        expected_outputs = {}
        for trace_prefill in (True, False, True):
            generator.prefill_forward_text(
                prompt,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=torch.tensor([32]),
                enable_trace=trace_prefill,
            )
            for params in (None, sampling_params):
                output = generator.decode_forward(
                    tokens,
                    positions,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=True,
                    sampling_params=params,
                    reset_inputs=True,
                    reset_batch=True,
                )
                values = output[0] if isinstance(output, tuple) else output
                # Only slot 0 is active. Decode does not define logits for slots
                # whose current position is -1.
                active_output = values[0]
                assert torch.isfinite(active_output).all()
                variant = params is not None
                if variant not in expected_outputs:
                    expected_outputs[variant] = active_output.clone()
                else:
                    torch.testing.assert_close(active_output, expected_outputs[variant], rtol=0, atol=0)

            global_cb = model.prefetcher_setup.global_circular_buffer
            layout = (global_cb.buffer_address(), global_cb.config_address(), global_cb.size())
            logger.info(f"Reserved GCB: data={layout[0]:#x}, config={layout[1]:#x}, size={layout[2]}")
            del global_cb  # The model owns the reservation across prefill.
            ids = tuple(
                generator.trace_ids_decode[generator._decode_preparation_key(page_table, mode, False, False)]
                for mode in (False, True)
            ) + (model.sampling.trace_ids,)
            assert all(ids) and ids[2]
            if expected_layout is None:
                expected_layout, captured_ids = layout, ids
            else:
                assert layout == expected_layout
                assert ids == captured_ids  # Replayed, not silently recaptured.

            if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                # Ask the checker under the prefill manager while deliberately
                # retaining the GCB. Do not submit an unsafe replay to hardware.
                mesh_device.load_sub_device_manager(model.mesh_sub_device_manager_id_prefill)
                try:
                    for trace_id in generator.trace_id_prefill.values():
                        if trace_id is not None:
                            unsafe = ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh_device, trace_id)
                            assert len(unsafe) >= 2
                            with expect_error(RuntimeError, "still alive before trace replay"):
                                trace_allocation_tracker.TraceAllocationTracker.verify_before_replay(
                                    mesh_device, trace_id
                                )
                finally:
                    mesh_device.load_sub_device_manager(model.mesh_sub_device_manager_id_decode)

        def read_first_shard(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()

        live_inputs = [
            tensor for inputs in generator.trace_inputs_decode.values() for tensor in inputs if tensor is not None
        ]
        cache_tensors = [tensor for layer_cache in kv_cache[0] for tensor in layer_cache]
        retained_state = [read_first_shard(tensor) for tensor in live_inputs + cache_tensors]
        # Measure the reservation transition with an idle device; this excludes
        # inference, model construction, and the unrelated allocator interference.
        transition_ms = []
        for _ in range(10):
            ttnn.synchronize_device(mesh_device)
            started = time.perf_counter()
            model.switch_mode("prefill")
            model.switch_mode("decode")
            ttnn.synchronize_device(mesh_device)
            transition_ms.append(1000 * (time.perf_counter() - started))
        logger.info(
            f"Idle prefill/decode transitions (ms): min={min(transition_ms):.3f}, "
            f"median={statistics.median(transition_ms):.3f}, max={max(transition_ms):.3f}; "
            f"samples={transition_ms}"
        )
        for tensor, expected in zip(live_inputs + cache_tensors, retained_state):
            torch.testing.assert_close(read_first_shard(tensor), expected, rtol=0, atol=0, equal_nan=True)
        mapping = model.prefetcher_setup.sender_receiver_mapping
        prefill_ids = dict(generator.trace_id_prefill)
        started = time.perf_counter()
        model.switch_mode("prefill")
        assert model.global_cb_trace_state.global_cb.is_suspended()
        blocker = ttnn.create_global_circular_buffer(mesh_device, mapping, 64 * 1024)
        # Ordinary allocations must not claim the suspended data reservation.
        assert (
            blocker.buffer_address() + blocker.size() <= expected_layout[0]
            or blocker.buffer_address() >= expected_layout[0] + expected_layout[2]
        )
        model.switch_mode("decode")
        global_cb = model.prefetcher_setup.global_circular_buffer
        assert not global_cb.is_suspended()
        assert (global_cb.buffer_address(), global_cb.config_address(), global_cb.size()) == expected_layout
        del blocker
        logger.info(f"Reserved GCB suspend/resume with blocker: {time.perf_counter() - started:.6f}s")
        for params in (None, sampling_params):
            output = generator.decode_forward(
                tokens,
                positions,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=params,
                reset_inputs=True,
                reset_batch=True,
            )
            values = output[0] if isinstance(output, tuple) else output
            torch.testing.assert_close(values[0], expected_outputs[params is not None], rtol=0, atol=0)
        assert (
            tuple(
                generator.trace_ids_decode[generator._decode_preparation_key(page_table, mode, False, False)]
                for mode in (False, True)
            )
            + (model.sampling.trace_ids,)
            == captured_ids
        )
        assert generator.trace_id_prefill == prefill_ids
    finally:
        # Trace IDs are scoped to the manager under which they were captured.
        model.switch_mode("prefill")
        for trace_id in generator.trace_id_prefill.values():
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)
        generator.trace_id_prefill.clear()
        mesh_device.load_sub_device_manager(model.mesh_sub_device_manager_id_decode)
        for trace_id in generator.trace_ids_decode.values():
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)
        generator.trace_ids_decode.clear()
        model.sampling.reset_trace()
        del generator
        gc.collect()
