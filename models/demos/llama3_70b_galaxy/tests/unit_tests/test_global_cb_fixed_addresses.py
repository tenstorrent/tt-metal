# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_global_cb_fixed_addresses_across_trace_mode_switches(
    mesh_device, expect_error, reset_seeds, monkeypatch, tmp_path
):
    """Reconstruct the GCB while retaining decode, sampling, and prefill traces."""
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

    def trace_ids():
        return (
            generator.trace_ids_decode[False],
            generator.trace_ids_decode[True],
            tuple(slot["id"] for slot in model.sampling._trace_states.values()),
        )

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
                    reload_inputs=True,
                    reload_page_table=True,
                    reload_sampling_params=params is not None,
                    reset_sampling_state=params is not None,
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
            logger.info(f"Reconstructed GCB: data={layout[0]:#x}, config={layout[1]:#x}, size={layout[2]}")
            del global_cb
            ids = trace_ids()
            assert all(ids) and all(ids[2])
            if expected_layout is None:
                expected_layout, captured_ids = layout, ids
            else:
                assert layout == expected_layout
                assert ids == captured_ids  # Replayed, not silently recaptured.

        # Once the GCB is released for prefill, another allocation can claim its
        # addresses. The model must fail before submitting a decode trace in that case.
        mapping = model.prefetcher_setup.sender_receiver_mapping
        ttnn.synchronize_device(mesh_device)
        model.switch_mode("prefill")
        blocker = ttnn.create_global_circular_buffer(
            mesh_device,
            mapping,
            expected_layout[2],
            buffer_address=expected_layout[0],
            config_address=expected_layout[1],
        )
        try:
            with expect_error(RuntimeError, "Requested buffer address"):
                model.switch_mode("decode")
            assert not model.is_decode_setup
            assert trace_ids() == captured_ids
        finally:
            ttnn.synchronize_device(mesh_device)
            blocker.deallocate()
        # Recovery is explicit: after the caller removes the conflict, the same
        # captured traces and both original addresses can still be used.
        model.switch_mode("decode")
        restored = model.prefetcher_setup.global_circular_buffer
        assert (restored.buffer_address(), restored.config_address(), restored.size()) == expected_layout
        for params in (None, sampling_params):
            output = generator.decode_forward(
                tokens,
                positions,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=params,
                reload_inputs=True,
                reload_page_table=True,
                reload_sampling_params=params is not None,
                reset_sampling_state=params is not None,
            )
            values = output[0] if isinstance(output, tuple) else output
            torch.testing.assert_close(values[0], expected_outputs[params is not None], rtol=0, atol=0)
        assert trace_ids() == captured_ids
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
