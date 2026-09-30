# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Blackhole cold-start coverage: no eager warmup may hide L1 placement errors."""

import pytest
import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.sampling import SamplingParams
from models.demos.llama3_70b_galaxy.demo.text_qwen_demo import create_tt_qwen_model
from models.demos.llama3_70b_galaxy.tests.unit_tests.test_prepared_trace_io import _release
from models.demos.llama3_70b_galaxy.tests.unit_tests.test_prepared_trace_io import (
    fresh_prefetcher_cache as fresh_prefetcher_cache,
)
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "worker_l1_size": 1345000,
            "l1_small_size": 16384,
            "trace_region_size": 64000000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_prefetcher", [True, False], ids=["prefetcher", "no-prefetcher"])
@pytest.mark.parametrize("first_capture", ["prefill", "decode"])
def test_blackhole_cold_trace_io(
    mesh_device, reset_seeds, monkeypatch, tmp_path, use_prefetcher, first_capture, fresh_prefetcher_cache
):
    if "blackhole" not in str(mesh_device.arch()).lower():
        pytest.skip("Exercises Blackhole Galaxy's two decode paths")
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    monkeypatch.setenv("HF_MODEL", "models/tt_transformers/model_params/Qwen3-32B")
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("QWEN_BH_PREFETCHER", str(int(use_prefetcher)))
    monkeypatch.setenv("QWEN_BH_UNFUSED_CCL", "1")
    args, model, page_table, kv_cache = create_tt_qwen_model(
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
    assert model.use_prefetcher == use_prefetcher
    generator = Generator(model, args, mesh_device)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    try:
        if first_capture == "prefill":
            generator.warmup_model_prefill(kv_cache, enable_trace=True, can_sample_on_device=True)
            assert any(generator.trace_id_prefill.values())
            assert not any(generator.trace_ids_decode.values())
        else:
            generator.decode_forward(
                torch.zeros((32, 1), dtype=torch.int32),
                torch.full((32,), -1, dtype=torch.int32),
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_params=params,
                reset_inputs=True,
                reset_batch=True,
            )
            assert any(generator.trace_ids_decode.values())
            assert not any(generator.trace_id_prefill.values())
        previous = None
        for repetition in range(3):
            results = []
            for batch, length in ((1, 97), (1, 777), (32, 97)):
                logits = generator.prefill_forward_text(
                    torch.ones((batch, length), dtype=torch.int32),
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=torch.full((batch,), length, dtype=torch.int32),
                    enable_trace=True,
                )
                assert torch.isfinite(logits).all()
                tokens = torch.zeros((32, 1), dtype=torch.int32)
                tokens[:batch, 0] = logits[:, 0, :].argmax(-1).to(torch.int32)
                positions = torch.full((32,), -1, dtype=torch.int32)
                positions[:batch] = length
                for step in range(3):
                    sampled = (
                        generator.decode_forward(
                            tokens,
                            positions,
                            page_table=page_table,
                            kv_cache=kv_cache,
                            sampling_params=params,
                            reset_inputs=step == 0,
                            reset_batch=step == 0,
                        )[0]
                        .reshape(-1)[:batch]
                        .clone()
                    )
                    results.append(sampled)
            state = (
                dict(generator.trace_id_prefill),
                dict(generator.trace_ids_decode),
                tuple(t.buffer_address() for *_, t in generator._prepared_trace_io._buffers),
                mesh_device.num_program_cache_entries(),
                model.global_cb_trace_state._layout,
            )
            if previous is not None:
                prior_state, prior_results = previous
                assert state == prior_state
                for actual, expected in zip(results, prior_results):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            previous = state, results
    finally:
        if any(generator.trace_id_prefill.values()) or any(generator.trace_ids_decode.values()):
            _release(generator)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "worker_l1_size": 1345000,
            "l1_small_size": 0,
            "trace_region_size": 64000000,
        }
    ],
    indirect=True,
)
def test_blackhole_prefetcher_requires_small_l1(
    mesh_device, reset_seeds, monkeypatch, tmp_path, fresh_prefetcher_cache
):
    with pytest.raises(RuntimeError, match="l1_small_size=16384"):  # allow-pytest.raises: Python-only preflight guard
        test_blackhole_cold_trace_io(
            mesh_device, reset_seeds, monkeypatch, tmp_path, True, "prefill", fresh_prefetcher_cache
        )
