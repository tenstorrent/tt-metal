# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Long-running traced prefill soak: fixed allocations, six resident slots, no device reset."""

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig


# The normal repository timeout is too short for a 600-iteration soak. The launcher
# optionally arms the dispatch-progress watchdog (TRIAGE=1) instead.
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 268435456, "l1_small_size": 0})
@pytest.mark.parametrize("num_iters", [1, 12, 20, 600], ids=lambda n: f"iters{n}")
@pytest.mark.parametrize("n_chunks", [1, 20, 32], ids=lambda n: f"chunks{n}")
@torch.no_grad()
def test_prefill_stress(mesh_device, num_iters, n_chunks, tmp_path):
    torch.set_num_threads(4)
    adapter = Gemma4PrefillAdapter()
    assert os.getenv("HF_MODEL", adapter.hf_model_id) == adapter.hf_model_id
    config = Gemma4ServiceConfig
    chunk_size = config.CHUNK_SIZE
    assert chunk_size == 8192, "This soak targets production 8192-token chunks"
    context_len = n_chunks * chunk_size
    assert context_len <= config.MAX_SEQ_LEN
    trace_dir = Path(os.getenv("PREFILL_TRACE_DIR", adapter.prefill_trace_default))
    token_ids = json.loads((trace_dir / "metadata.json").read_text())["token_ids"]
    assert len(token_ids) >= context_len, f"{trace_dir} needs at least {context_len} tokens"
    output_dir = Path(os.getenv("GEMMA4_STRESS_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    run = int(os.getenv("GEMMA4_STRESS_RUN", "1"))
    params = PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=config.NUM_LAYERS,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=config.MAX_SEQ_LEN,
        chunk_size=chunk_size,
        num_users=config.MAX_USER_SLOTS,
        capacity_factor=1,
        num_links=2,
        gate_mode_name="",
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path((8, 4)),
        use_trace=True,
    )
    logger.info(
        "Building Gemma4PrefillRuntime: layers={} slots={} capacity={} chunk={} chunks={} iterations={}",
        params.num_layers,
        params.num_users,
        params.max_seq_len,
        chunk_size,
        n_chunks,
        num_iters,
    )
    hf_config = adapter.load_hf_config()
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    compile_start = time.perf_counter()
    runtime.compile(caches)
    compile_s = time.perf_counter() - compile_start
    # Each slot gets a distinct, repeatable rotation of the real-text token stream.
    # Pre-stage host tensors, as in the reference perf test; device addresses stay fixed.
    host_chunks = []
    for slot in range(params.num_users):
        offset = slot * chunk_size % len(token_ids)
        prompt = (token_ids[offset:] + token_ids[:offset])[:context_len]
        host_chunks.append(
            [runtime._host_tokens(prompt[start : start + chunk_size]) for start in range(0, context_len, chunk_size)]
        )
    capture_start = time.perf_counter()
    runtime.capture_trace(caches)
    capture_s = time.perf_counter() - capture_start
    logger.info("Gemma4 stress forward loop: compile={:.2f}s capture={:.2f}s", compile_s, capture_s)

    baselines = {}
    device_times = []
    stage_times = []
    started = time.perf_counter()
    try:
        with (output_dir / f"timings_{run:02d}.csv").open("w", buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "slot", "chunk", "start", "end", "device_ms", "staging_ms"])
            for iteration in range(num_iters):
                slot = iteration % params.num_users
                iteration_start = time.perf_counter()
                for chunk in range(n_chunks):
                    start, end = chunk * chunk_size, (chunk + 1) * chunk_size
                    runtime.validate_chunk(slot, start, end)
                    stage_start = time.perf_counter()
                    ttnn.copy_host_to_device_tensor(host_chunks[slot][chunk], runtime.input_tokens)
                    runtime.model.prefill_metadata.update(slot_idx=slot, actual_start=start, actual_end=end)
                    ttnn.synchronize_device(mesh_device)
                    staging_ms = (time.perf_counter() - stage_start) * 1000
                    forward_start = time.perf_counter()
                    ttnn.execute_trace(mesh_device, runtime.trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    device_ms = (time.perf_counter() - forward_start) * 1000
                    runtime.slot_ends[slot] = end
                    device_times.append(device_ms)
                    stage_times.append(staging_ms)
                    writer.writerow([iteration, slot, chunk, start, end, device_ms, staging_ms])
                    logger.info(
                        "[chunk timing] iter={} slot={} chunk={}/{} {:.2f} ms",
                        iteration,
                        slot,
                        chunk + 1,
                        n_chunks,
                        device_ms,
                    )

                # A small correctness guard outside the timed forward: finite, nonconstant
                # final hidden states on one chip, and exact repeatability when reusing a slot.
                hidden = ttnn.to_torch(ttnn.get_device_tensors(runtime.output)[0])
                assert torch.isfinite(hidden).all(), f"Nonfinite output: iteration={iteration} slot={slot}"
                assert hidden.float().std() > 0.001, f"Constant output: iteration={iteration} slot={slot}"
                digest = hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                if slot in baselines:
                    assert digest == baselines[slot], f"Output changed on slot reuse: iteration={iteration} slot={slot}"
                else:
                    baselines[slot] = digest
                logger.info(
                    "iter {} done ({} chunks) in {:.3f}s slot={}",
                    iteration + 1,
                    n_chunks,
                    time.perf_counter() - iteration_start,
                    slot,
                )
    finally:
        runtime.release_trace()

    result = dict(
        status="passed",
        run=run,
        iterations=num_iters,
        chunks_per_iteration=n_chunks,
        chunk_size=chunk_size,
        context_len=context_len,
        num_slots=params.num_users,
        layers=params.num_layers,
        chunks_completed=len(device_times),
        compile_s=compile_s,
        capture_s=capture_s,
        forward_loop_wall_s=time.perf_counter() - started,
        mean_device_ms=sum(device_times) / len(device_times),
        mean_staging_ms=sum(stage_times) / len(stage_times),
        output_hashes=baselines,
        input_trace=str(trace_dir),
    )
    (output_dir / f"result_{run:02d}.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Gemma4 stress complete: {}", json.dumps(result))
