# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-checkpoint request isolation and long-context packed prefill checks."""

import json
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.kv_validation import cache_metrics, summarize_metrics
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def _output(runtime, *, striped):
    shards = ttnn.get_device_tensors(runtime.output)
    cp, tp = runtime.mesh_config.cp_degree, runtime.mesh_config.tp_degree
    rows = torch.stack([ttnn.to_torch(shards[rank * tp]).float().squeeze(0).squeeze(0) for rank in range(cp)])
    if striped:
        rows = rows.reshape(cp, -1, 32, rows.shape[-1]).permute(1, 0, 2, 3)
    return rows.reshape(-1, rows.shape[-1])


def _release_layouts(runtime):
    for trace_id in runtime.traces.values():
        ttnn.release_trace(runtime.mesh_device, trace_id)
    runtime.traces.clear()
    runtime.trace_id = None


def _compare(expected, actual, label, measurements, *, min_pcc=0.98):
    metrics = summarize_metrics([cache_metrics(expected, actual)])
    measurements.append({"case": label, **metrics})
    logger.info("[packed_check] {} PCC={:.6f} RRMSE={:.6f}", label, metrics["pcc"], metrics["relative_rmse"])
    assert metrics["pcc"] >= min_pcc, f"{label}: {metrics}"
    assert metrics["relative_rmse"] < 0.21, f"{label}: {metrics}"


def _legacy_reference(runtime, tokens, context_len):
    mesh, chunk = runtime.mesh_device, runtime.config.chunk_size

    def stage(values, start=0):
        values = values.tolist() + [0] * (chunk - len(values))
        ttnn.copy_host_to_device_tensor(runtime._host_rows(values), runtime.input_tokens)
        ttnn.copy_host_to_device_tensor(runtime._host_rows(list(range(start, start + chunk))), runtime.positions)
        runtime.model.prefill_metadata.update(slot_idx=0, kv_actual_global=start)

    def forward():
        embeddings = runtime.model.transform_and_embed_prefill_inputs_device(runtime.input_tokens)
        result = runtime.model(hidden_states=embeddings)
        ttnn.copy(result, runtime.output)
        result.deallocate(True)

    stage(tokens[:chunk])
    forward()
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        try:
            forward()
        finally:
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        for start in range(0, context_len, chunk):
            stage(tokens[start : start + chunk], start)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            logger.info("[packed_reference] legacy context {}/{}", start + chunk, context_len)
        return _output(runtime, striped=False)
    finally:
        ttnn.release_trace(mesh, trace_id)


def run_packed_prefill(mesh_device, hf_model_id, tokens, context_len, chunk_size, output_dir):
    mesh = MeshConfig(mesh_device)
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(hf_model_id))
    params = SimpleNamespace(
        mesh_shape=(8, 4),
        sp_axis=0,
        chunk_size=chunk_size,
        max_seq_len=context_len,
        num_users=6,
        num_layers=60,
        use_trace=True,
    )
    logger.info("[packed_check] allocating six {}-token slots across all 60 layers", context_len)
    caches = allocate_ring_kv_caches(mesh, args, num_users=6, max_seq_len=context_len, prefill_chunk_size=chunk_size)
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id=hf_model_id, tt_cache_path=None, config=params)
    measurements, cases, completions = [], [], []
    runtime.set_layer_completion_sink(SimpleNamespace(layers_completed=lambda *span: completions.append(span)))
    total_requests = 0
    report = {
        "passed": False,
        "model": hf_model_id,
        "layers": 60,
        "chunk_size": chunk_size,
        "context_len": context_len,
        "reference": "fixed-placement request isolation; legacy ring_mla for the 256k continuation",
        "measurements": measurements,
    }

    def run(segments, isolated_slot=None):
        nonlocal total_requests
        values = [
            tokens[s.slot_id * chunk_size + s.actual_start : s.slot_id * chunk_size + s.actual_end] for s in segments
        ]
        if isolated_slot is not None:
            values = [
                value if s.slot_id == isolated_slot else torch.zeros_like(value) for s, value in zip(segments, values)
            ]
        inp = runtime.make_batch_input([value.tolist() for value in values], segments)
        start = time.monotonic()
        runtime.prefill_batch(inp, caches, segments=segments)
        total_requests += len(segments)
        assert completions == [(0, params.num_layers, s.slot_id, s.actual_start, s.actual_end) for s in segments]
        completions.clear()
        assert len(runtime.traces) <= runtime.max_traces
        for segment in segments:
            assert runtime.slot_ends[segment.slot_id] == segment.actual_end
        logger.info(
            "[packed_check] batch={} wall={:.3f}s traces={}", segments, time.monotonic() - start, len(runtime.traces)
        )

    try:
        logger.info("[packed_check] loading and compiling the full checkpoint")
        runtime.compile(caches)
        runtime.capture_trace(caches)
        schedules = [
            [(i, 0, length) for i, length in enumerate((256, 512, 768, 1024, 1536, 2048))],
            [(0, 256, 512), (2, 768, 256), (5, 2048, 768)],
            [(5 - i, 0, length) for i, length in enumerate((256, 512, 768, 1024, 1536, 2048))],
            [(1, 0, 1600), (4, 0, 257)],
        ]
        first_trace = None
        for iteration, schedule in enumerate(schedules):
            segments = tuple(PrefillSegment(slot, start, start + length) for slot, start, length in schedule)
            run(segments)
            actual = _output(runtime, striped=True)
            assert torch.isfinite(actual).all()
            if iteration == 0:
                first_trace = runtime.trace_id
                runtime._forward()
                ttnn.synchronize_device(mesh_device)
                _compare(_output(runtime, striped=True), actual, "mixed trace/eager", measurements, min_pcc=0.999)
            elif iteration == 2:
                assert runtime.trace_id == first_trace, "Layout trace must survive slot and prefix changes"
            cases.append(actual)
        for slot in range(params.num_users):
            for iteration, schedule in enumerate(schedules):
                segments = tuple(PrefillSegment(s, start, start + length) for s, start, length in schedule)
                run(segments, isolated_slot=slot)
                isolated = _output(runtime, striped=True)
                offset = 0
                for segment in segments:
                    length = segment.actual_end - segment.actual_start
                    if segment.slot_id == slot:
                        expected = cases[iteration][offset : offset + length]
                        actual = isolated[offset : offset + length]
                        label = f"isolated batch {iteration} {segment}"
                        _compare(expected, actual, label, measurements, min_pcc=0.999)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=label)
                    offset += segment.padded_length()
        cases.clear()
        _release_layouts(runtime)
        legacy_tail = _legacy_reference(runtime, tokens, context_len)

        long_chunk = chunk_size - 2048
        for start in range(0, context_len, long_chunk):
            end = min(start + long_chunk, context_len)
            segments = (PrefillSegment(0, start, end), PrefillSegment(1, 0, 512), PrefillSegment(2, 0, 1536))
            run(segments)
        final_length = end - start
        actual = _output(runtime, striped=True)[:final_length]
        _compare(legacy_tail[-final_length:], actual, f"packed {context_len}-token continuation", measurements)
        report["passed"] = True
    finally:
        report["request_segments"] = total_requests
        (output_dir / "packed_prefill.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info("[packed_check] report: {}", output_dir / "packed_prefill.json")
        runtime.release_trace()
