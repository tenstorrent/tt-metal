# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded whole-model trace attribution including canonical device sampling."""

import functools
import hashlib
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.bounded_profile import check_artifact_budget
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import ARTIFACT_BUDGET, CASES, SCOPE
from models.demos.qwen38_27b_qb2.tests.test_bounded_layer_profile import digest
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_FULL_TRACE_PROFILE") != "1", reason="explicit allocated-Galaxy profile")
def test_full_trace_profile():
    import tracy

    path = Path(os.environ["QWEN_PROFILE_RECEIPT"])
    assert not path.exists(), "Preserve each attempt"
    length, batch = int(os.environ["QWEN_PROFILE_CONTEXT"]), int(os.environ["QWEN_PROFILE_BATCH"])
    assert (length, batch) in CASES
    recurrence = os.getenv("QWEN_PROFILE_RECURRENCE", "single_step_shared_qk")
    assert recurrence in (
        "single_step_shared_qk",
        "single_step_flat_prepare_epilogue",
        "single_step_compact_gdn",
    ), "Profile only the explicitly selected BFP8 decoder policy"
    source = Path(__file__).resolve().parents[1]
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        scope=SCOPE,
        input_tokens=length,
        batch=batch,
        expected_recurrence=recurrence,
        layer_indices=list(range(64)),
        prefill_calls=0,
        output_hashes=[],
        token_hashes=[],
        operand_hashes={},
        replays=[],
        source_sha256={
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.suffix in (".py", ".cpp", ".h", ".hpp")
        },
    )
    save(path, report)
    check_artifact_budget(path.parent, **ARTIFACT_BUDGET)
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = gen = None
    originals, backups = [], []
    loading_descriptor = Qwen38TPDecoder.__dict__["from_state_dict"]
    loading_method = Qwen38TPDecoder.from_state_dict

    def drain():
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        check_artifact_budget(path.parent, **ARTIFACT_BUDGET)

    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))

        @classmethod
        def load_layer(cls, *args, **kwargs):
            layer = loading_method(*args, **kwargs)
            drain()
            return layer

        Qwen38TPDecoder.from_state_dict = load_layer
        try:
            gen = build_generator(source, mesh, topology=ttnn.Topology.Linear)
        finally:
            Qwen38TPDecoder.from_state_dict = loading_descriptor
        assert gen.model.layer_indices == list(range(64)) and len(gen.model.layers) == 64
        assert not gen.model.decode_buckets, "Fixed-batch profiling must not substitute a smaller serving bucket"
        report.update(device_ids=list(mesh.get_device_ids()), precision=gen.model.precision)
        assert len(report["device_ids"]) == 4
        assert all(
            layer.policy.get("decode_recurrence") == recurrence
            for layer in gen.model.layers
            if layer.kind == "linear_attention"
        )
        assert all(
            layer.policy["kv_dtype"] == "bfloat8_b" for layer in gen.model.layers if layer.kind == "full_attention"
        )
        assert set(gen.model.precision["weight_groups"].values()) == {"bfloat8_b"}
        assert gen.model.precision["recurrent_dtype"] == "float32"
        report["profiler_program_support_count"] = os.getenv("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT")

        def no_prefill(*args, **kwargs):
            report["prefill_calls"] += 1
            raise AssertionError("Full trace diagnostic must not profile prompt prefill")

        for name in ("prefill", "prefill_batch"):
            originals.append((gen.model, name, getattr(gen.model, name)))
            setattr(gen.model, name, no_prefill)
        drain()
        report["state"] = "seeding_cache"
        save(path, report)
        cache = gen._ensure_cache(batch, length + 128)
        gen._ensure_history(128)
        drain()
        rng = torch.Generator().manual_seed(20261008 + length + batch)
        for index, state in enumerate(cache.layers):
            for name in ("key", "value", "conv", "recurrent"):
                target = getattr(state, name)
                if target is None:
                    continue
                host = torch.randn(tuple(target.shape), generator=rng)
                if name == "recurrent":
                    host *= 0.1
                else:
                    host = host.bfloat16()
                report["operand_hashes"][f"L{index}_{name}"] = digest(host)
                seeded = gen.model.upload(host, dtype=target.dtype, layout=target.layout)
                del host
                ttnn.copy(seeded, target)
                if name in ("conv", "recurrent"):
                    backups.append((target, seeded))
                else:
                    ttnn.deallocate(seeded)
                drain()
        tokens = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        tokens.reshape(-1)[:batch] = torch.arange(batch) + 100
        positions = torch.full((batch,), length, dtype=torch.int32)
        seeds = torch.arange(32, dtype=torch.int32) + 42
        gen.set_batch_sampling_params(top_k=[1] * 32, top_p=[1.0] * 32, temperature=[1.0] * 32, seed=seeds.tolist())
        report["operand_hashes"].update(tokens=digest(tokens), positions=digest(positions), pages=digest(gen.page_host))

        def restore():
            for target, backup in backups:
                ttnn.copy(backup, target)
            for host, target, counter in (
                (tokens, gen.tokens, "tokens"),
                (positions, gen.positions, "positions"),
                (positions, gen.rope_indices, "rope"),
                (seeds, gen.sampler.seeds_tt_tensor, "seeds"),
            ):
                gen._copy(host, target, counter)
            gen._reset_history()
            drain()

        def readback(logits):
            host = gen._host_logits(logits)
            assert torch.isfinite(host).all()
            report["output_hashes"].append(digest(host))
            report["token_hashes"].append(digest(gen._read_tokens()[:batch]))
            drain()

        report["state"] = "warming"
        save(path, report)
        for _ in range(2):
            restore()
            logits = gen._model_step()
            gen._sampling_step(logits)
            gen._append_history()
            drain()
            readback(logits)
            ttnn.deallocate(logits)
        assert len(set(report["output_hashes"])) == len(set(report["token_hashes"])) == 1
        restore()

        def mark(method, label):
            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                tracy.signpost(f"FULLTRACE_{label}_BEGIN")
                try:
                    return method(*args, **kwargs)
                finally:
                    tracy.signpost(f"FULLTRACE_{label}_END")

            return wrapped

        for index, layer in enumerate(gen.model.layers):
            originals.append((layer, "decode_forward", layer.decode_forward))
            layer.decode_forward = mark(layer.decode_forward, f"L{index}")
        report["state"] = "capturing"
        save(path, report)
        tracy.signpost("FULLTRACE_MODEL_BEGIN")
        gen.trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            gen.logits = gen._model_step()
        finally:
            ttnn.end_trace_capture(mesh, gen.trace, cq_id=0)
            tracy.signpost("FULLTRACE_MODEL_END")
        tracy.signpost("FULLTRACE_SAMPLER_BEGIN")
        gen.sample_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            gen._sampling_step(gen.logits)
            gen._append_history()
        finally:
            ttnn.end_trace_capture(mesh, gen.sample_trace, cq_id=0)
            tracy.signpost("FULLTRACE_SAMPLER_END")
        report.update(model_trace_id=int(gen.trace), sample_trace_id=int(gen.sample_trace), state="replaying")
        save(path, report)
        drain()
        for index in range(3):
            restore()
            started = time.perf_counter()
            ttnn.execute_trace(mesh, gen.trace, cq_id=0, blocking=False)
            ttnn.execute_trace(mesh, gen.sample_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - started
            report["replays"].append(dict(index=index, host_step_s=elapsed))
            ttnn.ReadDeviceProfiler(mesh)
            readback(gen.logits)
            assert len(set(report["output_hashes"])) == len(set(report["token_hashes"])) == 1
            save(path, report)
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        Qwen38TPDecoder.from_state_dict = loading_descriptor
        for obj, name, method in reversed(originals):
            setattr(obj, name, method)
        try:
            try:
                if gen is not None:
                    gen.close()
            finally:
                try:
                    if mesh is not None:
                        ttnn.close_mesh_device(mesh)
                finally:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        except BaseException as error:
            report.update(state="failed", passed=False, cleanup_error=str(error)[:2000])
            raise
        finally:
            save(path, report)
