# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profile three decode calls, without any prefill or full-model accuracy claim."""

import functools
import hashlib
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.bounded_profile import CASES, SCOPE, VARIANTS, check_artifact_budget
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt import decoder as decoder_module
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.skipif(os.getenv("QWEN_BOUNDED_LAYER_PROFILE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_bounded_layer_profile():
    import tracy

    path = Path(os.environ["QWEN_PROFILE_RECEIPT"])
    assert not path.exists(), "Preserve each capture"
    length = int(os.environ["QWEN_PROFILE_CONTEXT"])
    batch = int(os.environ["QWEN_PROFILE_BATCH"])
    recurrence = os.environ["QWEN_PROFILE_RECURRENCE"]
    assert (length, batch) in CASES and recurrence in VARIANTS
    check_artifact_budget(path.parent)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        scope=SCOPE,
        layers=[0, 3],
        recurrence=recurrence,
        prefill_calls=0,
        decode_calls=0,
        cells=[],
        output_hashes=[],
        operand_hashes={},
        max_decode_calls=3,
        source_sha256={},
    )
    source = Path(__file__).resolve().parents[1]
    for p in sorted(source.rglob("*")):
        if p.suffix in (".py", ".cpp", ".h", ".hpp"):
            report["source_sha256"][str(p.relative_to(source))] = hashlib.sha256(p.read_bytes()).hexdigest()
    save(path, report)
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = gen = None
    originals = []
    backups = []
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        gen = build_generator(source, mesh, layer_indices=[0, 3], topology=ttnn.Topology.Linear)
        report.update(device_ids=list(mesh.get_device_ids()), precision=gen.model.precision)
        assert len(report["device_ids"]) == 4
        assert [x.kind for x in gen.model.layers] == ["linear_attention", "full_attention"]
        assert gen.model.layers[0].policy.get("decode_recurrence", "native") == recurrence
        assert gen.model.layers[1].policy["kv_dtype"] == "bfloat8_b"

        def no_prefill(*args, **kwargs):
            report["prefill_calls"] += 1
            raise AssertionError("Prefill is prohibited in the bounded decode profile")

        for name in ("prefill", "prefill_batch"):
            originals.append((gen.model, name, getattr(gen.model, name)))
            setattr(gen.model, name, no_prefill)
        ttnn.ReadDeviceProfiler(mesh)
        cache = gen._ensure_cache(batch, length + 128)
        rng = torch.Generator().manual_seed(20261008 + length + batch)
        # Seed the actual cache geometry directly. This preserves the scan and
        # state-update workload without thousands of profiled prefill chunks.
        for state in cache.layers:
            for name in ("key", "value", "conv", "recurrent"):
                target = getattr(state, name)
                if target is None:
                    continue
                host = torch.randn(tuple(target.shape), generator=rng)
                if name == "recurrent":
                    host *= 0.1
                if name != "recurrent":
                    host = host.bfloat16()
                report["operand_hashes"][name] = digest(host)
                seeded = gen.model.upload(host, dtype=target.dtype, layout=target.layout)
                del host
                ttnn.copy(seeded, target)
                if name in ("conv", "recurrent"):
                    backups.append((target, seeded))
                else:
                    ttnn.deallocate(seeded)
                ttnn.ReadDeviceProfiler(mesh)
                check_artifact_budget(path.parent)
        tokens = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        tokens.reshape(-1)[:batch] = torch.arange(batch) + 100
        positions = torch.full((batch,), length, dtype=torch.int32)
        report["operand_hashes"].update(tokens=digest(tokens), positions=digest(positions), pages=digest(gen.page_host))

        def restore():
            for target, backup in backups:
                ttnn.copy(backup, target)
            gen._copy(tokens, gen.tokens, "token_refreshes")
            gen._copy(positions, gen.positions, "position_refreshes")
            gen._copy(positions, gen.rope_indices, "rope_refreshes")
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)

        def invoke():
            assert report["decode_calls"] < 3
            report["decode_calls"] += 1
            # Direct model decode keeps the cache position fixed; no sampler or
            # position advance is included in this stage-attribution boundary.
            result = gen.model.decode(
                gen.tokens, gen.positions, cache=cache, page_table=gen.page_table, rope_indices=gen.rope_indices
            )
            ttnn.synchronize_device(mesh)
            return result

        def readback(result):
            value = gen._host_logits(result)
            assert torch.isfinite(value).all()
            result_hash = digest(value)
            report["output_hashes"].append(result_hash)
            del value
            ttnn.deallocate(result)
            ttnn.ReadDeviceProfiler(mesh)
            return result_hash

        report["state"] = "warming"
        save(path, report)
        for _ in range(2):
            restore()
            readback(invoke())
        assert report["output_hashes"][0] == report["output_hashes"][1]
        restore()

        def mark(method, label, linear=False):
            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                suffix = "_" + str(args[1]).replace(".", "_") if linear else ""
                stage = label + suffix
                tracy.signpost(stage + "_BEGIN")
                try:
                    return method(*args, **kwargs)
                finally:
                    tracy.signpost(stage + "_END")

            return wrapped

        prefix = f"P0_S{length}_B{batch}"
        for index, layer in zip((0, 3), gen.model.layers):
            for name in (
                "decode_forward",
                "_norm",
                "_delta",
                "_delta_recurrence",
                "_full_decode",
                "_finish",
                "_linear",
                "_rope_decode",
            ):
                method = getattr(layer, name)
                originals.append((layer, name, method))
                setattr(layer, name, mark(method, f"{prefix}_L{index}_{name}", name == "_linear"))
        for name, index in (("paged_decode", 3), ("packed_decode_conv", 0)):
            method = getattr(decoder_module, name)
            originals.append((decoder_module, name, method))
            setattr(decoder_module, name, mark(method, f"{prefix}_L{index}_{name}"))
        report["state"] = "profiling"
        save(path, report)
        tracy.signpost(prefix + "_MODEL_BEGIN")
        started = time.perf_counter()
        result = invoke()
        elapsed = time.perf_counter() - started
        tracy.signpost(prefix + "_MODEL_END")
        ttnn.ReadDeviceProfiler(mesh)
        readback(result)
        assert len(set(report["output_hashes"])) == 1, "Instrumentation changed decoded logits"
        check_artifact_budget(path.parent)
        report["cells"] = [dict(input_tokens=length, batch=batch, state="completed", diagnostic_host_s=elapsed)]
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:2000]))
        raise
    finally:
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
