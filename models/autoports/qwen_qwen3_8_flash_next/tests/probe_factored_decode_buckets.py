"""Reduced exact-width factored trace proof; no production selection.

Device owner only; never run concurrently with another TT job. Tracked proof::

 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_factored_decode_buckets \
 --tracked --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/factored_decode_buckets_tracked_v1.json

Clean timing: same command without tracker environment/--tracked; use a new
output and --iterations 10. This B32 configuration factors every singleton;
standalone B1 production is untouched. Original traces remain diagnostic controls.
MoE widths1/2/4/8/16/32 duplicate the last valid input row to fill the bucket.
Only real rows scatter back; n1 retains its native FP32 reduction.
Layers0/1/2/3 include an ordinary nonfirst GDN for memory accounting.
Membership metadata changes only at scheduler transitions; fixed cache addresses stay put.
All segments retain exact original grouped MoE arithmetic, dtype and fidelity.
No trace eviction, cache migration, precision override, or custom sampler math.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path


class FactoredDecode:
    """Persistent per-slot working buffers; layers reuse them sequentially."""

    def __init__(self, gen, tt):
        self.gen, self.tt = gen, tt
        self.traces = {}
        self.capture_memory = []

        def buffers(width):
            return [
                tt.zeros(
                    (1, 1, 1, width),
                    device=gen.mesh,
                    dtype=tt.bfloat16,
                    layout=tt.TILE_LAYOUT,
                    memory_config=tt.DRAM_MEMORY_CONFIG,
                )
                for _ in range(32)
            ]

        self.residual, self.mixed, self.blocks, self.injection = buffers(2560), buffers(2560), buffers(2560), buffers(1)
        import torch

        self.slot_indices = tt.from_torch(
            torch.tensor([list(range(32)), [32] + [0] * 31], dtype=torch.int32),
            device=gen.mesh,
            dtype=tt.uint32,
            layout=tt.ROW_MAJOR_LAYOUT,
            memory_config=tt.DRAM_MEMORY_CONFIG,
        )
        self.packed = {
            width: tt.zeros(
                (1, 1, width, 2560),
                device=gen.mesh,
                dtype=tt.bfloat16,
                layout=tt.TILE_LAYOUT,
                memory_config=tt.DRAM_MEMORY_CONFIG,
            )
            for width in (1, 2, 4, 8, 16, 32)
        }
        self.active_slots = tuple(range(32))
        self.metadata_uploads = 0
        self.inactive = tt.full(
            tuple(gen.sample_logits.shape),
            -100.0,
            device=gen.mesh,
            dtype=tt.bfloat16,
            layout=tt.TILE_LAYOUT,
            memory_config=tt.DRAM_MEMORY_CONFIG,
        )

    def set_slots(self, slots):
        import torch

        from models.autoports.qwen_qwen3_8_flash_next.tests.factored_bucket_copy import metadata_words, validate_slots

        slots = validate_slots(slots)
        if slots != self.active_slots:
            values = torch.tensor(metadata_words(slots), dtype=torch.int32)
            self.gen._refresh(self.slot_indices, values)
            self.active_slots = slots
            self.metadata_uploads += 1

    def attention(self, index, slot):
        tt, gen = self.tt, self.gen
        layer = gen.model.layers[index]
        words = None
        if index == 0 or layer.has_ple:
            token = tt.reshape(gen.decode_tokens[:, :, :, slot : slot + 1], (1, 1))
            if layer.has_ple:
                words = tt.concat([tt.reshape(token, (1, 1, 1)), gen.word_suffix], dim=2)
        if index == 0:
            x = gen.model.embed(token)
        else:
            previous = gen.model.layers[index - 1]
            x = previous.mlp_mix.write(self.residual[slot], self.blocks[slot], self.injection[slot])
        residual, mixed, injection = layer._attention_chunk(x, gen.caches[slot].layers[index], words)
        tt.copy(residual, self.residual[slot])
        tt.copy(mixed, self.mixed[slot])
        tt.copy(injection, self.injection[slot])

    @staticmethod
    def bucket(count):
        from models.autoports.qwen_qwen3_8_flash_next.tests.factored_bucket_copy import bucket_for_count

        return bucket_for_count(count)

    def moe(self, index, width):
        from models.autoports.qwen_qwen3_8_flash_next.tests.factored_bucket_copy import indexed_copy

        mixed = self.packed[width]
        indexed_copy(self.slot_indices, mixed, self.mixed)
        blocks = self.gen.model.layers[index].mlp(mixed)
        indexed_copy(self.slot_indices, blocks, self.blocks, scatter=True)

    def terminal(self, slot):
        from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk

        layer = self.gen.model.layers[-1]
        residual = layer.mlp_mix.write(self.residual[slot], self.blocks[slot], self.injection[slot])
        write_token_chunk(self.gen.model.logits(residual), self.gen.sample_logits, slot)

    def eager(self, slots):
        self.set_slots(slots)
        self.tt.copy(self.inactive, self.gen.sample_logits)
        for index in range(len(self.gen.model.layers)):
            for slot in slots:
                self.attention(index, slot)
            self.moe(index, self.bucket(len(slots)))
        for slot in slots:
            self.terminal(slot)

    def capture(self, memory):
        def record(key, body):
            before = memory()
            self.traces[key] = self.gen._record_trace(body)
            after = memory()
            self.capture_memory.append(
                {
                    "key": list(key),
                    "retained_capture_bytes_per_rank": after - before,
                    "note": "Allocator growth across capture only, including trace-storage bank alignment",
                }
            )

        record(("clear",), lambda: self.tt.copy(self.inactive, self.gen.sample_logits))
        for index in range(len(self.gen.model.layers)):
            for slot in range(32):
                record(("attention", index, slot), lambda index=index, slot=slot: self.attention(index, slot))
            for width in (1, 2, 4, 8, 16, 32):
                self.set_slots(tuple(range(width)))
                record(("moe", index, width), lambda index=index, width=width: self.moe(index, width))
        for slot in range(32):
            record(("terminal", slot), lambda slot=slot: self.terminal(slot))

    def replay(self, slots):
        if self.gen.max_batch_size == 1 and slots == (0,):
            self.tt.execute_trace(self.gen.mesh, self.gen._capture(slots), cq_id=0, blocking=False)
            return
        assert slots == self.active_slots, "Membership metadata must be refreshed outside replay"
        self.tt.execute_trace(self.gen.mesh, self.traces["clear",], cq_id=0, blocking=False)
        for index in range(len(self.gen.model.layers)):
            for slot in slots:
                self.tt.execute_trace(self.gen.mesh, self.traces["attention", index, slot], cq_id=0, blocking=False)
            self.tt.execute_trace(
                self.gen.mesh, self.traces["moe", index, self.bucket(len(slots))], cq_id=0, blocking=False
            )
        for slot in slots:
            self.tt.execute_trace(self.gen.mesh, self.traces["terminal", slot], cq_id=0, blocking=False)

    def close(self):
        for trace in self.traces.values():
            self.tt.release_trace(self.gen.mesh, trace)


def cpu_check():
    from models.autoports.qwen_qwen3_8_flash_next.tests.factored_bucket_copy import cpu_check as check

    check()
    # The v1 CPU orchestration proof remains unchanged and separate.
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_factored_decode_traces import cpu_check as original_check

    original_check()
    from types import SimpleNamespace

    calls, uploads = [], []
    proof = object.__new__(FactoredDecode)
    proof.tt = SimpleNamespace(execute_trace=lambda mesh, key, **kw: calls.append(key))
    proof.gen = SimpleNamespace(
        mesh=None,
        max_batch_size=32,
        model=SimpleNamespace(layers=[0, 1, 3]),
        _refresh=lambda dst, value: uploads.append(value.clone()),
        _capture=lambda slots: ("original", slots),
    )
    proof.slot_indices, proof.active_slots, proof.metadata_uploads = None, tuple(range(32)), 0
    keys = (
        [("clear",)]
        + [("attention", i, s) for i in range(3) for s in range(32)]
        + [("moe", i, n) for i in range(3) for n in (1, 2, 4, 8, 16, 32)]
        + [("terminal", s) for s in range(32)]
    )
    proof.traces = {key: key for key in keys}
    for slots in ((3, 29), (1, 7, 31), (29,), (3, 29)):
        proof.set_slots(slots)
        prior = len(uploads)
        for _ in range(2):
            proof.set_slots(slots)
            calls.clear()
            proof.replay(slots)
            expected = [("clear",)]
            for i in range(3):
                expected.extend(("attention", i, s) for s in slots)
                expected.append(("moe", i, proof.bucket(len(slots))))
            expected.extend(("terminal", s) for s in slots)
            assert calls == expected
            assert len(uploads) == prior
    assert len(uploads) == 4
    print("CPU arbitrary-slot dispatch/transition-only metadata proof passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
        return
    if args.output is None:
        parser.error("Device probe requires --output")
    assert not args.output.exists() and 1 <= args.iterations <= 32
    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH") and "TT_METAL_WATCHER" not in os.environ
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
        assert os.environ.get(name) in (None, "", "0")
    if args.tracked:
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    else:
        for name in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS"):
            assert os.environ.get(name) in (None, "", "0")
    import torch
    from ttnn.tools import trace_allocation_tracker as tracker

    import ttnn as tt
    from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory
    from models.autoports.qwen_qwen3_8_flash_next.tests.whole_decode_control import build_generator
    from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import _state_tensors

    if not args.tracked:
        assert not tracker.TRACE_ALLOC_TRACKING and not tracker.TRACE_ALLOC_DIAGNOSTICS
    root = Path(__file__).resolve().parents[1]
    result = {
        "passed": False,
        "acceptance_eligible": False,
        "scope": "Reduced0/1/2/3 duplicate-row bucketed factored trace diagnostic; real common sampling; production decode factoring not selected",
        "tracked": args.tracked,
        "checks": [],
        "timings": {},
        "widths": [1, 2, 3, 5, 8, 9, 17, 32],
        "physical_moe_buckets": [1, 2, 4, 8, 16, 32],
        "slot_cases": [
            [0],
            [29],
            [0, 1],
            [3, 29],
            [1, 7, 31],
            [0, 3, 9, 16, 29],
            [1, 4, 7, 9, 16, 20, 27, 31],
            list(range(1, 18, 2)),
            list(range(1, 18)),
            list(range(32)),
            [29],
            [3, 29],
        ],
        "standalone_b1_production_untouched": True,
        "b32_all_singletons_factored_with_original_fp32_moe": True,
        "no_cache_migration": True,
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "tt/model.py",
                "tt/generator.py",
                "tt/multichip_decoder.py",
                "tests/diagnostic_sources/factored_before/generator.py",
                "tests/whole_decode_control.py",
                "tt/prefill_checkpoint.py",
                "tt/token_feedback.py",
                "tests/probe_factored_decode_buckets.py",
                "tests/factored_bucket_copy.py",
            )
        },
    }
    mesh = gen = factored = None
    started = time.monotonic()
    phase = "setup"

    def mark(value):
        nonlocal phase
        phase = value
        print(json.dumps({"phase": value, "wall_seconds": time.monotonic() - started}), flush=True)

    try:
        torch.set_num_threads(8)
        tt.set_fabric_config(tt.FabricConfig.FABRIC_1D_RING)
        mesh = tt.open_mesh_device(tt.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        gen = build_generator(
            root, mesh, layer_indices=(0, 1, 2, 3), max_seq_len=128, max_batch_size=32, defer_kv_cache=True
        )
        pool = gen.allocate_kv_cache_per_layer([((128, 1, 32, 256), torch.bfloat16, 0)])
        table = torch.arange(128, dtype=torch.int32).reshape(32, 4)
        factored = FactoredDecode(gen, tt)
        gen.prepare()
        mark("warm_factored")
        for width in (1, 2, 4, 8, 16, 32):
            gen.reset()
            factored.eager(tuple(range(width)))
        gen.reset()
        tt.synchronize_device(mesh)
        programs = mesh.num_program_cache_entries()

        def allocated():
            tt.synchronize_device(mesh)
            view = memory(mesh)
            return view["num_banks"] * view["total_bytes_allocated_per_bank"]

        result["memory_prepared"] = memory(mesh)
        mark("capture_all_factored")
        factored.capture(allocated)
        result["factored_trace_count"] = len(factored.traces)
        result["factored_capture_memory"] = factored.capture_memory
        result["factored_retained_capture_bytes_per_rank"] = sum(
            row["retained_capture_bytes_per_rank"] for row in factored.capture_memory
        )
        attention = {}
        for index in range(4):
            values = [
                row["retained_capture_bytes_per_rank"]
                for row in factored.capture_memory
                if row["key"][:2] == ["attention", index]
            ]
            assert len(set(values)) == 1
            attention[index] = values[0]
        moe_sums = [
            sum(
                row["retained_capture_bytes_per_rank"]
                for row in factored.capture_memory
                if row["key"][:2] == ["moe", index]
            )
            for index in range(4)
        ]
        terminal_and_clear = sum(
            row["retained_capture_bytes_per_rank"]
            for row in factored.capture_memory
            if row["key"][0] in ("terminal", "clear")
        )
        result["full48_capture_projection"] = {
            "measured_per_slot_attention_bytes": attention,
            "measured_moe_six_bucket_bytes_each_layer": moe_sums,
            "layer_multiplicities": {"first_gdn": 1, "ple_gdn": 1, "ordinary_gdn": 34, "qsa": 12},
            "projected_attention_bytes": 32 * (attention[0] + attention[1] + 34 * attention[2] + 12 * attention[3]),
            "projected_moe_bytes_using_max_measured_layer": 48 * max(moe_sums),
            "terminal_and_clear_bytes": terminal_and_clear,
            "scope": "C128 four-layer measured projection, not full48 fit proof; excludes weights/cache/workspaces/sampling/prefill and original diagnostic control traces",
        }
        result["full48_capture_projection"]["projected_total_bytes"] = sum(
            result["full48_capture_projection"][key]
            for key in (
                "projected_attention_bytes",
                "projected_moe_bytes_using_max_measured_layer",
                "terminal_and_clear_bytes",
            )
        )
        assert mesh.num_program_cache_entries() == programs
        seed = [785, 13458, 13, 785, 13, 13458, 42]
        prompts = [(seed * 5)[: (3, 7, 31, 33)[slot % 4]] for slot in range(32)]

        def prefill(slots):
            gen.reset()
            tokens = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(p) for p in [prompts[slot] for slot in slots]], batch_first=True
            )
            gen.prefill_forward(
                tokens,
                page_table=table[list(slots)],
                kv_cache=pool,
                prompt_lens=[len(p) for p in [prompts[slot] for slot in slots]],
                slots=slots,
            )

        def state_hash(slots):
            digest = hashlib.sha256()
            for slot in range(32):
                cache = gen.caches[slot]
                digest.update(str(cache.position).encode())
                for name, tensor in _state_tensors(cache):
                    digest.update(f"{slot}:{name}".encode())
                    for part in tt.get_device_tensors(tensor):
                        host = tt.to_torch(part).contiguous()
                        digest.update(host.view(torch.uint8).numpy().tobytes())
            for tensor in pool.pair_for_attention_layer(0):
                for part in tt.get_device_tensors(tensor):
                    host = tt.to_torch(part)
                    for slot in range(32):
                        position = gen.caches[slot].position
                        pages = table[slot, : (position + 31) // 32].long()
                        values = host[pages].permute(1, 0, 2, 3).reshape(1, -1, 256)[:, :position].contiguous()
                        digest.update(values.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def refresh_tokens(slots, step):
            values = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            values[..., list(slots)] = torch.tensor([42 + step + slot % 4 for slot in slots], dtype=torch.int32)
            gen._refresh(gen.decode_tokens, values)

        def replay(kind, slots):
            if kind == "original":
                if slots not in gen.traces:
                    before = allocated()
                    trace = gen._capture(slots)
                    after = allocated()
                    result.setdefault("original_capture_memory", []).append(
                        {"slots": list(slots), "width": len(slots), "retained_capture_bytes_per_rank": after - before}
                    )
                else:
                    trace = gen.traces[slots]
                tt.execute_trace(mesh, trace, cq_id=0, blocking=False)
            else:
                factored.replay(slots)

        for case_index, case in enumerate(result["slot_cases"]):
            slots, width = tuple(case), len(case)
            mark(f"exact_slots_{case_index}_{width}")
            reference = []
            for kind in ("original", "factored"):
                prefill(slots)
                factored.set_slots(slots)
                for step in range(3):
                    refresh_tokens(slots, step)
                    replay(kind, slots)
                    for slot in slots:
                        gen.caches[slot].position += 1
                    logits = gen._read_logits(gen.sample_logits)[list(slots)].clone()
                    state = state_hash(slots)
                    gen._sample(slots)
                    physical_tokens = gen._read_tokens()
                    tokens = [physical_tokens[slot] for slot in slots]
                    if kind == "original":
                        reference.append((logits, state, tokens))
                    else:
                        expected, expected_state, expected_tokens = reference[step]
                        check = {
                            "width": width,
                            "slots": list(slots),
                            "step": step,
                            "logits_exact": bool(torch.equal(expected, logits)),
                            "max_abs": float((expected - logits).abs().max()),
                            "state_exact": state == expected_state,
                            "tokens_exact": tokens == expected_tokens,
                        }
                        result["checks"].append(check)
                        assert check["logits_exact"] and check["state_exact"] and check["tokens_exact"], check
            if not args.tracked:
                result["timings"][str(case_index)] = {}
                for kind in ("original", "factored"):
                    prefill(slots)
                    factored.set_slots(slots)
                    refresh_tokens(slots, 0)

                    def iteration():
                        replay(kind, slots)
                        gen._sample(slots)

                    for _ in range(2):
                        iteration()
                    tt.synchronize_device(mesh)
                    metadata_before = factored.metadata_uploads
                    begin = time.perf_counter()
                    for _ in range(args.iterations):
                        iteration()
                    tt.synchronize_device(mesh)
                    assert factored.metadata_uploads == metadata_before
                    result["timings"][str(case_index)][kind] = {
                        "slots": list(slots),
                        "metadata_uploads_in_window": factored.metadata_uploads - metadata_before,
                        "token_out_ms": (time.perf_counter() - begin) * 1000 / args.iterations,
                        "iterations": args.iterations,
                        "token_feedback": "resident common sampled tokens",
                        "host_inputs_or_reads_in_window": False,
                    }
        mark("membership_transitions_without_reset")
        sequence = [(0, 1), (3, 29), (29,), (1, 7, 31), (0, 3, 9, 16, 29), tuple(range(32)), (29,), (3, 29)]
        reference = []
        for kind in ("original", "factored"):
            prefill(tuple(range(32)))
            for step, slots in enumerate(sequence):
                factored.set_slots(slots)
                refresh_tokens(slots, step)
                replay(kind, slots)
                for slot in slots:
                    gen.caches[slot].position += 1
                logits = gen._read_logits(gen.sample_logits).clone()
                state = state_hash(tuple(range(32)))
                gen._sample(slots)
                tokens = gen._read_tokens()
                if kind == "original":
                    reference.append((logits, state, tokens))
                else:
                    expected, expected_state, expected_tokens = reference[step]
                    check = {
                        "phase": "membership_without_reset",
                        "slots": list(slots),
                        "step": step,
                        "logits_exact": bool(torch.equal(logits, expected)),
                        "state_exact": state == expected_state,
                        "tokens_exact": tokens == expected_tokens,
                    }
                    result["checks"].append(check)
                    assert check["logits_exact"] and check["state_exact"] and check["tokens_exact"], check
        assert mesh.num_program_cache_entries() == programs
        result["memory_final"] = memory(mesh)
        result["membership_metadata_uploads"] = factored.metadata_uploads
        result["program_cache_entries"] = programs
        result["passed"] = True
    except BaseException as error:
        result.update(failed_phase=phase, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        if factored is not None:
            result["factored_capture_memory"] = factored.capture_memory
            result["factored_trace_count"] = len(factored.traces)
            result["factored_retained_capture_bytes_per_rank"] = sum(
                item["retained_capture_bytes_per_rank"] for item in factored.capture_memory
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if factored is not None:
            factored.close()
        if gen is not None:
            gen.close()
        if mesh is not None:
            tt.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
