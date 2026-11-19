"""Reduced exact-width factored trace proof; no production selection.

Device owner only; never run concurrently with another TT job. Tracked proof::

 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_factored_decode_traces \
 --tracked --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/factored_decode_tracked_v1.json

Clean timing: same command without tracker environment/--tracked; use a new
output and --iterations 10. The original whole-model B1 path remains selected.
Only prefix active slots2/32 are factored: arbitrary slot mapping is future work.
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
        self.inactive = tt.full(
            tuple(gen.sample_logits.shape),
            -100.0,
            device=gen.mesh,
            dtype=tt.bfloat16,
            layout=tt.TILE_LAYOUT,
            memory_config=tt.DRAM_MEMORY_CONFIG,
        )

    def attention(self, index, slot):
        tt, gen = self.tt, self.gen
        token = tt.reshape(gen.decode_tokens[:, :, :, slot : slot + 1], (1, 1))
        words = tt.concat([tt.reshape(token, (1, 1, 1)), gen.word_suffix], dim=2)
        if index == 0:
            if slot == 0:
                tt.copy(self.inactive, gen.sample_logits)
            x = gen.model.embed(token)
        else:
            previous = gen.model.layers[index - 1]
            x = previous.mlp_mix.write(self.residual[slot], self.blocks[slot], self.injection[slot])
        layer = gen.model.layers[index]
        residual, mixed, injection = layer._attention_chunk(
            x, gen.caches[slot].layers[index], words if layer.has_ple else None
        )
        # Every cross-segment value is copied into setup-time storage. No trace
        # output allocation survives capture; no broad corruptible exemption.
        tt.copy(residual, self.residual[slot])
        tt.copy(mixed, self.mixed[slot])
        tt.copy(injection, self.injection[slot])

    def moe(self, index, width):
        tt = self.tt
        mixed = tt.concat(self.mixed[:width], dim=2)
        blocks = self.gen.model.layers[index].mlp(mixed)
        for slot in range(width):
            tt.copy(blocks[:, :, slot : slot + 1, :], self.blocks[slot])

    def terminal(self, slot):
        from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk

        layer = self.gen.model.layers[-1]
        residual = layer.mlp_mix.write(self.residual[slot], self.blocks[slot], self.injection[slot])
        write_token_chunk(self.gen.model.logits(residual), self.gen.sample_logits, slot)

    def eager(self, width):
        for index in range(len(self.gen.model.layers)):
            for slot in range(width):
                self.attention(index, slot)
            self.moe(index, width)
        for slot in range(width):
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
                    "note": "Measured allocator growth across capture only; includes trace-storage bank alignment",
                }
            )

        for index in range(len(self.gen.model.layers)):
            for slot in range(32):
                record(("attention", index, slot), lambda index=index, slot=slot: self.attention(index, slot))
            for width in (2, 32):
                record(("moe", index, width), lambda index=index, width=width: self.moe(index, width))
        for slot in range(32):
            record(("terminal", slot), lambda slot=slot: self.terminal(slot))

    def replay(self, width):
        if width == 1:
            self.tt.execute_trace(self.gen.mesh, self.gen._capture((0,)), cq_id=0, blocking=False)
            return
        for index in range(len(self.gen.model.layers)):
            for slot in range(width):
                self.tt.execute_trace(self.gen.mesh, self.traces["attention", index, slot], cq_id=0, blocking=False)
            self.tt.execute_trace(self.gen.mesh, self.traces["moe", index, width], cq_id=0, blocking=False)
        for slot in range(width):
            self.tt.execute_trace(self.gen.mesh, self.traces["terminal", slot], cq_id=0, blocking=False)

    def close(self):
        for trace in self.traces.values():
            self.tt.release_trace(self.gen.mesh, trace)


def cpu_check():
    """Exercise segment boundaries, shared workspace reuse and dormant rows.

    Torch stand-ins validate orchestration only, never TT arithmetic or traces.
    No TTNN import or device access occurs.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    class FakeTT:
        bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = torch.float32, None, None

        @staticmethod
        def zeros(shape, **kwargs):
            return torch.zeros(shape)

        @staticmethod
        def full(shape, value, **kwargs):
            return torch.full(shape, value)

        reshape = staticmethod(torch.reshape)
        concat = staticmethod(torch.cat)
        copy = staticmethod(lambda source, destination: destination.copy_(source))

        @staticmethod
        def execute_trace(mesh, trace, **kwargs):
            trace()

    class Layer:
        def __init__(self, index):
            self.index, self.has_ple = index, index == 1
            self.mlp_mix = SimpleNamespace(write=lambda x, block, injection: x + block * injection)

        def _attention_chunk(self, x, state, words):
            state["updates"] += 1
            assert (words is not None) == self.has_ple
            if words is not None:
                assert tuple(words.shape) == (1, 1, 16)
            residual = x + self.index + 1
            return residual, residual * 2, torch.full((1, 1, 1, 1), 0.125)

        def mlp(self, mixed):
            return mixed * (self.index + 2) + 1

    layers = [Layer(index) for index in range(3)]
    model = SimpleNamespace(
        layers=layers,
        embed=lambda token: token.reshape(1, 1, 1, 1).float().expand(1, 1, 1, 2560).clone(),
        logits=lambda residual: residual[..., :64].clone(),
    )
    caches = [SimpleNamespace(layers=[{"updates": 0} for _ in layers]) for _ in range(32)]
    gen = SimpleNamespace(
        model=model,
        mesh=None,
        caches=caches,
        sample_logits=torch.full((1, 1, 32, 64), -100.0),
        decode_tokens=torch.arange(32).reshape(1, 1, 1, 32),
        word_suffix=torch.zeros(1, 1, 15, dtype=torch.int64),
        _record_trace=lambda body: body,
    )
    split = FactoredDecode(gen, FakeTT)
    split.capture(lambda: 0)
    assert len(split.traces) == 134

    def original(width):
        residuals = [model.embed(gen.decode_tokens[..., slot : slot + 1].reshape(1, 1)) for slot in range(width)]
        for layer in layers:
            prepared = []
            for slot, residual in enumerate(residuals):
                words = torch.zeros(1, 1, 16) if layer.has_ple else None
                prepared.append(layer._attention_chunk(residual, {"updates": 0}, words))
            blocks = layer.mlp(torch.cat([value[1] for value in prepared], dim=2))
            residuals = [
                layer.mlp_mix.write(value[0], blocks[..., slot : slot + 1, :], value[2])
                for slot, value in enumerate(prepared)
            ]
        output = torch.full_like(gen.sample_logits, -100.0)
        for slot, residual in enumerate(residuals):
            output[..., slot : slot + 1, :] = model.logits(residual)
        return output

    def write(source, destination, offset):
        destination[..., offset : offset + source.shape[-2], :].copy_(source)

    target = "models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output.write_token_chunk"
    with patch(target, write):
        for width in (2, 32, 2):
            expected = original(width)
            before = [[state["updates"] for state in cache.layers] for cache in caches]
            split.replay(width)
            assert torch.equal(gen.sample_logits, expected), (width, "segmented logits or stale inactive rows")
            for slot, cache in enumerate(caches):
                assert [state["updates"] for state in cache.layers] == [
                    value + int(slot < width) for value in before[slot]
                ]
            gen.decode_tokens.add_(17)
    print("CPU segment orchestration: exact2/32/2, stale workspace cleared, dormant updates preserved")


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
        "scope": "Reduced0/1/3 factored trace diagnostic",
        "tracked": args.tracked,
        "checks": [],
        "timings": {},
        "widths": [1, 2, 32],
        "original_whole_model_b1_unchanged": True,
        "no_cache_migration": True,
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "tt/model.py",
                "tt/generator.py",
                "tt/multichip_decoder.py",
                "tests/diagnostic_sources/factored_before/generator.py",
                "tests/whole_decode_control.py",
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
            root, mesh, layer_indices=(0, 1, 3), max_seq_len=128, max_batch_size=32, defer_kv_cache=True
        )
        pool = gen.allocate_kv_cache_per_layer([((128, 1, 32, 256), torch.bfloat16, 0)])
        table = torch.arange(128, dtype=torch.int32).reshape(32, 4)
        factored = FactoredDecode(gen, tt)
        gen.prepare()
        mark("warm_factored")
        for width in (2, 32):
            gen.reset()
            factored.eager(width)
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
        assert mesh.num_program_cache_entries() == programs
        seed = [785, 13458, 13, 785, 13, 13458, 42]
        prompts = [(seed * 5)[: (3, 7, 31, 33)[slot % 4]] for slot in range(32)]

        def prefill(width):
            gen.reset()
            tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in prompts[:width]], batch_first=True)
            gen.prefill_forward(
                tokens,
                page_table=table[:width],
                kv_cache=pool,
                prompt_lens=[len(p) for p in prompts[:width]],
                slots=tuple(range(width)),
            )

        def state_hash(width):
            digest = hashlib.sha256()
            for slot in range(width):
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
                    for slot in range(width):
                        position = gen.caches[slot].position
                        pages = table[slot, : (position + 31) // 32].long()
                        values = host[pages].permute(1, 0, 2, 3).reshape(1, -1, 256)[:, :position].contiguous()
                        digest.update(values.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def refresh_tokens(width, step):
            values = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            values[..., :width] = torch.tensor([42 + step + row % 4 for row in range(width)])
            gen._refresh(gen.decode_tokens, values)

        def replay(kind, width):
            if kind == "original":
                slots = tuple(range(width))
                if slots not in gen.traces:
                    before = allocated()
                    trace = gen._capture(slots)
                    after = allocated()
                    result.setdefault("original_capture_memory", []).append(
                        {"width": width, "retained_capture_bytes_per_rank": after - before}
                    )
                else:
                    trace = gen.traces[slots]
                tt.execute_trace(mesh, trace, cq_id=0, blocking=False)
            else:
                factored.replay(width)

        for width in result["widths"]:
            mark(f"exact_width_{width}")
            reference = []
            for kind in ("original", "factored"):
                prefill(width)
                for step in range(3):
                    refresh_tokens(width, step)
                    replay(kind, width)
                    for cache in gen.caches[:width]:
                        cache.position += 1
                    logits = gen._read_logits(gen.sample_logits)[:width].clone()
                    state = state_hash(width)
                    gen._sample(tuple(range(width)))
                    tokens = gen._read_tokens()[:width]
                    if kind == "original":
                        reference.append((logits, state, tokens))
                    else:
                        expected, expected_state, expected_tokens = reference[step]
                        check = {
                            "width": width,
                            "step": step,
                            "logits_exact": bool(torch.equal(expected, logits)),
                            "max_abs": float((expected - logits).abs().max()),
                            "state_exact": state == expected_state,
                            "tokens_exact": tokens == expected_tokens,
                        }
                        result["checks"].append(check)
                        assert check["logits_exact"] and check["state_exact"] and check["tokens_exact"], check
            if not args.tracked:
                result["timings"][str(width)] = {}
                for kind in ("original", "factored"):
                    prefill(width)
                    refresh_tokens(width, 0)

                    def iteration():
                        replay(kind, width)
                        gen._sample(tuple(range(width)))

                    for _ in range(2):
                        iteration()
                    tt.synchronize_device(mesh)
                    begin = time.perf_counter()
                    for _ in range(args.iterations):
                        iteration()
                    tt.synchronize_device(mesh)
                    result["timings"][str(width)][kind] = {
                        "token_out_ms": (time.perf_counter() - begin) * 1000 / args.iterations,
                        "iterations": args.iterations,
                        "token_feedback": "resident common sampled tokens",
                        "host_inputs_or_reads_in_window": False,
                    }
        assert mesh.num_program_cache_entries() == programs
        result["memory_final"] = memory(mesh)
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
