# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-model trace lifetime, feedback and page-table contract probe.

Run with TT_METAL_TRACE_ALLOC_TRACKING=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 for acceptance. --quick prepares an
explicit diagnostic subset through the helper's supported diagnostic API; it is
never acceptance evidence. No positive control or performance claim is made here.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import source_identity
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import RecurrentState
from models.autoports.qwen_qwen3_6_27b.tt.page_table_ownership import page_table_for_attention


class TraceProbe:
    def __init__(self, generator, report, output):
        self.gen, self.mesh, self.report, self.output = generator, generator.mesh, report, output
        self.phase = "prepare"
        self.writes, self.replays, self.bindings = [], [], []
        self.readbacks = 0
        self.original_write = generator._write
        self.original_execute = generator._execute
        self.original_decode = generator.model.decode
        self.original_sample = generator.sampling._run_sampling
        self.last_decode = self.last_sample = None
        generator._write = self.observe_write
        generator._execute = self.observe_execute
        generator.model.decode = self.observe_decode_capture
        generator.sampling._run_sampling = self.observe_sample_capture

    def save(self):
        self.report["probe_readbacks"] = self.readbacks
        self.output.write_text(json.dumps(self.report, indent=2) + "\n")

    def read(self, tensor, *, replicated=True):
        shards = ttnn.get_device_tensors(tensor)
        values = [ttnn.to_torch(shard).clone() for shard in shards]
        self.readbacks += len(values)
        if replicated:
            assert all(torch.equal(values[0], value) for value in values[1:]), "Replicated control tensor ranks differ"
        return values[0]

    def control(self):
        return {
            "tokens": self.read(self.gen.tokens).flatten().tolist(),
            "positions": self.read(self.gen.positions).flatten().tolist(),
            "rope_positions": self.read(self.gen.rope_positions).flatten().tolist(),
        }

    def binding(self, tokens, positions, rope_positions):
        return {
            "tokens": int(tokens.buffer_address()),
            "positions": int(positions.buffer_address()),
            "rope_positions": int(rope_positions.buffer_address()),
        }

    def observe_write(self, value, target, counter=None):
        if self.phase != "prepare":
            self.writes.append({"phase": self.phase, "address": int(target.buffer_address()), "counter": counter})
        return self.original_write(value, target, counter)

    def observe_decode_capture(self, tokens, positions, **kwargs):
        if ttnn.is_trace_capture_active(self.mesh):
            self.bindings.append({"kind": "decode", **self.binding(tokens, positions, kwargs["rope_positions"])})
        return self.original_decode(tokens, positions, **kwargs)

    def observe_sample_capture(self, *args, **kwargs):
        if ttnn.is_trace_capture_active(self.mesh):
            self.bindings.append(
                {
                    "kind": "sample",
                    "tt_out_tok": int(kwargs["tt_out_tok"].buffer_address()),
                    "logits": int(args[0].buffer_address()),
                }
            )
        return self.original_sample(*args, **kwargs)

    def observe_execute(self, key):
        # Reads are diagnostic observation only. They never reconstruct inputs.
        observe = self.phase.startswith("steady") and key in ("decode", "sample")
        before = self.control() if observe else None
        self.original_execute(key)
        after = self.control() if observe else None
        event = {"phase": self.phase, "key": key, "trace_id": str(self.gen.traces[key])}
        if observe:
            event.update(before=before, after=after)
            if key == "decode":
                assert after["tokens"] == before["tokens"], "Model decode overwrote the persistent token input"
                assert after["positions"] == [position + 1 for position in before["positions"]]
                assert after["rope_positions"] == [position + 1 for position in before["rope_positions"]]
                self.last_decode = event
            else:
                assert before["positions"] == after["positions"], "Sampling changed model positions"
                assert before["rope_positions"] == after["rope_positions"], "Sampling changed RoPE positions"
                self.last_sample = event
        if self.phase != "prepare":
            self.replays.append(event)

    def freeze(self):
        self.trace_ids = {key: str(value) for key, value in self.gen.traces.items()}
        self.cache_entries = int(self.mesh.num_program_cache_entries())
        self.capture_count = int(self.gen.counters["captures"])
        self.addresses = self.binding(self.gen.tokens, self.gen.positions, self.gen.rope_positions)
        decodes = [binding for binding in self.bindings if binding["kind"] == "decode"]
        samples = [binding for binding in self.bindings if binding["kind"] == "sample"]
        assert len(decodes) == len(samples) == 1, "Expected one captured execution signature per model/sampling graph"
        assert {key: decodes[0][key] for key in self.addresses} == self.addresses
        assert samples[0]["tt_out_tok"] == self.addresses["tokens"]
        assert samples[0]["logits"] == int(self.gen.logits.buffer_address())
        self.report.update(
            captured_bindings=self.bindings,
            trace_ids=self.trace_ids,
            captures_at_start=self.capture_count,
            program_cache_entries_at_start=self.cache_entries,
            prepared_lengths=self.gen.prefill_single.prepared_lengths,
        )
        self.phase = "request"
        self.save()

    def check_reuse(self):
        assert {key: str(value) for key, value in self.gen.traces.items()} == self.trace_ids, "Trace identity changed"
        assert self.gen.counters["captures"] == self.capture_count, "Request handling captured another trace"
        assert self.binding(self.gen.tokens, self.gen.positions, self.gen.rope_positions) == self.addresses
        entries = int(self.mesh.num_program_cache_entries())
        assert entries == self.cache_entries, f"Late program compilation: {self.cache_entries} -> {entries}"

    def step(self, *, expected_token, position, page_table=None, reload_page_table=False):
        self.phase = "steady_page_change" if reload_page_table else "steady"
        before = self.control()
        assert before["tokens"][0] == expected_token, "Sampled token did not reach the next decode input"
        assert before["positions"] == before["rope_positions"] == [position]
        write_start, replay_start = len(self.writes), len(self.replays)
        counters_before = dict(self.gen.counters)
        result = self.gen.decode_forward(
            None,
            None,
            page_table=self.gen.page_host if page_table is None else page_table,
            kv_cache=self.gen.cache,
            reload_inputs=False,
            reload_page_table=reload_page_table,
            sample_on_device=True,
        )
        writes = self.writes[write_start:]
        page_addresses = {
            int(page_table_for_attention(state).buffer_address())
            for state in self.gen.cache
            if not isinstance(state, RecurrentState)
        }
        if reload_page_table:
            assert writes and all(write["address"] in page_addresses for write in writes), writes
            assert self.gen.counters["page_table_refreshes"] > counters_before.get("page_table_refreshes", 0)
        else:
            assert not writes, f"Host input rebuild in steady decode: {writes}"
            assert self.gen.counters["page_table_refreshes"] == counters_before.get("page_table_refreshes", 0)
        for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "full_logits_readbacks"):
            assert self.gen.counters[name] == counters_before.get(name, 0), f"Unexpected steady {name}"
        events = self.replays[replay_start:]
        assert [event["key"] for event in events] == ["decode", "sample"]
        assert self.last_decode["before"] == before
        assert self.last_sample["before"] == self.last_decode["after"]
        after = self.last_sample["after"]
        assert int(result[0, 0]) == after["tokens"][0]
        assert after["positions"] == after["rope_positions"] == [position + 1]
        self.check_reuse()
        self.phase = "request"
        return after["tokens"][0]

    def setup(self, length, *, sampled=False, seed=1234, resume=None):
        self.phase = "request"
        self.gen.reset()
        self.gen.configure_sampling(
            temperature=0.7 if sampled else 0.0,
            top_k=32 if sampled else 1,
            top_p=0.9 if sampled else 0.0,
            seed=seed,
        )
        prompt = torch.arange(length, dtype=torch.int32)[None] + 100
        if resume is not None:
            prefix = self.gen.prefill_forward(
                prompt,
                page_table=self.gen.page_host,
                kv_cache=self.gen.cache,
                prompt_lens=[resume],
                return_device=True,
            )
            del prefix
        pieces = self.gen.prefill_forward(
            prompt,
            page_table=self.gen.page_host,
            kv_cache=self.gen.cache,
            prompt_lens=[length],
            start_pos=[0 if resume is None else resume],
            return_device=True,
        )
        value = ttnn.reshape(pieces[0], (1, 1, 1, self.gen.model.config.vocab_size // 4))
        padded = ttnn.pad(value, ((0, 0), (0, 0), (0, 31), (0, 0)), value=0)
        ttnn.copy(padded, self.gen.logits)
        del pieces, value, padded
        self.gen._execute("sample")
        sampled_token = int(self.read(self.gen.tokens).flatten()[0])
        self.gen._write(torch.tensor([length], dtype=torch.int32), self.gen.positions, "position_refreshes")
        self.gen._write(torch.tensor([length], dtype=torch.int32), self.gen.rope_positions, "rope_refreshes")
        # The first generation token came from sample -> tt_out_tok. No token
        # write occurs here or in the following resident-input decode steps.
        return sampled_token

    def run_request(self, length, *, sampled, seed, resume=None):
        print(f"TRACE_REQUEST length={length} resume={resume} sampled={sampled} seed={seed}", flush=True)
        token = self.setup(length, sampled=sampled, seed=seed, resume=resume)
        generated = [token]
        for offset in range(3):
            token = self.step(expected_token=token, position=length + offset)
            generated.append(token)
        case = dict(length=length, resume=resume, sampled=sampled, seed=seed, tokens=generated, passed=True)
        self.report["requests"].append(case)
        self.save()

    def key_row(self, state, page, row):
        # No device slice/program is dispatched after capture. Host extraction
        # retains only one small row from the diagnostic cache readback.
        host = ttnn.to_torch(ttnn.get_device_tensors(state.key)[0])
        self.readbacks += 1
        return host[page, 0, row].clone()

    def changed_pages(self):
        print("TRACE_CHANGED_PAGES position=32", flush=True)
        token = self.setup(31, seed=801)
        token = self.step(expected_token=token, position=31)
        state = next(state for state in self.gen.cache if not isinstance(state, RecurrentState))
        original_table = self.gen.page_host.clone()
        updated_table = original_table.clone()
        old_page, new_page = int(original_table[0, 1]), int(original_table[0, -1])
        updated_table[0, 1], updated_table[0, -1] = original_table[0, -1], original_table[0, 1]
        old_row, new_row = self.key_row(state, old_page, 0), self.key_row(state, new_page, 0)
        assert torch.count_nonzero(old_row) == torch.count_nonzero(new_row) == 0
        token = self.step(expected_token=token, position=32, page_table=updated_table, reload_page_table=True)
        for cache in self.gen.cache:
            if not isinstance(cache, RecurrentState):
                assert torch.equal(self.read(page_table_for_attention(cache)), updated_table)
        new_written = self.key_row(state, new_page, 0)
        old_unchanged = self.key_row(state, old_page, 0)
        assert torch.count_nonzero(new_written) > 0, "Decode did not update the newly mapped physical KV page"
        assert torch.equal(old_unchanged, old_row), "Decode retained the stale page mapping"
        # Finish this workload with unchanged resident replay as well.
        token = self.step(expected_token=token, position=33)
        self.report["changed_pages"] = {
            "position": 32,
            "old_physical_page": old_page,
            "new_physical_page": new_page,
            "pre_remap_rows_equal_zero": True,
            "new_page_written": True,
            "old_page_unchanged": True,
            "bound_device_tables_match": True,
            "final_token": token,
            "passed": True,
        }
        self.save()
        return token, 34


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Diagnostic subset only; never acceptance evidence")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    include_programs = os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    report = {
        "source_sha256": source_identity(),
        "layers": [0, 3],
        "context_allocation": 2048,
        "quick": args.quick,
        "tracking_enabled": tracking,
        "program_cache_allocations_included": include_programs,
        "positive_control": "Separate parent-owned positive-control artifact is required",
        "requests": [],
        "passed": False,
        "acceptance_ready": False,
    }
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = probe = None
    started = time.perf_counter()
    try:
        model = Model(mesh, layer_indices=[0, 3])
        gen = QwenGenerator(model, batch_size=1, max_seq_len=2048)
        probe = TraceProbe(gen, report, args.output)
        if args.quick:
            print("TRACE_QUICK_DIAGNOSTIC_PREPARE", flush=True)
            gen.prefill_single.prepare(_diagnostic_lengths=[2, 17, 31, 33, 128, 512])
        else:
            print("TRACE_FULL_PREPARE physical_lengths=1..512", flush=True)
        gen.prepare(prefill_lengths=(2,))
        probe.freeze()
        lengths = [2, 17, 33, 128] if args.quick else [2, 17, 33, 127, 128, 129, 511, 512, 513, 1025]
        for index, length in enumerate(lengths):
            probe.run_request(length, sampled=bool(index % 2), seed=1700 + index)
        probe.run_request(64 if args.quick else 545, sampled=False, seed=1901, resume=31)
        probe.run_request(545, sampled=True, seed=1902, resume=512)
        # Returning to greedy after regular sampling must reuse the same traces.
        probe.run_request(33, sampled=False, seed=1903)
        final_token, final_position = probe.changed_pages()
        # No new compilation or persistent allocation after the last check can
        # hide behind a workload ending at prefill: always end in decode replay.
        probe.step(expected_token=final_token, position=final_position)
        probe.check_reuse()
        decode_events = [event for event in probe.replays if event["key"] == "decode" and "before" in event]
        distinct_tokens = sorted({event["before"]["tokens"][0] for event in decode_events})
        distinct_positions = sorted({event["before"]["positions"][0] for event in decode_events})
        assert len(distinct_tokens) > 1 and len(distinct_positions) > 1, "Probe did not exercise changed trace inputs"
        assert gen.counters["full_logits_readbacks"] == 0, "Token-out probe crossed the full-logits host boundary"
        report.update(
            passed=True,
            acceptance_ready=not args.quick and tracking and include_programs,
            trace_ids_final={key: str(value) for key, value in gen.traces.items()},
            captures_final=int(gen.counters["captures"]),
            program_cache_entries_final=int(mesh.num_program_cache_entries()),
            counters=dict(gen.counters),
            distinct_decode_token_inputs=distinct_tokens,
            distinct_decode_positions=distinct_positions,
            request_writes=probe.writes,
            replays=probe.replays,
        )
    except BaseException as error:
        report["exception"] = f"{type(error).__name__}: {error}"
        if probe is not None:
            report.update(request_writes=probe.writes, replays=probe.replays, captured_bindings=probe.bindings)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        if probe is not None:
            probe.save()
        else:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
    print(f"TRACE_CONTRACT passed={report['passed']} acceptance_ready={report['acceptance_ready']}", flush=True)


if __name__ == "__main__":
    main()
