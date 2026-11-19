# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Actual common-sampler TP4 seeded replay probe, with no model weights.

Uses QwenGenerator._sample itself through a sampler-only object. The real
vocabulary is248320, sharded62080 per device, with32 physical sampler rows.
Obtain exclusive TT device ownership and preserve the accepted mesh descriptor:

timeout 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal python_env/bin/python \
    -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_seeded_sampling \
    --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/seeded_sampling_device.json

Prepared by a CPU-only audit; the author has not run this device script.
This tests sampler integration, not decoder correctness or throughput.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, choices=(32, 40), default=32)
    parser.add_argument("--sampler-capacity", type=int, choices=(32, 64), default=64)
    args = parser.parse_args()
    if args.top_k > args.sampler_capacity:
        parser.error("Sampler capacity must cover logical top-k")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Enable trace allocation tracking before Python starts")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program cache must remain tracked")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.device_seed import DeviceSeedState
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import PREFILL_BUCKETS, QwenGenerator
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives
    from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import SlotStateMover, remap_token_buffer
    from models.autoports.qwen_qwen3_8_flash_next.tt.token_feedback import DeviceTokenFeedback
    from models.common.sampling.generator import SamplingGenerator, SamplingParams

    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(4)
    report = {
        "passed": False,
        "scope": "actual common sampler through unmodified QwenGenerator._sample; no decoder",
        "mesh": [1, 4],
        "vocabulary": 248320,
        "local_vocabulary": 62080,
        "trace_tracking": True,
        "top_k": args.top_k,
        "sampler_capacity": args.sampler_capacity,
        "checks": [],
    }
    mesh, gen = None, None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024, l1_small_size=32768)
        gen = QwenGenerator.__new__(QwenGenerator)
        gen.mesh = mesh
        # Match constructor-owned lifecycle metadata even though this probe
        # supplies synthetic forward logits and never allocates a shared pool.
        gen.shared_pool = None
        gen.cache_binding_pending = False
        gen.cache_setup_error = None
        gen.owns_cache_storage = True
        gen.max_batch_size, gen.capacity, gen.prepared = 3, 128, True
        gen.traces = {}
        gen.sampling_traces = {}
        gen.counters = dict(
            captures=0,
            releases=0,
            model_replays=0,
            sampling_replays=0,
            prefill_replays=0,
            cache_resets=0,
            token_refreshes=0,
            position_refreshes=0,
            page_table_refreshes=0,
            readbacks=0,
            synchronizations=0,
        )
        gen.prefill_traces = {}
        gen.decode_tokens = gen._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32))
        gen.saved_tokens = gen._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32))
        gen.seed_skip = gen._upload(torch.full((32,), 2**32 - 1, dtype=torch.int64))
        sampler_args = SimpleNamespace(
            vocab_size=248320,
            padded_vocab_size=248320,
            max_batch_size=32,
            max_top_k=args.sampler_capacity,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            model_config={
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 1, "topology": ttnn.Topology.Ring}
            },
        )
        gen.sampler = SamplingGenerator(args=sampler_args, mesh_device=mesh, tt_ccl=Collectives(mesh))
        gen.sampler.tt_sampling._allow_penalties_sampling = False
        params = SamplingParams(temperature=[1.3] * 32, top_k=[args.top_k] * 32, top_p=[0.95] * 32)
        gen.set_sampling_params(params)
        gen.seed_state = DeviceSeedState(mesh, gen.sampler.tt_sampling.seeds_tt_tensor)
        gen.token_feedback = DeviceTokenFeedback(gen.decode_tokens, gen.saved_tokens, gen.seed_state.active)
        gen.seed_state.prepare()
        gen.token_feedback.prepare()
        gen.seed_state.remap(tuple(range(32)), warmup=True)
        remap_token_buffer(gen.decode_tokens, tuple(range(32)), warmup=True)

        # Exactly32 distinct candidates, distributed over all4 vocab shards.
        # Every physical row has identical logits, so row-order comparisons are
        # exact without a host logit update between decode steps.
        candidates = [rank * 62080 + 11 + index * 97 for rank in range(4) for index in range(8)]
        if args.top_k > 32:
            # All best40 lie on rank3: local32 cannot preserve this support.
            candidates = [3 * 62080 + 11 + index * 97 for index in range(40)] + [11 + index * 97 for index in range(24)]
        logits = torch.full((1, 1, 32, 248320), -80.0, dtype=torch.bfloat16)
        for index, token in enumerate(candidates):
            logits[..., token] = 0.6 - index * 0.025
        gen.sample_logits = ttnn.from_torch(
            logits,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        del logits
        # Synthetic terminal logits stand in for forward_chunk only. Keep this
        # immutable source separate from the sampler's mutable logits buffer.
        # This exercises the actual prefill API/slot/seed orchestration below,
        # without claiming any decoder/model correctness evidence.
        prefill_template = ttnn.clone(gen.sample_logits)
        gen.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=248320),
            forward_chunk=lambda tokens, words, *, cache: prefill_template[:, :, : tokens.shape[1], :],
        )
        gen.caches = [SimpleNamespace(layers=[], position=0, capacity=128, shared_kv=False) for _ in range(3)]
        gen.page_tables = {}
        gen.prefill_inputs = {n: gen._upload(torch.zeros((1, n), dtype=torch.int32)) for n in PREFILL_BUCKETS}
        gen.prefill_words = {n: gen._upload(torch.zeros((1, n, 16), dtype=torch.int32)) for n in PREFILL_BUCKETS}
        gen.prefill_output = ttnn.empty((1, 1, 32, 62080), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        gen.inactive_logits = ttnn.full(
            (1, 1, 1, 62080), -100.0, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        # Mirror constructor-owned canonical prefill bindings. The synthetic
        # caches have no model tensors, but use the real mover/host-position
        # contract rather than bypassing current prefill orchestration.
        gen.prefill_cache = SimpleNamespace(layers=[], position=0, capacity=128, shared_kv=False)
        gen.slot_mover = SlotStateMover(gen.caches, gen.prefill_cache)
        from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_checkpoint import PrefillCheckpoints

        gen.prefill_checkpoints = PrefillCheckpoints(gen.caches, mesh)
        gen.prefill_checkpoints.warmup(gen.caches)
        gen.slot_mover.warmup()
        for count in PREFILL_BUCKETS:
            gen.slot_mover._copy(gen.caches[0], gen.prefill_cache)
            gen._prefill_body(count)
            ttnn.full_like(gen.sample_logits, -100.0, optional_tensor=gen.sample_logits)
            gen._write_prefill_last(count, 0)
        ttnn.copy(prefill_template, gen.sample_logits)
        if hasattr(gen, "_warm_sampling"):
            gen._warm_sampling(all_configs=False)
        else:
            gen.sampler.precompile(
                gen.sample_logits, tt_out_tok=gen.decode_tokens, all_configs=False, compile_token_update=True
            )
        ttnn.copy(gen.decode_tokens, gen.saved_tokens)
        gen._restore_inactive_tokens(())
        ttnn.copy(gen.seed_skip, gen.sampler.tt_sampling.seeds_tt_tensor)
        # The restore concat shape is identical for every slot set; slice every
        # row/offset before any capture. Seed op prepare compiles without draws.
        ttnn.synchronize_device(mesh)

        request_seeds = [-7, 2**40 + 91, -7]
        initial_counter = 29
        expected_counters = [0] * 32
        previous_tokens = []

        def initialize(order=(0, 1, 2)):
            nonlocal expected_counters, previous_tokens
            all_seeds = [request_seeds[source] for source in order] + [None] * 29
            gen.seed_state.reset_slots(all_seeds, [initial_counter] * 32, list(range(32)))
            expected_counters = [initial_counter] * 32
            previous_tokens = [1000 + slot for slot in range(32)]
            gen._refresh(gen.decode_tokens, torch.tensor(previous_tokens, dtype=torch.int32).reshape(1, 1, 1, 32))

        def draw(slots, label):
            nonlocal previous_tokens
            slots = tuple(slots)
            gen._sample(slots)
            ttnn.synchronize_device(mesh)
            per_rank = [
                ttnn.to_torch(tensor).flatten().to(torch.int64).tolist()
                for tensor in ttnn.get_device_tensors(gen.decode_tokens)
            ]
            assert len(per_rank) == 4 and all(value == per_rank[0] for value in per_rank), (
                label,
                "rank disagreement",
                per_rank,
            )
            current = per_rank[0]
            for slot in range(32):
                if slot in slots:
                    assert current[slot] in candidates[: gen.formatted_sampling_params.top_k[slot]], (
                        label,
                        slot,
                        current[slot],
                    )
                    expected_counters[slot] += 1
                else:
                    assert current[slot] == previous_tokens[slot], (
                        label,
                        "inactive feedback changed",
                        slot,
                        previous_tokens[slot],
                        current[slot],
                    )
            for rank, state in enumerate(ttnn.get_device_tensors(gen.seed_state.state)):
                host = ttnn.to_torch(state).to(torch.int64)
                actual_counters = (host[:, 2] + (host[:, 3] << 32)).tolist()
                assert actual_counters == expected_counters, (
                    label,
                    rank,
                    "draw counter",
                    actual_counters,
                    expected_counters,
                )
            previous_tokens = current
            report["checks"].append({"name": label, "slots": list(slots), "tokens": [current[slot] for slot in slots]})
            return current[:3]

        # First invocation captures a cold key. It must consume exactly one
        # logical draw, and reproduce once the same trace key is already cached.
        initialize()
        baseline = [draw((0, 1, 2), f"baseline_{step}") for step in range(8)]
        assert all(row[0] == row[2] for row in baseline), "Same seed differs across physical rows"
        assert any(row[0] != row[1] for row in baseline), "Different seeds produced no variety"
        assert len({row[0] for row in baseline}) > 1, "Sampled stream did not vary over draws"
        initialize()
        repeated = [draw((0, 1, 2), f"reset_repeat_{step}") for step in range(8)]
        assert repeated == baseline, ("Reset reproducibility", baseline, repeated)

        order = (2, 0, 1)
        initialize(order)
        reordered = [draw((0, 1, 2), f"fresh_row_order_{step}") for step in range(8)]
        assert reordered == [[row[source] for source in order] for row in baseline]

        initialize()
        for step in range(3):
            assert draw((0, 1, 2), f"before_migration_{step}") == baseline[step]
        remap = list(order) + list(range(3, 32))
        gen.seed_state.remap(remap)
        remap_token_buffer(gen.decode_tokens, remap)
        previous_tokens = [previous_tokens[source] for source in remap]
        expected_counters = [expected_counters[source] for source in remap]
        for step in range(3, 8):
            assert draw((0, 1, 2), f"after_migration_{step}") == [baseline[step][source] for source in order]

        initialize()
        for step in range(3):
            assert draw((0, 1, 2), f"before_pause_{step}") == baseline[step]
        for step in range(3, 6):
            assert draw((0,), f"only_slot0_{step}")[0] == baseline[step][0]
        resumed = draw((0, 1, 2), "resume_inactive_slots")
        assert resumed == [baseline[6][0], baseline[3][1], baseline[3][2]], ("Inactive request progression", resumed)

        if args.top_k > 32:
            schedule = [
                ((0, 1, 2), [32, 20, 32]),
                ((0, 1, 2), [40, 20, 1]),
                ((1, 2), [40, 20, 1]),  # dormant k40 must not force width64 or reach native32
                ((0, 1, 2), [40, 20, 1]),
                ((0, 1, 2), [32, 20, 32]),
            ]
            streams = []
            for repeat in range(2):
                initialize()
                stream = []
                for step, (slots, ks) in enumerate(schedule):
                    gen.set_sampling_params(SamplingParams(temperature=[1.3] * 3, top_k=ks, top_p=[0.95] * 3))
                    stream.append(draw(slots, f"family_switch_{repeat}_{step}"))
                    expected_width = 64 if any(ks[slot] > 32 for slot in slots) else 32
                    assert gen.sampler.tt_sampling.candidate_width == expected_width
                    uploads_before = gen._sampling_configuration
                    gen._configure_sampling(slots)
                    assert gen._sampling_configuration is uploads_before, "unchanged metadata rebuilt parameters"
                    if expected_width == 32:
                        effective = ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.k_tensor)[0])
                        assert effective.to(torch.int64).max().item() <= 32, "inactive k40 reached native32"
                streams.append(stream)
            assert streams[0] == streams[1], ("family switch reset/replay reproducibility", streams)
            gen.set_sampling_params(params)

        # Exercise the public prefill first-token API, including plugin-like
        # duck-typed parameters and sparse/reordered empty_slots placement.
        prompt = torch.arange(29, dtype=torch.int64).reshape(1, 29).repeat(2, 1)
        first_params = SimpleNamespace(
            temperature=[1.3, 1.3],
            top_k=[args.top_k, args.top_k],
            top_p=[0.95, 0.95],
            seed=[request_seeds[0], request_seeds[1]],
            num_logprobs=None,
        )
        for destinations in ((2, 0), (0, 2)):
            initialize()
            tokens = gen.prefill_forward(
                prompt,
                page_table=None,
                kv_cache=None,
                prompt_lens=[29, 29],
                empty_slots=destinations,
                sampling_params=first_params,
            )
            assert tokens.shape == (2, 1)
            assert tokens[:, 0].tolist() == baseline[0][:2], (
                "Prefill first-token seed",
                destinations,
                tokens,
                baseline[0],
            )
            for state in ttnn.get_device_tensors(gen.seed_state.state):
                counters = ttnn.to_torch(state).to(torch.int64)[:, 2].tolist()
                assert [counters[slot] for slot in destinations] == [30, 30]
                assert counters[1] == 29, "Prefill changed dormant seed counter"
            for value in ttnn.get_device_tensors(gen.decode_tokens):
                assert ttnn.to_torch(value).flatten()[1].item() == 1001, "Prefill changed dormant token"
            report["checks"].append(
                {
                    "name": "prefill_first_token",
                    "empty_slots": list(destinations),
                    "tokens": tokens.tolist(),
                    "forward_chunk": "synthetic immutable logits",
                }
            )
        initialize()
        gen.prefill_forward(
            prompt, page_table=None, kv_cache=None, prompt_lens=[29, 29], empty_slots=(2, 0), return_device=True
        )
        for state in ttnn.get_device_tensors(gen.seed_state.state):
            assert ttnn.to_torch(state).to(torch.int64)[:, 2].tolist() == [29] * 32, "Intermediate prefill consumed RNG"
        report["checks"].append(
            {"name": "prefill_without_sampling_does_not_draw", "forward_chunk": "synthetic immutable logits"}
        )
        report.update(passed=True, captures=gen.counters["captures"], sampling_replays=gen.counters["sampling_replays"])
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if gen is not None:
                for trace in list(gen.sampling_traces.values()) + list(gen.prefill_traces.values()):
                    ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
