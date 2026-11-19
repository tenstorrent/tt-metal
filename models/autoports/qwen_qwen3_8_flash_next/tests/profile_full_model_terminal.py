# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer terminal attribution and equivalent greedy comparison.

Only layers 0/1/3 are supported: never use this harness to profile all layers.
Restore TT_MESH_GRAPH_DESC_PATH after sourcing lane-env. Run serially, with
watcher disabled. Clean traced timing (profiler and allocation tracker off)::

    TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
      timeout 3600 python -m \
      models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
      --iterations 12 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_timing.json

Per-op attribution uses the exact prepared device bodies BEFORE any captures;
the current checkout's aggregate trace profiler cannot attribute individual
ops within replay. Do not report these eager durations as traced token latency::

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_TRACE_ALLOC_TRACKING=0 \
      TT_METAL_TRACE_ALLOC_TRACEBACKS=0 timeout 3600 python -m tracy -r -p -v \
      --no-web-server --op-support-count 8192 \
      -o models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy \
      -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
      --profile-ops-only --profile-prompt-length 128 --capacity 262144 --iterations 1 \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_ops.json

Use the generated ops_perf_results CSV, preserving its original file, with
installed optimize skill references/perf-report.md (advice stays enabled)::

    tt-perf-report <ops_perf_results.csv> --start-signpost PERF_MODEL \
      --end-signpost PERF_MODEL_END --csv <artifact_dir>/model_perf_report.csv
    tt-perf-report <ops_perf_results.csv> --start-signpost PERF_MODEL \
      --end-signpost PERF_MODEL_END --no-summary > <artifact_dir>/model_perf_report.txt

Repeat those two commands for PERF_SPLIT_GREEDY / PERF_SPLIT_GREEDY_END and
PERF_FORCE_ARGMAX / PERF_FORCE_ARGMAX_END, using distinct output filenames.
The eager model region also contains PERF_TOKEN_ENTRY, PERF_EMBEDDING,
PERF_LAYER_L0, PERF_LAYER_L1, PERF_LAYER_L3, PERF_FINAL_MIXER, PERF_LM_HEAD
and PERF_ROW_ASSEMBLY, each with an _END marker. Weight the complete layers
35/1/12 and each entry/terminal leaf once. PERF_TERMINAL encloses final mixer
and LM head for convenience; do not add it again to its two leaf regions.
Console status from --csv belongs in *.console.log, not the rendered table.
Profile durations and clean timing are separate regimes. No source edits or
custom sampler math are used; force-argmax remains a candidate, not a default.

Before loading model weights, validate the common sampler/CCL contract alone::

    timeout 180 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
      --sampler-smoke-only --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/sampler_ccl_smoke_v1.json

This uses preallocated full-vocabulary inputs, both common greedy strategies,
changing winners on all four ranks, and65 equal maxima both distributed and
within one shard. It is correctness-only;
the force-argmax Linear topology is not a selected model policy.
"""

import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tracy import signpost
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

COMPONENT_REGIONS = {
    "PERF_TOKEN_ENTRY": {"weight": 1, "source": "QwenGenerator._decode_body before QwenModel.embed"},
    "PERF_EMBEDDING": {"weight": 1, "source": "QwenModel.embed (embedding, gather and reshape)"},
    "PERF_LAYER_L0": {"weight": 35, "source": "MultichipDecoder layer0 complete decode_forward"},
    "PERF_LAYER_L1": {"weight": 1, "source": "MultichipDecoder layer1 complete decode_forward including PLE"},
    "PERF_LAYER_L3": {"weight": 12, "source": "MultichipDecoder layer3 complete decode_forward including QSA"},
    "PERF_FINAL_MIXER": {"weight": 1, "source": "FinalMixer.__call__ including distributed norm and mixing"},
    "PERF_LM_HEAD": {"weight": 1, "source": "QwenModel.lm_head including projection layout conversions"},
    "PERF_ROW_ASSEMBLY": {"weight": 1, "source": "QwenGenerator._decode_body after QwenModel.decode_batch"},
}


@contextmanager
def model_component_signposts(gen, warmup):
    """Instrument only measured eager submissions; restore every binding on exit.

    Complete layer methods are wrapped, so the attribution includes attention,
    PLE, routed experts, collectives, residual work and device state advancement.
    No extra device operation, synchronization, read or trace is introduced.
    Warmup executes the identical methods with signposts disabled.
    """
    assert gen.max_batch_size == 1 and gen.factored_decode is None
    assert tuple(gen.model.layer_indices) == (0, 1, 3)
    assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces
    state = {"enabled": False, "calls": 0, "entry_open": False, "assembly_open": False}
    events = []

    def emit(label):
        signpost(label)
        events.append(label)

    def bracket(label, function):
        def wrapped(*args, **kwargs):
            if not state["enabled"]:
                return function(*args, **kwargs)
            emit(label)
            try:
                return function(*args, **kwargs)
            finally:
                emit(label + "_END")

        return wrapped

    original_embed = gen.model.embed
    original_decode_batch = gen.model.decode_batch

    def embed(*args, **kwargs):
        if state["enabled"]:
            assert state["entry_open"], "B1 attribution expects exactly one embedding call"
            emit("PERF_TOKEN_ENTRY_END")
            state["entry_open"] = False
        return bracket("PERF_EMBEDDING", original_embed)(*args, **kwargs)

    def decode_batch(*args, **kwargs):
        result = original_decode_batch(*args, **kwargs)
        if state["enabled"]:
            assert not state["entry_open"] and not state["assembly_open"]
            emit("PERF_ROW_ASSEMBLY")
            state["assembly_open"] = True
        return result

    def operation():
        state["enabled"] = state["calls"] >= warmup
        state["calls"] += 1
        if state["enabled"]:
            emit("PERF_TOKEN_ENTRY")
            state["entry_open"] = True
        try:
            gen._decode_body((0,))
            if state["enabled"]:
                assert not state["entry_open"] and state["assembly_open"]
        finally:
            # Keep failed profiles structurally closed; the driver still records
            # the original exception and never accepts such evidence.
            if state["entry_open"]:
                emit("PERF_TOKEN_ENTRY_END")
            if state["assembly_open"]:
                emit("PERF_ROW_ASSEMBLY_END")
            state.update(enabled=False, entry_open=False, assembly_open=False)

    with ExitStack() as stack:
        stack.enter_context(patch.object(gen.model, "embed", new=embed))
        stack.enter_context(patch.object(gen.model, "decode_batch", new=decode_batch))
        stack.enter_context(patch.object(gen.model, "logits", new=bracket("PERF_TERMINAL", gen.model.logits)))
        for attribute, label in (("final_mixer", "PERF_FINAL_MIXER"), ("lm_head", "PERF_LM_HEAD")):
            stack.enter_context(patch.object(gen.model, attribute, new=bracket(label, getattr(gen.model, attribute))))
        for layer_index, layer in zip(gen.model.layer_indices, gen.model.layers, strict=True):
            stack.enter_context(
                patch.object(layer, "decode_forward", new=bracket(f"PERF_LAYER_L{layer_index}", layer.decode_forward))
            )
        yield operation, events


def sampler_body(gen, sampler):
    """Same wrapper operations as QwenGenerator._sample; common sampler math."""
    gen.seed_state.set_active_slots((0,))
    ttnn.copy(gen.decode_tokens, gen.saved_tokens)
    gen.seed_state.forward()
    sampler._run_sampling(gen.sample_logits, penalties_on=False, grammar_on=False, tt_out_tok=gen.decode_tokens)
    gen._restore_inactive_tokens((0,))
    ttnn.copy(gen.seed_skip, sampler.tt_sampling.seeds_tt_tensor)


def prepare_eager_profile(gen, samplers):
    """Warm only measured bodies; this path never captures or serves requests.

    Full generator.prepare() also warms six prefill buckets and serving state
    transitions that this decode-only attribution does not measure. Earlier
    broad warmups exhausted Tracy's source-location table. Clean traced timing
    below still uses the complete current preparation contract.
    """
    assert not gen.prepared
    assert gen.max_batch_size == 1 and gen.factored_decode is None
    assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces
    gen._check_cache_setup()
    gen.inactive_logits = ttnn.full(
        (1, 1, 1, gen.model.config.vocab_size // 4),
        -100.0,
        device=gen.mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    gen.seed_state.prepare()
    gen.token_feedback.prepare()
    gen.reset()
    ttnn.synchronize_device(gen.mesh)
    ttnn.ReadDeviceProfiler(gen.mesh)
    gen._decode_body((0,))
    ttnn.synchronize_device(gen.mesh)
    ttnn.ReadDeviceProfiler(gen.mesh)
    for sampler in samplers:
        sampler.precompile(
            gen.sample_logits, tt_out_tok=gen.decode_tokens, all_configs=False, compile_token_update=True
        )
        sampler_body(gen, sampler)
        ttnn.synchronize_device(gen.mesh)
        ttnn.ReadDeviceProfiler(gen.mesh)
    assert not gen.prepared, "Eager-only warmup must not advertise full generator preparation"


def time_region(mesh, operation, iterations, warmup=2, label=None, profiler=False):
    for _ in range(warmup):
        operation()
        if profiler:
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
    ttnn.synchronize_device(mesh)
    if profiler:
        ttnn.ReadDeviceProfiler(mesh)
    if label:
        signpost(label)
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    ttnn.synchronize_device(mesh)
    elapsed = time.perf_counter() - started
    if label:
        signpost(label + "_END")
    if profiler:
        ttnn.ReadDeviceProfiler(mesh)
    return {
        "iterations": iterations,
        "warmup_iterations": warmup,
        "host_total_ms": elapsed * 1000,
        "host_ms_per_iteration": elapsed * 1000 / iterations,
        "synchronization": "One device synchronization at both measured window boundaries; read ops also block",
    }


def sampler_ccl_smoke(mesh, report):
    """No weights: prepare all tensors and bodies before the first capture."""
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives

    vocab, batch = 248320, 32
    ccl = Collectives(mesh)
    ccl.prepare_sampling_collectives()
    config = SimpleNamespace(
        vocab_size=vocab,
        padded_vocab_size=vocab,
        max_batch_size=batch,
        max_top_k=32,
        cluster_shape=(1, 4),
        sampling_all_gather_axis=1,
        model_config={
            "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 1, "topology": ttnn.Topology.Ring}
        },
    )
    hosts = []
    labels = []
    kinds = []
    for rank in range(4):
        values = torch.full((1, 1, batch, vocab), -32.0, dtype=torch.bfloat16)
        for row in range(batch):
            values[0, 0, row, rank * (vocab // 4) + 13 + row * 97] = 8.0
        hosts.append(values)
        labels.append(f"unique_winner_rank{rank}")
        kinds.append("unique")
    values = torch.full_like(hosts[0], -32.0)
    for row in range(batch):
        values[0, 0, row, (torch.arange(65) * (vocab // 65) + row * 7) % vocab] = 8.0
    hosts.append(values)
    labels.append("65_equal_global_maxima_distributed")
    kinds.append("distributed_ties")
    values = torch.full_like(hosts[0], -32.0)
    for row in range(batch):
        shard = row % 4
        values[0, 0, row, shard * (vocab // 4) + 13 + torch.arange(65)] = 8.0
    hosts.append(values)
    labels.append("65_equal_maxima_within_one_shard")
    kinds.append("local_shard_ties")

    def upload(value):
        return ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )

    inputs = [upload(value) for value in hosts]
    logits = upload(hosts[0])
    outputs = {}
    samplers = {}
    traces = {}
    greedy = format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), batch)
    report["scope"] = "No-model common-sampler CCL smoke; not sampler selection or model performance"
    report["regime"] = "no_model_sampler_ccl_correctness"
    report["layers"] = []
    report["force_argmax_policy"] = "Common1x4 candidate uses Linear topology; selected sampler remains unchanged"
    report["checks"] = []
    try:
        for name, force in (("split_greedy", False), ("force_argmax", True)):
            args = copy.deepcopy(config)
            args.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = force
            sampler = SamplingGenerator(args=args, mesh_device=mesh, tt_ccl=ccl)
            sampler.tt_sampling._allow_penalties_sampling = False
            sampler.reset_sampling_params(greedy)
            assert sampler.tt_sampling.force_argmax_sampling == force
            samplers[name] = sampler
            outputs[name] = ttnn.from_torch(
                torch.zeros(1, 1, 1, batch, dtype=torch.int32),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            sampler.precompile(logits, tt_out_tok=outputs[name], compile_token_update=True)

        def body(name):
            samplers[name]._run_sampling(logits, penalties_on=False, grammar_on=False, tt_out_tok=outputs[name])

        for source in inputs:
            ttnn.copy(source, logits)
            for name in samplers:
                body(name)
        ttnn.synchronize_device(mesh)
        cache_entries = mesh.num_program_cache_entries()
        for name in samplers:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                body(name)
            except BaseException:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.release_trace(mesh, trace)
                raise
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[name] = trace
        assert mesh.num_program_cache_entries() == cache_entries, "Capture compiled an unwarmed sampler program"
        # Only copies between precreated device tensors, replay, and token reads
        # occur below. No device allocations are permitted after first capture.
        replay_labels = labels + [
            "unique_winner_rank0",
            "65_equal_maxima_within_one_shard",
            "65_equal_global_maxima_distributed",
            "unique_winner_rank3",
        ]
        for label in replay_labels:
            index = labels.index(label)
            ttnn.copy(inputs[index], logits)
            observed = {}
            for name in samplers:
                repeats = []
                for _ in range(2):
                    ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    ranks = [
                        ttnn.to_torch(value).reshape(-1).long() for value in ttnn.get_device_tensors(outputs[name])
                    ]
                    assert all(torch.equal(ranks[0], rank) for rank in ranks[1:]), "Sampler tokens differ across ranks"
                    tokens = ranks[0]
                    assert len(tokens) == batch and ((tokens >= 0) & (tokens < vocab)).all()
                    selected = hosts[index][0, 0, torch.arange(batch), tokens]
                    assert torch.equal(
                        selected, hosts[index][0, 0].max(dim=-1).values
                    ), "Selected token is not globally greedy"
                    repeats.append(tokens.tolist())
                assert repeats[0] == repeats[1], "Tie policy is not deterministic on identical resident logits"
                observed[name] = repeats[0]
            if kinds[index] == "unique":
                assert observed["split_greedy"] == observed["force_argmax"], "Unique maxima disagree"
            report["checks"].append(
                {
                    "input": labels[index],
                    "kind": kinds[index],
                    "global_max_value_validated": True,
                    "deterministic_replay": True,
                    "tokens": observed,
                    "tie_choices_equal": observed["split_greedy"] == observed["force_argmax"],
                }
            )
        assert mesh.num_program_cache_entries() == cache_entries
        report["program_cache_entries_after_warmup"] = cache_entries
        report["all_inputs_preallocated_and_both_samplers_warmed_before_capture"] = True
        report["passed"] = True
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        for sampler in samplers.values():
            sampler.reset_trace()


def profile_capacity(value):
    """Parse the physical cache capacity without opening devices."""
    capacity = int(value)
    if not 128 <= capacity <= 262144 or capacity % 128:
        raise argparse.ArgumentTypeError("Capacity must be a multiple of 128 in [128, 262144]")
    return capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--capacity", type=profile_capacity, default=128)
    parser.add_argument("--profile-ops-only", action="store_true")
    parser.add_argument(
        "--profile-prompt-length",
        type=int,
        default=128,
        help="Natural prefix before eager-profile warmups; ignored by clean timing and sampler smoke",
    )
    parser.add_argument("--sampler-smoke-only", action="store_true")
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    parser.add_argument("--trace-region-size", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output artifact; preserve earlier capture evidence")
    if args.sampler_smoke_only and args.profile_ops_only:
        parser.error("Sampler smoke is correctness-only; disable the per-op profiler")
    if not 1 <= args.iterations <= 32 or not 0 <= args.warmup <= 8:
        parser.error("Use 1..32 measured iterations and 0..8 warmups")
    if args.profile_ops_only and args.iterations > 2:
        parser.error("Per-op capture is bounded to at most two reduced-stack iterations")
    if args.profile_ops_only and not 1 <= args.profile_prompt_length <= min(
        1024, args.capacity - args.warmup - args.iterations
    ):
        parser.error("Profile prefix must be1..1024 and leave capacity for warmups and measurement")
    started = time.monotonic()
    report = {
        "passed": False,
        "regime": "serial_reduced_eager_op_profile" if args.profile_ops_only else "serial_reduced_traced_clean_timing",
        "scope": "Debug/profile only; not all-layer acceptance or advertised throughput",
        "layers": [0, 1, 3],
        "batch_size": 1,
        "physical_sampler_rows": 32,
        "capacity": args.capacity,
        "greedy_semantics": {"top_k": 1, "top_p": 0.0, "temperature": 1.0, "physical_max_top_k": 32},
        "source_boundaries": {
            "model_trace": "tt/generator.py:QwenGenerator._decode_body",
            "embedding": "tt/model.py:QwenModel.embed",
            "decoder_stack": "tt/model.py:QwenModel.decode_batch -> tt/multichip_decoder.py",
            "final_norm_and_mix": "tt/model.py:FinalMixer.__call__",
            "vocab_sharded_lm_head": "tt/model.py:QwenModel.logits",
            "logits_row_assembly": "tt/generator.py:QwenGenerator._decode_body",
            "sampling_wrapper": "tt/generator.py:QwenGenerator._sample",
            "common_sampler": "models/common/sampling/tt_sampling.py:TTSampling",
            "caller_token_read": "tt/generator.py:QwenGenerator._read_tokens",
        },
        "strategies": {},
        "correctness": [],
        "expected_signposts": (
            [
                "PERF_MODEL",
                "PERF_MODEL_END",
                *(marker for label in (*COMPONENT_REGIONS, "PERF_TERMINAL") for marker in (label, label + "_END")),
                "PERF_SPLIT_GREEDY",
                "PERF_SPLIT_GREEDY_END",
                "PERF_FORCE_ARGMAX",
                "PERF_FORCE_ARGMAX_END",
            ]
            if args.profile_ops_only
            else []
        ),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[4])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                *sorted(
                    path
                    for path in (Path(__file__).resolve().parents[1] / "tt").rglob("*")
                    if path.suffix in (".py", ".cpp", ".h", ".hpp")
                ),
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "run_full_model_acceptance.py",
                Path(__file__).resolve().parents[4] / "models/common/sampling/generator.py",
                Path(__file__).resolve().parents[4] / "models/common/sampling/tt_sampling.py",
                Path(__file__).resolve().parents[4] / "models/common/tt_ccl.py",
            )
        },
    }
    mesh = gen = force_sampler = None
    split_sampler = None
    phase = "setup"
    try:
        assert not trace_allocation_tracker.TRACE_ALLOC_TRACKING, "Separate allocation tracking from measured timing"
        assert not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS, "Disable allocation diagnostics for timing"
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore accepted mesh descriptor after lane-env"
        assert (
            "TT_METAL_WATCHER" not in os.environ
        ), "TT_METAL_WATCHER must be unset; even an empty value or0 enables watcher"
        perf_counter_mode = int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0")
        profiler_on = os.environ.get("TT_METAL_DEVICE_PROFILER", "").startswith("1") or perf_counter_mode != 0
        assert profiler_on == args.profile_ops_only, "Enable device profiler only for --profile-ops-only"
        if args.profile_ops_only:
            assert (
                int(os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "0")) >= 8192
            ), "Use tracy --op-support-count 8192: a measured reduced decode exceeds the default1000-op buffer"
        assert os.environ.get("TT_METAL_STREAMING_PROFILER") in (
            None,
            "",
            "0",
        ), "This harness uses the DRAM per-op profiler; streaming profiling must be disabled"
        report["profiler_environment"] = {
            name: os.environ.get(name)
            for name in (
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILE_PERF_COUNTERS",
                "TT_METAL_STREAMING_PROFILER",
                "TT_METAL_WATCHER",
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT",
            )
        }
        assert os.environ.get("TT_METAL_TRACE_PROFILER") in (
            None,
            "",
            "0",
        ), "This profile uses per-op eager attribution"
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        if args.sampler_smoke_only:
            phase = "no_model_sampler_ccl_smoke"
            sampler_ccl_smoke(mesh, report)
            return
        options = {"layer_indices": (0, 1, 3), "max_batch_size": 1, "max_seq_len": args.capacity}
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator("models/autoports/qwen_qwen3_8_flash_next", mesh, **options)
        gen.model.ccl.prepare_sampling_collectives()
        split_sampler = gen.sampler
        # Both independently owned sampler states exist before any capture.
        force_args = copy.deepcopy(split_sampler.args)
        force_args.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = True
        force_sampler = SamplingGenerator(args=force_args, mesh_device=mesh, tt_ccl=gen.model.ccl)
        force_sampler.tt_sampling._allow_penalties_sampling = False
        greedy = format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), 32)
        for sampler in (split_sampler, force_sampler):
            sampler.reset_sampling_params(greedy)
        assert not split_sampler.tt_sampling.force_argmax_sampling, "Split candidate unexpectedly uses force-argmax"
        assert force_sampler.tt_sampling.force_argmax_sampling, "Force-argmax candidate did not select its common path"
        if args.profile_ops_only:
            prepare_eager_profile(gen, (split_sampler, force_sampler))
            report[
                "preparation_scope"
            ] = "Eager B1 decode plus exact two sampler bodies; no prefill buckets or captures"
        else:
            gen.prepare()
            force_sampler.precompile(gen.sample_logits, tt_out_tok=gen.decode_tokens, compile_token_update=True)
            # Warm the exact external wrapper as well as the common operations.
            for sampler in (split_sampler, force_sampler):
                sampler_body(gen, sampler)
            report["preparation_scope"] = "Full generator preparation for clean traced timing"
        assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces, "Prepare unexpectedly captured"
        gen.reset()
        force_sampler.reset_output_state()
        force_sampler.seed_manager.reset_seed([None] * 32, list(range(32)))
        force_sampler.seed_manager.get_new_values()
        ttnn.synchronize_device(mesh)
        report["prepared_seconds"] = time.monotonic() - started
        report["program_cache_entries_after_prepare"] = mesh.num_program_cache_entries()
        report["samplers_allocated_and_warmed_before_capture"] = True
        candidates = {"split_greedy": split_sampler, "force_argmax": force_sampler}

        if args.profile_ops_only:
            from models.autoports.qwen_qwen3_8_flash_next.tests import run_full_model_acceptance as acceptance

            phase = "eager_natural_prefix"
            model_dir = Path(__file__).resolve().parents[1]
            prompt = acceptance.natural_prompt(gen, model_dir, args.profile_prompt_length + 1)
            offset = 0
            # Native prefill construction, all outside signposts and before any
            # capture. Drain each full-model chunk so source/op buffers cannot
            # silently include a whole128-token prefill without a boundary.
            while offset < args.profile_prompt_length:
                count = min(32, 1 << ((args.profile_prompt_length - offset).bit_length() - 1))
                tokens = torch.tensor([prompt[offset : offset + count]], dtype=torch.int32)
                words = torch.zeros(1, count, 16, dtype=torch.int32)
                words[..., 0], words[..., 1] = tokens, 1
                gen._refresh(gen.prefill_inputs[count], tokens)
                gen._refresh(gen.prefill_words[count], words)
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
                gen._prefill_body(count)
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
                offset += count
            gen.caches[0].position = args.profile_prompt_length
            prompt_path = model_dir / "doc/full_model/hf_reference/prompt.txt"
            report["profile_prefix"] = {
                "logical_length": args.profile_prompt_length,
                "token_ids": list(map(int, prompt[:-1])),
                "source": str(prompt_path),
                "source_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                "teacher_forced_decode_token": int(prompt[-1]),
                "positions_before_decode_warmups": acceptance.qsa_positions(gen, args.profile_prompt_length),
                "measured_start_position": args.profile_prompt_length + args.warmup,
                "measured_end_position": args.profile_prompt_length + args.warmup + args.iterations,
                "prefill_outside_signposts": True,
                "drained_after_each_native_chunk": True,
            }
            phase = "eager_operation_attribution"
            # Repeated next natural token, with a real natural-prefix state.
            # It intentionally runs before any trace exists, so eager temporary
            # allocations cannot clobber a live trace or be mistaken for replay.
            seed = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            seed[..., 0] = prompt[-1]
            gen._refresh(gen.decode_tokens, seed)
            with model_component_signposts(gen, args.warmup) as (model_body, component_events):
                report["model_ops"] = time_region(
                    mesh,
                    model_body,
                    args.iterations,
                    args.warmup,
                    label="PERF_MODEL",
                    profiler=True,
                )
            gen.caches[0].position = args.profile_prompt_length + args.warmup + args.iterations
            report["profile_prefix"]["positions_after_measured_decode"] = acceptance.qsa_positions(
                gen, gen.caches[0].position
            )
            for label in (*COMPONENT_REGIONS, "PERF_TERMINAL"):
                assert component_events.count(label) == args.iterations, f"Missing/repeated component start: {label}"
                assert (
                    component_events.count(label + "_END") == args.iterations
                ), f"Missing/repeated component end: {label}"
            assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces
            report["component_attribution"] = {
                "leaf_regions": COMPONENT_REGIONS,
                "source_order_signposts": component_events,
                "parent_region": {"PERF_TERMINAL": ["PERF_FINAL_MIXER", "PERF_LM_HEAD"]},
                "aggregation": "Sum leaf device durations per iteration; weight L0/L1/L3 by35/1/12 and other leaves once; add split sampler once",
                "overlap_rule": "PERF_MODEL and PERF_TERMINAL are enclosing regions, never additive to their leaves",
                "warmup_signposts": False,
                "temporary_bindings_restored": True,
                "added_device_operations": 0,
                "scope": "Eager per-op attribution; weighted reduced-to-full estimate, not all-layer traced wall time",
            }
            frozen = gen._read_logits(gen.sample_logits)[0]
            expected = int(frozen.argmax())
            maximum = float(frozen.max())
            for name, sampler in candidates.items():
                phase = "eager_" + name
                report["strategies"][name] = time_region(
                    mesh,
                    lambda sampler=sampler: sampler_body(gen, sampler),
                    args.iterations,
                    args.warmup,
                    label="PERF_" + name.upper(),
                    profiler=True,
                )
                actual = int(gen._read_tokens()[0])
                assert (
                    0 <= actual < frozen.numel() and float(frozen[actual]) == maximum
                ), f"{name} is not greedy on the shared real logits"
                report["correctness"].append(
                    {
                        "strategy": name,
                        "reference_first_argmax": expected,
                        "actual": actual,
                        "selected_global_maximum": True,
                        "maximum_logit": maximum,
                        "tie_choice_matches_first_argmax": actual == expected,
                    }
                )
            report["profile_contract"] = {
                "no_live_traces": not gen.traces and not gen.prefill_traces and not gen.sampling_traces,
                "op_profile_not_trace_latency": True,
                "common_sampler_math": True,
                "frozen_real_logits_shared_by_both_candidates": True,
                "advice_required": True,
            }
            report["passed"] = True
            return

        prompt = [785, 13458, 13, 785, 13, 13458, 785, 13, 785, 13458, 13, 785, 13, 13458, 785, 13, 785]

        def reset_request(tokens, sampler):
            gen.sampler = sampler
            gen.reset()
            gen.prefill_forward(
                torch.tensor([tokens]),
                page_table=None,
                kv_cache=[gen.caches[0]],
                prompt_lens=[len(tokens)],
                slots=(0,),
                return_device=True,
            )
            gen._sample((0,))
            ttnn.synchronize_device(mesh)

        phase = "capture_and_same_logits_correctness"
        # Capture both sampler modes against the same persistent tensor. The
        # generator keys them separately; sampler flags are never flipped.
        model_trace = gen._capture((0,))
        for tokens in (prompt[:3], prompt[:7], prompt):
            reset_request(tokens, split_sampler)
            frozen = gen._read_logits(gen.sample_logits)[0]
            expected = int(frozen.argmax())
            observed = {}
            for name, sampler in candidates.items():
                gen.sampler = sampler
                gen._sample((0,))
                observed[name] = int(gen._read_tokens()[0])
                assert observed[name] == expected, f"{name} is not greedy at prompt length {len(tokens)}"
            report["correctness"].append({"prompt_length": len(tokens), "expected": expected, "tokens": observed})

        captured_ids = {str(k): str(v) for k, v in gen.sampling_traces.items()}
        assert len(captured_ids) == 2, "Expected exactly two fixed greedy sampling signatures"

        def model_replay():
            ttnn.execute_trace(mesh, model_trace, cq_id=0, blocking=False)
            gen.counters["model_replays"] += 1

        outputs = {}
        for name, sampler in candidates.items():
            phase = "timing_" + name
            gen.sampler = sampler
            rows = {}
            reset_request(prompt, sampler)
            rows["model_trace_logits_only"] = time_region(mesh, model_replay, args.iterations, args.warmup)
            reset_request(prompt, sampler)
            rows["sampling_trace_frozen_logits"] = time_region(
                mesh, lambda: gen._sample((0,)), args.iterations, args.warmup
            )
            rows["caller_token_read_frozen_output"] = time_region(mesh, gen._read_tokens, args.iterations, args.warmup)
            reset_request(prompt, sampler)
            before = dict(gen.counters)
            tokens_out = []

            def token_out():
                model_replay()
                gen._sample((0,))
                tokens_out.append(int(gen._read_tokens()[0]))

            rows["token_out_end_to_end"] = time_region(mesh, token_out, args.iterations, args.warmup)
            for field in ("token_refreshes", "position_refreshes", "page_table_refreshes"):
                assert gen.counters[field] == before[field], f"{name}: host feedback in timed token-out loop: {field}"
            rows["counter_delta"] = {key: gen.counters[key] - before[key] for key in before}
            outputs[name] = tokens_out
            report["strategies"][name] = rows
            assert {str(k): str(v) for k, v in gen.sampling_traces.items()} == captured_ids, "Sampling traces changed"

        assert outputs["split_greedy"] == outputs["force_argmax"], "Equivalent greedy paths diverged on free feedback"
        report["feedback_tokens"] = outputs
        report["sampling_trace_ids"] = captured_ids
        report["model_trace_id"] = str(model_trace)
        report["workload"] = {
            "prompt_length": len(prompt),
            "warmup_decode": args.warmup,
            "measured_decode": args.iterations,
        }
        elapsed = {
            name: rows["token_out_end_to_end"]["host_ms_per_iteration"] for name, rows in report["strategies"].items()
        }
        report["fastest_measured_candidate"] = min(elapsed, key=elapsed.get)
        report["candidate_selection_changes_default"] = False
        report["sampler_dominance"] = {}
        for name, rows in report["strategies"].items():
            sampling = rows["sampling_trace_frozen_logits"]["host_ms_per_iteration"]
            model = rows["model_trace_logits_only"]["host_ms_per_iteration"]
            share = sampling / (sampling + model)
            report["sampler_dominance"][name] = {
                "isolated_device_submission_share": share,
                "dominant_over_reduced_model": sampling >= model,
                "requires_terminal_investigation": sampling >= model,
                "scope": "Reduced-stack screen; final all-layer share needs all-layer clean timing and per-op attribution",
            }
        report["limitations"] = [
            "Host-window timings include trace submission and an amortized boundary synchronization.",
            "Isolated model and sampler timings are diagnostic windows, not additive device-kernel times.",
            "Correctness full-logit reads and host argmax occur only outside measured or feedback paths.",
            "A dominant sampler requires LM-head/sampling work; this script does not waive it or select a new default.",
        ]
        report["program_cache_entries_final"] = mesh.num_program_cache_entries()
        report["passed"] = True
    except BaseException as exc:
        report["failed_phase"] = phase
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            if split_sampler is not None:
                gen.sampler = split_sampler
            gen.close()
        if force_sampler is not None:
            force_sampler.reset_trace()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print(json.dumps({"passed": report["passed"], "phase": phase, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
