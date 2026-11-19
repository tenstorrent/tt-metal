# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare synchronous and barrier-protected async split-sampling gathers.

Run serially under an external timeout, for example::

    timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_sampling_collective_policy \
        --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/sampling_ccl_v1.json

This test-local adapter is deliberately not a production policy. It borrows the
sampler cases/configuration from profile_full_model_terminal.sampler_ccl_smoke,
but isolates the CCL change, keeps force-argmax disabled, and adds sampled modes.
Run watcher/trace tracking separately from clean timing. No TT device is opened
until main runs. The caller must serialize this job with every other device job.
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params


class BarrierSamplingCollectives(Collectives):
    """Experimental sampler-only async adapter with common TT_CCL ownership.

    All semaphore objects are allocated by prepare_sampling_collectives BEFORE
    capture and owned until every trace has been released. The shared TT_CCL
    cycles its two independent AG/barrier sets in call order; this probe uses
    one CQ and no concurrent submitters. Values and indices are separate calls.

    No persistent_output_buffer is passed: that factory option disables the
    inter-invocation barrier and can expose stale results under fast replay.
    A fresh eager/capture allocation is owned by the caller. Common TTSampling
    deallocates its temporary gathers after consumption, exactly as with the
    synchronous path. Trace capture freezes those temporary addresses in trace
    storage; replay does not invoke this Python allocation path. The adapter
    neither caches a freed output nor independently deallocates caller tensors.
    Inputs/outputs, sampler state and semaphores outlive all captured traces.
    """

    def __init__(self, device):
        super().__init__(device)
        self.calls = []
        self.prepare_sampling_collectives()

    def line_all_gather(self, tensor, *, dim, cluster_axis, memory_config, num_links, buffer_key=None):
        self.calls.append(
            dict(
                buffer_key=buffer_key,
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                dim=dim,
                cluster_axis=cluster_axis,
                num_links=num_links,
            )
        )
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=self.get_and_cycle_ag_semaphore_handles(cluster_axis),
            barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            cluster_axis=cluster_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=memory_config,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh --output path")
    if min(args.iterations, args.repeats) < 1:
        parser.error("Iteration counts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    vocab, batch = 248320, 32
    tracked = trace_allocation_tracker.TRACE_ALLOC_TRACKING
    diagnostics = tracked or os.environ.get("TT_METAL_WATCHER") or os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    report = dict(
        passed=False,
        scope="sampler-only CCL policy; no model or serving latency claim",
        vocab=vocab,
        logical_users=1,
        physical_rows=batch,
        mesh=[1, 4],
        regime="serial",
        force_argmax=False,
        trace_tracking=tracked,
        checks=[],
        rank_order_checks=[],
        timing={},
        ownership=BarrierSamplingCollectives.__doc__,
        environment={
            key: os.environ.get(key)
            for key in (
                "TT_VISIBLE_DEVICES",
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
            )
        },
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(name):
        report["phase"] = name
        save()
        print(name, flush=True)

    def read(tensor):
        return [ttnn.to_torch(rank).clone() for rank in ttnn.get_device_tensors(tensor)]

    mesh = None
    traces, samplers = {}, {}
    try:
        phase("mesh_open")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        sync_ccl = Collectives(mesh)
        sync_ccl.prepare_sampling_collectives()
        async_ccl = BarrierSamplingCollectives(mesh)
        ccls = {"sync": sync_ccl, "async_barrier": async_ccl}

        def upload(value, dtype, *, sharded=True, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3) if sharded else ttnn.ReplicateTensorToMesh(mesh),
            )

        phase("direct_rank_order_and_dtype_checks")
        # Exactly representable distinct payloads expose swapped rank blocks.
        # uint16 is the local top-k index dtype at shard width 62080; uint32
        # also covers the common sampler's wider-index variant.
        for dtype, torch_dtype in (
            (ttnn.bfloat16, torch.bfloat16),
            (ttnn.uint16, torch.int32),
            (ttnn.uint32, torch.int32),
        ):
            host = torch.arange(128).reshape(1, 1, 1, 128).expand(1, 1, batch, 128).to(torch_dtype).contiguous()
            tensor = upload(host, dtype)
            outputs = [
                ttnn.all_gather(tensor, dim=3, cluster_axis=None, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                async_ccl.line_all_gather(
                    tensor,
                    dim=3,
                    cluster_axis=None,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    num_links=1,
                    buffer_key="DIRECT_DTYPE_ORDER",
                ),
            ]
            for output in outputs:
                assert output.dtype == dtype
                assert all(torch.equal(value.to(host.dtype), host) for value in read(output))
                ttnn.deallocate(output)
            ttnn.deallocate(tensor)
            report["rank_order_checks"].append(dict(dtype=str(dtype), passed=True))
        phase("allocate_all_inputs_and_samplers")
        hosts, labels = [], []
        for rank in range(4):
            value = torch.full((1, 1, batch, vocab), -32.0, dtype=torch.bfloat16)
            for row in range(batch):
                value[0, 0, row, rank * (vocab // 4) + 13 + row * 97] = 8.0
            hosts.append(value)
            labels.append(f"unique_rank{rank}")
        for distributed in (True, False):
            value = torch.full_like(hosts[0], -32.0)
            for row in range(batch):
                indices = (
                    (torch.arange(65) * (vocab // 65) + row * 7) % vocab
                    if distributed
                    else (row % 4) * (vocab // 4) + 13 + torch.arange(65)
                )
                value[0, 0, row, indices] = 8.0
            hosts.append(value)
            labels.append("65_distributed_ties" if distributed else "65_local_ties")
        inputs = [upload(value, ttnn.bfloat16) for value in hosts]
        logits = upload(hosts[0], ttnn.bfloat16)
        config = SimpleNamespace(
            vocab_size=vocab,
            padded_vocab_size=vocab,
            max_batch_size=batch,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": False,
                    "num_links": 1,
                    "topology": ttnn.Topology.Ring,
                }
            },
        )
        modes = {"greedy": (1, 0.0), "topk": (8, 0.0), "topk_topp": (8, 0.8)}
        outputs = {}
        for mode, (k, p) in modes.items():
            params = format_sampling_params(SamplingParams(temperature=1.0, top_k=k, top_p=p), batch)
            for policy, ccl in ccls.items():
                name = f"{policy}/{mode}"
                sampler = SamplingGenerator(args=config, mesh_device=mesh, tt_ccl=ccl)
                sampler.tt_sampling._allow_penalties_sampling = False
                sampler.reset_sampling_params(params)
                assert not sampler.tt_sampling.force_argmax_sampling
                samplers[name] = sampler
                outputs[name] = upload(
                    torch.zeros((1, 1, 1, batch), dtype=torch.int32),
                    ttnn.uint32,
                    sharded=False,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
        # Common manual_seed executes immediately before each draw. Identical
        # explicit device seed inputs yield matched samples without host RNG or
        # SamplingGenerator's per-request seeded eager routing.
        seeds = ttnn.from_torch(
            torch.arange(101, 101 + batch).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
        )
        for sampler in samplers.values():
            ttnn.copy(seeds, sampler.tt_sampling.seeds_tt_tensor)

        def body(name):
            samplers[name]._run_sampling(logits, penalties_on=False, grammar_on=False, tt_out_tok=outputs[name])

        phase("prepare_all_variants_before_capture")
        for name, sampler in samplers.items():
            sampler.precompile(logits, tt_out_tok=outputs[name], all_configs=False, compile_token_update=True)
        for source in inputs:
            ttnn.copy(source, logits)
            for name in samplers:
                body(name)
        ttnn.synchronize_device(mesh)
        cache_entries = mesh.num_program_cache_entries()
        phase("capture_all_prepared_variants")
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
        assert mesh.num_program_cache_entries() == cache_entries, "Late program compile during capture"
        phase("changed_winner_tie_and_sampling_replay")
        order = list(range(len(inputs))) + [0, 5, 4, 3, 0]
        for index in order:
            observed = {}
            for mode in modes:
                for policy in ccls:
                    name = f"{policy}/{mode}"
                    repeats = []
                    for _ in range(2):
                        # Deliberately queue a different input/replay immediately
                        # before the tested one, with no host sync between them.
                        ttnn.copy(inputs[(index + 1) % len(inputs)], logits)
                        ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                        ttnn.copy(inputs[index], logits)
                        ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        ranks = [value.reshape(-1).long() for value in read(outputs[name])]
                        assert all(torch.equal(ranks[0], value) for value in ranks[1:]), name
                        tokens = ranks[0]
                        assert len(tokens) == batch and ((tokens >= 0) & (tokens < vocab)).all()
                        selected = hosts[index][0, 0, torch.arange(batch), tokens]
                        # Background probability is negligible at this 40-logit
                        # margin; all admissible top-k/top-p candidates are maxima.
                        assert torch.equal(selected, hosts[index][0, 0].max(-1).values), name
                        repeats.append(tokens.tolist())
                    assert repeats[0] == repeats[1], f"Fixed device seeds failed deterministic replay: {name}"
                    observed[name] = repeats[0]
                assert observed[f"sync/{mode}"] == observed[f"async_barrier/{mode}"], f"CCL changed sampling: {mode}"
            report["checks"].append(dict(input=labels[index], matched_tokens=observed, passed=True))
            save()
        assert mesh.num_program_cache_entries() == cache_entries
        phase("clean_split_greedy_timing" if not diagnostics else "diagnostic_timing_skipped")
        if not diagnostics:
            ttnn.copy(inputs[3], logits)
            for policy in ccls:
                name = f"{policy}/greedy"
                for _ in range(4):
                    ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples = []
                for _ in range(args.repeats):
                    begin = time.perf_counter()
                    for _ in range(args.iterations):
                        ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - begin) / args.iterations)
                report["timing"][policy] = dict(
                    ms_per_step=samples,
                    iterations=args.iterations,
                    nonblocking_replay=True,
                    per_step_readbacks=0,
                    per_step_input_refreshes=0,
                    boundary="full common split greedy sampler; preexisting logits to persistent tt_out_tok; excludes model and seed orchestration",
                )
        report["async_gather_calls"] = async_ccl.calls
        report["trace_ids"] = {name: str(trace) for name, trace in traces.items()}
        report["program_cache_entries"] = cache_entries
        report["allocation_exemptions"] = []
        # End on replay so allocation tracking checks all prior activity.
        ttnn.execute_trace(mesh, traces["async_barrier/greedy"], cq_id=0, blocking=True)
        report["passed"] = True
        phase("passed")
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        save()
        raise
    finally:
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            for sampler in samplers.values():
                sampler.reset_trace()
            ttnn.close_mesh_device(mesh)
        save()


if __name__ == "__main__":
    main()
