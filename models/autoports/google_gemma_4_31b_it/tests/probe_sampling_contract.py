# SPDX-License-Identifier: Apache-2.0
"""Synthetic TP4 sampler comparison; never a full-model performance result.

Both paths are semantically greedy: physical local top-k=32, semantic k=1,
versus force-argmax. Force-argmax pads logical rows to32 so its output matches
our persistent UINT32 [1,1,1,32] feedback buffer. TTSampling split-greedy
also pads B8/16: its fixed32 index/greedy buffers cannot broadcast against8/16
logical rows. B1 retains its valid singleton-broadcast path. Padding is measured.
Use allocation tracking for acceptance; --timing requires tracking disabled.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch

import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.common.sampling.tt_sampling import TTSampling

VOCAB = 262144


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32, 8, 16])
    parser.add_argument("--library", choices=["tt", "sampling1d"], default="tt")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--semantic-greedy",
        action="store_true",
        help="Require a maximum-valued token; still report lowest-index mismatches",
    )
    parser.add_argument(
        "--native-small-batch", action="store_true", help="Reproduce unadapted TTSampling B8/16 broadcast failure"
    )
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--cases", nargs="+", default=["unique", "cross_shard_tie", "many_ties", "negative"])
    args = parser.parse_args()
    if args.timing and os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0") != "0":
        raise ValueError("Measure timing separately with allocation tracking disabled")
    assert all(b in (1, 8, 16, 32) for b in args.batches)
    torch.set_num_threads(16)
    torch.manual_seed(22)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
    failures = []
    try:
        for batch in args.batches:
            failures.extend(run(mesh, batch, args))
    finally:
        ttnn.close_mesh_device(mesh)
    assert not failures, failures


def run(mesh, batch, args):
    ccl = TT_CCL(mesh)
    base = dict(vocab_size=VOCAB, max_batch_size=32, max_top_k=32, cluster_shape=tuple(mesh.shape))
    samplers = {}
    for mode in ("split_greedy", "force_argmax"):
        config = SimpleNamespace(
            **base,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": mode == "force_argmax",
                    "num_links": 1,
                    "topology": ttnn.Topology.Linear,
                }
            },
        )
        if args.library == "tt":
            sampler = TTSampling(mesh, ccl, config)
            assert sampler.force_argmax_sampling == (mode == "force_argmax")
        else:
            from models.common.modules.sampling.sampling_1d import Sampling1D

            sampler = Sampling1D.from_model_args(mesh, ccl, config)
            sampler.load_device_buffers()
        samplers[mode] = sampler

    def upload(value, *, shard=False, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value,
            device=mesh,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3) if shard else ttnn.ReplicateTensorToMesh(mesh),
        )

    logits = upload(torch.zeros(1, 1, batch, VOCAB, dtype=torch.bfloat16), shard=True)
    tokens = upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Stateless Sampling1D takes params explicitly; TTSampling owns the same defaults.
    k = upload(torch.ones(32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p = upload(torch.zeros(32), layout=ttnn.ROW_MAJOR_LAYOUT)
    temp = upload(torch.ones(32), layout=ttnn.ROW_MAJOR_LAYOUT)

    def needs_padding(mode):
        return batch < 32 and (
            mode == "force_argmax" or (args.library == "tt" and batch in (8, 16) and not args.native_small_batch)
        )

    def forward(mode):
        value = logits
        if needs_padding(mode):
            value = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0.0)
        sampler = samplers[mode]
        if args.library == "tt":
            result = sampler(value, tt_out_tok=tokens)
            ttnn.plus_one(sampler.seeds_tt_tensor)
        else:
            params = dict(k=k, p=p, temp=temp) if mode == "split_greedy" else {}
            result = sampler(value, tt_out_tok=tokens, **params)
        assert [t.buffer_address() for t in ttnn.get_device_tensors(result[0])] == [
            t.buffer_address() for t in ttnn.get_device_tensors(tokens)
        ], "Sampler must write the supplied token backing storage"
        assert result[1] is None

    def feedback_view():
        return ttnn.reshape(ttnn.slice(tokens, (0, 0, 0, 0), (1, 1, 1, batch)), (1, batch))

    def stage(host):
        source = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)
        )
        ttnn.copy_host_to_device_tensor(source, logits)

    # Preparation includes the exact decoder-facing token view before any capture.
    for mode in samplers:
        forward(mode)
    view = feedback_view()
    ttnn.to_torch(ttnn.get_device_tensors(view)[0])
    del view
    ttnn.synchronize_device(mesh)
    traces = {}
    for mode in samplers:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        forward(mode)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traces[mode] = trace
    failures = []
    try:
        for case in args.cases:
            # Two different token winners per case expose stale captured logits.
            for shift in (0, 101):
                host = torch.full((1, 1, batch, VOCAB), -4.0, dtype=torch.bfloat16)
                for row in range(batch):
                    winner = ((row % 4) * 65536 + 317 + row * 67 + shift) % VOCAB
                    host[0, 0, row, winner] = 2.0
                    if case == "cross_shard_tie":
                        host[0, 0, row, (winner + 65536) % VOCAB] = 2.0
                    elif case == "many_ties":
                        host[0, 0, row, winner : winner + 64] = 2.0
                    elif case == "negative":
                        host[0, 0, row, winner] = -1.0
                    elif case != "unique":
                        raise ValueError(case)
                expected = host.argmax(dim=-1).reshape(-1).tolist()
                stage(host)
                for mode in ("split_greedy", "force_argmax", "split_greedy"):
                    ttnn.execute_trace(mesh, traces[mode], cq_id=0, blocking=False)
                    view = feedback_view()
                    actual = [ttnn.to_torch(part).reshape(-1).tolist() for part in ttnn.get_device_tensors(view)]
                    del view
                    ok = all(row == expected for row in actual)
                    maxima = host.amax(dim=-1).reshape(-1)
                    semantically_greedy = all(
                        len(device_tokens) == batch
                        and all(
                            0 <= token < VOCAB and host[0, 0, slot, token] == maxima[slot]
                            for slot, token in enumerate(device_tokens)
                        )
                        for device_tokens in actual
                    )
                    record = dict(
                        library=args.library,
                        mode=mode,
                        batch=batch,
                        case=case,
                        shift=shift,
                        expected=expected,
                        actual=actual,
                        exact=ok,
                        semantically_greedy=bool(semantically_greedy),
                        token_storage_shape=list(tokens.shape),
                        feedback_shape=[1, batch],
                        sampler_logical_rows=32 if needs_padding(mode) else batch,
                    )
                    print(json.dumps(record), flush=True)
                    if not semantically_greedy or (not args.semantic_greedy and not ok):
                        failures.append((args.library, batch, mode, case, shift))
        if args.timing:
            for mode, trace in traces.items():
                for _ in range(5):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - start
                replay_ms = 1000 * elapsed / args.iterations
                start = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).reshape(-1)[:batch].tolist()
                with_readback_ms = 1000 * (time.perf_counter() - start) / args.iterations
                print(
                    json.dumps(
                        dict(
                            library=args.library,
                            mode=mode,
                            batch=batch,
                            iterations=args.iterations,
                            host_wall_ms_per_replay=replay_ms,
                            host_wall_ms_per_replay_with_token_readback=with_readback_ms,
                            scope="synthetic sampler only; includes declared row padding; host-wall not device time",
                            sampler_logical_rows=32 if needs_padding(mode) else batch,
                        )
                    ),
                    flush=True,
                )
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
    return failures


if __name__ == "__main__":
    main()
