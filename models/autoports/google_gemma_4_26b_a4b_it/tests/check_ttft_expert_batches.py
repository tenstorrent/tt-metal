# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experiment-only EP prefill batching; real reduced weights, unchanged arithmetic."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import MODEL_ID, REVISION, Gemma4Model
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _ExpertParallelExperts
from models.common.utility_functions import comp_pcc


def batch_variant(source, mesh, rows, gate_block=11, down_cores=88):
    """Share weights and precision; specialize only token batching geometry."""
    assert isinstance(source, _ExpertParallelExperts)
    if gate_block <= 0 or 88 % gate_block:
        raise ValueError("Gate K block must divide the 88 input tiles")
    if down_cores not in (88, 44, 22):
        raise ValueError("Down core count must be 88, 44, or 22")
    candidate = copy.copy(source)
    candidate.prefill_batch_tokens = rows
    # Keep the experiment's explicit width independent of deployment's
    # optional short-sequence policy; the original object remains unchanged.
    if hasattr(candidate, "short_prefill_batch_tokens"):
        candidate.short_prefill_batch_tokens = 32
    candidate.prefill_configs = dict(source.prefill_configs)
    candidate.route_indices = dict(source.route_indices)
    ownership = torch.arange(128, dtype=torch.int32).reshape(1, 1, 1, 128)
    for count in range(32, rows + 1, 32):
        # Historical default preserves the original production row32 config,
        # including its actual gate K (now22). Keep that baseline untouched.
        if count == 32 and gate_block == 11 and down_cores == 88:
            continue
        candidate.prefill_configs[count] = tuple(
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=per_n,
                out_block_h=1,
                out_block_w=per_n,
                per_core_M=count // 32,
                per_core_N=per_n,
                fuse_batch=False,
                mcast_in0=True,
            )
            for grid, block, per_n in (
                ((11, 4), gate_block, 1),
                ((11, down_cores // 11), 22, 88 // down_cores),
            )
        )
        if count not in candidate.route_indices:
            candidate.route_indices[count] = ttnn.from_torch(
                ownership.repeat(1, 1, count, 1),
                device=mesh,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
    for name in ("prefill_gate", "prefill_down", "prefill_compute", "prefill_memory"):
        assert getattr(candidate, name) is getattr(source, name)
    return candidate


def compare(reference, actual, pcc_threshold, relative_threshold):
    assert reference.shape == actual.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    passing, pcc = comp_pcc(reference, actual, pcc_threshold)
    relative = float(
        torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )
    return dict(
        passed=finite and bool(passing) and relative <= relative_threshold,
        finite=finite,
        pcc=float(pcc),
        relative_l2=relative,
        max_abs=float((actual - reference).abs().max()),
        exact_equal=torch.equal(reference, actual),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", choices=(32, 64, 128), default=[32, 64, 128])
    parser.add_argument("--gate-blocks", type=int, nargs="+", default=[11], help="Gate K tile blocks dividing 88")
    parser.add_argument("--down-cores", type=int, nargs="+", choices=(88, 44, 22), default=[88])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 33, 64, 127, 128, 129, 256])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-replays", type=int, default=2)
    parser.add_argument("--pcc", type=float, default=0.999)
    parser.add_argument("--relative-l2", type=float, default=0.01)
    args = parser.parse_args()
    if any(not 2 <= length <= 256 for length in args.lengths) or min(args.samples, args.warmup_replays) < 1:
        parser.error("lengths must be 2..256 and replay/sample counts positive")
    if any(block <= 0 or 88 % block for block in args.gate_blocks):
        parser.error("gate-blocks must be positive divisors of 88")
    widths = list(dict.fromkeys([32] + args.candidates))
    settings = list(
        dict.fromkeys(
            [(32, 11, 88)]
            + [
                (width, block, cores)
                for width in args.candidates
                for block in args.gate_blocks
                for cores in args.down_cores
            ]
        )
    )
    report = dict(
        scope="Isolated full layers 0 and 5 independently consume real token embeddings; not full-stack accuracy or serving TTFT",
        revision=REVISION,
        layers=[0, 5],
        candidates=widths,
        gate_blocks=args.gate_blocks,
        down_cores=args.down_cores,
        baseline="Original production row32 program objects; historical setting32/K11/down88 is a baseline sentinel",
        candidate_settings=[
            dict(batch=width, requested_gate_k_tiles=block, down_cores=cores) for width, block, cores in settings
        ],
        lengths=args.lengths,
        pcc_threshold=args.pcc,
        relative_l2_threshold=args.relative_l2,
        token_check="Diagnostic LM-head argmax after one isolated layer; not a generated model token",
        raw_tensors=str(args.output.with_suffix(".pt")),
        results=[],
        passed=False,
    )
    raw = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(4)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1024**3)
    trace = None
    try:
        model = Gemma4Model(mesh, max_seq_len=1024, layer_indices=(0, 5))
        cache, host_table = model.allocate_cache(slots=1, context=1024)
        table = model.upload(host_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        text = "Explain how sunlight, water, and soil help a tree grow, using concrete examples. " * 32
        ids = tokenizer.encode(text)[: max(args.lengths)]
        assert len(ids) == max(args.lengths)
        report["input_token_ids"] = ids
        report["precision"] = model.precision_summary()
        for number, (layer_index, layer) in enumerate(zip(model.layer_indices, model.layers)):
            owner = layer.layer.moe.experts
            original = owner.prefill
            assert original.prefill_batch_tokens == 32
            variants = {
                (width, block, cores): batch_variant(original, mesh, width, block, cores)
                for width, block, cores in settings
            }
            report.setdefault("expert_precision", {})[str(layer_index)] = dict(
                gate_dtype=str(original.prefill_gate.dtype),
                down_dtype=str(original.prefill_down.dtype),
                compute=str(original.prefill_compute),
                memory=str(original.prefill_memory),
                gate_grid=str(original.prefill_configs[32][0].compute_with_storage_grid_size),
                down_grid=str(original.prefill_configs[32][1].compute_with_storage_grid_size),
                gate_k_tiles=original.prefill_configs[32][0].in0_block_w,
                down_k_tiles=22,
                original_gate32=str(original.prefill_configs[32][0]),
                original_down32=str(original.prefill_configs[32][1]),
            )
            try:
                for length in args.lengths:
                    tokens = model.upload(
                        torch.tensor(ids[:length]).reshape(1, 1, 1, length).int(),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                    embedding = model.embed(tokens)
                    reference = None
                    baseline_token = None
                    baseline_us = None
                    for width, gate_block, down_cores in settings:
                        assert trace is None, "Release the previous trace before candidate setup/warmup"
                        owner.prefill = variants[width, gate_block, down_cores]
                        row = dict(
                            layer=layer_index,
                            layer_type=model.config.layer_types[layer_index],
                            length=length,
                            batch=width,
                            requested_gate_k_tiles=gate_block,
                            gate_k_tiles=owner.prefill.prefill_configs[32][0].in0_block_w,
                            requested_down_cores=down_cores,
                            down_cores=88 // owner.prefill.prefill_configs[32][1].per_core_N,
                            baseline=(width, gate_block, down_cores) == (32, 11, 88),
                            gate_dtype=str(owner.prefill.prefill_gate.dtype),
                            down_dtype=str(owner.prefill.prefill_down.dtype),
                            compute=str(owner.prefill.prefill_compute),
                            memory=str(owner.prefill.prefill_memory),
                            configs={
                                str(count): dict(gate=str(pair[0]), down=str(pair[1]))
                                for count, pair in owner.prefill.prefill_configs.items()
                                if count <= width
                            },
                        )
                        report["results"].append(row)

                        def forward():
                            return layer.prefill_forward(
                                embedding,
                                rope_mats=model.rope_prefill[model.config.layer_types[layer_index]],
                                page_table=table,
                                kv_cache=cache[number],
                                user_id=0,
                            )

                        # Projection/token diagnostics run before capture so their
                        # allocations cannot conflict with a live layer trace.
                        ttnn.synchronize_device(mesh)
                        started = time.perf_counter_ns()
                        output = forward()
                        ttnn.synchronize_device(mesh)
                        row["warmup_eager_us"] = (time.perf_counter_ns() - started) / 1000
                        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).float().clone()
                        raw_key = f"layer{layer_index}_s{length}_batch{width}"
                        if gate_block != 11:
                            raw_key += f"_gateK{gate_block}"
                        if down_cores != 88:
                            raw_key += f"_downCores{down_cores}"
                        raw[raw_key] = actual
                        row["raw_tensor_key"] = raw_key
                        logits = model.logits(output[:, :, -1:, :])
                        host_logits = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
                        row["diagnostic_token"] = int(host_logits.reshape(-1).argmax())
                        del logits, host_logits
                        output.deallocate(True)
                        del output
                        if row["baseline"]:
                            reference = actual
                            baseline_token = row["diagnostic_token"]
                        row.update(compare(reference, actual, args.pcc, args.relative_l2))
                        row["diagnostic_token_equal"] = row["diagnostic_token"] == baseline_token
                        row["programs_after_warmup"] = mesh.num_program_cache_entries()
                        ttnn.synchronize_device(mesh)
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        try:
                            output = forward()
                        finally:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        for _ in range(args.warmup_replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples = []
                        for _ in range(args.samples):
                            started = time.perf_counter_ns()
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter_ns() - started) / 1000)
                        replay = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).float()
                        row["replay_matches_eager"] = torch.equal(replay, actual)
                        row["trace_replay_us"] = samples
                        row["trace_median_us"] = statistics.median(samples)
                        row["programs_after_replay"] = mesh.num_program_cache_entries()
                        if row["baseline"]:
                            baseline_us = row["trace_median_us"]
                        row["baseline_over_candidate"] = baseline_us / row["trace_median_us"]
                        row["passed"] = row["passed"] and row["replay_matches_eager"]
                        ttnn.release_trace(mesh, trace)
                        trace = None
                        output.deallocate(True)
                        del output, replay, actual
                        save()
                    del embedding, tokens
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                    trace = None
                owner.prefill = original
        report["passed"] = all(row["passed"] for row in report["results"])
        save()
        assert report["passed"], "Expert batch correctness failed; inspect raw results"
    except BaseException as error:
        report["error"] = repr(error)
        save()
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        try:
            torch.save(raw, args.output.with_suffix(".pt"))
        finally:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
