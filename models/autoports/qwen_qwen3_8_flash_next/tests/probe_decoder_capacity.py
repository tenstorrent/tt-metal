# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PLE resident-memory bound with the actual decoder weights and chunk workspace.

This is a capacity diagnostic, not an HF accuracy test. Default/proxy modes keep
full logical I/O resident but repeat offset0 and selected tails; they are not a
complete public prefill. --full-public-prefill executes the actual public method
at every offset, with all32 physical math/hash signatures prepared. Even that
mode uses constant inputs and requires independent real-input HF accuracy before
accepting a capability reduction.
"""

import argparse
import gc
import json
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_input import prepare_chunk_io, read_token_chunk, read_token_words
from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), required=True)
    parser.add_argument(
        "--lengths",
        help="Comma-separated explicit probes instead of binary search; use a fresh process for boundary verification",
    )
    parser.add_argument(
        "--reserve-decode-input",
        action="store_true",
        help="Keep one decode hidden/token input resident during prefill, matching probe_ple_decoder",
    )
    parser.add_argument(
        "--decode-lifecycle",
        action="store_true",
        help="After chunk capacity checks, release output, warm/capture/replay decode, release trace/output, and repeat the resident-output checks",
    )
    parser.add_argument(
        "--prepare-programs",
        action="store_true",
        help="Before each candidate's full input allocation, warm its exact prefill32/tail/decode1 signatures, matching the public prepared harness",
    )
    parser.add_argument(
        "--prepare-all-tails",
        action="store_true",
        help="Prepare all32 physical prefill lengths plus decode1 before large I/O; exercise all counts in both lifecycle passes (implies --prepare-programs)",
    )
    parser.add_argument(
        "--prepare-once",
        action="store_true",
        help="Reuse one all-tail preparation and restore its initial state between candidates; requires --prepare-all-tails",
    )
    parser.add_argument(
        "--terminal-residues",
        type=int,
        help="Probe the32 consecutive lengths END-31..END; mutually exclusive with --lengths, requires --prepare-all-tails",
    )
    parser.add_argument(
        "--search-residues",
        help="Candidate-only fixed-tail binary searches: comma-separated residues1..32, or all; requires all-tail prepare-once/public-chunks-only/decode lifecycle",
    )
    parser.add_argument("--search-lower-length", type=int, default=32)
    parser.add_argument("--search-upper-length", type=int, default=3712)
    parser.add_argument(
        "--public-chunks-only",
        action="store_true",
        help="After all-tail preparation, check only two full chunks and actual tail, without the stronger full-I/O all-count sweep",
    )
    parser.add_argument(
        "--full-public-prefill",
        action="store_true",
        help="Execute actual public prefill on the complete input at every offset; implies all32 preparation, preserves optional decode/repeat lifecycle; still not HF accuracy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reserve-sparse-down-workspace",
        action="store_true",
        help="Test-only fragmentation contrast: reserve80MiB expanded sparse down output before state/preparation/large I/O; unchanged native math, not a runtime implementation",
    )
    parser.add_argument(
        "--split-gr-query-rows",
        type=int,
        choices=(16,),
        help="Test-only capacity hypothesis: split only two GR down projections into BF16 row groups16 before unchanged FP32 projection; up projections unchanged",
    )
    parser.add_argument(
        "--reserve-sparse-down-per-call",
        action="store_true",
        help="Test-only reserve80MiB at each MLP entry after GDN/GR and release after original MLP; incompatible with persistent reservation",
    )
    parser.add_argument(
        "--inplace-expert-activation",
        action="store_true",
        help="Test-only reuse private sparse gate output for native SiLU and activation multiply; guard exact gate/up weight identities",
    )
    parser.add_argument(
        "--reuse-private-reduction-buffer",
        action="store_true",
        help="Test-only private weighted-down padding fill and aligned view before unchanged native sum, avoiding nonaligned80MiB clone",
    )
    args = parser.parse_args()
    if args.reserve_sparse_down_per_call and args.reserve_sparse_down_workspace:
        parser.error("Per-call and persistent sparse-down reservations are mutually exclusive")
    residues = None
    if args.search_residues is not None:
        try:
            residues = (
                list(range(1, 33))
                if args.search_residues == "all"
                else [int(v) for v in args.search_residues.split(",")]
            )
        except ValueError:
            parser.error("--search-residues must be all or comma-separated integers1..32")
        if not residues or len(set(residues)) != len(residues) or any(v not in range(1, 33) for v in residues):
            parser.error("--search-residues requires distinct residues1..32")
        if args.lengths is not None or args.terminal_residues is not None or args.full_public_prefill:
            parser.error("Per-residue search is a proxy mode, incompatible with lengths/terminal/full-public modes")
        if not all(
            (
                args.prepare_all_tails,
                args.prepare_once,
                args.public_chunks_only,
                args.decode_lifecycle,
                args.reserve_decode_input,
            )
        ):
            parser.error(
                "Per-residue search requires --prepare-all-tails --prepare-once --public-chunks-only --decode-lifecycle --reserve-decode-input"
            )
        if not 32 <= args.search_lower_length <= args.search_upper_length:
            parser.error("Require32 <= search-lower-length <= search-upper-length")
        if any((args.search_lower_length - r + 31) // 32 > (args.search_upper_length - r) // 32 for r in residues):
            parser.error("Search interval must contain at least one length for every residue")
    if args.full_public_prefill:
        args.prepare_all_tails = True
        if args.public_chunks_only:
            parser.error("--full-public-prefill and --public-chunks-only select different execution modes")
    if args.prepare_all_tails:
        args.prepare_programs = True
    if (
        args.prepare_once or args.terminal_residues is not None or args.public_chunks_only
    ) and not args.prepare_all_tails:
        parser.error("--prepare-once, --terminal-residues and --public-chunks-only require --prepare-all-tails")
    if args.terminal_residues is not None and args.lengths is not None:
        parser.error("--terminal-residues and --lengths are mutually exclusive")
    if args.decode_lifecycle and not args.reserve_decode_input:
        parser.error("--decode-lifecycle requires --reserve-decode-input")
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    if residues is not None and args.search_upper_length >= config.max_position_embeddings:
        parser.error("Per-residue upper length must leave an advertised-context position for decode")
    reference, weights = load_reference(args.snapshot, config)
    del reference
    lengths = [int(value) for value in args.lengths.split(",")] if args.lengths else None
    if args.terminal_residues is not None:
        lengths = list(range(args.terminal_residues - 31, args.terminal_residues + 1))
    if lengths is not None and (not lengths or min(lengths) < 1 or max(lengths) > config.max_position_embeddings):
        parser.error("--lengths must be positive and within advertised context")
    if args.full_public_prefill and args.decode_lifecycle and lengths is not None:
        if max(lengths) >= config.max_position_embeddings:
            parser.error("Full public prefill plus decode requires one remaining advertised-context position")
    if args.prepare_all_tails and not args.full_public_prefill and lengths is not None and min(lengths) < 32:
        parser.error("--prepare-all-tails requires every explicit candidate length >=32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": (
            "Complete public prefill at every offset with constant input and all32 prepared signatures; capacity/lifecycle only, not HF accuracy"
            if args.full_public_prefill
            else "Resident-I/O allocation proxy with repeated offset0 chunks and selected tails; not complete public execution or HF accuracy"
        ),
        "full_public_prefill": args.full_public_prefill,
        "execution_mode": "full_public" if args.full_public_prefill else "chunk_proxy",
        "input_factory": "setup torch full/ones -> ttnn.from_torch; no ttnn.full",
        "chunk_io": "tt.chunk_input read_token_chunk/read_token_words; setup prepare_chunk_io(width=10240)",
        "reserve_decode_input": args.reserve_decode_input,
        "decode_lifecycle": args.decode_lifecycle,
        "prepare_programs": args.prepare_programs,
        "prepare_all_tails": args.prepare_all_tails,
        "prepare_once": args.prepare_once,
        "terminal_residues_end": args.terminal_residues,
        "public_chunks_only": args.public_chunks_only,
        "capacity_profile": (
            "actual public prefill, every offset, batch1 all32-prepared continuous-range profile"
            if args.full_public_prefill
            else "public chunk-path allocation proxy under all32 prepared shapes; no stronger sweep"
            if args.public_chunks_only
            else "stronger full-I/O sweep under all32 prepared shapes; fixed32,32,1..32,then actual tail"
            if args.prepare_all_tails
            else "candidate-specific physical signatures"
        ),
        "sweep_failure_is_not_public_prefill_failure": not args.full_public_prefill,
        "program_path": (
            "complete public prefill; optional warmed/captured/replayed decode1 then complete repeated public prefill"
            if args.full_public_prefill
            else "two repeated offset0 chunks plus optional all-count sweep and tail; optional decode lifecycle; repeated proxy"
        ),
        "explicit_lengths": lengths,
        "search_is_candidate_only": True,
        "advertised_context": config.max_position_embeddings,
        "expert_dtype": args.expert_dtype,
        "cache_dir": str(args.cache_dir),
        "probes": [],
        "capability_reduction_accepted": False,
        "reserved_sparse_down_workspace": {"enabled": args.reserve_sparse_down_workspace},
        "per_call_sparse_down_workspace": {"enabled": args.reserve_sparse_down_per_call},
        "inplace_expert_activation": {"enabled": args.inplace_expert_activation},
        "reuse_private_reduction_buffer": {"enabled": args.reuse_private_reduction_buffer},
        "split_gr_query_rows": {"enabled": args.split_gr_query_rows is not None, "rows": args.split_gr_query_rows},
        "per_residue_search": None
        if residues is None
        else {
            "residues": residues,
            "lower_length": args.search_lower_length,
            "upper_length": args.search_upper_length,
            "monotonicity_assumption": "Binary search assumes capacity decreases with block index at a fixed tail residue; fragmentation/history can violate even this assumption. No continuous-range proof.",
            "candidate_only": True,
            "explicit_test_policy": {
                "split_gr_down_query_rows": args.split_gr_query_rows,
                "reserve_sparse_down_workspace": args.reserve_sparse_down_workspace,
                "reserve_sparse_down_per_call": args.reserve_sparse_down_per_call,
                "inplace_expert_activation": args.inplace_expert_activation,
                "reuse_private_reduction_buffer": args.reuse_private_reduction_buffer,
                "scope": "Same explicit policy for every probe in this process; experimental policy results cannot be promoted to default-runtime capacity",
            },
            "results": [],
        },
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    workspace_patch = None
    workspace = None

    def hidden_input(count):
        return ttnn.from_torch(
            torch.full((1, 1, count, 10240), 0.02, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )

    def token_input(count):
        return ttnn.from_torch(
            torch.ones((1, count, 16), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
        )

    def memory():
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return {
            "banks": view.num_banks,
            "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
            "largest_contiguous_bytes_per_bank": view.largest_contiguous_bytes_free_per_bank,
        }

    try:
        report["empty_device"] = memory()
        packed = PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
        model = FunctionalDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=1,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, args.expert_dtype),
        )
        if args.split_gr_query_rows is not None:
            from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_gr_rows import install_split_gr_down

            report["split_gr_query_rows"] = install_split_gr_down(model, args.split_gr_query_rows)
        if args.inplace_expert_activation:
            from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_expert_activation import (
                InplaceExpertActivation,
            )

            model.mlp = InplaceExpertActivation(mesh, model.mlp)
            report["inplace_expert_activation"] = model.mlp.report
        if args.reuse_private_reduction_buffer:
            from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_private_reduction import (
                ReusePrivateReductionBuffer,
            )

            model.mlp = ReusePrivateReductionBuffer(model.mlp)
            report["reuse_private_reduction_buffer"] = model.mlp.report
        if args.reserve_sparse_down_workspace:
            from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_sparse_workspace import (
                ReservedSparseDownWorkspace,
            )

            before_reservation = memory()
            workspace = ReservedSparseDownWorkspace(mesh, model.mlp.down)
            workspace.report["before_reservation"] = before_reservation
            workspace.report["after_reservation"] = memory()
            report["reserved_sparse_down_workspace"] = workspace.report
            workspace_patch = workspace.intercept()
            workspace_patch.start()
        if args.reserve_sparse_down_per_call:
            from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_sparse_workspace import (
                PerCallSparseDownWorkspace,
            )

            model.mlp = PerCallSparseDownWorkspace(mesh, model.mlp)
            report["per_call_sparse_down_workspace"] = model.mlp.report
        # Match public state/hash preparation exactly for fresh explicit probes.
        # Multi-candidate searches retain a union and remain candidate-only evidence.
        hash_lengths = (
            tuple(sorted({1, 7, 31, 32, *(length % 32 or 32 for length in lengths)}))
            if args.prepare_programs and not args.prepare_all_tails and lengths is not None
            else tuple(range(1, 33))
        )
        state = model.create_state(hash_lengths=hash_lengths)
        prepare_chunk_io(mesh, width=10240)
        report["chunk_io_prepared_before_large_io"] = True
        report["hash_lengths"] = list(hash_lengths)
        report["resident_model"] = memory()
        prepared_signatures = None
        preparation_state = None
        preparation_source_length = None
        fresh_state = (
            [
                ttnn.from_torch(
                    ttnn.to_torch(getattr(state, name)),
                    dtype=getattr(state, name).dtype,
                    layout=getattr(state, name).layout,
                )
                for name in ("conv", "recurrent", "ple_conv", "token_history")
            ]
            if args.full_public_prefill
            else None
        )

        def probe(length):
            nonlocal prepared_signatures, preparation_state, preparation_source_length
            entry = {"length": length, "before": memory(), "phase": "input"}
            entry["reserved_sparse_down_workspace"] = args.reserve_sparse_down_workspace
            entry["reserve_sparse_down_per_call"] = args.reserve_sparse_down_per_call
            entry["inplace_expert_activation"] = args.inplace_expert_activation
            entry["reuse_private_reduction_buffer"] = args.reuse_private_reduction_buffer
            entry["split_gr_query_rows"] = args.split_gr_query_rows
            tensors = []
            chunk = None
            output = None
            decode_output = None
            trace = None
            passed = False
            entry["memory_phases"] = []

            def mark(phase):
                entry["phase"] = phase
                entry["memory_phases"].append({"phase": phase, **memory()})

            state_names = ("conv", "recurrent", "ple_conv", "token_history")

            def snapshot():
                return [
                    ttnn.from_torch(
                        ttnn.to_torch(getattr(state, name)),
                        dtype=getattr(state, name).dtype,
                        layout=getattr(state, name).layout,
                    )
                    for name in state_names
                ]

            def restore(values):
                for name, value in zip(state_names, values):
                    ttnn.copy_host_to_device_tensor(value, getattr(state, name))

            try:
                if fresh_state is not None:
                    restore(fresh_state)
                    ttnn.synchronize_device(mesh)
                    entry["fresh_request_state_restored"] = True
                reuse_preparation = args.prepare_once and prepared_signatures is not None
                if args.prepare_programs and not reuse_preparation:
                    mark("prepare_programs_before")
                    saved = snapshot()
                    signatures = [("prefill", 32 if args.prepare_all_tails else min(length, 32))]
                    if args.prepare_all_tails:
                        signatures.extend(("prefill", count) for count in range(1, 32))
                    elif length > 32 and length % 32:
                        signatures.append(("prefill", length % 32))
                    signatures.append(("decode", 1))
                    entry["prepared_signatures"] = []
                    for phase, count in signatures:
                        warm_hidden = warm_ids = warm_output = None
                        try:
                            restore(saved)
                            warm_hidden = hidden_input(count)
                            warm_ids = token_input(count)
                            entry["phase"] = "prepare_" + phase + str(count)
                            with runtime_audit():
                                operation = model.prefill_forward if phase == "prefill" else model.decode_forward
                                warm_output = operation(warm_hidden, state=state, token_ids=warm_ids)
                            ttnn.synchronize_device(mesh)
                            entry["prepared_signatures"].append(
                                {
                                    "phase": phase,
                                    "tokens": count,
                                    "hidden_shape": [1, 1, count, 10240],
                                    "token_shape": [1, count, 16],
                                }
                            )
                        finally:
                            ttnn.synchronize_device(mesh)
                            for temporary in (warm_output, warm_ids, warm_hidden):
                                if temporary is not None:
                                    ttnn.deallocate(temporary)
                            restore(saved)
                            ttnn.synchronize_device(mesh)
                    entry["preparation_state_restored"] = all(
                        torch.equal(ttnn.to_torch(getattr(state, name)), ttnn.to_torch(value))
                        for name, value in zip(state_names, saved)
                    )
                    assert entry["preparation_state_restored"], "Preparation must restore all four states"
                    mark("prepare_programs_after")
                    if args.prepare_once:
                        prepared_signatures = list(entry["prepared_signatures"])
                        preparation_state = saved
                        preparation_source_length = length
                elif reuse_preparation:
                    restore(preparation_state)
                    ttnn.synchronize_device(mesh)
                    entry["prepared_signatures"] = list(prepared_signatures)
                    entry["preparation_reused_from_length"] = preparation_source_length
                    entry["preparation_state_restored"] = all(
                        torch.equal(ttnn.to_torch(getattr(state, name)), ttnn.to_torch(value))
                        for name, value in zip(state_names, preparation_state)
                    )
                    assert entry["preparation_state_restored"]
                    mark("prepared_programs_reused")
                entry["phase"] = "input"
                hidden = hidden_input(length)
                tensors.append(hidden)
                entry["phase"] = "token_words"
                token_ids = token_input(length)
                tensors.append(token_ids)
                if args.reserve_decode_input:
                    entry["phase"] = "decode_inputs"
                    decode_hidden = hidden_input(1)
                    tensors.append(decode_hidden)
                    decode_ids = token_input(1)
                    tensors.append(decode_ids)
                entry["phase"] = "output"
                if not args.full_public_prefill:
                    output = ttnn.empty(
                        (1, 1, length, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                    )
                    tensors.append(output)
                entry["resident_io"] = memory()
                entry["resident_io_includes_output"] = not args.full_public_prefill
                entry["phase"] = "decoder_chunk"
                count = min(length, 32)
                chunks = [
                    {"start": 0, "count": count, "stage": "proxy_offset0_chunk"},
                    {"start": 0, "count": count, "stage": "proxy_offset0_chunk"},
                ]
                stronger_sweep = args.prepare_all_tails and not args.public_chunks_only and not args.full_public_prefill
                if stronger_sweep:
                    chunks.extend(
                        {"start": 0, "count": size, "stage": "stronger_full_io_sweep"} for size in range(1, 33)
                    )
                # The same fixed sweep prefix precedes every candidate's tail.
                # This removes the old tail-dependent allocator-history change.
                if length > 32 and length % 32:
                    chunks.append({"start": length - length % 32, "count": length % 32, "stage": "candidate_tail"})
                if args.full_public_prefill:
                    chunks = [
                        {"start": start, "count": min(32, length - start), "stage": "actual_public_chunk"}
                        for start in range(0, length, 32)
                    ]
                entry["chunk_lengths"] = [item["count"] for item in chunks]
                entry["chunk_plan"] = chunks
                entry["candidate_tail_after_stronger_sweep"] = stronger_sweep and bool(length % 32)
                entry["all_tails_checked_with_full_resident_io"] = False
                entry["completed_chunk_checks"] = []
                initial_state = snapshot() if args.decode_lifecycle else None

                def run_chunks():
                    nonlocal chunk, output
                    if args.full_public_prefill:
                        original_chunk = model._chunk
                        observed = []

                        def observe_chunk(value, state, token_ids=None):
                            index = len(observed)
                            planned = chunks[index]
                            assert value.shape[-2] == planned["count"]
                            entry["active_chunk"] = {"pass": entry["phase"], "index": index, **planned}
                            result = original_chunk(value, state, token_ids)
                            observed.append(dict(entry["active_chunk"]))
                            return result

                        model._chunk = observe_chunk
                        try:
                            with runtime_audit():
                                output = model.prefill_forward(hidden, state=state, token_ids=token_ids)
                            tensors.append(output)
                        finally:
                            model._chunk = original_chunk
                            entry["completed_chunk_checks"].extend(observed)
                        ttnn.synchronize_device(mesh)
                        assert len(observed) == len(chunks) and sum(x["count"] for x in observed) == length
                        assert tuple(output.shape) == (1, 1, length, 10240)
                        entry.setdefault("full_public_passes", []).append(
                            {
                                "phase": entry["phase"],
                                "logical_rows": length,
                                "completed_chunks": len(observed),
                                "all_logical_rows_visited": True,
                                "offset_observation": "Public method owns extraction; offsets are the cumulative logical chunk counts, not a content-correctness oracle",
                                "output_finite": bool(torch.isfinite(ttnn.to_torch(output)).all()),
                            }
                        )
                        assert entry["full_public_passes"][-1]["output_finite"]
                        return
                    with runtime_audit():
                        # Match the repaired public loop: release each chunk
                        # immediately after enqueueing its last-consumer write.
                        for index, planned in enumerate(chunks):
                            start, size = planned["start"], planned["count"]
                            entry["active_chunk"] = {"pass": entry["phase"], "index": index, **planned}
                            chunk = model.prefill_forward(
                                read_token_chunk(hidden, start, size),
                                state=state,
                                token_ids=read_token_words(token_ids, start, size),
                            )
                            write_token_chunk(chunk, output, start)
                            chunk = None
                            entry["completed_chunk_checks"].append(dict(entry["active_chunk"]))
                    ttnn.synchronize_device(mesh)

                run_chunks()
                entry["all_tails_checked_with_full_resident_io"] = stronger_sweep
                mark("first_chunks_complete")
                if args.decode_lifecycle:
                    # Nothing consumes these outputs after the capacity checks.
                    if chunk is not None:
                        ttnn.deallocate(chunk)
                    chunk = None
                    assert tensors[-1] is output
                    tensors.pop()
                    ttnn.deallocate(output)
                    mark("prefill_output_released")
                    prefix_state = snapshot()
                    restore(prefix_state)
                    mark("decode_warm")
                    with runtime_audit():
                        decode_output = model.decode_forward(decode_hidden, state=state, token_ids=decode_ids)
                    ttnn.synchronize_device(mesh)
                    ttnn.deallocate(decode_output)
                    decode_output = None
                    restore(prefix_state)
                    mark("decode_capture")
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        with runtime_audit():
                            decode_output = model.decode_forward(decode_hidden, state=state, token_ids=decode_ids)
                    finally:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    # Only this output is read synchronously after replay.
                    trace_allocation_tracker.acknowledge_corruptible(decode_output)
                    restore(prefix_state)
                    mark("decode_replay")
                    with runtime_audit():
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    entry["decode_output_finite"] = bool(torch.isfinite(ttnn.to_torch(decode_output)).all())
                    assert entry["decode_output_finite"], "Initialized decode produced nonfinite values"
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    # Never force-deallocate a live trace's output.
                    ttnn.deallocate(decode_output)
                    decode_output = None
                    mark("decode_trace_output_released")
                    restore(initial_state)
                    mark("repeat_output_allocation")
                    if not args.full_public_prefill:
                        output = ttnn.empty(
                            (1, 1, length, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                        )
                        tensors.append(output)
                    mark("repeat_resident_io")
                    entry["phase"] = "repeated_decoder_chunk"
                    run_chunks()
                    entry["all_tails_rechecked_after_trace"] = stronger_sweep
                    mark("repeated_chunks_complete")
                    entry["decode_lifecycle_completed"] = True
                passed = True
                entry["phase"] = "complete"
            except RuntimeError as exc:
                entry["python_traceback"] = [
                    {"filename": frame.filename, "name": frame.name, "lineno": frame.lineno, "source": frame.line}
                    for frame in traceback.extract_tb(exc.__traceback__)
                ]
                message = str(exc).split("backtrace:")[0]
                entry["error"] = message
                # A shape, kernel or assertion failure is not a physical limit.
                if not any(
                    term in message.lower()
                    for term in (
                        "out of memory",
                        "not enough space",
                        "allocation failed",
                        "cannot allocate",
                        "failed to allocate",
                    )
                ):
                    raise
            finally:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                if decode_output is not None:
                    ttnn.deallocate(decode_output)
                if chunk is not None:
                    ttnn.deallocate(chunk)
                for tensor in reversed(tensors):
                    ttnn.deallocate(tensor)
                gc.collect()
                entry["passed"] = passed
                entry["after"] = memory()
                if workspace is not None:
                    entry["reserved_sparse_down_last_call"] = dict(workspace.report["last_call"] or {})
                report["probes"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("CAPACITY_PROBE", json.dumps(entry), flush=True)
            return passed

        # Warm the chunk before the full attempt so compiled persistent resources
        # are present for every comparison. State values do not affect allocation.
        if not args.prepare_programs:
            assert probe(32), "The minimum decoder chunk must fit before searching capacity"
        # Prepared mode performs only the selected candidate's setup before its
        # full input allocation; it does not pre-capture an extra 32-token probe.
        report["implicit_probe32"] = not args.prepare_programs
        if residues is not None:
            from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_residue_search import search_residue

            report[
                "required_followup"
            ] = "Fresh-process candidate/next-failure checks for every limiting residue, complete public prefill and real-input HF checks, nonaligned neighbors and traced decode; search does not establish a continuous supported range."
            for residue in residues:
                result = search_residue(probe, residue, args.search_lower_length, args.search_upper_length)
                report["per_residue_search"]["results"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("RESIDUE_CANDIDATE", json.dumps(result), flush=True)
            return
        if lengths is not None:
            for length in lengths:
                probe(length)
            report["required_followup"] = (
                "Complete real-input HF comparison at the public boundary and nonaligned neighbor; constant-input capacity/lifecycle success alone does not accept a capability reduction."
                if args.full_public_prefill
                else "Complete public prefill and HF comparison; explicit allocation proxies alone do not accept a capability reduction."
            )
            return
        maximum = config.max_position_embeddings - int(args.full_public_prefill and args.decode_lifecycle)
        if probe(maximum):
            low, high = maximum, maximum + 1
        else:
            low, high = 32, maximum
            while high - low > 1:
                middle = (low + high) // 2
                if probe(middle):
                    low = middle
                else:
                    high = middle
        report["largest_resident_candidate"] = low
        report["first_failing_length"] = high if high <= maximum else None
        report[
            "required_followup"
        ] = "Recheck candidate and candidate+1 in fresh processes because cached programs can accumulate during binary search. Complete prefill/HF validation at final candidate and nonaligned neighbor, with traced final decode; trace workspace may lower this bound."
    finally:
        if workspace_patch is not None:
            workspace_patch.stop()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
