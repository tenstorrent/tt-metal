# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact bounded input copies, shape-independent allocation, and fixed-plan trace.

Run under the serialized device-usage workflow. Native-only is a diagnostic of
slice program retention, not a functional acceptance result. All Torch and
host/device transfers occur at explicit test boundaries outside runtime_audit.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit

WIDTH = 10240


def snapshot(mesh):
    gc.collect()
    ttnn.synchronize_device(mesh)
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        "program_cache_entries": mesh.num_program_cache_entries(),
        "allocated_bytes": view.num_banks * view.total_bytes_allocated_per_bank,
        "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
        "largest_contiguous_bytes_per_bank": view.largest_contiguous_bytes_free_per_bank,
    }


def release(*tensors):
    for tensor in tensors:
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


def fixtures(length, variant=0):
    # Independent row/column values exercise face boundaries and source offsets.
    generator = torch.Generator().manual_seed(3701 + length + variant * 100003)
    hidden = torch.randn((1, 1, length, WIDTH), generator=generator).bfloat16()
    words = torch.randint(0, 1 << 32, (1, length, 16), generator=generator, dtype=torch.int64)
    words[0, 0, :4] = torch.tensor([0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF])
    return hidden, words


def upload(values, mesh=None):
    hidden, words = values
    return (
        ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
        ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh),
    )


def offsets(length):
    last = ((length - 1) // 32) * 32
    return sorted({0, min(32, last), min(64, last), (last // 64) * 32, last})


def check_outputs(outputs, values, offset, count, padding_required):
    hidden, words = values
    actual_hidden = ttnn.to_torch(outputs[0])
    actual_words = ttnn.to_torch(outputs[1]).to(torch.int64)
    checks = {
        "bf16_logical_shape": list(outputs[0].shape) == [1, 1, count, WIDTH],
        "words_logical_shape": list(outputs[1].shape) == [1, count, 16],
        "bf16_exact": torch.equal(actual_hidden, hidden[:, :, offset : offset + count]),
        "words_exact": torch.equal(actual_words, words[:, offset : offset + count]),
    }
    if padding_required:
        # This host-only untilization must not add inspection programs to the
        # device cache whose growth is the subject of this experiment.
        padded = outputs[0].cpu().to_torch_with_padded_shape()
        checks["bf16_physical_shape"] = list(padded.shape) == [1, 1, 32, WIDTH]
        checks["bf16_padding_zero"] = bool(torch.all(padded[:, :, count:] == 0))
    checks["passed"] = all(checks.values())
    return checks


def negative_host_contracts(mesh):
    """Real tensor metadata; every invalid call must fail before generic dispatch."""
    from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_input import (
        prepare_chunk_io,
        read_token_chunk,
        read_token_words,
    )
    from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk

    tensors = []

    def tensor(shape, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory=ttnn.DRAM_MEMORY_CONFIG, tile=None):
        value = torch.zeros(shape, dtype=torch.int32 if dtype == ttnn.uint32 else torch.float32)
        kwargs = {} if tile is None else {"tile": tile}
        result = ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh, memory_config=memory, **kwargs)
        tensors.append(result)
        return result

    records = []
    try:
        hidden = tensor((1, 1, 33, WIDTH))
        words = tensor((1, 33, 16), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        source = tensor((1, 1, 32, WIDTH))
        target = tensor((1, 1, 64, WIDTH))
        hidden_float = tensor((1, 1, 32, WIDTH), dtype=ttnn.float32)
        hidden_rm = tensor((1, 1, 32, WIDTH), layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_l1 = tensor((1, 1, 32, WIDTH), memory=ttnn.L1_MEMORY_CONFIG)
        hidden_width = tensor((1, 1, 32, 64))
        hidden_tile = tensor((1, 1, 32, WIDTH), tile=ttnn.Tile((16, 32)))
        released_hidden = tensor((1, 1, 32, WIDTH))
        release(released_hidden)
        words_float = tensor((1, 33, 16), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        words_width = tensor((1, 33, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        words_l1 = tensor((1, 33, 16), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory=ttnn.L1_MEMORY_CONFIG)
        host_hidden = ttnn.from_torch(torch.zeros(1, 1, 32, WIDTH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        cases = []
        for name, reader, operand in (("hidden", read_token_chunk, hidden), ("words", read_token_words, words)):
            for label, offset, count in (
                ("negative_offset", -32, 1),
                ("unaligned_offset", 1, 1),
                ("zero_count", 0, 0),
                ("oversize_count", 0, 33),
                ("past_logical_tail", 32, 2),
                ("past_source", 64, 1),
                ("bool_offset", True, 1),
                ("float_count", 0, 1.0),
                ("bool_count", 0, True),
            ):
                cases.append((name + "_" + label, lambda op=reader, src=operand, a=offset, b=count: op(src, a, b)))
        for label, operand in (
            ("dtype", hidden_float),
            ("layout", hidden_rm),
            ("memory", hidden_l1),
            ("width", hidden_width),
            ("tile", hidden_tile),
            ("host_storage", host_hidden),
            ("released_storage", released_hidden),
        ):
            cases.append(("hidden_" + label, lambda src=operand: read_token_chunk(src, 0, 1)))
        for label, operand in (("dtype", words_float), ("width", words_width), ("memory", words_l1)):
            cases.append(("words_" + label, lambda src=operand: read_token_words(src, 0, 1)))
        cases.extend(
            [
                ("writer_alias", lambda: write_token_chunk(source, source, 0)),
                ("writer_negative_offset", lambda: write_token_chunk(source, target, -1)),
                ("writer_past_destination", lambda: write_token_chunk(source, target, 33)),
                ("writer_bool_offset", lambda: write_token_chunk(source, target, True)),
                ("writer_oversize_source", lambda: write_token_chunk(hidden, target, 0)),
                ("writer_wrong_dtype", lambda: write_token_chunk(hidden_float, target, 0)),
                ("writer_wrong_layout", lambda: write_token_chunk(hidden_rm, target, 0)),
                ("writer_wrong_memory", lambda: write_token_chunk(hidden_l1, target, 0)),
                ("writer_wrong_tile", lambda: write_token_chunk(hidden_tile, target, 0)),
                ("prepare_wrong_width", lambda: prepare_chunk_io(mesh, width=64)),
            ]
        )
        with mock.patch.object(
            ttnn, "generic_op", side_effect=AssertionError("Invalid host contract reached dispatch")
        ):
            for label, operation in cases:
                try:
                    with runtime_audit():
                        operation()
                except ValueError as error:
                    records.append({"case": label, "rejected_before_dispatch": True, "error": str(error)})
                else:
                    raise AssertionError(f"Invalid host contract accepted: {label}")
        return records
    finally:
        release(*tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--lengths", default="1,31,32,33,63,64,65,97,257,4097")
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) < 1:
        parser.error("--lengths must contain positive logical source lengths")
    if not args.native_only and not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        parser.error("Candidate trace validation requires TT_METAL_TRACE_ALLOC_TRACKING=1")

    if args.native_only:

        def read_pair(sources, offset, count):
            return sources[0][:, :, offset : offset + count, :], sources[1][:, offset : offset + count, :]

        prepare = None
    else:
        from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_input import (
            prepare_chunk_io,
            read_token_chunk,
            read_token_words,
        )

        def read_pair(sources, offset, count):
            return read_token_chunk(sources[0], offset, count), read_token_words(sources[1], offset, count)

        from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk

        prepare = prepare_chunk_io

    report = {
        "mode": "native_slice_diagnostic" if args.native_only else "candidate_chunk_input",
        "scope": "BF16 width10240 and UINT32 words16 input copies; not full decoder acceptance",
        "lengths": lengths,
        "checks": [],
        "growth": [],
        "writer_checks": [],
        "writer_growth": [],
        "negative_host_contracts": [],
        "trace": None,
        "audit": {
            "runtime_context": "TorchFunctionMode and forbidden from_torch/to_torch; throw_exception_on_fallback=True",
            "input_setup": "ttnn.from_torch only, outside runtime",
            "padding_inspection": "host-only Tensor.cpu().to_torch_with_padded_shape()",
            "offsets": "Python static plan arguments, multiples of32; dynamic trace offsets are not tested",
            "trace_allocation_tracking": bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        },
        "passed": False,
        "functional_acceptance": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 * 1024 * 1024)
    trace_id = None
    try:
        mesh.enable_program_cache()
        report["before_preparation"] = snapshot(mesh)
        if prepare is not None:
            # Preparation intentionally precedes audit: it may initialize fixed
            # scratch buffers, but no per-request setup is allowed in read_pair.
            prepare(mesh, width=WIDTH)
        sources = upload(fixtures(32), mesh)
        with runtime_audit():
            warm = read_pair(sources, 0, 32)
        report["warmup"] = check_outputs(warm, fixtures(32), 0, 32, not args.native_only)
        release(*warm, *sources)
        del warm, sources
        baseline = snapshot(mesh)
        report["baseline_after_fixed_descriptor_warmup"] = baseline
        assert report["warmup"]["passed"], "Warm-up output mismatch"

        if not args.native_only:
            # No request-specific read plan has run: only prepare() and the
            # fixed32/offset0 helper execution above. In particular, count7
            # and N97/start64 are first recorded in this capture, then executed
            # by its explicit initial replay. Capture alone does not execute.
            trace_values = [fixtures(97), fixtures(97, variant=1)]
            sources = upload(trace_values[0], mesh)
            host_changed = upload(trace_values[1])
            captured_addresses = [tensor.buffer_address() for tensor in sources]
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                with runtime_audit():
                    outputs = read_pair(sources, 64, 7)
            finally:
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            for output in outputs:
                trace_allocation_tracker.acknowledge_corruptible(output)
            # Capture records commands without execution. Populate outputs by
            # an explicit replay before any correctness inspection; this does
            # not warm the unseen plan before its first capture.
            with runtime_audit():
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            # All inspection is host-only, including physical padding.
            initial_replay_checks = check_outputs(outputs, trace_values[0], 64, 7, True)
            live_baseline = snapshot(mesh)
            eager_checks, eager_growth = [], []
            report["trace"] = {
                "source_length": 97,
                "fixed_offset": 64,
                "fixed_count": 7,
                "captures": 1,
                "initial_replay": initial_replay_checks,
                "capture_execution": "record_only",
                "total_replay_calls": 1,
                "first_capture_after_only_fixed32_setup": True,
                "exact_plan_warmed_before_capture": False,
                "exhaustive_sweep_started_before_capture": False,
                "live_baseline": live_baseline,
                "live_eager_checks": eager_checks,
                "live_eager_growth": eager_growth,
            }
            save()
            assert initial_replay_checks["passed"], "Initial explicit trace replay mismatch"
            for source_length, offset, count in ((129, 96, 17), (65, 32, 31), (257, 224, 32), (33, 32, 1)):
                eager_values = fixtures(source_length, variant=2)
                eager_sources = upload(eager_values, mesh)
                different_addresses = all(
                    tensor.buffer_address() != original for tensor, original in zip(eager_sources, captured_addresses)
                )
                assert different_addresses, "Independent live eager source must not alias captured inputs"
                target_value = torch.zeros_like(eager_values[0])
                target = ttnn.from_torch(target_value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                writer_offset = (source_length - count) // 2
                with runtime_audit():
                    eager_outputs = read_pair(eager_sources, offset, count)
                    write_token_chunk(eager_outputs[0], target, writer_offset)
                checks = check_outputs(eager_outputs, eager_values, offset, count, True)
                target_value[:, :, writer_offset : writer_offset + count] = eager_values[0][
                    :, :, offset : offset + count
                ]
                checks.update(
                    source_length=source_length,
                    offset=offset,
                    count=count,
                    independent_source_addresses=different_addresses,
                    writer_canaries_exact=torch.equal(ttnn.to_torch(target), target_value),
                )
                checks["passed"] = checks["passed"] and checks["writer_canaries_exact"] and different_addresses
                eager_checks.append(checks)
                release(*eager_outputs, target, *eager_sources)
                del eager_outputs, target, eager_sources
                state = snapshot(mesh)
                eager_growth.append(
                    {
                        "source_length": source_length,
                        "offset": offset,
                        "count": count,
                        **state,
                        "program_cache_delta": state["program_cache_entries"] - live_baseline["program_cache_entries"],
                        "persistent_dram_delta_bytes": state["allocated_bytes"] - live_baseline["allocated_bytes"],
                    }
                )
                assert checks["passed"], checks
                save()
                assert eager_growth[-1]["program_cache_delta"] == 0, eager_growth[-1]
                assert eager_growth[-1]["persistent_dram_delta_bytes"] == 0, eager_growth[-1]
            trace_checks = []
            for replay in range(3):
                if replay == 1:
                    for host, source in zip(host_changed, sources):
                        ttnn.copy_host_to_device_tensor(host, source)
                assert [tensor.buffer_address() for tensor in sources] == captured_addresses
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                checks = check_outputs(outputs, trace_values[int(replay > 0)], 64, 7, True)
                checks.update(replay=replay, changed_input=replay > 0)
                trace_checks.append(checks)
            report["trace"] = {
                "source_length": 97,
                "fixed_offset": 64,
                "fixed_count": 7,
                "captures": 1,
                "replays": trace_checks,
                "initial_replay": initial_replay_checks,
                "capture_execution": "record_only",
                "total_replay_calls": 4,
                "first_capture_after_only_fixed32_setup": True,
                "exact_plan_warmed_before_capture": False,
                "exhaustive_sweep_started_before_capture": False,
                "captured_input_addresses_stable": [tensor.buffer_address() for tensor in sources]
                == captured_addresses,
                "live_eager_checks": eager_checks,
                "live_eager_growth": eager_growth,
                "live_eager_growth_stable": all(
                    row["program_cache_delta"] == 0 and row["persistent_dram_delta_bytes"] == 0 for row in eager_growth
                ),
                "repeated_input_exact": trace_checks[1]["passed"] and trace_checks[2]["passed"],
                "acknowledgment": "Exactly the two captured outputs, consumed before replay; no blanket acknowledgment",
            }
            assert initial_replay_checks["passed"] and all(row["passed"] for row in trace_checks), "Trace mismatch"
            ttnn.release_trace(mesh, trace_id)
            trace_id = None
            release(*outputs, *sources)
            del outputs, sources, host_changed
            report["after_trace_release"] = snapshot(mesh)
            report["first_trace_released_persistent_growth_stable"] = all(
                report["after_trace_release"][key] == baseline[key]
                for key in ("program_cache_entries", "allocated_bytes")
            )
            assert report["first_trace_released_persistent_growth_stable"]
            save()

        # Preserve the exhaustive copy/padding proof, now strictly AFTER the
        # first unseen-plan capture and live eager/replay challenges.
        for length in lengths:
            values = fixtures(length)
            for offset in offsets(length):
                for count in range(1, min(32, length - offset) + 1):
                    sources = upload(values, mesh)
                    with runtime_audit():
                        outputs = read_pair(sources, offset, count)
                    checks = check_outputs(outputs, values, offset, count, not args.native_only)
                    checks.update(source_length=length, offset=offset, count=count)
                    report["checks"].append(checks)
                    if not args.native_only:
                        target_value = torch.zeros_like(values[0])
                        target = ttnn.from_torch(
                            target_value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                        )
                        writer_offset = (length - count) // 2
                        before_writer = snapshot(mesh)
                        with runtime_audit():
                            write_token_chunk(outputs[0], target, writer_offset)
                        after_writer = snapshot(mesh)
                        target_value[:, :, writer_offset : writer_offset + count] = values[0][
                            :, :, offset : offset + count
                        ]
                        writer_check = {
                            "source_length": length,
                            "source_offset": offset,
                            "destination_offset": writer_offset,
                            "count": count,
                            "exact_equal_with_canaries": torch.equal(ttnn.to_torch(target), target_value),
                        }
                        report["writer_checks"].append(writer_check)
                        report["writer_growth"].append(
                            {
                                **writer_check,
                                "program_cache_delta": after_writer["program_cache_entries"]
                                - before_writer["program_cache_entries"],
                                "persistent_dram_delta_bytes": after_writer["allocated_bytes"]
                                - before_writer["allocated_bytes"],
                                "largest_contiguous_bytes_per_bank": after_writer["largest_contiguous_bytes_per_bank"],
                            }
                        )
                        release(target)
                        del target
                        assert writer_check["exact_equal_with_canaries"], writer_check
                    release(*outputs, *sources)
                    del outputs, sources
                    state = snapshot(mesh)
                    growth = {
                        "source_length": length,
                        "offset": offset,
                        "count": count,
                        **state,
                        "program_cache_delta": state["program_cache_entries"] - baseline["program_cache_entries"],
                        "persistent_dram_delta_bytes": state["allocated_bytes"] - baseline["allocated_bytes"],
                    }
                    report["growth"].append(growth)
                    save()
                    assert checks["passed"], f"Copy mismatch: {checks}"
            print("CHUNK_INPUT_LENGTH", length, json.dumps(report["growth"][-1]), flush=True)

        report["growth_stable"] = all(
            row["program_cache_delta"] == 0 and row["persistent_dram_delta_bytes"] == 0 for row in report["growth"]
        )
        if args.native_only:
            report["diagnostic_completed"] = True
            report["diagnostic_note"] = "Growth is measured, not an acceptance failure; no candidate trace claim"
        else:
            report["writer_growth_stable"] = all(
                row["program_cache_delta"] == 0 and row["persistent_dram_delta_bytes"] == 0
                for row in report["writer_growth"]
            )
            report["writer_counts_checked"] = sorted({row["count"] for row in report["writer_checks"]})
            assert report["writer_counts_checked"] == list(range(1, 33))
            assert report["growth_stable"] and report["writer_growth_stable"], "Helpers retained programs/DRAM"
            report["negative_host_contracts"] = negative_host_contracts(mesh)
            report["negative_host_contracts_passed"] = bool(report["negative_host_contracts"]) and all(
                row["rejected_before_dispatch"] for row in report["negative_host_contracts"]
            )
            assert report["negative_host_contracts_passed"]
            report["passed"] = True
            report["functional_acceptance"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
        save()
    print(json.dumps({key: value for key, value in report.items() if key not in ("checks", "growth")}), flush=True)


if __name__ == "__main__":
    main()
