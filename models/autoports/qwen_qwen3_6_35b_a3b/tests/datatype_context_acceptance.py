# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ONE selected-datatype maximum-context acceptance pass. Do not use as a sweep.

Run only if the selected KV-cache dtype/layout changed. The normal default
build_generator path must load selected_precision_config.json. Historical
full-model plans remain historical evidence, never current policy assertions.

Run only after short correctness/serving gates, on an idle assigned TP4 mesh:
 timeout 14400 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_context_acceptance

The 262143-token prefill fills positions 0..262142. Its device-sampled token
is decoded at position 262143, filling the final legal position of context
262144. A 262144-token prefill followed by another decode would exceed that
contract. Preparation uses short fixtures outside the one long acceptance pass.
Existing output is rejected to prevent accidentally repeating acceptance;
select a fresh artifact explicitly only after investigating a failed run.
This runner never changes context_contract.json or retries at a smaller context.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 262144
TRACE_BYTES = 200000000


def config_arithmetic(config, precision):
    kinds = config["layer_types"]
    full, linear = kinds.count("full_attention"), kinds.count("linear_attention")
    hidden, vocab, heads = config["hidden_size"], config["vocab_size"], config["num_key_value_heads"]
    head = config["head_dim"]
    pages = ((CONTEXT + 255) // 256) * 8
    # Accepted TP4 duplicates each of two KV heads on two ranks; each chip
    # owns one head. Do not divide the per-chip KV formula by four again.
    local_kv_heads = math.ceil(heads / 4)
    conv_width = (
        2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
        + config["linear_num_value_heads"] * config["linear_value_head_dim"]
    ) // 4
    conv = linear * (config["linear_conv_kernel_dim"] - 1) * conv_width * 2
    recurrent = (
        linear
        * (config["linear_num_value_heads"] // 4)
        * config["linear_key_head_dim"]
        * config["linear_value_head_dim"]
        * 4
    )
    page_tables = full * pages * 4
    padded_vocab = (vocab + 127) // 128 * 128
    rope_dim = int(head * config.get("partial_rotary_factor", config["rope_parameters"]["partial_rotary_factor"]))
    tile_bytes = {"bf16": 2048, "bfp8": 1088, "bfp4": 576}
    kv_tile_bytes = tile_bytes[precision["kv_cache_dtype"]]
    kv_tiles = full * 2 * pages * local_kv_heads * math.ceil(head / 32)
    return {
        "scope": "Current selected policy per chip, batch1; packed tile arithmetic excludes allocator padding and program-owned buffers",
        "full_attention_layers": full,
        "linear_attention_layers": linear,
        "page_size": 32,
        "table_entries_per_layer": pages,
        "local_kv_heads": local_kv_heads,
        "kv_dtype": precision["kv_cache_dtype"],
        "kv_layout": precision["kv_layout"],
        "kv_update_semantics": "Prefill casts to backing dtype; decode uses native update/repack; FP32 recurrent backing is distinct",
        "kv_tile_payload_bytes": kv_tile_bytes,
        "kv_tiles": kv_tiles,
        "kv_bytes": kv_tiles * kv_tile_bytes,
        "historical_bf16_kv_bytes_same_context": kv_tiles * 2048,
        "kv_bytes_saved_vs_historical_bf16": kv_tiles * (2048 - kv_tile_bytes),
        "page_table_bytes": page_tables,
        "recurrent_fp32_bytes": recurrent,
        "conv_bf16_bytes": conv,
        "reset_recurrent_conv_bytes": recurrent + conv,
        "slot_recurrent_conv_page_position_activity_snapshot_bytes": recurrent + conv + page_tables + 8,
        "prefill_three_residual_workspaces_bytes": 3 * CONTEXT * hidden * 2,
        "shared_rope_bf16_bytes": 2 * CONTEXT * rope_dim * 2,
        "embedding_hidden_shard_bf16_bytes": vocab * (hidden // 4) * 2,
        "lm_head_weight_dtype": precision["weight_groups"]["lm_head"],
        "lm_head_vocab_shard_tile_bytes": (hidden // 32)
        * ((padded_vocab // 4) // 32)
        * tile_bytes[precision["weight_groups"]["lm_head"]],
        "terminal_norm_fp32_tiled_bytes": 32 * hidden * 4,
        "sampler_logits_bf16_tiled_bytes": 32 * (padded_vocab // 4) * 2,
        "sampler_zero_row_bf16_tiled_bytes": 32 * (padded_vocab // 4) * 2,
        "output_pool_two_logits_two_token_snapshots_bytes": 2 * 32 * (padded_vocab // 4) * 2 + 2 * 32 * 4,
        "trace_region_reserved_bytes": TRACE_BYTES,
        "remaining_terms": [
            "actual retained decoder weights and helpers",
            "common sampler and penalties buffers",
            "CCL semaphores and scratch",
            "prefill helper buffers",
            "slot/control vectors",
            "prepared MoE workspaces",
            "trace storage",
            "allocator alignment and temporary peak",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/datatype_sweep/context_acceptance/result.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Acceptance artifact already exists; refusing a repeated long run: {args.output}")
    assert not os.environ.get("TT_METAL_WATCHER"), "Use a separate watcher run"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1", "Acceptance is not a profiler run"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_runtime_identity import (
        runtime_identity,
        source_input_identity,
    )
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_MODEL, HF_REVISION
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import SELECTED_PATH, device_dtype, resolve_precision

    assert SELECTED_PATH.is_file(), "Select and publish the precision artifact before acceptance"
    selected_precision = resolve_precision()
    selected_sha256 = hashlib.sha256(SELECTED_PATH.read_bytes()).hexdigest()
    assert selected_precision["kv_layout"] == "TILE"

    contract_path = ROOT / "doc/context_contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["hf_advertised_context"] == contract["supported_context"] == CONTEXT
    assert contract["capability_reduction"] is None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), ROOT / "tests/runtime_audit.py"] + [
        ROOT / "tt" / name
        for name in (
            "model.py",
            "generator.py",
            "prefill.py",
            "prefill_transport.py",
            "multichip_decoder.py",
            "slot_state.py",
            "serving_state.py",
            "precision.py",
        )
    ]
    provenance_inputs = (
        Path(__file__),
        SELECTED_PATH,
        contract_path,
        ROOT / "doc/full_model/hf_reference/aime24_prompt_tokens.json",
    )
    result = {
        "status": "RUNNING",
        "passed": False,
        "command": sys.argv,
        "time_unix": time.time(),
        "model": HF_MODEL,
        "revision": HF_REVISION,
        "context": CONTEXT,
        "batch_size": 1,
        "selected_precision_artifact": str(SELECTED_PATH),
        "selected_precision_sha256": selected_sha256,
        "construction_path": "build_generator(ROOT, mesh, max_seq_len=262144, max_batch_size=1, host_sampling=False); no precision override",
        "historical_contract_policy_sections": {
            key: contract.get(key)
            for key in ("multichip_decoder_plan", "optimized_multichip_decoder_validation", "full_model_plan")
        },
        "historical_contract_note": "These earlier plans describe their original stage. Current KV/CCL policy is runtime_precision and arithmetic in this artifact.",
        "prefill_length": CONTEXT - 1,
        "decode_position": CONTEXT - 1,
        "long_prefill_calls": 0,
        "acceptance_decode_calls": 0,
        "capability_reduction": None,
        "boundary_explanation": "Prefill fills 0..262142; one sampled token is consumed at final valid position262143. Post-decode next-position262144 is a sentinel, never executed.",
        "contract_sha256_before": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "native_runtime_identity": runtime_identity(),
        "source_input_identity": source_input_identity(model_root=ROOT, explicit_paths=provenance_inputs),
        "native_identity_capture": "Actual acceptance process, before mesh construction and after teardown",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
            )
        },
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    model = generator = mesh = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=TRACE_BYTES)
        generator = build_generator(ROOT, mesh, max_seq_len=CONTEXT, max_batch_size=1, host_sampling=False)
        model = generator.model
        assert model.precision_config == selected_precision, "Normal construction did not consume selected artifact"
        assert tuple(model.layer_indices) == tuple(range(40)) and model.config.max_position_embeddings == CONTEXT
        result["runtime_precision"] = model.precision_summary()
        result["arithmetic"] = config_arithmetic(model.config.to_dict(), selected_precision)
        generator.configure_sampling(temperature=0.0, top_k=1, top_p=1.0, seed=0)
        assert generator.prefill_executor.workspace_bytes_per_chip == 3 * 1024**3
        full_groups = [
            (index, states[0]) for index, states in zip(model.layer_indices, generator.cache) if "k" in states[0]
        ]
        assert len(full_groups) == 10
        for _, state in full_groups:
            assert state["k"].dtype == state["v"].dtype == device_dtype(selected_precision["kv_cache_dtype"])
            assert tuple(state["k"].shape) == tuple(state["v"].shape) == (8192, 1, 32, 256)
            assert tuple(state["page_table"].shape) == (1, 8192)
        for layer in model.layers:
            assert layer.residual_layout == "replicated" and layer.ccl_dtype == selected_precision["ccl_dtype"]
        result["actual_kv_buffer_bytes_per_device"] = sum(
            int(state[name].buffer_page_size()) * int(state[name].buffer_num_pages())
            for _, state in full_groups
            for name in ("k", "v")
        )
        assert result["actual_kv_buffer_bytes_per_device"] == result["arithmetic"]["kv_bytes"]
        result["precision_policies"] = {
            str(index): layer.precision_policy for index, layer in zip(model.layer_indices, model.layers)
        }

        def last_pages():
            snapshots = {}
            for index, state in full_groups:
                pages = state["_host_pages"].flatten()
                assert int(pages[-1]) == 8191 and len(pages.unique()) == 8192
                physical = int(pages[-1])
                for name in ("k", "v"):
                    tail = state[name][physical : physical + 1, :, :, :]
                    try:
                        snapshots[f"layer{index}.{name}"] = [
                            ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(tail)
                        ]
                    finally:
                        ttnn.deallocate(tail)
            return snapshots

        def position_value():
            return int(ttnn.to_torch(ttnn.get_device_tensors(generator.positions[0])[0]).flatten()[0])

        def seed_prefill(ids, *, captured):
            logits = generator.prefill_forward(ids, prompt_lens=[ids.shape[-1]], device_output=True)[0]
            values = model.host_logits(logits)
            assert values.shape == (1, 1, model.vocab_size) and torch.isfinite(values).all()
            logits = ttnn.reshape(logits, (1, 1, 1, logits.shape[-1]))
            ttnn.copy(ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], 0), generator.sampler_logits)
            ttnn.deallocate(logits)
            generator._copy_host(torch.tensor([ids.shape[-1]], dtype=torch.int32), generator.positions[0])
            generator.counters["position_refreshes"] += 1
            generator._sample() if captured else generator._sampling_forward()
            token = generator._read_tokens([0])[0]
            assert 0 <= token < model.vocab_size
            return token

        # All diagnostic readback kernels (last cache page and both logits
        # shapes) are warmed before capture; no long-context traversal here.
        last_pages()
        for states in generator.cache:
            for name in ("conv", "recurrent"):
                if name in states[0]:
                    for part in ttnn.get_device_tensors(states[0][name]):
                        ttnn.to_torch(part)
        position_value()
        model.host_logits(generator.sampler_logits)
        seed_prefill(torch.tensor([[1, 2]], dtype=torch.int64), captured=False)
        generator.prepare(prefill_cases=((32, 0),))
        generator.reset()
        ttnn.synchronize_device(mesh)
        prepared_entries = int(mesh.num_program_cache_entries())
        result["program_cache_entries_after_prepare"] = prepared_entries
        result["trace_ids"] = {key: int(value) for key, value in generator.traces.items()}
        result["prepared_census"] = model.census("max_context_prepared_with_live_traces")

        # Inventory real allocations by their root buffer ID, not Python object
        # identity: cache aliases and immutable tensors are counted only once.
        seen_objects, seen_buffers, summaries = set(), set(), {}

        def visit(value, category):
            if isinstance(value, ttnn.Tensor):
                uid = value.buffer_unique_id()
                if uid is None or uid in seen_buffers:
                    return
                seen_buffers.add(uid)
                key = f"{category}/{value.memory_config().buffer_type}/{value.dtype}"
                row = summaries.setdefault(key, {"buffers": 0, "packed_page_bytes": 0, "aligned_page_bytes": 0})
                row["buffers"] += 1
                row["packed_page_bytes"] += int(value.buffer_page_size()) * int(value.buffer_num_pages())
                row["aligned_page_bytes"] += int(value.buffer_aligned_page_size()) * int(value.buffer_num_pages())
                return
            # Model/helper instances can implement __call__ while owning real
            # tensors. The module allowlist below excludes functions/methods
            # without accidentally dropping those callable owners.
            if isinstance(value, (str, bytes, int, float, bool, type(None), torch.Tensor)):
                return
            if id(value) in seen_objects:
                return
            seen_objects.add(id(value))
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, category)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, category)
            elif hasattr(value, "__dict__") and type(value).__module__.startswith("models."):
                for name, child in vars(value).items():
                    if name not in {"model", "mesh", "mesh_device", "workspace", "config", "checkpoint", "tokenizer"}:
                        visit(child, category)

        for category, value in (
            ("prefill_workspaces_and_helpers", generator.prefill_executor),
            ("shared_cache_and_pages", generator.cache),
            ("reset_sources", generator.zeros),
            ("slot_snapshots", generator.slot_state),
            ("output_pool", generator.output_pool),
            ("sampler", generator.sampler),
            ("decoder_weights_and_helpers", model.layers),
            ("model_embeddings_head_rope", model),
            ("ccl_workspace", model.workspace),
            ("generator_controls", generator),
        ):
            visit(value, category)
        result["actual_unique_tensor_allocation_inventory"] = summaries
        result[
            "inventory_note"
        ] = "Unique root mesh buffers; page payload/alignment arithmetic per device. Memory census also covers program-owned allocations, bank distribution, live traces and non-tensor storage; do not add inventory to census. Trace region is reported separately to avoid double-counting allocator reservations."
        census = result["prepared_census"]
        result["measured_dram"] = {
            "allocated_bytes_per_device": census["banks"] * census["total_bytes_allocated_per_bank"],
            "free_bytes_per_device": census["banks"] * census["total_bytes_free_per_bank"],
            "allocator_capacity_bytes_per_device": census["banks"] * census["total_bytes_per_bank"],
            "reserved_trace_region_bytes_per_device": TRACE_BYTES,
            "scope": f"Current selected all40 model, {selected_precision['kv_cache_dtype']} KV, recurrent/reset/slot snapshots, sampler/output buffers, 3GiB prefill workspace and live traces",
        }
        assert result["measured_dram"]["free_bytes_per_device"] > 0
        old_plan = contract.get("full_model_plan", {})
        result["historical_plan_vs_current_runtime"] = {
            "prior_plan_status": old_plan.get("status"),
            "prior_retained_decoder_estimate": old_plan.get("retained_decoder_weight_bytes_per_device"),
            "actual_loaded_weights_and_rope_census": next(
                row for row in model.memory_views if row["stage"] == "all_weights_and_rope"
            ),
            "historical_policy_is_not_current_policy": True,
            "current_kv_dtype": selected_precision["kv_cache_dtype"],
            "current_ccl_dtype": selected_precision["ccl_dtype"],
            "current_precision_config_id": selected_precision["config_id"],
            "supported_context_unchanged": True,
            "corrections": [
                "Large KV pools reset in place, with no full KV zero mirror",
                "Recurrent/conv reset mirrors and slot snapshots are separate",
                "Three replicated BF16 prefill workspaces total3GiB per chip",
                "Prepared sampler/output/CCL/trace buffers included in actual census",
            ],
        }
        # Repeat pinned real-token AIME input to exercise physical capacity;
        # this synthetic repeated prompt has no qualitative completion claim.
        fixture_path = ROOT / "doc/full_model/hf_reference/aime24_prompt_tokens.json"
        base = json.loads(fixture_path.read_text())
        ids = torch.tensor([(base * ((CONTEXT - 2) // len(base) + 1))[: CONTEXT - 1]], dtype=torch.int64)
        result["fixture"] = {
            "source": str(fixture_path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "kind": "repeated real-token capacity fixture",
            "length": ids.shape[-1],
            "qualitative_use": False,
        }
        save()
        print("MAX_CONTEXT_ACCEPTANCE_BEGIN", json.dumps(result["measured_dram"]), flush=True)
        result["long_prefill_calls"] += 1
        save()
        began = time.perf_counter()
        initial_token = seed_prefill(ids, captured=True)
        ttnn.synchronize_device(mesh)
        result["max_prefill_seconds"] = time.perf_counter() - began
        assert position_value() == CONTEXT - 1
        before_pages = last_pages()
        for key, chips in before_pages.items():
            assert len(chips) == 4
            for chip in chips:
                assert torch.isfinite(chip).all(), key
                assert (torch.count_nonzero(chip[..., :31, :], dim=-1) > 0).all(), (
                    key,
                    "Prefill left an unpopulated visible cache row",
                )
                assert torch.count_nonzero(chip[..., 31:, :]) == 0, (key, "Prefill wrote beyond logical prompt")
        assert int(mesh.num_program_cache_entries()) == prepared_entries
        result["prefill_last_page"] = {
            "all10_layers_all4_chips": True,
            "rows0_to30_finite_nonzero": True,
            "unfilled_row31_exact_zero": True,
        }
        model.census("after_max_prefill")
        save()
        before = dict(generator.counters)
        result["acceptance_decode_calls"] += 1
        save()
        with device_only():
            began = time.perf_counter()
            generator.decode_forward(None, [CONTEXT - 1], refresh_inputs=False, read_tokens=False)
            ttnn.synchronize_device(mesh)
            decode_seconds = time.perf_counter() - began
        counters = {key: generator.counters[key] - value for key, value in before.items()}
        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "readbacks", "captures"):
            assert counters[key] == 0, counters
        assert counters["model_replays"] == counters["sampling_replays"] == 1
        assert position_value() == CONTEXT
        final_token = generator._read_tokens([0])[0]
        assert 0 <= final_token < model.vocab_size
        final_logits = model.host_logits(generator.sampler_logits)[0, 0, 0, :]
        assert torch.isfinite(final_logits).all()
        after_pages = last_pages()
        for key, chips in after_pages.items():
            for index, chip in enumerate(chips):
                assert torch.isfinite(chip).all() and torch.count_nonzero(chip[..., 31:, :]) > 0, key
                assert torch.equal(chip[..., :31, :], before_pages[key][index][..., :31, :]), (
                    key,
                    "Decode changed preceding cache rows",
                )
        linear_checks = 0
        for states in generator.cache:
            for name in ("conv", "recurrent"):
                if name in states[0]:
                    for part in ttnn.get_device_tensors(states[0][name]):
                        assert torch.isfinite(ttnn.to_torch(part)).all(), name
                        linear_checks += 1
        assert int(mesh.num_program_cache_entries()) == prepared_entries
        result["decode"] = {
            "seconds": decode_seconds,
            "input_device_sampled_token": initial_token,
            "output_device_sampled_token": final_token,
            "position_after": CONTEXT,
            "loop_counter_delta": counters,
            "runtime_device_only_guard": "pass",
            "finite_logits": True,
            "linear_state_chip_checks": linear_checks,
            "final_cache_row31_written_all_layers_all_chips": True,
            "preceding_cache_rows_unchanged": True,
            "token_feedback": "Prefill sampling trace writes generator.tokens; model trace consumes that stable tensor with tokens=None and zero host refreshes; sampling trace overwrites it with the next token.",
        }
        assert result["long_prefill_calls"] == result["acceptance_decode_calls"] == 1
        model.census("after_max_context_decode")
        result["status"] = "PASS"
        result["passed"] = True
        print("MAX_CONTEXT_ACCEPTANCE_OK", flush=True)
    except BaseException:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        traceback.print_exc()
        # Preserve the acceptance failure before teardown touches device state.
        try:
            save()
        except BaseException:
            traceback.print_exc()
        raise
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_errors = []
        try:
            if model is not None:
                result["memory_views"] = model.memory_views
            if generator is not None:
                result["counters_before_release"] = dict(generator.counters)
                try:
                    generator.close()
                except BaseException:
                    cleanup_errors.append({"operation": "generator.close", "error": traceback.format_exc()})
                result["counters_after_release"] = dict(generator.counters)
            if mesh is not None:
                try:
                    ttnn.close_mesh_device(mesh)
                except BaseException:
                    cleanup_errors.append({"operation": "close_mesh_device", "error": traceback.format_exc()})
            if cleanup_errors:
                result["cleanup_errors"] = cleanup_errors
                result["status"] = "FAIL"
                result["passed"] = False
        finally:
            try:
                result["native_runtime_identity_after"] = runtime_identity()
                result["source_input_identity_after"] = source_input_identity(
                    model_root=ROOT, explicit_paths=provenance_inputs
                )
                if result["native_runtime_identity_after"] != result["native_runtime_identity"]:
                    raise RuntimeError("Native source/build identity changed during acceptance")
                if result["source_input_identity_after"] != result["source_input_identity"]:
                    raise RuntimeError("Source/input identity changed during acceptance")
            except Exception as exc:
                result["status"] = "FAIL"
                result["passed"] = False
                cleanup_errors.append({"operation": "native_provenance", "error": str(exc)})
            result["contract_sha256_after"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            result["selected_precision_sha256_after"] = hashlib.sha256(SELECTED_PATH.read_bytes()).hexdigest()
            if (
                result["contract_sha256_after"] != result["contract_sha256_before"]
                or result["selected_precision_sha256_after"] != selected_sha256
            ):
                result["status"] = "FAIL"
                result["passed"] = False
                cleanup_errors.append(
                    {"operation": "provenance", "error": "Contract or selected policy changed during acceptance"}
                )
            result["elapsed_seconds"] = time.time() - result["time_unix"]
            try:
                save()
            except BaseException:
                if not original_exception_pending:
                    raise
                traceback.print_exc()
        if cleanup_errors and not original_exception_pending:
            raise RuntimeError("Context acceptance cleanup failed; see artifact cleanup_errors")


if __name__ == "__main__":
    main()
