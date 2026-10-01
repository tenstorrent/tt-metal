# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent native-input layer gates and retained full-model output checks."""

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.device_utils import free_cache, snapshot
from models.demos.llama_3p1_8b_d_p.tests.model import device_utils
from models.demos.llama_3p1_8b_d_p.tests.model.reference import (
    chat_tokens,
    read_raw_weights,
    reference_layer,
    reference_prefill,
)
from models.demos.llama_3p1_8b_d_p.tests.utils import (
    CHUNKS,
    FULL_LIMITS,
    LAYER_NAMES,
    assemble_head,
    check_metric,
    full_hidden,
    join_hidden_tp_replicas,
    local_limits,
    metrics,
    score_row,
    selected_logit_positions,
    validate_observer_order,
    windows,
    write_pcc_summary,
)
from models.demos.llama_3p1_8b_d_p.tt.input import upload_token_chunk
from models.demos.llama_3p1_8b_d_p.tt.model import PrefillModel
from models.demos.llama_3p1_8b_d_p.tt.weights import CheckpointWeights


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_hidden(value):
    assert tuple(value.shape) == (1, 1, 256, 4096)
    assert value.dtype == ttnn.bfloat16 and value.layout == ttnn.TILE_LAYOUT
    assert value.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    shards = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]
    assert all(torch.isfinite(shard).all() for shard in shards)
    return join_hidden_tp_replicas(shards)


@pytest.fixture(scope="module")
def llama_checkpoint():
    """Use an explicit local checkpoint path instead of an evidence-specific machine layout."""
    configured = os.environ.get("LLAMA31_8B_CHECKPOINT")
    if not configured:
        pytest.fail("Set LLAMA31_8B_CHECKPOINT to a local Llama-3.1-8B-Instruct checkpoint", pytrace=False)
    checkpoint = Path(configured).expanduser().resolve()
    if not (checkpoint / "model.safetensors.index.json").is_file():
        pytest.fail(f"Missing checkpoint index: {checkpoint}", pytrace=False)
    return checkpoint


def _make_reference(checkpoint, slot, length, num_layers, *, user_text=None):
    ids, prompt = chat_tokens(checkpoint, slot=slot, length=length, user_text=user_text)
    length = len(ids)
    reference = reference_prefill(
        checkpoint, ids, num_layers=num_layers, selected_logit_positions=selected_logit_positions(length)
    )
    return ids, reference, prompt


# Capture actual embedding/layer outputs for two complete chunks in each slot. Reconstruct each
# layer's full native input and score an independent causal 2K reference at the existing decoder
# gates. Raw-global drift stays visible. Final logits and all exact cache/slot invariants stay
# hard, and an uninstrumented replay must preserve the instrumented final states and decoded cache.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("prompt_case", ["baseline", "held_out"])
def test_prefill_all_layers_from_native_inputs(llama_checkpoint, mesh_device, cache_dtype, prompt_case, tmp_path):
    dtype_name = "bfloat16" if cache_dtype == ttnn.bfloat16 else "bfloat8_b"
    directory = Path(os.environ.get("LLAMA_PREFILL_EVIDENCE_DIR", str(tmp_path))) / prompt_case / dtype_name
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="started",
        coverage="local_native_input_and_global_semantics",
        case_passed=False,
        result_scope="single native-input case; full 2K acceptance also requires the boundary cases and selected fixture/dtype set",
        cache_dtype=dtype_name,
        prompt_case=prompt_case,
        checkpoint=str(llama_checkpoint),
        records=[],
        local_rows=[],
        raw_global_summary_rows=[],
        final_global_rows=[],
        completed_calls=[],
        baseline_calls=[],
        replay_calls=[],
        loader_identities={},
        timing_scope="diagnostic_with_readbacks",
        boundary_restart_companion_required=True,
        counts={},
        instrumentation="synchronous host readbacks",
    )
    model = cache = output = tokens = None
    references = []
    context = {}
    original_embedding_call = None
    try:
        checkpoint = llama_checkpoint
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        checkpoint_files = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
        report["checkpoint_metadata"] = {name: sha(checkpoint / name) for name in checkpoint_files}
        report["checkpoint_shard_stats"] = {
            name: [(checkpoint / name).stat().st_size, (checkpoint / name).stat().st_mtime_ns]
            for name in sorted(set(index["weight_map"].values()))
        }
        user_text = None
        fixture_path = None
        if prompt_case == "held_out":
            fixture_path = Path(__file__).parent / "fixtures" / "held_out_operations_chat.txt"
            fixture_bytes = fixture_path.read_bytes()
            user_text = fixture_bytes.decode("utf-8")
            report["held_out_fixture"] = dict(
                path=str(fixture_path),
                sha256=hashlib.sha256(fixture_bytes).hexdigest(),
                bytes=len(fixture_bytes),
                slot=0,
                template_applications=1,
                required_prefix_tokens=2048,
            )
        references = [
            _make_reference(checkpoint, slot, 2048, 32, user_text=user_text if slot == 0 else None) for slot in (0, 1)
        ]
        assert not torch.equal(references[0][0], references[1][0])
        for slot, (ids, _, prompt) in enumerate(references):
            torch.save(ids, directory / f"tokens-slot{slot}.pt")
            dump(directory / f"prompt-slot{slot}.json", prompt)

        loader_order = []
        original_load = CheckpointWeights.layer

        def load_spy(instance, layer_idx):
            raw = original_load(instance, layer_idx)
            loader_order.append(layer_idx)
            report["loader_identities"][str(layer_idx)] = {name: tensor_sha(value) for name, value in raw.items()}
            return raw

        with patch.object(CheckpointWeights, "layer", load_spy):
            model = PrefillModel(mesh_device, checkpoint, cache_dtype=cache_dtype)
        assert loader_order == list(range(32))
        assert model.num_layers == len(model.layers) == 32
        assert [layer.layer_idx for layer in model.layers] == list(range(32))
        assert len({id(layer) for layer in model.layers}) == 32
        cache = device_utils.seed_cache(mesh_device, model, cache_dtype)
        # Baseline precedes capture. Its forward has no embedding hook or layer observer.
        for slot, start, end in CHUNKS:
            ids, golden, _ = references[slot]
            tokens = upload_token_chunk(mesh_device, ids[start:end], actual_start=start, actual_end=end)
            try:
                output = model.prefill_chunk(
                    tokens, cache, slot_idx=slot, actual_start=start, actual_end=end, skip_lm_head=True
                )
                ttnn.synchronize_device(mesh_device)
                actual = read_hidden(output)
                for sp in range(4):
                    expected = golden["layers"][-1]["hidden"][start + sp * 256 : start + (sp + 1) * 256]
                    report["final_global_rows"].append(
                        score_row(
                            metrics,
                            expected,
                            actual[sp * 256 : (sp + 1) * 256],
                            FULL_LIMITS,
                            phase="baseline",
                            slot=slot,
                            start=start,
                            sp=sp,
                            layer=31,
                        )
                    )
                torch.save(actual, directory / f"baseline-hidden-s{slot}-c{start}.pt")
                logits_device = model.head(output)
                try:
                    logits = device_utils.check_logits(
                        logits_device,
                        golden,
                        start=start,
                        end=end,
                        num_layers=32,
                        dtype=cache_dtype,
                        records=report["records"],
                    )
                    torch.save(logits, directory / f"baseline-logits-s{slot}-c{start}.pt")
                finally:
                    logits_device.deallocate(True)
                report["baseline_calls"].append(dict(slot=slot, start=start, end=end))
            finally:
                if output is not None:
                    output.deallocate(True)
                    output = None
                if tokens is not None:
                    tokens.deallocate(True)
                    tokens = None
        baseline_cache = dict(zip(("k", "v"), snapshot(cache)))
        free_cache(cache)
        cache = None
        cache = device_utils.seed_cache(mesh_device, model, cache_dtype)
        original_embedding_call = type(model.embedding).__call__

        def capture_embedding(instance, *args, **kwargs):
            assert instance is model.embedding
            result = original_embedding_call(instance, *args, **kwargs)
            try:
                natural = read_hidden(result)
                torch.save(natural, directory / f"embedding-s{context['slot']}-c{context['start']}.pt")
                return result
            except BaseException:
                result.deallocate(True)
                raise

        with patch.object(type(model.embedding), "__call__", capture_embedding):
            for slot, start, end in CHUNKS:
                context.update(slot=slot, start=start)
                ids, golden, _ = references[slot]
                tokens = upload_token_chunk(mesh_device, ids[start:end], actual_start=start, actual_end=end)
                input_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tokens)]
                before = snapshot(cache)
                observed = []

                def observe(layer_idx, hidden):
                    observed.append(layer_idx)
                    actual = read_hidden(hidden)
                    torch.save(actual, directory / f"hidden-s{slot}-l{layer_idx}-c{start}.pt")
                    if layer_idx == 31:
                        assert torch.equal(
                            actual, torch.load(directory / f"baseline-hidden-s{slot}-c{start}.pt", weights_only=True)
                        )
                    for sp in range(4):
                        sl = slice(sp * 256, (sp + 1) * 256)
                        check_metric(
                            golden["layers"][layer_idx]["hidden"][start + sp * 256 : start + (sp + 1) * 256],
                            actual[sl],
                            device_utils.hidden_limits(layer_idx + 1, cache_dtype),
                            f"layer output slot={slot} layer={layer_idx} chip={sp*8}",
                            report["records"],
                            enforce=layer_idx == 0,
                        )
                        if layer_idx == 31:
                            report["final_global_rows"].append(
                                dict(
                                    report["records"][-1], phase="instrumented", slot=slot, start=start, sp=sp, layer=31
                                )
                            )

                try:
                    ttnn.synchronize_device(mesh_device)
                    begin = time.perf_counter()
                    output = model.prefill_chunk(
                        tokens,
                        cache,
                        slot_idx=slot,
                        actual_start=start,
                        actual_end=end,
                        skip_lm_head=False,
                        layer_observer=observe,
                    )
                    ttnn.synchronize_device(mesh_device)
                    validate_observer_order(observed)
                    for actual, expected in zip(ttnn.get_device_tensors(tokens), input_before):
                        assert torch.equal(ttnn.to_torch(actual), expected)
                    device_utils.check_cache(
                        cache,
                        before,
                        golden,
                        slot=slot,
                        start=start,
                        end=end,
                        records=report["records"],
                        enforce_accumulated=False,
                    )
                    logits = device_utils.check_logits(
                        output,
                        golden,
                        start=start,
                        end=end,
                        num_layers=32,
                        dtype=cache_dtype,
                        records=report["records"],
                    )
                    torch.save(logits, directory / f"logits-s{slot}-c{start}.pt")
                    assert torch.equal(
                        logits, torch.load(directory / f"baseline-logits-s{slot}-c{start}.pt", weights_only=True)
                    )
                    del before, input_before
                    report["completed_calls"].append(
                        dict(
                            slot=slot,
                            start=start,
                            end=end,
                            layers=observed,
                            synchronized_wall_seconds=time.perf_counter() - begin,
                        )
                    )
                finally:
                    if output is not None:
                        output.deallocate(True)
                        output = None
                    if tokens is not None:
                        tokens.deallocate(True)
                        tokens = None
                dump(directory / "report.json", report)

        native_cache = dict(zip(("k", "v"), snapshot(cache)))
        assert all(torch.equal(a, b) for kind in ("k", "v") for a, b in zip(native_cache[kind], baseline_cache[kind]))
        report["baseline_cache_equal"] = True
        del baseline_cache
        for kind in ("k", "v"):
            for chip, value in enumerate(native_cache[kind]):
                torch.save(value, directory / f"cache-{kind}-chip{chip}.pt")

        # Run the same calls after removing capture hooks. A fresh sentinel cache prevents stale
        # cache reuse from concealing a missing write. There are no inside-forward host readbacks.
        free_cache(cache)
        cache = None
        cache = device_utils.seed_cache(mesh_device, model, cache_dtype)
        for slot, start, end in CHUNKS:
            ids, golden, _ = references[slot]
            tokens = upload_token_chunk(mesh_device, ids[start:end], actual_start=start, actual_end=end)
            try:
                output = model.prefill_chunk(
                    tokens, cache, slot_idx=slot, actual_start=start, actual_end=end, skip_lm_head=True
                )
                ttnn.synchronize_device(mesh_device)
                expected = torch.load(directory / f"hidden-s{slot}-l31-c{start}.pt", weights_only=True)
                actual_final = read_hidden(output)
                assert torch.equal(actual_final, expected), "Uninstrumented final hidden differs"
                for sp in range(4):
                    wanted = golden["layers"][-1]["hidden"][start + sp * 256 : start + (sp + 1) * 256]
                    report["final_global_rows"].append(
                        score_row(
                            metrics,
                            wanted,
                            actual_final[sp * 256 : (sp + 1) * 256],
                            FULL_LIMITS,
                            phase="after",
                            slot=slot,
                            start=start,
                            sp=sp,
                            layer=31,
                        )
                    )
                logits_device = model.head(output)
                try:
                    replay_logits = device_utils.check_logits(
                        logits_device,
                        golden,
                        start=start,
                        end=end,
                        num_layers=32,
                        dtype=cache_dtype,
                        records=report["records"],
                    )
                    assert torch.equal(
                        replay_logits, torch.load(directory / f"logits-s{slot}-c{start}.pt", weights_only=True)
                    )
                finally:
                    logits_device.deallocate(True)
                report["replay_calls"].append(
                    dict(slot=slot, start=start, end=end, final_hidden_equal=True, logits_equal=True)
                )
            finally:
                if output is not None:
                    output.deallocate(True)
                    output = None
                if tokens is not None:
                    tokens.deallocate(True)
                    tokens = None
        replay_cache = dict(zip(("k", "v"), snapshot(cache)))
        assert all(torch.equal(a, b) for kind in ("k", "v") for a, b in zip(native_cache[kind], replay_cache[kind]))
        report["replay_cache_equal"] = True
        del replay_cache
        free_cache(cache)
        cache = None
        model.close()
        model = None

        def load_chunks(prefix):
            return {start: torch.load(directory / f"{prefix}-c{start}.pt", weights_only=True) for start in (0, 1024)}

        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        embedding = read_raw_weights(checkpoint, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
        for slot in (0, 1):
            actual = full_hidden(load_chunks(f"embedding-s{slot}"))
            assert torch.equal(
                actual, F.embedding(references[slot][0], embedding).bfloat16()
            ), "Native embedding differs"
        del embedding
        for layer_idx in range(32):
            prefix = f"model.layers.{layer_idx}."
            loaded = read_raw_weights(checkpoint, [prefix + name for name in LAYER_NAMES])
            raw = {name.removeprefix(prefix): value for name, value in loaded.items()}
            assert all(value.dtype == torch.bfloat16 for value in raw.values())
            assert {name: tensor_sha(value) for name, value in raw.items()} == report["loader_identities"][
                str(layer_idx)
            ]
            for slot in (0, 1):
                input_prefix = f"embedding-s{slot}" if layer_idx == 0 else f"hidden-s{slot}-l{layer_idx-1}"
                actual_input = full_hidden(load_chunks(input_prefix)).float()
                local_hidden, local_k, local_v = reference_layer(actual_input, raw, config)
                actual_hidden = full_hidden(load_chunks(f"hidden-s{slot}-l{layer_idx}"))
                golden = references[slot][1]["layers"][layer_idx]
                pairs = [("hidden", None, local_hidden, actual_hidden, golden["hidden"])]
                for kind, expected in (("k", local_k), ("v", local_v)):
                    for head in range(8):
                        actual = assemble_head(native_cache[kind], slot * 32 + layer_idx, head)
                        pairs.append((kind, head, expected[head], actual, golden[kind][head]))
                for kind, head, expected, actual, raw_global in pairs:
                    limits = local_limits(dtype_name, kind)
                    coords = dict(slot=slot, layer=layer_idx, kind=kind, head=head)
                    report["local_rows"].append(
                        score_row(metrics, expected, actual, limits, scope="full", start=0, end=2048, **coords)
                    )
                    raw_limits = (
                        device_utils.hidden_limits(layer_idx + 1, cache_dtype)
                        if kind == "hidden"
                        else device_utils.kv_limits(layer_idx, cache_dtype)
                    )
                    report["raw_global_summary_rows"].append(
                        score_row(metrics, raw_global, actual, raw_limits, scope="full", **coords)
                    )
                    for chunk, sp, begin, end in windows():
                        report["local_rows"].append(
                            score_row(
                                metrics,
                                expected[begin:end],
                                actual[begin:end],
                                limits,
                                scope="stripe",
                                chunk=chunk,
                                sp=sp,
                                start=begin,
                                end=end,
                                **coords,
                            )
                        )
                dump(directory / "report.json", report)
                del actual_input, local_hidden, local_k, local_v, actual_hidden, pairs
            del loaded, raw
        report["local_misses"] = [row for row in report["local_rows"] if not row["within_limits"]]
        report["final_global_misses"] = [row for row in report["final_global_rows"] if not row["within_limits"]]
        report["raw_global_misses"] = [row for row in report["raw_global_summary_rows"] if not row["within_limits"]]
        assert len(report["local_rows"]) == 9792
        assert len(report["raw_global_summary_rows"]) == 1088
        assert len(report["baseline_calls"]) == len(report["completed_calls"]) == len(report["replay_calls"]) == 4
        assert {name: sha(checkpoint / name) for name in checkpoint_files} == report["checkpoint_metadata"]
        assert {
            name: [(checkpoint / name).stat().st_size, (checkpoint / name).stat().st_mtime_ns]
            for name in report["checkpoint_shard_stats"]
        } == report["checkpoint_shard_stats"]
        report["status"] = "native_input_checks_complete"
        report["local_gates_passed"] = not report["local_misses"]
        report["raw_final_global_within_characterization_limits"] = not report["final_global_misses"]
        report["counts"] = dict(
            local_rows=len(report["local_rows"]),
            raw_global_summary_rows=len(report["raw_global_summary_rows"]),
            final_global_rows=len(report["final_global_rows"]),
        )
        assert len(report["final_global_rows"]) == 48
        if fixture_path is not None:
            assert sha(fixture_path) == report["held_out_fixture"]["sha256"], "Held-out fixture changed"
        assert not report["local_misses"], "Existing decoder gates failed on same-native-input reference"
        report["case_passed"] = True
    except BaseException as error:
        report.update(status="native_input_checks_failed", exception=repr(error))
        raise
    finally:
        report["local_misses"] = [row for row in report["local_rows"] if not row["within_limits"]]
        report["final_global_misses"] = [row for row in report["final_global_rows"] if not row["within_limits"]]
        report["raw_global_misses"] = [row for row in report["raw_global_summary_rows"] if not row["within_limits"]]
        report["counts"] = dict(
            baseline_calls=len(report["baseline_calls"]),
            instrumented_calls=len(report["completed_calls"]),
            replay_calls=len(report["replay_calls"]),
            local_rows=len(report["local_rows"]),
            raw_global_summary_rows=len(report["raw_global_summary_rows"]),
            final_global_rows=len(report["final_global_rows"]),
            local_misses=len(report["local_misses"]),
            final_global_misses=len(report["final_global_misses"]),
            raw_global_misses=len(report["raw_global_misses"]),
        )
        dump(directory / "report.json", report)
        if output is not None:
            output.deallocate(True)
        if tokens is not None:
            tokens.deallocate(True)
        if cache is not None:
            free_cache(cache)
        if model is not None:
            model.close()
        if summary_root := os.environ.get("PREFILL_SUMMARIES"):
            write_pcc_summary(report, summary_root)
