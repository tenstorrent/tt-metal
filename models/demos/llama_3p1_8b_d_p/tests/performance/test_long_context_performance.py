# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full32-layer configured-length eager timing with final normalization and vocabulary logits."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from models.demos.llama_3p1_8b_d_p.tests.performance.performance_config import load_config
from models.demos.llama_3p1_8b_d_p.tests.performance.request_progress import (
    emit_progress,
    parse_warmup_stack_seconds,
    run_observed_request,
)
from models.demos.llama_3p1_8b_d_p.tests.performance.source_inventory import hash_files

# A missing opt-in skips this module; explicitly supplied invalid requests still fail below.
if "LLAMA_LONG_CONTEXT_PERF_CONFIG" not in os.environ:
    pytest.skip("Set LLAMA_LONG_CONTEXT_PERF_CONFIG to run this optional benchmark", allow_module_level=True)

CONFIG = load_config(os.environ.get("LLAMA_LONG_CONTEXT_PERF_CONFIG"))
WARMUP_STACK_SECONDS = parse_warmup_stack_seconds(os.environ.get("LLAMA_PREFILL_WARMUP_STACK_SECONDS"))

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.performance.book_fixture_loader import load_book_fixture
from models.demos.llama_3p1_8b_d_p.tests.performance.book_observation import (
    METADATA_KEYS,
    OBSERVATION_SCOPE,
    assemble_final_logits,
    final_position,
    rank_logits,
    validate_prompts,
)
from models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_helpers import (
    aggregate,
    chunk_plan,
    finalize_owned_resources,
    request_schedule,
    run_request,
)
from models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_validation import report_reasons
from models.demos.llama_3p1_8b_d_p.tt.input import upload_token_chunk
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
from models.demos.llama_3p1_8b_d_p.tt.model import PrefillModel

HERE = Path(__file__).resolve().parent
REPO = Path(CONFIG["repository"])
CHECKPOINT = CONFIG["checkpoint"]
TOKENS = CONFIG["context_length"]
CHUNKS = chunk_plan(TOKENS)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_pins(directory, phase):
    expected = json.loads(Path(CONFIG["source_pins"]["path"]).read_text())
    actual = hash_files(expected)
    dump(directory / (phase + "-source-pins.json"), actual)
    assert actual == expected, "Pinned snapshot or harness bytes changed"
    assert Path(ttnn.__file__).resolve().is_relative_to(REPO)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def logits_evidence(output, *, slot, start, phase, repetition, baseline, metadata, tokenizer, observations):
    assert tuple(output.shape) == (1, 1, 256, 16032)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    shards = ttnn.get_device_tensors(output)
    assert len(shards) == 32
    rows = []
    final_rows = []
    position = final_position(TOKENS)
    for chip, shard in enumerate(shards):
        host = ttnn.to_torch(shard)
        finite = bool(torch.isfinite(host).all())
        digest = tensor_digest(host)
        key = (slot, start, chip)
        if phase == "warmup":
            baseline[key] = digest
        equal = digest == baseline[key]
        rows.append(
            dict(
                slot=slot,
                start=start,
                end=start + 1024,
                phase=phase,
                repetition=repetition,
                chip=chip,
                sp=chip // 8,
                tp=chip % 8,
                shape=list(host.shape),
                elements=host.numel(),
                finite=finite,
                sha256=digest,
                repeat_equal=equal,
                reference="same-slot/chunk warmup logits SHA256",
                max_abs=float(host.abs().max()) if finite else None,
            )
        )
        if start == position["chunk_start"] and chip // 8 == position["sp"]:
            final_rows.append(
                dict(
                    tp=chip % 8,
                    sp=chip // 8,
                    local_row=position["local_row"],
                    position=position["position"],
                    values=host[0, 0, position["local_row"], :].float().tolist(),
                )
            )
        del host
    if start == position["chunk_start"]:
        values = assemble_final_logits(final_rows, TOKENS)
        observation = rank_logits(
            values,
            metadata["expected_next_token_id"],
            lambda token_id: tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        )
        observation.update(
            slot=slot,
            phase=phase,
            repetition=repetition,
            final_prompt_position=position["position"],
            chunk_start=start,
            sp=position["sp"],
            local_row=position["local_row"],
            tp_order=list(range(8)),
            vocab_size=128256,
            scope=OBSERVATION_SCOPE,
        )
        observation.update({key: metadata[key] for key in METADATA_KEYS})
        observations.append(observation)
    return rows


# Every timed call runs all 32 layers and the final norm/head. Inspect every output only after
# all chunks finish, so whole-prompt timing includes upload/sync but excludes readback and cleanup.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_full_prefill_long_context_performance(mesh_device):
    directory = Path(os.environ["LLAMA_PERFORMANCE_EVIDENCE_DIR"]) / f"captures-{TOKENS}"
    directory.mkdir(exist_ok=False)
    report = dict(
        schema_version=4,
        status="started",
        full_model_accepted=False,
        accuracy_status="No golden accuracy claim; book continuation predictions are observations only",
        execution_scope="error_free_full32_prefill",
        execution_authorization=CONFIG["resource_review"],
        run_mode="execution_and_performance",
        execution_verified=False,
        book_fixture=CONFIG["book_fixture"],
        next_token_observations=[],
        launch_config=CONFIG,
        scope="Synchronized eager full-prefill host wall; no accuracy, kernel-time or serving claim",
        cache_dtype="bfloat8_b",
        weights_dtype="bfloat16",
        activations_dtype="bfloat16",
        num_layers=32,
        mesh=[4, 8],
        slots=2,
        prompt_tokens=TOKENS,
        chunk_tokens=1024,
        include_final_norm_lm_head=True,
        traced=False,
        samples=[],
        output_checks=[],
        prompts=[],
        timing_definitions={
            "forward_wall_seconds": "Sum of all chunk intervals: after upload synchronization to completed model call synchronization",
            "prompt_wall_seconds": "Before first chunk upload to completion of last full-model forward; includes uploads, dispatch and synchronization",
            "tokens_per_second_per_user": f"{TOKENS} / prompt_wall_seconds for each sequential single-user request",
            "excluded": "Fixture/tokenizer loading, model/cache load, host logits readback/hash/ranking/decoding, output cleanup and report writes; no golden work is performed",
            "rerun_policy": "No automatic rerun for minor fixes. Reconsider measurement only for material execution changes such as precision, communication, chunking or algorithm/kernel work.",
        },
    )
    model = cache = None
    report["observability"] = dict(
        progress_file="progress.jsonl",
        warmup_stack_seconds=WARMUP_STACK_SECONDS,
        scope="Untimed phase events; optional warmup-only nonfatal stack dump may perturb cold timing",
    )
    retained = []
    baseline = {}
    try:
        check_pins(directory, "before")
        begin = time.perf_counter()
        fixture = load_book_fixture(
            CONFIG["book_fixture"]["manifest_path"], CONFIG["book_fixture"]["manifest_sha256"], TOKENS, CHECKPOINT
        )
        report["prompts"] = [dict(slot=slot, **record) for slot, record in enumerate(fixture["slots"])]
        validate_prompts(report["prompts"], TOKENS)
        prompts = [torch.tensor(record["token_ids"], dtype=torch.int64) for record in report["prompts"]]
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        report["fixture_and_tokenizer_load_wall_seconds"] = time.perf_counter() - begin
        emit_progress(directory / "progress.jsonl", "model_setup_begin", tokens=TOKENS)
        begin = time.perf_counter()
        model = PrefillModel(
            mesh_device, CHECKPOINT, num_layers=32, cache_dtype=ttnn.bfloat8_b, enable_lm_head=True, max_seq_len=TOKENS
        )
        assert len(model.layers) == model.num_layers == 32 and [layer.layer_idx for layer in model.layers] == list(
            range(32)
        )
        assert model.head is not None
        cache = allocate_kv_cache(mesh_device, model.mesh_config, cache_dtype=ttnn.bfloat8_b, max_seq_len=TOKENS)
        ttnn.synchronize_device(mesh_device)
        report["model_and_cache_load_wall_seconds"] = time.perf_counter() - begin
        emit_progress(directory / "progress.jsonl", "model_setup_complete", tokens=TOKENS)
        for slot, phase, repetition in request_schedule():
            ids = prompts[slot]

            def upload(start, end):
                return upload_token_chunk(
                    mesh_device, ids[start:end], actual_start=start, actual_end=end, max_seq_len=TOKENS
                )

            def forward(tokens, start, end):
                return model.prefill_chunk(
                    tokens,
                    cache,
                    slot_idx=slot,
                    actual_start=start,
                    actual_end=end,
                    skip_lm_head=False,
                    layer_observer=None,
                )

            def inspect(output, start, end):
                return logits_evidence(
                    output,
                    slot=slot,
                    start=start,
                    phase=phase,
                    repetition=repetition,
                    baseline=baseline,
                    metadata=report["prompts"][slot]["metadata"],
                    tokenizer=tokenizer,
                    observations=report["next_token_observations"],
                )

            sample, checks, readback_seconds = run_observed_request(
                run_request,
                emit=lambda event, **fields: emit_progress(directory / "progress.jsonl", event, **fields),
                warmup_stack_seconds=WARMUP_STACK_SECONDS,
                tokens=TOKENS,
                slot=slot,
                phase=phase,
                repetition=repetition,
                upload=upload,
                forward=forward,
                synchronize=lambda: ttnn.synchronize_device(mesh_device),
                program_count=mesh_device.num_program_cache_entries,
                inspect=inspect,
                release=lambda value: value.deallocate(True),
                clock=time.perf_counter,
                retained=retained,
                on_sample=report["samples"].append,
            )
            report["output_checks"].extend(checks)
            report.setdefault("readback_hash_wall_seconds", []).append(
                dict(slot=slot, phase=phase, repetition=repetition, seconds=readback_seconds)
            )
            dump(directory / "report.json", report)
            assert all(row["finite"] and row["repeat_equal"] for row in checks)
            if phase == "measured":
                assert sample["programs_before"] == sample["programs_after"], "Measured run compiled a new program"
        report["summary"] = aggregate(report["samples"])
        assert len(report["output_checks"]) == 8 * len(CHUNKS) * 32
        report["status"] = "performance_complete_no_accuracy_claim"
        assert (
            load_book_fixture(
                CONFIG["book_fixture"]["manifest_path"], CONFIG["book_fixture"]["manifest_sha256"], TOKENS, CHECKPOINT
            )
            == fixture
        )
        check_pins(directory, "after")
        assert not report_reasons(report, TOKENS), report_reasons(report, TOKENS)
    finally:
        actions = []
        if cache is not None:
            actions.extend(
                [
                    ("cache.k.deallocate", lambda: cache.k.deallocate(True)),
                    ("cache.v.deallocate", lambda: cache.v.deallocate(True)),
                ]
            )
        if model is not None:
            actions.append(("model.close", model.close))
        actions.append(("device.synchronize", lambda: ttnn.synchronize_device(mesh_device)))
        finalize_owned_resources(
            retained=retained,
            release=lambda value: value.deallocate(True),
            actions=actions,
            report=report,
            persist=lambda: dump(directory / "report.json", report),
            primary_error=sys.exc_info()[1],
        )
