# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A pure synthetic complete report for verifier tests; never imported by the native benchmark."""

from models.demos.llama_3p1_8b_d_p.tests.performance.book_observation import (
    METADATA_KEYS,
    OBSERVATION_SCOPE,
    final_position,
    rank_logits,
    synthetic_prompts,
)
from models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_helpers import (
    aggregate,
    request_sample,
    request_schedule,
)


def build_report(tokens=4096):
    samples = []
    checks = []
    observations = []
    prompts = synthetic_prompts(tokens)
    now = 10.0
    for slot, phase, repetition in request_schedule():
        begin = now
        chunks = []
        for start in range(0, tokens, 1024):
            chunks.append(
                dict(start=start, end=start + 1024, chunk_start=now, forward_start=now + 0.25, forward_end=now + 1)
            )
            now += 1
            for chip in range(32):
                checks.append(
                    dict(
                        slot=slot,
                        phase=phase,
                        repetition=repetition,
                        start=start,
                        end=start + 1024,
                        chip=chip,
                        sp=chip // 8,
                        tp=chip % 8,
                        finite=True,
                        repeat_equal=True,
                        shape=[1, 1, 256, 16032],
                        elements=4104192,
                        sha256="a" * 64,
                        max_abs=1.0,
                    )
                )
        samples.append(
            request_sample(
                tokens=tokens,
                slot=slot,
                phase=phase,
                repetition=repetition,
                prompt_start=begin,
                prompt_end=now,
                chunks=chunks,
                programs_before=7,
                programs_after=7,
            )
        )
        now += 5
        meta = prompts[slot]["metadata"]
        values = [0.0] * 128256
        values[0] = 1.0
        observation = rank_logits(values, meta["expected_next_token_id"], lambda i: f"token{i}")
        position = final_position(tokens)
        observation.update(
            slot=slot,
            phase=phase,
            repetition=repetition,
            final_prompt_position=position["position"],
            chunk_start=position["chunk_start"],
            sp=position["sp"],
            local_row=position["local_row"],
            tp_order=list(range(8)),
            vocab_size=128256,
            scope=OBSERVATION_SCOPE,
        )
        observation.update({k: meta[k] for k in METADATA_KEYS})
        observations.append(observation)
    return dict(
        schema_version=4,
        run_mode="execution_and_performance",
        execution_verified=False,
        status="performance_complete_no_accuracy_claim",
        full_model_accepted=False,
        execution_scope="error_free_full32_prefill",
        execution_authorization={"path": "/synthetic/resource.json", "sha256": "b" * 64},
        book_fixture={"manifest_path": "/synthetic/manifest.json", "manifest_sha256": "c" * 64},
        launch_config={},
        cache_dtype="bfloat8_b",
        weights_dtype="bfloat16",
        activations_dtype="bfloat16",
        num_layers=32,
        mesh=[4, 8],
        slots=2,
        prompt_tokens=tokens,
        chunk_tokens=1024,
        include_final_norm_lm_head=True,
        traced=False,
        samples=samples,
        output_checks=checks,
        prompts=prompts,
        next_token_observations=observations,
        readback_hash_wall_seconds=[dict(slot=s, phase=p, repetition=n, seconds=5.0) for s, p, n in request_schedule()],
        summary=aggregate(samples),
    )
