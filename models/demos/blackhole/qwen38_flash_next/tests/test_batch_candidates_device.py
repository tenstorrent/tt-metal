# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3g-1 device tests: the batch adapter's candidate rows for the sampled lanes, in-process on the real chain.

Ids (one per process; every id runs with the prefill pack off, so a request's lane holds the same state alone and among
others):

* ``cand_rows_b8`` / ``cand_rows_b32``: after a sampled decode step the device candidate row of EVERY lane equals the per-shard
  top-32 of that lane's full gathered logits (values bitwise, ids equal up to ties at a shard's 32nd value);
* ``seeded_b8``: a seeded sampled request draws the same tokens alone and among seven other requests (sampled, greedy, and
  rows that fall back to the full logits);
* ``mixed_b8``: greedy, candidate, and every fallback reason in one step: the counters, one full gather for all fallback rows, and
  every drawn token inside the support of the full-row law of that row.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_batch_candidates_device.py -v -s -k cand_rows_b8
Env as test_vllm_batch_contract_device.py.
"""

import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_candidates_no_device import full_law
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import (
    MAX_SEQ_LEN,
    initialize,
    plugin_warmups,
    ttnn_bf16,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    Qwen38HostSamplingPolicy,
    apply_penalties,
    greedy_batch_sampling_params,
    position_generator,
    sample_from_row,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38CandidateRow

BLOCK = 64
STEPS = 12
PROMPT_LEN = 100  # two 64-row blocks per request


def sampling(rows: int, **fields) -> SimpleNamespace:
    params = greedy_batch_sampling_params(rows)
    for name, values in fields.items():
        setattr(params, name, list(values) + getattr(params, name)[len(values) :])
    return params


class Lanes:
    """A tiny runner over one batch adapter: request i owns slot i and blocks ``[1 + 2i, 2 + 2i]``."""

    def __init__(self, model, lanes: int) -> None:
        self.model, self.lanes = model, lanes
        self.width = -(-MAX_SEQ_LEN // BLOCK)
        self.tokens = [0] * lanes
        self.positions: list[int | None] = [None] * lanes
        self.prompts: dict[int, list[int]] = {}
        self.outputs: dict[int, list[int]] = {}

    def table_row(self, slot: int) -> list[int]:
        return [1 + 2 * slot, 2 + 2 * slot] + [0] * (self.width - 2)

    def admit(self, slots: list[int], prompts: list[list[int]]) -> None:
        rows = len(slots)
        tokens = torch.zeros(rows, max(len(p) for p in prompts), dtype=torch.int32)
        for i, prompt in enumerate(prompts):
            tokens[i, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
        out, _ = self.model.prefill_forward(
            tokens=tokens,
            page_table=torch.tensor([self.table_row(s) for s in slots], dtype=torch.int32),
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(p) for p in prompts],
            start_pos=torch.zeros(rows, dtype=torch.int32),
            empty_slots=slots,
            sampling_params=greedy_batch_sampling_params(rows),
        )
        for i, slot in enumerate(slots):
            self.prompts[slot] = prompts[i]
            self.tokens[slot] = int(out.reshape(-1)[i])
            self.positions[slot] = len(prompts[i])
            self.outputs[slot] = []

    def release(self, slot: int) -> None:
        self.model.release_request(slot)
        self.positions[slot] = None

    def step(self, params, histories: dict[int, tuple[list[int], list[int]]] | None = None) -> torch.Tensor:
        """One decode step of every admitted slot; returns the next tokens ``[lanes, 1]`` and advances the slots."""

        lanes = self.lanes
        tokens = torch.zeros(lanes, 1, dtype=torch.int32)
        positions = torch.full((lanes,), -1, dtype=torch.int32)
        table = torch.zeros(lanes, self.width, dtype=torch.int32)
        for slot in range(lanes):
            if self.positions[slot] is None:
                continue
            tokens[slot, 0], positions[slot] = self.tokens[slot], self.positions[slot]
            table[slot] = torch.tensor(self.table_row(slot), dtype=torch.int32)
        kwargs = {}
        if histories:
            prompt_tokens = torch.full((lanes, PROMPT_LEN + STEPS + 2), -1, dtype=torch.int32)
            output_tokens = torch.full_like(prompt_tokens, -1)
            for slot, (prompt, output) in histories.items():
                prompt_tokens[slot, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
                if output:
                    output_tokens[slot, : len(output)] = torch.tensor(output, dtype=torch.int32)
            kwargs = {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
        out = self.model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            sampling_params=params,
            slot_remap=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            **kwargs,
        )
        for slot in range(lanes):
            if self.positions[slot] is not None:
                self.outputs[slot].append(int(out[slot, 0]))
                self.tokens[slot] = int(out[slot, 0])
                self.positions[slot] += 1
        return out


def make_prompts(count: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1000, 20000) for _ in range(PROMPT_LEN)] for _ in range(count)]


def open_model(mesh_device, lanes: int, monkeypatch, normalizer: bool = False):
    """The batch adapter on the real chain.  The P3g-1 ids run with the P3g-2 normalizer off (their counters are P3g-1's: a
    top-k-off row is a ``top_k`` fallback); the P3g-2 ids turn it on."""

    if not os.environ.get("MODEL_WEIGHTS_DIR") or not os.environ.get("QWEN38_CACHE_ROOT"):
        pytest.fail("set MODEL_WEIGHTS_DIR and QWEN38_CACHE_ROOT")
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    monkeypatch.delenv("QWEN38_BATCH_CANDIDATES", raising=False)  # unset = on
    monkeypatch.setenv("QWEN38_BATCH_NORMALIZER", "1" if normalizer else "0")
    mesh_device.enable_program_cache()
    num_blocks = 1 + 2 * lanes + 4
    model = initialize(mesh_device, Path(os.environ["MODEL_WEIGHTS_DIR"]), lanes)
    model.allocate_kv_cache((num_blocks, 1, BLOCK, 128), ttnn_bf16(), 12)
    plugin_warmups(model, lanes, BLOCK, num_blocks)
    assert model.batch_candidates and model.decoder.has_candidates
    assert model.batch_normalizer is normalizer and model.decoder.has_normalizer is normalizer
    return model


def candidate_rows_equal_the_topk_of_the_full_logits(mesh_device, lanes: int, monkeypatch) -> None:
    model = open_model(mesh_device, lanes, monkeypatch)
    try:
        driver = Lanes(model, lanes)
        driver.admit(list(range(lanes)), make_prompts(lanes, 1))
        params = sampling(
            lanes,
            temperature=[1.0] * lanes,
            top_k=[20] * lanes,
            top_p=[0.95] * lanes,
            seed=list(range(11, 11 + lanes)),
        )
        decoder = model.decoder
        reports = []
        for step in range(4):
            driver.step(params)  # a sampled step: the lane replay, then the candidate trace
            rows = decoder.read_candidates()  # replays the candidate trace again over the same logits
            assert tuple(rows.shape) == (lanes, 256)
            full = decoder.read_full_logits()
            assert tuple(full.shape) == (lanes, VOCAB_SIZE)
            for lane in range(lanes):
                actual = Qwen38CandidateRow.from_host_row(rows[lane].reshape(1, 1, 1, 256))
                agreement = actual.agreement(Qwen38CandidateRow.emulate(full[lane].to(torch.bfloat16)))
                reports.append(agreement)
                assert all(agreement["values_bitwise"]), (step, lane, agreement)
                assert all(agreement["ids_equal_up_to_boundary_ties"]), (step, lane, agreement)
        exchanged = sum(sum(r["ids_exchanged"]) for r in reports)
        logger.info(f"B={lanes}: {len(reports)} lane rows checked, ids exchanged at boundary ties: {exchanged}")
        counters = model.candidate_counters
        assert counters["candidate_rows"] == 4 * lanes and counters["fallback_steps"] == 0
        assert counters["candidate_steps"] == 4
    finally:
        model.release_persistent_capture()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "kind, lanes",
    [
        pytest.param("rows", 8, id="cand_rows_b8", marks=pytest.mark.timeout(5400)),
        pytest.param("rows", 32, id="cand_rows_b32", marks=pytest.mark.timeout(5400)),
        pytest.param("seeded", 8, id="seeded_b8", marks=pytest.mark.timeout(5400)),
        pytest.param("mixed", 8, id="mixed_b8", marks=pytest.mark.timeout(5400)),
        pytest.param("norm_lse", 8, id="norm_lse_b8", marks=pytest.mark.timeout(5400)),
        pytest.param("norm_lse", 32, id="norm_lse_b32", marks=pytest.mark.timeout(5400)),
        pytest.param("norm_mixed", 8, id="norm_mixed_b8", marks=pytest.mark.timeout(5400)),
        pytest.param("norm_seeded", 8, id="norm_seeded_b8", marks=pytest.mark.timeout(5400)),
    ],
)
def test_batch_candidates(mesh_device, kind, lanes, monkeypatch, record_property):
    if kind == "rows":
        candidate_rows_equal_the_topk_of_the_full_logits(mesh_device, lanes, monkeypatch)
    elif kind == "seeded":
        seeded_request_is_the_same_alone_and_in_a_batch(mesh_device, lanes, monkeypatch)
    elif kind == "mixed":
        mixed_step(mesh_device, lanes, monkeypatch)
    elif kind == "norm_lse":
        normalizer_equals_the_host_logsumexp(mesh_device, lanes, monkeypatch)
    elif kind == "norm_mixed":
        normalizer_mixed_step(mesh_device, lanes, monkeypatch)
    else:
        seeded_request_is_the_same_alone_and_in_a_batch(mesh_device, lanes, monkeypatch, normalizer=True)


def seeded_request_is_the_same_alone_and_in_a_batch(
    mesh_device, lanes: int, monkeypatch, normalizer: bool = False
) -> None:
    prompts = make_prompts(lanes, 2)
    model = open_model(mesh_device, lanes, monkeypatch, normalizer)
    try:
        # alone: slot 0 only (the other lanes idle), seeded, sampled
        alone_params = sampling(lanes, temperature=[1.0], top_k=[20], top_p=[0.95], seed=[1234])
        driver = Lanes(model, lanes)
        driver.admit([0], [prompts[0]])
        for _ in range(STEPS):
            driver.step(alone_params)
        alone = list(driver.outputs[0])
        assert model.candidate_counters["fallback_steps"] == 0
        driver.release(0)
        # among seven others: seeded sampled rows, a greedy row, a top-k-off row and a top_k 50 row (full-logits fallbacks)
        driver = Lanes(model, lanes)
        driver.admit(list(range(lanes)), prompts)
        batch_params = sampling(
            lanes,
            temperature=[1.0, 1.0, 0.7, 0.0, 1.0, 0.9, 1.0, 1.0],
            top_k=[20, 20, 10, 1, VOCAB_SIZE, 50, 32, 5],
            top_p=[0.95, 0.9, 0.9, 0.0, 0.95, 1.0, 1.0, 0.8],
            seed=[1234, 5, 6, None, 7, 8, 9, 10],
        )
        for _ in range(STEPS):
            driver.step(batch_params)
        batch = list(driver.outputs[0])
        counters = dict(model.candidate_counters)
        logger.info(f"alone {alone}\nbatch {batch}\ncounters {counters}")
        if normalizer:  # row 4 (top-k off, top_p 0.95) uses the normalizer or, on a flat row, the nucleus fallback
            assert (
                counters["normalizer_steps"] >= STEPS
                and counters["normalizer_rows"] + counters["fallback_nucleus"] >= STEPS
            )
            assert counters["fallback_top_k"] >= STEPS  # row 5: top_k 50
        else:
            assert counters["fallback_top_k"] >= 2 * STEPS and counters["fallback_steps"] >= STEPS
        assert alone == batch, "a seeded request must draw the same tokens alone and among other requests"
    finally:
        model.release_persistent_capture()


def mixed_step(mesh_device, lanes: int, monkeypatch) -> None:
    prompts = make_prompts(lanes, 3)
    model = open_model(mesh_device, lanes, monkeypatch)
    try:
        driver = Lanes(model, lanes)
        driver.admit(list(range(lanes)), prompts)
        # 0 greedy, 1 / 2 candidates, 3 top_k off, 4 top_k 50, 5 penalty that raises logits, 6 presence penalty (candidates),
        # 7 repetition penalty over a history (candidates)
        params = sampling(
            lanes,
            temperature=[0.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.7, 0.9],
            top_k=[1, 20, 32, VOCAB_SIZE, 50, 20, 20, 20],
            top_p=[0.0, 0.95, 1.0, 0.95, 0.9, 0.95, 0.8, 0.95],
            seed=[None, 11, 22, 33, 44, 55, 66, 77],
            presence_penalty=[0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.5, 0.0],
            repetition_penalty=[1.0] * 7 + [1.2],
        )
        equal = checked = 0
        for step in range(STEPS):
            histories = {
                5: (prompts[5], driver.outputs[5]),
                6: (prompts[6], driver.outputs[6]),
                7: (prompts[7], driver.outputs[7]),
            }
            before = dict(model.candidate_counters)
            positions = list(driver.positions)
            out = driver.step(params, histories)
            full = model.decoder.read_full_logits()
            delta = {k: model.candidate_counters[k] - before[k] for k in before}
            assert delta["candidate_steps"] == 1 and delta["candidate_rows"] == 4, delta  # rows 1, 2, 6, 7
            assert (
                delta["fallback_steps"] == 1 and delta["fallback_top_k"] == 2 and delta["fallback_penalty"] == 1
            ), delta
            for row in range(1, lanes):
                policy = Qwen38HostSamplingPolicy.from_contract(params, row)
                masked = full[row].clone()
                masked[TOKENIZER_SIZE:] = float("-inf")
                if policy.penalized:
                    history = (
                        torch.tensor(prompts[row], dtype=torch.long),
                        torch.tensor(
                            driver.outputs[row][:-1], dtype=torch.long
                        ),  # the step's own token is not history yet
                    )
                    apply_penalties(masked, *history, policy)
                token = int(out[row, 0])
                checked += 1
                expected = sample_from_row(masked.clone(), policy, position_generator(policy.seed, positions[row]))
                equal += token == expected
                if policy.top_k <= 32 and not policy.penalized or row in (6, 7):
                    support = full_law(masked, Qwen38HostSamplingPolicy(policy.temperature, policy.top_k, policy.top_p))
                    assert token in support, (step, row, token)
        logger.info(
            f"mixed step: {STEPS} steps, drawn token equal to today's full-row draw in {equal} of {checked} rows; "
            f"counters {json.dumps(model.candidate_counters)}"
        )
        assert equal >= 0.97 * checked
    finally:
        model.release_persistent_capture()


# -- P3g-2: the device normalizer ---------------------------------------------------------------------------------------

LSE_REL_BOUND = 3e-5  # on exp(device LSE) / exp(host LSE) - 1; see normalizer_equals_the_host_logsumexp


def normalizer_equals_the_host_logsumexp(mesh_device, lanes: int, monkeypatch) -> None:
    """The per-lane device LSE equals the host's logsumexp of the lane's gathered bf16 logits / T over the real vocabulary
    (float64 reference), with a different temperature per lane.

    Bound: the device runs fp32 throughout (bf16 -> fp32 is exact; x * fl32(1 / T) differs from the host's x / T by at most
    one fp32 rounding of 1 / T, a relative 6e-8 of |x| <= ~60 / T, so <= 4e-6 in the LSE; exp and log add a few fp32 ulps; the
    62080-term fp32 sum a relative ~1e-7).  Measured: 4e-6 on random logits (T 0.6 .. 1.4), 8.8e-6 on the real model's logits
    (B=8 8.8e-6, B=32 1.2e-5, T 0.5 .. 1.5: |x| / T reaches ~60).  The asserted bound, 3e-5 on exp(LSE) / exp(host) - 1,
    is ~2.4x above the largest measurement and two orders below the bf16 rounding (4e-3) a bf16 normalizer would carry.
    """

    model = open_model(mesh_device, lanes, monkeypatch, normalizer=True)
    try:
        driver = Lanes(model, lanes)
        driver.admit(list(range(lanes)), make_prompts(lanes, 4))
        temperatures = [0.5 + i / max(lanes - 1, 1) for i in range(lanes)]  # 0.5 .. 1.5
        params = sampling(
            lanes,
            temperature=temperatures,
            top_k=[VOCAB_SIZE] * lanes,
            top_p=[0.95] * lanes,
            seed=list(range(11, 11 + lanes)),
        )
        decoder = model.decoder
        inverse = [1.0 / t for t in temperatures]
        worst_shard = worst_combined = 0.0
        padding_over_real = []
        for step in range(4):
            driver.step(params)  # a top-k-off step: the lane replay, then the candidate + normalizer trace
            assert model.candidate_counters["normalizer_steps"] == step + 1
            rows, shard = decoder.read_candidates_normalized(inverse)  # replay again over the same logits
            assert tuple(shard.shape) == (lanes, 4) and bool(torch.isfinite(shard).all())
            full = decoder.read_full_logits().double()
            padding_over_real.append(
                float((full[:, TOKENIZER_SIZE:].max(dim=1).values - full[:, :TOKENIZER_SIZE].max(dim=1).values).max())
            )
            real = full[:, :TOKENIZER_SIZE]
            for lane in range(lanes):
                scaled = torch.full((VOCAB_SIZE,), float("-inf"), dtype=torch.float64)
                scaled[:TOKENIZER_SIZE] = real[lane] / temperatures[lane]
                per_shard = torch.stack([torch.logsumexp(scaled[d * 62080 : (d + 1) * 62080], 0) for d in range(4)])
                err = (torch.exp(shard[lane].double() - per_shard) - 1).abs().max()
                worst_shard = max(worst_shard, float(err))
                combined = float(torch.logsumexp(shard[lane].double(), 0))
                worst_combined = max(worst_combined, abs(math.exp(combined - float(torch.logsumexp(per_shard, 0))) - 1))
            candidates = Qwen38CandidateRow
            for lane in range(lanes):  # the variant's candidate rows are the plain trace's
                agreement = candidates.from_host_row(rows[lane].reshape(1, 1, 1, 256)).agreement(
                    candidates.emulate(full[lane].float().to(torch.bfloat16))
                )
                assert all(agreement["values_bitwise"]), (step, lane)
        logger.info(
            f"B={lanes}: max relative error of exp(shard LSE) {worst_shard:.3e}, of exp(combined LSE) {worst_combined:.3e}; "
            f"largest padded-column logit minus largest real logit: {max(padding_over_real):.2f}"
        )
        assert worst_shard < LSE_REL_BOUND and worst_combined < LSE_REL_BOUND
    finally:
        model.release_persistent_capture()


def normalizer_mixed_step(mesh_device, lanes: int, monkeypatch) -> None:
    """A step with top-k-off rows (different T, top_p), a top_k 20 row, a greedy row and fallback rows: the tokens against the
    host full-row path on the same gathered logits.  Mismatches are expected only where the nucleus boundary lies inside a
    rounding of the probabilities."""

    prompts = make_prompts(lanes, 5)
    model = open_model(mesh_device, lanes, monkeypatch, normalizer=True)
    try:
        driver = Lanes(model, lanes)
        driver.admit(list(range(lanes)), prompts)
        # 0 greedy, 1 top_k 20, 2..4 top-k off (normalizer), 5 top-k off top_p 1 (top_k), 6 top-k off + presence penalty
        # (penalty), 7 top_k 50 (top_k)
        params = sampling(
            lanes,
            temperature=[0.0, 1.0, 0.6, 0.8, 1.0, 1.0, 0.9, 1.0],
            top_k=[1, 20, VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE, 50],
            top_p=[0.0, 0.95, 0.9, 0.8, 0.6, 1.0, 0.9, 0.9],
            seed=[None, 11, 22, 33, 44, 55, 66, 77],
            presence_penalty=[0.0] * 6 + [1.5, 0.0],
        )
        equal = checked = 0
        mismatched_rows: list[tuple[int, int]] = []
        for step in range(STEPS):
            histories = {6: (prompts[6], driver.outputs[6])}
            before = dict(model.candidate_counters)
            positions = list(driver.positions)
            out = driver.step(params, histories)
            full = model.decoder.read_full_logits()
            delta = {k: model.candidate_counters[k] - before[k] for k in before}
            assert delta["normalizer_steps"] == 1 and delta["candidate_steps"] == 1, delta
            assert delta["normalizer_rows"] + delta["fallback_nucleus"] == 3, delta  # rows 2, 3, 4
            assert delta["fallback_top_k"] == 2 and delta["fallback_penalty"] == 1, delta  # rows 5 and 7, row 6
            assert delta["fallback_guard"] == 0 and delta["fallback_nonfinite"] == 0, delta
            for row in range(1, lanes):
                policy = Qwen38HostSamplingPolicy.from_contract(params, row)
                masked = full[row].clone()
                masked[TOKENIZER_SIZE:] = float("-inf")
                if policy.penalized:
                    history = (
                        torch.tensor(prompts[row], dtype=torch.long),
                        torch.tensor(driver.outputs[row][:-1], dtype=torch.long),
                    )
                    apply_penalties(masked, *history, policy)
                expected = sample_from_row(masked.clone(), policy, position_generator(policy.seed, positions[row]))
                checked += 1
                equal += int(out[row, 0]) == expected
                if int(out[row, 0]) != expected:
                    mismatched_rows.append((step, row))
        counters = dict(model.candidate_counters)
        logger.info(
            f"norm mixed: {STEPS} steps, token equal to the host full-row draw in {equal} of {checked} rows, mismatches "
            f"{mismatched_rows}; counters {json.dumps(counters)}"
        )
        assert counters["normalizer_rows"] >= STEPS, "the normalizer never drew a row: the test proves nothing"
        assert equal >= 0.98 * checked
    finally:
        model.release_persistent_capture()
