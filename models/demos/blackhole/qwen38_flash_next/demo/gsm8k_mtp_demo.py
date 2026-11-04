# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP decode on many-shot GSM8K chat prompts, one ISL per test id (128 ... 256k).

Prompt: one chat user message (thinking off) = an instruction line + N solved train examples ("Question: ...\\nAnswer:
<worked solution ending with #### n>") + one test question.  N is the largest count whose templated prompt is <= the
target ISL (0 shots at ISL 128 when the shots do not fit).  Greedy, stop at EOS, 512 tokens cap.
Tests: ``gsm8k_<isl>_mtp4`` (batch 1, K=4) and ``gsm8k_<isl>_b8_mtp3`` (lanes, K=3; B = the largest of 8..2 that the
lanes admission takes).  The data (train.jsonl, test.jsonl of openai/grade-school-math) is read from
the directory $QWEN38_GSM8K_DIR (the tests skip when it is unset or missing).

Run:  pytest models/demos/blackhole/qwen38_flash_next/demo/gsm8k_mtp_demo.py -v -s -k gsm8k_1k_mtp4
"""

import json
import os
import re
import time
from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import EOS_TOKEN_IDS, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import open_mtp_chain, run_mtp_lanes, run_mtp_single
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is the fixture)
    CONTEXTS,
    DEVICE_PARAMS,
    MTP_LANES,
    device_params,
    write_chat_report,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatChainError, Qwen38ChatSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

INSTRUCTION = "Solve the last math problem step by step. End your answer with '#### <number>'."
MAX_NEW_TOKENS = 512
WARMUP_NEW_TOKENS = 16
SHOT_OFFSET_STRIDE = 911  # lane i starts its shots at train[i * stride]
LANE_DRAFTS = 3
SINGLE_DRAFTS = 4
LANE_CHECK_ISLS = (128, 1024)  # lanes are also compared with their own batch-1 K=3 run
FULL_WARM_MAX_ISL = 32768  # up to here a same-ISL warm-up request runs first so the TTFT is not a compile

ISLS = [
    (128, "128", 5400),
    (256, "256", 5400),
    (1024, "1k", 5400),
    (2048, "2k", 5400),
    (4096, "4k", 7200),
    (8192, "8k", 7200),
    (16384, "16k", 9000),
    (32768, "32k", 10800),
    (65536, "64k", 14400),
    (131072, "128k", 21600),
    (262144, "256k", 28800),
]


def load_gsm8k():
    directory = os.environ.get("QWEN38_GSM8K_DIR")
    if (
        not directory
        or not (Path(directory) / "train.jsonl").is_file()
        or not (Path(directory) / "test.jsonl").is_file()
    ):
        pytest.skip("set QWEN38_GSM8K_DIR to a directory with train.jsonl and test.jsonl (openai/grade-school-math)")

    def read(name):
        return [
            json.loads(line)
            for line in (Path(directory) / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return read("train.jsonl"), read("test.jsonl")


def label_of(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_answer(text: str):
    """The number after the last '####', else the last number in the text; commas and a trailing '.0' removed."""

    tail = text.split("####")[-1] if "####" in text else text
    numbers = _NUMBER.findall(tail)
    if not numbers:
        return None
    value = (numbers[0] if "####" in text else numbers[-1]).replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


class PromptBuilder:
    def __init__(self, template, train, test):
        self.template, self.train, self.test = template, train, test
        self.tokenizer = template.tokenizer
        self.shots = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in train]
        self._shot_tokens = None

    def render_ids(self, content):
        rendered = self.template.render([{"role": "user", "content": content}], enable_thinking=False)
        return [int(t) for t in rendered.input_ids[0].tolist()]

    def content(self, offset, n_shots, question):
        parts = [INSTRUCTION] + [self.shots[(offset + j) % len(self.shots)] for j in range(n_shots)]
        parts.append(f"Question: {question}\nAnswer:")
        return "\n\n".join(parts)

    def build(self, target, offset, test_index, exclude=()):
        """(ids, info): the largest N with a templated length <= target.  A test question whose 0-shot prompt exceeds
        the target is skipped (the next test question is used)."""

        if self._shot_tokens is None:
            self._shot_tokens = [len(self.tokenizer.encode(s + "\n\n", add_special_tokens=False)) for s in self.shots]
        n_test = len(self.test)
        for skip in range(n_test):
            index = (test_index + skip) % n_test
            if index in exclude:
                continue
            question = self.test[index]["question"]
            base = len(self.render_ids(self.content(offset, 0, question)))
            if base <= target:
                break
        else:
            raise ValueError(f"no test question fits {target} tokens")
        # estimate N from the per-shot token counts, then fix it with exact renders
        total, n = base, 0
        while total + self._shot_tokens[(offset + n) % len(self.shots)] <= target and n < 100_000:
            total += self._shot_tokens[(offset + n) % len(self.shots)]
            n += 1
        ids = self.render_ids(self.content(offset, n, question))
        while len(ids) > target and n > 0:
            n -= 1
            ids = self.render_ids(self.content(offset, n, question))
        while True:
            more = self.render_ids(self.content(offset, n + 1, question))
            if len(more) > target:
                break
            n, ids = n + 1, more
        return ids, {
            "isl": len(ids),
            "shots": n,
            "shot_offset": offset,
            "test_index": index,
            "label": label_of(self.test[index]["answer"]),
            "question": question,
        }


def sizes(isl):
    allocated = min(context for context in CONTEXTS if context >= isl)
    max_seq_len = allocated - RESIDENT_CONTEXT_HEADROOM
    return allocated, max_seq_len, min(isl, max_seq_len - MAX_NEW_TOKENS)


def grade(tokenizer, tokens, label):
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = parse_answer(text)
    return {"parsed": parsed, "label": label, "correct": parsed is not None and parsed == label}, text


def common_setup():
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    template = Qwen38OfficialChatTemplate(checkpoint)
    train, test = load_gsm8k()
    return checkpoint, template, PromptBuilder(template, train, test)


def _single(mesh_device, isl, tmp_path, record_property):
    checkpoint, template, builder = common_setup()
    tokenizer = template.tokenizer
    allocated, max_seq_len, target = sizes(isl)
    ids, info = builder.build(target, 0, 0)
    logger.info(
        f"gsm8k {isl}: target {target} -> ISL {info['isl']} with {info['shots']} shots (allocated context {allocated})"
    )
    warm_ids = None
    if isl <= FULL_WARM_MAX_ISL:
        warm_ids, _ = builder.build(target, SHOT_OFFSET_STRIDE * 9, 5)
    short_ids, _ = builder.build(128, 0, 1)
    stop_ids = tuple(EOS_TOKEN_IDS)
    mesh_device.enable_program_cache()
    started = time.perf_counter()
    try:
        chain = open_mtp_chain(mesh_device, checkpoint, max_seq_len, SINGLE_DRAFTS)
    except Qwen38ChatChainError as error:
        pytest.skip(f"MTP K={SINGLE_DRAFTS} admission refused at context {allocated}: {error}")
    logger.info(f"MTP chain ready in {time.perf_counter() - started:.1f} s")
    try:
        session = Qwen38ChatSession(chain, template, prefill_mode="chunked")
        run_mtp_single(session, short_ids, WARMUP_NEW_TOKENS, stop_ids=stop_ids)
        if warm_ids is not None:
            run_mtp_single(session, warm_ids, WARMUP_NEW_TOKENS, stop_ids=stop_ids)
        run = run_mtp_single(session, ids, MAX_NEW_TOKENS, stop_ids=stop_ids)
        assert not session.poisoned
    finally:
        chain.close()
    result, text = grade(tokenizer, run["tokens"], info["label"])
    row = {
        "isl": info["isl"],
        "shots": info["shots"],
        "ttft_s": run["ttft_s"],
        "generated_tokens": run["generated_tokens"],
        "decode_s": run["decode_s"],
        "decode_ms_per_token": run["decode_ms_per_token"],
        "tok_s": (run["generated_tokens"] - 1) / run["decode_s"],
        "tokens_per_pass": run["tokens_per_pass"],
        "chain_tokens_per_pass": run["chain_tokens_per_pass"],
        "passes": run["passes"],
        "finish": run["finish"],
        **result,
    }
    logger.info(
        f"gsm8k_{isl}_mtp4: ISL {row['isl']} ({row['shots']} shots) | TTFT {row['ttft_s']:.3f} s | {row['generated_tokens']} "
        f"tokens {row['finish']} | decode {row['decode_s']:.3f} s | {row['decode_ms_per_token']:.2f} ms/tok | "
        f"{row['tok_s']:.2f} tok/s/user | tokens/pass {row['tokens_per_pass']} | answer {result['parsed']} vs {result['label']} "
        f"correct={result['correct']}"
    )
    logger.info(f"gsm8k_{isl}_mtp4 text: {text!r}")
    write_chat_report(
        tmp_path,
        record_property,
        "qwen38-gsm8k-mtp/v1-single",
        {
            "drafts": SINGLE_DRAFTS,
            "allocated_context": allocated,
            "info": info,
            "row": row,
            "decoded": text,
            "generated_token_ids": run["tokens"],
        },
    )
    assert run["finish"] in ("stop", "length") and run["generated_tokens"] > 8


def _lanes(mesh_device, isl, tmp_path, record_property):
    checkpoint, template, builder = common_setup()
    tokenizer = template.tokenizer
    allocated, max_seq_len, target = sizes(isl)
    # The table admission is conservative (it refuses 8 lanes at 32k, which run live): the live admission in the open
    # decides.  B starts at 8; after a live refusal the driver reruns with QWEN38_GSM8K_LANES=<B - 1> (a refused open
    # leaves the mesh to a new process).
    candidates = [int(os.environ.get("QWEN38_GSM8K_LANES", MTP_LANES))]
    if not candidates:
        pytest.skip(f"no lane count fits at allocated context {allocated}")
    stop_ids = tuple(EOS_TOKEN_IDS)
    lanes = candidates[0]
    built = []
    for user in range(lanes):  # own test question and own shot offset per lane
        built.append(
            builder.build(target, SHOT_OFFSET_STRIDE * user, user, exclude={info["test_index"] for _, info in built})
        )
    prompts = [ids for ids, _ in built]
    assert len({tuple(p) for p in prompts}) == lanes, "two lanes share a prompt"
    logger.info(
        f"gsm8k lanes {isl}: B={lanes} ISLs {[info['isl'] for _, info in built]} shots {[info['shots'] for _, info in built]}"
    )
    short_ids, _ = builder.build(128, 0, 1)
    warm_ids = builder.build(target, SHOT_OFFSET_STRIDE * 9, 5)[0] if isl <= FULL_WARM_MAX_ISL else None
    mesh_device.enable_program_cache()
    chain = lanes_session = None
    refusals = []
    for lanes in candidates[:1]:
        lanes_session = Qwen38LanesSession(lanes=lanes, drafts=LANE_DRAFTS, marker=logger.info)
        started = time.perf_counter()
        try:
            chain = open_mtp_chain(mesh_device, checkpoint, max_seq_len, LANE_DRAFTS, lanes_session=lanes_session)
            logger.info(f"MTP lanes chain B={lanes} ready in {time.perf_counter() - started:.1f} s")
            break
        except Qwen38ChatChainError as error:
            logger.info(f"B={lanes} refused by the live admission: {error}")
            refusals.append({"B": lanes, "reason": str(error)})
            lanes_session = None
            break  # a refused open leaves the mesh state undefined; a retry needs a new process (QWEN38_GSM8K_LANES)
    if chain is None:
        pytest.skip(f"LANES_REFUSED B={candidates[0]} at allocated context {allocated}: {refusals}")

    references = {}
    try:
        session = Qwen38ChatSession(chain, template, prefill_mode="chunked")
        run_mtp_single(session, short_ids, WARMUP_NEW_TOKENS, stop_ids=stop_ids)
        if warm_ids is not None:
            run_mtp_single(session, warm_ids, WARMUP_NEW_TOKENS, stop_ids=stop_ids)
        if isl in LANE_CHECK_ISLS:
            for user, ids in enumerate(prompts):
                references[user] = run_mtp_single(session, ids, MAX_NEW_TOKENS, stop_ids=stop_ids)
                logger.info(
                    f"batch-1 K={LANE_DRAFTS} lane {user}: {references[user]['generated_tokens']} tokens {references[user]['finish']}"
                )
        session.reset()
        lanes_session.capture(chain, session)
        run = run_mtp_lanes(lanes_session, prompts, LANE_DRAFTS, MAX_NEW_TOKENS, stop_ids=stop_ids)
        assert not session.poisoned
    except BaseException:
        logger.exception("gsm8k lanes run failed")
        raise
    finally:
        lanes_session.close()
        chain.close()

    results = []
    for user, row in enumerate(run["lanes_rows"]):
        info = built[user][1]
        out = row["tokens"]
        result, text = grade(tokenizer, out, info["label"])
        entry = {
            **{k: v for k, v in row.items() if k != "tokens"},
            "user": user,
            "isl": info["isl"],
            "shots": info["shots"],
            "tok_s": (row["committed"] - 1) / row["decode_s"],
            **result,
            "decoded": text,
            "tokens": out,
        }
        if user in references:
            ref = references[user]["tokens"]
            diffs = [i for i in range(min(len(out), len(ref))) if out[i] != ref[i]]
            first = diffs[0] if diffs else (None if len(out) == len(ref) else min(len(out), len(ref)))
            entry.update(
                {"equal_to_batch1_k3": first is None, "first_diff_index_vs_batch1": first, "batch1_tokens": len(ref)}
            )
        results.append(entry)
        logger.info(
            f"lane {user}: ISL {info['isl']} ({info['shots']} shots) | TTFT own {row['ttft_own_prefill_s']:.3f} s / from batch "
            f"start {row['ttft_from_batch_start_s']:.3f} s | {row['committed']} tokens {row['finish']} | decode "
            f"{row['decode_s']:.3f} s | {entry['tok_s']:.2f} tok/s | tokens/pass {row['tokens_per_pass']:.3f} | answer "
            f"{result['parsed']} vs {result['label']} correct={result['correct']} | vs batch-1: "
            f"{entry.get('equal_to_batch1_k3', 'n/a')} first_diff={entry.get('first_diff_index_vs_batch1', 'n/a')}"
        )
    logger.info(
        f"gsm8k_{isl}_b8_mtp3: B={lanes} total tokens {run['total_generated_tokens']} | decode window "
        f"{run['decode_window_s']:.3f} s | aggregate {run['aggregate_all_tokens_tok_s']:.1f} tok/s ({run['aggregate_tok_s']:.1f} "
        f"without token 1) | mean per-user {run['mean_per_user_tok_s']:.2f} tok/s | pass {run['pass_ms_mean']:.2f} ms | "
        f"correct {sum(r['correct'] for r in results)}/{lanes}"
    )
    write_chat_report(
        tmp_path,
        record_property,
        "qwen38-gsm8k-mtp/v1-lanes",
        {
            "drafts": LANE_DRAFTS,
            "lanes": lanes,
            "allocated_context": allocated,
            "run": {k: v for k, v in run.items() if k not in ("lanes_rows", "pass_ms")},
            "lanes_rows": results,
        },
    )
    for r in results:
        assert r["finish"] in ("stop", "length"), (r["user"], r["finish"])
        assert r["passes"] > 0 and 1.0 < r["tokens_per_pass"] <= LANE_DRAFTS + 1, (r["user"], r["tokens_per_pass"])
    checked = [r for r in results if "equal_to_batch1_k3" in r]  # reported, not asserted (as chat_b8_mtp3)
    if checked:
        logger.info(
            f"lanes equal to their batch-1 K=3 run: {sum(r['equal_to_batch1_k3'] for r in checked)}/{len(checked)}"
        )
        if os.environ.get("QWEN38_GSM8K_B8_EQ_ASSERT") == "1":
            for r in checked:
                assert r[
                    "equal_to_batch1_k3"
                ], f"lane {r['user']} differs from its batch-1 K=3 run at {r['first_diff_index_vs_batch1']}"


def _single_params():
    return [pytest.param(isl, id=f"gsm8k_{name}_mtp4", marks=pytest.mark.timeout(t)) for isl, name, t in ISLS]


def _lane_params():
    return [pytest.param(isl, id=f"gsm8k_{name}_b8_mtp3", marks=pytest.mark.timeout(t)) for isl, name, t in ISLS]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("isl", _single_params())
def test_gsm8k_mtp4(mesh_device, isl, tmp_path, record_property):
    _single(mesh_device, isl, tmp_path, record_property)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("isl", _lane_params())
def test_gsm8k_b8_mtp3(mesh_device, isl, tmp_path, record_property):
    _lanes(mesh_device, isl, tmp_path, record_property)
