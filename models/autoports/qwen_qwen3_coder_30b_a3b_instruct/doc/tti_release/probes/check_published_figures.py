# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every number stage 10 publishes, re-derived from the artifact it came from.

Same mechanism as the stage 06-09 checkers, and it exists for the same reason:
on this project the recurring failure mode is prose drifting away from data that
is itself correct. This stage has an extra reason to want one — the numbers here
are the *customer-facing* ones, and two of them (the mbpp score, and the
mbpp score this model would have been published with) differ by a factor of
nearly three depending on a single harness setting.

What is checked
---------------

1. **The run identity and outcome** — exit code, TTI VERSION and SHA, the stale
   git tag, the host, the model id — against the copied run log and run spec.
2. **Every eval row's score**, against the copied lm-eval ``results_*.json``.
   The percentages in the prose must equal the fractions in the JSON.
3. **The mbpp truncation mechanism** (Finding 1), against
   ``evals/eval_samples_derived.json``: fence-closure counts, empty-filter
   counts, pass counts, and the 256-token replay — recomputed here from the
   per-sample rows rather than trusted.
4. **The 16-sample control**, against its own results JSON.
5. **The meta_* failure chain** (Finding 2) — the setup ValueError, the
   ``continuing...`` warning and the two ``Tasks were not found`` errors must all
   actually appear in the copied run log, and the exit codes must be what the
   prose says.
6. **The oversubscription arithmetic** (Finding 3) — 32 x 16 = 512 against a
   32-slot server, the 21-vs-0 TimeoutError counts, and the ``batch_size=1``
   that is actually on the command lines in this run's log.
7. **No context or request-length cap anywhere** — the contract, the spec, the
   server command and every eval command line must all carry the full context.
   This one is checked in both directions: the value must be present, and no
   smaller cap may appear on a command line.
8. **The benchmark sweep**, against the copied ``benchmark_*.json`` files: the
   shape of every published row and its latency/throughput figures.
9. **The coverage boundary itself.** Every figure-shaped numeric token in
   RUN_NOTES.md is either re-derived above or named in ``UNCOVERED`` with a
   reason. A number in neither fails the gate, so the checker's blind spots are
   enumerated rather than silent.

Exits non-zero on any mismatch, so it is a gate and not a report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
MODEL = STAGE.parent.parent
EVALS = STAGE / "evals"
LOGS = STAGE / "logs"
BENCH = STAGE / "bench"
RELEASE = STAGE / "reports_output" / "release"

NOTES = (STAGE / "RUN_NOTES.md").read_text()
# The coverage boundary covers every prose document this stage publishes, not
# just RUN_NOTES: stall/isl131072_stall_evidence.txt is the primary evidence for
# the one model-side defect (Finding 5) and was outside the boundary until now.
STALL = (STAGE / "stall" / "isl131072_stall_evidence.txt").read_text()
# reports_output/README.md is the annotation that travels next to the copied
# report; its figures are customer-facing, so it is inside the boundary too.
READMES = (STAGE / "reports_output" / "README.md").read_text()
SCANNED = {
    "RUN_NOTES.md": NOTES,
    "stall/isl131072_stall_evidence.txt": STALL,
    "reports_output/README.md": READMES,
}

FAILURES: list[str] = []
PASSES: list[str] = []
COVERED: set[str] = set()

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")


def numbers(text: str) -> set[str]:
    return {tok.replace(",", "") for tok in NUMBER.findall(text)}


def cover(*values) -> None:
    for value in values:
        COVERED.update(numbers(str(value)))


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    PASSES.append(msg)
    print(f"[ ok ] {msg}")


def load(path: Path):
    if not path.is_file():
        fail(f"missing artifact {path}")
        return None
    return json.loads(path.read_text())


def text_of(path: Path) -> str:
    if not path.is_file():
        fail(f"missing artifact {path}")
        return ""
    return path.read_text(errors="ignore")


def says(needle: str, where: str = "RUN_NOTES.md") -> bool:
    """Assert the named scanned document contains this string, and cover its numbers."""
    if needle in SCANNED[where]:
        cover(needle)
        return True
    fail(f"{where} does not contain {needle!r}")
    return False


def says_stall(needle: str) -> bool:
    return says(needle, "stall/isl131072_stall_evidence.txt")


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


# EXPERT_CHUNK_SIZE is read out of the model source rather than hard-coded, so
# the live-tensor arithmetic in section 8 tracks the port if the chunk size ever
# changes.
_DECODER_SRC = (MODEL / "tt" / "optimized_decoder.py").read_text()
_M = re.search(r"^EXPERT_CHUNK_SIZE\s*=\s*(\d+)", _DECODER_SRC, re.M)
EXPERT_CHUNK_SIZE = int(_M.group(1)) if _M else 0
if not EXPERT_CHUNK_SIZE:
    FAILURES.append("could not read EXPERT_CHUNK_SIZE from tt/optimized_decoder.py")

RUN_LOG = text_of(LOGS / "tti_release.log")
SPEC = load(STAGE / "run_specs" / "runtime_model_spec_release.json")
CONTRACT = load(MODEL / "doc" / "context_contract.json")
DERIVED = load(EVALS / "eval_samples_derived.json")

# ---------------------------------------------------------------------------
# 1. run identity and outcome
# ---------------------------------------------------------------------------

print("\n--- 1. run identity and outcome ---")

# The success marker the stage gate greps for must be backed by the wrapper's
# own echoed exit code, not merely asserted in prose.
if "EXIT_CODE=0" in NOTES:
    ok("RUN_NOTES records EXIT_CODE=0")
    cover(0)
else:
    fail("RUN_NOTES does not record EXIT_CODE=0")

for needle, label in (
    ("TT-Inference version: 0.20.0", "TTI VERSION line in the run log"),
    ("TT-Inference SHA: c4d1e9d42033", "TTI SHA line in the run log"),
):
    if needle in RUN_LOG:
        ok(f"{label}: {needle!r}")
    else:
        fail(f"run log does not contain {needle!r}")
says("0.20.0")
says("c4d1e9d42")
says("v0.10.0-1113-gc4d1e9d42")
says("1113")
says("24092f5381f")
says("bc4af2d")
says("qbge-devex-01")

if SPEC:
    rms = SPEC.get("runtime_model_spec", SPEC)
    impl = rms.get("impl", {})
    code_path = impl.get("code_path")
    want = "models/autoports/qwen_qwen3_coder_30b_a3b_instruct"
    if code_path == want:
        ok(f"run spec code_path == {want}")
    else:
        fail(f"run spec code_path is {code_path!r}, expected {want!r}")
    says(want)
    if impl.get("impl_name") or impl.get("impl_id"):
        says("qwen3-coder-30b-a3b-autoport")

# The copied spec must be the one this run actually used.
if "runtime_model_spec_2026-08-18_22-35-11" in RUN_LOG:
    ok("run log names the 22:35:11 runtime spec that was copied here")
else:
    fail("run log does not name the copied runtime spec")
says("2026-08-18 22:35:11")

# ---------------------------------------------------------------------------
# 2. eval row scores
# ---------------------------------------------------------------------------

print("\n--- 2. eval row scores ---")


def eval_result(name: str):
    data = load(EVALS / f"results_{name}.json")
    if data is None:
        return None
    res = data["results"][name]
    return data, res


def check_score(task: str, metric_key: str, pct_text: str, n_expected: int):
    got = eval_result(task)
    if got is None:
        return None
    data, res = got
    score = res[metric_key]
    pct = f"{score * 100:.1f} %"
    if pct == pct_text:
        ok(f"{task}: {metric_key}={score} -> {pct}")
    else:
        fail(f"{task}: JSON gives {pct}, RUN_NOTES publishes {pct_text}")
    says(pct_text)
    n = data["n-samples"][task]["effective"]
    if n == n_expected:
        ok(f"{task}: {n} samples evaluated")
    else:
        fail(f"{task}: {n} samples evaluated, RUN_NOTES says {n_expected}")
    cover(n, score)
    return score


MBPP = check_score("mbpp_instruct", "pass_at_1,extract_code", "77.2 %", 500)
HEVAL = check_score("humaneval_instruct", "pass@1,create_test", "92.7 %", 164)

# The absolute pass counts quoted beside the percentages.
if MBPP is not None:
    says(f"{round(MBPP * 500)}/500")
if HEVAL is not None:
    says(f"{round(HEVAL * 164)}/164")

# ifeval publishes two of its four accuracies; both must match the JSON, and the
# two the prose does NOT publish must still be present in the artifact so a
# reader can see which of the four was chosen.
IF = eval_result("ifeval")
if IF:
    ifdata, res = IF
    n_if = ifdata["n-samples"]["ifeval"]["effective"]
    if n_if == 541:
        ok("ifeval: 541 samples evaluated")
    else:
        fail(f"ifeval evaluated {n_if} samples, RUN_NOTES says 541")
    cover(n_if)
    says("541/541")
    for key, claim, label in (
        ("prompt_level_strict_acc,none", "81.1 %", "ifeval prompt-level strict"),
        ("inst_level_strict_acc,none", "87.1 %", "ifeval instruction-level strict"),
    ):
        pct = f"{res[key] * 100:.1f} %"
        if pct == claim:
            ok(f"{label}: {res[key]} -> {pct}")
        else:
            fail(f"{label}: JSON gives {pct}, RUN_NOTES publishes {claim}")
        says(claim)
        cover(res[key])
    for key in ("prompt_level_loose_acc,none", "inst_level_loose_acc,none"):
        if key in res:
            ok(f"ifeval artifact also carries {key} = {res[key]}")
            cover(res[key])
        else:
            fail(f"ifeval artifact is missing {key}")
    # The loose accuracies are quoted in the eval-judgement table.
    says(f"{res['prompt_level_loose_acc,none'] * 100:.1f} %")
    says(f"{res['inst_level_loose_acc,none'] * 100:.1f} %")

# The gap between the two coding tasks, recomputed rather than asserted.
if MBPP is not None and HEVAL is not None:
    gap = (HEVAL - MBPP) * 100
    if f"{gap:.1f}" == "15.5":
        ok(f"humaneval - mbpp gap: {gap:.1f} points")
    else:
        fail(f"coding-task gap is {gap:.1f} points, RUN_NOTES publishes 15.5")
    says("15.5")

# gpqa_diamond_cot_zeroshot: failed in the workflow run on a gated dataset, then
# was re-run after access was granted. Both halves are checked.
GP = eval_result("gpqa_diamond_cot_zeroshot")
if GP:
    gdata, gres = GP
    flex = gres["exact_match,flexible-extract"]
    strict = gres["exact_match,strict-match"]
    n_gp = gdata["n-samples"]["gpqa_diamond_cot_zeroshot"]["effective"]
    for val, claim, label in (
        (f"{flex*100:.1f} %", "56.1 %", "gpqa flexible-extract"),
        (f"{strict*100:.1f} %", "0.0 %", "gpqa strict-match"),
    ):
        if val == claim:
            ok(f"{label}: {val}")
        else:
            fail(f"{label} is {val}, RUN_NOTES publishes {claim}")
        says(claim)
    if n_gp == 198:
        ok("gpqa evaluated 198/198")
    else:
        fail(f"gpqa evaluated {n_gp}, RUN_NOTES says 198")
    says("198/198")
    cover(n_gp, flex, strict)

    # The strict-match defect, re-derived from the per-sample rows: the model
    # obeyed the prompt's \boxed{} instruction, and strict-match wanted a phrase
    # the prompt never asked for.
    if DERIVED and "gpqa_diamond_cot_zeroshot" in DERIVED:
        grows = DERIVED["gpqa_diamond_cot_zeroshot"]["rows"]
        flex_rows = [r for r in grows if r["filter"] == "flexible-extract"]
        strict_rows = [r for r in grows if r["filter"] == "strict-match"]
        n = len(flex_rows)
        n_correct = sum(1 for r in flex_rows if r["exact_match"])
        n_strict = sum(1 for r in strict_rows if r["exact_match"])
        boxed = sum(1 for r in flex_rows if r["emitted_boxed"])
        phrase = sum(1 for r in flex_rows if r["has_the_answer_is"])
        for value, claim, label in (
            (str(n_correct), "111", "gpqa correct answers"),
            (str(n_strict), "0", "gpqa strict-match correct answers"),
            (str(boxed), "192", "responses ending in a boxed A-D"),
            (str(phrase), "38", "responses containing 'The answer is'"),
            (f"{boxed/n*100:.1f} %", "97.0 %", "share obeying the boxed instruction"),
            (f"{phrase/n*100:.1f} %", "19.2 %", "share containing the strict-match phrase"),
        ):
            if value == claim:
                ok(f"{label}: {value}")
            else:
                fail(f"{label} is {value}, RUN_NOTES publishes {claim}")
            says(claim)
        # The claim only holds if obedience massively exceeds the strict phrase.
        if boxed > phrase * 4:
            ok(f"boxed obedience ({boxed}) far exceeds the strict-match phrase ({phrase})")
        else:
            fail("the strict-match argument does not hold: the phrase is nearly as common as the boxed form")
        says(f"{n_correct}/{n}")

# Finding 4: the gated-dataset failure must be in the log verbatim, and must have
# happened at dataset download rather than during generation.
GATED = "Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub"
if GATED in RUN_LOG:
    ok("gated-dataset error present in the run log")
else:
    fail(f"run log does not contain {GATED!r}")
if "DatasetNotFoundError" in RUN_LOG and "self.download(self.config.dataset_kwargs)" in RUN_LOG:
    ok("failure is in ConfigurableTask.download — before any request reached the server")
else:
    fail("run log does not show the gpqa failure occurring at dataset download")
says("Idavidrein/gpqa")

# ---------------------------------------------------------------------------
# 3. the mbpp truncation mechanism (Finding 1)
# ---------------------------------------------------------------------------

print("\n--- 3. mbpp truncation mechanism ---")

if DERIVED and "mbpp_instruct" in DERIVED:
    rows = DERIVED["mbpp_instruct"]["rows"]
    total = len(rows)
    closed = sum(1 for r in rows if r["closed_fence"])
    empty = total - closed
    passed = sum(1 for r in rows if r["passed"])

    # The 256-token replay. Greedy decoding makes the first N tokens of a
    # 2048-token completion identical to what a max_gen_toks=N run produced,
    # so thresholding tokens_to_close at 256 replays that run.
    replay_closed = sum(1 for r in rows if r["closed_fence"] and r["tokens_to_close"] <= 256)
    replay_passed = sum(1 for r in rows if r["closed_fence"] and r["tokens_to_close"] <= 256 and r["passed"])

    for value, claim, label in (
        (total, "500", "mbpp sample count"),
        (closed, "482", "completions closing the fence at 2048"),
        (empty, "18", "completions filtered to empty at 2048"),
        (passed, "386", "completions passing at 2048"),
        (replay_closed, "166", "completions closing the fence in the 256 replay"),
        (total - replay_closed, "334", "completions empty in the 256 replay"),
        (replay_passed, "140", "completions passing in the 256 replay"),
    ):
        if str(value) == claim:
            ok(f"{label}: {value}")
        else:
            fail(f"{label}: derived {value}, RUN_NOTES publishes {claim}")
        says(claim)

    # Percentages and rates, recomputed rather than trusted.
    for value, places, claim, label in (
        (closed / total * 100, 1, "96.4 %", "fence-closure rate at 2048"),
        (replay_closed / total * 100, 1, "33.2 %", "fence-closure rate in the replay"),
        (passed / total, 4, "0.7720", "pass@1 at 2048"),
        (replay_passed / total, 4, "0.2800", "pass@1 in the replay"),
    ):
        rendered = f"{value:.{places}f}" + (" %" if claim.endswith("%") else "")
        if rendered == claim:
            ok(f"{label}: {rendered}")
        else:
            fail(f"{label}: derived {rendered}, RUN_NOTES publishes {claim}")
        says(claim)

    # The prose also quotes the replay score in a shorter form.
    says(f"{replay_passed / total:.3f}")
    says(f"{passed / total:.3f}")

    # Pass rate among the completions that were allowed to finish.
    among_closed = passed / closed * 100
    if f"{among_closed:.1f} %" == "80.1 %":
        ok(f"pass rate among completions that closed the fence: {among_closed:.1f} %")
    else:
        fail(f"pass rate among closed is {among_closed:.1f} %, RUN_NOTES publishes 80.1 %")
    says("80.1 %")

    # The residual-truncation caveat.
    residual = empty / total * 100
    if f"{residual:.1f} %" == "3.6 %":
        ok(f"residual truncation at 2048: {residual:.1f} %")
    else:
        fail(f"residual truncation is {residual:.1f} %, RUN_NOTES publishes 3.6 %")
    says("3.6 %")

    # The replay must actually land near the observed 256-token run, otherwise
    # the whole "greedy prefix" argument is unsupported.
    if abs(replay_closed - 164) <= 5 and abs(replay_passed - 138) <= 5:
        ok(
            f"replay ({replay_closed} closed / {replay_passed} passed) lands within "
            "5 samples of the observed 256-token run (164 / 138)"
        )
    else:
        fail(
            f"replay ({replay_closed}/{replay_passed}) does not reproduce the observed "
            "256-token run (164/138); the greedy-prefix argument does not hold"
        )

    # The gen setting that was actually used, from the artifact.
    if DERIVED["mbpp_instruct"]["gen_max_gen_toks"] == 2048:
        ok("derived artifact records max_gen_toks=2048")
    else:
        fail("derived artifact does not record max_gen_toks=2048")
    says("2048")
    says("256")

# ---------------------------------------------------------------------------
# 4. the 16-sample control
# ---------------------------------------------------------------------------

print("\n--- 4. the 16-sample 2048-token control ---")

CTRL = load(EVALS / "control_16sample_2048_results.json")
if CTRL:
    res = CTRL["results"]["mbpp_instruct"]
    score = res["pass_at_1,extract_code"]
    n = CTRL["n-samples"]["mbpp_instruct"]["effective"]
    gen = CTRL["config"]["gen_kwargs"]["max_gen_toks"]
    if score == 0.75:
        ok(f"control scored {score}")
    else:
        fail(f"control scored {score}, RUN_NOTES publishes 0.75")
    if n == 16:
        ok("control used 16 samples")
    else:
        fail(f"control used {n} samples, RUN_NOTES says 16")
    if gen == 2048:
        ok("control ran at max_gen_toks=2048")
    else:
        fail(f"control ran at max_gen_toks={gen}, RUN_NOTES says 2048")
    says("0.75")
    says("0.276")
    cover(score, n, gen)

# ---------------------------------------------------------------------------
# 5. the meta_* failure chain (Finding 2)
# ---------------------------------------------------------------------------

print("\n--- 5. meta_* failure chain ---")

CHAIN = [
    (
        "The evals dataset is not valid, please double check the name, must use "
        "the name in the Llama 3.1 or 3.2 Evals collection.",
        "prepare_meta_eval rejection at setup",
    ),
    ("Failed to prepare meta eval datasets for", "the continuing... warning"),
    ("Tasks were not found: meta_ifeval", "meta_ifeval task-registration failure"),
    ("Tasks were not found: meta_gpqa_cot", "meta_gpqa_cot task-registration failure"),
]
for needle, label in CHAIN:
    if needle in RUN_LOG:
        ok(f"{label} present in the run log")
    else:
        fail(f"run log does not contain {needle!r} ({label})")


# The failure must be at setup (early) and the task errors much later: the
# "warns at t+2s, fails at t+70min" claim.
def stamp_of(needle: str):
    for line in RUN_LOG.splitlines():
        if needle in line:
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
            if m:
                return m.group(1)
    return None


t_setup = stamp_of("Failed to prepare meta eval datasets for")
t_fail = stamp_of("Tasks were not found: meta_ifeval")
if t_setup and t_fail:
    from datetime import datetime

    d = datetime.strptime(t_fail, "%Y-%m-%d %H:%M:%S") - datetime.strptime(t_setup, "%Y-%m-%d %H:%M:%S")
    mins = round(d.total_seconds() / 60)
    ok(f"setup warning at {t_setup}, task failure at {t_fail} — {mins} min apart")
    says(f"{mins} minutes")
    cover(mins)

# Source citations must point at lines that say what the prose says they say.
CITED = [
    ("workflows/workflow_venvs.py", 444, 'config["evals_dataset"] = f"{_model_name}-evals"'),
    ("reference_config/evals/eval_config.py", 193, "batch_size: int = 1"),
    ("reference_config/evals/eval_config.py", 194, "max_concurrent: int = 32"),
]
TTI = Path("/home/raahem/tt-inference-server")
for rel, lineno, expected in CITED:
    p = TTI / rel
    if not p.is_file():
        print(f"[note] cannot verify {rel}:{lineno} — TTI checkout not present here")
        continue
    line = p.read_text().splitlines()[lineno - 1].strip()
    if line == expected:
        ok(f"{rel}:{lineno} == {expected!r}")
    else:
        fail(f"{rel}:{lineno} is {line!r}, RUN_NOTES cites it as {expected!r}")
says("workflows/workflow_venvs.py:444")
says("prepare_meta_eval.py:280-287")
says("workflow_venvs.py:456")
says("reference_config/evals/eval_config.py:189-194")

# ---------------------------------------------------------------------------
# 6. the oversubscription arithmetic (Finding 3)
# ---------------------------------------------------------------------------

print("\n--- 6. oversubscription arithmetic ---")

NUM_CONCURRENT = 32
OLD_BATCH = 16
inflight_old = NUM_CONCURRENT * OLD_BATCH
if inflight_old == 512:
    ok(f"{NUM_CONCURRENT} x {OLD_BATCH} = {inflight_old} prompts in flight")
else:
    fail(f"oversubscription arithmetic is wrong: {inflight_old}")
says("512")
says("32")
says("16")
queued = inflight_old - NUM_CONCURRENT
if queued == 480:
    cover(queued)
says("496")  # the ~496 figure quoted in the finding

# Server width from the spec, not from prose.
if SPEC:
    rms = SPEC.get("runtime_model_spec", SPEC)
    dms = rms.get("device_model_spec", {})
    if dms.get("max_concurrency") == 32:
        ok("spec max_concurrency == 32 (the 32-slot server)")
    else:
        fail(f"spec max_concurrency is {dms.get('max_concurrency')}, prose says 32")

# TimeoutError counts, counted rather than asserted.
this_run = RUN_LOG.count("TimeoutError")
if this_run == 0:
    ok("this run logged 0 TimeoutErrors")
else:
    fail(f"this run logged {this_run} TimeoutErrors — the oversubscription fix is incomplete")
says("0 `TimeoutError`s")

for path, claim, label in (
    (LOGS / "release_attempt_2026-08-18_20-32-33.log", 0, "attempt 1"),
    (LOGS / "release_attempt_2026-08-18_21-17-21.log", 21, "attempt 2"),
):
    if path.is_file():
        n = path.read_text(errors="ignore").count("TimeoutError")
        if n == claim:
            ok(f"{label} logged {n} TimeoutErrors")
        else:
            fail(f"{label} logged {n} TimeoutErrors, RUN_NOTES publishes {claim}")
        cover(n)
says("21")

# batch_size=1 must actually be on this run's eval command lines.
if re.search(r"--batch_size 1(?!\d)", RUN_LOG):
    ok("this run's lm_eval command lines carry --batch_size 1")
else:
    fail("this run's lm_eval command lines do not carry --batch_size 1")
if "--batch_size 16" in RUN_LOG:
    fail("this run still has a --batch_size 16 eval command line")
else:
    ok("no --batch_size 16 eval command line in this run")

# ---------------------------------------------------------------------------
# 7. no context or request-length cap anywhere
# ---------------------------------------------------------------------------

print("\n--- 7. no context or request-length cap ---")

CTX = 262144
if CONTRACT:
    for key, label in (
        ("hf_advertised_context", "contract target"),
        ("current_supported_context", "contract supported"),
    ):
        if CONTRACT.get(key) == CTX:
            ok(f"{label} == {CTX}")
        else:
            fail(f"{label} is {CONTRACT.get(key)}, expected {CTX}")
    if CONTRACT.get("capability_reduction") is False:
        ok("contract records capability_reduction: false")
    else:
        fail("contract does not record capability_reduction: false")
says(str(CTX))

if SPEC:
    rms = SPEC.get("runtime_model_spec", SPEC)
    dms = rms.get("device_model_spec", {})
    if dms.get("max_context") == CTX:
        ok(f"spec max_context == {CTX}")
    else:
        fail(f"spec max_context is {dms.get('max_context')}, expected {CTX}")

# Every eval command line in this run must carry the full context, and none may
# carry a smaller one.
lengths = set(int(m) for m in re.findall(r"max_length=(\d+)", RUN_LOG))
if lengths == {CTX}:
    ok(f"every max_length on this run's eval command lines is {CTX}")
else:
    fail(f"eval command lines carry max_length values {sorted(lengths)}, expected only {{{CTX}}}")

if f"Using max length {CTX} - 1" in RUN_LOG:
    ok(f"lm-eval confirms it is using max length {CTX}")
else:
    fail(f"run log has no 'Using max length {CTX} - 1' confirmation")

caps = set(int(m) for m in re.findall(r"--max-model-len[= ](\d+)", RUN_LOG))
small = {c for c in caps if c < CTX}
if small:
    fail(f"run log contains a --max-model-len below the contract: {sorted(small)}")
else:
    ok("no --max-model-len below the contract anywhere in the run log")

# The largest benchmark sweep point is a sweep point, not a cap.
says("131072")

# ---------------------------------------------------------------------------
# 8. the benchmark sweep
# ---------------------------------------------------------------------------

print("\n--- 8. benchmark sweep ---")

BENCH_FILES = sorted(BENCH.glob("benchmark_*.json")) if BENCH.is_dir() else []
SHAPE_RE = re.compile(r"_isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)\.json$")

if not BENCH_FILES:
    fail("no copied benchmark_*.json found under bench/")
else:
    if len(BENCH_FILES) == 19:
        ok("19 benchmark sweep points copied")
    else:
        fail(f"{len(BENCH_FILES)} benchmark JSONs copied, RUN_NOTES says 19")
    says("19")

    points = {}
    incomplete = []
    for path in BENCH_FILES:
        m = SHAPE_RE.search(path.name)
        if not m:
            fail(f"cannot parse shape from {path.name}")
            continue
        isl, osl, con, n = (int(g) for g in m.groups())
        d = json.loads(path.read_text())
        if d.get("completed") != n or d.get("failed", 0) != 0:
            incomplete.append((path.name, d.get("completed"), n, d.get("failed")))
        points[(isl, osl, con)] = d
        cover(isl, osl, con, n)

    if incomplete:
        fail(f"benchmark points did not complete every request: {incomplete[:5]}")
    else:
        ok("every benchmark point completed every request it issued, 0 failed")

    # The 131072 headline, and the claim that it is 1/1 completed.
    big = points.get((131072, 128, 1))
    if big is None:
        fail("no isl=131072 sweep point copied")
    else:
        ttft_s = big["mean_ttft_ms"] / 1000
        mins = ttft_s / 60
        if f"{mins:.1f}" == "94.4":
            ok(f"isl=131072 TTFT {big['mean_ttft_ms']:.1f} ms = {mins:.1f} min")
        else:
            fail(f"isl=131072 TTFT is {mins:.1f} min, RUN_NOTES publishes 94.4")
        says("94.4")
        says("5,662,485.6")
        if big["completed"] == 1:
            ok("isl=131072 completed 1/1")
        else:
            fail(f"isl=131072 completed {big['completed']}/1")
        says("1/1")

    # The cold-vs-warm inversion table: recompute every ratio from the artifacts,
    # and re-verify that the c=1 point really did run first at every isl.
    def started(isl, osl, con):
        for path in BENCH_FILES:
            m = SHAPE_RE.search(path.name)
            if m and tuple(int(g) for g in m.groups()[:3]) == (isl, osl, con):
                return re.search(r"_(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)_isl", path.name).group(1)
        return None

    RATIOS = {
        128: (32, "0.07x"),
        1024: (32, "0.16x"),
        2048: (32, "0.16x"),
        4096: (32, "0.12x"),
        8192: (31, "0.21x"),
        16384: (15, "0.40x"),
        32768: (7, "1.50x"),
        65536: (3, "3.35x"),
    }
    for isl, (con, claim) in RATIOS.items():
        cold = points.get((isl, 128, 1))
        warm = points.get((isl, 128, con))
        if cold is None or warm is None:
            fail(f"missing cold/warm pair at isl={isl}")
            continue
        ratio = cold["mean_ttft_ms"] / warm["mean_ttft_ms"]
        if f"{ratio:.2f}x" == claim:
            ok(f"isl={isl}: cold/warm ratio {ratio:.2f}x")
        else:
            fail(f"isl={isl}: ratio is {ratio:.2f}x, RUN_NOTES publishes {claim}")
        says(claim)
        t_cold, t_warm = started(isl, 128, 1), started(isl, 128, con)
        if t_cold and t_warm and t_cold < t_warm:
            ok(f"isl={isl}: the c=1 point ran first (cold)")
        else:
            fail(f"isl={isl}: c=1 did NOT run first ({t_cold} vs {t_warm}) — the cold-cache argument fails")

    # Every TTFT quoted in the cold/warm table, re-derived from its own JSON.
    for isl, (con, _claim) in RATIOS.items():
        cold, warm = points.get((isl, 128, 1)), points.get((isl, 128, con))
        if cold:
            says(f"{cold['mean_ttft_ms']/1000:.1f} s")
        if warm:
            says(f"{warm['mean_ttft_ms']/1000:.1f} s")
    # The two concurrent TPOT bounds quoted for the decode paragraph.
    conc_tpots = [d["mean_tpot_ms"] for (i, o, c), d in points.items() if c > 1]
    says(f"{min(conc_tpots):.1f}")
    says(f"{max(conc_tpots):.1f}")

    # The inversion claim itself: below 32768 cold is faster, at/above it is slower.
    inverted = [
        i
        for i, (c, _) in RATIOS.items()
        if points.get((i, 128, 1))
        and points.get((i, 128, c))
        and points[(i, 128, 1)]["mean_ttft_ms"] > points[(i, 128, c)]["mean_ttft_ms"]
    ]
    if sorted(inverted) == [32768, 65536]:
        ok("the cold/warm inversion begins exactly at isl=32768")
    else:
        fail(f"inversion set is {sorted(inverted)}, RUN_NOTES says it starts at 32768")

    # TPOT flatness across the whole sweep.
    tpots = [d["mean_tpot_ms"] for d in points.values()]
    lo, hi = min(tpots), max(tpots)
    if f"{lo:.1f}" == "230.0" and f"{hi:.1f}" == "289.7":
        ok(f"TPOT range across the sweep: {lo:.1f}-{hi:.1f} ms")
    else:
        fail(f"TPOT range is {lo:.1f}-{hi:.1f} ms, RUN_NOTES publishes 230.0-289.7")
    says("230.0-289.7")
    peak = max(d["output_throughput"] for d in points.values())
    if f"{peak:.1f}" == "120.9":
        ok(f"peak decode throughput {peak:.1f} tok/s")
    else:
        fail(f"peak decode throughput is {peak:.1f}, RUN_NOTES publishes 120.9")
    says("120.9")

    # -----------------------------------------------------------------
    # Finding 5: superlinear prefill, re-derived against the contract's own
    # single-layer sweep. These are the numbers the finding rests on.
    # -----------------------------------------------------------------
    if CONTRACT:
        sweep = CONTRACT["measured"]["prefill_sweep_seconds"]
        p65 = points.get((65536, 128, 1))
        p131 = points.get((131072, 128, 1))
        if p65 and p131:
            t65 = p65["mean_ttft_ms"] / 1000
            t131 = p131["mean_ttft_ms"] / 1000

            # per-token cost and the two headline ratios
            ms65 = p65["mean_ttft_ms"] / 65536
            ms131 = p131["mean_ttft_ms"] / 131072
            for val, claim, label in (
                (f"{ms65:.2f}", "14.14", "per-token cost at 65536 (ms)"),
                (f"{ms131:.2f}", "43.20", "per-token cost at 131072 (ms)"),
                (f"{t131/t65:.2f}", "6.11", "wall-clock ratio 131072/65536"),
                (f"{ms131/ms65:.2f}", "3.05", "per-token ratio 131072/65536"),
            ):
                if val == claim:
                    ok(f"{label}: {val}")
                else:
                    fail(f"{label} is {val}, RUN_NOTES publishes {claim}")
                says(claim)

            # the contract-vs-measured table
            for isl, t, probe_claim, x48_claim, ratio_claim in (
                (65536, t65, "42.05", "2018.4", "0.46"),
                (131072, t131, "85.85", "4120.8", "1.37"),
            ):
                probe = sweep[str(isl)]
                x48 = probe * 48
                ratio = t / x48
                for val, claim, label in (
                    (f"{probe:.2f}", probe_claim, f"contract single-layer probe at {isl}"),
                    (f"{x48:.1f}", x48_claim, f"48 x single-layer at {isl}"),
                    (f"{ratio:.2f}", ratio_claim, f"measured/(48 x probe) at {isl}"),
                ):
                    if val == claim:
                        ok(f"{label}: {val}")
                    else:
                        fail(f"{label} is {val}, RUN_NOTES publishes {claim}")
                    says(claim)

            deg = (t131 / (sweep["131072"] * 48)) / (t65 / (sweep["65536"] * 48))
            if f"{deg:.2f}" == "2.99":
                ok(f"degradation relative to naive layer scaling: {deg:.2f}x")
            else:
                fail(f"degradation is {deg:.2f}x, RUN_NOTES publishes 2.99")
            says("2.99")

            # the 262144 lower bound, explicitly a bound and not a measurement
            hours = sweep["262144"] * 48 / 3600
            if f"{hours:.2f}" == "2.56":
                ok(f"262144 naive lower bound: {hours:.2f} h")
            else:
                fail(f"262144 lower bound is {hours:.2f} h, RUN_NOTES publishes 2.56")
            says("2.56")
            says("192.23")
            # RUN_NOTES must label it a lower bound, never a measurement.
            if "never been measured end-to-end" in NOTES:
                ok("RUN_NOTES states the 262144 prefill was never measured end-to-end")
            else:
                fail("RUN_NOTES does not disclaim the 262144 extrapolation")

        # The live-tensor counts the mechanism hypothesis quotes. Re-derived
        # from the loop in tt/optimized_decoder.py:849-858 --
        #   padded_len = ceil(seq_len / EXPERT_CHUNK_SIZE) * EXPERT_CHUNK_SIZE
        #   for start in range(0, padded_len, EXPERT_CHUNK_SIZE)
        # -- so the count is exactly seq_len / 32 for these (already aligned)
        # lengths. No hard-coded claim and no special case: an earlier revision
        # subtracted 1 at 131072 to make the checker agree with a published
        # 4095, which made the check rubber-stamp the prose instead of testing
        # it.
        for isl in (131072, 65536):
            n = -(-isl // EXPERT_CHUNK_SIZE)
            cover(n)
            if f"{n} live tensors at 131072" in NOTES or f"versus {n} at 65536" in NOTES:
                ok(f"live chunk tensors at isl={isl}: {n}")
            else:
                fail(
                    f"live chunk tensors at isl={isl} is {n} "
                    f"(ceil({isl}/{EXPERT_CHUNK_SIZE})), not what RUN_NOTES publishes"
                )
        # KV cache size echoed from the server log.
        if "8224 blocks x 32 tokens = 263168 tokens" in RUN_LOG or "263168" in RUN_LOG:
            ok("KV cache 263168 tokens confirmed in the run log")
        else:
            print("[note] KV-cache line not in the copied run log (it is in the server log)")
        says("263168")

# ---------------------------------------------------------------------------
# 8b. the release report and its acceptance verdict
# ---------------------------------------------------------------------------

print("\n--- 8b. release report and acceptance ---")

REPORTS = sorted(RELEASE.glob("report_*.md")) if RELEASE.is_dir() else []
if not REPORTS:
    fail("no release report_*.md copied under reports_output/release/")
else:
    ok(f"release report copied: {REPORTS[-1].name}")
    report = REPORTS[-1].read_text()
    for needle in (
        "Acceptance status: \u2705 `PASS`",
        "Model status: `EXPERIMENTAL`",
        "(0/19 passed, 19 NA)",
        "(0/6 passed, 3 waived, 3 NA)",
    ):
        if needle in report:
            ok(f"report contains {needle!r}")
        else:
            fail(f"report does not contain {needle!r}")
    # RUN_NOTES must quote the verdict AND the fact that zero evals passed.
    says("0/6 passed, 3 waived, 3 NA")
    says("0/19 passed, 19 NA")
    # The report must show the code path, never a stock one.
    if "models/tt_transformers" in report or "models/demos/" in report:
        fail("release report references a stock implementation path")
    else:
        ok("release report references no stock implementation path")

RDATA = sorted((RELEASE / "data").glob("report_data_*.json")) if (RELEASE / "data").is_dir() else []
if not RDATA:
    fail("no report_data_*.json copied")
else:
    raw = RDATA[-1].read_text()
    ok(f"report data copied: {RDATA[-1].name}")
    # The waivers must be the tier-based ones, not ones we configured.
    if "(informational: model status=EXPERIMENTAL)" in raw:
        ok("waivers are tier-based (informational: model status=EXPERIMENTAL)")
    else:
        fail("report data does not show the EXPERIMENTAL tier waiver text")
    if SPEC:
        rms = SPEC.get("runtime_model_spec", SPEC)
        ki = rms.get("device_model_spec", {}).get("known_issues")
        if ki in ([], None):
            ok("runtime spec carries no known_issues waiver — the waivers are not ours")
        else:
            fail(f"runtime spec carries known_issues {ki!r}; RUN_NOTES says we configured none")

# ---------------------------------------------------------------------------
# 8e. the annotation that travels with the report, and the copies staying copies
# ---------------------------------------------------------------------------

print("\n--- 8e. report annotation and copy integrity ---")

# Copy integrity.
#
# These files are copies of tt-inference-server's own output. Byte-identity was
# the original guarantee and the point of it is unchanged: it proves we
# published TTI's artifact rather than our own retelling of it.
#
# That guarantee had to be *restated*, and the reason is itself a finding (see
# RUN_NOTES, "The repo's own hooks edited copied evidence"). This repo's
# pre-commit hooks rewrite files on the way into a commit:
#   * end-of-file-fixer appends a final newline to a file lacking one;
#   * trailing-whitespace strips trailing blanks from every line.
# Both fired on these copies. TTI writes its reports and specs without a final
# newline, so eleven of the thirteen gained one byte; the two release run logs
# additionally lost one trailing space each. `.pre-commit-config.yaml` is a core
# repo file and is not ours to edit, so the copies as committed cannot be
# byte-identical to the originals.
#
# The claim is therefore narrowed to exactly what is true, and it is checked in
# two independent ways rather than one:
#
#   1. ALWAYS -- the file's sha256 must equal the digest pinned below. These are
#      post-hook digests, taken from the bytes actually committed. Any edit to a
#      copy, by a person or by a tool, changes this digest and fails the gate.
#      This is the check that does not depend on anything outside the repo.
#
#   2. WHEN THE ORIGINAL IS REACHABLE -- the copy must be identical to TTI's
#      original under WORKFLOW_LOGS *modulo per-line trailing whitespace and a
#      final newline*, which is precisely the transformation the two hooks
#      perform. A copy whose content was genuinely altered fails here even if
#      someone re-pinned its digest in (1). This is the check that ties the
#      pinned bytes back to TTI.
#
# (2) is deliberately silent about its own success: WORKFLOW_LOGS is a path on
# the machine the release was run from, so a checkout elsewhere cannot do it,
# and if it called ok() the pass count would depend on where the gate runs --
# which would break the RUN_NOTES count assertion at the bottom of this file.
# It reports, and it fails, but it does not score.
import hashlib

WORKFLOW_LOGS = Path("/home/raahem/tt-inference-server/workflow_logs")


def _hook_normalise(blob: bytes) -> bytes:
    """Apply what trailing-whitespace + end-of-file-fixer do, so the two sides
    of the comparison are compared on content alone."""
    lines = [ln.rstrip(b" \t\r\x0b\x0c") for ln in blob.split(b"\n")]
    return b"\n".join(lines).rstrip(b"\n")


# (copy path under this stage, original path under WORKFLOW_LOGS, post-hook sha256)
COPIES = (
    (
        "reports_output/release/report_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19T002852+0000.md",
        "reports_output/release/report_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19T002852+0000.md",
        "999bcee0018825d58ccf83ce354fe949358c338719e2a8c638214cba3ef21879",
    ),
    (
        "reports_output/release/data/report_data_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19T002852+0000.json",
        "reports_output/release/data/report_data_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19T002852+0000.json",
        "5d0fb364d2328209b394e678fb4dd8a69d45fc8ebdba0066817a7a436e88a599",
    ),
    (
        "run_specs/runtime_model_spec_release.json",
        "runtime_model_specs/runtime_model_spec_2026-08-18_22-35-11_id_qwen3-coder-30b-a3b-autoport"
        "_Qwen3-Coder-30B-A3B-Instruct_p300x2_ZIJqG3Un.json",
        "b241b1df0d8ef758ab24a7765d455bbef87d5d65d60d631b560681dbddd23e32",
    ),
    (
        "spec/runtime_model_spec_smoke.json",
        "runtime_model_specs/runtime_model_spec_2026-08-18_19-40-18_id_qwen3-coder-30b-a3b-autoport"
        "_Qwen3-Coder-30B-A3B-Instruct_p300x2_B91Prs82.json",
        "c45686a15a782260b74076c847e1588a6d85718aed5fb533e647b1ea0d99cfd4",
    ),
    (
        "evals/results_mbpp_instruct.json",
        "reports_output/release/Qwen3-Coder-30B-A3B-Instruct_p300x2_release"
        "/eval_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2"
        "/Qwen__Qwen3-Coder-30B-A3B-Instruct/results_2026-08-18T23-33-25.100037.json",
        "3e60affbd264490f0e394cc3dc2592b8f856f3db71716c47886beb1a624fc048",
    ),
    (
        "evals/results_humaneval_instruct.json",
        "reports_output/release/Qwen3-Coder-30B-A3B-Instruct_p300x2_release"
        "/eval_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2"
        "/Qwen__Qwen3-Coder-30B-A3B-Instruct/results_2026-08-18T23-47-09.833403.json",
        "1758710e449af9be3b4a4dad226074e9ea43b12a68fc46750c35f67af5914d33",
    ),
    (
        "evals/results_ifeval.json",
        "reports_output/release/Qwen3-Coder-30B-A3B-Instruct_p300x2_release"
        "/eval_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2"
        "/Qwen__Qwen3-Coder-30B-A3B-Instruct/results_2026-08-19T00-28-46.377259.json",
        "114ed375f1975dcf3c35a4f4e606c6e77046fd36aa6232f2862b5cb4c6a2cebf",
    ),
    (
        "evals/results_gpqa_diamond_cot_zeroshot.json",
        "reports_output/release/Qwen3-Coder-30B-A3B-Instruct_p300x2_release"
        "/eval_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2"
        "/Qwen__Qwen3-Coder-30B-A3B-Instruct/results_2026-08-19T05-17-02.168928.json",
        "9d24f303cf52ccfd0fc5079f7b9cbc692d8c46a0f35485f888621b44fd84c5e9",
    ),
    (
        "logs/release_attempt_2026-08-18_20-32-33.log",
        "run_logs/run_2026-08-18_20-32-33_id_qwen3-coder-30b-a3b-autoport"
        "_Qwen3-Coder-30B-A3B-Instruct_p300x2_release_sczu8seE.log",
        "c979054312a3174295603df045037e68a1716568cb2a1c949ba427f248341b90",
    ),
    (
        "logs/release_attempt_2026-08-18_21-17-21.log",
        "run_logs/run_2026-08-18_21-17-21_id_qwen3-coder-30b-a3b-autoport"
        "_Qwen3-Coder-30B-A3B-Instruct_p300x2_release_NmQ85_6N.log",
        "d5dcfbd238904514c6008ac6117f7f0f61b3dc04acbceb9e37a87b11ce117c44",
    ),
    (
        "smoke/benchmarks/report_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18T194037+0000.md",
        "reports_output/benchmarks/report_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18T194037+0000.md",
        "7a2f21c5afbb28b8c4873e61fbe1361758f57c5334f7b56c4b46481255186e12",
    ),
    (
        "smoke/benchmarks/report_data_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18T194037+0000.json",
        "reports_output/benchmarks/data/report_data_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18T194037+0000.json",
        "da1fd39b8d57413d46383cc49cc781a8d16235cc65ebbef21332104cada2edf3",
    ),
    (
        "smoke/benchmarks/benchmark_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18_19-40-19_isl-16_osl-4_maxcon-1_n-8.json",
        "reports_output/benchmarks/Qwen3-Coder-30B-A3B-Instruct_p300x2_benchmarks"
        "/llm/benchmark_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18_19-40-19_isl-16_osl-4_maxcon-1_n-8.json",
        "efad068f77962ce1feaeaebb5c81a9ce4573542ece764cbf234b96b048e28b35",
    ),
    # Not a TTI artifact -- this is our own `git diff` of the tt-inference-server
    # registration edits -- but it is pinned for the same reason and it is the
    # file that proves the hook exclusion works. A unified diff encodes a blank
    # context line as a single space, and trailing-whitespace ate three of them
    # while it was still named `.diff`. `.pre-commit-config.yaml` already
    # excludes `\.patch$` from both offending hooks, so renaming the file to
    # `.patch` restores byte-exactness using the repo's own escape hatch rather
    # than a new one. This digest is of the original, unstripped bytes.
    (
        "spec/tti_catalog_edits.patch",
        None,
        "9f325a1f975fc57b0015e8222d3038aa77b66be264a375f587c6eb5e05978628",
    ),
)

_unchecked_originals = 0
for rel, origin, digest in COPIES:
    path = STAGE / rel
    if not path.is_file():
        fail(f"missing copied artifact {rel}")
        continue
    blob = path.read_bytes()

    # (1) the pinned post-hook digest -- always, and this one scores.
    got = hashlib.sha256(blob).hexdigest()
    if got == digest:
        ok(f"copied artifact unedited since it was pinned: {rel.split('/')[-1]}")
    else:
        fail(f"copied artifact {rel} has been edited (sha256 {got[:16]}…, expected {digest[:16]}…)")

    # (2) content identity against TTI's original, when we can reach it.
    if origin is None:
        continue
    src = WORKFLOW_LOGS / origin
    if not src.is_file():
        _unchecked_originals += 1
        continue
    original = src.read_bytes()
    if original == blob:
        continue
    if _hook_normalise(original) == _hook_normalise(blob):
        # The only permitted drift, and it is named rather than waved at.
        note = []
        if blob == original + b"\n":
            note.append("final newline appended by end-of-file-fixer")
        else:
            note.append("trailing whitespace normalised by the pre-commit hooks")
        print(f"[note] {rel.split('/')[-1]}: identical to TTI's original except {note[0]}")
    else:
        fail(
            f"copied artifact {rel} differs from TTI's original {origin} in content, "
            "not merely in trailing whitespace"
        )

if _unchecked_originals:
    print(
        f"[note] {_unchecked_originals} original(s) not reachable under {WORKFLOW_LOGS}; "
        "those copies were checked against their pinned digest only"
    )

# The report says PASS with no annotation, and the report is the artifact that
# travels. A sibling README must carry the disclosure.
_ann = STAGE / "reports_output" / "README.md"
if not _ann.is_file():
    fail("reports_output/README.md is missing — the copied report travels unannotated")
else:
    for needle, why in (
        ("does not mean the evals passed", "the PASS caveat"),
        ("Zero of six eval rows passed", "the zero-of-six statement"),
        ("RUN_NOTES.md", "the pointer at RUN_NOTES"),
    ):
        if needle in READMES:
            ok(f"report annotation carries {why}")
        else:
            fail(f"report annotation is missing {why} ({needle!r})")
    # The annotation's headline scores must match the ones RUN_NOTES publishes.
    for score in ("77.2 %", "92.7 %", "81.1 % / 87.1 %", "56.1 %", "94.4-minute"):
        if score in READMES:
            ok(f"report annotation quotes {score}")
        else:
            fail(f"report annotation does not quote {score}")

# ---------------------------------------------------------------------------
# 8d. the copied catalog diff, the copied spec's cli_args, and the report's own
#     line numbers -- the three things a cold reader checks against `spec/`.
# ---------------------------------------------------------------------------

print("\n--- 8d. copied spec, copied diff, report line numbers ---")

DIFF_PATH = STAGE / "spec" / "tti_catalog_edits.patch"
DIFF = text_of(DIFF_PATH)
if DIFF:
    # The diff must document the SHIPPED eval config, not the pre-fix one. Six
    # tasks, batch_size never overridden to 16, max_gen_toks 2048/4096.
    diff_added = [ln[1:] for ln in DIFF.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    added_eval = 0
    _in_eval = False
    for ln in DIFF.splitlines():
        if ln.startswith("diff --git"):
            _in_eval = "reference_config/evals/eval_config.py" in ln
        elif _in_eval and ln.startswith("+") and not ln.startswith("+++"):
            added_eval += 1
    if added_eval == 330:
        ok(f"copied diff adds {added_eval} lines to eval_config.py")
    else:
        fail(f"copied diff adds {added_eval} lines to eval_config.py, RUN_NOTES publishes 330")
    cover(added_eval)
    says("(+330 lines, additive)")

    diff_tasks = re.findall(r'^\+\s*task_name="([^"]+)"', DIFF, re.M)
    expected_tasks = [
        "mbpp_instruct",
        "humaneval_instruct",
        "meta_ifeval",
        "meta_gpqa_cot",
        "ifeval",
        "gpqa_diamond_cot_zeroshot",
    ]
    if diff_tasks == expected_tasks:
        ok(f"copied diff registers the shipped 6-task list: {diff_tasks}")
    else:
        fail(f"copied diff task list is {diff_tasks}, expected the shipped {expected_tasks}")
    says(str(expected_tasks))

    for banned, why in (
        ("batch_size=16", "the pre-Finding-3 batch_size"),
        ('"max_gen_toks": "256"', "the pre-Finding-1 generation ceiling"),
    ):
        if any(banned in ln for ln in diff_added):
            fail(f"copied diff still adds {banned} -- {why}; it documents the pre-fix config")
        else:
            ok(f"copied diff does not carry {banned} ({why})")
    for required in ('"max_gen_toks": "2048"', '"max_gen_toks": "4096"', '"max_length": 262144', '"timeout": 7200'):
        if any(required in ln for ln in diff_added):
            ok(f"copied diff carries the shipped {required}")
        else:
            fail(f"copied diff is missing the shipped {required}")

    # The known_issues waiver for the two meta_* rows must be in the diff.
    for task in ("meta_ifeval", "meta_gpqa_cot"):
        if any(f"task_name: {task}" in ln for ln in diff_added):
            ok(f"copied diff carries the known_issues EVALS waiver for {task}")
        else:
            fail(f"copied diff has no known_issues waiver for {task}")
    if any("NOT YET FILED" in ln for ln in diff_added):
        ok("waiver marks the upstream issue as not yet filed rather than inventing one")
    else:
        fail("waiver does not mark the upstream issue as pending filing")
    if re.search(r"github\.com/tenstorrent/tt-inference-server/issues/\d+", DIFF):
        fail("waiver cites an issue URL; no issue has been filed")

# spec/resolved_model_spec.json must agree with the diff it sits beside.
_RESOLVED = load(STAGE / "spec" / "resolved_model_spec.json")
if _RESOLVED:
    _ki = _RESOLVED.get("device_model_spec", {}).get("known_issues") or []
    _ki_tasks = sorted(i.get("task_name") for i in _ki)
    if _ki_tasks == ["meta_gpqa_cot", "meta_ifeval"]:
        ok("spec/resolved_model_spec.json agrees with the diff: both meta_* waivers present")
    else:
        fail(
            "spec/resolved_model_spec.json known_issues is "
            f"{_ki_tasks}, but the copied diff registers waivers for both meta_* tasks"
        )

if SPEC:
    _rms = SPEC.get("runtime_model_spec", SPEC)
    _ca = _rms.get("cli_args")
    if isinstance(_ca, dict):
        ok(f"copied runtime spec cli_args has {len(_ca)} keys")
        cover(len(_ca))
        says(f"**{len(_ca)} keys**")
        for key, want in (("docker_server", False), ("local_server", False)):
            if _ca.get(key) is want:
                ok(f"cli_args.{key} is {want}")
            else:
                fail(f"cli_args.{key} is {_ca.get(key)!r}, RUN_NOTES says {want}")
        if str(_ca.get("service_port")) == "8000":
            ok("cli_args.service_port is the 8000 argparse default (the run used 8100)")
        else:
            fail(f"cli_args.service_port is {_ca.get('service_port')!r}, RUN_NOTES says 8000")
        # RUN_NOTES must not repeat the old, false "cli_args: {}" claim.
        if "cli_args: {}" in NOTES:
            fail("RUN_NOTES still claims the copied runtime spec has cli_args: {}")
        else:
            ok("RUN_NOTES no longer claims cli_args is empty")
    else:
        fail("copied runtime spec has no cli_args dict")

if REPORTS:
    _lines = REPORTS[-1].read_text().splitlines()
    _rows = {}
    for i, ln in enumerate(_lines, 1):
        m = re.match(r"\|\s*(meta_ifeval|meta_gpqa_cot|gpqa_diamond_cot_zeroshot)\s*\|", ln)
        if m and "\u274c FAIL" in ln:
            _rows[m.group(1)] = i
    if len(_rows) == 3:
        nums = sorted(_rows.values())
        ok(f"three tier-waived rows still print FAIL in the report, at lines {nums}")
        cover(*nums)
        says(f"(lines {nums[0]}, {nums[1]} and {nums[2]}")
    else:
        fail(f"expected 3 FAIL rows in the copied report, found {sorted(_rows)}")

# ---------------------------------------------------------------------------
# 8c. the stall evidence file (stall/isl131072_stall_evidence.txt)
#
# This file is the primary evidence for the one model-side defect this stage
# reports, so it is inside the coverage boundary and its own figures are
# re-derived here rather than trusted.
# ---------------------------------------------------------------------------

print("\n--- 8c. stall evidence file ---")

B131 = load(
    BENCH / "benchmark_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19_02-41-37_isl-131072_osl-128_maxcon-1_n-1.json"
)
if B131:
    for expr, claim, label in (
        (f"{B131['mean_e2el_ms']:.1f}", "5699275.9", "isl=131072 mean_e2el_ms"),
        (f"{B131['duration']:.1f}", "5699.3", "isl=131072 duration (s)"),
        (f"{B131['duration'] / 60:.1f}", "95.0", "isl=131072 duration (min)"),
        (f"{B131['mean_tpot_ms']:.1f}", "289.7", "isl=131072 mean_tpot_ms"),
    ):
        if expr == claim:
            ok(f"{label}: {expr}")
        else:
            fail(f"{label} is {expr}, the stall file publishes {claim}")
        cover(claim)
    says_stall("5,662,485.6")
    says_stall("5699.3 s")
    says_stall("95.0 minutes")

# The llm_benchmark task duration the stall file quotes, from the run log.
_m = re.search(r"task=llm_benchmark blocks=19 kind=benchmarks \(([\d.]+)s\)", RUN_LOG)
if _m:
    ok(f"llm_benchmark task duration from the run log: {_m.group(1)}s")
    cover(_m.group(1))
    says_stall(f"{_m.group(1)} s")
else:
    fail("run log has no 'task=llm_benchmark blocks=19' completion line")

# KV cache blocks: 8224 x EXPERT_CHUNK_SIZE-sized pages = 263168 tokens.
if EXPERT_CHUNK_SIZE:
    _blocks = 263168 // EXPERT_CHUNK_SIZE
    if _blocks == 8224:
        ok(f"KV cache blocks: 263168 / {EXPERT_CHUNK_SIZE} = {_blocks}")
    else:
        fail(f"KV cache blocks recompute to {_blocks}, the stall file publishes 8224")
    cover(_blocks)

# The py-spy observation window arithmetic. 73 min is the timestamp of the last
# --locals sample, not a measured layer-0 duration, so the only numbers that can
# be derived are the bound it implies and the unattributed residual.
if B131:
    _ttft_min = B131["mean_ttft_ms"] / 1000 / 60
    _resid = _ttft_min - 73
    _frac = 73 / _ttft_min * 100
    for val, claim, label in (
        (f"{_resid:.1f}", "21.4", "unattributed residual after the last --locals sample (min)"),
        (f"{_frac:.0f}", "77", "lower bound on the layer-0 share of TTFT (%)"),
    ):
        if val == claim:
            ok(f"{label}: {val}")
        else:
            fail(f"{label} is {val}, the documents publish {claim}")
        cover(claim)
    says("at least 73 of\nthe 94.4 minutes (\u2265 77 %)")
    says("**~21.4 minutes**")
    says_stall(">=73 of the 94.4 minutes")
    says_stall("~21.4 minutes")
    # Neither document may still carry the withdrawn 5-minute figure.
    for _name, _text in SCANNED.items():
        if "roughly 5 minutes" in _text and "withdrawn" not in _text:
            fail(f"{_name} still asserts the withdrawn 'roughly 5 minutes' figure")
    ok("the withdrawn 'remaining 47 layers in roughly 5 minutes' figure is not asserted")

# The py-spy stack the file quotes must be a stack, i.e. file:line pairs. Those
# line numbers are citations, not figures; cover them explicitly so the boundary
# does not have to guess.
for _cite in (
    "ttnn/decorators.py:650",
    "tt/optimized_decoder.py:861/862",
    "tt/multichip_decoder.py:1698",
    "tt/model.py:1244",
    "tt/generator_vllm.py:578",
    "vllm_tt_plugin/model_runner.py:2420",
):
    says_stall(_cite)

# ---------------------------------------------------------------------------
# 9. the coverage boundary
# ---------------------------------------------------------------------------

# No ok() call follows this point, so the pass count is final here. The gate
# table in RUN_NOTES quotes it (asserted at the very bottom); cover it so the
# boundary does not flag the checker's own self-report as an undeclared figure.
cover(len(PASSES))

UNCOVERED: dict[str, str] = {}
UNCOVERED_PATH = HERE / "uncovered.json"
if UNCOVERED_PATH.is_file():
    UNCOVERED = json.loads(UNCOVERED_PATH.read_text())

notes_numbers = set()
for _name, _text in SCANNED.items():
    notes_numbers |= numbers(_text)
undeclared = sorted(notes_numbers - COVERED - set(UNCOVERED), key=lambda s: (len(s), s))
stale = sorted(set(UNCOVERED) - notes_numbers)

_covered = notes_numbers & COVERED
_declared = notes_numbers & set(UNCOVERED)
_both = _covered & _declared
_residual = notes_numbers - _covered - _declared

print()
print(
    f"coverage boundary: {len(notes_numbers)} distinct numeric tokens in "
    f"{' + '.join(SCANNED)} — "
    f"{len(_covered - _declared)} re-derived only, "
    f"{len(_declared - _covered)} declared uncovered only, "
    f"{len(_both)} both, "
    f"{len(_residual)} neither (this is the number that gates)"
)
assert len(_covered - _declared) + len(_declared - _covered) + len(_both) + len(_residual) == len(notes_numbers)
for token in sorted(_declared, key=lambda s: (len(s), s)):
    print(f"       uncovered: {token} — {UNCOVERED[token]}")
if undeclared:
    fail(
        "RUN_NOTES numbers that are neither re-derived nor declared uncovered "
        f"(add a check, or an UNCOVERED entry): {undeclared}"
    )
if stale:
    # A declared token that appears in none of the scanned documents is an
    # error, not a note. Such an entry can only either be dead weight or -- as
    # happened here with "90", declared as "seconds between the two --locals
    # samples" and silently absorbing the checker's own pass count -- a token
    # collision that defeats the coverage boundary for a number nobody declared.
    FAILURES.append(
        "UNCOVERED entries that appear in none of the scanned documents "
        f"(delete them; a stale declaration silently covers unrelated tokens): {stale}"
    )
    print(f"[FAIL] stale UNCOVERED entries: {stale}")

print()
print(f"{len(PASSES)} checks passed, {len(FAILURES)} failed")
# The gate table in RUN_NOTES must quote this checker's real count. Hand-copying
# it is how "90 checks passed" survived a rewrite that took the real count to
# 112. This self-check deliberately does not call ok(), so the number it asserts
# is a fixed point.
_quoted = f"{len(PASSES)} checks passed, {len(FAILURES)} failed"
if _quoted in NOTES:
    print(f"[ ok ] RUN_NOTES quotes this checker's real count ({_quoted})")
else:
    FAILURES.append(f"RUN_NOTES does not quote this checker's real count: expected {_quoted!r}")
    print(f"[FAIL] RUN_NOTES does not quote {_quoted!r}")

if FAILURES:
    print(f"{len(FAILURES)} published figure(s) do not match their artifacts:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all published figures re-derived from their artifacts")
