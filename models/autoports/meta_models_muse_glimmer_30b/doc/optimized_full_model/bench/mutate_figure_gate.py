# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mutation-test ``check_reported_figures.py``: every defeat a stage review demonstrated.

A figure gate that passes proves nothing on its own -- rounds 3, 6, 7 and 8 each found the
gate passing over a wrong figure, and each time the review had to construct the mutation by
hand to show it.  This runs them: each mutation is applied on its own to a scratch copy of
the model directory, the gate is run there, and the mutation must make it **fail**.  A
``SURVIVED`` line is a hole in the gate.

It is read-only with respect to the repo (it copies to ``/tmp``) and needs no hardware.

Usage::

    python doc/optimized_full_model/bench/mutate_figure_gate.py
"""

import pathlib
import shutil
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
SCRATCH = pathlib.Path("/tmp/muse_glimmer_figure_gate_mutations")
WORK = SCRATCH / "models/autoports" / SRC.name
GATE = WORK / "doc/optimized_full_model/bench/check_reported_figures.py"

MUTATIONS = [
    # (name, relative path, old, new)
    (
        "contract token-out figure",
        "doc/context_contract.json",
        '"token_out_decode_ms_per_token": 23.298420512033115',
        '"token_out_decode_ms_per_token": 99.9',
    ),
    ("contract TTFT figure", "doc/context_contract.json", '"ttft_ms": 63.679240993224084', '"ttft_ms": 9.9'),
    (
        "perf_summary stored roofline fraction",
        "doc/optimized_full_model/perf_summary.json",
        '"roofline_fraction_of_e2e": 0.379',
        '"roofline_fraction_of_e2e": 0.9',
    ),
    (
        "perf_summary logits-only field",
        "doc/optimized_full_model/perf_summary.json",
        '"decode_ms_per_token_e2e_logits_only": 22.656',
        '"decode_ms_per_token_e2e_logits_only": 99.999',
    ),
    (
        "perf_summary prose TTFT figure",
        "doc/optimized_full_model/perf_summary.json",
        "63.68 -> 49.76 ms",
        "63.68 -> 12.34 ms",
    ),
    (
        "perf_summary prose replay figure",
        "doc/optimized_full_model/perf_summary.json",
        "22.656 ms logits-only replay",
        "21.111 ms logits-only replay",
    ),
    (
        "work log token-out figure",
        "doc/optimized_full_model/work_log.md",
        "**23.298 / 22.656 ms**, TTFT 63.68",
        "**99.999 / 22.656 ms**, TTFT 63.68",
    ),
    (
        "work log two-trace residual",
        "doc/optimized_full_model/work_log.md",
        "account for the step to within 9.9 µs",
        "account for the step to within 900 µs",
    ),
    (
        "work log baseline row (two regenerations stale)",
        "doc/optimized_full_model/work_log.md",
        "| 23.844 / 23.164 ms, TTFT 65.41 |",
        "| 23.815 / 23.164 ms, TTFT 65.94 |",
    ),
    (
        "fabricated sixth arm row",
        "doc/optimized_full_model/README.md",
        "| **twelve: the ten + both opt-in `prefill_trace` cases** | ✓ | ✓ | ✓ | ✓ | 6 | **6** |",
        "| **twelve: the ten + both opt-in `prefill_trace` cases** | ✓ | ✓ | ✓ | ✓ | 6 | **6** |\n"
        "| sixteen: the twelve + four more sampling cases | ✓ | ✓ | ✓ | | 9 | **0** |",
    ),
    (
        "prose-only contradictory tally",
        "doc/optimized_full_model/README.md",
        "The work-matched arm is the one round 6 asked for.",
        "On a later re-run the count-matched arm came back 5 of 6. The work-matched arm is the one round 6 asked for.",
    ),
    (
        "control-arm bullet reversed",
        "doc/optimized_full_model/README.md",
        "* **a preceding workload alone *is* sometimes sufficient**",
        "* **the control arm never tripped without the pair, so the pair is required**",
    ),
    (
        "headline table columns swapped (TTFT row)",
        "doc/optimized_full_model/README.md",
        "| **TTFT**, prompt 128, shipped default | 65.41 ms (min of 3) | 63.68 ms (min of 3) |",
        "| **TTFT**, prompt 128, shipped default | 63.68 ms (min of 3) | 65.41 ms (min of 3) |",
    ),
    (
        "headline table columns swapped (token-out row)",
        "doc/optimized_full_model/README.md",
        "| **token-out decode** | 23.844 ms/token · 41.94 t/s/u | **23.298 ms/token · 42.92 t/s/u** |",
        "| **token-out decode** | 23.298 ms/token · 42.92 t/s/u | **23.844 ms/token · 41.94 t/s/u** |",
    ),
    (
        "Bonferroni values falsified",
        "doc/optimized_full_model/README.md",
        "leaves the pooled contrast at p = 0.0012",
        "leaves the pooled contrast at p = 0.0500",
    ),
    (
        "audit SdpaDecode µs",
        "doc/optimized_full_model/README.md",
        "| `SdpaDecode` | 3168 | 15.136 |",
        "| `SdpaDecode` | 3168 | 25.136 |",
    ),
    (
        "audit id moved out of its group",
        "doc/optimized_full_model/README.md",
        "| `plus_one` x2 | 3143, 3254 | 1.845 |",
        "| `plus_one` x2 | 3254 | 1.845 |",
    ),
    (
        "device time in the accounting section",
        "doc/optimized_full_model/README.md",
        "| device-time decode | **22.838 ms/token**",
        "| device-time decode | **12.838 ms/token**",
    ),
    (
        "fallback synchronizations counter",
        "doc/optimized_full_model/README.md",
        "| synchronizations | **0** | **0.0** |",
        "| synchronizations | **32** | **1.0** |",
    ),
    (
        "accuracy top-5",
        "doc/optimized_full_model/README.md",
        "| decode (`run_teacher_forcing`) | bf16 | 0.990 | **1.000** | **1.000** |",
        "| decode (`run_teacher_forcing`) | bf16 | 0.990 | **0.960** | **1.000** |",
    ),
    ("force-argmax flipped", "doc/optimized_full_model/README.md", "force-argmax **off**", "force-argmax **on**"),
    (
        "context reduced",
        "doc/optimized_full_model/README.md",
        "| context | **131072**, unreduced",
        "| context | **65536**, reduced",
    ),
    (
        "suite size",
        "doc/optimized_full_model/README.md",
        "**56** cases (46 inherited + 10 new)",
        "**57** cases (46 inherited + 11 new)",
    ),
    (
        "eligibility bound",
        "doc/optimized_full_model/README.md",
        "config.prefill_chunk_size` (**8192**)",
        "config.prefill_chunk_size` (**4096**)",
    ),
    (
        "eligibility user_id condition dropped",
        "doc/optimized_full_model/README.md",
        "only for `user_id == 0`",
        "for any `user_id`",
    ),
    (
        "arm row tally",
        "doc/optimized_full_model/README.md",
        "| the two opt-in `prefill_trace` cases **alone** | | ✓ | ✓ | ✓ | 4 | **0** |",
        "| the two opt-in `prefill_trace` cases **alone** | | ✓ | ✓ | ✓ | 4 | **2** |",
    ),
    (
        "work log accounting device figure",
        "doc/optimized_full_model/work_log.md",
        "device time **22.838 ms/token**",
        "device time **12.838 ms/token**",
    ),
    (
        "work log per-layer floor row",
        "doc/optimized_full_model/work_log.md",
        "| per-layer floor | `logs/layer_ab_after.log` | 0.4473 / 0.4164 |",
        "| per-layer floor | `logs/layer_ab_after.log` | 0.4573 / 0.4164 |",
    ),
    (
        "accuracy fp32 row top-100",
        "doc/optimized_full_model/README.md",
        "| prefill (`run_prefill_check`) | fp32 control | 0.990 | **1.000** | **1.000** |",
        "| prefill (`run_prefill_check`) | fp32 control | 0.990 | **1.000** | **0.980** |",
    ),
    (
        "the fail-closed negative control's discriminating assertion",
        "doc/optimized_full_model/logs/trace_release_failclosed_negative_control.log",
        "- AssertionError: the failed decode release must still clear the replayed slot",
        "- AssertionError: some other assertion entirely",
    ),
    (
        "the fail-closed negative control's verdict",
        "doc/optimized_full_model/logs/trace_release_failclosed_negative_control.log",
        "1 failed",
        "1 passed",
    ),
    (
        "roofline row in the reconciliation table",
        "doc/optimized_full_model/README.md",
        "| roofline | **8.829 ms/token** |",
        "| roofline | **6.829 ms/token** |",
    ),
]


#: This harness's own log, which ``check_reported_figures.py`` reads: it is written at the
#: *end*, not streamed, because every mutation run copies the model directory -- a log being
#: appended to as the run proceeds would be copied half-written and would fail the very gate
#: the baseline pass is checking.  Redirecting this script's stdout into that path defeats it
#: for the same reason; run it plainly and let it write the file.
LOG = SRC / "doc/optimized_full_model/logs/mutate_figure_gate.log"
transcript: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    transcript.append(line)


def reset():
    shutil.rmtree(WORK.parent, ignore_errors=True)
    WORK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC, WORK)


def run_gate() -> bool:
    out = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True, cwd=str(SCRATCH))
    return "FIGURES_OK" in out.stdout


def _bootstrap_log() -> None:
    """Keep the scratch copy self-consistent when the mutation table has just grown.

    ``check_reported_figures.py`` asserts that this harness's committed log records as many
    caught mutations as the table has entries -- so that a case cannot be quietly dropped from
    the table while the log still advertises it.  That makes the log an input to the very gate
    the baseline run checks, so adding a mutation would fail the baseline until the log is
    rewritten, which only a completed run does.  Placeholder lines break the cycle; the real
    transcript overwrites them at the end of this run, and the committed artifact is always a
    real one.
    """
    wanted = len(MUTATIONS)
    existing = LOG.read_text() if LOG.exists() else ""
    if len([line for line in existing.splitlines() if line.startswith("CAUGHT ")]) == wanted:
        return
    placeholder = ["baseline: FIGURES_OK", ""]
    placeholder += [f"CAUGHT  bootstrap placeholder {i + 1}" for i in range(wanted)]
    placeholder += ["", f"ALL {wanted} MUTATIONS CAUGHT"]
    LOG.write_text("\n".join(placeholder) + "\n")
    print(f"(bootstrapped {LOG.name} to {wanted} entries; this run overwrites it)")


def main() -> int:
    _bootstrap_log()
    reset()
    if not run_gate():
        say("BASELINE FAILS -- the scratch copy is not clean; aborting")
        LOG.write_text("\n".join(transcript) + "\n")
        return 2
    say("baseline: FIGURES_OK")
    say()
    survivors = []
    for name, rel, old, new in MUTATIONS:
        reset()
        path = WORK / rel
        text = path.read_text()
        if text.count(old) != 1:
            say(f"SKIP  {name}: anchor found {text.count(old)} times")
            survivors.append(f"{name} (anchor)")
            continue
        path.write_text(text.replace(old, new))
        caught = not run_gate()
        say(f"{'CAUGHT' if caught else 'SURVIVED'}  {name}")
        if not caught:
            survivors.append(name)
    say()
    if survivors:
        say(f"MUTATION_SURVIVORS: {len(survivors)}")
        for s in survivors:
            say(f"  {s}")
    else:
        say(f"ALL {len(MUTATIONS)} MUTATIONS CAUGHT")
    LOG.write_text("\n".join(transcript) + "\n")
    say(f"wrote {LOG}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
