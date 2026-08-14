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

import hashlib
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
        "| device-time decode | **22.908 ms/token**",
        "| device-time decode | **12.908 ms/token**",
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
        "**58** cases (46 inherited + 12 new)",
        "**59** cases (46 inherited + 13 new)",
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
        "device time **22.908 ms/token**",
        "device time **12.908 ms/token**",
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
    # Round 10's set: thirteen survived the round-9 gate, every one a figure that occurs twice
    # (so the document-wide search found the other copy) or was never resolved at all.
    (
        "the opening token-out claim",
        "doc/optimized_full_model/README.md",
        "**2.3 % faster token-out decode**",
        "**5.1 % faster token-out decode**",
    ),
    (
        "the opening TTFT claim",
        "doc/optimized_full_model/README.md",
        "plus **22 % faster TTFT** from an",
        "plus **42 % faster TTFT** from an",
    ),
    (
        "the softcap pair total",
        "doc/optimized_full_model/README.md",
        "| pair total | 36.85 µs | **23.79 µs** |",
        "| pair total | 46.85 µs | **13.79 µs** |",
    ),
    (
        "a softcap row against its CSV",
        "doc/optimized_full_model/README.md",
        "| `tanh` (`UnaryDeviceOperation`) | **17.71 µs** | **11.64 µs** |",
        "| `tanh` (`UnaryDeviceOperation`) | **27.71 µs** | **11.64 µs** |",
    ),
    (
        "the roofline layer-weight bytes",
        "doc/optimized_full_model/README.md",
        "4,327,784,448 of layer weights",
        "5,327,784,448 of layer weights",
    ),
    (
        "the assumed DRAM bandwidth",
        "doc/optimized_full_model/README.md",
        "**512 GB/s per device**",
        "**812 GB/s per device**",
    ),
    (
        "the @256 layer-stack floor row",
        "doc/optimized_full_model/README.md",
        "| sliding x39 | 0.4473 | **0.4390** | −1.9 % |",
        "| sliding x39 | 0.4473 | **0.3390** | −21.9 % |",
    ),
    (
        "the layer-stack floor total at 256",
        "doc/optimized_full_model/README.md",
        "| **layer-stack floor** | 22.858 | **22.421** | −1.9 % |",
        "| **layer-stack floor** | 22.858 | **21.421** | −6.3 % |",
    ),
    (
        "the L1 peak column",
        "doc/optimized_full_model/README.md",
        "| **softcap in L1 (shipped)** | **217,088 B** | **1,238,144 B** |",
        "| **softcap in L1 (shipped)** | **117,088 B** | **1,238,144 B** |",
    ),
    (
        "the quoted watcher verdict block",
        "doc/optimized_full_model/README.md",
        "watcher/watcher.log.gz: 6991 lines",
        "watcher/watcher.log.gz: 991 lines",
    ),
    (
        "the qualitative character counts",
        "doc/optimized_full_model/README.md",
        "the same 406/716/638/609/556/682 characters",
        "the same 406/716/638/609/556/482 characters",
    ),
    (
        "the traced-prefill retained DRAM",
        "doc/optimized_full_model/README.md",
        "| DRAM retained per device, 128 rows | 3.3 MB |",
        "| DRAM retained per device, 128 rows | 0.3 MB |",
    ),
    (
        "the shared sampler's retry discarding its orphans",
        "models/common/sampling/generator.py",
        "        self._orphaned_traces = still_held\n        return len(still_held)",
        "        self._orphaned_traces = []\n        return 0",
    ),
    (
        "the measured invalidation cost",
        "doc/optimized_full_model/README.md",
        "with a trace live and the cache unmoved",
        "with a trace live and the cache moved",
    ),
    # Round 11's set: a sample of the 33 that survived the round-10 gate, chosen to cover each
    # mechanism it exposed -- an unbound cell in a bound table, a table whose row count was not
    # asserted, the dtype/fidelity evidence, a unit swap, and a work-log figure.
    (
        "the dtype table's fidelity",
        "doc/optimized_full_model/README.md",
        "| `mlp_down` | 32 x 5120 x 6656 | `BF16 x BFP4 => BF16` | LoFi |",
        "| `mlp_down` | 32 x 5120 x 6656 | `BF16 x BFP4 => BF16` | HiFi4 |",
    ),
    (
        "the dtype table's datatypes",
        "doc/optimized_full_model/README.md",
        "| LM head | 32 x 6656 x 50688 | `BF16 x BFP4 => BF16` | LoFi |",
        "| LM head | 32 x 6656 x 50688 | `BF16 x BFP8 => BF16` | LoFi |",
    ),
    (
        "the headline sampling-trace row",
        "doc/optimized_full_model/README.md",
        "| sampling trace | 0.632 ms | 0.632 ms | unchanged |",
        "| sampling trace | 0.932 ms | 0.632 ms | improved |",
    ),
    (
        "a headline delta cell",
        "doc/optimized_full_model/README.md",
        "**22.656 ms/token · 44.14 t/s/u** | **-2.19 %** |",
        "**22.656 ms/token · 44.14 t/s/u** | **-9.19 %** |",
    ),
    (
        "the fallback audit's trace-replay row",
        "doc/optimized_full_model/README.md",
        "| trace replays | 32 | 1.0 |",
        "| trace replays | 96 | 3.0 |",
    ),
    (
        "the @2048 layer-stack floor row",
        "doc/optimized_full_model/README.md",
        "| sliding x39 | 0.4473 | **0.4390** | −1.9 % |",
        "| sliding x39 | 0.4173 | **0.4390** | −1.9 % |",
    ),
    (
        "a fabricated row in the SwiGLU grid table",
        "doc/optimized_full_model/README.md",
        "| **80** | **1.5248** |",
        "| **80** | **1.5248** |\n| 96 | 1.4111 |",
    ),
    (
        "a unit swap in the invalidation paragraph",
        "doc/optimized_full_model/README.md",
        "signature itself is 7.19 µs of the 7.57",
        "signature itself is 7.19 ms of the 7.57",
    ),
    (
        "a unit swap in the sampling-trace sentence",
        "doc/optimized_full_model/README.md",
        "the sampling trace is 632 µs",
        "the sampling trace is 632 ms",
    ),
    (
        "the CCL host-probe table",
        "doc/optimized_full_model/README.md",
        "| `all_gather_async`, persistent output buffer | 56.18 | 69.09 |",
        "| `all_gather_async`, persistent output buffer | 26.18 | 69.09 |",
    ),
    (
        "the DRAM budget table",
        "doc/optimized_full_model/README.md",
        "| **total long-lived** | **7,178,958,080** | **7.18 GB** of 31.46 GiB |",
        "| **total long-lived** | **9,178,958,080** | **9.18 GB** of 31.46 GiB |",
    ),
    # Round 12's set: the carried-forward decoder contract (ten of its twelve cells were
    # flippable), the terminal term, the stale mutation count, and an indented-table row.
    (
        "the carried-forward KV-cache dtype",
        "doc/optimized_full_model/README.md",
        "| KV-cache dtype | `BFLOAT8_B` | no |",
        "| KV-cache dtype | `BFLOAT16` | no |",
    ),
    (
        "the carried-forward persistent-buffer verdict",
        "doc/optimized_full_model/README.md",
        "| persistent CCL staging buffers | **off**",
        "| persistent CCL staging buffers | **on**",
    ),
    (
        "the carried-forward residual core count",
        "doc/optimized_full_model/README.md",
        "`WIDTH_SHARDED` L1, 16 cores, `[32, 416]` shards",
        "`WIDTH_SHARDED` L1, 8 cores, `[32, 416]` shards",
    ),
    (
        "the carried-forward changed? column",
        "doc/optimized_full_model/README.md",
        "| decode SDPA | `max_cores_per_head_batch=32` | no |",
        "| decode SDPA | `max_cores_per_head_batch=32` | yes |",
    ),
    (
        "the carried-forward SDPA grid",
        "doc/optimized_full_model/README.md",
        "| decode SDPA | `max_cores_per_head_batch=32` | no |",
        "| decode SDPA | `max_cores_per_head_batch=16` | no |",
    ),
    (
        "the carried-forward prefill collective",
        "doc/optimized_full_model/README.md",
        "BFP8 payload, 4 RS workers, AG barrier on",
        "BFP4 payload, 2 RS workers, AG barrier off",
    ),
    (
        "the terminal term",
        "doc/optimized_full_model/README.md",
        "terminal term is now derived from a named id list: **3145**",
        "terminal term is now derived from a named id list: **3146**",
    ),
    (
        "the mutation count in the Artifacts table",
        "doc/optimized_full_model/README.md",
        "mutation-tests that gate: 65 mutations,",
        "mutation-tests that gate: 96 mutations,",
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
    # The gate also reads the shared sampler (round 9 brought the sampling trace inside the
    # fail-closed policy there), so the copy has to include it or the baseline cannot run.
    for shared in (pathlib.Path("models/common/sampling/generator.py"),):
        target = SCRATCH / shared
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC.parents[1] / shared.relative_to("models"), target)
    _bootstrap_scratch_log()


def run_gate() -> bool:
    out = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True, cwd=str(SCRATCH))
    return "FIGURES_OK" in out.stdout


def digest(mutation) -> str:
    """A short hash of one mutation's full content -- name, path, and both texts.

    The gate recomputes these from the table and requires the log to carry the same set, so a
    log cannot outlive an edit to what it claims to have tested.  Round 9 defeated the previous
    form by *neutering* a mutation without re-running: the gate only counted entries, so the
    old log still satisfied it.
    """
    name, rel, old_text, new_text = mutation
    return hashlib.sha256("\x00".join((name, rel, old_text, new_text)).encode()).hexdigest()[:12]


def _bootstrap_scratch_log() -> None:
    """Write a self-consistent placeholder log **into the scratch copy only**.

    The gate asserts that this harness's log matches the mutation table, so the log is an input
    to the very gate the baseline run checks: adding or editing a mutation would fail the
    baseline until a completed run rewrote the log, which cannot happen. Writing the placeholder
    into the copy breaks the cycle without ever putting one in the repo -- round 9 pointed out
    that the previous form bootstrapped the *committed* log, which handed anyone a one-command
    way to produce a passing log with no mutation run behind it.
    """
    placeholder = ["baseline: FIGURES_OK", ""]
    placeholder += [f"CAUGHT  {name}  [{digest(m)}]" for m in MUTATIONS for name in (m[0],)]
    placeholder += ["", f"ALL {len(MUTATIONS)} MUTATIONS CAUGHT"]
    (WORK / LOG.relative_to(SRC)).write_text("\n".join(placeholder) + "\n")


def main() -> int:
    reset()
    if not run_gate():
        say("BASELINE FAILS -- the scratch copy is not clean; aborting")
        LOG.write_text("\n".join(transcript) + "\n")
        return 2
    say("baseline: FIGURES_OK")
    say()
    survivors = []
    for mutation in MUTATIONS:
        name, rel, old, new = mutation
        reset()
        # Paths outside the autoport are relative to the scratch root, not to the model dir.
        path = (SCRATCH / rel) if rel.startswith("models/") else (WORK / rel)
        text = path.read_text()
        if text.count(old) != 1:
            say(f"SKIP  {name}: anchor found {text.count(old)} times  [{digest(mutation)}]")
            survivors.append(f"{name} (anchor)")
            continue
        path.write_text(text.replace(old, new))
        caught = not run_gate()
        say(f"{'CAUGHT' if caught else 'SURVIVED'}  {name}  [{digest(mutation)}]")
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
