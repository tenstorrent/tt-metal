# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prove that every assertion in ``check_published_figures.py`` can actually fail.

The stage-05 review of that checker found an assertion whose second clause was
already proven by an assertion above it -- so it could not fail, and a check that
cannot fail is worse than no check because it reads like cover. Finding that one
took a human reading 678 lines. This finds them mechanically.

Method: copy the whole model directory to a scratch tree (hard links, so it is
instant and costs nothing), then apply one **mutation** at a time -- corrupt a
document, corrupt an artifact, rename a file the documents name, flip a boolean
in the contract -- and re-run the checker against the mutated copy. Record which
assertions failed. At the end, **any assertion that never failed under any
mutation is reported**, and the script exits non-zero.

Three things it reports, and any of them makes it exit non-zero:

* an assertion **no mutation could make fail**;
* an assertion made to fail **only by one of the four shotgun mutations**
  (``SHOTGUN`` below), which corrupt every digit or every word of a whole
  document. How many assertions that trips is measured and printed by the run;
  no figure for it is written down here, because the two that were written down
  here both went stale. That coverage is coverage in name only: something else
  in the file failed first for a reason unrelated to the assertion credited;
* a **mutation that broke nothing**, which proves nothing and inflates the
  mutation count the README quotes.

Crediting is keyed by the checker's **stable check id**, not by the formatted
check name. The stage-06 review found that the name was the key and that many
names embed the artifact value under test, so mutating the artifact renamed the
check, the ``FAIL`` line matched nothing from the clean run, and the failure was
silently discarded -- crediting the artifact side of those assertions with zero
failures while the tester still reported full coverage.

Passing this still does not prove the assertions are the *right* ones. What it
establishes is narrower, and the README states it in those terms.

    python mutation_test_checker.py            # full run, ~1 minute
    python mutation_test_checker.py --list      # just name the mutations
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent
MODEL_DIR = DOC.parents[1]
RELATIVE = HERE.relative_to(MODEL_DIR)


# --- mutations ---------------------------------------------------------------
#
# Each takes the copied model root and edits it in place. They are deliberately
# blunt: the question is only "can this assertion ever be false", so a mutation
# that breaks a whole family at once is fine.


def _rewrite(path: Path, text: str) -> None:
    # The copy is made with hard links, so the original must not be written
    # through. Unlink first, then create a fresh file.
    path.unlink()
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            handle.write(text)
        return
    path.write_text(text, encoding="utf-8")


def _read(path: Path) -> str:
    """Read an artifact whether or not it is gzipped.

    The three ``tt-perf-report`` transcripts are archived as ``.txt.gz``: at
    800-1330 KB each they are over the repo's 500 KB artifact limit uncompressed,
    and at 30-56 KB they are comfortably under it compressed. They are still
    plain text as far as every mutation below is concerned.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="ignore").read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _digits(text: str) -> str:
    return re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), text)


def _letters(text: str) -> str:
    return re.sub(r"[A-Za-z]{4,}", "zzzz", text)


def _scale_numbers(value, factor: float = 1.7):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * factor
    if isinstance(value, list):
        return [_scale_numbers(v, factor) for v in value]
    if isinstance(value, dict):
        return {k: _scale_numbers(v, factor) for k, v in value.items()}
    return value


def _mutate_strings(value):
    if isinstance(value, str):
        return "MUTATED"
    if isinstance(value, list):
        return [_mutate_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _mutate_strings(v) for k, v in value.items()}
    return value


def mutate_json_strings(relative: str):
    """Replace every string leaf. Numbers survive, labels do not."""

    def apply(root: Path) -> None:
        path = root / relative
        _rewrite(path, json.dumps(_mutate_strings(json.loads(path.read_text())), indent=2))

    return apply


def mutate_json_op_labels(relative: str):
    """Rename every ``op`` label in a ranking, leaving the numbers alone."""

    def apply(root: Path) -> None:
        path = root / relative
        data = json.loads(path.read_text())

        def rename_ops(node):
            if isinstance(node, dict):
                if "op" in node and isinstance(node["op"], str):
                    node = {**node, "op": "RenamedOp " + node["op"]}
                return {k: rename_ops(v) for k, v in node.items()}
            if isinstance(node, list):
                return [rename_ops(v) for v in node]
            return node

        _rewrite(path, json.dumps(rename_ops(data), indent=2))

    return apply


def mutate_json_list_item(relative: str, index: int, key: str, new):
    """Change one field of one row of a list-shaped artifact."""

    def apply(root: Path) -> None:
        path = root / relative
        data = json.loads(path.read_text())
        data[index][key] = new
        _rewrite(path, json.dumps(data, indent=2))

    return apply


def mutate_ranking_row(relative: str, key: str, op: str, field: str, new):
    """Change one field of the ranking row whose ``op`` label matches."""

    def apply(root: Path) -> None:
        path = root / relative
        data = json.loads(path.read_text())
        row = next(r for r in data[key] if r["op"] == op)
        row[field] = new
        _rewrite(path, json.dumps(data, indent=2))

    return apply


def mutate_json_reverse(relative: str, key: str):
    """Reverse a ranking, so 'X is the largest' becomes false without touching values."""

    def apply(root: Path) -> None:
        path = root / relative
        data = json.loads(path.read_text())
        data[key] = list(reversed(data[key]))
        _rewrite(path, json.dumps(data, indent=2))

    return apply


def mutate_json_swap_largest(relative: str, key: str, field: str):
    """Make the last row of a ranking the largest, leaving every label alone."""

    def apply(root: Path) -> None:
        path = root / relative
        data = json.loads(path.read_text())
        biggest = max(row[field] for row in data[key])
        data[key][-1][field] = biggest * 10.0
        _rewrite(path, json.dumps(data, indent=2))

    return apply


def mutate_json_one_value(relative: str, dotted: str, new):
    """Change a single leaf, so sum/identity checks break where scaling does not."""

    def apply(root: Path) -> None:
        path = root / relative
        data = json.loads(path.read_text())
        node = data
        parts = dotted.split(".")
        # A path segment that is all digits indexes a *list*. Without this the
        # four ``pooled_fit_held_out.N....`` mutations raised "list indices must
        # be integers" and were reported as unappliable -- so the pooled-fit
        # assertions had no targeted mutation at all.
        keys = [int(part) if part.lstrip("-").isdigit() else part for part in parts]
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = new
        _rewrite(path, json.dumps(data, indent=2))

    return apply


def mutate_log_got(relative: str):
    """Change the *measured* side of every boundary tally, not the expected side."""

    def apply(root: Path) -> None:
        path = root / relative
        text = re.sub(
            r"(\S+)\s+(\d+) / (\d+)\s+ok",
            lambda m: f"{m.group(1)}  {int(m.group(2)) + 7} / {m.group(3)}     ok",
            path.read_text(encoding="utf-8", errors="ignore"),
        )
        _rewrite(path, text)

    return apply


def mutate_json(relative: str, factor: float = 1.7):
    def apply(root: Path) -> None:
        path = root / relative
        _rewrite(path, json.dumps(_scale_numbers(json.loads(path.read_text()), factor), indent=2))

    return apply


def mutate_text(relative: str, transform):
    def apply(root: Path) -> None:
        path = root / relative
        _rewrite(path, transform(_read(path)))

    return apply


def _bump_widest(text: str, name: str) -> str:
    """Move the width the README quotes for one non-shotgun mutation by one.

    Limitation 10 writes these as ``` `profile_decode` (80) ```. The value is
    re-derived from each run, so the mutation increments whatever is there
    rather than replacing a literal that would rot into a no-op.
    """
    return re.sub(
        rf"(`{re.escape(name)}` \()(\d+)\)",
        lambda m: f"{m.group(1)}{int(m.group(2)) + 1})",
        text,
    )


def _duplicate_routing_run(root: Path) -> None:
    """Make the third prompt's routing a copy of the second's.

    Three runs of the same prompt are not a three-prompt sample, and the
    assertion that says so has to be able to fail. ``mutate_text`` only sees one
    file's text, so this one takes the tree.
    """
    probes = root / RELATIVE
    source = (probes / "moe_routing_across_tokens_prompt2.json").read_text()
    target = probes / "moe_routing_across_tokens_prompt3.json"
    target.unlink()
    target.write_text(source)


def rename(relative: str):
    def apply(root: Path) -> None:
        path = root / relative
        path.rename(path.with_suffix(path.suffix + ".moved"))

    return apply


def mutate_log_tallies(relative: str):
    """Turn every boundary-check 'ok' into a mismatch, and drop a repeat check."""

    def apply(root: Path) -> None:
        path = root / relative
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"(\d+) / (\d+)\s+ok", lambda m: f"{m.group(1)} / {int(m.group(2)) + 5}   MISMATCH", text)
        text = text.replace("identical to the preceding pass  ok", "DIFFERENT from the preceding pass  MISMATCH")
        _rewrite(path, text)

    return apply


DOCDIR = str(DOC.relative_to(MODEL_DIR))

MUTATIONS: dict = {
    # documents
    "readme_digits": mutate_text(f"{DOCDIR}/README.md", _digits),
    "readme_letters": mutate_text(f"{DOCDIR}/README.md", _letters),
    "work_log_digits": mutate_text(f"{DOCDIR}/work_log.md", _digits),
    "both_docs_digits": lambda root: (
        mutate_text(f"{DOCDIR}/README.md", _digits)(root),
        mutate_text(f"{DOCDIR}/work_log.md", _digits)(root),
    ),
    "both_docs_letters": lambda root: (
        mutate_text(f"{DOCDIR}/README.md", _letters)(root),
        mutate_text(f"{DOCDIR}/work_log.md", _letters)(root),
    ),
    "part1_work_log": mutate_text(f"{DOCDIR}/profile_48layer_work_log.md", _letters),
    # the contract
    "contract_numbers": mutate_json("doc/context_contract.json"),
    # the profile artifacts
    "profile_decode": mutate_json(f"{DOCDIR}/probes/profile_summary_decode.json"),
    "profile_prefill": mutate_json(f"{DOCDIR}/probes/profile_summary_prefill.json"),
    "profile_decode_small": mutate_json(f"{DOCDIR}/probes/profile_summary_decode.json", 1.001),
    # the performance artifacts
    "perf_shipped": mutate_json(f"{DOCDIR}/probes/perf_full_model_p128_argmaxrows.json"),
    "perf_1024": mutate_json(f"{DOCDIR}/probes/perf_full_model_p1024_argmaxrows.json"),
    "perf_4096": mutate_json(f"{DOCDIR}/probes/perf_full_model_p4096_argmaxrows.json"),
    "perf_before_128": mutate_json(f"{DOCDIR}/probes/perf_full_model_p128_before.json"),
    "perf_before_4096": mutate_json(f"{DOCDIR}/probes/perf_full_model_p4096_before.json"),
    "perf_after_128": mutate_json(f"{DOCDIR}/probes/perf_full_model_p128_after.json"),
    "perf_part1_suffixed": mutate_json(f"{DOCDIR}/probes/perf_full_model_part1_preadoption.json", 0.6),
    "perf_canonical": mutate_json(f"{DOCDIR}/probes/perf_full_model.json", 0.6),
    "perf_baseline_stage05": mutate_json("doc/full_model/probes/perf_full_model.json"),
    # the other artifacts
    "footprint": mutate_json(f"{DOCDIR}/probes/footprint_262144.json"),
    "moe_skew": mutate_json(f"{DOCDIR}/probes/moe_skew_analysis_final.json"),
    "audit": mutate_json(f"{DOCDIR}/probes/runtime_fallback_audit.json"),
    "argmax_probe": mutate_json(f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json"),
    "sdpa_depth": mutate_json(f"{DOCDIR}/probes/sdpa_depth_probe.json"),
    "sdpa_sweep": mutate_json(f"{DOCDIR}/probes/sdpa_sweep_confirm_bf16.json"),
    "sdpa_prefill": mutate_json(f"{DOCDIR}/probes/sdpa_prefill_confirm.json"),
    "sdpa_pcc": mutate_json(f"{DOCDIR}/probes/sdpa_hf_pcc_at_depth.json", 0.5),
    "autoregressive_meta": mutate_json("readiness_autoregressive/autoregressive_meta.json"),
    # the logs
    "window_decode_log": mutate_log_tallies(f"{DOCDIR}/logs/window_full_model_48_final.log"),
    "window_prefill_log": mutate_log_tallies(f"{DOCDIR}/logs/window_full_model_48_prefill.log"),
    "prefill_check_log": mutate_text(f"{DOCDIR}/logs/run_prefill_check_argmaxrows.log", _digits),
    "teacher_forcing_log": mutate_text(f"{DOCDIR}/logs/run_teacher_forcing_argmaxrows.log", _digits),
    "stage05_teacher_forcing_log": mutate_text("doc/full_model/run_teacher_forcing.log", _digits),
    "stage05_prefill_log": mutate_text("doc/full_model/run_prefill_check.log", _digits),
    "pytest_log": mutate_text(f"{DOCDIR}/logs/pytest_argmax_rows.log", _digits),
    "degeneracy_log": mutate_text(
        f"{DOCDIR}/logs/check_degenerate_argmaxrows.log",
        lambda t: _digits(t).replace("No degenerate output detected", "DEGENERATE OUTPUT DETECTED"),
    ),
    # the profile reports and rankings the checker parses
    "tt_perf_report": mutate_text(
        f"{DOCDIR}/tt_perf_report_full_model_48layer_decode.txt.gz",
        lambda t: t.replace("MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", "SOMETHING_ELSE").replace(
            "340 GB/s  66.4 %", "111 GB/s  99.9 %"
        ),
    ),
    "part1_ranking": mutate_text(f"{DOCDIR}/rank_full_model_48layer_decode_part1_preadoption.txt", _digits),
    "stage04_window": mutate_text("doc/optimized_multichip_decoder/window_decode.txt", _digits),
    "stage04_perf_csv": mutate_text("doc/optimized_multichip_decoder/perf_decode.csv", _digits),
    # source files the checker reads to keep its own disclosures honest
    # Roll the sampler docstring back to the part-1 figures it used to carry.
    "model_py_docstring_part1": mutate_text(
        "tt/model.py",
        lambda t: t.replace("0.928 ms against 6.155 ms", "0.901 ms against 6.155 ms")
        .replace("6.6x\n        faster", "6.8x\n        faster")
        .replace("**19.693 ms, 50.78 t/s/u**", "**21.461 ms, 46.60 t/s/u**"),
    ),
    # Put the withdrawn cross-accounting claim back, and take the 48-layer
    # pricing that replaced it away.
    "model_py_docstring_cross_accounting": mutate_text(
        "tt/model.py",
        lambda t: t.replace("the claim is withdrawn", "and the sum is essentially all of the non-layer work")
        .replace("essentially all of the\n    ", "essentially all of the ")
        .replace("**366.5 us of an 18889.5 us decode iteration, 1.94%**", "essentially all of the non-layer work"),
    ),
    # Drop the qualifier that explains why the 2-layer window's shares do not
    # scale, leaving the shares themselves in place.
    "model_py_docstring_drops_window_size": mutate_text(
        "tt/model.py", lambda t: t.replace("stage 05's **2-layer** window", "stage 05's window")
    ),
    # Put the two superseded disclosures back into the README, verbatim.
    "readme_reinstates_closed_disclosures": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace(
            "## The rejection ledger",
            "6. **The qualitative six-prompt suite was not re-run this stage.**\n"
            "7. **`probes/perf_full_model.{csv,json}`, unsuffixed, are the part-1 measurement**\n\n"
            "## The rejection ledger",
            1,
        ),
    ),
    # Strip the digits out of both docstrings without touching any artifact.
    "model_py_docstring_digits": mutate_text(
        "tt/model.py",
        lambda t: re.sub(
            r"\d+\.\d+", lambda m: str(round(float(m.group(0)) * 1.7, 3)), t[: t.index("class Qwen3CoderModel")]
        )
        + t[t.index("class Qwen3CoderModel") :],
    ),
    "stage05_2layer_report": mutate_text(
        "doc/full_model/tt_perf_report_full_model_decode.txt",
        lambda t: t.replace("27.5 %", "31.9 %").replace("26.5 %", "30.1 %"),
    ),
    "stage05_2layer_report_rows": mutate_text(
        "doc/full_model/tt_perf_report_full_model_decode.txt",
        lambda t: t.replace("889 μs", "111 μs").replace("859 μs", "222 μs"),
    ),
    # the qualitative suite
    "qualitative_json": mutate_json_list_item(
        f"{DOCDIR}/probes/vllm_qualitative_outputs_argmaxrows.json", 0, "sampled_completion", ""
    ),
    "qualitative_score_log": mutate_text(
        f"{DOCDIR}/logs/check_degenerate_vllm_argmaxrows.log",
        lambda t: t.replace("No degenerate output detected", "DEGENERATE OUTPUT DETECTED")
        .replace("greedy_completion", "greedy_leg")
        .replace("'adjacent_duplication': 0.0,", "'adjacent_duplication': 0.5,"),
    ),
    "qualitative_probe_local_prompts": mutate_text(
        f"{DOCDIR}/probes/qualitative_probe.py",
        lambda t: t.replace("vllm_prompts.txt", "local_prompts.txt").replace(
            "vllm_qualitative_outputs_argmaxrows.json", "doc/full_model/qualitative_check.log"
        ),
    ),
    "qualitative_drop_prompt": mutate_text(
        f"{DOCDIR}/probes/vllm_qualitative_outputs_argmaxrows.json",
        lambda t: json.dumps(json.loads(t)[:-1], indent=2),
    ),
    # This tester's own archived run is an artifact the checker reads.
    "mutation_log_incomplete": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: re.sub(
            r"(\d+) assertions; (\d+) were made to fail", r"\1 assertions; 3 were made to fail", t
        ).replace(
            "every assertion was made to fail by at least one mutation",
            "5 assertions NEVER failed under any mutation",
        ),
    ),
    "rename_mutation_log": rename(f"{DOCDIR}/logs/mutation_test_checker.log"),
    "rename_qualitative_json": rename(f"{DOCDIR}/probes/vllm_qualitative_outputs_argmaxrows.json"),
    "rename_qualitative_score": rename(f"{DOCDIR}/logs/check_degenerate_vllm_argmaxrows.log"),
    "rename_stage05_qualitative": rename("doc/full_model/qualitative_check.log"),
    "rename_perf_part1": rename(f"{DOCDIR}/probes/perf_full_model_part1_preadoption.csv"),
    "model_py_pins_workers": mutate_text(
        "tt/model.py",
        lambda t: t.replace(
            "    def _distributed_argmax_local_vocab(self):",
            "    _pinned = dict(num_workers_per_link=1)\n\n    def _distributed_argmax_local_vocab(self):",
        ),
    ),
    "model_py_sub_core_grids": mutate_text(
        "tt/model.py",
        lambda t: t.replace(
            "    def _distributed_argmax_local_vocab(self):",
            "    _grids = dict(sub_core_grids=None)\n\n    def _distributed_argmax_local_vocab(self):",
        ),
    ),
    "moe_probe_default_csv": mutate_text(
        f"{DOCDIR}/probes/moe_skew_analysis.py",
        lambda t: t.replace(
            "ops_perf_full_model_48layer_decode.csv.gz",
            "ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz",
        ),
    ),
    # This file is itself an artifact the checker reads, to count its own size.
    # The copy under mutation is not the copy being run, so this is safe.
    "mutation_tester_unimportable": mutate_text(
        f"{DOCDIR}/probes/mutation_test_checker.py", lambda t: "raise RuntimeError('mutated')\n" + t
    ),
    "argmax_probe_guard": mutate_text(
        f"{DOCDIR}/probes/argmax_outer_dim_probe.py", lambda t: t.replace("sub_core_grids", "some_other_knob")
    ),
    # artifacts the documents promise exist
    "rename_decode_window": rename(f"{DOCDIR}/ops_perf_full_model_48layer_decode.csv.gz"),
    "rename_decode_report": rename(f"{DOCDIR}/tt_perf_report_full_model_48layer_decode.txt.gz"),
    "rename_decode_rank": rename(f"{DOCDIR}/rank_full_model_48layer_decode.txt"),
    "rename_part1_window": rename(f"{DOCDIR}/ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz"),
    "rename_part1_report": rename(f"{DOCDIR}/tt_perf_report_full_model_48layer_decode_part1_preadoption.txt.gz"),
    "rename_prefill_window": rename(f"{DOCDIR}/ops_perf_full_model_48layer_prefill_s128.csv.gz"),
    "rename_prefill_report": rename(f"{DOCDIR}/tt_perf_report_full_model_48layer_prefill_s128.txt.gz"),
    "rename_part1_log": rename(f"{DOCDIR}/profile_48layer_work_log.md"),
    "rename_ccl_reproducer": rename("doc/full_model/probes/ccl_watcher_ab.py"),
    "rename_argmax_reproducer": rename(f"{DOCDIR}/probes/argmax_outer_dim_probe.py"),
    # the watcher evidence
    "watcher_tripped": lambda root: _corrupt_gz(root / f"{DOCDIR}/logs/watcher_argmaxrows.log.gz"),
    # --- string / label / single-value mutations, which the numeric scalings
    # --- above cannot express
    "profile_decode_labels": mutate_json_op_labels(f"{DOCDIR}/probes/profile_summary_decode.json"),
    "profile_prefill_labels": mutate_json_op_labels(f"{DOCDIR}/probes/profile_summary_prefill.json"),
    "audit_strings": mutate_json_strings(f"{DOCDIR}/probes/runtime_fallback_audit.json"),
    "argmax_probe_labels": mutate_json_strings(f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json"),
    "sdpa_sweep_labels": mutate_json_strings(f"{DOCDIR}/probes/sdpa_sweep_confirm_bf16.json"),
    "sdpa_pcc_leg": mutate_json_strings(f"{DOCDIR}/probes/sdpa_hf_pcc_at_depth.json"),
    "footprint_one_row": mutate_json_one_value(
        f"{DOCDIR}/probes/footprint_262144.json", "stages_gb_per_die.kv_cache", 1.0
    ),
    "footprint_context": mutate_json_one_value(f"{DOCDIR}/probes/footprint_262144.json", "context", 4096),
    "footprint_headroom": mutate_json_one_value(f"{DOCDIR}/probes/footprint_262144.json", "headroom_gb_per_die", 0.5),
    "profile_one_region": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "regions_us.terminal_pre", 500.0
    ),
    "profile_devices": mutate_json_one_value(f"{DOCDIR}/probes/profile_summary_decode.json", "devices", 3),
    "profile_spread": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "device_spread_percent", 9.9
    ),
    "profile_ops_per_device": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "ops_per_device.3", 17
    ),
    "profile_sampler_us": mutate_json_one_value(f"{DOCDIR}/probes/profile_summary_decode.json", "sampler_us", 400.0),
    "prefill_iteration_us": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_prefill.json", "iteration_us", 900000.0
    ),
    "perf_shipped_token_out": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_argmaxrows.json", "token_out_ms", 40.0
    ),
    "perf_shipped_context": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_argmaxrows.json", "context", 4096
    ),
    "perf_shipped_split_token": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_argmaxrows.json", "sampler_split_token", 7
    ),
    "perf_shipped_split_ms": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_argmaxrows.json", "sampler_split_ms", 3.0
    ),
    "perf_4096_token_out_faster": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p4096_argmaxrows.json", "token_out_ms", 60.0
    ),
    "perf_after_128_slower": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_after.json", "token_out_ms", 1.0
    ),
    "perf_before_128_ttft": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_before.json", "ttft_ms", 400.0
    ),
    # the naming rule, both ways: the part-1 file must stay distinguishable from
    # the shipped one, and the canonical file must stay identical to it.
    "perf_part1_matches_shipped": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_part1_preadoption.json", "token_out_ms", 19.692513975314796
    ),
    "perf_part1_context_differs": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_part1_preadoption.json", "context", 8192
    ),
    "perf_canonical_is_superseded": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model.json", "token_out_ms", 21.460929478053004
    ),
    "perf_canonical_row_drift": mutate_json_one_value(f"{DOCDIR}/probes/perf_full_model.json", "sampler_split_ms", 3.0),
    "perf_after_128_context": mutate_json_one_value(
        f"{DOCDIR}/probes/perf_full_model_p128_after.json", "context", 4096
    ),
    "moe_recovery_flag": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "expert_count_recovery.sums_to_top_k_in_every_layer", False
    ),
    "moe_chi2_matches_pass3": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "skew_is_combinatorial.chi2_vs_uniform", 8.062184838677412
    ),
    "moe_corr": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "reduce_scatter_is_wait.corr_lag_vs_attn_rs", 0.95
    ),
    "moe_idle_above_floor": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "budget.measured_idle_ms_per_iteration", 9.0
    ),
    "audit_layers": mutate_json_one_value(f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.num_layers", "2"),
    "audit_host_readback": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.host_logit_readback_on_token_out_path", "True"
    ),
    "audit_vocab_padding": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.vocab_padding", "128"
    ),
    "audit_rows": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "stage06_measured_path.sampler_dist_active_rows", 32
    ),
    "audit_clamped": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "stage06_measured_path.sdpa_decode_k_chunk_clamped", True
    ),
    "audit_prefill_wired": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json",
        "stage06_measured_path.sdpa_prefill_program_config_passed",
        "WIRED",
    ),
    "audit_cache_entries": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "stage06_measured_path.sdpa_decode_config_cache_entries", 96
    ),
    "audit_dist_taken": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json",
        "stage06_measured_path.sampler_distributed_argmax_taken",
        False,
    ),
    "audit_steady_state": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "steady_state_two_tokens.only_replays_moved", False
    ),
    "audit_replays": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "steady_state_two_tokens.counters_after.replays", 99
    ),
    "audit_token_copies": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "steady_state_two_tokens.counters_after.token_host_copies", 5
    ),
    "contract_capability_reduction": mutate_json_one_value("doc/context_contract.json", "capability_reduction", True),
    "contract_context": mutate_json_one_value("doc/context_contract.json", "current_supported_context", 4096),
    "contract_measured_depth": mutate_json_one_value(
        "doc/context_contract.json", "stage06_context_flatness.measured_to_context_tokens", 262144
    ),
    "contract_op_depth": mutate_json_one_value(
        "doc/context_contract.json", "stage06_context_flatness.op_level_evidence_to_cur_pos", 12345
    ),
    "meta_token_count": mutate_json_one_value("readiness_autoregressive/autoregressive_meta.json", "tt.num_tokens", 64),
    "window_decode_log_got": mutate_log_got(f"{DOCDIR}/logs/window_full_model_48_final.log"),
    "window_prefill_log_got": mutate_log_got(f"{DOCDIR}/logs/window_full_model_48_prefill.log"),
    "window_decode_log_devices": mutate_text(
        f"{DOCDIR}/logs/window_full_model_48_final.log",
        lambda t: "\n".join(line for line in t.splitlines() if " device 3 " not in line),
    ),
    "prefill_repeat_counts": mutate_text(
        f"{DOCDIR}/logs/window_full_model_48_prefill.log",
        lambda t: t.replace("4606 ops, identical", "4444 ops, identical"),
    ),
    "pytest_watcher_tally_differs": mutate_text(
        f"{DOCDIR}/logs/pytest_argmax_rows.log", lambda t: t.replace("146 passed", "143 passed")
    ),
    "stage04_window_label": mutate_text(
        "doc/optimized_multichip_decoder/window_decode.txt",
        lambda t: t.replace("last-iteration window rows", "some other rows"),
    ),
    "stage04_csv_ctx": mutate_text(
        "doc/optimized_multichip_decoder/perf_decode.csv", lambda t: t.replace("128,", "129,")
    ),
    "tt_perf_report_no_lm_row": mutate_text(
        f"{DOCDIR}/tt_perf_report_full_model_48layer_decode.txt.gz",
        lambda t: t.replace("MatmulDeviceOperation 32 x 2048 x 37984", "MatmulDeviceOperation 32 x 2048 x 37985"),
    ),
    "degeneracy_metrics_only": mutate_text(
        f"{DOCDIR}/logs/check_degenerate_argmaxrows.log",
        lambda t: re.sub(r"'num_tokens': \d+", "'nothing': 0", t),
    ),
    "prefill_check_aggregate": mutate_text(
        f"{DOCDIR}/logs/run_prefill_check_argmaxrows.log", lambda t: t.replace("AGGREGATE", "SUMMARY")
    ),
    "teacher_forcing_rate": mutate_text(
        f"{DOCDIR}/logs/run_teacher_forcing_argmaxrows.log",
        lambda t: t.replace("decode=42.25 t/s/u", "decode=50.78 t/s/u"),
    ),
    # --- ordering, selective values and missing files ------------------------
    "profile_post_largest": mutate_json_swap_largest(
        f"{DOCDIR}/probes/profile_summary_decode.json", "terminal_post_ranking", "us"
    ),
    "prefill_sdpa_share": mutate_ranking_row(
        f"{DOCDIR}/probes/profile_summary_prefill.json", "ranking", "SDPA 1x8x128x128", "percent", 9.9
    ),
    "decode_sdpa_biggest": mutate_ranking_row(
        f"{DOCDIR}/probes/profile_summary_decode.json",
        "per_layer_ranking",
        "SdpaDecode 1x1x32x128",
        "us_per_layer",
        999.0,
    ),
    "lm_head_far_off_bandwidth": mutate_text(
        f"{DOCDIR}/tt_perf_report_full_model_48layer_decode.txt.gz",
        lambda t: t.replace("340 GB/s  66.4 %", "340 GB/s   1.4 %"),
    ),
    "prefill_tally_family_missing": mutate_text(
        f"{DOCDIR}/logs/window_full_model_48_prefill.log",
        lambda t: "\n".join(line for line in t.splitlines() if "TopKDeviceOperation" not in line),
    ),
    "decode_tally_family_missing": mutate_text(
        f"{DOCDIR}/logs/window_full_model_48_final.log",
        lambda t: "\n".join(line for line in t.splitlines() if "GatherDeviceOperation" not in line),
    ),
    "accuracy_bar_top5_decode": mutate_text(
        f"{DOCDIR}/logs/run_teacher_forcing_argmaxrows.log", lambda t: t.replace("top5=1.000", "top5=0.500")
    ),
    "accuracy_bar_top100_prefill": mutate_text(
        f"{DOCDIR}/logs/run_prefill_check_argmaxrows.log", lambda t: t.replace("top100=1.000", "top100=0.900")
    ),
    "pytest_trailing_failure": mutate_text(
        f"{DOCDIR}/logs/pytest_argmax_rows.log", lambda t: t + "\nERROR: 1 failed after teardown\n"
    ),
    "watcher_trailing_failure": lambda root: _append_gz(
        root / f"{DOCDIR}/logs/watcher_argmaxrows.log.gz", "\nERROR: 1 failed after teardown\n"
    ),
    "sdpa_depth_one_leg": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_depth_probe.json", 0, "ms", 9.9),
    "sdpa_sweep_flat_default": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_sweep_confirm_bf16.json", 0, "us", 9999.0),
    "sdpa_prefill_config_wins": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_prefill_confirm.json", 1, "us", 1.0),
    "sdpa_pcc_leg_name": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_hf_pcc_at_depth.json", 1, "leg", "other"),
    "argmax_keepdim_faster": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "argmax_keepdim_false.ms", 9.9
    ),
    "argmax_slice_slower": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "rm_slice1_then_argmax.ms", 9.9
    ),
    "argmax_slice_token": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "rm_slice1_then_argmax.first4", [7]
    ),
    "argmax_probe_drop_leg": mutate_text(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json",
        lambda t: t.replace('"argmax_keepdim_false"', '"argmax_keepdim_false_RENAMED"'),
    ),
    "part1_ranking_matches_shipped": mutate_text(
        f"{DOCDIR}/rank_full_model_48layer_decode_part1_preadoption.txt",
        lambda t: t.replace("396.904", "384.791"),
    ),
    "part1_sdpa_row": mutate_text(
        f"{DOCDIR}/rank_full_model_48layer_decode_part1_preadoption.txt",
        lambda t: t.replace("20.704", "88.888"),
    ),
    "part1_argmax_cheap": mutate_text(
        f"{DOCDIR}/rank_full_model_48layer_decode_part1_preadoption.txt",
        lambda t: t.replace("366.098", "10.500"),
    ),
    "audit_vocab_not_prime": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.lm_head_local_vocab", "37888"
    ),
    "audit_topology": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.collective_topology", "Topology.Linear"
    ),
    # ``bfloat8_b`` would not do here: the README's audit section names it for
    # the LM head weights, so the required rendering would still be present.
    "audit_kv_dtype": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.kv_cache_dtype", "float32"
    ),
    "audit_kv_paged": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.kv_cache_paged", "False"
    ),
    "audit_lm_dtype": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json", "audit.lm_head_weight_dtype", "DataType.BFLOAT16"
    ),
    "audit_sampling_line": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json",
        "audit.sampling_greedy",
        "Sampling1D force-argmax (all-gather vocab -> ttnn.argmax)",
    ),
    "audit_max_cores": mutate_json_one_value(
        f"{DOCDIR}/probes/runtime_fallback_audit.json",
        "stage06_measured_path.sdpa_decode_max_cores_per_head_batch",
        # not 7: a bare 7 occurs all over a 55 KB README, so the mutation has to
        # ask the README for a number it certainly does not carry.
        4321,
    ),
    "meta_four_matches": lambda root: _make_four_matches(root),
    "accuracy_bar_top5": mutate_text(
        f"{DOCDIR}/logs/run_prefill_check_argmaxrows.log", lambda t: t.replace("top5=1.000", "top5=0.500")
    ),
    "accuracy_bar_top100_decode": mutate_text(
        f"{DOCDIR}/logs/run_teacher_forcing_argmaxrows.log", lambda t: t.replace("top100=1.000", "top100=0.900")
    ),
    "teacher_forcing_rate_gone": mutate_text(
        f"{DOCDIR}/logs/run_teacher_forcing_argmaxrows.log", lambda t: t.replace("decode=", "rate=")
    ),
    "stage05_teacher_forcing_rate_gone": mutate_text(
        "doc/full_model/run_teacher_forcing.log", lambda t: t.replace("decode=", "rate=")
    ),
    "watcher_not_a_session": lambda root: _corrupt_gz_session(root / f"{DOCDIR}/logs/watcher_argmaxrows.log.gz"),
    "prefill_summary_ops": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_prefill.json", "ops_per_device.2", 999
    ),
    "composite_gather_share": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "composite_gather_us", 4321.0
    ),
    "device_spread_us": mutate_json_one_value(f"{DOCDIR}/probes/profile_summary_decode.json", "device_spread_us", 55.0),
    # renames of the text artifacts the documents promise
    "rename_window_decode_log": rename(f"{DOCDIR}/logs/window_full_model_48_final.log"),
    "rename_window_prefill_log": rename(f"{DOCDIR}/logs/window_full_model_48_prefill.log"),
    "rename_degeneracy_log": rename(f"{DOCDIR}/logs/check_degenerate_argmaxrows.log"),
    "rename_pytest_log": rename(f"{DOCDIR}/logs/pytest_argmax_rows.log"),
    "rename_watcher_log": rename(f"{DOCDIR}/logs/watcher_argmaxrows.log.gz"),
    "rename_part1_rank": rename(f"{DOCDIR}/rank_full_model_48layer_decode_part1_preadoption.txt"),
    "rename_autoregressive_meta": rename("readiness_autoregressive/autoregressive_meta.json"),
    # --- the sampler ledger, from the two argmax probe runs ------------------
    "argmax_probe_a_numbers": mutate_json(f"{DOCDIR}/probes/argmax_outer_dim_probe.json"),
    "argmax_probe_a_labels": mutate_text(
        f"{DOCDIR}/probes/argmax_outer_dim_probe.json",
        lambda t: t.replace('"argmax_kd_true_cores', '"argmax_kd_true_RENAMEDcores'),
    ),
    "argmax_sub_core_not_monotonic": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe.json", "argmax_kd_true_cores8.ms", 0.001
    ),
    "argmax_topk_k1_built": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "topk_k1_rm_32rows", {"ms": 1.0}
    ),
    "argmax_tile_slice_wins": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "tile_slice1_plus_untilize.ms", 0.001
    ),
    "argmax_padding_rows_nonzero": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json",
        "padding_rows_produce_token_zero.tokens",
        [0, 5, 5, 5],
    ),
    "argmax_ties_missing": mutate_json_one_value(f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "ties", {}),
    "argmax_drop_full_shipped": mutate_text(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json",
        lambda t: t.replace('"full_shipped_keepdim_true"', '"full_shipped_RENAMED"'),
    ),
    # --- targeted replacements for coverage the shotguns used to supply -------
    #
    # The stage-06 review found 51 assertions whose only "coverage" was one of
    # the four document-wide shotguns. These are the ones worth a mutation of
    # their own: the README's three self-accounting figures, the four audit
    # fields, and the nine artifact names the documents promise.
    "readme_mutation_count": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(r"(\d+) mutations", lambda m: f"{int(m.group(1)) + 2} mutations", t),
    ),
    "readme_figure_count": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(
            r"re-derives (\*{0,2})(\d+) figures", lambda m: f"re-derives {m.group(1)}{int(m.group(2)) + 2} figures", t
        ),
    ),
    "readme_assertion_count": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(
            r"\*\*(\d+) assertions, (\d+) made to",
            lambda m: f"**{int(m.group(1)) + 2} assertions, {int(m.group(2)) + 2} made to",
            t,
        ),
    ),
    # The composite gather's row count, which nothing checked until the stage-06
    # review found the README saying 14 where the artifact says 16.
    "composite_gather_rows": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "composite_gather_rows", 14
    ),
    # The LM-head headroom's two operands, which used to come from two devices.
    "lm_head_slow_device": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "lm_head_us_all_devices.3", 300.0
    ),
    "lm_head_reported_device": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "lm_head_us_all_devices.0", 226.5
    ),
    # Stage 05's own agreement figure, which the README's "Stage 05 measured N
    # matching tokens" sentence is now read from rather than hardcoded against.
    "stage05_agreement": mutate_text(
        "doc/full_model/check_degenerate_output.log",
        lambda t: t.replace("'matching_tokens': 4", "'matching_tokens': 6"),
    ),
    # The free-running run's matching *positions*, as opposed to their count.
    "meta_matching_indices": lambda root: _move_matches(root),
    # Every sampled leg collapses onto its greedy leg: the count goes 4 -> 6 and
    # the README's sentence has to stop being true.
    "qualitative_all_collapse": mutate_text(
        f"{DOCDIR}/probes/vllm_qualitative_outputs_argmaxrows.json",
        lambda t: json.dumps(
            [{**row, "sampled_completion": row["greedy_completion"]} for row in json.loads(t)], indent=2
        ),
    ),
    # Delete the padding-row leg outright. This used to cause zero failures.
    "argmax_drop_padding_leg": mutate_text(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json",
        lambda t: t.replace('"padding_rows_produce_token_zero"', '"padding_rows_RENAMED"'),
    ),
    # --- more targeted replacements, from the first honest coverage run ------
    #
    # Everything below exists because the fixed tester reported the assertion it
    # covers as reachable *only* by a document-wide shotgun.
    "perf_before_1024": mutate_json(f"{DOCDIR}/probes/perf_full_model_p1024_before.json"),
    "perf_after_1024": mutate_json(f"{DOCDIR}/probes/perf_full_model_p1024_after.json"),
    "perf_after_4096": mutate_json(f"{DOCDIR}/probes/perf_full_model_p4096_after.json"),
    "sdpa_depth_leg_1": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_depth_probe.json", 1, "ms", 91.9),
    "sdpa_depth_leg_2": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_depth_probe.json", 2, "ms", 92.9),
    "sdpa_depth_leg_3": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_depth_probe.json", 3, "ms", 93.9),
    "sdpa_sweep_leg_1": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_sweep_confirm_bf16.json", 1, "us", 8881.0),
    # rows 8 and 9 are the cur_pos 255 legs the README quotes, not rows 2 and 3.
    "sdpa_sweep_leg_255_default": mutate_json_list_item(
        f"{DOCDIR}/probes/sdpa_sweep_confirm_bf16.json", 8, "us", 8882.0
    ),
    "sdpa_sweep_leg_255_configured": mutate_json_list_item(
        f"{DOCDIR}/probes/sdpa_sweep_confirm_bf16.json", 9, "us", 8883.0
    ),
    "sdpa_prefill_leg_0": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_prefill_confirm.json", 0, "us", 8884.0),
    "sdpa_prefill_leg_1": mutate_json_list_item(f"{DOCDIR}/probes/sdpa_prefill_confirm.json", 1, "us", 8885.0),
    "profile_terminal_post": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "regions_us.terminal_post", 7777.0
    ),
    "profile_lm_head_us": mutate_json_one_value(f"{DOCDIR}/probes/profile_summary_decode.json", "lm_head_us", 7778.0),
    # Rewrite only the device-0 summary line of the superseded ranking, keeping
    # its shape so the checker's regex still parses it -- the whole-file digit
    # mutation breaks the parse instead, which skips the checks rather than
    # failing them.
    "part1_rank_device0_line": mutate_text(
        f"{DOCDIR}/rank_full_model_48layer_decode_part1_preadoption.txt",
        lambda t: t.replace(
            "device 0: total  19,926.5 us = pre    53.3 + 48 layers  19,051.4 + terminal   821.7   "
            "(per layer 396.904 us)",
            "device 0: total  18,111.2 us = pre    41.7 + 48 layers  17,222.3 + terminal   847.2   "
            "(per layer 358.799 us)",
        ),
    ),
    "stage04_ctx128_median": mutate_text(
        "doc/optimized_multichip_decoder/perf_decode.csv",
        lambda t: t.replace("128,0.4286,", "128,0.8642,"),
    ),
    "readme_drops_figure_claim": mutate_text(f"{DOCDIR}/README.md", lambda t: t.replace("re-derives", "covers")),
    "stage04_window_kernel_time": mutate_text(
        "doc/optimized_multichip_decoder/window_decode.txt",
        lambda t: re.sub(
            r"last-iteration window rows ([\d-]+):\s*([\d,]+\.\d+) us", r"last-iteration window rows \1: 8,886.41 us", t
        ),
    ),
    "prefill_check_top1": mutate_text(
        f"{DOCDIR}/logs/run_prefill_check_argmaxrows.log", lambda t: t.replace("top1=0.980", "top1=0.870")
    ),
    "teacher_forcing_top1": mutate_text(
        f"{DOCDIR}/logs/run_teacher_forcing_argmaxrows.log", lambda t: t.replace("top1=0.990", "top1=0.870")
    ),
    "watcher_tally_only": lambda root: _rewrite_gz(
        root / f"{DOCDIR}/logs/watcher_argmaxrows.log.gz", lambda t: t.replace("146 passed", "8887 passed")
    ),
    "tt_perf_report_lm_bandwidth": mutate_text(
        f"{DOCDIR}/tt_perf_report_full_model_48layer_decode.txt.gz",
        lambda t: t.replace("340 GB/s  66.4 %", "777 GB/s  66.4 %"),
    ),
    "tt_perf_report_lm_device": mutate_text(
        f"{DOCDIR}/tt_perf_report_full_model_48layer_decode.txt.gz",
        lambda t: t.replace(
            "MatmulDeviceOperation 32 x 2048 x 37984                           3       236",
            "MatmulDeviceOperation 32 x 2048 x 37984                           1       236",
        ),
    ),
    "lm_head_devices_equal": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "lm_head_us_all_devices.3", 226.13
    ),
    "lm_head_drop_slow_device": lambda root: _drop_json_key(
        root / f"{DOCDIR}/probes/profile_summary_decode.json", "lm_head_us_all_devices", "3"
    ),
    "lm_head_reported_device_disagrees": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_decode.json", "lm_head_us_all_devices.0", 226.5
    ),
    "moe_independence_caveat": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json",
        "skew_is_combinatorial.independence_caveat",
        "the counts are independent draws",
    ),
    "moe_chi2_df": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "skew_is_combinatorial.chi2_df", 17
    ),
    "moe_chi2_p": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "skew_is_combinatorial.chi2_p_value", 0.71
    ),
    "moe_per_die_chi2": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "skew_is_combinatorial.per_die_chi2", 71.77
    ),
    "moe_per_die_p": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json", "skew_is_combinatorial.per_die_p_value", 0.71
    ),
    "moe_per_die_totals": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json",
        "skew_is_combinatorial.per_die_total_active_experts",
        [71, 117, 107, 89],
    ),
    "moe_chi2_pooling": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_skew_analysis_final.json",
        "skew_is_combinatorial.chi2_observed_pooled",
        {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1},
    ),
    "argmax_candidate_batch1": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "full_candidate_batch1.ms", 0.4321
    ),
    "readme_drops_host_boundary_fields": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace("host_logit_readback_on_token_out_path", "logits_never_reach_the_host"),
    ),
    "readme_drops_audit_section": mutate_text(
        f"{DOCDIR}/README.md", lambda t: t.replace("## Runtime fallback audit", "## Some other section")
    ),
    "readme_drops_bug_list": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t[: t.index("## Three upstream bugs")]
        + "## Three upstream bugs\n\nNothing to report.\n\n"
        + t[t.index("## Named limitations") :],
    ),
    "readme_drops_part1_perf_name": lambda root: (
        mutate_text(f"{DOCDIR}/README.md", lambda t: t.replace("perf_full_model_part1_preadoption", "perf_old"))(root),
        mutate_text(f"{DOCDIR}/work_log.md", lambda t: t.replace("perf_full_model_part1_preadoption", "perf_old"))(
            root
        ),
    ),
    "readme_reinstates_arbitrary_partition": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace(
            "## Runtime fallback audit",
            "The shipped layout is already better than the expectation for an arbitrary partition, "
            "so a permutation is negative in expectation.\n\n## Runtime fallback audit",
            1,
        ),
    ),
    "readme_drops_one_token_caveat": mutate_text(
        f"{DOCDIR}/README.md", lambda t: t.replace("single decode token", "representative sample")
    ),
    "part1_csv_drops_a_collective": lambda root: _drop_csv_rows(
        root / f"{DOCDIR}/ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz",
        "ReduceScatterMinimalAsyncDeviceOperation",
        3,
    ),
    "part1_csv_reduce_scatter_slower": lambda root: _scale_csv_rows(
        root / f"{DOCDIR}/ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz",
        "ReduceScatterMinimalAsyncDeviceOperation",
        1.31,
    ),
    "rename_stage05_degeneracy_log": rename("doc/full_model/check_degenerate_output.log"),
    "mutation_log_shotgun_only": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: t + "\n3 assertions failed ONLY under a shotgun mutation, which trips 200+\n",
    ),
    "mutation_log_dead_mutation": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: t + "\n2 mutations BROKE NOTHING. A mutation that changes no assertion's outcome is\n",
    ),
    "mutation_log_targeted_tally": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: re.sub(
            r"\d+ of those were made to fail by a mutation that is not one of the",
            "7 of those were made to fail by a mutation that is not one of the",
            t,
        ),
    ),
    # --- the multi-token routing sample, which nothing mutated until the
    # --- round-2 review found the README quoting a superseded n=2 range -------
    "routing_run1_numbers": mutate_json(f"{DOCDIR}/probes/moe_routing_across_tokens.json"),
    "routing_run2_numbers": mutate_json(f"{DOCDIR}/probes/moe_routing_across_tokens_prompt2.json"),
    "routing_run3_numbers": mutate_json(f"{DOCDIR}/probes/moe_routing_across_tokens_prompt3.json"),
    "routing_one_token_only": mutate_json_one_value(f"{DOCDIR}/probes/moe_routing_across_tokens.json", "tokens", 1),
    "routing_hotness_uniform": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_across_tokens.json", "per_expert_hotness.mean_top8_share", 0.0625
    ),
    "routing_per_die_counts": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_across_tokens.json",
        "per_die_counts.per_die_total_selections",
        [9111, 9222, 9333, 9444],
    ),
    "routing_no_overfitting_gap": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_across_tokens.json",
        "permutation_search.fitted_mean_max_k_per_layer_held_out",
        1.0,
    ),
    "routing_held_out_gain": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_across_tokens_prompt2.json",
        "permutation_search.held_out_gain_ms_per_iteration",
        9.111,
    ),
    "cross_prompt_numbers": mutate_json(f"{DOCDIR}/probes/moe_routing_cross_prompt.json"),
    "cross_prompt_min": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json", "gain_ms_per_iteration_min", 9.222
    ),
    "cross_prompt_max": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json", "gain_ms_per_iteration_max", 9.333
    ),
    "cross_prompt_mean": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json", "gain_ms_per_iteration_mean", 9.444
    ),
    "cross_prompt_directions": mutate_json_one_value(f"{DOCDIR}/probes/moe_routing_cross_prompt.json", "directions", 2),
    "cross_prompt_pooling_worse": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "pooled_fit_held_out.0.pooling_transfers_better",
        False,
    ),
    "cross_prompt_pooled_gain": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "pooled_fit_held_out.1.pooled_gain_ms_per_iteration",
        9.555,
    ),
    "cross_prompt_pooled_leaks": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "pooled_fit_held_out.0.pooled_over_indices",
        [0, 1, 2],
    ),
    # -- the round-2 review's additions ---------------------------------------
    #
    # The committed readiness artifact was unchecked once the checker moved to
    # the evidence-tree copy; these two break the tie that now binds them.
    "readiness_qualitative_moved": rename("readiness_qualitative/vllm_qualitative_outputs.json"),
    "readiness_qualitative_drifts": mutate_text(
        "readiness_qualitative/vllm_qualitative_outputs.json",
        lambda t: t.replace("sampled_completion", "sampled_completion_DRIFTED", 1),
    ),
    # the structural corroboration of the cross-prompt fit
    "cross_prompt_structure_overlap": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "shared_structure.mean_top8_overlap_over_pairs",
        0.51,
    ),
    "cross_prompt_structure_chance": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "shared_structure.top8_overlap_under_independent_routing",
        9.666,
    ),
    "cross_prompt_structure_rank_corr": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "shared_structure.mean_rank_correlation_over_pairs",
        9.777,
    ),
    # the pooled ceiling that sets the top of the published range
    "cross_prompt_pooled_ceiling": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_cross_prompt.json",
        "pooled_fit_held_out.2.pooled_gain_ms_per_iteration",
        0.001,
    ),
    # the published range itself, in the sentence that publishes it
    "readme_drops_published_range": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace("0.024\u20130.112 ms/iteration", "0.024\u20130.111 ms/iteration"),
    ),
    "readme_drops_published_percent": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace("0.12\u20130.57%", "0.12\u20130.56%"),
    ),
    "readme_undeclines_the_lever": mutate_text(f"{DOCDIR}/README.md", lambda t: t.replace("declined", "accepted")),
    # the measured shotgun breadth and the coverage gap measuring it exposed
    "mutation_log_shotgun_breadth": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: re.sub(
            r"(measured breadth of the \d+ declared shotguns: )\d+-\d+ assertions \([^)]*\)",
            r"\g<1>90001-90004 assertions (both_docs_digits 90004, both_docs_letters 90002, "
            r"readme_digits 90003, readme_letters 90001)",
            t,
        ),
    ),
    "mutation_log_coverage_gap": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: re.sub(
            r"(\d+) of (\d+) assertions have NO mutation narrower than (\d+)",
            r"1 of \2 assertions have NO mutation narrower than \3",
            t,
        ),
    ),
    # Targeted mutations for assertions the round-2 run found were covered only
    # by a document-wide shotgun, or by nothing at all.
    #
    # Ratios survive a whole-file scale, so the mutation has to move ONE operand.
    "prefill_iteration_only": mutate_json_one_value(
        f"{DOCDIR}/probes/profile_summary_prefill.json", "iteration_us", 99999.0
    ),
    "argmax_candidate_only": mutate_json_one_value(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json", "full_candidate_batch1.ms", 0.27
    ),
    # the two per-run overfitting-gap assertions on prompts 2 and 3
    "routing_p2_no_overfitting_gap": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_across_tokens_prompt2.json",
        "permutation_search.fitted_mean_max_k_per_layer_held_out",
        0.001,
    ),
    "routing_p3_no_overfitting_gap": mutate_json_one_value(
        f"{DOCDIR}/probes/moe_routing_across_tokens_prompt3.json",
        "permutation_search.fitted_mean_max_k_per_layer_held_out",
        0.001,
    ),
    # three prompts that are really the same prompt three times
    "routing_p3_duplicates_p2": _duplicate_routing_run,
    # the MoE ledger row itself
    "readme_drops_moe_ledger_row": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: "\n".join(line for line in t.splitlines() if "Permuting experts across dies" not in line),
    ),
    # the two anchored mutation-count sentences, each corrupted where it stands
    "readme_miscounts_perturbations": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(r"(a specific, listed set\s+of )\d+( perturbations)", r"\g<1>212\g<2>", t),
    ),
    "readme_miscounts_corruptions": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(r'("this set of )\d+( corruptions is)', r"\g<1>212\g<2>", t),
    ),
    "readme_reinstates_two_prompt_range": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace(
            "## Runtime fallback audit", "It is worth 0.024\u20130.028 ms.\n\n## Runtime fallback audit", 1
        ),
    ),
    "readme_reinstates_zero_ms": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: t.replace("## Runtime fallback audit", "achievable saving **0 ms**\n\n## Runtime fallback audit", 1),
    ),
    "argmax_drop_ledger_legs": mutate_text(
        f"{DOCDIR}/probes/argmax_outer_dim_probe_b.json",
        lambda t: t.replace('"topk_k32_tile_32rows"', '"RENAMED_a"')
        .replace('"tile_slice1_plus_untilize"', '"RENAMED_b"')
        .replace('"untilize_37984"', '"RENAMED_c"')
        .replace('"full_candidate_batch1"', '"RENAMED_d"'),
    ),
    # --- the two sites that quote the measured shotgun breadths --------------
    #
    # These exist to prove the *anchoring*, not just the numbers. The README
    # states the four measured breadths twice: once in limitation 10 and once in
    # the "coverage that comes only from a shotgun" bullet. While either site
    # was checked by a document-wide `appears(README, value)` search, a garbled
    # copy at one site was held up by the intact copy at the other -- which is
    # exactly how limitation 10 sat at "235 and 259" through a green run.
    #
    # Each mutation below corrupts ONE site. Each must break ONE assertion: its
    # own. If either mutation trips both, the anchoring is not anchoring, and if
    # either trips none, the site has no assertion of its own. The bare-integer
    # `appears()` checks stay green under both, because the *other* site still
    # carries all four values -- which is the point being demonstrated.
    "readme_limitation10_breadths": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(
            r"(the four declared shotguns trip \*\*)[\d,\s]+?and\s+\d+(\*\* assertions)",
            r"\g<1>90001, 90002, 90003 and 90004\g<2>",
            t,
        ),
    ),
    # NOTE the anchor here is a regex over the *current* wording, matched with
    # ``\s+`` between words so it survives re-wrapping, and it must be kept in
    # step with the phrase ``check_published_figures.py`` anchors that site to.
    # It read ``the round-2 review measured it: they trip **...**`` until that
    # wording was replaced -- the mutation then matched nothing, silently became
    # a no-op, and the second site went uncovered. A literal that names a thing
    # that can be edited out is a mutation waiting to rot; if this one ever
    # reports BROKE NOTHING, the wording moved and the anchor must follow it.
    "readme_shotgun_section_breadths": mutate_text(
        f"{DOCDIR}/README.md",
        lambda t: re.sub(
            r"(the\s+shipped\s+run\s+reports\s+\*\*)[\d,\s]+?and\s+\d+(\*\*)",
            r"\g<1>90001, 90002, 90003 and 90004\g<2>",
            t,
        ),
    ),
    # --- the three widest mutations that are NOT declared shotguns -----------
    #
    # Limitation 10 names these to make its point that "shotgun" is a declared
    # list rather than a width. All three figures are read from the archived log
    # and had no assertion of any kind before this stage; now each has one, so
    # each gets a mutation of its own.
    # Written as increments rather than literal replacements: the three widths
    # are re-derived from every run, so a mutation that hardcoded the current
    # value would silently become a no-op the moment the figure moved.
    # ``\w+`` would run past the closing backtick, so the name is delimited --
    # otherwise ``profile_decode`` also matches inside ``profile_decode_small``.
    "readme_widest_profile_decode": mutate_text(f"{DOCDIR}/README.md", lambda t: _bump_widest(t, "profile_decode")),
    "readme_widest_profile_decode_small": mutate_text(
        f"{DOCDIR}/README.md", lambda t: _bump_widest(t, "profile_decode_small")
    ),
    "readme_widest_perf_shipped": mutate_text(f"{DOCDIR}/README.md", lambda t: _bump_widest(t, "perf_shipped")),
    # ...and the log line those three are read from. Removing it is what covers
    # the assertion that the archived run *reports* its widest non-shotgun
    # mutations at all; the three README assertions above then have nothing to
    # compare against and stop being emitted, so this one is wider than a single
    # assertion by construction.
    "mutation_log_drops_widest": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: "\n".join(line for line in t.splitlines() if "widest mutations NOT declared shotgun:" not in line),
    ),
    # --- the archive's provenance --------------------------------------------
    #
    # A ``--bootstrap`` run is a stepping stone whose measured breadths are
    # inflated, and it says so in a banner. Nothing read the banner: pasting one
    # on top of the archive, together with a clean-tree line reporting failures,
    # passed every assertion the checker made. This mutation is that laundering,
    # performed exactly as it would be, and it must break the assertion that
    # says the archive came from a run whose clean tree was green.
    "mutation_log_launders_bootstrap": mutate_text(
        f"{DOCDIR}/logs/mutation_test_checker.log",
        lambda t: "*" * 78 + "\n"
        "*  --bootstrap: running against a clean tree that is NOT green. 2 check(s)\n"
        "*    - PERMITTED BY --bootstrap: some self-referential count check\n"
        "*  THIS LOG IS NOT THE ARCHIVE. Archive it, re-derive the\n"
        + "*" * 78
        + "\n"
        + re.sub(r"clean tree: (\d+) checks, 0 failing", r"clean tree: \1 checks, 2 failing", t, count=1),
    ),
}


#: Mutations that corrupt an **entire document** -- every digit, or every word.
#: "This assertion failed under ``readme_digits``" is nearly no evidence about
#: that assertion in particular: something else in the same file failed first for
#: an unrelated reason. Coverage that comes only from these is reported
#: separately and counted as a gap.
#:
#: This is a **declared** property, not a measured breadth, and the round-2
#: review was right that the two were being conflated: "200+ assertions at once"
#: was true of the two digit shotguns and badly wrong about the two letter ones.
#:
#: The four measured breadths are deliberately **not** written here. They moved
#: twice while this comment stood -- it said "236, 39, 260 and 44" two revisions
#: after the run stopped measuring that, and nothing asserted it, which is the
#: exact defect this tool exists to remove. The run measures them, prints them
#: on the ``measured breadth of the N declared shotguns`` line, and
#: ``check_published_figures.py`` ties the README's two statements of them to
#: that line. Read the numbers there. The same goes for the widest mutations
#: that are *not* shotguns: the run prints them on the next line, and they are
#: routinely wider than the letter shotguns, which is the point -- breadth and
#: "corrupts a whole document" are different axes and the gate is the second.
SHOTGUN = {"readme_digits", "readme_letters", "both_docs_digits", "both_docs_letters"}


#: The artifacts the documents promise by name. ``check_published_figures.py``
#: asserts, for each, both that the file exists and that a document names it.
#: The second half used to be covered only by the letter shotguns, so each name
#: gets a mutation that removes exactly that one mention from both documents.
NAMED_ARTIFACTS = (
    "ops_perf_full_model_48layer_decode.csv.gz",
    "tt_perf_report_full_model_48layer_decode.txt.gz",
    "rank_full_model_48layer_decode.txt",
    "ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz",
    "tt_perf_report_full_model_48layer_decode_part1_preadoption.txt.gz",
    "rank_full_model_48layer_decode_part1_preadoption.txt",
    "ops_perf_full_model_48layer_prefill_s128.csv.gz",
    "tt_perf_report_full_model_48layer_prefill_s128.txt.gz",
    "profile_48layer_work_log.md",
)


def _unname_artifact(artifact: str):
    """Stop both documents naming one artifact, leaving the file itself alone.

    The longer names contain the shorter ones as substrings, so the replacement
    has to run longest-first and put a placeholder in that contains none of
    them.
    """
    placeholder = "SOME_UNNAMED_FILE"

    def transform(text: str) -> str:
        for other in sorted(NAMED_ARTIFACTS, key=len, reverse=True):
            if other != artifact and artifact in other:
                text = text.replace(other, f"KEPT_{len(other)}")
        text = text.replace(artifact, placeholder)
        for other in sorted(NAMED_ARTIFACTS, key=len, reverse=True):
            if other != artifact and artifact in other:
                text = text.replace(f"KEPT_{len(other)}", other)
        return text

    return transform


for _artifact in NAMED_ARTIFACTS:
    MUTATIONS[f"unname_{_artifact.split('.')[0]}"] = lambda root, a=_artifact: (
        mutate_text(f"{DOCDIR}/README.md", _unname_artifact(a))(root),
        mutate_text(f"{DOCDIR}/work_log.md", _unname_artifact(a))(root),
    )


def _move_matches(root: Path) -> None:
    """Keep the number of matching tokens but move *which* positions match."""
    path = root / "readiness_autoregressive/autoregressive_meta.json"
    data = json.loads(path.read_text())
    hf, tt = data["hf"]["token_ids"], data["tt"]["token_ids"]
    matching = [i for i, (a, b) in enumerate(zip(hf, tt)) if a == b]
    free = [i for i in range(len(hf)) if i not in matching]
    moved = tt[:]
    for i in matching:
        moved[i] = hf[i] + 1
    for i in free[: len(matching)]:
        moved[i] = hf[i]
    data["tt"]["token_ids"] = moved
    _rewrite(path, json.dumps(data, indent=2))


def _make_four_matches(root: Path) -> None:
    """Rewrite the TT completion so exactly four tokens match HF again."""
    path = root / "readiness_autoregressive/autoregressive_meta.json"
    data = json.loads(path.read_text())
    hf = data["hf"]["token_ids"]
    data["tt"]["token_ids"] = [hf[i] if i in (0, 1, 2, 3) else hf[i] + 1 for i in range(len(hf))]
    _rewrite(path, json.dumps(data, indent=2))


def _rewrite_gz(path: Path, transform) -> None:
    import gzip

    body = transform(gzip.open(path, "rt", errors="ignore").read())
    path.unlink()
    with gzip.open(path, "wt") as handle:
        handle.write(body)


def _drop_json_key(path: Path, dotted: str, key: str) -> None:
    data = json.loads(path.read_text())
    node = data
    for part in dotted.split("."):
        node = node[part]
    node.pop(key)
    _rewrite(path, json.dumps(data, indent=2))


def _csv_rows(path: Path):
    import csv
    import gzip

    with gzip.open(path, "rt") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _write_csv_rows(path: Path, fieldnames, rows) -> None:
    import csv
    import gzip
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.unlink()
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(buffer.getvalue())


def _drop_csv_rows(path: Path, op_code: str, count: int) -> None:
    """Delete rows from a windowed profile, so a per-device tally stops holding."""
    fieldnames, rows = _csv_rows(path)
    dropped = 0
    kept = []
    for row in rows:
        if dropped < count and row["OP CODE"] == op_code:
            dropped += 1
            continue
        kept.append(row)
    _write_csv_rows(path, fieldnames, kept)


def _scale_csv_rows(path: Path, op_code: str, factor: float) -> None:
    """Scale one op's kernel durations, leaving the row count alone."""
    fieldnames, rows = _csv_rows(path)
    for row in rows:
        if row["OP CODE"] == op_code:
            row["DEVICE KERNEL DURATION [ns]"] = str(int(int(row["DEVICE KERNEL DURATION [ns]"]) * factor))
    _write_csv_rows(path, fieldnames, rows)


def _append_gz(path: Path, text: str) -> None:
    import gzip

    body = gzip.open(path, "rt", errors="ignore").read() + text
    path.unlink()
    with gzip.open(path, "wt") as handle:
        handle.write(body)


def _corrupt_gz_session(path: Path) -> None:
    """Strip the pytest session header, so the log stops being the whole run."""
    import gzip

    text = gzip.open(path, "rt", errors="ignore").read().replace("test session starts", "something else")
    path.unlink()
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def _corrupt_gz(path: Path) -> None:
    """Rewrite the gzipped watcher log with a tripped assert and a different tally."""
    import gzip

    text = gzip.open(path, "rt", errors="ignore").read()
    text = text.replace("146 passed", "144 passed")
    text += "\nDevice 0 worker core(x= 0,y= 0): BRISC tripped an assert on line 119.\n"
    path.unlink()
    with gzip.open(path, "wt") as handle:
        handle.write(text)


LINE = re.compile(r"^(PASS|FAIL)  \[([^\]]+)\] (.*)$")


def run_checker(root: Path) -> dict[str, tuple[bool, str]]:
    """Run the checker inside ``root``; return ``{stable id: (passed, name)}``.

    **Keyed by id, not by name.** The stage-06 review found the bug this fixes:
    many check names embed the artifact value they check, e.g. ``README quotes
    the degeneracy metric 109``. Mutating that artifact to 77 produced ``FAIL
    README quotes the degeneracy metric 77`` -- a name that appears nowhere in
    the clean run -- and the old ``ever_failed |= failed & set(clean)`` threw it
    away. Every such assertion was credited with zero failures on the artifact
    side while the tester still reported 473 of 473.
    """
    result = subprocess.run(
        [sys.executable, str(root / RELATIVE / "check_published_figures.py")],
        capture_output=True,
        text=True,
    )
    outcome: dict[str, tuple[bool, str]] = {}
    for line in result.stdout.splitlines():
        match = LINE.match(line)
        if match:
            outcome[match.group(2)] = (match.group(1) == "PASS", match.group(3).split("  -- ")[0].strip())
    return outcome


def _per_line(outcome: dict[str, tuple[bool, str]]) -> dict[str, int]:
    """How many checks each source line emitted -- the guard on ordinal drift.

    An id is ``L<line>.<ordinal>``. The line is stable under any mutation; the
    ordinal is stable only while the line keeps emitting the same number of
    checks. This counts them so a run whose loop shed an iteration can be caught
    rather than credited to whichever assertion slid into the vacated ordinal.
    """
    counts: dict[str, int] = {}
    for identity in outcome:
        line = identity.split(".")[0]
        counts[line] = counts.get(line, 0) + 1
    return counts


def copy_tree(destination: Path) -> Path:
    root = destination / MODEL_DIR.name
    shutil.copytree(
        MODEL_DIR,
        root,
        copy_function=lambda a, b, follow_symlinks=True: __import__("os").link(a, b),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".benchmarks", "generated"),
    )
    return root


#: The clean tree must be green before a single mutation is applied -- a
#: mutation is only evidence if the assertion it breaks was passing first. Two
#: of this checker's assertions are **self-referential**, though, and they make
#: that rule unsatisfiable from a standing start:
#:
#: * "the archived mutation run covered exactly the assertions this run makes"
#:   compares the assertion count recorded in ``logs/mutation_test_checker.log``
#:   against the count the live run reaches,
#: * "README's figure count matches what this run reports" compares the same
#:   count against the README, and
#: * "the archived mutation run is a normal run over a green clean tree"
#:   requires the archive to open with ``clean tree: <this run's count> checks,
#:   0 failing`` and to carry no bootstrap banner. That one is new: without it,
#:   pasting a banner and a ``2 failing`` line onto the archive passed.
#:
#: Adding an assertion therefore breaks all three until a *new* archived log
#: exists, and the only thing that can produce that log is a run of this tester
#: -- which refuses to start. ``--bootstrap`` is the way out and nothing more:
#: it permits exactly these three checks to fail on the clean tree, says loudly
#: which ones it permitted, and still refuses if anything else is failing.
#:
#: A ``--bootstrap`` log is a stepping stone, **not** the archive. While these
#: are failing on the clean tree they also "fail" under every mutation, so every
#: measured breadth in a bootstrap run is inflated by up to that many. The
#: settling procedure is: bootstrap, archive, re-derive the documents' figures
#: from that log, then run **normally** and archive that log, repeating until a
#: normal run reproduces the archived log byte for byte.
#:
#: The provenance check adds one step to that procedure, because a bootstrap log
#: can never satisfy it: a bootstrap log reports its clean tree as *not* green,
#: so archiving one leaves the clean tree failing, so only another bootstrap run
#: can proceed, forever. Break the cycle with a **scaffold**: take the bootstrap
#: log, delete its banner and set its clean-tree line to ``0 failing``, put that
#: in place of the archive and derive the documents' figures from it. The clean
#: tree is then green, a **normal** run is possible, and its log -- a real one --
#: replaces the scaffold. Iterate from there as above. The scaffold is thrown
#: away; what ships is a log a normal run reproduces byte for byte, which is
#: exactly what the checker's three assertions above then confirm.
BOOTSTRAP_SELF_REFERENTIAL = (
    "the archived mutation run covered exactly the assertions this run makes",
    "README's figure count matches what this run reports",
    "the archived mutation run is a normal run over a green clean tree",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="permit ONLY the self-referential assertion-count checks to fail on the clean tree, "
        "so a first log can be produced after the assertion count changes. Every other failure "
        "still aborts. The resulting log has inflated breadths and must be settled by a normal run.",
    )
    args = parser.parse_args()
    if args.list:
        for name in MUTATIONS:
            print(name)
        return 0

    with tempfile.TemporaryDirectory(prefix="mutation_") as tmp:
        base = copy_tree(Path(tmp) / "clean")
        clean = run_checker(base)
        if not clean:
            print("the checker produced no output on the clean tree", file=sys.stderr)
            return 2
        names = {identity: name for identity, (_, name) in clean.items()}
        broken_on_clean = [names[i] for i, (ok, _) in clean.items() if not ok]
        print(f"clean tree: {len(clean)} checks, {len(broken_on_clean)} failing")
        if broken_on_clean:
            permitted = [
                name for name in broken_on_clean if args.bootstrap and name.startswith(BOOTSTRAP_SELF_REFERENTIAL)
            ]
            blocking = [name for name in broken_on_clean if name not in permitted]
            for name in broken_on_clean:
                label = "PERMITTED BY --bootstrap" if name in permitted else "FAILING ALREADY"
                print(f"  {label}: {name}")
            if blocking:
                if args.bootstrap:
                    print(
                        f"\n--bootstrap permits only the {len(BOOTSTRAP_SELF_REFERENTIAL)} self-referential "
                        f"count checks to fail; {len(blocking)} other check(s) are failing on the clean tree."
                    )
                return 2
            print()
            print("*" * 78)
            print(f"*  --bootstrap: running against a clean tree that is NOT green. {len(permitted)} check(s)")
            print("*  are failing and were PERMITTED because they are the self-referential")
            print("*  assertion-count checks, which cannot pass until this run's log exists:")
            for name in permitted:
                print(f"*    - {name}")
            print("*  Every measured breadth below is inflated by up to that many assertions,")
            print("*  because a check already failing on the clean tree also 'fails' under")
            print("*  every mutation. THIS LOG IS NOT THE ARCHIVE. Archive it, re-derive the")
            print("*  documents' figures from it, then run again WITHOUT --bootstrap and")
            print("*  archive that log instead.")
            print("*" * 78)
            print()

        per_line = _per_line(clean)
        ever_failed: set[str] = set()
        targeted_failed: set[str] = set()
        no_ops: list[str] = []
        did_not_apply: list[str] = []
        unstable_ordinals: list[tuple[str, int]] = []
        #: measured breadth: how many assertions each mutation actually tripped,
        #: and for each assertion the breadth of the narrowest mutation that
        #: tripped it. Both used to be described by hand and neither was true.
        breadth: dict[str, int] = {}
        narrowest: dict[str, int] = {}
        for index, (name, mutation) in enumerate(MUTATIONS.items(), 1):
            root = copy_tree(Path(tmp) / f"m{index}")
            try:
                mutation(root)
            except Exception as error:  # a mutation that cannot apply is a bug here
                print(f"  {name:<38} -> MUTATION FAILED TO APPLY -- {error}")
                did_not_apply.append(f"{name}: {error}")
                shutil.rmtree(root)
                continue
            outcome = run_checker(root)
            failed = {identity for identity, (ok, _) in outcome.items() if not ok} & set(clean)
            # An id is ``L<line>.<ordinal>``, and the ordinal is stable only if
            # the number of ``check()`` calls that line makes is stable. A
            # mutation that removes a NON-LAST iteration of a loop shifts every
            # later ordinal, so the id credited belongs to a different assertion
            # than the one that failed: dropping device "1" from
            # ``device_kernel_us`` credits ``L0352.1``, whose clean name is
            # "device 1's kernel total" while the mutated run's is "device 2's".
            # No shipped mutation triggers it, and it was unguarded. Detect it
            # by cardinality: if a line emits a different number of checks than
            # it did on the clean tree, its ordinals no longer line up, so none
            # of that line's credit is trustworthy and all of it is dropped.
            #
            # Cardinality and not the name. Comparing names would re-open the
            # round-1 bug this whole id scheme exists to fix -- names embed the
            # artifact value under test, so a mutation of the artifact changes
            # the name of exactly the assertion it is supposed to be credited
            # with, and rejecting on a name mismatch would throw that credit
            # away again. The number of checks a line emits does not depend on
            # any artifact value; it depends only on how many times the line
            # ran, which is precisely what an ordinal shift changes.
            mutated_per_line = _per_line(outcome)
            shifted = {
                identity
                for identity in failed
                if per_line[identity.split(".")[0]] != mutated_per_line.get(identity.split(".")[0])
            }
            if shifted:
                unstable_ordinals.append((name, len(shifted)))
                failed -= shifted
            # A mutation that makes the checker crash before finishing also
            # counts for the checks it did reach, but not for the ones it did
            # not -- so record only what it actually reported.
            ever_failed |= failed
            if name not in SHOTGUN:
                targeted_failed |= failed
            if not failed:
                no_ops.append(name)
            breadth[name] = len(failed)
            for identity in failed:
                narrowest[identity] = min(narrowest.get(identity, len(clean) + 1), len(failed))
            print(
                f"  {name:<38} -> {len(failed):>4} checks failed"
                f"{'  SHOTGUN' if name in SHOTGUN else ''}"
                f"{'  *** BROKE NOTHING ***' if not failed else ''}"
                f"{'  (checker aborted early)' if len(outcome) < len(clean) else ''}"
            )
            shutil.rmtree(root)

    never = sorted(set(clean) - ever_failed)
    shotgun_only = sorted(ever_failed - targeted_failed)
    print()
    print(f"{len(clean)} assertions; {len(ever_failed)} were made to fail by at least one mutation")
    print(
        f"{len(targeted_failed)} of those were made to fail by a mutation that is not one of the "
        f"{len(SHOTGUN)} shotguns ({', '.join(sorted(SHOTGUN))})"
    )

    # -- measured breadth ----------------------------------------------------
    #
    # ``SHOTGUN`` is a declared property -- these four mutations corrupt every
    # digit or every word of a whole document -- and it is the gate. It is NOT
    # a statement about breadth, and the round-2 review was right that the code
    # and the README were describing it as one ("trip 200+ assertions at once",
    # which is true of two of the four and false of the other two). Breadth is
    # measured here instead of asserted, and the two axes are printed side by
    # side so the gap between them is visible rather than papered over.
    shotgun_breadth = sorted(breadth[name] for name in SHOTGUN if name in breadth)
    widest_targeted = sorted(((b, name) for name, b in breadth.items() if name not in SHOTGUN), reverse=True)[:5]
    print(
        f"\nmeasured breadth of the {len(SHOTGUN)} declared shotguns: "
        f"{shotgun_breadth[0]}-{shotgun_breadth[-1]} assertions "
        f"({', '.join(f'{n} {breadth[n]}' for n in sorted(SHOTGUN) if n in breadth)})"
    )
    print("widest mutations NOT declared shotgun: " + ", ".join(f"{name} {b}" for b, name in widest_targeted))
    print("  -- so breadth and 'corrupts a whole document' are different axes, and the gate is the")
    print("     second one. The narrowest coverage each assertion has, by measured breadth:")
    buckets = ((1, 1), (2, 5), (6, 20), (21, 50), (51, len(clean)))
    for low, high in buckets:
        n_in = sum(1 for b in narrowest.values() if low <= b <= high)
        label = f"{low}" if low == high else f"{low}-{high}"
        print(f"       narrowest mutation trips {label:>9} assertions: {n_in:>3} of {len(clean)}")
    coarse = sorted(i for i, b in narrowest.items() if b > 20)
    print(
        f"     {len(coarse)} of {len(clean)} assertions have NO mutation narrower than 21 assertions " "covering them."
    )
    print("     That is a real weakness in this evidence and it is a named limitation in the README,")
    print("     not a failure here: the gate is the declared-shotgun one above.")
    status = 0
    if unstable_ordinals:
        # Credit that was dropped because the mutation changed how many checks a
        # line emits, so its ordinals no longer identify the same assertions.
        print(f"\n{len(unstable_ordinals)} mutations shifted a line's check ordinals; that line's credit was")
        print("dropped rather than misattributed:")
        for name, count in unstable_ordinals:
            print(f"  - {name}: {count} ids not credited")
    if never:
        print(f"\n{len(never)} assertions NEVER failed under any mutation -- each is a check that may not be able")
        print("to fail at all. Either strengthen it or add a mutation that should break it:")
        for identity in never:
            print(f"  - [{identity}] {names[identity]}")
        status = 1
    else:
        print("every assertion was made to fail by at least one mutation")
    if shotgun_only:
        # "Covered" by a mutation that corrupts every digit in the document is
        # very weak evidence: some other assertion fails first for a reason that
        # has nothing to do with this one. These are reported, not tolerated
        # silently.
        print(f"\n{len(shotgun_only)} assertions failed ONLY under a shotgun mutation, which corrupts a whole")
        print(
            f"document at once ({shotgun_breadth[0]}-{shotgun_breadth[-1]} assertions tripped here) and so says "
            "little about any one of them:"
        )
        for identity in shotgun_only:
            print(f"  - [{identity}] {names[identity]}")
        status = 1
    if no_ops:
        # A mutation that breaks nothing proves nothing and inflates the count
        # the README quotes. The old tester printed "0 checks failed" and moved
        # on; now it is a failure of the tester.
        print(f"\n{len(no_ops)} mutations BROKE NOTHING. A mutation that changes no assertion's outcome is")
        print("not evidence of anything and must be repaired or dropped:")
        for name in no_ops:
            print(f"  - {name}")
        status = 1
    if did_not_apply:
        print(f"\n{len(did_not_apply)} mutations could not be applied at all:")
        for line in did_not_apply:
            print(f"  - {line}")
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
