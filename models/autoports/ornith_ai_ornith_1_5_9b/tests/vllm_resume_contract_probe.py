# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-host information-loss probe for the pinned resumed-prefill interface.

Extracts selected real definitions with AST; never imports the plugin or TTNN.
This records a known contract gap and is not a passing serving-correctness test.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PINNED_PLUGIN_SHA = "042a94863ad27d88207bcf71abdf67122c6f4c27"


class HostValues(list):
    def tolist(self):
        return list(self)


def extract(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    # Execute only named definitions from the pinned local sources, without runtime imports.
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def require(condition, description):
    if not condition:
        raise RuntimeError(description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plugin-sha", default=PINNED_PLUGIN_SHA)
    args = parser.parse_args()
    plugin = args.plugin.resolve()
    model_root = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "-C", str(plugin), "rev-parse", "HEAD"], text=True).strip()
    require(sha == args.expected_plugin_sha, f"Expected plugin {args.expected_plugin_sha}, found {sha}")
    runner_path = plugin / "src/vllm_tt_plugin/model_runner.py"
    input_path = plugin / "src/vllm_tt_plugin/model_input.py"
    state_path = model_root / "tt/sampling_state.py"
    namespace = {"Any": Any, "TTModelInput": Any, "fields": fields, "dataclass": dataclass, "SEED_NONE_SENTINEL": -1}
    params_type = extract(input_path, "TTSamplingParams", namespace)
    submit = extract(runner_path, "submit_prefill", namespace, "TTModelRunner")
    release = extract(runner_path, "_release_model_request", namespace, "TTModelRunner")
    normalize = extract(state_path, "normalize_histories", {"Integral": Integral})
    params = params_type(
        **{
            key: HostValues([value])
            for key, value in {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "presence_penalty": 0.75,
                "frequency_penalty": 0.5,
                "repetition_penalty": 1.0,
                "seed": -1,
                "num_logprobs": -2,
                "enable_log_probs": False,
            }.items()
        }
    )

    def transcript(resumed):
        calls = []

        def prefill(**kwargs):
            kwargs["sampling_params"] = asdict(kwargs["sampling_params"])
            calls.append({"call": "prefill_forward", "kwargs": kwargs, "returned_tokens": [[9]]})
            return [[9]]

        model = SimpleNamespace(
            prefill_forward=prefill,
            release_request=lambda slot: calls.append({"call": "release_request", "slot": slot}),
        )
        runner = SimpleNamespace(
            model=model,
            trace_mode="decode_only",
            kv_caches="same opaque model-owned cache",
            request_specific_rope=False,
            async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
            _req_state_slot={"old": 0},
        )

        def invoke(tokens, prompt, output):
            model_input = SimpleNamespace(
                input_tokens=[tokens],
                block_tables=[[0]],
                prompt_lens=[len(tokens)],
                input_positions=[0],
                block_tables_per_layer=None,
                multi_modal_kwargs={},
                stage_inputs=None,
                perform_device_sampling=True,
                tt_sampling_params=params,
                prefill_empty_slots=[0],
                # Deliberately populated, although the current builder only populates histories for decode.
                prompt_tokens=[prompt],
                output_tokens=[output],
            )
            submit(runner, model_input, [1])

        invoke([7, 8], [7, 8], [])
        release(runner, "old")  # Same actual call for finish and preemption.
        invoke([7, 8, 9], [7, 8] if resumed else [7, 8, 9], [9] if resumed else [])
        return calls

    fresh = transcript(False)
    resumed = transcript(True)
    same = fresh == resumed
    omitted = all(
        key not in resumed[-1]["kwargs"] for key in ("prompt_tokens", "output_tokens", "original_prompt_lens")
    )
    require(same, "Fresh and resumed calls no longer have identical transcripts; reassess the pinned contract")
    require(omitted, "The prefill boundary now conveys history; reassess the pinned contract")

    def expected(output_history):
        counts = Counter(output_history)
        scores = {
            token: score - 0.75 * (counts[token] > 0) - 0.5 * counts[token]
            for token, score in {9: 2.0, 10: 1.5}.items()
        }
        return {
            "output_history": output_history,
            "adjusted_logits": scores,
            "greedy_token": max(scores, key=scores.get),
        }

    fresh_expected, resumed_expected = expected([]), expected([9])
    require(
        fresh_expected["greedy_token"] == 9 and resumed_expected["greedy_token"] == 10,
        "The chosen penalty witness no longer distinguishes the histories",
    )
    absent_empty = normalize(None, slots=[0], vocab_size=16) == ((),)
    require(absent_empty, "Absent canonical history no longer normalizes to empty")
    evidence = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "contract_gap_reproduced",
        "plugin_sha": sha,
        "plugin_root": str(plugin),
        "scope": "AST-extracted host methods with recorder doubles; no plugin/torch/TTNN imports, hardware, or servers",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (runner_path, input_path, state_path)
        },
        "checks": {
            "identical_complete_model_visible_transcript": same,
            "prefill_omits_even_populated_internal_histories": omitted,
            "required_penalty_outputs_differ": True,
            "absent_history_normalizes_to_empty": absent_empty,
        },
        "shared_transcript": fresh,
        "expected": {"fresh": fresh_expected, "resumed": resumed_expected},
        "limits": "Recorder outputs are illustrative; no actual model logits or scheduler execution. Source inspection establishes legal finish/preemption traces.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": evidence["status"], "plugin_sha": sha, "checks": evidence["checks"], "output": str(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
