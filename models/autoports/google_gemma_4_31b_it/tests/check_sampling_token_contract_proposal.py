# SPDX-License-Identifier: Apache-2.0
"""CPU-only PC-03 proposal checks using retained serving and HF evidence."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    root = Path(__file__).parents[1]
    lines = (root / "doc/vllm_integration/PC-03-sampling-token-contracts.patch").read_text().splitlines()
    namespace = {}
    for name, signature in (
        ("_first_generated_token_ids", "responses"),
        ("_assert_frequency_batch_controls", "mixed, homogeneous, solo"),
    ):
        start = lines.index(f"+def {name}({signature}):")
        block = []
        for line in lines[start:]:
            if not line.startswith("+"):
                break
            block.append(line[1:])
        exec(compile(ast.parse("\n".join(block)), "PC-03 proposed helper", "exec"), namespace)
    first = namespace["_first_generated_token_ids"]
    check = namespace["_assert_frequency_batch_controls"]

    def response(row):
        return SimpleNamespace(choices=[SimpleNamespace(token_ids=row["token_ids"], text=row["text"])])

    fixture = [response(dict(token_ids=[10], text=" A")), response(dict(token_ids=[11], text=" B"))]
    assert len({r.choices[0].text[:1] for r in fixture}) == 1 and len(set(first(fixture))) == 2
    assert len(set(first([fixture[0], fixture[0]]))) == 1
    probe = json.loads((root / "readiness_vllm/sampling_failure_probe.json").read_text())
    waves = {wave["name"]: wave["rows"] for wave in probe["waves"]}
    mixed = list(map(response, waves["frequency_native_mixed"]))
    homogeneous = {
        0.0: list(map(response, waves["frequency_host_clean"])),
        2.0: list(map(response, waves["frequency_host_penalized"])),
    }
    solo = {p: response(waves[f"frequency_solo_host_{p}"][0]) for p in (0.0, 2.0)}
    check(mixed, homogeneous, solo)
    rejected = []
    bad = mixed[:]
    bad[2] = response(dict(token_ids=[-123], text="visible"))
    row = mixed[0]
    for name, args in (
        ("slot_contamination", (bad, homogeneous, solo)),
        ("missing_penalty_effect", ([row] * 4, {0.0: [row] * 4, 2.0: [row] * 4}, {0.0: row, 2.0: row})),
    ):
        try:
            check(*args)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError(f"Missed {name}")
    hf = json.loads((root / "readiness_vllm/raw_prompt_hf_controls.json").read_text())
    cases = {case["name"]: case for case in hf["results"]}
    clean, penalized = cases["frequency_raw_0"], cases["frequency_raw_2"]
    assert clean["text"].count("a") == penalized["text"].count("a") == 1
    assert clean["output_ids"] != penalized["output_ids"]
    report = dict(
        scope="CPU-only unapplied PC-03 assertions on retained evidence",
        served_mixed_rows_checked=len(mixed),
        homogeneous_and_solo_match=True,
        same_first_character_different_tokens_detected=True,
        identical_first_tokens_still_fail_variety_threshold=True,
        invalid_controls_rejected=rejected,
        full_hf_counterexample=dict(character_a_counts=[1, 1], token_sequences_differ=True),
        plugin_patch_applied=False,
    )
    output = root / "readiness_vllm/pc03_host_regression.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
