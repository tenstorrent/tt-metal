"""Prompt-correct, on-device qualitative controls through the live API."""

import argparse
import hashlib
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--repeat-greedy", action="store_true", help="Check identical output after warming prefill/decode"
    )
    args = parser.parse_args()
    model = Path(__file__).resolve().parents[1]
    control_path = model / "doc/full_model/hf_qualitative_extended.json"
    controls = json.loads(control_path.read_text())
    baseline_path = model / "doc/datatype_sweep/qualifications/down4_l24to34_head8_lofi_qualification_quality.json"
    baselines = {row["id"]: row for row in json.loads(baseline_path.read_text())}
    revision = "036114ce8d46c32b24c15423211069abb9c5d25e"
    tokenizer = AutoTokenizer.from_pretrained(
        "IFM/K2-Horizon-7B", revision=revision, trust_remote_code=True, local_files_only=True
    )
    result = dict(
        hf_model="IFM/K2-Horizon-7B",
        revision=revision,
        prompt_mode="chat template rendered to token IDs",
        hf_control=str(control_path),
        hf_control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        full_model_control=str(baseline_path),
        outputs=[],
    )
    for control in controls:
        ids = tokenizer.apply_chat_template(
            control["messages"], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        assert ids == control["prompt_token_ids"]
        modes = [("greedy", 0.0)]
        if args.repeat_greedy:
            modes.append(("greedy_repeat", 0.0))
        modes.append(("sampled", 0.7))
        first_greedy = None
        for mode, temperature in modes:
            payload = dict(
                model="IFM/K2-Horizon-7B",
                prompt=ids,
                max_tokens=512,
                temperature=temperature,
                top_k=32,
                top_p=0.9,
                seed=17,
                return_token_ids=True,
            )
            response = requests.post(args.url + "/v1/completions", json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            assert "error" not in data, data
            output_ids = data["choices"][0]["token_ids"]
            if mode == "greedy":
                first_greedy = output_ids
            elif mode == "greedy_repeat":
                assert output_ids == first_greedy, (control["id"], "warmed greedy differs")
            baseline_ids = baselines[control["id"]]["tt_token_ids"]
            common = next(
                (i for i, (a, b) in enumerate(zip(output_ids, baseline_ids)) if a != b),
                min(len(output_ids), len(baseline_ids)),
            )
            result["outputs"].append(
                dict(
                    id=control["id"],
                    mode=mode,
                    messages=control["messages"],
                    rendered_prompt=control["rendered_prompt"],
                    request=payload,
                    response=data,
                    full_model_greedy_common_prefix=common if temperature == 0.0 else None,
                )
            )
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print(control["id"], mode, len(output_ids), data["choices"][0]["finish_reason"], flush=True)


if __name__ == "__main__":
    main()
