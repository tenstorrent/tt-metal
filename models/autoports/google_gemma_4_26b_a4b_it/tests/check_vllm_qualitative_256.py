# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Extend selected-policy standalone controls to the saved 256-token serving text."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import MODEL_ID, REVISION
from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_policy import SELECTED_CONFIG, resolve_precision_config
from models.common.sampling.generator import SamplingParams

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {
    "shared_2": {"greedy": ["tiny-brass-heart"], "sampled": ["pair-o-glasses", "brass-bound-and-etched"]},
    "shared_3": {"greedy": ["In any own-contained system"], "sampled": []},
}


def common_prefix(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def flagged_spans(text, phrases):
    return {
        phrase: {
            "present": (start := text.find(phrase)) >= 0,
            "character_offset": start if start >= 0 else None,
            "context": text[max(0, start - 60) : start + len(phrase) + 60] if start >= 0 else None,
        }
        for phrase in phrases
    }


def text_comparison(tokenizer, generated, serving_text):
    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    visible_ids = tokenizer.encode(text, add_special_tokens=False)
    serving_ids = tokenizer.encode(serving_text, add_special_tokens=False)
    prefix = common_prefix(text, serving_text)
    return {
        "text_exact": text == serving_text,
        "text_common_prefix_characters": prefix,
        "first_text_difference": (
            None
            if text == serving_text
            else {
                "standalone": text[max(0, prefix - 60) : prefix + 100],
                "serving": serving_text[max(0, prefix - 60) : prefix + 100],
            }
        ),
        "serving_generated_token_ids_available": False,
        "token_comparison_basis": "Both visible texts retokenized; serving artifact did not retain generated token IDs",
        "retokenized_text_ids_exact": visible_ids == serving_ids,
        "retokenized_text_common_prefix_tokens": common_prefix(visible_ids, serving_ids),
        "serving_text_retokenized_ids": serving_ids,
        "standalone_text_retokenized_ids": visible_ids,
    }


def prepare(tokenizer, metadata, serving, prior, prompt_ids):
    assert metadata["hf_model"] == prior["hf_model"] == MODEL_ID
    assert metadata["revision"] == prior["revision"] == REVISION
    assert metadata["chat_template_present"] and tokenizer.chat_template
    metadata_rows = {row["id"]: row for row in metadata["prompts"]}
    prior_rows = {row["id"]: row for row in prior["prompts"]}
    serving_rows = {row["prompt"]: row for row in serving}
    rows = []
    for prompt_id in prompt_ids:
        source, control = metadata_rows[prompt_id], prior_rows[prompt_id]
        messages = source["messages"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        assert rendered == source["rendered_prompt"] == control["rendered_prompt"], prompt_id
        assert ids == source["prompt_token_ids"] == control["prompt_tokens"], prompt_id
        assert len(messages) == 1 and messages[0]["role"] == "user"
        current = serving_rows[messages[0]["content"]]
        assert current["prompt_mode"] == "chat"
        rows.append(
            {
                "id": prompt_id,
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_token_ids": ids,
                "prior_selected_tokens": control["tt_tokens"],
                "prior_selected_text": control["tt_completion"],
                "serving_greedy_text": current["greedy_completion"],
                "serving_sampled_text": current["sampled_completion"],
                "flagged_serving_phrases": {
                    mode: flagged_spans(current[f"{mode}_completion"], FLAGS.get(prompt_id, {}).get(mode, []))
                    for mode in ("greedy", "sampled")
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate cached pinned prompts without device access"
    )
    parser.add_argument("--prompt-ids", nargs="+", default=["shared_2", "shared_3"])
    parser.add_argument(
        "--sampled-seed", type=int, help="Optional new sampled control; original request seed was not saved"
    )
    parser.add_argument("--sampled-top-k", type=int, help="Explicit top-k for the optional new sampled control")
    args = parser.parse_args()
    if (args.sampled_seed is None) != (args.sampled_top_k is None):
        parser.error("The optional sampled control requires both --sampled-seed and --sampled-top-k")
    if args.sampled_top_k is not None and not 1 <= args.sampled_top_k <= 32:
        parser.error("This generator's common sampler supports explicit top-k between 1 and 32")

    metadata_path = ROOT / "readiness_vllm/qualitative_prompt_format.json"
    serving_path = ROOT / "readiness_vllm/vllm_qualitative_outputs.json"
    prior_path = ROOT / "doc/datatype_sweep/selected/qualitative_tt.json"
    metadata, serving, prior = (json.loads(path.read_text()) for path in (metadata_path, serving_path, prior_path))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    rows = prepare(tokenizer, metadata, serving, prior, args.prompt_ids)
    policy = resolve_precision_config()
    params = {"greedy": SamplingParams(temperature=0.0, top_k=1, top_p=1.0)}
    if args.sampled_seed is not None:
        params["sampled"] = SamplingParams(temperature=0.7, top_k=args.sampled_top_k, top_p=0.9, seed=args.sampled_seed)
    source_paths = [
        metadata_path,
        serving_path,
        prior_path,
        SELECTED_CONFIG,
        ROOT / "tt/generator.py",
        ROOT / "tt/model.py",
        ROOT / "tt/multichip_decoder.py",
        ROOT.parents[1] / "common/sampling/generator.py",
        ROOT.parents[1] / "common/sampling/tt_sampling.py",
    ]
    report = {
        "status": "prepared",
        "hf_model": MODEL_ID,
        "revision": REVISION,
        "tokenizer": type(tokenizer).__name__,
        "chat_template_present": True,
        "prompt_mode": "chat",
        "rendering": "pinned tokenizer.apply_chat_template(add_generation_prompt=True), exact saved-ID validation",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "precision_config": policy,
        "max_new_tokens": 256,
        "full_model": True,
        "stop_on_eos": True,
        "control_sampling_parameters": {mode: asdict(value) for mode, value in params.items()},
        "sampled_control_is_new_draw": args.sampled_seed is not None,
        "automatic_quality_verdict": None,
        "original_sampled_reproduction": {
            "available": False,
            "reason": "Shared harness saved text only and sent no request seed/top_k; effective random seed is unavailable",
            "known_request_parameters": metadata["sampled"],
            "engine_seed_is_not_request_seed": True,
        },
        "prompts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if args.prepare_only:
        print(
            f"Validated {len(rows)} pinned prompts and preserved original flagged text; no device opened.", flush=True
        )
        return
    generator = mesh = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
        generator = Gemma4Generator(mesh, max_seq_len=512, precision_config=policy)
        assert not generator.model.reduced_probe
        report["runtime_precision"] = generator.model.precision_summary()
        for row in rows:
            row["controls"] = {}
            for mode, sampling_params in params.items():
                tokens = generator.generate(
                    row["prompt_token_ids"], 256, sampling_params=sampling_params, stop_on_eos=True, buffer_tokens=False
                )
                visible = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                result = {
                    "generated_tokens": tokens,
                    "num_generated_tokens": len(tokens),
                    "text": visible,
                    "text_with_special_tokens": tokenizer.decode(tokens, skip_special_tokens=False),
                    "comparison": text_comparison(tokenizer, tokens, row[f"serving_{mode}_text"]),
                    "flagged_standalone_phrases": flagged_spans(visible, FLAGS.get(row["id"], {}).get(mode, [])),
                    "sampled_seed_comparable_to_original": False if mode == "sampled" else None,
                }
                if mode == "greedy":
                    prior_tokens = row["prior_selected_tokens"]
                    result["prior_selected_token_prefix_exact"] = tokens[: len(prior_tokens)] == prior_tokens
                    result["prior_selected_common_prefix_tokens"] = common_prefix(tokens, prior_tokens)
                row["controls"][mode] = result
                save()
                print(f"CONTROL {row['id']} {mode}\n{visible}", flush=True)
        report["status"] = "controls_complete_requires_review"
        report["all_greedy_text_exact"] = all(row["controls"]["greedy"]["comparison"]["text_exact"] for row in rows)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
