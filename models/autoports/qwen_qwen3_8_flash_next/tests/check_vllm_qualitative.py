# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the six shared quality prompts with settings matching the HF/TT controls.

Thinking is disabled per request; served defaults are unchanged. Run this CPU
client against an already-running full model. Successful collection is not a
quality verdict: inspect the outputs against both controls afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import requests
from run_vllm_quality_controls import MODEL_DIR, MODEL_ID, REVISION, assert_no_ttnn, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, default=MODEL_DIR / "readiness_vllm/quality_hf_v1")
    parser.add_argument("--publish-output", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        parser.error("Use a fresh output directory to preserve previous evidence")

    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    assert_no_ttnn()
    checkpoint = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned checkpoint has no chat template")
    source = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = [p.strip() for p in source.read_text().split("\n\n") if p.strip()]
    control_prompts = json.loads((args.controls_dir / "prompts.json").read_text())
    if len(prompts) != 6 or len(control_prompts) != 6:
        raise RuntimeError("Expected exactly six shared prompts and six control prompts")
    profiles = {
        "greedy": dict(temperature=0.0, top_p=1.0, top_k=1, seed=42),
        "sampled": dict(temperature=0.7, top_p=0.9, top_k=20, seed=42),
    }
    control_manifest = json.loads((args.controls_dir / "manifest.json").read_text())
    expected_control = dict(
        model_id=MODEL_ID,
        revision=REVISION,
        chat_template_kwargs=dict(enable_thinking=False),
        max_new_tokens=256,
        profiles=profiles,
    )
    mismatched = [key for key, value in expected_control.items() if control_manifest.get(key) != value]
    if mismatched:
        raise RuntimeError(f"Control settings do not match this served suite: {mismatched}")
    session = requests.Session()
    response = session.get(args.server_url.rstrip("/") + "/v1/models", timeout=30)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    manifest = dict(
        complete=False,
        model_id=MODEL_ID,
        revision=REVISION,
        checkpoint=str(checkpoint),
        served_model=model,
        server_url=args.server_url,
        endpoint="/v1/chat/completions",
        tokenizer_class=type(tokenizer).__name__,
        transformers_version=transformers.__version__,
        chat_template_present=True,
        chat_template=tokenizer.chat_template,
        chat_template_kwargs=dict(enable_thinking=False),
        prompt_mode="chat",
        rendering="apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
        prompt_source=str(source),
        controls_dir=str(args.controls_dir),
        controls_complete_at_collection_start=control_manifest.get("complete"),
        profiles=profiles,
        max_new_tokens=256,
        outputs_completed=0,
        expected_outputs=12,
        quality_verdict="not_assessed; compare with matched HF and same-code TT controls",
        command=sys.argv,
    )
    write_json(manifest_path, manifest)
    rendered_rows, raw_rows, shared_rows = [], [], []
    started = time.monotonic()
    try:
        for index, (prompt, control) in enumerate(zip(prompts, control_prompts), 1):
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
            )
            encoded = tokenizer.encode(rendered, add_special_tokens=False)
            tokenize_request = dict(
                model=model,
                messages=messages,
                add_generation_prompt=True,
                chat_template_kwargs=dict(enable_thinking=False),
            )
            token_response = session.post(
                args.server_url.rstrip("/") + "/tokenize", json=tokenize_request, timeout=args.timeout
            )
            token_response.raise_for_status()
            server_tokens = token_response.json()
            row = dict(
                prompt_id=index,
                prompt=prompt,
                messages=messages,
                rendered_prompt=rendered,
                prompt_token_ids=ids,
                encoded_token_ids=encoded,
                server_tokenize_request=tokenize_request,
                server_tokenize_response=server_tokens,
                controls_prompt_id=control["prompt_id"],
                parity=(
                    ids == encoded == control["prompt_token_ids"] == server_tokens.get("tokens")
                    and prompt == control["prompt"]
                    and rendered == control["rendered_prompt"]
                ),
            )
            rendered_rows.append(row)
            write_json(args.output_dir / "prompts.json", rendered_rows)
            if not row["parity"]:
                raise RuntimeError(
                    f"Prompt{index}: canonical/HF-control/server tokenization mismatch; see prompts.json"
                )
            shared_row = dict(prompt=prompt, prompt_mode="chat")
            for profile, settings in profiles.items():
                body = dict(
                    model=model,
                    messages=messages,
                    max_tokens=256,
                    chat_template_kwargs=dict(enable_thinking=False),
                    return_token_ids=True,
                    **settings,
                )
                request_start = time.monotonic()
                response = session.post(
                    args.server_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=args.timeout
                )
                try:
                    data = response.json()
                except ValueError:
                    data = {"raw_body": response.text}
                raw = dict(
                    prompt_id=index,
                    profile=profile,
                    request=body,
                    status_code=response.status_code,
                    response=data,
                    seconds=time.monotonic() - request_start,
                )
                raw_rows.append(raw)
                write_json(args.output_dir / f"prompt_{index:02d}_{profile}.json", raw)
                write_json(args.output_dir / "outputs.json", raw_rows)
                response.raise_for_status()
                if data.get("prompt_token_ids") != ids:
                    raise RuntimeError(f"Prompt{index}/{profile}: generation input IDs differ from canonical template")
                choice = data["choices"][0]
                content = choice["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(f"Prompt{index}/{profile}: missing or empty required text content")
                output_ids = choice.get("token_ids")
                if not isinstance(output_ids, list) or not output_ids:
                    raise RuntimeError(f"Prompt{index}/{profile}: missing output token IDs")
                if any(not 0 <= token < len(tokenizer) for token in output_ids):
                    raise RuntimeError(f"Prompt{index}/{profile}: out-of-vocabulary output token")
                if data["usage"]["prompt_tokens"] != len(ids):
                    raise RuntimeError(f"Prompt{index}/{profile}: logical prompt length mismatch")
                shared_row[profile + "_completion"] = content
                manifest["outputs_completed"] += 1
                write_json(manifest_path, manifest)
                assert_no_ttnn()
                print(f"Prompt{index} {profile}: {len(output_ids)} tokens, {choice.get('finish_reason')}", flush=True)
            shared_rows.append(shared_row)
            write_json(args.output_dir / "vllm_qualitative_outputs.json", shared_rows)
        manifest["complete"] = True
        manifest["all_prompt_token_parity"] = all(row["parity"] for row in rendered_rows)
        if args.publish_output:
            args.publish_output.parent.mkdir(parents=True, exist_ok=True)
            if args.publish_output.exists():
                backup = args.publish_output.with_name(args.publish_output.stem + ".before_targeted.json")
                if backup.exists():
                    raise RuntimeError(f"Refusing to overwrite preserved original artifact: {backup}")
                shutil.copy2(args.publish_output, backup)
                manifest["preserved_prior_shared_artifact"] = str(backup)
            shutil.copy2(args.output_dir / "vllm_qualitative_outputs.json", args.publish_output)
            manifest["published_output"] = str(args.publish_output)
            write_json(args.publish_output.with_name("qualitative_prompt_format.json"), manifest)
    except Exception:
        manifest["complete"] = False
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["seconds"] = time.monotonic() - started
        manifest["ttnn_imported"] = "ttnn" in sys.modules
        write_json(manifest_path, manifest)
        session.close()
    print("SERVED_QUALITY_COLLECTION_COMPLETE; qualitative verdict still requires review", flush=True)


if __name__ == "__main__":
    main()
