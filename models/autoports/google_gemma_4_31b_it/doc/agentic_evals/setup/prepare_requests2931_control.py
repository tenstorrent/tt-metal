#!/usr/bin/env python3
"""Offline tokenizer-only preparation; runs only --validate-only diagnostics."""
import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[6]
os.chdir(ROOT)
D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
RID = "chatcmpl-9c756f9505f95abf"
REV = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SERVER = D / "current/post-gpqa-controls-head-bfp8/server/server-b32/server.log"
RAW = D / "current/swe-extended85-head-bfp8/live-tasks/psf__requests-2931-first/responses.private.log"
BASE = D / "anomalies/swe85-precision/requests2931-call20"


def write(path, value, private=False):
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    if private:
        path.chmod(0o600)


def normalize(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        for tc in m.get("tool_calls") or []:
            f = tc["function"]
            if isinstance(f.get("arguments"), str):
                f["arguments"] = json.loads(f["arguments"])
    return messages


def main():
    for line in SERVER.open():
        if RID in line and "eval_call=" in line:
            record = json.loads(line.split("eval_call=", 1)[1])
            request = record["request"]
            request = json.loads(request) if isinstance(request, str) else request
            break
    else:
        raise ValueError("Exact request unavailable")
    call = 0
    for line in RAW.open():
        if "response_body=" not in line:
            continue
        call += 1
        if RID in line:
            response = json.loads(line.split("response_body=", 1)[1].strip()[1:-1])
            break
    assert call == 20 and response["id"] == RID
    message = response["choices"][0]["message"]
    content = message["content"]
    assert not message.get("reasoning")
    runaway = re.search(r"\^{1000,}", content)
    assert runaway and len(runaway.group()) == 130608
    tool = request["messages"][39]
    assert tool["role"] == "tool"
    observed = json.loads(tool["content"])["output"]
    legitimate = re.search(r"\^{40}", observed)
    assert legitimate and observed[legitimate.end()] == "\n"
    assert content[runaway.start() - 80 : runaway.start()] == observed[legitimate.start() - 80 : legitimate.start()]
    onset = runaway.start() + 40
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-4-31B-it",
        revision=REV,
        cache_dir="/home/mvasiljevic/.cache/huggingface/hub",
        local_files_only=True,
    )
    kw = dict(tools=request.get("tools"), tokenize=False, **request.get("chat_template_kwargs", {}))
    messages = normalize(request["messages"])
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kw)
    full_assistant = tokenizer.apply_chat_template(messages + normalize([message]), add_generation_prompt=False, **kw)
    opening = "<|channel>thought\n"
    is_open = rendered.endswith(opening)
    # Empty reasoning was not returned; completing this open channel is an explicit assumption.
    inferred = "<channel|>" if is_open else "<|channel>thought\n<channel|>"
    prompt = rendered + inferred + content[:onset]
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    full_ids = encode(prompt)
    chosen = encode(content[onset:])[:64]
    correct = encode(observed[legitimate.end() :])[:64]
    assert len(chosen) == len(correct) == 64
    request_ids = encode(rendered)
    sampling = {
        "temperature": request.get("temperature", 1.0),
        "top_p": request.get("top_p", 0.95),
        "top_k": request.get("top_k", 20),
    }
    note = (
        "Exact request rendered with pinned local tokenizer; JSON tool arguments normalized only for template. "
        "Native generated IDs absent. Request ends in open thought channel despite no returned reasoning and reasoning_tokens0. "
        "An empty thought closure is inferred before saved content; native full-assistant template omits this empty channel, so canonical full rendering does not establish native serialization equivalence. "
        "Chosen64 begins at first excess caret after40 matching native traceback underline characters. "
        "Correct64 is only next newline/File traceback text copied from native tool output, not a task solution. No inference or reproduction claim."
    )
    old = D / "anomalies/swe85-precision/call88/canonicalprefix"
    for variant, ids in [("canonicalprefix", full_ids), ("suffix512", full_ids[-512:])]:
        out = BASE / variant
        out.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            native_response_id=RID,
            call_number=call,
            hf_revision=REV,
            sampling=sampling,
            seed=42,
            native_seed=request.get("seed"),
            reconstruction_caveat=note,
            chosen_teacher_forcing_positions=64,
            correct_continuation_scope="Local quoted traceback continuation only.",
            diagnostic_prompt_tokens=len(ids),
            native_prompt_tokens=response["usage"]["prompt_tokens"],
        )
        case = out / "case.private.json"
        pp = out / "prompt.private.txt"
        rp = out / "request.private.json"
        write(
            case,
            {"prompt_token_ids": ids, "continuations": {"chosen": chosen, "correct": correct}, "metadata": metadata},
            True,
        )
        write(pp, prompt if variant == "canonicalprefix" else tokenizer.decode(ids, skip_special_tokens=False), True)
        write(rp, request, True)
        write(
            out / "cases.json",
            [
                {
                    "name": "requests2931.call20." + variant,
                    "case": str(case),
                    "branches": ["chosen", "correct"],
                    "free_run_tokens": 256,
                    "free_run_trials": 3,
                }
            ],
        )
        commands = json.loads((old / "commands.json").read_text())
        commands["device_controls"] = {
            mode: [v.replace(str(old), str(out)) for v in cmd] for mode, cmd in commands["device_controls"].items()
        }
        commands["preferred_variant"] = "canonicalprefix"
        commands[
            "diagnostic_scope"
        ] = "Chosen64 and local-traceback-correct64 TF;3x256 free seeds42/43/44. No performance claim."
        write(out / "commands.json", commands)
        validation = []
        for mode, command in commands["device_controls"].items():
            result = subprocess.run(command + ["--validate-only"], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            validation.append(dict(mode=mode, exit_code=result.returncode, stdout=result.stdout.strip()))
        manifest = dict(
            status="prepared_not_executed",
            task="psf__requests-2931",
            variant=variant,
            preferred=variant == "canonicalprefix",
            native_response_id=RID,
            call_number=call,
            revision=REV,
            raw_reply_source=str(RAW),
            raw_request_response_source=str(SERVER),
            native_prompt_tokens=response["usage"]["prompt_tokens"],
            native_completion_tokens=response["usage"]["completion_tokens"],
            reconstructed_request_prompt_tokens=len(request_ids),
            diagnostic_prompt_tokens=len(ids),
            full_reconstructed_conditioning_tokens=len(full_ids),
            teacher_forcing_positions=64,
            correct_teacher_forcing_positions=64,
            correct_branch_scope="Only quoted traceback newline/File continuation from request.messages[39].content.output; not whole-task correctness.",
            reply_char_offset=onset,
            caret_run_char_offset=runaway.start(),
            consecutive_carets=len(runaway.group()),
            legitimate_carets=40,
            request_ends_open_thought_channel=is_open,
            inferred_empty_reasoning_delimiter=inferred,
            inferred_delimiter_token_ids=encode(inferred),
            native_full_assistant_render_starts_with_request=full_assistant.startswith(rendered),
            native_full_assistant_render_starts_with_diagnostic_prompt=full_assistant.startswith(prompt),
            rendered_boundary_mismatch="Full-assistant template omits thought channel for empty reasoning; request generation prefix opens thought channel. Diagnostic explicitly closes it; actual generated protocol unavailable.",
            native_prompt_token_count_mismatch=len(request_ids) - response["usage"]["prompt_tokens"],
            sampling=sampling,
            free_running={"tokens": 256, "trials": 3, "seeds": [42, 43, 44]},
            tokenization_note=note,
            cpu_validation=validation,
            reproduction_claim="none; CPU validation only",
            builder=str(Path(__file__).relative_to(ROOT)),
            request_end_token_ids=request_ids[-12:],
        )
        for key, path in [("private_case", case), ("private_prompt", pp), ("private_request", rp)]:
            manifest[key] = str(path)
            manifest[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(out / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    k: manifest[k]
                    for k in [
                        "variant",
                        "reconstructed_request_prompt_tokens",
                        "diagnostic_prompt_tokens",
                        "reply_char_offset",
                        "native_full_assistant_render_starts_with_request",
                        "native_full_assistant_render_starts_with_diagnostic_prompt",
                    ]
                }
            )
        )


if __name__ == "__main__":
    main()
