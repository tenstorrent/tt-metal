# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native HF discovery control for existing plugin chat-test token allowances.

Never imports vLLM or TTNN. Uses the original CPU HF model and the pinned
serving installation's xgrammar compiler, with constraints deferred until the
Qwen reasoning-end token. This is budget evidence, not a serving-test result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import string
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessor

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"
BAD_WORDS = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
CHOICES = ["red", "green", "blue", "yellow"]
SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "lane": {"type": "integer", "minimum": 0, "maximum": 31},
    },
    "required": ["status", "lane"],
    "additionalProperties": False,
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def no_ttnn():
    assert not any(x == "ttnn" or x.startswith("ttnn.") for x in sys.modules), "TTNN imported"


def cases():
    return [
        dict(
            name="choice_0",
            prompt="Pick one color for request 0.",
            base_cap=8,
            temperature=0,
            presence_penalty=0.5,
            grammar="choice",
        ),
        dict(
            name="regex_1",
            prompt="Return one code matching LANE-[0-9] for request 1.",
            base_cap=16,
            temperature=0,
            grammar="regex",
        ),
        dict(
            name="json_2",
            prompt="Return a compact JSON object with status ok and lane 2.",
            base_cap=64,
            temperature=0,
            grammar="json",
        ),
        dict(name="plain_3", prompt="Reply with a short sentence for request 3.", base_cap=16, temperature=0),
        *[
            dict(
                name=f"bad_words_{seed}",
                prompt="Say hello to me",
                base_cap=100,
                temperature=1,
                top_p=1,
                seed=seed,
                bad_words=BAD_WORDS,
            )
            for seed in range(5)
        ],
    ]


def bad_word_ids(tokenizer):
    # sampling_params.py::update_from_tokenizer: preserve both accepted forms.
    result = []
    for word in BAD_WORDS:
        for prefix in ("", " "):
            ids = tokenizer.encode(prefix + word.lstrip(), add_special_tokens=False)
            if not ids:
                if not prefix:
                    raise ValueError("Empty unprefixed bad-word encoding")
                continue
            if not prefix or (ids[0] != result[-1][0] and len(ids) == len(result[-1])):
                result.append(ids)
    return result


class MatchedProcessors(LogitsProcessor):
    """Output-only presence/bad-word rules, then post-reasoning grammar."""

    def __init__(self, prompt_length, end_think, case, banned, xgr, compiled, eos_ids, token_log):
        self.prompt_length, self.end_think = prompt_length, end_think
        self.case, self.banned, self.xgr = case, banned, xgr
        self.matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=eos_ids) if compiled else None
        self.bitmask = xgr.allocate_token_bitmask(1, compiled.tokenizer_info.vocab_size) if compiled else None
        self.accepted = 0
        self.logged = 0
        self.token_log = token_log

    def __call__(self, input_ids, scores):
        assert input_ids.shape[0] == 1
        generated = input_ids[0, self.prompt_length :].tolist()
        no_ttnn()
        for index in range(self.logged, len(generated)):
            self.token_log.write(
                json.dumps(dict(index=index, token_id=generated[index], monotonic=time.monotonic())) + "\n"
            )
        self.token_log.flush()
        self.logged = len(generated)
        penalty = self.case.get("presence_penalty", 0)
        if generated and penalty:
            scores[0, list(set(generated))] -= penalty
        if self.case.get("bad_words"):
            for ids in self.banned:
                n = len(ids) - 1
                if n <= len(generated) and (generated[-n:] if n else []) == ids[:-1]:
                    scores[0, ids[-1]] = -float("inf")
        if self.matcher and self.end_think in generated:
            answer = generated[generated.index(self.end_think) + 1 :]
            for token in answer[self.accepted :]:
                assert self.matcher.accept_token(token), f"Grammar rejected generated token {token}"
            self.accepted = len(answer)
            self.matcher.fill_next_token_bitmask(self.bitmask, 0)
            self.xgr.apply_token_bitmask_inplace(scores, self.bitmask)
        return scores


def assert_answer(case, answer):
    if case.get("grammar") == "choice":
        assert answer in CHOICES
    elif case.get("grammar") == "regex":
        import re

        assert re.fullmatch(r"LANE-[0-9]", answer)
    elif case.get("grammar") == "json":
        value = json.loads(answer)
        assert set(value) == {"status", "lane"}
        assert value["status"] == "ok" and isinstance(value["lane"], int) and 0 <= value["lane"] <= 31
    else:
        assert answer
    if case.get("bad_words"):
        punctuation = string.punctuation.replace(">", "")
        words = [word.strip(punctuation) for word in answer.split()]
        assert not any(word in words for word in BAD_WORDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grammar-site", type=Path, default=Path("/home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages")
    )
    parser.add_argument("--max-new-tokens", type=int, choices=(2048, 4096), default=4096)
    parser.add_argument("--max-seconds-per-case", type=int, default=7200)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_seconds_per_case <= 0:
        parser.error("--max-seconds-per-case must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    no_ttnn()
    # Append only after importing original HF/torch: never replace either package.
    original = (torch.__file__, transformers.__file__)
    sys.path.append(str(args.grammar_site.resolve()))
    import xgrammar as xgr
    from huggingface_hub import snapshot_download

    assert original == (torch.__file__, transformers.__file__)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    generation = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    eos_ids = list(generation.eos_token_id)
    end_think = tokenizer.encode("</think>", add_special_tokens=False)
    assert len(end_think) == 1
    tool_start = tokenizer.encode("<tool_call>", add_special_tokens=False)
    assert tool_start
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size), max_threads=8
    )
    compiled = {
        "choice": compiler.compile_grammar('root ::= "red" | "green" | "blue" | "yellow"'),
        "regex": compiler.compile_regex(r"LANE-[0-9]"),
        "json": compiler.compile_json_schema(json.dumps(SCHEMA), any_whitespace=False),
    }
    requests = []
    for case in cases():
        messages = [{"role": "user", "content": case["prompt"]}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert ids == tokenizer.encode(rendered, add_special_tokens=False)
        assert "Reasoning effort is set to xhigh." in rendered
        assert rendered.endswith("<think>\n")
        requests.append({**case, "messages": messages, "rendered_prompt": rendered, "prompt_token_ids": ids})
    write(args.output_dir / "requests.json", requests)
    banned = bad_word_ids(tokenizer)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("full_model_hf_reference.py"),
        args.grammar_site / "vllm/v1/structured_output/__init__.py",
        args.grammar_site / "vllm/v1/structured_output/backend_xgrammar.py",
        args.grammar_site / "vllm/sampling_params.py",
        args.grammar_site / "vllm/v1/sample/ops/bad_words.py",
        args.grammar_site / "vllm/model_executor/layers/utils.py",
        args.grammar_site / "xgrammar/contrib/hf.py",
        Path("/home/ttuser/qwen38-flash/vllm-tt-plugin/tests/tt/test_structured_output_dp1.py"),
        Path("/home/ttuser/qwen38-flash/vllm-tt-plugin/tests/tt/test_host_only_params.py"),
        snapshot / "chat_template.jinja",
        snapshot / "generation_config.json",
    ]
    manifest = dict(
        complete=False,
        preflight_only=args.preflight_only,
        model=MODEL_ID,
        revision=REVISION,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        torch_path=torch.__file__,
        transformers_path=transformers.__file__,
        xgrammar_path=xgr.__file__,
        xgrammar_version=importlib.metadata.version("xgrammar"),
        eos_ids=eos_ids,
        end_think_token=end_think[0],
        tool_start_token_ids=tool_start,
        bad_words_token_ids=banned,
        max_new_tokens=args.max_new_tokens,
        max_seconds_per_case=args.max_seconds_per_case,
        thinking="checkpoint default on/xhigh; no chat-template overrides",
        grammar="same pinned xgrammar; any_whitespace=False; masks only after </think>; checkpoint EOS preserved",
        sampler_note="Native HF sampling RNG differs from vLLM; request seeds and probability controls matched, not token identity",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        checkpoint_generation_config=generation.to_dict(),
        results=[],
        ttnn_imported=False,
    )
    write(args.output_dir / "manifest.json", manifest)
    if args.preflight_only:
        print(json.dumps(dict(preflight="passed", compiled_grammars=list(compiled), request_count=len(requests))))
        return
    from models.autoports.qwen_qwen3_8_flash_next.tests import full_model_hf_reference

    full_model_hf_reference.EVIDENCE_DIR = args.output_dir / "hf_reference_loader"
    model = full_model_hf_reference.load_exact_text_model()
    no_ttnn()
    for case in requests:
        started = time.monotonic()
        print(json.dumps(dict(event="case_start", name=case["name"], max_new_tokens=args.max_new_tokens)), flush=True)
        torch.manual_seed(case.get("seed", 0))
        inputs = torch.tensor([case["prompt_token_ids"]], dtype=torch.long)
        with (args.output_dir / f"{case['name']}.tokens.jsonl").open("w") as token_log:
            processor = MatchedProcessors(
                inputs.shape[1], end_think[0], case, banned, xgr, compiled.get(case.get("grammar")), eos_ids, token_log
            )
            with torch.inference_mode():
                sequence = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=args.max_new_tokens,
                    max_time=args.max_seconds_per_case,
                    do_sample=case["temperature"] > 0,
                    temperature=case["temperature"] or 1.0,
                    top_k=generation.top_k,
                    top_p=case.get("top_p", generation.top_p),
                    repetition_penalty=1.0,
                    eos_token_id=eos_ids,
                    forced_eos_token_id=None,
                    pad_token_id=generation.pad_token_id,
                    logits_processor=[processor],
                    use_cache=True,
                )
            tokens = sequence[0, inputs.shape[1] :].tolist()
            for index in range(processor.logged, len(tokens)):
                token_log.write(
                    json.dumps(dict(index=index, token_id=tokens[index], monotonic=time.monotonic())) + "\n"
                )
        normal_stop = bool(tokens and tokens[-1] in eos_ids)
        boundary = tokens.index(end_think[0]) if end_think[0] in tokens else None
        answer_tokens = tokens[boundary + 1 :] if boundary is not None else []
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        valid, error = False, None
        try:
            assert normal_stop, "Discovery ceiling/time limit reached; usable budget unestablished"
            assert boundary is not None, "No reasoning-end token"
            assert tokens.count(end_think[0]) == 1, "Duplicate reasoning-end marker; parser equivalence unestablished"
            assert not any(
                tokens[i : i + len(tool_start)] == tool_start for i in range(len(tokens))
            ), "Tool-call marker requires implicit reasoning/tool parser handling; control equivalence unestablished"
            assert_answer(case, answer)
            valid = True
        except (AssertionError, ValueError) as exc:
            error = str(exc) or "Final-answer assertion failed"
        row = dict(
            name=case["name"],
            base_cap=case["base_cap"],
            raw_token_ids=tokens,
            raw_text=tokenizer.decode(tokens, skip_special_tokens=False),
            reasoning_tokens=boundary,
            framing_tokens=1 if boundary is not None else 0,
            answer_tokens_including_eos=len(answer_tokens),
            answer=answer,
            finish_reason="stop" if normal_stop else "length" if len(tokens) >= args.max_new_tokens else "time_limit",
            elapsed_seconds=time.monotonic() - started,
            established=valid,
            error=error,
        )
        write(args.output_dir / f"{case['name']}.json", row)
        manifest["results"].append(row)
        no_ttnn()
        write(args.output_dir / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "name",
                        "finish_reason",
                        "reasoning_tokens",
                        "answer_tokens_including_eos",
                        "elapsed_seconds",
                        "established",
                        "error",
                    )
                }
            ),
            flush=True,
        )
    manifest["complete"] = True
    manifest["budget_established"] = all(row["established"] for row in manifest["results"])
    if manifest["budget_established"]:
        manifest["suggested_extra_allowance"] = max(
            max(2 * (row["reasoning_tokens"] + row["framing_tokens"]), len(row["raw_token_ids"]) - row["base_cap"])
            for row in manifest["results"]
        )
    write(args.output_dir / "manifest.json", manifest)
    raise SystemExit(0 if manifest["budget_established"] else 1)


if __name__ == "__main__":
    main()
