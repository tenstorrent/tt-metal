"""Insert the r1_gpqa_diamond EvalTask into the Qwen/Qwen3.6-27B EvalConfig.

Mirrors the Qwen/Qwen3.8-27B entry that tt-inference-server branch
vvukoman/add-8-models-to-release-flow adds, because that is this model's closest
sibling and the pattern the release flow is standardising on.

Data-only: no logic in tt-inference-server is changed. Run against a checkout based on
origin/vv-8models (= main + vvukoman's commit).
"""

import re
import sys

PATH = "reference_config/evals/eval_config.py"

TASK = '''            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond, mirroring the
                # Qwen/Qwen3.8-27B entry. Chosen over gpqa_diamond_cot_zeroshot
                # for four reasons, each measured on TT silicon for THIS model
                # (see tt-metal models/autoports/qwen_qwen3_6_27b/doc/):
                #   * cot_zeroshot sets no max_gen_toks, so lm-eval falls back to
                #     256 tokens, which cannot escape this model's <think> block;
                #   * cot_zeroshot sets greedy decoding, which drives this model
                #     into a repetition loop on hard items -- 16,384 tokens at
                #     50.06% duplicate 12-grams, one 12-gram repeated 1,241 times,
                #     while the same question answers correctly in 1,849 tokens
                #     with thinking disabled;
                #   * cot_zeroshot's `until` is ["</s>"], not a Qwen stop token;
                #   * cot_zeroshot's strict-match regex looks for "The answer is"
                #     while its own prompt asks for \\boxed{}, so that arm is
                #     always 0.00.
                # r1_gpqa_diamond fixes all four: a 32768 budget in its YAML, the
                # correct <|im_end|> stop list, non-greedy sampling, and its own
                # extractor scoring exact_match,none.
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=87.8,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    # No gpu_reference_score: this checkpoint has not been run on an
                    # H100 reference server. Via resolve_eval_reference() +
                    # compute_accuracy_check() the bar becomes
                    # published_score * (1 - tolerance) = 87.8 * 0.95 = 83.41% on TT
                    # silicon. That is strict and early runs should be expected to
                    # fail; published numbers run optimistic against a real serving
                    # stack. Status is EXPERIMENTAL so evals are informational
                    # (evals_enforced is False) and a failure does not block
                    # acceptance -- it becomes a real gate at FUNCTIONAL and above.
                    # Replace with a measured gpu_reference_score before promoting.
                    #
                    # Also no mode_reference_scores, so under --ci-mode the subset
                    # score is compared against the FULL-set 87.8 and will read low
                    # until a subset reference is measured.
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                # Chat endpoint so the SERVER applies the chat template, which is
                # what carries thinking mode; client-side apply_chat_template on
                # /v1/completions would bypass it.
                use_chat_api=True,
                model_kwargs={
                    # Matches the P300X2 spec's max_context. Declared explicitly
                    # because lm-eval's local backends otherwise default to
                    # max_length=2048 and truncate prompts client-side.
                    "max_length": 262144,
                },
                gen_kwargs={
                    # stream=false is REQUIRED: lm-eval's local-chat-completions
                    # streaming parser raises KeyError 'message' on every response.
                    "stream": "false",
                    # 32*1024 rather than the sibling's 80*1024. Measured on this
                    # hardware decode is ~56 ms/token, so 80*1024 is up to ~76 min
                    # PER DOCUMENT and ~12.7 h for the 10-document CI subset, while
                    # 32768 is ~31 min worst case. Observed convergence on real
                    # Diamond documents under this sampling ranged ~5k to ~19k
                    # tokens, so 32768 has headroom; raise it for a published-number
                    # run on faster silicon.
                    "max_gen_toks": 32 * 1024,
                    "until": [],
                    "do_sample": "true",
                    # Qwen3.6-27B generation_config.json, thinking mode:
                    # temperature 1.0 / top_k 20 / top_p 0.95. Greedy is what breaks
                    # this model, so these are not optional.
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                # Reasoning eval at low batch: documents run effectively
                # sequentially and dominate CI runtime.
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
'''


def main():
    src = open(PATH).read()

    anchor = 'hf_model_repo="Qwen/Qwen3.6-27B",'
    i = src.find(anchor)
    if i < 0:
        sys.exit("FAIL: could not find the Qwen/Qwen3.6-27B EvalConfig anchor")
    if src.count(anchor) != 1:
        sys.exit(f"FAIL: expected exactly one anchor, found {src.count(anchor)}")

    # Scope every subsequent check to THIS EvalConfig only. A fixed-size window
    # spills into the next entry -- the Qwen3.8-27B config that follows legitimately
    # contains r1_gpqa_diamond, which made an earlier version of this script report
    # "already applied" and silently do nothing.
    nxt = src.find("hf_model_repo=", i + len(anchor))
    end = nxt if nxt > 0 else len(src)
    block = src[i:end]
    if 'task_name="r1_gpqa_diamond"' in block:
        print("  already applied to THIS entry, nothing to do")
        return
    print(f"  entry spans {end - i} chars; tasks currently present:")
    for line in block.splitlines():
        if "task_name=" in line:
            print("   ", line.strip())

    # insert immediately after this EvalConfig's `tasks=[`
    m = re.compile(r"\n(\s*)tasks=\[\n").search(src, i)
    if not m or m.end() > end:
        sys.exit("FAIL: could not find tasks=[ inside this EvalConfig")
    at = m.end()

    out = src[:at] + TASK + src[at:]
    open(PATH, "w").write(out)
    print(f"  inserted r1_gpqa_diamond task at offset {at}")
    print(f"  file grew {len(src)} -> {len(out)} chars")


if __name__ == "__main__":
    main()
