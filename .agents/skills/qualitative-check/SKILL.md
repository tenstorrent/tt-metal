---
name: qualitative-check
description: Run and review qualitative or prompt-based evaluation checks for Hugging Face text models during TTNN bringup. Use whenever a stage generates text, judges output quality, runs vLLM/TTI/API qualitative smokes, compares HF and TT outputs, investigates wrong-language or base-model autocomplete behavior, or reviews whether prompt format and shifted-left qualitative evidence are valid.
---

# Qualitative Check

## Purpose

Make model text-quality checks comparable and prompt-correct. Use this skill for any qualitative generation, prompt-based eval, API smoke, vLLM qualitative run, TTI release text check, or stage review that relies on generated text.

## Prompt Format

Use the prompt format declared by the Hugging Face checkpoint.

1. Load the tokenizer/config for the exact HF model id or local checkpoint used by the stage.
2. If the tokenizer has a non-empty `chat_template` or supports `apply_chat_template` with that template, treat the model as chat/instruct:
   - render message prompts with `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`, or send equivalent messages through `/v1/chat/completions`;
   - keep system/user/assistant roles exactly as the prompt suite defines them;
   - do not judge the model from raw `/v1/completions` prompts alone.
3. If the tokenizer has no chat template, treat the model as base/completion:
   - use plain continuation prompts;
   - do not invent a chat wrapper or judge poor chat-style outputs as model failure.
4. Record the decision in the evidence: HF model/revision, tokenizer class, whether `chat_template` was present, prompt mode (`chat` or `completion`), endpoint or rendering method, generation parameters, and prompt source path.

Raw completion output from a chat/instruct model is allowed only as labeled continuation stress coverage. It is not a pass/fail quality verdict unless a correctly formatted run is also present.

## Controls

Every quality verdict needs a control rendered with the same prompt format:

- Prefer HF reference generation from the same model id/revision and tokenizer.
- For serving regressions, also compare against the most recent full-model or previous-stage TT output on the same prompt suite.
- If HF fails the same prompt in the same way, record that as a model/control behavior, not a TT serving bug.
- If TT output is materially worse than the HF or previous-stage control, treat it as stage work: token feedback, cache/position handling, sampling, trace replay, dtype/fidelity, or adapter state are common causes.

## Artifacts

Leave small, inspectable artifacts under the stage evidence directory:

- prompt-format metadata, for example `qualitative_prompt_format.json`;
- rendered prompts or prompt token ids for each prompt id;
- HF control outputs;
- TT/full-model/vLLM/TTI outputs;
- degenerate-output check result when available;
- a short verdict that cites concrete prompt ids and output snippets, not only a summary.

Do not store secrets, auth files, model weights, or bulky tensor/profiler dumps as qualitative evidence.

## Verdict Rules

More work is required if:

- an instruct/chat model is judged only from raw completion prompts;
- a base model is judged through invented chat prompts;
- the prompt-format decision is missing or contradicted by artifacts;
- a stage from full-model onward skips the shared qualitative suite without a concrete capability blocker;
- generated text shows wrong language, prompt echo, mechanical repetition, doubled subwords, control-token leakage, cross-request leakage, repeated or corrupt first token, or gibberish and no matching HF/control behavior explains it;
- a runner, API endpoint, or eval harness cannot send the correct prompt format and the stage uses its output anyway.
- the verdict includes stale or pre-fix debug captures instead of only current post-fix outputs.
