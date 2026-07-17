# AutoDebug: Gemma Stage 11 IFEval schema failure

## Verdict

The failure is a deterministic preparation/validation bug, not a model, API, or hardware failure. `prepare_gemma_meta_eval.py` deliberately deletes the source `prompt` column while retaining `input_question`; the copied Meta IFEval scorer then unconditionally reads `doc["prompt"]`. The cache validator compounds the bug by declaring every parquet containing `prompt` invalid, so it positively certifies data that cannot be scored.

The smallest raw-base repair is to publish a raw `prompt` alias equal to `input_question`, make the task generate from `prompt`, and validate equality plus the scorer's remaining required schema. This preserves the intended base-model prompt exactly and does not restore the removed Llama-Instruct rendering.

## Headline finding: preparation violates the scorer contract

Evidence-ranked confidence: **certain**.

1. `workflows/prepare_gemma_meta_eval.py:194-204` copies each IFEval row while excluding both `prompt` and `input_final_prompts`, then uses `input_question` as the raw generation text.
2. Its generated YAML uses `doc_to_text: input_question` (`prepare_gemma_meta_eval.py:32-47`), so all 541 requests can be built and served successfully without exposing the missing metric field.
3. The staged cookbook scorer's `process_results` reads `kwargs`, `key`, `instruction_id_list`, and `prompt` unconditionally (`.../work_dir/ifeval/utils.py:111-123`). The observed `KeyError: 'prompt'` at line 120 occurs only after generation because `process_results` is the first consumer that needs it.
4. `prompt` is semantic, not dead compatibility data: strict and loose scoring pass it to instructions whose argument set contains `prompt` (`utils.py:39-42` and `89-92`). It therefore must be the exact prompt the model saw.
5. The prepared parquet was statically inspected and contains `input_question`, `instruction_id_list`, `kwargs`, and `key`, but no `prompt`. This accounts for the complete observation matrix: context construction succeeds, 541 raw completions succeed, and scoring immediately fails on the first document.

## Minimal schema-preserving raw-base fix

In the IFEval preparation loop:

- Continue deleting `input_final_prompts` and never reuse the old rendered Llama `prompt`.
- Rebuild each row with `prompt = input_question` after copying its source fields.
- Change the generated IFEval YAML to `doc_to_text: prompt`. Keeping `input_question` would generate identical text if equality is enforced, but using `prompt` makes the scorer and request path share one authoritative field and prevents later drift.
- Continue checking raw prompts for emptiness and Llama control markers, now against `prompt` (and require it to equal `input_question`).

The scorer-visible IFEval row contract is:

- `prompt`: non-empty string; exact equality with `input_question`; used for generation and prompt-dependent instruction scoring.
- `input_question`: non-empty raw source question retained as provenance/equality anchor.
- `key`: required by `process_results` when constructing `InputExample`.
- `instruction_id_list`: required list of instruction IDs; scoring iterates it.
- `kwargs`: required list aligned one-to-one with `instruction_id_list`; each item must be a mapping. This staged utility also directly indexes `item["nth_paragraph"]`, so that key must exist in every item (nullable is valid).

`previous_is_correct` and `output_prediction_text` are present in the cookbook parquet but are not read by this task's request or metric path. They may be preserved; they are not required to resolve this failure.

## Cache validator and test corrections

`workflows/workflow_venvs.py:138-164` currently requires `doc_to_text: input_question` and rejects any IFEval parquet with a `prompt` column. Invert those two assumptions and fail closed on metric viability:

- require `doc_to_text: prompt`;
- require columns `prompt`, `input_question`, `key`, `instruction_id_list`, and `kwargs`;
- reject any row where `prompt` is empty, differs from `input_question`, or contains a Llama control marker;
- reject malformed instruction metadata: non-list/empty IDs, non-list kwargs, unequal lengths, non-mapping kwargs items, or absent `nth_paragraph` keys;
- continue rejecting `input_final_prompts`, preserving 541 rows, and enforcing the 113280 context manifest without truncation or alignment.

`tests/test_workflow_venvs_meta.py` currently encodes the bug in `_write_gemma_required`: its IFEval fixture has only `input_question`, and the setup assertion expects `doc_to_text: input_question` (lines 74-93 and 191-193). Update the fixture to create scorer-complete rows with `prompt == input_question`, `key`, `instruction_id_list`, and aligned `kwargs`; expect `doc_to_text: prompt`. Add focused negative tests for missing prompt, prompt/input mismatch, Llama markers in prompt, each missing scorer-required column, unequal ID/kwargs lengths, and malformed/missing `nth_paragraph`. A cheap scorer contract test should feed one prepared row and a dummy result to the copied `process_results` and assert that all four metric keys are returned; this would have caught the current crash before a 541-request release run.

## Can the completed generations be reused?

**No safe reuse artifact exists; run a fresh release/eval after the fix.** The logged command has `--log_samples` but no lm-eval `--use_cache`, and the release output directory contains no completed sample/result JSONL for this crashed task. The responses remained in the failing lm-eval process until `process_results` raised, before normal result/sample serialization. Reconstructing them from server logs would lack a harness-proven document/result mapping and would not constitute a valid release artifact. A fresh run is required; with `prompt == input_question`, it sends the same raw prompt text as the failed run.

## Ruled out

- **Context cap/alignment:** the request command uses `max_length=113280`, and all 541 generations completed. The exception is a Python document-key lookup after inference.
- **Server/model correctness:** the server returned all requested generations; no response/API error precedes the traceback.
- **Cookbook scorer defect:** the scorer's required `prompt` field is internally consistent with its `InputExample` and prompt-dependent instruction semantics. The local preparation removed that required field.

## Intervention boundary

Repair `prepare_gemma_meta_eval.py`, `_meta_eval_artifacts_valid`, and their tests only. Do not modify the installed/copy of the Meta IFEval scorer, do not reintroduce Llama chat-control text, and do not change prompt lengths, generation budgets, task sample counts, or server/model code.
