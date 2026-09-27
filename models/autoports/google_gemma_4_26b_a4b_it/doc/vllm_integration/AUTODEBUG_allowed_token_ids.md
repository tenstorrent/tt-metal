# Shared allowed-token test diagnosis and repair

Date: 2026-09-27. Scope: shared sampling-test assertions and request helper only. No model/plugin runtime was changed, TT device opened, or server request issued by this investigation. Main-stage live rerun is still required.

## Verified finding

`tests/tt/test_host_only_params.py:79` asks for allowed IDs `[1,2,3]`, `[4,5,6]`, `[7,8,9]`, `[10,11,12]`, and `[13,14,15]`. It previously tested only that decoded text was nonempty. The first allowlist is incompatible with that assertion on the pinned Gemma tokenizer: **every allowed token is a special token removed by default detokenization**. A fully correct allowed-token sampler therefore fails this test.

CPU proof used `AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it", revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52", local_files_only=True)`. It did not download anything. Full results are in `readiness_vllm/allowed_token_ids_tokenizer_proof.json`.

| Allowed IDs | Token strings | `decode(ids, skip_special_tokens=True)` |
| --- | --- | --- |
| 1, 2, 3 | `<eos>`, `<bos>`, `<unk>` | Empty; each individual token also decodes empty |
| 4, 5, 6 | `<mask>`, `[multimodal]`, `<unused0>` | `[multimodal]<unused0>`; ID 4 alone decodes empty |
| 7, 8, 9 | `<unused1>`, `<unused2>`, `<unused3>` | All three visible strings |
| 10, 11, 12 | `<unused4>`, `<unused5>`, `<unused6>` | All three visible strings |
| 13, 14, 15 | `<unused7>`, `<unused8>`, `<unused9>` | All three visible strings |

The installed vLLM0.26 completion request defaults `skip_special_tokens=True` (`entrypoints/openai/completion/protocol.py:82`). It supports `return_token_ids` at line 159. The response builder uses `as_list(output.token_ids)` for each choice at `completion/serving.py:593` and derives `usage.completion_tokens` from the same IDs at lines 599-605. The checked-out vLLM source also exposes the option (`vllm/entrypoints/openai/completion/protocol.py:127`). This provides a direct correctness oracle without re-tokenizing decoded text or enabling unrelated logprob behavior.

The original test also had a false-positive gap: any nonempty text passed, even if its generated IDs violated the allowlist. Replacing the text check with token membership strengthens the intended requirement.

## Controlled experiment and smallest repair

Added `vllm/plugins/vllm-tt-plugin/tests/test_allowed_token_ids_assertions.py`. Before editing the shared test, its valid-special-token control supplied empty text for the first allowlist. The original test failed at the expected assertion:

```text
AssertionError: should produce non-empty output for request 0
1 failed in 0.07s
```

After that source/tokenizer/CPU verification, changed only shared test files:

- `tests/tt/utils.py`: optional `RequestConfig.return_token_ids=False`, forwarded through both completion request helpers when explicitly true.
- `tests/tt/test_host_only_params.py`: enable the option for the existing five cases and request full responses. Assert that generated IDs exist, at least one was generated, the count is at most `max_tokens`, **every ID belongs to that request's allowlist**, usage exists, and `completion_tokens` equals the returned ID count. Early EOS is valid and must not be forced to generate all ten tokens.
- `tests/test_allowed_token_ids_assertions.py`: valid empty-text/special-token output passes; six negative controls reject a disallowed ID, absent IDs, empty IDs, excess IDs, wrong usage count, and absent usage. Two additional cases verify forwarding through completion and chat helpers without network access.

The final CPU checks passed **9 tests in 0.06 seconds**. Exact commands, from `/workspace/tt-metal/vllm/plugins/vllm-tt-plugin`:

```bash
python -m pytest tests/test_allowed_token_ids_assertions.py::test_allowed_special_tokens_with_empty_text_pass -q --disable-warnings --tb=short
python -m pytest tests/test_allowed_token_ids_assertions.py -q --disable-warnings --tb=short
python -m black --check --target-version py310 --line-ranges 79-124 tests/tt/test_host_only_params.py
python -m black --check --target-version py310 tests/test_allowed_token_ids_assertions.py tests/tt/utils.py
git diff --check
```

The first command is the recorded pre-repair negative control; the second is the repaired suite. Formatter checks passed; formatting of the existing host-only file was limited to the changed method to preserve unrelated formatting. Whitespace checks passed. Baseline and repaired transcripts are `readiness_vllm/allowed_token_ids_baseline.log` and `readiness_vllm/allowed_token_ids_host_tests.log`. No C++ build is needed for these Python-only test changes.

## AutoDebug runner limitation and remaining evidence

The repo-local `.agents/scripts/autodebug.sh` was invoked from `doc/vllm_integration/allowed_token_ids_debug` for an independent inspection-only pass. Its nested Codex could not run even `pwd` or write a report because its sandbox launcher lacked `bubblewrap`; it produced no source-backed diagnosis. The runner transcript is `allowed_token_ids_debug/autodebug_runner.log`. That environmental failure was not treated as evidence about the sampler. The direct code reads, cached tokenizer proof, and pre-/post-repair CPU controls above verified the diagnosis in this working agent.

This repair corrects a proven assertion defect. It does **not** independently establish that the original live response contained only allowed IDs, because the original request did not return them. The main stage must rerun the corrected test against its full-model server after the current suite completes. A new token-membership failure would be a real runtime investigation and must not be hidden by further assertion changes.
