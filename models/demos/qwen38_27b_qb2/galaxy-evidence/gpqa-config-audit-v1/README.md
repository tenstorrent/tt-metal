# GPQA configuration and prompt audit — October 9, 2026

The evaluator was not preserving the benchmark's answer text. The pinned
`tstescoTT/lm-evaluation-harness` revision
`321e3bb68cb750a58c76606ab57832533302be73` imports the zero-shot GPQA
preprocessor into `r1_gpqa_diamond`. That preprocessor applies
`re.sub("\\[.*?\\]", "", text)` to every answer choice.

This removes bracketed scientific notation from **38 choices in 12 questions**.
Seven of those questions were marked incorrect in the completed BFP8 run.
At row 171, four originally distinct answers become the same displayed string;
the legacy `choices.index(correct_answer)` then derives the label from the
first collision. This is a confirmed evaluation defect, not proof that fixing
it will make all seven answers correct or explain every model error.

A synthetic reproduction, containing no benchmark content:

| Original answer | Legacy displayed answer |
| --- | --- |
| `Vector [1, 2]` | `Vector ` |
| `Vector [2, 3]` | `Vector ` |
| `The concentration is [H+] = 0.1 M` | `The concentration is = 0.1 M` |

## Correction and validation

`tests/gpqa_documents.py` retains answer text except for surrounding whitespace,
shuffles tagged answer entries in the existing order, and derives the gold
letter from the original correct entry. The client records
`choice_processing=preserve_scientific_notation_v1` in its protocol receipt.
The prompt template, dataset revision, 198-question denominator, sampling,
output budget and numerical qualification gate remain unchanged.

- **28 CPU tests pass**, including scientific vectors, concentrations,
  intervals, nested brackets, duplicate strings, deterministic permutations,
  empty-choice rejection, streaming/scoring and protocol reporting.
- Preparation of the actual pinned 198-question dataset confirms every answer
  choice is retained, all choice permutations are unchanged, and row 171's gold
  label is restored to the original answer entry.
- Fifteen rendered prompts change: twelve lose the bracket-removal defect;
  three additional prompts retain internal whitespace previously normalized
  by the legacy preprocessor. The source answer text is authoritative.
- Original input SHA256:
  `73f40d9f36c2cf2070120e34c425f0a10aafaa7edd8c1f6878b002e7bf756114`.
- Corrected input SHA256:
  `1fea5e48b7d2faa1ee1257ee9de7f63a30d44b5150906311e447a9190365cf1c`.

The CPU tests ran in an isolated task directory on the existing host. They did
not open devices, modify the running server, overwrite existing inputs/results,
or run fresh model inference. The original job completed with its frozen source.
A corrected full-model GPQA run has **not** been launched by this audit.

## Other configuration checks

The live checkpoint's config, generation config, tokenizer config, chat template,
tokenizer JSON, vocabulary, merges and weight index match the official pinned
revision byte-for-byte. Full weight shards were not rehashed during this audit;
the earlier copy receipts are retained separately.

The live endpoint's CPU `/tokenize` route exactly matched the pinned tokenizer's
58 token IDs for a synthetic public prompt. Both the file and embedded templates
have SHA256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
Default thinking equals explicitly requesting `xhigh` and `preserve_thinking`.
Installed serving versions are Transformers 5.12.1, vLLM 0.26.0 and Tokenizers
0.22.2. The model's `qwen3_5` architecture identifier is expected for Qwen3.8.

The GPQA client requests temperature 1, top-p 0.95, top-k 20, seed 42 and 65,536
output tokens. The R1 task YAML's different generation defaults are bypassed by
the explicit API payload. vLLM defaults are min-p 0, presence/frequency penalties
0 and repetition penalty 1. Generation config declares both EOS IDs 248046 and
248044. This verifies configuration, not statistical equivalence of the TT and
GPU samplers. Source inspection confirms the requested k=20 is supported and
temperature is converted to the inverse expected by the native sampler.

At the 196-response audit snapshot, all saved-response hashes matched the scored
receipts, rerunning the installed scorer reproduced 170 correct, and combining
reasoning plus final content did not recover a correct answer. An independent
boxed-letter extraction agreed on all 195 nonempty final answers. One natural
stop had no final content. The final independent completion audit covers all
198 responses; it does not repair the prompt defect.

The official [pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)
reports 89.2% GPQA Diamond and recommends the sampling settings above. It does
not establish an identical GPQA prompt, choice permutation, output budget or
aggregation protocol. Its `avg@3` footnote concerns QwenSWEBench, not GPQA.
The [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
also documents `xhigh` as the default. The numerical gate remains 177/198;
passing it alone would not prove an exact reproduction of the published score.

## Completed original run and next comparison

The unchanged BFP8/HiFi2 run finished at 11:26:26 UTC:
**171/198 (86.36%)**, no length cutoffs, 59m51s. Its completion audit and durable
capture both completed successfully. Mean client decode rate was 12.93 tok/s/user;
aggregate output was 340.45 tok/s, including the entire evaluation interval.

Against the previous native BFP4 run, 160 questions were correct in both,
10 only in BFP4, 11 only in BFP8, and 17 in neither. That single sampled
comparison, using corrupted prompts, cannot establish the numerical cause of
the remaining quality gap. BFP8 still improves the separate short HF-logit
comparison, but that is not a corrected benchmark qualification.

Next, run the full corrected protocol from a fresh output directory, keeping
precision and sampling fixed. Do not retrofit corrected labels onto old
responses, splice new answers into old scores, select favorable seeds, or treat
the seven affected failures as recovered. A matched higher-precision reference
and a concurrency control remain useful if the corrected run still falls short.

## Evidence and audit limitations

`configuration-audit.json` contains checkpoint/tokenizer verification and the
196-response snapshot. `preparation.json` contains full-dataset validation,
affected IDs and hashes, without benchmark question or answer text.
`final-validation/unit.log.gz`, `final-validation/unit.xml.gz`,
`final-validation/manifest.json` and archived audit scripts make the final
checks reproducible; the top-level receipts preserve the earlier test snapshot. The final original result is in `../decoder-gpqa-result-v1/`.

Audit development failures are preserved rather than hidden: the first isolated
test staging omitted `models.perf`, so 27 tests passed and one import failed.
The second staging passed all 28 tests, then its diagnostic incorrectly required
the legacy collision-derived label to stay unchanged. The final preparation
allows a label correction only where that collision is proven, verifies the
original answer text, and confirms every shuffle position is unchanged.
An initial scratch check compared normalized answer text with unnormalized text,
and a tokenizer probe compared a `BatchEncoding` mapping with token IDs. Both
were corrected in the audit; neither was a serving defect or changed a run.

The repository pre-commit hook required `expect_error` instead of `pytest.raises`
in the new negative tests. The fixture was adopted and the complete 28-test CPU
suite plus 198-question preparation were repeated in `final-validation/`.
