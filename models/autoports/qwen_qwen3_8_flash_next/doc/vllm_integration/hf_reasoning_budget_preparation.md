# Native reasoning-budget control preparation

Prepared and then executed `tests/run_hf_reasoning_budget.py` on 2026-11-16.
CPU tokenizer, grammar and synthetic-logit checks passed. The completed native
run is **not a combined pass**: eight cases reached final answers, while
`bad_words_3` stopped at EOS before the reasoning-end marker. See
`hf_reasoning_budget_outcome.md` and `readiness_vllm/hf_reasoning_budget_v1/manifest.json`. This controls the existing `--tt-reasoning-token-budget` option; it does
not change the served recipe or replace any sampling assertion.

The nine requests are the exact structured-test choice/regex/JSON/plain inputs
with request IDs 0/1/2/3, plus the bad-word prompt with seeds 0 through 4.
The checkpoint chat template is used without overrides: thinking on, effort
xhigh. The original HF loader executes all 48 layers with original checkpoint
dtypes and exact mmap weights/PLE rows. The script never imports vLLM or TTNN.
It imports original `torch 2.11.0+cpu` and `transformers 5.22.2` first, then
appends the pinned serving site directory to resolve its unchanged xgrammar
and tvm_ffi packages. No package installation or environment mutation is needed.

The running acceptance server records `reasoning_parser='qwen3'`,
`enable_in_reasoning=False`, `disable_any_whitespace=True`. Pinned vLLM
`v1/structured_output/__init__.py` begins grammar constraints after the reasoning
end marker and passes `sampling_params.all_stop_token_ids` to the matcher.
`backend_xgrammar.py` compiles JSON with
`any_whitespace=not disable_any_whitespace`; choice uses the same EBNF literals,
and regex uses the same compiler. The automatic backend tries xgrammar first;
the present schema has none of the unsupported features listed in
`has_xgrammar_unsupported_json_features`. The native control uses these same
grammar settings and overrides matcher stop tokens with checkpoint EOS IDs
248046 and 248044. With whitespace disabled, xgrammar's canonical JSON spaces
remain required (for example `{"status": "ok", "lane": 2}`); whitespace is not
arbitrarily removed.

Presence penalty subtracts 0.5 once per previously generated token, excluding
the prompt, matching the choice request and pinned vLLM penalty definition.
Bad-word tokenization preserves vLLM's accepted unprefixed and space-prefixed
forms, and its masking uses generated-token history only. Synthetic logits
matched the pinned vLLM bad-word function on 14 histories. The sampled bad-word
requests use temperature 1, explicit top_p 1, checkpoint top_k 20, and seeds
0–4. HF and vLLM RNG algorithms differ; exact sampled token identity is not a
claim of this control. Greedy requests retain their original temperature 0.

`hf_reasoning_budget_validation.json` records the synthetic checks. Full
xgrammar decoded-vocabulary SHA-256 is
`89cda81bb3e45c2e4d6b10b8bd9f33972847d9ac6583f3fa530ccec6cdfc2b63`
in both original transformers 5.22.2/Qwen3_5Tokenizer and serving
5.14.1/Qwen2Tokenizer. Their xgrammar metadata also matches: vocab_type 2,
vocab_size 248320, add_prefix_space false, tokenizer stop ID 248046. The
additional checkpoint stop ID is supplied explicitly to each matcher.

An independent source review confirmed the Qwen3 parser preserves content
after `</think>` without injecting or stripping a newline. A generated tool
call or duplicate reasoning-end marker follows additional parser branches;
the control flags either case as an unestablished parser match instead of
claiming an equivalent result.

The discovery limit is 4096 new tokens per request, with normal EOS stopping,
7200 seconds per request and an outer 86400-second process bound. A length or
time limit, missing reasoning-end token, or failed original answer assertion
leaves the budget **unestablished**. The control requires normal EOS even though
the plugin can accept a valid truncated final answer; this is conservative
budget discovery, not an equivalent replacement test. The suggested extra
allowance is the maximum of twice the observed reasoning-plus-boundary tokens
and observed total tokens minus each test's original answer cap. Thus observed
answer/EOS tokens also fit. It remains an empirical allowance, not a bound on
unseen generations.

Every generation retains raw token IDs/text, reasoning and answer counts,
finish reason and elapsed time. Per-token JSONL flushes during generation;
per-case JSON and the manifest update after each request. A hard process kill
can leave the manifest incomplete; it must never be interpreted as a pass.
Source paths/hashes, model revision, original package paths, full generation
config and exact rendered prompts are recorded at launch.

Historical `readiness_vllm/quality_hf_v1` generated 3072 raw tokens in
2898.433 seconds (0.943 seconds/token overall). Each 256-token request took
169.782–330.855 seconds. A linear **planning estimate**, not a performance
measurement for this workload, is 9.7 hours for all nine requests reaching
4096 tokens at that average, or 13.2 hours using the slowest historical case.
Normal EOS may shorten this. Longer sequences, cold pages and contention can
make the estimate optimistic; a cap hit does not establish a budget.

The loader maps 251,487,307,800 parameter bytes (234.2 GiB), not 68 GiB. It
does not materialize all mapped pages simultaneously; the historical control
successfully used page-cache-backed weights. At preparation time `free -h`
reported 249 GiB RAM, 97 GiB available and 3.3 GiB swap used while serving was
active. Wait for root authorization and server cleanup, recheck memory/process
ownership, and run with CPU/memory otherwise dedicated. A concurrent TT probe
would add host pressure and make the runtime estimate less reliable.

Executed launch command (also used OMP_NUM_THREADS=8 and PYTHONUNBUFFERED=1):

```bash
source /home/ttuser/qwen38-flash/lane-env.sh
M=models/autoports/qwen_qwen3_8_flash_next
timeout --signal=TERM --kill-after=60 86400 python_env/bin/python \
  "$M/tests/run_hf_reasoning_budget.py" \
  --output-dir "$M/readiness_vllm/hf_reasoning_budget_v1" \
  --max-new-tokens 4096 --max-seconds-per-case 7200 \
  > "$M/doc/vllm_integration/logs/hf_reasoning_budget_v1.log" 2>&1
```

The output directory must not already exist; the script preserves earlier
attempts by refusing overwrite. After a fully established reference, pass the
measured allowance through inherited
`PYTEST_ADDOPTS='--tt-reasoning-token-budget=N'` to the unchanged shared runner's
full sampling profile. This does not resolve unrelated penalty-sensitivity or
seed/variety failures.
