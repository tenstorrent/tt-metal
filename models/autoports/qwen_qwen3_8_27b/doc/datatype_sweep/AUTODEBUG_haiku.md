# AutoDebug: selected-policy haiku instruction failure

Date: 2026-09-13. Source-only fresh-context investigation under the repository
AutoDebug, AutoFix, and qualitative-check skills. The coordinator owns all device
work. This investigation ran no Torch/TTNN imports, device tests, resets, or model
generation, and changed only this report. No nested reviewer was started.

## Verdict

The selected output has a real instruction-following failure: its final third
line, “A new truth emerges,” conventionally has six syllables (1+1+1+3), making
the answer 5/7/6. The pinned HF and Stage 7 controls both deliver 5/7/5. This is
not explained by truncation: selected p0 reaches EOS at 347 tokens. The saved
output explicitly miscounts “emerges” as two syllables before emitting that
line. The observation does not establish a tokenizer, cache, trace, sampler, or
native-kernel implementation bug.

Head precision is the leading falsifiable explanation for the different
free-running branch. No source-proven runtime defect was found. The earlier
Stage 6 cache-capacity caveat is relevant to experiment design, but it does not
by itself explain this Stage 7/Stage 8 comparison: both extended p0 artifacts
indicate the same newly allocated 1088-token, 34-page cache.

A focused current-source, fixed-geometry selected/baseline/head-fidelity control
is required before classifying the failure as a demonstrated precision quality
regression or a reproducibility/state issue. A clean degeneration check and the
passing AIME24 gate do not make this particular answer correct. Conversely, a
correct answer at higher precision would not alone prove that LoFi is implemented
incorrectly or justify an untested runtime patch.

## Inspected evidence and contract

- `goal_contract.md`: select the fastest evaluated accuracy-passing **traced
  teacher-forcing** policy, retain capabilities, obtain independent stage review;
  do not begin vLLM. The report does not silently replace that ranking rule with
  an alternative precision-selection criterion.
- `work_log.md`, `README.md`, `commands.log`, `selected_readiness.json`,
  `selected_readiness.environment.json`, and its source snapshots. The selected
  readiness launcher has recorded exit status 0 and resolves the selected policy
  through the normal constructor with no `QWEN_PRECISION_CONFIG` override.
- Current `tt/generator.py`, `tt/model.py`, `tt/precision.py`, relevant cache and
  SDPA code in `tt/optimized_decoder.py` and `tt/multichip_decoder.py`,
  `tests/tt_qualitative.py`, and `tests/run_readiness.py`.
- Selected, Stage 7, and pinned HF extended p0 tokens/text; the original
  256-token selected and Stage 7 p0 artifacts; and
  `doc/full_model/qualitative_review.md` plus the Stage 7 qualitative review.

Current base HEAD is `624f6352a9ca1c339870dbe517e564910bb71b15`; the working tree
contains ongoing stage changes. All five inspected current runtime files
(`generator.py`, `model.py`, `precision.py`, `optimized_decoder.py`,
`multichip_decoder.py`) are byte-identical to their
`source_snapshots/selected_readiness/` copies at investigation time.

The existing AIME24 result is a separate, limited measurement: full64 B1/S203/G100,
selected prefill 100/100/100% and decode 99/100/100% top-1/top-5/top-100. The recorded
selected TF rate is 40.878129 tokens/s/user; BFP4/HiFi2 head is 40.734779, and the
policy-plumbed BFP8/HiFi2 head baseline is 40.141822. No new performance was
measured by this investigation.

## What the saved comparison establishes

All three extended p0 controls use the exact same rendered chat prompt, including
the checkpoint's xhigh system message and assistant `<think>` prefix, and the
same 60 prompt token IDs. All have a 1024-token budget. The pinned revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, tokenizer `Qwen2Tokenizer`.
The HF control is a left-padded batch of four; TT processes prompts singly.
That HF batching difference is preserved, not confused with a same-runtime A/B.

| Output | Saved tokens including EOS | Final third line | Syllables |
| --- | ---: | --- | --- |
| Selected BFP4/LoFi head | 347 | A new truth emerges | 6 |
| Stage 7 BFP8/HiFi2 head | 418 | Clarity forms now | 5 |
| Pinned HF | 284 | answers emerge now | 5 |

Selected and Stage 7 tokens first differ at zero-based generated index **31**:
selected token 6970 versus Stage 7 token 30, in the early “Ensure 5-7-5” wording.
The selected text continues “maybe. Could be:” and Stage 7 continues “? Maybe:”.
Both first differ from HF at index 16. Those early differences precede the final
answer; comparing logits on the eventual third line after independently running
each branch would mix head effects with different autoregressive histories.

For generated index 31, `generate` predicts from decoder input at absolute
position `S + 31 - 1 = 90`: page 2, in-page offset 26 with 32-token pages. With
k-chunk 64 this is chunk 1, offset 26. The first mismatch is not at an absolute
page/chunk boundary. An index near a generated-token multiple of 32 is not by
itself evidence of a paged-cache or history-buffer cliff.

## Source findings

### 1. Precision is actually consumed and isolated from final norm

`tt/precision.py:43-96` chooses explicit override, environment, selected artifact,
then baseline; `build_generator` passes its explicit precision argument to
`QwenModel` (`tt/generator.py:705-713`). `QwenModel` forwards decoder policies
at `tt/model.py:67-78`, uploads the head with the chosen weight dtype at lines
89-93, and builds its head compute config from the chosen fidelity at lines
110-115. Both head implementations use that config and the BF16 logits dtype
at lines 215-241. DRAM decode weights are derived from the uploaded head at
lines 97-109.

Final RMSNorm has a separate, fixed HiFi2 compute config at lines 116-121 and
190-209. The selected head LoFi does **not** accidentally lower final norm
fidelity. Sampling remains BF16 with uint32 token storage. The candidate JSONs
are full policies, not partial overrides.

For the three recommended controls, all decoder groups remain BFP4/LoFi with
FP32 destination accumulation; attention/MLP activations, residuals, CCL, logits,
sampling, and norm remain BF16; KV remains BFP8; final norm remains HiFi2. Only
the head dtype/fidelity and descriptive `config_id` differ. There is no reason
to change KV precision, decoder groups, head geometry, or sampler in the first A/B.

### 2. Generation length and prior requests can alter cache geometry

`generate` asks for `S + G - 1` cache capacity at `tt/generator.py:611`.
`_ensure_cache` retains an existing same-batch cache when large enough and rounds
new capacities to 32 at lines 190-210. With S60/G1024, a new allocation is 1088
tokens and 34 pages. Both selected and Stage 7 extended p0 counters record one
`page_table_allocations`, one `history_allocations`, and identical trace/history
event counts. The current source plus these counters supports equivalent newly
allocated geometry, although neither artifact directly records the cache shape.

The SDPA decode choice at `tt/optimized_decoder.py:747-770` uses a special short
grid below 16 mapped pages; its k-chunk is reduced until mapped capacity is
divisible. TP4 sets nominal `sdpa_k=128` in `tt/multichip_decoder.py:61`.
For 34 pages, both extended p0 executions should use the general grid and
k-chunk 64. The older fresh S60/G256 request allocated 320 tokens/10 pages and
used the short grid/k-chunk 32. That earlier comparison changes numerical
execution. The current extended-to-extended comparison does not cross that
branch on the inspected evidence.

Directly record capacity, page-table shape/content, KV shape/dtype, page size,
local head geometry, and resolved SDPA k/grid in the new controls. Relying on a
generation budget alone is insufficient because a previously larger cache is
retained. Request order must also be recorded.

### 3. Reset, token feedback, and trace paths have no demonstrated defect here

`generate` resets cache/request state before prefill (lines 611-614);
`QwenModel.reset_cache` zeroes key/value/conv/recurrent state (model lines 151-156).
The first sampled token occupies the same device buffer later decoded; positions
and RoPE indices are set to S (generator lines 631-645), then each decode advances
them once (lines 331-345). Deferred output appends one uint32 history row per
sampling replay and reads the valid prefix (lines 82-114 and 648-669).

`_capture` preserves recurrent/convolution state, positions, RoPE, sampled tokens,
seeds, and history cursor across warmup (lines 351-379). KV warmup writes the
current row and the actual replay overwrites that same row; causal attention
hides future rows. The absence of a full KV backup alone is not a proven bug.
No changed capture or delivery logic appears in the Stage 8 generator diff;
changes there are precision construction and external-cache dtype validation.

`tests/tt_qualitative.py:25-29` asserts the exact control prompt IDs, calls the
normal generator, and trims saved output at first configured EOS. It does not
rewrite the answer or enforce syllables. Its copied `do_sample=False` metadata
agrees with the actual `generate` defaults k=1, p=0, temperature=1, seed=0.
`run_readiness.py` performs full64 AIME controls and the original qualitative
suite before extended p0. The extended p0 then grows the cache, releases old
traces, and captures new traces. Reproducing this order is a second control if a
fresh isolated p0 behaves differently.

## Falsifiable hypothesis ledger

| Hypothesis | Prediction and smallest discriminating evidence | Current verdict |
| --- | --- | --- |
| H1: head quantization/fidelity changes greedy ranks, leading to an ordinary approximate-model instruction error | Fixed-shape repeated selected output is stable; current-source baseline changes the branch/answer; BFP4/HiFi2 isolates head fidelity with BFP4 weights unchanged. At the common-prefix mismatch, hidden states agree while head logits/ranks differ. | Leading explanation; unverified causal claim. |
| H2: retained cache shape or request/capture history explains the Stage 7/8 delta | With a fixed policy and cache geometry, changing only prior request/capture history reproduces a token change; or explicitly logged old/new capacity differs. | Possible generally, but same newly allocated 34-page geometry in both current artifacts weakens a capacity-only explanation. |
| H3: incorrect sampling/token delivery or trace state causes the failure | Same-geometry repeated selected runs disagree, or device greedy output disagrees with its own captured logits outside legitimate equal-max ties; immediate versus deferred delivery changes the same model token stream. | No supporting symptom/source defect found; investigate only if controls expose it. |
| H4: final-norm fidelity accidentally followed the head | Selected runtime final norm is LoFi while baseline is HiFi2. | Refuted by inspected source/config: both are fixed HiFi2. No norm patch justified. |
| H5: output budget or malformed prompt created the bad final line | Selected lacks EOS, has different prompt IDs/template, or the line is only a truncated proposal. | Refuted for saved p0: same prompt, complete EOS answer, wrong syllable count. |

## Smallest controlled experiment for the coordinator

Use the existing packaged device environment and normal TP4 Ring construction.
Run serially with watcher/profiler disabled as in selected readiness. Each policy
must use current source and a fresh model/process; do not alter the selected
artifact, use reduced layer counts, or edit a launcher while it runs.

1. Run selected `configs/head_bfp4_lofi.json` p0 at G1024 twice in one fresh
   generator. Explicitly reserve cache capacity 1088 and history capacity 1023
   before the first request; assert both capacities remain identical. The first
   run exercises capture, the second its warmed prefill/decode replays. Preserve
   both outputs even if one changes. `generate` already resets request state.
2. Run current-source `configs/baseline_bfp4_lofi_head_bfp8_hifi2.json` in a fresh
   process using exactly the same two-request protocol.
3. Run `configs/head_bfp4_hifi2.json` using the same protocol. This changes only
   head fidelity relative to selected. If necessary to distinguish weight/fidelity
   interaction, add `configs/head_bfp8_lofi.json` to complete the 2x2 comparison;
   it is unnecessary before the first three outcomes are known.

The existing helper already exposes the needed prompt filter. A temporary
coordinator-owned probe can call the following without changing model code:

```python
gen = build_generator(root, mesh, precision_config=config_path)
gen._ensure_cache(1, 1088)
gen._ensure_history(1023)
for repeat in range(2):
    artifact = run(
        gen,
        root,
        "hf_qualitative_extended.json",
        max_new_tokens=1024,
        prompt_ids=[0],
        output_dir=probe_output_dir,
        output_name=f"{config_id}_repeat{repeat}.json",
    )
    assert gen.cache.capacity == 1088
    assert tuple(gen.page_table.shape) == (1, 34)
    assert gen.history_capacity == 1023
```

Here `build_generator` is from `tt.generator`, and `run` is from
`tests.tt_qualitative`. The probe must supply the normal `configure_fabric`, mesh
open/close, exception cleanup, source/precision hashes, command/exit status, and
metadata sidecar. These proposed device steps **were not executed** by AutoDebug.
Use explicit full policy paths; `run_readiness` does not currently expose a p0
CLI filter, so a new flag is not required merely for this one diagnostic probe.

For each result record the full policy, allocated head/cache dtypes, actual
compute fields, cache capacity, page-table content/shape, history capacity,
resolved full-attention k/grid, prompt IDs/rendering, generation knobs, source
hashes, trace/capture counters, full output tokens, first EOS, and final answer.
Keep the pinned HF control unchanged; do not add a special haiku system prompt,
postprocess a corrected line, or treat a new answer as continuation of old tokens.

If isolated selected p0 does not reproduce, rerun selected under the original
readiness order or reproduce its prior requests before concluding it improved.
If either repeat differs at fixed geometry, preserve both runs and inspect
same-config first-divergence logits/state before changing precision.

If selected is stable and baseline restores 5/7/5, this verifies a **precision-
sensitive qualitative regression at this prompt**, not a native LoFi arithmetic
bug. If BFP4/HiFi2 also restores it, head fidelity is a sufficient intervention
for this output; if it does not, BFP8 weights or the interaction remain candidates.
Neither result licenses a claim that every prompt improves or changing the
user's fastest-top-k-passing selection rule without reconciling the stage review.

For stronger localization only when needed, force the common prefix through the
first changed token (currently generated index 31), retaining capacity 1088 and
history geometry. Compare BF16 full logits/top ranks from the captured model
output, then final hidden states or a CPU/reference head on the same hidden
state. Fixed decoder policy and identical prior tokens should isolate the head
before free-running histories diverge. A CPU argmax check must allow documented
equal-max tie behavior. Do not compare arbitrary later tokens from different
branches and call the difference a head error.

If cache shape is implicated, vary only preallocated capacity at fixed G1024
(for example 1088 versus 1152, which changes k64 to k128) with the same BFP8 KV
policy. That probes shape sensitivity without changing the completion budget.
The earlier short-cache branch can separately be probed at G256 with preallocated
320 versus 1088 and equal history buffers. These are follow-ups, not an initial
KV dtype sweep or grounds for changing cache allocation now.

## Evidence identities and remaining work

```text
datatype_sweep/tt_qualitative_extended.json
  174256c4ff40cde8bbfbad6069f3184b71422a23b84387a2fc3ebf2c557f7809
optimized_full_model/tt_qualitative_extended.json
  bd26d7f96b52ad76be6fc9f5cdf77e06bdd96056d011793664e70a3f8b13d990
full_model/hf_qualitative_extended.json
  61221a3a42f6d2ae431a13fd583c45c3dacc957e0aad20fc5cf6af938465fe39
datatype_sweep/selected_precision_config.json
  33a8fa95c118a85ceb53e6ba707ae4d6eb74631263d5120c1842afa6605628d1
```

Host checks performed: repository/source reads, `git status --short`, working-tree
diff review, source-snapshot byte equality, SHA256 computation, JSON inspection,
and exact saved prompt/token comparisons using only Python standard-library
`pathlib`, `json`, and `hashlib`. This is a docs-only report; no build was needed.

No implementation fix is proposed before the controls. The haiku finding remains
open for AutoFix experiment handling and independent stage review. If the
controls establish stable, precision-sensitive ordinary model error without a
runtime defect, record that limitation explicitly alongside the passing numeric
gate; do not describe the observed six-prompt suite as wholly instruction-correct.
