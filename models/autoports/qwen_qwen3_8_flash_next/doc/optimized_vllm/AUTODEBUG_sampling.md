# AutoDebug: four final vLLM sampling failures

**Leading diagnosis: admission-dependent decode reproducibility, with a concrete singleton-versus-pair arithmetic difference to test. No host/device sampler switch occurs in the reconstructed failing windows. No corrupt state transfer or RNG defect is proven by inspection.** Do not call this harmless numerical drift until the same-history logits are compared.

Inspection only; no requests, hardware operations, source edits, plugin edits, or subagents. Only this report was written. The model `tt/` tree matches HEAD `ff80569624d60dfdce236aee2cb432e849b3515e`; plugin is clean at `629f07cb385494b14d0ee3d138de039d71d09546`.

Path abbreviations below:

- `M` = `models/autoports/qwen_qwen3_8_flash_next`; `T` = `M/tt`.
- `S` = `M/readiness_vllm/stage10_final_acceptance/server/server.log`.
- `F` = `M/readiness_vllm/stage10_final_acceptance/checks/sampling/sampling_tests.log`.
- `S9` = `M/readiness_vllm/acceptance_tracking_v3/server.log`.
- `P` = `/home/ttuser/qwen38-flash/vllm-tt-plugin`; plugin source references are relative to `P`.

## Observations and exact scope

`F:153–557` contains four output-equality failures; `F:586` reports **4 failed, 63 passed, 75 skipped**, with two additional reproducibility warnings. The tests use legacy completions, not chat templates (`P/tests/tt/utils.py:55–93,136–188`). All return strings, so reported divergence indices in these four assertions are **word-ish units, not token indices** (`utils.py:288–319`). In particular, top1's “index 0” does not establish a bad prefill/first token. Its alternatives share leading zero characters.

| Test | Actual workload and failed comparison |
|---|---|
| `mixed_params_batch` | `max_batch_size=2` truncates ten listed configs to **only** `Count: `, greedy, 5 tokens; and `Random: `, temperature 2, k32, seed42, 5 tokens. No penalties or host-only companion. Seed42 returns `2x105` versus `2x10 =` (`F:159–261`). Assertion index1 means the two-element shuffle retained config order; network admission order is not thereby proven. |
| `same_seeds_reproduce_across_batches` | `Generate: `, temperature1, k50, 15 tokens, seeds100/101; second config order is 101/100. The reported failing pair is `15 mcq, 15 fill in blanks with options, ` versus `15 mcq, 15 t/f, 10 short`. The traceback does not identify which seed's loop iteration failed. A disagreeing pair has no strict majority (`F:267–399`). |
| `specific_seed_reproducible[0]` | `Text 0:`/`Text 1:`, temperature1, k50, 10 tokens, seeds0/1000, same submitted order twice. Seed0 differs after the common text beginning ` {'text': '\n` (`F:405–443`). Seeds42/123/999 pass. |
| `top1_is_greedy` | Two runs of **four requests**, despite server capacity2: `Number: `, 10 tokens; one temperature0, three temperature2/k1. All first-run outputs agree. Second-run submitted indices1/3 differ; concatenated assertion indices5/7 are **not physical slots**. Majority `000000491 Category`; alternatives `000000691 Category` and `0000\n\nName:\n\n## Description` (`F:450–555`). |

Server is full48, B2, context262144, async, budget9, alignment8, prefill/decode interleave1:2, device sampling mode `all` (`S:14,44,173`). These prompts have logical lengths3/4; every relevant prefill starts at0 and samples a final row. Internal length3 processing uses chunks2+1; length4 uses one chunk4. They do not exercise scheduler partial-prefill relocation.

## Admission and route ledger

Notation: `P0`, `P1`, `P01` identify physical slots in a prefill submission; `D×n` counts subsequent decode submissions until the next prefill. Counts include scheduler overlap; they are not per-request token indices. `|` separates test invocations. All prefill sample masks below are true.

| Test | Reconstructed final server sequence | Route, counts, lines |
|---|---|---|
| Mixed | `P0 D×2 P1 D×4 | P01 D×4` | Entirely **device**: 3 prefills, 10 decodes; `S:4619–4659`. First run changes one→two→one active rows; second stays paired. |
| Same seeds | `P0 D×2 P1 D×14 | P0 D×2 P1 D×14` | Entirely **host**: 4 prefills, 32 decodes; `S:4710–4763`. Config order reverses, but the log lacks seed-to-slot identity. |
| Seed0 | `P0 D×2 P1 D×9 | P01 D×9` | Entirely **host**: 3 prefills, 20 decodes; `S:4876–4910`. |
| Top1 | `P01 D×9 P01 D×9 | P0 D×2 P1 D×8 P1 D×2 P1 D×9` | Entirely **device**: 6 prefills, 39 decodes; `S:10915–11036`. First run is two paired waves; second has staggered admission, completion/compaction and reuse. |

Windows are reconstructed from ordered pytest cases, JUnit durations, monotonic event times, prompt lengths, output counts, and adjacent workload transitions. `checks/sampling.xml` starts at 2026-11-17 13:12:07 UTC. The final server did **not** enable adapter trace diagnostics: events omit request IDs, tokens, submitted positions, active decode slots and seeds. Thus the per-forward sequence above is observable, but a complete per-request sequence—especially matching seed100 to a physical slot, or assigning top1 alternatives to slot histories—is unavailable. `active_decode_slots` on prefill is retained-cache metadata, not an authoritative live-request roster. Do not invent either mapping.

All host decodes in these windows reload inputs; device decodes reload at transitions and otherwise retain device feedback. No new capture occurs inside these four windows.

The plugin makes one sampling decision per submitted forward: any stochastic row outside `1..max_device_top_k` selects host (`src/vllm_tt_plugin/model_runner.py:4028–4085`). The model bound is40 (`T/generator.py:50–57`); the adapter directly translates presence of `sampling_params` into device mode (`T/generator_vllm.py:151–178`). Hence k50 is host from the **first prefill**, while the mixed test's k32 and top1 stay device. The known per-step route-switch defect is a valid general concern, but its necessary condition is absent here.

## Passing stage9 contrast and source mechanism

`M/readiness_vllm/sampling_tests.full_budget1528_v1.log:113–153` records all four passing and **67 passed/75 skipped**. Same plugin commit and B2; `acceptance_tracking_v3/vllm_run_summary.json` records budget256 without interleave1:2. The actual corresponding windows show:

- Mixed: `P01 D×4 | P0 P1 D×4`, `S9:5330–5372`.
- Same seeds: `P0 P1 D×14 | P0 P1 D×14`, `S9:5420–5469`.
- Seed0: `P0 P1 D×9 | P0 P1 D×9`, `S9:5567–5601`.
- Top1: `P0 P1 D×9 P01 D×9 | P01 D×9 P01 D×9`, `S9:10551–10663`.

Stage9 diagnostic decode events explicitly show active slots `[0,1]` throughout those windows. **Separate admission alone is not the difference; singleton decode between admissions is.** Current scheduler interleave deliberately inserts those decode steps (`P/src/vllm_tt_plugin/scheduler.py:189–203,893–908`). For length3/4 prompts, budget9 is not itself forcing partial chunks.

The concrete model path is `QwenModel → MultichipDecoder → MeshMoe`, not the similarly named single-chip `OptimizedSparseMoe` (`T/model.py:146`; `T/multichip_decoder.py:1006`). `FactoredDecode` packs active rows into bucket1 or bucket2, calls that MLP, then scatters results (`T/factored_decode.py:95–106,153–167`). Their reduction arithmetic differs:

- **One row:** `MeshMoe._reduce_single`, `T/multichip_decoder.py:934–946`, masks nonowned expert values and uses a weighted matmul with FP32 destination accumulation inherited from `WeightLoader` (`T/decoder_components.py:33–43`), then BF16 output.
- **Two rows:** `_reduce`, lines893–932, pads reduction inputs to32 rows and invokes native `post_combine_reduce`. Concrete logical combine shape is `[1,1,32,10,2560]`: 32 token rows, 10 expert entries, three 1024-element CB pages per expert row. Its factory sets `fp32_dest_acc_en=false` (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:39–63,327–331`); compute repeatedly packs expert products into BF16 L1 accumulation (`.../kernels/deepseek_moe_post_combine_reduce_compute.cpp:80–147`).

This establishes **different rounding paths**, not a low-level contract violation. A request traversing different singleton/pair histories can acquire different recurrent state and logits before its output diverges. It predicts a shared failure mechanism across host sampling, device sampling and greedy top1, unlike a device RNG-only explanation. Seed999 also passes a staggered-versus-paired contrast (`S:4841–4875`), so admission changes are not sufficient to cause every seed to fail; the differing distribution must cross that seed's selection boundary or change an argmax. Actual logit deltas/margins are unmeasured.

Stage9 is not a one-variable experiment. Since its sampling checkpoint `fe932db2`, model-tree changes are in `generator.py`, `slot_state.py`, and new `pending_prefill.py`; shared/native sampling also gained wider candidates. MoE/factored arithmetic is unchanged. Traced state copies and sampler configuration changes therefore remain alternative explanations. Current source being unchanged since `ff805696` does not make it identical to stage9.

## Competing explanations and adjudication

| Candidate | Evidence and remaining uncertainty | Smallest discriminator / intervention if confirmed |
|---|---|---|
| Occupancy-dependent model numerics | Strongest sequence-matched lead above; explains both sampler routes and top1. No identical-history logits saved at failure. | Compare bucket1/bucket2 logits and first differing layer; unify the offending arithmetic only if confirmed. Do not change scheduler policy merely to make tests pass. |
| Adapter state/async transition defect | Staggered runs exercise prefill during decode and slot reuse. No contradictory copy/remap found. New traced scratch copies merit a control. | Replay the same token/position/page/slot history standalone versus serving; compare synchronous versus queued execution, then eager versus traced state copies if necessary. |
| RNG ownership or parameter reset | Could explain sampled failures, cannot alone explain correct k1 on identical logits. Host generators are request-owned and seed0 is accepted explicitly (`input_batch.py:155–160`); re-keying preserves objects (`model_runner.py:1423–1450`). Device seeded counter starts at prompt length and advances only active rows (`T/generator.py:981–985`; `T/device_seed.py:83–92`). | Identical logits plus different draws require comparing request seed/counter or host generator state, candidate IDs and effective parameters before changing RNG code. |
| Host/device switching | No switch in these windows; k50 never starts device. | No plugin routing change justified for these failures. Plugin stays pinned. |

Async boundaries were inspected: model/copy/sampling traces enqueue on CQ0; decode readback enqueues a nonblocking CPU copy followed by an event (`T/generator.py:1357–1366`). Subsequent same-CQ writes are ordered after that read; plugin finalization waits the event before consuming host data (`src/vllm_tt_plugin/async_decode.py:1653–1663`). Prefill token/logit reads synchronize. A submission log is not completion evidence, but no missing wait is demonstrated in this path.

Other potential issues/test gaps: pending-prefill relocation cannot explain these starts-at0 windows (`T/pending_prefill.py:18–20`). Wider k40 sampling is not selected by the two device failures: active k≤32 selects width32, inactive rows are neutralized, and capture keys include candidate width (`T/generator.py:375–392,615–623`). Shared narrow-kernel changes still need fixed-logit validation if raw logits match. Greedy tie handling already targets both temperature0 and k1 (`models/common/sampling/generator.py:856–863`; `tt_sampling.py:963–971,989–1011`); its documented >32 equal-maxima-per-shard limitation is unproven here. Do not promote it without candidate data.

## Smallest discriminating follow-up, after the root's live gates

1. **Start with `Number: `, full48, selected policy, capacity2.** Save exact prompt/output token IDs, then compare prefill logits and up to nine teacher-forced decode steps. Force the **same input prefix** in every arm: singleton slot0, singleton slot1, paired duplicate rows, and staggered `P0 → D×2 → P1`, with two repeats and reversed row order. Use the served cache/layout contract. Save full-vocabulary logits before sampling, active slots/bucket, positions, and page identities. Record exact equality, maximum absolute delta, argmax IDs, top-two margins and first differing step. This removes stochastic selection and divergent histories as confounders. Stop at the first reproducible discrepancy; only then localize a layer/reduction.
2. **Repeat through actual vLLM with the final launch configuration.** First collect a small host-logprob comparison at the same prompt/prefixes, with real admission/slot evidence. Then cover multi-step decode: a one-token request containing the whole prefix tests prefill, not the decode path. For decisive device-path evidence, use a temporary model-side diagnostic tap that snapshots pre-sampling logits before shared-buffer reuse, preserves the original route, and records completion events plus token/position/slot metadata. No plugin edit is needed. Adding `logprobs` to a device request selects host and is therefore only a labeled control, not proof about the failing device path. A readback can alter overlap; preserve a queued control if the synchronized diagnostic passes.
3. If needed, extend the same short check to `Random: ` with `Count: `, and host k50 `Generate: ` seeds100/101 and `Text 0:`/`Text 1:` seeds0/1000. Compare each history only until its first true token divergence. With identical logits, freeze the candidate row and inspect seeded draws/greedy selection; with standalone equality but served inequality, investigate adapter/plugin state ordering instead of retuning precision.

Existing `tests/probe_vllm_logit_baseline.py` and `tests/check_vllm_logit_determinism.py` provide useful scaffolding but their saved stage9 logit comparisons cover **prefill next-token logits** on different 33/47-token prompts. Served checks use `max_tokens=1`, selected top20 host logprobs; they cannot clear these steady-decode failures. Do not overwrite those passing artifacts or relax the four assertions.

The installed [vllm-integration skill](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/vllm-integration/SKILL.md) classifies reproducibility-only failures separately **only when correctness, logprobs, crash-free serving and qualitative output pass**, and requires standalone plus served logit reproducibility checks. This suite's exercised logprob tests pass, but that alone does not establish every required gate. Preserve **4 failed** as the raw pytest outcome; carry these as reproducibility investigations pending the root's independent gates and the discriminating evidence above, not as proven routing defects or an unconditional acceptance pass.
