# Findings — driving PR #46283's pipeline on Voxtral-TTS

Running log of every issue hit while driving the PR's `bring-up → emit-e2e → optimize` pipeline end
to end, across two experiments, and what was done about it. Written so each entry stands alone:
symptom first (what a user actually sees), then the real cause, then the fix, then how it was
verified.

**Why this file exists.** The failures have not announced themselves — each surfaced several layers
downstream of its cause, wearing a diagnosis that pointed at the wrong thing. A log of "what it
said" next to "what it was" is the useful artifact.

---

## READ THIS FIRST

Two experiments against the same hand-built TTNN implementation, on one Blackhole p150b:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| model | Block 1 only (3.4B backbone, stock `MistralForCausalLM`) | **full three-block Voxtral-TTS** (4.0B: backbone + flow + codec) |
| bring-up | 5/5 components, ~17 min of real work | **7/7 components, 114/114 ops on device, one round, 42 min** |
| e2e | — | **PCC 0.9999834**, exact audio-code match, **0 code flips** |
| optimize | ran unattended and worked | 22 attempts, 11 wins, **−15.5%** — see F46 |

**The port is correct, and it is 9.6× slower than the hand-built one.** Measured like for like on
the shipped path: **258.26 ms per audio frame against 26.9** — RTF 3.319 against 0.357, i.e. the
generated port cannot keep up with real-time speech and the hand-built one does with margin. The gap
is structural (no KV cache, eager dispatch), not a tuning deficit. See THE COMPARISON below.

Correctness itself is solid and was measured by running the tests directly, not by reading the
tool's status files. Bring-up in particular is genuinely impressive: a 4B three-stack
model, no per-model code, entirely on device in one round.

**The optimizer's headline number does not describe the delivered model** — it reported −17.2%,
measured on a decode path the demo never calls, while the shipped path improved **−13.2%** (297.69 →
258.26 ms/frame, measured directly). The gains are real and reach the product through shared code;
the measurement simply is not of the product. That is F46, and it is the first thing to read.

### The five that matter most

| # | one line |
|---|---|
| **F46** | the optimizer measures `decode_step`, which the demo never calls, while the shipped loop recomputes the whole prefix every frame — it reported −17.2%, the product got −13.2% |
| **F42** | the correctness gate returned `pcc: 33.612, pcc_verified: true` — no range check on a value that cannot exceed 1.0 |
| **F36 + F37** | the graduation gate feeds `torch.randn` to components whose real captured inputs sit unread on disk, and its defaults can silently corrupt the golden |
| **F34** | the overlay store restores a model deleted from HEAD, so a from-scratch run is unreachable and two runs from one commit differ invisibly |
| **F29** | the CLI's `--pcc-target` default of 0.95 overrides the engine's documented 0.99 — measured here as the difference between a 0.9586 and a 0.9986 port |

Everything else is indexed in the OWNERSHIP tables below. Entries marked **OURS** (S1–S7) are our
own setup problems, recorded so they are never mistaken for tool defects.

---

## Experiment 1 — Block 1

Port Voxtral-TTS **Block 1** (the 3.4B autoregressive backbone) with the tool, blind, and compare
against a hand-built TTNN implementation of the same block that has been through 74 recorded
optimization experiments.

| | |
|---|---|
| model given to the tool | `/localdev/lserbedzija/hf_models/voxtral-tts-backbone` |
| what it is | Block 1 exported to HF format — stock `MistralForCausalLM`, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RoPE θ=1e6 |
| provenance | `scripts/export_backbone_hf.py` from the hand-port's own history, recovered at `a4b9382c96` |
| verified | PCC **1.00000298** vs the fp32 CPU reference, max\|diff\| 2.6e-04 on rms 2.59 |
| contains | weights + `config.json` only — **no TTNN code, no tuning, no findings** |
| blindness | run from a worktree of `pr46283`, which has no `voxtral_tts` directory at all |
| hardware | one Blackhole p150b, `--box P150 --mesh 1,1` |
| target to beat | **~15.9 ms per decode step**, batch 1 (the hand-port's Block 1, measured device-bound: eager 15.907 / traced 15.922) |

---

### Experiment 1 result — the port is CORRECT; the tool could not tell

Bring-up **succeeded**. Verified by running the generated tests directly on the P150, not by
reading the tool's status files:

```
[bringup] achieved PCC=0.9998358  target=0.99  component=attention
[bringup] achieved PCC=0.9999971  target=0.99  component=decoder_layer
[bringup] achieved PCC=0.9999944  target=0.99  component=m_l_p
[bringup] achieved PCC=0.9999874  target=0.99  component=r_m_s_norm
[bringup] achieved PCC=0.9999992  target=0.99  component=rotary_embedding
                                                        pytest exit=0, 5/5
```

All five components were finished by **07:22 on 2026-08-13**, about 17 minutes into the run. The
loop then ran a further ~1.5 h re-doing completed work, reporting `graduated 0`, and would not
have stopped on its own. See F6: the agent had no way to report what it had done.

---

## ★ FOR THE HAND-PORT — what to actually do, ranked

The point of the experiment. Everything below is derived from a tool that never saw the hand-port
(see the blindness audit). Full detail in the O4* entries; this is the short list.

### Take these

| # | change | evidence | risk |
|---|---|---|---|
| **1** | **Rotate Q+K in ONE call** — slice q+k out together, split heads once into a 40-head tensor, rotate once, slice apart. NOT ttnn's fused q+k rope operator (§6.23 rejected that correctly, for the interleaved convention); the *same* `rotary_embedding_hf` on a wider tensor. | exact arithmetic, PCC bit-identical, −1.99% on the identical model. 52 rotary launches/token → 26; `[gpt-24]` priced a comparable 26-launch saving at 0.405 ms | **check first:** decode mode wants cos/sin sharded to match, and §6.44 documents the trap (*"RoPE on a core whose cos/sin table lives elsewhere returns 3.4e38"*) |
| **2** | **Give `_PRG_W2` its own grid and re-sweep it.** One `_MM_GRID` for all five decode matmuls cannot be right for both K-light projections and the deepest K-reduction in the model. | measured curve on the identical op: `96c 0.1589 / 48c 0.0960 / 32c 0.1137 / **24c 0.0934** / 16c 0.1084`, a 1.70× spread. `_PRG_W2` sits at 72c. | none — precision-neutral. Sweep and keep what wins |
| **3** | **Audit the decode path for the batch-1 / seq-1 pathology** (see ★ THE PATTERN). Four instances in one model. | argmax 32× the bytes; rotation 160 tiles where 4 had data; head creation collapsed to one core; a join moving 8 MB where 64 KB was real | none — these are pure waste |
| **4** | **Keep the decode residual stream in ONE shard, the whole way down.** `_norm` returns `sharded_norm(..., _L1)` — it shards internally, converts back to **L1 interleaved** on the way out, and `ttnn.linear` converts straight back in. The tool went further than the one handoff: both norms in a block and the next layer's are built on the same dim, so the stream *never has to leave the shard* across all 26 layers. | 0.80 ms of layout ops removed on the QKV handoff alone (`sharded→interleaved` 1152→896 calls, `interleaved→sharded` 1024→768), then a further −0.11 ms/token carrying it through the residual adds | **the catch:** the chain only exists where the grids agree. Norm is `(8,4)`=32, `_MM_GRID` is `(12,6)`=72. The tool paid 3.72→3.98 ms moving its norm 32→48 to meet the projection. Do this **together with #2** — pick `wqkv`'s re-swept grid to be one the norm can share. Note `[gpt-27]` (residual as matmul bias) already removes the *add* launches; this is about the *layout* round-trip, and the two compose |

### Re-open these, don't assume they're settled

| # | what | why |
|---|---|---|
| **4** | **§6.8 — device argmax rejected in favour of host.** | The A/B was scored against a **single-core** kernel that didn't have to be single-core. `ttnn.argmax` picks its path from input LAYOUT; a TILE input single-cores the scan *and* pads the row 32×. Re-run with `to_layout(ROW_MAJOR)` in front. Host may still win on a 33 KB reduce that already ends in a D→H copy — but the number you have doesn't answer that. |
| **5** | **§6.44 — fused K/V cache write deleted as 0.687 ms/step slower.** | The tool measures it **faster** on the same board: 0.402 → 0.210 ms/token, moving V one core exactly as `[gpt-24]` did. The layouts are materially similar — you already call `nlp_create_qkv_heads_decode` on a sharded operand — so I **cannot** explain the disagreement, and my first attempt to (a layout you hadn't adopted) was wrong. An unexplained 0.19 ms/token swing is still worth one A/B. |
| **6** | **`_MM_GRID` generally.** | Tuned at §6.52, then §6.65/§6.67/§6.72 changed the structure around it. A structural change invalidates earlier knobs silently (O4m: a stale cap cost 15% on one op). |

### Explicitly do NOT take

- **`down_proj`/`w2` → bf8_b.** This is §6.16, which you measured and declined. The tool takes it only because a 0.95 gate has no reason not to. Your call stands.
- **DRAM-sharding the LM head weight.** Real bandwidth insight (interleaved buffers round-robin across all 8 banks; the head ran at 256 GB/s where projections manage 340–360), but it costs a **second 226 MB copy**, and Block 1's LM head isn't on your critical path.

### What the experiment confirms about work you already did

Six independent agreements, each reached without sight of your code: the width-sharded decode norm
(§6.67), the traced decode loop (§6.65), decode matmul program configs (§6.52), the fused `wqkv`,
the hand-rolled head split over the fused op (§6.72 — *"dispatches fell 3413 → 2867 yet it got
slower; these were view ops doing no work"*), 2 cores/head on the decode SDPA (`[gpt-21]`), and that
`activation=` never fuses while `fused_activation` does (`[gpt-26]`).

**And one thing the tool structurally cannot check:** `[gpt-21]` records SDPA settings that were
faster but *"NOT SAFE — position sweep"*. The tool gates on PCC at a single length. Nothing in it
would have caught that.

---

## ★★★★ THE COMPARISON — measured, like for like

The experiment's actual question: how does a tool-generated port compare with a hand-built one?
Answered here by measuring both the same way — steady state, per audio frame, after warm-up — rather
than by quoting the tool's own headline, which F43/F44/F45/F46 establish does not describe the
delivered model.

**Method.** The optimize run was stopped after ~17.5 h and 11 banked commits. Its in-flight
unbanked work was stashed, so the tree measured is exactly the 11 wins. `run_tts` — the shipped path,
the one `demo_tts.py` calls — was run at the demo's own settings (real prompt, `--max-frames 24`,
`early_stop` at its default), once to warm up and once timed. `run_tts` already returns
`{prefill_s, decode_s, codec_s}`, so no new instrumentation was introduced.

```
frames generated : 24
prefill          :   142.2 ms   (once)
decode           :  6198.2 ms   total   ->  258.26 ms PER FRAME
codec            :    31.6 ms   (once)
total utterance  :  6371.9 ms   for 1.92 s of audio
RTF              :     3.319
```

### Result

| | tool-generated port | hand-built port (§6.72) |
|---|---|---|
| per audio frame | **258.26 ms** | **26.9 ms** |
| RTF (compute ÷ audio) | **3.319** | **0.357** |
| against real time (80 ms/frame) | **3.2× slower** | **2.8× faster** |
| execution mode | eager, full re-prefill per frame | traced (trace+1cq), incremental |

**9.6× slower**, and on the wrong side of real time: the generated port cannot sustain speech, the
hand-built one does with margin.

### The gap is structural, not tuning

Three separable causes, in the order they were established:

1. **No KV cache in the shipped path (F46).** `run_tts` re-runs all 26 layers over the whole padded
   224-token sequence every frame and keeps one row. An incremental, cache-resident `decode_step`
   exists in the same file and is never called from the demo.
2. **Eager dispatch (F45).** `execute_trace` appears once in the entire pipeline, inside the
   capture selftest. The delivered generation loop replays no trace, so every op is dispatched from
   the host, every frame.
3. **Tuning.** The 11 optimisations are real, correctness-preserving, **and they do reach the
   shipped path** — see the before/after below. They are already in the 258.26 ms.

Only the third is the kind of gap more optimisation closes. The first two are structural choices the
generated port made and its gates never questioned.

### And the proportions confirm F44 with a hard number

The optimizer's objective timed **one prefill against one decode**, weighting prefill at roughly
half the pair. The real utterance:

```
decode  6198.2 ms   97.3%
prefill  142.2 ms    2.2%
codec     31.6 ms    0.5%
```

**Decode is 97.3% of the work and entered the objective once.** Prefill is 2.2% and carried
comparable weight. That is the whole of F44 in three lines: an edit that helped prefill scored, an
edit that helped decode barely moved the number, and the ranking that followed was not a ranking of
what this model actually spends its time on.

### What this does NOT say

- **Not that the port is wrong.** It is correct: `e2e PCC 0.9999803900718689`, exact audio-code
  equality, zero code flips, measured directly on this same tree.
- **Not that bring-up failed.** Bring-up is the strong half of this tool: a 4B three-stack model,
  no per-model code, 7/7 components and 114/114 operations on device, in one round, in 42 minutes.
### MEASURED: the optimisations DO reach the shipped path — a correction

An earlier draft of this document said the 11 commits improve only `decode_step` and therefore
"none of it reaches a user". **That was wrong, and the error mattered.** Most of the commits land in
`tt_common.py` (6) and `tt_backbone.py` (3) — the helper module and the attention/MLP bodies that
**both** paths execute — plus `_stubs/flow_matching.py` and `_stubs/codec_decoder.py`, which the demo
runs directly. Only the changes specific to `decode_step`'s own structure are stranded.

The shipped path was measured at the pre-optimise commit (`51e208f40c`) and at the optimised tree,
same script, same settings:

| | before 11 commits | after | change |
|---|---|---|---|
| **per frame** | **297.69 ms** | **258.26 ms** | **−13.2%** |
| prefill | 156.4 ms | 142.2 ms | −9.1% |
| codec | 37.1 ms | 31.6 ms | −14.8% |
| total utterance | 7337.9 ms | 6371.9 ms | −966 ms |
| RTF | 3.822 | 3.319 | |

**So the work is not stranded and it did not pessimise the shipped path** — the concern that tuning
for `M=32` might hurt a path running `M=224` is not borne out in aggregate.

What remains true, and is the accurate form of the criticism: **the optimizer reported 17.2% and the
product got 13.2%.** The metric is optimistic by four points, not fictional. The gains arrive by
side effect — through shared code — rather than because anything measured the thing being shipped.
A change that helped `decode_step` and hurt `run_tts` would still be banked, and nothing in the
pipeline would notice; it simply did not happen this time.

- **Not that the optimisation work is wasted.** It reached the product, as measured above. Wiring
  `decode_step` into `run_tts` would additionally recover the structural gap (F46).

The honest summary is narrow and specific: **the tool ports correctly, and measures its own
performance on a code path the user never runs — so its headline (−17.2%) overstates what the
product received (−13.2%).**

---

## OWNERSHIP — who should fix what

Split by owner so the PR feedback can be lifted straight out of this file. **Only the first table
is for the PR author.** The second is our own setup, recorded so it is never mistaken for a tool
defect.

### A. TOOL DEFECTS — for the PR author

| # | one line | fix? | effort |
|---|---|---|---|
| **F47** | "host-free" is certified by `host_op_selftest` with `early_stop=False`, while the demo takes the `early_stop=True` default and reads one value back to host every frame; matters once F46 lands | **YES** | small |
| **F45** | `E2E_REQUIRE_TRACE` is satisfied by proving each stage CAN be captured; `run_tts` — the shipped generation path — never calls `execute_trace`, so an eager pipeline ships (**measured 258.26 ms/frame**) while perf is quoted from trace replay | **YES — first, with F42** | medium |
| **F44** | the optimize objective times a capture-and-verify harness (eager pass + trace capture + 2 readbacks + release + host PCC, ONE decode step) rather than inference — so capture cost is weighted 4×/iteration and the 24-frame decode that dominates deployment enters once | **YES — with F42/F43** | medium |
| **F43** | `TRACE_PER_TOKEN_MS` is the per-CALL time, not per token — inflated by OSL (4×) plus prefill; `per_token_ms == forward_wall_ms` in the ledger is the signature, and `tokens_per_sec` inherits it | **YES — with F42** | one line |
| **F46** | the profiled decode (`decode_step`, incremental KV, O(1)/frame) and the shipped decode (`run_tts`, full re-prefill of the whole prefix every frame, O(n)/frame) are different algorithms — `decode_step` is called only from the trace harness, never from the demo | **YES — first** | medium |
| **F42** | the correctness gate returned `pcc: 33.612, pcc_verified: true` against threshold 0.99 — no range check anywhere, and the pytest exit code is deliberately ignored, so the regex scrape is the only signal | **YES — first, one line** | trivial |
| **F6** | `mcp` is **declared nowhere** → all 10 agent tools silently absent → 11 h stall | **YES — first** | trivial |
| **F2** | the READY verdict can never fire (`lambda _: []`); best compat report = surest failure | **YES** | one line |
| **F1** | a local model dir gets a reduced probe → refused 3 stages later, wrong diagnosis | **YES** | small |
| **F7** | one reporting channel, no cross-check against disk, unbounded retries | **YES — highest value** | medium |
| **F3** | a Python exception reported as "the PCC gate rejected the output" | **YES** | small |
| **F9** | a local model dir is mistaken for a demo dir (`optimize` unusable by model id) | **YES** | small |
| **F8** | the in-place refusal returns `rc=1`, so the supervisor resets the card 3x | **YES** | one line |
| **F10** | the F9 workaround loses the model id → `optimize` cannot build its own PCC gate, though it wrote one | **YES** | small |
| **F5** | systemic-pattern detector counts error NAMES, not families | **YES — low prio** | small |
| **F4** | deliberately-wrong constructor, repaired on retry | **NO — works as designed** | — |
| **F11** | documented `--max-rounds` default is 20; the real one is 3, and it is the ONLY exit | **YES** | one line |
| **F12** | the fusion rung reaches for a grid where it should reach for a program config | **YES** | medium |
| **F13** | generated stubs swallow fast-path exceptions, so a perf regression passes the PCC gate | **YES** | small |
| **F14** | "producer emits the consumer's shard" must check the consumer's PROGRAM CONFIG grid | **YES** | medium |
| **F15** | `plan` and `compat` disagree about what the model IS | **YES** | small |
| **F16** | the block table degrades to EMPTY, which reads identically to "nothing needed" | **YES** | small |
| **F17** | machine-readable structure is declared and never read | **YES** | small |
| **F18** | the architecture gate tests the model's NAME, not its structure | **YES — fix tested** | small |
| **F19** | template dispatch silently runs a DIFFERENT model; the template can be the tool's own prior output | **YES** | medium |
| **F20** | ⚠ REVISED — the meta-plan is wired to stdout, not control flow; on this run the pipeline ignored it and was RIGHT | **partial** | — |
| **F21** | `trust_remote_code` is a ONE-MODEL allowlist; the two halves of the pipeline disagree | **YES** | small |
| **F22** | the isolation worktree silently ignores uncommitted edits to the tool's own source | **YES** | small |
| **F23** | ⚠ CORRECTED — capture drivers guess where the config already says they should not; 3 of the 4 misses were OURS (S5) | **partial** | small |
| **F25** | decomposition children lose the parent path prefix, and the plan is copied from another model | **YES** | small |
| **F26** | report what the gate MEASURED, not what was collected (`captured 7/7` ≠ used) | **YES** | small |
| **F27** | the captured input is DISCARDED where one `deepcopy` would have kept it | **YES** | one line |
| **F28** | the entire end-to-end verdict rests on ONE prompt (n=1) | **YES** | small |
| **F29** | the CLI's 0.95 `--pcc-target` overrides the engine's documented 0.99 — the threshold SETS quality, it does not merely gate it | **YES — highest value of the three-block run** | one line |
| **F30** | the drift gate detects the stale template and is wired never to block (*"Never raises"*) | **YES** | small |
| **F31** | a profiled child that aborted (SIGBUS) is reported as a missing CSV | **YES** | small |
| **F32** | `termination_check()` blocks 30 min with no progress channel; the retry never returns | **YES** | medium |
| **F33** | `worktree-list` can never print ORPHAN (`id(s) in orphans`), so dead worktrees accumulate looking active; `PermissionError` is also misread as dead | **YES** | one line |
| **F34** | the overlay store silently restores a deleted model over a clean HEAD, so a from-scratch run is unreachable and two runs from one commit differ invisibly; `overlay-drop` also fails to empty its scope | **YES — reproducibility** | small |
| **F35** | backend selection is non-deterministic — identical runs picked different templates, the LLM ranker overriding its own top score, choosing between two entries whose paths are both missing | **YES — reproducibility** | small |
| **F36** | "PCC tests will use real inputs" is false — the graduation gate builds inputs with `torch.randn` and never loads the 43 MB captures or the captured `output.pt`; raising the threshold measures the wrong thing more precisely | **YES — the sharpest one** | small |
| **F37** | the generated test calls `_captured_submodule_path()`, defined in 0 of 7 emitted files → guaranteed first-round `NameError` for every model; and its input defaults drop the causal mask (non-causal golden) and stage index tensors as bf16 (8191→8192) | **YES — with F36** | small |
| **F38** | `optimize --devices` defaults to `"0,1"`, so a 1-chip box is planned as 2-chip TP=2 — overriding an explicit `--mesh 1,1` and ignoring the `parallelism_manifest.json` the tool wrote itself | **YES** | one line |
| **F39** | the e2e report prints `Verdict: PASS` beside `e2e PCC n/a` — verdict and measurement come from different sources and only one survived the cc-engine refactor (real value was 0.99998) | **YES** | small |
| **F40** | the baseline measurement completes (`756.4513 ms`, 8 iters, 4 stages PCC-clean) and is then discarded by a segfault in `close_mesh_device` at teardown; the run reports a missing CSV and optimizes on with no baseline | **YES — with F31** | small |
| **F41** | the depth-knob sanity check compares op sequences truncated at 50000 and reads the shared limit as "cap never reached the builder"; the knob is also only backbone-deep while its comment claims every stack | **YES** | small |

**If only one thing is taken: F46.** The optimize stage spent twelve hours improving a decode
function the product never calls (its gains reach the product only via shared code), while the shipped generation loop recomputes the entire prompt
plus every prior frame, for every frame. No perf number in this report describes the delivered
model. F42 is the next one, and is a one-line fix.

**On the correctness half: F42.** One line of range validation. Without it the correctness gate
can return `pcc_verified: true` on a number that is not a correlation coefficient, and every perf
win the tool banks rests on that gate. It was found only because the impossible value happened to be
read by a human; nothing in the system objected.

**If only one thing is taken for BRING-UP: F6.** It is the difference between "this tool does not
work" and "this tool ported a 3.4B model correctly in ten minutes".

**If one thing is taken from the three-block run: F29** — a one-line default that this document
measures as the difference between e2e PCC 0.9586 and 0.9986 on the same code. F30 is its
structural twin: a gate the tool already has and declines to enforce.

### B. OUR SIDE — not tool defects, do NOT report these

| # | what | whose fault | note |
|---|---|---|---|
| **S1** | the HF export shipped no tokenizer | **ours** | our exporter predates this use; fixed by converting `tekken.json` (15/15 vs ground truth) |
| **S2** | `tt-perf-report` not installed → `optimize` preflight failed | **ours** | it **is** declared in `requirements-agent.txt`; we simply never ran that install. The tool detected it, refused cleanly, and named the fix — correct behaviour |
| **S3** | the model had to be converted to HF format at all | **ours / inherent** | Voxtral ships in Mistral-native format; every model this tool handles arrives as a `transformers` model. Not a defect |
| **S4** | six packaging defects, all in `transformers`, all hit in one afternoon | **ours / upstream** | see §S4; none of them the tool's |
| **S5** | the first HF wrapper exposed 26 empty `nn.Module()` placeholders | **ours** | caused 3 of the 4 capture misses originally written up as F23 |
| **S6** | our own `conftest.py` bootstrap shadowed the built `ttnn` inside the planner's scratch copy | **ours** | fixed; looks exactly like a tool defect because the tool creates the copy |
| **S7** | parking the Block-1 demo left its `family_backends` entry dangling | **ours** | this is what exposed F30 — and it left a local absolute path in a shared registry |

### C. CREDIT — things the tool got right

Worth saying, so the feedback is not one-sided:

- **It refuses to optimize while its own test suite is red** (`rc=3`, "a decision, not a crash"),
  and the message named the failing tests and the override flag. That gate did its job.
- **The supervisor distinguishes refusals from crashes** — a dedicated exit code, no retry, no
  device reset. F8 is one call site returning the wrong code, not a missing capability.
- **It caught a real silent-wrongness statically**: this config uses HF's newer `rope_parameters`
  while `tt_transformers` reads only `rope_scaling`, so scaling would be silently ignored at long
  context. Found before anything ran, with two concrete fixes offered.
- **It kept honest accounting**: the e2e report says "1/5 (20%) actually graduated (native stub)"
  and labels the other four `REUSE-wired`, rather than claiming credit for all five.
- **The port is correct.** 5/5 components at PCC 0.9998-0.99999, verified independently.
- **It climbed the whole ladder, including the hand-kernel rung, and the reasoning carried across
  rungs.** On the 2026-08-16 run the ladder went grid → fidelity → dtype → shard → structural →
  **cpp**, and the last rung landed a measured win: `projections: fuse the hi/lo pair into one
  Metalium kernel, weight streamed once`, −9.2 ms, PCC 0.9999804 intact. Its stated hypothesis cites
  the *failed* dtype experiment as the source:

  > *"straight from this op's own dtype rung (bytes are the lever but the weight must not get
  > lossier): `tt_linear_hp` runs `x_hi@W` and `x_lo@W` as two matmuls against the SAME W…"*

  A rejected attempt on one rung became the premise for a successful attempt on a later one. That is
  the behaviour the ladder exists to produce, and it is worth saying plainly alongside the defects.

---

## F1 — a LOCAL model directory gets a degraded probe, and the run dies three stages later

**Status: FIXED** (`probe.py`) · severity: blocks any local-path model · reported: not yet

### What the user sees

```
Step 1/6  Static analysis (plan + compat)
  Summary: 11 ready  /  0 partial  /  0 missing
  Memory fit gate SKIPPED: no LLM-style memory model produced — typically a vision /
  multi-modal model whose memory budget is dominated by per-op scratch, not weights.

Step 2/6  Scaffold the demo folder
ERROR: unexpected compat verdict 'UNKNOWN'; refusing to scaffold
RUN ENDED: pre-flight/setup failed — model could not be loaded, scaffolded, or prepared
```

A model the tool had **just declared 100% supported** (11/11 components drop-in from
`tt_transformers`) is refused at the next stage, with an error naming neither the cause nor the
input that triggered it.

### The actual cause

`probe_model()` has two paths and only one of them does the work:

```python
def probe_model(model_id: str) -> ModelProbe:
    _validate_hf_id(model_id)
    if _is_local_model_dir(model_id):
        return _probe_local_model(model_id)     # returns early — no arch_spec, no memory_model
    ...                                          # Hub path: builds both
```

`_probe_local_model` reproduces the Hub path's category/dtype/param logic but stops before the
arch-spec and memory-model section. So a local directory returns `arch_spec=None`,
`config_status=None`, `memory_model=None` — while the Hub path returns all three.

### The cascade, which is the interesting part

Four layers, each degrading quietly and re-describing the problem as something else:

| layer | what it does with the missing memory model |
|---|---|
| `_probe_local_model` | returns `memory_model=None`. Says nothing. |
| memory-fit gate (`cli.py:1330`) | reports *"typically a vision / multi-modal model"* — **a guess, contradicted by the probe's own `category='LLM'`** |
| same gate | returns `("unknown", …)`; its own docstring says **"Caller SHOULD proceed."** |
| `scaffold.py:214` | treats `UNKNOWN` as fatal — **the opposite of what the gate asked for** |

Nothing anywhere says "local paths take a reduced probe".

### Evidence it is not the model

The probe read the config perfectly — 32 keys, `hidden_size 3072`, `num_hidden_layers 26`,
`num_attention_heads 32`, `model_type mistral`, `head_dim 128`, `intermediate_size 9216` — and
counted `total_params 3429020008`, `weight_bytes 6858040016`. Both correct. `plan` had already
sized the model across nine boards and called it an LLM with **CONFIDENCE: HIGH**.

### Fix

Extract the Hub path's arch/memory block into `_attach_arch_and_memory(probe, cfg, total_params,
weight_bytes)` and call it from **both** paths. Deliberately one function, not a copy — two copies
is how the paths diverged.

### Verified

```
                 before        after
config_status    None          True
arch_family      None          dense
memory_model     None          DenseTransformerModel
arch_spec        None          layers 26, hidden 3072, heads 32, kv 8, head_dim 128
```

### Worth noting

The tool's own XTTS-v2 registry entry uses a **local path** as its canonical id
(`canonical_hf_id="/local/ttuser/apande/models/XTTS-v2-hf"`), so the only TTS family in the
registry points at exactly the input shape that fails here. `optimize`'s documented usage also
takes local directories.

### Still open, not fixed here

The gate/caller contract is still contradictory: the gate documents `unknown` as "caller SHOULD
proceed" and `scaffold.py` refuses. F1's fix removes the *trigger* for this model but leaves the
disagreement in place — any other source of `unknown` will reproduce it.

---

## F2 — the READY verdict can never fire: an all-green compat report is refused

**Status: FIXED** (`compatibility.py`) · severity: blocks any architecturally-ready model that
is not already demo-wired · reported: not yet

### What the user sees

Identical to F1's symptom, which is what made it confusing — fixing F1 did NOT clear it:

```
  Memory fit gate PASSED: mesh `1,1` on `P150` -> FITS (comfortable)
  Summary: 11 ready  /  0 partial  /  0 missing
  Overall verdict:   UNKNOWN
Step 2/6  Scaffold the demo folder
ERROR: unexpected compat verdict 'UNKNOWN'; refusing to scaffold
```

### The actual cause

`_aggregate_overall` leaves `report.overall` at its `"UNKNOWN"` default and overwrites it only if
a predicate in `_OVERALL_FROM_STATUSES` matches. The third entry is the unconditional catch-all:

```python
(
    lambda _: [],        # <-- an empty list. `if predicate(report):` is ALWAYS False.
    "READY",
    "All required blocks already exist in models/tt_transformers/...",
),
```

| compat result | verdict |
|---|---|
| any block MISSING | `BLOCKED` — fires correctly |
| any block PARTIAL | `FEASIBLE WITH WORK` — fires correctly |
| **everything READY** | no predicate matches → stays `UNKNOWN` → scaffold refuses |

**The better the compatibility result, the more certainly the run dies.** A perfect report is the
one input that cannot produce a verdict.

### Why it survived this long

`_aggregate_overall` returns `ALREADY SUPPORTED` early for anything in `SUPPORTED_HF_MODELS` or
found by discovery — which is most models the tool is pointed at. The dead branch is only reached
by a model that is architecturally ready but **not yet wired as a demo**, i.e. exactly the
new-model case the tool exists to serve.

### Fix

`lambda _: []` → `lambda _: True`, with a comment recording what it broke, since the next person
reading a bare `True` will wonder why it is not simply `else`.

### Verified

```
before:  Overall verdict: UNKNOWN     11 ready / 0 partial / 0 missing
after:   Overall verdict: READY       11 ready / 0 partial / 0 missing
```

### Note on F1

F1 and F2 produce the *same* error message from the *same* line. F1's fix was necessary (the
memory gate genuinely had no model) but not sufficient, and the identical symptom made it look
like the first fix had not worked. Two independent defects, one error string.

---

## F3 — a Python exception is reported as "the PCC gate rejected the output", and triggers hours of the wrong work

**Status: OPEN (tool)** · severity: sends the loop down its most expensive path for a non-numerical
failure · reported: not yet

### What the user sees

```
FAILED simple_text_demo.py::test_demo_text[...] -
  Exception: No fallback tokenizer found for base model: voxtral-tts-backbone
...
  ESCALATING on PCC fail  model=/localdev/.../voxtral-tts-backbone
  The ALREADY-SUPPORTED routing produced output the PCC gate rejected. Drafting a NEW
  backend via auto-onboard and re-invoking `up` so the scaffold + per-component iterate
  loop runs.
```

Nothing was measured. The test raised before producing a number, and an **environment** problem
(no tokenizer on disk) was classified as an **accuracy** problem ("the PCC gate rejected"). The
response is the most expensive path the tool has: draft a new backend, re-scaffold, and port every
component with LLM agents. On this run that consumed ~1.5 h and 39+ agent rounds before it was
diagnosed by hand.

### Why it matters

The distinction is cheap to make — a pytest ERROR/exception is not a PCC failure — and the two
call for opposite responses: an environment fault should be reported and fixed in seconds, an
accuracy fault genuinely warrants the port. Mapping the first onto the second converts a
one-line fix into hours of device and agent time, and the log tells the user the model's numbers
were wrong when they were never computed.

### Suggested fix

Classify the gate result before escalating: if the test errored (exception / collection failure /
missing dependency) rather than producing a PCC below threshold, surface the exception and stop.
`_cli_helpers/failure_classifier.py` and `error_patterns.py` already exist for this kind of
triage; the escalation path does not consult them.

---

## S1 — scaffolding gap (OURS, not the tool's): the HF export shipped no tokenizer

**Status: FIXED** (`hf_models/make_tekken_hf_tokenizer.py`)

`export_backbone_hf.py` emits `config.json` + weights only — its original consumer
(`tt_transformers`) sourced the tokenizer separately. A real HF model ships one, so this is our
gap, and it is what triggered F3.

**Fixed faithfully rather than substituted.** Tekken is tiktoken-shaped, so `tekken.json` converts
directly: vocab entries are `{rank, token_bytes(b64), token_str}`, ids are `rank + 1000` (ids
0–999 are the 1000 special tokens), and only the first 130072 ranks are in the released vocabulary
(1000 + 130072 = 131072 = the embedding width). Verified `len(tokenizer) == 131072`.

**Validated against ground truth: 15/15 fixture cases match `mistral_common` exactly** — the text
ids the tokenizer produces appear verbatim inside the recorded prompt ids, across 8 languages,
digits, a symbol run, emoji and literal tab/newline.

### The trap in it, worth recording

The first conversion was wrong in a way **no round-trip test would catch**: it kept only the best
split per token instead of every valid split, and never sorted merges globally by the merged
token's rank. The result decoded to **byte-identical text** while emitting **55 tokens where the
truth has 26** — a correct-looking tokenizer that silently doubles sequence length, i.e. doubles
prefill cost and changes every downstream measurement. Only comparing ids against a known-good
tokenizer catches it.

---

## F4 — the deliberately-wrong constructor: the tool emits a call it expects to fail, and the repair never came

**Status: root cause identified; F5 is the fix** · severity: no component can graduate

### What the user sees

Five components, all matching the PyTorch reference — and **not one graduates**:

```
component          best PCC        failure class      gate is 0.99
decoder_layer      0.99999594      SHAPE
r_m_s_norm         0.99998738      UNEXPECTED_KWARG
rotary_embedding   0.99996236      UNEXPECTED_KWARG
m_l_p              0.99981016      MISSING_KWARG
attention          0.99969689      MISSING_KWARG
```

The port is CORRECT. Every failure is the test wrapper failing to construct the tt_transformers
class, before any number is computed.

### The actual cause — and it is by design

`bringup_loop.py:1977` emits ONE hardcoded constructor call into every component's test:

```python
canonical = {canonical_import_target}(
    mesh_device=device, args=args, state_dict=..., layer_num=0, dtype=ttnn.bfloat16,
)
```

with a comment that states the problem plainly:

> The exact constructor signature varies per class (Attention takes 14 args, MLP takes 11,
> RMSNorm different, RotaryEmbedding different). **The LLM refines this call on PCC failure.**

So the strategy is deliberate: **emit a call known to be wrong, let it crash, repair on retry.**
That is defensible — the signatures genuinely vary — but it makes the repair loop load-bearing.
On this run the repair never happened: `attempts` is **1** for every component after 39+ rounds
and 9 hours.

**This is not a stale-API problem.** The tool is running from its own branch; `tt_transformers`
there requires `tt_ccl`, `weight_cache_path`, `transformation_mats`, `configuration`, and the
template passes none of them. It would fail identically on any checkout.

### CORRECTION (2026-08-13): the repair DID happen — through the agent's generic tools

The first write-up of F4 said "the repair never came". That is wrong, and the truth is more
interesting. `allowed_tools` grants the agent ten MCP tools **and six ordinary ones** — `Read`,
`Edit`, `Write`, `Bash`, `Grep`, `Glob`. The MCP half was dead (F6); the ordinary half was not.

So the agent read the real `__init__` signatures, repaired every constructor, ran the tests with
`Bash`, and — unable to call `record_result` — wrote the graduation snapshots **by hand** with
`Write`. The file timestamps show it working through the list one component at a time:

```
07:16:40  attention.py          + attention.py.last_good_native
07:18:21  decoder_layer.py      + decoder_layer.py.last_good_native
07:19:37  m_l_p.py              + m_l_p.py.last_good_native
07:20:50  r_m_s_norm.py         + r_m_s_norm.py.last_good_native
07:22:21  rotary_embedding.py   + rotary_embedding.py.last_good_native
```

Six minutes, five components, all passing. **F4's fail-first/repair-on-retry design worked.** What
failed was reporting it.


---

## F5 — the systemic-pattern detector counts CLASS NAMES, so the broadest bugs are the ones it misses

**Status: FIXED, but LATENT — it was NOT the cause of this run's stall.** `termination_check` was
never callable (F6), so the systemic hint was never requested and this code never ran. The counting
bug is real and verified against our exact failure map, but it is insurance for future runs, not
the explanation for what went wrong here. The first write-up implied otherwise; corrected.

**Status: FIXED** (`bringup_mcp.py`) · severity: disables the tool's own escape hatch exactly when
it is most needed · reported: not yet

### The design being defeated

The agent prompt is explicit that per-component repair is the wrong response to a shared bug:

> If `termination_check` returns a non-null `systemic_hint`: STOP iterating per-component and
> address the shared root cause first. A systemic hint means 3+ components are failing with the
> same class — the fix belongs in `tests/pcc/conftest.py` or the common `_make_arg_for` helper,
> not in each stub. **Individual repairs will keep re-hitting the same wall.**

That is precisely our situation. It never fired.

### Why

```python
_hot = [(cls, cs) for cls, cs in _class_counts.items() if len(cs) >= 3]
```

It counts components sharing an **identical class string**. F4's single template produces two
different strings from the same line of code — `MISSING_KWARG` where a class needs more args,
`UNEXPECTED_KWARG` where a class rejects `mesh_device`:

```
MISSING_KWARG      attention, m_l_p              2
UNEXPECTED_KWARG   r_m_s_norm, rotary_embedding  2
SHAPE              decoder_layer                 1        -> nothing reaches 3, hint stays None
```

**The broader the shared bug, the more different symptoms it produces, and the less likely the
"same class" test is to fire.** A bug that breaks every class in the same way trips it; a bug that
breaks classes in *different* ways does not.

### Fix

Count by **family**, not by class name. `MISSING_KWARG`, `UNEXPECTED_KWARG` and `API_SIGNATURE`
are one family (`CONSTRUCTOR_SIGNATURE`) because they share a fix location. The hint also now
names the classes actually seen, and for this family points at the canonical-constructor call
rather than only at conftest.

### Verified — replaying the stalled run's exact failure map

```
OLD (by class name)   {'MISSING_KWARG': 2, 'UNEXPECTED_KWARG': 2, 'SHAPE': 1}  -> fires: False
NEW (by family)       {'CONSTRUCTOR_SIGNATURE': 4, 'SHAPE': 1}                 -> fires: True
```

---

## F6 — the agent's tool server never starts: `mcp` is an undeclared, missing dependency

**Status: FIXED (environment)** · severity: **the root cause of an 11-hour stall** · reported: not yet

### What the user sees

Nothing. That is the whole problem. Round after round of:

```
BRING-UP (cc) round 36 ... target=`?` rung=? (graduated 0) → invoke claude → gate
  · round 36 working… 45s, 8 tool calls
```

No error, no warning. Indistinguishable from a model that is genuinely hard to port.

### The actual cause

`_cli_helpers/bringup_cc.py` writes an MCP config telling Claude to launch the tool server:

```json
"command": "/opt/venv/bin/python",
"args": ["/…/scripts/tt_hw_planner/bringup_mcp.py"]
```

`bringup_mcp.py:65` does `from mcp.server.fastmcp import FastMCP` — an unguarded module-level
import. **`mcp` was not installed**, so the server died on startup and all ten tools silently did
not exist:

```
termination_check  list_components  run_component  record_result  restore_best
decompose_component  fall_back_to_cpu  mark_harness_skipped
resolve_reference_loader  get_shard_plan
```

`scripts/tt_hw_planner/` ships **no requirements file at all**. The only one in the PR
(`models/experimental/perf_automation/requirements-agent.txt`) lists `claude-agent-sdk`, not `mcp`.

### Every symptom this explains

| symptom | because |
|---|---|
| `target=?` `rung=?` | `termination_check` uncallable — nothing to name a target |
| `attempts` frozen at 1 | `record_result` uncallable — it is the only thing that bumps it |
| `graduated 0` while tests pass | graduation is recorded through `record_result` |
| 36 rounds × 45 s of no progress | the agent had only generic tools (see F4 correction) |
| the systemic hint never fired | it is returned BY `termination_check` |

### Fix

`uv pip install "mcp<2"` into the interpreter named in the config. **The pin matters**: the current
major renamed `mcp.server.fastmcp` → `mcp.server.mcpserver`, so a bare `pip install mcp` installs a
version that still fails the import. Installed 1.29.0; `from mcp.server.fastmcp import FastMCP`
then succeeds and `termination_check()` immediately returned `can_stop: True` with all five
components graduated.

### Suggested fix for the tool

1. Declare the dependency (`scripts/tt_hw_planner/requirements.txt`), pinned `mcp<2`.
2. **Pre-flight it.** `ttnn_preflight.py` already exists to check `import ttnn` before any device
   test; the same pattern applied to the MCP server would have turned an 11-hour silent stall into
   a one-line error at startup.

---

## F7 — all progress flows through one channel, and nothing notices when it is dead

**Status: OPEN (design)** · severity: converts any reporting fault into unbounded wasted time

F6 was survivable in principle: the agent finished the work anyway. What made it cost eleven hours
is that **the harness has exactly one way to learn anything** — the MCP tools — and no cross-check
against the filesystem it can see.

At 07:22 the work was complete and on disk: five repaired stubs, five `.py.last_good_native`
snapshots, tests passing. The harness's own graduation predicate is
`_is_graduated()` = *"snapshot exists AND stub is native"* — a pure filesystem check that would
have returned **True for all five**. It was simply never consulted outside the dead MCP path.

Meanwhile the loop's response to "no progress reported" is to start another identical round,
indefinitely (`max_consecutive_timeouts` defaults to 1000).

**Suggested:** re-derive graduation from disk at the top of each round, and halt with a loud error
after N rounds in which nothing on disk changed AND no tool call succeeded — "the agent reported
nothing and changed nothing" is a diagnosable state, not a reason to keep spending.

---

## F8 — a clean, deliberate refusal is misread as a hardware crash, and the card gets reset

**Status: OPEN (tool)** · severity: resets the user's device for a config error; burns 3 retries
· reported: not yet

### What the user sees

```
[optimize/cc] refusing to mutate an existing demo in place. Pass --in-place to override.
[optimize/supervisor] orchestrator exited rc=1 (likely native crash / device wedge)
                      -- resetting device + restarting (restart 2/3)
[optimize/supervisor] reclaimed device (killed holders none) + tt-smi -r 0 rc=0
```

The refusal is correct, intentional, and printed a clear message one line earlier. The supervisor
sees only `rc=1`, assumes *"likely native crash / device wedge"*, runs **`tt-smi -r 0` to reset the
accelerator**, and retries the identical command — three times, each ending the same way.

### Why it matters

Resetting hardware is not a neutral act: on this branch a board reset is the documented recovery
for a wedged card, and it interrupts anything else using it. Doing it in response to a *policy
refusal* is both useless (the refusal is deterministic — retrying cannot help) and disruptive.

`rc=1` here carries no information: the tool used the same exit code for "I decline" and "the
device died". The distinction exists one line above in plain text.

### Suggested fix

Give deliberate refusals a distinct exit code (or a sentinel line the supervisor greps for), and
only treat rc as a wedge when the device is actually unresponsive — which is cheap to test by
opening it. Verified after the three resets: the card was healthy the whole time
(`ttnn.open_device` fine, grid 13x10).

### CORRECTION (2026-08-13): the mechanism EXISTS — this is one wrong exit code

A later refusal in the same run printed:

```
[optimize/cc] refusing to start against a tool whose own tests fail.
[optimize/supervisor] child REFUSED to start (rc=3) — a decision, not a crash. Not restarting; the reason is above.
```

So the supervisor **does** distinguish a deliberate refusal from a crash, via a dedicated exit code
3, and correctly declines to reset the device or retry. The in-place refusal simply returns `rc=1`,
which falls into the crash path. **F8 is therefore a one-line fix — return 3, not 1 — not a design
gap.** The original write-up overstated it; corrected here.


---

## F9 — a local model directory is mistaken for a demo directory (same family as F1)

**Status: WORKED AROUND** · severity: `optimize` unusable when the model is a local path
· reported: not yet

`optimize` accepts either a model id or a demo directory, resolved by `_resolve_target`:

```python
p = Path(target)
...
if p.is_dir():
    return p.resolve()          # any directory is assumed to BE the demo
```

Our model is a local folder of weights, so passing it as the target made the tool treat
`/localdev/.../hf_models/voxtral-tts-backbone` as the demo to optimize. It is outside the repo, so
worktree isolation failed; it does not look planner-emitted, so `kind` became `"existing"`; and the
run refused (then F8 reset the card three times).

**Workaround:** pass the demo directory instead of the model id —
`optimize models/demos/voxtral_tts_backbone`. The classification then flips to `(emitted)` and it
runs in place, correctly.

**Suggested fix:** resolve a directory that contains `config.json` + weights as a MODEL (look it up
via `bringup_status.json` like the model-id path does), not as a demo. Same root cause as F1: local
paths are second-class throughout, and the failure surfaces far from the cause.

---

## S2 — OURS: `tt-perf-report` was never installed, so `optimize`'s preflight failed

**Status: FIXED (our environment)** · **NOT a tool defect — do not report**

### What we saw

```
[optimize/cc] preflight FAILED
  FAILED test_before_loop.py::test_before_loop_all_mocks_produces_manifest_and_baseline
  FAILED test_tracy_tool.py::test_tracy_tool_orchestrates_runs_and_median
  ... 4 failures
[optimize/cc] refusing to start against a tool whose own tests fail.
```

Root cause of all four: `FileNotFoundError: [Errno 2] No such file or directory: 'tt-perf-report'`.

### Why it is ours

`tt-perf-report==1.2.2` **is** listed in `models/experimental/perf_automation/requirements-agent.txt`,
with install instructions at the top of that file. We never ran it. Fixed with
`uv pip install -r models/experimental/perf_automation/requirements-agent.txt`; the tool's suite
then read **2617 passed, 7 skipped**.

### Contrast with F6 — this is the distinction that matters

| | F6 (`mcp`) | S2 (`tt-perf-report`) |
|---|---|---|
| declared anywhere? | **no** — `scripts/tt_hw_planner/` ships no requirements file | **yes**, with install instructions |
| detected? | **no** — silent, 11 h of empty rounds | **yes** — refused in seconds, named the tests and the override |
| owner | **tool** | **us** |

Same symptom class (a missing dependency), opposite verdicts. The difference is entirely whether
the tool declared it and checked for it.

---

## F10 — the F9 workaround loses the model id, so `optimize` cannot build its own correctness gate

**Status: WORKED AROUND** · severity: `optimize` unusable without a hand-supplied `--pcc-test`
· reported: not yet

### What the user sees

```
Step 6/10  Mapping the model's pipelines & building perf tests
  CANNOT CONTINUE — no usable correctness gate.
  no --pcc-test supplied, and no cached HF reference for None. There is no ground truth to check
  correctness against, so optimize would be free to commit edits that silently degrade the model.
  PLEASE GIVE A PCC TEST TO RUN OPTIMIZE: pass --pcc-test <file>::<test>.
```

**The refusal itself is correct and worth crediting** — it will not make a model faster if it
cannot prove the model is still right. The defect is the reason it got there.

### The actual cause — two findings interacting

Note `no cached HF reference for **None**`: the model id is `None`.

1. `optimize <model-id>` fails, because F9 resolves any directory argument as a demo dir and our
   model is a local folder.
2. The workaround is `optimize <demo-dir>` — which works, but **the model id is then never
   resolved**, so the stage that would auto-generate the PCC gate has no reference model to
   compare against.

So the documented "just point it at the directory" path cannot auto-generate a correctness gate,
and the model-id path that could is broken by F9. Either one alone is survivable; together they
close both routes.

### The gate existed the whole time

`emit-e2e` had already emitted `tests/e2e/test_e2e_pipeline.py`, which compares against the HF
golden, declares `PCC_THRESHOLD = 0.95`, and prints exactly the format asked for:

```python
print("e2e PCC=%s" % min(float(pcc_call1), float(pcc_call2)), flush=True)
```

The tool produced its own correctness gate one stage earlier and could not find it.

### Workaround

Pass it explicitly:

```
--pcc-test models/demos/voxtral_tts_backbone/tests/e2e/test_e2e_pipeline.py::test_e2e_pipeline
```

### Suggested fix

When the target is a planner-emitted demo, look for `tests/e2e/test_*.py` in that demo before
declaring there is no gate — the tool wrote it. And resolve the model id from
`bringup_status.json`, which sits in the same directory and records it.

---

## F8 (addendum) — a second refusal path also returns rc=1 and gets restarted

The `CANNOT CONTINUE — no usable correctness gate` refusal also exits **rc=1**, so the supervisor
retried it three times before giving up:

```
[optimize/cc] run failed (see messages above)
[optimize/supervisor] child exited rc=1; 3 restart(s) exhausted.
```

Same root cause as F8: deliberate refusals must return the dedicated refusal code (**rc=3**), which
the supervisor already handles correctly. At least two call sites return 1 instead — the in-place
refusal and this one.

---

## R — `optimize` RESULTS (running; snapshot 2026-08-13 20:04, 4h50m elapsed, ~2h30m in the loop)

Once the bring-up defects above were cleared, the optimize half ran unattended and **worked**. This
section is the credit half of the ledger and is the material for the comparison write-up.

### Headline

| | device_ms (whole test) | decode ms/token (trace+1cq) | e2e PCC |
|---|---|---|---|
| baseline as generated by `auto-up` | **1121.293** | 28.878 | 0.9976 |
| after the grid/structural rungs (20:04) | 348.026 | 23.168 | 0.9795 |
| after the dtype rung on MLP + attention | — | 18.282 | 0.9715 |
| after the down_proj shard-width sweep (O4b) | 281.8 | 16.763 | 0.9708 |
| after the same sweep generalised to K/V (O4b) | 276.7 | 16.135 | 0.9897 |
| after fusing Q/K/V into one decode projection (O4c) | 273.6 | 15.976 | 0.9708 |
| after fusing RoPE into one op (O4d) | 271.9 | 15.212 | 0.9903 |
| run 2 — Q+K rotated in one call (O4f) | 269.5 | 14.909 | 0.9903 |
| run 2 — decode-native Q/K/V layout (O4g) | 261.9 | 13.975 | 0.9903 |
| run 2 — head creation fed the projection's shard (O4h) | 254.5 | 13.277 | 0.9903 |
| run 2 — DRAM-sharded LM head weight (O4i) | — | 13.228 | 0.9904 |
| run 2 — untilize vocab blocks before joining (O4j) | — | 13.139 | 0.9904 |
| run 2 — fused K/V cache write (O4k) | 234.1 | 12.890 | 0.9904 |
| run 2 — decode SDPA 16 cores/head -> 2 (O4l) | 231.8 | 12.633 | 0.9904 |
| run 2 — fused-QKV grid re-swept 32 -> 48 (O4m) | 229.6 | 12.427 | 0.9903 |
| run 2 — SiLU folded into the SwiGLU multiply (O4n) | 229.0 | 12.352 | 0.9903 |
| run 2 — norm's shard chained into QKV (O4o) | — | 12.299 | 0.9903 |
| run 2 — gate/up plan, reshard now free (O4p) | — | 12.038 | 0.9903 |
| run 2 — residual stream kept in the norm's shard (O4q) | — | 11.928 | 0.9903 |
| run 2 — o_proj/down_proj hand shards to the residual add (O4q) | — | 11.839 | 0.9903 |
| run 2 — LM head vocab blocks written as bf8_b (O8) | — | **11.827** | **0.9774** |
| tool's own roofline target | 338.541 | — | gate 0.95 |
| **hand-port, for reference** | — | **15.907** | — |

**11.827 against 15.907 — the tool is 25.7% AHEAD of 74 human experiments** — but see O8: the last
step is where the run stops being worth watching., autonomously, with
PCC bit-identical across its last four wins (0.990347151783074, unchanged to every decimal). Every
one of those four was a layout or dispatch result. None spent accuracy.

State this with the accuracy bar attached, every time it is quoted: the tool is at e2e PCC 0.9708
against its 0.95 gate; the hand-port's p150 decode holds 0.981 (`STATUS.md`, and 0.99991 on the
N150 branch). Those two numbers are not measured over the same thing, so they are not a clean
comparison — but the tool is certainly not *tighter*, and part of the last 3 ms came from
`bfloat8_b` on `down_proj`, which is §6.16's `w2` decision the hand-port took and then handed back.

**On the dtype walk, and a correction.** Three commits took `gate/up`, `down_proj` and `q/k/v/o` to
`bfloat8_b`, and PCC did walk 0.9795 → 0.9715 alongside 23.168 → 18.282. But the ladder then
stopped itself: pushing `q_proj` on to `bfloat4_b` measured **faster** (18.282 → 17.770) and was
**reverted on PCC 0.7707**, with reasoning worth quoting —

> *q_proj is the most exposed weight in the block for this lever: its output goes through RoPE into
> the attention scores, so a coarser weight perturbs WHICH positions attend to which, not just by
> how much — a change the softmax then amplifies.*

That is the same argument §6.17 makes about top-2 gaps in a discrete decision, reached
independently. PCC has since recovered to 0.9897. So the fair statement is **not** "it trades
accuracy until the gate stops it" — it stopped one rung early, on structure, and its best result is
its most accurate recent one. O1 still stands as a gap (there is no per-weight axis, and
`down_proj → bf8_b` is the §6.16 `w2` decision the hand-port took and then deliberately gave back),
but the ladder is more discriminating than the raw PCC trend suggested.

**3.22× on device time, autonomously, with the PCC gate held the whole way** (0.9795 against its
0.95 e2e threshold; one attempt fell to 0.8638 and was correctly rejected and reverted).
`modeled_floor_ms` 400.68; throughput 43.16 tok/s against a 71.31 theoretical, now `IN_BAND`.

### Against the hand-port (this is the number the experiment was for)

| Block 1 decode step | ms | source |
|---|---|---|
| N150 branch this port forked from | 23.15 | `STATUS.md` header table |
| **tool, fully autonomous, ~2.5 h in the loop** | **23.168** | `perf_mcp_stage_ms`, trace+1cq |
| hand-port at §6.39 (p150 fork, pre-§6.65/§6.67) | 21.2 | `STATUS.md` header table |
| **hand-port current, §6.72** | **15.907** eager / **15.922** traced | `STATUS.md:4148`, `:4699` |

**The tool is at the hand-port's starting line, not its finish line.** 23.168 vs 23.15 is a dead
heat with the N150 build the p150 port forked from — i.e. it independently reached in one
afternoon what that branch already had — but the hand-port's *current* Block 1 is **15.9 ms**, so
the tool is **1.46× slower** than the target stated at the top of this file. It has not closed the
gap; it has covered the first third of it.

Two things separate them, and only one is a fair fight:

1. **Precision.** The tool is buying part of its 23.168 with trades the hand-port refused —
   `bfloat4_b` on the LM head and a LoFi compute config — against a hand-port whose decode holds
   PCC 0.981–0.99991. Its 0.95 e2e gate permits what §6.16/§6.17 rejected on quality grounds. So
   the honest read is that the tool is 1.46× slower *and* less accurate, not trading one for the
   other.
2. **The rungs it has not reached.** It is still on `knob:grid` and `knob:dtype` for the remaining
   matmuls. The hand-port's last 5 ms came from §6.65/§6.67-class structural work plus per-weight
   precision (§6.16), and the per-weight axis does not exist in this tool (O1).

*(Corrected 2026-08-13: an earlier draft of this section compared against the stale 21.2 ms figure
from the `STATUS.md` header table and claimed the tool was within 9.3%. §6.72 superseded that
number; the header table was not updated. The real gap is 46%.)*

### What it found, in order (14 commits)

```
argmax        untilize logits to ROW_MAJOR              1121.3 -> 545.7   (-51.3%)
lm_head       bf8_b weight, then bf4_b, then LoFi        545.7 -> 397.3
rmsnorm       width-shard the decode norm                397.3 -> 370.4   (-2.11 ms/token)
matmul        full-grid 1D-mcast plans, q/k/v/o/down     370.4 -> 356.7
host          capture the decode step, replay per token  356.7 -> 354.9
datamove      producers emit the consumer's shard        354.9 -> 348.0
```

### Independent rediscovery of the hand-port's findings

Three of the hand-port's recorded wins were re-derived from scratch by a tool that had never seen
that code:

- **`perf(rmsnorm): width-shard the decode norm so it fills the grid`** — the hand-port's **§6.67**,
  its single largest win at −5.399 ms/frame. Same lever, same stated mechanism (the norm was at
  `grid=tiny`, one core).
- **`perf(decode): capture the decode step once and replay it per token`** — **§6.65**, −4.244 ms/frame.
- **the five full-grid decode projection plans** — **§6.52**, −4.24 ms/frame.

Note §6.67 was a *reversal* — the hand-port shipped the sharded norm, reverted it at §6.39, then
reinstated it at §6.67. The tool went straight to the end state.

### O4 — the argmax finding is new, and it explains a hand-port rejection

The first and largest single win, worth **51.3% of device time**, is a cause the hand-port measured
but never diagnosed. The tool's note:

> `ttnn.argmax` picks its parallel path from INPUT LAYOUT, not a flag — `uses_multicore_path()`
> (`argmax_device_operation.cpp:16`) bails to the single-core kernel for any non-ROW_MAJOR input,
> and `ttnn.linear` hands it TILE, which also pads the `[1,1,131072]` decode row to the 32-row tile
> height so one core scanned ~32× the needed bytes.

`STATUS.md` §6.8 measured exactly this pathology on Block 2's `semantic_code` — *"argmax over 8320
values, 490.1 us, 39.9%, 33 KB at 0.07 GB/s — ALL overhead"* — and worked around it by moving the
reduce to the host (option C, 1.439×, shipped). **That rejection is worth re-running with a
`ttnn.to_layout(logits, ROW_MAJOR)` in front of the device argmax.** The host path may still win on
a 33 KB reduce that already ends in a D→H copy, but the A/B was scored against a single-core kernel
that did not have to be single-core.

This is the clearest case in the whole experiment of the tool contributing something the human pass
missed, and it came from reading the kernel's C++ dispatch condition — not from a sweep.

### O4b — "fill the grid" is WRONG for a K-heavy decode projection, and this is the second finding to take

The second-largest win of the run, and it is precision-neutral — a pure shape result. Having earlier
committed full-grid plans for every decode projection, the tool contradicted its own heuristic on
`down_proj` (`32 x 9216 x 3072`, the deepest K-reduction in the model). Its diagnostic: the op takes
**the same 0.159 ms/call at bf16 and at bf8_b**, so it is not bandwidth-bound and the limit is the
shape of its k-reduction, not the bytes. It then swept the activation shard width — which sets
`in0_block_w` and `per_core_N` together — instead of assuming the widest grid:

```
cores    96      48      32      24      16      12       8
ms    0.1589  0.0960  0.1137  0.0934  0.1084  0.1342  0.1964
                               ^ pinned
```

Non-monotonic, with a **1.70× spread** between the widest grid and the best one. Kept:
`18.282 → 16.763 ms/token` (−8.3%), PCC held at 0.9708. Its own note calls it *"a correction to my
own earlier heuristic: 'occupy the full grid' is wrong for a K-heavy decode projection."*

**Why this matters for the hand-port.** `tt/ttnn_voxtral_gpt.py:76` defines **one** grid for all
five decode matmuls:

```python
_MM_GRID = (12, 6)      # 72 of the 130 cores; 13x10 measured 0.31 ms WORSE
...
_PRG_W2 = _mm1d(4, 2)   # K=9216  N=3072  Nt=96 -- the deepest reduction in the model
```

§6.52 swept that choice **globally** (72 against 130, −4.24 ms/frame) but never **per-op**. `w2` is
the identical op to the one swept above — same K, same N — and it is running on 72 cores, which
sits on the wrong side of the curve the tool measured (48c 0.0960, 96c 0.1589). The other four
projections are K-light and may well want the wide grid they have; the point is that one global
`_MM_GRID` cannot be right for both.

**Concrete suggestion: give `_PRG_W2` its own `compute_with_storage_grid_size` and sweep it
independently.** The tool's curve says the win is in the 24–48 core range and is worth ~1.7× on
that op.

**It then generalised the lesson itself**, re-sweeping K/V at the current dtype rather than
trusting the widest even split:

```
K/V (32 x 3072 x 1024)   default   32c      16c      8c       4c
                          0.0528  0.0379  0.0251  0.0314  0.0535 ms
                                          ^ pinned                  a further -34%
```

`16.763 → 16.135 ms/token`, and **PCC improved** 0.9708 → 0.9897. Q measured best at 32 cores, so it
no longer shares a grid with K/V at all.

**Caveat on how far this transfers — worth stating before acting on it.** The tool's model has
`q_proj` and `k/v` as separate ops; the hand-port **fuses** them into one `wqkv`
(`_PRG_QKV = _mm1d(2, 3)`, K=3072 N=6144). A per-projection grid cannot be applied to a fused QKV
without unfusing it, and unfusing costs a dispatch the hand-port deliberately paid once. So:

- **`w2` / `down_proj` — transfers directly.** Standalone op in both, identical shape. Act on it.
- **Q/K/V — does not transfer as-is.** It is evidence that the *optimum differs per projection*
  (Q wants 32, K/V want 16), which is an argument about whether the fusion is still worth it, not a
  drop-in change.

**SUPERSEDED within the hour, by the tool itself.** The very next attempt was the structural rung on
QKV, and it independently arrived at the hand-port's design: stage the three weights **concatenated**
into one 3072×6144 tensor and run **one wide matmul per token** instead of three, opening one shard
instead of two, then slicing. `16.135 → 15.976 ms/token`. Its own explanation of why the win is small
is the interesting part:

> *Modest rather than dramatic because the three separate projections were already individually tuned
> (their own core counts and a shared input shard), so what is left to win is the per-op dispatch and
> one reshard, not bytes — the fused read moves exactly the same weight bytes.*

So the caveat above was right about the mechanism and wrong about the conclusion: the per-projection
optima are real, and fusing still wins anyway, because what fusion buys is dispatch, not bandwidth.
The hand-port's fused `wqkv` is vindicated by a tool that tried it both ways.

It also recorded what it deliberately did **not** take: `nlp_create_qkv_heads_decode`, which would
collapse the 3 reshapes + 3 permutes as well, because it emits `[1,B,H,D]` while this SDPA path
consumes `[1,H,S,D]` — a contract change across the cache write and SDPA, so it belongs in its own
attempt rather than riding along. That is the same boundary §6.68/§6.72 negotiated by hand.

### O4d — RoPE as one op, and where the tool's remaining lead actually comes from

The generated stub wrote `x*cos + rotate_half(x)*sin` out longhand — two slices, a neg and a concat
to build `rotate_half`, then two multiplies and an add. **Seven dispatches for one elementwise
rotation, twice per layer, 26 layers ≈ 360 launches per token** against tensors of a few KB. That is
why the roofline tagged this model's `BinaryNg`/`Reshape`/`Concat`/`Slice` ops `bound_by=dispatch`.
It replaced the chain with `ttnn.experimental.rotary_embedding_hf`: `15.976 → 15.212 ms/token`
(−4.79%), **and PCC rose** 0.9714 → 0.9903, because the fused kernel accumulates the rotation
internally instead of round-tripping two bf16 products through DRAM.

Two things in that commit are worth lifting on their own:

- **It explained a metric disagreement rather than picking the flattering number.** *"device_ms
  barely moves because it SUMS per-op durations — what shrank is the inter-op gap those ops were
  tagged for, which only the wall metric sees."* 273.62 → 271.87 device, against −4.79% wall.
- **It shipped a latched fallback** — the explicit chain is retained for operands the fused op
  refuses, and the flag latches after the first refusal *"so it cannot raise inside a captured
  trace."*

**This one is NOT a finding for the hand-port — it is the tool catching up.** `ttnn_voxtral_gpt.py`
has used `rotary_embedding_hf` from the start (`_rope`, and both decode call sites). Worth noting
the hand-port is still ahead on this specific op: it calls the **decode-specialised** path
(`is_decode_mode=True`, with cos/sin sharded to match, per the comment at line 53), where the tool
calls prefill mode with `s=1`.

#### So where does the tool's 0.7 ms lead actually come from?

Now that both implementations agree on fused QKV and fused RoPE, the remaining delta is a short
list — and it splits cleanly into "free" and "paid for":

| the tool has | hand-port equivalent | free? |
|---|---|---|
| per-op narrow grids: `w2` 24c, K/V 16c | one global `_MM_GRID` (12,6) = 72c for all five | **FREE — precision-neutral. This is the one to take.** |
| `down_proj` at `bfloat8_b` | `w2` at bf16 **on purpose** (§6.16) | PAID — 77% of the precision stack's accuracy cost |
| LM head at `bfloat4_b` | n/a — Block 2 consumes the hidden state | n/a |
| `ttnn.argmax` on a ROW_MAJOR input | argmax on the **host** (§6.8) | FREE — but re-measure, see O4 |

**The conclusion for the comparison write-up:** roughly the whole of the tool's lead is (a) per-op
grid sweeps the hand-port never ran, which cost nothing in accuracy, and (b) one precision trade the
hand-port evaluated and deliberately declined. Take (a); (b) is already settled and settled
correctly.

### F11 — the documented `--max-rounds` default is 20; the real one is 3, and it is the ONLY exit

**Status: found 2026-08-13, run 1 ended on it** · severity: silently caps every run at ~1/7 of the
advertised effort · reported: not yet

Run 1 ended after 7h20m with this line, and nothing else:

```
pipeline main: 3 round(s), can_stop=False
```

`can_stop=False` is the tool's own gate saying **it should not have stopped**. It stopped because
`DEFAULT_MAX_ROUNDS = 3` (`cc_optimize/run.py:39`, and `cli.py`'s `--max-rounds` default). The
documentation says otherwise:

```
GETTING_STARTED.md:272
  `--max-rounds N` — cc engine: max `claude -p` optimization rounds per pipeline (default `20`).
```

**20 documented, 3 in code.** A user who reads the guide and does not pass the flag gets 3 rounds.

**This compounds with O7 and that is the real severity.** O7 records that the throughput band can
never fire `can_stop` when the parameter count is estimated — which is every locally-supplied
model. So for a local model the band exit is unreachable, the floor exit is unreachable, and
`--max-rounds` is the *only* thing that ends a run. The documented behaviour ("the deterministic
gate can still stop earlier once each op is at its floor") is therefore not what happens: **every
such run ends at exactly 3 rounds, mid-climb, and reports it in one line at the bottom of a long
report.** Ours had just produced its two largest wins in round 3 when the cap hit.

**Fix:** align the default with the documentation (or the documentation with the default), and make
the terminal line say *why* it stopped — `stopped: round cap (3) reached with can_stop=False` reads
very differently from `3 round(s), can_stop=False`.

**Verified:** relaunched with `--max-rounds 20`, pid 1067756, from the same tree.

### O4e — the tool independently corroborates §6.72, the hand-port's most contested experiment

In its last round it tried `nlp_create_qkv_heads` with `transpose_k_heads=False` to replace the
head-split's 9 ops (3 slices + 3 reshapes + 3 permutes) with one fused call. PCC was bit-identical,
so this was a pure data-movement question — and it **regressed**: device 273.62 → 281.07 (+2.7%),
per-token 15.21 → 16.19 (+6.45%). Its conclusion:

> *Dispatches fell 3413 → 2867 yet it got slower — decisive: these were view ops doing no work.*

That is §6.72 reached from the other direction. The hand-port went fused → hand-rolled and measured
−0.775 ms/frame bit-exact; the tool went hand-rolled → fused and measured +0.98 ms/token. Both land
on the same conclusion, and the tool's phrasing supplies the mechanism §6.68 got wrong when it
"counted one op short": **dispatch count is not the cost when the ops being removed are views.**

Two independent confirmations of a reversal that was the hardest call in the hand-port's log.

### O4f — a THIRD finding to take: the same win §6.23 rejected, by a route that avoids what it rejected

Run 2's first win, and the one most worth acting on, because it looks like a contradiction of the
hand-port and is not one.

**What the hand-port rejected (`NOTES.md [gpt-23]`, §6.23):**

> *Two calls, not ttnn's fused q+k rope: that one implements the INTERLEAVED convention via a
> trans_mat, and our wq/wk are permuted to HALF-SPLIT at load. Measured 0.236 ms/frame for reverting
> that permute, disjoint q/k cores and losing bit-exactness.*

A correct rejection. The objection is to **ttnn's fused q+k rope operator**, whose convention
disagrees with the half-split layout the weights are permuted into at load.

**What the tool did instead:** it did not use that operator at all. Q and K are adjacent in the
fused QKV output and RoPE applies per head against the same cos/sin, so it slices **q+k out
together**, splits heads once into a **40-head** tensor (32 q + 8 kv), calls the *same*
`rotary_embedding_hf` **once**, and slices query/key apart afterwards.

```
trace+1cq 15.2117 -> 14.9090 ms/token (-1.99%)    device 271.92 -> 269.45
PCC 0.990347 -> 0.990347, UNCHANGED               dispatches 3413 -> 3257
```

Same convention, same op, one dispatch instead of two — plus it drops a slice and a reshape+permute
pair, so the block is 2 ops lighter per layer on top of halving the rotary count. **The fusion is
exact, not an approximation**, which the unchanged PCC confirms.

**Why this matters:** §6.23 measured the cost of a *convention change* and rejected it, correctly.
This route asks nothing of the convention. The hand-port keeps half-split, keeps its permute, keeps
bit-exactness, and still gets the launch-count win — 52 rotary launches per token become 26. For
scale, `[gpt-24]` measured a comparable 26-launch saving (the fused KV write) at **0.405 ms/frame,
bit-identical**.

**The one thing to check before taking it.** The hand-port's decode RoPE runs `is_decode_mode=True`,
and `ttnn_voxtral_gpt.py:53` records that *"rotary_embedding_hf's decode mode requires cos/sin
sharded as well as the input"*. Concatenating to 40 heads changes the shard shape, so the question
is whether a 40-head sharded rotation is expressible on this grid — not whether the arithmetic
holds, which it does. (The tool calls prefill mode with `s=1`, so it never met this constraint.)

**Status: strongest candidate of the three.** Unlike O4 (argmax, needs a re-measurement) and O4b
(`w2` grid, needs a sweep), this one is a known quantity: exact arithmetic, a measured win on the
identical model, and it sidesteps the specific objection that got it rejected the first time.

### O4g — decode was wearing prefill's layout, and 31 of every 32 rows were padding

The largest run-2 win, and the one with the most general lesson in it. The block shaped Q/K/V as
`[1, heads, seq, hd]` because that is what a **prefill** SDPA reads — but nothing in a decode step
reads that. `paged_update_cache` wants `[1, batch, kv_heads, hd]`; `scaled_dot_product_attention_decode`
wants `[1, batch, q_heads, hd]`. Bridging the two cost a reshape, a permute and three slices up
front, a permute plus a reshard before each cache write, and a permute in and out of the attention:

> *19 ops per layer where 8 do the work, and the profiler tags every one of the other 11
> `bound_by=dispatch`: their cost IS the launch.*

`nlp_create_qkv_heads_decode` emits that layout directly out of the fused projection, L1-sharded, so
K and V arrive already in the memory config the cache write takes and Q already in the one the
decode SDPA takes — no permute at either end. **`14.909 → 13.975 ms/token` (−6.3%), device 269.45 →
261.85, PCC bit-identical.**

Note this is the op it *declined* two attempts earlier as "a contract change across the cache write
and SDPA, so it belongs in its own attempt rather than riding along." It came back for it. The
deferral was not avoidance.

**The general lesson — and the tool has now found it twice.** Buried in that commit:

> *That also drops the rotation from 160 tiles to 4: in the old layout 31 of every 32 rows were
> tile padding.*

A decode step is one row. In TILE layout one row occupies a 32-row tile, so **any decode-path op
that has not been given a decode-shaped input does 32× the work it needs to.** This is the same
finding as O4 (`ttnn.argmax` scanning ~32× the bytes because `ttnn.linear` handed it a TILE row).
Two independent discoveries of one pathology in one run.

**Worth a systematic pass on the hand-port:** for every op on the decode path, check whether its
input is a padded tile row. It is a class of waste that profiles as "this op is slow" rather than
"this op is doing nothing," which is why it survives casual inspection.

#### An apparent fourth agreement — WITHDRAWN two commits later, by the tool itself

It initially declined `paged_fused_update_cache`, reasoning that the op parallelises K and V across
**disjoint cores** and rejects operands sharing one, and at batch 1 head-creation puts K and V on
the same single core. That matched `NOTES.md [gpt-19]` / §6.44, which **deleted** `_V_SHARD` —
existing solely to let that op accept K and V — because on Blackhole *"that fused write is 0.687
ms/step SLOWER than two plain writes."* I recorded it as a fourth independent agreement.

**It is not. See O4k — the tool came back and landed the fused write.**

**This also arms O4f's open question.** §6.44 records the silent failure mode that went with
`_V_SHARD`: *"RoPE on a core whose cos/sin table lives elsewhere returns 3.4e38 from uninitialised
L1."* That is precisely the hazard waiting for anyone sharding a 40-head rotation per O4f — the
hand-port's own notes already document the trap.

### O4h — the bottleneck the previous fix created, and the loop catching it

Immediately after O4g, head creation became the largest open gap — an op producing **12 tiles** was
costing **33 µs**. The diagnosis:

> *`nlp_create_qkv_heads_decode` is batch-parallel, so at batch 1 its outputs live on ONE core — and
> handed an interleaved operand, that one core had to pull the whole fused row, 192 tiles, out of
> DRAM by itself.*

The projection had just written that row across 32 cores' L1, and the op's sharded program factory
reads a width-sharded operand from exactly there. Feeding it directly turns a single-core DRAM read
into a fan-in over L1:

```
head creation   33 -> 5.9 us/call   (8.44 -> 1.50 ms)
rotation         7.35 -> 3.14 ms    (follows onto the same shard)
datamove        26.8 -> 14.6 ms
                13.975 -> 13.277 ms/token (-5.0%)   PCC bit-identical
```

This is the measure → attack → re-measure loop doing exactly what it is supposed to: the previous
win moved the bottleneck, and the next round found where it went.

---

## ★ THE PATTERN — one pathology, found three times, and the best thing to take from this experiment

Three of the tool's largest wins are the same underlying bug wearing different clothes. **Ops
written for a batch or sequence dimension degenerate when decode gives them neither.** Each profiles
as "this op is slow", never as "this op is doing nothing", which is why all three survived a human
pass:

| where | what decode gave it | what it did | cost |
|---|---|---|---|
| **O4** `ttnn.argmax` | a TILE row | took the **single-core** path (dispatch keys off input LAYOUT) and read the row padded to 32 | 51.3% of device time |
| **O4g** RoPE / rotation | a TILE row | processed **160 tiles where 4 had data** — 31 of every 32 rows were padding | part of −6.3% |
| **O4h** `nlp_create_qkv_heads_decode` | batch 1 | **batch-parallel** op collapsed to ONE core, which then pulled 192 tiles from DRAM alone | 33 µs for a 12-tile op |
| **O4j** LM-head vocab-block join | a TILE row | concatenated 4 blocks of **2 MB where 64 KB was real** — again 31 of 32 rows | 0.080 ms/token, over half of O4i's win |

The unifying rule: **at batch 1, seq 1, a "parallel" op may be running on one core, and a "small"
tensor may be 32× its logical size.** Neither is visible in the source — both are properties of what
the op does with the shape it is handed.

**O4j is the clearest illustration**, because the fix costs nothing at all. Only row 0 of those
blocks is real — the decode activation is one token padded to the tile — so untilizing each block
*before* the join turns it from 2 MB into 64 KB and the join moves **256 KB instead of 8 MB**. And
it is not even an added step: `_greedy_token` already has to untilize to reach the multi-core argmax
path (O4), so this only moves existing work in front of the join instead of behind it. **The O4 fix
is what made O4j free** — the wins compound.

**Recommended action on the hand-port:** a systematic decode-path audit against this rule. For every
op in the decode step, ask (a) does its parallel path key off layout or batch, and does decode
satisfy that? and (b) is its input a padded tile row? The tool found three instances in one model;
there is no reason to think a hand-written port has zero.

### O4i — a real bandwidth finding, on a trade the hand-port should NOT take

Diminishing returns begin here: **−0.4%** (13.277 → 13.228). The diagnosis is still good —

> *The LM head was the worst matmul in the model: 226 MB of bfloat4_b weight at **256 GB/s**, half
> the board's DRAM bandwidth, where the layer projections manage **340–360**. The reason is
> placement, not size — an interleaved buffer round-robins its pages across all eight banks, so each
> of the 128 cores gathers its column slice from every bank at once.*

Width-sharding the weight in DRAM makes each core's slice contiguous in **one** bank, which is what
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` reads. Genuinely useful knowledge: an
interleaved DRAM weight can cost ~30% of achievable bandwidth purely through bank placement.

**But the cost is a second copy of the weight — 226 MB — because the DRAM-sharded kernel requires an
in0 height of exactly one tile, so prefill (logits at every prompt position) must keep the
interleaved copy.** The tool judges that against 32 GB of board DRAM and takes it.

**Do not transfer this one.** The hand-port runs three blocks co-resident on the same board and
treats headroom as a real budget; spending 226 MB for 0.05 ms/token is the wrong side of that trade,
and Block 1's LM head is not even on the hand-port's critical path (Block 2 consumes the hidden
state, per §6.8's semantic head). Recorded as a **bandwidth lesson worth keeping and a change worth
declining** — the placement insight generalises to any large interleaved weight; the duplication
does not.

### O4k — §6.44's reversal may itself be CONDITIONAL, and this is the finding with the most at stake

**`12.890 ms/token` (−1.9%), PCC unchanged.** The tool went back to the fused cache write it had
declined, and its solution is verbatim the hand-port's own N150-era trick:

> *Moving V one core over is enough to make it legal, and a 2 KB shard move is far cheaper than the
> launch it buys back.*

Compare `NOTES.md [gpt-24]`: *"V is moved to core (1,0) first because the op refuses an overlap."*
Same fix, found independently.

**The conflict.** §6.44 measured the fused write **0.687 ms/step SLOWER** on Blackhole and deleted
the machinery. The tool measures it **faster** on Blackhole — 0.402 → 0.210 ms/token across 26
layers. Both are Blackhole p150b. Both are this model.

**CORRECTION (2026-08-14).** I first proposed that §6.44 was conditional on a layout the hand-port
had not adopted — the decode-native `nlp_create_qkv_heads_decode` output that O4g introduced — and
called this the highest-value item to test. **That mechanism is wrong.** Reading
`ttnn_voxtral_gpt.py::_layer_step` directly: the hand-port **already calls
`nlp_create_qkv_heads_decode`**, and already feeds it a sharded operand
(`to_memory_config(reshape(qkv), _QKV_SHARD)`). It has both halves of what I claimed it lacked.

So the honest position is narrower and more interesting: **two careful measurements of the same
change, on the same hardware, against materially similar layouts, disagree.** §6.44 records the
fused write losing 0.687 ms/step; the tool records it winning 0.402 → 0.210 ms/token. Both moved V
one core to satisfy the disjoint-operand rule. I cannot account for the difference from the
artifacts I have.

Still worth an A/B — a 0.19 ms/token swing is real either way — but as an *unexplained
disagreement*, not as a mechanism I have identified. Downgraded accordingly in the ranked list.

#### A fifth agreement, in the same commit's rejected attempt

It tried fusing the MLP's SiLU into the gate projection and reverted it (+13 ms):

> *`activation=` alone does not fuse — with no program config ttnn appends a `unary_chain` op, which
> is the same launch under another name.*

`NOTES.md [gpt-26]` says exactly this: *"`activation="silu"` never fused; `fused_activation` does."*
Independently confirmed. It also added a detail the hand-port's note does not have: naming a core
grid to reach the fused path made gate/up **43.35 → 57.50 ms on the same 96 cores**, because *"the
router's auto-derived config for a named grid is not the one it picks for itself."*

### O4l — a SIXTH agreement, arrived at through a different knob

Handed no program config, `scaled_dot_product_attention_decode` spends the whole grid: at batch 1
with 8 KV heads that is **16 cores per head, 128 active**. Its diagnosis is the pattern again —

> *Sixteen ways is far too fine for a 256-deep cache — 16 positions per core, half a tile of work —
> and the kernel then pays a 4-round tree reduction ACROSS those cores to put each head back
> together. **The reduction, not the read, was what the op cost.***

It swept `max_cores_per_head_batch`: `default(16) 5.29 / 4 → 3.25 / 2 → 3.04 / 1 → 2.99 ms`.
Monotone, *"which is the tell that the reduction dominates throughout."* `12.890 → 12.633 ms/token`.

**The hand-port is already there, by another route.** `_SDPA_PRG` passes
`compute_with_storage_grid_size=CoreCoord(8, 2)` to the decode SDPA — 16 cores total, 8 KV heads,
so **2 cores per head**. The tool swept a per-head knob and landed on **2**. Same effective
parallelism, reached from opposite directions. (`[gpt-21]` records the hand-port also found faster
settings that were *"NOT SAFE — position sweep"*, a correctness dimension the tool's single-length
gate cannot see.)

Worth noting the tool took **2 rather than 1** despite 1 measuring faster: *"the last step is worth
0.05 ms, whole-model came out marginally better at 2, and 2 still has somewhere to go when a deeper
cache makes the read matter again."* It declined 0.05 ms to keep headroom at longer context — a
judgment about future conditions, not a greedy pick.

### O4m — a structural change silently invalidated an earlier knob, and it went back for it

> *The 32-core cap was tuned when Q, K and V were three separate projections. Fusing them into one
> 3072→6144 read widened N and moved the optimum, but the cap stayed where it was.*

Re-swept: `16 → 17.47 / 32 → 15.98 / 48 → 13.61 / 96 → 35.71 ms`. **48 wins by 15% over the value in
place.** And the cliff at 96 is the pathology `down_proj` already documented in O4b: `in0_block_w`
falls to 1, so each core walks 96 sequential single-tile k-blocks to produce 2 output tiles with
nothing to overlap the reduction against.

**The meta-lesson, and it applies directly to the hand-port: a structural change invalidates every
knob tuned before it, silently.** Nothing errors; the old value simply stops being optimal. The
hand-port's `_MM_GRID = (12, 6)` was fixed at §6.52 and has since been through §6.65 (traced loop),
§6.67 (sharded norm) and §6.72 (head split). It has not been re-swept since. That is the same
staleness this commit found, and it sharpens O4b: **re-sweep against the current structure, not from
the historical setting.**

### O4n — the hand-port's answer is better, and the tool's own note says why it couldn't reach it

The SiLU was a standalone unary fetching 9216 values out of DRAM and writing them back *"to do
almost nothing — 26 launches per token."* The tool's fix: the product is its only consumer and a
binary op can apply a unary to an input **as it reads it**, so `silu(gate) * up` becomes one launch.
`12.427 → 12.352`.

**The hand-port does not have that unary at all.** `_PRG_W1 = _mm1d(2, 4, UnaryWithParam(SILU))`
puts `fused_activation` inside the matmul program config, so the activation happens **in the
projection**, with no separate launch to fold anywhere. That is strictly better than folding it into
the consumer.

**And the tool explains its own gap**, in the attempt it reverted:

> *Deliberately not `ttnn.linear(activation="silu")`: with no program config to put it in, ttnn
> appends a `unary_chain` op — the same launch under another name — and **naming a core grid** to
> reach the genuinely fused path costs far more than the unary did (measured: gate/up 43.35 → 57.50
> ms on the same 96 cores, because the router's auto-derived config for a named grid is not the one
> it picks for itself).*

That is the whole story: the tool reaches for fusion by **naming a grid and letting the router
derive a config**, and the derived config is worse than the router's own default. The hand-port
writes the **full** `MatmulMultiCoreReuseMultiCast1DProgramConfig` by hand — `in0_block_w`,
`per_core_N`, `out_subblock_w`, and `fused_activation` together — so it gets the fusion without
inheriting a bad config. `NOTES.md [gpt-26]` records exactly this: *"`activation="silu"` never
fused; `fused_activation` does."*

**→ Tool defect, and a concrete one for the PR (see F12).** Not a wrong answer — a missing lever.

### F12 — the fusion rung reaches for a grid when it should reach for a program config

**Status: found 2026-08-14** · severity: leaves activation fusion unreachable, and misprices it as a
loss · reported: not yet

When the ladder wants to fuse an activation into a matmul, it does so by naming a core grid and
letting ttnn's router derive the rest of the program config. Measured cost on this model: **gate/up
43.35 → 57.50 ms on the same 96 cores** — a 33% regression that has nothing to do with the fusion
and everything to do with the derived config. The attempt is then correctly reverted, and the
catalogue records activation fusion as a **loss**, when the lever simply was not pulled.

**Fix:** when fusing an activation, emit a complete program config (`in0_block_w`, `per_core_M/N`,
`out_subblock_h/w`, `fused_activation`) rather than a grid. The tool already builds exactly such
configs on the `grid` rung — O4b/O4m sweep `in0_block_w` and `per_core_N` directly — so the
machinery exists; the fusion rung just does not use it. The hand-port reached +0 launches this way
and the tool reached +1; the difference is one code path.

### O4o — a local sacrifice for a global win, plus a ttnn fact worth knowing

> *At decode the sharded norm converted its result back to interleaved on the way out and the
> projection converted it straight back in — two launches per layer for a tensor that never needed
> to leave L1.*

The fix requires both to agree on a grid, so the norm **moves off its own optimum** — swept
`8 → 4.15 / 32 → 3.72 / 96 → 4.35 ms`, an interior minimum at 32 — and pays `3.72 → 3.98 ms` at 48
to sit on the projection's grid, against **0.80 ms of layout ops removed**. Taking a 0.26 ms local
loss for a 0.80 ms global win is a trade a per-op ladder is not obviously able to make, and it made
it.

**The ttnn fact, which is load-bearing and not obvious:** `to_memory_config` does **not** treat
"already in the requested config" as a no-op — **it dispatches a copy**. Any code that defensively
normalises memory configs is paying for launches it may not need.

**This is take-item #4 for the hand-port** — `_norm` returns `sharded_norm(x, gamma, NORM_EPS, _L1)`,
i.e. it shards internally and hands back **L1 interleaved**, and `_layer_step` feeds that straight
into `ttnn.linear(..., DECODE_PRG["wqkv"])`. Same round trip. See the ranked list for the grid catch.

#### And [gpt-28] is the precedent that makes re-opening rejections reasonable

The hand-port has already lived through exactly this. `NOTES.md [gpt-28]`:

> *the decode RMSNorm is width-sharded again, +5.399 ms/frame. **6.39/6.40 rejected this at +4.4 ms
> WORSE, but that cost was the RESHARD DISPATCH, which 6.65 traced away.***

A rejection that was correct when measured, invalidated by a later, unrelated change, and reversed.
That is documented precedent in the hand-port's own log for the general claim behind re-open items
#4–#6: **a rejection is only valid under the conditions it was measured in, and structural changes
move those conditions.** §6.67 is the proof that this repo already knows it.

### O4p — and the thing this whole run is actually demonstrating

> *gate/up kept ttnn's default routing on the strength of an old sweep: seven core counts, all lost,
> "the plan's two reshard ops cost more than they buy". **That accounting was correct and is now
> obsolete.*** *The norm ahead of this block emits its result IN a 48-core width shard, and gate and
> up read the SAME activation — so the plan's input reshard is not two ops, or even one. It is zero,
> and what is left is the matmul routing on its own, which is the part the old sweep could never see
> separately.*

`43.28 → 41.32 ms` in the slice, `12.299 → 12.038 ms/token`.

**This is the fourth time in one run that a correct rejection went stale**, and it is the most
transferable observation in this document:

| | rejection | what invalidated it |
|---|---|---|
| O4m | QKV grid capped at 32 | fusing Q/K/V widened N |
| O4k | fused K/V write unavailable | (V relocation made it legal) |
| O4o→**O4p** | gate/up plan "costs more than it buys" | the norm now emits the shard for free |
| hand-port `[gpt-28]` | sharded norm, +4.4 ms worse (§6.39/§6.40) | §6.65 traced the reshard dispatch away → §6.67 reversed it |

**A rejection records a measurement under conditions, not a fact.** Every structural change silently
re-prices every knob and every earlier "no". The tool's real advantage over a human pass is not that
it finds cleverer optimizations — five of its wins the hand-port already had, and two of them the
hand-port does better (O4n, and the decode-mode RoPE). **It is that it keeps re-opening its own
closed questions, cheaply, in an order driven by fresh measurements.** A human does that once or
twice, when something prompts it; §6.67 and §6.72 are the hand-port's two.

**For the hand-port this is the standing recommendation behind items #2, #4, #5 and #6:** the
rejections in `STATUS.md` are dated, and the structure has moved underneath several of them.

### O4q — the shard chain, extended to the whole block

The last of the layout-chaining family, and the one that shows how far it goes:

> *The stream is one tile row of 3072 values that every op in the block already touches in L1, but
> it went back to interleaved between them, so each norm re-opened the same shard from DRAM — twice
> per layer, 26 layers deep. [...] Both norms in a block are built on the same dim, and so is the
> next layer's, so **the stream can stay in the shard the whole way down**.*

`12.038 → 11.928 ms/token`, then a further `→ 11.839` closing the loop from the other end — `o_proj`
and `down_proj` now emit the residual add's shard directly, so the stream is sharded end to end
through the block rather than only on the norm side. Folded into take-item #4 rather than listed separately, because for the
hand-port it is the same recommendation carried further: not just norm→QKV, but the residual stream
never leaving its shard across the depth of the model.

Worth noting the hand-port already solved the *adjacent* problem better — `[gpt-27]` passes the
residual as the matmul's `bias`, so the add itself costs no launch at all. That is orthogonal to
this and the two compose.

### O8 — the accept test has no exchange rate, and it shows at the end of a run

The last commit bought **0.012 ms/token (0.10%)** and cost **0.0129 of PCC** — 0.9903 → 0.9774, about
a quarter of the remaining headroom above the 0.95 gate, for a gain indistinguishable from noise.

Nothing in the ladder objects, because the accept test is:

```
faster?  AND  PCC still above the floor?   ->  keep
```

There is no notion of an **exchange rate** — no "is this gain worth this much accuracy?" So once the
structural ideas run out, a run will keep converting PCC headroom into arbitrarily small wins until
it reaches 0.95. The run's own arc shows the transition cleanly: twelve consecutive layout and
dispatch wins at **bit-identical PCC**, and then this.

**Suggested fix, and it is small:** require a precision-spending change to clear a ratio, not just
the floor — e.g. reject when `Δpcc / Δms` is worse than some rate, or simply require dtype-rung
wins to exceed a materiality threshold (the tool already has `material_gap_threshold_ms = 0.25`
for choosing targets; the same idea applied to *accepting* precision trades would have rejected
this). Cheap to add, and it is the difference between a run that stops at its best result and one
that grinds its accuracy down for noise.

**This is the mechanised form of §6.16.** The hand-port faced the identical question, computed the
exchange rate explicitly — *w2 costs 77% of the precision stack's accuracy for 15% of its speed* —
and handed back 2.5 ms. That reasoning has no place to live in the current design. Together with O1
(no per-weight dtype axis) this is the clearest gap between the tool and a careful human pass.

### F13 — the generated stubs swallow fast-path exceptions, so a perf regression passes the PCC gate

**Status: found by the agent itself, 2026-08-14** · severity: a broken optimization reports as
correct · reported: not yet

The emitted `tt/pipeline.py` guards its fast paths like this:

```python
try:
    ...fast matmul plan...
except Exception:
    self.lm_head_dram = None      # then fall through to plain ttnn.linear
```

The fallback computes **the same math**, so `check_pcc` passes. What is lost is only speed — and
PCC cannot see a performance-only failure. The guard converts an exception into a **silent policy
change** rather than an error.

**Measured instance.** Setting the LM head's output dtype to `bfloat4_b` made
`ttnn.to_layout(part, ROW_MAJOR_LAYOUT)` throw — block-float formats require TILE layout, and while
`bfloat8_b` is accepted, `bfloat4_b` is not. PCC still reported **ok at 0.9774**, while all 128
decode tokens ran the interleaved fallback: **device_ms 223.65 → 244.64**, and the op table showed
`32 x 3072 x 131072` at n=128 where `32 x 3072 x 32768` at n=512 should have been.

**Why it matters here specifically:** this tool's entire design premise is that *the AI proposes and
the harness verifies*. A guard that turns a failed optimization into a passing test defeats that
premise inside the generated code itself — the harness is verifying honestly, but it is being handed
a program that lies about which path it took.

**Fix, two options:** (a) do not emit bare `except Exception` around fast paths — let the failure be
loud during bring-up, and gate the fallback on an explicit capability check instead; or (b) if the
guard must stay (the RoPE commit's latched fallback exists for a real reason — an exception inside a
captured trace is fatal), then have `measure_candidate` assert the **expected op signature and call
count** are present in the profile, not just that the run was faster. The tool already parses that
table.

**Credit where due: the agent caught this itself**, diagnosed it from a collapsed call count in the
per-op table, recorded the general rule (*"never trust `check_pcc` alone after touching a guarded
fast path"*), and settled on `bfloat8_b`, which does not trip the guard.

### O5 — the ladder's escalation is real, and it gives up in the right place

On the LM head matmul it climbed `grid → dtype → shard → fidelity → structural → cpp` across 11
attempts, and the C++ rung came back *slower* (400.19 vs 397.27) and was reverted. It then moved on
rather than grinding. The rung ordering by `bound_by` behaved as documented.

### O6 — `weight_dtype` reads `null`, so the dtype rung is recommended blind

Every entry in `blocking_ops` reports `"weight_dtype": null` and the advice text renders as *"lower
the weight dtype (now unknown) to bf8_b/bf4_b"*. The profiler is not recovering the dtype actually
in use, so the tool cannot tell an untouched bf16 weight from one it already stepped down, and will
re-propose the same rung on a weight that is already at bf4_b. Minor — the agent notices from its
own diff — but it wastes attempts and weakens the report. **Worth fixing.**

### O7 — the throughput band can never stop the run

`termination_check` carries `"band_stop_disarmed": "divisor is an estimate (device census: 7.18 GB
resident at served dtype), not an exact param count"`. Reaching `IN_BAND` therefore does not stop
the loop, by design. Defensible — an estimated divisor should not end a run — but it means the
documented "stops when at the theoretical floor" exit is unreachable for any model whose parameter
count is inferred rather than declared, which is every model that arrives as a local directory.

---

### Blindness audit — evidence that the comparison is honest

The whole experiment is worthless if the tool saw the hand-port. Audited 2026-08-13 rather than
assumed. Method: grep the agent's full 6.7 MB tool transcript, and enumerate **every** file-access
tool call it made.

| check | result |
|---|---|
| `models/experimental/voxtral_tts` in the tool's checkout | absent (not in the working tree) |
| transcript hits: `repos/tt-metal` outside `-pr46283` | **0** |
| transcript hits: `experimental/voxtral_tts`, `ttnn_voxtral`, `STATUS.md`, `ONBOARDING.md` | **0** |
| transcript hits: the hand-port's headline numbers (`45.4`, `26.9`, `RTF`) | **0** |
| `WebSearch` / `WebFetch` calls | **0** |
| every `Read`/`Glob`/`Grep` target | all inside `pr46283`, except two memory files |

The two exceptions are `.claude/projects/-localdev-lserbedzija-repos-tt-metal/memory/`
`python-env-wrapper-fixes-planner-gates.md` and `perf-mcp-env-traps-look-like-device-faults.md`
(the latter written by the agent itself). That scope holds four memories, all created *during this
experiment* and all about the tool's own environment traps — `python_env` wrappers, `PYTHONPATH`,
the `ttnn` namespace-package trap. No optimization content. The hand-port's own memories are in a
different scope (`-localdev-lserbedzija/memory/`) which appears nowhere in the transcript.

Eight matches for a `6.6x` pattern were checked individually and are all coincidental decimals
(`316.6906`, timestamp fractions), not `STATUS.md` section references.

**Two honest qualifications.** (1) Both branches share the `tt-metal` remote, so the hand-port is
reachable from inside this checkout via `git show <sha>:<path>` even though it is not checked out —
it was never impossible to reach, it simply was never reached, and the transcript is the evidence.
(2) The real contamination vector was the operator, not the tool: what was handed to it was
tool-source fixes (F1/F2/F5), `conftest.py`, env wrappers, and the HF export. None encodes tuning,
and the tool justified its own wins from primary sources — the argmax finding cites
`argmax_device_operation.cpp:16` by line, not a measurement from the hand-port.

A write/read tripwire on the hand-port tree (1,424 files fingerprinted at `fa57362fe5`, clean) ran
for the remainder of the experiment.

---

## ★ TRANSFER RESULTS — what happened when the tool's findings were applied to the hand-port

The reason the experiment existed. Every precision-neutral finding was implemented against the
hand-port and measured on the same p150, with its own audio-tier gate (45 utterances, WER, MOS,
6 PCCs, ms_per_frame) run twice in one session so nothing is judged against a stored number.

Baseline, reproducing §6.71 to the noise floor: **ms_per_frame 27.751, RTF 0.3653, WER 0/894,
MOS 4.6101, 132/132 tests, 32 metrics with no nulls.**

| # | tool finding | outcome on the hand-port |
|---|---|---|
| O4o/O4q | residual stream kept in the norm's shard | **−0.99 ms/frame** — and **WER 0 → 2**, fails the gate |
| O4f | Q+K rotated in one call | **refused** — `nlp_create_qkv_heads_decode` returns q/k already `HEIGHT_SHARDED`; concat-then-rotate is a TT_FATAL |
| O4k | fused K/V cache write | **0.510 ms/token SLOWER** — §6.44 independently reproduced |
| — | SwiGLU product → `w2` as a sharded operand | **refused** — matmul rejects a sharded in0 |
| — | head-merge → `wo` as a sharded operand | **accepted, but slower AND wrong** (PCC 0.954, `nan` at 16 cores) |
| — | reshard the QKV activation once | **moot** — the hand-port already fuses q/k/v into one projection |
| O4b | per-op grid for `w2` | **~0.15 ms, under the instrument's resolution** — §6.52 stands |

**Nine further findings were already in the hand-port** (§6.67 sharded norm, §6.65 traced decode,
§6.52 program configs, fused `wqkv`, fused RoPE, `[gpt-05]` decode-native layout — worth **6.6
ms/frame** there against the tool's 0.93 — `[gpt-21]` SDPA core count, §6.72 head split, and
`[gpt-26]` `fused_activation`, which the hand-port does *better*, see F12).

### The headline, stated plainly

**Essentially nothing from this tool transfers to a mature hand-port of the same model.** That is
not a criticism of the tool — it is a statement about where its value lies. Its wins were real and
large *on its own output*: its `w2` sat at 96 cores, its RoPE was a seven-op chain, its Q/K/V wore a
prefill layout, its argmax was single-cored. The hand-port had already fixed all of those. What the
tool recovers is the distance between generated code and hand-tuned code — and by construction that
distance is zero once someone has done the hand-tuning.

### The one thing that DID transfer, and why it still failed

O4o/O4q is a genuine **0.99 ms/frame** — double the gate's 0.5 ms tolerance. It failed on WER
because the chain forced `_NORM_GRID` from 32 to 48 cores, which changes the RMSNorm's reduction
tree (48 partials, not 32). Not bit-identical, `decode_min_pcc` 0.999316 → 0.999288, and 2 of 894
long-form words flipped. **A variant pinning `per_core_N=3` on the two residual matmuls — which puts
their output on 96/3 = 32 cores, matching the norm's existing grid — is under test.**

### ★ O9 — what the 0.95 gate actually cost, measured in the metrics it cannot see

The tool's optimizer accepted `down_proj → bfloat8_b` (commit `074ec705a8`) on the only two
criteria it has: faster, and PCC still above 0.95. The hand-port evaluated the same change at
§6.16 and **handed the speed back**. This prices that disagreement by making the identical change
to the hand-port and running its audio-tier gate — everything else held fixed, both tags in one
session.

```
                    baseline      w2 -> bf8_b
wer_longform             0    ->      3        *** WORSE ***   (tolerance is 0)
codes_real_n            45    ->     59        *** WORSE ***   +31% code flips
codes_real_pct         5.2    ->    6.8        *** WORSE ***
decode_mean_pp        0.97    ->   1.09        *** WORSE ***
mos_min             2.6597    -> 2.4145                        -0.245 on the WORST utterance
mos_longform        4.6101    -> 4.5895                        within tolerance
ms_per_frame        27.751    -> 30.271        *** WORSE ***   SLOWER by 2.52 ms
rtf                 0.3653    -> 0.4113        *** WORSE ***
pytest                 132    ->    131        *** WORSE ***   the §6.16 guard fired
                                               8 metrics worse, 15 within tolerance
```

**Three of those four quality columns are invisible to the tool's gate.** It has no word-error
metric, no perceptual metric, and no exact-match check on discrete codes — and `mos_min` is the
worst-case utterance, which a mean-based gate would hide even if it had one. Its own PCC reading
for this change stayed comfortably above 0.95 throughout.

**And on this build the change is SLOWER.** §6.16 measured BFP8 `w2` as ~2.5 ms *faster* on the
N150; on p150 it costs 2.52 ms. So the trade is now loss-loss.

#### O9b — the tool re-opens its NOs but never its YESes

The sharpest part. In run 2 the tool's own diagnostic for this exact op reported:

> *this op takes the SAME 0.159 ms/call at bf16 and at bf8_b, so it is not bandwidth-bound and the
> limit is the shape of its k-reduction*

**It measured that `w2`'s dtype does not affect its speed — after having already spent accuracy
lowering that dtype in run 1 — and never went back.** Section O4p credits this tool for re-opening
four *rejections* in a single run, which is the best thing it does. But nothing re-opens an
**acceptance**. Precision is the only irreversible cost the ladder pays, and it is the one class of
decision never revisited when later evidence undermines it.

**Suggested fix, and it is symmetric with what the tool already does well:** when a measurement
shows a lever is inert on an op (same time at two dtypes, same time at two grids), re-open every
*applied* change that rung made to that op and try reverting it. A revert that costs nothing in
time and buys back accuracy is a strict win, and the ladder currently cannot express it.

### F14 — "producer emits the consumer's shard" needs to check the consumer's PROGRAM CONFIG grid

The most useful mechanism the tool found is also the one most likely to misfire, and the reason is
worth handing to the PR author directly:

> `memory_config=` on a matmul carrying a program config is **only a request for the layout**. The
> shard spec follows the matmul's OWN grid. Asking `DECODE_PRG["wo"]` (`_MM_GRID=(12,6)`,
> `per_core_N=2`) for a 32-core shard returns a **48-core** one, and the downstream norm refuses it:
> *"shard_spec.grid size 12x4 does not fit within program_config grid 8x4"*.

So a tool applying this lever has three choices, and it should know which it is making: move the
consumer to the producer's grid (what the tool did — and what perturbs a reduction's arithmetic),
move the producer to the consumer's (`per_core_N`, arithmetic-preserving), or give up. **The tool
took the first without recording that it had changed the numerics** — its PCC gate at 0.95 could not
see a 2-word WER shift, and nothing in the ladder flags "this lever altered a reduction tree".

**Suggested fix:** when a shard-chaining lever changes a *reduction's* core count (norm, softmax,
argmax, any tree reduce), record it as a precision-affecting change rather than a layout one, and
prefer the `per_core_N` route when it exists.

### Process note — a measurement error of mine, and how it was caught

My first fused-K/V-write measurement showed it **8.72 µs faster** and I nearly reported §6.44 as
overturned. The V relocation was hoisted **out** of the timed region, so it measured the fused write
as if V were already on a disjoint core — which it never is, because head creation puts K and V on
the same core at batch 1. With the move inside the loop the same comparison reads **19.63 µs
slower**, and the V move alone costs 23.49 µs. §6.44 recorded 0.687 ms/step; the corrected
measurement says 0.510 ms/token. Same sign, same magnitude, independently reproduced.

The general form: **when testing a fused op that requires an operand to be relocated, the relocation
is part of the cost unless the producer can emit it in place.**

---

## ★★ WHAT THE OPTIMIZER IS MISSING — the analysis this experiment was for

The ladder, as defined in `run.py:102`:

```
knob:grid -> knob:fidelity -> knob:dtype -> knob:shard -> structural -> tt-lang -> cpp
                                            structural levers named: trace | kv-cache | gather
```

### 1. Coverage map — the hand-port's shipped wins against the rung that would find them

| hand-port win | worth | rung that finds it |
|---|---|---|
| sharded decode RMSNorm (§6.67) | **−5.399 ms/frame** | `knob:shard` ✅ found it |
| decode matmul program configs (§6.52) | **−5.06** | `knob:grid` ✅ — but the silu half is **unreachable**, see F12 |
| whole frame graph traced (§6.65) | **−4.244** | `structural:trace` ✅ found it |
| sdpa for Block 2's attention interior (§6.45) | −2.555 | ❌ **no rung** — swap a hand-rolled interior for a library primitive |
| residual as matmul bias (§6.62) | −1.918/step | ❌ **no rung** — algebraic rewrite |
| in-place elementwise, Block 1 (§6.47) | +0.929 | ❌ **no rung** — allocation elimination |
| two plain KV writes + 1-core qkv shard (§6.44) | +0.907 | ⚠ `knob:shard` in reverse — the rung only ever ADDS sharding |
| in-place elementwise, Block 2 (§6.48) | +0.790 | ❌ **no rung** |
| hand-rolled 9-op head split (§6.72) | −0.775 bit-exact | ❌ **no rung** — this is DE-fusion |
| `_SDPA_PRG` (§6.46) | +0.197 | `knob:grid` ✅ |

Plus everything inherited and still shipping — **CFG batch-fold into rows (2.23×)**, qkv weight
fusion, `SCALE` baked into wqkv's q rows, `_trunk` projecting before it narrows, the semantic argmax
**on the host**, the codec's gather-based pad. **None of those has a rung either.**

**By magnitude the ladder reaches about two-thirds of the device-time wins and none of the algebraic
ones.** And that is the generous reading — it assumes `knob:grid` reaches `fused_activation`, which
F12 shows it does not.

### 2. The real finding: `structural` is where the value was, and it is the least specified rung

Every large win the tool itself landed in run 2 came from `structural` — fused QKV, the decode-native
layout, one-call RoPE, the shard chain. Yet the rung names only **three** levers (trace, kv-cache,
gather), none of which is any of those. **The agent improvised all of it.** That is why the run was
good and also why it is not reproducible: the ladder's most valuable rung is a blank cheque.

**Recommendation: populate `structural` with a named sub-catalogue**, each with a firing condition
and a guard. Every one below is drawn from measured evidence, with both signs where they exist:

| sub-lever | fires when | evidence |
|---|---|---|
| **`bias-fold`** | an elementwise add's only consumer is a matmul → make it that matmul's `bias` | §6.62, **−1.918 ms/step**. Guard: one tile of rows only — a bias broadcasts and is silently wrong on prefill |
| **`in-place`** | an elementwise operand is dead immediately after → use the `_` variant | §6.47 + §6.48, **+1.72 ms** combined; allocation was ~12 µs of a ~65 µs op |
| **`reorder project↔narrow`** | a projection is adjacent to a slice/gather/duplicate → try BOTH orders | `_trunk` projects before narrowing (**win**); §6.34 project-then-duplicate is **0.785×, a loss**. Both signs — must be measured, never assumed |
| **`weight-bake`** | a constant scalar multiplies a projection's output → fold it into the weights at load | `SCALE` into wqkv's q rows |
| **`weight-concat`** | sibling projections consume the same activation → concatenate at load | the tool DID find this (O4c); worth naming so it is not re-derived |
| **`de-fuse`** | a library op can be expressed as primitives, **and trace is applied** | §6.72, **−0.775 ms bit-exact**, 9 ops beating 1. The tool found this too (O4e) — from the opposite direction, and only by accident |
| **`library-swap`** | a hand-rolled interior matches a library primitive's contract | §6.45 sdpa, **−2.555 ms** |
| **`revert`** | a previously-applied config → try REMOVING it | §6.43: `wo`'s tuned config was inert, and deleting it was bit-exact |

### 3. Three structural blindnesses — things the design cannot express

**(a) The host is forbidden as a destination.** `test_e2e_pipeline` asserts `torch_ops == 0` and
`test_forward_fires_no_host_op` asserts zero host aten ops. So "this work belongs on the host" is
**inexpressible**. §6.8 moved a semantic argmax to the host for **1.439×** — an 8320-value reduce
that already ended in a D→H copy, so it added no round trip. §6.50 is the control: moving the other
three host steps ON device is 7–29× slower. A tool that can only ever move work onto the chip will
never find either. **Suggested fix:** allow a host fallback when the op is already adjacent to a
transfer, and gate it on total wall time rather than on op location.

**(b) Op count is treated as monotone-good.** Every lever reduces launches. §6.72 and the tool's own
O4e both show the reverse winning once trace has removed launch cost — *"dispatches fell 3413 → 2867
yet it got slower; these were view ops doing no work."* **De-fusion should be a scheduled rung after
`structural:trace`, not a lucky accident.**

**(c) PCC at one length is the whole correctness model.** `[gpt-21]` records SDPA settings that were
faster and **"NOT SAFE — position sweep"**; §6.31 holds back a 2.079× bf16 semantic head because
*"one flip redirects the whole utterance"*. Neither is visible to a single-length PCC gate. **A
generative model needs its gate run at several positions/lengths**, and discrete outputs (argmax,
codes) need an exact-match check, not a correlation.

### 4. Already filed, restated here because they belong to this analysis

- **O1** — no per-weight dtype axis. §6.16 kept `w2` in bf16 while everything else went BFP8, because
  w2 alone was 77% of the precision stack's accuracy cost for 15% of its speed.
- **O8** — the accept test has no exchange rate: `faster AND above the floor` keeps a 0.10% gain that
  costs a quarter of the PCC headroom.
- **F12** — the fusion rung reaches for a grid when it should emit a full program config, so
  activation fusion is recorded as a loss when the lever was never pulled.
- **F14** — shard-chaining must check the consumer's program-config grid, and must classify a change
  to a REDUCTION's core count as precision-affecting.

### 5. What to keep — the tool's genuine structural advantage

Worth saying plainly, because the rest of this section is criticism. **The tool re-opened its own
closed questions four times in a single run** (O4m, O4k, O4o→O4p, plus the stale 32-core cap) and
that is where its best late wins came from. The hand-port did it twice in 74 experiments (§6.67,
§6.72) and its own ledger calls the rule out: *"a rejection is stale when its premise is a cost
someone has since removed."*

**That behaviour should be promoted from emergent to designed:** when any structural change lands,
mark every knob measured before it as stale and re-open it. The tool nearly has this already — it is
the single thing it does better than a careful human, and it is currently an accident of the agent's
judgement rather than a property of the ladder.

---

## ★★ THE THREE-BLOCK EXPERIMENT — packaging Voxtral-TTS as one HF model

The Block-1-only run was criticised, correctly, for testing the model as a text LM when the
deployment drives it with audio. So the whole pipeline was packaged as a single
`trust_remote_code` HuggingFace model and handed to the tool. **It works**: 4.00 B parameters,
three blocks, bit-exact to the reference (`torch.equal`, maxdiff 0.0 on all three), self-contained,
text ids -> 24 kHz audio in one `forward`.

### S4 — six packaging defects, all in `transformers`, all hit in one afternoon (OURS/upstream)

Not tool defects, but every one is a barrier between a real research model and this tool, and the
tool's adoption depends on people clearing them:

| # | what | how it fails |
|---|---|---|
| 1 | `save_pretrained` drops `auto_map` unless `register_for_auto_class()` was called | later load says *"Transformers does not recognize this architecture"* — blames the architecture, not the missing key |
| 2 | `trust_remote_code` does not support **subpackages** | `from .reference import x` is resolved as a file `reference.py`; all custom code must be flat |
| 3 | its import scanner only matches `from .module import name` | the `from . import module` form is **invisible**, so those files are never copied and fail at runtime |
| 4 | only `.py` files are copied to the module cache | any asset resolved from `__file__` (voice presets, `params.json`, vocab) breaks, because `__file__` is now the cache |
| 5 | `ModelOutput` subclasses need `@dataclass` | fires on the **return** path, after the entire forward has already run |
| 6 | `nn.ParameterDict` forbids `.` in keys | every real checkpoint uses dotted names |

### F15 — `plan` and `compat` disagree about what the model IS

Same model, same directory, two stages of the same tool, run minutes apart:

```
plan   :  Category: TTS  (pipeline_tag=None, library=transformers)
          Category guidance (TTS): Text-to-speech. Closest template: models/demos/qwen3_tts/
          CONFIDENCE: LOW

compat :  Architecture: unknown / non-LLM (fingerprint: unknown)
          Overall verdict: ARCHITECTURE NOT RECOGNIZED (non-LLM) — no confident block plan
          Summary: 0 ready / 0 partial / 0 missing
```

`plan` identified it as TTS **from the config alone** (`pipeline_tag=None`, so not from Hub
metadata) and even named a TTS template it could copy. `compat` then declared the architecture
unrecognised and emitted an **empty** block table. A user reading these in order gets a green light
and then a wall, with no explanation of which stage to believe.

**Credit where due:** `plan`'s `CONFIDENCE: LOW` here is *correct* and is an improvement on the
Block-1 run, which printed `CONFIDENCE: HIGH` while admitting it had omitted the KV term (O3).

### F16 — the block table degrades to EMPTY, which reads identically to "nothing needed"

Block 1 alone produced `11 ready / 0 partial / 0 missing`. The three-block model produces
`0 ready / 0 partial / 0 missing` — not "these blocks are missing", but **no analysis at all**,
printed in the same format. Combined with **F2** (the READY verdict can never fire because the
predicate is `lambda _: []`), the summary line is now unable to express either success or failure
for a custom architecture: 0/0/0 is what you get whether everything is supported or nothing was
examined.

**Suggested fix:** distinguish *not analysed* from *analysed, nothing missing*. A third state, or
simply refusing to print the summary line when the analyser bailed.

### F17 — machine-readable structure is declared and never read

The config states the model's shape in fields designed to be read:

```json
"task": "text-to-speech",
"block_stacks": ["backbone", "flow", "codec"],
"decode_input": "audio_code_embedding"
```

`plan` used `task`. **Nothing used `block_stacks`.** `compat`'s advice is
*"inspect subfolders (dit/, vae/, text_encoder/, ...) and bring up per-component"* — the Stable
Diffusion **folder** convention. This model's three stacks are `nn.Module` attributes, not
subfolders, and are named in the config. The tool has multi-stack support (its own commits
`emit-e2e: a multi-stack model must expose one depth knob per stack` and `G6 refuses a model whose
block stacks the profiler cannot see`), but discovery is folder-shaped only.

**Suggested fix, and it is small:** when the config declares stacks, walk those attributes. A model
that says what it is should not have to also be laid out in a particular directory shape.

### What still works on an unrecognised architecture — worth keeping

`compat` did not simply bail. It read the real config and produced genuine kernel-level findings:

- `ttnn.topk (sampling)` — `vocab_size=131072` needs a power of two **< 65536** for the multi-core
  path, else single-core, with the throughput consequence stated
- per-TP divisibility, correctly deriving that `TP=32` fails on `num_key_value_heads=8` while
  TP=1/2/4/8 are fine, and correctly framing it as *"rules out that mesh shape, not the model"*

So the kernel-constraint half degrades gracefully where the architecture half does not.

---

## ★ F18 — the architecture gate tests the model's NAME, not its structure (with a tested fix)

`compat` refused the three-block model outright. The chain, in `compatibility.py`:

```python
family = detect_family(cfg)              # matches cfg["model_type"] against hardcoded name lists
is_unknown = family.startswith("unknown")
fpr = arch_descriptor(model_type, architectures, is_encoder_decoder)
if not fpr.startswith("decoder-only"):
    return report                        # early return -> EMPTY block table
```

**Every input to that decision is a name.** `model_type`, `architectures`, `is_encoder_decoder`.
`model_type` is free text chosen by whoever wrote the config, so a model that is *structurally* a
Llama-family decoder is refused for being called `"voxtral_tts"` — while the config carries every
field the block checks actually read: `num_attention_heads`, `num_key_value_heads`, `head_dim`,
`intermediate_size`, `rope_theta`, `rms_norm_eps`.

**The intent is right and worth preserving.** The checklist it runs is the LLM-decoder one; run it
against a VAE and it would report *"GQA attention: ready"* for a model with no attention. Families
genuinely differ — MLA, SSM and MoE need different handling. Refusing to guess beats a confident
plan for the wrong architecture.

**The implementation tests the label instead of the thing.**

### Tested fix

Added `_looks_like_decoder(cfg)` — requires the four fields the checks read, plus one
attention-shape hint (`num_key_value_heads` or `head_dim`) and one position hint (`rope_theta`,
`rope_parameters`, `rope_scaling` or `max_position_embeddings`), so a bag of integers does not
qualify. Used as a fallback in `detect_family` after the name lists.

Same model, same command:

```
BEFORE   ARCHITECTURE NOT RECOGNIZED (non-LLM)      0 ready / 0 partial / 0 missing
AFTER    Llama-family causal LM (INFERRED from config fields, model_type='voxtral_tts'
         is not a known name; config declares 3 stacks: backbone, flow, codec)
         Overall verdict: READY                     10 ready / 0 partial / 0 missing
```

Also added `declared_stacks(cfg)`, which reads `block_stacks` — the field F17 showed nothing was
reading.

### The fix is INCOMPLETE, and the remainder is the real recommendation

`Overall verdict: READY` is **wrong**. It analysed the *backbone* and declared the whole *model*
ready. The flow matcher and the codec are not covered by that checklist at all, and the codec is
built on `conv1d` — which §6.13 records as the op that caused this port's hang. So the patch trades
"refuses to analyse anything" for "analyses one stack and overclaims for three": **F16 in a new
costume.**

The correct output is per-stack, and the tool already has the names:

```
backbone : 10 ready / 0 partial / 0 missing
flow     : NOT ANALYSED — no checklist for flow-matching
codec    : NOT ANALYSED — no checklist for neural codecs
```

**Three states, not two: supported / missing / not-analysed.** That is the change worth making —
it is what lets the tool be honest about scope instead of choosing between silence and
over-confidence, and it is the same gap F16 identified from the other direction.

*(Note: the `READY` verdict can only fire at all because F2 was fixed in this checkout —
`lambda _: []` -> `lambda _: True`. Unpatched, that verdict is unreachable.)*

---

## ★★ F19 — template dispatch silently runs a DIFFERENT model, and the template can be the tool's own earlier output

The highest-severity finding of the three-block experiment, because it produces a complete,
plausible-looking run whose results are not the model you asked for.

Handed `/localdev/.../voxtral-tts-full` (three blocks, 4.00 B params), `auto-up` printed:

```
  Step 2/6  Scaffold the demo folder for /localdev/.../voxtral-tts-full
  GENERIC LLM BACKEND. No per-model tt/ folder needed. Skipping scaffold and
  routing directly to `prepare --execute`.
  ALREADY SUPPORTED via tt_transformers/simple_text_demo. Skipping scaffold.

  BRING-UP TEMPLATE — /localdev/.../voxtral-tts-full on P150 mesh [1,1]
  Backend: Voxtral TTS Backbone (mistral decoder)
  Runs canonical HF id out-of-the-box: /localdev/.../voxtral-tts-backbone     <-- DIFFERENT MODEL
  Compat verdict: READY
```

It dispatched to `models/demos/voxtral_tts_backbone/`, whose demo loads a **different checkpoint
directory** — the Block-1-only export from the previous experiment. Left overnight this yields a
finished run, with metrics, for a model nobody asked about.

**Three things make this dangerous rather than merely wrong:**

1. **The warning is buried.** It exists — *"adapt encoder/decoder/IO ... before expecting correct
   outputs"* — as a prose bullet under `Notes:`, below a header stating the other model's id as a
   feature (*"Runs canonical HF id out-of-the-box"*) and directly beneath `Compat verdict: READY`.
2. **The template is the tool's OWN prior output.** `models/demos/voxtral_tts_backbone/` was
   generated by an earlier run of this same tool. It found its own artifact and reused it as a
   template for a different model. Nothing checks that a template's `canonical_hf_id` is the model
   under port.
3. **Skipping scaffold is silent about consequence.** "No per-model tt/ folder needed" reads as an
   optimisation. What it means is that no port happens at all.

**Suggested fixes, cheapest first:**

- **Refuse, do not warn, when `template.canonical_hf_id != model_id`** unless an explicit
  `--allow-template-substitution` is passed. A different checkpoint is not a detail to note.
- Never treat a directory the tool generated as a template for a *different* model id.
- Say what skipping scaffold costs: *"no TTNN port will be produced for this model"*.

### F18 (correction) — my own fix caused F19 to fire, and that is the sharper finding

The routing gate is:

```python
if compat.overall == "READY" and not _missing and not _partial and _generic_backend_picked:
    _route_via_generic_llm = True          # -> skip scaffold, run a sibling demo
```

F18's patch made `compat` return `READY` for a family inferred from config structure. That flipped
this gate. **So the patch did not merely overclaim in a report — it caused the port to be skipped
entirely.** Recorded against myself because it is the clearest possible demonstration of the
underlying design problem: `READY` is doing two incompatible jobs, "the checklist passes" and
"this model needs no work", and any change to the first silently changes the second.

**Corrected patch.** Generic routing now additionally requires that the family came from a known
`model_type` **and** that the config declares at most one stack, and it states why when it declines:

```
NOT routing to the generic LLM demo: family was INFERRED from config structure, not a
known model_type; config declares 3 stacks (backbone, flow, codec) and the block
checklist covers the decoder stack only. Scaffolding this model's own stubs instead.
```

`CompatReport` now carries `family_inferred` and `declared_stacks` so the caller can tell an
inferred verdict from a matched one — which is the general form of the fix: **a verdict should
carry how confident it is and what it covered, not just what it concluded.**

---

## ★★★ F20 — the meta-plan already knows. It is wired to stdout, not to control flow.

**This is the most actionable finding in this document, and the cheapest to fix, because the
analysis already exists and is good.**

Before failing, `auto-up`'s advisory meta-plan wrote this about the three-block model — unprompted,
with no access to anything in this file:

> *This is a three-stage TTS pipeline (an LLM-style 26-layer backbone, an audio codec decoder, and
> a flow-matching vocoder) being routed into `tt_transformers/simple_text_demo` purely by
> category-default*

Its six listed risks are, one for one, the findings the rest of this document arrived at the hard
way:

| the meta-plan said | this document filed it as |
|---|---|
| *"Backend selection is a category-default match, not a genuine 'voxtral_tts' match — simple_text_demo assumes causal text-token generation"* | **F19** (template dispatch runs a different model) |
| *"Early success graduating the 26-layer backbone (the 'easy' REUSE/ADAPT part) may create false confidence that masks the fact that the two hard components are architecturally out of scope"* | **F16 / F18** (READY overclaims scope) — stated better than I stated it |
| *"Do not evaluate codec_decoder/flow_matching against simple_text_demo's logit-based PCC harness — they need a waveform/mel-level comparison harness, which doesn't exist in this backend"* | the **whole correctness-gate argument** (O9, §3 of the optimizer analysis) |
| *"Both audio-specific components show leaves=1, meaning the discovery tracer could not see inside them"* | **F17** (structure declared, not read) |
| *"flow-matching ... a multi-step numerical integration loop, not a fixed op graph — which the op catalog has no [coverage for]"* | the `structural` rung's missing sub-levers |
| *"variable-step sampling overlaps with the already-flagged unsupported 'DynamicShape' NEW-op category"* | — |

It then recommended, correctly:

> *"Cap auto-iterate retry budget on codec_decoder and flow_matching specifically and escalate to
> human review early, rather than letting the loop retry against op patterns that were never in its
> catalog."*

And immediately printed:

```
(advisory only; proceeding with auto-iterate loop. Disable via --no-meta-plan.)
```

**It identified two components as architecturally out of scope, recommended capping their retry
budget and escalating to a human, and then proceeded to do exactly what it had just warned
against.** The run subsequently died on an iteration-budget timeout — spent on the components the
meta-plan had named half an hour earlier.

### Why this reframes most of this document

The gap in this tool is **not analysis**. A component of it already produces a better architectural
critique than the rest of the pipeline acts on. Every finding above — the name-based family gate,
the overloaded `READY`, the template substitution, the logit-only correctness harness — is visible
to the meta-plan and invisible to the code that makes decisions.

### Suggested fix, in increasing order of ambition

1. **Let the meta-plan set budgets.** It already emits per-component risk. Feeding
   `cap_iterations(component, n)` and `escalate_to_human(component)` back into the loop is a
   plumbing change, not a research one — and it is the tool's own recommendation.
2. **Let it veto a backend.** When it says *"backend selection is a category-default match, not a
   genuine match"*, that is F19's check already written in prose. Make it a refusal.
3. **Let it select the correctness harness.** *"they need a waveform/mel-level comparison harness"*
   is the gate-design decision the optimizer half needs (O9). The meta-plan knows which harness a
   component requires; nothing asks it.

**Until then the tool prints an accurate diagnosis and then ignores it, which is a worse user
experience than not producing the diagnosis at all** — the run looks informed and behaves as though
it is not.

*(Recorded verbatim in `autoup_full2.log` lines 175-193.)*

### ⚠ F20 REVISED — the pipeline ignored the meta-plan and was RIGHT

The framing above is too strong, and the run that finally completed refutes it. The meta-plan
called `voxtral_flow_matching` and `voxtral_codec_decoder` *"architecturally out of scope for this
backend"* and recommended capping their retry budget and escalating to a human. The loop ignored
that and **ported both**:

```
23:03  ✓ GRADUATED  voxtral_codec_decoder   5/7
29:57  ✓ GRADUATED  voxtral_flow_matching   6/7
Graduated (ON_DEVICE): 7/7 (100%) actually graduated (native stub, PCC-verified)
```

Verified as real TTNN, not a torch shim that would pass a PCC gate trivially: `voxtral_codec_decoder.py`
is 363 lines with 11 `ttnn.linear`, 6 `ttnn.slice`, 5 `ttnn.reshape/multiply/add`, 4
`ttnn.permute/concat`, 2 `ttnn.matmul/embedding` and `ttnn.transformer`, **no `except`/fallback
path**, and `from_torch` only for weight upload.

**So the corrected recommendation is narrower and better:** the meta-plan's architectural pessimism
should inform **budgets and ordering** — try the hard components last, cap their share of the
iteration budget, warn the user what is at risk — but it must **not veto attempts**. It was wrong
about both hard components, and a veto would have cost the run's most valuable result.

What survives from F20 unchanged: its *factual* observations were all correct (the category-default
backend match, `leaves=1` discovery blindness, the logit-only PCC harness being wrong for a codec
and a vocoder), and none of those is wired to anything. **Route the facts into control flow; leave
the predictions advisory.**

---

## ★★★ THE OVERNIGHT RESULT — full three-block port, 7/7, 34 minutes

The experiment the whole detour was for.

```
Component classification: 0 REUSE, 0 ADAPT, 4 NEW (total 4)

00:00  ✓ layers_0_input_layernorm    1/7
01:34  ✓ layers_0_mlp                2/7
06:16  ✓ layers_0_self_attn          3/7
09:33  ✓ module                      4/7
23:03  ✓ voxtral_codec_decoder       5/7
29:57  ✓ voxtral_flow_matching       6/7
33:48  ✓ voxtral_tts_backbone        7/7

Graduated (ON_DEVICE): 7/7 (100%) actually graduated (native stub, PCC-verified)
RUN ENDED: bring-up complete — gate can_stop
```

**2,553 lines of generated TTNN** across 11 stubs in `models/demos/voxtral_tts_full/`, from a model
the tool had never seen, with `0 REUSE / 0 ADAPT` — nothing was copied; all of it was written.

### The qualification that must travel with that number

**Only 3 of 7 components were gated on REAL inputs.**

```
[capture] selected AutoModelForCausalLM (VoxtralTtsForConditionalGeneration) resolving 4/7
[capture] layers_0_input_layernorm: submodule not resolved; skipping.
[capture] layers_0_mlp:             submodule not resolved; skipping.
[capture] layers_0_self_attn:       submodule not resolved; skipping.
[preflight] captured 3/7 components; per-component PCC tests will use real inputs
```

`voxtral_tts_backbone`, `voxtral_codec_decoder` and `voxtral_flow_matching` got captured IO. The
three decomposed sub-components did not, and graduated against synthetic inputs — the §6.54 trap
(29.5% code flips on synthetic against 3.9% on real), which the tool's own documentation warns about.

### ⚠ F23 CORRECTED — three of the four capture misses were OURS, not the tool's

**Filed first as a tool finding; most of it is not.** The `captured 3/7` result had two causes and
only one belongs to the tool.

**Ours (S5).** The first version of the HF wrapper exposed the backbone's layer stack as

```python
self.layers = nn.ModuleList([nn.Module() for _ in range(26)])   # 26 EMPTY placeholders
```

with a comment saying, in as many words, that this existed *"so a structural walk finds a 26-deep
stack"*. The weights lived in a flat `ParameterDict` and `forward` called the reference function
over it. So the model **advertised structure it did not have**. The tool believed the
advertisement, decomposed `decoder_layer` into `layers.0.input_layernorm` / `.self_attn` / `.mlp`,
tried to hook those paths, and correctly reported:

```
[capture] layers_0_input_layernorm: submodule not resolved; skipping.
[capture] layers_0_mlp:             submodule not resolved; skipping.
[capture] layers_0_self_attn:       submodule not resolved; skipping.
```

Three of the four misses. **The tool's message was precise and correct; the model was lying to it.**

**Fixed.** The backbone is now real per-layer `nn.Module`s whose `forward`s call the reference's own
primitives (`rms_norm`, `split_heads`, `apply_rope`, `gqa_attention`, `merge_heads`, `swiglu`)
composed in `_layer`'s exact order — still bit-exact (prefill, prefill+cache and steps all
`maxdiff 0.0`), now with 138 named modules instead of 27, and verified by *firing hooks on a real
prompt* rather than by assuming:

```
hooks fired: {input_layernorm: 7, self_attn: 7, mlp: 7, flow: 3, codec: 1}
```

— the multiplicities of the actual frame loop, which is the check that would have caught the
placeholder immediately.

**The lesson, and it is general enough to be worth stating in the proposal:** a porting tool reads
the model's declared structure as ground truth. A wrapper that fakes structure to satisfy a
discovery pass will be believed, and the damage surfaces somewhere unrelated — here, as
"synthetic inputs" three stages later. **Verify a wrapper by hooking it, not by listing it.**

**What remains genuinely the tool's**, unchanged, is below.

### F23 — the capture drivers guess, and the config already says they should not

The clearest evidence yet for the representative-inputs recommendation, all from one run:

```
[capture] running drivers with pixel_values shape (1, 3, 224, 224) on 4 hook(s)
[capture] driver `model(pixel_values=...)`: ValueError: give input_ids or inputs_embeds
```

It drove a **text-to-speech** model with a **224x224 image tensor**, against a config declaring
`task: "text-to-speech"` and `modality_in: "text"` (F17 again, third instance).

```
[capture] driver `model(input_ids=..., attention_mask=...) [10 tokens]`:
  AssertionError: prompt has 0 audio placeholders but the preset has 169 rows.
[capture] auto-onboard: closed-loop iteration exhausted after 3 attempts:
  runtime ok but fired 0/3 target(s)
```

The generic driver feeds a 10-token prompt; this model needs one with 169 voice-specific audio
placeholders. The assertion message names the exact command that generates one
(`dump_prompt_ids.py --text '...' --voice <name>`) and **the tool has no way to act on a
remediation printed by the model it is porting**. Three LLM-drafted attempts, none valid.

**One fixture file would have given all seven components real inputs.** The tool already insists on
real activations over random ones — its own docs explain why — and then has no channel through
which a user can supply them.

**Suggested fix (restated, now with evidence):** `--calibration-inputs <path>`. A tensor fixture or
a callable. It is the smallest change on this list with the largest effect on correctness, because
every downstream PCC number inherits the quality of these inputs.

### Honest scope of this result

- **This is the tool plus five of my patches** — F1, F2, F5 (earlier), F18/F19 routing, F21
  `trust_remote_code`. Stock, it refuses this model at `compat` and again at the demo loader.
- The graduation gate is **per-component PCC against captured reference IO**. It is not end-to-end
  audio, and the tool has no WER, no MOS and no exact-match check on discrete codes (O9).
- `emit-e2e` — the independent grader — had not yet reported when this was written.

---

## F21 — `trust_remote_code` is a ONE-MODEL allowlist, and the two halves of the pipeline disagree

**Status: FIXED in this checkout** · severity: any custom-architecture checkpoint passes preflight
and dies in the demo · reported: not yet

```
FAILED models/tt_transformers/demo/simple_text_demo.py::test_demo_text[...]
  ValueError: The repository /localdev/.../voxtral-tts-full contains custom code which
  must be executed to correctly load the model.
```

The bring-up half loads with custom code enabled — `bringup_loop.py:486`,
`_cls.from_pretrained(HF_MODEL_ID, trust_remote_code=True, ...)`. The demo half, via
`model_config.py`, enables it like this:

```python
if self.base_model_name in ["Phi-3-mini-128k-instruct"]:
    self.trust_remote_code_hf = True
```

**Custom-architecture support is an allowlist containing exactly one model.** So a
`trust_remote_code` checkpoint clears Step 0 (*"transformers can load ... [ok]"*), clears static
analysis, and then fails at execution — and the message blames the repository rather than the
loader's configuration.

**Fix applied:** decide from the checkpoint, not the name. `auto_map` in `config.json` **is** the
declaration that a model ships custom modelling code — HF refuses to load such a model without
`trust_remote_code`, so its presence is decisive and needs no allowlist. `TT_TRUST_REMOTE_CODE=0`
restores the old behaviour for a checkpoint that should not be trusted.

### The theme these three share, and it is worth stating once in the proposal

| finding | decided by | available instead |
|---|---|---|
| **F18** | is `model_type` a known **name**? | do the config's fields describe a decoder? |
| **F19** | is there a template with a similar **name/family**? | does the template's `canonical_hf_id` equal the model being ported? |
| **F21** | is the model's **name** on an allowlist? | does `config.json` declare `auto_map`? |

**Three gates, three times deciding by identity when the answer was available from structure.** In
each case the structural signal is present in data the tool has already loaded, and in each case
the name-based answer fails on the first model that is not already known to it — which is the exact
population a porting tool exists to serve.

### F19 (addendum) — the generic demo has a SECOND entrance

The routing gate patched under F19 fired correctly this run:

```
NOT routing to the generic LLM demo: family was INFERRED from config structure ...
```

and the model still reached `simple_text_demo`, via a different path — `scaffold` raising
`ColdStartScaffoldError`, which the CLI handles as *"COLD-START PATH (no per-model `tt/` folder
needed)"* at `cli.py:9189`. **Two independent routes reach the same generic backend, and closing
one does not close the other.** Recorded rather than patched: one gate is a defect, two gates
reaching the same place by different reasoning is a design note the author should see.

---

## F22 — the isolation worktree silently ignores uncommitted changes to the tool's own source

**Status: found 2026-08-14, cost one run** · severity: developer iterating on the tool gets stale
behaviour with no warning · reported: not yet

`auto-up` runs in a private worktree — a good design, and the reason nothing it does can damage the
caller's checkout:

```
[isolation] worktree: /tmp/tt_hw_planner__..._1786737198
```

That worktree is created from **`HEAD`**. An edit sitting in the working tree is not in `HEAD`, so
it does not exist inside the run — **and nothing says so**.

Concretely: the F21 patch was written to `models/tt_transformers/tt/model_config.py`, compile-checked,
and the run launched. It failed with the *identical* `ValueError` the patch fixes. The worktree was
on `81814c5383` while the patch landed in `5ee438f04b`; `grep -c auto_map` in the worktree returned
**0** against **3** in the main checkout.

**Why this matters more than an ordinary footgun.** The people most likely to edit this tool's source
are the people extending it — adding a family, a block, an op-registry entry — and the natural loop
is *edit, re-run, observe*. That loop silently observes the previous version. The failure looks
exactly like "my fix didn't work", which is the most expensive possible misdiagnosis.

**Suggested fixes, cheapest first:**

- At worktree creation, if `git status --porcelain` is non-empty for tracked tool source, print a
  one-line warning naming the files that will NOT be included.
- Offer `--include-uncommitted` (a `git stash`-and-apply, or worktree-from-working-tree).
- Print the worktree's commit sha in the banner. It is already printing the path; the sha is the
  thing that determines behaviour.

**Verification I now use before every relaunch:** grep the created worktree for the patch itself,
rather than trusting the main checkout —

```
worktree: /tmp/tt_hw_planner__..._1786737523
patch present in worktree: 3        # was 0 on the run that failed
```

### Process note (mine)

I launched the run before committing the patch, then committed while it was in flight. Ordinary
sequencing error, but it is worth recording next to the finding: the tool's design made a routine
mistake produce a result indistinguishable from a failed fix, and I only caught it because the error
message was byte-identical to the previous run's — which is a weak signal to rely on.

---

## ★ F25 — decomposition children lose their parent's path prefix, and the plan is copied from another model

**Status: found 2026-08-15, worked around; the real fix is one line** · severity: silently degrades
per-component gates to synthetic inputs · reported: not yet

Two independent defects that compound, both visible in the same log.

### (a) The tool computes the correct path and then discards it

```
line 472  [recompose-link] `decoder_layer` (backbone.layers.0) -> 3 on-device child component(s)
line 468  [reinject] re-added decomposition child `layers_0_input_layernorm`
                     (layers.0.input_layernorm) of `decoder_layer`
line 487  [capture] layers_0_input_layernorm: submodule not resolved; skipping.
```

The recompose-link step records the parent **fully qualified** — `backbone.layers.0`. The reinject
step records the children **relative** — `layers.0.input_layernorm`. The capture hook uses the
children's path, looks up `layers.0.input_layernorm` on a model whose stack lives at
`backbone.layers.*`, finds nothing, and skips.

**The correct path is four lines away in the same log.** Children should inherit the parent's
qualified prefix.

### (b) `decomposition_plan.json` is COPIED from the closest existing demo

```
line 202  A  models/demos/voxtral_tts_full/decomposition_plan.json
line 203        copied from models/demos/voxtral_tts_backbone/decomposition_plan.json
```

and that file contains, correctly for **its own** model:

```
"layers.0.input_layernorm", "layers.0.mlp", "layers.0.self_attn"
```

`voxtral_tts_backbone` is a bare `MistralForCausalLM` whose stack genuinely is at top level. The
three-block model's is not. So a plan describing one model's topology was applied to another's —
**F19's template substitution, resurfacing in the decomposition plan rather than the demo.** The
demo it copied from is itself a previous artifact of this same tool.

### Consequence

Three components could not be hooked, so they graduated against **synthetic** inputs while the run
reported success — the §6.54 trap, reached without anyone making a mistake at the point of failure.

### Fixes

1. **Qualify child paths with the parent's prefix** at reinject time. The value is already computed.
2. **Do not copy `decomposition_plan.json` across models.** Regenerate per model, or at minimum
   validate every recorded path resolves against the model being ported and discard the plan if not.
3. Cheap defence that catches both: **after building the hook list, assert each path resolves**, and
   fail loudly rather than `skipping`. A skipped hook silently changes what the PCC gate measures.

**Workaround used here:** the Block-1 demo was moved out of `models/demos/` so nothing could be
copied from it, forcing the plan to regenerate against the model actually being ported.

### CONFIRMED — both fixes were necessary, and each was necessary alone

Measured across three runs of the same model, changing one thing at a time:

| run | model structure | template pool | capture result |
|---|---|---|---|
| 1 | 26 empty `nn.Module()` placeholders (**ours**, S5) | Block-1 demo present | `resolving 4/7`, **captured 3/7** |
| 2 | real per-layer submodules | Block-1 demo present | `resolving 7/10`, still 3 × `submodule not resolved` |
| 3 | real per-layer submodules | **empty** | `copied from (skeleton — no sibling source)` · **`resolving 7/7`**, zero unresolved |

**Neither fix alone was sufficient.** Real submodules did not help while the plan carried another
model's paths; deleting the stale plan would not have helped while the modules were hollow. That is
worth stating to the PR author, because it is why this failure is hard to diagnose from a single
run: two independent causes produce one identical symptom (`submodule not resolved`), and fixing
either one leaves the symptom unchanged.

**A side benefit worth noting:** with real modules to inspect, classification moved from
`0 REUSE, 0 ADAPT, 4 NEW (total 4)` to **`3 REUSE, 0 ADAPT, 4 NEW (total 7)`** — the tool recognised
three components as things `tt_transformers` already implements rather than writing them from
scratch. Honest structure did not just fix the gate; it made the port cheaper.

### Run 4 — `captured 7/7`, and the third fix was also ours

Runs 1-3 reached 3/7, then 5/7. The last two misses were **Blocks 2 and 3**, and the cause was again
the wrapper, not the tool:

```
driver `model(input_ids=..., attention_mask=...) [10 tokens]`:
   AssertionError: prompt has 0 audio placeholders but the preset has 169 rows
driver `submodule[backbone](**['inputs_embeds'])`: ok          <- backbone ONLY
```

Unable to construct a valid whole-model prompt, the framework fell back to driving the **backbone
submodule alone** — which reaches every backbone component and never executes the flow matcher or
the codec, so those two were gated on synthetic inputs while the run reported success.

**The model could not be run with no arguments.** Its `forward` required a prompt whose
audio-placeholder count is voice-specific (169 rows for the default voice), and no such prompt was
shipped with it. A generic driver cannot invent one. Fixed by carrying `default_prompt_ids` in
`config.json` — deliberately in the config rather than a sidecar file, because trust_remote_code
copies only `.py` into its module cache (S4 #4), so anything resolved from `__file__` is absent.

```
[capture] tts_backbone / decoder_layer / r_m_s_norm / attention / m_l_p /
          codec_decoder / flow_matching:  captured
[preflight] captured 7/7 components; per-component PCC tests will use real inputs
```

**Trajectory: 3/7 → 5/7 → 7/7, across three independent causes** — two ours (hollow modules, no
default prompt), one the tool's (F25's copied decomposition plan). Every per-component gate is now
measured against tensors the deployment actually produces.

**A caveat that keeps F23 intact.** The driver that finally succeeded is
`model(pixel_values=...): ok` — the *image-tensor* driver, on a text-to-speech model. It works only
because `forward` ignores unknown kwargs and falls through to the bundled default. **The tool still
does not know how to drive this model; it merely can no longer fail.** The recommendation is
unchanged: a `--calibration-inputs` channel, so representative inputs are supplied rather than
guessed.

**Packaging lesson for anyone wrapping a model for this tool:** it must be runnable with **no
arguments**. Every automatic driver, capture and smoke test depends on that, and a model whose
`forward` demands a specially-constructed input is undrivable no matter how correct it is.

### ⚠ CORRECTION — `captured 7/7` does NOT mean the tests use the captured tensors

I reported that reaching `captured 7/7` put every per-component gate on deployment tensors. **It does
not.** The capture succeeded — 23 files on disk under `_captured/` — and **none of the seven tests
read them**:

```
test_attention       captured-refs: 0   synth-refs: 5
test_codec_decoder   captured-refs: 0   synth-refs: 5
test_flow_matching   captured-refs: 0   synth-refs: 5
    ... all seven identical
```

`captured N/M` counts **recordings made**, not recordings consumed. Two different things, and I
conflated them.

**There are three tiers of input quality here, not two:**

| tier | what it is | where the run actually is |
|---|---|---|
| 1 | random tensors from name-guessing (`_make_arg_for`) | where it started — tests **crashed**: `cis` got `randn(1,64,3072)` where a COMPLEX rope table was required |
| 2 | inputs built from the reference's OWN primitives (`rope_cis`, `causal_bias`) | **where all 7 component tests are** |
| 3 | the recorded deployment activations | captured, on disk, **unused** |

**And tier 3 is declined for a genuine reason**, which the agent-rewritten harness documents:

> *`_captured/attention/args.pt` holds a real deployment step: `h=[1,1,3072]` with a 208-deep KV
> cache. It is not usable as-is for a unit test. The cache dict is **MUTATED** by
> `VoxtralAttention.forward`, and the harness hands the same object to the torch reference and then
> to the ttnn stub*

Feeding one mutable cache to the reference and then to the stub means the stub sees the reference's
write — a comparison against contaminated state that would look correct. Declining the capture is
the right call; **silently substituting tier 2 while the run reports `captured 7/7` is not.**

**F26 — report what the gate MEASURED, not what was collected.** A line reading `captured 7/7`
directly above per-component PCC results invites exactly the reading I gave it. The gate should
state its input provenance per component — `real-capture` / `synthetic-from-reference` /
`synthetic-guessed` — because those three carry very different confidence and §6.54 measured the
difference at 29.5% vs 3.9% error on the same code.

**What is unaffected:** the END-TO-END test genuinely uses the real prompt through the whole
pipeline and compares waveform against waveform. That is the number that decides whether the audio
is right, and it is honest.

---

## ★ F27 — the captured input is DISCARDED where one `deepcopy` would have kept it

The harness captures a real deployment activation for `attention`, correctly works out that it
cannot hand the same object to both sides, and then **throws it away** rather than copying it.

Its own note, in full:

> *`_captured/attention/args.pt` holds a real deployment step: `h=[1,1,3072]` with a 208-deep KV
> cache. It is not usable as-is for a unit test. The cache dict is MUTATED by
> `VoxtralAttention.forward` (`cache[cache_key] = (k, v)`), and the harness hands the same object to
> the torch reference and then to the ttnn stub — so the stub would attend over a cache one position
> longer than the golden did. **Dropping the cache instead makes the test vacuous**: at S=1 with no
> cache the softmax is over a single key, so it returns 1.0 whatever q and k are, and RoPE — the
> thing most likely to be wrong in a port — stops affecting the output at all.*

It considers exactly two options — **share the object** (contaminated) or **drop the cache**
(vacuous) — and takes neither, substituting a synthetic 64-token causal prefill.

**The third option is absent.** Give each side its own copy:

```python
ref_out  = reference(h, cis, bias, deepcopy(cache), key)
stub_out = stub(h, cis, bias, deepcopy(cache), key)
```

`grep` confirms it never occurred to the harness: **no `deepcopy`, no `.clone()`, no `copy.` anywhere
in `tests/pcc/conftest.py`.** The cost is negligible — one layer's cache at 208 positions is
≈ 208 × 8 heads × 128 dims × 2 tensors × 4 B ≈ **1.7 MB**.

**And the copy would be strictly better than the substitute**, in exactly the dimension that
matters. The synthetic prefill exercises RoPE at positions 0-63 with an empty cache; the real
captured step exercises it at **position 208 with a 208-deep cache**. RoPE errors are
position-dependent — `[gpt-21]` records SDPA settings that were correct at one length and
*"NOT SAFE — position sweep"* across others. The harness itself calls RoPE *"the thing most likely
to be wrong in a port"*, and then tests it at the positions least likely to expose the bug.

**Fix:** deep-copy mutable captured args per side. One line, and it converts the whole capture
pipeline from collected-but-unused (F26) into actually-used.

*(Credit to the reviewer who spotted this: the harness's reasoning is sound about why sharing fails
and why dropping fails, which makes the missing third option easy to overlook.)*

## ★★ F28 — the entire end-to-end verdict rests on ONE prompt

```
tests/e2e/test_e2e_pipeline.py     pytest parametrize: 0
CLI flags for prompts / cases:     none found in scripts/tt_hw_planner/cli.py
```

One text, one voice, one seed, one horizon. For a **generative speech model**, that is the whole
correctness gate — `Verdict: PASS` is decided on a single utterance.

**For contrast, the hand-port's gate on the same model** runs **45 utterances across 3 seeds**, and
its own history says why: `§6.21` records that a case's frame count depends on what ran before it in
the same process, so arms need identical history; `§6.62`'s `tail_probe.py` exists specifically to
*"count failures, not means"* because damage concentrates in rare bad utterances; and the
`w2 -> bf8_b` experiment moved `mos_min` by **0.245** while `mos_longform` moved 0.021 — a
mean-preserving change that mauled the worst case. **A one-utterance gate cannot see any of that.**

This compounds every other correctness finding in this document:

- **O9** — no WER, no MOS, no exact-match on discrete codes. Now also: no sample size.
- **F26** — per-component gates run on synthetic inputs, so the e2e test carries the correctness
  burden alone. It carries it on n=1.
- **F27** — the one component whose real input WAS captured has it discarded, so even that single
  sample is not deployment-representative at the component level.

**Suggested fix, in order of cost:**

1. `pytest.mark.parametrize` the e2e test over a prompt list, and take the **worst** PCC as the
   verdict rather than the only one.
2. A `--eval-prompts <file>` flag, so a user supplies the set — the same channel F23 asks for on the
   input side.
3. Report the distribution (min / mean / n), not a scalar. A single number invites exactly the
   confidence it cannot support.

**Why this is arguably the most important finding here:** every other defect in this document is a
thing the tool does wrong that a reader could catch. This one is a thing it *doesn't do*, and its
absence is invisible — the report says `PASS` and shows a PCC, and nothing on the page hints that
`n=1`.

### F28b — PROPOSAL: get the sample size from the BATCH dimension, not from N sequential runs

The obvious objection to F28 is cost: running 45 utterances the way the hand-port's gate does takes
~18 minutes, and an inner-loop correctness gate cannot afford that. **The batch dimension makes it
close to free**, and for this model two of the three blocks already support it:

```
Block 2  predict_velocity   [B,36], [B,3072], [B,3072] -> [B,36]
         semantic_code      h [B,3072] -> [B,1]
         decode_frame       [B,1], [B,3072] -> [B,36]              <- already batched
Block 3  reference_decode   codes [B,37,T] -> waveform [B,1,T*240*8]  <- already batched
Block 1  reference_forward  [1, S, 3072] -> [1, S, 3072]           <- pinned at 1
```

Block 2 is *already* running batched in production — CFG folds the batch to 2x (§6.35). So the gate
would be exercising a path the model already uses.

**What blocks Block 1 is deployment concerns a TEST does not have:**

| deployment problem | why a gate can ignore it |
|---|---|
| prompts differ in length | pad to a common length; the causal mask already handles padding |
| each utterance stops at its own `[END_AUDIO]` | run a FIXED frame count and ignore termination |
| per-sequence retirement scheduling | not needed when every row runs the same number of frames |

And the shape works out: a tile is 32 rows, so **B <= 32 still occupies one tile**. `per_core_M=1` /
`fuse_batch=True` are not violated. `nlp_create_qkv_heads_decode` already emits
`[1, batch, heads, head_dim]`, and `paged_update_cache` / `sdpa_decode` both take a batch dimension.
Batch-5 is untested here, not structurally blocked.

**The proposal:**

1. Run the e2e gate at **B = 5-8 prompts**, padded, fixed horizon, reference batched identically.
2. Report **min / mean / n** across rows, and gate on the **worst** row, not the mean — §6.62's
   `tail_probe.py` exists because damage concentrates in rare utterances, and `w2 -> bf8_b` moved
   `mos_min` by 0.245 while `mos_longform` moved 0.021.
3. Cost is roughly one utterance's wall time for 5-8 samples, which is what makes it viable as an
   inner-loop gate rather than a nightly one.

**Two caveats the PR author should hear with it:**

- **A port validated only at B=1 may silently assume it.** If the generated stubs break at B=5, that
  is itself the finding — and a cheap one to surface, since the gate would catch it on day one.
- **It changes the performance regime.** Batching is a throughput lever, so a B=5 measurement is not
  comparable to B=1 deployment timing. Use it for CORRECTNESS only; letting `optimize` tune against
  a batched measurement would optimise the wrong operating point.

*(Proposed, deliberately NOT implemented here — this is a design suggestion for the PR, not a change
we validated.)*

---

## ★★★ F29 — the threshold does not just GATE quality, it SETS it. And the two defaults disagree.

The single cleanest experiment in this document, and the most actionable finding for the PR.

### The two defaults

```
cli.py:10986      pe2e.add_argument("--pcc-target", type=float, default=0.95,
                     help="PCC threshold for the final HF-vs-TT comparison (default: 0.95)")

e2e_mcp.py:20     E2E_MCP_PCC   required e2e PCC threshold (default 0.99)
e2e_mcp.py:43     _PCC = float(os.environ.get("E2E_MCP_PCC", "0.99"))
```

The gate engine documents **0.99** as *"required"*. The CLI passes **0.95** and overrides it. A user
who never touches the flag gets the loose one, and nothing says a stricter default exists.

### What that costs, measured

Same port, same model, same machine. Only the threshold changed:

| threshold | measured e2e PCC | rounds the fix-loop needed |
|---|---|---|
| `0.95` (CLI default) | **0.9586** | `rounds=1 can_stop=True` — passed immediately, loop never worked |
| `0.99` (engine default) | **0.9986** | round 1, 45+ tool calls of actual repair |

**The 0.9586 was not a ceiling.** It was not a precision limit either — the test's own
device-precision bound reads `1.0000 at N=4`, and the comparison ran at N=4 (waveform 7680 samples
at `SAMPLES_PER_FRAME=1920`). It was simply **where the loop stopped, because the gate let it.**

Given a target it could not trivially clear, the same tool on the same code found another four
points of accuracy.

### The tool hands the user the loose default, in writing

Confirmed again on the 2026-08-15 re-run. Bring-up ends by printing the command to run next, and
that command carries no threshold (`run2_autoup.log:461`):

```
  NEXT STEP: wire the pipeline:
    python -m scripts.tt_hw_planner emit-e2e /localdev/lserbedzija/hf_models/voxtral-tts-full
```

Copy the line the tool just gave you and you get 0.95 — the 0.9586 column of the table above. The
0.99 column is reachable only by knowing to add a flag the tool never mentions. This is what makes
F29 a default problem rather than a documentation one: the recommended path *is* the loose path.

### Why this is the important version of O8

O8 recorded that the accept test has no exchange rate — it keeps any change that is faster and above
the floor. This is the mirror image and it is worse: **the threshold is not a floor, it is the
target.** Quality delivered ≈ quality demanded. A default set four points below the engine's own
documented requirement therefore does not merely permit worse ports — it *produces* them.

For a model where `§6.31` records that one flipped semantic code redirects an entire utterance, and
where the same port at 0.95 shipped `code exact-match: all codebooks 0.8649` — one acoustic code in
seven differing, against the hand-port's measured `codes_real_pct 5.2` — that gap is not academic.

### Fixes

1. **Make the CLI inherit the engine's default rather than override it.** One line. If 0.99 is
   documented as *required*, the CLI should not silently ask for less.
2. **Print the threshold's provenance in the report** — `pcc>=0.95 (CLI default; engine default is
   0.99)`. The banner currently states the number with no indication that it was lowered.
3. **Derive the floor from the measured precision bound.** The test already computes it
   (`1.0000 at N=4, 0.9458 at the 8-frame cap`). A fixed constant is either unreachable or too
   generous depending on the horizon; the bound is neither.

*(Recommendation 3 matters because 0.99 is NOT universally safe: at the 8-frame horizon this model's
own reference is only reproducible to 0.9458, so a hard 0.99 would be unsatisfiable there. The right
target is a function of the measured bound, not a constant.)*

---

## ★★★ F30 — the drift gate exists, detects the stale template, and is wired never to block

**Status: live in this checkout** · severity: bring-up selects a template directory that does not
exist, and says so only in a line it also suppresses · reported: not yet

The tool ships a registry drift check whose entire stated purpose is to stop this. Run against this
tree on 2026-08-15 it works perfectly:

```
$ python -m scripts.tt_hw_planner sync-registry --check
  [MISSING] family_backends[Voxtral TTS Backbone (mistral decoder)].demo_path
            -> models/demos/voxtral_tts_backbone
[sync-registry] FAIL: 27 registry path(s) missing from the checkout — fix the registry or restore the paths.
rc=1
```

`sync_registry.py:1-8` says why it exists: *"``--check`` exits non-zero on hard drift (a mapped path
that is gone) so CI / a pre-plan gate fails loudly instead of the planner silently mis-pointing at a
stale sibling."*

`up` / `auto-up` reach the same function through `_warn_on_registry_drift()` (`cli.py:8103`), whose
docstring states the opposite as a design commitment:

> *"Never raises: neither a fetch nor a drift check may block bring-up."*

On hard drift it prints exactly one line (`cli.py:8142-8149`):

```
[registry] N mapped registry path(s) are stale on this checkout — run `tt_hw_planner sync-registry` for detail.
```

…followed by the full `format_drift(issues)` listing, naming every stale path. **Verified against a
live `auto-up` on 2026-08-15**, which printed all 26 before proceeding.

That the detail appears at all is an accident worth its own line. It is guarded by
`if os.environ.get("TT_HW_PLANNER_VERBOSE")` (`cli.py:8147`), and the default is set at
`cli.py:8082` as:

```python
os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0")
```

The string `"0"` is **truthy** in Python, so the guard is always true and verbose output is
permanently on for this branch — including for a user who sets `TT_HW_PLANNER_VERBOSE=0` explicitly
to turn it off. The check wants `not in ("", "0", "false")`, the idiom the same file already uses
for `TT_HW_PLANNER_NO_WRAP` (`__main__.py:34`).

**So the tool prints everything it knows and proceeds anyway** — which is the finding in its
strongest form. This is not a reporting gap that hides the problem; the operator is shown 26 stale
paths, by name, and the run continues into template selection regardless. The whole body is also
wrapped in `except Exception: pass`, so a drift check that itself throws is indistinguishable from
a clean checkout.

### What it cost here, measured

`models/demos/voxtral_tts_backbone/` was removed from this tree at 09:29 on 2026-08-15. The bring-up
run at 15:11 selected it anyway — `models/demos/voxtral_tts_full/RUN_REPORT.md`:

```
Backend picked:    Voxtral TTS Backbone (mistral decoder)  (TEMPLATE-FALLBACK — model_type mismatch)
Closest template:  models/demos/voxtral_tts_backbone/        <- absent from the checkout
Sibling base:      /localdev/lserbedzija/hf_models/voxtral-tts-backbone (model_type=mistral)
```

This is **F19 with the safety net already built and switched off.** F19 showed template dispatch can
silently run a different model; F30 shows the tool can *prove* the template is gone, on the same
run, and proceed regardless. It is also F20's exact shape a third time: the knowledge exists and is
wired to stdout instead of to control flow.

**This is not only our mess.** The dangling Voxtral entry was ours (S7) and we removed it. The
drift check still fails afterwards, with **26 mapped paths missing** — `XTTS-v2 (multilingual TTS)`
→ `models/demos/xtts_v2`, `tt_dit/minimax_h3 (auto-upstream)`, and 24 more, all entering the
registry through the tool's own commits (`589a4d121a`, `12bd4e4ef8`). So the shipped registry
points at 26 paths that do not exist in the checkout it ships with, and the gate that knows this is
the one guaranteed never to fire. Any of those 26 can be selected as a template exactly the way
ours was.

### Fixes

1. **Make hard drift on the *selected* backend fatal.** Global drift can stay advisory — 27 stale
   paths in unrelated families should not block a bring-up. But once template selection has
   *picked* an entry, a missing `demo_path` on that entry is not a warning, it is a broken run.
2. **Fix the verbosity guard** — `os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0")` plus a bare
   truthiness test means the flag can never be off, and `TT_HW_PLANNER_VERBOSE=0` does not turn it
   off. Compare against `("", "0", "false")` as `__main__.py:34` already does. (The drift detail
   itself should stay visible — print it unconditionally rather than by accident.)
3. **Narrow the `except Exception: pass`.** A drift check that crashes currently reports as a clean
   checkout.

---

## ★★ F31 — the profiler reports a missing CSV where the child actually died of a bus error

**Status: live** · severity: the optimizer agent is handed a plumbing error instead of a crash ·
reported: not yet

`termination_check()` returned this to the optimizer agent at 16:27:45 on 2026-08-15:

```
can_stop: false
error: "profiler crashed: tracy run exit 1 (log: /tmp/perf_mcp_4tvrspfx/run0_tracy.log)
        AssertionError: cpp_device_perf_report.csv not found and legacy device log
        profile_log_device.csv is also missing in /tmp/perf_mcp_4tvrspfx/tracy_out/.logs."
```

Read literally that is a profiler-output-plumbing problem, and an agent told to keep optimizing will
go looking for one. It is not what happened. The surviving log of a sibling run
(`/tmp/perf_mcp_h4kudb_0/run0_tracy.log`) shows the profiled child aborting mid-forward:

```
Fatal Python error: Bus error

Current thread (most recent call first):
  ttnn/ttnn/decorators.py:650 in __call__
  models/demos/voxtral_tts_full/_stubs/m_l_p.py:38 in __call__
  models/demos/voxtral_tts_full/tt/pipeline.py:165 → 195 → 369 decode_stack → 413 run_tts
  tests/e2e/test_main_perf.py:204 in _eager_forward → 251 in test_main_perf
Aborted (core dumped)
```

The CSV never appears **because the process died before writing it.** The postprocess then walks its
fallback chain — `process_ops_logs.py:1136` warns that `cpp_device_perf_report.csv` is missing and
falls back to legacy parsing, `process_ops_logs.py:755` finds that missing too and raises — and the
raise is the only thing that reaches the caller. The abort, the signal, and the stack are all in the
log the caller cites but does not read.

**The masking is the finding, not the bus error.** Whether the bus error is a ttnn defect, a
profiler-buffer overrun on a three-block forward, or a fault in the perf test itself is not
established here and is not claimed. What is established is that an abort was reported as a missing
file.

**CORRECTION (2026-08-15, from the re-run).** An earlier draft of this entry called
`test_main_perf.py` "our own hand-written perf test". It is not ours. The optimize stage generates
it — the re-run's own log shows it being produced from the PCC test:

```
  auto-gen perf from pcc (agentic) -> tests/e2e/test_main_perf.py::test_main_perf
  auto-gen perf from pcc           -> tests/e2e/test_main_perf.py::test_main_perf
```

The file in the aborted session carried the same name because that session was running `optimize`,
which had generated it. So the crash is inside the tool's own generated artifact on the tool's own
path, and the attribution above should be read accordingly. See F40, where the same measurement
crashes again — this time in device teardown, after producing its numbers.

### Fixes

1. **Check the child's exit status first.** A child that exited by signal (or non-zero) should be
   reported as that — `tracy child aborted (SIGBUS) at <last stack frame>` — before any assertion
   about its outputs runs.
2. **Include the log tail in the error.** The caller is already given the log path; the last ~40
   lines would have carried `Fatal Python error: Bus error` into the agent's context.
3. **Do not let the fallback chain's terminal assertion be the reported cause** when an earlier,
   more specific failure was already observed.

---

## ★★ F32 — `termination_check()` blocks for 30 minutes with no progress channel, and the retry never returns

**Status: live** · severity: an unattended optimizer run cannot be distinguished from a hung one ·
reported: not yet

Timings from the driving session's own transcript (`0b038219-…jsonl`), 2026-08-15:

| time | event |
|---|---|
| 15:57:53 | agent calls `termination_check()` |
| 16:27:45 | returns — **29 min 52 s later** — with the F31 error |
| 16:28:02 | agent retries `termination_check()` |
| — | never returns; the transcript ends here |

Nothing is emitted in between. The tool re-profiles the model inside the call, so half an hour of
silence is the *normal* case, not the failure case — which means the failure case is
indistinguishable from it. The run above was abandoned by its operator as hung; it had in fact
returned one error and was sitting inside a second identical call.

This is **F7 recurring at the optimizer stage** (*"all progress flows through one channel, and
nothing notices when it is dead"*), and it compounds F31: the one message that does come back after
thirty minutes describes the wrong failure.

### Fixes

1. **Emit progress from inside the call** — at minimum the sub-step (`profiling`, `parsing`,
   `checklist`) and the elapsed time.
2. **Bound the call and return partial status** rather than blocking indefinitely on the retry.
3. **Make a repeated identical call cheap or refused.** The retry re-ran the same 30-minute profile
   against an unchanged tree.

---

## ★ F33 — `worktree-list` can never print ORPHAN, so dead worktrees accumulate looking healthy

**Status: live in this checkout** · severity: the operator is told there is nothing to reclaim while
2.7 GB is reclaimable · reported: not yet

Six bring-up worktrees on this box, one per `auto-up` run:

```
$ python -m scripts.tt_hw_planner worktree-list
  /tmp/tt_hw_planner__…_1786735901   …voxtral-tts-full   1541336   22.0   active
  /tmp/tt_hw_planner__…_1786736367   …voxtral-tts-full   1548859   21.9   active
  … 4 more, all "active"

$ for p in 1541336 1548859 1552915 1557766 1797227 1801185; do ps -p $p ...; done
  1541336 DEAD   1548859 DEAD   1552915 DEAD
  1557766 DEAD   1797227 DEAD   1801185 DEAD
```

Every creator is dead. Every row says `active`. The cause is one expression —
`commands/worktree_list.py:20`:

```python
status = "ORPHAN" if id(s) in orphans else "active"
```

`id(s)` is CPython's builtin object-address `id()`. `list_orphans()` (`worktree.py:169-174`) returns
`List[WorktreeSession]` — objects, not addresses. An `int` is never `in` a list of
`WorktreeSession`, so the ORPHAN branch is unreachable and every worktree prints `active` forever.

**This is F2's shape again**: a verdict that cannot fire, where the failing branch is the one that
signals work is needed.

**The two commands contradict each other in the same checkout.** Immediately after `worktree-list`
called all six `active`, `worktree-cleanup` was run and printed, for the very same PIDs:

```
orphan worktree: /tmp/tt_hw_planner__…_1786735901 (… creator-pid=1541336 dead, age=22.1h)  -> removing
… removed 6 orphan worktree(s)
```

Same predicate, same process, opposite answers — because cleanup asks `list_orphans()` and the
listing asks `id()`. 2.7 GB was reclaimable the whole time the listing said otherwise.

**The reclaim path itself is correct.** `cleanup_orphans()` (`worktree.py:195`) calls
`list_orphans()` and iterates the objects properly, so `worktree-cleanup` *does* remove them. The
damage is confined to the display — but the display is the only thing telling an operator whether
running cleanup is worthwhile, and it says no. Six worktrees × ~430 MB accrued unnoticed, one per
run, on a box where the model checkpoint alone is 16 GB.

**Secondary, and worse in a shared setting.** `_pid_alive()` (`worktree.py:176-186`) treats
`PermissionError` as not-alive:

```python
    except (ProcessLookupError, PermissionError):
        return False
```

`os.kill(pid, 0)` raises `PermissionError` precisely when the process **exists but belongs to
another user**. That PID is alive. Classified orphan, it becomes a `git worktree remove --force`
(and an `shutil.rmtree` fallback) against a worktree whose creator is still running.

### Fixes

1. **Compare identity, not `id()`** — `if s in orphans`, or match on `s.path`.
2. **`PermissionError` means alive.** Only `ProcessLookupError` means gone.
3. **Have `worktree-cleanup` print the same status `worktree-list` computes**, so the two can never
   disagree about what is reclaimable.

---

## ★★★ F34 — deleting the model does not delete the model: the overlay store silently restores it, and a from-scratch run is unreachable

**Status: live in this checkout** · severity: two runs from the same HEAD start from different
states, and the difference is invisible · reported: not yet

To re-run the pipeline cleanly at PCC 0.99, `models/demos/voxtral_tts_full/` was deleted and the
deletion **committed** (`42e9bee5f7`). HEAD contained no demo directory; `git status` was clean.
`auto-up` was then launched.

The isolation worktree was created from that HEAD — correctly, and `git log` inside it confirms
`42e9bee5f7`. Then one line went past:

```
  [isolation] applied 0 _shared + 1 model overlay(s)
```

After that line, the worktree contained the **entire previous port**: 63 files, including every
graduated stub, their `.best_native` / `.last_good_native` graduation snapshots,
`.bringup_cc_state.json` (16234 B), `bringup_status.json` (5928 B) and the previous run's
`RUN_REPORT.md` — all stamped with the current run's timestamp, all reinstated on top of a HEAD
that does not contain them.

The overlay store had retained a whole-directory patch for the model. Nothing in the run says a
previously-ported demo directory has just been reinstated; the notice is a count of overlays.

**Why this is worse than a surprising default**

1. **A from-scratch run is not reachable through the documented surface.** Delete the model's
   directory, commit, re-run — and the model comes back. There is no `--no-overlays` /
   `--from-scratch`.
2. **Reproducibility inverts.** Two runs from the same commit produce different starting states
   depending on overlay state that is not in the tree, not in the log, and not in the report.
3. **It is invisible in exactly the place it matters.** The RUN_REPORT records placements and PCC
   for what it *believes* it built this run.

`--reverify` does mitigate part of it — it clears restored graduation snapshots so each component
re-earns its gate — and we passed it. But it is opt-in, it addresses only the markers, and the
restored *implementations* remain regardless. A run that begins with a finished port is not a
bring-up, whatever the markers say.

**The documented wipe does not wipe.** `overlay-drop <model_id>` is documented as *"Omit rel_path to
wipe ALL overlays for the scope."* Run against this model it dropped every patch and left:

```
scripts/tt_hw_planner/overlays/_localdev_…_voxtral-tts-full/locked_modules.json
  {"decoder_layer": {"locked_ts": 1786786489.8,
                     "reason": "children all on device; recomposed as whole-module target"}}
```

A pin from the previous run, recording a structural decision about `decoder_layer`, surviving the
command whose stated job is to wipe the scope. It had to be removed by hand.

**A tell in the same log.** The successful overlay was preceded by ~30 lines of the form
`skipped <path> — git apply --check returned rc=1 … already exists in working directory`. The
overlay system was largely failing to apply against a tree it had itself just populated; the patch
that *did* apply was the whole-directory one. The mechanism is noisy about its failures and silent
about its one consequential success.

### Fixes

1. **Say what was restored, not how many.** `restored models/demos/voxtral_tts_full/ (63 files,
   incl. 5 graduation markers) from the overlay store` is one line and ends the entire class of
   confusion.
2. **Provide `--no-overlays` (or `--from-scratch`)**, and name it in the bring-up docs as the way to
   reproduce a clean port.
3. **Make `overlay-drop <scope>` empty the scope** — `locked_modules.json` included — or state what
   it deliberately preserves and why.
4. **Never carry graduation markers in an overlay.** Ported source is legitimate to reuse; a
   "this component already passed its PCC gate" marker earned in a different run under a different
   threshold is not — see F26 (report what the gate measured) and F29 (the threshold sets quality).

---

## ★★ F35 — backend selection is not reproducible: identical runs pick different templates

**Status: live in this checkout** · severity: the template that shapes the generated port is chosen
non-deterministically · reported: not yet

Two `auto-up` runs, same model, same commit (`42e9bee5f7`), ~4 minutes apart. The deterministic
ranking was **identical** both times:

```
  Sibling candidates (top 2, exact first):
    1. hf_eager universal (TTS)          [score=40; category 'TTS' default (generic runner)]
    2. XTTS-v2 (multilingual TTS)        [score=30; category 'TTS' default]
```

The selection was not:

```
run 1:  Backend match: LLM-RESOLVED  (hf_eager universal (TTS))     <- rank 1
run 2:  Backend match: LLM-RESOLVED  (XTTS-v2 (multilingual TTS))   <- rank 2, score 30
```

Same inputs, same scores, different answer — and in run 2 the LLM ranker overrode its own
deterministic top pick in favour of a candidate scored 10 points lower, with no stated reason
beyond the boilerplate *"the registry-constrained LLM ranker chose this as the closest
architectural sibling"*.

**Why it matters.** The backend is the template, and the template shapes the scaffold: which demo is
copied, which attention/RoPE conventions are assumed, which reuse map applies. F19 already showed
template dispatch can silently run a *different model*; F35 says which template you get is not
stable across identical invocations. A bring-up that cannot be reproduced cannot be bisected, and a
regression cannot be attributed to a code change rather than to the ranker's mood.

**Both winners point at directories that do not exist** — `models/demos/hf_eager/demo.py` and
`models/demos/xtts_v2` are both on F30's list of 26 stale registry paths. The ranker is choosing
between two broken entries and the run continues either way.

**What saved this run.** F18's corrected routing gate declined the generic route in *both* runs,
identically and for the right reason:

```
NOT routing to the generic LLM demo: family was INFERRED from config structure, not a known
model_type; config declares 3 stacks (backbone, flow, codec) and the block checklist covers the
decoder stack only. Scaffolding this model's own stubs instead.
```

So for *this* model the divergence is contained — the port is scaffolded from the model's own
structure rather than from either template. That containment is luck of architecture, not design:
a single-stack model with a known `model_type` would have been routed to whichever template the
ranker happened to name that day.

### Fixes

1. **Make the deterministic score authoritative unless the LLM gives a stated, logged reason to
   depart from it** — and log the reason next to the score it overrode.
2. **Seed / cache the ranker per (model, commit)** so a re-run reproduces the earlier choice, and
   record the resolved backend in the run report as an input, not a narration line.
3. **Exclude candidates whose `demo_path` is missing** before ranking. Both candidates here were
   unusable; the ranker was choosing between two dead links (F30).

---

## ★★★ F36 — "PCC tests will use real inputs" is false: the gate runs on `torch.randn`, and the real capture is never loaded

**Status: live in this checkout, observed during the 0.99 re-run** · severity: every component
graduates against synthetic data while its real activations sit unused on disk · reported: not yet

This is F26 confirmed by direct inspection, and worse than F26 recorded it, because the tool now
*states* the opposite in its own log.

### The claim

Preflight, run 2, 2026-08-15:

```
  [capture] attention: captured args=5 kwargs=0 output=tensor
  …
  [preflight] captured 7/7 components; per-component PCC tests will use real inputs
```

It captured well. `_captured/attention/` and `_captured/decoder_layer/` are **43 MB each** — real
deployment activations with a deep KV cache — and every component directory holds `args.pt`,
`kwargs.pt`, **`output.pt`** (the real reference output) and `manifest.json`.

### What the gate actually runs

`bringup_mcp.py:337` runs the per-component gate as
`_run_focused_pytest(test_files=[tests/pcc/test_<comp>.py])`. That generated test:

- contains **no `torch.load`, no `args.pt`, no `.pt` reference of any kind** — verified across the
  whole `tests/pcc/` directory, 0 hits in 7 files;
- builds every input from the forward signature by **argument name**, via `_make_arg_for`:

```python
if arg_name in ("hidden_states", "inputs_embeds", "embeddings"):
    shape, _ = _detect_hidden_shape(torch_module, model=model)
    return torch.randn(*shape).to(md)
…
if primary is None:
    primary = ("(synthetic)", torch.randn(1, 64, 64))
```

- uses the capture for exactly one thing — deciding **which submodule** to test:

```python
_captured_path = _captured_submodule_path(COMPONENT_NAME)
if _captured_path:
    torch_module = _resolve(model, _captured_path)
```

All six `_captured` references in each test are that path lookup. The tensors are never opened.

### So what graduation at 0.99 means here

A component is graduated when this test reports PCC ≥ 0.99 — against `torch.randn` at whatever
shape the name heuristic infers, compared to the HF module fed *the same* synthetic tensor. For
`attention` that replaces a real 208-deep KV cache with `torch.randn(1, 64, …)`; the captured
`output.pt` that would have made it a true golden comparison is never read.

Raising the threshold does not help. **0.99 on synthetic input is not a stronger claim than 0.95 on
synthetic input — it is a more precise measurement of the wrong thing.** This is the limit of what
F29's threshold fix can buy, and the reason F28's point stands: the whole real-correctness signal
rests on the e2e stage.

### It is not that the captures are unusable

The demo path does consume them — `demo_wiring.py:80-81` requires `args.pt`/`kwargs.pt` to exist,
and `bringup_loop.py` emits a `_load_captured()` helper into generated demo code whose docstring
says it "matches the PCC test convention so the demo passes whenever the PCC test passes". The
convention it claims to match is the one the PCC test does not implement. The plumbing to do this
right is already written and already paid for; the gate simply does not call it.

### Fixes

1. **Load `args.pt`/`kwargs.pt` in the generated PCC test when the capture exists**, and fall back
   to `_make_arg_for` only when it does not — the fallback is the current behaviour, so this is
   additive.
2. **Compare against the captured `output.pt`**, not only against a re-run of the HF module on
   synthetic input. That turns the gate into a golden test at no extra cost.
3. **Make the log line honest.** If the test ran on synthetic inputs, say `captured 7/7; gate ran on
   SYNTHETIC inputs (captures used for submodule resolution only)`. Reporting what the gate
   measured rather than what was collected is F26, and this is the same sentence needing the same
   repair.
4. **State it in the report too.** `RUN_REPORT.md` records components as "graduated, native ttnn,
   PCC verified" with no indication of what they were verified against.

---

## ★★★ F37 — the generated PCC test cannot express this model, and four of its defaults are silently wrong

**Status: live, hit in round 1 of the 0.99 re-run** · severity: one defect is a `NameError` on every
model; two others corrupt the *golden* rather than the port · reported: not yet

Round 1 of bring-up spent 13 minutes and 43 tool calls on `attention`. It graduated — but the
repair went almost entirely into `tests/pcc/conftest.py`, not into the stub. The agent's own header
is the cleanest statement of the problem:

> *"Everything here fixes a defect in `tt_hw_planner`'s test TEMPLATE, so it belongs in one shared
> conftest rather than in seven generated files (which get re-emitted) and never in a stub (the stub
> is not what is broken)."*

Four defects, in the tool's template. **(1) and (4) verified here independently; (2) and (3) are the
repair agent's analysis, recorded as such.**

### 1. `_captured_submodule_path()` is called by every generated test and defined by none — VERIFIED

The first gate run of the first component died on:

```
NameError: name '_captured_submodule_path' is not defined
   at stage=build_torch_reference
```

Checked across the emitted suite: **used in 7 of 7 tests, defined in 0 of 7.** This is not
model-specific — the template emits the call unconditionally, so the first component of every
bring-up, for every model, fails its first gate on the tool's own generation bug and burns repair
rounds on it. The fix here was to define it in `conftest.py` and publish it into `builtins`, which
works only because an unresolved global in a test module falls back there.

### 2. `_make_arg_for` cannot build inputs for a functional-style reference — reported by the agent

The template infers inputs from argument **names** (F36). This reference takes its own arguments,
not HF's canonical ones, and the name heuristic fails four different ways:

- **`cis`** — a complex RoPE table `[S, head_dim/2]`, required by `attention` and `decoder_layer`.
  The fallback would hand it `randn(1, 64, 3072)`.
- **`bias`** — the additive causal mask. It defaults to `None` and is not in the "well known" set,
  so the template's own rule drops it:
  ```python
  if not is_required and not is_well_known:
      continue
  ```
  **Dropping it silently makes the golden non-causal.** The port is then measured against a
  reference that is not the model — a test that can fail a correct port, or pass an incorrect one,
  with equal confidence.
- **`x_0`** — `flow_matching` draws `torch.randn` for it when `None`, so golden and port integrate
  **different noise**; and it wants `llm_hidden` `[B, 3072]` (2-D) where the template supplies a 3-D
  activation.
- **`codes`** — `codec_decoder` wants an integer tensor `[T, 37]`; the fallback hands it a float
  activation.

### 3. The native probe forbids marshalling the side inputs — reported by the agent

`models/common/native_probe.py` graduates a stub only on `torch_ops == 0`, and `ttnn.from_torch`
itself counts (it surfaces as `__dlpack__`). So a stub **may not** convert `cis` / `bias` / `x_0`
per call. The workaround is to rebuild them inside `build()` (not probed) and ignore the passed
values — meaning the graduated stub deliberately ignores three of its own inputs to satisfy the
probe.

### 4. The harness stages every primary input as bfloat16 — VERIFIED

`bringup_loop.py:619`:

```python
def _ttnn_from_torch_mesh_safe(tensor, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = tensor.to(torch.bfloat16) if dtype == ttnn.bfloat16 else tensor
```

called at `bringup_loop.py:791` with no dtype override. bfloat16 carries 8 mantissa bits, so it is
exact only to 256:

- **For an index tensor this is destructive.** A codebook id of 8191 becomes 8192 — the harness
  corrupts the input before the port ever sees it, and the port is blamed for the mismatch.
- **For an activation it injects ~4e-3 of input error**, which lands in the same PCC number the
  0.99 gate is judging.

The repair replaces it with: integer → `uint32`/`ROW_MAJOR` (which is also what `ttnn.embedding`
wants), float → `float32`/`TILE`.

### Why this one matters most

F36 says the gate measures synthetic data. F37 says that for a model whose reference is not
HF-canonical, the synthetic data the template invents is not merely unrepresentative but
**invalid** — and in the `bias` and bf16 cases the error is on the *reference* side, where no
amount of work on the port can fix it and no threshold can detect it.

**The threshold itself was left honest.** The repair agent noted explicitly:

> *"NOTE the assertion itself is untouched: `PCC_TARGET = 0.99` and the single `assert ok` in each
> generated test are exactly as emitted."*

So this run's 0.99 gate is intact — it is the inputs and the golden that needed repair, not the bar.

### Fixes

1. **Emit `_captured_submodule_path` into the template** (or import it) — a one-line generation fix
   that removes a guaranteed first-round failure for every model.
2. **Load the capture** (F36 fix 1) — it makes defect 2 largely moot, since real `cis`, `bias`,
   `codes` and `x_0` are already on disk in `args.pt`/`kwargs.pt`.
3. **Never drop a defaulted argument that changes semantics.** A mask defaulting to `None` is not
   an optional input; if the template cannot build it, it must fail loudly rather than produce a
   non-causal golden.
4. **Choose the staging dtype from the tensor**, not a constant: integer → `uint32`/`ROW_MAJOR`,
   float → `float32`/`TILE`.
5. **Let a stub marshal its own side inputs**, or exclude input staging from the `torch_ops == 0`
   probe — otherwise the probe's definition of "native" forces stubs to ignore their arguments.

---

## ★★★ F38 — `optimize --devices` defaults to `"0,1"`, so a single-chip box gets a 2-chip plan — and an explicit `--mesh 1,1` does not stop it

**Status: live, hit launching the optimize stage** · severity: the perf stage optimizes and gates
under a topology the model was never ported to · reported: not yet

`cli.py:11101`:

```python
popt.add_argument("--devices", default="0,1", help="single | all | explicit ids like '0,1'")
```

A literal two-device default. This box has **one** chip:

- `HARDWARE['P150'].chips == 1`
- one device node exists: `/dev/tenstorrent/3`
- bring-up, minutes earlier, wrote `models/demos/voxtral_tts_full/parallelism_manifest.json`:
  ```json
  {"chips": 1, "tp": 1, "dp": 1, "mesh": [1, 1]}
  ```
- the whole port was brought up and e2e-verified at TP=1

Launched with the topology stated explicitly — `--box P150 --mesh 1,1` — `optimize` printed:

```
  engine   : cc · devices 0,1 · mesh 1,1 · metric device_ms
  topology : 2-chip -> mesh 1x2 (TP=2 DP=1) [1D default]
```

It echoes `mesh 1,1` on one line and resolves `2-chip -> mesh 1x2 (TP=2 DP=1)` on the next. The
device-list default wins over the explicitly requested mesh, silently. Adding `--devices 0` fixes
it:

```
  engine   : cc · devices 0 · mesh 1,1 · metric device_ms
  topology : single chip -> mesh 1x1
```

**Why it matters.** Every measurement the optimize stage takes — and the e2e PCC gate it runs to
protect correctness — would have run against a 1x2 mesh with TP=2 on hardware that has one chip.
Either it fails somewhere late and expensively, or it "succeeds" and reports device_ms for a
topology the port does not have. The tool wrote the correct topology into its own
`parallelism_manifest.json` during bring-up and does not read it back.

This is F20's shape again — the tool already knows, and the knowledge is not wired to the decision.

### Addendum — the perf record does not carry the topology either

Once the run was pinned to `--devices 0` and had resolved `topology : single chip -> mesh 1x1`, the
scorecard it then wrote reads:

```
[full-pipeline-gate] PERF_SCORECARD mesh=unknown TP=unknown DP=unknown shard=unknown
                     on_device=True ISL=128 OSL=128 batch=1 TTFT_ms=NA
                     prefill_path=n/a decode_ms=1735.2656 decode_path=trace+1cq TSU=0.58 TS=0.58
```

`mesh=unknown TP=unknown DP=unknown` — in the same process that printed the resolved topology
minutes earlier, and for a metric whose meaning depends entirely on it. A latency number without
its mesh cannot be compared against another run, which is the whole purpose of a scorecard. Same
root as the finding above: the topology is known and not propagated.

### Fixes

1. **Default `--devices` to what is present**, not to `"0,1"`. `single` is already an accepted value
   of the flag; it is the safe default.
2. **Read `parallelism_manifest.json`.** The tool records the port's own chips/TP/DP during
   bring-up; the perf stage should inherit it rather than re-derive a different answer.
3. **Never let a device-list default override an explicit `--mesh`.** If they disagree, stop and say
   so — the current behaviour prints both, one line apart, and proceeds with the one the user did
   not ask for.
4. **Stamp the resolved topology into `PERF_SCORECARD`.** It is computed and printed at startup;
   emitting `mesh=unknown TP=unknown` alongside a latency makes the record uncomparable.

---

## ★★ F39 — the e2e report prints `Verdict: PASS` beside `e2e PCC n/a`

**Status: live, observed on the 0.99 re-run** · severity: the headline artifact omits the one number
the verdict is about · reported: not yet

`emit-e2e` finished PASS. Both the console summary and `RUN_REPORT.md` render:

```
    task         e2e PCC   demo (real I/O)            trace perf test
    tts          n/a       demo/demo_tts.py           test_tts_perf.py
```

**The verdict is sound.** `can_stop` comes from `e2e_mcp.py`'s combined gate, which the module
docstring describes as *"Tool-run, NOT agent-reported: the tool runs tests/e2e and measures PCC
itself, so the agent cannot self-declare done, fake the number, or xfail/skip past it"* — good
design, and it held. A direct `pytest` run confirms the port is excellent:

```
e2e PCC = 0.9999834299087524
frames tt=(8,37) ref=(8,37)  exact_match=True  code_flips=0
per-step hidden PCC: 1.000000 x9
7 passed in 50.96s
```

**The report just cannot see it.** The cell is computed at `commands/emit_e2e.py:434`:

```python
pcc_s = f"{pcc:.4f}" if isinstance(pcc, float) else "n/a"
```

and `pcc` is sourced (`emit_e2e.py:377-386`) from `demo_dir/grader_report.json` → `calls[].final_pcc`.
**That file does not exist** — the cc-engine fix-loop path never writes it; it belongs to the older
grader-agent path. The verdict beside it is a literal, passed in at `emit_e2e.py:1487`:

```python
if final.get("can_stop"):
    emit_e2e_report(model_id, demo_dir, verdict="PASS")
```

So verdict and measurement come from two different places, and only one of them survived the
refactor to the cc engine.

**Why it matters more than a cosmetic gap.** F29 established that the PCC threshold is the single
most consequential knob in this pipeline — the difference between a 0.9586 port and a 0.9986 one.
`RUN_REPORT.md` is the artifact a reviewer reads to judge a port. Printing `PASS` with no number
next to it is precisely F26's complaint (*report what the gate measured*) applied to the headline
result: a reader cannot tell 0.9999 from 0.9900 from "the gate was satisfied some other way".

### The number is not lost — another part of the tool writes it down

The optimize stage, running an hour later against the same demo, recorded this in
`/tmp/perf_mcp_gate_verdicts_voxtral_tts_full_main.json`:

```json
{
  "full_pipeline": {"status": "ok", "full_pipeline_ms": 1735.2656, "method": "trace", "sha": ""},
  "pcc":           {"status": "ok", "pcc": 0.9999834299087524, "sha": ""}
}
```

The same value, to the digit, that a direct `pytest` run produces — captured, structured, and stored
by `perf-mcp` without difficulty. So the e2e report's `n/a` is not a measurement that could not be
taken; it is a number the tool holds in one hand and cannot render with the other. That also makes
fix 1 concrete rather than aspirational.

### Fixes

1. **Surface the number the gate already measured.** `_run_deterministic_gates` computes the e2e PCC
   to compare against the threshold — and `perf-mcp` demonstrably persists exactly that value in its
   gate-verdicts file. Return it and render it, rather than re-reading a file written by a different
   code path.
2. **Have the cc-engine path write `grader_report.json`** (or drop the file dependency entirely).
3. **Never print a verdict beside `n/a`.** If the value is genuinely unavailable, say why — `PASS
   (gate satisfied; PCC not recorded by this path)` is honest; `PASS … n/a` reads as an oversight
   the reader must not notice.

---

## ★★★ F40 — a teardown segfault throws away a measurement that already succeeded

**Status: live, hit at Step 10/10 of the optimize stage** · severity: the baseline the whole perf
run is judged against is discarded after being computed · reported: not yet

The optimize stage's final discovery step is *"Measuring the baseline latency (trace+1CQ)"*. It ran,
and it ran **well** — from its own tracy log:

```
[perf] iter 4/8  744.0 ms/call
[perf] iter 5/8  745.9 ms/call
[perf] iter 6/8  749.7 ms/call
[perf] iter 7/8  753.3 ms/call
[perf] iter 8/8  756.4 ms/call
[perf] stages: {'prefill': ok pcc=1.000002, 'decode': ok pcc=1.000000,
                'flow': ok pcc=1.000000, 'vocode': ok pcc=1.000001}
FORWARD_WALL_MS=756.4513
TRACE_PER_TOKEN_MS=756.4513
TRACE_REPLAY_PATH=trace+1cq native batch=1
```

Eight iterations, all four pipeline stages traced and PCC-clean, the baseline printed. The very next
line:

```
Fatal Python error: Segmentation fault

Thread (most recent call first):
  ttnn/ttnn/distributed/distributed.py:689 in close_mesh_device
  models/demos/voxtral_tts_full/tests/e2e/test_main_perf.py:108 in _close_device
  models/demos/voxtral_tts_full/tests/e2e/test_main_perf.py:233 in test_main_perf
Aborted (core dumped)
```

**The crash is in device teardown, after the work.** `close_mesh_device` segfaults once the
measurement is complete, killing the process before tracy's post-processing writes its device CSV.
The tool then reports:

```
  ✗ discovery failed (TracyRunError):
      tracy run exit 1
      Fatal Python error: Segmentation fault
      Aborted (core dumped)
      AssertionError: cpp_device_perf_report.csv not found and legacy device log
      profile_log_device.csv is also missing …
```

and continues without a baseline:

```
  [optimize/cc] discovery exited 1 but the manifest is complete; continuing.
```

### It recurs, and it costs a rung rather than just a baseline

Not a one-off. Hours later, on the `knob:dtype` rung, the ledger recorded another attempt lost the
same way:

```
{'op_signature': 'MatmulDeviceOperation 32 x 4096 x 3072', 'kernel_kind': 'dtype',
 'fullpipe_ms': None, 'fullpipe_delta_ms': None,
 'note': "wedged/crashed when tried: tracy run exit 1 (log: /tmp/perf_mcp_d9qaacck/run0_tracy.log)
          Fatal Python error: Segmentation fault
          Aborted (core dumped)"}
```

`fullpipe_ms: None` — the attempt produced no measurement at all. To the tool's credit the entry is
recorded honestly as *"wedged/crashed when tried"* rather than silently dropped, so the ladder knows
the rung was attempted. But the effect is that a profiler crash converts a candidate optimisation
into a hole in the search: not rejected on merit, just unmeasurable.

### Three separable problems

1. **`close_mesh_device` segfaults.** On the same device, in the same session, `test_tts_e2e.py`
   opens and closes cleanly seven times over (7 passed in 50.96s). Whatever the generated perf test
   leaves open — a trace buffer, a captured program — makes teardown fatal. This is the underlying
   defect and it is in the tool's own generated artifact (see the F31 correction).
2. **A completed measurement is discarded because of what happened afterwards.** `756.4513 ms` was
   on stdout. The run is judged a total failure over a CSV that a *post-measurement* crash prevented
   from being written. Nothing tries to recover the number that was already printed.
3. **It proceeds anyway, without a baseline.** `discovery exited 1 but the manifest is complete;
   continuing` is the worst of both: not a halt the operator must resolve, and not a usable
   reference point either. Every later "win" is measured against nothing.

F31 recorded the reporting half of this (an abort surfaced as a missing file). F40 is the same
masking with the additional insult that the data existed.

### Fixes

1. **Parse the numbers the run already emitted.** `FORWARD_WALL_MS` / `TRACE_PER_TOKEN_MS` are
   printed on stdout in the tool's own format; a run that produced them has produced a baseline,
   whatever happened during teardown.
2. **Do not let teardown fail a run.** Wrap `_close_device` so a crash after the measurement is a
   warning; better, let the process exit without closing — the OS reclaims the device, and this is a
   short-lived profiling child.
3. **Report the abort as the cause, not the missing CSV** (F31 fix 1, unchanged).
4. **Refuse to optimize with no baseline**, or say loudly that improvements will be unverifiable.
   `continuing` after a failed baseline measurement is a decision worth surfacing.

---

## ★★ F41 — the "is the depth knob inert?" check saturates against its own truncation limit, and the knob it tests is only half-wired

**Status: live, hit sizing the profiling window** · severity: the tool concludes its depth cap does
not work, and sizes the profiling window from an explicitly unverified floor · reported: not yet

At the start of pipeline optimization the tool checks whether the depth knob actually reduces work:

```
[optimize/cc] depth knob is INERT: capping to 2 produced the SAME work signal (50000) as the full
              model, so the cap never reached the builder. Refusing to report a coverage window
              measured against an uncapped model.
[optimize/cc] coverage (unverified-floor): 437 distinct op(s) -> TT_PERF_LAYERS=2
[optimize/cc] coverage-sized profiling window: TT_PERF_LAYERS=2 (covers all block types)
```

**The refusal is good practice** — it declines to trust a coverage window it could not verify, and
says so. The conclusion it refuses on, however, looks wrong in two independent ways.

### 1. The work signal is truncated at exactly the value it reported — INFERRED

`models/experimental/perf_automation/cc_optimize/_op_sig_probe.py:616`:

```python
print("PERF_OP_SIG_SEQUENCE=" + json.dumps(_SEQ[:50000]), flush=True)
```

The op-signature sequence is cut to its first **50000** entries before being emitted — and `50000`
is precisely the "SAME work signal" reported for both the capped and the uncapped run. Any two runs
that each emit ≥50000 op invocations produce byte-identical signals here, so the comparison cannot
distinguish *"the cap did nothing"* from *"both runs exceeded the truncation limit"*. For an 8-frame
run over a 26-layer backbone plus flow and codec stacks, exceeding 50000 op invocations at **both**
depths is entirely expected.

The next line of the same run prints the comparison in its rawest form:

```
[optimize/cc] depth-knob bridge: {'TT_PERF_LAYERS': '2'} did not reduce work
              (op-count 50000->50000); ignoring
[optimize/cc] measuring FULL-model end-to-end (BEFORE) — ALL layers (uncapped), no tracy
```

`op-count 50000->50000` — the truncation constant on both sides of the arrow, reported as an
op-count. A depth cap from 26 layers to 2 producing *byte-identical* op counts is not a plausible
measurement; a clipped sequence compared against another clipped sequence is.

*(Still marked inferred rather than proven: I did not dump the two raw `_SEQ` lists to show both
exceeded 50000. What is established is that the compared value equals the truncation constant
exactly, on both sides, and that saturation produces precisely this symptom.)*

The consequence is immediate and visible in the next line: the knob is ignored and the tool measures
the **full uncapped model**, which is the opposite of the bounded profiling window the knob exists
to provide.

### 2. The knob really is only half-wired — VERIFIED

The generated perf test documents itself as capping everything:

```python
# DEPTH. A POSITIVE TT_PERF_LAYERS caps every repeated stack; ABSENT means ALL LAYERS …
```

and then builds with only the backbone capped (`test_main_perf.py:123`):

```python
pipe = P.build_pipeline(device, model=model, layers=PERF_LAYERS)
```

No `flow_layers`, no `vocode_layers` — both of which `build_pipeline` accepts, and both of which the
e2e suite's own `test_layer_cap_is_not_inert` exercises:

```python
capped = P.build_pipeline(device, model=hf_model, layers=2, flow_layers=1, vocode_layers=1)
assert len(capped.backbone_layers) == 2
assert len(capped.flow_layers) == 1
assert capped.depths["vocode"] == 1
```

That test **passes**. So the pipeline's cap works; the generated perf test simply does not use two
thirds of it, while claiming in a comment that it does. Capping the backbone alone leaves the flow
and vocode stacks at full depth — and the tool's own `partial_stage_coverage` note records that
those untimed stages dominate a real run.

### Why the two compound

The tool ends up believing its depth cap is inert when the more likely truth is *"the cap is partial
and the detector is saturated"*. It then profiles a window sized from an unverified floor — i.e.
more work than intended — which is the same direction that produces the profiler crashes in F31 and
F40. A depth cap that genuinely bounded all three stacks is the documented remedy for exactly those
overflows.

### Fixes

1. **Do not compare truncated sequences.** Compare lengths before truncation, or hash the full
   sequence, or raise the cap and record that it was hit. A detector whose two inputs are both
   clipped to the same constant cannot report anything but "identical".
2. **Say when the signal saturated.** `work signal 50000 == truncation limit; comparison
   inconclusive` is a different message from `the cap never reached the builder`, and it points at
   the right fix.
3. **Wire the whole knob.** Pass `flow_layers` / `vocode_layers` through from `TT_PERF_LAYERS`, or
   change the comment to say only the backbone is capped.

---

## ★★★★ F42 — the correctness gate reported `pcc: 33.612, pcc_verified: true`

**Status: live, observed during the optimize run** · severity: the gate that protects every perf
edit can return "correctness verified" on a value that is not a correlation coefficient ·
reported: not yet

This is the most serious finding in this document. Everything the optimize stage banks — every
"win", every commit — rests on one question: *did the edit preserve correctness?* On 2026-08-16 at
~02:07 the tool answered that question like this, and handed the answer to the agent verbatim:

```json
{
  "status": "ok",
  "pcc": 33.612,
  "pcc_verified": true,
  "threshold": 0.99
}
```

A Pearson correlation coefficient is bounded in **[-1, 1]**. `33.612` is not a PCC that is very
good; it is not a PCC at all. The gate compared it against 0.99, found it larger, and declared
correctness **verified**.

### Why nothing caught it

Three mechanisms compose, each defensible alone.

**1. The parse is permissive** (`agent/pcc_runner.py:22`):

```python
_PCC_RE = re.compile(r"(?i)pcc[^\n]*?[:=]\s*(-?\d+\.\d+)")
```

`pcc[^\n]*?[:=]` allows *arbitrary text* between the word "pcc" and the delimiter, so any line that
merely mentions pcc and later contains `: <float>` matches — a duration, a percentage, a byte count,
a core id.

**2. The reduction is `min`, and `min` cannot detect this** (`pcc_runner.py:25-34`). Taking the
worst observed PCC is good design, and its docstring explains real bugs it fixed. But `min` only
protects against values that are too *low*. When every match is spurious, the smallest spurious
value is returned with full confidence — here, 33.612.

**3. The exit code is deliberately not consulted** (`pcc_runner.py:105-117`):

> *"PCC IS the correctness signal for a perf edit. A non-zero pytest EXIT with PCC>=threshold is NOT
> an edit-induced regression: the e2e gate also enforces BRING-UP checks (Gate-2) and the process
> prints benign nanobind teardown leaks at interpreter shutdown — BOTH set a non-zero exit while the
> math is perfect… Gating on the raw return code here rejected every edit. So gate on PCC"*

That reasoning is sound and hard-won. But it means the **parsed number is the only signal left**.
Once it is wrong, nothing else is watching.

```python
effective = max(float(threshold or 0.0), _operator_pcc_floor())
return ({"status": "ok", "pcc": pcc, "pcc_verified": True, "threshold": effective}
        if pcc >= effective else {"status": "pcc_low", ...})
```

There is no upper-bound check anywhere in the path.

### What it means

A test run whose output does not contain the expected `e2e PCC=0.9999…` line — because the test
errored earlier, because a format changed, because a different case ran — but which does contain any
line matching `pcc…: <number ≥ threshold>` is reported as `status: ok`, `pcc_verified: true`, and
the edit that produced it is banked as correct. **The failure is silent and it is in the unsafe
direction.**

The gate does work in the ordinary case: minutes earlier the same gate correctly caught the
`bfloat8_b` weight experiment at `pcc: 0.349, status: pcc_low`, and the agent reverted it. That is
what makes this dangerous rather than merely broken — it is right often enough to be trusted.

### Fixes (the first is one line)

1. **Reject out-of-range values as parse failures — WITH a tolerance.** A real correlation can
   exceed unity by floating-point rounding (this port's own trace selftest prints `pcc=1.000017`),
   so a strict `-1.0 <= pcc <= 1.0` test would fail valid runs. Use
   `not (-1.0 - 1e-3 <= pcc <= 1.0 + 1e-3)` → `status: crash`, `pcc_verified: False`. The observed
   bogus values span 1.525 to 47.779 — all outside that band, while this port's legitimate
   overshoot (1.000017) sits inside it. Tight matters: a `±0.1` tolerance would admit 1.525.
2. **Tighten the regex.** `pcc[^\n]*?[:=]` should not span arbitrary text; require the delimiter to
   follow `pcc` closely (`pcc\s*[:=]` or `\bpcc\b[^,\n]{0,12}[:=]`).
3. **Cross-check the exit code when the parse is implausible.** Ignoring the return code is
   reasonable *given a trustworthy number*; when the number is out of range, the return code is the
   only remaining evidence and should decide.
4. **Report `pcc_verified: false` when the value is not in range**, rather than asserting
   verification of a number the tool cannot have measured.

### IT RECURS — a second instance, and the mechanism is now exact

2026-08-16, ~11:20, same run, a different edit:

```json
{"full_pipeline": {"status": "regressed", "full_pipeline_ms": 1572.2449, ...},
 "pcc":           {"status": "ok", "pcc": 47.779}}
```

**`pcc: 47.779`, accepted as ok.** And half an hour later, a third:

```json
{"pcc": {"status": "ok", "pcc": 1.525}}
```

**Four instances in roughly twelve hours of optimisation — `33.612`, `47.779`, `1.525`, `11.021` —
all `status: ok`.** Roughly one every three hours. This is not rare.

**And the reason nothing was banked is not the correctness gate.** In all four windows the *other*
gate independently said no:

| # | pcc verdict | full_pipeline verdict | banked? |
|---|---|---|---|
| 33.612 | ok (bogus) | regressed | no — blocked by perf |
| 47.779 | ok (bogus) | regressed | no — blocked by perf |
| 1.525 | ok (bogus) | regressed | no — blocked by perf |
| 11.021 | ok (bogus) | regressed | no — blocked by perf |

`gates_allow_banking()` requires **both** `pcc.status == "ok"` and `full_pipeline.status == "ok"`.
The two are meant to be independent checks on different properties. Across four observed instances
**only one of them was actually functioning**, and it happened to be the one saying no. Had any of
those edits been faster as well as unverified, it would have been committed as a correctness-checked
win. The protection here came from the edits being slow, not from the correctness gate.

**The third value is the dangerous one.** `33.612` is self-evidently not a correlation; anyone
glancing at it would stop. `1.525` looks almost plausible — close enough to unity that a reader
might file it under rounding, and close enough that a *loose* tolerance would admit it. The observed
range of bogus values therefore runs from 1.525 to 47.779, while legitimate overshoot in this port
is `1.000017`. The margin is wide, but only if the tolerance is tight: a careless `±0.1` or `±0.5`
would pass 1.525 straight through.

**The mechanism, established by running the tool's own regex over a real passing e2e log:**

```
matches in a healthy run: 11        min (what parse_pcc returns) = 0.9999   <- correct
```

In a healthy run the real per-stage PCC lines are present, the minimum of them is a true PCC, and
the gate behaves. The bogus readings therefore mean **no sub-1.0 PCC line was present at all** —
the e2e run did not reach its comparison — leaving only a spurious match, which by being the only
match becomes the `min` and clears the threshold. The failure mode is precisely: *the test produced
no PCC, and the absence was reported as a pass.*

### The one-line fix needs a tolerance — a strict range check would break valid runs

Worth correcting, because the naive version of this fix is wrong. The same healthy log contains two
legitimate matches **above 1.0**:

```
  1.000017  <-  pcc=1.000017      (trace selftest, prefill stage)
```

A real correlation can exceed unity by rounding. So `if not (-1.0 <= pcc <= 1.0): reject` would fail
correct runs. The check must carry a tolerance:

```python
if pcc is not None and not (-1.0 - 1e-3 <= pcc <= 1.0 + 1e-3):
    return {"status": "crash", "pcc_verified": False,
            "error": f"parsed PCC {pcc} is out of range — the matched line was not a correlation"}
```

`33.612` and `47.779` are four orders of magnitude outside that band; `1.000017` sits inside it.

### This verdict is the sole correctness input to "may I bank this win?"

The false green is not confined to a report. `git_commit` refuses to bank a win unless
`gates_allow_banking()` says so, and that function reads exactly one correctness field
(`perf_mcp.py:3055-3067`):

```python
def gates_allow_banking() -> tuple:
    """...Absent verdicts are refused, not assumed: an unrun gate is not a passed gate."""
    v = gate_verdicts()
    pcc, fp = v.get("pcc") or {}, v.get("full_pipeline") or {}
    if not pcc:                              return False, "check_pcc has not run since the last commit"
    if str(pcc.get("status")) != "ok":       return False, "check_pcc status=%s" % pcc.get("status")
```

`pcc.status` is precisely the field F42 sets to `"ok"` on 33.612, 47.779 and 1.525. So the gate that
exists to stop an incorrect edit being committed consults a verdict that can be wrong in the unsafe
direction, and has no other correctness signal to fall back on.

**The enforcement design is right and worth crediting** — the docstring's *"an unrun gate is not a
passed gate"* is exactly the correct default, and the comment above `git_commit` records that it was
added *after* two wins (`d54438bb4b`, `7fac4ae685`) were banked while the end-to-end best had not
moved, because "nothing here asked the gates". The mechanism learned from a real failure. It is
simply only as trustworthy as the one number feeding it, which is why F42's one-line range check
matters more than its severity as a reporting bug would suggest.

### No harm done on this run — the severity is potential, not realised

Stated plainly, because it matters for how urgently this is read: **nothing was banked during the
false-green window.** The commit count on the isolation branch was 6 before the `33.612` reading and
6 after it; the best full-pipeline time was 1481.57 ms on both sides. The agent happened to be
mid-experiment rather than at a decision point, and the very next gate reading caught the same edit
honestly:

```
pcc gate: ok (33.612)  ->  pcc_low (0.0536)
```

So this run's six commits were each measured against a real PCC. The defect is that the guarantee
does not hold *in general*, not that it failed here — the window in which a wrong answer is accepted
exists, is silent, and opens in the unsafe direction. It was found only because a human read an
impossible number; nothing in the system objected to it.

### Settled — the port's correctness, measured independently of this gate

The optimize run was stopped after ~17.5 h and 11 banked commits, and `test_e2e_pcc` was run
directly against the resulting tree — not read from the gate that produced 33.612, 47.779, 1.525 and
11.021:

```
e2e PCC = 0.9999803900718689
frames tt=(8,37) ref=(8,37)  exact_match=True  code_flips=0
per-step hidden PCC: 1.000000 x9
```

Against the pre-optimisation 0.9999834299087524, a drop of 3e-6 — negligible, far above the 0.99
threshold, with exact audio-code equality and zero code flips preserved. **The eleven optimisations
did not damage correctness.**

That is a good outcome and it is worth being precise about what it does *not* show: it does not
retroactively validate the gate. Four times in twelve hours the gate reported `pcc_verified: true`
on a number that cannot be a correlation, and each time it was the *perf* gate that stopped the
commit. The port survived because the edits it would have waved through happened to be slow.

*(Two other tests failed in that run, neither about correctness: `gate1` reports
`flow_matching: live stub differs from its graduated snapshot` — the documented `stub_edits_break_gate1`
behaviour, since the optimizer edited that stub — and `trace_capture_selftest` hits a
`FileNotFoundError` on `_captured/tts_backbone/args.pt`, which the isolation worktree never staged.)*

---

## ★★★ F43 — `TRACE_PER_TOKEN_MS` is not per token: it is per call, so throughput is wrong by the output length

**Status: live in this checkout** · severity: the headline throughput metric is off by a factor of
`OSL`, silently · reported: not yet

The generated perf test times a whole forward and prints the same number twice
(`tests/e2e/test_main_perf.py:209-228`):

```python
_iters = int(os.environ.get("TT_PERF_REPLAY_ITERS", "8"))
for _i in range(_iters):
    out = _forward()                       # prefill(ISL) + OSL audio frames
    ttnn.synchronize_device(device)
    ...
_ms = (time.monotonic() - _t0) * 1000.0 / max(_done, 1)   # ms per CALL

print("FORWARD_WALL_MS=%.4f" % _ms)
print("TRACE_PER_TOKEN_MS=%.4f" % _ms)     # <- the SAME value, not divided by tokens
```

One call is **`PERF_ISL_TOKENS=32` prompt rows plus `PERF_OSL_TOKENS=4` audio frames**. The per-call
wall time is published as the per-token time, so:

- `TRACE_PER_TOKEN_MS` is too large by roughly `OSL` (4× here), plus the whole one-off prefill;
- the ledger inherits it — `perf_mcp_baseline…json` records `forward_wall_ms: 598.94` and
  `per_token_ms: 598.94`, **identical**, which is the signature of the bug;
- `tokens_per_sec: 1.6696` is wrong by the same factor, and it is the number a reader would quote.

It scales with a knob, which is what makes it dangerous: raise `TT_PERF_OSL_TOKENS` to amortise
prefill — normally the right instinct — and the reported "per token" cost rises proportionally,
making a *better* measurement look like a regression.

### Why it matters beyond cosmetics

This is the metric a port is judged and compared by. A hand-written implementation of this model
reports **26.9 ms/frame** (RTF 0.357). Comparing that against the tool's `per_token_ms` of 598.94
implies a ~22× gap; the true per-frame gap cannot be read off these numbers at all, because one side
is per frame and the other is per (prefill + 4 frames). Any comparison table built from the tool's
own output is wrong by construction, in the direction that flatters the hand-written port.

### Fixes

1. **Divide.** `per_token = _ms / max(PERF_OSL_TOKENS, 1)`, and keep `FORWARD_WALL_MS` as the
   undivided call time — both are useful, they are just not the same number.
2. **Subtract prefill, or report it separately.** A per-token metric that includes a one-off prefill
   is not a decode rate; `TTFT_ms` already exists in the scorecard for that half and is currently
   `NA`.
3. **Assert the invariant.** `per_token_ms <= forward_wall_ms` whenever `OSL > 1`; equality is only
   valid at `OSL == 1`, and the two being equal is exactly what this bug looks like.

---

## ★★★★ F44 — the number every optimisation is judged by is a capture-and-verify harness, not an inference measurement

**Status: live; it is the objective function of the entire optimize stage** · severity: the metric
being minimised is dominated by costs that do not exist in deployment · reported: not yet

The optimize stage ranks every candidate edit by `full_pipeline_ms`. That number comes from the
generated perf test's `_forward()`, which is:

```python
def _forward():
    """The model's OWN self-recording function: per stage it stages inputs resident,
    then captures / executes / releases its own trace."""
    return pipe.trace_capture_selftest(device, verbose=True)
```

`trace_capture_selftest` (`tt/pipeline.py:597`) is documented as *"Capture **ONE step per stage** in
begin/end_trace_capture, execute it, check it against the eager result, then RELEASE before the next
stage."* Per stage, inside the timed region:

```python
eager  = ttnn.to_torch(step()).float()   # 1. a full EAGER execution
tid    = ttnn.begin_trace_capture(...)   # 2. trace CAPTURE (record + compile)
out    = step()
         ttnn.end_trace_capture(...)
         ttnn.execute_trace(...)         # 3. ONE trace replay  <- the only deployment-shaped work
traced = ttnn.to_torch(out).float()      # 4. a second device->host readback
         ttnn.release_trace(device, tid) # 5. trace RELEASE
p      = ref.pcc(eager, traced)          # 6. a HOST-side PCC computation
```

Steps 1, 2, 4, 5 and 6 do not occur in a deployed inference loop at all. Capture and release happen
once at startup; the eager pass never happens; a production step does not read its own output back
twice and correlate it on the host. Only step 3 resembles serving.

### What the metric therefore weights

Measured shape of the run, from the test's own runtime output:

```
PERF_ISL_TOKENS=32
PERF_OSL_TOKENS=4
[perf] vocode input pinned to 4 audio frames
[trace] prefill: OK C=224   decode: OK C=64   flow: OK C=3   vocode: OK C=32
```

- **one** decode step, **one** prefill, **one** flow, vocode over 4 frames;
- against a product workload of a 200-id prompt and **24 frames** (`demo_tts.py --max-frames 24`);
- so the repeated cost that dominates deployment — decode, 24× — enters the objective **once**,
  while per-stage capture/release/eager/readback overhead enters **four times, every iteration**.

An edit that makes trace *capture* cheaper scores. An edit that makes steady-state *decode* cheaper
barely moves the number. The optimizer is not being dishonest — it is faithfully minimising what it
was handed.

### The ops profiled are not the ops that run — they are different SHAPES

A small workload would still be defensible if it exercised the same operations. It does not. The
sequence axis is pinned to a capacity derived from the workload
(`tt/pipeline.py:353`):

```python
cap = int(capacity or min(self.max_context, 32 * ((prompt_len + n_max + 31) // 32)))
```

Two runs on the same build, same day:

```
PERF  (ISL=32, OSL=4)     [trace] prefill: C=32    decode: C=64    flow: C=3   vocode: C=32
REAL  (200-id, 24 frames) [trace] prefill: C=224   decode: C=224   flow: C=3   vocode: C=32
```

**CORRECTION — the damage is narrower than "every matmul differs", and the distinction matters.**
An earlier draft of this entry said every backbone matmul is a different shape between the two runs.
That is wrong, and being wrong in the tool's *disfavour* is worth fixing explicitly.

Decode emits **one token at a time**, and the device computes in 32×32 tiles, so that single row is
padded to a full tile. Decode projections are therefore `M=32` **permanently** — in the small job
and in production alike. Splitting the ops by what they actually depend on:

| op class | small job | real job | same shape? |
|---|---|---|---|
| decode projections (QKV, MLP, out) | M=32 (1 padded token) | M=32 (1 padded token) | **yes — tuned correctly** |
| decode attention (scores, probs) | context **64** | context **224** | no — 3.5× |
| prefill (all of it) | M=**32** | M=**224** | no — 7× |

The ledger shows both classes. `32 x 9216 x 3072` and `32 x 4096 x 3072` are projections, whose
shape does not move with context. `32 x 128 x 32` is attention — 128 is `head_dim` and the trailing
**32 is the context length**; that same op is `32 x 128 x 224` in production.

So the bandwidth-bound reasoning quoted below is **genuinely correct for decode projections**: M is
one tile there permanently, the weight really is streamed for a single row of work, and "only bytes
help" holds at any horizon. The custom hi/lo kernel was built on a premise that is true in
production. What is mis-tuned is **attention**, profiled at under a third of its real context, and
**the whole of prefill**, profiled at a seventh of its real size — plus the proportion and depth
distortions below.

The tool established that sensitivity itself, in a commit message:

```
c90bc44d52 hilo kernel: pick the core count per shape, because full grid is the wrong answer here
```

It learned that the right core count depends on the shape, and learned it at the wrong shape. So the
concern is not merely that the *proportions* are off (below) — it is that a config tuned here has no
guarantee of being the right config there, and may be a regression at the real size.

**Why token count changes a matmul at all, and why it changes the STRATEGY.** A matmul here is
activations × weights. The weights are fixed; the activations carry one row per token, so the token
count sets M. The device computes in 32×32 tiles, so 32 tokens is **one tile row** and 224 tokens is
**seven**. At one tile row the entire weight matrix is streamed in and each weight is used once —
bandwidth-bound. At seven, the same weights are reused across seven rows of work for the same
traffic — seven times the arithmetic intensity, a materially different regime with different levers.

Every matmul the optimizer profiled sits at the extreme end of that scale. From its own ledger:

```
MatmulDeviceOperation 32 x 9216 x 3072
MatmulDeviceOperation 32 x 4096 x 3072
MatmulDeviceOperation 32 x 3072 x 1024
MatmulDeviceOperation 32 x 128  x 32
```

**Every one begins with 32** — a single tile row. And the strategy it derived says so explicitly:

> *"these projections are DRAM-bandwidth-bound (M is a single tile, so the whole cost is streaming
> the weight in), and the profile puts them at 329–512 GB/s — already ON this board's roofline.
> Cores cannot help; only BYTES can"*

That reasoning is **sound and stays sound** for the decode projections it was applied to: decode
emits one token, padded to one tile, at every horizon — so M=32 is not an artefact of the small job,
it is the permanent production shape. The custom hi/lo kernel rests on a premise that holds.

Where M *does* grow with the workload is **prefill** (M = padded prompt length: 32 here, 224 real).
There, seven tile rows give 7× the reuse for the same weight traffic, so "already at the bandwidth
roof, cores cannot help" is exactly the claim that can invert — and prefill was never profiled at
its real size to find out.

**This also disposes of the obvious cheap fix.** "Keep the small job for profiling, run the real job
for timing" is not sufficient: per-op attribution gathered at C=64 mis-ranks the ops that dominate
at C=224, so the ladder would still aim at the wrong targets. Three further distortions compound it:

- **Context depth.** Decode is timed at step 1 of 1, when the KV cache is shortest and attention is
  cheapest. In a 24-frame run every later frame attends over more context, so attention's true share
  is understated.
- **Proportions.** One prefill against one decode makes prefill look ~24× more important than a real
  run makes it; the ladder picks its `next_target` by largest `gap_ms`, so it aims accordingly.
- **Cold execution.** A single captured-and-released execution carries first-touch effects that a
  replayed steady state does not.

**The mechanism for doing this right already exists and was not used.** The tool resolves Tracy
signposts to capture a *window* inside a run (`agent/probes.py:1295 resolve_signposts`,
`start_signpost`/`end_signpost`, `perf_mcp.py:_signpost_blocks`). For this model it found none and
said so, then proceeded:

```
Step 9/10  Locating profiler signposts
   WARN signpost: no tracy signposts in .../tests/ -- using default 'start'/'stop' (full capture)
```

Signposting a steady-state window — say frames 10–12 of the real 24-frame run — keeps the marker
count inside the 12000 budget *and* profiles the real shapes at a realistic context depth. That is
the correct fix, and it is F20's pattern once more: the capability exists, its absence is detected,
and the run continues without it.

### WHY it is shaped this way — and why the reason does not apply to the timing

There is a real constraint behind the small workload, and it is worth stating fairly before the
criticism. The tool documents it (`agent/perf_test_gen.py:1184`):

> *"BOUNDED + profiler-safe so tracy's 12000-marker buffer never overflows: cap the work (decode …)"*

Every device op writes markers into a fixed buffer holding ~12000. A 24-frame run drives the
26-layer backbone 24 times plus flow and codec each frame — far past that. Overflow means dropped
markers and a partial profile, which is the same failure `tt_metal/impl/profiler/profiler.cpp` was
patched to tolerate. Capping the work so a profile is *valid* is correct engineering.

A second, independent limit shapes the per-stage structure: the trace region is
`DEFAULT_TRACE_REGION_SIZE = 200_000_000`, and `trace_capture_selftest` notes that **"stage traces
must not co-reside"** — all four stage traces do not fit at once, so each is captured, executed and
released in turn. That is also a real device constraint, not a choice.

**The defect is that this bound is then applied to a measurement that does not profile.** The
win/lose timing runs with Tracy off — the log says so plainly:

```
[optimize/cc] measuring FULL-model end-to-end (BEFORE) — ALL layers (uncapped), no tracy
```

With no profiler attached there is no marker buffer to overflow, so nothing stops the timing run
from executing the real 24-frame workload. It inherits a restriction that exists solely for
profiling. The two jobs — *attribute time to ops* (needs Tracy, must be small) and *decide whether
an edit is faster* (needs realism, has no marker limit) — have been collapsed into one workload
sized for the first.

### Why this is separable from F41 and F43

F41 is about the depth knob failing to bound the window. F43 is about the per-token label being the
per-call time. **F44 is about the workload itself being the wrong shape** — even with the depth knob
working and the label fixed, timing a capture-verify-release cycle would still not be timing
inference.

### It also explains a number that cannot be compared

A hand-written TTNN port of this model reports **26.9 ms/frame** steady state (RTF 0.357). The
tool's `full_pipeline_ms` is **1466.79 ms** for one capture-and-verify pass over four stages. These
are not the same quantity and no ratio between them is meaningful. Any comparison table that places
them side by side is wrong, and the error is large — not a few percent, but a different measurement
entirely.

### Fixes

0. **Profile a signposted WINDOW inside the real run, not a shrunken run.** The mechanism already
   exists (`resolve_signposts`, `start_signpost`/`end_signpost`) and reported itself missing here.
   Marking a steady-state window — e.g. frames 10–12 of the real 24-frame workload — stays inside
   the 12000-marker budget while profiling the real shapes (C=224, not C=64) at a realistic context
   depth. Simply keeping the small job for profiling is NOT sufficient: attribution gathered at C=64
   mis-ranks the ops that dominate at C=224.
0b. **Run the accept/reject timing on the real workload.** It executes with Tracy off, so no marker
   budget constrains it. These two are the root fixes; the rest are refinements.
1. **Time replay, not capture.** Capture once outside the timed region; time `execute_trace` only.
   The pipeline already separates `<stage>_trace_setup` from `<stage>_trace_step`, so the split
   exists.
2. **Drop the verification work from the timed path.** The eager execution, the second readback and
   `ref.pcc` are correctness machinery; they belong in the PCC gate, which already runs separately.
3. **Weight decode by the real horizon.** Run OSL decode steps per iteration rather than one, so the
   objective's shape matches the product's (`TT_PERF_OSL_TOKENS` and `PERF_MCP_FULLPIPE_TOKENS`
   already exist as the knobs).
4. **Report the shape next to the number.** `1466.79 ms (prefill 1 + decode 1 + flow 1 + vocode 4,
   capture+verify included)` is honest; `FULL-model end-to-end` is not.

---

## ★★★★★ F46 — the profiled decode path and the shipped decode path are different ALGORITHMS

**Status: live** · severity: twelve hours of optimisation were applied to a function the product
never calls; its gains reach the product only as a side effect via shared helpers · reported: not yet

This subsumes the shape and mode concerns of F44/F45. Those describe a measurement taken at the
wrong size, in the wrong execution mode. This one is worse: **the code being measured is not the
code that runs.**

### VERIFIED — the exhaustive check, reproducible in four commands

This claim is large enough that it was checked exhaustively rather than read off one function.

**(a) Every reference to the fast path in the whole demo directory:**

```
$ grep -rn "decode_step\|decode_prefill" models/demos/voxtral_tts_full --include=*.py
tests/e2e/test_tts_perf.py:7   docstring: "The unit is ONE decode frame: … decode_step"
tt/pipeline.py:258             comment on the _kv field
tt/pipeline.py:470,502,507     definitions + an assert
tt/pipeline.py:536             self.decode_prefill(...)      <- inside decode_trace_setup
tt/pipeline.py:545             return self.decode_step(...)  <- inside decode_trace_step
```

**Two call sites, both inside the trace harness.** Nothing else in the port reaches them.

**(b) The demo calls the other one** — `demo/demo_tts.py:66` is `pipe.run_tts(...)`, and the file's
own header states: *"Runs the SAME `tt/pipeline.py::run_tts` the e2e test asserts on — there is
exactly one copy of it."* So demo and correctness test agree with each other, and both differ from
what is profiled.

**(c) No KV cache exists on the shipped path.** `_stubs/tts_backbone.py` iterates
`layer(x, causal=True)` with no cache argument, and `_stubs/decoder_layer.py` **accepts a cache and
discards it**:

```python
def __call__(self, x, cis=None, bias=None, cache=None):
    return self.layer(x, causal=bias is not None)     # `cache` is never used
```

So even if a caller supplied one it would be silently dropped. The `cache`/`cache_key` parameters on
the attention stub (`_stubs/attention.py:44`) are likewise unreached from `run_tts`.

**(d) Both perf tests target the fast path.** `test_main_perf.py` times `trace_capture_selftest`
(→ `decode_trace_step` → `decode_step`), and `test_tts_perf.py` says so in its first line: *"The unit
is ONE decode frame: `tt/pipeline.py::decode_step` advancing a single position against [the resident
cache]."* Neither times `run_tts`.

### Two decode implementations, only one of them reachable

`tt/pipeline.py` contains an incremental, KV-resident decode:

```python
def decode_prefill(self, inputs_embeds, capacity=None):   # line 470  — seed resident KV
def decode_step(self, emb, pos=None):                     # line 502  — ONE token against it
```

`decode_trace_setup` documents the intent exactly: *"Seed resident KV outside the trace, pin the
step position, pre-upload the step input."* One token in, cached keys and values, constant work per
frame.

Every caller of that pair, in the whole demo directory:

```
pipeline.py:536   self.decode_prefill(...)     <- inside decode_trace_setup
pipeline.py:545   return self.decode_step(...) <- inside decode_trace_step
```

Two call sites, both inside the trace-capture harness. **`run_tts` does not call either.** The
shipped generation loop does this instead:

```python
for i in range(n_max):
    codes  = self._run_flow(h)
    emb    = self.embed_frame(codes)
    embeds = ttnn.concat([embeds, emb], dim=1)
    length = prompt_len + len(frames)
    # "the whole prompt plus every frame so far, read at the last real position"
    h = self._row(self._run_backbone(self._pin(embeds, length, cap)), length - 1)
```

Every frame re-runs **all 26 layers over the entire padded 224-token sequence** and keeps one row.
No cache is consulted. Per-frame cost is O(sequence), so an utterance is O(n²) where the profiled
path is O(n).

### What this means for every number in the perf half

- The optimizer's target list, its roofline gap, its 19 attempts and 11 wins were all computed
  against `decode_step`.
- The product calls `_run_backbone` on a growing prefix.
- **They share ops but not shapes, not counts, and not complexity.** A win on the profiled path is
  not evidence of a win on the shipped path, and can be a loss there.

It also explains the magnitude of the gap against a hand-written port far better than eager-vs-traced
alone: 26.9 ms/frame (traced, incremental) against ~500 ms/frame from this port's own README
(eager, full re-prefill). Two of the three differences — caching and dispatch mode — are structural,
not tuning.

### The correctness result is unaffected

Worth stating plainly so this is not over-read. `test_e2e_pcc` drives `run_tts` — the shipped path —
and it measured **0.9999834 with exact code match and zero code flips**. The port is correct. What
is not established is anything about its speed.

### WHY TWO PATHS EXIST — the stub contract and the trace contract are incompatible

Established from the tool's own plan and the reference implementation, not inferred.

**The HF reference is the efficient one.** `modeling_voxtral_tts.py:198 prefill_then_step` prefills
once into a cache and then advances one position at a time:

```python
def prefill_then_step(self, inputs_embeds, step_embeds):
    cache, P = {}, inputs_embeds.shape[1]
    for layer in self.layers:
        x = layer(x, cis, common.causal_bias(P, x.dtype), cache)   # prefill INTO the cache
    for t in range(step_embeds.shape[1]):
        x = step_embeds[:, t: t + 1]                               # ONE position
        for layer in self.layers:
            x = layer(x, cis_t, None, cache)                       # reuse it; no mask needed
```

So `decode_step` is the **faithful** mirror of the reference, and `run_tts` is the divergence — it
reaches the same numbers by recomputation instead of caching.

**The demo path cannot cache — but not because anything forbids it.** Gates 1 and 2 require it to
route through the graduated stubs, and those stubs simply have no cache implementation:

```python
_stubs/attention.py:44
    def __call__(self, h, cis=None, bias=None, cache=None, cache_key=None):
        return self.attn(h, causal=bias is not None)     # `cache` accepted and DISCARDED

tt_backbone.py:103
    def __call__(self, h, causal=True):                  # no cache parameter at all
```

A path obliged to call those stubs has exactly one way to get the right answer: hand them the whole
prefix again, every frame.

**And the stubs lack that implementation because nothing ever asked for it.** No constraint blocks
caching; the specification simply never exercises it. The generated PCC test builds arguments by
name and drops anything optional it does not recognise (`tests/pcc/test_attention.py:461-463`):

```python
is_well_known = name in _WELL_KNOWN_INPUTS
if not is_required and not is_well_known:
    continue                      # `cache=None` is optional and unrecognised -> skipped
```

`cache` is not in `_WELL_KNOWN_INPUTS`, and the same file explicitly forces HF's standard cache
names to `None` as well:

```python
if arg_name in ("past_key_values", "cache_position", "use_cache", ...):
    return None
```

So the only specification the stub is written against **never passes a cache, ever**. A
cache-ignoring implementation and a cache-using one score identically — both PCC 0.9999 — so the
simpler one was written, passed, and graduated. Nothing downstream asks for more.

This makes the defect far more tractable than "the stubs are incapable" would suggest. Nothing needs
inventing: the reference already defines the cache contract (`layer(x, cis, bias, cache)`), the
capture already recorded real 208-deep cache contents, and the stub signature already carries the
parameter. What is missing is a test that passes it. Which is what `run_tts` does — and what the plan then **codified as the
specification**:

```json
"gate_2_invoked": { "how": "... tts_backbone == 1 + n_frames ..." }
```

The recomputation is not an oversight the builder made; it is the invocation count the plan
required.

**The trace gate demanded the opposite.** A host-free, static-shape, single-position step is only
possible with a resident cache — so `decode_step` had to **bypass the stubs** and inline the
`tc.*` helpers to build one.

**Root cause, and it traces back to F27.** The attention capture contained a real 208-deep KV cache;
the harness discarded it rather than `deepcopy` it, so the component was generated *and PCC-verified*
with no cache plumbing. Every downstream consequence follows from that single dropped argument: a
cacheless stub → a demo path that must recompute → a plan that codifies the recomputation → a trace
gate that cannot be met through the stubs → a second, unreachable implementation carrying all the
performance work.

**This also reframes the fix.** "Wire `decode_step` into `run_tts`" would break Gate 1 (the stubs
would no longer be the routed bodies) and Gate 2 (`tts_backbone` would drop to 1). The real fix is
further upstream: give the stubs the cache contract their captures already contained, so one
implementation can satisfy correctness, invocation and trace gates at once.

### Why the demo does not simply call it — the capture is position-locked

There are two reasons, and the second explains why this was not merely an oversight.

**No gate links the two paths.** Correctness runs `run_tts` and passed at 0.9999834, so that path's
cost never surfaced as a failure. Perf and the trace gate both run `trace_capture_selftest` →
`decode_step`, so that path's speed never had to reach the product. Nothing asks whether the
function measured for speed is the function measured for correctness.

**And the captured decode trace is only valid at one position.** `decode_step` takes `pos` as a
plain Python integer and uses it as a literal offset throughout:

```python
cos  = ttnn.slice(attn.tables.cos,  [0, 0, p, 0], [1, 1, p + 1, HEAD_DIM])
ttnn.update_cache(kc, ttnn.typecast(k, ttnn.bfloat16), p)
bias = ttnn.slice(attn.tables.bias, [0, 0, p, 0], [1, 1, p + 1, c])
```

Host-side integers are frozen into a recording at capture time. A trace captured at position 200
would, on **every** replay, write the new key/value into cache slot 200, read the rotary tables at
position 200, and apply position 200's causal mask row. Replayed across a 24-frame loop it would not
merely be slow — it would be **numerically wrong**.

*(Inferred from the code rather than tested: a replay at a different position was not attempted. The
mechanism — Python ints baked into a captured command stream — is standard, and the docstring below
describes exactly this constraint.)*

The author states the gap explicitly:

> *"`pos` is a fixed integer at capture time, which is what makes the traced shapes static **while a
> real generation loop still advances it**."*

Fixed at capture; must advance in a real loop. Reconciling those is the genuinely hard part of traced
incremental decode — the position has to live on the device as an index tensor so one recording is
valid at every step. **The builder stopped precisely where the gate stopped asking.** The gate asks
"can this stage be captured?" — yes, at one position. It never asks "is the capture replayable at
the next position?", which is the property that makes traced decode useful.

**`decode_step` itself is not broken — this matters for the fix.** Read as ordinary Python it is
correct at *any* position: `pos` is a parameter (`def decode_step(self, emb, pos=None)`), and a
caller advancing it 200, 201, 202… gets the right answer every time, with a real cache and no
recomputation. It is a proper incremental decode. The position-locking bites **only at trace-capture
time**, when the Python integer is frozen into the recording.

So there are two independent wins here, not one blocked one:

| | what it needs | difficulty |
|---|---|---|
| **1. call `decode_step` from `run_tts`** | wire it up, pass an advancing `pos` | **easy, available today** |
| **2. trace that loop** | move the position on-device as an index tensor so one recording is valid at every step | real work |

Win 1 removes the recompute-everything cost on its own and needs no tracing at all. Win 2 is where
the remaining distance to a hand-written port's numbers lives. The accurate summary is therefore:
**the demo runs a wasteful implementation while a correct, efficient one sits unused in the same
file — and the efficient one was additionally built in a form that can only be traced at one fixed
position.**

### Why nothing caught it

The trace gate (F45) asks whether each stage *can* be captured. `decode_trace_setup`/`decode_step`
answer yes, honestly, for a decode step that exists. Nothing asks whether the generation entry point
reaches that code. Coverage was checked by op signature, and both paths run backbone ops — so an
op-level check sees the same signatures and reports the model covered.

### AUDIT — is anything else in the port like F46?

F46 is severe enough that the rest of the generated port was swept for the same class of defect:
code that a gate or a measurement exercises but the product never runs. **The result bounds the
problem: decode is the only stage that diverges.** Recorded because a finding that overstates its
own blast radius is worth less than one that fences it.

**Method.** Four passes over `models/demos/voxtral_tts_full/`:

1. compare each `<stage>_trace_step` against what `run_tts` calls for that stage;
2. AST sweep for parameters accepted and never referenced in the body (the `cache` pattern);
3. reachability sweep for methods reachable only from trace/perf code;
4. check whether the depth caps differ between the traced and shipped builds.

**Result 1 — stage-by-stage, only decode differs:**

| stage | traced / profiled | shipped (`run_tts`) | same code? |
|---|---|---|---|
| prefill | `_run_backbone(x, depths["prefill"])` | `_run_backbone(x)` | **yes** — see result 4 |
| flow | `self.flow(h)` | `self.flow(h)` via `_run_flow` | **yes** |
| vocode | `_run_codec(codes)` | `_run_codec(codes)` | **yes** |
| **decode** | `decode_step` — incremental, resident KV | `_run_backbone(full padded seq)` | **NO** |

Decode is also the stage that runs 24× per utterance, so the one divergence is in the one place that
dominates the cost.

**Result 2 — accepted-and-ignored parameters.** Two matter, both supporting F46 rather than adding
to it:

```python
_stubs/attention.py:44
    def __call__(self, h, cis=None, bias=None, cache=None, cache_key=None):
        return self.attn(h, causal=bias is not None)      # cache, cache_key dropped

_stubs/decoder_layer.py:33
    def __call__(self, x, cis=None, bias=None, cache=None):
        return self.layer(x, causal=bias is not None)     # cache dropped
```

Both accept a KV cache and discard it, confirming that no cache can reach the shipped path even if a
caller supplied one. The ignored `cis` is the documented consequence of F37's third defect (the
native probe forbids marshalling side inputs, so stubs rebuild them in `build()`). The remaining
hits are benign: pytest's `device_params` indirect-fixture idiom, an abstract base raising
`NotImplementedError`, and the F37 repair conftest deliberately choosing staging dtype from the
tensor rather than the argument.

**Result 3 — no other dead paths.** Every `<stage>_trace_*` method is reached, via
`getattr(self, f"{stage}_trace_setup")` inside `trace_capture_selftest`. (A naive static grep reports
them as uncalled; that is a false positive and was checked.) Nothing else in `tt/pipeline.py` is
unreachable from either path.

**Result 4 — depths match.** `_depth(None, total)` returns `total`, and `demo_tts.py` builds with
`layers=args.layers`, defaulting to `None`. So `depths["prefill"] == 26` and
`_run_backbone(x, 26)` takes the `n == len(layers)` branch, which is exactly `self.backbone(x)` —
byte-identical behaviour to the shipped call. The capped path exists only for depth-limited
profiling builds.

**Conclusion.** The defect is real, confined to decode, and not symptomatic of a port that is
divergent throughout. Three of four stages are measured on precisely the code that ships. That makes
F46 a specific bug with a specific fix, rather than grounds for distrusting the whole artefact — and
it is the reason the correctness result (0.9999834 through `run_tts`) can still be believed.

---

### Fixes

1. **Profile what the entry point reaches.** Derive the profiled workload from the demo's own call
   graph, or assert that the perf test's hot function is reachable from `run_tts`. A single static
   reachability check would have caught this before the first measurement.
2. **Then decide which implementation is meant to ship.** `decode_step` is almost certainly the
   right one — it exists, it is trace-capable, and it is what the perf numbers already describe.
   Wire it into `run_tts` and the port becomes both faster and consistent with its own report.
3. **Until then, report the shipped path's number.** `run_tts` already returns
   `timings: {prefill_s, decode_s, codec_s}`; the honest headline is that measurement, not a
   trace-replay of an unreachable function.

---

## ★★★ HOW THE OPTIMIZE STAGE SHOULD MEASURE — consolidated, in dependency order

F40/F41/F43/F44/F45 are five symptoms of one root problem: **the stage never establishes what a
production step costs.** It profiles a shrunken, capture-and-verify workload in an execution mode
the product does not use, then ranks candidate edits by that number. The fixes below are ordered so
each one is worth doing even if the later ones are not.

### 1. Measure the mode you ship (F45)

Decide, and be consistent: either the generated pipeline replays traces, or it does not.

- If it stays eager, profile and time **eager**. Quoting trace-replay timings for an eager product
  makes the perf story unfalsifiable from outside.
- If it should be traced — and for this hardware it should — wire the existing
  `decode_trace_setup`/`decode_trace_step` hooks into `run_tts`: capture once, replay per frame.
  The apparatus is already built and proven by `trace_capture_selftest`; only the loop is missing.

Gate on **use**, not capability: fail if `execute_trace` is unreachable from the generation entry
point, or if a real run replays zero traces.

### 2. Profile a signposted WINDOW inside a real-sized run (F44)

The 12000-marker budget is real, but shrinking the workload is the wrong way to respect it.

The tool already resolves signposts (`agent/probes.py:1295`), scanning `tests/` for
`signpost("…")` string literals, and `perf_mcp._signposts_usable` verifies they interleave with ops
rather than clumping. For this model it found **none** and said so at Step 9/10, then continued with
full capture.

`perf_test_gen` should **emit** them:

```python
run_to_frame(12)            # real prompt (200 ids), real horizon (24 frames)
signpost("start")
one_decode_step()           # ONE step, at a realistic KV depth
signpost("stop")
```

One short window per stage — prefill at its real padded capacity, one mid-run decode step, one flow
step, one vocode call — keeps every capture inside the marker budget while profiling **production
shapes at production context depth**. Extra signposts between blocks give per-block attribution,
which is what `_signpost_blocks` exists to consume.

### 3. Rank targets by total contribution, not by per-step cost

Even a perfect per-op profile mis-ranks if each op is counted once. In one utterance a decode op
runs **24×** and a prefill op runs **1×**; the ladder picks `next_target` by largest `gap_ms`, so
with a one-step profile it aims at whatever is biggest in a single step.

Rank by `per_op_time × invocations_per_utterance`. The tool already collects the op-signature
sequence (`_op_sig_probe`), so the multiplicities are in hand — and fixing the truncation in F41 is
a prerequisite for counting them correctly.

### 4. Time accept/reject on the real workload, separately from profiling

These are two jobs with different constraints, and only one has a marker budget:

| job | profiler | workload |
|---|---|---|
| attribute time to ops | Tracy on | short signposted window |
| decide if an edit is faster | **Tracy off** | **the real 24-frame utterance** |

The accept/reject measurement already runs with Tracy off (`no tracy` in the log), so nothing forces
it to inherit the profiling workload. `run_tts` already returns
`timings: {prefill_s, decode_s, codec_s}` — the instrumentation exists.

### 5. Take capture and verification out of the timed region (F44)

`trace_capture_selftest` per stage runs an eager pass, a capture, a replay, two readbacks, a release
and a host-side PCC. Capture and release are startup costs; the eager pass and the PCC are
correctness machinery that the PCC gate already runs separately. Time `execute_trace` only.

### 6. Validate the profile before ranking on it

A dropped-marker profile is worse than none, because it looks complete. Two cheap checks:

- **Coverage**: sum of attributed per-op time should approximate the independently measured stage
  wall time. A large shortfall means markers were dropped — reject the profile rather than rank on
  it.
- **Sanity**: a PCC outside [-1, 1] is a parse failure (F42); an op count identical to the
  truncation limit is a saturated comparison (F41). Both are one-line assertions.

### 7. Never discard a measurement that succeeded (F40)

The baseline run produced `FORWARD_WALL_MS=756.4513` and then segfaulted in `close_mesh_device`
during teardown; the whole run was reported as failed and the number thrown away. Parse what the run
printed, and treat a post-measurement teardown crash as a warning rather than a failed measurement.

---

**Net effect.** With 1–4 in place the stage would minimise *time to speak one utterance at
production shapes in the shipped execution mode*, attribute it to ops weighted by how often they
actually run, and reject profiles it cannot trust. That is a different — and answerable — question
from the one it currently asks.

---

## ★★★★ F45 — the trace gate is satisfied by proving traces are POSSIBLE; the delivered pipeline never uses them

**Status: live** · severity: a model can pass "host-free / trace-capturable" and ship an eager
pipeline ~18× slower than a traced one, with perf numbers quoted from trace replay · reported: not
yet

`E2E_REQUIRE_TRACE=1` was set for this run, and the e2e gate passed it. What that gate verifies is
that each stage **can** be captured into a trace. It does not verify that the pipeline **does**
capture one.

`tt/pipeline.py` defines the full trace apparatus — `prefill_trace_setup/step`,
`decode_trace_setup/step`, `flow_…`, `vocode_…` — and `trace_capture_selftest` exercises all four.
But `execute_trace` appears **exactly once in the entire file**, at line 621, inside that selftest.

`run_tts`, documented as *"THE pipeline. Real prompt -> real 24 kHz waveform, through all seven
graduated stubs"* — the function the demo calls to make audio — contains no trace call at all. Its
decode loop is eager, frame by frame:

```python
for i in range(n_max):
    codes = self._run_flow(h)
    ...
    embeds = ttnn.concat([embeds, emb], dim=1)
    h = self._row(self._run_backbone(self._pin(embeds, length, cap)), length - 1)
```

Every frame re-dispatches the whole 26-layer backbone from the host.

### The cost, from the port's own README

```
Timings for the 8-frame gate: build ~30 s, prefill 2.2 s, decode 0.5 s/frame, codec 1.9 s
```

That README figure was an estimate for the 8-frame gate. **Measured directly** on the shipped path
at 24 frames, after the optimize run: **258.26 ms/frame** (297.69 before the 11 commits) — see THE
COMPARISON. A hand-written TTNN port of this same model on the same board reports **26.9 ms/frame**
with trace+1cq (RTF 0.357). Real time is 80 ms/frame, so the tool's port runs **3.2× slower than
real time** while the hand port runs 2.8× faster than it.

Trace replay is not a marginal optimisation on this hardware — removing per-op host dispatch is most
of the difference between those two numbers. The gap is not solely eager-vs-traced (implementation
quality differs too), but eager-vs-traced is the dominant term and it is a structural choice the
generated pipeline never made.

### Why this compounds F44

F44 established that the perf metric times `trace_capture_selftest`. Put together:

- the number the optimizer minimises is measured **on trace replay**,
- the pipeline that ships **does not use trace replay**,
- so every optimisation was ranked in an execution mode the product never enters.

An edit that helps traced replay and hurts eager dispatch would be banked as a win and would slow
the delivered model down. Nothing in the loop would notice.

### Why the gate let this through

The requirement is phrased as capturability, and capturability is what was demonstrated. A stage
that can be captured, executed and compared once is genuinely host-free *in that window* — the gate
is not lying. It simply never asks the question that matters: **does the generation path replay a
captured trace?** That question is answerable statically (`execute_trace` reachable from `run_tts`)
and dynamically (count trace replays during a real generation).

### Fixes

1. **Gate on use, not on capability.** Require that the generation entry point replay a trace —
   statically, by reachability from the demo's entry, or dynamically, by counting `execute_trace`
   calls during a real run and failing at zero.
2. **Wire the apparatus that already exists.** The pipeline has per-stage setup/step hooks and a
   proven capture for each; what is missing is a decode loop that captures once and replays per
   frame. The parts are built and unused.
3. **Measure the mode you ship.** Either report eager numbers for an eager pipeline, or make the
   pipeline traced and report traced numbers. Reporting trace-replay timings for an eager product is
   the specific error, and it is what makes the perf story unfalsifiable from the outside.
4. **State the execution mode in the report.** `RUN_REPORT.md` says nothing about whether the
   delivered pipeline is traced or eager.

---

## ★★ F47 — "host-free" is certified with the demo's per-frame host readback switched off

**Status: live** · severity: the host-freedom claim does not cover the configuration that ships ·
reported: not yet

The shipped generation loop performs a device→host readback **every frame**:

```python
def _is_stop(self, codes):
    """Host readback of ONE value -- generation control, not arithmetic (the graduated codec
    stub draws the same line: its [END_AUDIO] cut is 'host-side generation control')."""
    sem = int(ttnn.to_torch(ttnn.slice(codes, [0, 0], [1, 1])).flatten()[0])
    return sem in self.stop_ids
```

`run_tts` declares `early_stop=True` by default, and `demo/demo_tts.py:66` calls
`pipe.run_tts(inputs, max_frames=args.max_frames, verbose=True)` — so the demo takes that default and
syncs once per frame.

The check that certifies host-freedom does not:

```python
tt/pipeline.py:647   # `early_stop=False` because the [END_AUDIO] readback is generation control
tt/pipeline.py:653   self.run_tts(uploaded, max_frames=max_frames, early_stop=False, ...)
```

So `host_op_selftest` observes "a pure device chain" only because the one host op in the loop was
disabled for the observation.

**The reasoning is defensible and openly stated.** A stop condition genuinely is generation control
rather than model arithmetic, and every autoregressive implementation has to decide where that line
sits. The objection is narrower: the certification is reported without the qualifier, so a reader of
`RUN_REPORT.md` sees a host-free pipeline and the demo runs one that syncs 24 times per utterance.

**It also becomes a real cost exactly when the other fixes land.** At today's ~500 ms/frame a
single-value readback is noise. On a traced incremental decode at tens of ms/frame, a per-frame
device sync serialises the pipeline and is a material fraction of the frame. F46's fix makes F47
matter.

### Fixes

1. **State the configuration next to the claim** — "host-free with `early_stop=False`; the demo
   default performs one readback per frame".
2. **Or remove the readback from the loop**: compare the semantic code against the stop ids
   on-device and read the flag every N frames, accepting up to N-1 frames of overrun; or read the
   previous frame's flag while the current frame computes.
3. **Certify the shipped configuration.** `host_op_selftest` should run what the demo runs, and
   report the host ops it finds rather than removing them first.

---

## Sweep results — what was checked and found SOUND

Recorded because a report that only lists defects misrepresents the artefact, and because knowing
what was checked bounds what the findings claim.

**Every stub forward is genuinely host-op free.** An AST walk of all seven `__call__` bodies found
no `torch.*`, no `to_torch`, no `.item()`, no `.cpu()`:

```
attention CLEAN · codec_decoder CLEAN · decoder_layer CLEAN · flow_matching CLEAN
m_l_p CLEAN · r_m_s_norm CLEAN · tts_backbone CLEAN
```

The `torch` usage that exists in `flow_matching.py` (8 sites) and `codec_decoder.py` (3) is all
inside `build()` — staging constant tables, ALiBi rows and the pinned `x_0`. That is the documented
consequence of F37's third defect, and it is the correct place for it.

**F13's pattern is absent.** No `try/except` appears in any stub, so there is no swallowed fast-path
exception silently degrading to torch. `_runtime_fallbacks.json` is `{}`.

**Gate 1 is rigorous.** It byte-compares each live stub against its `.last_good_native` graduation
snapshot, asserts `native_probe.torch_ops == 0` **and** `ttnn_dispatch > 0`, requires
`_runtime_fallbacks.json == {}`, and then checks the objects actually wired into the chain are those
classes. Editing a graduated stub de-graduates it and fails the suite.

**Gate 2 is rigorous.** It uses *this run's* invocation deltas rather than lifetime totals — so
another test's calls cannot stand in — and asserts exact counts: `tts_backbone == 1 + n_frames`,
`decoder_layer == n_backbone`, `r_m_s_norm == 2 × n_backbone`, `flow_matching == n_frames`,
`codec_decoder == 1`.

**The golden is real.** `cached_reference_tts` runs the fp32 HF reference and caches it on disk under
a key covering prompt, voice, horizon and `x_0`, so a stale golden cannot silently apply to different
inputs. **And the correctness run uses the demo's own configuration** — `tt_run` calls
`run_tts(inputs, max_frames=horizon)` with `early_stop` at its default `True`.

**Only decode diverges.** Per the audit under F46: prefill, flow and vocode are measured on exactly
the code that ships.

### One consequence for whoever fixes F46

Gate 2 encodes the current architecture as correct:

```python
n_backbone = 1 + tt_run["n_frames"]          # prompt prefill + one per emitted frame
assert counts["tts_backbone"] == n_backbone
```

It **requires** the backbone to run once per frame — which is precisely the recomputation F46 says
to remove. Wiring `decode_step` into `run_tts` would drop `tts_backbone` to 1 and fail this
assertion. The gate is not wrong today; it simply describes the implementation it was written
against, and the fix must update these expected counts in the same change. Worth knowing before
someone starts, so a correct fix is not mistaken for a regression.

---

## S6 — OURS: our own `conftest.py` bootstrap shadows the built `ttnn` inside the planner's scratch copy

**Status: FIXED (uncommitted at time of writing)** · not a tool defect

`_bootstrap_ttnn_import_paths()` in the repo-root `conftest.py` is ours — added 2026-08-15 in
`bb71494984`, absent from `origin/main` and from PR #46283. It published `<this tree>/ttnn` onto
`sys.path` whenever `ttnn/ttnn/__init__.py` existed.

The hw-planner copies the checkout into a scratch root (`/tmp/tt_hw_planner__<model>_<ts>/`) and runs
pytest from there. That copy carries the `ttnn` **sources** but no compiled `_ttnn*.so`, so the
bootstrap put a source-only regular package ahead of the real one and every test in the copy died at
`ModuleNotFoundError: No module named 'ttnn._ttnn'`.

Fixed by selecting the `sys.path` entries from whichever tree actually holds `_ttnn*.so` (falling
back to `TT_METAL_HOME`) while keeping the calling tree first, so `models`/`tests` still resolve to
the copy under test.

**Recorded because it is easy to mistake for a tool defect** — the failure only appears inside the
planner's own scratch copy, and the tool is what creates that copy. The bug is ours.

---

## S7 — OURS: parking the Block-1 demo left its registry entry dangling

**Status: open at time of writing** · not a tool defect — but it is what exposed F30

`family_backends.py:289-299` registers `Voxtral TTS Backbone (mistral decoder)` with
`demo_path='models/demos/voxtral_tts_backbone/'` and
`canonical_hf_id='/localdev/lserbedzija/hf_models/voxtral-tts-backbone'`. We added that entry in
`3dfdc8b4a5` during the Block-1 experiment; we then deleted the directory it points at in
`9251fa6026` ("park the Block-1 demo out of the template pool") without removing the entry.

Parking the directory alone does not park the backend: selection still ranks and picks it (RUN_REPORT
15:11), and it is now a template with no template. Either drop the registry entry or restore the
directory — and never leave a local absolute path in a shared registry.

---

## Corrections to this document

- **"None of the optimisation work reaches a user" was WRONG.** An earlier draft of F46 said the 11
  commits improve only `decode_step`, so nothing reached the product. Measurement contradicts it:
  most commits land in `tt_common.py` and `tt_backbone.py`, which **both** paths execute, and the
  shipped path improved **297.69 → 258.26 ms/frame (−13.2%)** across those commits. The accurate
  claim is narrower — the optimizer *reported* −17.2% and the product *received* −13.2%, so the
  headline overstates by four points and the gains arrive as a side effect of touching shared code
  rather than because anything measured what ships. Corrected in five places on 2026-08-16.
- **A strict `[-1, 1]` PCC range check would break valid runs.** F42's fix was first written as
  `if not (-1.0 <= pcc <= 1.0): reject`. This port's own trace selftest legitimately prints
  `pcc=1.000017` — rounding can push a real correlation over unity — so the check needs a tolerance
  (1e-3). Observed bogus values run 1.525 to 47.779, all far outside it.
- **`test_main_perf.py` is NOT ours.** F31 originally called it "our own hand-written perf test". The
  optimize stage generates it (`auto-gen perf from pcc (agentic)`), so both profiler crashes are on
  the tool's own path.
- **"Every backbone matmul is a different shape" was too broad.** Decode emits one token padded to a
  32×32 tile, so decode *projections* are `M=32` in both the profiled and shipped runs — tuned at the
  right shape. Only decode *attention* (context 64 vs 224) and *prefill* (M 32 vs 224) differ.

- **`beat_baseline: false` on 24/24 kernel records is BY DESIGN, not a defect.** I flagged it as a
  possible reporting bug before reading `perf_mcp.py:4430`, which stores the agent's argument as
  `claimed_beat_baseline` and pins `beat_baseline` to `False` unconditionally; `_ledger().is_win`
  owns the verdict from measurements. The comments at 3941 and 4412 record that trusting the
  agent's flag previously double-counted wins. This is the same anti-self-certification design as
  F6's tool split, and it is correct.
- **The optimizer's state was tracking correctly.** I earlier read a `state.json` showing
  `iteration: 0` and reported a reporting gap. That file is not the optimize ledger; the live state
  is `/tmp/perf_mcp_*_voxtral_tts_backbone_main.json`, which tracked every commit accurately. No
  defect.
- **"Voxtral and XTTS are not on HuggingFace" is wrong** where it appears above. Both are on the
  Hub; they ship under `library=vllm` and `library=coqui` respectively, in their own native
  formats, not in `transformers` format. The tool's loader requires the latter — that is the actual
  constraint, and S1 is the consequence of it.

---

## Observations (not defects — recorded for the comparison write-up)

**O0 — the emitted e2e test is STRICTER than PCC, and it is right to be.** Credit, recorded on the
2026-08-15 re-run. `test_e2e_pcc` asserts **exact equality of every emitted audio code** in addition
to waveform PCC ≥ threshold:

```
frames: tt=(8, 37) ref=(8, 37) exact_match=True code_flips=0
e2e PCC=0.9999834299087524
```

This is the criterion §6.31 of the hand-port arrived at only after measuring it — one flipped
semantic code redirects an entire utterance, so waveform PCC alone is the wrong bar for a codec
model. The tool got there from the model's structure, unprompted.

It also has teeth. The optimize stage's own opening analysis reads:

> *"exact_code_match_beyond_pcc: `test_e2e_pcc` asserts exact equality of all emitted audio codes in
> addition to waveform PCC >= 0.99, and the accuracy notes show 67 code flips at plain-matmul
> precision, so any precision relaxation for speed fails outright."*

So the whole precision axis — the cheapest speed knob, and the one O1 notes the tool otherwise
applies model-wide — is closed off by its own correctness gate before the first measurement. That
is the exchange-rate O8 says the accept test lacks, supplied here by the test rather than by the
optimizer.

**But "closed off" turned out to be too strong, and the tool found the distinction that I did not.**
Measured over the run: every *weight-dtype* attempt failed the gate (`bfloat8_b` → PCC 0.349), while
a *math-fidelity* attempt **passed and won** −7.8 ms, on this reasoning:

> *"MathFidelity is how many passes the FPU makes over the operand mantissa, and the hi/lo split
> should make two passes sufficient here"*

HiFi2 is safe exactly because `stage_weight_split` already carries the precision in a hi/lo pair —
the mantissa bits the FPU would have gathered on later passes are in `lo`, not in the operand. So
the closed axis is *lossier weights*, not *fewer FPU passes*, and the two live on the same ladder
rung group. A blanket "no precision changes on a codec model" would have left that win on the table.

**O10 — `git_commit` sweeps whatever sits in the model dir, including untracked scratch.** Observed
on the 2026-08-16 run: commit `a7cdd41139` ("attention: split the fused QKV into heads with
nlp_create_qkv_heads") carried a stray `hifi3_verified_uncommitted.patch` alongside its two source
files, and the agent noticed and removed it in the next commit (`2ecac422b7`, "drop a stray scratch
patch that was swept into the previous commit").

No harm here — a `.patch` file is inert, so the measurement attributed to that commit came from the
two `.py` files as described. But `git_commit` stages the whole model-dir pathspec, so a scratch
`.py` dropped in the demo directory would be committed *and* would change behaviour, while the
commit message described something else. Worth a `git add` of only the files the attempt touched, or
at minimum a warning when the staged set exceeds them.

Credit alongside it: the agent spotted its own contamination unprompted and cleaned it up in a
separate, honestly-titled commit.

**O1 — whole-model dtype only.** `plan` recommends "N150 with bfp8_b weights", one dtype for the
whole model. The hand-port's §6.16 measured per-weight precision as the deciding factor: BFP8 on
FF and attention but **w2 in bf16**, because w2 alone is 77% of the accuracy cost for 15% of the
speed. There is no per-weight axis in the recommendation.

**O2 — a real static catch, worth crediting.** Compat flagged that this config uses HF's newer
`rope_parameters` field while `tt_transformers/tt/model_config.py:2736` only reads `rope_scaling`,
so the runtime would **silently** treat the model as having no scaling — safe at short context,
divergent at long. Found before anything ran, with two concrete fixes offered.

**O3 — confident tone on a fallback path.** `plan` prints `CONFIDENCE: HIGH` while also stating
"weights-only estimate (no transformer config); no KV math applied". The KV term is genuinely
omitted; here it does not change the verdict (218 MB at max_seq_len 2048 against ~24 GB headroom),
but the label and the caveat disagree.

---

## Process notes (mine, recorded so the numbers above can be judged)

Three mistakes cost real time and one of them destroyed evidence:

- **Trusted `pgrep` three times.** It matches any process whose command line contains the search
  text, including the lingering shell wrappers of finished jobs. It reported dead runs as alive
  and a live run as dead. `ps` is the reliable check.
- **Piped two long jobs through `tail`.** `tail` buffers until the process exits, so the build's
  CMake error was lost and the 9-hour overnight run wrote **nothing** to its log. Everything about
  that run had to be reconstructed from the tool's own state files and file timestamps. Use
  `python -u` and a direct redirect.
- **Reported progress from a log line instead of a verified state**, and called a run "past the
  wall" moments before it died at the same place.
- **Committed findings to the branch the tool was optimizing on.** The tool stamps the current HEAD
  into its own measurement records, so `full_pipeline_baseline` for the 15.976 result carries the
  sha of a *documentation* commit. Harmless here, and the agent noticed unprompted (*"the extra HEAD
  commit is the harness's own findings entry sitting on top of my perf commit"*), but the findings
  should have lived on a separate branch. `git log --grep '^perf('` isolates the tool's commits.

All PCC numbers in this file come from a direct `pytest` run I executed, not from the tool's
status files.
