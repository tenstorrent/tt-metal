# CosyVoice2 TTNN bring-up — status handoff

Written 2026-09-18, last updated 2026-09-22, for a future Claude Code session
picking this up (this one may not survive). **Start with "STATE AS OF
2026-09-22" right below the header block; where it conflicts with anything
further down, it wins.** Everything below is grounded in
real, verified repo/test state — verify it again yourself before trusting it,
same discipline this whole bring-up has used. Don't take this file's claims
about what "used to be here" on faith any more than you'd take a verbal
summary — `git log`, `git show`, and re-running the test suite are always the
ground truth; this file is a map, not the territory.

**IMPORTANT — user instruction, applies to this whole repo/session lineage:
do NOT run `git commit` or `git push` unless the user explicitly asks for
that specific action in that specific turn.** Writing/updating a file
(including this one) does not imply permission to commit it. This was a real
correction mid-session (2026-09-20) — see memory file
`git_commit_permission.md` if it still exists, but don't rely on memory
alone, this file is the durable copy of that rule too.

**Bounty**: tenstorrent/tt-metal issue #54104 ($2,000; the issue's label is
`bounty_difficulty/medium`, not Hard as this file used to say). CosyVoice2
(Alibaba FunAudioLLM) TTNN bring-up: LLM (Qwen2-0.5B) → flow-matching decoder
→ HiFT vocoder → waveform.

**Repo**: `~/tt-metal`, fork `github.com/Sedherthe/tt-metal`, branch
`bringup/cosyvoice2-istft`. Reference repo (read-only, architecture patterns
only, NOT directly transferable — CosyVoice2 differs from CosyVoice1 in real
ways): `~/reference-cosyvoice1` (`ayewo/tt-metal`, `bringup/cosy-voice-01`).
This reference repo has real, filed, upstream-relevant bug reports in its
`tt/hifigan/` code and `scripts/` dir — see "Two real hardware bugs found via
the reference repo" below, this matters a lot for where to look next.

**Hardware**: real N150, `device.arch() == ttnn.Arch.WORMHOLE_B0` — confirmed
freshly this session, matters for both hardware-bug sections below (both are
Wormhole-specific or Wormhole-relevant).

## STATE AS OF 2026-09-22 (read this first; supersedes conflicting text below, including the 2026-09-21 section)

Everything here was measured on the N150 in the 2026-09-22 session unless it says
"estimate" or "projected". Re-verify before trusting, as always. This round's scripts are
in `scripts/perf_2026_09_22/` (its own README explains each one and summarizes the
findings below).

### Git state

Committed this round (2026-09-22), NOT pushed (the user pushes): `tt/flow/decoder.py`,
`tt/flow/encoder.py`, `tt/flow/flow.py`, `tests/pcc/test_flow_decoder.py`,
`tests/pcc/test_upsample_conformer_encoder.py`, `scripts/perf_2026_09_22/`, this file.
Check `git log`/`git status` for whether it's been pushed by the time you read this.

### Traced CFM Euler-step solver — built, one real bug found and fixed

Ported the CosyVoice1 reference's `tt/flow/cfm.py` recipe into
`TtCausalConditionalCFM` (`tt/flow/decoder.py`): one Euler step (CFG concat, estimator,
CFG blend, update) captured as a single trace, replayed once per step from the host; `t`
(as its pre-time_mlp raw sinusoidal-embedding form, since `_sinusoidal_pos_emb` computes
on the host, unlike CosyVoice1's on-device sin/cos) and `dt` as device tensors, refreshed
per replay via `ttnn.copy`; warm up twice before capture; output allocated inside the
capture. Opt-in: `COSYVOICE2_FLOW_CFM_TRACE=1` / `use_trace=True` (off by default, same
convention as `TtQwen2LM`'s `use_decode_trace`). Single-slot cache keyed by
`(t_len, channels)`, same design as the CosyVoice1 reference — a NEW length always evicts
and recaptures, never held alongside another length.

**Confirmed, via explicit regression test
(`test_device_cfm_trace_cache_across_utterances_and_replays`), that the trace captures
exactly ONE Euler step and is replayed N times from the host — step count is never baked
into the capture.** Proven by reusing the SAME cached trace across DIFFERENT
`n_timesteps` values on the same instance.

**Real bug found and fixed while validating this**: a second solve on an
already-captured, cached trace (same geometry, either the same or a different
`n_timesteps`) replayed against corrupted buffer contents — PCC ~0.6 against the eager
reference, despite the first solve on the same trace scoring PCC 0.999+. Root cause:
the per-Euler-step device tensors (`t`/`dt`'s device-tensor form) were pre-built as a
full list upfront, before the reuse/capture dispatch — several small device tensors held
alive simultaneously while an already-captured trace was active. Matches ttnn's own
runtime warning verbatim ("Allocating device buffers is potentially unsafe due to the
existence of an active trace"). Ruled out first, empirically, with zero effect on the
corrupted PCC value: giving the conditioning tensors explicit `DRAM_MEMORY_CONFIG`, and
adding an explicit `ttnn.synchronize_device()` between the reuse-copy and the first
replay. Fixed by building each step's device tensor(s), using them, and deallocating
them immediately — one pair alive at a time, never a pre-built list (`_temb_device`/
`_dt_device` in `TtCausalConditionalCFM`). All four regression cases (same-step reuse,
cross-step-count reuse, cross-geometry recapture, back-to-first-geometry recapture) now
PCC 0.9996+. Full detail (including why two other plausible root causes were ruled out)
in `TtCausalConditionalCFM`'s and `_temb_device`'s docstrings, and in memory file
`cosyvoice2_trace_reuse_hazard.md`.

**Costs, measured cleanly** (`trace_lifecycle_experiment.py`, isolating tracing's own
overhead from unavoidable kernel-compile cost):
- A trace's cold-capture cost (tens of seconds) is **almost entirely kernel compile that
  eager pays too at a new shape** — NOT overhead tracing itself adds. Direct isolation:
  eager warm-kernel call 675.3 ms vs traced first-capture-with-warm-kernels 697.7 ms —
  tracing costs about one extra eager-equivalent pass once kernels are compiled, not the
  22-28 s the raw "cold" number suggests.
- Steady-state, same shape, no release (8 solves in a row, T=510): 496.9–504.1 ms per
  full 10-step solve, mean 498.8 ms (49.88 ms/Euler-step), stable to within 1.5%.
- Release: ~1.6 ms.

**Streaming trace-safety hazard, flagged in writing per instruction, NOT solved**: this
trace's safety today relies on running strictly after the LLM decode trace releases (same
reasoning as the LLM trace's own scoping note in `qwen2lm.py`). Streaming will interleave
LLM decode, flow encode, and CFM solve across chunks, breaking that non-overlap assumption
for both traces at once. Needs either per-stage trace-region partitioning or a documented
ordering constraint the streaming scheduler enforces — out of scope this round.

### Cached/traced flow encoder — does NOT work, real architectural blocker found

Built `TtUpsampleConformerEncoder`'s trace path the same way (opt-in
`COSYVOICE2_FLOW_ENCODER_TRACE=1`, off by default), sized for the real Stage 1
(whole-utterance) token length, not the 100-frame streaming number — measured that
separately too, clearly labeled `STREAMING PROBE`.

**Capture always fails**: `TtRelPositionMultiHeadedAttention._rel_shift` does a
deliberate host round-trip once per Conformer layer (see that method's own docstring;
this predates this round, not a regression) — all 10 layers this encoder calls hit it,
so `begin_trace_capture`/`body()`/`end_trace_capture` always raises `TT_FATAL: Reads are
not supported during trace capture`, at every geometry tried, not just some. The fallback
to eager is correct (every "traced" PCC test for this module actually exercises this
fallback, hence passing with PCC 1.0 vs eager — not a false positive, just not evidence
tracing works) and now remembers the failure (`_trace_unavailable`) so it stops
re-attempting a proven-doomed capture (measured cost of NOT doing this: ~2.5x plain eager
on every call, from two wasted warm-up passes plus an aborted capture attempt, repeated
every time). Making this module genuinely traceable needs `_rel_shift` ported to run
natively on device — real, separate work, not attempted this round.

### WER scoring bug found and fixed

`jiwer.ExpandCommonEnglishContractions()` (the standard fix for a "we're" vs "we are"
mismatch) also expands ANY bare `'s` to `" is"`, unconditionally — this incorrectly hits
possessives too (measured: `"Layton's"` → `"Layton is"`). Fixed with a precise,
contraction-only transform (`jiwer.SubstituteRegexes`, the same rule list as
`ExpandCommonEnglishContractions` minus that one ambiguous rule) in
`scripts/perf_2026_09_22/rescore_wer_final.py`. This is a scoring-methodology fix only —
no production code changed, since WER scoring only ever happens in eval/perf scripts, not
in the `tt/` package itself.

Re-scored with the precise fix: this round's four sentences — only utt1 changed (9.52% →
0.00%, a genuine "We're"/"We are" contraction, confirmed real by comparing two independent
HiFT noise draws of the identical mel: tokens and mel bit-identical across two independent
LLM+flow runs, both HiFT draws transcribe identically despite audibly different waveforms
— see `wer_repro_and_stage_breakdown.py`); utt0/utt2/utt3 unchanged. **The original Stage 1
eval's reported 4.17% WER (`REF_IDX=0, TGT_IDX=3`, "Leighton's" vs "Layton's") was
reproduced faithfully and re-scored with the precise fix: 4.17% → 4.17%, unchanged.**
Confirms that miss is a genuine ASR name-spelling error, not a scoring artifact — **Stage
1 WER is confirmed 4.17% (PASS, target <5.0%).**

### Regression check, steps=10 (still the validated count), real measured

Full 4-sentence warm regression, CFM traced + encoder eager (encoder trace doesn't help,
see above), corrected WER scoring:

| Metric | 2026-09-21 baseline | 2026-09-22 (measured) |
|---|---|---|
| Warm RTF | 0.58–1.01 | **0.428–0.523** |
| WER (corrected scoring) | 4.17% (single utterance) | 0.00% / 0.00% / 0.00% / 2.78% (four sentences) |
| Speaker similarity | 0.8885 | 0.56 (utt0, matches this file's own documented inherent limit for that exact sentence) / 0.84 / 0.79 / 0.87 |

RTF genuinely improved (real, measured, not composed) — the traced CFM solver is the
reason; the encoder trace contributes nothing (see above). Fresh per-stage breakdown
(warm, CFM traced): **LLM decode now dominates, 52–60% of total time**; CFM 21–32%;
encoder 7–11%; HiFT 8–9% — the bottleneck moved since the 2026-09-21 estimate (which
predated any of this round's tracing work).

### Step count: DECIDED — hold at 10

Measured real RTF at steps=5, via a real isolated CFM-only measurement (not the old
pre-trace estimate) at the four real T values this round's sentences produce, then
projected the full pipeline using that real measured delta against the real steps=10
per-stage baseline:

| Utt | Audio | RTF@10 (real) | RTF@5 (projected, real per-stage data) | RTF reduction |
|---|---|---|---|---|
| 0 | 4.36s | 0.523 | 0.466 | 10.9% |
| 1 | 6.32s | 0.462 | 0.418 | 9.5% |
| 2 | 7.36s | 0.446 | 0.403 | 9.7% |
| 3 | 11.36s | 0.428 | 0.393 | 8.2% |

**Decision (user, 2026-09-22): hold at steps=10.** An 8–11% RTF gain isn't worth an
unvalidated quality risk given current priorities. **If step reduction is revisited
later, start with 8 or 6 steps** (much smaller mel error per this file's existing
"Euler steps vs 10" table: 8 → 1.0-1.9%, 6 → 3.1-4.5%, vs 5 → 3.0-6.9%, torch-measured,
unchanged by anything this round did since the traced path is PCC-proven numerically
identical to eager at any step count) **and run full WER/speaker-similarity/listening on
real sentences before considering 5.** Do not re-litigate this decision from scratch —
this is a settled call, not an open question, unless something material changes.

### Next steps (2026-09-22 order; updates progress as it happens)

Items 4 and 5 from the 2026-09-21 list (LLM decode profiling, traced Euler-step solver)
are **done** — see above. Item 3's second half (op-level traced profile of the estimator,
SineGen2 cumsum-vs-mod-1 check) is still **not started**. Items 6-8 (torch-LLM silence
check, step-count accuracy at 8/6, `release_caches()`) are **not started** — item 7 is now
scoped to steps 8/6 specifically per the decision above, not 5. The progress note has been
updated with this round's real numbers (steps=10, RTF 0.428-0.523, corrected WER) —
**still not posted**, the user reviews first.



### Git state (check `git status -sb` / `git log` — this drifts)

- Pushed to origin (by the user; verified with `git fetch` on 2026-09-21): `1cefdec6fc`
  (TtStft fix + conv resolver + tests), `090bf1cbb2` (diagnostic scripts + README),
  `a930a13dcb` (opt-in traced LLM decode, `use_decode_trace`), `cf20ed17ad` (DRAM conv
  config tensors, fused QKV/SDPA flow estimator, bf16 flow defaults).
- Committed locally, **not pushed**: `f1e28c1266` (`scripts/perf_2026_09_21/`, this round's
  measurement scripts + README), `cbbf3ff708` (conv-resolver port: `TtCausalConv1d`/
  `TtPaddedConv1d` now subclass `hifigan.conv.TtConv1d`, which gained asymmetric
  `(pad_left, pad_right)` padding support; full suite re-run after: 136 passed).
- **Uncommitted**: this file only. Nothing gets committed or pushed unless the user asks
  in that turn.

### Flags (all read at construction time) and their defaults now

| Flag | Default | Meaning |
|---|---|---|
| `COSYVOICE2_CONV_CONFIG_IN_DRAM` | on | `config_tensors_in_dram=True` on every conv, so L1_SMALL stays flat across lengths (was: 64 KB bank exhausted by the 2nd different length). Bit-identical outputs. |
| `COSYVOICE2_FLOW_FUSED_QKV` | on | one fused Q/K/V matmul in the flow estimator; bit-identical |
| `COSYVOICE2_FLOW_SDPA` | on (flipped 2026-09-21 after the user listened and could not tell arms apart) | fused `scaled_dot_product_attention` (bf16; q128/k256 chunks). Assumes an all-ones mask; the CFM host guard raises on a partial mask (use `=0` for masked inputs) |
| `COSYVOICE2_FLOW_MATMUL_CC=accurate` | off (opt-in) | HiFi4 + fp32 accumulation for flow linears/matmuls |
| flow dtype | bf16 (the `TtCausalMaskedDiffWithXvec` default; only my scripts passed fp32) | target dtype per the user: bf16 |

Test suite: **136 passed in 320 s on 2026-09-21** with these defaults and no env
vars (was 118 on 2026-09-18; the additions are the STFT, conv-verification,
iSTFT-length and traced-LLM tests).

### Measured performance (RTF = whole-request wall time / audio seconds, device-synchronized)

Warm = identical request repeated in the same process, LLM decode traced, flow
(bf16, fused QKV + SDPA) and HiFT eager; three repeats each:

| Audio | Warm RTF | First request at that length (fresh kernels) |
|---|---|---|
| 4.36 s | 0.97 to 1.01 | 50.2 |
| 6.32 s | 0.76 to 0.77 | 39.8 |
| 7.36 s | 0.71 | 35.5 |
| 11.36 s | 0.58 | 28.0 |

- The flow estimator is **host-bound when eager**: about 270 ms per call, 2.7 s per
  solve at every length. Traced device time per call (bf16, fused, measured at
  T = 342/392/492/592/660/792): 36.7/40.5/47.7/56.8/62.4/74.0 ms, about
  7.8 + 0.083*T ms, valid only inside that T range. Replay PCC vs eager 1.0.
  Baseline before the estimator work was 348 ms per traced call at T = 660.
- LLM: 41 -> 9.8 ms/token traced (bit-exact), about 0.1 s fixed per `generate`.
  **Trace-lifetime trap**: a trace kept alive across the flow/vocoder hung the card;
  the trace is scoped to one `generate()`. Recovery: kill -9, `tt-smi -r`.
- New-length cost: first request at a new length is RTF 28 to 50 when kernels are
  not on the disk cache (first HiFT call about 190 s), about 2 to 3 when they are.
- **Unfixed leak**: DRAM grows per new utterance length (prepared conv weights are
  per-geometry and not owned by the program cache): 90.9 MB/bank after the first
  length, 134.1 after the fourth in the 2026-09-21 warm run (capacity is about 1 GB per bank).
  `device.clear_program_cache()` fixes L1_SMALL but not this.
- **Stage 1 RTF < 1.0 (non-streaming)**: met warm for every length above; the
  shortest (4.36 s) is about 1.0. The issue does not define warm vs cold or the
  utterance set (asked nowhere yet; a draft progress note that asks the maintainers is at
  `/home/user/cosyvoice2_stft_fix_wavs/issue54104_progress_note_DRAFT.md`, NOT posted;
  `gh` is unauthenticated, the user posts).

### Accuracy and listening (2026-09-21)

- Listening arms A (defaults), B (+SDPA), C (+SDPA+bf16), D (+SDPA+HiFi4/fp32acc),
  R (torch flow mel through TT HiFT), seven everyday sentences, shared tokens and
  vocoder noise. Mel relative L2 vs pure torch: A 1.02%, B 1.00%, C 1.03%, D 0.55%.
  WER identical in every arm and in R (1.76% mean). The user could not tell them apart.
  Wavs: `/home/user/cosyvoice2_stft_fix_wavs/listening_arms/`.
- Speaker similarity, seven sentences: 0.55 to 0.89, mean about 0.78; the 4.36 s
  sentence ("Please close the door...") scores 0.554 in every arm. **Inherent, not
  ours**: fully torch flow + torch HiFT on the same tokens gives 0.56; that clip has
  only 1.84 s of speech in 4.36 s (about 1.9 s trailing silence); real same-speaker
  clips under 3.5 s have median 0.51 (9 of 11 below 0.60). Untested: whether a torch
  LLM would also emit the trailing silence.
- **dtype correction**: the "fp32" flow is NOT fp32 weights + bf16 activations.
  Estimator activations are mostly fp32 after the first conv; ALL conv weights are
  bf16 in every run (each conv class defaults `weights_dtype=bf16`, no caller
  overrides); LLM has some bfloat8_b weights. This resolves the earlier "conv weights
  stored bf16 even in fp32 run" open lead: it is true, and not audible.
- Euler steps vs 10 (torch reference, mel rel L2): 8: 1.0-1.9%, 6: 3.1-4.5%,
  5: 3.0-6.9%, 4: 6-13%, 3: 14-19%. WER/listening at fewer steps NOT evaluated.

### Streaming (bounty Stages 2 and 3): entirely unbuilt

Gaps: no flow streaming/finalize/chunk-causal masks (official CosyVoice2 uses
chunk-causal masks, the whole growing token prefix + 3-token lookahead per chunk,
hop 25 tokens, first chunk 25 + prompt padding); torch reference has no streaming
mode; HiFT has no `cache_source`/mel cache (8 frames)/Hamming crossfade (3,840
samples); `generate()` returns the list only at the end. The design is written up
(Part 6 of the Claude Doc below).

Latency budget (mostly **estimates** from the measured fit; see Part 8 / Part A.3):
official-shape time to first packet about 0.94 s, chunk RTF 0.8 to 1.1, so neither
TTFP < 500 ms nor streaming RTF < 0.4 is reachable with the current design. LLM +
HiFT alone are about 0.28 s per audio second, leaving about 0.12 s per audio
second for encoder + estimator. Needs: traced solver, cached/traced encoder, and
fewer Euler steps or far fewer prompt frames (both change the output, unvalidated).

### CosyVoice1 reference comparison (read, not run; Claude Doc below)

Their RTF is a composed warm figure for one fixed 3.27 s utterance (LLM step time x
token count + one warm flow call with the cached trace + one warm vocoder call), no
cold end-to-end figure. n300 (one Wormhole chip, same silicon as N150): RTF 0.553;
Blackhole 0.34 to 0.40; streaming first audio 1.31 to 1.49 s (Blackhole; their
Wormhole latency test hangs). Re-costed their way, our estimate with a traced solver
is about 0.46 at 3.3 s and about 0.39 at 11.4 s (estimate; today 0.58 to 1.0).

### Where the write-ups live

- Claude Doc "CosyVoice2 on TTNN: debugging and performance log" (id
  `52353727-45d2-4b7e-a0a9-28c51e6dec3a`, Parts 1 to 8 plus glossary): every experiment in order.
- Claude Doc "CosyVoice2 on N150: analysis and how CosyVoice1 does it" (id
  `438ba0ec-110e-46d4-80f6-c83ada4a5003`): this round's analysis + CosyVoice1 comparison.
- Wavs, arm results, progress-note draft: `/home/user/cosyvoice2_stft_fix_wavs/`.
- Measurement scripts for this round: `scripts/perf_2026_09_21/` (README explains each;
  uncommitted; paths made portable via `COSYVOICE2_SCRATCH` and `OUT_DIR`).

### Next steps (2026-09-22 order; updates progress as it happens)

1. **Done.** Progress-note draft updated with this round's warm numbers, warm defined
   precisely, first-request column dropped, asks for the RTF/TTFP definitions (not to
   relax targets). Still not posted (the user wants the re-costed "best case" number
   first). `/home/user/cosyvoice2_stft_fix_wavs/issue54104_progress_note_DRAFT.md`.
2. **Done.** Masked/chunk-causal SDPA benchmark, T 156-792: masking costs real time
   (mask *presence*, not content, per T=156-792 op-level bench), re-costed A.3's chunk
   RTF with it; also checked for a causal fast path (`is_causal=True` exists, is even
   faster than no-mask, but is strict token-causal, not block-causal -- not a drop-in);
   took a real (non-extrapolated) T=156 measurement. Best real case so far: cached
   encoder + 5 Euler steps + masking, chunk RTF 0.45. See Claude Doc 438ba0ec, A.5/A.6.
3. **Done (this file's "Uncommitted" line above).** Conv-resolver port for the
   estimator and encoder convs. Next half of this item, not started: an op-level traced
   profile of the estimator, and checking the SineGen2 phase accumulation against a
   mod-1 reduction (CosyVoice1 found `ttnn.cumsum` inaccurate on long scans).
4. Profile LLM decode: traced replay device time vs host readback and sampling: where
   does the 9.8 ms/token go. Not started.
5. Traced Euler step for one fixed shape (recipe in the CosyVoice1 section of the
   second Claude Doc), plus a short design note on how streaming chunks map to a small
   set of T keys (bucketed padding + masks vs a fixed window), stating what changes the
   output. Not started.
6. Run the torch LLM on CPU for the 4.36 s sentence with the same sampler; compare token
   count and trailing silence (the 0.554 speaker-similarity investigation's last open
   lead). Not started.
7. WER / speaker similarity / listening at 5 and 6 Euler steps. Not started.
8. `release_caches()` for the DRAM leak only if the fixed-shape design still needs it.
   Not started.

Confirming the RTF/TTFP definition with the maintainers is folded into item 1 above
(the progress note asks it) rather than a separate step.

### Session habits that paid off

Run device jobs as `timeout -s KILL N /opt/venv/bin/python ...` with
`PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal`
from a directory other than the repo root. Time with the trace-allocation tracker
env vars OFF (they inflate replay time; use them only in separate safety passes).
Do not `pkill -f` a pattern that can match your own shell.

## What's built and verified (all real, all tested; as of 2026-09-18 committed+pushed, see the git state above for later work)

Everything below has a real PCC test in `models/demos/audio/cosyvoice2/tests/pcc/`
and passes on real N150 hardware. Full suite as of 2026-09-18:
**118 passed, 0 failed** (`pytest models/demos/audio/cosyvoice2/tests/pcc/ -q`).
**Update 2026-09-21: the suite is now 136 tests and 136 passed** with the new
defaults; re-run it yourself first, don't trust either number once time has passed.

- iSTFT-as-matmul, HiFT vocoder (upsample/resblock/3-source-branch stack),
  `ConvRNNF0Predictor` (5 plain `Conv1d(k=3)+ELU` layers + `Linear(512,1)` +
  `abs()`, no recurrent layer despite the name), SineGen2/NSF excitation,
  Qwen2-0.5B LLM backbone with real sequence assembly/prefill/decode/RAS
  sampling, causal Conformer flow encoder, CFM flow-matching estimator (with
  real classifier-free guidance, `inference_cfg_rate=0.7`), outer
  `CausalMaskedDiffWithXvec`/`TtHiFTGenerator` wiring. Full pipeline (speech
  tokens → flow decoder → mel → real F0 predictor → HiFT vocoder → waveform)
  wired end-to-end, zero synthetic stand-ins anywhere.
- **Real CosyVoice2-0.5B checkpoint weights load into all four modules**
  (`FunAudioLLM/CosyVoice2-0.5B` on HF — `llm.pt`/`flow.pt`/`hift.pt`), each
  verified against its own real-checkpoint PCC test. See `tt/checkpoint.py`
  for shared download/remap helpers (`load_checkpoint_file`, `sub_state_dict`,
  `build_local_qwen2_checkpoint_dir`).
- **HiFT needs fp32, not bf16, with real weights** — bf16 PCC collapses to
  ~0.49 (real weights push `conv_post`'s pre-`exp()` values into a range bf16
  can't track precisely; random init never did this).
- **`TtQwen2LM.generate()`'s stop-token/min_tokens handling is correct**,
  matching real upstream `Qwen2LM.inference_wrapper`/`sampling_ids` exactly:
  `stop_token_ids` checks all three real stop IDs, the break is unconditional,
  `min_tokens` only masks `eos_token`'s logit pre-sampling. Fixed in commit
  `c44cb8131a`, regression-tested.

Real Stage 1 targets (from the bounty issue itself — re-fetch via `gh api
repos/tenstorrent/tt-metal/issues/54104` if unsure, don't trust from memory),
measured with real infra (real LibriSpeech test-clean sample, real
`campplus.onnx`/`speech_tokenizer_v2.onnx`, real Whisper ASR):

| Target | Goal | Measured | Result |
|---|---|---|---|
| Token-level accuracy | >95% | 100% | PASS |
| WER | <5.0% | 4.17% | PASS |
| Speaker similarity | >0.60 cosine | 0.8885 (re-measured 2026-09-20 after the `TtStft` fix; was 0.857 with the corrupt STFT — mel was bit-identical, so the gain is the vocoder fix) | PASS |
| RTF (non-streaming) | <1.0 | **Superseded 2026-09-21: warm 0.58 to 1.01 (0.97-1.01 at 4.36 s, 0.58 at 11.36 s); first request at a new length 28 to 50 (cold kernels).** Historical: 14.3× on the 2026-09-20 post-fix re-run, cold caches (132.8 s for 9.28 s of audio: LLM 23.0 s, flow decoder 102.1 s, HiFT 7.7 s) | Met warm except the shortest clip (about 1.0); cold/new-length not met. The issue does not define which counts. See the state section at the top. |

(Token accuracy 100% and WER 4.17% re-measured too, unchanged. The one WER miss is a name spelling, "Leighton's" vs "Layton's".)

## SUPERSEDES THE SECTION BELOW (2026-09-20, later session): the dominant cause was `TtStft`, not F0 drift

**`TtStft` (`tt/hifigan/stft.py`) silently returned garbage for any input >= 65,536
samples** (~2.7 s at 24 kHz; correct at 64,000, wrong at 65,536+; PCC ~0.27, output
~30x too small). It framed the signal with a conv whose weight was hoisted via
`ttnn.prepare_conv_weights` -- the #55545 defect class, at a site with no output
check. The same conv given the raw weight was correct at every length tried (to 1.9M
samples). It feeds the source branch (`source_downs` -> `source_resblocks`) at every
upsample stage of every real synthesis; no test went past 9,600 samples. Found by a
stage-by-stage decode bisect (float64 torch reference, same mel and same excitation on
both sides, every sub-module tapped): every module was locally accurate, and the
accumulated error entered through the STFT.

Measured effect of fixing it (real 464-frame utterance, fp32, same noise draw):
TT-generator-with-torch-F0 sample PCC 0.10 -> 0.999, log-mel distance to the torch
reference 1.54 -> 0.63 dB, overall level offset -1.9 -> -0.55 dB.

**Fixed in the working tree (see `git status`/`git log` for whether it has been
committed):** `TtStft` now always passes the raw weight (`tests/pcc/test_stft.py` pins
4,800 / 64,000 / 65,536 / 222,720 samples); `TtConv1d`/`TtConvTranspose1d`'s
verify-and-resolve now judges agreement by relative L2 (5%) and, on a disagreement,
arbitrates with a float64 host conv over {prepared+accurate, raw+safe, raw+accurate}
instead of assuming the raw+safe reference is right (`tests/pcc/test_conv1d_verification.py`,
including injected-corruption tests); `test_istft.py` now also covers 55,681 and 65,537
frames (`TtIStft` was checked and is fine through 240,000 frames).

**Corrections to the F0 section below:** (1) its F0 numbers (PCC 0.9999744, max 4.9 Hz)
are fp32, the production dtype -- not bf16. (2) The claim that F0 drift is "the" cause
was wrong: the STFT bug dominated and masked it. After the STFT fix, F0 error is a real
but secondary effect: with the device F0 the waveform PCC vs torch stays ~0.11 (phase
drift) and log-mel distance is ~1.0 dB, vs 0.999 / 0.63 dB with torch F0 injected.
Measured F0 error in voiced frames: mean 3.3 cents, p99 28 cents, max 116 cents; one
unvoiced->voiced flip (frame 145). (3) The `Conv1d(128->128, k=11) length 18560`
warning was NOT a weight-prep disagreement (prepared == raw exactly); it was the
compute-config axis, and the old resolver picked the *less* accurate result ("safe"
config: 2.7% error at 128ch/L=18560, 10% at 64ch/L=55681, vs 0.4-0.6% for accurate).

**Still open:** the ~0.63 dB residual with perfect F0, and the F0 predictor's own
precision. Untested leads: conv weights are stored bf16 even in an fp32 run
(`TtConv1d.from_module(..., dtype=dtype)` leaves `weights_dtype` at its bf16 default),
and the F0 classifier `ttnn.linear` takes no compute config.

**USER CONFIRMED BY EAR (2026-09-20): the audio quality complaint is FIXED.** With the
`TtStft` fix in place, the TT vocoder output (with its own F0) sounds the same as the
official vocoder's -- the robotic resonance / hiss / metallic character is gone. The
residual ~1 dB log-mel distance and the F0 phase drift are NOT audible, so the open
leads above are optional polish, not blockers. (Do not confuse with the pre-fix
render, which still has the artifacts.)

## (SUPERSEDED — see above) REAL ROOT CAUSE (2026-09-20): F0 predictor precision drift, amplified by phase integration

**User confirmed: the excitation-noise fix below did NOT fix the audio — it
sounds exactly the same.** Verified numerically why: the fix is real but its
effect is tiny (waveform 83% of samples differ, but correlation 0.9997, max
deviation ~3% of full scale) — far too small to be audible, and definitely
not the dominant cause. Keep it (it's correct and harmless), but the search
continued and found something much bigger.

**Confirmed via a controlled 3-way experiment** (not just eyeballing a PCC
number): built the actual TT device `TtHiFTGenerator` and compared it against
our (already proven fully correct, PCC 0.9999999+) torch reference, at the
REAL production length (464 mel frames / 222720 samples) for the first time
ever — every existing PCC test only checks short synthetic lengths (8, 20,
250 at most). Full pipeline PCC at this real length: **0.09** (near-total
decorrelation). Bisecting stage by stage found the F0 predictor's device
output is very close to torch (PCC 0.9999744, max diff 4.9 Hz out of ~194 Hz
range) — looks harmless. But:

- Feed the SAME f0 curve (torch's numbers, or device's numbers, doesn't
  matter which) into BOTH the torch and device excitation modules
  (`SineGen2`/`SourceModuleHnNSF`): PCC ~0.9999 either way. The excitation
  module itself is correct on both sides.
- Feed EACH side its OWN (very slightly different) f0 curve, exactly what the
  real pipeline actually does: PCC collapses to **0.19**.

Mechanism: `SineGen2` integrates frequency into phase via a running
`cumsum`. A tiny, persistent frequency deviation, integrated over 222,720
samples, accumulates into large absolute phase drift — two clocks with
slightly different rates ending up completely out of sync, even though each
tick was almost identical. Checked whether this is the SAME
`accurate_compute_config`/`safe_compute_config` defect already fixed
elsewhere in this file — it is NOT; no disagreement warning fires for the F0
predictor's convs at this geometry/length. This is ordinary TT-hardware-vs-
CPU floating-point non-determinism (5 stacked conv+ELU layers,
`TtConvRNNF0Predictor` in `tt/hifigan/f0_predictor.py`), not a discrete bug —
made catastrophic purely by the oscillator's zero tolerance for accumulated
input drift.

**This is a harder, different class of problem than everything fixed so far
in this file** — a real numerical-sensitivity property of the
integrate-frequency-to-phase architecture meeting ordinary cross-hardware
float differences, not a patchable discrete bug. Upstream never faces this
at all (single-hardware reference, nothing to diverge from).

**NOT YET IMPLEMENTED — awaiting user direction**, presented as of this
write-up:
1. Try to shave down the F0 predictor's absolute device-vs-torch divergence
   further (check for any remaining precision lever — e.g. verify
   `ttnn.elu`/conv aren't silently using an approximate mode — even though
   already at the highest fidelity setting available). Likely reduces but
   may not eliminate the drift.
2. Make the phase integration itself more robust to small input drift (e.g.
   periodically re-anchor/dampen phase instead of letting error accumulate
   unboundedly over the whole utterance). Diverges from upstream's exact
   math, needs careful validation it doesn't add its own artifacts.
3. Document as a known hardware-precision limitation and move on.

**Reproducing this finding** (scripts in `/tmp` scratch as of this write-up,
NOT yet copied into `scripts/vocoder_debug_2026_09_20/` — do that first if
this session survives, or rebuild from this description): `tt_vs_torch_real_length.py`
(the full-pipeline PCC-0.09 discovery, real mel + shared noise, TT vs torch
at mel_frames=464), `tt_vs_torch_stage_bisect.py` (the stage-by-stage
localization to the excitation stage), then the follow-up one-off diagnostic
(not saved as a file — rerun inline) that ran the 3-way same-f0/cross-f0
experiment isolating the F0-predictor-precision-plus-phase-integration
mechanism specifically, using `TtConvRNNF0Predictor` vs
`TorchConvRNNF0PredictorRef` directly on `stage1_v4_mel.npy`.

## EARLIER FIX (2026-09-20, kept but insufficient on its own): zero excitation noise in real synthesis

**Confirmed real and correct, but confirmed by the user NOT to have fixed
the audible complaint on its own** — see "REAL ROOT CAUSE" above, written
after this section, for what's actually driving the reported audio quality
issue and why this fix's effect turned out to be numerically real but too
small to hear.

Bisected our own torch reference (`TorchHiFTGeneratorInferenceRef`, proven
downstream of a correct flow-decoder mel by the vocoder isolation test below)
against the real official `HiFTGenerator`+`ConvRNNF0Predictor` classes,
stage by stage, on the identical real mel. First pass showed the excitation
signal `s` diverging (PCC 0.95) — but this turned out to be a **diagnostic-
script artifact**: the comparison call hadn't passed the real module's random
noise draw, defaulting ours to zero while the real module drew real noise
internally. Capturing the real module's actual `torch.randn_like` draws (via
a monkeypatch) and feeding them into our own reference gave **PCC
0.9999999+ at every single stage** (F0 curve, excitation, every upsample
stage, iSTFT, final waveform) — our torch reference is fully correct.
Lesson, worth remembering for any future "found a divergence" moment: always
check whether an omitted random draw explains it before trusting the number.

That redirected the search: since the actual noisy audio came from the **TT
device path**, and existing PCC tests only validate at short synthetic
lengths, checking there found a real, structural bug already half-documented
in the code: `TtSineGen2.__call__`'s own docstring
(`tt/hifigan/source.py`) says its `noise` argument is "Deterministic zero if
omitted... wrong for synthesis" — and `TtHiFTGenerator.inference()`
(`tt/hifigan/generator.py`) called `self.source(f0_audio)` with **no noise
argument at all**, on every real synthesis call. Real upstream's
`SineGen2.forward` draws real random noise here unconditionally, every real
call (no eval-mode bypass) — this is a genuine part of the NSF/HiFTNet
excitation model, not training-only regularization. Zero noise collapses
`sine_waves * uv + n` to an exact zero in every unvoiced frame (`uv=0`)
instead of the natural noise floor the model was trained on — a real,
structural, always-on defect in every real synthesis call this port has ever
made, fully consistent with "noisy, robotic" (dead unvoiced segments,
unnaturally clean voiced segments, both out-of-distribution for the trained
resblock stack).

(The *other* random draw upstream computes, `SourceModuleHnNSF.forward`'s
"branch noise", is genuinely dead code in real upstream — its `noise`/`uv`
outputs are discarded, `s, _, _ = self.m_source(s)` — correctly NOT
reproduced here, confirmed by reading the real source directly, not assumed.)

**Fixed**: `TtHiFTGenerator.inference()` now draws real `torch.randn` noise
per call by default (shape `[batch_size, mel_frames * upsample_scale,
harmonic_num+1]`), with an explicit `sine_noise=` override for reproducible
tests. `TorchHiFTGeneratorInferenceRef.inference()` and
`tests/pcc/test_hift_generator_inference.py`'s device-vs-reference test were
updated the same way (capture one shared `torch.randn` tensor, pass it to
both sides) so the test still checks the *computation*, not two independent
random draws. **Full suite still 118 passed, 0 failed** after this change.

Regenerated the real test-sample audio with the fix
(`scripts/vocoder_debug_2026_09_20/stage1_eval_v4_noisefix.py` if it made it
into a commit, else rebuild from `stage1_eval_v3_savemel.py` + this fix) and
added it to the artifact (see below) for the user to listen to. **If a new
session picks this up and the user says this fixed it: done, move to
whatever's next (RTF/perf is Stage 2/3, out of scope for Stage 1). If they
say it's still off: this exact fix was real and correct on its own terms
(confirmed by the bisection), so any remaining issue is something else
layered on top, not this.**

## THE EARLIER HEADLINE FINDING (2026-09-20): the bug is in our HiFT vocoder, not the flow decoder

The user listened to real synthesized audio and reported it still sounds
noisy/robotic with a voice mismatch, despite WER/speaker-sim both passing.
A long investigation (detailed below, mostly now a *closed* side-quest) had
been chasing this inside the flow decoder's CFM estimator. **That direction
turned out to be a red herring.** The decisive test, borrowed from the
CosyVoice1 reference PR's own methodology (isolate mel-correctness from
waveform-correctness as two separate questions):

**Took our flow decoder's real mel output for the exact zero-shot test
sample, fed it into the REAL, unmodified, official `cosyvoice.hifigan.generator.HiFTGenerator`
+ `ConvRNNF0Predictor` classes** (fetched fresh from `FunAudioLLM/CosyVoice`
on GitHub, run completely standalone — no TT, no our port at all — with the
real `hift.pt` weights, **0 missing / 0 unexpected keys on a strict
`load_state_dict`**, confirming this is genuinely the right class for this
checkpoint). Also generated our own port's vocoder output on the exact same
mel, for a fair A/B.

**Result, confirmed by the user listening to both: the real official vocoder
produces clean, crisp audio from our mel. Our own vocoder does not.** This
proves the flow decoder's mel output is fine — the bug is entirely downstream,
in our own HiFT vocoder / F0 predictor TT port.

Both clips (plus the original reference and v1/v2 synthesis clips) are on a
published artifact: **https://claude.ai/artifact/9KQKjr5nHX9aR4Ez7A95t2**
(private, "CosyVoice2 Stage 1 Results", version 3 as of this write-up — look
for the "Vocoder isolation test" section). If a new session needs to
regenerate this, the recipe is below under "Reproducing the vocoder isolation
test."

### Why the earlier flow-decoder investigation was a red herring (closed, but instructive)

Extensive bisection (attention math proven bit-exact against real
`diffusers.Attention`, FFN proven bit-exact against real `diffusers.FeedForward`,
resnet blocks proven bit-exact against the ONNX graph) eventually found the
flow decoder's CFM estimator genuinely diverges from `flow.decoder.estimator.fp32.onnx`
(PCC 0.944 on the raw vector field, compounding to visible divergence over a
10-step Euler solve) — and traced the exact cause: **the ONNX graph applies a
real causal-with-50-frame-lookahead attention mask** (query `i` attends to
keys `[0, i+50]`) that our torch reference and TT port do not replicate.

This looked like a smoking gun, but isn't one: real upstream
`cosyvoice/flow/decoder.py`'s `CausalConditionalDecoder.forward` only applies
that chunked mask when `streaming=True`; for `streaming=False` (the actual
offline zero-shot synthesis path our reference/eval implements) it reduces to
the plain padding mask our reference already uses. And `cosyvoice/flow/flow.py`'s
own ONNX-export-adjacent code calls `model.inference(..., streaming=True,
finalize=True)` — strong evidence `flow.decoder.estimator.fp32.onnx` was
exported with **streaming mode baked in**, making it the wrong ground truth
for a non-streaming comparison. The apparent divergence is very likely just
streaming-vs-non-streaming attention patterns, not a bug in our code.

**Not fully closed** — the rigorous way to settle it would be installing the
real `cosyvoice` Python package and running its actual
`CausalConditionalDecoder.forward(..., streaming=False)` directly, as a
config-matched ground truth, instead of the ONNX file. Given the vocoder
isolation test above conclusively shows the flow decoder's *mel output* is
fine in practice, this is now low-priority — worth a footnote if a future
session has spare time, not worth chasing before the vocoder bug is fixed.

All the flow-decoder bisection technique/scripts (attention/FFN substitution
against real `diffusers` classes, ONNX intermediate-tensor extraction by
adding graph outputs, per-Euler-step divergence tracking) are reusable
methodology if the vocoder fix doesn't fully resolve the audio-quality
complaint — see "Reproducing the flow-decoder bisection" below.

## Two real hardware bugs found via the reference repo (2026-09-20) — one applies to us, one doesn't

The user pointed at two real, filed tt-metal issues discovered by the
CosyVoice1 reference PR's author (`ayewo`) via this exact HiFT vocoder
architecture, both potentially relevant to our own noisy vocoder output.
Investigated both directly on our own hardware/build — not assumed by
analogy.

### tt-metal issue #55542 (`ttnn.cumsum` fp32 catastrophic divergence) — does NOT apply, verified

Their finding: `ttnn.cumsum`'s fp32 accumulation diverges badly at long scan
lengths (their case: ~72k *audio-rate* samples in CosyVoice1's `SineGen`),
and bf16 is actually more accurate there. Our own `tt/hifigan/source.py`
already uses `ttnn.cumsum(..., dtype=ttnn.float32)` for `SineGen2`'s phase
integration — but **CosyVoice2's `SineGen2` integrates phase at MEL rate**
(hundreds to low-thousands of samples), architecturally different from
CosyVoice1's audio-rate cumsum. This was already flagged in our own
`source.py` docstring as "not assumed safe by analogy," with a test
(`tests/pcc/test_sine_gen2.py::test_cumsum_precision_at_mel_rate`) — but that
test only checked `mel_frames=250`, and our real test utterance is 464.

Extended the measurement directly (fp32 and bf16 `ttnn.cumsum` vs. a float64
torch reference, `mel_frames` from 250 to 4000):

```
mel_frames=  250  fp32 err=2.7332e-06  bf16 err=1.5618e-02
mel_frames=  464  fp32 err=4.5270e-06  bf16 err=3.1603e-02
mel_frames=  800  fp32 err=1.1486e-05  bf16 err=6.2625e-02
mel_frames= 1200  fp32 err=1.1582e-05  bf16 err=6.2625e-02
mel_frames= 2000  fp32 err=3.3032e-05  bf16 err=1.2594e-01
mel_frames= 4000  fp32 err=2.0105e-04  bf16 err=2.5084e-01
```

fp32 stays safely under our test's `0.01` gate at every length up to 4000
(16x our real utterance), and is **consistently 1000–6000x more accurate than
bf16** at every length — the opposite of what #55542 found for the long
audio-rate case, exactly as expected given the different (much shorter)
regime. **Conclusion: not our bug, and our existing fp32 choice is correct —
now backed by real measurement at production-relevant lengths, not just the
original 250-frame test.** Worth adding `mel_frames=464` (or higher) as a
permanent case to `test_cumsum_precision_at_mel_rate` at some point, but this
is a nice-to-have, not urgent.

### tt-metal issue #55545 (`ttnn.conv1d`'s `prepare_conv_weights` silently disagrees with the op's own prep, on Wormhole) — DOES apply, unfixed

Their finding: `ttnn.prepare_conv_weights` (hoisted weight preparation, used
to make convs traceable) silently disagrees with the op's own internal
preparation at some input lengths on Wormhole, off by up to `1e37` — a wrong
number, not an exception. Found via `Conv1d(128->128, k=11, pad=5)`, which is
architecturally identical to our own HiFT source resblocks' 128-channel,
kernel-11 convs (CosyVoice2's `resblock_kernel_sizes=(3,7,11)` includes k=11
too).

**Reproduced directly on our own build**: ran their exact
`~/reference-cosyvoice1/models/demos/cosyvoice/scripts/repro_conv1d_wormhole.py`
unmodified. It reproduces — disagreement at `L=9217` (prepared weight path
gives `5.55e37`, raw weight gives the correct `9.438`) for the identical
`Conv1d(128->128, k=11)` geometry. **The specific bad length differs from
theirs** (their reported bad band 8193/8321/8577/8705 is all *fine* on our
build; ours breaks at 9217 instead) — confirming this is build/version-
specific, not a fixed list that can be avoided by construction.

Also tested our own real-workload-derived lengths directly (128-channel
resblock length = `mel_frames * 40`, confirmed via our own runtime warning
logs showing `Conv1d(128->128, k=11) length 18560` at `mel_frames=464`):
lengths `2000, 4000, 8000, 18560, 24000, 32000, 40000, 60000` (covering
`mel_frames` 50–1500) **all agree, 0 disagreements** — so this exact resblock
is not currently hitting the defect for realistic utterance lengths *on this
build, today*. But since the bad lengths are sparse and build/version-
specific (confirmed by the 9217-vs-8193 mismatch above), this is not a
durable guarantee — a `ttnn` update or a different utterance length could
land on a bad one at any time, silently.

**We currently have zero protection against this.** `TtConv1d._prepared` in
`tt/hifigan/conv.py` only falls back to raw weights on an *exception* from
`ttnn.prepare_conv_weights` — never checks for silently-wrong output.

**Also affected, same risk class, not covered by either filed issue**:
`TtConvTranspose1d` in `tt/hifigan/upsample.py` has the analogous unverified
`ttnn.prepare_conv_transpose2d_weights` call (same hoisted-prep pattern, same
hardware) — worth the same fix for consistency, though not independently
confirmed to have a live bug the way `TtConv1d`'s was.

**Important context: our `tt/hifigan/conv.py` already has a *different*,
independently-discovered verification mechanism** — `TtConv1d._verify_and_resolve`
checks `accurate_compute_config` (HiFi4 + `fp32_dest_acc_en=True` +
`packer_l1_acc=True`) against `safe_compute_config` (same but
`fp32_dest_acc_en=False`) once per geometry, because that specific
combination was independently found to silently corrupt `ttnn.conv1d` at
CosyVoice2's `source_downs` shape (`in_channels=18, kernel=16, stride=8,
padding=4`, PCC 0.0011 vs. expected ~0.9999). **This is a different bug axis
from #55545** (compute fidelity config vs. weight preparation) — our existing
fix does NOT cover the weight-prep axis at all, and the reference repo's fix
does NOT cover the compute-config axis at all (they don't have that
mechanism). Both are real, independent, and need covering.

**IMPLEMENTED 2026-09-20 (in the working tree — NOT committed/pushed yet,
per the user's standing instruction; verify with `git status` whether this
made it into a commit by the time you read this).** Extended
`TtConv1d._verify_and_resolve` (`tt/hifigan/conv.py`) and the equivalent in
`TtConvTranspose1d` (`tt/hifigan/upsample.py`) into a *unified* check rather
than bolting on a second separate verification pass like the reference repo
did. On first sighting of a new `(input_length, batch_size)` geometry, the
fast path's output (prepared weight + accurate compute config) is compared
against ONE maximally-conservative reference computed together (raw/
unprepared weight + safe compute config); whichever `(weight, bias,
compute_config)` triple agreed is cached per geometry in `_verified_config`
(previously this dict only stored a bare `compute_config`, covering the
compute-fidelity axis alone). Same cost as before — one extra conv call,
only once per new geometry, amortized to nothing over an utterance — but now
catches either defect (or both at once) without needing to know in advance
which lengths are dangerous on whatever build/hardware/ttnn-version this
runs on.

**Verified**: full suite still **118 passed, 0 failed** after the change
(`pytest models/demos/audio/cosyvoice2/tests/pcc/ -q`). As expected, this did
NOT change today's real-utterance audio output — the earlier direct sweep
already found 0 disagreements at our actual production conv lengths on this
build/day; this is a safety net against a sparse, build-version-specific
defect (confirmed to still exist on our build, just not currently triggered
by our specific workload), not a fix for a symptom we could reproduce
end-to-end. **This did not explain the noisy audio** — see below for where
the investigation goes next.

## (HISTORICAL, superseded 2026-09-20/21) "CURRENT STATUS: root cause fixed, awaiting user confirmation it fixed the audio"

**Resolved: the user confirmed by ear on 2026-09-20 that the audio is fixed (the
real cause was `TtStft`, see the SUPERSEDES section near the top). The text below is
kept for the record only; its "zero excitation noise" fix was real but not the audible cause.**

The vocoder bisection plan below (this section used to describe it as
in-progress) is DONE — see "ROOT CAUSE FOUND AND FIXED" above for the full
result. Short version: torch reference is fully correct (verified stage by
stage against the real official classes, PCC 0.9999999+ everywhere once a
diagnostic-script noise-matching mistake was caught and fixed); the real bug
was `TtHiFTGenerator.inference()` running every real synthesis call with
zero excitation noise instead of upstream's real per-call random draw; fixed;
118/118 tests still pass; new audio regenerated and posted to the artifact.
**Not yet re-confirmed by the user listening** as of this write-up — if
they've since said it's fixed, this whole vocoder investigation is closed and
whatever's next is a fresh topic (RTF/perf, most likely, per Stage 1's own
scope). If they've said it's not (fully) fixed, this exact defect and fix
were still real (confirmed independently, not just by ear) — treat this as
one real problem solved, not the disproven starting point for further vocoder
digging in the same spot.

## Diagnostic scripts also live in the repo now (committed; the newest ones in `090bf1cbb2`, pushed 2026-09-20)

*(Update 2026-09-21: the "uncommitted / user will handle committing" wording in this
section is stale. `git status` shows the whole directory tracked; `090bf1cbb2` added
`decode_stage_bisect.py`, `f0_dtype_check.py`, `f0_ablation_ab.py`, `conv_config_matrix.py`,
`repro_prepare_conv_weights_2p16.py`, `stft_length_sweep.py` and the README section. Their
outputs go to `$COSYVOICE2_DEBUG_OUT`.)*

`models/demos/audio/cosyvoice2/scripts/vocoder_debug_2026_09_20/` — copies of
the `/tmp` scratch scripts described below (frontend reimplementation, the
real-`cosyvoice` package shim, the vocoder isolation test, the flow-decoder
bisection scripts, the vocoder stage-by-stage bisection + its noise-matching
correction, `stage1_v3_mel.npy`), plus its own `README.md` explaining what
each one is for. Copied there specifically so they'd survive this session
ending; NOT committed as of this write-up (the user said they'd handle
committing themselves) — check `git status`/`git log` to see if that
happened by the time you're reading this. If it's there, prefer these copies
over anything still claimed to be in `/tmp` (which won't have survived). The
vocoder bisection scripts are all in this directory too:
`vocoder_stage_bisect.py` (the misleading first pass, kept for the lesson,
not the numbers), `vocoder_stage_bisect_fair_noise.py` (the noise-capture
correction), `vocoder_stage_bisect_v2.py` (the corrected full bisection
showing PCC 0.9999999+ everywhere), `stage1_eval_v4_noisefix.py`/
`stage1_v4_mel.npy`/`build_artifact_v4.py` (the fix applied end-to-end and
posted to the artifact).

## Reproducing the vocoder isolation test (scripts in `/tmp` scratch, likely gone)

All diagnostic scripts this session lived in `/tmp/claude-*/scratchpad/` —
NOT committed, NOT guaranteed to survive a session boundary. Key ones for the
headline finding, in order:

1. `stage1_eval_v3_savemel.py` — a copy of the real zero-shot Stage 1 eval
   (see below) with one addition: `np.save(f"{SCRATCH}/stage1_v3_mel.npy",
   mel.numpy())` right after `tt_flow.inference(...)`. Produces the real mel
   for the exact test sample plus our own vocoder's waveform on it
   (`stage1_synth_v3_ourvocoder.wav`).
2. **Building a local, unmodified real-`cosyvoice` package shim** — needed
   because `cosyvoice.hifigan.generator.HiFTGenerator` imports real
   `cosyvoice.transformer.convolution`/`activation` and
   `cosyvoice.utils.common`, which aren't installed (installing the full
   `cosyvoice` pip package risks repeating the torchaudio-ABI-breakage
   incident from earlier this bring-up — see environment gotchas). Instead:
   ```bash
   mkdir -p real_cosyvoice_pkg/cosyvoice/{transformer,utils,hifigan}
   touch real_cosyvoice_pkg/cosyvoice/__init__.py real_cosyvoice_pkg/cosyvoice/transformer/__init__.py \
         real_cosyvoice_pkg/cosyvoice/utils/__init__.py real_cosyvoice_pkg/cosyvoice/hifigan/__init__.py
   curl -fsSL https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/transformer/convolution.py \
       -o real_cosyvoice_pkg/cosyvoice/transformer/convolution.py
   curl -fsSL https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/transformer/activation.py \
       -o real_cosyvoice_pkg/cosyvoice/transformer/activation.py
   # cosyvoice/utils/common.py and cosyvoice/hifigan/generator.py + f0_predictor.py were already
   # fetched earlier this session into real_src/ (cosyvoice_utils_common.py,
   # cosyvoice_hifigan_generator.py, cosyvoice_hifigan_f0_predictor.py) -- copy those in too,
   # renamed to their real module paths (cosyvoice/utils/common.py, cosyvoice/hifigan/generator.py,
   # cosyvoice/hifigan/f0_predictor.py). If real_src/ is gone, re-fetch fresh from
   # raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/... (all public, Apache-2.0).
   ```
   The real, unmodified `ConvRNNF0Predictor` class (not `CausalConvRNNF0Predictor`
   — confirmed by checkpoint key compatibility, `f0_predictor.pt` keys match
   the plain class's `condnet.{i}` naming, not the causal variant's) needs no
   further deps once `cosyvoice.transformer.convolution`/`utils.common` exist
   on `sys.path`.
3. **`real_vocoder_check.py`** — loads `stage1_v3_mel.npy`, builds the real
   `HiFTGenerator`+`ConvRNNF0Predictor` with the real CosyVoice2 config
   (`upsample_rates=[8,5,3]`, `upsample_kernel_sizes=[16,11,7]`,
   `resblock_kernel_sizes=[3,7,11]`, `source_resblock_kernel_sizes=[7,7,11]`,
   `n_fft=16, hop_len=4`, `sampling_rate=24000` — all confirmed against our
   own port's already-verified config in `tt/hifigan/generator.py`), loads
   real `hift.pt` via `generator.load_state_dict(hift_sd, strict=False)`
   (confirm 0 missing/0 unexpected — that's the check that this is really the
   right class), calls `generator.inference(speech_feat=mel_cf)` (note:
   channel-FIRST `[B,80,T]`, transpose from our port's channel-last
   convention first), saves `stage1_synth_v3_REALVOCODER.wav`.
4. **Artifact update** — `build_artifact_v3.py` reads the previously-published
   artifact's saved HTML (via `Artifact` tool's `read` action, which saves
   the full page to a local file — read that file with plain Python, NEVER
   with the `Read` tool, it's ~1-3MB of base64 audio and will blow up context
   for no reason, a mistake already made and fixed once this bring-up), splices
   in a new "Vocoder isolation test" section with both wav files as base64
   `<audio>` tags, republishes via `Artifact` tool's `publish` action with the
   existing `url`.

## Reproducing the flow-decoder bisection (now low-priority, but the technique is reusable)

If the vocoder fix doesn't fully resolve the audio-quality complaint, or a
future session wants to properly close the streaming/non-streaming question:

1. Real ONNX graph node names carry the real module scope (e.g.
   `/down_blocks.0.1.0/attn1/to_out.0/Add_output_0`), matching `flow.pt`'s own
   key naming — confirmed via `onnx.load(...).graph.node[i].name`. Any
   intermediate tensor can be exposed as a graph output via
   `model.graph.output.append(onnx.helper.make_tensor_value_info(name,
   onnx.TensorProto.FLOAT, None))` then rebuilding the `InferenceSession` —
   no need to re-export, just mutate the loaded `onnx.ModelProto` in memory.
2. Real attention/FFN weight tensors ARE directly extractable as ONNX
   initializers (contrary to what an earlier pass of this investigation
   assumed) — e.g. `onnx::MatMul_9071` for `to_q`'s weight, shape `[in,out]`
   (i.e. `nn.Linear.weight.T`). Diffed bit-exact against `flow.pt`.
3. To substitute a real `diffusers` class for a hand-rolled piece and compare
   bit-for-bit: build the real activation entering that piece exactly as our
   own `forward()` does (verified by checking output `std`/`mean` match the
   corresponding stage in a full run), construct the real class
   (`diffusers.models.attention_processor.Attention` /
   `diffusers.models.attention.FeedForward`), copy weights in directly
   (`real_module.to_q.weight.copy_(our_module.to_q.weight)` etc.), run both
   on the identical input, `comp_pcc` + max-abs-diff. Both attention and FFN
   were proven bit-exact (PCC 1.0, diff 0.0) this way — the real divergence
   traced to the causal+lookahead attention *mask*, not any weight/formula.
4. The per-Euler-step divergence table (independent running state each side,
   same initial noise, same `t_span`/`dt` schedule) is the right technique to
   distinguish "constant per-step bug," "compounding ODE error," and
   "sudden schedule-specific jump" — see the analysis in this session's
   transcript if the exact script is gone, it's straightforward to rebuild
   from `CausalConditionalCFMRef.solve_euler`'s own loop structure.

## Environment gotchas (see also memory files, same content)

- **Fresh build needs `uv pip install -e .` after `build_metal.sh`** — the
  C++ build alone doesn't install the `ttnn` Python package.
- **Never `pip`/`uv pip install` anything torch-adjacent without
  `--extra-index-url https://download.pytorch.org/whl/cpu`** — this venv's
  `torch` is a CPU-only build; installing something that pulls in
  `torchaudio` from default PyPI silently breaks its ABI (`OSError: Could not
  load this library: .../torchaudio/lib/_torchaudio.abi3.so`) even though
  `import torchaudio` succeeds. Diagnose via `uv pip show torch torchaudio` —
  `+cpu` suffix should match both. Fix: `uv pip install --extra-index-url
  https://download.pytorch.org/whl/cpu --reinstall "torchaudio==<version>"`.
  This is why the real-`cosyvoice` package was shimmed by hand (see above)
  rather than `pip install`ed.
- **Git**: the user does NOT want `Co-Authored-By: Claude` in commits —
  author only. **And, as of 2026-09-20: do not commit or push at all unless
  explicitly asked for that specific action in that turn** — see the note at
  the top of this file.

## Memory files (survive within this same environment, not across a full VM reset)

`/home/user/.claude/projects/-home-user-tt-metal/memory/` —
`MEMORY.md` (index), `tt_metal_vm_env_setup.md`, `cosyvoice2_stft_corruption_finding.md`,
`cosyvoice2_rtf_findings.md`, `cosyvoice1_reference_insights.md`,
`bounty_54104_criteria_and_cfm_trace_pattern.md` (the older `cosyvoice2_bringup_state.md`,
`tt_metal_build_env_gotcha.md` and `git_commit_permission.md` may no longer exist; the
commit rule is at the top of this file). Read these too if they're still there; this file
is the durable (committable, if asked) version of the same information, plus
everything from 2026-09-20's session that hasn't been folded into memory yet.
