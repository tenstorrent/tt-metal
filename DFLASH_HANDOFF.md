# DFlash speculative decoding on Qwen3.6-27B (T3K) — handoff

**Branch:** `ign/qwen_3.6_27B_dFLASH` · **Last commit at handoff:** `ad60b404d4a`

You are picking this up on a fresh machine. Read §1–§3, then go to §5. §7 and §8 exist so you
don't repeat work that has already been done and measured.

---

## 0. Update 2026-09-15 (second machine) — both open problems now have a mechanism

The environment reproduces: `test_dflash_traced_throughput.py` gives acceptance **7.000** and
**24.01 tok/s** traced on this box. Everything below was measured after that gate passed.

**The failure in §4 is a HANG, not a death, and it is deterministic.** py-spy put every occurrence at
the identical state — `drafter.py:585 _kv_heads`, `layer_idx=4`, `start=110`, `hist_len=109`,
`new_ctx=1`, `q_len=2`, `kv_seq=3` — and gdb put the native stack in
`SystemMemoryManager::fetch_queue_reserve_back`, spinning on a dispatch fetch queue that never
drains. The host is stuck **pushing**; the device has stopped consuming. So the Python frame is
where the queue filled, not the op at fault — as §4 already suspected.

**Root cause: a program shape first compiled AFTER the capture.** §3's ordering rule ("one eager
generation before `enable_traced_verify()`") is necessary but **not sufficient** — a warm-up compiles
only the shapes *it* touches. `test_dflash_prose_throughput.py` warmed with `bench, max_new_tokens=8`
but measured a 64-token prompt at `NEW_TOKENS=48`, whose final step is
`verify_size = min(16, 112-110) = 2`. That 2-wide block was a drafter shape the warm-up never
produced, so it compiled under a parked trace, and hung.

*Proof:* warming with the real prompts at the real budget before the capture makes the test **pass**
(243 s) where it had hung twice, deterministically. Committed in that file.

### CORRECTION: the capture is NOT offset-specific -- that finding was a fixture artifact

Commits 4a57957c44f and 0aa99e9361d claim the verify trace is only valid at the chunk_start it was
captured at, and localise the bake to the first full-attention layer. **Both claims are wrong.**
With the fixture bug removed, every arm is exact:

    capture@0   replay@0     logits pcc 1.0   argmax 1.0000   all taps 1.0
    capture@0   replay@128   logits pcc 1.0   argmax 1.0000   all taps 1.0     <- was 0.843 / 0.0625
    capture@128 replay@128   logits pcc 1.0   argmax 1.0000   all taps 1.0

ONE capture serves every chunk_start as well as every valid_len. The 0.843 came from the test
itself: `capture_verify_trace` runs real forwards, those include `paged_fill_cache`, and capturing
at offset 0 wrote its dummy zero tokens over KV pages 0..1 -- the prefix the test had just primed.
The capture@128 arm looked healthy for the mirror-image reason. The "first divergence at L3 [attn]"
was L3 being the first layer to READ the clobbered KV, not the first to bake anything.

WHAT SURVIVES: the fix. `lo == 0` moves demo acceptance 1.138 -> 4.950 and the 200-token crossing
arm 1.118 -> 4.471, matching pure-eager exactly. Those are end-to-end measurements, so replaying
past the anchor IN THE LOOP is genuinely broken -- the explanation was wrong, not the effect.

LEADING HYPOTHESIS NOW: the anchor GDN snapshot interacting with the parked trace. The passing test
takes a FRESH snapshot (`save_gdn_state()`), while `TtTarget.forward` re-anchors with
`save_gdn_state(into=self._anchor_gdn)`. That reuse is independently implicated -- DFLASH_FRESH_ANCHOR=1
moves post-anchor acceptance 1.118 -> 1.617 and removes the output corruption, and the same reuse in
`reset()` SIGBUSed the drafter. Next experiment: re-run test_verify_traced_at_offset.py with the
snapshot reused instead of freshly allocated, which is the one difference left between it and the loop.

IGNORE the "routes to the remaining 2.6x" list below that is premised on a ttnn SDPA fix; the op is
not at fault. The ~2.6x is still on the table, but the path to it runs through whatever makes a
traced replay misbehave after a re-anchor in the loop.

### FIXED: the capture is only valid at chunk_start=0, and the loop replayed it everywhere

The anchor cliff below is real, but re-anchoring was never the culprit. The TRACE was.
`tests/unit/test_verify_traced_at_offset.py` compares replay against eager at two offsets:

    chunk_start=0     logits pcc 1.0     argmax agree 1.0000   taps 1.0   / 1.0   / 1.0
    chunk_start=128   logits pcc 0.843   argmax agree 0.0625   taps 0.979 / 0.938 / 0.880

**One row in sixteen has the right argmax at the anchor.** `capture_verify_trace` hardcodes
`chunk_start=0`, and its docstring's guarantee -- "ONE capture serves every valid_len below the
bucket" -- is about valid_len. Nobody ever checked chunk_start, and it does not hold. `TtTarget`
re-anchors every 128 tokens, so every verify after the first bucket replayed a trace that was not
valid for it.

That explains the whole picture at once. With acceptance 0 the only logit row reaching the output is
row 0, which happens to survive -- so the text stays fluent English while rows 1..15 are wrong, so
EVERY draft is rejected, and the taps are degraded exactly as `take_taps` warns ("reads as a
plausible-looking tap ... and drafts pure garbage").

THE FIX (`targets.py::_run`): use the trace only at `lo == 0`, eager elsewhere. Measured:

    gen200 arm                 time     tok/s   acceptance after the anchor   non-ascii
    traced (broken)           97.47 s    2.05            1.118                 37/728
    fully eager               57.57 s    3.47            4.471                  0/810
    trace restricted to 0     41.36 s    4.84            4.471                  0/810

Strictly better than both: it keeps the trace where it is valid and drops it only where it is not.

    demo spec_128   acceptance 1.138 -> 4.950, 87 steps -> 20, 3.86 -> 6.19 tok/s, 0.22x -> 0.35x

`test_dflash_traced_throughput.py` is unchanged (7.000 / 22.17 tok/s): it never crosses an anchor,
so it keeps the trace throughout.

LOCALIZED FURTHER (same test, capture@0 replay@128, tapping early layers and labelling each):

    L0 [gdn ] pcc 1.0        L3 [attn] pcc 0.9825   <- FIRST divergence
    L1 [gdn ] pcc 1.0        L4 [gdn ] pcc 0.9822
    L2 [gdn ] pcc 1.0        L30[gdn ] pcc 0.8798

The GDN layers ahead of it are BIT-EXACT, which is the control: GDN never receives chunk_start. The
error enters at the FIRST FULL-ATTENTION LAYER and every later layer merely inherits it. So the
baked value lives in the chunked-SDPA / paged-attention path.

And the capture is exact at whatever offset it was taken at:

    capture@0   replay@0     logits pcc 1.0     argmax 1.0000   taps 1.0 / 1.0 / 1.0
    capture@0   replay@128   logits pcc 0.843   argmax 0.0625   taps 0.979 / 0.938 / 0.880
    capture@128 replay@128   logits pcc 1.0     argmax 1.0000   taps 1.0 / 1.0 / 1.0

(`capture_verify_trace` now takes `capture_chunk_start` so this is testable.)

WHAT IS *NOT* THE CAUSE, checked in tt/attention/tp.py's `forward_prefill_paged`: in the
`chunk_start_idx_tensor is not None` branch the Python int is used in exactly two places, and
neither bites. `qk_chunk` is pinned at 128 regardless of it, and `needed_blocks` cannot truncate the
page table because `target_blocks = max(needed_blocks, page_table.shape[-1])` keeps the full width.
The staged inputs are all correct too -- cos/sin, the chunk page table, and `_vt_csi` itself are
re-staged per replay.

That leaves the ttnn op: `chunked_scaled_dot_product_attention` appears to specialise something on
the chunk_start it first sees, so the staged `chunk_start_idx_tensor` does not fully override it.
Confirming that, and fixing it, is a TTNN-side change, not a model-side one -- which is why the
model-side fix here is to restrict the trace rather than to stage harder.

ROUTES TO THE REMAINING ~2.6x, in order of preference:

1. Fix the ttnn op so one capture serves every offset. Ceiling: step ~300 ms at the demo's
   acceptance 4.95 -> ~16.5 tok/s (0.92x), or ~19.8 tok/s (1.11x) with the narrow head.
2. Capture ONE TRACE PER ANCHOR and pick by offset. Correct today (capture@128 replay@128 is
   pcc 1.0) but costs a trace region per anchor: ~250 MB each, so ~500 MB for the demo's two
   buckets and 8 GB for the full 4096-token capacity. Viable only for bounded generations.
3. Re-capture on every anchor advance. Correct, but a capture is ~4.3 s against ~7.8 s of stepping
   per 128 tokens -- roughly 55 % overhead, which gives back most of the win.

NOTE the ceiling in (1) is ~1.1x, not the 1.25x this document originally claimed. The demo's
acceptance is 4.95 where the reference prompt's is 7.000, and that gap is genuine drafter
prompt-dependence. The original headline was measured on the prompt the drafter happens to nail.

### The anchor cliff, as originally diagnosed (cause now known to be the trace)

Everything else in §0 is downstream of this. `tests/reference/test_dflash_anchor_crossing.py` holds
the prompt constant (the same five tokens) and varies only `max_new_tokens`, so the single
difference between arms is whether `start` walks past `ANCHOR`. Per-step accepted lengths,
`start:accepted`, for the 200-token arm:

    before   5:3  8:2  10:16  26:4  30:16  46:2  48:5  53:7  60:10 ... 121:6  127:1
    cross    128:9
    after    137:1 138:1 139:1 140:1 ... 202:1 203:1        <- EVERY step, to the end

    acceptance before the anchor 4.241 (29 steps), after 1.118 (68 steps)
    non-ascii characters in the output: 37 / 728            <- the token soup, same run

After the anchor advances, **every single step accepts exactly 1** — the target's own bonus token.
The drafter never lands another token for the rest of the generation. This is not degradation, it is
total and permanent failure, and it starts at the boundary.

That accounts for everything that looked mysterious:

* **the demo.** Its prompt is 128 tokens, so it crosses immediately and spends nearly every step on
  the far side. Post-crossing acceptance 1.118 vs the demo's 1.138 — the same number.
* **"prompt-dependence" (§6).** Wrong. Healthy runs are the ones that never cross (5+64=69);
  sick ones are the ones that do. Prompt length mattered only because it decides how soon you cross.
* **the token soup.** Same runs, same cause: every post-crossing verify computes from the
  re-anchored state, so the target's own argmax is wrong and greedy verification no longer pins the
  tokens to what the 27B would emit. This is a CORRECTNESS bug, not a throughput bug, and
  `_assert_output_quality` passes it because it only detects repetition.

PRE-EXISTING, not introduced by any change in this branch: the same demo configuration reproduces
acceptance 1.138 and the same soup on the commit before the narrow head.

WHERE TO LOOK. `forward` re-anchors with `self._anchor_gdn = self.model.save_gdn_state(into=...)`
after each whole bucket, and every later `_run` begins `restore_gdn_state(self._anchor_gdn)`. If the
re-anchored snapshot is wrong, every subsequent verify is computed from bad state — which is exactly
the observed signature (drafts always rejected AND garbage tokens). The `into=` buffer-reuse path is
already independently implicated: adding the same reuse to `reset()` SIGBUSed the drafter. The next
experiment is to drop `into=` from the re-anchor save so it allocates fresh, and see whether
post-crossing acceptance recovers.

Note the whole-bucket run also takes the EAGER fallback (`length == ANCHOR`), so a crossing step
mixes an eager verify with traced ones — a second candidate worth separating from the snapshot.

**TESTED, and it is half the answer.** `DFLASH_FRESH_ANCHOR=1` (in `targets.py`, uncommitted) makes
the re-anchor save allocate instead of reusing. Same `gen200` arm:

                             baseline      DFLASH_FRESH_ANCHOR=1
    before the anchor          4.241            4.241     (unchanged, as it must be)
    after  the anchor          1.118            1.617
    non-ascii in output       37 / 728          0 / 810   <- the CORRUPTION IS GONE
    overall / throughput   2.052 / 2.05     2.618 / 5.67 tok/s

So the `into=` buffer reuse at re-anchor is a real correctness bug: dropping it produces clean
English for the whole generation and 2.8x the throughput. The "keeps DRAM flat" rationale costs
little to give up — re-anchoring happens once per 128 tokens, not once per block. Combined with the
SIGBUS that `into=` caused in `reset()`, the buffer-reuse path on this snapshot should be considered
unsafe generally.

TWO THINGS IT DOES NOT FIX, both important:

1. **Acceptance still falls across the boundary** — 1.617 against 4.241 before. A second defect
   remains; the eager whole-bucket verify is the next suspect.
2. **The demo still hangs with it on.** Changing acceptance changes which `(new_ctx, q_len)` pairs
   the loop produces, and any shape the warm-up did not compile hangs under the parked trace. It
   hung at `drafter.py:766`, `start=213`, `q_len=15`, `kv_seq=16` — `verify_size = min(16, 228-213)`.
   This is the SAME bug as §0's opening, resurfacing: warming "the real prompts at the real budget"
   only covers the shapes THAT run happened to hit, so it is not robust to anything that shifts
   acceptance.

   The durable fix is to warm the drafter across its whole width space before the capture —
   `q_len` 1..16 against the `new_ctx` values the loop can produce — rather than relying on a
   warm-up run to stumble on them. `tests/unit/test_drafter_block_width.py` shows each new width
   costs ~3.3 s to compile once, so the space is affordable to cover exhaustively, and doing so
   would make every future acceptance change safe instead of hang-prone.

### On the acceptance oscillation (a separate, smaller effect)

**It is not a decay, it OSCILLATES with period 2.** Everything
below described the acceptance loss as a decay that compounds per generation. That was wrong.
`tests/reference/test_dflash_warmup_position.py` varies only how many traced generations sit between
the capture and the measured one, holding every other value at the known-good run's, each arm in its
own fixture instance:

    N=0   eager 7.000 -> capture -> MEASURED (traced gen 1) = 7.000
    N=1   eager 7.000 -> capture -> gen 1 7.000 -> MEASURED (gen 2) = 1.500
    N=2   eager 7.000 -> capture -> gen 1 7.000 -> gen 2 1.500 -> MEASURED (gen 3) = 7.000

Traced generations run 7.000 / 1.500 / 7.000: ODD ones are healthy, EVEN ones are broken, and the
third recovers completely. Nothing compounds.

A clean period-2 alternation is the signature of a DOUBLE-BUFFERED resource toggled once per use.
`TT_CCL` is exactly that: `get_and_cycle_ag_semaphore_handles` advances `(current_idx + 1) % 2` over
**two** semaphores per axis. A captured trace bakes whichever semaphore it held at capture time, so
on alternate generations the live parity collides with the baked one. Note this arm does NOT share
`tt_ccl` with the drafter, so the earlier "shared TT_CCL refuted" result (below) tested the wrong
half — the suspect is the TARGET's own cycling against its own parked trace, not sharing.

**This makes the demo's acceptance an artifact of parity.** The demo runs eager warm -> capture ->
traced warm -> measure, so it measures traced generation **2** every time: the bad parity.

**The §6 acceptance gap is real and specific to the traced path.**
`test_dflash_generation_repeat.py` (new) holds everything still and varies only generation index:

    eager   gen 1-5   6.200  6.200  6.200  6.200  6.200     flat to three decimals
    traced  gen 1-3   4.429  3.875  1.409                   collapsing

Five eager generations on a reused drafter drift **not at all**, so `dflash_generate`'s
reset/snapshot/restore bookkeeping is sound. The decay appears only once the trace is enabled and
compounds per generation. That also explains §6's "identical tokens, different acceptance" and the
control readings, which track generation index exactly: 7.000 (2nd generation) → 2.611 (3rd) →
1.270 (4th).

Prime suspect, **not yet proven**: `dflash_generate` calls `target.reset()` at the top of *every*
generation, and §8 already records that `reset()` reallocates the anchor's GDN snapshot and that
Metal flags this as unsafe with an active trace. The first traced generation is safe because the
capture just happened; each later one re-does that reallocation under a parked trace. Next step is
to make `reset()` leave the trace-baked buffers alone (or release/re-capture per generation) and see
whether the decay disappears.

**THE PROMPT IS EXONERATED — §6's framing is wrong.** §6 explains the acceptance gap as *"acceptance
is strongly prompt-dependent"* and doubts the 22 tok/s headline on that basis. Measured by running
the demo's own loop on the reference prompt (`DFLASH_PROMPT="The capital of France is"`, which the
demo now honours):

    test_dflash_traced_throughput.py   "The capital of France is"   acceptance 7.000
    demo spec_128                      condiment, 128 tok           acceptance 1.138
    demo spec_128, prompt overridden   "The capital of France is"   acceptance 1.031

The reference prompt does WORSE in the demo than the prose one, and 7x worse than the same prompt in
the reference test. Whatever prompt-dependence exists, it is not what makes the demo lose — the
demo's own structure (its generation index, i.e. the decay above) is.

**Unexplained, and it matters: step time degrades WITH acceptance, not independently of it.** The
"step time is near-constant" model in §6 holds across every run in this work except the low-
acceptance ones:

    demo + condiment prompt    22.9 s /  88 steps =  260 ms/step   (acceptance 1.138)
    demo + reference prompt   151.2 s /  97 steps = 1559 ms/step   (acceptance 1.031)
    repeat test traced gen 1    2.65 s /   7 steps =  379 ms/step   (acceptance 4.429)
    repeat test traced gen 3   20.32 s /  22 steps =  924 ms/step   (acceptance 1.409)

A 6x step-time regression is a SECOND effect on top of the acceptance collapse and has no mechanism
yet. That the two degrade together across four independent runs argues for one shared cause rather
than two — most likely whatever the traced path is corrupting. Do not model throughput as
`acceptance / 295 ms` in this regime; that formula is only calibrated on healthy runs.

**The two bugs are entangled, which is why §4 resisted bisection.** The decay changes the accept
counts, so `new_ctx` takes values no warm-up produced, `kv_seq = new_ctx + q_len` becomes an
uncompiled shape, and *that* hangs. The repeat test hung on its own traced arm at `q_len=10,
kv_seq=11` — a novel width reached only because acceptance had already collapsed. So neither
"generation count ≥24" nor "prompt length 64" is causal; both merely raise the odds of reaching a
novel shape after enough decay.

**Corrected premise.** §4 dismissed the drafter-attention localisation because "that op's shapes are
constant across steps — always `ctx_pad(16) + block(16) = 32` rows". They are constant only while
`q_len == block_size`. `_layer_attention` computes `kv_seq = kv_src.shape[-2]` from
`new_ctx + q_len`, which was **3** on the hung step. Constant across every step but the last.

**Device note.** Three hangs, three `tt-smi -r`, all recovered. All three ETH failures named the
**same** ASIC — `10226819085, ETH core e8-6` — matching the single-ASIC pattern §5.5 flagged.

### A step-cost win that is already measured: the NARROW VERIFY HEAD (+21 %)

`TtTarget.forward` returns `logits[:, -S:]` with S ≤ 16, but the capture ran `norm` + `_lm_head`
over the whole 128-row bucket and read back up to 128 rows — and `_lm_head` ends in an **all-gather
that replicates the vocab**. Up to 8× of the tail matmul, that collective, and the 26.8 ms readback
was spent on rows that are thrown away. The verify is device-bound (replay 107.1 ms vs 105.8 ms
device), so removing device work is the only lever that helps.

This was previously rejected, in a comment at `prefill_block_all_logits`: *"a pre-head slice would
need tile alignment and would recompile per length."* Both objections are about cutting the window
**exactly**. `Qwen36Model.lm_head_window` snaps it out to tile boundaries instead: the start is
always 32-aligned, and since `keep_rows ≤ 16` the window spans at most two tile rows, so its width
is only ever **32 or 64** — two program shapes for the whole loop, both warmed before the capture
(which is what stops it re-introducing the compile-under-a-parked-trace hang). Proven exhaustively,
without hardware, in `tests/unit/test_lm_head_window.py`.

MEASURED (T3K, 64 new tokens, separate processes, `tests/reference/test_dflash_narrow_head.py`):

    wide head     64 tokens in 2.90 s = 22.08 tok/s (45 ms/tok), acceptance 7.000
    narrow head   64 tokens in 2.40 s = 26.68 tok/s (37 ms/tok), acceptance 7.000
                  -> +20.8 %, tokens BIT-IDENTICAL, 1.24x -> 1.49x production

Enable with `target.enable_traced_verify(narrow_head=True)`. **Default is OFF** — validated so far
on one prompt at one length; widen it (start with `test_dflash_target_trace_replay.py`'s PCC gate)
before making it the default. Independently, `forward` now passes `keep_rows=0` for the whole-bucket
iterations whose logits are discarded in full, skipping their head entirely; that is on by default
and `test_dflash_traced_throughput.py` still reports acceptance 7.000 with identical text.

Note this is *worth more than shrinking ANCHOR*, which was the other candidate: 128→32 buys ~+20 %
on rows but gives ~13 % of it straight back to `max_block` truncation (`1/(1 + acceptance/ANCHOR)`),
costs PCC via extra GDN carries, and needs a page-size change. Net ~+4 % against this +21 %.

---

## 1. What this is, and where it stands

A 1.73B DFlash block-diffusion **drafter** proposes a 16-slot block of tokens; the 27B **target**
verifies all 16 in **one** forward and commits every slot it would have produced anyway. Verification
is greedy, so **the emitted tokens are exactly the tokens the 27B would have emitted alone** — the
drafter can only change how fast they arrive, never what they are. Acceptance length (tokens
committed per target forward) is therefore the entire speedup story.

| | |
|---|---|
| Production traced decode (the baseline that matters) | **17.87 tok/s** (text_demo.py, `traced_128`, ISL 128) |
| DFlash at start of this work | 18.82 tok/s |
| DFlash now | **22.40 tok/s** ≈ **1.25×** production |
| Acceptance on `"The capital of France is"` | **7.000** |
| Acceptance on ordinary prose | **~2.0–2.3** ⚠️ |

**Two things are open, and the second one is more important than the first:**

1. **The demo does not pass.** [`demo/dflash_demo.py`](demo/dflash_demo.py) dies on every case. See §4.
2. **The 22.40 tok/s headline may not generalise.** Every throughput and acceptance number in this
   work was measured on a five-token prompt whose continuation the drafter nails. Throughput is
   `acceptance / step_time` and step time is near-constant, so if prose acceptance is really ~2.2
   against 7.0, the honest figure could be closer to **~0.4× production, i.e. a regression**. This is
   not established either way — it is the single most valuable thing to settle. See §6.

Do not quote 22.40 tok/s as a result until §6 is resolved.

---

## 2. Setting up the new machine

```bash
# repo env (this is what source.sh does; it is not committed)
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH="${TT_METAL_HOME}"

# the run env for every command in this document
export DFLASH_RUN_TARGET=1              # without this every DFlash test skips
export MESH_DEVICE=T3K                  # 8-chip; DFlash is a T3K path only
export HF_MODEL=Qwen/Qwen3.6-27B        # target
export DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash   # drafter, 1.73B
export TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B
```

Both checkpoints come from HuggingFace. The **first run on a new machine pays two separate cold
costs** and neither is a hang:

* **Weight conversion into `TT_CACHE_PATH`** — several minutes, visible as `Loaded cache for …` /
  cache-miss lines.
* **Kernel compilation** — see §5.2. Budget ~30 min for the very first generation.

**Sanity check before anything else** (small, fast, no drafter):

```bash
pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_traced_throughput.py
```

This is the known-good reference. It should report acceptance **7.000** and ~22 tok/s. If it does
not, your environment is wrong and nothing below will mean anything.

---

## 3. How the pieces fit

```
generate.py::dflash_generate          the speculative loop (host)
  ├─ drafter.propose(taps, ...)       TtDrafter → TtDFlashDrafter   (tt/dflash/drafter.py)
  │                                   runs EAGER, deliberately (see §7)
  └─ target.forward(block, start)     TtTarget                      (reference/dflash/targets.py)
                                      verify forward runs TRACED
```

Three facts about `TtTarget` that explain most of the code's shape:

* **`ANCHOR = 128`.** Every forward re-runs the span `[anchor, start+S)` as one 128-row bucket at
  `chunk_start = anchor`. `ANCHOR` is simultaneously the mask bucket size *and* the `chunk_start`
  alignment. `_run` asserts both.
* **The trace is only used when `length < ANCHOR`** ([targets.py:434](reference/dflash/targets.py#L434)).
  A whole-bucket span (prompt prefill, `length == ANCHOR`) takes an **eager fallback**. So a single
  run exercises both paths.
* **`max_block(start) = ANCHOR - (start % ANCHOR)`** — returns **1** when `start % 128 == 127`. That
  is the bug fixed in `158aa0194ae`; see §8.

**Ordering is load-bearing:** you must complete at least one **eager** generation *before*
`enable_traced_verify()`. Capturing first leaves the drafter projections, tap gather, LM head at the
drafted width, and the eager fallback uncompiled — the first traced generation then compiles them
*with a trace parked*, which hangs the process and wedges the device.
`TtTarget.enable_traced_verify` now asserts this (`_has_generated`), and `_has_generated`
deliberately survives `reset()`.

---

## 4. The open failure: what actually happens

**Symptom.** Any generation of **≥24 new tokens** after a capture fails. All three demo cases
(`spec_128` 100 tok, `spec_128_long` 256 tok, `spec_512` 100 tok) exceed that.

**Reproduction:** [`tests/unit/test_dflash_hang_repro.py`](tests/unit/test_dflash_hang_repro.py) —
a bisection over generation count, prompt length, and traced-vs-eager, logging one arm at a time.

**Last full measurement (2026-09-15, under Watcher):**

```
eager-1              1689.19s   acceptance 3.000   <- ALL kernel recompile, not a hang
eager-2                 2.80s   acceptance 3.000
capture                 4.27s
traced-1                0.48s   acceptance 3.000
traced-2                0.52s   acceptance 3.000
traced prompt len 16  137.72s   acceptance 3.000
traced prompt len 24  282.75s   acceptance 1.500
traced prompt len 32  141.36s   acceptance 3.000
traced prompt len 48    8.68s   acceptance 3.000
traced prompt len 64      --    PROCESS DIED, silently
```

### What the evidence does and does not say

**It is a silent process death, not a hang.** No traceback, no pytest summary, no Watcher trip, no
readable OOM record — the process stopped between one log line and the next. Earlier in this work it
was characterised as a *hang* in the drafter's RoPE at
[`rope_tp.py:730`](tt/attention/rope_tp.py#L730). **Treat that localisation as unreliable:**

* ttnn dispatch is **asynchronous**, so a Python traceback points at wherever the host first
  *blocks*, not at what the device is stuck on.
* That op's shapes are **constant across steps** — `k` there is always `ctx_pad(16) + block(16) = 32`
  rows ([drafter.py:697](tt/dflash/drafter.py#L697)). It cannot itself be what degrades with
  generation count.
* Moving it to DRAM was tried and **did not help** (reverted).

**Already ruled out as the trigger:**

| Hypothesis | Verdict |
|---|---|
| Number of generations after a capture | ✗ — traced-1 and traced-2 both complete in <0.6 s |
| Prompt length alone | ✗ — lengths 16/24/32/48 all pass at 4 new tokens |
| L1 vs DRAM placement of the RoPE slice | ✗ — reverted, no change |
| The traced verify itself | ✗ — eager arms reach the same territory |

**The remaining correlate is generation count ≥24**, i.e. how far the *drafter's* KV history has
grown, not the target's.

---

## 5. How to debug it — do these in order

### 5.1 First, reproduce on the new hardware

```bash
pytest -svq models/demos/blackhole/qwen36/tests/unit/test_dflash_hang_repro.py
```

Watch for the `>>>>>` markers. If it now completes all arms, the fault was machine-specific and you
should say so loudly before doing anything else.

### 5.2 Understand the Watcher trap before you use Watcher

`TT_METAL_WATCHER` adds `-DWATCHER_ENABLED`, which **changes the kernel build hash**, so the entire
27B kernel set recompiles from scratch. That looks exactly like a hang: ~28 minutes with no output.
It is not one. **Before calling anything a hang under Watcher, check:**

```bash
find ~/.cache/tt-metal-cache/*/kernels -newermt '-90 seconds' -type f | wc -l
ps -o pid,etime,time,pcpu -p <pid>
```

Hundreds of files per minute and >100% CPU means it is compiling. Every *new prefill shape* pays a
smaller version of the same toll (hence 137 s / 282 s / 141 s, then 8.7 s once warm).

Watcher also needs most of its sub-checks disabled or the **erisc build fails** with
`section '.text' is not within region 'ERISC_APP_KERNEL_CODE'`:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 \
TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 \
TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1 \
TT_METAL_WATCHER_DISABLE_ETH_LINK_STATUS=1 \
pytest -svq models/demos/blackhole/qwen36/tests/unit/test_dflash_hang_repro.py
```

Output lands in `generated/watcher/watcher.log`. **Copy it out immediately** — the next run deletes it.

### 5.3 Catch the death itself

The 2026-09-15 run learned little because the process vanished without a word. Close that gap first:

```python
import faulthandler; faulthandler.enable()          # SIGSEGV/SIGABRT → Python stack
faulthandler.dump_traceback_later(120, repeat=True) # periodic stacks, distinguishes hang from death
```

and run under a supervisor that records the **exit signal**:

```bash
pytest ... ; echo "rc=$?"     # 137 = SIGKILL (OOM), 139 = SIGSEGV, 134 = SIGABRT
dmesg -T | tail -50           # if readable on the new box — it was not on the old one
```

Distinguishing SIGKILL-by-OOM from SIGSEGV-in-Metal splits the search space in half and costs one
run. **Do this before more Watcher runs.**

### 5.4 Then bisect what grows with generation count

Since generation count is the only surviving correlate, instrument what actually grows:

* Drafter KV history length per step (`_ctx_len`, `hist_len`) — does it exceed a capacity bound?
* `ttnn.dump_device_memory_state` on the drafter's device once per step; watch L1 and DRAM
  high-water marks climb toward a cliff at the failing step.
* Run with the drafter in **fixed-capacity** mode (`ctx_capacity`, see
  [`tests/unit/test_drafter_fixed_capacity.py`](tests/unit/test_drafter_fixed_capacity.py)) vs dynamic.
  If fixed capacity survives ≥24 tokens, the fault is in the growing-concat path and you have it.

### 5.5 Device hygiene

* A process killed while a trace is parked **wedges the device**. Symptom on the next run:
  `RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: … Stuck at 0x…`.
* Recovery is `tt-smi -r`. **This is shared hardware — confirm with the owner before resetting.**
* Six resets went into this work. Prefer letting a run finish over killing it.
* **Unexplained, worth a look:** in every Watcher round, devices 0–4 answer within milliseconds of
  each other while **devices 5, 6 and 7 each take ~33 s**. Both ETH-heartbeat failures this work hit
  named a single ASIC. May be nothing; may be the whole story.

---

## 6. The other open problem: is 22.40 tok/s real?

Acceptance is **7.000** on `"The capital of France is"` and **~2.0–2.3** on ordinary prose. Since
throughput ≈ `acceptance / step_time`, that gap is the difference between 1.25× production and a
regression.

* [`tests/reference/test_dflash_prompt_length.py`](tests/reference/test_dflash_prompt_length.py)
  measures acceptance vs prompt length (gated: `DFLASH_PROBE_PROMPT_LEN=1`). Its ordering flaw is
  **fixed** — it now runs the whole eager sweep, captures once, then the traced sweep.
* [`tests/reference/test_dflash_prose_throughput.py`](tests/reference/test_dflash_prose_throughput.py)
  is committed **UNTRUSTWORTHY and says so**: its control disagrees with
  `test_dflash_traced_throughput.py` by 4× on the control's own prompt (1.808 vs 7.000). **Do not
  quote its numbers.** Fix its control first — until the control reproduces 7.000/22 tok/s, nothing
  it reports means anything.

**A live clue, recorded but not chased:** two runs at prompt length 64 (both traced) reported
acceptance **1.917 vs 2.091 while emitting identical tokens**. Under greedy decoding the tokens are
fixed, so a differing accept/verify split means the drafter entered the second run with **different
state** — i.e. state leaks across generations on a reused drafter object. That would also
contaminate any A/B that reuses a drafter, which is most of the tests here.

---

## 7. Dead ends — measured, not assumed. Do not redo these.

| Attempt | Result |
|---|---|
| `block_size` 8 and 32 | 16 is a genuine optimum; both directions lose (`7c4aa0d2fbb`) |
| `ANCHOR` 128 → 64 | Identical tokens, **−10 % acceptance**. A wash (`728ef5f5562`) |
| Tracing the drafter | Built and measured at 1.00× / 0.91× / 0.71× across three configs. Staging and the KV commit — neither capturable — cost what traced dispatch saves (`test_dflash_drafter_trace.py`) |
| Fused KV commit (20 ops → 4) | **0.71×** end-to-end; strided scatter. Reverted |
| Verify-side argmax on device | 22.19 → **4.48 tok/s**. Reverted |
| Drafter op-count trimming | Below the noise floor (`8f57ca4711f`) |
| "Lever B" (device-side taps) | Already implemented (`device_taps=True`) |
| SDPA dilution fix | Zero acceptance |
| DRAM for the RoPE slice | No effect on the failure |
| Sharing `model.tt_ccl` with the drafter as the cause of the broken control | ✗ — structurally suspect (one round-robin semaphore pool shared with the target's *traced* collectives) but giving the drafter a fresh `TT_CCL` leaves the control at 2.611, unchanged. Not the cause of either bug (2026-09-15) |
| Narrow final block as the cause of the hang | ✗ — `test_drafter_block_width.py` runs the drafter standalone at widths 16/8/4/3/2/1 and reproduces the exact hung state (109 rows history, `new_ctx=1`, `q_len=2`) in **0.20 s**. The width is fine; what hangs is compiling it under a parked trace (2026-09-15) |

---

## 8. ttnn landmines found the hard way

* **`ttnn.Tensor.__eq__` is an elementwise device op.** `if tensor in some_list` silently dispatches
  a `binary_ng` and fails with `is_allocated()`. Never use membership tests on tensors.
* **`argmax` is ~45× slower in TILE than ROW_MAJOR.** A device-argmax experiment measured 0.26×
  (i.e. 4× *slower*) in TILE and 4.50× in ROW_MAJOR. A real win was nearly discarded over a layout.
* **`ttnn.slice` aliases on full-width slices but *copies* on partial ones.** A "fused"
  `hist_k = ttnn.slice(...)` made a later in-place commit write to a temporary, and the KV history
  silently stopped growing. No error.
* **ttnn SDPA attends tile padding.** This made a legacy path the *broken* baseline and produced a
  0.913 PCC "regression" that was actually a fix. Gate against an fp32 host oracle, never against
  the previous implementation.
* **Host I/O is illegal inside a trace capture**, and capture bakes in shapes, buffer addresses
  *and* op attributes (`slice_start` / `slice_end`). An offset that moves with the accept count
  cannot live inside the capture — see `commit_staged_context`.
* **`reset()` reallocates the anchor's GDN snapshot**, which Metal flags as *"Allocating device
  buffers is potentially unsafe due to the existence of an active trace"*. Calling it between the
  capture and a measured run produced fluent-looking garbage.
* **`_assert_output_quality` only detects repetition.** It passed multilingual token soup. Coherence
  is a human judgement here; read the text.
* **Exchange rate for op-count work:** ~0.69 ms wall per dispatch. The drafter is only ~9 %
  device-utilised, so it is dispatch-bound, not compute-bound.
* **Beware coincidence.** One A/B "confirmed" a prediction to the millisecond on pair 1; pairs 2–5
  had sd 11.4 ms. Always run ≥5 pairs.

---

## 9. File map

**Core**

| Path | What |
|---|---|
| [`reference/dflash/generate.py`](reference/dflash/generate.py) | the speculative loop; `pending_taps` lives here |
| [`reference/dflash/targets.py`](reference/dflash/targets.py) | `TtTarget`: anchoring, `max_block`, trace enable/capture |
| [`reference/dflash/drafters.py`](reference/dflash/drafters.py) | `TtDrafter` / `HostDrafter` adapters |
| [`tt/dflash/drafter.py`](tt/dflash/drafter.py) | `TtDFlashDrafter`: fixed capacity, staging, KV commit |
| [`demo/dflash_demo.py`](demo/dflash_demo.py) | the end-to-end demo (**currently failing**) |

**Tests worth knowing**

| Path | What |
|---|---|
| `tests/reference/test_dflash_traced_throughput.py` | **the known-good reference.** 7.000 / ~22 tok/s |
| `tests/unit/test_dflash_hang_repro.py` | the bisection for §4 |
| `tests/reference/test_dflash_prompt_length.py` | acceptance vs prompt length (`DFLASH_PROBE_PROMPT_LEN=1`) |
| `tests/reference/test_dflash_prose_throughput.py` | ⚠️ control is broken; do not quote |
| `tests/unit/test_drafter_fixed_capacity.py` | fixed-capacity drafter mode |
| `tests/unit/test_drafter_block_width.py` | **new (§0).** drafter standalone at block widths 16→1; proves the narrow tail is not the fault |
| `tests/reference/test_dflash_generation_repeat.py` | **new (§0).** acceptance vs generation index, eager vs traced; isolates the leak |
| `tests/perf/*_sweep.py` | matmul / dtype / sharding sweeps behind the tuning |

---

## 10. If you only do three things

1. **Reproduce §4 on the new hardware**, with `faulthandler` on and the exit signal recorded (§5.3).
   Knowing whether it is SIGKILL or SIGSEGV is worth more than another Watcher run.
2. **Fix the control in `test_dflash_prose_throughput.py`** until it reproduces 7.000 on
   `"The capital of France is"`, then measure prose throughput (§6). This decides whether the whole
   result stands.
3. **Chase the drafter state leak** (§6, last paragraph) — identical tokens with different
   acceptance across two runs. It is cheap to test (fresh drafter per run) and it contaminates
   every A/B in this suite if real.
