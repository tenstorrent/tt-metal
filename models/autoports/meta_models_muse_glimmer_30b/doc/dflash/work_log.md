# DFlash speculative decoding — work log

Goal: implement the DFlash drafter Muse-Glimmer-30B ships with, and win decode
t/s/u at batch 1.  The model card advertises **3.1× on an RTX 5090** (74.9 →
233.4 tok/s) from this feature, and no stage of the original bring-up ported it.

Status: **7.3× faster than the first working run; ahead of non-speculative decode
on the highest-acceptance prompt and behind it on the rest.**  26.2 t/s/u against a
42.9 t/s/u baseline at ISL 67 / OSL 128, and 25.4–44.2 t/s/u across six prompts
(mean 32.6, up from 22.0) where one prompt reaches **1.03×**.

Two independent things cap it, both now measured rather than assumed:

* **one target forward costs 55–67 ms whatever it computes** (F13), because the
  prefill path is host-dispatch bound — so the verify cannot be made cheaper by
  verifying fewer rows, and it alone exceeds the break-even budget;
* **acceptance is capped by the target port, not the drafter** (F15) — the real HF
  drafter scores no better against this target — so the CPU oracle's 4.41
  accepted/forward is not reachable here.

The *shipped* non-speculative path is unaffected: measured against the pristine
pre-DFlash commit it is 42.92 vs 42.96 t/s/u, a -0.10 % difference inside the
run-to-run spread, with identical output tokens (F17).  "DFlash is slower" is a
statement about the feature, not about the release path.

F16 records three redesigns that would each have closed the gap and did not
survive measurement.

---

## What DFlash actually is, on this checkpoint

The drafter is a *separate published artifact*,
[`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)
— public, ungated, 5.11 GB, `MuseGlimmerAssistantModel`.  The reference
implementation ships in the installed `transformers` 5.15.0, so the port is
graded against real HF math rather than a re-implementation:

| | |
|---|---|
| drafter | 5 layers, all `sliding_attention`, window 2048 |
| dims | hidden 6656, FFN 19968, 32 Q / 8 KV heads, head_dim 128 |
| block | `block_size` 16 → 1 anchor + 15 drafted tokens |
| context tap | target hidden states at layers `[1, 13, 25, 37, 49]`, concatenated → 33280 |
| cache | `DFlashCache` — context K/V persists, the diffusion window is appended then cropped |
| driver | `DFlashTokenCandidateGenerator`, selected by `speculation_type="dflash"` |

**The drafter has no `embed_tokens` and no `lm_head`.**  Its 58 tensors are 5
decoder layers + `encoder.fc` + `encoder.output_norm_enc` + `norm`.  Confirmed
two ways: read from the safetensors header, and by arithmetic — 5 × 467 M +
221.5 M ≈ 2.56 B params × 2 bytes = **5,111,976,608 bytes**, which is the file
size exactly.  Input embeddings come from the *target's* table via a plain
lookup; candidate logits come from the *target's* `lm_head`.

---

## Four ways this differs from the target decoder

Each of these produces a silent accuracy loss rather than a crash if ported by
analogy from the existing `functional_decoder.py`:

1. **Plain RMSNorm, not centered.**  The target's text layers use
   `MuseGlimmerTextCenteredRMSNorm` (`x * (1 + w)`), which the target port
   pre-folds the `+1` into at setup.  The drafter uses ordinary `x * w`.
   Reusing `_MuseGlimmerNorm` here adds a spurious `+1` to every weight.

2. **QKV cannot be fused.**  Q is projected from the 16-token window; K/V are
   projected from `concat(context, window)`.  The target port fuses QKV into one
   matmul — correct there, wrong here.

3. **Attention is bidirectional.**  `is_causal = False`; the mask is
   `bidirectional_mask_function AND sliding_window_overlay(w)`, and since the
   former is unconditionally true the whole condition collapses to
   `kv_idx > q_idx - 2048` — **a lower bound only, no causal upper bound**.  The
   16 window positions see each other in both directions.  A causal mask still
   runs and still emits plausible tokens; it just lowers the acceptance rate,
   which is invisible unless measured.

4. **The context half of K/V is not re-normalised per layer.**  Each layer
   applies `input_layernorm` to the window only; the context entering K/V is the
   encoder output, shared unchanged by all five layers.

---

## Findings

### F1 — The speedup is real: 4.80× fewer target forwards, measured

`tests/dflash_cpu_oracle.py` runs the genuine end-to-end loop on CPU with real
target + real drafter weights, and counts target forwards via a hook.

| | baseline greedy | DFlash |
|---|---|---|
| tokens | 48 | 48 |
| target forwards | 48 | **10** |
| wall clock | 68.9 s | **16.6 s** |

**5.33 accepted tokens per target forward** (ceiling 16), **4.80×** fewer
forwards, **4.14×** wall-clock. Above the card's 3.1× claim — CPU flatters the
drafter, since it is 11.7× smaller than the target and the ratio is
bandwidth-bound, but the forward-reduction figure is hardware-independent and is
the one that transfers.

### F2 — DFlash is *not* output-lossless in bf16, and that is the target's fault

The oracle found DFlash diverging from plain greedy at token 39 of 48.  That
should be impossible: a token is only accepted when it equals the target's own
argmax, so the accepted sequence should *be* the greedy sequence.

`tests/dflash_divergence_probe.py` settles it **without the drafter involved at
all** — take the greedy sequence, re-score it with one wide teacher-forced
forward, compare per-position argmax:

```
positions compared: 48
argmax mismatches between incremental and teacher-forced: 1
  idx 39: incremental 1574 vs wide-forward 20694 | top2 gap 0.06250 | incr rank 1
VERDICT (B): the target model's own argmax depends on forward width in bf16.
```

A top-2 gap of **0.0625** is one bf16 ulp at that magnitude.  Baseline greedy
computes logits with `query_len == 1`; DFlash verifies with `query_len == 16`;
those are different reduction orders through identical weights, so near-tied
logits argmax differently.

**Consequence for the port:** "token-identical to greedy" is not an achievable
correctness gate and must not be used as one.  Gate instead on (a) accepted
tokens being the target's argmax from the verify forward *by construction*,
(b) divergence rate staying at the bf16 near-tie noise floor, (c) eval parity.
The card's "producing identical output quality" is true only in exact arithmetic.

### F3 — `to_empty()` + `assign=True` silently produced a garbage RoPE table

Cost roughly an hour, and is worth recording because the failure is invisible.

The first reference harness built the model on `meta`, then `to_empty(device=
"cpu")` and `load_state_dict(..., assign=True)`.
`MuseGlimmerAssistantRotaryEmbedding.inv_freq` is a **non-persistent buffer**,
so it is absent from the state dict, `to_empty` gave it uninitialised memory,
and nothing ever filled it.  HF's `inv_freq` came out as
`[0.0, 1.9e-19, 0.0, 4.7e-18, 0.0, 0.0]` — garbage — instead of
`[1.0, 0.815, 0.664, ...]`.

The model ran without complaint and produced plausible activations, so the
goldens were quietly wrong and the port was graded against noise: end-to-end PCC
0.73–0.92, *degrading with context length*, which looks exactly like a real
attention bug.

Localised by reimplementing the layer in **pure torch** first: it failed
identically (0.92 vs the port's 0.916), which proved the fault was in the shared
understanding rather than in any ttnn op.  Hooking HF's layer-0 internals then
showed everything matching to ≥0.999996 up to `q_norm`, and the RoPE tables
disagreeing at PCC 0.011 with `max|Δ| = 2.0`.

Fixed by loading through `from_pretrained`, plus `_assert_rope_initialised()`,
which checks `inv_freq` against the analytic default table so this can never
recur silently.

**After the fix, the pure-torch reimplementation matches the HF drafter at
0.99994 on every layer** — confirming the bidirectional mask, plain RMSNorm,
unfused QKV, context-as-KV, QK-norm and RoPE slicing are all right.

### F3b — The golden harness silently disabled the sliding window

Same shape as F3, found on device.  End-to-end PCC came back 0.9954 / 0.9966 /
0.9970 / 0.9977 for context 1 / 16 / 128 / 2048 and **0.9288 for 4096** — the one
case exceeding the 2048 window.

`reference_forward` called the model with `use_cache=False` and no
`attention_mask`.  With both absent, `create_bidirectional_sliding_window_mask`
takes its `allow_is_bidirectional_skip` path and returns **`None`**, so the
reference ran with no window whatsoever.  Below 2048 that is unobservable; above
it, the golden rewards the wrong implementation — a CPU reimplementation with the
window *disabled* scored **0.99997** against that golden, while the correct
windowed port scored 0.9294.

The real driver passes an explicit `attention_mask` *and* a `DFlashCache`, so the
window does apply in production.  The harness now does the same.  The cache is
what makes the mask the right **size**: it is constructed inside `forward` before
K/V are appended, so the base `kv_length` is only `block_size`, and
`DFlashCache.get_mask_sizes` adds `_previous_number_of_accepted_tokens` back.

**After the fix, device PCC is 13/13:**

| context | encoder | drafter end-to-end |
|---|---|---|
| 1 | 0.99960 | 0.99541 |
| 16 | 0.99956 | 0.99659 |
| 128 | 0.99952 | 0.99701 |
| 2048 | 0.99914 | 0.99801 |
| 4096 | 0.99914 | 0.99803 |

End-to-end PCC now *rises* with context instead of falling, which is the
signature of a correct mask.

**Generalisable lesson, and the one worth carrying to other ports:** both F3 and
F3b were references that ran, produced plausible activations, and graded the port
against something other than production behaviour.  A golden harness must mirror
how the model is *actually driven*, not merely call it in a way that does not
raise.

### F4 — Device contention is not handled by anything in this repo

Device PCC was blocked for ~4.5 h by another job's `VLLM::EngineCore` (from
`/home/ttuser/dev/laguna/.../poolside_laguna_xs_2_1`, orphaned to init) holds
`CHIP_IN_USE_0_PCIe`.  tt-metal blocks on the lock with a bare warning and no
timeout, so a test run simply hangs until the 300 s pytest timeout fires and
reports as a test failure rather than as "hardware busy".

Note also that `pgrep -x 'VLLM::EngineCore'` cannot match it — the name exceeds
the 15-char comm limit — and `pgrep -f` matches the polling script's own argv.
Both traps were hit while writing the waiter, and `pkill -f` later killed its own
invoking shell for the same reason.

A waiter that fired during the other job's *teardown* then wedged the device:
the first test hit the 300 s pytest timeout inside `open_mesh_device` and every
subsequent test failed with `RuntimeError: Query mappings failed on device 0`.
It cleared on its own once the chips were fully released — but note that
pytest reports "hardware busy" and "hardware wedged" as ordinary test failures,
which is how a contended run gets misread as a broken port.

---

### F5 — Projected device win: ~4.5×, from measured inputs

Not yet measured on device (F4).  Stated as a projection so it can be checked
against the real number rather than quietly forgotten:

| term | value | source |
|---|---|---|
| baseline decode | 23.03 ms/token, 43.4 t/s/u | TTI sweep, ISL 128 batch 1 |
| accepted tokens / target forward | 5.33 | CPU oracle (F1) |
| drafter : target parameter ratio | 2.56 B : 30 B ≈ 8.5 % | checkpoint sizes |

Decode at batch 1 is weight-bandwidth bound, so a 16-position verify forward
costs about the same as a 1-position decode step, and the drafter forward scales
roughly with its parameter share:

```
per iteration ≈ 23 ms (verify) + ~2-4 ms (draft)  ≈ 25-27 ms
                → 5.33 tokens                     ≈ 4.7-5.1 ms/token
                → ~195-210 t/s/u                  ≈ 4.5x over 43.4
```

That would put Muse-Glimmer-30B on 4 Blackhole dies at roughly the RTX 5090's
DFlash figure (233.4 tok/s) rather than a third of it.  **The dominant
uncertainty is the acceptance rate**, which is workload-dependent — 5.33 came
from one coding prompt, and the sweep should measure it per ISL rather than
assume it.

### F6 — The drafter's 1.09 s was ttnn **program compilation**, not arithmetic

The first working run blamed the drafter on replicated weights, `repeat_interleave`,
DRAM attention and a host round trip.  All four are real, and none of them is the
cost.  `tests/dflash_drafter_bench.py` loads only the 5.11 GB drafter — no 30 B
target, so an attempt takes a minute — and drives it with the shape sequence a real
generation produces:

| what the drafter ran | ms per call |
|---|---|
| real context lengths 67, 3, 14, 4, 11, … (shape changes every call) | **1201.7** |
| the identical work at one constant shape, after call 1 | **14.3** |

**82×**, on the same mesh, weights and inputs.  Per-op attribution across a churning
run: `from_torch` 21.3 %, `matmul` 17.0 %, `linear` 12.5 %, `reshape` 10.7 %,
`permute` 10.7 %, `softmax` 9.6 % — and ~3 ms *per op* against the ~19 µs/op the
30 B target achieves.  A 3 ms `reshape` is not arithmetic; every distinct shape a
ttnn op sees costs a compilation, and the incremental drafting path produces a new
context length **and** a new cache length on every iteration, so it recompiles for
as long as generation continues.

**The confound that makes this easy to mismeasure.** ttnn caches compiled kernels
**on disk** (`~/.cache/tt-metal-cache`), across processes.  After earlier runs had
warmed it, the same incremental path timed **44.5 ms** per call over 48 tokens —
and then **671 ms** per call over 128 tokens, because the longer run reaches shapes
nothing had compiled yet.  Any drafter timing is therefore meaningless without
stating its generation length *and* its cache state, and a "fix" can look like a
30× win purely by running second.

### F7 — Bucketing the context is free, and the run that said otherwise was too short

`forward_padded` pads the accumulated prefix to one of seven widths
(`CONTEXT_BUCKETS`), bounding a whole generation to seven programs. Measured over
128 tokens / 41 blocks, ISL 67:

| drafting path | ms/drafter call | accepted/forward | mean matches | t/s/u |
|---|---|---|---|---|
| incremental | 671 | 3.05 | 2.10 | 3.93 |
| **padded** | **120** | 3.05 | 2.10 | **12.91** |

Acceptance is **identical**, so the wider attention reduction padding introduces —
real, at the 1e-3 level — costs no acceptance at all.

A 48-token run concluded the opposite (incremental 4.00 vs padded 2.82
accepted/forward) and was wrong.  Over 11 blocks on one prompt, acceptance spans
**2.82 – 4.00 across mathematically equivalent configurations**: incremental 4.00,
padded-exact 3.69, bf16 weights 3.00, padded 2.82.  The metric only discriminates
anything at ~40 blocks.  **Do not rank drafting configurations on a 48-token run**,
and note this cuts against F1's 5.33 as a target too — that also came from one
prompt.

`--drafting padded-exact` is retained as the control that separates padding from the
rest of the rewrite: same accumulated-prefix path, exact width.

Padding does need one thing the sliding window will not do for you.  A pad row
parked at position 0 satisfies `kv > q - 2048` for every query below position 2033,
so the window bound admits it as an ordinary key, and because drafter attention is
bidirectional it then corrupts the real slots rather than being ignored the way
padding is on the target's causal path.  `bidirectional_sliding_mask(kv_valid=...)`
blocks them explicitly; `test_mask_blocks_padding_the_window_bound_would_admit`
asserts the window bound alone does not.

### F8 — The drafter was running on ttnn's default math fidelity, unexamined

No `compute_kernel_config` was passed anywhere in the drafter, so it inherited a
default rather than a decision.  The target decoder deliberately runs `LoFi` with
`fp32_dest_acc_en=False`, but the two are graded on different things: the target on
output quality, the drafter on **argmax agreement with the target over a 202k
vocabulary**, where a near-tie that flips costs a whole accepted token.  The drafter
is also ~8.5 % of the target's parameters, so fidelity is cheap here.

`HiFi4` + `fp32_dest_acc_en=True` (`drafter_compute_kernel_config`), 13/13 still
passing:

| case | before | after |
|---|---|---|
| encoder ctx 1 | 0.99960 | **0.999991** |
| encoder ctx 128 | 0.99952 | **0.999899** |
| encoder ctx 4096 | 0.99914 | **0.999902** |
| end-to-end ctx 128 | 0.99701 | **0.998606** |
| end-to-end ctx 4096 | 0.99803 | **0.999098** |

Encoder error falls ~5–10×.  It did **not** move measured acceptance (3.05 either
way at 41 blocks), so on this evidence drafter fidelity is not what limits
acceptance — worth knowing before anyone spends more effort there.

### F9 — What now blocks the win: verify is O(prefix) through the prefill path

Per iteration at ISL 67 / OSL 128, padded drafting:

```
draft   120 ms   (drafter forward, noise embed, LM head, 202k logits gather, host argmax)
verify  103 ms   (one target forward)
       ------
        223 ms  ->  3.12 tokens/iteration  ->  71.5 ms/token  ->  12.91 t/s/u
baseline                                       23.31 ms/token      42.90 t/s/u
```

To break even at 3.12 tokens/iteration the whole iteration must fit in
**3.12 × 23.31 = 72 ms**.  Verify alone is 103 ms and *grows with the prefix*, so no
amount of drafter work can reach it: the drafter could cost zero and DFlash would
still lose.

The verify forward re-forwards the **whole prefix from position 0** every iteration.
That is not an algorithmic requirement — K/V for those positions is already in the
paged cache — it is two mechanical blockers (recorded in the
`first working end-to-end run` commit): `paged_fill_cache` writes from virtual block
0 of the page table it is handed, so a multi-token prefill must start on a 64-token
boundary; and a sliding-window layer refuses any `start_pos > 0` without the previous
call's K/V tail.  `chunked_scaled_dot_product_attention` does not rescue the sliding
layers — it takes no `sliding_window_size`, and its `chunk_start_idx` must be a
multiple of both the q- and k-chunk size.

Projected, with a 16-row verify costing about one decode step (~25–35 ms, weight
bandwidth bound at batch 1) and the drafter at its constant-shape floor:

```
draft ~20 ms + verify ~30 ms = 50 ms  ->  3.12 tokens  ->  16 ms/token  ->  ~62 t/s/u  ~1.5x
                                          6.3 tokens   ->   8 ms/token  ->  ~125 t/s/u ~2.9x   (if acceptance reaches F1's 5.33)
```

So the remaining work is, in order: (1) an O(block) verify — page-table slicing to
a 64-aligned restart plus sliding-tail threading, bounded by re-forwarding at most
63 committed rows; (2) the drafter's constant-shape floor, which needs the
*incremental* cache made shape-stable via in-place writes at fixed capacity so it is
both O(delta) and one program; (3) acceptance, which at 2.10 matches per block of 15
is the largest untouched multiplier and is not explained by drafter fidelity (F8).

### F10 — Acceptance, measured properly: 2.26 matches per block of 15

The number that decides whether speculation can win at all, pooled over **234 blocks
across 6 prompts** at OSL 128 (`--prompts-file`, one model load):

| prompt | blocks | matches/block | accepted/forward | t/s/u |
|---|---|---|---|---|
| merge two sorted lists | 41 | 2.10 | 3.05 | 19.92 |
| why the sky is blue | 42 | 2.02 | 2.98 | 20.47 |
| SQL second-highest salary | 29 | **3.38** | 4.27 | 30.21 |
| optimistic vs pessimistic locking | 45 | 1.82 | 2.78 | 18.75 |
| `def quicksort(arr):` | 46 | **1.76** | 2.72 | 15.01 |
| translate to French | 31 | 3.10 | 4.00 | 27.85 |
| **pooled** | **234** | **2.26** | **3.26** | |

Two things follow.

**F1's 5.33 accepted/forward should not be used as a planning number.**  It came from
one CPU prompt over ~10 blocks; the device pools to 3.26 over 234.  The per-prompt
range here is 2.72–4.27, which brackets 5.33 nowhere near its middle, and t/s/u tracks
acceptance almost linearly (15.0 at 1.76 matches, 30.2 at 3.38) — so acceptance, not
any single op, is the multiplier the whole feature rides on.

**The viability budget.**  At 3.26 tokens per iteration, break-even against
23.31 ms/token is `3.26 x 23.31 = 76.0 ms` per iteration, against 157.6 ms today
(F9).  Every remaining item is needed to get there, and the optimistic floor is:

```
verify (aligned, ~48 rows, cannot go below one target forward)   ~30 ms
drafter forward (sharded weights, fused SDPA, L1)                ~12
candidates (device argmax instead of a 202k host gather)          ~4
noise embed                                                       ~1
taps -> host                                                      ~6
                                                                 ----
                                                                 ~53 ms  ->  16.4 ms/token  ->  ~61 t/s/u  ~1.4x
```

So DFlash *can* win here, but there is no single change that does it: the win needs
the verify forward, the drafter forward, the candidate argmax and the tap readback all
fixed, and it is worth ~1.4x rather than the card's 3.1x at this acceptance rate.
Raising acceptance is worth more than any of them and is the least explored — and per
F8 it is not explained by drafter numerical fidelity.

### F11 — The aligned-restart verify: scaffolded, and failing two ways

Attempted, behind `--verify aligned` / `DFlashRunner(aligned_verify=True)`, and left
**off by default**.  Recorded in detail because the hard part turned out not to be
the part everyone expects, and the next attempt should not re-derive it.

**Both supposed blockers are already solved in the port.**  The
`first working end-to-end run` commit lists two reasons the verify starts at 0:
`paged_fill_cache` writing from virtual block 0, and sliding layers refusing
`start_pos > 0` without a K/V tail.  Neither needs new machinery:

* `FunctionalDecoder._chunk_page_table` already shifts the page-table row by
  `start_pos / block_size` and enforces the alignment.
* A sliding tail is a contiguous run of K/V rows, so **the tail for an earlier
  position is a prefix of the tail already held** — exactly what
  `trim_sliding_tails` produces.  The commit's objection, that `prefill_forward`
  emits its tail at the call's *end* while the next verify restarts earlier, is
  therefore not an obstacle: trim it back.
* The chunked-SDPA offset constraint is handled too — `chunked_q` is halved until it
  divides `start_pos`, and q/k chunk sizes are set together.

**Failure 1: it is slower.**  39 of the target's 52 layers are `sliding_attention`,
so a per-iteration trim is **78 device slices** plus 39 tail concats, and verify went
**106.8 → 157 ms** per iteration: the bookkeeping costs more than the shorter forward
saves.  The cause is that `prefill_forward` *consumes* the tail it is handed, so a
fresh trim is needed every call.  The fix is a borrow mode that neither frees the
tail nor replaces it, after which the trim is only needed when `aligned_start`
actually advances — once per 64 tokens instead of every iteration.

**Failure 2: one committed token is wrong.**  Reproducible at OSL 128: the aligned
path emits two tokens (34302, 14166) where both from-0 and greedy emit one (26382),
at produced index 32 / absolute position 99, after which the streams re-align.  A
committed token is always the target's own argmax, so this is wrong *history* in the
verify forward, not a bad guess by the drafter — and note it is a far sharper gate
than acceptance rate, which barely moved (2.84 vs 3.05).

Ruled out by inspection, so as not to be re-checked: the chunked-SDPA offset;
tile-padding garbage entering the tail (always trimmed off, because
`aligned_start' <= anchor_pos + block`); and rejected candidates inside the tail
(everything below `aligned_start'` is a prefix token or an accepted candidate, and an
accepted candidate *is* the committed token, so its K/V is correct).  Not yet
checked: whether `sliding_kv_tail_len` means `min(window, start_pos)` rows ending at
`start_pos` in the same sense the trim produces, and the interaction with the
prompt prefill's own tile padding on the very first verify (prompt 67 → padded 96 →
trimmed to 64).

### F12 — Tooling traps that silently corrupt DFlash measurements

Four, each of which produced a plausible wrong number rather than an error:

1. **tt-metal resolves dispatch kernel *sources* relative to cwd.**  Running from a
   tt-metal checkout compiles that checkout's
   `tt_metal/impl/dispatch/kernels/*.cpp` against the *installed* build's headers.
   On this branch they disagree: `'COMPLETION_COUNTER_OFFSET' was not declared in
   this scope`, then a segfault inside the JIT builder.  `TT_METAL_HOME` does **not**
   fix it — only a neutral cwd does.
2. **`models` is a multi-root namespace package** and this venv's `ttnn-custom.pth`
   hard-codes a second checkout onto `sys.path`.  The dangerous outcome is not an
   `ImportError` but silently benchmarking a mixture of two branches, so the harness
   asserts its own import path against `$DFLASH_EXPECTED_ROOT`.
3. **The baseline needs a warm-up trial.**  One trial measured 32.5 t/s/u where two
   measured 38.41 then **42.91**, because a fresh generator's first decode pays trace
   capture and program-cache population.  Comparing warm DFlash against a cold
   baseline flatters DFlash by ~30 %; `--baseline-trials` keeps the fastest.
4. **`nohup … &` inside a job launcher reports the launcher's exit, not the job's.**
   A goldens run "exited 0" having written nothing.  Wait on log *content*.

Also: `reference_dflash.py` needs `/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv`
(transformers 5.15.0 with `muse_glimmer_assistant`); a bare `python` is
`.tenstorrent-venv` at 5.11.0, where the import cannot resolve.  That install came
from a wheel that no longer exists, backed up under `vendor/`, so nothing may be
pip-installed over it.  `tests/dflash_checkpoint.py` reads config + weights straight
from the snapshot so perf work does not depend on it at all.

### F13 — A prefill forward costs the same whatever it computes

Measured directly (`tests/dflash_verify_cost.py`, warm medians), which every
end-to-end comparison had been confounding:

| start_pos | 32 rows | 64 rows | 128 rows | 256 rows |
|---|---|---|---|---|
| 0 (plain SDPA) | 65.5 | 60.6 | 54.7 | 57.5 |
| 64 – 1024 (chunked SDPA over the paged cache) | 66–67 | 57–61 | — | — |

**Flat**, and 32 rows costs *more* than 128.  So the forward is host-dispatch bound,
not work bound, and three things follow at once.  Re-forwarding fewer committed rows
saves nothing, so the aligned restart's value is not the shorter forward.  The
chunked paged path costs the same as the plain one, so continuing at an offset is
free.  And at 64.2 ms one forward already exceeds the entire break-even budget of
2.72 tokens × 23.31 ms = 63.4 ms — no drafter or I/O work can win without changing
the forward itself.

Corollary worth keeping: the aligned verify's real benefit was never the forward, it
was running the 202k-wide LM head over 2 tiles instead of 7.  Device-side argmax got
most of that back on *both* paths.

### F14 — Nothing needed a sliding K/V tail

The `first working end-to-end run` commit recorded two reasons the verify had to
start at position 0, and neither binds.  `FunctionalDecoder._chunk_page_table`
already shifts the page-table row by `start_pos / block_size`, which is the whole of
the `paged_fill_cache` constraint.  And a sliding layer needs the previous call's K/V
tail only because `chunked_scaled_dot_product_attention` has no sliding-window mask —
but **while the sequence fits inside the 2048 window the mask is vacuous**: query `i`
attends `(i - w, i]`, and if the last query is below `w` the lower bound clips
nothing, so the layer *is* a full-attention layer over the chunk and the paged path is
exact.  `FunctionalDecoder.sliding_window_is_inert` says so and routes it.

An earlier version did thread tails via `trim_sliding_tails`.  It was both slower —
39 of the target's 52 layers are sliding, so trimming cost **78 device slices per
iteration** and took verify from 106.8 to 157 ms — and it appeared to produce a wrong
token, which was itself a mis-diagnosis: gating on token equality with greedy is
invalid (F2), and `tests/dflash_verify_probe.py` shows the aligned continuation
matches a from-zero prefill at 3 differences in 72 positions, **all** at near-ties
(gaps 0.0625–0.1875), none wide.

### F15 — Acceptance is capped by the *target* port, and the drafter is exonerated

The device accepts 2.72 tokens per target forward where the CPU oracle accepts 4.41
on the identical prompt and length.  `tests/dflash_acceptance_probe.py` drafts every
block twice from byte-identical inputs — once with the TTNN drafter, once with the
**real HF drafter on CPU** — and scores both against the same device target:

| | accepted per block |
|---|---|
| device drafter | 2.50 |
| HF drafter, same target | **2.57** |
| candidate agreement | 0.900 per position |

Swapping in the genuine drafter buys **nothing**.  So what limits acceptance is that
the port's target argmax differs from the model the drafter was trained against: a
faithful prediction gets marked wrong.  That closes four avenues at once, all of which
were tried and moved acceptance by ~0: drafter math fidelity (F8), drafter weight
dtype (BFP8 2.72 vs BF16 2.78, at 4× the drafter cost), the LM head's tanh softcap,
and the padded-context rewrite (F7).

It also means the card's 3.1× is not reachable by optimising DFlash.  Raising it
requires raising the *target's* numerical fidelity, which trades directly against the
decode speed DFlash is being compared to.

### F16 — Three redesigns that would have closed the gap, and why each failed

Recorded in full because each is the obvious next idea.

**Verify as one batched decode step.**  The strongest of the three.  A *traced* decode
step costs 23.3 ms against 64.5 ms for any prefill, and decode cost is independent of
how many rows are active (70.35 ms at 16 active against 72.12 at 1, measured), so the
16 verify positions should verify for the price of one decode step — exactly the
break-even gap.  It is **unsound**: decode rows do not reliably observe each other's
same-step K/V writes.  Two runs of the identical step, each from a freshly re-seeded
cache, disagreed at row 14, and the winner differed from the prefill reference by a
**0.59** logit gap — far outside the near-tie floor.  End to end it produced degenerate
output (one token repeated) whose acceptance *looked* excellent, 15/15 blocks, because
repetition is trivial to draft.

Two traps made it look correct first, and both are easy to repeat: running the prefill
reference *before* the decode step pre-warms the cache for exactly those positions, so
the decode never has to chain; and running the decode twice without re-seeding does
the same to the second run.  `tests/dflash_decode_verify_probe.py` now re-seeds before
every measurement.

**Tracing the verify forward.**  Flat cost (F13) means a *fixed-width* verify is free,
and fixed width with `start_pos = 0` is one static graph — so the whole verify should
capture as a single trace, worth ~15 ms/iteration.  It wedges: a live trace requires
that nothing else allocate device buffers (`"Allocating device buffers is unsafe due
to the existence of an active trace"`), and DFlash allocates on every iteration — the
drafter's activations, the context upload, the logits tiles.  The shipped decode trace
coexists with its loop only because that loop allocates *nothing*.  Doing this needs
the DFlash loop converted to persistent buffers throughout, which is a larger change
than the trace.  The attempt also wedged the board and needed `tt-smi -r`.

**GQA by grouping queries instead of copying K/V.**  Reshaping `[1, 32, block, d]` to
`[1, 8, 4*block, d]` serves 32 query heads from 8 K/V heads with no `repeat_interleave`
and is exactly equivalent on paper.  Wrong here: TILE layout pads the `-2` dimension to
32, so at `block_size` 16 every head owns a 32-row tile that is half padding, and a
reshape folding heads into the row axis crosses it.  15 of 25 drafter tests fail while
the encoder tests, which touch no head dimension, still pass.  It would be valid at
`block_size >= 32`.

### F17 — The DFlash commits do not regress non-speculative decode

Worth settling with a direct A/B rather than by assertion, because "DFlash made things
slower" is ambiguous between *the feature is slower than not using it* (true, and the
whole subject of F13/F15) and *merging this work slowed the shipped path* (not true).

`doc/dflash/bench/baseline_ab.py` measures plain greedy decode and TTFT.  It lives
outside the model tree deliberately, so the identical script runs against two
checkouts via `PYTHONPATH`, with the interpreter, the tt-metal build and the device
held fixed:

| | pristine `0dd37ce6ee3` | DFlash tip `b20c1de1858` |
|---|---|---|
| decode t/s/u, best of 3 | **42.96** | **42.92** |
| per-trial | 37.55 / 42.96 / 42.92 | 37.63 / 42.92 / 42.91 |
| TTFT, best of 3 | 60.0 ms | 62.4 ms |
| TTFT per-trial | 60.4 / 60.0 / **73.7** | 64.2 / 62.4 / 65.5 |
| first 16 tokens | — | **identical** |

Decode differs by **-0.10 %**, inside the run-to-run spread, and the generated tokens
are identical.  TTFT's 2.4 ms difference sits well inside the pristine arm's own
60.0-73.7 ms spread, so it is not resolvable at three trials; it is worth re-checking
if the sliding-window routing added in F14 is ever suspected, since that is the one
change touching the shipped prefill path.

Both arms show the same cold-first-trial pattern (37.6 then 42.9), which is the trace
capture and program-cache population F12 warns about -- another reason to report the
best of several trials rather than a single one.

**Note for anyone repeating this**: `git worktree add <relative-path>` resolves against
the *current* directory, so adding a worktree from inside another worktree nests it.
The first run of this A/B pointed `PYTHONPATH` at a path that did not exist, `models`
silently fell back to the shared checkout, and both arms measured the same tree.  The
harness prints the resolved package root for exactly that reason -- check it.

### F18 — A traced 32-row prefill costs one decode step. That is the whole remaining win

The single most useful number this project has produced.  Measured with the port's own
`prefill_trace_probe.py`, warm, 10 replays:

| rows | eager ms | **traced ms** | speedup |
|---|---|---|---|
| **32** | 66.77 | **24.48** | **2.73x** |
| 64 | 61.16 | 40.99 | 1.49x |
| 128 | 60.65 | 47.28 | 1.28x |

Two things fall out.

**A traced 32-row verify would cost one decode step** (24.48 against 23.3 ms), not the
45 ms previously assumed.  That earlier figure came from the 44.96 ms on record, which is
at **128 rows** -- the wrong shape to plan a 16-token verify around.  At 32 rows the
prefill path already dispatches the *decode* matmuls, the *decode* collectives and
sharded norms, because `_prefill_projection` branches on `rows == TILE_SIZE` and
forwards to `_decode_projection`.  So the kernels are not the problem at verify shapes;
host dispatch is, and tracing removes it.

**32 rows is a cliff, not a slope.**  The DRAM-sharded decode matmul asserts
`M == per_core_M` *and* `M == 1` -- exactly one tile row -- so 64 rows falls back to
mcast2d at roughly half the DRAM bandwidth, and the traced time jumps 24.5 -> 41.0 ms.
Any verify design that needs 33 rows loses most of the win.

That also explains F13's oddity that a 32-row forward measured *slower* than a 128-row
one eagerly (65.45 vs 54.74): at 32 rows the decode-grade kernels do less device work but
cost more host calls, and eagerly the host is the binding constraint.

**Confirmed by building it.**  A traced from-zero verify at `verify_width = 256` works
and is *correct* -- 0 token mismatches against greedy over 48 tokens, which validates the
whole capture/replay path including the tapped hidden states as trace outputs -- and it
is **not faster**: verify 64.2 -> 59.0 ms, with capture amortisation making the iteration
worse overall.  That is the row-count curve above doing exactly what it says.  The trace
is not the win; the *32-row* trace is.  `--trace-verify` therefore stays off by default.

**What it would take.**  The verify window must be exactly 32 rows, which means an
*aligned* window (`page_block_size = 32`, restart at `32*floor(anchor/32)`, draft
`31 - (anchor mod 32)` candidates -- 15.5 on average, i.e. today's 15 for free).  A
traced window at a varying `start_pos` needs either one trace per distinct start (only
~5 for OSL 128, each ~140 ms to capture, so this is viable) or the runtime-offset form:
`chunked_scaled_dot_product_attention` accepts a `chunk_start_idx_tensor` read from
device at replay, documented for exactly this.  Prefill RoPE would also need the
decode-style on-device gather, since it currently slices with host ints.

### F19 — Tracing the verify needs an allocation-free loop, and the loop is not one yet

The rule a live trace imposes is **lifetime-based**: the allocator's own comment says
buffers allocated while a trace exists must have "a lifetime that ends before the trace
is executed".  It is a warning, not a gate, so nothing fails loudly -- it corrupts.

That means the drafter's ~250 per-call intermediates are *fine* (created and freed
between blocking replays); only what survives a replay matters.  Three real faults were
found and fixed here, and a fourth is still open:

1. `tt_page_table` was baked into the trace and then freed at the end of `generate()` --
   a use-after-free inside a paged attention op, which is a **board wedge**, not a wrong
   number.  This project already has the identical finding recorded for a cloned KV cache.
2. `_verify_tokens` was reallocated per `generate()` while the trace still read the first
   buffer, so a second prompt would silently replay the first prompt's tokens.
3. `DFlashDrafter._forward_with_positions` never freed the last layer's hidden state
   (`final_norm(hidden)` returns a new tensor and the input is dropped on the floor).
4. **Open:** ~16 KB/iteration is still allocated and live across the replay, isolated to
   the replay-plus-argmax region.  It grows linearly and at ~176 KB/bank the run hangs --
   which is the allocation rule biting exactly as documented.

`DFlashRunner._dram_allocated()` and the drift check are the instrument that made all of
this visible: it names a leak in bytes at a labelled checkpoint instead of wedging the
board.  Anyone continuing this work should keep `alloc_drift_budget = 0` and fix (4)
before anything else; `--trace-verify` is off by default until then.

### F20 — Per-shard argmax is correct and slower

Avoiding the all-gather in `_argmax_rows` by reducing each vocab shard in place and
combining four (index, max) pairs on the host is exactly equivalent -- it produced
byte-identical tokens over 128 tokens -- and it is a clear **loss**: 26.67 -> 21.12
t/s/u, candidates 9.7 -> 10.5 ms, verify logits 4.9 -> 6.5 ms.

The premise was that the all-gather of a `32 x 202752` tile (~13 MB) dominated.  It does
not.  Trading one readback of one small tensor for **eight** small per-shard readbacks
costs more than the gather saves, which says this path is bound by per-transfer latency
rather than by bytes.  Kept as `_argmax_rows_sharded`; the only variant worth retrying
folds ids and maxima into one tensor per device, halving the readbacks to four.

The vocab padding is handled by *detection* rather than masking, which is worth keeping
whatever the outcome: only the last shard is padded (49984 real of 50688), and a padded
winner is identifiable from its index alone, needing no assumption about padded values.

## Artifacts

| file | what |
|---|---|
| `tests/reference_dflash.py` | HF reference + golden generator; asserts the 58-tensor contract, absence of `embed_tokens`/`lm_head`, pinned config, initialised RoPE |
| `tests/dflash_goldens.pt` | goldens at context 1 / 16 / 128 / 2048 / 4096 (4096 is the only one exceeding the window) |
| `tests/dflash_cpu_oracle.py` + `.json` | end-to-end CPU oracle: acceptance rate, forward reduction, losslessness check |
| `tests/dflash_divergence_probe.py` + `.json` | isolates F2 to target-model numerics |
| `tt/dflash_drafter.py` | the TTNN drafter, plus context/noise assembly helpers |
| `tt/dflash_accept.py` | the accept/reject rule, device-free |
| `tests/test_dflash_drafter.py` | PCC parity + mask-semantics unit tests |
| `tests/test_dflash_accept.py` | 71 tests, incl. 64 randomised blocks vs the HF rule |
| `tt/model.py` | `arm_hidden_state_taps()` / `take_hidden_state_taps()` |
| `tests/dflash_drafter_bench.py` | drafter-only perf harness: loads 5.11 GB not 30 B, replays real context-length sequences, `--breakdown` for per-op attribution, `--fixed-shape` to separate compilation from work |
| `tests/dflash_checkpoint.py` | drafter config + weights from the HF snapshot with **no** `transformers` import, so perf work runs where the architecture is not installed. Not a correctness reference |
| `tests/test_dflash_padded.py` | gates `forward_padded`: padding must not move the port away from HF (incl. a padded case past the 2048 window), plus the pad-blocking mask unit tests |
| `tests/conftest.py` | one session mesh shared by every module — two modules each defining `mesh_device` fail the second open with what looks like a hardware fault |

## Next

In impact order, with the arithmetic in F13 and F15:

1. **Make one target forward cheaper.**  It is 64.2 ms of a 104 ms iteration and, at
   2.72 tokens per iteration, exceeds the 63.4 ms break-even budget on its own.  It is
   host-dispatch bound, so the only mechanism is tracing — which needs the DFlash loop
   converted to persistent buffers first, because a live trace forbids concurrent
   device allocation (F16).  Worth ~15 ms.
2. **Persistent buffers for the drafter too.**  Its forward is 16.3 ms for 5 layers —
   3.3 ms/layer against the target's 1.15 — for the same reason, and its shapes are
   already static per bucket, so it is the easier of the two to trace.  Worth ~12 ms.
3. **Cheaper candidates and taps** — 9.6 ms goes on the LM head plus an all-gather of
   32x202752, and 8.5 ms on moving tapped hidden states host-ward and back.  A
   per-shard argmax avoids the gather; assembling the context on device avoids the
   round trip.  Worth ~12 ms together.

Together those are ~39 ms of a 104 ms iteration, which reaches break-even on the
median prompt and a clear win on the better half.  Beyond that the ceiling is
acceptance (F15), and that is a property of the target port's fidelity rather than
of DFlash.

Not worth pursuing on current evidence: drafter weight dtype, drafter math fidelity,
the LM head softcap, and the padded-context rewrite — all measured, all moved
acceptance by ~0 (F15).  And verify as a batched decode step is unsound, not merely
unfinished (F16).
