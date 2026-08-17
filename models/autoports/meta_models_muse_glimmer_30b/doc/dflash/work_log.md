# DFlash speculative decoding — work log

Goal: implement the DFlash drafter Muse-Glimmer-30B ships with, and win decode
t/s/u at batch 1.  The model card advertises **3.1× on an RTX 5090** (74.9 →
233.4 tok/s) from this feature, and no stage of the original bring-up ported it.

Status: **end to end on device and 3.3× faster than the first working run, but
still 0.30× of non-speculative decode.**  12.91 t/s/u against a 42.90 t/s/u
baseline at ISL 67 / OSL 128.  The drafter is no longer the bottleneck; the
verify forward is, and F9 gives the arithmetic and the remaining design.

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

### F10 — Tooling traps that silently corrupt DFlash measurements

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

In impact order, with the arithmetic in F9:

1. **O(block) verify.**  The only change that can flip DFlash to a win — verify is
   103 ms of a 72 ms break-even budget, so nothing else can compensate.  Needs a
   page-table slice to a 64-aligned restart plus sliding-tail threading (deep-copied,
   since `prefill_forward` consumes its tail and consecutive verifies restart at the
   same position).  Bounded by re-forwarding at most 63 committed rows.
2. **The drafter's constant-shape floor** — 14.3 ms measured, 120 ms today.  Wants
   the *incremental* cache made shape-stable through in-place writes at fixed
   capacity, so it is O(delta) *and* one program, rather than the bucketed
   whole-prefix recompute that trades O(bucket) work for shape stability.
3. **Acceptance**, at 2.10 matches per block of 15 — the largest untouched
   multiplier, and per F8 *not* explained by drafter numerical fidelity.  Measure
   over ≥40 blocks and several prompts (F7); the CPU oracle's 5.33 is one prompt and
   should not be treated as a target until reproduced.
4. Cheaper follow-ons, all inside the 120 ms draft figure: device-side argmax instead
   of gathering a 202k-wide logits tile to host each iteration, and skipping the
   mask entirely below position 2033, where it is uniformly permissive.

Not worth pursuing on current evidence: drafter weight dtype (bf16 measured *worse*
than BFP8 on acceptance, and both are within F7's noise band), and further drafter
math fidelity (F8 moved PCC 5–10× and acceptance not at all).
