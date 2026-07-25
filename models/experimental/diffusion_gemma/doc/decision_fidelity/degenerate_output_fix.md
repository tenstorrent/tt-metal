# Degenerate output: cause, fix, and the safety net (#48291)

**Status 2026-07-25: root cause identified and fixed. The served Gumbel default is back to `host`.**
Matched 4-seed A/B on device, one variable: `host` answered correctly **4/4**, `device` corrupted
**2/4**. A degenerate-canvas detector is added as defense in depth, calibrated on real blocks.

## 1. What the failure actually looks like

GPQA-Diamond doc_id=0 (`gpqa_thinking3072_sub40_20260723.samples.jsonl.gz`) produced 3256
characters and no reasoning at all. It opens

```
níníníenianíeniaenianíníní… ní1ní1_1111111… the the the1 \1111 the the the \ \ \ the
 \ \ \ \ \ \ \ \ \ \ … 1111111111111111111111111111111111111111111…
```

Three tokens do all the damage, and their ids matter:

| id | token | where it shows up |
| --- | --- | --- |
| 621 | `▁\` | the `\ \ \ \` run |
| 236770 | `1` | the 2000-character wall of `1` |
| 1 | `<eos>` | (termination, *not* part of the failure) |

Ids 621 and 236770 are **the two most frequent tokens of the HEALTHY blocks of this same prompt**
— it is a LaTeX-heavy physics question. So the canvas is not emitting noise; it is collapsing onto
the prompt's own most probable token. That is the signature of positions losing their independence,
not of a model that has forgotten the task.

## 2. Reproducing it required fixing the prompt contract first

The failing traces were produced in **thinking mode**, and `serving_smoke` renders through
`tokenize_prompt`, which could not emit the `<|think|>` turn at all until `--enable-thinking` was
wired in. Before that, the offline harness could not reproduce the regime it exists to diagnose.
With it, doc_id=0 replays at `prompt_len=157` — the exact length the server logged — so the offline
path is a valid stand-in.

## 3. The A/B that identifies the cause

`serving_smoke --upfront --enable-thinking`, 12 blocks, `--max-seq-len 4096`, reveal span 4096,
48-step cap, identical prompt and seed per pair. **One variable: `--gumbel-mode`.**

| seed | `host` (IID) | `device` (the shipped default) |
| --- | --- | --- |
| 0 | `… is $10^{-4}$ eV.  Answer: \boxed{C}` | `$\Gamma_1 \approx 6.558times 10^{-^{-}} \text{}$  $\_22 \  ..5 \times  100^{-88` |
| 1 | `… clearly resolved is $10^{-4}$ eV.  Answer: \boxed{C}` | `Answer: \boxed{C}10^{- } \^{-text}} \} \text{ ( \: $1.054 \times 10^{-25}…` |
| 2 | `… Answer: \boxed{C}` | `… Answer: \boxed{C}` (identical text) |
| 3 | `… Answer: \boxed{C}` | `… Answer: \boxed{C}` |

**host 0/4 corrupted, device 2/4 corrupted.**

The corruption is not random noise, it is token-level duplication and dropout:
`6.558times` (the `\` between them lost), `10^{-^{-}}` (the superscript group repeated),
`100^{-88` (digits doubled). Neighbouring canvas positions are producing the same or overlapping
tokens — exactly what correlated per-position noise does.

## 4. Why `device` does this

`sample_gumbel_noise_with_permuted_vocab` keeps the vocab axis off `ttnn.rand`'s innermost axis by
collapsing every other axis into one trailing axis. For the production noise shape
`(1, 1, 256, vocab)` that trailing axis **is the 256 canvas positions**, so the draw becomes
`ttnn.rand((vocab, 256))` with positions on the width axis — and `ttnn.rand`'s width axis is not
independent:

* only 24 of every 32 column streams are distinct (columns `c` and `c-24` are byte-identical for
  `c % 32 >= 24`), so 64 of 256 positions carry a **copy** of another position's noise;
* the remaining columns stay correlated in value, so even the non-duplicate positions agree far
  more often than IID allows: 119/256 distinct flat-logit winners, against 255/256 for host.

Full measurement, root cause, and the two workarounds that do *not* fix it:
`gumbel_position_correlation.md`. Upstream regression:
`tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py` (xfail(strict)).

## 5. The fix

`DEFAULT_VLLM_GUMBEL_MODE` and the launcher default go back to **`host`**. `device` remains
selectable via `DG_VLLM_GUMBEL_MODE=device` for throughput work where the generated text is not
the product — it does buy the ~313 ms/step host RNG and the ~256 MiB/step replicated PCIe copy
(~1.8x denoise/step) recorded in `upfront_gumbel_overlap_devicemode_20260724.md`, which is now
banner-marked as superseded on the default question only. The throughput numbers there stand.

## 5b. The ttnn.rand fix, and the device default restored (2026-07-25, later)

`host` met the correctness bar but not the throughput bar (~36.3 vs ~53.6 tokens/block/s), so the
kernel defect itself was fixed rather than routed around.

**What the defect actually was.** Mapping one 32x32 `ttnn.rand` tile element by element: 94 distinct
values in 1024 slots, the PRNG per-lane (32/32 distinct inside one 32-lane SFPU vector) but only 20
of a tile's 32 vector draws distinct, with the exact relation

    (face f, vector 2k)  ==  (face f+1, vector 2k+1)

which in tile coordinates is "column c is byte-identical to column c-24 for c % 32 >= 24". So
element `(read t, lane i)` carries `stream[t + i]`: **one sliding window that advances about one
element per read while all 32 lanes read overlapping positions.**

**The fix.** `ckernel_sfpu_rand.h` (Blackhole) now consumes several PRNG values per stored element,
so the window moves past its own width instead of being re-read. Measured:

| | tile distinct | distinct vector draws | byte-identical rows | max abs r | max argmax mult |
| --- | --- | --- | --- | --- | --- |
| before | 94/1024 | 20/32 | 64/256 | 1.00000 | 11 |
| after | 214/1024 | 32/32 | **0/256** | 0.618 | 5 |
| host IID | — | — | 0/256 | 0.035 | 2 |

The uniform [0,1) distribution is untouched: mean 0.4994, std 0.2887, top decile 0.0996 over 524288
samples (ideal 0.5 / 0.28868 / 0.1).

**Rejected alternatives, all measured, so nobody repeats them.** NOP spacing after the PRNG read
(0 through 32 NOPs: byte-identical output — it is not a pipeline hazard, and the Wormhole kernel's
extra NOPs are irrelevant here); xorshift32 mixing of two draws (no gain — any combination of reads
is still a function of `t + i`); `SFPTRANSP` across four draws with an XOR fold (modest, and it
re-introduced duplicate rows). One trap worth recording: holding `scale`/`from` in lreg4/lreg5
across a transpose silently broke the output range to mean 0.35 / std 0.64, which for a while looked
like a decorrelation success.

**What the fix does not do.** It dilutes the lane/stream degeneracy rather than removing it —
cross-position max |r| is 0.618 against 0.035 for host IID. A full fix needs a counter-based RNG
keyed on each element's own position, which this instruction sequence has no lane index to build
from. `test_rand_independence.py` keeps that half as `xfail(strict)`; the duplicate-column half now
passes and its marker is gone.

**Gate.** Same 4-seed A/B, shipped serving configuration (EOS stop on, degeneracy guard at its
default), GPQA doc0 in thinking mode:

| seed | device answer | device tok/blk/s | host answer | host tok/blk/s |
| --- | --- | --- | --- | --- |
| 0 | C | 23.8 (6 blocks) | C | 37.5 |
| 1 | C | 52.3 | C | 27.2 |
| 2 | C | 53.8 | C | 34.5 |
| 3 | C | 54.5 | C | 37.0 |

8/8 correct, guard never fired on any run, and the served default is back to `device`.

## 6. Defense in depth: the degenerate-canvas detector

Block diffusion commits a whole 256-token canvas at once, so degeneration is directly measurable
on the committed tensor before it reaches the KV cache — no entropy proxy needed. `tt/degeneracy.py`
reports `top_frac` (share taken by the most frequent id) and `max_run` (longest consecutive repeat);
both are needed, because the `\ \ \ \` 2-cycle has `max_run == 1`.

Calibration on real traced blocks (host Gumbel, seed 0, 12 blocks forced past the natural end):

| | distinct ids / 256 | top_frac | max_run |
| --- | --- | --- | --- |
| healthy blocks | 54–106 | 0.06–0.08 | 1–2 |
| collapsed blocks | 1–16 | 0.94–1.00 | 240–256 |

The defaults (`top_frac >= 0.5`, `max_run >= 32`) sit in that gap by an order of magnitude.

**Termination is not degeneration.** Once the answer is complete the model fills the canvas with
`<eos>`, scoring `top_frac 1.0 / max_run 256` — the same numbers, the opposite meaning. So the
verdict takes the caller's stop-token set and never flags a canvas whose dominant id is a stop
token. This exclusion is inert unless the caller passes its stop ids, which the on-device
validation caught: `serving.decode_block` was calling the commit path without them, and the first
terminating canvas raised. Both `generate_blocks` and the serving session now thread them.

The check runs **after denoise and before `commit_fn`** — placement is the point. A degenerate
canvas that reaches the KV cache conditions every later block, which is what makes the state
near-absorbing (P(nonhalt | prev nonhalt) = 85.7% against an 8.2% base rate).

`DG_DEGENERACY_POLICY` is `off` by default (no measurement, no behaviour change), `warn` logs
per-block statistics, `stop` raises `DegenerateBlockError` so generation ends with everything that
was healthy and the collapsed canvas is never committed or emitted.

## 6b. Policy default: `warn`, validated; `stop` opt-in, also validated

`DG_DEGENERACY_POLICY` defaults to **`warn`**: it measures every committed canvas (one bincount
over 256 ids) and never changes behaviour, so a future collapse appears in the log as it happens
instead of being reconstructed from sample dumps afterwards.

`stop` is validated on device and does not interfere. Four seeds, EOS stop enabled (the real
serving configuration), `DG_DEGENERACY_POLICY=stop`:

| seed | exit | stopped early | blocks | tokens | text chars |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | no | 7 | 1792 | 3798 |
| 1 | 0 | no | 8 | 2048 | 5028 |
| 2 | 0 | no | 5 | 1280 | 3294 |
| 3 | 0 | no | 7 | 1792 | 4167 |

Zero false positives; every run still terminated naturally on EOS with full text.

It is still **not** the default. `max_run >= 32` has a plausible false-positive surface in ordinary
content -- a markdown horizontal rule, a table separator row, padding inside a code block, ASCII
art -- and four physics answers do not clear that. Promoting it needs a false-positive study over
that kind of text. The degeneration itself is fixed at the root (§5), so `stop` is defence in
depth rather than the fix.

Getting here took two wiring defects that only running it on device exposed: `serving.decode_block`
was calling the commit path without `stop_token_ids` (so the termination exclusion was inert on the
path that matters, and the first `<eos>` canvas raised), and `stop_token_ids` can be a bare int
from `eos_token_id`, so `set()` on it raised TypeError.

## 6c. What the fix does and does not cover (measured, 10 GPQA docs)

The 10 worst-behaving docs from the served run were replayed under the shipped defaults (host
Gumbel, thinking mode, EOS stop enabled, 12-block cap). Across **192 committed canvases** measured
here and in both 4-seed sweeps:

| | n | max top_frac | max max_run |
| --- | --- | --- | --- |
| healthy, not stop-dominated | 136 | 0.1836 | 18 |
| degenerate | 1 | 0.8516 | 86 |

**doc0 — the trace that emitted a 2000-character wall of `1` in serving — now answers `C`,
correctly.** Nine of ten docs committed no degenerate canvas at all.

**COVERED.** A canvas that collapses onto one content token. `top_frac 0.5` sits 2.7x above the
healthy maximum and 1.7x below the degenerate one; `max_run 64` is 3.5x above the healthy maximum.
On the serving path the request now ends gracefully — the collapsed canvas is refused, the session
is marked finished, and a zero-token terminal emission hands the caller every healthy block it
already received. (First attempt failed the whole request with an exception, which loses the good
text; "no degenerate output" must not mean "no output".)

**NOT COVERED.** `host` Gumbel does not eliminate degeneration, it only makes it rare: doc7 still
collapsed, at the very end of a 12-block run that never finished its answer. And degradation is
*progressive* — the block immediately before the refused one was already repetitive
(`the the the ... ,,,1,1111`) at `top_frac` under 0.5, so it was emitted. Catching that precursor
would need a bound near 0.2–0.3, against a measured healthy maximum of 0.1836; that margin is too
thin to spend, so it is left uncaught rather than traded for false positives on ordinary text.

The residual sits in the over-generation regime — doc7 was one of the traces that was TRUNCATED in
the served run too, i.e. it ran out of context before it ran out of reasoning. That is the
context-length bottleneck `gpqa_thinking3072_sub40_20260723.md` already identified
("Bottleneck is context length (4096), not the sampler"), and it is a separate piece of work.

## 7. What the halt telemetry says, and why it is not the detector

`traced_denoise` now emits the per-step `(entropy, mismatch)` trace it always computed, with a
`halt_blocking_gate` verdict. On a healthy 10-block run every block halted (non-halt 0%), and
`halt_entropy_first` fell from ~4.9 nats to ~1e-4 once the answer was complete — with clean prose
committed either way. So entropy reports "converged", which a finished answer and a collapsed
canvas share. It is the right instrument for *why 48 steps were burned*; it cannot substitute for
measuring the committed tokens.

## 8. Reproduce

```bash
# the A/B (one variable)
for arm in host device; do
  DG_TRACE_REGION_SIZE=12884901888 MESH_DEVICE=P150x4 \
  python -m models.experimental.diffusion_gemma.demo.serving_smoke \
    --max-seq-len 4096 --num-blocks 12 --gumbel-mode "$arm" --upfront --reveal-pmax 4096 \
    --enable-thinking --disable-eos-stop --seed 0 --prompt "<gpqa doc 0>"
done

# per-block degeneracy statistics on any run
DG_DEGENERACY_POLICY=warn  ...   # logs DG_DEGENERACY start_pos=... top_frac=... max_run=...
DG_DEGENERACY_POLICY=stop  ...   # ends generation instead of committing a collapsed canvas
```
