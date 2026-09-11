# Qwen3.6-27B-DFlash

DFlash speculative decoding on Qwen3.6-27B, with **one loop that runs against either a host or a
device target**. A companion to the primary [README.md](README.md), which documents the ttnn port
of the Qwen3.5/3.6 family.

One loop, two swappable halves — a target and a drafter, each host or device:

| Target | Drafter | Purpose |
| --- | --- | --- |
| `HFTarget` (host `Qwen3_5ForCausalLM`) | `HostDrafter` | golden reference; proves the checkpoint pair works |
| `TtTarget` (ttnn `Qwen36Model`) | `HostDrafter` | real device target numerics, real KV and GDN state |
| `TtTarget` | **`TtDrafter`** (ttnn) | fully on device — only token ids cross PCIe |

Code in [tt/dflash/](tt/dflash/) (the ttnn drafter) and [reference/dflash/](reference/dflash/) (the
host drafter, the targets, the loop); tests in
[tests/test_dflash_drafter_tp.py](tests/test_dflash_drafter_tp.py),
[tests/reference/test_dflash_host.py](tests/reference/test_dflash_host.py) and
[tests/reference/test_dflash_device.py](tests/reference/test_dflash_device.py).

## What DFlash is

[DFlash](https://github.com/z-lab/dflash) ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036))
is speculative decoding where the drafter is a **block diffusion** model rather than a small
autoregressive LM. Two checkpoints are involved:

| Role | Checkpoint | Params | Env var |
| --- | --- | --- | --- |
| Target | `Qwen/Qwen3.6-27B` | 27B, 64 layers | `HF_MODEL` |
| Drafter | `z-lab/Qwen3.6-27B-DFlash` | 1.73B, 5 layers (3.5 GB) | `DFLASH_HF_MODEL` |

Per decode step:

1. The target's residual stream is tapped at layers **`[1, 16, 31, 46, 61]`** and the five
   5120-wide vectors are concatenated into a 25,600-wide context feature.
2. The drafter's `fc` projects that back to 5120, and one drafter forward fills a **16-slot
   block**: slot 0 is the confirmed anchor token, slots 1–15 start as `mask_token_id` (248070).
   All 15 are drafted **in parallel** — the drafter's last layer is bidirectional
   (`full_attention` with `is_causal=False`), so masked slots see each other. Its first four
   layers are causal sliding-window (2048).
3. The drafter has no embedding and no LM head of its own. Its input is the **target's** raw input
   embedding of the block; its output hidden states go through the **target's** `lm_head`.
4. The target verifies all 16 slots in one forward; the longest matching prefix is accepted and a
   bonus token is emitted from the target's own distribution.

Greedy decoding accepts only exact argmax matches and the sampling path uses standard rejection
sampling, so **output is distributionally identical to plain autoregressive decoding**. Acceptance
length is a speedup number, never a quality knob — which is what makes it testable: a correct
implementation must reproduce the baseline token stream exactly.

The drafter's modeling code is vendored in [reference/dflash/dflash.py](reference/dflash/dflash.py).
It has to be: the checkpoint declares `auto_map: {"AutoModel": "dflash.DFlashDraftModel"}` but ships
no modeling file, so there is nothing for `trust_remote_code` to load.

## Layout

| File | What it is |
| --- | --- |
| [tt/dflash/drafter.py](tt/dflash/drafter.py) | **the ttnn drafter**, replicated on every device |
| [tt/dflash/weights.py](tt/dflash/weights.py) | weight upload + the `fc` row permutation |
| [tt/dflash/config.py](tt/dflash/config.py) | `DFlashDrafterConfig`, checkpoint resolution |
| [reference/dflash/dflash.py](reference/dflash/dflash.py) | the host drafter, vendored from upstream z-lab (MIT) |
| [reference/dflash/targets.py](reference/dflash/targets.py) | `SpeculativeTarget` + `HFTarget` + `TtTarget` |
| [reference/dflash/drafters.py](reference/dflash/drafters.py) | `SpeculativeDrafter` + `HostDrafter` + `TtDrafter` |
| [reference/dflash/generate.py](reference/dflash/generate.py) | the speculative loop, backend-agnostic |

Two small protocols keep the loop from knowing which backend it drives, and they *are* the port
spec. `SpeculativeTarget` is six methods (`reset`, `forward`, `snapshot`, `restore`, `embed`,
`lm_head`); `SpeculativeDrafter` is two (`reset`, `propose`). Everything about *how* a draft
happens — noise embedding, KV history, LM head, sampling — sits behind `propose`, which is what
lets the ttnn drafter keep the whole draft on the mesh. Neither HF cache objects nor ttnn tensors
leak into the loop.

## Running it

The host tiers need no device.

```bash
# 1. Structural correctness. Seconds, no checkpoint, no network.
#    Randomly-initialised models that keep Qwen3.6's structure (interleaved DeltaNet /
#    full attention) at toy dimensions.
pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv

# 2. + the real drafter (3.5 GB download, strict state-dict match).
DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
  pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv

# 3. + the real 27B target, end to end on host. Minutes on CPU, ~60 GB RAM.
HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash DFLASH_RUN_TARGET=1 \
  pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv
```

On a Tenstorrent mesh:

```bash
# The ttnn drafter, PCC'd against the host reference module by module. No 27B needed.
MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
  pytest -svq models/demos/blackhole/qwen36/tests/test_dflash_drafter_tp.py

# Device target primitives (taps, block forward, rollback) on a few layers — minutes.
MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \
  pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_device.py

# + the full 64-layer 27B generating, with the host drafter AND with the ttnn drafter.
MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
  DFLASH_RUN_TARGET=1 \
  pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_device.py
```

Both env vars take a hub id or a local checkpoint directory; hub ids are `snapshot_download`ed, the
same handling `Qwen36ModelArgs` gives `HF_MODEL`.

Driving it directly:

```python
from models.demos.blackhole.qwen36.reference.dflash import dflash_generate, load_drafter, load_target

drafter, target = load_drafter(), load_target()
stats = dflash_generate(drafter, target, input_ids, max_new_tokens=64, return_stats=True)
print(stats.mean_acceptance_length)          # tokens committed per target forward
baseline = dflash_generate(drafter, target, input_ids, max_new_tokens=64, block_size=1)
assert (stats.output_ids == baseline).all()  # block_size=1 disables speculation
```

`block_size=1` is the control: it turns the loop into plain autoregressive decoding through the
same code path, which is what every equivalence assertion compares against.

## Measured — host

Real `Qwen/Qwen3.6-27B` + real `z-lab/Qwen3.6-27B-DFlash`, greedy, on a 2×Xeon Silver 4309Y host
(`test_real_end_to_end_speculation`, prompt `"The capital of France is"`, 32 new tokens):

| | |
| --- | --- |
| Output | `" Paris.\n\n<think>\n\n</think>\n\nThat is correct. Paris is the capital and most populous city of France. It is located in the north-central part of the"` |
| Equivalence | identical to the `block_size=1` baseline, token for token |
| Mean acceptance | **2.21 tokens/step** over 14 steps |
| Per-step | `[3, 1, 1, 2, 4, 5, 4, 2, 2, 1, 2, 1, 1, 2]` |
| Rollbacks | 14 of 14 steps |

So the pair works: the drafter loads, drafts, and buys ~2.2× fewer target forwards while leaving
output unchanged.

### Throughput on host: speculation is a 3x LOSS

Same three-row shape as the device benchmark, so the two are directly comparable
(`test_host_throughput`, warmed, 24 new tokens):

| path | tok/s | s/tok | acceptance |
| --- | --- | --- | --- |
| HF `generate` (native decode) | 0.808 | 1.24 | — |
| our loop, `block_size=1` | 0.812 | 1.23 | 1.00 tok/step |
| speculative, `block_size=16` | **0.267** | 3.75 | 2.56 tok/step |

**0.33x**, and the middle row is the important one: our loop with speculation off runs at **1.01x**
HF's own `generate`, and the test asserts its tokens are identical to HF's greedy output. So the
harness costs nothing on host and 0.33x is speculation's own effect, not measurement overhead.

Put the two platforms side by side and the cause isolates cleanly:

| | native decode | harness at `block=1` | speculative |
| --- | --- | --- | --- |
| host | 0.808 tok/s | 0.812 (**1.01x**) | 0.267 (**0.33x**) |
| T3K | 2.14 tok/s | 0.74 (**0.35x**) | 3.18 (**1.48x**) |

On device, speculation overcomes a 2.9x harness penalty (the 128-token bucket) and still wins 1.48x.
On host, with *no* harness penalty at all, it still loses 3x. The difference is not the loop.

#### Why: verify forwards scale with token count on a CPU

Fit `forward = f + c * tokens` to the host runs (two equations, two unknowns) and it reproduces the
measurements to within 1% — predicted 3.81 s/tok against 3.78 measured:

| | |
| --- | --- |
| `f` = **0.90 s/forward** | streaming 54 GB of weights, paid once per forward whatever the token count |
| `c` = **0.39 s/token** | the actual compute |

Both land on sensible hardware constants — 54 GB / 0.90 s = **60 GB/s** memory bandwidth, 54 GFLOP /
0.39 s = **138 GFLOPS** effective — so this is a cost model, not a curve fit to noise.

| | cost |
| --- | --- |
| 1-token decode forward | 0.90 + 0.39 = **1.30 s** |
| 16-token verify forward | 0.90 + 6.26 = **7.17 s** (5.5x a decode) |
| 2.56-token replay forward | **1.90 s** (GDN rollback; host only, `TtTarget` anchors instead) |
| per step | 9.07 s for 2.56 tokens = 3.55 s/tok, + 0.27 drafter = **3.81 s/tok** |

The single number that explains everything: **`c / f` = 0.43 — each extra token costs 43% of the
entire weight stream.** Speculative decoding assumes that ratio is near zero; that is what makes
verifying 16 tokens nearly free once the weights have been read. A CPU target is compute-bound, so
the verify forward scales almost linearly with block size and you do 16 tokens of work to keep 2.56.

Useful ceiling to keep in mind: **in the ideal memory-bound case with no replay, speedup equals the
acceptance rate exactly** (verify costs `f`, decode costs `f`, and you commit 2.56 tokens per
verify). Three things eat it here, in order: `c` being large (intrinsic to CPU), the replay forward
(~26% on top of every step), and 83% of the drafted tokens being discarded.

So the host path is a **correctness reference only** — which is what it was built for. Do not read
its acceptance numbers as a throughput proxy in either direction.

Note that no step accepted all 15 drafted tokens, so **every** step took the rollback path. At this
acceptance rate a block of 16 mostly drafts tokens that are thrown away; sweeping `block_size` is
worth doing before committing the device design to 16.

## Measured — device (T3K)

The same loop, same drafter, target swapped for the ttnn `Qwen36Model`
(`test_tt_speculation_matches_autoregressive`, same prompt, 24 new tokens):

| | |
| --- | --- |
| Output | `" Paris.\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   -"` |
| Equivalence | identical to the device's own `block_size=1` baseline, token for token |
| Mean acceptance | **4.60 tokens/step** over 5 steps |
| Per-step | `[2, 1, 2, 16, 2]` — one step accepted all 15 drafts |

### Throughput: 5.6x SLOWER than the production traced decode

> ⚠️ **Read this before the table.** An earlier version of this section headlined "1.48x real
> decode". That baseline was `Qwen36Model.decode_tp`, which is the **eager bespoke validation**
> decode — per token it host-uploads the token, does host RoPE trig, runs an eager per-layer Python
> loop and reads the logits back. The **production** decode path is traced
> (`text_demo.py`, every case `use_trace=True`) and runs at **17.87 tok/s / 56.0 ms/tok** at ISL 128,
> batch 1, greedy — see [README-T3K-27B.md](README-T3K-27B.md#performance). `decode_tp` is 8.3x
> slower than that, so "1.48x" was measured against the wrong reference.

Against the real decode path:

| path | tok/s | ms/tok | vs production |
| --- | --- | --- | --- |
| **production traced decode** (README-T3K-27B) | **17.87** | 56.0 | 1.00x |
| speculative, ttnn drafter | 3.18 | 314 | **0.18x** |
| speculative, host drafter | 2.70 | 370 | 0.15x |
| `decode_tp` (eager validation path) | 2.14 | 467 | 0.12x |
| bucket-only, `block_size=1` | 0.74 | 1360 | 0.04x |

**The speculative path is 5.6x slower than just running the model normally.** The within-harness
ratios below are still valid and still useful for comparing drafters — the ttnn drafter is 1.18x the
host drafter and 4.32x the bucket-only baseline, and both accept 5.17 tok/step identically — but
none of that is a speedup over production.

The cause is that **the entire speculative path is eager while production is traced**:

| | ttnn drafter | host drafter |
| --- | --- | --- |
| `target.forward` (eager 128-token bucket, untraced) | **9.4 s (94%)** | 9.7 s (82%) |
| `drafter.propose` (230 eager dispatches, 13% device utilization) | 0.6 s (6%) | 2.1 s (17%) |
| loop | 0.1 s | 0.1 s |

Drafting is only 6% of a step, so the drafter is no longer where the time is. The target's untraced
bucket forward is.

**What the algorithm is still worth:** speculation commits **5.17 tokens per target forward**. If a
traced verify forward cost ~2x a traced decode step (112 ms) and still committed 5.17 tokens, that
is ~22 ms/tok — roughly 2.5x production. That is arithmetic from the acceptance rate and the
published traced-decode cost, **not a measurement**, and the last projection in this file was badly
wrong (see below), so treat it as motivation rather than a target.

Priority order is therefore: **trace the speculative step**, then shrink the 128-token bucket to 32,
then shard the LayerNorm.

### UPDATE: the verify forward is now traced — 1.96x

Priority 1 is done for the target half. The masked bucket CAN be traced; the blocker was never
`valid_len` masking (that is a fixed-shape `multiply` whose contents can be staged) but eight
per-step host uploads, the last of which was `ttnn.arange` inside the conv1d one-hot selector.
See `tt/model.py::capture_verify_trace` and the evidence in
`tests/reference/test_dflash_target_trace_replay.py`.

MEASURED on this branch (64 tokens, "The capital of France is", greedy — a different harness from
the 3.18 tok/s figure above, so compare the two rows below, not across tables):

| | tok/s | ms/tok | vs production traced decode |
| --- | --- | --- | --- |
| speculative, eager verify | 3.68 | 272 | 0.21x |
| **speculative, TRACED verify** | **7.21** | **139** | **0.40x** |
| production traced decode | 17.87 | 56.0 | 1.00x |

Tokens are bit-identical between the two and acceptance is unchanged at 7.000 tok/step, as greedy
speculation requires. **Speculation still loses to simply running the model** — this closes about
half the gap, not all of it.

Remaining, in the order they now matter: the drafter is still eager (~120 ms of a ~970 ms step,
12 % and rising as the target gets faster, and its KV design currently blocks tracing — see
`tests/perf/test_dflash_drafter_trace_probe.py`); the 128-row bucket still verifies a 16-token
block; and ~89 % of steps still pay a second target forward for the rollback. See [DFLASH_DRAFTER_OP_MAPPING.md](DFLASH_DRAFTER_OP_MAPPING.md).

> **Absolutes are soft.** This is a shared machine: `decode_tp` alone has measured 0.93, 1.05, 1.14,
> 1.64, 2.08 and 2.14 tok/s on identical code as other jobs came and went. Only within-run ratios
> are trustworthy, which is why every variant runs in one process.

#### The warmup matters more than anything else measured here

The `warmup` in that benchmark is not hygiene, it is the difference between two opposite
conclusions. Without it, the first variant to touch the drafter absorbs every kernel compile inside
the timed region. Unwarmed, `propose` measured **7.9 s** for the tensor-parallel drafter and
**23.0 s** for this replicated one, which read as "the drafter is CCL-bound" and then as "replication
made it worse". Warmed, `propose` is **0.6 s**. The JIT cache line in the log is the tell: 961/1007
hits unwarmed, 1007/1007 warmed.

So treat the tensor-parallel-vs-replicated comparison in this file's history as **unresolved**: the
TP version was never measured warmed, and its 22-collectives-per-step diagnosis came from a
compile-dominated number. Replication is what ships, it is comfortably fast enough (6% of a step),
and re-litigating it could win 6% at best — but the claim "TP was too collective-heavy" was not
established.

> **Absolutes are soft.** This is a shared machine: `decode_tp` alone has measured 0.93, 1.05, 1.14,
> 1.64, 2.08 and 2.14 tok/s on identical code as other jobs came and went. Only within-run ratios
> are trustworthy, which is why every variant runs in one process.

Do **not** read 4.60 against the host's 2.21 as a device-vs-host result — the two runs diverge into
different continuations after the first token, and this one lands in a `<think>` preamble whose
boilerplate is trivially predictable. Both numbers are single short prompts.

An earlier measurement of this same case read 5.75 tok/step (`[3, 2, 13, 5]`). It was taken before
the drafter's context-starvation fix (bug 3 above), which changed what the drafter proposes; on this
one prompt acceptance did not improve. That is a sample of one either way — the fix is right on
first principles (the drafter now sees more than the last accepted block), not because it moved this
number. What the device run establishes is correctness: the taps, the block forward, and the state
handling all agree with the target's own autoregressive output.

## The ttnn drafter

[tt/dflash/](tt/dflash/) is the drafter on device, TP across the mesh. With it, a speculative step
crosses PCIe only for the chosen token ids: the target's taps stay on the mesh, feed the drafter's
`fc` directly, and the draft logits come off the target's own resident LM head.

**Parallelism: none.** The drafter is **replicated** on every device and each one runs the whole
thing redundantly. It is 1.73B params (~1.8 GB per device in bf8) over a 16-token block — ~55 GFLOP
— so redundant compute is nearly free, every projection and norm is local, and the only collective
in a step is a single all-gather of the target's taps. Drafting costs 6% of a step this way.

**The `fc` row order** is the one place the target's TP layout leaks in. `fc` consumes 5
concatenated taps in tap-major order, but the taps arrive as the target's residual stream — already
fractured on hidden — so each device holds 5 *scattered* blocks of the 25,600 input rows. The
drafter concatenates its local slices and all-gathers, which yields **device-major** columns, so
`reorder_fc_rows` permutes the weight's rows the same way once at load time instead of shuffling
activations every step.

**KV history** is a plain growing tensor per layer, concatenated per step, not a paged cache. That
sidesteps the target's bucket-alignment constraint entirely (a paged multi-token write must start
at a 128-aligned offset, and speculation advances 1-16 tokens a step), and it is cheap here: 1 KV
head x 128 dim is 256 B per token per layer, so a 4096-token context is 1 MB per layer per tensor.

**Validated** against the host drafter in
[tests/test_dflash_drafter_tp.py](tests/test_dflash_drafter_tp.py), as a ladder so a regression
localises rather than just saying "the drafter is wrong":

| rung | PCC vs host reference |
| --- | --- |
| `fc` + `hidden_norm` (the tap projection) | 0.99990 |
| one sliding-attention layer | 0.99894 |
| one bidirectional full-attention layer | 0.99896 |
| all 5 layers | 0.99783 |
| two steps, with carried context | 0.99818 |

End to end on the full 27B (`test_tt_drafter_end_to_end`): output identical to the device target's
own autoregressive baseline, **4.60 tok/step** over 5 steps (`[2, 1, 2, 16, 2]` — one step accepted
all 15 drafts).

Between those two sits `test_tt_drafter_agrees_with_host_on_real_taps`, which runs both drafters on
the target's *real* taps. It exists because per-module PCC feeds the drafter taps the test builds
itself — validating arithmetic but not the hand-off — while end-to-end acceptance catches a bad
hand-off without saying where. Its load-bearing assertion is tap-vs-host-tap PCC (1.0), which is
what pinned bug 2 below.

One thing that test deliberately does **not** assert: agreement past drafted slot 0. DFlash samples
a whole block jointly over masked slots, so from slot 1 on the per-slot argmax is a coin toss
between near-tied modes and bf16-vs-fp32 flips it. The two drafters agree on 9/15 tokens and
diverge at slot 1, while reaching 4.6 and 5.2 tok/step respectively. Draft quality is an
acceptance-rate question, not a token-equality one.

### Three bugs the port found

Worth recording, because two of them are silent and one was in the *host* path:

1. **`tt_all_reduce` with `cluster_axis=1` on a `(1, N)` mesh is a no-op.** It hits an
   `1 in mesh_shape` early-out and returns its input unchanged — no error. It read as `fc` PCC 0.35,
   i.e. exactly one device's partial. `cluster_axis` must be 0 here. And on N300/T3K the function
   does not all-reduce at all: it reduce-*scatters*, so completing the reduction needs a following
   gather — via `tp_common.tuned_vocab_all_gather`, because `ccl.tt_all_gather` takes `cluster_axis`
   literally and axis 0 holds one device.
2. **Device taps must be sliced to `valid_len` before they leave `take_taps`.** A masked bucket is
   padded to 128 rows; handing the full bucket on gave the caller *padding* rows. That read as tap
   PCC 0.38 against the host taps and drafted pure garbage — 1.00 tok/step, every step. The host
   branch had always sliced; the device branch has to as well.
3. **The host drafter was running context-starved.** Upstream crops its KV cache to `start` after
   each step, which keeps the context rows the step appended and drops only the noise block. An
   earlier version here snapshotted before the forward and restored after — which looks equivalent
   but returns the cache to its *pre-forward* length, i.e. permanently empty. Nothing failed: the
   drafter still drafted, because `target_hidden` alone carries a lot. It just never saw anything
   older than the last accepted block. Fixed in `truncate_kv`, with the length assertions that
   would have caught it.

## The device path: anchoring

Swapping in `TtTarget` ran into a hard constraint in the device prefill path, measured on T3K
(`test_tt_block_forward_needs_bucket_alignment`):

> A masked-bucket prefill is exact only when `chunk_start` is a multiple of the **bucket size**
> (128) — not of the 64-token paged block, and not of `valid_len`.

| `chunk_start` | 64 | 128 | 192 | 256 | 320 |
| --- | --- | --- | --- | --- | --- |
| PCC vs one-shot prefill | 0.16 | **1.00** | 0.27 | **1.00** | 0.25 |

Aligned offsets come back at exactly 1.0 every run. *How badly* an unaligned offset degrades varies
with the token sample — the 0.16/0.27/0.25 above are from one seeded sweep; another run of the same
test gave 0.97 and 0.99. So the test only asserts that an unaligned offset is **not exact**.

The cause is the KV write: `paged_fill_cache` writes the whole padded bucket starting at block
`chunk_start // 64`, so consecutive segments are spaced by the bucket rather than by their real
length. There is no device primitive for a multi-token write at an arbitrary offset —
`paged_fill_cache` starts at a block boundary, and `paged_update_cache` writes one token per batch
element. Isolating the layer types confirmed it: attention-only PCC was 0.16 at offset 64 and
exactly 1.0 at 128.

Speculation advances 1–16 tokens a step, so `start` is arbitrary and cannot be used as
`chunk_start`. `TtTarget` **anchors** instead: it holds a 128-aligned `anchor` with a GDN snapshot,
and every forward re-runs the whole span `[anchor, start + S)` as one bucket at `chunk_start=anchor`.
Re-running up to 127 already-computed tokens is free — the bucket costs 128 positions either way.
The only thing the caller must respect is `max_block(start)`: a block may not cross the anchor's
boundary, so near one the loop drafts a shorter block.

With anchoring, the device is exact at arbitrary starts (`test_tt_target_forward_matches_one_shot`):

| `start` | 40 | 100 | 128 | 130 |
| --- | --- | --- | --- | --- |
| PCC vs one-shot prefill | **1.00** | **1.00** | 0.9995 | 0.9995 |

**Anchoring also makes rollback disappear.** Each forward restores GDN to the anchor and rewrites
the entire `[anchor, anchor+128)` KV span, so a rejected block is simply overwritten — nothing to
undo, and no replay to pay for (`test_tt_rejected_block_leaves_no_trace`, PCC 1.0). That is why
`TtTarget.replays_after_rollback` is False while `HFTarget`'s is True. The device ends up with the
*cheaper* rollback of the two.

### What was added to the model

Three additive changes in [tt/model.py](tt/model.py), all inert unless a caller opts in:

- `set_residual_taps(layer_ids)` / `take_taps(valid_len)` — capture the residual stream after
  chosen checkpoint layers. Disarmed, `_record_tap` is one falsy check and the forwards are
  unchanged (`test_tt_taps_are_off_by_default`).
- `prefill_block_all_logits(...)` — masked-bucket prefill returning logits for *every* real
  position, which verification needs, rather than only the last.
- `save_gdn_state(into=None)` / `restore_gdn_state(...)` — GDN snapshot/restore that works on
  **both** paths. The existing `_save_deltanet_states` reads `recurrent_state` / `fused_conv_state`
  and is single-device only; `TPGatedDeltaNet` keeps `rec_state` / `conv_carry` / `conv_states` in
  the module instead, so TP crashed with `AttributeError` on the old helper.

## The Gated DeltaNet rollback (host target)

This is the one part of the loop that is **not** a port of upstream's, and the reason the device
path above needed designing rather than just wiring up.

Speculative decoding needs to undo a rejected block: after verifying 16 slots and accepting *k*,
the target's cache must go back to the state it had after *k* tokens. For attention layers that is
a KV truncation. For Qwen3.6-27B's **48 Gated DeltaNet layers** it is not — they carry a recurrent
state and a conv state, not a per-token history.

Upstream handles this with `cache.crop()` after `cache.activate_past_recording()`. Neither works
against Qwen3.6-27B on transformers 5.12.1:

- `activate_past_recording` does not exist in 5.12.1; and even on transformers `main` it only
  preserves the linear-attention **conv** states — `recurrent_states` are overwritten in place and
  never rolled back.
- `crop()` on a `linear_attention` layer is an unconditional no-op
  (`LinearAttentionCacheLayerMixin.crop`: *"We don't crop the linear attention cache, so simply do
  nothing here"*).

So a naive port silently keeps 48 recurrent states advanced past the accepted prefix while
`get_seq_length()` reports the cropped length. There is no error — just wrong tokens. On a shrunk
Qwen3.6 config, feeding an 8-token junk block and cropping it away flips **~25% of subsequent
argmaxes**. `test_linear_attention_crop_is_a_noop_tripwire` pins that behaviour, and will fail (as
a signal to simplify) if a transformers upgrade ever fixes it.

[generate.py](reference/dflash/generate.py) instead rolls back exactly:

1. Snapshot every GDN conv + recurrent state before the verify forward (~200 MB for the 27B —
   negligible against the forward itself).
2. On a partial acceptance, restore the snapshot, truncate the attention KV, and **replay the
   accepted tokens**. The replay is what re-advances the GDN state correctly, and it also produces
   the tap hidden states the next draft needs, so it is not wasted work.

The drafter's own cache is handled by snapshot/restore rather than `crop` for a different reason:
its four sliding-window layers cannot be cropped once past their 2048-token window
(`DynamicSlidingWindowLayer.crop` raises, because the older keys needed to refill the window are
already gone). Snapshot/restore is exact at any context length, and the drafter is small enough
(~40 MB of K/V at a full window) that it costs nothing. It is also unconditional: the block's noise
K/V must **never** persist as context, so every step undoes its own drafter forward, and only
accepted tokens re-enter — as `target_hidden` on the following step.

The replay costs a second target forward on every partially-accepted step. On host that is merely
slow; the device sidesteps it entirely by anchoring (see above), which is the more interesting
result — the constraint that forced anchoring also removed the need for a rollback.

## What is left for a real device port

`TtTarget` proves the algorithm on hardware; it is not yet a serving path.

- **The block pays a full 128-token bucket — now 70% of wall clock.** The smallest masked bucket is 128, so a 16-token
  block does ~8× the attention and GDN work it needs. Anchoring rides along inside that bucket for
  free, but the bucket itself is the thing to shrink — either a smaller bucket in
  `_PREFILL_MASK_BUCKETS`, or a KV write that respects `valid_len` so blocks need not be padded.
- **No trace.** Every forward is eager dispatch. The serving path traces prefill chunks; a
  speculative block is a fixed shape, so it should trace well.
- **Nothing in the speculative path is traced**, while production decode is. That is the 5.6x gap
  and the first thing to fix.
- **The target is 94% of a speculative step**, so it is where the remaining work is. The 128-token
  bucket a 16-token block has to pay for is the first item below. `_record_tap` does a `to_torch` inside the layer loop
  because the drafter is on host. Moving the drafter onto the mesh would remove five PCIe
  round-trips per step, and the drafter is only 1.73B — a 5-layer Qwen3-style GQA stack that maps
  onto `tt/attention/`, `tt/mlp.py`, `tt/rms_norm.py` and `tt/rope.py` as they already stand.
- **`block_size` is unswept.** 16 is upstream's default, not a measured optimum for this pair.

## Known limitations

- **Batch size 1.** The rejection sampler and the acceptance bookkeeping are both written for a
  single sequence.
- **Text only.** `load_target` builds `Qwen3_5ForCausalLM` from the checkpoint's text config; the
  vision tower plays no part in drafting.
- **CPU speed, not device speed.** Host acceptance length is architecture-accurate and transfers to
  device; wall-clock timings from a host run do not.
- Drafter runs fp32 (it is the golden reference); target runs bf16, since fp32 would be 108 GB of
  weights for no added fidelity in the verify path.
