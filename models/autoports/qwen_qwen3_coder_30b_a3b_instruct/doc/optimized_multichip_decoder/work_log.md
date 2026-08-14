# Work log — stage 04, optimized multichip decoder

Chronological. Every number is either a cell of a file in this directory or a
line of a script in `probes/`, and the command that produced it is given. Where
something did not work, the measurement that says so is given too.

Baseline: `tt/multichip_decoder.py` at `56366307f44` (stage 03, clean-pass).
The optimization is **in place** in that same file, which is why stage 03's
`doc/multichip_decoder/perf_*.csv` are never regenerated — they are the frozen
"before" half of every table here, and `tests/test_perf.py` now writes the
multichip rows into this directory instead.

---

## 0. What the audit said before anything was changed

`topology_audit.md` is the audit and it is not repeated here. The two findings
that carried the stage:

* **Both residual RMSNorms run on one core** — rows 134 and 159 of the stage-03
  decode profile, `CORE COUNT` 1, 20.081 and 20.127 µs, 40.208 µs and 9.7% of
  the layer for 128 KB of work.
* **The router projection reads its weight DRAM-interleaved** — row 160, 24.916
  µs on 4 cores for a 0.5 MB weight, 21 GB/s.

Both are *replicated* work, so both sit inside the 25.18% of the layer that
`mesh_plan.md` §4.2 identified as the Amdahl cap. The task framing called the
router "a distribution question". It is not: the router matmul is 4 tiles wide
and `topk` over one 128-wide row is one core, so there is nothing to distribute
and no collective cheap enough to be worth adding (the measured decode
collective floor is ~11 µs, and this profile's `AllGatherAsync` on a 32 KB
payload costs 12.2). What the router *did* have was a memory-config bug hiding
in plain sight, and so did the norms.

## 1. The sharded RMSNorm — 4× faster and 4× more accurate

`probes/norm_router_probe.py`, then `probes/norm_accuracy_probe.py`. Trace
slope, median of 30, at the shipped decode shape `[1,1,32,2048]` bf16.

The first probe measured the sharded norm against the shipped one and reported
`max|diff| 1.562e-02`, which says *different* and not *better*. The second probe
adds a torch fp64 reference computed from the bf16-rounded inputs the device
actually sees, so the comparison is against the mathematical answer:

```
interleaved, no compute config (shipped)   19.82 us   max|err vs fp64| 6.711e-02  rel 1.21e-02
sharded  4 cores, default                   7.36          6.711e-02       1.21e-02
sharded  4 cores, HiFi4 fp32acc             7.53          1.439e-02       2.60e-03
sharded  8 cores, default                   4.26          3.586e-02       6.48e-03
sharded  8 cores, HiFi4 fp32acc             4.92          1.686e-02       3.05e-03
                                 8 cores: i2s 0.51 us, s2i 0.53 us
```

**8 cores, HiFi4, `fp32_dest_acc_en=True`.** 4.0× faster and 4.0× more accurate
than the call it replaces, which passed no compute config at all and therefore
accumulated the sum of squares in bf16. **`norm_accuracy_probe.py` crashed at its 16-core leg** and the run
was not repeated. The failure is the probe's, not the op's:

```
TT_FATAL: Illegal kernel placement for writer_unary_sharded,
          Kernels cannot be placed on dispatch cores!  (assert.hpp:104)
RuntimeError: ... program.cpp:149: not on_dispatch_core
```

— the probe builds its core range as a rectangle and at 16 cores that rectangle
reaches a row holding dispatch cores. So **the 16-core row published in
`README.md` is `norm_router_probe.log`'s, not this probe's**, and there is
consequently *no* fp64 accuracy figure at 16 cores. That is a gap, and it is
tolerable only because 16 cores was already rejected on speed by the first
probe and rejecting it does not need the accuracy column: a 2048-wide row is 64
tiles, so 8 cores already hold 8 each, the norm stops improving (4.26 at 8 vs
4.14 at 16) and the reshard at both ends grows with the core count (i2s+s2i
0.51/0.53 at 8 against 0.75/0.76 at 16), which is net worse. Had 16 cores been
*ahead* on speed, this crash would have had to be fixed rather than recorded.

The shard spec is deliberately `optimized_decoder._width_sharded_l1(2048)` —
8 cores, one per DRAM bank, `[32, 256]` — which is bit-for-bit the memory config
`attention_decode_optimized` reshards its input into. So the first norm's output
crosses into the qkv projection with **no conversion at all**: stage-03 row 135
(`InterleavedToSharded`, 0.915 µs) is gone from the stage-04 window.

`ttnn.rms_norm`'s sharded program factory reads its weight as ROW_MAJOR
`[1, 1, dim/32, 32]` rather than the tiled `[1,1,1,dim]` the interleaved factory
takes, so `MultichipWeights` carries a second copy of each residual norm vector
— 4 KB each against the layer's 95.5 MB.

Prefill keeps the interleaved norm. `decode_residual_norm` asserts `S <= 32` and
prefill norms are 0.2% of the prefill profile.

## 2. The router projection — 24.62 → 5.85 µs shipped, bit-identical

Same probe. `[1,1,32,2048] × [2048,128] → fp32`:

```
router matmul interleaved (shipped)      24.62 us
router matmul in0 L1 wsh  4 cores         4.30 us   max|diff| 0.000e+00
router matmul in0 L1 wsh  8 cores         5.85 us   max|diff| 0.000e+00
router matmul DRAM-sharded N=256 fp32 bf16 w   7.37 us (+s2i 0.45)  max|diff| 6.873e-02
router matmul DRAM-sharded N=256 fp32 bfp8 w   7.40 us (+s2i 0.50)  max|diff| 5.028e-02
router matmul DRAM-sharded N=256 bf16 bf16 w   7.34 us (+s2i 0.51)  max|diff| 6.525e-02
router matmul DRAM-sharded N=256 bf16 bfp8 w   7.39 us (+s2i 0.51)  max|diff| 5.205e-02
```

N = 128 is 4 tiles, so the matmul uses 4 cores either way and the speedup is
entirely the activation's placement: L1 instead of DRAM-interleaved. **The
shipped figure is the 8-core leg, 24.62 → 5.85 µs, 4.2×** — 8 is what the norm
shards over, and the last paragraph of this section is why the layer takes the
8-core pair rather than the faster-looking 4-core one. The 4-core leg's 4.30 µs
is a leg of the sweep and not a shipped number; where it appeared as one, review
caught it. **The
output is bit-identical**, which matters more here than the speed — the four
dies must agree on the top-8 exactly or the four 32-expert windows stop being a
partition of it, and `test_router_windows_partition_global_routing` asserts
`max |stitched − global| = 0.0` against the *single-chip* router at S = 1, 33,
128. A matmul that changed the logits in the last ulp would put that test at the
mercy of near-ties.

The DRAM-sharded spelling was built because it is the shape stage 02 tuned the
other two decode projections into; `_dram_sharded_ok` needs both dims divisible
by `8 banks × 32 = 256` and N = 128 fails, so the weight was padded to 256 with
zero columns and sliced back. It runs, and it is **rejected on both counts**:
1.26× slower than the shipped L1-activation spelling (7.34–7.40 plus its own
0.45–0.51 µs sharded→interleaved, against 5.85) *and* numerically different.

Because the norm now emits an 8-core width shard and the router wants a 4-core
one, the shipped path uses the 8-core shard for both (5.85 µs standalone,
6.241 µs in the layer at row 182): 8-core norm + 8-core matmul is 4.92 + 5.85 =
10.77 against a 4-core norm + 4-core matmul's 7.53 + 4.30 = 11.83.

## 3. What that bought, on the layer and in the profile

`probes/layer_levers.py`, one process, one mesh, one set of real weights. The
"stage 03" leg is a verbatim copy of the committed stage-03 layer **body**, so
the before/after pair is measured minutes apart on the same hardware.

**It is not a copy of stage 03's whole path, and the difference biases the
result.** The leg calls `MC.all_reduce`, which is stage 04's — persistent
collective buffers and one-link decode included — so it measures only the norms
and the router projection and silently gives stage 03 two of stage 04's four
changes. The bias is **conservative**: it understates the pass, which is why
this probe's 1.098× is smaller than the 1.112× the two committed trees give.
The CSV pair is the headline ratio; this is the same-session corroboration of
its direction and rough size, not an independent measurement of the same thing:

```
stage 03 (before)                  0.4700 ms   (reference)
04 (the shipped default)           0.4282 ms   (max|diff| 1.465e-03)
04: + threshold router tail        0.4355 ms
04: + CCL in L1                    0.4377 ms   (confounded -- see below)
04: + num_links=1 forced           0.4286 ms   (== the default; see section 6)
```

The `+ CCL in L1` leg is **confounded** and is kept only as corroboration: it
read `ctx.num_links` (2) where the shipped `all_reduce` reads `_links` (1 at
decode), so it paid for the second ethernet link on top of the L1 staging. Its
0.4377 is an upper bound on the L1 penalty, not a measurement of it. The lever
is rejected either way — `layer_levers2/3.py` reject the neighbouring L1
placements with matched link counts — but the probe is fixed (`MC._links`) and
the row is labelled in `README.md`.

**0.4700 → 0.4282 ms, 1.098×**, and the two perf CSVs — measured in a different
session by `tests/test_perf.py` — say 0.4767 → 0.4286, **1.112×**. The
`max|diff| 1.465e-03` against the stage-03 leg is the norm's changed accuracy
showing up downstream — §1's table says the new norm is the closer of the two to
fp64 — and the PCC gates in `tests/test_multichip_decoder.py` are what decide
whether that is acceptable (§7).

In the profile (`probes/profile_layer.py decode`, then `probes/window.py`),
device 0, the last of two decode iterations: **414.661 → 362.828 µs, 1.143×**,
and the eleven/twelve-range decomposition in `topology_audit.md` sums exactly to
both.

## 4. Re-audit, and three things that did not work

Every round of this model's optimization has promoted a new top op, so the
profile was re-read before deciding anything else. It said:

* `TopK` 26.356 µs on one core — now the biggest op in the layer outside the
  expert matmuls, 7.3% — with a 4.190 µs `FillPad` in front of it;
* the two reduce-scatters cost **18.871 and 15.018 µs for the same shape**, the
  first fed from DRAM-interleaved and the second from L1-interleaved.

`probes/layer_levers3.py`, two interleaved passes so a leg's spread against
itself is on the page:

```
pass1 04 default                     0.4348 ms
pass1 04 + attn out in L1            0.4403 ms
pass1 04 + persistent CCL            0.4343 ms
pass1 04 + persistent CCL + L1 in    0.4389 ms
pass1 04 + topk logits in L1         0.4380 ms
pass2 04 default                     0.4346 ms
pass2 04 + attn out in L1            0.4399 ms
pass2 04 + persistent CCL            0.4337 ms
pass2 04 + persistent CCL + L1 in    0.4397 ms
pass2 04 + topk logits in L1         0.4383 ms
```

* **Feeding the first reduce-scatter from L1: 1.2% worse.** The asymmetry
  between the two collectives is real and repeats in the stage-03 profile
  (20.413 vs 16.322), but it is not the input's buffer type — putting the
  attention output in L1 costs more than it saves, both alone and combined with
  persistent buffers.
* **Staging the logits in L1 before `topk`: 0.8% worse.** With `sorted=False`
  (33.78 vs 33.81 µs standalone) and a bf16 input (31.81, and forbidden anyway
  — routing must select in fp32 logit space) that closes the `topk` question.
  A 128-wide `topk` over one row is one core by construction.
* **Persistent CCL buffers: 0.2% better, and adopted.** See §5.

## 5. Persistent collective buffers

The first attempt raised

```
reduce_scatter_validate_utils.cpp:77: output_tensor.layout() == input_tensor.layout
```

because the probe passed `[intermediate, penult, output]` where the op wants
`[intermediate, output, penult]`
(`tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py:200`). A first
API error is not a rejection; corrected, it runs and is consistently faster:

| | pass 1 | pass 2 |
|---|---|---|
| allocating (stage-04 default) | 0.4348 | 0.4346 |
| persistent, `layer_levers2.py` | 0.4335 | 0.4333 |
| persistent, `layer_levers3.py` | 0.4343 | 0.4337 |

~0.2%, against a leg-vs-itself spread of 0.05%, in four paired measurements
across two probe runs. Adopted as `_decode_ccl_buffers`, owned by `MeshContext`
alongside the semaphores, keyed by **`(logical shape, padded shape, dtype)`**
— the logical component is load-bearing and the next two paragraphs are why —
and **only for
`S <= 32`** — prefill runs at a different `S` on every call and the prefill
collective is bandwidth-bound rather than allocation-bound, so caching there
would allocate a set per sequence length for no measurable gain.

All 48 layers of the stacked model share the cache. That is safe because the
trace serialises them and the all-gather's result is cloned out before the next
collective starts. The clone is the price and it is inside the 0.2%.

**The colliding pair is the priming prefill and the decode call, not the
layer's two all-reduces.** (Review corrected this paragraph; the code was
always right, the explanation was not.) Both of a decode layer's partials are
`[1, 1, batch, 2048]` — the attention one comes out of `wo` after
`_concat_heads_decode` has sliced the padded tile back to `batch`, and the
expert one is `batch` by construction — so they have the *same* key and share
one buffer set, correctly and by design. The profile shows exactly that: decode
iteration 1 allocates **one** set, four ops, and the layer's second all-reduce
allocates nothing.

What collides is the **priming prefill**. Every harness runs a short prefill
before the first decode to fill the cache, and at `S <= 32` that prefill is
inside the branch too — it gets its own set, logically `S` rows. Keyed on the
padded shape alone, a 32-token priming prefill and a decode at `batch < 32` are
the same key: both are `[1, 1, ·, 2048]` padded to one 32-row tile. A persistent
output buffer imposes *its* logical shape on the op's result, so decode
inherited the prefill's **32** rows and the layer's output silently became a
32-row tensor. Six decode tests caught it, and the shapes in the failures name the
mechanism exactly (transcribed here rather than archived: the failing run wrote
`pytest_full.log`, which the passing re-run then overwrote —
`pytest_decode_retry.log` is the 38-test re-run that confirmed the fix):

```
test_multichip_decode_vs_single_chip[contiguous]  RuntimeError: The size of tensor a (2048)
                                                  must match the size of tensor b (65536)
test_multichip_decode_contiguous_batch8           binary_ng_device_operation.cpp:224
                                                  Invalid subtile broadcast type
```

65536 = 32 × 2048 — the prefill's row count, in a decode result. The key now
carries the logical shape as well, which is what
`probes/layer_levers2.py` and `layer_levers3.py` did — their `_persist_ar` keys
on `x.shape`, so the probes that measured the 0.2% were measuring the *correct*
two-buffer arrangement all along, and only the port into the module was wrong.
Worth recording because it is the exact failure mode a preallocated-buffer
optimization has in a 48-layer stack, and because it is silent in every test
that only compares a path against itself: the 20-step determinism test and the
trace tests all passed with the wrong shape.

## 6a. The two levers the contract names by hand

### `matmul_reduce_scatter_async` on the `wo` → RS edge

Stage 03 named this as the one untried collective lever, bounded at 5.3% of the
layer. Wiring it into the layer raised
`matmul_reduce_scatter_async.cpp:36 mesh_device != nullptr`; that was not taken
as the answer. `probes/mmrs_probe.py` rebuilds the edge standalone at the
shipped decode shapes — in0 `[1,1,32,1024]`, weight `[1024,2048]` K-sharded,
scatter dim 3 — where it runs:

```
unfused: matmul (2D default)   18.82 us
unfused: reduce-scatter        10.73 us
fused matmul+RS grid (8, 6) rs@(0, 6)    30.91 us   max|diff| vs unfused 2.734e-02
fused matmul+RS grid (8, 4) rs@(0, 4)    30.90 us
fused matmul+RS grid (8, 8) rs@(0, 8)    30.85 us
```

**Rejected.** Fused is 4.4% slower than the unfused pair at the same program
config, and that comparison already flatters it: the fused op takes a 2D
`MatmulMultiCoreReuseMultiCast` config, so adopting it means `wo` gives up the
DRAM-sharded config that runs it at **8.228 µs** in the layer (row 174) and pays
18.82 instead. It is also numerically different from the unfused pair by
2.734e-02. Three grid/offset splits were tried; all three land within 0.06 µs of
each other, so the loss is not a tuning failure.

This matches the note in `models/demos/blackhole/qwen36/tt/tp_common.py:524` —
"unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses)"
— which is a different model on the same silicon reaching the same conclusion.

### Removing the router's ROW_MAJOR round trip

Stage 02 recorded `untilize → scatter → tilize` as **not removable**, because
`ttnn.scatter` takes only ROW_MAJOR and every consumer of the dense vector needs
TILE. That reasoning is wrong. `topk(sorted=True)` puts the 8th-largest logit in
column 7, so

    dense = exp(logits − max) · (logits ≥ top_logits[..., 7])

computed over all 128 columns is the same vector and never leaves TILE.
`router_forward_threshold` implements it and deletes rows 190–197 — a
`zeros_like`, two typecasts, three untilizes, a scatter and a tilize, **17.007
µs** of profile.

It is **0.8% slower**: 0.4382 / 0.4382 ms against 0.4348 / 0.4346, in two
interleaved passes, and 1.7% slower against the final default (0.4355 vs
0.4282). Widening the softmax's `sub` and `exp` from 8 columns to
128, plus the `ge` and the `mul`, costs more than the layout conversions save.

The output is **bit-identical on all four dies** (`max|diff| 0.000e+00`), which
is also the evidence for the tie question the construction raises: with fp32
logits accumulated over K = 2048, no two logits tie at rank 8. Kept in the
module as a measured alternative rather than deleted, because the arithmetic is
the useful part.

## 6b. One ethernet link for decode, which stage 03 measured and dismissed

Stage 03 measured `num_links=1` at 0.4738 ms against two links' 0.4766 — 0.6% —
called it noise-level and kept 2 links for a single code path. Against the
stage-04 layer the collectives are a larger share of a smaller layer and no
longer allocate, so the lever was re-opened.

**Review then found the probe could not reproduce its own result.** `_links`
returned `ctx.num_links` only when it *differed* from `NUM_LINKS`, so a caller
asking explicitly for 2 links at decode silently got 1 — and `links_probe.py`
builds its two-link leg exactly that way. The probe was measuring one link
against one link. `_links` now reads two separate fields, `num_links` for
prefill and `decode_num_links` for decode, so each mode's count is
independently settable and an explicit override is honoured.

Repairing it and re-running returned **the same gap**, which is suspicious
rather than reassuring: the broken probe had produced that gap between two
*identical* configurations. Either the lever is real, or the leg that runs first
in each pass is slower and the published 1.2% was always an artifact of the
probe running 2 links first, every pass. So the probe now alternates the leg
order across six passes:

```
posA  2 links 0.4342  0.4341  0.4340     1 link 0.4290  0.4288  0.4286
posB  2 links 0.4341  0.4337  0.4339     1 link 0.4291  0.4283  0.4287

mean  2 links 0.43400      1 link 0.42875      1.22%
```

**Each configuration reads the same at both positions**, so the gap follows the
link count and not the running order. The lever is real, and the adoption
stands. Output is bit-identical on all four dies in all twelve legs.

The reconciliation is that the old log **predates `_links`**: it was taken while
`all_reduce` read `ctx.num_links` directly, which did distinguish the legs. So
the figure was correct when it was measured, and adopting `NUM_LINKS_DECODE` is
what destroyed the probe's ability to re-derive it — a lever that broke its own
evidence. That is the part worth remembering.

## 6c. The rotary lever: built, measured, 3.05× — and backed out

Review would not accept limitation 4's original wording, and it was right not
to: it dismissed `rotary_embedding_llama` with "128 wide, so a width shard has
almost nothing to spread", which is an argument about a *different op*. Stage 03
lost a review for exactly that move.

**Standalone it is a large win.** `probes/rope_probe.py`, trace slope at the
shipped per-die decode shape `[1,1,32,128]`:

```
rotary_embedding      (shipped, HF order, DRAM)   3.84 us
rotary_embedding_llama (decode, Meta order, L1)   1.26 us   max|diff| 0.000e+00  PCC 1.0000000
  interleaved -> height-sharded [32,128]          0.20 us
```

3.05×, bit-identical, and *not* by spreading: the llama decode factory shards
over **batch**, so at batch 1 both run on one core. The gain is L1 residency and
a resident 32×32 matrix instead of a DRAM cos/sin gather — finding B's lever on
a different op. Against rows 163–164's 9.358 µs that is 1.4–1.7% of the layer,
more than the ethernet-link lever this stage did adopt.

**So it was implemented, not argued about.** The Meta channel permutation of the
Q and K row blocks of `wqkv` plus Qwen3's per-head `q_norm`/`k_norm`
(`weight_mapping.permute_wqkv_to_meta`, `permute_head_vector_to_meta`), a
host-side assertion of the whole convention (`test_meta_rope_weights_match_hf`),
a decode-only Meta weight twin, and the position gather hoisted onto the first
eager call — legitimate because `token_index` is a Python int for the shipped op
too, so neither spelling can move the rotary position inside a replayed trace.

**Then `test_multichip_decode_vs_single_chip` read PCC 0.876.** The dies agreed
exactly (spread 0.0), so it was systematic, not a race.
`probes/rope_layer_probe.py` isolated it in one run:

```
fresh KV cache          attention out  max|diff| 1.221e-04   PCC 0.9999697
prefill-primed cache    attention out  max|diff| 8.911e-02   PCC 0.1932974
```

**RoPE runs before K is written, so the KV cache inherits the rotary's channel
convention.** Prefill is untouched by this lever and writes HF-ordered keys; a
Meta-ordered decode Q scores against them and the dot products are meaningless.
The op-level probe looked clean only because its cache was fresh and every other
key was zero — which is the trap, and the reason a standalone rotary probe can
never settle this question on its own.

The lever is therefore **not decode-local**. It needs the llama rotary in
prefill as well, prefill's interleaved `wqkv` permuted, and a changed KV-cache
channel convention — which `test_per_die_kv_heads_stitched` compares against a
single-chip cache. That is a whole-layer change to `optimized_decoder.py`'s RoPE
convention and would invalidate the prefill numbers this stage reports as
unchanged.

One more cost, found on the way and worth keeping because it bounds the prize:
`ATTENTION_WEIGHT_DTYPE` is `bfloat8_b`, whose 16-element blocks share an
exponent, so permuting channels **regroups the blocks and requantizes**. Even on
a fresh cache the layer is not bit-identical (1.221e-04 on the attention output,
3.125e-01 on the K cache after permuting back). Bit-identity is a property of
the op at fixed input, not of the layer at permuted weights.

Backed out of the shipped path and left runnable behind
`upload_multichip_weights(meta_rope=True)`, on the same principle as
`router_forward_threshold`: the measurement is the useful part, and the stage
that can change prefill should inherit it rather than rediscover it.

## 7. Correctness, and the one number that moved

**112 passed, 0 failed** over the whole 4-die suite with the perf tests
deselected (`pytest_full.log`) -- stage 03's 111, plus the host-only
`test_meta_rope_weights_match_hf` that section 6c added. The PCC table is in `README.md`; every cell is a
line of `pcc_log.txt`. Nine of the ten rows are stage 03's numbers to the digit.

The tenth is `decode vs single-chip`: **0.99997 → 0.99994**. That comparison is
against `optimized_decoder.py` replicated on the mesh, and stage 04 did not
change `optimized_decoder.py`'s numerics (its one edit is the optional `rope=`
seam of section 6c, which defaults to the shipped op) — it still runs the one-core interleaved RMSNorm
with no compute config. §1 measured that call at 6.711e-02 from a torch fp64
reference and the new sharded one at 1.686e-02, so **the multichip path moved
towards the true answer and away from the reference**, and the two now differ
slightly more.

Against HF, which is the arbiter, the honest statement is not "nothing
regressed" — three of the four decode steps are marginally *lower*:

| step | stage 03 | stage 04 | Δ |
|---|---|---|---|
| 0 | 0.9992853 | 0.9992450 | −4.0e-05 |
| 1 | 0.9987817 | 0.9987388 | −4.3e-05 |
| 2 | 0.9993126 | 0.9993928 | +8.0e-05 |
| 3 | 0.9989592 | 0.9989421 | −1.7e-05 |

All four deltas are at the fifth decimal and they go both ways, which is what a
norm whose rounding changed looks like — not a regression, and not "nothing".
The band moved from 0.99878–0.99931 to 0.998739–0.999393, i.e. it widened by
about 4e-05 at the bottom and 8e-05 at the top. The gate is 0.99.

Two assertions are new, and both exist because of something that went wrong here
rather than because of something that might:

* `test_decode_output_layout_matches_input` asserts the layer returns exactly the
  tensor contract it takes — logical shape, dtype, layout, memory config, and
  bit-identity across the four dies. That is the inter-layer residual contract
  written as a test, and it is what a persistent-buffer shape collision breaks
  (§5). Every test that compares a path against *itself* — the 20-step
  determinism test, the trace tests — passed while the layer was returning a
  32-row tensor.
* `test_no_runtime_fallbacks` gained `norm_shard_feeds_qkv_directly`, the
  equality between the sharded norm's output shard and the qkv projection's
  input shard. Breaking it costs a silent reshard, which is the same class of
  defect as the three fallbacks that test already covered.

## 8. Watcher

`TT_METAL_WATCHER_DISABLE_ETH=1` is still required — the watcher's active-eth
program does not fit alongside the `FABRIC_1D_RING` router
(`../multichip_decoder/work_log.md` §11) — and every worker core is still
instrumented, including the `sparse_matmul` reader where the `nnz` assert lives.
Evidence is `watcher.log` and `pytest_watcher.log`.

## 9. Context contract

Unchanged, and `doc/context_contract.json` records why rather than being
silently left alone. Nothing stage 04 changed touches KV dtype, KV layout,
paging, or activation sharding *of the cache*: the two new per-layer tensors are
the ROW_MAJOR copies of the residual norm vectors (4 KB each, 0.384 MB over 48
layers per die against 4.596 GB of sharded weights), and the persistent
collective buffers are two sets of ~0.5 MB owned by the mesh context rather than
by a layer. The measured 262144-token capability and the 22.350 GB/die of
headroom stand.

## 10. Numbers, before and after

See `README.md`; every cell is a CSV in this directory, and both ratio columns
are computed from those cells by `probes/summarize_perf.py`.
