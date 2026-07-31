# AttnRes bringup — phase results

Kimi K3 attention residuals on Blackhole. One-page status: what each phase produced, what
its gate was, what it measured, and what it did **not** establish. Full reasoning, tables and
refutations live in `bringup_log.md`; this file is the index to them.

- **Branch:** `nmilicevic/bringup/kimi-k3-attnres-2026-07-30`, off `main` @ `6d526e8d61d`. Nothing pushed.
- **Machine:** LoudBox, 8 × Blackhole. Meshes exercised: `(1,1)`, `(8,1)`, `(2,4)`.
- **Shape:** `T = 5120`, `d = 7168`, `L = 93` layers, `Bk = 12`, `S ∈ 0..8`, 186 executed reads.
- **Gate:** PCC ≥ 0.9999 **and** rel err ≤ 2e-2 per read; depth gated *relatively*, device against torch-bf16 on the same walk.

**11 of 12 phases closed, 183 tests passing. Phase 11 (PP boundary) not started.**

---

## Phase results

| # | Phase | Gate | Result | Status |
|---|---|---|---|---|
| 0 | Scope + decisions | worktree builds; `ttnn.__file__` resolves under it | **FAIL** first — stale `_ttnn.so`, uninitialized submodules. **PASS** after `git submodule update --init --recursive` + `build_metal.sh` | PASS |
| 1 | Infra map | every in-tree analog at `file:line` **plus its thresholds** | Back-filled 9 phases late. Found the gate is **not** inherited — ours is 100× stricter than the nearest analog's (0.9999 against `pcc=0.99`), and **mHC is not in this tree at all**, so nothing was inheritable from the prior residual bringup | PASS, late |
| 2 | Delta analysis | AttnRes as a countable delta against the analogs | 6-row delta, 5 settled in composed ops. **Delta 4 (the mixture) became Phase 10's entire mandate.** Surfaced `fast_reduce_nc`, which P9 then refuted | PASS, late |
| 3 | API spec | contract written before code, never rewritten after | `API_SPEC.md`. Amended once *inside* the phase, before any code depended on it | PASS |
| 4 | Torch reference | three-rung numeric ladder | **41/41.** Rung-1 error 1.5e-7 … 4.0e-7 against a 1e-5 gate. Two harness defects found and fixed before the reference was trusted | PASS |
| 5 | TTNN composite | forward runs single-device | **30/30.** PCC ≥ 0.9999, rel err ≤ 2e-2 across `S ∈ {0,1,4,8}` × `d ∈ {256,7168}`. Split form reproduces direct at all 24 sites. Three divergences from the frozen spec recorded rather than edited in | PASS |
| 6 | Device correctness + depth | 93-layer walk under the relative gate | **74/74.** 186 chained reads, 8 seals, `S` ramping 0→8: device **0.9999408** against torch-bf16 0.9999741 at `d = 7168` (worst layer 92: 0.9999105 / 0.9999501), norm ratio 1.003655. Reference and device are the *same* walk, so the seal schedule cannot silently diverge | PASS |
| 7 | Remove fallbacks | production `T = 5120`, no host fallback | **79/79.** `block_residual` 560 MiB, 660 MiB of candidates. PCC 0.9999804 at `S = 8`. `T` confirmed a **pure batch axis** — the shared token slice is **bit-identical** between `T = 64` and `T = 5120`, after the full 186-read walk. One warm read moves 7.33 GB in 21.6 ms (~339 GB/s), which PCIe cannot do | PASS |
| 8 | Distribution | memo → judgment gate → TP impl → roofline | **92/92** on a real `(2,4)` mesh. PCC 0.9999778 at `d = 7168`, `S = 8`. Two gates only a mesh can run: sequence-sharded vs replicated agree at **max\|Δ\| = 0**, and stubbing `_reduce_stats` to identity drops PCC to **0.5757** — so the collective is proven load-bearing. Depth on mesh is *better* than single-device (0.9999500 vs 0.9999408) because statistics reduce in fp32 | PASS |
| 9 | Perf harness + loop | numbered hypothesis → measure → keep-or-refute | **P1–P9** plus a tracy attribution. **380 ms traced against 622 ms untraced** at the start; **191.2 ms** at the end. **8 refutations recorded**, three of them of this memo's own claims | PASS |
| 10 | Fused C++ op | composed floor measured against the roofline **first** | `ttnn.experimental.fast_weighted_reduce_nc`, **19/19** at PCC 0.9999 vs torch fp32. Mixture **687.1 → 257.3 µs (2.67×)**, landing 1.13× above an unweighted reduce over the same bytes. Forward **191.2 → 153.6 ms** | PASS |
| 11 | PP boundary | `(1+S)·d` layout round-tripped through a `MeshSocket` pair | Not started | **OPEN** |

---

## The perf ladder

Per forward, traced, `(2,4)`, `T = 5120`, over the real 186-read schedule. Each step was
measured against the step above it, and each changed the op rather than the measurement:

| step | lever | per forward | on the read |
|---|---|---|---|
| Phase 9 start, untraced | — | 622 ms | launch-bound |
| Phase 9 start, traced | `begin_trace_capture` | **380 ms** | 1.64×, and free |
| P6 | statistics fold into `[1,1,N,2(S+1)]` | −15.3 ms | 7.4× on the collective alone |
| P7 | both `d`-reductions in one pass over `v` | ~367 → **267 ms** | 1.37× |
| P8 | `inter_block`'s 24 dots batched into one matmul | 265.0 → **191.2 ms** | 23.4× on that half; 49 → 26 passes per block |
| P10 | the mixture fused into one MAC pass | 191.2 → **153.6 ms** | 2.67× on the op |

**380 → 153.6 ms, 2.47× end to end** — with the caveat that the baseline changes form at P8
(direct → split), so this is a chain of separately-measured steps, not one controlled A/B. The
direct form's own chain is 265.0 → 216.2 ms.

### Where the floor is

Every conclusion in this project is a ratio to a bandwidth floor, and the floor is measured
four independent ways at **228.2–229.1 µs** for one pass over the 79 MiB candidate tensor —
across two axes, two kernels, and fp32 dest accumulation on and off. That agreement is why
the ratios below are trustworthy:

| the mixture, traced `(2,4)`, 79 MiB | µs | × floor |
|---|---|---|
| floor: `fast_reduce_nc` (unweighted, same bytes) | 228.2 | 1.00 |
| **fused: `fast_weighted_reduce_nc`** | **257.3** | **1.13** |
| fused, fp32 weight | 266.9 | 1.17 |
| composed: `mul` + `sum` | 687.1 | 3.01 |

The 29 µs between the fused op and an unweighted reduce is the *entire* cost of the weighting
— the per-candidate multiply plus a weight fetch that is 3% of the read. There is no third
pass hiding in it, which is the row that says the kernel is finished rather than merely faster.

---

## The eight refutations

Recorded because a bringup that only lists its successes is not a record. Three of these
refute claims made earlier in this same memo:

| # | claim | verdict |
|---|---|---|
| 1 | `ROOFLINE.md` §6: "launch-bound even at production shape" | **Refuted.** Dispatch is pipelined; the criterion is `max(host, device)`, not the sum. At production shape the two terms are within 2% |
| 2 | §4's fabric model: 88.5 µs at `num_links = 2` | **Off 2.7×** — 235.9 measured. A collective reaching 18–25% of fabric peak is core-limited (AllGather runs on **2 worker cores**), not link-limited |
| 3 | §4: defer the 32× statistics padding fix | **Correct untraced, wrong traced.** The decision inverts between the regimes |
| 4 | D10: "~12 ops per read" | **28.3** measured |
| 5 | `num_links` should follow production's 2 | **Refuted, stays 1.** With the fold in place the second link buys 4.7 µs per read; they are alternatives, not additive |
| 6 | The `N`-batched matmul reaches the mixture's floor | **Refuted, 1.09×.** All of a 3.2× arithmetic win spent on the two permutes that reach the layout |
| 7 | `fast_reduce_nc` beats `ttnn.sum` on the dim-1 reduce | **Refuted, 0.08%** on a two-run mean against a 0.35% band. The reduce half is already at the floor |
| 8 | P7's mixture cost 791 µs | **Corrected to 688.** P7 multiplied by a `[1,1,1,d]` broadcast; `_mix` multiplies by `[1,C,N,1]`. Same bytes, **102 µs apart** |

A separate source-only correction pass found that **two rows labelled "default fidelity vs
HiFi4" varied no fidelity at all** — `ttnn.sum`, `fast_reduce_nc` and `rms_norm_pre_all_gather`
all default to HiFi4 on Blackhole. Every timing conclusion survived; both stated *mechanisms*
were wrong. `init_device_compute_kernel_config` takes the default as its third positional
argument, so nothing had to be measured to catch it.

---

## Tests

**183 passed, 0 failed** in one run over the whole tree on the committed branch, `(2,4)` where
the test asks for a mesh:

| suite | tests | covers |
|---|---|---|
| `test_torch_attn_res.py` | 41 | the reference against a vendored oracle, three-rung ladder |
| `test_tt_attn_res.py` | 30 | device `forward` + the block-split form, single device |
| `test_tt_attn_res_depth.py` | 3 | the 93-layer, 186-read walk under the relative gate |
| `test_tt_attn_res_production.py` | 8 | `T = 5120`, no host fallback, `T` as a pure batch axis |
| `test_tt_attn_res_distributed.py` | 13 | TP on a real `(2,4)` mesh, SP equality, collective load-bearing |
| `test_attn_res_perf.py` | 69 | the perf loop; logs, asserts nothing |
| `test_fast_weighted_reduce_nc.py` | 19 | the fused op against torch fp32 at PCC 0.9999 |

The op's suite covers by *kernel path*, not shape variety: `C ∈ {1,5,8,9,12,13}` to take the
granularity cap, a clean factor, two primes that fall back to granularity 1, and the
degenerate `C = 1`; `Wt = 1` so the weight set turns over on every output tile and `Ht = 1` so
it never does; `B = 2` for both index chains; 100 rows for the padding path; the bf16×fp32
pair; the full production shape. Plus program-cache reuse with both tensors held past the
assertion, equivalence with `mul` + `sum` at matched precision, and five rejection cases each
pinned to its own error message.

---

## What this does **not** establish

The honest half. Nothing below is measured, and some of it is where the risk lives:

- **Galaxy `(8,4)` and `[LINE, RING]` are modelled, never run.** A ring axis sustains 2
  directions and halves the fabric term; that is arithmetic, not a measurement. `(4,2)` was
  skipped deliberately — it sits between two measured points on both axes. This is the largest
  untested surface, and the end goal runs on it.
- **The remaining ~3× of `ROOFLINE.md` §7.** Folding the two `d`-reductions and the
  cross-candidate softmax into the mixture's pass is not started, and it is not a continuation
  of Phase 10: that op changed no arithmetic, while this one owns the statistics all-reduce and
  the fp32 score chain P7 spent four iterations getting right in composed form.
- **No decode.** `T = 1` has never run. Every number here is prefill.
- **No PP boundary.** Phase 11 in full — the `(1+S)·d` canonical layout and the socket
  round-trip that a pipeline of Galaxies requires.
- **No real K3 weights and no real module outputs.** `apply_module` is an elementwise `h ⊙ w`,
  which exercises the residual bookkeeping but not KDA / MLA / MoE traffic patterns.
- **The perf harness asserts nothing.** It logs. None of the numbers above is regression-guarded,
  so a future change can undo any of them silently.
- **Depth cannot catch a scale defect.** Mutation testing shows the 93-layer gate passes both
  `ttnn.softmax(dim=1)` and an unshifted softmax; the op-level gate is the detector for that
  class. Depth dilutes what it was built to catch.
- **`dim = 0` and non-bf16 input are rejected, not implemented,** in the fused op.
- **The collective's per-call global semaphores are still unhoisted** — 481 µs enqueue against a
  152 µs baseline. Priced, untested, and untraced-only, so it moves none of the numbers above.

---

## Commits

Local only, on `nmilicevic/bringup/kimi-k3-attnres-2026-07-30`:

| commit | contents |
|---|---|
| `2091daec06e` | `attn_res`: the mixture through the fused op — 1.24× on the forward |
| `a14bc26b4d0` | `ttnn`: `fast_weighted_reduce_nc`, a dim-1 reduce that MACs its weight in |
| `3b3e33e6c65` | `attn_res`: fix two fidelity mislabels |
| `871bbdf968b` | Phase 8 — distribution, roofline, TP on `(2,4)` |
