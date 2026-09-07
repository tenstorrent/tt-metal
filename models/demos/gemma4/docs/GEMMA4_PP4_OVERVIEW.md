# Gemma4 prefill on a Blackhole galaxy: pipeline parallelism (PP=4 × [8,1])

Start here. This is the summary — what the change is, how it works, what it measures, and what to
do next. The two companion documents are:

* [`GEMMA4_PP4_PLAN.md`](GEMMA4_PP4_PLAN.md) — the design and the arithmetic behind it, written
  before any code, and left as written so the reasoning can be checked against the outcome.
* [`GEMMA4_PP4_RESULTS.md`](GEMMA4_PP4_RESULTS.md) — the full measurement record, including the
  four things the plan got wrong and the estimate-vs-actual table.

---

## 0. Checkpoint — status as of 2026-09-07

One session, one commit, on `gemma4-pp4` off `gemma4-prefill-pr`. PP=4 runs end to end at 256k and
is faster; the split is proven not to change the numbers; the layer split and several suspected perf
levers are closed by measurement. Nothing is merged and nothing is wired into CI.

### Verified — every one of these came back BIT-EXACT, not merely high-PCC

| what | how | result |
|---|---|---|
| Splitting the layer stack does not change the answer | `verify_pp_split.py --layers 17 --split 8` | exact, 3/3 chunks |
| …at the real rank0/rank1 boundary, with stage B starting on **global layer 17** | `--layers 30 --split 17` | exact, 3/3 chunks |
| …including the final-norm / last-rank gate | `--layers 17 --split 11 --last` | exact, 3/3 chunks |
| The TP=1 head-grouped `concat_heads` equals its definition, `permute(0,2,1,3).reshape` | `verify_concat_heads.py`, 8 head shapes TP=8→TP=1 | exact, 8/8 |
| The TP=4 baseline is untouched by these edits | re-ran `test_prefill_long_context_traced` | 13.6 s / 19,287 vs archived 13.6 s / 19,271 |

### Measured

| | |
|---|---|
| 256k, 1 request | **11.64 s / 22,517 tok/s — 1.17×** the 8×4 baseline |
| 256k, 4 in flight | **23,912 tok/s — 1.24×** |
| max stage compute | 9.78 s = **1.39×**, and *at theory* for TP=1 |
| the 16% not delivered | 0.67 s fill + 0.56 s inbound socket sync + 0.64 s residual imbalance |

### Closed by measurement — do not re-spend time here

* **The layer split.** `17,13,17,13` is the unique optimum of all 32,509 contiguous splits (§4).
* **TP=1 compute efficiency.** At theory; predicted 295 ms/chunk, measured 277–303.
* **The fabric-link lease cycle.** Prime suspect for the per-chunk overhead; measures **0.1 ms**.
* **Chunk 16384.** The first 18.19 s reading was a cold-JIT artifact; warm it is still 1.0 s worse.
* **PP=2 × [8,2].** Balances the globals 5/5 but keeps half the collectives; modelled ~12.2 s.

### Remaining, correctness first

1. Wire Gemma4 into the multi-rank golden-KV PCC harness — §5.5 item 1. Closes three gaps at once.
2. TP=1 vs TP=4 equivalence run — the one TP-shaped change `verify_concat_heads` does not cover.
3. Promote `verify_pp_split.py` to a marked pytest so the guarantee cannot rot.
4. Restore a CPU reference for CP prefill — the only thing that answers whether the path Gemma4 PP
   sits on is right at all. Inherited from the branch, not introduced here.
5. Perf: the ~18 ms/chunk inbound socket sync (shared runner, helps every pipelined model), then
   the global layers themselves — 7.4× a sliding layer and ~60% of a stage's cost.
6. Migration wiring for PP, and a numerical KV-shape check per rank.

---

## 1. What it does

Gemma4-31B prefill previously ran as **one rank on the whole 32-chip galaxy**, an 8×4 mesh split
CP=8 × TP=4: every chip holds a quarter of every layer's weights, and each of the 60 layers pays
a round of tensor-parallel collectives across its 4-chip TP group.

PP=4 carves the same galaxy into **four Z-connected `[8,1]` column sub-meshes**, one MPI rank each.
Each rank owns a contiguous slice of the layers and runs them at **CP=8 × TP=1** — no tensor
parallelism at all. The hidden state hops stage → stage over a ttnn `MeshSocket` on fabric.

```
   producer ──H2D socket──▶  rank 0        rank 1        rank 2        rank 3
                            [8,1] mesh    [8,1] mesh    [8,1] mesh    [8,1] mesh
                            8 chips       8 chips       8 chips       8 chips
                            layers 0-16   layers 17-29  layers 30-46  layers 47-59
                            embed +       ────D2D────▶  ────D2D────▶  ────D2D────▶
                            17 layers     13 layers     17 layers     13 layers
                                                                      + final norm
                              │             │             │             │
                              └─ its own KV ┴─ its own KV ┴─ its own KV ┴─ its own KV
```

Chunks flow through continuously: while rank 3 works on chunk *k−3*, rank 0 is already on chunk
*k*. Steady-state throughput is therefore **1 / max(stage)**, not the sum — which is why stage
balance (§4) matters and why you must never add the four stage times together.

**Within a stage nothing changes.** Each rank still runs Gemma4's context-parallel prefill exactly
as the single-rank path does: sequence sharded 8 ways, the paged KV cache CP-sharded alongside it,
and `ring_joint` SDPA gathering the prefix around the CP axis. CP stays at 8, so the 1024-token
sliding window still fits a per-rank Q slab and **chunk 8192 remains the canonical size** — the
same one the baseline uses, so the comparison is like for like.

## 2. Why it is faster

The branch's own per-layer attribution says the TP-axis collectives are the biggest single bucket
in a global layer — **1344.8 µs, 30.6%, more than SDPA**. At TP=1 they are simply gone:
`ccl_allreduce` and `ccl_allgather` short-circuit at `tp <= 1`.

The trade would normally be a memory disaster, and here it is exactly free. Gemma4 shards weights
**only on the TP axis**; the SP axis replicates. So TP 4→1 quadruples per-device weight bytes while
PP=4 quarters the layer count each device holds. Net zero — measured: the TP=1 weight cache is
35 GB against the TP=4 cache's 37 GB, and every stage fits the same 256k / 2-user KV as the
baseline.

Working it through: with `W` the TP-parallelisable work, `X` the TP collectives and `R` the CP ring
collective (whose per-device traffic also scales as 1/TP),

```
1 rank, 60 layers @ TP=4:   60·(W/4 + X + R₄)  =  15W + 60X + 60R₄
PP=4, 15 layers @ TP=1:     15·(W  + 0 + 4R₄)  =  15W        + 60R₄
```

Everything except `X` cancels. Predicted ceiling **1.44×**; measured stage compute **1.39×** of it.

## 3. Results

All measured on `bh-glx-120-b03u02`, one session, baseline re-run alongside rather than quoted.

| config | 256k prefill | tok/s | vs baseline |
|---|---:|---:|---:|
| **8×4 single rank** (CP=8 × TP=4), chunk 8192 | **13.6 s** device / 14.1 s wall | **19,287** | — |
| **PP=4 × [8,1]** (CP=8 × TP=1), chunk 8192, 1 request | **11.64 s** | **22,517** | **1.17×** |
| PP=4 × [8,1], 4 requests back to back | 43.85 s / 1,048,576 tok | **23,912** | **1.24×** |

Where the time goes on the single-request run:

```
max stage compute        9.76 s   ->  1.39x    <- what TP=1 bought; at theory, see below
+ pipeline fill         +0.67 s              3 stages before rank 3 starts
+ inbound socket sync   +0.56 s              ~18 ms x 31 chunks, an untraced device op
+ residual imbalance    +0.64 s              stages differ by 8%; the pipeline runs at the max
= measured span         11.64 s   ->  1.17x
```

**The compute is at theory.** Predicting a TP=1 stage as `4 × (per-layer TP=4 cost − TP CCL)` gives
295 ms/chunk; the four stages measure 277–303 ms. There is no TP=1 inefficiency left to recover —
every remaining gain has to come from the bubble or from the layers themselves.

## 4. Stage balance is by GLOBAL-layer count, and the split is provably optimal

`layer_types` is 50 `sliding_attention` + 10 `full_attention`, with a global every 6th layer
(5, 11, … 59). Sliding layers are flat in context; global layers pay the whole prefix. Fitting
per-layer costs to a measured run gives, summed over the 32 chunks of a 256k prefill:

| layer type | cost |
|---|---:|
| sliding | **0.301 s** |
| global | **2.234 s** — a **7.4×** premium |

Those two numbers predict a *different* split's stage times to under 1%, so the model is trustworthy.
Exhaustively searching all **32,509** ways to cut 60 layers into 4 contiguous stages, the shipped
`PREFILL_PP_LAYER_COUNTS=17,13,17,13` is the **unique optimum** at 9.712 s, against an unreachable
perfectly-balanced 9.348 s. The naive even split is 7th (10.314 s, and measures 12.04 s end to end).

The residual 3.9% is structural: **10 global layers do not divide by 4.** No split fixes it. (A
PP=2 × [8,2] pipeline would balance them 5/5 exactly — but it keeps half the TP collectives, and
modelling that puts its stage period around 12.2 s, well worse than 9.71. It is not a perf option.)

## 5. Correctness

### 5.1 What Gemma4 had before PP=4

Gemma4 has a large PCC surface — some 15 test files assert against HuggingFace references
(`test_model.py`, `test_attention.py`, `test_layer.py`, `test_vllm_parity.py`, `test_spec_decode.py`,
`test_lm_head.py`, `test_moe.py`, …). **None of it touches the context-parallel prefill path.** Those
tests run the decode and non-CP paged-prefill paths on small meshes (1×N); the `ring_joint` CP
prefill that this whole branch is about is exercised only by
`text_demo_prefill.py::test_prefill_long_context_traced`, which asserts:

* every chunk's hidden state is finite,
* the output is not degenerate (`std > 0.001`),
* ring attention was actually called at least once per layer.

**No numerical comparison at all.** `cpu_prefill_reference.py` was deleted in `e56882c307b` ("a
replacement reference is coming"), and the branch's other prefill test,
`test_prefill_layer_perf_chunk_n`, is a perf test and is red at HEAD anyway. The numerical evidence
for CP prefill therefore rests on the author's earlier single-layer PCC against HF (sliding 0.99932,
global 0.99986) and an op-level 0.99975 `ring_joint` probe — real work, but inspection-era artifacts
rather than a running gate.

There **is** a strong multi-rank PCC harness in the common runner —
`models/demos/common/prefill/runners/ci/run_multirank_pcc.sh`. The runner publishes a KV chunk table
under real migration; the producer pushes a golden trace's tokens, reads the device KV back through
that table, PCCs it per layer against golden KV, and writes a per-rank JSON verdict that a gate
enforces. That is the right shape of check, and it is already multi-rank aware.

### 5.2 Does any of it cover PP=4? No — and not because of PP

The multi-rank harness is **Gemma4-blind**, pipeline or not, on two counts:

* Gemma4 has no golden trace — `Gemma4PrefillAdapter.prefill_trace_default = ""`.
* `prefill_producer._read_slot_kv_and_check_pcc` dispatches on adapter name for `minimax_m3` and
  `gpt_oss_d_p` and otherwise **falls through to the MLA reader**, which would misread Gemma4's
  packed-global `[Krot128 | Vordered512]` CP-sharded ring cache entirely.

So before this work there was no numerical gate on Gemma4 CP prefill at any rank count, and PP
inherited exactly that.

### 5.3 What is now established: the split changes nothing, bit for bit

`tests/perf/pp4/verify_pp_split.py` asks the one question PP introduces — **splitting the layer
stack must not change the answer** — without needing a golden trace or an HF reference. The same
layer window, over the same chunks, runs two ways on two different columns of the galaxy, with the
second path relaying the activation through the host using *exactly* the mapper the D2D socket uses
(`[Shard(2), Replicate()]`):

| check | reference | split | result |
|---|---|---|---|
| mid-window boundary | layers `[0,17)` | `[0,8)` + `[8,17)` | **bit-exact**, 3/3 chunks |
| the real rank0/rank1 boundary | layers `[0,30)` | `[0,17)` + `[17,30)` | **bit-exact**, 3/3 chunks |
| final-norm gate (`--last`) | `[0,17)` post-norm | `[0,11)` + `[11,17)` post-norm | **bit-exact**, 3/3 chunks |

The two paths run identical ops in an identical order, so `exact=True` — not merely high PCC — is
the expected outcome and anything less would be a defect. The second case is the one that matters
most: stage B begins on **global layer 17**, which is the configuration that segfaulted before the
use-after-free fix, and it is the actual shipped boundary. The third toggles the final norm on both
sides (visible as the output's std moving from 3.2 pre-norm to 24.4 post-norm), covering the one
PP-specific change the first two could not.

`tests/perf/pp4/verify_concat_heads.py` covers the other numerically-novel change: the head-grouped
`concat_heads` needed for TP=1. It compares against the *definition*
(`x.permute(0,2,1,3).reshape(1,1,S,H·D)`) rather than against another device run, at all eight head
shapes Gemma4-31B reaches between TP=8 and TP=1 — **all bit-exact**, with grouping firing only where
intended (global TP=1, 2 groups; everything else 1 group, i.e. the original single call).

Alongside those, the structural evidence: the per-stage cost model matched measurement to under 1%
(a stage running the wrong layer *types* would cost visibly differently), each rank logs the global
layer indices it believes it owns, and `Gemma4DecoderLayer` now raises rather than silently
defaulting `layer_scalar` to 1.0 when a lookup misses.

### 5.4 What is still NOT established

Be precise about the scope of the above:

1. **That Gemma4's CP prefill is numerically right in the first place.** The split checks prove PP
   preserves whatever the single-rank path computes. If that is wrong, PP is wrong identically. This
   is inherited, not introduced — but it is the biggest open item on the whole branch.
2. **The D2D socket transport itself.** The checks relay through the host using the socket's mapper,
   so the *layout* is validated; the socket is not. Mitigating: it is model-agnostic shared code
   already gated by the kimi27/glm52 CI.
3. **TP=1 vs TP=4 end-to-end.** Both sides of every split check are TP=1, so a wrong TP-dependent
   quantity would be wrong identically on both. `concat_heads` is covered directly; the other TP=1
   change — `local_heads = num_key_value_heads // tp` for global layers — is not.
4. **Migration.** Off throughout (`PREFILL_ENABLE_MIGRATION=0`). `build_kv_chunk_table` has not been
   checked for `first_layer_idx` awareness.

### 5.5 What to expand, in order

1. **Wire Gemma4 into the existing multi-rank PCC harness.** This is the real gate and most of it
   already exists. Needs: (a) a Gemma4 golden trace — token IDs plus per-layer golden KV from HF or
   vLLM; (b) a `_read_slot_kv_and_check_pcc_gemma4` in the producer that understands the packed
   global `[Krot128 | Vordered512]` and CP-sharded ring layout, undoing CP the way
   `export_paged_kv_cache_natural_order` does; (c) `build_kv_chunk_table` verified per rank. That
   buys golden-KV validation, over the real sockets, per rank, in CI — and it closes items 1, 2 and
   4 of §5.4 at once.
2. **A TP=1 vs TP=4 equivalence run.** Same layers, same tokens, `[8,1]`/TP=1 against `(8,4)`/TP=4,
   PCC on the hidden state. Expect ~0.999 rather than exact — the reduction order differs — which is
   why it is a separate check from the split ones. Closes §5.4 item 3 cheaply.
3. **Promote the split check to pytest.** `verify_pp_split.py` runs in ~2 minutes at 17 layers and
   needs only the TP=1 weight cache; as a marked test it would keep the guarantee from rotting.
4. **Restore a CPU reference for CP prefill** (§5.4 item 1) — the author's stated intent, and the
   only thing that answers whether the path is right rather than merely self-consistent.

## 6. Next steps, ranked

### Already closed — do not re-spend time here

* **The layer split.** `17,13,17,13` is optimal over all 32,509 splits (§4). Exhausted.
* **TP=1 compute efficiency.** At theory (§3). Nothing to recover.
* **PP=2 × [8,2].** Modelled at ~12.2 s stage period vs 9.71 s. Worse, not a fallback worth taking.
* **The fabric-link lease cycle.** Suspected as the per-chunk overhead; instrumented and measured
  at **0.1 ms**. Not the problem.
* **Chunk 16384.** Measures 12.62 s against 11.64 s — the extra chunk size buys nothing on stage
  compute (max 9.67 vs 9.76) and costs 0.5 s more pipeline fill. Note the first measurement of this
  read 18.19 s and was a **cold-JIT artifact**: rank 0's first chunk alone was 5,904 ms, and every
  later chunk matched its peer rank to within 1%. Always re-run a new shape warm before believing it.

### Worth doing, in order

1. **Fold the inbound socket sync into the trace, or make it cheaper.** ~18 ms per chunk per rank,
   0.56 s of an 11.64 s run, and it is a floor rather than pipeline idle — rank 0 pays it with
   nothing upstream to wait for. It is an untraced device op (`inbound_socket_service_sync`) costing
   a full dispatch round trip per chunk. This is the largest single remaining item and it lives in
   the shared runner, so it would benefit every pipelined model, not just Gemma4.
2. **Attack the global layers themselves.** They are 7.4× a sliding layer and 10 of the 60 layers
   account for ~60% of a stage's cost. Everything the branch already knows about CCL and SDPA in
   the global path applies, and PP does not change it — but PP does make it the dominant term.
   A per-stage Tracy capture (`tests/perf/pp4/analyze_layer_budget.py`) is the way in; read its
   header first, because an op's cost across a stage's 8 concurrent chips is the MAX, not the sum,
   and `InboundSocketServiceSyncOperation` is ~99% of a PP stage's device time and is pure idle.
3. **Serve concurrent requests.** Fill is 0.67 s of a single request's 11.64 s and is pure pipeline
   ramp; it amortises to nothing across back-to-back requests. If the deployment shape is concurrent
   prefill — which is the point of a disaggregated prefill server — this is free throughput that a
   single-request benchmark cannot see. See §7 for the measured numbers.
4. **Double-buffer the stage-to-stage hand-off.** A rank's receive is serialised with its compute.
   `inbound_socket_service_sync` is blocking with no async variant exposed, so this needs op-level
   work before the runner can use it — but it would remove item 1's cost from the critical path
   rather than merely shrinking it.
5. **Re-examine `nlp_concat_heads` head-grouping.** The TP=1 fix costs one extra pass over the
   activation on global layers (~0.3 ms each, ~31 ms over a 256k run — 0.3%). Feeding `o_proj` per
   head-group and summing the partial products would remove it entirely, at the cost of slicing a
   `DramShardedLinear` weight. Low value; listed for completeness.
6. **Wire migration for PP.** Turned off throughout this work (`PREFILL_ENABLE_MIGRATION=0`). The
   engine already publishes a per-rank `KvCacheStage`, but `build_kv_chunk_table` has not been
   checked for `first_layer_idx` awareness. Note the runner rejects `PREFILL_MOCK_MIGRATION` only
   when real migration is *off*; the CI PCC harness runs multi-rank with both enabled, which is the
   path §5.5 item 1 would use.
7. **Correctness — see §5.** The split itself is now proven bit-exact, but Gemma4 is still not
   wired into the multi-rank golden-KV gate, and the CP prefill path it sits on has no numerical
   reference of its own. §5.5 ranks what to build.

## 7. Concurrency measurements

Same pipeline, same 256k per request, varying how many are in flight:

| in-flight requests | tokens | span | tok/s | max stage | fill | bubble | bubble/chunk |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 262,144 | 11.64 s | 22,517 | 9.78 s | 0.67 s | 1.87 s | 58 ms |
| 2 | 524,288 | 22.66 s | 23,136 | 19.75 s | 1.20 s | 2.91 s | 45 ms |
| 4 | 1,048,576 | 43.85 s | **23,912** | 40.13 s | 0.74 s | 3.72 s | 29 ms |

**+6.2% from 1 to 4 in flight**, and it converges rather than keeps climbing. The reason is visible
in the last column: only the *fill* part of the bubble amortises. The ~18 ms inbound socket sync is
paid per chunk however many requests are running, so it survives concurrency untouched — 3.72 s of
bubble at 128 chunks is 0.74 s of fill plus 23 ms per chunk of floor.

That is the honest reading of item 3 in §5: concurrency is worth taking (1.17× → 1.24× against the
baseline, for free) but it is not a substitute for item 1. Fixing the socket sync would help every
shape; concurrency only helps the one that was paying for fill.

## 8. Running it

```bash
# once per machine: the [8,1] column -> device map is PER-GALAXY and a wrong map does NOT error,
# it builds four stages that are not columns and reports plausible, wrong numbers
python models/demos/gemma4/tests/perf/pp4/gen_gemma4_pp4_binding.py

# once per machine: build the TP=1 weight cache one layer window at a time. A single [8,1] mesh
# cannot hold all 60 layers at TP=1 (~29 GB/chip of weights), so it has to be done in pieces.
export GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 GEMMA4_WEIGHT_CACHE_MESH_ONLY=1
for w in "0 17 --first" "17 13" "30 17" "47 13 --last"; do set -- $w
  python models/demos/gemma4/tests/perf/pp4/run_stage.py --first-layer $1 --num-layers $2 ${3:-} \
    --max-seq 32768 --chunks 4
done
unset GEMMA4_PREFILL_LOAD_FULL_WEIGHTS

# the pipeline
RUN_TAG=pp4_256k PP_MAX_SEQ_LEN=262144 PP_USERS=1 PP_REQUESTS=1 \
  models/demos/gemma4/tests/perf/pp4/run_pp4_gemma4.sh

# correctness (§5): splitting the layer stack must not change the answer
python models/demos/gemma4/tests/perf/pp4/verify_pp_split.py --layers 30 --split 17 --chunks 3
python models/demos/gemma4/tests/perf/pp4/verify_pp_split.py --layers 17 --split 11 --chunks 3 --last
python models/demos/gemma4/tests/perf/pp4/verify_concat_heads.py     # seconds, no weights
```

`run_stage.py` also runs **one** stage in a single process with no MPI and no sockets, which is the
cheap way to bisect anything that breaks: point it at a layer window and it reports that stage's
per-chunk time and the global-layer indices it believes it owns.

Two rules that cost real time when broken: never run a device job in a foreground call with a
timeout (a killed wrapper is a SIGKILL mid-fabric, and the next mesh open dies on an ethernet-core
timeout — use `setsid nohup … &` and poll), and check weight-cache provenance before debugging any
hang, because a mesh-mismatched cache deadlocks in layer 0 with no diagnostic.
