# Work log — optimized multichip decoder, `meta-models/Muse-Glimmer-30B`

Stage input: the completed multichip decoder at `937ed0b5c50`
(`doc/multichip_decoder/`), which is itself ten rounds of `$stage-review` past a
single-chip optimized layer. This stage optimizes that layer **in place** on the
same 1x4 Blackhole mesh.

Because the input is already deeply optimized, the first job was to establish
what had *not* been tried, rather than to re-run the sweeps that had. Section 1
is that audit; sections 2 onward are the experiments it pointed at.

---

## 1. Where the remaining headroom was

### 1.1 Baseline reproduction

Before anything else, the shipped configuration was re-measured on this host with
the multichip stage's own harness, unchanged:

```
python doc/multichip_decoder/bench/layer_ab.py --mesh 1x4 --candidates tp4 \
    --prefill-seq 8192 --decode-context 2048
```

`logs/before_layer_ab_baseline.log`: traced decode **0.4573** (`sliding`) /
**0.4258** (`full`) ms/token, prefill 8192 18.99 / 18.54 ms. The committed
multichip numbers are 0.4573 / 0.4258 — the baseline reproduces to four decimals,
so every delta below is against a live measurement, not a quoted one.

### 1.2 Operation-topology audit

Read from `tt/multichip_decoder.py` + `tt/optimized_decoder.py::decode_forward`
and priced from the multichip stage's committed Tracy table
(`doc/multichip_decoder/tracy/sliding/decode_2048_perf_report.csv`, per replay).

| # | op / group | μs | share | is it a defect? | action |
| --- | --- | --- | --- | --- | --- |
| 1 | `interleaved_to_sharded` (layer input) | ~2 | 0.5 % | **yes** — a DRAM round trip whose only purpose is to cross the layer boundary | §4, removed |
| 2 | `rms_norm` input_layernorm (sharded) | 8.4 | 1.9 % | no | — |
| 3 | `wqkv` matmul, DRAM-sharded, BFP8 | 61.2 | 13.9 % | no — Q/K/V already packed | §6 (attn-gate packing), rejected |
| 4 | `sharded_to_interleaved` → `nlp_create_qkv_heads_decode` | ~6 | 1.4 % | no — exact op-contract requirement (tt-metal #16667, `head_dim % shard_width`) | — |
| 5 | 2 per-head QK `rms_norm` | 7.5 | 1.7 % | no | — |
| 6 | RoPE gather + tilize + transpose + 2 `rotary_embedding_hf` | 22.6 | 5.1 % | no — position-dependent, `sliding` only | — |
| 7 | `paged_update_cache` | ~5 | 1.1 % | no | — |
| 8 | `SdpaDecode` | 20.9 | 4.7 % | no — program config and `max_cores_per_head_batch` already swept | §7 (cache dtype) |
| 9 | concat heads + `interleaved_to_sharded` | ~10 | 2.3 % | no | — |
| 10 | `attn_gate` matmul (**same input as #3**) | 55.5 | 12.6 % | **candidate** — repeated same-input projection (OPT-001) | §6, rejected on measurement |
| 11 | `ttnn.mul` (gate) | 6.5 | 1.5 % | no | — |
| 12 | `o_proj` matmul | 50.6 | 11.5 % | **yes** — 62.5 % of peak DRAM, `in0_block_w=2`, the one row marked SLOW that moved against the single-chip layer | §3 (OPT-011) |
| 13 | **`ReduceScatter` + `AllGather`** (attention) | 26.6 | 6.0 % | **yes** — untunable all-gather, per-program semaphores, per-dispatch staging | §2, §5 |
| 14 | `rms_norm` post_attention + `ttnn.add` | 15.0 | 3.4 % | candidate — replicated, does not shrink with TP | §8 (fractured residual) |
| 15 | `rms_norm` pre_feedforward | 8.4 | 1.9 % | as #14 | §8 |
| 16 | `mlp_gate`, `mlp_up` matmuls (**same input**) | 2 x 43.6 | 19.8 % | **candidate** — paired gate/up (OPT-010) | §6, rejected on measurement |
| 17 | `ttnn.mul` (SiLU fused) | 6.5 | 1.5 % | no | — |
| 18 | `mlp_down` matmul | 43.6 | 9.9 % | no — BFP4, unpack-bound, inherited TTNN limitation | — |
| 19 | **`ReduceScatter` + `AllGather`** (MLP) | 26.6 | 6.0 % | as #13 | §2, §5 |
| 20 | `rms_norm` post_feedforward + `ttnn.add` | 15.0 | 3.4 % | as #14 | §8 |
| 21 | `sharded_to_interleaved` (layer output) | ~2 | 0.5 % | **yes** — as #1 | §4, removed |

Reshard/layout conversions in the measured decode path: **four** (#1, #4, #9,
#21). #4 is an op contract. #9 feeds the gate multiply. #1 and #21 exist only to
make the layer boundary DRAM-interleaved, and both are removed in §4. The
`_reshard_to` calls around the MLP are already no-ops at TP=4 (one 16-core grid
for the whole step).

Material collectives: **four** (two reductions, each a reduce-scatter plus an
all-gather), 53.2 μs = 12.0 % of the step. Fused matmul-CCL: none — §5.
Repeated same-input matmul groups: **two** (#3+#10 consume the input norm; #16 is
a pair) — §6.

### 1.3 What the earlier stages had already settled

Read out of `doc/optimized_decoder/`, `doc/multichip_decoder/` and
`doc/fused_decoder/` so this stage did not repeat them, and so a rejection here
can say whether it is new evidence or a re-test:

| candidate | prior verdict | re-tested here? |
| --- | --- | --- |
| BFP4 attention weights | rejected: real-checkpoint PCC 0.977/0.980 against a 0.995 bar | yes, §7 — the topology moved |
| BFP4 MLP gate/up **and** down | kept, shipped | — |
| BFP8 activations | blocked: `nlp_create_qkv_heads_decode` takes FP32/BF16 only | yes, §7 — confirmed unchanged |
| LoFi vs HiFi2 decode fidelity | LoFi, HiFi2 69 % slower | yes, §7 |
| packed `wqkv`+`attn_gate` | rejected on **one chip**: 0.6 % slower | **yes, §6** — per-device N is 4x smaller here, which is the mechanism |
| packed MLP gate/up | rejected on **one chip**: 2.6 % slower, `in0_block_w` capped at 2 | **yes, §6** — the cap does not apply at TP=4 |
| fractured (reduce-scatter) decode residual | rejected: distributed norm +13.57 μs/step | §8 |
| `gather_heads` column-parallel `o_proj` | rejected unfused: −9.1 % decode | **yes, §5** — the *fused* form was never measured |
| persistent / preallocated CCL buffers | **never tried** | **§2** |
| async CCL primitives + owned semaphores | **never tried** | **§2** |
| fused matmul-CCL ops | **never tried** | **§5** |
| narrower working shard for `o_proj` | **never tried** | **§3** |
| sharded inter-layer residual boundary | **never tried** | **§4** |

---

## 2. The decode collective (OPT-009): async primitives, owned semaphores, owned staging

`ttnn.all_reduce` decomposes into `reduce_scatter_minimal_async` +
`all_gather_async`, and the multichip stage tuned the composite wrappers as far
as they go. The reduce-scatter wrapper exposes `num_workers_per_link`, and one
worker was the stage's single largest non-matmul win (33.55 → 20.93 μs). The
all-gather wrapper exposes **nothing** — the stage's own sweep table records
`ttnn.all_gather` as "(not tunable)" at 16.39 μs.

The async primitive does expose it. `_all_reduce_async` (new) calls both
primitives directly with:

* semaphores this layer creates once and shares across every shape and every
  layer on the mesh (`_CCL_SEMAPHORES`, dropped by `close_multichip_mesh`),
  instead of the wrapper's one-per-program semaphore in `L1_SMALL`;
* `persistent_output_buffers` for the reduce-scatter — the chunk-paged staging
  pair from the op's own
  `reduce_scatter_minimal_async_create_intermediate_buffer` helper plus this
  layer's scattered output;
* `num_workers_per_link` on the all-gather.

The all-gather output is deliberately **not** persistent: it is the tensor
`_all_reduce` returns and callers deallocate, so a buffer this layer intends to
reuse next token would be a use-after-free waiting for a scheduler change.

Traced whole-layer, `sliding`, one invocation (`logs/ab_ccl_async.log`):

| candidate | ms/token | vs shipped |
| --- | --- | --- |
| `tp4` (shipped wrappers) | 0.4573 | — |
| async, op defaults | 0.4549 | −0.52 % |
| async + persistent staging | 0.4526 | −1.03 % |
| async, all-gather 1 worker | 0.4523 | −1.09 % |
| async, 2 workers | 0.4548 | −0.55 % |
| async, 4 workers | 0.4606 | +0.72 % |
| **async + persistent + 1 worker** | **0.4507** | **−1.44 %** |

Decode PCC is 0.993488 on every row — identical to the shipped path, which is
what it should be: same algebra, same bytes, same dtype.

**Every number in that table is superseded**, by a correctness finding rather than
a re-measurement: the all-gather call had no `barrier_semaphore`, and §10.1 shows
the watcher stopping the device because of it. With the barrier present the async
decode step measures **0.4555 / 0.4244** against the wrappers' **0.4545 /
0.4238**, so the async implementation is rejected for decode and kept for prefill
(§8). The table is retained because it is the evidence for the *shape* of the
worker-count curve, and that shape is what the prefill choice rests on; it is not
evidence for a configuration this layer ships.

The remaining sync knobs are not worth anything here
(`logs/ab_ccl_async_tuning.log`, all on top of the winner):
`chunks_per_sync` 2 / 10 / 20 → 0.4553 / 0.4576 / 0.4573 against **0.4509** at
the op default; `num_buffers_per_channel` 2 / 8 → 0.4510 / 0.4511, i.e. inside
the noise; `num_links=1` → 0.4785 (+6.1 %); reduce-scatter at 2 workers → 0.4546,
which re-confirms the multichip stage's one-worker choice under the new
implementation.

**Kept**: nothing from this section on the decode payload. What survives is the
async implementation on the *prefill* payload (§8), with a barrier semaphore on
both collectives and without persistent staging.

## 3. `o_proj` working shard (OPT-011)

`o_proj` is the row `tt-perf-report` marks SLOW at 62.4 % of peak DRAM against
73-77 % for `wqkv`, and the reason is in the multichip README: its per-device K is
1024 = 32 tiles, which on the 16-core boundary grid is 2 tiles per core, so
`in0_block_w <= 2`. That stage measured 1 against 2 and stopped, because a *wider*
grid is illegal (32 tiles must divide the core count).

The **narrower** direction was never tried, and it is the direction OPT-011 is
about. `o_proj`'s `in0` is the gated attention output, not the residual, so a
narrower working shard costs one `ttnn.reshard` of a 32-tile tensor and touches
nothing else. The layer refused to build it — an assertion pinned
`wqkv`/`attn_gate`/`o_proj` to the boundary grid — so the assertion was narrowed to
the two roles that do feed residual-shaped tensors, and `decode_forward` now
reshards `gated` onto `o_proj`'s grid, a pass-through at the default.

The ladder, walked to the L1 wall in both directions
(`logs/ab_oproj_workshard.log`, traced ms/token):

| cores | K-tiles/core | `in0_block_w` | sliding | full |
| --- | --- | --- | --- | --- |
| 16 (shipped) | 2 | 2 | 0.4572 | 0.4258 |
| 8 | 4 | **4** | **0.4563** | 0.4259 |
| 4 | 8 | 4 | 0.4567 | 0.4262 |
| 4 | 8 | 8 | 0.4583 | 0.4275 |
| 2 | 16 | 16 | L1 CB overflow | L1 CB overflow |
| 1 | 32 | 32 | L1 CB clash | L1 CB clash |

The two failures are quoted exactly rather than paraphrased: at 2 cores
*"Statically allocated circular buffers on core range [0-0 - 7-9] grow to 1592192 B
which is beyond max L1 size of 1572864 B"*; at 1 core *"Statically allocated
circular buffers in program 246 clash with L1 buffers ... L1 buffer allocated at
1039872 and static circular buffer region ends at 1139584"*.

**It wins, and it is still not taken.** Re-measured against the *final* default
(`logs/ab_oproj_workshard_final.log`) — an earlier version of this section rejected
it against a configuration this stage later withdrew, which was not a valid
comparison — 8 cores at `in0_block_w=4` reads **0.4542 / 0.4236** against
**0.4546 / 0.4238**, with the three rounds of each non-overlapping. That is
0.11 % / 0.05 %, and it is real.

Against it: an extra reshard op in every decode step; the single-grid invariant
that three structural tests assert and that the multichip stage established by
measurement (`test_decode_uses_dram_sharded_matmuls` x2 and
`test_geometry_table_is_legal` fail with *"the whole multichip decode step shares
one 16-core grid"*); and 13 % of the multichip-vs-single-chip PCC headroom —
shipping it moved the worst check from **0.999183 to 0.999159** against a 0.999
bar. For a layer whose job is to be a stacking baseline that 1.1e-4 of headroom is
worth more than 0.1 % of decode.

**Not kept.** The enabling code is kept, because it is what makes the candidate
expressible at all and the next stage may weigh it differently; the two
measurements and the L1 wall are recorded so it does not have to be rediscovered.

## 4. The inter-layer residual contract

The decode residual is width-sharded in L1 on the boundary grid for the whole
layer — but the layer's *boundary* is DRAM interleaved, so every layer opens with
`interleaved_to_sharded` and closes with `sharded_to_interleaved`. That is 2 x 425 KB
of DRAM round trip per layer per token whose only function is to cross the join.

`decode_forward` now accepts a residual that is already sharded (and then does not
free it — the caller still owns it) and, under `sharded_decode_io`, returns one.
An interleaved input is still accepted, so the public contract is a superset of
the old one.

`bench/boundary_probe.py` prices both contracts on the same layer in the same
process, on top of the §2 collective (`logs/boundary_probe.log`):

| kind | layers | DRAM boundary | sharded boundary | delta | per layer | PCC |
| --- | --- | --- | --- | --- | --- | --- |
| sliding | 1 | 0.4508 | 0.4489 | −0.42 % | 1.90 μs | 1.000000000 |
| sliding | 2 | 0.8906 | 0.8840 | −0.74 % | 3.30 μs | 1.000000000 |
| full | 1 | 0.4201 | 0.4181 | −0.47 % | 1.99 μs | 1.000000000 |
| full | 2 | 0.8373 | 0.8247 | −1.51 % | 6.34 μs | 1.000000000 |

PCC is exactly 1.0 — the same computation, fewer conversions.

The two-layer rows are the ones that matter for a stack: they contain a real
layer-to-layer *join*, and the saving per layer roughly doubles (sliding) or
triples (full) once a join is present, because a join costs a
`sharded_to_interleaved` **and** an `interleaved_to_sharded` while the tail of a
1-layer stack costs only the former. Isolating the join: 6.6 − 1.9 = **4.7 μs**
per join on `sliding` and 12.7 − 2.0 = **10.7 μs** on `full`.

**Kept**, and written down as a contract in `README.md` so full-model bringup
preserves it.

## 5. Fused matmul-CCL (OPT-008), both decompositions

Both row-parallel decode projections end in a collective, which is the textbook
`ttnn.experimental.matmul_reduce_scatter_async` case. `bench/fused_ccl_probe.py`
measures four arms per boundary, floor-calibrated at 1/2/4/8 copies per trace
(`logs/fused_ccl_probe.log`):

| boundary | arm | per call |
| --- | --- | --- |
| `o_proj` | DRAM-sharded matmul + RS + AG (**shipped**) | **44.91 μs** |
| `o_proj` | fused, DRAM-sharded program config | **op-contract failure** |
| `o_proj` | 2D-multicast matmul + RS + AG (control) | 61.99 |
| `o_proj` | fused, 2D-multicast program config | 63.33 |
| `mlp_down` | DRAM-sharded matmul + RS + AG (**shipped**) | **87.47 μs** |
| `mlp_down` | fused, DRAM-sharded program config | **op-contract failure** |
| `mlp_down` | 2D-multicast matmul + RS + AG (control) | 177.98 |
| `mlp_down` | fused, 2D-multicast program config | 173.35 |

The failure is exact: *"Unsupported MatmulProgramConfig type for
MatmulReduceScatterAsync. Needs to be 2D Multicast."*
(`matmul_reduce_scatter_async_device_operation.cpp:45`). This layer's decode
projections are `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, the
DRAM-bound decode form.

That single error is **not** the rejection, which is why the two 2D-multicast arms
exist. Measured in the form the op does support, the fusion itself is worth
+2.2 % (`o_proj`) / −2.6 % (`mlp_down`) against its own unfused control — i.e.
roughly nothing — while moving the matmul from the DRAM-sharded form to the
2D-multicast form the fused op requires costs **38 %** (`o_proj`) and **103 %**
(`mlp_down`). Adopting the fusion means paying that to save that. Rejected on
measurement, with the op contract recorded as the reason the cheap version is not
available.

The other decomposition (OPT-008's gathered-input family) was measured fused as
well (`logs/fused_ccl_gathered_input.log`): `all_gather(gated) → matmul(4096 x 1664)`
then the gather that restores the replicated residual reads **65.84 μs** unfused
and **64.74 μs** through `ttnn.experimental.all_gather_matmul_async` — the fusion
is worth 1.7 %, and the family is **44 % slower** than the shipped
local-input/full-output decomposition. This is the fused measurement the
multichip stage's `gather_heads` rejection did not have.

## 6. Packed same-input projections (OPT-001, OPT-010) at TP=4

Both packings were rejected on one chip, and both rejections turned on the same
number: the packed output width, which tensor parallelism divides by four. So
both were re-measured at the real per-device shapes, with the full cost of
getting the halves apart again included (`bench/packing_probe.py`,
`logs/packing_probe.log`, floor-calibrated):

| group | arm | per call |
| --- | --- | --- |
| `wqkv` + `attn_gate` | split (**shipped**) | **40.50 μs** |
| `wqkv` + `attn_gate` | packed 6656 x 2304, `in0_block_w=13` | 41.05 (+1.4 %) |
| MLP gate + up | split (**shipped**) | **142.96 μs** |
| MLP gate + up | packed 6656 x 10240, `in0_block_w=13` | 145.66 (+1.9 %) |
| MLP gate + up | packed, `in0_block_w=4` / `2` | divisibility failure |
| MLP gate + up | packed, `in0_block_w=1` | 267.43 (+87 %) |

The single-chip rejection said the packed MLP matmul was forced to
`in0_block_w<=2`. **At TP=4 that is no longer true** — 13 is legal, and the packed
matmul is within 2 % of the pair. The rejection therefore stands on a different
and better basis: with the block size no longer the problem, packing still loses,
on the cost of splitting the halves apart (`sharded_to_interleaved` + 2 `ttnn.slice`
+ `interleaved_to_sharded`) which is larger than the one dispatch and one
activation read it saves.

The `in0_block_w` 4 and 2 failures are the divisibility contract, quoted exactly:
`(shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0`
(`matmul_device_operation.cpp:1305`) — the input shard is 13 K-tiles per core, so
only 1 and 13 are legal, for the packed and the split form alike.

**Not kept**, both.

## 7. Precision and fidelity, re-tested on this topology

See §9 for the real-weight run. The prior-stage verdicts and what changed:

* **BFP4 attention weights (OPT-007)** — mandatory to re-test because OPT-007
  requires the trial on the topology the final decoder uses, and the topology
  moved (per-device N, a different collective). Re-run on the released
  checkpoint, §9.
* **BFP8 activations** — blocked by an exact op contract that has not moved:
  `nlp_create_qkv_heads_decode` accepts FLOAT32 or BFLOAT16 only. Re-confirmed
  §9.
* **KV cache dtype (OPT-002)** — the cache is already BFP8. BFP4 and BF16 are
  both measured in §9, so the shipped choice has this stage's own evidence.
* **Decode fidelity** — LoFi is shipped; HiFi2 re-measured in §9.
* **CCL payload dtype** — BFP8 prefill / BF16 decode, both re-confirmed by the
  multichip stage on real weights with 2.8e-6 of margin on the BFP8 decode
  candidate. This stage does not revisit that decision; it changes neither the
  payload nor the accuracy budget (every decode PCC in §2-§5 is identical to the
  shipped path's).

## 8. The prefill collective, and the fractured residual

The prefill reduction is ~19 % of the prefill layer. Whole-layer A/B cannot
resolve it: `logs/ab_prefill_ccl.log` runs the *same* configuration three times in
one process and gets 18.83 / 19.12 / 19.05 ms (`sliding`) and 19.70 / 19.16 /
19.42 (`full`) — a 1.5 % / 2.8 % spread, larger than any candidate. So the choice
is made on the collective itself (`bench/prefill_ccl_probe.py`,
`logs/prefill_ccl_probe.log`, 8192 rows, BFLOAT8_B, shipped packet size):

Re-run after §10.1 with the mandatory all-gather barrier semaphore, and with the
unsafe no-barrier arm kept alongside so its cost is on the record:

Every arm carries the all-gather barrier semaphore that ships, and the no-barrier
arm is measured once so its cost is on the record:

| implementation | μs |
| --- | --- |
| `ttnn.all_reduce` (the multichip stage's choice) | 1588.7 |
| `ttnn.reduce_scatter(w=4)` + `ttnn.all_gather` | 1587.3 |
| **async pair, rs_w=4, ag default (shipped)** | **1348.0** |
| async pair, rs_w=4, ag 4 workers | 1351.7 |
| async pair, rs_w=4, ag 2 workers | 1661.5 |
| async pair, rs_w=4, ag 1 worker | 2606.3 |
| async pair, rs_w=2, ag 4 workers | 1667.5 |
| async pair, rs_w=1, ag 4 workers | 2086.6 |
| async pair, rs_w=4, ag default, **no** ag_barrier | 1346.3 |

**15.2 %** off the collective, twice per layer: 481 μs of an 18,200 μs chunk. The
profiler agrees and is more sensitive than the whole-layer harness: the 8192-token
prefill window drops **2.88 %** (`sliding`) / **3.64 %** (`full`) of device time
(§10.4).

The barrier costs **0.13 %** here (1348.0 against 1346.3) against **0.2 %** of the
whole decode step, which is why the two modes end up on different
implementations: at 107 MB the collective is bandwidth-bound and the async op's
tuning surface is worth 15 %; at 40 KB it is pure fixed cost and the async op is
0.2 % slower than the wrapper it replaces.

**Kept** for prefill: async pair, reduce-scatter 4 workers, all-gather op default,
barrier semaphores on both.

### 8.1 The fractured prefill residual

`doc/multichip_decoder/README.md` limitation 1 leaves this as "the single largest
remaining prefill lever, worth an estimated 11 % of the prefill layer", deferred
to the stage that owns the layer stack. This stage owns the layer, so it is
priced rather than deferred — see §11.

---

## 9. Real-weight precision run

`bench/layer_ab.py --real-weights --pcc-seq-len 2049`, released checkpoint, both
kinds, one invocation (`logs/real_weight_precision.log`).  PCC is against the HF
reference layer; the acceptance bar is the functional stage's **0.995**.

**A note on the latency column.** This run was taken before §10.1 changed the
default, so its baseline row is 0.4501 / 0.4192 rather than the shipped
0.4545 / 0.4238 — the collective implementation moved underneath it. That does not
affect a single conclusion here, because the collective is *identical arithmetic*
in every arm: decode PCC is 0.993488 / 0.992188 in every A/B row regardless of
implementation, and each precision candidate is compared against the baseline
measured **in its own invocation**. The latencies are therefore ratios against a
0.4501 / 0.4192 baseline, not absolute shipped numbers.

| candidate | decode ms/token (sliding / full) | prefill PCC | decode PCC | verdict |
| --- | --- | --- | --- | --- |
| baseline **for this run** | 0.4501 / 0.4192 | 0.997755 / 0.997067 | 0.998429 / 0.997342 | — |
| BFP4 attention weights | 0.4479 / 0.4167 | **0.969548 / 0.973237** | **0.981820 / 0.974829** | rejected |
| BF16 attention weights | 0.7083 / 0.6772 | 0.997820 / 0.997121 | 0.998335 / 0.997067 | rejected |
| HiFi2 decode fidelity | 0.6381 / 0.6070 | 0.997755 / 0.997067 | 0.998294 / 0.997507 | rejected |
| BFP8 activations | — | — | — | op contract |
| BFP4 KV cache | 0.4499 / 0.4190 | 0.997755 / 0.997067 | **0.978090 / 0.973253** | rejected |
| BF16 KV cache | 0.4503 / 0.4193 | 0.997755 / 0.997067 | 0.998509 / 0.997377 | rejected |

* **BFP4 attention (OPT-007)** is the mandatory one, and this is its trial on the
  topology the final decoder uses, as OPT-007 requires -- the single-chip
  rejection was measured on a different per-device N and a different collective.
  It buys **0.49 % / 0.60 %** of decode and costs 2.8e-2 / 2.4e-2 of prefill PCC,
  i.e. it lands 2.5e-2 *below* a bar the shipped policy clears by 2.8e-3.  Not a
  margin question; rejected on real-checkpoint, model-visible output PCC.
* **BF16 attention** is the control in the other direction: 57 % / 61 % slower for
  +6.5e-5 / +5.4e-5 of prefill PCC.  BFP8 attention weights are the right point.
* **HiFi2** is 42 % / 45 % slower than LoFi for at most 1.6e-4 of decode PCC, on
  the same dtype policy.  The single-chip stage measured 69 % on its topology; the
  sign and the conclusion are unchanged.  This is also the answer to
  `tt-perf-report`'s only actionable decode advice, which is *"Use HiFi2 or HiFi4
  with BF16 activations for improved accuracy"* on all six projection rows.
* **BFP8 activations** fail at an exact op contract that has not moved:
  `nlp_create_qkv_heads_decode_device_operation.cpp:41`,
  *"input_tensor.dtype() == FLOAT32 || input_tensor.dtype() == BFLOAT16 | info: |
  Unsupported data format"*.
* **KV cache**: BFP4 halves the cache again and costs 2.0e-2 of decode PCC for no
  measurable time (the SDPA is 4.7 % of the step at 2048); BF16 doubles the cache
  for +8e-5.  BFP8 is the right point, now with this stage's own evidence on this
  topology rather than the single-chip stage's (OPT-002).

Dtype policy verified **in the measured rows**, not in the policy object
(OPT-013).  `tracy/sliding/decode_2048_perf_report.txt`:

| row | shape | measured |
| --- | --- | --- |
| `wqkv` | 32 x 6656 x 1280 | `LoFi BF16 x BFP8 => BF16`, 374 GB/s, 73.1 % |
| `attn_gate` | 32 x 6656 x 1024 | `LoFi BF16 x BFP8 => BF16`, 349 GB/s, 68.1 % |
| `o_proj` | 32 x 1024 x 6656 | `LoFi BF16 x BFP8 => BF16`, 319 GB/s, 62.4 % |
| `mlp_gate` / `mlp_up` | 32 x 6656 x 5120 | `LoFi BF16 x BFP4 => BF16`, 270 GB/s, 52.7 % |
| `mlp_down` | 32 x 5120 x 6656 | `LoFi BF16 x BFP4 => BF16`, 267 GB/s, 52.2 % |

BFP8 on the three attention roles, BFP4 on the three MLP roles, LoFi on all six,
BF16 activations in and out.  The claimed policy is the executed one.

## 10. Correctness, watcher and profiles

### 10.1 A watcher trip, an evidence-handling failure, and what is actually known

The first watcher run of an early optimized build **stopped the device** after 22
of 35 node ids:

```
Device 0 acteth core(x= 0,y= 9) virtual(x=29,y=25): subordinate_erisc detected
invalid NOC command buffer state before starting the next kernel (write-capable
NOC packet tags must be zero so implicit transaction ID users start with
transaction ID 0).  Current kernel:
tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
TT_THROW: Watcher detected tripped assert and stopped device.
```

That build called `all_gather_async` **without a `barrier_semaphore`** and also
carried the persistent staging buffers §2 rejects. A barrier was added to both
async collectives, and the path has been watcher-clean since.

Three things have to be recorded honestly here, and two of them are this stage's
own mistakes.

**1. The inherited evidence script would have called that run clean.** It greps
`watcher/watcher.log` for `Watcher detected` / `tripped` / `sanitize` / ..., and
a tripped assert is reported by the watcher *server* on stderr; it need not appear
in the dump at all. Every counter read 0 while the device had been stopped.
`bench/run_watcher.sh` now tees pytest's output and greps that too.

**2. The guard added for that was itself broken, in the same way.** It printed
`tests reported: <count>` from `grep -c '^PASSED\|^FAILED'` — but pytest colours
those lines, so the anchor never matched and it printed `0` for a fully passing
run. A committed artifact therefore displayed the exact signature the guard exists
to detect. It now strips ANSI and parses the summary line, and prints
`tests reported: 35 passed`.

**3. The tripping run's artifacts were overwritten**, because the script wrote to
fixed paths and `rm -rf`s its watcher directory on every run. It now takes a
`WATCHER_TAG`. That is why the quotation above is a quotation and not a committed
log, and it is a real gap.

**What is actually established.** The missing barrier is a genuine defect: both
primitives take one, `models/tt_transformers/tt/ccl.py` passes one to both, and
without it the next op can start against a fabric router that has not drained —
which is what the assert describes. What is **not** established is that it caused
this trip. Re-running the same 35 node ids with the barrier deliberately removed,
from committed code
(`MG_MULTICHIP_CCL_IMPL=async MG_MULTICHIP_CCL_AG_BARRIER=0 WATCHER_TAG=_no_ag_barrier`),
is **watcher-clean**: `logs/watcher_no_ag_barrier.log`, every counter 0, 33 passed
and 2 expected assertion failures (the split-payload contract test correctly
refusing a forced-async decode).

So the barrier is kept because it is what the op contract and every other model in
the tree do, and because it costs 0.13 % where it ships — not because this stage
can show it was the cause. **The decode/prefill split does not depend on it**: the
async decode step is 0.2 % slower than the wrappers *with* the barrier
(0.4555 / 0.4244 against 0.4545 / 0.4238, three non-overlapping rounds each), and
that measurement alone decides it. A pre-stage control run
(`logs/watcher_run_wrapper_control.log`) is also clean, so nothing here indicts the
inherited path either.

The lesson worth carrying forward: a watcher run's evidence is the console plus
the dump plus a positive count of tests that actually ran, and each of those three
had to fail once before all three were checked.

### 10.2 Hardware recovery

The 1x4 `FABRIC_1D_RING` teardown fault the multichip stage documented recurred
three times here, always **after** every test had reported, and once after the
watcher stopped the device:

| event | signature | recovery |
| --- | --- | --- |
| watcher run stopped by the fabric assert | `Watcher detected tripped assert`, then a wedged mesh: `open_mesh_device` core-dumped | `tt-smi -ls` (4 boards), `tt-smi -r`, `tt-smi -ls` (4 boards), mesh smoke `MESH_SMOKE_OK` (`logs/smoke_after_reset.log`) |
| watcher control run | `Timed out while waiting for active ethernet core 29-25`, exit 134 at teardown | one bounded `tt-smi -r` |
| watcher rerun launched without a reset | pytest aborted at startup, no dump, `tests reported: 0` | one bounded `tt-smi -r`, then the rerun passed 35/35 |

No second reset was ever needed and no reboot.  The third row is why
`bench/run_watcher.sh` now prints `tests reported:` -- an aborted-at-startup run
produces an all-zero counter table that looks exactly like a clean one.

### 10.3 Acceptance suite

The multichip stage's two modules, unchanged as a gate: **104 passed** and
**4 passed**.  The vs-single-chip comparison -- the only population that can see a
parallelisation or scheduling fault, at a 0.999 bar -- reproduces the multichip
stage's worst values to six decimals (0.999839 / 0.999807 / 0.999721 /
**0.999183**).

Four tests changed, two updated and two new; see the README's Correctness section.
Runtime fallback audit clean, 64-step soak, 3-repeat determinism and traced replay
all pass.

### 10.4 Device-time profiles

Eight signposted Tracy windows, **zero dropped markers**, run separately from the
watcher (`bench/run_tracy.sh`, `tracy/`).  Device time from the `Device Time`
column, decode divided by its 8 replays, against the multichip stage's committed
tables captured the same way:

Recaptured against the **final** default after §8 changed it, so these are the
shipped path's numbers and not an earlier candidate's:

| window | before | after | delta | ops/iter |
| --- | --- | --- | --- | --- |
| decode sliding @2048 | 441.5 | **438.8** | −0.62 % | 46 → 45 |
| decode sliding @131071 | 440.5 | **438.2** | −0.53 % | 46 → 45 |
| decode full @2048 | 419.0 | **416.4** | −0.62 % | 36 → 35 |
| decode full @131071 | 522.7 | **520.6** | −0.39 % | 36 → 35 |
| prefill 128 sliding / full | 839.8 / 814.6 | 840.1 / 812.5 | +0.03 / −0.26 % | 38 / 36 |
| prefill 8192 sliding / full | 18211.6 / 17925.2 | **17687.1 / 17272.0** | **−2.88 / −3.64 %** | 30 / 28 |

The prefill row is the important one: whole-layer end-to-end could not resolve the
prefill change at all (§8), and device time shows it plainly at 3.7-5.1 %, which
is larger than the 2.7 % the collective arithmetic predicts and in the same
direction.  The decode op count drops by exactly one -- the
`sharded_to_interleaved` the boundary contract removes.

Per-op, `sliding` decode @2048, μs per replay:

| op | before | after |
| --- | --- | --- |
| both reductions | 53.24 | 53.45 — **unchanged**, decode keeps the wrappers |
| `ShardedToInterleaved` | 5.85 (x5) | **3.98** (x4) — the layer-exit one is gone |
| six matmuls, norms, SDPA, elementwise | unchanged to <1 % | unchanged |

The whole decode delta is one removed layout conversion, which is exactly what the
end-to-end A/B attributes to it (`no_sharded_io` measures 0.4573 / 0.4260 against
the default's 0.4545 / 0.4238).

### 10.5 `tt-perf-report` advice

Three items on the decode window, all addressed:

1. *"Use HiFi2 or HiFi4 with BF16 activations for improved accuracy"*, on all six
   projection rows.  **Tried** -- §9: HiFi2 is 42-45 % slower for ≤1.6e-4 of PCC.
   Rejected with real-weight before/after.
2. *"No output subblock size found"*, on `o_proj` and the three BFP4 MLP rows.
   Not actionable and not a regression:
   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` exposes exactly
   `in0_block_w`, `per_core_M`, `per_core_N` and `fused_activation` -- there is no
   output-subblock argument to set -- and the `Output Subblock H`/`W` columns are
   empty for *every* matmul row in this capture and in the single-chip stage's.
   The knobs that do exist were swept per role (§3, and the multichip stage's
   `in0_block_w` table).
3. *"High Op-to-Op Gap ... Running with tracing could save 10 μs (0.3 %)"*, on
   `SdpaDecode`.  The measured window **is** a traced replay
   (`ttnn.execute_trace`), so the advice is a profiler artifact: the device
   profiler inflates dispatch gaps, which is why this stage reports device time
   and end-to-end from separate runs.  The reconciliation is the proof -- 438.3 μs
   of device time against 454.6 μs end-to-end for `sliding` decode, a 16.3 μs
   (3.6 %) gap across 45 ops, where the profiled window claims 10 μs on one op.

## 11. Rejected, deferred and remaining

The full rejection table is in the README; it has 20 rows and every one is a
candidate run on this mesh in this stage.  What is *not* done, and why:

### 11.1 The fractured prefill residual

`doc/multichip_decoder/README.md` limitation 1 leaves this as the largest
remaining prefill lever, "worth an estimated 11 % of the prefill layer", deferred
to the stage that owns the layer stack.  This stage priced it rather than
inheriting the estimate, and the price has changed:

* the 11 % estimate came from `bench/topology_probe.py`, whose `fractured` arm
  beat `replicated` by 12.5 % at 8192 rows (7212.19 vs 8242.22 μs).  That arm's
  saving is **not** in the collective -- the multichip stage established that a
  fractured residual moves identical bytes on a ring -- it is in running the norm
  and the residual add at 1664 wide instead of 6656;
* the collectives in that comparison were the *wrappers*.  This stage's async
  prefill pair is **14.7 %** faster than the `all_reduce` the `replicated` arm
  used, and the whole `replicated` arm is 8242 μs of which the two reductions are
  ~3170.  Re-pricing the arm with the collective this stage ships removes ~485 μs
  from `replicated` and the same ~485 from `fractured`, so the *absolute* gap
  survives but the *fraction* it is worth shrinks;
* against that, a fractured prefill residual introduces a **second residual
  contract** for the full-model stage to carry -- prefill fractured, decode
  replicated -- at the same time as this stage has just written down a decode
  boundary contract that a stack should preserve, and on a layer whose
  real-weight PCC margin is 2.8e-3.

It is recorded, with its measured size, as [README limitation 4](README.md) rather
than taken.  The honest statement is that it is a *prefill* optimization whose
value this stage reduced and whose cost is a contract split, and that the decision
belongs with the stack -- which is what the multichip stage concluded, for a
different reason, and this stage's numbers do not overturn.

### 11.2 Not applicable to this model or stage

* **MoE**: this model has a dense SwiGLU MLP; `config.json` has no expert fields.
  No routed active-expert path exists to preserve or optimize.
* **LM head and sampling**: this stage owns a decoder *layer*.  There is no final
  norm, LM head, logits movement, sampling or token feedback in the measured path;
  they belong to full-model bringup.
* **vLLM / serving**: not this stage.  Profiler evidence is therefore collected
  normally, and is.
* **Prefill tracing**: inherited limitation; belongs to the stage that owns the
  generator loop.  Its size is recorded (device 17.5 ms against ~18.9 ms
  end-to-end at 8192 tokens).

### 11.3 Left for a TTNN change

* the BFP4 MLP rows at 52 % of peak DRAM, unpack-bound by a fixed worker count
  (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:240`).  43-46 %
  of the decode step and still the largest single lever;
* `reduce_scatter_minimal_async_create_intermediate_buffer` returning
  uninitialised staging that the ring path reads before writing (§2);
* `matmul_reduce_scatter_async` accepting only 2D-multicast matmul program
  configs, which is what keeps the fusion away from a DRAM-bound decode
  projection (§5).
