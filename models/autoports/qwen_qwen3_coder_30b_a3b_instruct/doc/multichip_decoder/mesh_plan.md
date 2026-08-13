# Mesh plan — Qwen3-Coder-30B-A3B-Instruct on 4 Blackhole dies

Stage 03 design document. **Written before any of `tt/multichip_decoder.py`
exists**, as the goal contract requires ("Choose the target mesh/hardware
strategy before coding the final path"). Everything marked *measured* was run on
this machine during the design phase; everything marked *estimated* is
arithmetic on top of measured numbers and is flagged as such at the point of
use.

Baseline: `tt/optimized_decoder.py` at `0ba65b321c1` — prefill S=512
**69.12 µs/token**, traced decode ctx128 **0.5634 ms**, decode device time
**512.65 µs** (rows 45–103 of `../optimized_decoder/ops_perf_optimized_decode.csv`),
layer PCC 0.99870–0.99910.

---

## 0. The target hardware is not quite what the task assumed

`tt-smi` and the UMD cluster descriptor say this host is
**`ClusterType.P300_X2`**: two **p300** boards, two Blackhole dies each, four
dies total. The repo's `models/common/modules/tt_ccl._determine_device_name()`
labels any 4-die Blackhole host `"P150x4"` *purely by device count*, which is
where the name in the task comes from. Nothing in the plan below depends on the
label, but two physical facts do:

**It is a ring, not a line.** From the serialized cluster descriptor
(`ttnn.cluster.serialize_cluster_descriptor()`, archived at
`probes/cluster_descriptor.yaml`):

| hop | chips | links | kind |
|---|---|---|---|
| 0–1 | board `…2856` | 2 (chan 8/9 ↔ 3/2) | intra-board |
| 1–2 | across boards | 2 (chan 4/5 ↔ 4/2) | cabled |
| 2–3 | board `…2847` | 2 (chan 8/9 ↔ 3/2) | intra-board |
| 3–0 | across boards | 2 (chan 2/4 ↔ 5/4) | cabled |

So: a closed 4-ring with **two ethernet links on every hop**, and a 1×4
`MeshDevice` enumerates as device ids `[3, 2, 1, 0]` — adjacent in the mesh means
adjacent on the ring.

**The repo default gets this wrong.** `tt_ccl.default_topology()` returns
`ttnn.Topology.Linear` for any 4-device mesh (it only special-cases 8-device T3K
and Galaxy for `Ring`). Measured, that default costs a lot — see §5. The
multichip decoder must set `FabricConfig.FABRIC_1D_RING` before mesh open and
pass `Topology.Ring` with `num_links=2` explicitly rather than taking the
helper's default.

**DRAM.** Measured by allocating a single interleaved DRAM tensor per device on
the 1×4 mesh: 30 GB succeeds on every die (probe stopped there, it is not an
upper bound). 8 DRAM banks per die, 11×10 worker grid — identical to the
single-die baseline, so every program config in `optimized_decoder.py` that is
written against `_DRAM_BANKS = 8` keeps its meaning per die.

---

## 1. Model shape and the four "awkward numbers"

hidden 2048 · 32 Q heads · **4 KV heads** (8:1 GQA) · head_dim 128 ·
128 experts · **top-8** · moe_intermediate 768 · SwiGLU · 48 layers ·
vocab 151936, untied · context 262144. (Read back from
`AutoConfig.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")`.)

The task flagged four numbers as awkward. Three are fine and one is a real
constraint, but not the one expected:

| number | verdict |
|---|---|
| **4 KV heads / 4 dies = 1 each** | Fine, and **the DRAM-sharded attention config survives** — see below. This is the hard cap on the TP factor: TP=8 would need KV-head replication. |
| **128 experts / 4 = 32** | Clean. 32 whole experts per die, no padding. |
| **hidden 2048 / 4 = 512** | Clean: 16 tiles, and 512 is still a multiple of `8 banks × 32 = 256`, so even a hidden-sharded DRAM-sharded matmul would be legal. |
| **moe_intermediate 768 / 4 = 192** | The task says "not a multiple of 32". **It is** — 192 = 6 × 32, six whole tiles, no padding needed. The real reason not to split the intermediate is different and much more serious: it collapses expert-matmul core occupancy and caps `in0_block_w`. See §3.2. |

**The recommended scheme needs zero load-time padding.** Every division is
exact: 2048/4, 32/4, 4/4, 128/4, 151936/4 = 37984 = 1187 whole tiles. Padding is
therefore *not* a lever we have to spend here, and the contract's allowance for
it goes unused.

### 1.1 The one shape that had to be checked, not assumed

Splitting attention 4 ways gives each die **8 Q heads + 1 K head + 1 V head**, so
the fused QKV weight is `[2048, 8·128 + 1·128 + 1·128] = [2048, 1280]` and `wo`
is `[8·128, 2048] = [1024, 2048]`. `optimized_decoder._dram_sharded_ok()`
requires *both* dims divisible by `8 banks × 32 = 256`:

* wqkv: K 2048/256 = 8 ✓, N 1280/256 = 5 ✓ → `per_core_N = 5`
* wo: K 1024/256 = 4 ✓ → `in0_block_w = 4`, N 2048/256 = 8 ✓

**Measured** (`shape_probe.py`, single die, real program configs pulled out of
`optimized_decoder.py`, trace-slope timing, random weights):

| op | single-die shape | µs | TP=4 per-die shape | µs | ratio |
|---|---|---|---|---|---|
| qkv, DRAM-sharded | K=2048 N=5120 | 27.24 | K=2048 N=1280 | **9.48** | 2.87× |
| wo, DRAM-sharded | K=4096 N=2048 | 21.55 | K=1024 N=2048 | **8.79** | 2.45× |

Both ran, both produced correctly-shaped output, and `27.24 / 21.55` reproduce
the profiled `27.33 / 21.91`, which is what makes the TP-side numbers
trustworthy. The batch-32 cap (`num_users_supported = 32`) is unaffected by TP
and carries over unchanged.

**The whole 1-KV-head attention path is legal**, decode and prefill, measured
(`probes/probe3a.py`):

| op | single-die 32Q/4KV | TP=4 8Q/1KV |
|---|---|---|
| `nlp_create_qkv_heads_decode` | `q[1,32,32,128] k[1,32,4,128] v[1,32,4,128]` | ✅ `q[1,32,8,128] k[1,32,1,128] v[1,32,1,128]` |
| `paged_scaled_dot_product_attention_decode` | ✅ out `[1,1,32,128]` | ✅ out `[1,1,8,128]` |
| `nlp_create_qkv_heads` (prefill) | `q[1,32,512,128] k[1,4,512,128]` | ✅ `q[1,8,512,128] k[1,1,512,128]` |
| `scaled_dot_product_attention` (prefill, causal) | ✅ out `[1,32,512,128]` | ✅ out `[1,8,512,128]` |

`paged_update_cache` raised in *both* legs identically
(`paged_update_cache_device_operation.cpp:255: input_tensor.is_sharded()`) —
that is the probe handing it an interleaved update tensor, not a TP finding;
`attention_decode_optimized` already restores `kv_sharded_mem` before the call.
It is still worth re-checking at 1 KV head during implementation, since the
shard spec for a single head is what the probe did not exercise.

This is the single most load-bearing legality check in the plan, because if
`_dram_sharded_ok` had returned false at N=1280 the whole attention
optimization of stage 02 would have silently fallen back to interleaved on the
multichip path.

---

## 2. Candidate schemes

Notation: **TP** = tensor parallel (fracture a weight's rows/columns),
**EP** = expert parallel (fracture the expert dimension),
**DP** = data parallel (different users per die).
All per-die byte figures count block-float at true footprint —
bfloat4_b 0.5625 B/elem, bfloat8_b 1.0625 B/elem — as
`../optimized_decoder/perf_summary.json` established.

### Scheme A — pure TP=4 (everything fractured, experts on `moe_intermediate`)

| tensor | full | per-die | shard axis | padding |
|---|---|---|---|---|
| wqkv | [2048, 5120] bfp8 | [2048, 1280] | N, head-interleaved | none |
| wo | [4096, 2048] bfp8 | [1024, 2048] | K, by Q head | none |
| gate_up | [128, 2048, 1536] bfp4 | [128, 2048, **384**] | N (= 2·192) | none |
| down | [128, 768, 2048] bfp4 | [128, **192**, 2048] | K | none |
| router | [2048, 128] bf16 | replicated | — | — |
| KV cache | [B, 4, ctx, 128]×2 | [B, 1, ctx, 128]×2 | head | none |

Per-die weights/layer: 56.62 (gate_up) + 28.31 (down) + 5.57 (wqkv ×2 copies)
+ 4.46 (wo ×2) + 0.52 (router) = **95.5 MB** — identical to the recommendation,
because TP and EP fracture the expert weights by the same factor.

Collectives per layer: all-reduce after `wo`, all-reduce after `down`. Same as
the recommendation.

**Why it loses: the expert matmuls barely speed up.** `_sparse_matmul_config`
parallelises over N only, so N-tiles cap usable cores (this is the whole reason
stage 02 packed gate+up: 768→24 cores, 1536→48, 2048→64). Splitting the
intermediate takes packed gate/up from N=1536 (48 tiles, 48 cores) to N=384
(12 tiles, **12 cores** of the 8×8 = 64 the helper can address) — and since
`per_core_N = n_tiles // num_cores` is **1 in both cases**, the work per core is
literally unchanged; the matmul just uses a quarter of the grid. And `down`'s K falls
to 192 = 6 tiles, so `_tuned_sparse_matmul_config` must drop `in0_block_w` from
the tuned **12 to 6**, straight back up the block-width curve stage 02 spent its
largest win climbing down.

**Measured** (`shape_probe.py`, decode M=1, batch 1, bfp4 weights, LoFi,
DRAM outputs, trace-slope):

| configuration | gate_up | down | pair | vs baseline |
|---|---|---|---|---|
| E=128, I=768, nnz=8 — *single-die baseline* | 139.45 | 125.20 | 264.65 | 1.00× |
| **E=128, I=192, nnz=8 — scheme A** | 72.21 | 102.74 | **174.95** | **1.51×** |

1.51× from a 4× fracture. `down` gets 1.22× for a 4× reduction in weight bytes,
which is the `in0_block_w` regression showing up exactly where predicted.

### Scheme B — pure EP=4 (experts fractured, attention replicated)

| tensor | per-die | shard axis |
|---|---|---|
| gate_up | [**32**, 2048, 1536] bfp4 = 56.62 MB | expert dim |
| down | [**32**, 768, 2048] bfp4 = 28.31 MB | expert dim |
| wqkv / wo | full, replicated | — |
| KV cache | [B, **4**, ctx, 128]×2 | replicated |

Collectives: one all-reduce after the expert reduce. Attention needs none.

**Why it loses, decisively, on DRAM.** Replicated attention means a replicated
KV cache:

```
48 layers × 262144 tokens × 4 kv-heads × 128 × 2 B × 2 (K,V) = 25.77 GB per die
```

plus per-die weights 4.07 GB (experts) + 1.93 GB (full wqkv/wo ×2 copies ×48)
+ 0.02 GB (routers) + ~1.24 GB (embed + lm_head, replicated) = **7.26 GB**.
Total **33.0 GB against ~30 GB usable**. It does not fit at the advertised
context, and it also leaves attention's 113.9 µs of decode device time
completely unshared. Rejected on a hard physical limit.

### Scheme C — hybrid: TP=4 attention + EP=4 experts + replicated router  ✅ **recommended**

| tensor | full shape | dtype | mesh mapping | per-die shape | per-die bytes/layer |
|---|---|---|---|---|---|
| `input_layernorm` | [2048] | bf16 | replicate | [2048] | 4 KB |
| `wqkv` (prefill copy, interleaved) | [2048, 5120] | bfp8 | **shard N, head-interleaved** | [2048, 1280] | 2.785 MB |
| `wqkv_decode` (DRAM width-sharded, 8 banks) | [2048, 5120] | bfp8 | same | [2048, 1280] | 2.785 MB |
| `q_norm`, `k_norm` | [128] | bf16 | replicate | [128] | 512 B |
| K cache / V cache | [B, 4, ctx, 128] | bf16 | **shard KV head** | [B, 1, ctx, 128] | 512 B/token |
| `wo` (prefill copy) | [4096, 2048] | bfp8 | **shard K by Q head** | [1024, 2048] | 2.228 MB |
| `wo_decode` (DRAM width-sharded) | [4096, 2048] | bfp8 | same | [1024, 2048] | 2.228 MB |
| `post_attention_layernorm` | [2048] | bf16 | replicate | [2048] | 4 KB |
| `router` | [2048, 128] | bf16 | **replicate** | [2048, 128] | 0.524 MB |
| `gate_up_proj` | [1, 128, 2048, 1536] | bfp4 | **shard expert dim** | [1, 32, 2048, 1536] | 56.623 MB |
| `down_proj` | [1, 128, 768, 2048] | bfp4 | **shard expert dim** | [1, 32, 768, 2048] | 28.312 MB |
| **total per layer per die** | | | | | **95.48 MB** |

Shard specs / program configs, per die:

| config | single-die | per-die under scheme C |
|---|---|---|
| `_dram_sharded_program_config(qkv)` | `in0_block_w=8, per_core_M=1, per_core_N=20` | `in0_block_w=8, per_core_M=1,` **`per_core_N=5`** |
| `_dram_sharded_program_config(wo)` | `in0_block_w=16, per_core_M=1, per_core_N=8` | **`in0_block_w=4`**`, per_core_M=1, per_core_N=8` |
| `_width_sharded_l1` activation (qkv in) | [32, 2048/8=256] | unchanged, [32, 256] — input is replicated |
| `_width_sharded_l1` activation (qkv out) | [32, 5120/8=640] | [32, **160**] |
| `_width_sharded_l1` activation (wo in) | [32, 4096/8=512] | [32, **128**] |
| `_width_sharded_l1` activation (wo out) | [32, 2048/8=256] | unchanged, [32, 256] — output is a full-width partial |
| DRAM weight shard (qkv) | [2048, 640] × 8 banks | [2048, **160**] × 8 banks |
| DRAM weight shard (wo) | [4096, 256] × 8 banks | [**1024**, 256] × 8 banks |
| `_tuned_sparse_matmul_config` gate/up | M=1, N=1536, `in0_block_w=16` | **unchanged** (N and K are not fractured) |
| `_tuned_sparse_matmul_config` down | M=1, N=2048, `in0_block_w=12` | **unchanged** |
| `_decode_expert_memory_config` budget | 40 MB threshold vs `B × 29.4 MB` | intermediates are 4× smaller: `B × 7.34 MB`. The 40 MB constant must be **re-derived**, not inherited — see §7. |
| `EXPERT_CHUNK_SIZE`, `EXPERT_WEIGHT_DTYPE`, `EXPERT_MATH_FIDELITY`, `ATTENTION_WEIGHT_DTYPE`, `fp32_dest_acc_en=False` | | **all unchanged** |

Every tuned constant stage 02 measured on the expert path survives untouched,
because EP fractures the *batch* (expert) dimension of `sparse_matmul` and
leaves M, N and K exactly as they were. That is the central reason to prefer C
over A.

**The wqkv column split is a permutation, not a slice.** The checkpoint's fused
weight is `[Wq(4096) | Wk(512) | Wv(512)]`. A contiguous 4-way column split
would give die 0 nothing but Q heads. Die *d* must take Q heads `8d…8d+7`,
K head `d`, V head `d`, re-concatenated as `[Q_local(1024) | K_local(128) |
V_local(128)] = 1280` so `nlp_create_qkv_heads_decode(num_heads=8,
num_kv_heads=1)` reads it correctly. `wo`'s row split *is* contiguous once the Q
head assignment is fixed: die *d* takes rows `1024d … 1024d+1023`. Both are
load-time transforms in `upload_optimized_weights`'s successor.

---

## 3. MoE: keeping gate-selected active experts under EP

The contract requires the active-expert path, not dense. It survives, but there
is one non-obvious blocker.

### 3.1 The `nnz` contract forbids the obvious EP implementation

`ttnn.sparse_matmul` takes `nnz` as a **host scalar baked into the kernel as a
compile-time arg**, and
`ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:205-211`
spells out the contract: *"The op therefore requires `count_nonzero(sparsity) ==
nnz`; a mismatch deadlocks the device"* (tt-metal #45943). The sender kernel
asserts it on-device — loudly under watcher, as a **silent hang needing a board
reset** without.

A TTNN mesh op is SPMD: one program, one `nnz`, all four dies. Under EP the
number of locally-active experts is the number of the global top-8 that landed
in this die's 32-expert window — **data-dependent, and different on every die**,
anywhere in 0…8 with a mean of 2. There is no single correct `nnz`.

The escape is documented in the same file and in
`reader_bmm_tile_layout_in0_sender_padding.cpp:187`: pass **`nnz=None`**. Then
`get_batch_from_reader` is set, the sender reads the sparsity page at runtime and
multicasts a per-slot valid/ignore flag, and the receiver and compute kernels
follow. The loop runs over all `E_local` slots but only does weight reads and
math for the live ones.

### 3.2 Measured: EP is the right fracture, and dynamic `nnz` is affordable

Same harness as §2 (decode M=1, batch 1, bfp4/LoFi, DRAM outputs, trace-slope,
random weights):

| configuration | gate_up µs | down µs | pair µs | vs baseline |
|---|---|---|---|---|
| E=128, nnz=8 — **single-die baseline** | 139.45 | 125.20 | 264.65 | 1.00× |
| E=32, nnz=2 — EP, mean load, *exact nnz (illegal, see below)* | 35.39 | 32.83 | 68.22 | 3.88× |
| E=32, nnz=4 — EP, tail load, exact | 51.26 | 40.99 | 92.25 | 2.87× |
| E=32, nnz=8 — EP, worst case, exact | 82.72 | 58.21 | 140.93 | 1.88× |
| **E=32, `nnz=None` — EP, dynamic** ✅ | 60.67 | 63.29 | **123.96** | **2.13×** |
| E=128, `nnz=None` — dynamic at full E | 243.08 | 249.03 | 492.11 | 0.54× |
| E=128, I=192, nnz=8 — scheme A (TP experts) | 72.21 | 102.74 | 174.95 | 1.51× |

Two clean readings fall out:

1. **Dynamic-`nnz` overhead is linear in `E_local` at ≈ 0.8 µs per slot per
   matmul.** E=128: (243.08 − 139.45)/128 = 0.81; (249.03 − 125.20)/128 = 0.97.
   E=32: (60.67 − 35.39)/32 = 0.79; (63.29 − 32.83)/32 = 0.95. So dynamic mode
   costs ≈ 28 µs per matmul pair at E=32 and would cost ≈ 114 µs at E=128 —
   which is exactly why EP is what makes dynamic `nnz` affordable at all. The
   two decisions are coupled.
2. **EP (2.13×) beats expert-TP (1.51×) even paying that overhead**, and it does
   so while leaving every stage-02 expert program config untouched.

A capacity-padding scheme — fix `nnz = C`, pad the local sparsity to exactly `C`
live slots with zero-weight dummies — is faster whenever `C` is small
(C=2 → 3.88×). It is **rejected on correctness**: the only value of `C` that can
never be exceeded is 8, and C=8 measures 1.88×, worse than dynamic's 2.13×. Any
smaller `C` requires dropping experts on the tail, which changes the model's
output. Recorded in §6.

### 3.3 Prefill gets to keep exact `nnz`

Prefill's sparsity is per 32-token *tile*, and with 32 tokens × top-8 = 256
selections over 128 experts, essentially every expert is live — the profile
reads `active=128/128`. That is a limitation of the op (stage 02 measured the
per-token alternative at **2.1× slower**), but under EP it becomes an asset:
every die has all 32 of its local experts live, deterministically, so
`nnz = 32 × group_size` is **exact and identical across dies**. Prefill uses the
fast path; only decode needs dynamic mode.

### 3.4 The router stays replicated, and it is the thing that limits decode

Top-8 of 128 needs a global view of the logits, so the router matmul
(`[.,2048]×[2048,128]`, N = 4 tiles → 4 cores) and `ttnn.topk` (a single 128-wide
row → **1 core**) cannot be fractured usefully; splitting N four ways would give
each die one tile and one core. Each die therefore computes the full 128-way
routing on the replicated normed activation and slices its own 32-expert column
window out of the dense scatter. **No collective is needed for routing** — which
is the good news — but the 88.86 µs the router block costs on one die costs
88.86 µs on four.

**Correctness risk this creates:** all four dies must agree on the top-8, or the
union of the four local windows is not the global top-8 and the output is
silently wrong. The inputs are bit-identical (a replicated tensor through an
identical program), so `ttnn.topk` must return identical indices — but this is a
*tie-breaking determinism* assumption, and it needs an explicit test rather than
an argument. See §7.

---

## 4. Recommendation and expected performance

**Adopt scheme C**: attention TP=4 (8 Q heads, 1 KV head, 1 KV-cache head per
die), experts EP=4 (32 experts per die, `nnz=None` in decode / exact `nnz` in
prefill), router + both RMSNorms + residual **replicated**, `FABRIC_1D_RING` /
`Topology.Ring` / `num_links=2`, two all-reduces per layer.

### 4.1 Why the residual stays replicated

The skill asks for a topology table before choosing. Both consumers of the
residual — `wqkv` and the router+`gate_up` pair — need the **full 2048-wide**
hidden. So:

| contract | collective sequence per layer | primitive hops |
|---|---|---|
| **replicated residual** (recommended) | AR after `wo`; AR after `down` | 2 all-reduces |
| hidden-sharded residual (512/die) | RS after `wo`; AG before router/gate_up; RS after `down`; AG before next `wqkv` | 2 RS + 2 AG = the *same* traffic |
| hidden-sharded + distributed RMSNorm | as above **plus** a stats all-gather per norm | strictly worse |

An all-reduce *is* an RS followed by an AG, so the sharded-residual contract
recreates exactly the communication it is supposed to avoid, and then adds
distributed-norm stats gathers on top. This is the case the skill warns about
("a path that does reduce-scatter → all-gather immediately... mostly recreates
the communication the fused path was meant to avoid") — here it is the *sharded*
variant that has that shape, because hidden is only 2048 and both matmuls are
column-parallel over the whole of it. Replicated wins, and it also keeps the two
RMSNorms local and keeps the decoder's input/output layout a plain replicated
`[1, 1, B, 2048]`, which stacks across 48 layers with no boundary conversion.

`ttnn.experimental.all_gather_matmul_async` / `matmul_reduce_scatter_async`
would fuse a collective into a neighbouring matmul. Not evaluated in this phase
— flagged as the first optimization to try in implementation (§7), not as part
of the baseline.

### 4.2 Decode: expected ≈ 1.6×, and why it is not 4×

Scaling the profiled single-die decode budget (512.65 µs) term by term. *Ratios
in the "scaling" column are measured; the resulting per-die µs are estimates.*

| block | single-die µs | scaling | scheme C µs |
|---|---|---|---|
| `input_layernorm` | 20.04 | replicated, 1.00× | 20.04 |
| attention projections (qkv + wo) | 49.24 | **measured 2.87× / 2.45×** | 18.27 |
| attention body (head split, 2 per-head norms, 2 RoPE, 2 paged_update_cache, SDPA, concat, layout moves) | 64.63 | ~4× on work, launch-floor bound | ~30 |
| **all-reduce (attention out)** | — | **measured** | **19.96** |
| residual add | 2.00 | 1.00× | 2.00 |
| `post_attention_layernorm` | 20.19 | replicated, 1.00× | 20.19 |
| router block (matmul, topk, scatter, normalise) | 88.86 | **replicated, 1.00×** | 88.86 |
| expert `sparse_matmul` pair | 92.07 | **measured 2.13×** | 43.2 |
| expert reshape/eltwise tail (M-padding compaction) | 173.87 | ~4× (scales with E) + floors | ~55 |
| **all-reduce (expert out)** | — | **measured** | **19.96** |
| residual add | 1.76 | 1.00× | 1.76 |
| **total device** | **512.65** | | **≈ 319** |

**≈ 1.61×, 40% parallel efficiency.** End-to-end traced replay should land near
**0.37 ms** against the baseline's 0.5634 ms (the ~51 µs of host dispatch inside
a blocking replay does not scale).

The reason is Amdahl, and it is worth stating plainly because it is the main
thing that might make someone want a different plan:

> **129.1 µs of the 512.65 µs single-die decode layer — 25.18% — is replicated
> work that no parallelisation on this mesh can remove**: the two RMSNorms
> (40.23 µs, rows 45 and 68) and the router block (88.86 µs, rows 69–88). Both
> are latency-bound, not
> bandwidth-bound — a `[1,1,32,2048]` bf16 RMSNorm is 128 KB in 20 µs, i.e.
> 6.5 GB/s, and `topk` over a 128-wide row occupies one core. Even at infinite
> dies the decode ceiling is **3.97×**. At four dies, with 40 µs of collective
> added, 1.6× is close to what this decomposition can give.

> *Arithmetic correction, applied after stage-03 review; the plan is otherwise
> as written.* This paragraph originally read "132.9 µs — 25.9% — … ceiling
> 3.9×". Its own two components, which are unchanged and are cells of
> `../optimized_decoder/ops_perf_optimized_decode.csv`, sum to 40.234 + 88.858 =
> **129.092**, which is **25.18%** of 512.655 and a ceiling of **3.97×**. The
> conclusion the plan drew from it does not move.

Sharding the norms does not help: a `[1,1,32,512]` RMSNorm is not four times
faster than a `[1,1,32,2048]` one at these sizes, and it would add a stats
all-gather whose latency floor (≈ 11 µs, §5) exceeds any saving.

### 4.3 Prefill: expected ≈ 3.6×

Prefill's expert path is dense-all-expert per 32-token tile, so EP fractures it
with **zero load imbalance and exact `nnz`** — the best case for this mesh. From
the S=512 profile (35.23 ms/layer, `SparseMatmul` 71.8%, Unary 9.6%, Binary
6.5%, Permute 4.0%, Slice 3.4%):

| block | single-die ms | scaling | scheme C ms |
|---|---|---|---|
| `SparseMatmul` | 25.30 | 4.00× (exact E fracture) | 6.33 |
| expert-path eltwise/permute/slice | 8.28 | ~4× | 2.07 |
| attention + router + norms | ~1.65 | ~1.3× | ~1.3 |
| **2 × all-reduce at S=512** | — | **measured 76.85 µs each** | **0.154** |
| **total** | **35.23** | | **≈ 9.85** |

**≈ 3.58×**, i.e. **≈ 19.3 µs/token** at S=512 against 69.12. Collectives are
1.7% of the layer. Prefill is where this mesh pays.

### 4.4 What dominates, in one line each

* **Decode**: the replicated router (28% of the multichip layer) first,
  collectives (13%) second, the expert M-padding tail (17%) third. The M-padding
  that dominated single-die decode is *reduced* 4× by EP — it stops being the
  top line.
* **Prefill**: compute, overwhelmingly. Collectives are noise.
* **Capacity**: see §8 — this is the real reason the mesh is required.

---

## 5. Measured collective costs

Method: 1×4 mesh, bf16, DRAM interleaved in and out, `num_links` as stated,
persistent global semaphores created before capture. Cost per op is a
**trace slope** — median-of-30 blocking replay of a 17-op trace minus a 1-op
trace, divided by 16 — which removes the ≈ 57 µs host-dispatch floor that would
otherwise swamp every decode-sized measurement. Script:
`probes/ccl_probe2.py`.

**`all_gather_async` (dim 3), µs per op:**

| per-device input | Linear ×1 | Linear ×2 | Ring ×1 | **Ring ×2** |
|---|---|---|---|---|
| 32×512 (32 KB) — *decode* | 16.65 | 14.29 | 12.20 | **11.76** |
| 128×512 (128 KB) | 33.71 | 24.05 | 22.55 | **16.84** |
| 512×512 (512 KB) | 90.51 | 53.08 | 53.70 | **34.63** |
| 2048×512 (2 MB) | 323.86 | 168.73 | 173.54 | **94.42** |

**`reduce_scatter_minimal_async` (dim 3), µs per op:**

| per-device input | Linear ×1 | Linear ×2 | Ring ×1 | **Ring ×2** |
|---|---|---|---|---|
| 32×2048 (128 KB) — *decode* | 16.62 | 14.19 | 11.98 | **10.83** |
| 128×2048 (512 KB) | 31.43 | 22.20 | 22.01 | **15.46** |
| 512×2048 (2 MB) | 90.20 | 53.99 | 66.56 | **42.68** |
| 2048×2048 (8 MB) | 313.62 | 170.81 | 154.16 | **104.90** |

Findings:

* **Ring + 2 links is 1.21× better than Linear + 2 links at decode size and
  1.79× at 2 MB**, and 2.8×/3.4× better than the Linear + 1 link that a naive
  setup would get. Since `tt_ccl.default_topology()` returns `Linear` for a
  4-device mesh, this must be overridden explicitly.
* **Below ~128 KB per device the collective is pure latency**: ~11 µs floor,
  essentially flat from 32 KB to 128 KB. Decode lives entirely in this regime,
  so decode collective cost is *per-call*, not per-byte. Reducing the number of
  collectives matters; reducing their size does not.
* At 2 MB, ring AG achieves 6 MB inbound / 94.42 µs ≈ **63.5 GB/s** per die.

**All-reduce, the two ways to build it** (`probes/ar_probe.py`, Ring ×2):

| shape | RS + AG | AG(dim 0) of partials + local `sum` |
|---|---|---|
| `[1,1,32,2048]` — decode | 23.69 µs | **19.96 µs** ✅ |
| `[1,1,512,2048]` — prefill | **76.85 µs** ✅ | 121.72 µs |

Mode-dependent, and both are cheap to implement: **decode uses AG-of-partials**
(one latency-bound collective beats two), **prefill uses RS+AG** (bandwidth
wins once the payload is large). `ttnn.experimental.all_reduce_async` raised on
every argument spelling tried and was **not** successfully measured — flagged as
a guess-free item to retry in implementation.

Collective volume, per token per layer, decode, bf16:

* attention all-reduce: 131,072 B of logical payload per die; AG-of-partials
  moves 3 × 131,072 = **393,216 B inbound per die**
* expert all-reduce: identical
* **786,432 B moved per die per layer**, 39.92 µs, **1.92 ms/token over 48
  layers** — a real cost, and the reason §7 lists collective fusion as the top
  follow-up.

---

## 6. Rejected alternatives

| # | alternative | evidence | why rejected |
|---|---|---|---|
| 1 | **Pure TP=4, experts split on `moe_intermediate`** (scheme A) | measured: expert pair 174.95 µs vs EP's 123.96 and baseline 264.65 | 1.51× vs EP's 2.13×. `_sparse_matmul_config` spreads over N only, so N=1536→384 drops usable cores 48→12 with no change in per-core work; `down`'s K=192 forces `in0_block_w` 12→**6**, undoing stage 02's largest single lever. Note 192 *is* tile-aligned — the problem is occupancy, not padding. |
| 2 | **Pure EP=4, attention replicated** (scheme B) | 25.77 GB replicated KV + 7.26 GB weights = **33.0 GB > ~30 GB usable** | Hard DRAM limit at the advertised 262144 context, and leaves 113.9 µs of attention unshared. |
| 3 | **Data parallel (one user per die)** | per-die weights 19.50 GB + per-die KV 25.77 GB = **45.3 GB** | Fails DRAM by more than scheme B, and gives exactly 0% single-user latency improvement. |
| 4 | **2D 2×2 mesh (TP=2 × EP=2)** | per-die: experts 169.9 MB/layer, 2 KV heads → 12.9 GB KV, total ≈ 22.9 GB — *fits*, but expert block only /2 and attention only /2 | Still needs two collectives per layer (one per axis), and decode collectives are latency-bound at ~11 µs regardless of device count, so a 2-device collective saves a few µs at best while giving up half the compute fracture. Strictly worse on a mesh this small. *(2-device collective latency estimated from the flatness of the size sweep, not separately measured — see §9.)* |
| 5 | **Capacity-padded exact `nnz` under EP** (`nnz=C`, pad local sparsity to C live slots) | measured: C=2 → 68.22 µs (3.88×), C=4 → 92.25, **C=8 → 140.93 (1.88×)** | Faster than dynamic only for C < 8; but the local active count is data-dependent in 0…8, so C=8 is the only value that can never be exceeded, and C=8 is *slower* than dynamic `nnz=None` (123.96). Any smaller C means dropping experts, which changes the model output. |
| 6 | **Dense all-expert decode** (all 32 local experts every token) | equivalent to E=32 `nnz=32`; E=32 `nnz=8` already measures 140.93 µs | Slower than gate-selected dynamic (123.96) and forbidden by the contract absent evidence it is faster. It is not. |
| 7 | **Hidden-sharded residual + distributed RMSNorm** | §4.1 topology table; measured RS 10.83 / AG 11.76 at decode size | Both residual consumers need the full 2048, so the sharded contract needs 2 RS + 2 AG = the same traffic as 2 all-reduces, *plus* per-norm stats gathers. No saving to buy the extra complexity with. |
| 8 | **`Topology.Linear` (the repo default for a 4-device mesh)** | measured, §5: 14.29 vs 11.76 µs at decode; 168.73 vs 94.42 at 2 MB | 1.2–1.8× slower. The cluster descriptor shows a genuine 4-ring; `tt_ccl.default_topology()` only returns `Ring` for 8-device T3K/Galaxy and must be overridden. |
| 9 | **`num_links=1`** | measured, §5 | 1.04× slower at decode (latency-bound), **1.84× slower at 2 MB** — i.e. free money in prefill. Both links are present on every hop. |
| 10 | **Sharding the router / top-k** | `topk` over a 128-wide row is 1 core; router matmul N=128 = 4 tiles = 4 cores | Fracturing N four ways gives 1 tile and 1 core per die. Top-8-of-128 also needs the global logit vector, so a fractured router would need a collective *inside* the routing path. Strictly worse. |
| 11 | **TP=8 (splitting KV heads below 1)** | 4 KV heads is the hard cap | Would require KV-head replication and takes wqkv's N to 640, which is not divisible by `8 banks × 32 = 256` — `_dram_sharded_ok` returns false and stage 02's DRAM-sharded attention path silently disappears. Not reachable on this mesh anyway; recorded because it is the natural "what about more dies" question. |

---

## 7. Risks

Ordered by how much damage they do if they turn out to be true.

1. **`sparse_matmul` `nnz` mismatch is a silent device hang.** The single most
   dangerous item. Under EP the live-expert count is data-dependent, so decode
   *must* pass `nnz=None`; any regression to an exact `nnz` computed on the host
   will deadlock the board on the first token whose routing is unbalanced, and
   without `TT_METAL_WATCHER` it hangs rather than asserting
   (`sparse_matmul_device_operation.cpp:205-211`, tt-metal #45943). Mitigation:
   run the whole EP bring-up under the watcher; pin the `nnz=None` choice with a
   test that forces a maximally unbalanced routing (all 8 experts inside one
   die's window) and a maximally empty one (0 experts on some die); keep
   `tt-smi -r` in the loop.
   **The `E_local = 0` case deserves its own test** — a die where none of the
   global top-8 landed must contribute an exact zero to the all-reduce, not
   garbage from an uninitialised output buffer.
2. **Top-k determinism across dies.** Scheme C relies on four dies independently
   computing the *same* top-8 from bit-identical replicated logits. If
   `ttnn.topk` ever breaks ties non-deterministically, the four 32-expert
   windows stop being a partition of the global top-8 and the layer is silently
   wrong — no shape error, just PCC drift. Must be asserted directly (compare
   the four dies' `top_indices` tensors for exact equality across many random
   inputs), not argued from "same program, same input".
3. **The DRAM-sharded attention config's width assumptions.** They survive at
   TP=4 (verified by *running* the op at N=1280 / K=1024, §1.1), but they
   survive by exactly one factor of 2: `_dram_sharded_ok` needs divisibility by
   256, and wqkv's per-die N=1280 = 5×256. `_width_sharded_l1` also hardcodes a
   32-row shard and `per_core_M=1`, which is what caps this path at batch 32.
   Anything that changes the head split — a different TP factor, a padded head
   count, KV replication — has to re-check `_dram_sharded_ok` or it will
   silently fall back to interleaved and quietly give back stage 02's 1.11×.
4. **`_DECODE_EXPERT_L1_BUDGET_BYTES = 40 MB` is inherited from a different
   problem.** Stage 02's own comment says the threshold is "asserted, not
   measured". Under EP the intermediates are `batch × 32 experts × 32 rows ×
   (1536 + 2048) × 2 B = batch × 7.34 MB`, so the constant now admits batch 5
   rather than batch 1 — a behaviour change nobody chose. It must be re-derived
   against the *multichip* L1 budget, which is also smaller than stage 02's
   because fabric and CCL persistent buffers now live in L1 too.
5. **Prefill attention with 1 KV head is checked but the prefill program config
   is not.** `nlp_create_qkv_heads` / prefill SDPA at 8Q/1KV was probed (§9);
   the interleaved prefill `ttnn.linear` on a `[2048,1280]` weight was not
   separately timed, though attention is 0.8% of prefill so this is low-value.
6. **Trace + CCL interaction.** Global semaphores and persistent CCL buffers
   must be allocated before `begin_trace_capture`, and nothing may allocate
   inside the trace. The `_ONES_COLUMN` cache in `router_forward_optimized`
   already encodes this pattern for one tensor and will need the same treatment
   per-mesh; note its `id(device)` key must now key on the *mesh*, and the
   comment about CPython address reuse applies with equal force.
7. **Non-aligned sequence lengths.** Low risk but must be shown, not assumed:
   the collectives run on dim 3 (hidden), which is 2048 and independent of S, so
   the existing internal chunk-padding in `moe_prefill_optimized` is the only
   padding in play and the public contract is unchanged. The multichip test must
   still cover S = 33, 100, 257 as stage 02 does.
8. **Watcher cost on 4 dies.** `TT_METAL_WATCHER` inflates device timings ~8×
   on one die; on four with fabric traffic it will be worse. The stage-02 rule
   — never run the watcher over the perf tests, they rewrite the published CSVs
   — carries over and now matters more.

---

## 8. Full-model feasibility at 48 layers — the number that justifies the mesh

Per-layer weights, single die, at the shipped dtypes:

```
gate_up  128 × 2048 × 1536 = 402,653,184 elem × 0.5625 B = 226.49 MB
down     128 ×  768 × 2048 = 201,326,592 elem × 0.5625 B = 113.25 MB
wqkv           2048 × 5120 =  10,485,760 elem × 1.0625 B =  11.14 MB  ×2 copies
wo             4096 × 2048 =   8,388,608 elem × 1.0625 B =   8.91 MB  ×2 copies
router         2048 ×  128 =         262,144 elem × 2 B =   0.52 MB
                                                    total = 380.37 MB / layer
```

Embeddings (untied): `embed_tokens` and `lm_head` are each
151936 × 2048 = 311,164,928 params = 622.3 MB at bf16.

| | one die (no mesh) | scheme C, per die |
|---|---|---|
| decoder weights, 48 layers | 380.37 MB × 48 = **18.26 GB** | 95.48 MB × 48 = **4.58 GB** |
| embed (replicated) + lm_head (column-parallel /4) | 1.24 GB | 0.62 + 0.16 = **0.78 GB** |
| **weights total** | **19.50 GB** | **5.36 GB** |
| KV at 262144, batch 1 | 48 × 262144 × 2048 B = **25.77 GB** | 48 × 262144 × 512 B = **6.44 GB** |
| **total** | **45.3 GB** | **11.80 GB** |
| against ~30 GB usable | ❌ **does not fit** | ✅ **18.2 GB spare** |

**One die cannot hold this model at the advertised context** — it runs out at
≈ 107,000 tokens with nothing left for trace or activations. The 4-die mesh is
therefore a capability requirement, not only a speed one, and scheme C clears it
with 60% of DRAM free.

### 8.1 Proposed `context_contract.json` update

**No capability reduction.** `current_supported_context` stays **262144**,
`capability_reduction` stays `false`. The multichip section should record:

```jsonc
"device": { "arch": "blackhole", "board": "p300 ×2 (ClusterType.P300_X2)",
            "mesh": "1x4", "dram_per_die_gb": 32, "usable_dram_per_die_gb": 30,
            "topology": "ring, 2 eth links per hop" },
"parallelism": { "attention": "TP=4", "experts": "EP=4",
                 "router": "replicated", "residual": "replicated" },
"per_die": {
  "kv_bytes_per_token_per_layer": 512,        // 1 kv head × 128 × 2 B × 2 tensors
  "kv_bytes_at_full_context_all_layers": 6442450944,
  "weight_bytes_all_layers": 5360000000,      // decoder 4.58 GB + embed/lm_head 0.78 GB
  "total_at_full_context_batch1_gb": 11.80,
  "headroom_gb": 18.2
},
"largest_feasible": {
  "context_at_batch1": 262144,                // HF-advertised, not a limit here
  "batch_at_full_context": 3,                 // (30 − 5.36 − ~2 trace) / 6.44 GB
  "context_at_batch32": 28788                 // ~22.6 GB / (32 × 48 × 512 B)
}
```

The `forward_looking_note` in the current file — "the full 48-layer model would
need ~24 GiB on a 32 GiB die, so stage 05 will have to weigh KV dtype, paging
across dice, or a served-context cap" — is **discharged by this plan**: TP=4
across four dies removes the problem without touching KV dtype or the served
context. That note should be rewritten rather than deleted, since it is the
question this stage answers.

*This file is deliberately not edited in the design phase.* The values above are
the proposal for review; they go into `doc/context_contract.json` when
`multichip_decoder.py` exists and the per-die footprint is measured rather than
computed.

---

## 9. Probe inventory

Everything measured for this document, so a reviewer can rerun it. All probes
are small; none of them implements any part of the decoder.

| probe | what it establishes | script |
|---|---|---|
| cluster descriptor dump | P300_X2, 4-ring, 2 links/hop, mesh order [3,2,1,0] | `ttnn.cluster.serialize_cluster_descriptor()` |
| DRAM capacity sweep | 30 GB single allocation succeeds per die on the 1×4 mesh | inline |
| AG / RS size × topology × links sweep | §5 tables; ring beats linear; ~11 µs latency floor | `probes/ccl_probe2.py` |
| all-reduce composition | AG-of-partials wins at decode, RS+AG wins at prefill | `probes/ar_probe.py` |
| expert `sparse_matmul` E / `nnz` / intermediate-split sweep | §3.2 table — the EP-vs-TP decision | `probes/shape_probe.py` |
| TP=4 attention shape legality + timing | `_dram_sharded_ok` holds at N=1280 / K=1024; `nlp_create_qkv_heads_decode(8, 1)` runs | `probes/shape_probe.py` |
| paged SDPA + prefill head-split at 1 KV head | §1.1 — all legal at 8Q/1KV | `probes/probe3a.py` |
| 2-device submesh collective | **did not complete** — see below | `probes/probe3.py` |

### Where this document is guessing rather than measuring

Stated explicitly, as requested:

* **The scheme-C decode total (≈ 319 µs) and prefill total (≈ 9.85 ms) are
  estimates.** The per-block *ratios* for the attention projections (2.87× /
  2.45×) and the expert matmuls (2.13×) and the collective costs (19.96 /
  76.85 µs) are measured; the attention body (~30 µs) and the expert
  reshape/eltwise tail (~55 µs) are scaled by argument with a hand-added launch
  floor, and could each be off by 10–15 µs.
* **The `sparse_matmul` probe uses random weights and DRAM outputs**, while the
  shipped decode path uses real weights and L1 intermediates at batch 1. That is
  why the probe's E=128 baseline reads 264.65 µs against the profile's 92.07 µs
  for the same two ops. Only the *ratios* between rows of that table are used
  here, never the absolute values.
* **`all_reduce_async` was not measured** — every argument spelling tried
  raised. The recommendation uses composed primitives, which are measured.
* **Fused `all_gather_matmul_async` / `matmul_reduce_scatter_async` were not
  evaluated.** They are the most promising unexplored lever for the 39.92 µs/layer
  of decode collective.
* **The 2×2 rejection's collective cost is inferred** from the flatness of the
  4-device size sweep below 128 KB, not from a separate 2-device measurement.
  The attempt to measure it — `mesh.create_submesh(MeshShape(1,2))` on a
  `FABRIC_1D_RING` 1×4 mesh, then `all_gather_async` on the submesh — **hung and
  had to be killed by its `timeout`, leaving the boards needing `tt-smi -r`**
  (`probes/probe3.py`, exit 143). That is not evidence the 2×2 scheme is
  impossible, but it is evidence that a ring-fabric submesh is not free to set
  up, which makes the 2×2 alternative *more* expensive to explore than its
  arithmetic already suggested it is worth. Anyone revisiting it should
  configure the fabric for the submesh topology rather than inheriting the
  parent's, per the skill's "treat CCL failures from raw `open_mesh_device` as
  setup evidence, not hardware evidence".
* **Prefill's per-die interleaved `wqkv` timing at N=1280 was not measured**
  (attention is 0.8% of prefill).
* **The 48-layer footprint is computed, not allocated.** A load-time probe that
  actually allocates 48 layers' worth of per-die weights should be run in
  implementation before the contract file is updated.
