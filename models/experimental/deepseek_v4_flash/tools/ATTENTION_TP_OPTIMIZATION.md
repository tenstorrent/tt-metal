# DeepSeek V4 Flash TP attention optimization log

Goal: minimize TP=4 decode attention critical-path latency while preserving the
real-weight PCC threshold. All latency conclusions must come from device profile
data; correctness is checked with `test_attention_real_weights_tp.py`.

## Candidate ideas

1. **Head/group/N sharding (current foundation)** — shard `q_b` heads, `o_a`
   groups, and `o_b` output N. Gather `o_a` before `o_b`, then gather hidden N.
2. **Dedicated q_a and KV ranks** — execute `q_a` only on rank 0 and `kv_proj`
   only on rank 1, then explicitly broadcast both outputs.
3. **Balanced q_a/KV N sharding** — compute N/TP on every rank and gather before
   RMSNorm/RoPE. This uses all chips and one collective per projection.
4. **Replicated fused q_a+KV projection** — one larger matmul with no CCL. This
   spends replicated FLOPs but may beat small-message fabric overhead.
5. **One heterogeneous q_a+KV workload** — dispatch q_a on rank 0 and KV on rank
   1 in one mesh workload so their matmuls overlap, followed by broadcasts.
6. **Row-parallel o_b** — keep local `o_a` group output, compute full-N partials,
   then all-reduce. Compare one all-reduce against column-parallel `o_b`'s two
   all-gathers and reduced N compute.
7. **Fused o_b matmul + output all-gather** — use matmul_decode's fused gather
   and compare against the separate collective. Previous local work indicated it
   may be slower, so retain only if a fresh profile wins.
8. **Collective tuning** — compare linear/ring topology, link count, packet/page
   sizing, and async collectives for the small decode tensors.
9. **Eliminate layout traffic** — keep producer/consumer memory configs aligned,
   especially around P2P/collectives and `o_a -> o_b`.

## Experiments

### E0 — TP head/group/N sharding before dedicated q_a/KV

- Historical TP profile:
  `generated/profiler/reports/2026_08_28_09_03_50/ops_perf_results_2026_08_28_09_03_50.csv`
- Baseline:
  `generated/profiler/reports/2026_08_28_09_07_15/ops_perf_results_2026_08_28_09_07_15.csv`
- Result from `compare_attention_tp_profiles.py`: selected projection +
  collective path was 258.37 us versus 285.63 us baseline (1.11x).
- Caveat: this profile predates split `o_b`; `o_b` remained full N and only one
  all-gather was present.

### E1 — dedicated q_a rank 0 / KV rank 1 with direct P2P broadcasts

- Added optional restricted mesh dispatch to `matmul_decode`.
- Restricted dispatch is incompatible with the tensor prefetcher because pages
  sent to inactive ranks would not be acknowledged; q_a/KV use transient L1
  weights in this experiment.
- q_a output is copied from rank 0 to ranks 1–3; KV output is copied from rank 1
  to ranks 0, 2, and 3.
- Correctness: TP=4 real weights, layer 1, sequence length 2 passed.
  - Position 0 PCC: 0.9826032040
  - Position 1 PCC: 0.9773883339
- Profile: `reports/dedicated_qkv/2026_08_28_11_21_29`.
- Result: rejected. Six serialized P2P transfers cost about 158 us/token. The
  selected path is about 454 us/token after correctly summing repeated P2P and
  all-gather calls.

### E2 — balanced q_a/KV N sharding

- Per-rank matmuls are q_a `[4096, 256]` and KV `[4096, 128]`.
- Both outputs are gathered before their replicated consumers.
- Profile: `reports/balanced_qkv_no_prefetch/2026_08_28_11_27_56`.
- Result: 283.49 us/token including projection matmuls and all gathers. Correct,
  but four all-gathers consume 99.93 us/token.

### E3 — fused balanced q_a+KV

- Interleaved rank-local q_a/KV weight chunks into one `[4096, 384]` matmul,
  followed by one gather and reshape/split.
- Profile: `reports/fused_balanced_qkv/2026_08_28_11_30_14`.
- Result: 246.68 us/token, saving about 37 us over E2.

### E4 — fused replicated q_a+KV

- One replicated fused projection removes the q_a/KV collective.
- Partial-K layout profile: `reports/fused_replicated_qkv/2026_08_28_11_31_57`.
- Result with column-parallel o_b: 243.79 us/token. It narrowly beats fused
  balanced because small-message all-gather latency exceeds the extra FLOPs.

### E5 — row-parallel o_b

- Keep each rank's 2,048-wide local o_a result and K-shard o_b. Compute a
  full-hidden partial and all-reduce it.
- Profile: `reports/row_parallel_ob/2026_08_28_11_36_03`.
- Result: 220.67 us/token versus 243.79 us for column parallel. The o_b matmul
  drops from 35.11 to 16.31 us, and one all-reduce replaces two all-gathers.

### E6 — collective and receiver-grid tuning

- Ring all-reduce on the physical line: rejected; no forwarding route exists
  from endpoint rank 0 to endpoint rank 3.
- 8 KiB fabric payload: rejected; fused-balanced selected time regressed from
  246.68 to 254.89 us.
- Batched o_a with 16 N blocks: rejected; 235.10 us versus 220.67 us with 32.
- Fused q_a+KV full-width layout:
  - 24 receiver cores reduced q_a+KV from 34.38 to 19.64 us.
  - 48 receiver cores reduced it further to about 16.0 us.

### E7 — sequential local-group o_a

- Replaced the two-group batched matmul with two ordinary `[4096, 1024]`
  matmuls. Each local group gets a mesh-sharded weight.
- 16 receiver cores/group: 20.92 us per matmul.
- 32 receiver cores/group: 18.32 us per matmul.
- Best profile: `reports/sequential_oa_n32/2026_08_28_11_43_31`.
- Correctness:
  - Position 0 PCC: 0.9831063555
  - Position 1 PCC: 0.9780676831
  - Heavily-compressed layer 5, sequence length 128: passed at positions
    124–127 with PCC 0.9748–0.9800.
- Result: 139.95 us/token for projections + collectives, or 148.67 us/token
  when slice/concat device kernels are included.

### E8 — prefetcher on the optimized layout

- Profile: `reports/optimized_prefetch/2026_08_28_11_45_03`.
- Result: rejected for this TP layout. It reduced matmul time by 4.42 us but
  raised measured collective time enough to move the selected total from
  148.67 to 153.54 us/token.

## Current ranking

1. Fused full-width replicated q_a+KV (48 cores), sequential local-group o_a
   (32 cores/group), row-parallel o_b, no prefetcher: 139.95 us/token for the
   comparison script's projection + collective scope (2.04x over the 285.63 us
   single-chip baseline).
2. Same with prefetcher: 153.54 us/token including slice/concat kernels.
3. Fused replicated q_a+KV + batched o_a + row o_b: 214.53 us/token.
4. Fused replicated q_a+KV + column o_b: 243.79 us/token.
5. Fused balanced q_a+KV + column o_b: 246.68 us/token.
6. Balanced separate q_a/KV: 283.49 us/token.
7. Dedicated q_a/KV ranks with P2P: about 454 us/token.
