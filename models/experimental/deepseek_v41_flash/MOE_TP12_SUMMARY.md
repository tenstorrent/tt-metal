# DeepSeek V4.1 Flash: MoE decode at TP12

Summary of the TP12 MoE decode experiment on the Blackhole 8x4 Galaxy: what was built, what was
measured, why the gains are what they are, and what is left. All numbers are for one MoE layer
(layer 39), one decode token, batch 1, measured with Tracy device profiling on Nov 2, 2026.

## TL;DR

Splitting the routed experts over 12 chips instead of 4 brings the MoE critical path from
**~193 µs to ~112 µs per token** (about 1.7x faster). The first attempt, which padded the
intermediate size, gained only 15%. Most of the gain came from removing that padding, and the
rest from two `fused_experts` kernel fixes. The rest of the MoE time no longer shrinks with more
chips. About 24 µs of router and small ops runs on every chip. Inside `fused_experts`, about
28 µs of setup comes before any weight is read.

| Per token, critical path | TP4 (1x4) | TP12 padded | TP12 unpadded | + shared-core move | + DRAM bank balancing |
|---|---|---|---|---|---|
| Router gate matmul (replicated) | 15 µs | 15 µs | 15 µs | 15 µs | 15 µs |
| Small ops (gather, unary, binary, reshard) | ~9 µs | ~9 µs | ~9 µs | ~9 µs | ~9 µs |
| `fused_experts` (6 routed + shared expert) | 165 µs | 130 µs | 89 µs | 80.6 µs | **77.8 µs** |
| All-reduce, real transfer | ~5 µs (1 axis) | ~10 µs (2 axes) | ~10 µs | ~10 µs | ~10 µs |
| **Total** | **~193 µs** | **~164 µs** | **~123 µs** | **~115 µs** | **~112 µs** |

The last two columns are estimates. They take the measured `fused_experts` time and add the
fixed costs and reduce cost that earlier runs measured on the critical path. PCC against the
torch reference is identical in every TP12 variant and matches TP4 (aggregate 0.9955). The output
is bit-identical on all 12 chips.

## 1. Setup

- **Mesh:** TP4 runs on a 1x4 submesh of the 8x4 Galaxy and TP12 on a 3x4 submesh, both created
  with `create_submesh` under fabric `FABRIC_2D_TORUS_XY`.
- **Sharding:** each rank holds an `I/TP` slice of every expert. The gate_up weights shard the
  N dimension and the down weights shard K. Partial outputs are summed with an all-reduce.
- **Two-axis reduce:** TP12 spans the whole 3x4 submesh, so the all-reduce runs on cluster axis 1
  and then axis 0 (`_tp_cluster_axes` in `deepseek_v4_flash/tt/decode/moe.py`). TP4 keeps its
  single-axis reduce.
- **Intermediate tiles per chip:** I = 2304, which is 18 tiles per chip at TP4 and 6 at TP12.
- **Test:** `test_moe_decode_tp` in `deepseek_v41_flash/tests/decode/test_moe.py`.
  - Parametrized by `tp_size` (4 and 12) and by layout (`whole_k` and `banked`).
  - Skips 4 warmup tokens.
  - Brackets each measured token with `MOE_TP<tp>_START` / `_END` signposts.
  - Checks that all chips hold the same output, and requires PCC 0.95.

### How the numbers were read

The tests dispatch ops one at a time, without trace replay, so chips start each token at
slightly different times. The all-reduce absorbs that skew. Chips that started early wait inside
it, so its time looks like 35 µs in the slowest-chip-per-op column even though the transfer
itself takes about 4.7 µs. The non-reduce work is the same on every chip, so the critical path is
the chip that starts last: its own work plus about 4.7 µs per reduce hop. Under trace replay most
of this start-time skew should disappear.

## 2. Step 1: padded TP12 (~164 µs, only 15% better than TP4)

`fused_experts` needs a whole number of I-tiles per gate_up core. The 6-expert path had 16 or 18
cores per expert, so I was padded to 6144 with zeros, giving 16 tiles per chip.

**Why 3x the chips gave so little:**

1. **Padding.** Each TP12 chip still read 16/18 of TP4's routed bytes (26.5 MB against 29.9 MB),
   and 10 of its 16 tiles were zeros. The cost follows bytes read per chip, not chip count.
   `fused_experts` went from 165 µs to only 130 µs.
2. **Fixed cost.** About 24 µs per token (the router matmul plus small ops) is replicated on
   every chip and does not shrink.
3. **The extra reduce hop.** The second all-reduce axis adds about 5 µs.

## 3. Step 2: unpadded TP12 (~123 µs)

`fused_experts` now runs 6 I-tiles per rank without padding:

- **Factory, `fused_experts_program_factory.cpp`:**
  - `num_producers` is the largest divisor of `i_tiles` that is at most `cores_per_expert`
    (6 of 16 cores at TP12). The other cores in the expert group stay idle in phase 1.
  - The divisibility TT_FATAL is gone.
  - The tall 9-row grid (18 cores per expert) is used only when the I-tiles divide by 18 and not
    by 16.
- **Kernels:**
  - The readers, compute and writer compute the SwiGLU tiles from `num_producers`.
  - Non-producer cores skip the gate_up fetch and the input CB wait.
- **Python (V4.1 `fused_experts_config`):** pads only when no producer count fits. TP 1, 2, 4,
  8, 12 and 24 need no padding.

Result: `fused_experts` dropped from 130 µs to 89 µs. TP4 is unchanged.

The device zones on one chip showed the shared-expert core (local core 1 of each group) spending
about 50 µs in gate_up. It computed its routed tile and also the shared-expert tile, while the
other producers took 17–23 µs.

## 4. Step 3: move the shared-expert tile to an idle core (89 → 80.6 µs)

The shared-expert tile now goes to the first idle core after the producers (local core 6 at
TP12), so no core reads both a routed tile and the shared tile. When there is no idle core before
hub1, which covers TP4 and TP1, it stays on core 1, so their layout is unchanged. This adds a
`shared_first_local` compile-time arg to the readers' shared-expert block, the writer and the
compute kernel.

## 5. Step 4: CB sizes (evaluated, no change)

Bigger circular buffers would not help:

- **`cb_weights`:** 540 KB already holds a producer's whole gate_up slice, read in one go, and
  all of its down slices, which are prefetched before the sync.
- **`cb_shared_gate_up`:** 2 x 24 KB chunks. The shared core reads at the same ~10 GB/s as the
  routed producers, so it is limited by the busy DRAM bank, not by chunk size.
- **The rest:** a few KB each and not on the critical path.

The real limit was DRAM bank imbalance:

- **Bank mapping:** ND shards are placed round-robin, so shard `p` sits on bank `p % 8`, and
  every expert tensor starts again at bank 0.
- **TP12 imbalance:** each expert has only 6 gate_up shards, so all of them landed on banks 0–5,
  about 1.45 MB per bank. Banks 6 and 7 got no gate_up traffic at all.

## 6. Step 5: spread gate_up over all 8 DRAM banks (80.6 → 77.8 µs)

A per-expert bank rotation would have been the simpler fix, but it does not guarantee balance:
the 6 experts chosen for a token often share a rotation. Instead, each gate_up column shard is
cut into K-row pieces, so every expert's shard count is a multiple of 8:

- **Piece count:** `k_shards = 8 / gcd(col_shards, 8)` (`_gate_up_k_shards`).
- **TP12:** 6 column shards x 4 K-pieces = 24 shards, 3 per bank.
- **Other configs:** TP4 also uses 4 pieces. TP1 (72 column shards) and V4 (64) stay at 1,
  because they already cover all 8 banks.
- **Kernel:** `fetch_weight_one` walks each CB reserve across the K-pieces
  (shard id `k_shard * col_shards + col`), so the CB contents and the compute side are unchanged.
- **Op:** the factory derives `gate_up_k_shards` from the tensor's shard shape and passes it to
  all three readers. The device op accepts any whole-tile divisor of K as the gate_up shard rows.
- **Opt-in:** `balance_dram_banks=False` by default on `DeepSeekV4PreloadedExperts` and
  `DeepSeekV41PreloadedExperts`.
  - Prefill (`fused_experts_prefill`) reads the same expert tensors and still requires full-K
    gate_up shards, so it would stop with an error on this layout.
  - The full model keeps the whole-K layout until the prefill op supports K-pieces.
- **Weight cache:** the gate_up cache name gets a `.k<N>` tag. A cache hit restores the shard
  layout the tensor was saved with and ignores the requested `memory_config`. Without the tag, the
  first banked run silently loaded the whole-K layout.

| TP12 `fused_experts` | Whole-K | Banked |
|---|---|---|
| Op time (`ops_perf` report) | 80.6 µs | 77.8 µs |
| Span on the device (first core start to last core end) | 86.0 µs | 83.0 µs |
| Phase 1 (gate_up) starts | 28.0 µs | 28.0 µs |
| Phase 1 ends | 69.5 µs | 66.5 µs |
| Phase 2 (down) ends | 79.8 µs | 76.8 µs |
| Median phase-1 time per core | 7.7 µs | 6.3 µs |
| Slowest phase-1 core | 40.8 µs | 37.4 µs |

The gain is real but small, because bank contention was only part of phase 1's time. The rest is
below.

## 7. Where the remaining ~112 µs goes

| Item | Time | Shrinks with more chips? |
|---|---|---|
| Setup inside `fused_experts` before phase 1 starts | ~28 µs | No |
| Phase 1 (gate_up), bounded by a few slow cores | ~38 µs | Partly |
| Phase 2 (down) | ~10 µs | Yes |
| Router gate matmul, replicated on every chip | 15 µs | No |
| Small ops | ~9 µs | No |
| All-reduce transfer (2 hops) | ~10 µs | Grows slightly |

Next steps, by expected payoff:

1. **The ~28 µs before phase 1.** Every core enters its first zone about 28 µs after the op
   starts, in every variant. It is probably the kernel setup: reading the expert ids and routing
   weights, and broadcasting the input. The next step is to add device zones around those steps
   in the reader kernels (no rebuild needed).
2. **The slow row-8 cores in phase 1.** The cores at x = 1, 3, 5, 7, 11 on row 8 take 25–37 µs
   against a 6 µs median, one per 2-column expert group. They are probably the hubs, whose zone
   includes waiting for every producer's gather. This needs confirming with the core-to-role
   mapping.
3. **Shard the router matmul.** Each chip would score its own 1/12 of the 384 experts, followed by
   a small gather before top-k. That saves most of the 15 µs.
4. **Shared expert gate_up in bfp4 instead of bf8.** This halves the largest single read
   (348 KB to 174 KB) but needs a PCC check.
5. **Split K across the idle cores.** At TP12, 10 of each group's 16 cores are idle in phase 1.
   This only helps now that the banks are balanced, and it needs a partial-sum reduce.
6. **Trace replay.** It removes the start-time skew between chips that the all-reduce currently
   absorbs.

Getting the MoE under 50 µs per token needs at least items 1–3.

## 8. Correctness notes

- All TP12 variants give the same PCC per token as TP4, and the K-split read path is bit-identical
  to the whole-K layout.
- Per-token PCC ranges from 0.80 (token 14) to 0.997, and token 14 is low at TP4 too, so the dip
  is not caused by TP12. The aggregate PCC is 0.9955.

## 9. Files touched

| File | Change |
|---|---|
| `deepseek_v4_flash/tt/decode/moe.py` | `_tp_cluster_axes` two-axis all-reduce, `_fused_parallel_grid_y`, `_gate_up_k_shards`, `balance_dram_banks`, `.k<N>` gate_up cache tag |
| `deepseek_v41_flash/tt/decode/moe.py` | Pad only when no producer count fits; pass `balance_dram_banks` through |
| `deepseek_v41_flash/tests/decode/test_moe.py` | `test_moe_decode_tp` (TP4 / TP12, `whole_k` / `banked`) |
| `fused_experts_program_factory.cpp` | `num_producers` divisor rule, grid-y rule, `shared_first_local`, `gate_up_k_shards` compile-time args |
| `fused_experts_device_operation.cpp` | Accept K-split gate_up shards |
| `kernels/dataflow/fetch_gate_up.h` | Producer gating, `SharedExpert::first_local`, K-piece reads in `fetch_weight_one` |
| `kernels/dataflow/{compute_expert_ids,broadcast_input,wait_expert_ids,write_gate_up}.cpp`, `kernels/compute/matmul_gate_up.cpp` | New compile-time args, producer-count based SwiGLU tiles |

## 10. Reproducing

The tests open the device, so check that the card is free first. Tracy re-joins its arguments
without quotes, so use `-k` filters that contain no spaces:

```
python -m tracy -p -r -v -m pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_moe.py::test_moe_decode_tp' -k banked-tp12 -x --tb=short
python -m tracy -p -r -v -m pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_moe.py::test_moe_decode_tp' -k whole_k-tp12 -x --tb=short
pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_moe.py::test_moe_decode_tp' -k tp4 -x --tb=short
```

Profiler reports under `generated/profiler/reports/`:

| Run | Report folder |
|---|---|
| TP4 (padded-run baseline) | `2026_11_02_07_22_04` |
| TP12 padded | `2026_11_02_07_28_16` |
| TP4 (unpadded-run baseline) | `2026_11_02_07_40_33` |
| TP12 unpadded | `2026_11_02_07_43_42` |
| TP12 shared-core move, whole-K | `2026_11_02_08_20_03` (first half of the signposts) |
| TP12 banked | `2026_11_02_08_25_36` |
