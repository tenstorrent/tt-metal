# DeepSeek V4.1 Fused Attention Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 14 ops that run one non-index decode attention layer on each TP4 chip with one op, `ttnn.experimental.deepseek.fused_attention_decode`. The op covers q_a / kv projection, the sliding-window ring update, SDPA over `[ring | selected]` and o_a / o_b with the TP all-reduce. It takes the layer from ~132 µs to ~25-30 µs with weights resident in L1, or ~50 µs when they are streamed.

**Architecture:** There is one program per chip, a `WorkloadDescriptor` so it can own fabric resources. It runs on the full 12x10 Tensix grid in three core roles: matmul cores, SDPA cores, and combine/all-reduce cores. The tensor prefetcher (DRISC) streams the layer's five weights into PrefetcherPipe rings in dependency order. Those rings are deep enough to hold the whole layer, so the next layer's weights land in L1 while the MoE of the current layer runs. That takes DRAM off the attention critical path. KV for SDPA is loaded at op start, because it does not depend on this step's q. The fresh kv row is patched into that L1 copy. Then q is the only dependency SDPA has to wait for.

**Tech Stack:** C++ TT-Metal (`WorkloadDescriptor` / `ProgramDescriptor`, Metal 2.0 `experimental::PrefetcherPipe`, tt-fabric 1D line multicast). JIT-compiled Tensix dataflow/compute kernels, nanobind, Python (ttnn, pytest).

**Spec:** None separate; sections 1-4 below are the spec. Sources:
- `models/experimental/deepseek_v41_flash/tt/decode/attention.py` (`_project_qkv`, `_qkv`, `_sdpa`, `_attend`, `_project_out`, `decode`)
- `ttnn/cpp/ttnn/operations/experimental/deepseek/flash_mqa_decode/` (two-segment KV, sink, tree merge)
- `ttnn/cpp/ttnn/operations/experimental/matmul_decode/` (pipe relay `publish_in1_block_from_pipe`, fused RMSNorm transport, RM height-sharded A)
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/` (fused fabric all-reduce: senders off-grid, `GlobalSemaphore`, scratch slots; see `docs/superpowers/plans/2026-11-04-fused-experts-all-reduce.md`)
- `tt_metal/impl/buffers/prefetcher_matmul_design.md` §6, `models/experimental/deepseek_v41_flash/tt/decode/decode_prefetch.py`
- `models/experimental/deepseek_v41_flash/DECODE_PARALLELISM_SUMMARY.md` §6 (whole-block preload idea), `DECODE_SDPA_IMPROVEMENTS.md`

---

## 1. Today's cost (measured)

Source: `generated/profiler/reports/dsv41_decode_32k_20L/2026_11_05_11_00_24/` (traced decode at ~32k, prefetcher on, `flash_mqa_decode`, Blackhole 12x10 = 120 workers). It takes the median over 612 compressed-consumer and sliding layer instances on all 32 chips. The 14-op pattern runs from `AllGatherForMatmul` through `WidthShardedAllReduce`.

| # | op | µs (p50) | cores | note |
|---|---|---|---|---|
| 1 | `AllGatherForMatmul` (tokens -> B grids) | 2.4 | 80 | |
| 2 | q_a `[5120,1280]` + fused RMSNorm, mcast to q_b grid | 18.1 | 8 (B) | compute-bound on 8 cores: 3.7 MB in 18 µs ≈ 200 GB/s |
| 3 | q_b `[1280,8192]` | 6.9 | 64 | weights partly prefetched during q_a |
| 4 | kv `[5120,512]` + RMSNorm | 10.3 | 16 | compute-bound on 16 cores, ≈ 140 GB/s |
| 5 | q RoPE (`FusedPartialRope`) | 5.6 | 64 | |
| 6 | kv RoPE | 2.4 | 1 | |
| 7 | ring write (`PagedUpdateCache`) | 4.6 | 1 | |
| 8 | `AllGatherForMatmul` (q -> SDPA) | 1.6 | 64 | |
| 9 | `flash_mqa_decode` | 17.3 (640 keys) / 8.1 (128 keys) | 16 | ~8 µs fixed, ~4.5 µs per 64-key chunk round |
| 10 | output RoPE | 20.8 | **1** | |
| 11 | `AllGatherForMatmul` (attn -> o_a) | 1.1 | 1 | |
| 12 | o_a (2 local groups, `[4096,1024]` each) | 8.1 | 80 | |
| 13 | o_b `[2048,5120]` row-parallel | 2.6 | 80 | whole slab already in its ring |
| 14 | `WidthShardedAllReduce` (5120 bf16, TP4) | 22.0 | 82 | 10 KB: latency + skew, not bandwidth |
| | **kernel sum** | **123.7** | | |
| | op-to-op gaps (13) | 8.7 | | |
| | **per layer** | **~132** | | |

What the numbers say:
- **Weights:** 21.7 MB per chip per layer at bfloat4_b (576 B/tile). The five matmuls take 46 µs in total, ≈ 470 GB/s, so they already run at DRAM speed while they run. But **DRAM is idle for the other ~86 µs** (RoPE, SDPA, CCLs, gaps): the rings are 2-16 pages deep, so the prefetcher cannot run ahead of the matmuls.
- **Non-matmul time is 78 µs of kernels.** About half of that is pathologies: single-core RoPE/cache ops (28 µs) and an all-reduce of 10 KB that costs 22 µs.
- **In scope:** 32 of 40 layers (2 sliding + 30 compressed consumers) ≈ **4.2 ms per token**. The 8 index-source layers are left alone.

## 2. Design

### 2.1 Dataflow on one chip (TP4, 16 local heads, S = 1, B = 1)

```
t=0  ┌ SDPA cores: read lengths, ring chunks and selected chunks DRAM -> L1 (655 KB total, KV is independent of q)
     ├ AR senders: open fabric connections
     └ matmul cores: wait for tokens; weights already sit in their rings (resident mode)
     tokens [1,5120] (width-sharded input) ──gather+mcast──► 96 phase-1 cores
P1   q_a raw [1,1280] on 80 cores (40 N-tiles x 2 K-halves)  ║  kv raw [1,512] on 16 cores (off critical path)
     q_a: partner K-reduce; sum-of-squares all-reduce among 40 -> r_q          kv: RMSNorm -> RoPE(tail) -> fresh row
     q_a slices (x r_q, gamma folded into q_b) ──mcast──► q_b cores           fresh row ──► SDPA core owning slot
P2   q_b [1,8192] on 64 cores (4 N-tiles = 1/4 head each); q RoPE on the tail core of each head                (patch L1 tile row,
     q slices ──mcast──► SDPA cores (16 heads x 512)                                                             RMW write ring tiles to DRAM)
P3   SDPA: 10 cores x 64 keys x 16 heads (one 16-row half tile = all local heads), HiFi2
     partials (m, l, O) ──► 8 combine cores (2 heads each): flat merge + sink + inverse RoPE
     attention rows (RM, 512 per head) ──mcast──► o_a A buffers of the head's group
P4   o_a: 2 groups x 32 N-tiles = 64 cores x [4096,32]
     o_a out [1,2048] ──mcast──► o_b cores
P5   o_b: 80 cores x [2048,64] -> partial [1,5120]
     partial -> scratch slot[my_chip]; 2 sender cores line-mcast to the 3 peers; Σ slots 0..3 in fixed order
out  [1,5120] in today's `width_sharded_all_reduce` output layout
```

### 2.2 Core map on the 12x10 grid (starting point, tune in Task 7)

| role | cores | per-core work | ring L1 / core (bf4) |
|---|---|---|---|
| q_a | x0-9, y0-7 (80) | `[2560, 32]` slab, 80 tiles | 46 KB |
| kv | x0-7, y8-9 (16) | `[5120, 32]`, 160 tiles | 92 KB |
| q_b | x0-7, y0-7 (64) | `[1280, 128]`, 160 tiles | 92 KB |
| o_a | x0-9 y8-9 + x8-9 y0-7 + 28 of x0-7 (64) | `[4096, 32]`, 128 tiles | 74 KB |
| o_b | x0-9, y0-7 (80) | `[2048, 64]`, 128 tiles | 74 KB |
| SDPA | x10, y0-9 (10) | 64 keys x 16 heads (KV 64 KB bf16) | none |
| combine | x11, y2-9 (8) | 2 heads, 10 partials each | none |
| AR senders | x11, y0-1 (2) | one fabric link each | none |

Every receiver set has a core count divisible by the 8 DRAM banks, as the slab-to-bank stride requires. The ring footprint is 21.7 MB per chip, about 217 KB average and ~290 KB peak on matmul cores. That is **less than today's attention rings**: the shared ring at ~288 KB on 64 cores, plus q_a (180 KB x 8), kv (72 KB x 16) and o_b (144 KB x 80), ≈ 32 MB. Task 1 checks this against the L1 high-water marks of `fused_experts`, Engram and the select.

The phases run one after another on each matmul core, so a core is a matmul kernel that runs a list of phases (CT args per phase). Separate SDPA and combine cores keep the code size of each RISC bounded.

### 2.3 Weights: prefetcher, ring depth and order

- **Rings.** There are five new PrefetcherPipe sets, one per weight, on the receiver sets above (`FUSED_ATTN_RINGS` in V4.1's `decode_prefetch.py`). Each is sized to **one whole slab per receiver** (resident mode) or 2 K-blocks (streaming fallback).
- **Order.** One request per layer per ring, queued `q_a, kv, q_b, o_a, o_b`. That is dependency order, so a shallow ring can never deadlock: every weight a core waits on is either ahead of it in the FIFO or on a core that does not depend on it.
- **Senders.** Attention rings use one of the two DRISC senders per bank; MoE / Engram rings use the other (`start_tensor_prefetcher` provisions 2 per bank). Layer L+1's attention weights then stream while layer L's MoE runs, instead of queueing behind `shared_down`'s 2-page ring. The MPFE weights give `fused_experts`' own DRAM reads priority.
- **Why resident mode works.** A layer is ~530 µs and only ~62 MB of DRAM per chip (~150 µs of DRAM time). Refilling 21.7 MB takes ~48 µs and runs from the end of attention L to the start of attention L+1 (MoE, mHC, Engram, norm: >300 µs).
- **Weight prep (host, cached).** q_a, kv, q_b, o_a and o_b are stored DRAM ND-sharded, one slab per receiver in ring order (`_store_dram_sharded_weight`). `q_a_norm.weight` is folded into q_b's rows (`W_qb' = diag(γ_qa) W_qb`). The scalar `r_q = rsqrt(mean(q_a²)+eps)` is applied to q_a's slices before they are mcast; this replaces the norm epilogue, and the cross-core sum is the only reduction left on the path. kv keeps a real RMSNorm epilogue, which is off the path.

### 2.4 TP4 parallelization

The chain inside one chip is ~25 µs, and a hop between chips is ~1-2 µs (still unmeasured, `h` in `DECODE_PARALLELISM_SUMMARY.md`). Every CCL on the path therefore costs a sizeable fraction of the layer. The layout keeps **exactly one** cross-chip exchange and makes it one round.

| option | DRAM / L1 per chip | CCL on critical path | verdict |
|---|---|---|---|
| **A. Head-parallel q_b / SDPA / o_a, row-parallel o_b, q_a and kv replicated (recommended)** | 21.7 MB | one all-reduce of 10 KB at the end | lowest latency once weights are resident: the 5.2 MB replicated q_a/kv cost is paid during MoE, off the path |
| B. As A, with q_a/kv column-sharded (320 + 128 cols/chip) and one fused all-gather of 896 B/chip | 17.8 MB (−3.9) | + one all-gather (~3-4 µs) between P1 and P2 | wins only in streaming mode (saves ~8 µs of DRAM there); fallback if 21.7 MB of rings will not fit |
| C. Column-parallel o_b (all-gather o_a out, compute 1280 of 5120 cols) | same | all-gather before o_b **and** after | two rounds instead of one; no |
| D. Split SDPA keys across chips | same | partial-merge exchange | heads already split 16/16/16/16 with no traffic; no |
| E. Reduce-scatter instead of all-reduce | same | one round, same hops | same latency for 10 KB, and the residual and mHC that follow need the full row; no |

The all-reduce inside the op is **one-shot**, the `fused_experts` pattern. Each o_b core writes its 64-col partial into slot `my_chip` of a persistent L1 scratch. Two sender cores (one per link, `num_links=2`) line-multicast their cores' slots to every peer's same slot and bump a `GlobalSemaphore`. Each o_b core then sums slots `0..3` in fixed order, so all chips get bit-identical output, which the replicated residual needs. On `Linear` 1x4 the worst case is 3 hops; `CCL_TOPOLOGY=Ring` makes it 2 where the wrap link exists. Connections open at t=0 in parallel with P1, so the only cost at the end is hops plus a ~0.3 µs sum.

KV stays replicated: MQA has one K==V head, every chip writes the identical fresh row, and the select already produces identical rows on every rank.

### 2.5 SDPA inside the op

- **Q layout:** all 16 local heads in one 16-row half tile. The FPU cost is per 16-row face, so 8 heads per core, as `flash_mqa_decode` uses today, wastes half of it. With 64 keys per core that is 32 tile matmuls for QKᵀ and 32 for PV, about 1-1.5 µs.
- **Keys:** chunk `c` of `[ring | selected]` goes to SDPA core `c`. The ring has 2 chunks of 64 and the selected rows up to 8. `lengths[0..1]` mask the tail chunk to −inf and skip empty chunks, and `lengths[2]` (`ring_offset`) picks a session's ring rows. All of these are read on device, so trace replays stay correct. With ring-only layers (128 keys), `keys_per_core=32` spreads the 4 chunks over 4 cores.
- **Fresh row:** kv cores mcast the RoPE'd row to the 2 ring-owning SDPA cores. The core whose chunk holds `sliding_pos` overwrites that row in its L1 tiles before QKᵀ. It then writes those 16 tiles back to `scache.kv` in DRAM, asynchronously, and must finish before the op ends. That is the read-modify-write `paged_update_cache` does today, but with the read already done. For `S > 1` (Task 9), row `j` patches rows `0..j`.
- **Merge:** flat, not a tree. Each SDPA core sends `m, l` (fp32, 16 values) and unnormalized `O` (16 x 512) rows to the combine core of each head pair. A combine core merges 10 partials plus the sink logit, `l += exp(sink − m)`. It applies the inverse RoPE to columns 448-511 and mcasts the 512-wide bf16 RM row into the o_a A buffers of its group at offset `(h % 8) * 512`. That removes the 3-round tree, the 1-core RoPE and the gather.
- **Scale:** `scaling = head_dim^-0.5` as today; `r_q` is already in q (section 2.3).

### 2.6 Perf model

Per-stage estimates assume a bf4 M=1 tile matmul from L1 at ~25 ns. That is calibrated from q_a today, 800 tiles in 18 µs on 8 cores; Task 1 measures it properly. Hops are taken at ~1-1.5 µs.

| stage | resident (µs) | streaming (µs) |
|---|---|---|
| launch + tokens gather/mcast | 1.5 | 1.5 |
| P1 q_a (80 tiles) + K-reduce + r_q | 2.5 | ~11 (5.2 MB of DRAM) |
| q_a mcast | 0.5 | 0.5 |
| P2 q_b (160 tiles) + q RoPE | 4.3 | ~12 (5.9 MB) |
| q -> SDPA | 0.5 | 0.5 |
| P3 SDPA (KV preloaded) | 1.5 | 1.5 |
| merge + sink + inverse RoPE + mcast | 2.5 | 2.5 |
| P4 o_a (128 tiles) | 3.2 | hidden (streamed during P2-P3) |
| o_a mcast | 0.5 | 0.5 |
| P5 o_b (128 tiles) | 3.2 | ~10 (rest of o_a + o_b: 10.6 MB from ~t=23) |
| fused all-reduce | 4-5 | 4-5 |
| **per layer** | **~25-28** | **~45-50** |
| today | 132 | 132 |

Per token, 32 layers in scope: today ≈ 4.2 ms → resident ≈ 0.85 ms (**−3.3 ms**), streaming ≈ 1.55 ms (**−2.6 ms**). Against the 27.9 ms greedy token of `DSPARK_100TPS.md`, that is **35.8 → ~40.6 tok/s/u** resident, or ~39.5 streaming. DSpark rounds gain the same amount only once the verify rows (S > 1, Task 9) use the op.

What decides the resident number, most uncertain first:
1. The L1 compute rate of bf4 M=1 tile matmuls; q_b at 160 tiles per core is the longest phase.
2. Hop latency `h`.
3. Whether the 21.7 MB of rings fit.
4. NoC mcast contention in P1 and P2.

Two further cuts, out of scope, each worth −2 to −6 µs a layer:
- Fold `input_layernorm` into the op: fold γ into q_a/kv, and r into P1. That deletes a 6.4 µs op and a gap.
- Use LoFi for SDPA, if its accuracy holds.

## Global Constraints

- **Device commands need approval.** Every `pytest`, tracy capture, demo or test binary that opens the card must be shown to the user as a copy-pasteable command and run only after they approve. Never background or "just try" one.
- **Build only with** `cd /home/tt-admin/smanoj/metal_1 && ./build_metal.sh`. Kernel `.cpp` under `.../kernels/` is JIT-compiled and needs no rebuild; host `.cpp`/`.hpp`/nanobind edits do.
- **No thin wrappers** (workspace rule): no helper that only forwards arguments.
- **Do not touch the indexer**: `DeepSeekV41Indexer`, `fused_lightning_select_kv`, `merge_select_kv`, the compressor and index-key paths. Layers with `select=True` keep today's path.
- **Scope of v1:** `S == 1`, `B == 1`, the `sdpa_lengths` path (single session or `ring_offset`), `tp_size ∈ {1, 4}` on a 1xTP mesh, Blackhole 12x10, `weight_dtype=bfloat4_b`. Every other call keeps today's ops.
- **Switch:** `DEEPSEEK_V41_FUSED_ATTENTION=0` restores the 14-op path, like `DEEPSEEK_V4_FUSE_ALL_REDUCE`.
- **Numerics:** the output must match `DeepSeekV41Attention.decode` to PCC ≥ 0.999 (bf16 path) and the ring row to PCC ≥ 0.9999. Every TP rank must produce bit-identical output and ring rows.
- **Uncommitted work in `experimental/CMakeLists.txt` and `experimental_nanobind.cpp`** (engram_gate_prefill) must be preserved; add the new op next to it.

## Review Focus

1. **Ring wrap at `pos ≥ 128`:** slot `pos % 128` holds the key of position `pos − 128`. The patch has to replace it, not add a 129th key. Test at positions 127, 128, 129, 255 and 32767.
2. **Early positions:** `ring_valid < 128` and `selected_valid ∈ {0, 1, 37}` mean partial or empty chunks, and position 0 has a single key. A wrong mask shows up as NaN or as attention to zeros.
3. **Several sessions:** `lengths[2] = ring_offset` and `sliding_pos` include the slot offset. A ring read or write at the wrong slot corrupts another session silently.
4. **Trace replay across positions:** every position-dependent value (lengths, `sliding_pos`, cos/sin) must be read on device. Capture once, replay at 3 positions, and compare to eager.
5. **Prefetch order across layers:** 5 layers back to back on one stage, with an interleaved non-attention op and an index layer on the old path. A wrong FIFO order delivers the wrong weights with no error, so check every layer's output.

---

## File Structure

| path | responsibility |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/deepseek/fused_attention_decode/fused_attention_decode.{hpp,cpp}` | public API |
| `.../fused_attention_decode/fused_attention_decode_nanobind.{hpp,cpp}` | `ttnn.experimental.deepseek.fused_attention_decode` |
| `.../fused_attention_decode/device/fused_attention_decode_device_operation{.hpp,.cpp,_types.hpp}` | attributes, validation, hash, output spec |
| `.../fused_attention_decode/device/fused_attention_decode_program_factory.{hpp,cpp}` | `create_workload_descriptor`: core map, CBs, semaphores, scratch, per-chip programs |
| `.../device/kernels/dataflow/{reader,writer}_matmul_phases.cpp`, `compute/matmul_phases.cpp` | matmul cores: pipe relay, A mcast, K-reduce, r_q, RoPE, kv norm |
| `.../device/kernels/dataflow/{reader,writer}_sdpa.cpp`, `compute/sdpa.cpp` | KV preload, fresh-row patch and ring write-back, QKᵀ/softmax/PV |
| `.../device/kernels/dataflow/{reader,writer}_combine.cpp`, `compute/combine.cpp` | flat merge, sink, inverse RoPE, o_a mcast |
| `.../device/kernels/dataflow/ar_sender_reader.cpp` | wait for partials, feed the stock `all_reduce_async` `worker_writer` |
| `.../fused_attention_decode/{CMakeLists.txt,sources.cmake}` + `experimental/CMakeLists.txt`, `experimental_nanobind.cpp` | build/registration |
| `models/experimental/deepseek_v41_flash/tt/decode/fused_attention.py` | `FusedAttentionDecode`: weight prep (slabs, γ fold), rings, call |
| `models/experimental/deepseek_v41_flash/tt/decode/decode_prefetch.py` | `FUSED_ATTN_RINGS`, receiver sets, sender choice |
| `models/experimental/deepseek_v41_flash/tt/decode/attention.py` | `decode` dispatch to the fused op |
| `models/experimental/deepseek_v41_flash/tests/decode/test_fused_attention_decode_op.py` | op tests vs the module |

---

### Task 1: Baseline, L1 audit and matmul calibration (no op code)

**Files:** Modify `models/experimental/deepseek_v41_flash/tests/decode/test_attention.py` (prefetcher on in `test_attention_decode_profile_tp4`); write the findings into section 1 of this plan.

- [ ] **Step 1:** Make `test_attention_decode_profile_tp4` run inside `tensor_prefetcher_session` with `use_prefetcher=True`, using the current `flash_mqa_decode` path.
- [ ] **Step 2 (device, ask):** `DEEPSEEK_V4_TRACED_DECODE=0 python -m tracy -p -r -v -m pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_attention.py::test_attention_decode_profile_tp4'`. Record per-op p50 for one consumer and one sliding layer. Expected: within 10% of section 1.
- [ ] **Step 3:** L1 audit. In `test_full_model_decode_demo.py`, log `ttnn.get_memory_view(device, L1)` high-water marks around `fused_experts`, Engram and the select, plus the current ring allocations. Record the free L1 per core (x0-9, y0-9). Pass criterion for resident mode: ≥ 300 KB free per matmul core after removing the q_a, kv, shared(q_b/o_a) and o_b rings.
- [ ] **Step 4 (device, ask):** time `matmul_decode` with L1-resident (`dram_sharded_weight=False`) bf4 slabs of 80, 128 and 160 tiles at M=1. This sets the per-tile constant in section 2.6.
- [ ] **Step 5:** Update section 2.6 with the measured constants, then commit.

### Task 2: Golden test + op skeleton

**Files:** Create the op directory (all files in File Structure; kernels empty) and `tests/decode/test_fused_attention_decode_op.py`.

**Interfaces (Produces):**

```python
ttnn.experimental.deepseek.fused_attention_decode(
    tokens: ttnn.Tensor,            # [1,1,1,5120] bf16, today's input_layernorm output layout
    q_a_weight, kv_weight, q_b_weight, o_a_weight, o_b_weight: ttnn.Tensor,  # DRAM ND-sharded slabs, bfloat4_b
    kv_norm_weight: ttnn.Tensor,    # [1,1,1,512] bf16 (q_a gamma is folded into q_b)
    rot: ttnn.Tensor,               # [64,64] interleaved rotate matrix
    cos: ttnn.Tensor, sin: ttnn.Tensor,     # TILE [1,1,1,64]
    kv_cache: ttnn.Tensor,          # scache.kv, TILE DRAM [1,1,N*128,512], updated in place
    sdpa_lengths: ttnn.Tensor,      # INT32 [1,2|3]
    sliding_pos: ttnn.Tensor,       # INT32 [1], includes the session offset
    attention_sink: ttnn.Tensor,    # TILE [16,32], col 0 = sink/scale
    *, selected_kv: ttnn.Tensor | None, scale: float, eps: float,
    num_heads: int, head_dim: int, rope_dim: int, window: int, o_groups: int,
    cluster_axis: int | None, num_links: int = 2, topology=ttnn.Topology.Linear,
    prefetcher_pipes: dict[str, list] | None = None,   # keys q_a, kv, q_b, o_a, o_b
    keys_per_core: int = 64, memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor                    # [1,1,1,5120], width_sharded_all_reduce's output layout
```

- [ ] **Step 1: Write the failing test** `test_fused_attention_decode_matches_module[tp1|tp4][ring_only|ring_selected][pos 0, 5, 127, 128, 1000, 32767]`. It builds a random-weight `DeepSeekV41Attention` (layer 21 config, plus layer 0 for ring-only), runs `decode(...)` with `decode_inputs(pos, ...)` as the golden, and runs the op on identical caches. Assertions: `comp_pcc(out, golden) ≥ 0.999`; `comp_pcc(kv_cache_after, golden_kv_cache_after) ≥ 0.9999`; at TP4, `torch.equal` across ranks for both.
- [ ] **Step 2 (device, ask):** run it; expect `AttributeError: ... fused_attention_decode`.
- [ ] **Step 3:** API, nanobind and device-op validation. Check shapes and dtypes as in the interface; `num_heads % 16 == 0` per chip; `head_dim == 512`; `window % keys_per_core == 0`; at least 120 workers. The hash covers everything except tensor addresses and the position tensors. `create_workload_descriptor` returns an empty program per chip.
- [ ] **Step 4:** `./build_metal.sh`. Rerun; expect a PCC failure, not a crash. Commit.

### Task 3: P1 + ring update (q_a, kv, norms, RoPE, fresh row)

**Files:** factory and `matmul_phases` kernels; weights read from DRAM ND-sharded slabs (no prefetcher yet).

**Interfaces:** debug attribute `stop_after: Literal["qkv","sdpa","out"] = "out"` (test-only; part of the program hash, since it changes the program). At `"qkv"` the op returns `(q [1,1,1,8192], kv_row [1,1,1,512])`, both RoPE'd, plus the updated `kv_cache`.

- [ ] **Step 1:** Test `test_fused_attention_qkv_stage`: q and kv_row vs `DeepSeekV41Attention._qkv` with PCC ≥ 0.9995, and `kv_cache` vs `_update_cache_at`. Positions as in Task 2; multi-session with `ring_offset ∈ {0, 128, 384}`.
- [ ] **Step 2:** Implement the tokens gather + mcast to 96 cores, q_a (40x2, partner reduce), sum-of-squares across the 40 cores → `r_q`, q_a slice mcast to q_b cores, and q_b with the folded γ. Add q RoPE on each head's tail core: `x·cos + (x @ rot)·sin` on 2 tiles. Add kv with RMSNorm transport (reuse `full_width_rms_norm_transport.hpp`), kv RoPE, and the ring write. Until Task 4 the ring write is a plain RMW of 16 tiles from a kv core.
- [ ] **Step 3 (device, ask):** run the test; expect PASS. Commit.

### Task 4: P3 SDPA + merge

**Files:** `sdpa` and `combine` kernels; factory.

**Interfaces:** `stop_after="sdpa"` returns attention `[1,1,16,512]` with inverse RoPE applied, equal to `_attend`'s tensor before `_project_out`.

- [ ] **Step 1:** Test `test_fused_attention_sdpa_stage`: vs `flash_mqa_decode` followed by `_apply_rope(..., neg_sin)` with PCC ≥ 0.999. Grid: positions {0, 1, 127, 128, 129, 255, 32767} × `selected_valid` {0, 1, 37, 512} × `keys_per_core` {32, 64}, plus `selected_kv=None` and `ring_offset=128`.
- [ ] **Step 2:** Implement the KV preload at t=0 and the per-chunk −inf mask from the lengths. The ring write moves to the SDPA core that owns `sliding_pos`: patch its L1 tile row, then write its 16 tiles back with a barrier before the op ends. Then QKᵀ, softmax and PV with 16 heads in one half tile; reuse `compute_sdpa_chunk` from `flash_mqa_decode`. Then the fp32 stats and O rows to the combine cores, the merge with the sink, the inverse RoPE, and the RM output.
- [ ] **Step 3 (device, ask):** run; expect PASS. Then rerun Task 3's test (the ring now comes from the SDPA cores); expect PASS. Commit.

### Task 5: P4-P5 at TP1

**Files:** `matmul_phases` (o_a, o_b phases), combine writer (mcast into the o_a A buffers).

- [ ] **Step 1:** Test: Task 2's test, `tp1` ids only.
- [ ] **Step 2:** Implement the combine → o_a A mcast at `(h % 8) * 512`, o_a over 2x32 cores, the o_a output mcast to o_b, and o_b. Output goes to `memory_config` (TP1: o_b's width-sharded output).
- [ ] **Step 3 (device, ask):** run `-k tp1`; expect PASS. Commit.

### Task 6: Fused TP all-reduce

**Files:** factory (`GlobalSemaphore`, scratch `[4, 64]` per o_b core in L1_SMALL, 2 sender cores x11 y0-1), `ar_sender_reader.cpp`, the o_b writer and compute (sum of the 4 slots in fixed order).

- [ ] **Step 1:** Test: Task 2's test, `tp4` ids, with the bit-identity assertion.
- [ ] **Step 2:** Implement it, following `docs/superpowers/plans/2026-11-04-fused-experts-all-reduce.md` Task 4: the stock `all_reduce_async` `worker_writer` on the senders. Connections open at t=0. The workload is created before trace capture.
- [ ] **Step 3 (device, ask):** `pytest models/experimental/deepseek_v41_flash/tests/decode/test_fused_attention_decode_op.py -k tp4 -x`; expect PASS. Commit.

### Task 7: Prefetcher rings and resident mode

**Files:** `decode_prefetch.py` (`FUSED_ATTN_RINGS`: per-weight receiver core lists from section 2.2, depth `"resident"` = one slab per receiver, or `"stream"` = 2 K-blocks; sender 0 per bank). `layers.py` `make_shared_decode_pipes` takes an explicit receiver core list instead of a rectangle anchored at (0,0). `fused_attention.py` (`fetch_weights` queues the 5 tensors in order). The factory and kernels bind the 5 pipes with `bind_relay` per phase.

**Interfaces:** `FusedAttentionDecode(config, layer_idx, weights, device, cache, weight_dtype, tp_size, prefetch_mode: Literal["off","stream","resident"])`, `.fetch_weights()`, `.__call__(tokens, scache, **decode_inputs) -> ttnn.Tensor`.

- [ ] **Step 1:** Tests: Task 2's test parametrized over `prefetch_mode`. Plus `test_fused_attention_prefetch_order`: 5 consecutive layers (one of them an index layer on the old path) with a `ttnn.matmul` in between, and each layer's output vs its golden.
- [ ] **Step 2:** Implement it. In resident mode, assert the ring fits L1 using Task 1's budget, with a message naming the shortfall.
- [ ] **Step 3 (device, ask):** run both tests; expect PASS. Profile `test_fused_attention_decode_op.py -k "tp4 and resident and pos32767"` under tracy. Record the op time against section 2.6, then tune the core map, `keys_per_core` and the q_b core count. Commit.

### Task 8: Model integration, trace, end-to-end

**Files:** `attention.py`. In `DeepSeekV41Attention.__init__`, build `FusedAttentionDecode` when `not self.is_index_source` and the switch is on, and skip the five `LinearDecode`s and their rings. `decode` takes the fused branch for `s == 1 and b == 1 and sdpa_lengths is not None`. `prefetch_weights` routes to it.

- [ ] **Step 1:** Test: `tests/decode/test_attention.py::test_attention_decode` with `DEEPSEEK_V41_FUSED_ATTENTION=1` and `=0`; the same tolerance as today on both.
- [ ] **Step 2:** Implement it. Add the trace-replay test from Review Focus 4.
- [ ] **Step 3 (device, ask):** `test_full_model_decode_demo.py` at `DEEPSEEK_V4_START_POS=32768`: text vs the switch off, and tok/s/u. Then the 20L tracy profile; diff against `dsv41_decode_32k_20L/2026_11_05_11_00_24`. Expected: −2.6 to −3.3 ms/token. Commit.

### Task 9 (follow-up, separate plan once Task 8 lands)

- **`S > 1` verify rows** (DSpark): M = S ≤ 16 in the matmul phases; per-row causal patch of rows `0..j`; Q tile of `S·16` rows split over SDPA cores.
- **Front/back split for index layers:** `stop_after="qkv"` returns normalized q_a for the indexer and a `start_at="sdpa"` entry, so index layers use the same rings and drop the old ones.
- **Option B** (sharded q_a/kv) if Task 1 shows the rings do not fit.
