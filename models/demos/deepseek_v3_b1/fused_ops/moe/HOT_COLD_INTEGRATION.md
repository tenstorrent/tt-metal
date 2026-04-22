# Hot/Cold Expert MoE Integration — Design Doc

Integrates the `matmul_expert` (hybrid SRAM + DRAM, compressed weights) kernels
into the fused MoE kernel (`moe_kernel.cpp`). Requires a TP8 Megatron refactor
of the routed-expert path.

## 0. Status (2026-04-22)

**Phase 1A — TP8 plumbing: ✅ COMPLETE**
**Phase 1B — matmul_expert DRAM swap: ✅ COMPLETE (all 8 experts, DRAM-only)**
**Phase 2 — SRAM hot path + new index encoding: ⏳ NOT STARTED**

Anchor-test results on `test_moe_fused_with_reduce[blackhole-True-
NOC_MODE.DM_DYNAMIC_NOC-True-fabric_2d]` (100 iterations, 8 devices):

| Check                 | Baseline (pre-TP8) | After Phase 1B   |
|-----------------------|--------------------|------------------|
| Reduce output PCC     | 0.9911             | **0.9915**       |
| Reference MoE PCC     | 0.9911             | **0.9915**       |
| Uniform-scale robust? | n/a                | 0.9911 (varied)  |

Pre-reduce per-device check (`test_moe_fused_no_reduce`): all 8 devices
≥ **0.9988 PCC** (was d2=0.9627, d5=0.9679 failing the 0.97 threshold
before the shared-expert face-view fix — see §11).

## 1. Context

### 1.1 Current fused MoE routed pipeline (12 steps)
`moe_kernel.cpp` runs a full routed + shared MoE block in one kernel. The
routed DRAM matmul sub-pipeline is:

- Step 2: gate matmul + sigmoid (64 cores)
- Step 3: gate gather
- Step 4: TopK (sender)
- Step 5: mcast index + expert scale to 8 DRAM streamer cores
- Step 6: `gate_proj` DRAM MM + SiLU (8 cores, K=7168, N=2048 per device)
- Step 7: `up_proj` DRAM MM (8 cores, K=7168, N=2048 per device)
- Step 8: fused mul (gate × up × scalar)
- Step 9: `down_proj` gather (gate_proj cores → sender)
- Step 10: `down_proj` mcast (sender → all gate_proj cores)
- Step 11: `down_proj` DRAM MM (8 cores, K=2048, N=7168 per device)
- Step 12: eltwise add with shared-expert output

`gate_proj` / `up_proj` today are **full-K, N-per-device=2048**. One matmul
processes one token × one selected expert on one of 8 DRAM streamer cores.
Topk=8 is distributed 1 expert per DRAM core per device.

### 1.2 Hot/cold `matmul_expert` kernels
Two sibling headers under `unified_kernels/`:

- `matmul_expert_compressed_sram.hpp` — weights resident in L1/SRAM (compressed)
- `matmul_expert_compressed_dram.hpp` — weights streamed from DRAM (compressed)

Called back-to-back from a single kernel entry (see
`micro_ops/matmul_expert/kernels/matmul_expert_kernel.cpp`):
```
sram_mm();
dram_mm();
```
Each inspects an expert-index array. Bit 15 (1-indexed 16th) of each entry
selects which path processes that expert: `1` → SRAM, `0` → DRAM. The other
path skips it. This is how "hot" experts (in L1) and "cold" experts (in DRAM)
can coexist in one call.

Key properties:
- **Accum mode** (`accum_experts=True`): sums across selected experts into
  one output. Used by `down_proj`. SRAM path uses dst regs (no expert count
  limit). DRAM path uses L1_ACC slots 0+1 (cap of **2 experts per call**).
- **Non-accum mode** (`accum_experts=False`): one output per selected expert,
  concatenated/pushed per iteration. Intended for `gate_proj` / `up_proj`.
- **Compressed tile formats per expert**: each tile has its own format
  (bfp8 / bfp4 / bfp2 / bfp0) looked up from a per-expert format table.
- **Inner-K slicing** (`sram_k_per_core`, `sram_k_offset`) is already
  supported — one core can own a K-slice.
- **Outer-N slicing** is already supported via `per_core_n`.
- SRAM path has **no SiLU fusion**. DRAM path has `dram_fuse_silu`.

### 1.3 Why today's MoE and matmul_expert don't compose directly
- Today's routed path is **EP (expert-parallel) over 8 cores on one device**:
  each of 8 DRAM cores handles one of the 8 selected experts in full.
  `per_core_n = 2048`, `per_core_k = 7168`.
- `matmul_expert` is designed to be **TP over many cores**: one expert is
  split across cores by N (for gate/up) or by K (for down). An SRAM-resident
  expert can only be "hot" if every core in the grid holds a slice of it.
- `matmul_expert` also assumes weights are compressed and selected via an
  expert-index array with a flag bit — a format the routed path doesn't
  produce today.

## 2. Anchor test — `test_moe_fused_with_reduce`

`models/demos/deepseek_v3_b1/tests/unit_tests/test_moe_mlp.py::test_moe_fused_with_reduce`
is **the** correctness target for this integration. Both Phase 1 (DRAM-only
TP8) and Phase 2 (SRAM + new index encoding) design and debug against it.

**Setup:**
- Fixture: `bh_2d_mesh_device`, submesh `4×2` = 8 devices, `FABRIC_2D`.
- Weights: 256 routed experts, full DeepSeek V3 layer dims (K=7168, intermediate=2048).
- Gate is rigged via `rig_moe_gate_for_expected_experts` so topk=8 is deterministic.
- `num_iterations = TestConfig.NUM_ITERATIONS` (= 100) in-kernel loops per invocation.
- `reconfig_moe_cbs ∈ {True, False}`, `noc_mode=DM_DYNAMIC_NOC`.
- Env: `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_VISIBLE_DEVICES=<viable 4x2 group>`.

**Two PCC checks against per-device ROOT1 reduce output:**
1. **Reduce output PCC (threshold 0.97)** — vs `MoeOp.golden(...)` (B1's own
   Python golden of the kernel math). Validates the kernel matches its spec.
2. **Reference MoE PCC (threshold 0.95)** — vs `create_reference_moe_model`
   (upstream DeepSeek reference built from the state_dict). Validates the
   spec matches the authoritative HF-style implementation.

Plus: TT vs golden topk-indices must match (gate correctness).

**Baseline (pre-integration, 2026-04-18, reconfig=True):**
- Reduce output PCC = **0.9911242884032705**
- Reference MoE PCC = **0.9911318818015358**
- 100 iterations in ~2 minutes, PASSED.

**Known quirk — magnitude-blind today:** weights pass through
`ReplicateTensorToMesh`, so all 8 devices compute the same full output and
the sum is 8×. PCC normalizes scale → passes regardless. The **TP8
integration must change this to sharded weights** and make the magnitude
itself correct (mirroring what `test_mlp_with_reduce(use_mlp_weights=True)`
already does for the dense path at `test_moe_mlp.py:1542-1573`).

**What "PASSED" looks like after TP8 refactor:**
- Both PCCs remain ≥ thresholds (ideally similar to baseline, ~0.99).
- Magnitude comparison is now meaningful: per-device golden shards gate/up
  by N-column and down by K-row, residual added only on ROOT1, sum-reduce
  equals the full MoE output.
- No hangs across all 100 iterations, both `reconfig_moe_cbs` values.

**Fast-iterate command (single variant):**
```bash
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_VISIBLE_DEVICES=0,1,4,5,24,25,28,29 \
  pytest -svv models/demos/deepseek_v3_b1/tests/unit_tests/test_moe_mlp.py::test_moe_fused_with_reduce \
  -k "True"
```
(The `-k "True"` filter still matches both `reconfig_moe_cbs` variants because
`blackhole-True-...` is a fixture marker; use a more specific `-k` if needed.)

## 3. Target: TP8 Megatron across 8 devices

### 3.1 Partitioning
Eight devices form a TP8 group. Routed-expert weights partition Megatron-style:
- `gate_proj` / `up_proj`: **column-parallel** → each device holds
  7168 × 256 (intermediate_dim = 2048 → 256 per device).
- `down_proj`: **row-parallel** → each device holds 256 × 7168.
- Gate (router) runs replicated on all devices (small).
- Each device still has its own 8 DRAM streamer cores. On one device, a
  gate/up matmul is K=7168, N_per_device=256 → N_per_core=32.
  A down matmul is K_per_device=256, N=7168 → N_per_core=896.
- Gather+mcast between gate/up and down is still needed (it's within one
  device, sending the hidden-dim slice to all local down-proj cores).

### 3.2 Shared-expert coexistence
Shared expert is already TP8 across the 8 devices. After both paths:
- Routed: each device holds a `[1, 7168]` **partial** (sum of its owned
  N-columns of down_proj output, over the topk selected experts).
- Shared: each device holds a `[1, 7168]` **partial** (its slice of the
  shared down_proj output).
- Per-device sum (eltwise add) → per-device routed+shared partial.
- One cross-device `ReduceToOne` over the 8 devices → full
  `[1, 7168]` = routed_full + shared_full.

**Math correctness note:** TP8 on the routed path is a **requirement**, not
just a perf choice. With today's EP-on-one-device routed path + TP8 shared,
the shared output is already a partial while routed is full. Combining them
would need a different reduce pattern. With both as TP8 partials, a single
per-device add + one cross-device reduce produces the correct full output.

### 3.3 What stays / what changes
- **Keep**: gate matmul, sigmoid, gate gather, TopK, mcast index, expert scale
  mcast, gather+mcast between gate/up and down, residual add, shared-expert
  sub-pipeline.
- **Change**: step 6/7 (gate/up DRAM MM) → `matmul_expert` non-accum;
  step 11 (down DRAM MM) → `matmul_expert` accum; step 8 (fused mul) must
  handle per-expert scalars (see §5).
- **Remove/reshape**: step 9/10 gather+mcast still exist but carry a smaller
  per-device hidden slice. Remove the old per-expert scalar broadcast since
  accum'd down no longer multiplies by expert scalar downstream — scalar
  multiply moves into step 8.
- **Add**: one cross-device `ReduceToOne` across 8 devices after step 12
  (per-device routed+shared partial → full).

## 4. Index tensor encoding

### 4.1 Current encoding
TopK produces a `[1, 8]` uint16 tensor of selected expert IDs (`eid`).
Today `matmul_expert` kernels mask `raw_idx & 0x8000` to pick SRAM vs DRAM,
and use the lower 15 bits as an `eid` fed through a `table_idx_arr` to map
`eid → table row`. The table indirection exists because one device holds a
subset of experts and the weight/format tables are packed by **local
position**, not global `eid`.

### 4.2 New encoding (upstream)
The selection tensor passed into `moe_kernel` will be encoded upstream:
- **Bit 15 (1-indexed 16th)**: `is_sram` (1 = SRAM, 0 = DRAM).
- **Lower 15 bits**: **position in the per-device packed expert table**
  (not `eid`).

Implications:
- No flag-insert step inside `moe_kernel` — the selection tensor is already
  correctly encoded when it arrives.
- `matmul_expert` kernels can **drop the `table_idx_arr` indirection**: the
  lower 15 bits are already the table row. Whether this simplification
  happens in this integration or as a follow-up is deferred — we can keep
  the indirection by feeding a trivial identity `table_idx_arr` at first.
- The upstream producer of this encoding (likely TopK or a thin layer after
  it) is out of scope for this doc but must exist before integration.

## 5. EltwiseMul per-expert looping (option 1)

**Status**: loop implementation landed in `unified_kernels/eltwise_mul.hpp`
(BRISC scalar feed + TRISC per-expert in0*s*in1); `num_experts=1`
default preserves old behavior. Currently `moe_kernel.cpp` instantiates
EltwiseMul with `num_experts=1` — the loop is exercised only when a
future refactor consolidates gate/up matmul output from 8 EP streamer
cores to one non-accum matmul_expert call per device.

### 5.1 Problem
Today's `eltwise_mul.hpp` handles one `(gate × up × scalar)` multiply per
invocation — one selected expert's worth of tiles. With non-accum gate/up,
a single `matmul_expert` call emits **num_experts** groups of output tiles
back-to-back into `cb_in0` and `cb_in1`, one group per selected expert,
each needing its own expert scalar. We need the mul to loop over groups.

### 5.2 Chosen approach — internal loop in the kernel (option 1)
Extend `EltwiseMul` to loop `num_experts` times inside the op:

- **Compile-time arg**: `num_experts` (new, default 1 → current behavior).
  `tiles_per_expert = num_tiles / num_experts`.
- **BRISC**:
  - Wait for `cb_scalar_src` (populated by upstream mcast — shape
    `[1, num_experts]` of fp16 scalars).
  - For `i in 0..num_experts-1`: read `src_ptr[scalar_index_offset + i]`,
    write to `cb_scalar` (reserve 1, push 1). One scalar per expert.
  - Pop `cb_scalar_src` once at the end.
- **TRISC**:
  - Wait inputs upfront: `cb_in0_wait` (num_tiles total), `cb_in1_wait`
    (num_tiles total). Matmul emits all experts' tiles into these CBs.
  - Reserve `cb_out` num_tiles upfront (or per-group — per-group simplifies
    pack indexing; upfront is a micro-opt).
  - Per-group (`g = 0..num_experts-1`):
    - `cb_wait_front(cb_scalar, 1)`.
    - `tile_regs_acquire()`.
    - `in0[g*tiles_per_expert .. +tiles_per_expert] * scalar → dest`.
    - `dest * in1[g*tiles_per_expert .. +tiles_per_expert] → dest`.
    - `tile_regs_commit(); tile_regs_wait();`
    - Pack `tiles_per_expert` tiles into `cb_out[g*tiles_per_expert ..]`.
    - `tile_regs_release();`
    - `cb_push_back(cb_out, tiles_per_expert)` (pushes incrementally so down
      gather can start on group 0 without waiting for group 7).
    - `cb_pop_front(cb_scalar, 1)`.
  - After loop: pop `cb_in0_wait`, `cb_in1_wait` if `PopInputs`.
- **`num_experts=1`** degenerates to current behavior.

### 5.3 Why option 1 (vs calling EltwiseMul 8 times from moe_kernel)
- Avoids 8× `hw_startup` / `reconfig_data_format` overhead in TRISC.
- Avoids 8× wait/pop bookkeeping in moe_kernel.
- Keeps incremental output push (down gather overlaps with mul on later
  experts) by pushing per group inside the loop.
- Mirrors how `matmul_expert` already handles num_experts internally.

### 5.4 Edge cases / constraints
- `num_tiles % num_experts == 0` must hold (assert at compile time).
- `cb_scalar_src` must hold at least `num_experts` scalars contiguously
  starting at `scalar_index_offset`.
- `cb_out` downstream consumer (down_proj gather) must be OK receiving
  tiles in per-expert chunks — it already processes per-expert tiles
  separately so this matches.

## 6. matmul_expert integration into moe_kernel

### 6.1 Gate_proj / up_proj (steps 6, 7) — non-accum
- Replace `dram_streaming_matmul` call with `matmul_expert` call,
  `accum_experts=False`.
- `per_core_n = 32` (256 per device / 8 DRAM cores).
- Non-accum emits `num_experts=topk=8` groups of `N_per_core/32=1` tile
  × `M/32=1` tile = 1 tile per expert per core → 8 tiles total into cb_out.
- Gate_proj needs SiLU. DRAM path has `dram_fuse_silu` already. SRAM path
  does **not** — if any hot expert maps to gate_proj, SiLU has to be fused
  into the SRAM path (follow-up work, see §8).
- Index tensor: the `[1, 8]` encoded selection from §4.2 goes to both
  gate_proj and up_proj cores on every DRAM streamer.

### 6.2 down_proj (step 11) — accum
- Replace `dram_streaming_matmul` call with `matmul_expert` call,
  `accum_experts=True`.
- `per_core_n = 896` (7168 / 8 cores).
- **DRAM accum cap of 2 experts** is load-bearing. With topk=8, a single
  `matmul_expert` call can accumulate at most 2 cold experts. Options:
  - (a) Rely on enough hot coverage that ≤2 experts per call hit DRAM;
    SRAM path handles the rest. The `sram_mm(); dram_mm();` order already
    accumulates SRAM contributions into dst before DRAM runs, so SRAM
    results carry into the DRAM pass.
  - (b) Call `matmul_expert` multiple times, chaining accum across calls
    (would need API change — not current behavior).
  - (c) Require the router to produce ≤2 cold experts per token (policy).
  - **Decision: (a) for initial bring-up, revisit if unachievable.**
- Output is one `[1, 7168]` per device (the device's N-slice of the
  post-reduce-over-experts vector).

### 6.3 Expert scalar placement
With `matmul_expert` accum on down_proj, the per-expert scalar multiply
can't happen **after** accum (scalars differ per expert). Move the scalar
multiply into step 8 (EltwiseMul before down_proj), applied per-expert
via the num_experts loop (§5). Down_proj then accumulates
already-scaled values.

## 7. Test coverage gaps

`test_matmul_expert.py` is very thorough on the matmul kernel itself but
doesn't cover the exact shape/mode combinations needed for MoE integration.
Three add-tests-first items:

1. **gate/up_proj with TP8 + topk=8 + production shape**
   - K=7168, N_per_device=256, N_per_core=32, num_experts=8, accum=False
   - Production-shape tests today (e.g. `test_moe_gate_proj_shape` ~line
     1771) run `tp_expert=False` (EP mode, 1 expert per device). Nothing
     tests `tp_expert=True` at production shape with non-accum num_experts=8.
2. **down_proj with TP8 + accum + topk=8 + 8-DRAM-core grid**
   - K_per_device=256, N=7168, N_per_core=896, num_experts=8, accum=True
   - Current accum tests cap at 2 experts (DRAM limit). Need to validate
     the SRAM+DRAM split under policy (a) from §6.2.
   - `test_hybrid_expert_irregular_sram_down_grid` (~line 1745) is the
     closest existing test but uses a different grid.
3. **Cross-device ReduceToOne after per-device accum down_proj**
   - Verify routed+shared partial add + ReduceToOne produces correct
     full output at the fused-MoE level.
   - Belongs in `test_moe_mlp.py`, not `test_matmul_expert.py`.

## 8. Known gaps / follow-ups
- **SRAM gate_proj SiLU fusion**: SRAM path has no SiLU today. Needs to be
  added before any hot gate_proj experts are supported. Could be a new
  template bool `fuse_silu` plumbed into the compute side.
- **DRAM accum > 2**: policy (a) is a workaround. A proper fix would chain
  accum across multiple `matmul_expert` calls (needs kernel API change).
- **Flag-insert upstream producer**: the step that produces the new
  encoded selection tensor (§4.2) must exist before integration. Where
  it lives (inside TopK? a thin following op?) is TBD.
- **`table_idx_arr` simplification**: lower-15-bit-is-already-position
  means the indirection table can go away. Safe as a follow-up cleanup.

## 9. Rollout order

Current implementation is staged as **Phase 1A → 1B → 2**. Phase 1A defers
the `matmul_expert` kernel swap so TP8 plumbing + reduce + EltwiseMul loop
can be validated independently.

### Phase 1A — TP8 plumbing with existing `DRAMStreamingMatmul` ✅ COMPLETE
Goal: prove TP8 sharding + cross-device reduce end-to-end. Keeps the
existing routed matmul kernel to minimize delta.
1. ✅ Shard routed gate/up weights column-parallel (N/8=256 per device)
   and down weights row-parallel (K/8=256 per device). Replaced
   `ReplicateTensorToMesh` with `ShardTensor2dMesh` via
   `moe_routed_expert_tp8_torch_for_cache` in `weights/transforms/moe.py`.
2. ✅ Extend `EltwiseMul` with `num_experts` looping (§5). Loop is wired
   in `unified_kernels/eltwise_mul.hpp` (BRISC scalar feed + TRISC per-
   expert `in0*s*in1` iteration, pushes per-group so downstream
   consumer overlaps). `num_experts=1` default preserves old behavior.
   **Note**: not yet wired into `moe_kernel.cpp` — today gate/up matmul
   processes 1 expert × 8 streamer cores EP-style and the old per-expert
   scalar broadcast path still applies. EltwiseMul sees num_experts=1.
3. ✅ `enable_reduce_to_one=True` through routed path. Per-device
   partial routed+shared → ReduceToOne at ROOT1.
4. ✅ `test_moe_fused_with_reduce` computes per-device golden; residual
   added only on ROOT1; sum equals full MoE output.
5. ✅ **Hit**: Reduce-output PCC 0.9915 (target ≥ 0.97), Reference MoE
   PCC 0.9915 (target ≥ 0.95), 100 iterations no hangs.

### Phase 1B — swap DRAM matmuls to `matmul_expert` ✅ COMPLETE
Goal: use the compressed-weight expert matmul kernel for gate/up/down.
All three matmuls now go through `MatmulExpertCompressedDRAM` with all
8 selected experts running on the DRAM path (no SRAM, no hot experts
yet). Anchor test passes (see §0).

New infrastructure required in `op.py`:
- Build `CompressedTensor` for each of gate_proj / up_proj / down_proj
  (replaces the flat DRAM weight buffers). Uniform bfp4_b → assigner =
  `UniformPrecisionAssigner(ttnn.bfloat4_b)` (see
  `compressed_tensor/assigners.py`). 256 CTs per proj, sharded TP8 via
  `mesh_mapper_config=Shard2dMeshMapper(dims=(0,1))` + the TP8 torch
  layout from `moe_routed_expert_tp8_torch_for_cache`.
- Per-device DRAM meta + fmt tables via `create_dram_expert_tensors_multi_device`
  (at `micro_ops/matmul_expert/op.py:617`). Returns per-device tuple
  `(in1_backing, meta_tensors, fmt_tensors, meta_l1_addr, fmt_l1_addr,
  per_core_values, num_in1_buffers)`. `in1_backing` is a REPLICATED L1
  tensor that serves as the CB backing for triple-buffered DRAM streaming
  (its addr seeds `cb_in1_base_shifted`) — NOT a weight tensor.
- `fmt_dram_tensor`: one big DRAM width-sharded buffer per device, indexed
  by `bank_id * fmt_per_bank + core_in_bank_idx * fmt_per_core +
  expert_idx * fmt_per_expert`. Created inside
  `create_dram_expert_tensors_multi_device`.
- Meta table layout (per-core, L1-resident): `meta_stride = 2 +
  num_subblocks_k * per_core_n`; for each expert `e`, entries are
  `[in1_addr, dram_col_offset, block_size_0, ..., block_size_{n-1}]`,
  indexed by GLOBAL expert_idx (no table_idx indirection).
- `cb_fmt` CB (size `fmt_per_expert_bytes`, triple-buffered conceptually
  but one slot at a time is pushed/popped) to stage per-expert fmt data
  on each streamer core.
- Pipeline semaphore (`pipeline_sem_id`) and per-core pipeline coords
  (`next_core_noc_x/y`, `core_in_bank_idx`, `cores_per_bank`). For the
  MoE case where each DRAM core owns a full bank (topk=8, 8 DRAM cores,
  8 banks), `cores_per_bank=1` is the degenerate self-signal case — the
  pipeline semaphore is still compile-time-referenced but inactive.
- L1-resident expert index array per streamer core (`index_l1_addr`,
  populated by mcast of encoded `[1, 8]` uint16 from TopK). `table_idx`
  is passed (template arg) but the DRAM path doesn't use it — the kernel
  indexes meta/fmt directly by global expert_idx. Can feed a trivial
  identity array or skip (pass any valid L1 addr).

**Kernel CTArgs summary (from `unified_kernels/matmul_expert_compressed_dram.hpp`):**
- `ReaderCTArgs` (26 params): `cb_in0, cb_in1, cb_out, cb_index,
  num_tiles_k, subblock_k, num_subblocks_k, per_core_n, bank_id, vc,
  meta_l1_addr, cb_in1_size_bytes, noc_max_page_size, core_in_bank_idx,
  pipeline_sem_id, next_core_noc_x, next_core_noc_y, cores_per_bank,
  num_active_experts, table_idx_l1_addr, index_l1_addr, cb_fmt,
  fmt_dram_addr, fmt_per_expert_bytes, fmt_per_core_bytes,
  accum_experts=0, index_offset=0`.
- `ComputeCTArgs` (16 params): `cb_in0, cb_in1, cb_out, cb_index,
  num_tiles_k, subblock_k, num_subblocks_k, per_core_n, fmt_l1_addr,
  num_active_experts, table_idx_l1_addr, index_l1_addr, cb_fmt,
  accum_experts=0, fuse_silu=0, index_offset=0`.
- `WriterCTArgs`: empty.
- `Op<CTArgs, IsActiveCore, pop_in0=true, pop_index=true>`: `IsActiveCore`
  false = no-op. `pop_*` control whether to consume inputs (relevant when
  chaining gate_proj → up_proj which share cb_in0 activations and cb_index).

Kernel-side changes in `moe_kernel.cpp`:
- Replace include `../../unified_kernels/dram_streaming_matmul.hpp` with
  `../../unified_kernels/matmul_expert_compressed_dram.hpp`.
- Swap `DRAMStreamingMatmul::ReaderCTArgs` / `ComputeCTArgs` for
  `MatmulExpertCompressedDRAM::ReaderCTArgs` / `ComputeCTArgs` at:
  - gate_proj: `moe_kernel.cpp:146-162` (reader), `750` (compute), `1181` (op)
  - up_proj:   `moe_kernel.cpp:165-181` (reader), `763` (compute), `1191` (op)
  - down_proj: `moe_kernel.cpp:200-216` (reader), `796` (compute), `1301` (op)
- gate_proj uses `fuse_silu=1`, `accum_experts=0`, `pop_in0=false,
  pop_index=false` (activations + index reused by up_proj).
- up_proj uses `fuse_silu=0`, `accum_experts=0`, `pop_in0=true,
  pop_index=true`.
- down_proj uses `fuse_silu=0`, `accum_experts=1`, `pop_in0=true,
  pop_index=true`.
- **Accum cap (RESOLVED in kernel)**: the DRAM matmul_expert kernel was
  updated to reconfigure L1_ACC for every expert iteration, not just
  `i==0..1`. All 8 topk experts accumulate correctly on down_proj.
  Policy (a) / "≤2 cold experts per token" is no longer a kernel
  constraint — it's purely a future perf trade-off for SRAM hit-rate.
- **Scalar placement (not yet moved)**: today the per-expert scalar is
  still applied through the pre-Phase-1B scalar-broadcast path (one
  scalar per expert × 8 EP-style streamer cores). Moving the scalar
  into EltwiseMul's num_experts loop (§6.3) is tracked as follow-up; it
  becomes a requirement only when matmul_expert moves to non-accum
  gate/up emitting num_experts groups into one CB (currently each
  expert still flows through its own streamer core).

### Phase 2 — SRAM path + new index encoding
1. Upstream encoding producer for §4.2 selection tensor (`is_sram | position`).
2. SRAM SiLU fusion (unblocks hot gate_proj experts).
3. Hard-code 2 hot experts (rig gate) for initial validation; PCC similar.

## 10. Open questions
- Which upstream op produces the `is_sram | position` encoding? TopK
  followup, or a dedicated flag-insert op? (Probably the former for
  perf; the latter for modularity.)
- Is policy (a) "≤2 cold experts per token" realistic at production-scale
  hot/cold splits? Needs measurement on real routing data.
- For cross-device ReduceToOne: reuse existing reduce_to_one_b1 pattern,
  or does TP8 call for a different topology? (Phase 1B shipped with
  `reduce_to_one_b1` — answered unless a topology change is needed.)

## 11. Bugs hit & fixes landed during Phase 1A/1B

Chronological list of non-obvious bugs that cost real debug time. Keep
this updated as a reference so the next person doesn't re-hit them.

### 11.1 Per-device golden must sum all top-8 experts, not per-expert
- **Symptom**: PCC 0.8629 on anchor test right after Phase 1A weight
  sharding landed.
- **Root cause**: initial per-device golden in `test_moe_fused_with_reduce`
  summed only the expert assigned to that device, missing the other 7
  selected experts whose N-slices also land on this device.
- **Fix**: compute per-device golden as `sum_over_topk( gate[e] @
  W_gate[e][:, device_n_slice] * up[e] @ W_up[e][:, device_n_slice] *
  scale[e] ) @ W_down[e][device_k_slice, :]`, not just one expert.

### 11.2 d=4↔6, d=5↔7 row swap on tray 3
- **Symptom**: d4/d5/d6/d7 showed systematically lower PCC than d0-d3 on
  the 4×2 submesh; pairs d4↔d6 and d5↔d7 matched when logical-row-swapped.
- **Root cause**: REV_B galaxy tray 3 has device IDs 24-31 but the
  physical row order on the 4×2 submesh is non-sequential; our per-
  device weight preprocessor was shape-mapping by logical device index.
- **Fix**: weight TP8 preprocessor now maps along the mesh row/col axes
  that match the physical 4×2 layout, not `device_id` order.

### 11.3 Shared-expert face-view CB30 override (most impactful)
- **Symptom**: `test_moe_fused_no_reduce` d2=0.9627, d5=0.9679 (below 0.97
  threshold); shared-expert output measured at ~11–47% of expected
  magnitude on half of the devices; reduce-to-one variant still passed
  because correlation survived.
- **Root cause**: shared gated_reduce uses **face-view** tile format —
  `tiles_per_k=8`, `k_num_tiles=1`, page_size=512, tile=[32,32] (each
  "tile" slot is a 16×16 face = 256 bf16). `setup_gated_reduce` sets
  CB30/CB31/CB32/CB33 correctly, but `_overlap_cbs_with_sdpa_buffer`
  in `op.py` re-created CB30's format descriptor during hybrid kv_buf/
  out_buf allocation and used the default `page_size=64, tile=[1,32]`.
  Result: `add_tiles` read only 32 bf16 per tile slot (of 256 expected)
  and `pack_tile` wrote only 32 bf16 — 1/8 of the shared-expert output
  reached downstream.
- **Fix**: `_overlap_cbs_with_sdpa_buffer` now pulls `face_tile_size`
  and `face_tile_desc` from `shared_ctx.gated_reduce_params` for **all**
  CBs involved (CB30 + CB32 + CB33), matching what `setup_gated_reduce`
  configured. See `project_hot_cold_moe_shared_zero.md` memory file for
  the full RCA and DPRINT recipe.
- **Result**: per-device PCC went from d2=0.9627/d5=0.9679 to ≥0.9988
  on all 8 devices.

### 11.4 Expert index must be read from cb_index CB, not CTArgs::index_l1_addr
- **Symptom**: matmul_expert reading wrong expert weights per iteration
  after the CTArgs swap; partial/all-zero outputs.
- **Root cause**: original `matmul_expert` test harness writes the
  selection into an L1 array at `index_l1_addr` and the kernel reads
  that pointer. In the fused MoE pipeline, TopK mcasts the selection
  into `cb_index` — the L1 address is not pre-populated; the kernel
  must consume from the CB.
- **Fix**: kernel reads `get_read_ptr(cb_index)` per iteration; the
  CTArgs `index_l1_addr` is now unused on the MoE path (kept for
  standalone-test backwards compat).

### 11.5 num_active_experts / tile descriptor consistency
- **Symptom**: per-core meta-tensor indexing off by one; occasional
  crashes or wrong weights.
- **Root cause**: `num_active_experts` has to thread consistently from
  `setup_matmul_expert_dram` → CTArgs → meta/fmt tensor strides → CB
  sizes. Early wiring set `num_active_experts=1` as a temporary
  single-expert test rig and missed updating some call sites.
- **Fix**: `num_active_experts=8` is threaded through all three projs
  in `op.py` lines 1359, 1380, 1475 and the kernel CTArgs.

### 11.6 CompressedTensor TP8 sharding path
- **Infra added**: `CompressedTensor` uniform bfp4_b for each of gate/
  up/down (256 CTs per proj); `Shard2dMeshMapper(dims=(0,1))` + the
  TP8 torch layout from `moe_routed_expert_tp8_torch_for_cache`.
- **Per-device DRAM meta + fmt tables**: via
  `create_dram_expert_tensors_multi_device` in
  `micro_ops/matmul_expert/op.py:617`. Returns per-device
  `(in1_backing, meta_tensors, fmt_tensors, meta_l1_addr, fmt_l1_addr,
  per_core_values, num_in1_buffers)`. `in1_backing` is a REPLICATED L1
  tensor used as the CB backing for triple-buffered DRAM streaming,
  **not** a weight tensor.
- **cb_fmt**: sized `fmt_per_expert_bytes`, triple-buffered slots
  push/pop per expert on the streamer core.
- **Pipeline semaphore**: `cores_per_bank=1` (one streamer per bank
  for MoE) is the degenerate self-signal case — sem still compile-
  time-referenced but inactive.
