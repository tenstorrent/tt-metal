# SRAM Routed Expert — Separate Pipeline Plan

Hot-expert (SRAM-resident) routed pipeline that runs alongside the existing
DRAM routed path and shared expert path, structurally mirroring the shared
expert sub-pipeline.

This supersedes the same-core hybrid dispatch in
[`HOT_COLD_INTEGRATION.md`](./HOT_COLD_INTEGRATION.md) §2 for the SRAM phase.
Phase 1A/1B (TP8 plumbing + DRAM expert kernel swap) remain as documented
there.

## Why a separate pipeline

The hot/cold doc places SRAM and DRAM expert kernels on the **same 8 routed
DRAM streamer cores**, with each core owning the full K-slice for its bank.
That design has no K-shard, so no reduce step — but L1 per slot is large
(~129 KB for down_proj), capping practical hot coverage at ~4 experts per
proj.

This plan instead places SRAM gate/up on the **64 shared gate/up cores** and
SRAM down on the **112 shared down cores** (mirroring shared expert
placement), with KN-sliced compute. Per-slot L1 drops by ~15× on down_proj
(~8 KB), enabling 50+ hot slots if budget allows. The cost is the K-direction
reduce after gate/up — addressed by extending the existing `GatedReduce`.

## Pipeline at a glance

```
Sender:
  TopK → encoded indices → reorder by bit-15 (SRAM-first, DRAM-after) → mcast

SRAM routed sub-pipeline (parallel to DRAM routed + shared):
  64 SRAM gate cores: gate_proj SRAM matmul (K-sliced, per-expert outputs)
  64 SRAM up cores:   up_proj   SRAM matmul (K-sliced, per-expert outputs)
  Gate gather A:      64 cores → sender
  Up gather B:        64 cores → sender
  Extended GatedReduce on sender:
                      per expert: silu(sum_K(gate_e)) * sum_K(up_e) * scale_e
                      output: expert-major, full N per expert, n_sram dense + pad

DRAM routed sub-pipeline (existing):
  8 DRAM cores: gate_proj/up_proj/eltwise_mul + scalar
  down_proj_gather: 8 → sender, expert-major, n_dram dense + pad

Two mcasts → same merged in0 CB on down cores:
  SRAM mcast: src[0..n_sram] → dst[0..n_sram]                    (runtime size)
  DRAM mcast: src[0..n_dram] → dst[n_sram..n_sram+n_dram]        (runtime size + offset)

112 down cores:
  shared_down_matmul (existing)
  sram_down_matmul   (new, accum_experts=true, bit-15 filter)
  eltwise_add: shared_down + sram_down → combined_down (NEW)
  residual_add: combined_down + residual → residual_add_out (re-routed in0)
  output_gather → output_mcast (existing) → 8 DRAM cores into add_cb_in1

8 DRAM cores:
  eltwise_add: DRAM_down + add_cb_in1 → final (existing)
```

## 1. Index encoding (no reorder)

### Static map encoding
`weights/prepare.py:create_gate_indices_tensor` is updated to take a
`sram_expert_ids: list[int] = ()` parameter. For each global expert ID e
∈ [0, 256):

- e ∈ sram_expert_ids: encoded[e] = `(1 << 15) | slot_for[e]` (compact slot
  in 0..T-1, position = slot)
- else: encoded[e] = e (DRAM, bit-15 = 0)

Default empty preserves Phase 1B behavior (identity arange).

### No sender reorder needed
The TopK output flows through `index_mcast` (`moe_kernel.cpp:1322`)
unchanged. Bit-15 propagates as an opaque label.

The merged `in0` layout (SRAM-first dense, DRAM-after dense) creates an
**activation offset mismatch only in the down_proj** kernels, since gate/up
matmuls share one cb_in0 across all experts (no per-expert in0 stride).
Down_proj solves it with a **per-path counter** internal to each kernel —
no global index sort. See §6 for kernel logic.

## 2. SRAM gate_proj / up_proj matmul (64 + 64 cores)

### Cores
- gate: 64 `is_shared_gate_compute_core` cores (K-parallel × N-parallel = 64)
- up:   64 `is_shared_up_compute_core` cores (same layout)

Sequential timeslot with shared expert KN-sliced matmul (or new dedicated
slot — see open questions).

### Inputs
- `cb_in0`: `rmsnorm_output_cb` (already mcasted, `K_per_core` tiles)
- `cb_in1`: new sharded SRAM weight tensor, T expert slabs of
  `[K_per_core × N_per_core]` bfp4_b per core
- `cb_index`: TopK output (encoded values, original score order)

### Output
- `cb_out`: `num_active_experts × N_per_core` tiles per core, with the
  kernel's existing padding (real SRAM-flagged experts pushed in iter
  order, padding tail)

### Kernel
`MatmulExpertCompressedSRAM::Op` with:
- `accum_experts = false`
- `sram_k_per_core = K_per_core` (this core's K-slice)
- `sram_k_offset = (core_idx // n_parallel) × K_per_core`
- `cb_out_sram = 0` (write directly to cb_out)

### L1 per core
```
shared_gate slab     = K_per_core × N_per_core × bfp4_tile (≈ few KB)
T SRAM routed slabs  = T × same
sram_base_addrs[T]   = 4T bytes
sram_fmt[T]          = T × meta_words_per_expert × 4 bytes
```

T bound determined by L1 budget (TBD in bring-up).

### Host setup (new)
`MoeRoutedExpertOp.setup_matmul_expert_sram` modeled on
`setup_matmul_expert_dram` (`op.py:470–733`) but smaller — no DRAM streaming,
no fmt double-buffer, just per-core sharded weight + L1 base_addrs/fmt
tables.

## 3. Gate gather A + Up gather B

Reuse `MoeGather` exactly as shared expert does (`moe_kernel.cpp:1289`,
`:1351`). Each of the 64 SRAM gate/up cores sends its
`num_active_experts × N_per_core` tiles to sender. Per-core sender_idx
matches existing `setup_kn_per_core_values` layout.

Sender CBs (per group, expert-major, K-partials packed for each expert):
- `sram_group1_cb`: gate K-partials (n_sram experts × k_parallel × N_per_core
  tiles, padded tail)
- `sram_group2_cb`: up K-partials (same)

These can reuse shared expert's `group1_cb` / `group2_cb` if sequential,
or be new CBs if concurrent.

## 4. Extended GatedReduce (sender)

Wrap the existing `gated_reduce.hpp:83–134` body in an outer
`for e in 0..num_active_experts` loop. Three changes:

1. **Skip filter**: `if (!is_sram_expert(index_ptr[e])) { drain CBs; continue; }`
2. **Inner body unchanged**: pairwise add over `tiles_per_k` K-partials,
   silu(gate_sum) * up_sum
3. **Per-expert scale mul**: extra `mul_tiles` against `expert_scale[e]`
   from `mul_cb_scalar_src` (already mcasted by existing `expert_scale_mcast`,
   `moe_kernel.cpp:1336`)

Output: `num_active_experts × N_per_core` tiles into `sram_mcast_src_cb`,
expert-major, dense-packed (n_sram real, pad tail).

### Sizing match
The output stride must match the DRAM gather's `expert_dst_stride`
(= `num_gate_proj_cores × per_core_n × tile_size = 8 × 32 × 2 = 512` bytes
per expert, op.py:1530). This makes the two source CBs layout-compatible
for the dual mcast.

## 5. Two mcasts to merged in0

### Sender-side counts (single 8-entry walk of reordered index)
```cpp
n_sram = count(idx[i] s.t. is_sram_expert(idx[i]))
n_dram = num_active_experts - n_sram
```

### Dispatch
```
SRAM mcast: src=sram_mcast_src_cb,        len=n_sram × 8 tiles, dst_off=0
DRAM mcast: src=down_proj_gather_dst_cb,  len=n_dram × 8 tiles, dst_off=n_sram × 8 tiles
```

Both target the same `down_in0_cb` on **both** the 8 DRAM down streamer
cores and the 112 SRAM down cores. Merged dst is expert-major:
`[SRAM_0_256, SRAM_1_256, ..., DRAM_0_256, DRAM_1_256, ...]`.

### New infra
A runtime-sized variant of `Mcast::Op` is required. Existing op has
compile-time `num_pages_to_send`; new variant takes both length and
dst_offset as runtime args fed from BRISC counters.

## 6. SRAM down_proj matmul (112 cores)

### Cores
112 `is_shared_mcast_receiver_core` cores (same as shared expert down).
N-parallel only (no K-split), mirroring shared down's `Matmul::Op` pattern.

### Inputs
- `cb_in0`: merged dst CB (expert-major, 8 tiles × 8 experts = 64 tiles
  per core)
- `cb_in1`: new sharded SRAM down weights, T expert slabs of
  `[K × N_per_core]` bfp4_b per core
- `cb_index`: reordered index (same one used everywhere)

### Output
- `cb_out`: 1 tile per N-position per core (single accumulated tile)

### Kernel
`MatmulExpertCompressedSRAM::Op` with `accum_experts = true`. Existing
`is_sram_expert` filter skips DRAM-flagged entries naturally.

### Per-core dimensions
```
K_total           = 8 tiles    (256 / 32 elements per tile)
out_w (N_per_core)= 2 tiles    (7168 / 112 = 64 elements)
cb_in0_num_pages  = 64         (= K_total × num_active_experts)
sram_k_per_core   = 8          (full K, no split)
sram_k_offset     = 0
in0_page_size     = 64         (1×32 bf16 face tile)
```

### Activation offset — per-path counter
The merged `in0` is dense-packed (SRAM block first, DRAM block after), but
the index is in TopK score order. So `exp_i` (iteration position) does not
match the slot in `in0`. Both kernels track an internal counter that
increments only for the experts they actually process.

**SRAM down kernel** (small edit to `matmul_expert_compressed_sram.hpp`
accum branch):
```cpp
uint32_t sram_count = 0;
for (exp_i = 0; exp_i < num_active_experts; exp_i++) {
    if (!is_sram_expert(idx[exp_i])) continue;
    in0_offset = sram_count × num_tiles_k × in0_page_size;
    // override cb_in0 rd_ptr, run compressed_custom_mm_block, ...
    sram_count++;
}
```

**DRAM down kernel** (small edit to `matmul_expert_compressed_dram.hpp`
accum branch): same pattern + a runtime `n_sram` arg fed from sender BRISC
(already needed for mcast sizing in §5):
```cpp
uint32_t dram_count = 0;
for (exp_i = 0; exp_i < num_active_experts; exp_i++) {
    if (is_sram_expert(idx[exp_i])) continue;
    in0_offset = (n_sram + dram_count) × num_tiles_k × in0_page_size;
    // ...
    dram_count++;
}
```

`n_sram` propagates as a runtime arg via the existing common-RT-arg base.
~5 lines per kernel.

### L1 per core
```
shared_down slab  = K × N_per_core × bfp4 ≈ 8 KB
T SRAM slots      = T × ~8 KB
base_addrs/fmt    = small
```

With ~600 KB free L1, T can grow to 50+. Bring-up at lower T (4–8).

## 7. Post-down: shared + SRAM merge → existing pipeline

Insert ONE new `EltwiseAdd::Op` on the 112 cores between the two down
matmuls and the existing residual_add:

```cpp
shared_down_matmul()                                     // existing
sram_down_matmul()                                       // new
eltwise_add(shared_down_out, sram_down_out)              // NEW: single per-tile add
                       → combined_down_out_cb
residual_add(combined_down_out, residual)                // existing, re-routed in0
                       → residual_add_out_cb
output_gather  (112 → sender)                            // unchanged
output_mcast   (sender → 8 DRAM cores into add_cb_in1)   // unchanged

// 8 DRAM cores:
eltwise_add(DRAM_down, add_cb_in1) → final               // existing
```

Residual gets added once. Final 8-core add unchanged.

## 8. Summary of changes

### New host-side (op.py)
- `MoeRoutedExpertOp.setup_matmul_expert_sram(...)` — per-proj sharded
  weight + base_addrs/fmt L1 tables (gate, up, down)
- `setup_gated_reduce_routed(...)` — extended GatedReduce sizing
- New CT args threaded through `_build_compile_time_args`
- New RT arg (`n_sram`) plumbed to DRAM down kernel via common-RT-arg base

### New kernel-side
- `gated_reduce.hpp`: outer expert loop + per-expert scalar mul + bit-15 skip
- New `Mcast::Op` variant (or extension) with runtime length + dst_offset
- `matmul_expert_compressed_sram.hpp`: per-path `sram_count` counter in
  accum branch (~5 lines) — replaces `exp_i` for in0 offset
- `matmul_expert_compressed_dram.hpp`: per-path `dram_count` counter +
  runtime `n_sram` offset in accum branch (~5 lines)
- moe_kernel.cpp orchestration: SRAM gate/up matmul calls, two-mcast
  block, SRAM down matmul call, post-down eltwise_add

### New CBs
- `sram_gate_weights_cb`, `sram_up_weights_cb`, `sram_down_weights_cb`
- `sram_mcast_src_cb` (extended GatedReduce output)
- `sram_down_out_cb` (SRAM down matmul output)
- `combined_down_out_cb` (shared + SRAM eltwise_add output)
- L1-resident `sram_base_addrs`, `sram_fmt` per proj per core (small)

### Modified existing
- `weights/prepare.py:create_gate_indices_tensor`: take `sram_expert_ids`,
  encode bit-15 + compact slot
- `residual_add` in0 source: re-routed from `shared_down_out_cb` to
  `combined_down_out_cb`
- Test files: decode bit-15 + slot→eid inverse map for index comparisons

### Unchanged
- `deepseek_moe_gate.hpp` (TopK kernel — bit-15 propagates as opaque label)
- `MatmulExpertCompressedSRAM` kernel (already filter + pad)
- `MatmulExpertCompressedDRAM` kernel (already filter + pad)
- DRAM gate_proj/up_proj/down_proj path
- Shared expert gate/up/down/gather/mcast path
- `MoeGather` kernel
- Final 8-core `eltwise_add`
- ReduceToOne path

## 9. Open questions

- **T (slot count) per proj**: bring-up at 4–8; production target tied to L1
  budget. Compute exact ceiling once `setup_matmul_expert_sram` lands.
- **Sequential vs concurrent SRAM/shared on same cores**: sequential is
  simpler. Concurrent would need disjoint CBs and TRISC slot scheduling.
- **CB reuse vs new**: `sram_group1/2` reuse shared `group1/2`? Bring-up
  with new CBs first, alias later if L1-tight.
- **Hot expert placement policy**: hard-coded list (bring-up) → profile-driven
  (production). Out of scope here.
- **DRAM accum cap interaction**: with most experts routed to SRAM,
  DRAM accum sees ≤2 experts naturally — perf-friendly without extra policy.

## 10. Step-by-step implementation plan

Strategy: **modify existing ops first** (small, behavior-preserving changes
that put encoding + plumbing in place), **then add new ops one at a time**,
each with a no-op default and a single isolated change. Anchor test
(`test_moe_fused_with_reduce`) must pass at PCC ≥ 0.9915 after every step
when `num_sram_experts = 0`.

Each step lists: Files touched • Behavior change • Test gate.

---

### Phase 0 — Existing-op modifications (no new behavior)

Goal: land all encoding, reorder, and Mcast plumbing while
`num_sram_experts = 0`. Anchor test PCC must be unchanged after every step.

**Step 0.1 — Static-map encoding**
- Files: `weights/prepare.py:create_gate_indices_tensor`
- Change: add `sram_expert_ids: list[int] = ()` kwarg. When empty, behavior
  identical to today (`arange(256)`). When populated, encode bit-15 + slot
  for those eids.
- Test: anchor test PCC unchanged (default empty).
- Risk: low. Pure data change.

**Step 0.2 — Verify DRAM kernel filter + padding**
- Files: read `unified_kernels/matmul_expert_compressed_dram.hpp`
- Change: none expected — kernel already filters (`:400, :711, :1007`) and
  pads (`:1085–1089`).
- Test: anchor test PCC unchanged (filter never fires when no encoded
  entries have bit-15 set).
- Risk: zero. Read-only validation.

**Step 0.3 — Runtime-sized Mcast variant**
- Files: `unified_kernels/mcast.hpp` (new template variant or new struct
  `RuntimeMcast`)
- Change: add `Mcast::Op` variant taking runtime `num_pages` and
  `dst_offset_bytes` from BRISC RT args. Existing CT-arg-sized variant
  unchanged.
- Test: standalone unit test (NCRISC sender writes N tiles at offset O,
  receiver verifies). Anchor test unchanged (variant not yet called).
- Risk: medium. New kernel infrastructure; bake in unit test before use.

**Step 0.4 — Test golden decode**
- Files: `tests/unit_tests/test_moe_mlp.py`,
  `tests/unit_tests/test_moe_routed_expert.py`
- Change: read encoded indices with `& 0x7FFF`; for SRAM-flagged entries,
  decode via slot→eid inverse map. With `num_sram_experts = 0`, mask is
  no-op.
- Test: anchor test PCC unchanged.
- Risk: zero. Defensive only.

---

### Phase 1 — First new op: SRAM gate_proj matmul

Goal: stand up the SRAM gate_proj matmul on the 64 shared gate cores,
validated in isolation against torch reference. SRAM output goes nowhere
downstream yet — drained locally — so anchor test PCC unchanged.

**Step 1.1 — `setup_matmul_expert_sram` host helper (gate only)**
- Files: `fused_ops/moe/op.py`
- Change: new static method
  `MoeRoutedExpertOp.setup_matmul_expert_sram(...)` for gate_proj. Allocate
  per-core sharded weight tensor (T slabs of `K_per_core × N_per_core`
  bfp4_b), per-core L1 `sram_base_addrs[T]` and `sram_fmt[T × meta_words]`
  tables. Returns dict matching `setup_matmul_expert_dram` shape.
- Test: instantiate with T=2; assert allocation succeeds, base_addrs
  match expected L1 layout, fmt encodes a known per-tile format.
- Risk: medium. New host-side surface; needs careful L1 budget assertion.

**Step 1.2 — Wire SRAM gate_proj kernel call into `moe_kernel.cpp`**
- Files: `fused_ops/moe/moe_kernel.cpp`, `fused_ops/moe/op.py`
  (CT args + flag)
- Change: add `#include matmul_expert_compressed_sram.hpp`. Add
  `MatmulExpertCompressedSRAM::Op` call gated by
  `Core::is_shared_gate_compute_core && sram_active`, after the existing
  `KNSlicedMatmul` call (`:1276`) for shared expert. Output to new
  `sram_gate_out_cb`. Drain locally (`pop_out=true`) so no downstream
  effect. Keep `pop_in0=false, pop_index=false` (DRAM up still needs them).
- Test 1 (T=0): `sram_active=false` → kernel call is no-op (`if constexpr`).
  Anchor test PCC unchanged.
- Test 2 (T=1, single hot expert): `sram_active=true`, place 1 expert in
  SRAM, encoding produces bit-15+slot for it. SRAM kernel runs, output
  drained locally. DRAM kernel skips this expert via existing filter.
  **DRAM-only result is now MISSING that expert's contribution → anchor
  test PCC will drop.** Validate via DPRINT that SRAM output tiles match
  torch reference for that expert. Don't gate on PCC at this step.
- Risk: medium. First time bringing the SRAM kernel online inside MoE.

**Step 1.3 — Bypass: copy SRAM gate_out into DRAM gate_out path**
- Files: `fused_ops/moe/moe_kernel.cpp`
- Change: optional debug bypass — after SRAM kernel writes its output,
  NOC-write the SRAM tiles into the DRAM gate_proj's cb_out at the right
  per-expert offset (so eltwise_mul sees them). Allows anchor test PCC
  to recover at T=1 without requiring gather/reduce/mcast yet.
- Test: anchor test PCC ≥ 0.9915 with T=1 SRAM expert.
- Risk: medium. Throwaway code; remove once Phase 4 gather/reduce lands.

---

### Phase 2 — SRAM up_proj matmul (mirror Phase 1)

**Step 2.1 — `setup_matmul_expert_sram` for up_proj**
- Mirror Step 1.1. Same shape, separate sharded tensor.
- Test: allocation succeeds; L1 budget per core (gate + up combined) within
  cap.

**Step 2.2 — Wire SRAM up_proj kernel call**
- Mirror Step 1.2. Now `pop_in0=true, pop_index=true` (last consumer).
- Test (T=1): both gate and up SRAM kernels run; with bypass copy from
  Step 1.3 also wired for up_proj's cb_out, anchor test PCC ≥ 0.9915.
- Risk: low (pattern repeats Phase 1).

---

### Phase 3 — Gather A + Gather B (SRAM gate/up → sender)

**Step 3.1 — Gate gather A**
- Files: `fused_ops/moe/op.py`, `fused_ops/moe/moe_kernel.cpp`
- Change: new `MoeGather::Op` call, 64 SRAM gate cores → sender, into new
  `sram_group1_cb` on sender. Per-core sender_idx values match
  `setup_kn_per_core_values` layout.
- Test: DPRINT sender's `sram_group1_cb` after gather; assert tile values
  match expected K-partials × N-tiles × num_active_experts layout.
- Risk: medium. New gather wiring; sender_idx computation must match
  layout.

**Step 3.2 — Up gather B**
- Mirror Step 3.1 for up; output to `sram_group2_cb`.
- Test: same DPRINT validation.

---

### Phase 4 — Extended GatedReduce

**Step 4.1 — Add `num_experts` outer loop to `gated_reduce.hpp`**
- Files: `unified_kernels/gated_reduce.hpp`,
  `fused_ops/moe/op.py:setup_gated_reduce`
- Change: outer loop `for e in 0..num_experts`. Inside: `is_sram_expert`
  skip filter (drain CBs and continue), then existing K-reduce body, then
  per-expert `mul_tiles` against `mul_cb_scalar_src[e]`. Output stride
  matches DRAM gather's `expert_dst_stride` (= 8 face tiles per expert,
  512 bytes).
- Test (standalone): unit test with known K-partials, expert_scale, and
  bit-15 mask → check output bit-exact against torch reference.
- Risk: medium-high. Touches the kernel that handles the actual reduce
  math; bit-exactness against torch is essential.

**Step 4.2 — Wire extended GatedReduce in moe_kernel.cpp**
- Files: `fused_ops/moe/moe_kernel.cpp`
- Change: replace shared-only `GatedReduce::Op` call with
  routed+SRAM-aware variant on sender. Output to `sram_mcast_src_cb`.
- Test: with bypass from Steps 1.3/2.2 still in place but extended
  GatedReduce now producing the post-reduce-and-scale tiles, manually
  copy `sram_mcast_src_cb` content into the right slots of DRAM
  `down_proj_gather_dst_cb` so down_proj sees correct activations.
  Anchor test PCC ≥ 0.9915 at T=1.
- Risk: medium. Connects gather to reduce; first time SRAM path produces
  end-to-end-correct intermediates.

---

### Phase 5 — Two mcasts to merged in0

**Step 5.1 — Sender BRISC computes `n_sram` / `n_dram`**
- Files: `fused_ops/moe/moe_kernel.cpp` (sender BRISC)
- Change: 8-entry walk of reordered index, count bit-15 entries, write
  `n_sram` + `n_dram` to L1 scratch words consumed by the runtime Mcast.
- Test: DPRINT counts; assert sum equals `num_active_experts`.

**Step 5.2 — Two-mcast block + DRAM down kernel counter**
- Files: `fused_ops/moe/moe_kernel.cpp`,
  `unified_kernels/matmul_expert_compressed_dram.hpp`,
  `fused_ops/moe/op.py` (RT arg plumbing)
- Change: SRAM mcast (length=`n_sram × 8 tiles`, dst_off=0), then DRAM
  mcast (length=`n_dram × 8 tiles`, dst_off=`n_sram × 8 tiles`). Both
  target merged `down_in0_cb` on 8 DRAM cores AND 112 SRAM down cores
  (when wired in Phase 6).
- Kernel edit: DRAM down accum-experts branch uses `dram_count` counter
  with `n_sram` runtime offset for cb_in0 read (replaces `exp_i × num_tiles_k`
  offset). `n_sram` plumbed as common-RT-arg.
- Test: at T=1 hot expert, DRAM down reads in0 starting at offset
  `n_sram × num_tiles_k`, increments per processed DRAM expert.
- Test: anchor test PCC ≥ 0.9915 at T=1 (now without bypass).
- Risk: high. First step where the merged-in0 layout is exercised end-to-
  end with real DRAM down. Most likely place to hit alignment / sizing
  mismatches.

**Step 5.3 — Remove bypass copies from Steps 1.3/2.2**
- Files: `fused_ops/moe/moe_kernel.cpp`
- Test: anchor test PCC still ≥ 0.9915 at T=1.

---

### Phase 6 — SRAM down_proj matmul

**Step 6.1 — `setup_matmul_expert_sram` for down**
- Files: `fused_ops/moe/op.py`
- Change: T slabs of `K × N_per_core` bfp4_b on each of 112 cores.
- Test: allocation succeeds; per-core L1 budget within cap.

**Step 6.2 — Wire SRAM down_proj kernel call + SRAM kernel counter**
- Files: `fused_ops/moe/moe_kernel.cpp`,
  `unified_kernels/matmul_expert_compressed_sram.hpp`
- Change: add `MatmulExpertCompressedSRAM::Op` with `accum_experts=true`
  on `is_shared_mcast_receiver_core`, between shared down matmul (`:1449`)
  and existing residual_add (`:1471`). Output to new `sram_down_out_cb`.
- Kernel edit: SRAM accum branch uses `sram_count` counter (starts at 0)
  for cb_in0 read offset, replaces `exp_i × num_tiles_k`. ~5 lines.
- Test (T=1): anchor test PCC SHOULD drop (SRAM down output not yet
  combined with shared down). Validate `sram_down_out_cb` tiles against
  torch reference for that expert.
- Risk: medium. First accum-experts SRAM call + first kernel-counter edit.

---

### Phase 7 — Post-down merge

**Step 7.1 — Eltwise_add: shared_down + sram_down**
- Files: `fused_ops/moe/op.py` (setup), `fused_ops/moe/moe_kernel.cpp`
- Change: new `EltwiseAdd::Op` on 112 cores, in0=`shared_down_out_cb`,
  in1=`sram_down_out_cb`, out=new `combined_down_out_cb`. Re-route
  `residual_add`'s in0 from `shared_down_out_cb` to `combined_down_out_cb`.
- Test: anchor test PCC ≥ 0.9915 at T=1.
- Risk: medium. The first time SRAM path is fully end-to-end without
  bypass.

---

### Phase 8 — PCC scaling

**Step 8.1 — T = 2, 4, 8 sweep**
- Test: anchor test PCC ≥ 0.9915 at T ∈ {2, 4, 8}.
- Validate: hot expert L1 budget; index-reorder edge cases (all-SRAM,
  all-DRAM, mixed).

**Step 8.2 — Cleanup**
- Remove debug DPRINTs, throwaway bypass code, dead branches.
- Document final L1 budget ceiling per proj.

---

### Validation matrix (anchor test PCC at each step)

| Step                    | num_sram | Bypass on | Expected PCC      |
|-------------------------|----------|-----------|-------------------|
| 0.1 – 0.4 (Phase 0)     | 0        | n/a       | ≥ 0.9915          |
| 1.1 – 1.2               | 0        | off       | ≥ 0.9915          |
| 1.2                     | 1        | off       | drop expected     |
| 1.3                     | 1        | on        | ≥ 0.9915          |
| 2.1 – 2.2               | 1        | on        | ≥ 0.9915          |
| 3.1 – 3.2               | 1        | on        | ≥ 0.9915          |
| 4.1 – 4.2               | 1        | partial   | ≥ 0.9915          |
| 5.2                     | 1        | off       | ≥ 0.9915          |
| 6.2                     | 1        | n/a       | drop expected     |
| 7.1                     | 1        | n/a       | ≥ 0.9915          |
| 8.1                     | 2,4,8    | n/a       | ≥ 0.9915          |
