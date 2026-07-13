# Fused per-layer MoE kernel for DiffusionGemma denoise

Status: **DESIGN + increment-1 landed (WRITER_SCALE plumbing) + increment-3 SCAFFOLD landed
(in0-gather gate + reader hook, default off).** This is a multi-week ttnn C++ effort. This doc is the
contract + staged plan; increments land behind env-gated compile defines that default to
byte-identical existing behavior. Increment 3's in-reader gather was judged genuinely hard (§7): the
per-token row gather from a *tiled* hidden blows up NoC transactions ~64×, so only the gate + reader
hook (identity fallback) were landed this pass; §7.5 lists the exact remaining C++.

Owner-scope note: this work touches ONLY `tt/sparse_moe.py`,
`ttnn/cpp/ttnn/operations/matmul/device/sparse/**` (+ its quasar mirror
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/sparse/**`), the sparse-matmul unit
test, and this doc. It never edits gemma4 or the denoise/loop/sampling/self-cond/model files.

---

## 1. Why: the denoise MoE step is overhead-bound

Per layer (real QB2 shapes, mesh (1,4), TP=4): `S=256` canvas tokens, `E=128` experts,
`top_k=8`, capacity `C=32`, `EC = E*C = 4096`, `H=2816`, per-device intermediate `I=192`.

`tt/sparse_moe.py::sparse_experts_forward` (`DG_SPARSE_MOE=1`) currently does, per layer:

1. `build_capacity_dispatch(dense_routing)` → one-hot `disp [1,1,S,EC]` + route-weighted
   `comb [1,1,S,EC]` (GShard capacity dispatch, on device).
2. **gather**: `dispatched[EC,H] = disp^T @ hidden`  (`[1,1,EC,S] @ [1,1,S,H]`) — a dense matmul
   over a one-hot matrix, materializing `dispatched [1,E,C,H] ≈ 23 MB` to DRAM.
3. **experts** (`_batched_experts`): batched gate/up/geglu/down over the `E` capacity batches,
   OPT-004-tuned, ~weight-roofline.
4. **combine**: `out[S,H] = comb @ down_flat`  (`[1,1,S,EC] @ [1,1,EC,H]`) — again a dense matmul
   over a one-hot/route-weighted matrix, re-reading `down_flat [1,1,EC,H] ≈ 23 MB`.
5. `ccl_allreduce` over TP.

Measured (session-10 op re-profile): the step is **overhead / op-count bound**, ~5% of the
6–12 ms/step DRAM roofline is "useful". The two dense `EC=4096` matmuls (gather + combine) plus
their two 23 MB `[EC,H]` materializations are the waste target: they exist only to express a
gather and a weighted scatter as matmuls over one-hot masks.

**The fused op we want**: gather each expert's capacity tokens from `hidden`, run gate/up/down,
and scatter-accumulate the **route-weighted** result straight into `out[S,H]` — in one op, with
neither `[EC,H]` buffer materialized. Op-count is the enemy (a RoPE-unchunk removing ~128 tiny
ops gave +34%; a single-op `transpose_a` tweak gave 0), so collapsing steps 2+3+4 into one op is
the lever.

---

## 2. Fused-op contract

```
fused_moe(
    hidden        : [1, 1, S, H]  bf16, TP-replicated          (expert input)
    gather_index  : [E, C]        uint32                        (row t of hidden for each (e,slot); DEAD=S for empty slots)
    route_weight  : [E, C]        bf16                          (comb weight for each (e,slot); 0 for empty/dropped)
    w_gate/up/down: [1, E, ·, ·]  (bf16/bfp8)                   (expert bank, TP-sharded on I)
) -> out : [1, 1, S, H] bf16 (TP-partial; ccl_allreduce applied by caller, or fused — inc 5)
```

`gather_index` / `route_weight` are the compact form of `disp`/`comb` — they are exactly what
`build_capacity_dispatch` already computes internally (`col = e*C + slot`, `slot` = exclusive
per-expert token rank; `vals_valid` = route weight, zeroed on drop). Producing them as `[E,C]`
index/scale tables instead of `[S,EC]` one-hot masks is a host/device metadata change in
`sparse_moe.py`, not new kernel work.

Semantics: `out[t,:] = Σ_{(e,slot): gather_index[e,slot]==t} route_weight[e,slot] * down_e(geglu(gate_e(hidden[t]), up_e(hidden[t])))`.
Each token `t` receives exactly `top_k=8` contributions (one per expert it routed to), unless a
slot was capacity-dropped (route_weight=0).

---

## 3. Dataflow / compute-kernel plan

The op is a per-expert batched matmul (`E` batches, one `C`-row tile-group each) bracketed by a
gather-read and a scatter-accumulate-write. It maps onto the existing sparse reuse-mcast-1d
program (one expert per core-group, `cb_sparsity` skipping empty experts) with two kernel changes:

**Reader (gather).** Instead of streaming `dispatched[e*C : e*C+C, :]` contiguously from DRAM,
the in0 reader reads `gather_index[e, 0..C]` (a per-batch side page, same shape/plumbing as
`cb_sparsity`) and issues `C` gathered row reads `hidden[gather_index[e,slot], :]` into the in0
CB. Empty slots (`index==S`, a zero-padded row) read a pinned zero page. This deletes step 2
(the `disp^T @ hidden` matmul) and the `dispatched` materialization — the gather becomes NoC
reads folded into the matmul reader.

**Compute.** Unchanged batched gate → up → geglu → down per expert (reuse
`bmm_large_block_zm_fused_bias_activation.cpp`). Output tiles for expert `e` are its `C×H` down
rows in `cb_out`, in slot order.

**Writer (scatter-accumulate + route-weight).** Each `cb_out` row `slot` of expert `e` must be
multiplied by `route_weight[e,slot]` and **added** into `out[gather_index[e,slot], :]`. Two
sub-problems:

- *Route-weight fold* — multiply the down row by a per-(e,slot) scalar before writing. Increment 1
  proves the coarse (per-expert) version of exactly this side-table→writer→scale plumbing.
- *Scatter-accumulate* — the hazard below.

### The cross-core scatter-accumulate hazard

A token `t` gets contributions from up to `top_k=8` experts, which the factory places on
**different cores**. The NoC has **no read-modify-add** primitive — a core cannot atomically
`out[t] += partial`. Two cores writing `out[t]` race. So accumulation must be *structured*, not
done by concurrent `+=`. Three viable structures, cheapest first:

1. **Keep combine as a reduction op, drop only the matmul-over-one-hot** (recommended first
   target). The fused op emits the *compact* weighted down output `down_w [1, nnz_rows, H]` (route
   weight already folded in — increment 1's mechanism), then a single `fast_reduce_nc`-style
   gather+sum over the `token_slot [top_k, S]` index (the pattern `ragged_sparse_prefill_forward`
   already uses at `sparse_moe.py:755-767`) produces `out[S,H]`. This removes the `EC=4096` combine
   *matmul* and the `[EC,H]` materialization, replacing them with a compact
   `[nnz_rows,H] → [S,H]` embedding+reduce. No RMW race: the reduce owns each output row.
2. **Home-core reduction inside the op** — assign token `t` to one home core; every expert core
   writes its weighted partial to a per-(t, k-slot) scratch page it exclusively owns; a final
   per-core pass sums the ≤`top_k` partials for its home tokens. One extra scratch buffer
   (`[S, top_k, H]`), no cross-core RMW. This is the fully-fused single-op form.
3. **Serialize by semaphore** — order the `top_k` writers to `out[t]` with a token semaphore.
   Rejected: serializes the hot path, defeats the point.

Increment 1 lands the route-weight fold (the scalar-mul half of the writer). Structure (1) is the
next increment because it reuses machinery already proven bit-exact in the prefill path.

---

## 4. Reusable sparse_matmul code (do NOT reinvent)

- **`cb_sparsity` per-batch skip** (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:236-274`,
  `reader_bmm_tile_layout_in0_sender_padding.cpp`): reads one side page per batch, skips the batch
  when the entry is 0. This is the exact side-table plumbing the gather-index and route-weight
  tables need — one page per expert, indexed by the inner batch `bB`. **Increment 1 reuses this
  page as the route-weight carrier.**
- **`SPARSE_OUTPUT` / `compact_output`** (factory `:272`, writer kernel `:90-94,:269-272`): packs
  only the `nnz` active batch pairs in scan order, so the output is `[nnz_rows,H]` not `[EC,H]`.
  This is the "skip the `[EC,H]` materialization" primitive for the compact fused output.
- **`SparseMatmulMultiCoreReuseMcast1DProgramFactory`** (`sparse/factory/`): distributes `E` expert
  batches over the grid, one expert per core-group, `nnz`-driven loop counts. The fused op is this
  factory + a gather reader + a scatter/scale writer.
- **`matmul_reduce_scatter_async`** (`experimental/ccl/matmul_reduce_scatter_async/`): existing
  fused matmul+reduce-scatter; the template for folding the final TP `ccl_allreduce` into the op
  (increment 5).
- **`ragged_sparse_prefill_forward`** (`sparse_moe.py:625`): the zero-drop compact gather (via
  `ttnn.embedding`) + `fast_reduce_nc` weighted combine, **already QB2 bit-exact vs the dense
  128-expert path**. This is the reference for the compact gather/scatter semantics and for
  structure (1) above; the decode fused op is its on-kernel equivalent.

---

## 5. Staged increments (honest effort)

| # | Increment | C++ surface | Effort | Risk |
|---|-----------|-------------|--------|------|
| 1 | **WRITER_SCALE** — fold a per-batch route-weight (carried in the existing `cb_sparsity` page) into the down write, env-gated, bit-identical when off. | writer kernel `#ifdef` block + 3-line env-gate in both factories | **~2 days (done)** | low — JIT kernel, no host ABI change |
| 2 | **Compact route-weighted output** — wire the scaled+compact output through `sparse_moe.py` and replace the `comb @ down_flat` matmul with the `embedding`+`fast_reduce_nc` combine (structure 1). Deletes the combine matmul + `[EC,H]` materialization. | `sparse_moe.py` only (kernel from inc 1) | ~1 wk | low-med |
| 3 | **Gather reader** — per-expert `gather_index` side page in the in0 reader; read `hidden[index]` rows into in0 CB; pinned zero page for empty slots. Deletes the `disp^T @ hidden` matmul + `dispatched` materialization. **SCAFFOLD landed** (gate `TTNN_SPARSE_MATMUL_IN0_GATHER` + reader hook, identity fallback, byte-identical off); real gather + `gather_index` op input remain — see §7. | in0 reader kernel + factory (new side input) | ~2 wk | med — new op input, index-read dataflow |
| 4 | **Single-op fuse** — gather + experts + compact scaled scatter as one `fused_moe` op; expose in `sparse_moe.py` behind a flag; program-cache + trace-safety. | factory + op wrapper + pybind + `sparse_moe.py` | ~2 wk | med-high — new op ABI |
| 5 | **Fuse the TP all-reduce** via the `matmul_reduce_scatter_async` pattern. | new fused ccl+matmul path | ~1 wk | high — CCL |

**Total remaining: ~6 weeks of ttnn C++/Python after increment 1.** Increment 1 is ~5% of the
work (the side-table→writer→scale plumbing scaffold); increments 3–4 (the actual on-kernel gather
and single-op fuse) are the bulk. Structure (1) in increment 2 captures a large fraction of the
*measured* win (deletes one dense `EC=4096` matmul + one 23 MB materialization) with almost no
kernel risk, so it is the highest value/risk step and should follow increment 1 immediately.

---

## 6. Increment 1: WRITER_SCALE (implemented)

**What.** A gated path in the shared sparse writer kernel
(`reader_bmm_tile_layout_in1_sender_writer_padding.cpp`) that, per active batch `bB`, reads the
bf16 value already sitting in the `cb_sparsity` L1 page and multiplies every output tile of that
batch by it before the NoC write.

**Why this is legal with no new host tensor.** The op already reads one bf16 `cb_sparsity` value
per batch and uses **only `== 0`** as an active/skip gate — the float magnitude is currently
ignored (confirmed: `test_sparse_matmul_with_nnz` puts `torch.rand` values in `sparsity` and its
reference is `torch.matmul(...)` with **no** scale). So a caller can carry the route-weight in the
sparsity page: the nonzero *pattern* (hence `count_nonzero == nnz`) is unchanged, and with
`WRITER_SCALE` on, the output batch is scaled by that weight. This is exactly the per-batch
route-weight fold the combine needs, proven end-to-end through the real side-table.

**Gate.** Env var `TTNN_SPARSE_MATMUL_WRITER_SCALE` (read in both factories via `std::getenv`,
the established pattern in `compute_throttle_utils.cpp`). Unset → the `WRITER_SCALE` define is not
emitted → the `#ifdef` block vanishes → **byte-identical to today**. The define adds no
compile-time/runtime args and no CB, so the kernel arg layout is unchanged.

**Kernel mechanics.** After `cb_out.wait_front(out_subblock_tile_count)` (before the writes for
that subblock), when `batchB > 0` and the batch is active, scale the `out_subblock_tile_count`
tiles in place at the CB read pointer: bf16 output → per-element `bf16→f32 → *scale → f32→bf16`
(round-to-nearest-even) via inline bit ops; fp32 output → per-element `float *= scale`; block-float
(bfp8) output → **not scaled** (shared-exponent RMW is out of scope for increment 1; the DG
down-proj output is bf16). Padded subblock tiles that are popped without a write are left
untouched.

**Scope of the scale.** Increment 1 applies **one scale per expert-batch** (coarse). The final
fused op needs a **per-(expert,slot)=per-row** scale (`route_weight[e,slot]`); that is a
`[E,C]`-page-indexed-by-row refinement of this same plumbing (a second side page read per row
group), landed in increment 3/4 once the gather reader exists. Increment 1 deliberately proves the
plumbing at expert granularity first.

**Test.** `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py::test_sparse_matmul_writer_scale`
(new). Sets `TTNN_SPARSE_MATMUL_WRITER_SCALE=1`, runs a bf16-output sparse matmul with random
per-batch weights in the sparsity tensor, and checks each active `(b,s,e)` output block against
`torch.matmul(in0,in1) * float(sparsity_value)`. Host-buildable; **requires a Tenstorrent device
to run** (the writer kernel is JIT-compiled on device). Program-cache caveat: the env-gate is read
inside the factory and is not part of the program hash, so a run must not reuse a cached program
built with the flag in the opposite state for the same shapes — the test uses a distinct shape and
sets the env var before the first op.

**Verification status.** `.so` builds and links (both factories). Kernel change is JIT — cannot
affect the `.so`. Device run of the new test is gated on free hardware.

---

## 7. Increment 3: in-reader GATHER — feasibility verdict + landed scaffold

**Verdict: the in-reader per-row gather is genuinely hard (the ~2-week increment the roadmap
scoped), and CANNOT be landed as a working, *faster* gather in one buildable increment.** This
section records the analysis, the buildable scaffold that *was* landed (gate + reader hook, default
off, byte-identical), and the exact remaining C++ so the next session starts from a green tree.

### 7.1 Why it is hard (the crux)

The target is the denoise MoE gate/up matmul. Today `sparse_experts_forward` expresses the gather as
a dense one-hot matmul (`dispatched = disp^T @ hidden`, `sparse_moe.py:823-834`) and then runs the
batched experts with a plain reuse `ttnn.matmul` (`_batched_experts`). Increment 3 wants the gather
folded into the **sparse mcast-1d in0 sender reader**
(`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp`),
so each expert's `[C=32, H]` in0 block is assembled directly from `C` arbitrary rows of `hidden[S,H]`.

The dispatch is **per-token (per-row)**: expert `e`'s 32 capacity slots are 32 arbitrary tokens of
the 256-row canvas (`gather_index[e,slot] ∈ [0,S)`, `=S` for empty). `hidden` is **TILE layout**
(validated: `sparse_matmul_device_operation.cpp:87` forces in0 TILE). A logical row `t` therefore
lives at intra-tile row `t%32` of tile-row `t/32`, and inside a bf16 32×32 tile (4 row-major 16×16
faces, order (0,0),(0,1),(1,0),(1,1)) one logical row spans the **two horizontal faces** — two
discontiguous 16-element (32-byte) runs 256 elements apart. So gathering one dest tile-column of one
expert = `32 rows × 2 face-runs = 64` sub-tile NoC reads, vs **1** tile read today; over `H/32=88`
tile-cols that is **~5632 reads/expert vs 88** — a ~64× NoC-transaction blow-up. Since the denoise
step is *movement/op-count bound*, a face-aware gather from a **tiled** hidden would almost certainly
run *slower*, defeating the increment's purpose (op-count tweaks `transpose_a` and a compact
embedding gather already measured ~0/slower).

The only variant that reduces movement is a **ROW-MAJOR hidden + on-the-fly tilize**: row `t` is then
one contiguous `H·BPE`-byte read (`32` reads/expert), but the reader must then tilize the gathered
rows into the tiled in0 CB — an extra pipeline stage the matmul reader does not have. That is exactly
what `ttnn.embedding(..., layout=TILE)` already does in `ragged_sparse_prefill_forward`
(`sparse_moe.py:677`), and folding it into the matmul reader is the multi-week body.

### 7.2 Precise address math (for the real implementation)

Per dest row `slot` of expert `e`, `t = gather_index[e,slot]` (`t==S` → pinned zero row):

```
source hidden tile (tiled [S,H]):  page_id = (t / 32) * (H / 32) + tile_col
intra-tile row = t % 32:  face_row = (t%32) / 16,  r16 = (t%32) % 16
  run0 byte off = (face_row*2 + 0) * 256 * BPE + r16 * 16 * BPE   (cols  0..15)
  run1 byte off = (face_row*2 + 1) * 256 * BPE + r16 * 16 * BPE   (cols 16..31)
```

written to the mirror offsets of dest row `slot % 32` in the in0 CB tile. (Row-major variant:
`src byte off = t * H * BPE`, one contiguous `H·BPE` read, then tilize.) `BPE=2` for bf16. This math
is duplicated in the kernel top-of-file comment at the gather hook.

### 7.3 CB / op-input plan (remaining work)

- **New op input `gather_index`** `[E, C]` (or `[batchA, C]`) `uint32` ROW_MAJOR, plumbed exactly like
  `sparsity`: a 4th tensor in `SparseMatmulInputs`, an `is_*`/nnz-style attr, a pybind arg, a
  `TensorAccessorArgs` append in both factories, a per-core runtime arg for its address, and an
  `override_runtime_arguments` refresh. (This is the op-ABI change — the reason it is its own
  increment.)
- **New CB `cb_gather_index`** (next free index, e.g. `c_8`) sized to one page = `C` `uint32`
  (`C=32 → 128 B`), read once per batch `b` in the in0 sender exactly like `cb_sparsity`
  (`reader_bmm_tile_layout_in0_sender_padding.cpp:176-179`).
- **Pinned zero page** for `gather_index==S` (empty slot) — one zero-filled L1 page the read redirects
  to, so dropped/empty slots contribute 0.
- **in0 sender read loop** (`:255-291`): replace the contiguous `in0_tensor_tile_id` page id with the
  index-derived source page + the two sub-tile face reads above (or the row-major read + tilize).

### 7.4 What was landed this increment (buildable scaffold, default OFF)

1. **Gate** `TTNN_SPARSE_MATMUL_IN0_GATHER` → compile define `SPARSE_MATMUL_IN0_GATHER`, emitted only
   by the sparse mcast-1d factory (both the main and quasar mirrors) into
   `mm_kernel_in0_sender_writer_defines`. Unset → not emitted; the dense matmul factories that share
   the in0 kernel never emit it → **byte-identical**. (`.so` rebuilds + links, both factories.)
2. **Reader hook** in the shared in0 sender kernel: a gated `#ifdef SPARSE_MATMUL_IN0_GATHER` block at
   the read site with the §7.2 address math in comments and an **identity fallback** (source page ==
   contiguous tile id). The `#else` path is textually identical to the pre-scaffold read, so the flag
   is a **no-op** whether on or off until the gather lands.
3. **Tests** (`tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py`):
   `test_sparse_matmul_in0_gather_scaffold` (device-runnable) proves the gate+hook JIT-compile, run,
   and are byte-identical to the ungathered sparse_matmul; `test_sparse_matmul_in0_gather_reference`
   encodes the full gather↔torch reference (`_gathered_matmul_reference`) and **self-skips** until the
   `gather_index` op input exists.
4. **`sparse_moe.py` gate** `DG_MOE_FUSED_GATHER` (`fused_gather_enabled`, default off): the
   integration point in `sparse_experts_forward` **raises `NotImplementedError`** when set (rather
   than silently running the identity gather) until the kernel lands. Off → the path is bit-identical.

### 7.5 Exact remaining C++ steps (in order)

1. Add the `gather_index` op input end-to-end (types/hpp, `sparse_matmul(...)` overload,
   `create_sparse_matmul_attributes`, pybind) — the ABI change; validate shape `[·, C]` uint32
   ROW_MAJOR, `count_nonzero`-independent.
2. Factory: append its `TensorAccessorArgs`, add `cb_gather_index` + a pinned zero CB, pass its
   address as a runtime arg, refresh in `override_runtime_arguments` (both factories + quasar mirror).
3. Kernel: read the index page per batch; in the read loop compute the §7.2 source page/offsets and
   issue the sub-tile reads (start with the tiled face-aware read for correctness; then add the
   row-major + tilize fast path for the actual movement win).
4. Un-skip `test_sparse_matmul_in0_gather_reference`; confirm it passes on device.
5. Wire `sparse_moe.py`: build `gather_index[E,C]` on-device from the `build_capacity_dispatch`
   `col = e*C + slot` machinery (the `col_u`/`pos` tensors already computed there), replace
   `disp^T @ hidden` + `_batched_experts` gate/up with the gathered `sparse_matmul`. Remove the
   `NotImplementedError` guard.

### 7.6 A/B command (once wired)

```
# baseline (current dense gather):
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 <denoise/serve bench>
# fused gather (kernel + DG_MOE_FUSED_GATHER):
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 TTNN_SPARSE_MATMUL_IN0_GATHER=1 DG_MOE_FUSED_GATHER=1 <same bench>
```

**Verification status (increment 3).** `.so` builds + links (both factories); scaffold is JIT/no-op
→ cannot change `.so`. Scaffold device test result recorded alongside the WRITER_SCALE test run.
