# Fused per-layer MoE kernel for DiffusionGemma denoise

Status: **DESIGN + increment-1 landed (WRITER_SCALE plumbing).** This is a multi-week ttnn C++
effort. This doc is the contract + staged plan; increment 1 is the smallest buildable, testable,
tree-green step and is implemented behind an env-gated compile define.

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
| 3 | **Gather reader** — per-expert `gather_index` side page in the in0 reader; read `hidden[index]` rows into in0 CB; pinned zero page for empty slots. Deletes the `disp^T @ hidden` matmul + `dispatched` materialization. | in0 reader kernel + factory (new side input) | ~2 wk | med — new op input, index-read dataflow |
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
