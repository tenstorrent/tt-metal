# First-world kernel args — cross-op outlier survey (deviations & recipe gaps)

**Author:** Claude (Metal 2.0 recipe development).

**Date:** 2026-08-20.

**Purpose:** Stress-test the [named-kernel-args porting recipe](../ai/post_port/style/named_kernel_args.md)
against a **diverse sample of ~20 TTNN ops across domains**, to surface argument / flag / binding
patterns and special situations the recipe does not yet anticipate. Compiled from parallel,
read-only subagent explorations (no builds, no tests, no device).

> **Method / decaying snapshot.** Findings come from *static reading* of program factories + kernels
> (legacy or Metal 2.0), mapping legacy arg patterns to the Metal 2.0 CTA/RTA/CRTA concepts
> (`get_compile_time_arg_val` → CTA, `get_arg_val` → RTA, `get_common_arg_val` → CRTA; conditional
> `SetRuntimeArgs` / CB creation → conditional args / bindings). They are not device-validated, and
> line numbers drift — re-verify a specific op before acting on it. This file *tracks* candidate gaps;
> a change to the recipe follows only after review.
>
> **Exception — permute is device-validated.** Its tiled interleaved factory was actually ported +
> tested on 2026-08-20 (then rolled back): varargs-partial and padding-writer hard-stop confirmed on
> hardware, `test_permute.py` **1605 → 1605 passed**. Those two entries below are grounded; everything
> else is still a static read. Findings folded into the recipe's
> [permute field notes](../ai/post_port/style/named_kernel_args.md#field-notes-permute-2026-08-20).

---

## What the recipe already handles (the baseline these findings are measured against)

- Kernel → `template <uint32_t CTA…> TT_KERNEL void entry(uint32_t RTA/CRTA…)`; a no-CTA kernel is a
  bare, non-template `TT_KERNEL void entry(…)`.
- **All named args are `uint32_t`**, and the kernel's parameter names must **exactly equal** the
  host-registered arg names (CTAs ↔ template list; RTAs/CRTAs ↔ function list).
- **Hard stop** (kernel not convertible): the factory registers **any** CTA/RTA/CRTA *conditionally*
  (arg set varies by build/config), or a named arg is **non-`uint32_t`**.
- **Partial**: a kernel reaching args through **varargs** (`get_vararg` / `get_compile_time_vararg`) —
  named args convert, vararg reads stay manual.
- **Flags**: a *pure-value* flag (gates only always-present code) → promote to a `uint32_t` NTTP +
  `if constexpr`; a flag gating a **conditional resource** (a `dfb::`/`tensor::`/`sem::` token or arg
  provided only on some path) → **leave the `#ifdef`**.
- **Conditional DFB/tensor/semaphore bindings** → the `#ifdef` stays. **Shared kernels** → fork.

An "outlier" below is anything outside this model, or a variation its rules / stops / pitfalls do not
mention.

---

## Scope filter — Metal-2.0 kernels only (the view that actually matters)

**This recipe applies only to kernels already ported to Metal 2.0** — kernels that read
`get_arg(args::…)` and bind resources via `dfb::` / `tensor::` / `sem::`. Pre-Metal-2.0 kernels
(legacy positional `get_arg_val` / `get_compile_time_arg_val`, *and* the transitional
`get_named_compile_time_arg_val` hybrid) are **out of scope** — they need the base Metal 2.0
kernel-arg port first. The initial survey below deliberately swept broadly; **filtering to the
in-scope set changes the conclusion sharply.**

**In-scope ops in this survey** (kernels on `get_arg(args::…)`, not yet `TT_KERNEL`): `permute`,
`fold`, `rmsnorm_distributed` (the `*_metal2` forks under `layernorm_distributed`), and `topk`'s
single-core factory — plus `softmax` (the recipe's basis) and `kda` (already full `TT_KERNEL`; a
reference, not a target). **Everything else surveyed is pre-Metal-2.0 at the kernel level** (Metal
2.0 *host* + positional/hybrid *kernels*) and is out of scope until ported.

### What remains once filtered — the recipe already handles all of it

- **Conditional args / RTA schemas → Rule 5 hard stop.** permute padding writers
  (`#ifdef NEEDS_PADDING` + host `push_back` behind `if`, `permute_tiled_program_factory.cpp:415-423`);
  topk `GENERATE_INDICES` (`reader_create_index_local_topk.cpp:40-46`). Exactly the softmax-reader
  pattern, now confirmed across ops — and, for permute's two tiled writers, **device-validated
  2026-08-20** (left legacy, sentinel stayed green).
- **Varargs → partial.** permute's rank-scaled `advanced_options.num_runtime_varargs` + `get_vararg`
  (`permute_tiled_program_factory.cpp:154,851`) — the concrete in-scope varargs example softmax lacked.
  **Device-validated 2026-08-20:** the tiled readers' named CTAs/RTAs converted to a `TT_KERNEL`
  signature while the `get_vararg` reads stayed manual — builds and passes; `check_name_sets` ignores
  varargs (a count, never a name), so the "partial" split is real, not just plausible.
- **Conditional DFB + `#ifdef` → Rule 4 (leave it).** fold (`FOLD_RM_NOT_L1_ALIGNED` → `dfb::in1`),
  rmsnorm (`#ifdef FUSE_GAMMA` → `dfb::x_normed`/`out`, natural fallback) — textbook.
- **Per-flag decision on "entangled" flags.** permute `needs_x_padding` (a CTA, convertible) vs
  `NEEDS_Y_PADDING` (gates `dfb::cb_pad`, stays `#ifdef`) — the recipe's *per-flag* rule resolves it.
- **Same source, multiple bindings, *same* arg-name set → convertible.** fold's `is_reader`
  dual-instance and cliff/full compute specs — one signature, multiple runtime bindings.
- **Cross-op shared donor kernel → fork.** permute reuses a transpose reader.

**No new recipe gap survives the filter.** The in-scope ops are all handled by the existing
rules / stops.

### Why the scary themes evaporate — most were pre-Metal-2.0 artifacts the port itself removes

- **Unnamed `TensorAccessorArgs` CTA blocks (Theme B)** → the Metal 2.0 port *replaces* these with
  named `tensor::` bindings (`TensorAccessor(tensor::name)`); in-scope ops (permute, fold, softmax)
  already use `tensor::` and carry no accessor-CTA block.
- **`Buffer*` RTAs (Theme G)** → the port replaces them with `tensor::` bindings (base address via the
  accessor). Gone once ported.
- **Semaphores / fabric coords as plain `uint32_t` (Theme F)** → the port replaces legacy semaphores
  with `sem::` bindings; multi-device/fabric ops are out of scope until ported anyway.
- **Hybrid / positional prerequisite state (Theme C)** → this *is* the out-of-scope population.

### The only in-scope-worthy additions (small)

1. A concrete Metal-2.0 **varargs** example (permute: `advanced_options.num_runtime_varargs` +
   `get_vararg`) — the recipe mentions varargs but softmax had none. **Done** — permute ported +
   tested 2026-08-20, folded into the recipe's field notes (with the dead-CTA and silent-JIT-log
   nuances the run also surfaced).
2. One line that a source **bound multiple times with the same arg-name set is convertible** (fold) —
   distinct from the cross-op fork and from the different-schema hard stop.
3. One line that a scalar arg may carry a **non-integer bit pattern** (`kda`'s `epsilon_bits`, norm
   `eps`) — legal `uint32_t`, unchanged spelling.

**Deferred (only matter once those ops are ported):** injected-define op chains (Theme D) and
multi-device semaphore/fabric arg passing (Themes A2/F) appear when eltwise/matmul/CCL reach Metal
2.0 — re-survey then; they may reshape post-port.

---

## Ops surveyed

> The table and themes below include **pre-Metal-2.0 ops** (swept for completeness). For the actual
> recipe scope, read the [Scope filter](#scope-filter--metal-20-kernels-only-the-view-that-actually-matters)
> above — only `permute`, `fold`, `rmsnorm_distributed`, `topk` (single-core), `softmax`, and `kda`
> are in scope.

*(populated on subagent completion)*

| Op | Domain | Port status | Headline outlier(s) |
|---|---|---|---|
| `concat` | data_movement | Mixed — `ProgramDescriptor` host, legacy kernels | Variable-count positional RTAs (`3 + 3×N` per tensor); unnamed `TensorAccessorArgs` / `make_tensor_accessor_args_tuple<N>` CTAs; one `.cpp` bound as reader+writer with different CTAs; shared writers from other ops |
| `permute` | data_movement | Metal 2.0 (still `kernel_main`) | **Conditional RTA schema** — host `push_back` padding RTAs behind `if`, kernel `#ifdef NEEDS_PADDING` (hard stop, Rule 5); rank-scaled **varargs** (`num_runtime_varargs = 2–3×rank`, partial); conditional *compute-kernel* binding (whole kernel absent off-path) |
| `pad` | data_movement | Mixed — `ProgramDescriptor` host, legacy kernels | Variable-length RTA blob addressed by pointer arithmetic (`num_cores_read`-driven); zero-RTA active cores; positional index holes; unnamed `TensorAccessorArgs` CTAs; 8+ factories |
| `fold` | data_movement | Metal 2.0 (still `kernel_main`) | Conditional DFB + matching `#ifdef` (clean Rule 4 case ✓); dual-instance same source, role split via an `is_reader` CTA; duplicate compute `KernelSpec` (cliff vs full); mutable RTA used as loop state |
| `binary_ng` | eltwise | Metal 2.0 host (`ProgramDescriptor`), legacy positional kernels | Conditional RTA counts (ISCLOSE / scalar-B / RM-vs-tile) → Rule 5; **host-injected activation define chains** (`PROCESS_*_ACTIVATIONS`) + `HAS_ACTIVATIONS`→conditional CBs; unnamed `TensorAccessorArgs` CTAs; `Buffer*` RTAs |
| `unary` | eltwise | Metal 2.0 host, legacy kernels (unbound `_metal2` forks exist) | Conditional RTA count across layout modes; SFPU op-chain via injected defines (`SFPU_OP_CHAIN_0`); packed float scalars in `uint32_t` RTAs |
| `ternary` (`where`) | eltwise | Metal 2.0 host, legacy kernels; reuses `binary_ng` headers | Variant-dependent CB topology + CTA prefix (TTS/TST/TTT); 27-slot positional reader RTA schema; `Buffer*` RTAs; broadcast define grid; noop-core arg padding |
| `argmax` | reduction | Mixed — NC factory legacy, single/multi-core `ProgramDescriptor` | Legacy semaphores as CTAs (multicore); layout-driven kernel+CTA switching; per-core-group compute duplication; 28+ accessor CTAs; template specialization on `data_format` |
| `matmul` | matmul | Mixed — some `ProgramDescriptor`, some legacy; **hybrid** named+positional kernels | Conditional CTA/RTA for bias & sharding → Rule 5; shared compute across **8+ factories** with *positional* layout drift; `get_named_compile_time_arg_val` + positional hybrid; `Buffer*` RTAs; `TensorAccessorArgs` tails |
| `conv2d` | conv | Mixed — `ProgramDescriptor` host, legacy kernels | ~38 CTAs; mcast-topology-varying RTA counts → Rule 5; `Buffer*` RTAs; legacy `Semaphore<>` from a CTA; **dummy-CTA padding** to stabilize layout; `CONFIG_TENSOR_IN_DRAM` conditional reads |
| `pool/generic` | pool | Mixed — `ProgramDescriptor` host, legacy kernels | **Superset CTA vector** shared by two kernel files (dead slots); ~55 positional CTAs; conditional accessor / `return_indices` args → Rule 5; signed-`int` reinterpretation of `uint32_t` CTAs; split-reader duplication |
| `topk` | reduction | Mixed — single-core `KernelSpec`+named schema, multi-core `KernelDescriptor`+positional | `GENERATE_INDICES` conditional RTA → Rule 5; single-core already reads `get_arg(args::)` in `kernel_main` (recipe's exact "before"); **two host paradigms in one op**; semaphore IDs as CTAs |
| `layernorm` | normalization | Mixed — `ProgramDescriptor` host, **hybrid** `get_named_compile_time_arg_val` kernels | Conditional RTA (`input_is_row_major`) → Rule 5; same compute bound twice with different RTA schemas (all-to-all vs not); variable-length mcast NOC-coord arrays (`get_arg_addr`); `DO_COL_MASK`/`IDLE_CORE`; `constexpr`/`volatile` CTA dup; shared writer across ops |
| `groupnorm` | normalization | Mixed — `ProgramDescriptor` host, hybrid named-CTA kernels | Variable-count mcast sender RTAs + pointer/array tails → Rule 5; sender/receiver same-file different named-CTA maps; optional mask tensors (`FUSE_NEGATIVE_MASK`); `PAD_CORRECTION`/`ARCH_*`; dual kernel-groups |
| `rmsnorm_distributed` | normalization | Mixed/delegated — Metal 2.0 `get_arg(args::)` forks (no `TT_KERNEL`), legacy siblings kept | Natural token fallback (`#ifdef FUSE_GAMMA` → `dfb::x_normed`/`out`) — **textbook Rule 4 ✓**; conditional LN-only intermediate DFBs (RMS vs LN fork); `get_arg(args::)`-without-`TT_KERNEL` = recipe's target form |
| SDPA (`transformer/sdpa`) | transformer | Mixed — Metal 2.0 host, legacy positional kernels; **7+ factories** | Causal vs non-causal reader RTA sets → Rule 5; **conditional CRTA registration**; variable-count semaphore CTAs; `override_runtime_arguments` re-patching; ring/fabric/mux extra kernels; `ARCH_*` in shared headers |
| `all_gather` | ccl | Legacy (`CreateKernel`/`SetRuntimeArgs`) | `USE_WORKER_MUX` conditional CTAs+RTAs → Rule 5; **helper-appended variable-length** fabric/routing RTAs; semaphore L1 addrs + NOC coords + chip/mesh IDs as RTAs; `GlobalSemaphore`; `FABRIC_2D` build-time flag toggles includes/types |
| `reduce_scatter` | ccl | Legacy host; delegates to `experimental/ccl` kernels | 4 global semaphores; `*_IS_SHARDED` conditional CTA-tail blocks inside `#ifdef`; `USE_WORKER_MUX`; 24+ fixed CTAs + route-info slabs; semaphore ID→address; hybrid `get_named_compile_time_arg_val` in `kernel_main` |
| `embedding` | embedding | Mixed — `ProgramDescriptor` host, legacy kernels | `PADDED` conditional `pad_token` RTA → Rule 5; `PADDED`/`BINARY` gate conditional CBs (Rule 4); `Buffer*` RTAs; `TensorAccessorArgs` tail; output-sharded writer absent; shared writer from `ttnn/kernel` |
| `kv_cache` | kv_cache | Mixed — `ProgramDescriptor` host, legacy kernels | `INPUT_SHARDED` conditional-resource gate (Rule 4); **stale `#ifdef BACKWARDS`** (define never set); `Buffer*` RTAs; two compute bindings of one source with different `CTA[6]`; `override_runtime_arguments` refresh; compute is CTAs-only (Rule 1 no-RTA variant ✓) |
| `kda` (`sigmoid_gated_rms_norm`) | experimental | **New-syntax reference** — full `ProgramSpec` + `TT_KERNEL` | Reference form ✓: per-kernel distinct CTA lists (reader 4 / writer 3 / compute 1); `epsilon_bits` = **float bit-pattern** in a `uint32_t` CTA; uses `tensor::`/`dfb::` (not `args::`) for resources; **no `#ifdef`** |

---

## Deviations & special situations (candidate recipe gaps)

The five surveys converged hard: the *same* handful of patterns recur across almost every domain.
Grouped by theme, with representative evidence (not exhaustive — see the raw survey transcripts for
the full `file:line` lists).

### A. Variable-count / conditionally-registered argument streams — **the dominant blocker**

The recipe's Rule 5 ("conditional args → hard stop") is correct but its softmax example badly
*undersells* how pervasive and how varied this is. Two mechanisms, same effect:

- **A1 — Conditional arg *count* on one kernel entry**, via host `push_back` / `insert` /
  `append_*_rt_args` behind an `if`, or a helper that appends N args. Seen in: permute padding RTAs
  (`permute_tiled_program_factory.cpp:415-423`); matmul bias CTAs/RTAs + `in0`-sharded-vs-interleaved
  CTA branches (`matmul_multicore_reuse_optimized_program_factory.cpp:371-373,469-474`,
  `..._mcast_1d_...:369-436`); conv mcast-topology + activation-reuse RTAs
  (`conv2d_op_sharded_program_factory.cpp:1315-1382,1372-1380`); pool `return_indices` / accessor
  args (`pool_multi_core_program_factory.cpp:822-824,964-973`); SDPA causal reader RTAs +
  **conditional CRTA** (`sdpa_program_factory.cpp:1420-1435`, `ring_joint_sdpa_program_factory.cpp:2563-2581`);
  layernorm `input_is_row_major` + role-split compute (`layernorm_op_multi_core.cpp:597-599`,
  `sharded_layernorm_factory_helpers.cpp:1482-1517`); groupnorm mcast sender
  (`groupnorm_mcast_program_factory.cpp:1198-1241`); embedding `pad_token`
  (`embeddings_rm_program_factory.cpp:271-273`); topk `GENERATE_INDICES`
  (`topk..._program_factory.cpp:358-359`); CCL `USE_WORKER_MUX` / topology / `num_iters`
  (`all_gather_unicast_factory.cpp:458-461,602-630`). → **Rule 5 hard stop.**
- **A2 — Variable-length positional arg *blobs* addressed by pointer/index arithmetic**
  (`get_arg_addr(i)`, `arg_idx++` loops) — **not** `get_vararg`. Seen in: concat block-sharded
  (`num_transfers` + 9×N: `reader_writer_block_sharded_concat.cpp:31-42`) and N-tensor readers
  (`3+3×N`: `reader_concat_interleaved_start_id.cpp:38-42`); pad sharded-height blob
  (`reader_pad_dims_rm_sharded.cpp:17-22`); layernorm/groupnorm mcast NOC-coord arrays
  (`writer_unary_sharded_ln.cpp:47`); CCL route-info slabs + `append_routing_plane_connection_manager_rt_args`
  (`line_reduce_scatter_minimal_async_writer.cpp:59-125`).

**Verdict.** The recipe treats "varargs" (`get_vararg`) as the only variable-count shape (→ partial)
and "conditional registration" as a separate stop. In the corpus these are one family, and A2 is
*more* common than `get_vararg`. Both should collapse into one concept with detection guidance.

### B. Unnamed `TensorAccessorArgs` / accessor-tuple CTA blocks — **pervasive, uncovered**

Almost every reader/writer appends `TensorAccessor` metadata into the CTA vector with **no registered
names** (`TensorAccessorArgs::append_to(...)`, `make_tensor_accessor_args_tuple<N>()`), consumed
in-kernel as `TensorAccessorArgs<offset>()`. Seen in: concat
(`reader_concat_stick_layout_interleaved_start_id.cpp:57`), pad, binary_ng
(`binary_ng_program_factory.cpp:1321-1326`), ternary, argmax (`...multicore.cpp:299-300`), matmul
(`..._mcast_1d_...:499-501`), conv, pool, CCL (`unicast_reader.cpp:33-34`), embedding
(`embeddings.cpp:38`). The recipe assumes **named `uint32_t` CTAs only** and has *no* story for this
variable-width, unnamed block. This is the #1 recurring *structural* gap.

### C. Prerequisite / hybrid named-arg state — Step 0 is satisfied by almost nobody

The corpus is a **spectrum**, and the recipe's Step 0 ("already ported to Metal 2.0, kernels use
`get_arg(args::…)`") sits near the *end* of it:

1. **Legacy positional** (`get_arg_val` / `get_compile_time_arg_val`) — most kernels, even under a
   Metal 2.0 `ProgramDescriptor` host (concat, pad, binary_ng, unary, ternary, matmul, conv, pool,
   embedding, kv_cache, argmax, SDPA, all_gather).
2. **Hybrid**: `get_named_compile_time_arg_val("cb_*")` for CB indices **+ positional** for the rest
   (layernorm `layernorm.cpp:49-66`, groupnorm `groupnorm.cpp:93-140`, matmul
   `bmm_large_block_zm_fused_bias_activation.cpp:199-226`, reduce_scatter ring kernels).
3. **`get_arg(args::…)` without `TT_KERNEL`** — the recipe's exact "before" (rmsnorm
   `rmsnorm_post_allgather_metal2.cpp:55-61`, topk single-core `topk.cpp:119-138`).
4. **Full `TT_KERNEL`** — only `kda`.

**Verdict.** Only levels 3–4 are eligible. Step 0 should say so explicitly and route levels 1–2 back
to "needs the base Metal 2.0 kernel-arg port first" — this "Metal 2.0 *host* but positional/hybrid
*kernels*" state is the single most common "not eligible yet" verdict in the survey.

### D. Host-injected define-string op chains (fused activations / SFPU) — a new flag class

Eltwise/matmul/norm factories inject **code fragments** as defines (`PROCESS_LHS/RHS/POST_ACTIVATIONS`,
`SFPU_OP_CHAIN_0`, activation init/func snippets), meta-macro'd (`HAS_ACTIVATIONS`), often co-gating
conditional CBs (`c_3`/`c_4`). Seen in: binary_ng (`binary_ng_utils.cpp:574-591`), unary
(`unary_op_utils.cpp:1129-1146`), ternary, layernorm (`layernorm_op_multi_core.cpp:413-419`), matmul,
conv. These are **neither** pure-value flags **nor** simple token gates — they inject code. The flag
census (Step 2) has no column for them. **Verdict:** they stay `#define`/`#ifdef` (unpromotable,
not `if constexpr`-convertible); the recipe should name the class so the porter recognizes and leaves
them.

### E. Same kernel `.cpp`, multiple bindings with different schemas — *intra*-factory

Distinct from the recipe's "shared kernel across ops → fork": here one source is bound **N times
within one factory / one op** with different arg schemas. Seen in: concat reader+writer same file
(`concat_s2s_rm_program_factory.cpp:104-187`); fold `is_reader` split; argmax per-core-group
(`argmax_nc_program_factory.cpp:157-162`); layernorm compute bound twice (all-to-all vs not); groupnorm
sender/receiver; matmul shared compute w/ *positional* layout drift across 8+ factories
(`bmm_large_block_zm_fused_bias_activation.cpp:28-29`); pool superset vector for two files
(`pool_multi_core_program_factory.cpp:880-903`); kv_cache two compute bindings different `CTA[6]`
(`update_cache_multi_core_program_factory.cpp:352-362`). **Verdict / two sub-cases:** identical arg
**name set** per binding (differ only in values) → **convertible** (one signature, many runtime
bindings); different **count/names** per binding → the union problem → **hard stop or per-binding
fork**.

### F. Semaphores, fabric coords, and global semaphores as plain `uint32_t` args

The recipe (single-device softmax) never covers multi-device/fabric. The corpus passes semaphore
IDs/addresses, NOC (x,y), chip/mesh IDs, and `GlobalSemaphore.address()` as plain args, resolved via
`get_semaphore(get_arg_val(...))` — **not** `sem::` tokens. Seen in: argmax multicore
(`reader_argmax_interleaved_multicore.cpp:429-435`), conv, topk, all CCL
(`all_gather_unicast_factory.cpp:573-599`), SDPA ring. **Verdict:** these become plain named `uint32_t`
RTAs when unconditional; when conditional (mux/topology) they fold into Theme A (hard stop). CCL/fabric
ops are largely **out of scope until Metal-2.0-ported**, and even then their conditional fabric args
make most of them hard stops.

### G. `Buffer*` / tensor-ref runtime bindings + cache-hit re-patching

Descriptor-style factories register RTAs as `Buffer*` / tensor refs (framework resolves to an
address) for cache-hit patching, refreshed via `override_runtime_arguments` / `DynamicRuntimeArg`.
Seen in: matmul (`matmul_multicore_program_factory.cpp:174-188`), conv
(`conv2d_op_sharded_program_factory.cpp:1306-1308`), embedding, kv_cache
(`update_cache_multi_core_program_factory.cpp:393-405,436+`), ternary. **Verdict:** a `Buffer*` RTA
converts to a `uint32_t` **address** parameter (framework supplies the value); the
`override_runtime_arguments` path must keep the same name/slot. Worth a pitfall row + a cross-ref to
the base port recipe's override guidance.

### H. One-off curiosities (note, don't act)

- **Float bit-pattern CTAs** — `std::bit_cast<uint32_t>(eps)` (`layernorm_op_multi_core.cpp:593`),
  `kda`'s `epsilon_bits` `memcpy` (`sigmoid_gated_rms_norm_program_factory.cpp:91-92`). Legal `uint32_t`;
  worth one line noting "a CTA/RTA may carry a non-integer bit pattern."
- **`constexpr`/`volatile` duplicate of one CTA** for the LLKs (`layernorm_sharded.cpp:31-34`).
- **Signed reinterpretation** of a `uint32_t` CTA (`reader_pool_2d.cpp:170`).
- **Stale/dead `#ifdef`** whose define is never set (`reader_fill_cache_interleaved_start_id.cpp:33-38`).
- **Noop-core arg padding** (`CoreRuntimeArgs(count, 0)`, ternary `:705-707`); **`IDLE_CORE`**
  whole-kernel no-op via define (`layernorm_sharded.cpp:22-24`).
- **Conditional whole-*kernel* binding** (permute `swap_hw` compute, embedding sharded writer absent)
  — the factory builds a different *kernel set* per config; each present kernel just converts normally.
- **Runtime-arg budget** (341-arg limit documented in concat `concat_device_operation.cpp:220`).
- **Two host paradigms in one op** (topk `KernelSpec` single-core vs `KernelDescriptor` multi-core).

### Positive confirmations (recipe is right here)

- **`kda`** is the reference `TT_KERNEL` shape: each kernel entry has its *own* template CTA list; a
  float bit-pattern rides a `uint32_t` CTA; resources use `tensor::`/`dfb::`, not `args::`; no `#ifdef`.
- **rmsnorm_distributed** `#ifdef FUSE_GAMMA` → `dfb::x_normed`/`out` is a **textbook Rule 4** case
  (leave the `#ifdef`, natural fallback) — exactly as the recipe prescribes.
- **kv_cache compute** is a clean **no-RTA, CTAs-only** kernel — validates the Rule 1 no-CTA/RTA-only
  variants.
- **fold** has a clean conditional-DFB + matching `#ifdef` (Rule 4), converted signature-only.

---

## Recommended recipe follow-ups

> **Read the [Scope filter](#scope-filter--metal-20-kernels-only-the-view-that-actually-matters)
> first.** The tiers below were derived from the *full* survey (including pre-Metal-2.0 ops). Once
> filtered to in-scope Metal-2.0 kernels, **only the three small "in-scope-worthy additions" in the
> Scope filter are actionable now.** The Tier-1 items below are almost all either (a) resolved by the
> base Metal 2.0 port itself (accessor CTAs → `tensor::`, `Buffer*` → `tensor::`, legacy sem → `sem::`),
> or (b) future concerns for when eltwise/matmul/CCL reach Metal 2.0 — kept here as a forward-looking
> record, not an immediate queue.

Ordered by how often the *full* (pre-filter) survey hit them. **None applied yet.**

**Tier 1 — genuine gaps worth a rule / section:**

1. **Generalize "conditional args" → "variable-count / conditionally-registered argument streams"
   (Theme A).** Fold `get_vararg` (partial), conditional registration (hard stop), *and*
   pointer-addressed variable-length blobs (`get_arg_addr` / `arg_idx++` / host `append_*` helpers)
   into one concept. Give the porter a **detection procedure**: scan the *factory* for arg
   `push_back`/`insert`/`append_*_rt_args` inside a loop or behind an `if`, and the *kernel* for
   `get_arg_addr` / `arg_idx++` / `get_vararg`. This is the most common blocker in every domain.
2. **Add a story for unnamed `TensorAccessorArgs` / accessor-tuple CTA blocks (Theme B).** They
   appear in nearly every reader/writer and the recipe is silent. Decide and document: does the
   accessor block stay a trailing `TensorAccessorArgs<offset>()` construct (with only the *scalar*
   named CTAs lifted into the template list), and does that compose with the PR #46623 shim? This
   likely needs a small experiment before the recipe can prescribe.
3. **Sharpen Step 0's precondition and the "not ready" states (Theme C).** State the exact gate
   (kernels on `get_arg(args::…)`), and explicitly exclude legacy-positional and
   `get_named_compile_time_arg_val`-hybrid kernels as "needs the base Metal 2.0 kernel-arg port
   first." "Metal 2.0 *host* + positional/hybrid *kernels*" is the corpus norm.
4. **Name the injected-define op-chain flag class (Theme D)** in the Step 2 flag census — code-string
   defines (`PROCESS_*_ACTIVATIONS`, `SFPU_OP_CHAIN_*`) are unpromotable and stay `#ifdef`; the
   porter should recognize and leave them.
5. **Add a "same source, multiple bindings" rule (Theme E):** convertible iff every binding registers
   the *same arg-name set*; otherwise hard stop or per-binding fork — distinct from the existing
   cross-op shared-kernel fork.

**Tier 2 — a pitfall row / short note:**

6. **Semaphore / fabric-coordinate / global-semaphore args as plain `uint32_t` (Theme F)** — map to
   named `uint32_t` RTAs; note CCL/fabric ops are out of scope until ported and mostly hard-stop then.
7. **`Buffer*` / tensor-ref RTAs → `uint32_t` address params, preserve `override_runtime_arguments`
   (Theme G)** — cross-ref the base port recipe's override handling.
8. **Float-bit-pattern and signed-reinterpreted `uint32_t` args (Theme H)** — one line: a scalar arg
   may legally carry a non-integer bit pattern or signed value; the `uint32_t` spelling is unchanged.

**Tier 3 — note only, no recipe change:** the remaining Theme H curiosities (constexpr/volatile CTA
dup, stale `#ifdef`, noop-core padding, `IDLE_CORE`, arg-budget limit, dual host paradigms).

**Cross-cutting caveat for the recipe's scope.** The survey strongly suggests the named-args pass, as
written, applies cleanly to a *small* slice today (single-device, already-`get_arg(args::)` ops like
softmax, rmsnorm_distributed forks, topk single-core, and greenfield `kda`). The large, sharded,
fused, and multi-device ops are blocked *upstream* — either not yet on `get_arg(args::)` at all, or
carrying variable-count arg streams / accessor CTA blocks / conditional fabric args that are hard
stops regardless. That is useful scoping information in its own right: **the recipe should state that
its clean target is the simple single-device op, and that the corpus's heavy hitters need
prerequisite work (base Metal 2.0 kernel-arg port; unbound accessors #52179) before they qualify.**
