# variable_matmul refactor backlog

Pain points and proposed fixes for `tt-train/sources/ttml/metal/ops/variable_matmul/`.
Update as new pain is encountered. Open items are tracked status `[ ]`; resolved are `[x]`
with the commit short-SHA.

## 1. OFFSETS_ROLE as numeric preprocessor enum — IN PROGRESS

**Pain:** Six role values (1..6) encode combinations of orthogonal axes (M-axis recompute,
in0-row offset, out-row offset, in0-K offset, in1-K offset). Each kernel branches with
`#if OFFSETS_ROLE == X || OFFSETS_ROLE == Y || ...`, repeated in `dm_in0_sender.cpp`,
`dm_in1_sender_out.cpp`, `compute.cpp`. Adding any new combination = new enum value +
N more `#if` branches.

**Fix:** Public `OffsetsRole` enum stays as the user-facing API. Factory derives **orthogonal
boolean flags** (`OFFSET_M_AXIS`, `OFFSET_IN0_ROW`, `OFFSET_OUT_ROW`, `OFFSET_IN0_K`,
`OFFSET_IN1_K`) and passes them as defines. Kernels branch on flags only.

## 2. `use_offset` CTA computed in three places — OPEN

**Pain:** Same big OR expression in `in0_sender`, `in0_receiver`, `in1_sender` CTA setup —
each kernel must list every flag combination that needs the offset address path. Bug
`890fec52a6c` was exactly this (receiver missing `input_and_weight_k_active`).

**Fix:** Derive `use_offset_in0`, `use_offset_in1` once in the factory; pass to all three
kernels with the same value.

## 3. RT-arg layout as magic indices — OPEN

**Pain:** Host writes by `args[IDX_CONSTANT]`, kernel reads by `argidx++`,
`override_runtime_arguments_callback` writes by `IDX_CONSTANT` again. Three independent
copies of the layout that must stay in sync; index constants are only declared once
(in the override callback). Adding/removing one arg shifts every downstream index in
two callsites silently.

**Fix:** Define a single `struct VariableMatmulRTArgs` (or at minimum a shared header with
`enum class In0ArgIdx : uint32_t`) used by host, override, and kernel.

## 4. Duplicated dataflow logic between dm_in0_sender and dm_in1_sender_out — OPEN

**Pain:** Defer-write state machine, chain handshake, OFFSETS_ROLE parsing, mcast
forwarding — all near-identical between the two files. The OFFSETS_ROLE block alone is
~80 duplicated lines. `matmul_dataflow_common.hpp` exists but only covers tile-read
helpers.

**Fix:** Lift the shared structure into `matmul_dataflow_common.hpp` (or template into a
single source) so kernel-side logic lives in one place.

## 5. "Unused placeholder" CTAs kept for layout compat — OPEN

**Pain:** Several CTAs marked `// 0, 1, 9 are unused placeholders (kept for compile-time
arg layout compatibility)` — exist only so TensorAccessor finds its args at the right
offset. Subtle source of mistakes when reordering.

**Fix:** Use `TensorAccessorArgs<N>` indirection or pad explicitly via a tag struct, so the
unused slots aren't load-bearing.

## 6. `cb_ctrl` (c_8) one-shot mailbox is fragile — OPEN

**Pain:** Single-page CB used as a one-shot publish/wait mailbox between a dm kernel and
compute. If a dm kernel exits without pushing (e.g. some future early-return path),
compute deadlocks at `cb_wait_front(cb_ctrl)` with no diagnostic. No watcher hint either.

**Fix options:** (a) Make publish unconditional even on early-return paths (always push a
"sentinel" value), or (b) compute reads via `noc_async_read` from a known L1 address
instead of CB handshake.

## 7. 1000-line factory `create()` — OPEN

**Pain:** Inline kernel discovery, CB creation, semaphore creation, RT-arg construction —
all in one function. The override-callback then mirrors a subset by hand; bug
`d3e6f8396dc` (`override_runtime_arguments needs_offsets mismatch`) was a direct
consequence of forgetting to mirror a host-side condition.

**Fix:** Split into `build_kernels()`, `build_cbs()`, `build_rt_args()`. The override
callback should literally call the same `build_rt_args()` rather than re-implement it.

## 8. `defer_write_k_block` computation now split between host (passes core.y) and kernel — OPEN (minor)

**Pain:** After `b73367f090e`, the host passes raw `core.y` at the
`defer_write_k_block` RT-arg slot and the kernel recomputes the stagger. The RT-arg
name still says `defer_write_k_block` — misleading.

**Fix:** Rename the RT-arg slot to `core_y_index` end-to-end. Drop the legacy name from
docs/comments.
