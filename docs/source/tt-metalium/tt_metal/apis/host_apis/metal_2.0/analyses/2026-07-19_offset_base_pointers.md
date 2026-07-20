# Metal 2.0 — Offset base pointers: porting triage

**Author:** Audrey and Claude

**Purpose:** Enumerate every way a legacy op gets a **non-base (offset) device pointer** to a kernel, classify them into four types, and flag which ops need a light refactor **before** a Metal 2.0 port can proceed.

---

## The core constraint

Metal 2.0 delivers a kernel only the **base address of a bound memory object** — via a kernel `TensorBinding` (the base rides a CRTA; the kernel reconstructs addresses through a `TensorAccessor` whose args the framework auto-builds) or via a DFB `borrowed_from` a `TensorParameter` (base only). There is **no mechanism to hand a kernel a non-base pointer** — an interior address, a `base + offset`, or a base the framework can't regenerate.

So any legacy op that produces an *offset* base pointer is a porting consideration. There are exactly four ways it happens. Two need an ops-team refactor before porting; one is out of current scope; one is a non-issue.

### At a glance

| Type | Source | Ops-team action before port? | Priority |
|---|---|---|---|
| **1** | Offset pointer passed as an arg, used **raw** | **Yes** — mechanical (move the offset into the kernel) | routine |
| **2** | Offset pointer passed as an arg, used in a **`TensorAccessor`** | **⚠ Needs design discussion** — not mechanical | **flag early** |
| **3** | Offset **borrowed-memory CB** (`address_offset`) | No — Python ops, out of current scope | framework note |
| **4** | In-place op **`MeshTensor` offset trick** (`narrow`) | **No action** — ports correctly | none |

> How ops are detected (grep signatures per type) will live in the **audit recipe**, which flags these at audit time. This document is the findings + per-op tables the recipe references.

*Factory paths in Types 1–2 and 4 are relative to `ttnn/cpp/ttnn/operations/`; Type 3 paths are relative to `models/demos/deepseek_v3_b1/`. Arguments are named by their receiving kernel + role rather than a numeric slot, since roles survive edits (indices and line numbers do not).*

---

## Type 1 — Offset pointer passed as an argument, used raw

**What it is.** The host folds an offset into a buffer's base (`buffer()->address() + <offset>`) and passes the *combined* value as a (common) runtime argument. The kernel uses it **directly as a NoC address** (the `.addr` of a `noc_async_read`/`noc_async_write`), never through a `TensorAccessor`.

**How to recognize it.** An address-valued RTA whose value is `…->address()` with host arithmetic folded in (often via a local variable a few lines away), consumed on-device as a raw bank/NoC address.

**Remedy (ops team, before porting — straightforward).** Pass the **base** (which becomes a `TensorBinding`) and the **offset as a separate scalar** runtime arg; move the addition into the kernel. The offset must be deterministic from cache-miss inputs (attrs / tensor args / mesh coords) — for all ops below it is (shard/bank/head geometry), so the fix is a mechanical arg-split, no design decision.

| op | factory | argument | offset expression | caveat |
|---|---|---|---|---|
| `roll` | `data_movement/roll/device/`<br>`roll_program_factory.cpp` | reader RTA — `dst_bank_base`<br>reader RTA — `src_bank_addr` | `buf->address()`<br>`+ (shard_idx / num_dram_banks)`<br>`· dram_shard_size`<br>`(+ intra-shard offset)` | single reader kernel. `src_bank_addr` is a **per-transfer** field, not a fixed slot. DRAM_TILE folds the intra-shard offset in; DRAM_RM passes it as a separate arg |
| `nlp_create_qkv_heads` | `experimental/transformer/`<br>`nlp_create_qkv_heads/device/`<br>`nlp_create_qkv_heads_program_factory.cpp` | reader RTA — `q_start_addr`,<br>`kv_base_addr`, `kv_start_addr`<br>writer RTA — `q_start_addr`,<br>`v_base_addr`, `v_start_addr` | `<q/k/v base>`<br>`+ head_start_idx · head_size` | fused-QKV path only; the separate-KV path passes clean bases |
| `nlp_create_qkv_heads_boltz` | `experimental/transformer/`<br>`nlp_create_qkv_heads_boltz/device/`<br>`nlp_create_qkv_heads_boltz_program_factory.cpp` | reader RTA — `q_start_addr`,<br>`kv_base_addr`, `kv_start_addr`<br>writer RTA — `q_start_addr`,<br>`v_base_addr`, `v_start_addr` | `<q/k/v base>`<br>`+ head_start_idx · head_size` | same as `nlp_create_qkv_heads` |

*Notes.* `roll`'s own **DRAM_RM mode already passes base + offset separately** — the target shape exists in the same file. The `nlp_create_qkv_heads` offsets are conditional on the **fused-QKV** path (`read_from_input_tensor_kv == false`); the separate-KV path passes a clean base.

---

## Type 2 — Offset pointer passed as an argument, used in a `TensorAccessor` ⚠

**What it is.** Same host-side fold, but the kernel feeds the offset address into a **`TensorAccessor` as its base**. Because Metal 2.0 auto-builds the accessor's args from the *base* binding, there is no seam to inject a pre-offset base — the offset is entangled in the accessor's addressing, not a trailing `+`.

**How to recognize it.** An offset address (`…->address() + <offset>`) in an RTA whose receiving kernel constructs `TensorAccessor(args, that_addr, …)`.

**Remedy — needs exploration and discussion (NOT mechanical).** Moving one addition into the kernel does not suffice. Options to weigh (per design discussion; unresolved):
- The affected variants are all **row-major**; if the RM path is seldom-used / unoptimized, the ops team may be willing to **rejigger** it to a base-binding + kernel-side offset (as the Type-1 ops do), possibly at a perf cost.
- Confirm the offset-into-accessor pattern **isn't already buggy** (worth a spot check).
- A first-class **tensor-view / sub-region** binding feature (thought-experiment `MeshTensorView`) — but the RM fit is awkward.

These should be surfaced early (to Audrey / framework), not handed to a porter as a mechanical task.

| op | factory | argument | offset expression | caveat |
|---|---|---|---|---|
| `slice` | `data_movement/slice/device/`<br>`slice_program_factory_rm.cpp` | reader RTA[0] — input base | `input->address()`<br>`+ begins_bytes − misalignment` | the canonical case |
| `padded_slice` | `experimental/padded_slice/device/`<br>`padded_slice_rm_program_factory.cpp` | reader RTA[0] — input base | `start_addr`<br>`+ begins_bytes − misalignment`<br>`(+ per-core width_offset)` | two-stage host fold |
| `slice_write` | `experimental/slice_write/device/`<br>`slice_write_rm_sharded_input_program_factory.cpp` | writer RTA[0] — output base | `output->address()`<br>`+ output_start[-1] · elem_size`<br>`+ width_offset` | **BLOCK_SHARDED only**; HEIGHT_SHARDED folds offset 0 (bare base) |

*Note.* Only the **row-major** slice-family factories are affected. The **tiled** variants of the very same ops (`slice_program_factory_tile`, `padded_slice_tile`, `slice_write_tiled_sharded_input`) pass a clean **tile-index scalar** and are unaffected — the wall is an RM-layout phenomenon.

---

## Type 3 — Offset base pointer via borrowed-memory CB (`address_offset`)

**What it is.** A `CircularBuffer` built on *borrowed* memory at a **non-base offset**. Legacy borrows a CB's L1 from a user-managed allocation (`set_globally_allocated_address(Buffer&/MeshTensor&)`), which captures the **base**. An offset is added *only* through the newer **`address_offset`** field (`set_address_offset` / `CBDescriptor.address_offset` / the 4-arg `UpdateDynamicCircularBufferAddress`); the borrowed CB address resolves to `shadow_buffer.address() + address_offset`. It is used to pack **many CBs into one backing tensor at running byte offsets**.

**Metal 2.0 status.** `DataflowBufferSpec::borrowed_from` is a `TensorParameter`, **base only — no offset field**. So this pattern has no Metal 2.0 expression today.

**Scope — out of the current (C++ op) porting effort.** The *only* users are **DeepSeek v3 Python model ops** (`models/demos/deepseek_v3_b1/`) — ~106 non-zero `address_offset=` sites, via the `ttnn.cb_descriptor_from_sharded_tensor(address_offset=…)` binding and the `cb_descriptor_from_overlapped_tensor` helper. No C++ op in scope uses a non-zero offset.

**Action — none for the ops team.** This flags a **Metal 2.0 capability decision** — whether `borrowed_from` should grow offset support — for when/if these Python ops enter porting scope.

| Python op / file (`models/demos/deepseek_v3_b1/…`) | Notes |
|---|---|
| `fused_ops/moe/op.py` | ~38 sites, running `kv_offset` cursor |
| `fused_ops/attention_block/op.py` | ~42 sites, `sdpa_*_running_offset` |
| `fused_ops/pre_sdpa/op.py` | ~19 sites |
| `fused_ops/post_sdpa/op.py` | 4 sites |
| `fused_ops/lm_head_sampling/op.py` | ~13 sites (mixed non-zero / 0) |
| `fused_ops/moe_routed_expert/op.py` | via `cb_descriptor_from_overlapped_tensor` |
| `micro_ops/matmul_expert/op.py` | region-packing into one DRAM tensor |
| `circular_buffer_utils.py` | the `cb_descriptor_from_overlapped_tensor` helper (funnel) |
| `models/experimental/ops/descriptors/fusion/cb_allocator.py` | fusion library — propagates offsets |
| `micro_ops/sampling/op.py` | uses the API at `address_offset=0` (benign) |

*Note.* The C++ `deepseek_moe_gate` / `generalized_moe_gate` descriptor builders call the helper at offset 0 (benign; no action).

---

## Type 4 — Offset base pointer from the in-place-op `MeshTensor` offset trick (`narrow`)

**What it is.** `ttnn::narrow` returns a **zero-copy sub-tensor view**: an externally-owned `MeshTensor` whose `.address()` is `parent_base + offset` (an interior pointer), built via the explicit-address overload `MeshBuffer::create(…, parent_base + offset)` and kept alive by holding the parent's storage. It violates the documented `MeshTensor` sole-owner invariant, but skips the copy.

**Metal 2.0 status — NOT a problem.** Both binding paths were traced end-to-end:
- **TensorBinding:** the interior base is delivered to the kernel *verbatim*, and the accessor args are sourced from the (matching) narrowed spec → correct addressing. The base is **re-emitted fresh on every enqueue**, so it is cache-safe (it rides the typed channel, not a smuggled RTA).
- **DFB `borrowed_from`:** the adopted view reports the **narrowed** per-bank size, so the size/L1 checks are sound.

Correct **provided the narrowed `TensorSpec` matches the subtensor** — which `ValidateTensorArgs` enforces (spec equality). No framework backstop beyond the spec, but the spec check is the load-bearing guard and it holds.

| op | factory | argument | offset expression | action |
|---|---|---|---|---|
| `narrow` | `data_movement/narrow/`<br>`narrow.cpp` | `TensorBinding` / `borrowed_from` base<br>(the view's own reported base) | `parent_base + offset`<br>via `MeshBuffer::create(…, addr)`<br>— DRAM interleaved & L1 sharded | **None** — ports as-is |

*Caveat on the precondition (on record, not a port blocker).* The "ports correctly *provided the narrowed spec matches the subtensor*" verdict rests on `narrow` emitting a base that is actually consistent with the spec it reports — and Metal 2.0 validates the **spec only**, so it is not a backstop if `narrow` gets that wrong. One case where it does: last-dim (width) narrowing of a **DRAM-interleaved row-major** tensor silently mis-addresses for `start > 0` — the page-granular offset math truncates (`start_page_id = start / width → 0`), so the view points at element 0. Reachable and untested; code-traced, not yet run. This is a `narrow` defect for the op owner to fix, independent of Metal 2.0 — noted here only because it qualifies the Type 4 verdict.
