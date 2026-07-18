# TEMPORARY — CB/DFB reference info (frozen snapshot)

> **What this is.** A frozen copy of sections from
> [`metal2_port_patterns.md`](metal2_port_patterns.md) that the Quasar audit
> ([`cb_dfb_kernel_audit_helper.md`](cb_dfb_kernel_audit_helper.md)) links into.
> It exists only to keep those links pointing at stable content while the
> porting recipe is refactored for the **CB→DFB Gen1-parity pivot** (under which
> the live patterns doc has diverged — e.g. DM self-loops are now legal on Gen1).
>
> **This is a pre-pivot snapshot — do not treat it as current porter guidance.**
> The authoritative porter-facing docs are the live
> [porting recipe](port_op_to_metal2_recipe.md) and
> [patterns catalog](metal2_port_patterns.md).
>
> **Delete this file** once `cb_dfb_kernel_audit_helper.md` is reworked to be
> self-contained (it will absorb the Quasar-uplift guidance it needs). Sections
> are copied verbatim; only same-file anchor links were rewritten to point back
> at `metal2_port_patterns.md`.

---

## Pattern: Sync-free and single-ended CBs → self-loop DFB (interim workaround)

**Category**: Pattern (interim workaround)

**A word on legacy CBs — read this first.** Legacy gave kernel authors essentially *one* primitive for "a chunk of L1 the kernel touches": the CircularBuffer. So they used it for everything — genuine producer→consumer FIFOs, but also private scratch memory, base-pointer windows onto resident tensors, and output staging nothing ever drains. When all you have is a CircularBuffer, everything looks like one. Metal 2.0 splits these back apart (DFBs, plus the coming kernel-scratchpad and local-`TensorAccessor` features), so **you** now look at each legacy "CB" and decide what it actually is — along **two orthogonal axes that must not be fused**:
- **Synchronized vs. sync-free** — does the kernel drive the FIFO machinery (`reserve_back`/`push_back`/`wait_front`/`pop_front`) for a genuine cross-kernel hand-off (**synchronized** → a real DFB), or just grab the base pointer and walk the bytes, ignoring sync (**sync-free** → really scratch memory or a tensor view, not a CB at all)?
- **Endpoint multiplicity** — how many FIFO producers/consumers per node? The SPSC legality axis, checked separately ([DFB endpoint legality](port_op_to_metal2_audit.md#dfb-endpoint-legality-spsc)).

This entry handles the CBs the spec validator rejects because they can't present a **FIFO producer *and* a FIFO consumer on distinct kernels**. The interim binding workaround is the bridge until the real Metal 2.0 constructs land — but **the workaround now forks by the touching kernel's type** (a self-loop lowers only on compute), so read the **Decision** below before binding; some cases now STOP rather than bridge.

**Recognition signal**: A CB that lacks a usable FIFO producer–consumer pair, so the validator (which requires **≥1 PRODUCER and ≥1 CONSUMER** binding) rejects it. Two shapes land here:
- **Sync-free** — the kernel uses the CB purely as an *address source*: a base-pointer grab via `get_read_ptr` / `get_pointer_to_cb_data` (or `get_write_ptr`), then a direct memory access, with **no FIFO ops at all** — nothing `push_back`s, nothing `wait_front`s. The legacy idiom borrows a CB onto a resident tensor's buffer because, pre–Metal 2.0, that was the most convenient way to hand a kernel a base pointer to resident memory. (Sync-free CBs aren't really CBs — they're scratch memory or tensor views; see *Classify it*.)
- **Single-ended** — the CB *does* use the FIFO machinery, but on **one** side only: a FIFO producer with no consumer (or vice versa). Canonical case: the compute packer produces tiles (`reserve_back` / `push_back`) straight into an output-tensor-backed CB that nothing drains — a **synchronized** CB (real `push_back`), just missing its consumer. **Examples:** conv2d / pool `OUT` (compute packer → output shard); pool `out_idx_cb` (a DM kernel writes the argmax-index output).

The audit flags both as **FYI-P** for the common cases — the workaround keeps the port unblocked. The **one exception** is a **single-ended producer on a DM kernel** (a DM "packer"): it has no port-time workaround yet and **STOPs** the factory (see Decision; the audit gates it under [DFB endpoint legality](port_op_to_metal2_audit.md#dfb-endpoint-legality-spsc)).

> **Don't fuse this with the sync axis.** "Single-ended" is a *count* property (one endpoint), orthogonal to synchronized/sync-free (a *sync* property). The packer (`OUT`, above) is **synchronized but single-ended**; a base-pointer read of a resident lookup table (`recip`, below) is **sync-free**. Both need the bridge for the same reason — no usable FIFO producer+consumer pair on distinct kernels — but they have different long-term homes (*Classify it*).

**Decision — the resolution forks by the touching kernel's type, because a self-loop lowers only on compute.** A self-loop DFB (PRODUCER + CONSUMER on the *same* kernel) lowers **only on a compute kernel**. On a **DM** kernel a self-loop is *accepted by the spec validator but has no backend lowering* — `producer_risc_mask == consumer_risc_mask`, which the disjoint-mask requirement rejects (`dataflow_buffer.cpp`) — so a DM self-loop is a spec-accepts / backend-rejects footgun. The fork:

- **Compute kernel** (sync-free scratch/view, accumulator, or single-ended packer) → **self-loop it**: PRODUCER + CONSUMER both on that compute kernel, borrowing the [Self-loop DFB binding](metal2_port_patterns.md#pattern-self-loop-dfb-binding) mechanism. The white lie is contained to one kernel and lowers cleanly.
- **DM kernel, sync-free** (pure scratch / pointer-only, **no FIFO ops at all**) → **fabricate the consumer on a *different* kernel**: bind PRODUCER on the touching DM kernel, and CONSUMER on *any other* kernel in the program (which never references the DFB). Distinct kernels → disjoint masks → it lowers. Safe **only because nothing drives the FIFO**: no credits flow, so the fabricated consumer is inert at runtime.
- **DM kernel, single-ended *producer*** (real `reserve_back` / `push_back`, no consumer — a DM "packer") → **STOP and surface to the API owner** (record it in the port report; capitulate on the factory). This case is **under active design**: a DM producer-into-nothing is a gratuitous FIFO — a DM kernel can write its output tensor directly — so the likely resolution is an op-owner rewrite, but it is not finalized. (The fabricate-the-consumer trick *would* lower here, since a single-ended producer never fills beyond capacity and so never waits on the fiction — but this case has no settled end state, so we deliberately do **not** enshrine the hack.) Do **not** self-loop (broken on DM) and do **not** improvise.
- **DM kernel that *waits to receive*** (`wait_front` / `pop_front` with no real producer) → **STOP.** Here you'd have to fabricate the *producer*, and a fabricated producer never pushes, so the waiter hangs forever. No port-time workaround. (Rare-to-nonexistent — it would have hung in legacy too.)

> ⚠ **Cardinal safety rule.** Only ever fabricate the **consumer**, and only when the real kernel never blocks waiting on it — i.e. the CB drives **no FIFO synchronization at all**. Confirm zero `reserve_back` / `push_back` / `wait_front` / `pop_front` on it before applying the DM hack. **If you are not certain it is 100% pure scratchpad, STOP the port.** A blocked port is recoverable; a hardware deadlock is not.

**Correct port**:

```cpp
// (A) COMPUTE kernel, sync-free (e.g. a reciprocal LUT read by base pointer).
// Self-loop it on the one compute kernel:
KernelSpec compute{
    // ...
    .dfb_bindings = {
        DFBBinding{.dfb_spec_name = RECIP, .accessor_name = "recip", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = RECIP, .accessor_name = "recip", .endpoint_type = DFBEndpointType::CONSUMER},
    },
};

// (B) DM kernel, sync-free scratch. A self-loop would NOT lower on DM (disjoint
// masks are required), so bind PRODUCER on the real kernel and fabricate the
// CONSUMER on any OTHER kernel — which never touches the DFB:
KernelSpec reader{   // the kernel that actually needs the scratch
    .dfb_bindings = {
        DFBBinding{.dfb_spec_name = SCRATCH, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER},
    },
};
KernelSpec writer{   // any other co-resident kernel; this binding is pure paperwork
    .dfb_bindings = {
        DFBBinding{.dfb_spec_name = SCRATCH, .accessor_name = "scratch_unused", .endpoint_type = DFBEndpointType::CONSUMER},
    },
};

// Kernel side (both cases) — unchanged from legacy; base-pointer access, no FIFO ops:
experimental::DataflowBuffer dfb_scratch(dfb::scratch);
// ... read / write via base pointer as before ...
```

(Shared `accessor_name` for both endpoints relies on the per-kernel accessor-name dedup relaxation noted under [Self-loop DFB binding](metal2_port_patterns.md#pattern-self-loop-dfb-binding); the two-distinct-names form also works. For a **single-ended** CB that already has one genuine endpoint — e.g. the packer's producer — bind that one for real and fabricate only the missing side. Whether you can avoid fabricating at all turns on **one question: does any kernel other than the packer actually touch the CB?** If a co-resident kernel also accesses it — e.g. a DM kernel that reads the packed result to drain it elsewhere — bind *that* kernel as the consumer; it is a real endpoint, nothing fabricated. If the packer is the **sole** toucher — the usual shape for **sharded output**, where packed tiles land straight in the resident output shard and nothing drains them — there is no kernel to bind. **Here the kernel type decides** (per the Decision fork): a **compute** packer (the canonical case) → **self-loop** it; a **DM** sole-toucher producer → **STOP** (a DM self-loop doesn't lower, and you can't fabricate a consumer for a real producer without risking deadlock). Decide by reading the kernel *bodies* for a real access, not by kernel names: a `writer_*`-named kernel can be a weights- or activation-mover that never touches the output CB.)

**Document the hack prominently in the port report.** This is an *interim* workaround, not the intended end state. Record each fabricated-endpoint binding — a **compute self-loop**, or a **DM fabricated consumer** (naming the bystander kernel it was parked on) — in the report's [Open items for downstream](port_op_to_metal2_recipe.md#capture-the-port-report), stating plainly that the fabricated endpoint is a validator-satisfying device and **not** a genuine FIFO — so the eventual migration can find and replace every one.

**Classify it** (which kind it is, and where it's eventually headed — record this in the port report). Selected by the CB's **backing** first, then its access. The load-bearing rule: *borrowed-from-a-tensor backing is a tensor view, never scratch.*
- **Sync-free · regular-backed → kernel scratchpad.** The kernel reads and writes **private** (non-borrowed) L1 as scratch → the forthcoming **Metal 2.0 kernel scratchpad resource**.
- **Sync-free · borrowed · read-only → local `TensorAccessor`.** The kernel only *reads* a resident tensor's L1 by base pointer → a forthcoming read-only **"local" `TensorAccessor`** variant (still being designed).
- **Sync-free · borrowed · read-write → read-write local `TensorAccessor`.** The kernel reads *and* writes borrowed memory that **aliases a tensor** (e.g. an in-place accumulator on the output buffer) → the **read-write** form of that variant, delivered by the [compute-kernel `TensorBinding` fix](port_op_to_metal2_recipe.md#kernel-side-whitelist) (compute kernels can't bind a `TensorAccessor` today). A tensor view, *not* a scratchpad. Example: conv2d `MATMUL_PARTIALS` under `partials_cb_uses_output` (TILE output).
- **Synchronized · single-ended → an ordinary DFB, once its missing endpoint is bound.** This one is *not* sync-free — it's a genuine FIFO producer into resident output, missing only a consumer. Its end state is a normal borrowed-memory DFB (consumer bound to a real drain kernel, or the degenerate consumer accepted); it does **not** migrate to scratchpad/tensor-view. Example: the **compute** packer `OUT` above (→ self-loop interim). **Interim caveat:** on a **DM** kernel this case has no working interim — it **STOPs** (per the Decision), since a DM self-loop doesn't lower and a real producer can't take a fabricated consumer safely.

Until those land, the workarounds above are the sanctioned bridge — a **compute self-loop** or a **DM fabricated consumer** for the sync-free cases — with the **DM single-ended producer** the one case that STOPs instead.

**The sync verdict can flip per config — classify per instantiation, not per CB.** Whether a CB lands here (sync-free / single-ended) or is a genuine producer→consumer FIFO is *not* a fixed property of the CB; it can change across an op's configs. The same `buffer_index` can be a sync-free **scratchpad** (compute tilizes in place → self-loop) under one sharding and a **real FIFO** (a DM reader produces, compute consumes → ordinary DFB) under another. Re-run the litmus per code-path — one verdict applied across all configs mis-classifies the rest, and it flips on the **real/fake axis** (the Quasar-breakage predictor), so a miss here is the costly kind. **Canonical confuser:** conv2d `ACT_TILIZED` — height-sharded → sync-free scratchpad (self-loop); block/width-sharded → real FIFO.

**Orthogonal — SPSC.** Endpoint *multiplicity* is a separate check. A CB here that *also* has **2+ FIFO endpoints of one kind on a node** is an SPSC violation, **not** a self-loop case — it cannot be self-looped, and it's an op-owner pre-port fix. See [DFB endpoint legality](port_op_to_metal2_audit.md#dfb-endpoint-legality-spsc).

**See also**: [Self-loop DFB binding](metal2_port_patterns.md#pattern-self-loop-dfb-binding) (the legitimate accumulator case whose mechanism this borrows — there the producer/consumer do genuine work); [DFB endpoint legality](port_op_to_metal2_audit.md#dfb-endpoint-legality-spsc).
