# Empty-expert mailbox compaction contingency

Source-only audit, 2026-11-17. No implementation change, build, device test or latency claim. The candidate is worthwhile as a control-overhead experiment, but there is no evidence that it closes the remaining single-user latency deficit.

## What is actually avoidable

In `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/`:

- `moe_fused_swiglu_reader.cpp:434` broadcasts expert IDs, counts and offsets once per dispatch. Each worker already has the complete device-resident routing metadata. At `:503` it visits every local expert; `:565–627` applies ID/band/region validation and derives `m_t`/`m_blocks`. At `:640` it publishes each expert, including zero-block experts, to both local mailbox FIFOs.
- `moe_fused_swiglu_compute.cpp:524–552` consumes one local mailbox event per expert. UNPACK also sends two words to each of MATH and PACK; both threads receive them even for zero blocks.
- `moe_fused_swiglu_writer.cpp:267–283` consumes the second local FIFO event and builds accessors for every expert.
- Empty experts already execute **no M-block arithmetic, weight reads or collective rounds** in this bias-free Qwen path. Removing their local control turns cannot be advertised as removing collectives.

With one actual token and unique global top8 IDs, at most eight experts are active across four ranks, while the current loops visit 256 rank-local expert entries. A schedule-header design requires four headers plus eight expert publications across those ranks, instead of 256. This removes 244/256 = 95.3125% of these logical mailbox turns, repeated independently on each worker. This is a count of protocol events, not a bound or forecast for latency. Dense traffic adds one header per rank and can have no skipped experts.

## Concrete minimal protocol

Prefer one two-word active bitmap header, followed by the existing per-active-expert mailboxes. No new device tensor, host read, count refresh, sorting, weight packing or trace input is needed. Initially support at most64 local experts, no bias, deterministic rounding only; this exactly covers the selected Qwen grouped decode configuration. Reject incompatible explicit requests rather than silently changing behavior.

1. After the existing `COUNTS_BCAST` completes, the reader derives `mask[2]` from the existing validated runtime `m_blocks != 0` predicate. Keep the existing ID bound, active-token band, `M_T_MAX` clamp, aligned offset and capacity checks. Extract the reader’s current count/region calculation into one small inline helper used for both the header scan and ordinary expert body, so their predicates cannot diverge. A raw `counts != 0` test alone is insufficient for the shared op.
2. Reserve both existing mailbox CBs and publish a header using `MAILBOX_MAGIC + EXPERTS_PER_CHIP`, outside the range of existing per-expert magic values. Reuse `Mailbox.m_t` and `.m_blocks` as the two bitmap words **only for this distinct header**. Keep the existing payload-before-magic fence and FIFO push sequence.
3. Before its expert loop, UNPACK waits for the header, snapshots both words, broadcasts them once to MATH and PACK using the established two-word inter-TRISC sequence, then pops its FIFO. All three threads retain identical bitmaps in local scalar storage. The writer independently consumes and snapshots the same header through its own FIFO.
4. All three kernels retain `for (local_expert_id = 0; local_expert_id < EXPERTS_PER_CHIP; ++local_expert_id)` and immediately continue when the corresponding bit is clear. Use `(mask[id / 32] >> (id % 32)) & 1u`, avoiding shifts by32. The reader skip must precede weight-accessor construction and any bias traffic; the writer skip must precede weight-accessor construction and mailbox wait; compute skips before its ordinary mailbox wait.
5. Every selected expert executes the original body unchanged, with its **original** local expert ID, original `COMMON_WEIGHTS + local_expert_id` addressing, original mailbox magic, count, start-row and region. The schedule remains ascending. `gb`, `hfree_seq`, `hsend_seq`, `wd_done`, every block-relative counter, CB padding/rotation and arithmetic remain unchanged because the removed experts previously had zero M-blocks and zero `h_cursor`.
6. An all-empty rank publishes/consumes exactly the header and visits no expert bodies. It still runs the reader’s final `noc.async_write_barrier()` (`reader.cpp:1368`) that drains metadata multicast acknowledgements. The writer’s `out_pending` epilogue (`writer.cpp:642–646`) remains unconditional in control flow and drains the last output when present. Keep every existing read/write/atomic barrier inside the retained bodies. The header is not a reason to remove a barrier.

The mailbox CBs are aliased single-page handshake buffers (`moe_fused_swiglu_geometry.hpp:69–73`), so one extra fully consumed event wraps to the same base. Each consumer must snapshot the entire bitmap before popping; the reader reserves both consumer FIFOs before reusing the shared payload. Use the existing helpers rather than direct unguarded reads of the header words. No new shared mailbox fields, semaphores or L1 CB allocations are needed.

## Opt-in and build boundary

The existing runtime count mechanism is sufficient to implement the compact schedule **without changing the tensor/runtime-input API**. There is, however, no existing option that safely selects this protocol while preserving the default kernel for other callers. Reusing `stochastic_rounding=False`, a token band, a geometry coincidence or an unkeyed environment variable as the selector would conflate contracts.

Use an appended public keyword `compact_empty_experts=False` through `moe_fused_swiglu.hpp/.cpp`, the nanobind default, and `device/moe_fused_swiglu_types.hpp::OperationArguments`. Add it to both `attribute_names` and `attribute_values` so the program cache cannot mix protocols. Factory validation enforces the initial bounds and emits `COMPACT_EMPTY_EXPERTS=1` for **all three** kernel descriptors only when enabled. The default path must compile exactly its present protocol. This adds a native API keyword, but no plugin or vLLM API change.

The model-facing opt-in can be `QwenModel(..., compact_empty_moe_experts=False)`, propagated to decoder layers and passed only for deterministic grouped decode in `_grouped_moe` (`tt/multichip_decoder.py:1735`). Prefill remains on the default path. B2/B4/B32/solo0 signatures and sampler traces remain unchanged; cache keys select the correct compiled program. No scheduler-side Python branch depends on live expert counts.

This requires rebuilding the affected TTNN C++ operation and Python binding, followed by cold compilation of the three device kernels. A kernel-only source edit would avoid the host build only by giving up the explicit default-preserving selector; that is not the proposed experiment. No elapsed build estimate is justified by this audit. Existing hybrid routed-expert kernels are separate source copies, and must remain untouched; no shared mailbox layout or compile-argument order change is needed.

## Focused validation before adoption

First use a small native grouped-op exact discriminator, stochastic rounding disabled, comparing the original and compact protocols with identical weights, activation buffers and sentinel-initialized output buffers. Cover:

- all-empty ranks; one expert at local0,31,32,63; sparse separated experts; all64 experts; asymmetric rank populations;
- counts1,31,32 and larger legal multi-block/ragged counts to exercise block counters and CB realignment;
- unchanged metadata plus changed active bitmaps and original expert IDs/offsets on one retained trace, including active→empty→active;
- output bytes in valid computed regions and untouched sentinel regions for inactive experts, on every rank;
- reference and candidate programs alternated with the same tensor metadata to prove program-cache separation, and fresh output addresses to exercise runtime argument rebinding;
- tracker1/skip0, exact captured-output acknowledgements, nonblocking replay, final trace release, and repeated back-to-back dispatches to catch ACK leakage across kernels.

Then use the existing route/solo/recurrent conditional-commit reduced B2/B4/B32/solo0 test, all40 full-logit and serial96 controls, and real serving gates. Count-changing replay is mandatory: a fixed sparse warmed shape does not validate the schedule header. Measure matched single-user native and real-vLLM serving, plus mixed-shape and dense burst traffic. Report dense-header regression if present. Keep the context contract and sampling settings fixed.

Main risks are inter-TRISC header-count mismatch (hang), stale bitmap reuse at replay, accidental loss of original weight IDs, predicate divergence between scan/body, premature shared-mailbox reuse, and ACK leakage on all-empty ranks. Unchanged arithmetic is source-supported, but bit-exactness still requires hardware proof. The selected deterministic FPU rounding avoids a separate stochastic-stream scheduling concern. Bias is deliberately excluded from the first opt-in: its current fetch/pop protocol runs outside the M-block loop and requires extra coverage.

An archived implementation and focused probe were subsequently prepared as `empty_expert_schedule_candidate.patch` and `tests/run_empty_expert_schedule_probe.py`; see `empty_expert_schedule_candidate.md`. Production files remain untouched by that preparation. Compilation and protocol tests remain required; the design is not a hardware-proven result.
