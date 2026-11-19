# MMRS dedicated-NoC repair candidate (source verified, device unverified)

2026-11-13. AutoFix inspection only. No source implementation, build, device,
TTNN execution, or reset was performed. This report is the only file written.
The existing diagnosis is strong enough to proceed without another AutoDebug.

## Result

Route the reader's W reads through its owned NoC0, leaving the BRISC writer on
NoC1 and retaining all writer completion checks. The current reader already
uses `w_read_noc` for both address generation and read issue; therefore this
change needs **no manual coordinate remap and no multicast/linked-packet
change**. A kernel-only JIT experiment can establish the result without a host
library rebuild. The clean factory-level repair requires a TTNN host rebuild.
Neither correctness on hardware nor a performance improvement is claimed yet.

## Current-source verification and earliest inconsistency

Inspected HEAD: `8502f42bca20dcad1507fbc8cebf7032220b7d8e`.
Paths below are relative to the checkout. Let `D` denote
`ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device`.

| File | SHA256 before any candidate patch |
|---|---|
| `D/matmul_reduce_scatter_prefill_program_factory.cpp` | `f5ec0087d726ead9d4df4f43b3093507bcb86a1132bf3b8610f18ac5fd00bd78` |
| `D/kernels/matmul_reduce_scatter_prefill_reader.cpp` | `6810c85603c84909c0ef68a290cc8e01978b0207aca682cefca3cde041d94a4d` |
| `D/kernels/matmul_reduce_scatter_prefill_writer.cpp` | `0ab01998677a1a2ac844b2e638baa7b4f4702dcd383802eff654c6bcc7a2f966` |

The factory hash matches the old AutoDebug report despite the changed HEAD.
The defect is still present:

1. Factory lines 525–536 set A reads to 0 and W reads to
   `(resident && !w_noc0_thin_wide) ? 1 : 0`. This W selector is the earliest
   inconsistent policy decision.
2. Factory lines 663–687 configure dedicated NCRISC/NoC0 reader and dedicated
   BRISC/NoC1 writer. `DataMovementConfigDescriptor` defaults to dedicated mode
   in `tt_metal/api/tt-metalium/program_descriptors.hpp:98`.
3. Reader lines 114–119 still claim ownership of every read on both NoCs. This
   comment overlooks the other RISC's startup snapshot and exit checks.
4. `tt_metal/hw/firmware/src/tt-1xx/brisck.cc:66` snapshots the writer's NoC1
   hardware counters; lines 85–96 check them after `kernel_main()`.
   `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:587`
   increments the issuing RISC's private counter, while lines 599–600 compare
   that private counter to the shared hardware read-response count.
5. Reader NoC1 W reads therefore change the writer's hardware count without
   changing the writer's private count. The reader's per-step transaction-ID
   barriers at lines 468–472 can complete successfully while the writer exit
   assertion still fails. Adding a writer read barrier would wait on the same
   inconsistent equality; resetting counters would hide rather than repair it.

The current model still partitions native calls at 128 rows in
`tt/multichip_decoder.py:429–454`; selecting more than 8192 logical rows takes
the separate path. Thus merely running a large public B/S shape does not test
the native tall-grid repair.

## Candidate A: smallest JIT-only discriminating patch

Apply only this change in `D/kernels/matmul_reduce_scatter_prefill_reader.cpp`
at the current lines 114–118. This is the **experiment**, not a claim that the
factory's advertised W NoC has been changed:

```diff
-    // NoC of each operand's DRAM reads. This kernel owns every read on both NoCs of its core (the writer only writes),
-    // so the second NoC adds DRAM-response links for the fill
+    // Dedicated ownership: BRISC owns NoC1, including its cumulative exit counters.
+    // Keep both operands on the reader's NoC until the host policy is repaired.
     constexpr uint8_t a_read_noc = get_compile_time_arg_val(ct::A_READ_NOC);
-    constexpr uint8_t w_read_noc = get_compile_time_arg_val(ct::W_READ_NOC);
+    constexpr uint8_t w_read_noc = noc_index;
+    static_assert(a_read_noc == noc_index && noc_index == 0);
```

The following existing `two_read_nocs` expression becomes false. The existing
transaction-ID setup and per-step barriers then operate only on NoC0, covering
both operands with the same ID. No read, write, atomic, exit-counter, semaphore,
or source-lifetime protection is removed.

This changes a runtime-compiled device source only. A new process and fresh
`TT_METAL_CACHE` must be used for the experiment; no `_ttnn`/`_ttnncpp` rebuild
is needed. Record the edited reader hash and successful compilation evidence.
Do not treat a same-path kernel identifier as proof of a rebuild:
`tt_metal/impl/kernels/kernel.cpp:882` hashes the supplied source path, with
dependency tracking handled separately. A fresh cache avoids ambiguity.

The unused legacy W_READ_NOC compile argument may still equal 1. In particular,
`tests/probe_multichip_prefill.py:195–216` infers the old W NoC from geometry;
its `inferred_weight_read_noc` is **not actual routing evidence** after this
kernel-only patch. Attach the effective source policy to experiment provenance.

## Candidate B: canonical factory repair after the hypothesis passes

In the factory, replace the explanatory thin-wide/NoC1 policy block at lines
526–536 with:

```cpp
    // Each dedicated DM kernel owns its NoC's cumulative completion counters.
    // The NoC0 reader cannot issue reads on the NoC1 writer's interface.
    reader_ct[args::reader_ct::W_READ_NOC] = 0;
```

Keep `A_READ_NOC = 0`. Remove the now-unused `w_noc0_thin_wide` local. Restore
the reader's `w_read_noc = get_compile_time_arg_val(ct::W_READ_NOC)` and replace
its incorrect ownership comment with the dedicated-ownership explanation.
Optionally retain a static assertion that both operand NoCs equal the reader's
NoC to prevent silently reintroducing the defect. Do not apply both variants
and then credit the kernel override as proof that the new host policy loaded.

Factory changes require recompiling/linking the TTNN host library and starting
a fresh process. The existing `build` resolves to `build_Release`; its Ninja
file exposes target `ttnn` (`cmake --build build_Release --target ttnn`). This
command was not run. Verify that the executing Python environment loads the
rebuilt extension, then run with a new kernel cache. A factory-only source edit
with an old loaded host library cannot fix routing. No firmware source change
or whole-device firmware ownership change is needed.

## Why address mapping and linked packets already work

Reader lines 331–336 call both
`w_acc.get_noc_addr(page, 0, w_read_noc)` and
`noc_async_read(..., w_read_noc)`. The NoC choice is propagated together;
changing only the final argument of `noc_async_read` would be wrong, but the
constant change above changes both uses.

For these DRAM-interleaved tensors, the path is
`api/tensor/tensor_accessor.h:554,652` →
`internal/dataflow/dataflow_api_addrgen.h:299` →
`internal/dataflow/noc_address_backend_xy.h:50–58`, where
`dram_bank_to_noc_xy[noc][bank_index]` selects the proper coordinates. No
NoC1-encoded DRAM address is reused on NoC0. A and W page arithmetic is
unchanged. Do not manually invert the coordinates a second time.

Factory line 755 already attaches all A/W multicast families on NoC0. Neither
their host-generated rectangles nor receiver/sender coordinates change.
`ttnn/cpp/ttnn/kernel_lib/mcast/kernel/pipes/mcast_pipe.inl:144–148` calls
`send_data_` immediately followed by `signal_ready_`; lines 240–248 link Flag
payload writes, and lines 285–288 terminate the chain with an unlinked ready
write. Therefore no linked chain is left open across the next reader issue or
poll. The existing beginning-of-step `noc_async_writes_flushed()` and final
write/atomic barriers remain. Sharing read and multicast traffic on a NoC
owned by the same RISC is already used by thin-wide/stream plans; it does not
create the cross-RISC cumulative-counter defect.

This source argument does not promise equal contention or latency on tall
grids. Those are the measurements the larger-row candidate must provide.

## Discriminating hardware probe for the parent

Run after the parent's current profiler job ends and under its established
device-usage/hang-capture workflow. Use the existing direct-native topology
harness: its `fused()` calls native MMRS once at the requested M, bypassing the
model's 128-row partition. The `separate()` control disables fused MMRS. It
checks a sharded residual-add/norm consumer, eager PCC >=0.995, finite output,
first replay and 50 queued trace replays. It tests the native boundary, not all
decoder gates.

From the repo root, in the parent's established lane environment:

```bash
mmrs_probe_cache=$(mktemp -d /tmp/qwen-mmrs-noc0-attention-XXXXXX)
env -u TT_METAL_DEVICE_PROFILER \
  TT_METAL_WATCHER=10 TT_METAL_CACHE="$mmrs_probe_cache" \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_topology \
  --family mmrs --layer 0 --role attention --length 1056 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/mmrs_noc0_attention_m1056.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/mmrs_noc0_attention_m1056.log 2>&1
```

Repeat serially with `--role mlp`, a new `/tmp/qwen-mmrs-noc0-mlp-XXXXXX`
cache, and distinct `mmrs_noc0_mlp_m1056.{json,log}` artifacts. The two local K
widths are 1536 and 4352, covering the reported resident weight-read branches.
Preserve watcher log/kernel names and actual native plan/compiled arguments;
require a resident, non-thin-wide plan before calling this a discrimination of
the old defect. Live L1 availability can change the plan. Retain original
known-failing watcher artifacts as the control rather than unnecessarily
recreating a crash. If a fresh failure occurs, capture its live evidence before
the parent performs any termination/reset.

Do not use `probe_multichip_prefill --mmrs-chunk-rows 1056` alone to claim this
test: it wraps each already-128-row model call and cannot combine those calls
into 1056 rows. After native passes, the parent must explicitly adapt the
model's outer row partition (candidate-only initially), then run the normal
decoder oracle for both layer kinds and awkward shapes before selecting it.

## Remaining risks and decision rule

- NoC0 contention can make a single tall native call slower than another
  chunk size. Existing factory comments record up to 1.59x slower NoC0 on some
  M-on-X shapes. Compare larger chunks, whole-M native and separate MM+RS using
  the same final consumer, after watcher passes; this historical range is not
  a measurement on this model.
- Watcher-clean boundary results do not establish all-layer integration,
  per-user/state accuracy, resource pressure, arbitrary tails, or changed-input
  trace safety. Preserve logical flatten/pad/crop order and test actual B32/S33,
  awkward tails and both meaningful layer kinds. Recheck 64-owner pressure
  because native planning depends on free L1. The old fixed 128-row restriction
  must only be removed once the new path has that evidence.
- Retaining two read NoCs through dynamic mode is a separate alternative:
  both matmul DM kernels must use dynamic mode and the relevant APIs must be
  audited. That requires factory rebuild and expands the change. It is not
  necessary to test the ownership repair or remove repeated tiny-row dispatch.
- Do not add a writer read barrier, reset its counters, disable watcher, or
  change linked-packet semantics as a first intervention. None repairs the
  earliest ownership inconsistency as directly as routing W reads to NoC0.

Decision: pursue Candidate A first as the cheapest discriminating experiment;
if watcher and native correctness pass, evaluate larger-row performance, then
land the canonical factory policy with matching host-build provenance and
integrated decoder validation. Report hardware claims only from those runs.
