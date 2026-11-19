# CPU follow-up: probe contracts and final profiler plan

This review reads the current source and existing artifacts only. No TT device
command was run, and no decoder default or implementation file was changed.

## Probe findings

`tt/pipelined_striped_gather.py` preserves the original compile/runtime ABI,
TensorAccessor offsets, tile-element mapping, work assignment, and16KiB CB.
Each outstanding read uses a distinct64-byte aligned slot at
`12288 + slot*64`; batch64 ends exactly at16384. The source read stays inside
one2048/4096-byte source tile, and table addresses use TensorAccessor rather
than32-bit total-table byte arithmetic. Thus large table addresses retain the
original addressing contract. Invalid physical padding is never dereferenced,
the whole output tile is zero-initialized, and source/output scratch reuse is
protected by the corresponding read/write barriers. Empty batches skip the
read barrier. No new source-level bounds or layout defect was found.

The standalone lookup probe prepares all shape/dtype/kernel variants and stable
buffers before its first capture. It refreshes column/stripe inputs outside
capture, retains captured buffers and traces through shutdown, checks exact
payload bits per rank plus zero physical padding, and excludes program-cache
tracking exemptions. The CPU-only path passed earlier. This review did not run
the device path; real large-table PLE tests remain necessary because the small
synthetic table does not validate the device's large address mapping itself.

`tests/probe_multichip_fused_contract.py` has legal-looking initial shape/API
contracts, now supported by `fused_rs_qsa_v1.json`: native and fused RS complete
on QSA WO, all reported comparisons pass. That eager experiment measured
ordered-BF16 control1.09409ms, native RS1.26579ms, fused RS1.24674ms. It truly
keeps the640-wide result through residual write and distributed norm before
normalized GR input gathering. These results reject neither the unmeasured
AG-MM family nor a compatible reduced-precision family.

The new `--trace` path has three evidence limitations to handle explicitly:

1. It captures `inputs[0]` and never changes that stable input before replay.
   Eager comparisons use changed inputs, but this is not changed-input trace
   evidence. Prepare host-mapped variants before capture, copy each into
   `inputs[0]` outside capture, replay, and compare with its corresponding
   eager/reference output. Preserve independent timing with input copy outside
   the measured interval.
2. `plans/fused_families.json` deliberately sets allocation tracking0. This is
   clean timing evidence, not an allocation-tracked trace gate. A representative
   final candidate needs the identical path with tracking1 and program-cache
   tracking enabled. The probe currently records the flag but does not require
   it, so `passed:true` alone does not establish the tracked gate.
3. The class docstring and `persistence.ownership` still say no live traces.
   Actual behavior retains setup buffers during each trace, then releases that
   trace before the next candidate. Exact-tensor acknowledgments cover only
   the three capture-produced outputs. Record this one-live-trace invariant,
   and use exact eager-candidate parity as the replay check in addition to PCC
   against the control; changing arithmetic can legitimately differ from the
   ordered control, whereas replay must reproduce its own eager result.

These are missing trace evidence/metadata, not proof of a kernel error. No
whole-decoder trace gate should be waived by this screening probe. Its
activations remain seeded features, its fidelity option also changes the
control's fidelity, and both row/column weight layouts coexist only for the
one-layer experiment. Precision/default acceptance still comes from real
model activations and the integrated decoder.

## Final profile execution plan

`plans/final_profiles.json` contains three exact, bounded commands in the same
list-of-job format as the active stage plans. They use the complete1x4 mesh,
the current real layer0/1/3 references (`stage5_real_l{0,1,3}.pt`), logical33
prefill plus single-token decode, default decoder selection, and the existing
`process_multichip_profile --capture` adapter. Execute only after defaults are
selected and final source is formatted; no candidate monkeypatch is included.
Watcher and trace tracking are disabled for these profiling jobs. The adapter
preserves the raw capture and performs ordinary strict host/device joins after
re-exporting valid operation zones. It does not synthesize missing durations.

The existing `probe_multichip_decoder --profile` path warms prefill three times,
restores state outside signposts, uses `PERF_PREFILL`/`PERF_PREFILL_END` and
`PERF_DECODE`/`PERF_DECODE_END`, and uses the proven bounded profiler drains.
Run only one profile command at a time through the stage's serialized runner.
Each job has900-second timeout and its own provenance/console output. Check
capture exit status and all four device joins before generating tables. If
the capture overflows, preserve it and repair collection; do not bypass joins.

After those captures finish, the following is CPU-only:

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/final_profile_reports.py
```

The helper checks capture status, correctness/profile finiteness and four raw
device IDs. It requires one unambiguous raw CSV per capture directory, copies
that CSV into the stage artifact tree, produces advice-enabled human-readable
tables and separate CSV/console outputs for both phases, and records commands
and provenance. It also writes `final_operator_summary.json` with per-op
family sums, leading rows, dtype/fidelity and core counts. It does not suppress
advice or derive throughput from profiler-drain gaps. Raw CSVs may exceed commit
file-size limits; retain their on-disk paths and provenance. Use the separate
clean traced final-default harness for headline latency.

## What to look for in final advice and topology evidence

* Injection: verify selected K partition reduces the two original one-core
  matmuls without shifting more cost into layout or reduction. Include complete
  GR producer/consumer time and whole-layer/B32 effect.
* QSA: verify the actually executed ordered indexer is the new fused-valid-row
  implementation. Output-column sharding by itself decreases cores per chip
  and may leave each core's dot-product critical path unchanged; independent K
  blocks preserving the ordered combination are the measured stronger family.
  This qualifies the starting audit's speculative column-sharding priority.
* Lookup: verify the actual UINT32 striped-table GenericOp becomes faster and
  include unchanged untilization, scale decoding and PLE state costs. Batch32
  versus64 should be chosen by integrated latency/stress, not only queue depth.
* Collective topology: async/persistent ordered AG and BFP8 payload have
  whole-layer experiments; slower global policies do not reject selective
  subgroups or fused families. Complete AG-MM versus MM-RS under compatible
  residual/activation policy, or record the optimize stop-rule handoff.
* Native matmuls: the DRAM-sharded reader/grid/K-block experiments must use the
  selected dtype/fidelity, including lower-core/wider-shard geometry. Record
  exact padding/L1/API failures and retry material candidates before rejection.
* Shared gate/up: packed and split precision trials exist; inspect full
  gate/up/activation/down/collective cost before keeping the small local win.
* Precision: QSA HiFi2/LoFi trials exist; BFP4 is a candidate, not a new stage5
  default. Verify profiler rows match selected policies, and avoid using only
  synthetic topology-probe PCC to reject a faster real-weight candidate.
* Remaining untried candidates include reduced KV dtype (which triggers max
  context validation), packed-only indexed B1 MoE, and material policy/layout
  cross-products not covered by the current plans. Preserve grouped selected
  expert execution at diverse B32. If the stop rule fires, enumerate these as
  handoff notes, not completed experiments.

No clean-pass or completion verdict is asserted by this CPU follow-up.
