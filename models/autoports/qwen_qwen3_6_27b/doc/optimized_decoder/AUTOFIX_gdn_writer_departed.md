# Defer remote-write completion while preserving source-buffer ownership

The recurrence already has two heads of input/output buffering. Profiler163's
reader completes around 39 us before the writer, and the writer finishes only
about 3.4 us after the pack thread. These totals do not justify deeper buffers
or a claim of a large writer bottleneck. Splitting every head into more writer
chunks would add barriers and may merely reduce startup/drain time.

A narrower unnecessary dependency is explicit in the source: the writer waits
for remote DRAM acknowledgements before releasing every head's local output
buffers. `Noc::async_writes_flushed()` waits until all writes have departed,
which is sufficient to reuse their local source pages. The low-level function
waits for `ncrisc_noc_nonposted_writes_sent`; the stronger completion barrier
waits for `ncrisc_noc_nonposted_writes_flushed`. Both invalidate the local cache
after waiting. See `api/dataflow/noc.h` and `api/dataflow/dataflow_api.h`.

The candidate uses departure flush before each nonfinal head's CB pops and keeps
the original completion barrier on the final head. Its proof obligations are:

- A core reads and writes each global state's head exactly once during a call.
  Later heads never depend on earlier heads' persistent state.
- State pages are disjoint across heads. Output tile pages can be shared, but
  their written 64-byte row slices are disjoint; padding writes target unused
  rows only.
- Local source pages are released only after all payloads depart. Pending
  remote acknowledgements do not retain local CB ownership.
- The final completion barrier covers every outstanding nonposted write before
  the kernel returns and a later invocation can read persistent state/output.
- Single-head cores take the original completion path unchanged.

This changes no address, transfer count, NoC selection, CB size or computation.
It removes up to 13 per-core remote-completion dependencies at B32. Existing
double buffering may hide most or all of those waits, so this is a bounded
measurement candidate rather than a predicted material gain.

Frozen artifacts, based on the exact-probe-passing delta helper:

- `candidates/gdn_sfpu_writer_departed.py`
- `candidates/integrated_writer_departed134.py`
- `candidates/make_gdn_writer_departed_candidate.py`
- `candidates/check_gdn_writer_departed_cpu.py`
- `candidates/gdn_writer_departed_cpu.json`

CPU/source checks passed: the complete helper matches its parent except the
writer barrier replacement, B1–32 output slices are disjoint, and 512 delayed-ack
ring simulations preserve all payloads. A deliberately unsafe early-pop control
corrupts data in all 512 cases. Python compilation passed. No hardware or
main/shared source was touched. Device correctness and latency remain unverified.

Parent probe: `run_recurrence_exact_comparison.py` with candidate
`gdn_sfpu_writer_departed.py`, reference `gdn_sfpu_fused_delta_row.py` and
`--state-dtypes bf16 fp32`. Also use `--batches 3 --cores 1` to exercise 144
heads through repeated ring reuse and final acknowledgement draining. Require
exact state/output bits, then compare whole B32 short/512 and B1 with the parent.
Reject the candidate if warmed latency does not improve consistently.
