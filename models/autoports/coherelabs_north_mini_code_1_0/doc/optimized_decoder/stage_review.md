# Independent stage review

## Initial verdict

`more-work-needed`

The fresh xhigh reviewer identified six blocking classes:

1. The documented MoE sharding rejection was not reproducible from the
   then-current dense-only policy guard.
2. Static sparse `nnz=8` used routing values as its mask and did not prove
   eight nonzero entries under sigmoid underflow.
3. Final dense topology lacked precision-locked K-block/fidelity evidence,
   and retained MoE movement lacked a concrete adapted-path blocker.
4. Dominant all-expert prefill matmuls had not been attacked.
5. Selected real-weight coverage was missing for layers 0 and 4.
6. Context/review state and wall/device profiler accounting were ambiguous.

## Remediation

- Added `dram_sharded_moe_attention`; measured actual BFP8 DRAM-sharded QKV/O
  at 0.567961 ms versus the selected 0.543472 ms.
- Created an exact mask by scattering ones separately from routing scores.
  Twenty zero-score trace replays pass PCC 1.0 under watcher.
- Crossed the final dense topology with K-block cap 4, BFP8/LoFi, and
  BFP4/LoFi. Selected 0.173888 ms BFP8/LoFi at real PCC 0.999259; rejected
  0.158435 ms BFP4 at PCC 0.960960.
- Documented fixed 8-bank geometry and the DRAM-sharded program API's lack of
  output-subblock controls.
- Tried sparse output tile heights 1 and 16. Both hit the exact kernel
  requirement that output height equal the 32-row in0 tile.
- Tried dominant prefill 48/64 and 24/32 programs. The former cannot fit the
  legal device/M-tile grid; the latter is faster but fails PCC at 0.968541.
- Added real layer-0 and layer-4 prefill plus cache-consuming traced decode.
  All selected paths exceed PCC 0.9992.
- Corrected context stage provenance and reconciled headline wall, same-run
  profiler wall, device duration, op gaps, and profiler overhead.
- Reran exact-final context probes, profiler reports, normal suite (21/21),
  and watcher suite (21/21, clean).

## First rereview

The fresh xhigh rereviewer cleared the original findings and returned three
remaining evidence gaps:

1. The faster final-topology dense BFP4 rejection needed real layer-0
   prefill and cache-consuming trace evidence.
2. The sparse movement trials changed only output tile height instead of
   testing a matched input/output L1 chain.
3. The dominant prefill sweep held K block at 2 and therefore did not cover
   larger legal K blocks.

All three are now remediated:

- Real layer-0 BFP4 prefill/decode PCC is 0.966624/0.960101, below 0.995.
- A matched 32x32 L1 chain improves layer-4 decode to 0.530140 ms and
  preserves real prefill/decode PCC 0.999650/0.999604. Source inspection also
  proves that public device retile APIs cannot produce matched 1x32/16x32
  tiles without an out-of-scope kernel.
- Legal 24/32-core prefill K blocks 4, 8, and 16/12 were measured. K block 8
  is fastest at 8.003327 ms and passes non-aligned seq-1025 PCC 0.999516.

The combined final revision passed 21/21 normally and 21/21 under watcher,
and fresh MoE Tracy reports were collected.

## Final rereview

`clean-pass`

No required work remains. The fresh xhigh reviewer independently inspected
the exact current code, tests, docs, context contract, real-weight candidate
logs, final4 profiler compressed-raw/filtered/human artifacts, exact-final
JUnit, and compressed raw watcher log without editing files or using
hardware.

Controlled anomalies:

- Faster dense BFP4 is invalid on real target evidence: prefill/decode PCC
  0.966624/0.960101.
- Smaller sparse tiles have an exact 32-row input-match/API blocker; the
  adapted matched 32x32 L1 chain is correct and faster.
- Layer-4 router output remains DRAM because L1 changes a near-boundary top-8
  selection.
- Initial stale profiler text artifacts were replaced with the internally
  consistent final4 bundle before this verdict.

Residual risk is limited to dense Tracy predating the MoE/prefill-only final
remediation; code inspection plus fresh dense performance, correctness, and
watcher runs confirm the dense runtime was unchanged.
