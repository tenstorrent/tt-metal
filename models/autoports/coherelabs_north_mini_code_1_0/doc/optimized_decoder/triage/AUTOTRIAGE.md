# AUTOTRIAGE

## Diagnosis

- The North-Mini failure is caused by an invalid sparse 1D matmul program configuration reaching the device: the model fixes `out_block_w=1` but the failing sweep sets `out_subblock_w=3` (the second reproduction is the same contract violation with `out_subblock_w=2`). The sparse operation does not apply the dense matmul block/subblock validator before dispatch. Integer division then makes `in1_num_subblocks = 1 / 3 = 0`, so the compute kernel consumes no in1 subblocks while the dataflow kernels and host runtime arguments retain nonzero output-subblock geometry. Their CB/loop protocol diverges and dispatch never completes.

## Triage Evidence

- Exact live reproduction:

  ```text
  python_env/bin/python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
    --mode decode --batch 1 --layer 1 --warmups 1 --iterations 1 \
    --sparse-gate-up-out-subblock-w 3
  ```

  It remained alive without output for more than 50 seconds; the retained `1/1` configuration completes in the existing candidate evidence.
- The live capture is preserved at
  `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/triage/tt-triage.txt.gz`
  and its generated summary at
  `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/triage/triage-summary.txt`.
- Triage directly proves that device 2 remained responsive: ARC heartbeats were approximately 10/s, telemetry reported 1350 MHz AI clock and 48 C, and the board was a Blackhole p300. It does **not** provide usable RISC call stacks, running-op aggregation, CB state, or NoC counters on this installation. Those readers all skipped because `tt-triage` passed a `memoryview` to the installed `tt_umd 0.9.5` `noc_read` overload, which rejects that signature. Consequently, the generated all-`pass` summary is not an all-clear and must not be interpreted as one.
- After preserving evidence, only the exact reproducer PID was terminated. `python_env/bin/tt-smi -ls --local` then returned successfully and listed all four p300c boards.

## Source Evidence

- The North helper in
  `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:574-587`
  accepts arbitrary `out_subblock_w`, while independently fixing `out_block_w=1`.
  The failing CLI value is forwarded unchanged at lines 612-625.
- The generic dense matmul validator in
  `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:119-165`
  expressly requires `out_block_w % out_subblock_w == 0`; it would reject both
  `1/3` and `1/2`. Sparse matmul builds dense attributes, but its own
  `SparseMatmulDeviceOperation::validate_on_program_cache_miss` does not call
  this validator or duplicate that contract.
- In the sparse factory,
  `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:477-501`,
  the bad `1/3` geometry produces this ledger:

  | Participant | Derived count |
  | --- | ---: |
  | in1 compute subblocks | `out_block_w / out_subblock_w = 0` |
  | in1 block tiles presented to compute | `3 * in0_block_w * 0 = 0` |
  | output tiles per compute subblock | `1 * 3 = 3` |
  | output/accumulator CB capacity | `out_block_h * out_block_w = 1` tile |

  Thus the compute-side subblock and CB geometry cannot describe the same
  transaction.
- The writer runtime arguments at factory lines 631-641 and 710-735 likewise
  combine a zero full-subblock count (`1/3`) with one nonzero tail subblock and
  `last_subblock_of_last_block_w=1`. This explains why the device kernels can
  wait rather than returning a host API error.
- The passing `out_subblock_w=1` contrast satisfies the same equations:
  `1/1=1`, one in1 subblock, one output tile, and one-tile CB capacity.

## Downstream Effects

- The host waiting for command completion and the need to terminate the
  benchmark are downstream effects of the inconsistent dataflow/compute loop
  counts. Device responsiveness and normal ARC heartbeats rule out ARC,
  Ethernet, temperature, or board discovery as the initiating failure.
- Because the installed triage reader cannot access live RISC/CB state, this
  capture cannot identify the first individual kernel waiter. The source
  contract and the exact passing/failing geometry isolate the earlier host-side
  defect independently of that missing stop-site detail.

## Proposed Fix

- Smallest correctness fix: validate sparse matmul's normalized 1D program
  configuration on the host before program creation, at minimum enforcing
  nonzero block/subblock dimensions and
  `out_block_{h,w} % out_subblock_{h,w} == 0`, plus the existing
  per-core/block divisibility rules used by dense matmul.
- Smallest model-stage workaround: do not sweep an output subblock wider than
  the selected output block. With North's current `out_block_w=1`, retain
  `out_subblock_w=1` for both sparse projections.
- Smallest verify experiment: add the sparse host validation, then rerun the
  exact `--sparse-gate-up-out-subblock-w 3` command under a short timeout. It
  should raise the explicit `out_block_w (1) must be divisible by
  out_subblock_w (3)` error before dispatch; immediately rerun the `1/1`
  command to prove the valid path still completes.

## Uncertainty

- The precise first blocked RISC and CB cannot be stated because all relevant
  `tt-triage` readers were skipped by the `tt_umd` API mismatch. A watcher
  reproduction would likely turn the protocol mismatch into a louder device
  assertion, but is unnecessary to justify rejecting an arithmetically invalid
  program configuration and would incur another intentional device hang.
- A wider output subblock may be legal only if `out_block_w` is increased to a
  compatible multiple and all CB/L1 constraints are revalidated. That is a
  separate geometry experiment; changing `out_subblock_w` alone is invalid.
