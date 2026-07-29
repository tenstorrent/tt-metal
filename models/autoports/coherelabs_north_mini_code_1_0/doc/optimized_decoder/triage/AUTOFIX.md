# AutoFix Report

## Starting Evidence

- Source: `AUTOTRIAGE.md` and the live capture under
  `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/triage/`.
- Original failure:

  ```text
  python_env/bin/python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
    --mode decode --batch 1 --layer 1 --warmups 1 --iterations 1 \
    --sparse-gate-up-out-subblock-w 3
  ```

  This remained stuck beyond 50 seconds. The analogous
  `--sparse-down-out-subblock-w 2` also reproduced previously.

## Hypothesis Experiments

- Hypothesis: North's fixed `out_block_w=1` cannot support output subblocks 2
  or 3 tiles wide; sparse matmul misses the dense host validation and dispatches
  the invalid geometry.
- Experiment: call `OptimizedDecoder._sparse_matmul_program` directly with
  gate/up subblock 3 and down subblock 2, without opening a device.
- Result before fix: AutoTriage proved both values were accepted and the gate
  command hung on device. Source lowering produced zero in1 subblocks from
  integer division while retaining nonzero writer/tail geometry.
- Verdict: verified.
- Fix: `optimized_decoder.py` now rejects nonpositive output subblocks and
  values that exceed or do not divide its fixed `out_block_w=1`. Shared TTNN
  core code was not changed. A device-independent regression test covers 0, 2,
  and 3.
- Verification:

  ```text
  python_env/bin/python -m pytest -q \
    models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py::test_sparse_program_rejects_invalid_output_subblocks_before_device
  ```

  Result: `3 passed in 1.05s`.

  Direct gate=3 and down=2 constructor probes both raised:

  ```text
  sparse out_subblock_w (...) must be positive, no greater than, and divide out_block_w (1)
  ```

  The retained hardware control completed and closed its mesh normally:

  ```text
  timeout 120 python_env/bin/python \
    models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
    --mode decode --batch 1 --layer 1 --warmups 1 --iterations 1 \
    --sparse-gate-up-out-subblock-w 1 --sparse-down-out-subblock-w 1
  ```

  Result: exit 0, mean/min `0.882483 ms`.

## Final Status

- Fixed in the model-owned sparse program builder. Invalid candidates now fail
  before device dispatch; the retained valid candidate still runs.
- The valid benchmark closed device drivers cleanly. A subsequent bounded
  `python_env/bin/tt-smi -ls --local` listed all four p300c boards. No reset or
  stale-process cleanup was needed after verification.
- Remaining framework issue: shared sparse matmul should eventually apply the
  same block/subblock validation as dense matmul, but patching shared TTNN core
  is outside this stage and unnecessary for the model-side repair.
