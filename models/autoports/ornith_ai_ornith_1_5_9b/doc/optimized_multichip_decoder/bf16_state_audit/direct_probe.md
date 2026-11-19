# Quantized-control direct probe

Prepared source only; no hardware execution. New files:

- `tests/multichip_bf16_state_staged_probe.py`
- `doc/optimized_multichip_decoder/bf16_state_staged_direct_jobs.json`

Run the job JSON through the existing serialized root runner. Its command is:

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf16_state_staged_probe --steps 8 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/bf16_state_staged_direct_watcher.json
```

The job sets Watcher=10, NOINLINE=1, APPEND=1, and allows 1800 seconds. No profiler is enabled. A longer follow-up may use `--steps 32` with a fresh output path and suitably bounded runner timeout.

## Exact comparison and rounding timing

Three independently allocated states start from identical CPU BF16-rounded values: candidate BF16 state, FP32 quantized-control state, and FP32 unquantized-control state. Both controls call the current production `tt.flat_ordered_post_loop`, not a historical test candidate. All paths receive the exact same device tensors for conv/g/beta/gamma/gate.

For each step, run candidate, quantized control, and unquantized control. Each produces output from its prior state and then updates its own state. Only after producing that step's output, typecast the quantized control's updated state to BF16, widen it to FP32, and copy it back into that control's existing persistent allocation. This is the next token's initial state. Candidate state is never used to seed either control. The unquantized control receives no further state rounding after initialization.

Require finite values and global/per-head PCC >= .995 for candidate output versus quantized control, and for candidate BF16 state versus rounded control state. Also require bitwise equality for that state comparison: the recurrence/order is unchanged and the expected difference is only the final BF16 state pack. A failed bitwise check is preserved and stops the probe; there is no tolerance override. The BF16→FP32 restoration is separately required to be bitwise exact. Zero/constant unequal tensors fail. PCC uses double accumulation to avoid correlation underflow on tiny heads.

Separate metrics report candidate and quantized-control output/state divergence from the never-requantized FP32 control. These are explicitly precision-loss diagnostics, not a substitute for unchanged decoder/PCC acceptance gates. A pass here only supports correctness of the BF16 implementation, not acceptance of its precision policy.

## Coverage and changes from the previous postnorm probe

Defaults: B1,2,6,13,14,16,17,26,27,31,32; waves104/7; eight steps. Reset RNG to 7905+B for each wave so narrow-wave CB reuse receives the same input sequence. Persistent state lives in DRAM for the first wave and L1 for the second. Inputs alternate DRAM/L1 every step. Inputs include signed random initial states, signed gamma/gates, zero/tiny QK, varying decay, and beta lanes 0, 1e-12, 1e-6, 1. Every rank of the 1×4 mesh is checked. Wave7 crosses head/request boundaries and exercises repeated full-capacity CB reuse. Direct B1 exercises actual BF16 state, unlike the decoder candidate's B1 private-state path.

Compared with the earlier postnorm probe, replace separate normalization/Torch controls with production post-loop; add explicit state rounding between steps and a third unquantized control; gate each head; add narrow waves, source hashes including shared scan, independent allocation assertion, and failure-only tensor artifacts. Device operations remain inside `device_only()`; upload/download are outside as in existing harnesses. No trace or subset/pool isolation behavior is claimed by this direct probe.

Python compilation passed. The audit's original queued-helper/kernel SHA256 manifest was rechecked unchanged. Hardware execution and numerical acceptance remain pending root's serialized job.
