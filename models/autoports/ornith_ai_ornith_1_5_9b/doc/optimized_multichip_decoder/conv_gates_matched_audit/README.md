# Matched production-convolution direct probe

Fresh `tests/multichip_conv_gates_matched_probe.py` and `doc/optimized_multichip_decoder/conv_gates_precise_matched_jobs.json`; no hardware run. Original `multichip_conv_gates_probe.py`, precise wrapper, and prior results remain unchanged.

The main control uses current `tt.tiled_conv` with independent persistent history, the existing separate TTNN sigmoid/add/softplus/multiply gate calculations, and current `tt.flat_ordered_post_loop` with independent FP32 recurrent state. The candidate uses precise conv-wide gates and its flat gate reader/recurrence. Both receive identical packed inputs, taps, bias, decay, gamma. All public batches and wave/step defaults match the original strict probe.

The separate-TTNN convolution calculation and concat/copy history control are retained as an additional independent path. Each step reports and gates candidate conv versus production conv (bitwise), production conv versus separate TTNN conv (.995), candidate conv versus separate TTNN conv (.995), and candidate conv versus Torch (.995). Candidate, production, and TTNN history must agree bitwise. No convolution discrepancy is silently discarded.

A third recurrent state follows production conv with the **candidate's exact g/beta values**, reshaped through TTNN into the production reader contract. This branch is separate across the full sequence and must match candidate state bitwise and candidate output at global/per-head PCC>=.995. It isolates the reader/layout/recurrence from gate numerical differences. The main separate-gate production branch retains global/per-head .995 state/output gates; candidate beta/g comparisons with separate TTNN remain required. The identical-input branch never seeds either other state.

This is a targeted reference-contract experiment: the current accepted convolution rounding is held fixed while gate fusion changes. Passing it does not erase the separate-TTNN-conv end-to-end failure or establish decoder accuracy. Both original and matched results must be retained, and unchanged long decoder quality gates remain required.

Python compilation passed; source hashes are in `validation.json`. The new Watcher job is ready for root's serialized runner and has not been launched.
