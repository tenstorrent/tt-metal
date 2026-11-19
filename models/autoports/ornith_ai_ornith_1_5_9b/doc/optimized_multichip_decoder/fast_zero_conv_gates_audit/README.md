# Conditional fast-zero + precise conv-wide gate candidate

Prepared source-only. Run only after the precise-gate family passes its required gates. No devices were opened or kernels executed; no existing candidate, queued kernel, or test gate was edited.

New modules: `tests/multichip_fast_zero_conv_gates.py` (producer), `_recurrent.py` (consumer), `_candidate.py` (production decoder subclass), and `_probe.py` (unchanged strict direct harness with process-local module substitutions). New kernel directories: `tests/kernels/fast_zero_conv_gates` and `fast_zero_conv_gates_recurrent`.

## Merge contract

The producer helper is the precise conv-wide gate helper, copied with only its kernel directory changed. Its three kernels are byte-identical to `conv_gates_precise`. This preserves BF16 sigmoid arithmetic, FP32 gate intermediate unpack settings, supported ComputeConfigDescriptor field assignment, convolution arithmetic, packed projection source offsets, and persistent history advancement.

The recurrent helper is copied from `multichip_flat_ordered_conv_gates.py` with only its kernel directory changed. Its compute, normalization header, and writer are byte-identical to current production `tt/kernels/flat_ordered_post_loop`. There is no identity elimination, altered scan order, fidelity reduction, or postnorm change. FP32 state and wave104 remain the decoder default; the direct probe also tests wave7.

The recurrent reader starts from `fast_zero/reader.cpp`. Relative to that reader, only these source-address changes are made:

- g and beta now read source tile0, because the producer stores shapes `[1,1,B,8]`, rather than `[B,1,8]` with one tile per request.
- Extract scalar index `(batch/16)*512 + (batch%16)*16 + hv` from that tile. This is the element offset for rows0–31, cols0–7 in tiled face order; FP32 g and BF16 beta use their respective element pointer sizes.
- Packed z begins at tile64: `64 + hv*4 + t`, preserving columns2048–3071 of the original width3136 packed projection. The standalone gate reader's tile0 offset would be incorrect here.

Diffs against both source readers are saved next to this file. An exhaustive CPU index check covers every logical head at B1–32. All 4224 logical head addresses are within the batch-wide gate tile, and z addresses stay in tiles64–95. No padding is exposed as a logical request.

NoC zero fills cover reserved CB capacities only: Q/K/V and z each8192 bytes; g4096, beta2048, unit2048, eye4096. Each zero group is followed by `write_zeros_l1_barrier()` before overwrite or publication. Existing CPU fences before scalar/unit/eye publication are preserved. Producer/consumer ownership and full-capacity CB ring transactions are unchanged. `tt_metal/hw/inc/api/dataflow/noc.h:815–845` defines the local zero endpoint and required barrier. DRAM row reads retain aligned64-byte staging; L1 reads retain direct32-byte face rows.

## Execution artifacts and limits

`doc/optimized_multichip_decoder/fast_zero_conv_gates_jobs.json` contains four fresh-label jobs: strict direct Watcher probe, decoder B6/B32 16-step warmed/trace checks, and B32 4095-prefix/1024-answer test. Decoder jobs use the production packed-stack runner with only its decoder class replaced. The direct wrapper retains the existing TTNN convolution/gate references, history bitwise checks, and global/per-head .995 gates; its control recurrence explicitly uses the current production module. Thus the existing strict TTNN-convolution discrepancy remains visible and is not silently replaced with a more permissive control.

Parent source hashes were verified unchanged; new-file hashes and source-only checks are in `validation.json`. Python compilation passed. This is a conditional experiment, not a selected path, performance result, or quality pass.
