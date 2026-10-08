# BFP4 numerical check using ttsim

The CPU simulator completed 60 matmuls on ten actual Qwen weight submatrices.
All results were finite and the single virtual device closed cleanly. **This
measures sampled projection error, not full-model or GPQA accuracy.** The model's
precision and physical Galaxy experiments were unchanged.

The observed BFP4 error is dominated by quantization in these samples. Increasing
matmul fidelity reduces execution error but hardly changes total BFP4 error.
The small decrease in total BFP4 error with LoFi is an interaction of errors;
it is not evidence that LoFi has better model accuracy.

## Results

Values are medians across ten submatrix cases, in percent. Relative RMS is
`norm(actual - reference) / norm(reference)`. The columns have different
references and are not additive; see the definitions below.

| Weight format | Fidelity | Weight quantization | Output from quantization alone | Execution on quantized operands | Total versus checkpoint |
|---|---|---:|---:|---:|---:|
| BFP4 | LoFi | 11.611% | 11.803% | 0.439% | 11.793% |
| BFP4 | HiFi4 | 11.611% | 11.803% | 0.167% | 11.804% |
| BFP8 | LoFi | 0.757% | 0.762% | 1.888% | 2.032% |
| BFP8 | HiFi4 | 0.757% | 0.762% | 0.168% | 0.782% |
| BF16 | LoFi | 0.000% | 0.000% | 2.623% | 2.623% |
| BF16 | HiFi4 | 0.000% | 0.000% | 0.169% | 0.169% |

BFP4/LoFi total output error ranged **11.25%-12.57%** across slices; the largest
individual input-row error was **15.50%**. Its execution error against already
quantized operands ranged **0.433%-0.448%**. These are relative tensor errors,
not percentages of wrong tokens or task-score losses.

The BF16/LoFi control also had **2.623%** median execution error, versus **0.169%**
with HiFi4. BFP8/LoFi had **2.032%** median total error, versus **0.782%** with
HiFi4. Format and fidelity therefore need separate controls; do not infer model
quality from a format name alone. This experiment does not establish a kernel
bug, hardware equivalence, or the cause of the earlier GPQA score gap.

## Method and limits

- Use pinned checkpoint revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
  Sample two aligned blocks of 128 output channels from each of layer-0
  linear-attention QKV, layer-3 full-attention Q, layer-31 MLP down, layer-63
  MLP gate, and LM head. Preserve complete 16-value exponent-sharing groups.
- Use full K=5120 except the down projection, which uses K=4352, one TP4 input
  shard. Compute M=32 independent seeded BF16 Gaussian input rows. These inputs
  are **not real captured activations**. No scale calibration, clipping or
  weight changes are applied before standard TTNN conversion.
- Run each slice with BFP4/BFP8/BF16 and LoFi/HiFi4. All cases retain BF16
  activations/output, FP32 destination accumulation, accurate math configuration
  and L1 packer accumulation. This uses interleaved small-submatrix matmul, not
  the full model's DRAM-sharded projection/collective path.
- Round-trip device weights to establish the actual TTNN-quantized operands.
  Host float32 references separate four errors: `Wq` versus checkpoint `W`;
  `A @ Wq` versus `A @ W`; simulator output versus `A @ Wq`; simulator output
  versus `A @ W`. Error metrics use float64 reductions.
- Record per-case input/weight hashes, probe hash, checkpoint-index hash and
  pinned simulator/native revisions. Raw data includes PCC, maximum absolute
  error, mean signed error and worst-row relative RMS, alongside total RMS.
- Require finite outputs and successful cleanup. There is no arbitrary
  "accuracy passed" threshold in this exploratory probe. Full-model logit,
  perplexity/reference-eval comparison and physical kernel parity remain open.
- Do not use CPU simulator elapsed time for hardware latency or bandwidth.

## Pins and execution

Official simulator repository: <https://github.com/tenstorrent/ttsim>.
Use release `v1.11.2`, `libttsim_bh.so`, SHA256
`d01be2a094f9f0f6a00e771defeae2311cb82f051b8af89c2df509e78b8432b7`.
The local clone is `f6150d114139a0b5265b1497d996bc6139af2cad`; the experiment uses
the release binary, not a build from that clone. Native Metal is
`a08819ddbe23077f8037d3802303939064868ff6` on the existing Linux/x86_64 host.

`real-weights-launch-v1.json` records the exact persistent command/environment.
It sets `TT_METAL_SIMULATOR`, slow dispatch and a separate JIT cache, caps usage
at one CPU and 8 GiB, and bounds the run to one hour. The probe validates the
binary digest and requires one visible virtual chip before opening a mesh.
No device lock, reset, firmware update or physical Galaxy access is needed.

Observed log interval: **2026-10-08 04:55:01-04:55:23 UTC**. Terminal report is
`completed`, `cleanup_completed=true`; collected service state is inactive,
exit status 0. The script prints a result and atomically saves progress after
each variant, so a later failure would preserve earlier comparisons.

First smoke v1 failed before matmul on the absent
`ttnn.BlackholeComputeKernelConfig` Python alias. Smoke v2 uses the pinned
runtime's `ttnn.WormholeComputeKernelConfig`, also used by this model on
Blackhole. It also disables simulator inspector RPC to avoid a port collision
with the hardware job. No arithmetic checks or unsupported-instruction checks
were disabled. Smoke v2 completed successfully.

## Receipts and reproduction

- [Probe source](../../demo/probe_bfp4_simulator.py)
- [Raw 60-case numerical receipt](real-weights-v1.json)
- [Aggregated summary](summary.json)
- [Persistent launch command](real-weights-launch-v1.json)
- [Pinned binary metadata](binary.json)
- [Service states and collection time](collection.json)
- [Complete simulator log](real-weights-v1.log.gz)
- [Exact executed source](probe-real-weights-v1.py.gz)
- [Failed smoke log](smoke.log.gz), [failed source](smoke.py.gz)
- [Corrected smoke receipt](smoke-v2.json), [log](smoke-v2.log.gz)

Source/log gzip files preserve the exact executed bytes (deterministic gzip
mtime=0). The initial and corrected smoke launch receipts remain alongside
these artifacts. To repeat, reconstruct the environment from the launch receipt
and invoke the probe with a **new output path**; it refuses to overwrite an
existing numerical receipt. The simulator library and checkpoint must already
match the recorded pins.

The primary performance effort remains placement/bandwidth at 128K and 256K.
A substantial 32K gain can justify a small longer-context regression when the
tradeoff is measured and explicitly flagged; this probe changes no such policy.
