# tt-lab accuracy reference review

Reviewed Oct 8, 2026 UTC at
[`e6baf562a72491ed9d594cddbdf42e0925b1f79e`](https://github.com/tenstorrent/tt-lab/tree/e6baf562a72491ed9d594cddbdf42e0925b1f79e).
Cloned read-only into `/private/tmp/tt-lab-accuracy-reference`; read `AGENTS.md`.
No tt-lab build, model run, device command or source modification was performed.
The repository's reported hardware matches below have not been independently
reproduced in our Qwen environment.

## Useful patterns for Qwen

1. **Separate arithmetic correctness from model-quality drift.** The host device
   proxy mirrors Tensix/SFPU arithmetic and consumes the same quantized tensors
   as simulation/hardware. `require_exact_match` compares float bit patterns and
   reports the first mismatching logit. CPU model versus proxy is a separate
   measurement. See
   [exact-match checks](https://github.com/tenstorrent/tt-lab/blob/e6baf562a72491ed9d594cddbdf42e0925b1f79e/src/tt_backend.cpp#L3237)
   and [CPU/proxy checks](https://github.com/tenstorrent/tt-lab/blob/e6baf562a72491ed9d594cddbdf42e0925b1f79e/src/tt_backend.cpp#L3647).
   Our completed simulator probe isolates quantization from execution error,
   but its float32 matmul reference is **not** an exact Tensix arithmetic oracle.
   Its nonzero execution RMS alone therefore does not establish a kernel bug.

2. **Model actual instruction phases and reduction order.**
   [tt_matvec.cpp](https://github.com/tenstorrent/tt-lab/blob/e6baf562a72491ed9d594cddbdf42e0925b1f79e/src/tt_matvec.cpp)
   implements operand unpacking, per-phase mantissa selection, exponent
   alignment, accumulation and destination encoding. Their two-phase BF16 and
   one-phase BFP8 paths are specific broadcast MVMUL programs. Qwen TTNN kernels
   need a reference matching their own generated operations and layout; copying
   this proxy wholesale would not establish equivalence. This also reinforces
   controlling the number/order of partial reductions during placement sweeps.

3. **Use real token inputs and intermediate checkpoints.** CPU/proxy comparison
   feeds the same prompt token IDs at each position, reports layer residual,
   normalization and MoE drift, then reports final logits. Tensor dumps include
   token/layer/operation names and a manifest. For Qwen, the useful follow-up is
   frozen real activations and teacher-forced tokens, not two freely generated
   sequences that diverge after one token. Preserve model/revision/packing/fidelity
   pins and measure raw logits before sampling.

4. **Report more than PCC.** Their
   [logit diagnostics](https://github.com/tenstorrent/tt-lab/blob/e6baf562a72491ed9d594cddbdf42e0925b1f79e/src/tt_backend.cpp#L3486)
   include relative RMS, maximum error, top-5/10/20 overlap, reciprocal ranks and
   softmax KL on the union of selected top tokens. The latter is a restricted,
   renormalized distribution, not full-vocabulary KL. These are useful additions
   when we have genuine full-model Qwen logits; sampled LM-head columns with
   synthetic inputs are insufficient for token-agreement claims.

5. **Quantizer details matter.**
   [requant.cpp](https://github.com/tenstorrent/tt-lab/blob/e6baf562a72491ed9d594cddbdf42e0925b1f79e/src/requant.cpp#L337)
   tests both the maximum shared exponent and one smaller exponent, selecting
   the lower reconstructed squared error; magnitudes round via
   `floor(abs(value) / scale + 0.5)` and clamp. Our standard Metal packer chooses
   the group's maximum exponent and uses ties-to-even. Thus differing bytes
   would not automatically imply either packer is broken. Their active expert
   layout shares exponents across 16 input columns per output; Qwen's tensor
   orientation and grouping must be stated explicitly in any comparison.
   Exponent selection is a possible separate accuracy experiment at the same
   nominal storage size, not an approved production quantizer change or a
   demonstrated model-quality improvement.

## Boundaries

- Current tt-lab supports GPT-OSS-20B/120B MXFP4 GGUF inputs, not Qwen3.8-27B.
  Its [current active expert format](https://github.com/tenstorrent/tt-lab/blob/e6baf562a72491ed9d594cddbdf42e0925b1f79e/README.md#L168)
  is BFP8, without a format-selection knob. Legacy BFP4 packing helpers/tests
  exist, but the active preload path excludes BFP4 strip tensors. Do not call
  its reported exact matches BFP4 Qwen qualification.
- Full-model simulator/silicon exact matches are **their reported results** on
  tested GPT-OSS inputs. The README explicitly leaves broader teacher-forced
  and task-level quality evaluation open.
- The runner's physical-device path takes device ownership and changes device
  state. We reviewed source only and did not run it alongside the Galaxy job.
- Qwen's completed [60-case BFP4 probe](../galaxy-evidence/bfp4-simulator-v1/README.md)
  remains a real-weight/synthetic-input numerical sample. Hardware parity,
  full-model accuracy and a causal account of GPQA drift remain open.
- Main effort remains long-context placement and bandwidth, weighted toward
  128K/256K with explicitly reported 32K tradeoffs. This reference work does not
  promote a new model precision policy or delay the running hardware experiment.
