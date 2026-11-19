# Projection order follows public HF length

For the installed CPU oracle and these contiguous BF16 projection operands, the supported rule is:

```python
projection_k_block = 512 if public_hf_sequence_length == 1 else 1024
```

The rule depends on the **whole HF linear call**, not the TT internal chunk length, tile padding, absolute cache position, or projection microbatch length. In particular, every one-token tail of a multi-token public prefill must retain K=1024.

CPU/source-only evidence is `projection_context_order_cpu.py`, `.json`, `.log` in this directory. The probe executes actual full `F.linear` calls at M = 1, 2, 3, 15, 16, 31, 32, 33, 64, 127, 128, 129, 257, 4096, 4097, for original QKV/Z/A/B weight widths. Inputnorm is computed on the complete M rows. For each output it checks the first, middle, and last row against both candidate schedules using all channels. This is sampled-row validation per M, supplementing the prior exhaustive M=128 prefix audit. No TTNN import, hardware, F edit or test edit was performed.

All M=1 kernels have a 64-iteration JIT loop consuming 8 K elements per iteration (K=512). All emitted kernels for every tested M>1, including internal M-tail kernels and A/B kernels, have 128 iterations (K=1024). Backend verbose output consistently identifies `brg_matmul:avx512_core_bf16`. The JSON records each newly emitted JIT file and loop count; reused A/B kernels need no second dump.

| Public M | Significant differences with K=512 | Significant differences with K=1024 |
|---:|---:|---:|
| 1 | 0 | 2 |
| 2 | 6 | 0 |
| 3 | 9 | 0 |
| 15 | 8 | 0 |
| 16 | 7 | 0 |
| 31 | 5 | 0 |
| 32 | 6 | 0 |
| 33 | 8 | 0 |
| 64 | 8 | 0 |
| 127 | 11 | 0 |
| 128 | 5 | 0 |
| 129 | 5 | 0 |
| 257 | 9 | 0 |
| 4096 | 5 | 0 |
| 4097 | 9 | 0 |

Counts aggregate sampled rows over QKV/Z/A/B and exclude canonical magnitudes <=1e-30 solely to separate known instruction denormal handling. Acceptance thresholds are unchanged. The correct schedule has zero ordinary numeric differences in every case. A/B sampled outputs alone often do not discriminate the schedules; their JIT loops do.

Source: oneDNN's pinned [AVX512 blocking heuristic](https://raw.githubusercontent.com/uxlfoundation/oneDNN/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul_utils.cpp) selects default K=1024 when K>1024 and the source tag is not transposed (lines 813–817). Its subsequent M/N/thread tuning does not change that K block for these BF16 forward cases. Degenerate one-row descriptors take the observed K=512 route. The normal contiguous M>1 forward source layout therefore supplies a source basis beyond merely extrapolating the tested lengths. This is the installed oracle's numeric contract, not a backend-independent mathematical identity.

## Necessary integration detail

`FusedDecoder.prepare_prefill` splits a public call into aligned chunks up to 4096 plus one-token boundaries. `prefill_forward` currently calls `_forward(..., decode=n == 1)` on those internal pieces. That flag controls implementation choice, but cannot choose projection order. Derive the projection K block once from `hidden_states.shape[1]` at the public entry, and pass it explicitly through the linear-attention frontend to every internal chunk. Alternatively store the public projection policy in the prepared plan. Avoid mutating a shared decoder-global flag since public batch rows can have different lengths.

Examples:

- Public M=31: all 31 internally decoded tokens need K=1024.
- Public M=33: the 32-token chunk and final one-token tail both need K=1024.
- Public M=4097: the 4096-token chunk and final token both need K=1024.
- Public M=1: use K=512, whether exposed as a decode or a one-token prefill.
- Projection microbatch T=1 within a longer prefill still uses K=1024.
- A nonzero or unaligned start position does not change the public projection policy.

A concrete large-context tail discriminator is M=4097 row 4096, QKV channel 6289: whole-call HF is `-0.859375`, while applying the decode K=512 schedule gives `-0.86328125`. Z channel 1404 is HF `0.388671875` versus K=512 `0.390625`. Thus the tail issue is observable, not merely a possible rounding difference.

Keep the existing once-per-public-call recurrent cache boundary. Changing projection chunking is not a reason to introduce BF16 state rounding at internal boundaries. The frontend can process small outer-product microbatches and preserve the public numerical policy without changing recurrence/state semantics.

Reproduce, optionally adding `ONEDNN_JIT_DUMP=1` from a scratch working directory to capture machine code:

```sh
ONEDNN_VERBOSE=1 python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/projection_context_order_cpu.py > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/projection_context_order_cpu.log 2>&1
```

Full CPU probe elapsed 11.82 seconds in this run. JIT artifacts from the audited run are under `/tmp/projection_context_order_jit`.
