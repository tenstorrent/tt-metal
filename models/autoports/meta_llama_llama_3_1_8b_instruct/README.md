# Llama 3.1 8B Instruct — optimized TP4 full model

Full-model readiness is complete on a P300 1x4 ring. The primary batch-1
prompt-128/generate-128 workload measures **20.51 ms warmed TTFT** and
**110.45 token-out t/s/u** through the public generator, including one token-ID
readback per output. The isolated device-feedback model-plus-sampler trace is
**111.17 t/s/u**; teacher forcing is **101.50 t/s/u** because it intentionally
copies a forced token at every step. Sampling consumes 0.795 ms/token, or 8.84%
of separate trace time, so it is not the token-out bottleneck.

| Final full-stack gate | Result |
| --- | ---: |
| Prefill top-1 / top-5 / top-100 | 86% / 100% / 100% |
| Decode teacher-forcing top-1 / top-5 / top-100 | 86% / 100% / 100% |
| Warm TTFT, prompt 128, batch 1 | 20.51 ms |
| Public traced token-out, generate 128, batch 1 | 110.45 t/s/u |
| Device-feedback trace pair, batch 1 | 111.17 t/s/u |
| Teacher-forcing decode, batch 1 | 101.50 t/s/u |
| Supported logical context | 131,072 tokens |
| Real near-context hardware gate | prompt 131,071, all 2,048 pages |
| Shared qualitative suite | 6/6 coherent, no degeneration findings |

The implementation preserves the accepted optimized multichip decoder policy:
TP=4 Q8/KV2 ownership, P300 ring with two links, BFP4/LoFi projections, BF16
residual/norm/CCL state, BFP8 default paged KV cache with BF16 compatibility,
and the 16-core L1 inter-layer residual. The canonical greedy path uses split
Sampling1D on device; full-vocabulary gather plus argmax is comparison-only.
The exact 131,072-token prompt boundary is accepted for one generated token;
the generator accounts for prefill producing that first token rather than
allocating a nonexistent extra decode slot.

See [doc/full_model/README.md](doc/full_model/README.md) for commands, evidence,
artifacts, limitations, and the runtime fallback audit. This stage does not
contain vLLM integration.
