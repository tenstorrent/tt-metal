# Mistral-Small-24B-Instruct-2501 TTNN autoport

Full-model status: complete. Implementation and hardware gates pass, and the
independent `$stage-review` verdict is `clean-pass`. This stage contains the
complete TP4 autoregressive model and Metal readiness generator. It does not
contain vLLM integration.

| Batch-1 full-model metric | Result |
| --- | ---: |
| TTFT, AIME24 teacher-forcing prompt (327 tokens) | 2808.99 ms |
| TTFT, shared-suite token-out prompt (128 tokens) | 5812.15 ms |
| TTFT, chat-template token-out prompt (238 tokens) | 6057.47 ms |
| teacher-forcing decode, including forced-token callback/copies | 34.78 t/s/u |
| teacher-forcing trace-only interval | 54.77 t/s/u |
| traced shared-suite token-out decode, post-capture, device feedback | 55.00 t/s/u |
| traced free-running token-out decode, post-capture, device feedback | 55.94 t/s/u |
| prefill accuracy, top-1 / top-5 / top-100 | 99% / 100% / 100% |
| decode accuracy, top-1 / top-5 / top-100 | 97% / 100% / 100% |

The measured free-running path uses separate model and canonical Sampling1D
traces. The sampled token is the next model input at the same device address;
positions advance inside the model trace. Its 53 decode replays perform zero
full-logit readbacks and no per-token token, position, sampling-parameter, or
page-table host copies. The comparison-only common force-argmax sampler was
rejected: semantically identical greedy tokens measured 1.266483 ms through
the distinct common SamplingGenerator versus 0.314182 ms through Sampling1D
local-top-k split greedy.

Supported context remains the HF maximum of 32,768. The 1x4 p300c capacity gate
keeps both decode and prefill matrix representations for all 40 layers, batch
32 full-context BFP8 paged caches, TP endpoints, a 200 MB-per-DRAM-bank
(1.6 GB/device) trace region, and a physical 1.5 GiB/rank runtime reserve
resident. It executes a batch-32 prefill chunk and then decodes position
32,767 without releasing prefill weights. Arithmetic leaves 1,189,509,248
bytes/rank after the reserve and a page-aligned physical ceiling of 34,464.

Implementation: `tt/model.py` and `tt/generator.py`. Full commands, accuracy,
trace proof, qualitative review, limitations, failures/adaptations, and exact
artifacts are recorded in `doc/full_model/README.md` and
`doc/full_model/work_log.md`.
