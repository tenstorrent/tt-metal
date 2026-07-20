# Falcon3-10B-Base TTNN full model

**Full-model TTFT (128-token prompt): 2444.08 ms cold, 34.99 ms warm. Trace-verified batch-1 decode: 52.10 t/s/u teacher-forcing and 51.98 t/s/u caller-visible token-out.** The canonical readiness runner, which includes its own untraced boundary costs, measured 4219.68 ms TTFT and 44.56 decode t/s/u. All performance numbers are per user on four Blackhole p300c devices in a 1x4 TP mesh.

Status: full-model implementation and device validation complete; independent stage review returned `clean-pass`.

The full autoregressive path is implemented in `tt/model.py` and `tt/generator.py`. It preserves the optimized multichip decoder contract: TP4 on a 1x4 `FABRIC_1D_RING` with two links, BFP4_B/LoFi decoder and LM-head weights, BFP8_B paged KV cache, BF16 residual/activation/CCL state, persistent cross-layer all-reduce buffers, and the native L1 width-sharded residual layout. There is no single-chip, replicated-model, host-layer, or reduced-precision fallback path.

The public generator supports prompt lengths 1 through 32767 before token-out at the advertised 32768-token model context. It owns non-aligned padding, masking, 2048-token prefill chunking, cache fill, logical positions, page-table construction, and output slicing. Its lower-level interfaces expose explicit KV cache, page table, prompt length, fixed slot, active-row, and position state for later serving integration; this stage does not contain vLLM work.

Canonical AIME24 accuracy against the fresh exact-revision HF reference is:

| Path | Top-1 | Top-5 | Top-100 |
|---|---:|---:|---:|
| Prefill | 91/100 | 99/100 | 100/100 |
| Traced teacher-forcing decode | 91/100 | 99/100 | 100/100 |

Greedy token-out uses canonical split sampling: per-rank BF16 local argmax, a small FP32 rank-candidate gather, and a device-side winner written directly into the next decode token buffer. The model and sampler are captured as separate traces. The measured steady state performs no host token feedback, position rebuild, rotary-position rebuild, page-table rebuild, sampling-parameter copy, full-logits readback, or host argmax. An explicit host-sampling compatibility mode remains available for readiness tests that need logits on the host.

Detailed results, commands, qualitative review, policy ledger, fallback audit, and capacity evidence are in [`doc/full_model/README.md`](doc/full_model/README.md). The supported context remains 32768: full-stack batch-1 execution reached the last page, and a physical batch-32 full-context allocation left 4.12 GB DRAM free per device.
