# Qwen3.6-27B TP4 full model

Status: **final review pending**. On four Blackhole p300c devices, the full
64-layer model reports AIME24 prefill top-1/top-5/top-100 of
**92% / 100% / 100%** and traced teacher-forcing decode of
**97% / 100% / 100%**. At B1/S128, final full-model TTFT is **4.420 s** and canonical
split-trace token-out is **17.52 t/s/u** over 128 replays. Teacher-forcing
compatibility decode is **6.96 t/s/u**; it is reported separately because it
reads logits and refreshes teacher inputs on host.

## Runtime contract

`tt/model.py` implements replicated device embedding, the unchanged 64-layer
`MultichipDecoder` stack, device final RMSNorm, and a TP4 vocab-sharded LM head.
The residual between layers remains replicated BF16. The decoder's selected
precision and geometry remain binding: BFP8 paged KV/recurrent state, BF16
activation and CCL payloads, BF16/HiFi2 full-attention projections,
BFP4/LoFi linear projections and MLPs, DRAM-sharded decode weights, and the
TP4 ring collectives. There is no single-chip, host-layer, replicated-weight,
or lower-performance fallback.

The LM head owns contiguous vocabulary shards. Each 62,080-column local shard
is split into 30,976 and 31,104 columns, DRAM-sharded, projected 32 rows at a
time, concatenated locally, and sliced to non-aligned public lengths. This is
the fastest correct terminal contract found; an unsplit shard exceeds L1.

`tt/generator.py` exposes explicit low-level prefill/decode state: caller-owned
cache, page table, positions, prompt lengths, active rows, and fixed slots.
Public prefill owns padding masks, per-64-token linear-attention selectors,
cache fill, and logical output slicing. The inherited layer-0 mixed S65/S63
probe is `logs/mixed_prompt_state_final.log`; it is not full-wrapper evidence.
The reduced full-wrapper probe covering embedding, three linear layers, one
full-attention layer, terminal LM head, common sampler, and the public token-out
API is `tests/full_model_mixed_slots.py` and requires a serialized hardware run.
Inactive slots are empty/reset slots; preserving a paused occupied slot is not
claimed. The reduced full wrapper (embedding, layers 0--3, terminal, sampler)
passes mixed S65/S63 at B2 with public non-greedy top-k=5/top-p=0.9 and an
inactive row (`logs/full_model_mixed_slots_lm_head_fix_v4.log`).

## Sampling and tracing

The optimized greedy path uses common `SamplingGenerator`/`TTSampling` and
semantic full-vocabulary all-gather+argmax. Model decode and sampling are two
device traces; the sampler writes directly to the persistent `tt_out_tok`
consumed by the next model replay. Position increments are also traced and
active-row masked. No full logits, host argmax, per-token host token/position
refresh, page-table rebuild, or untraced sampler occurs on this path. Only the
small sampled token is read for API output.

The alternative common local-top-k path produced the same reduced token
sequence but was slower: its fixed TopK costs 9.697 ms in the matched reduced
profile. It is rejected. The selected force-argmax path slices to active rows;
its remaining sampler all-gather is 0.830 ms (24.4% of named device time),
below the 1.198 ms LM head and not dominant. `Sampling1D` is also rejected because it lacks the common wrapper's
trace/state contract. A named `host_sampling_compatibility=True` mode retains
traced logits readback for readiness teacher forcing; it is not the measured
token-out path.

Serving callers use public `setup_token_out_decode(...)` and
`token_out_decode_step(...)`. Setup accepts explicit token, position,
page-table, cache, active-slot, and `SamplingParams` state. Replay returns the
persistent device token by default, so greedy and top-k/top-p-capable modes do
not cross a logits boundary; sampled-token readback is opt-in.

Trace evidence (`logs/reduced_trace_state_page_table_final.log`) shows device
positions advancing only for active rows, unchanged page tables causing zero
refreshes, a changed table causing exactly one refresh, and zero host token or
position refreshes. Repeated reset/capture lifecycles preserve the canonical
greedy token sequence.

## Correctness and qualitative evidence

The fresh reference is `readiness_aime24_chat.refpt`, generated with the exact
local revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, HF tokenizer and chat
template, 161 prompt tokens, 100 continuation tokens, and top-100 logits.

| Gate | top-1 | top-5 | top-100 | Evidence |
|---|---:|---:|---:|---|
| Full prefill | 92% | 100% | 100% | `logs/run_prefill_check_split_lm_head_v2.log` |
| Traced teacher forcing | 97% | 100% | 100% | `logs/run_teacher_forcing_final.log` |

The standard 100-token autoregressive run saved both completions under
`autoregressive_final/`. HF and TT are coherent English, non-repetitive, and
follow the same algebraic setup; neither reaches the numerical answer inside
100 tokens. TT diverges stylistically after token 3 but does not drift language
or fail semantically. The machine checker reports zero adjacent duplication,
5.26% trigram overlap, and no degeneracy (`artifacts/degenerate_output.json`).

The six-prompt shared suite is complete in
`artifacts/qualitative/shared_suite.json`. All prompts use the exact chat
template. Manual review found six coherent, prompt-relevant TT outputs matching
the HF control style, with no repetition, language drift, prompt echo,
control-token leakage, or suspicious early divergence.

## Capacity and audit

Actual resident weights are **13,460,453,888 bytes/device**: decoder
10,599,141,888; replicated embedding 2,542,796,800; final norm 10,240; physical
TP4 BFP8 LM head 318,504,960. B1 C262,144 KV is 2,281,701,376 bytes/device.
The corrected physical probe passes B1 C262,144 and brackets B32 at C72,192
pass / C72,256 fail when terminal weights are resident. A second B1 C262,144
probe also passes while reserving the full token-out trace snapshot copy of
every KV/recurrent/conv state (`artifacts/capacity_trace/`).

The runtime fallback audit is `artifacts/runtime_fallback_audit.txt`. Host
logits occur only at public prefill results, low-level compatibility decode,
and explicit host teacher forcing. Optimized decode has device token feedback
and sampled-token-only readback. Cache reset zeros every layer state on device;
trace teardown releases model and sampler traces before state rebinding.

The inherited rejection ledger remains in force: fractured distributed-norm
residual, packed MLP, BFP8 CCL, preallocated CCL, L1 prefill, fused
matmul→reduce-scatter, and TP4 all-gather→matmul were not re-enabled.

## Limitations

- Public prompt/context capacity is 192,511 at B1, the largest physically
  demonstrated full-layer prefill before the measured contiguous-allocation
  limit. Decode cache allocation and absolute positions extend to 262,144 for
  generation after a supported prompt; that separate cache capability is not
  advertised as prompt capacity.
- High-level `generate` is B1. Mixed prompts/fixed slots use the explicit
  low-level serving API.
- Host compatibility sampling is for tests, not performance claims.
- vLLM integration is intentionally outside this stage.
- Full-wrapper Watcher cannot initialize TP4 fabric in this checkout: Watcher
  instrumentation makes ACTIVE_ETH firmware 27,920 bytes versus the 25,600-byte
  kernel-config limit, before model construction. Inherited decoder Watcher
  evidence remains valid; final full-wrapper integrity instead uses named
  device profiling, traced determinism/reset, and runtime logs.
