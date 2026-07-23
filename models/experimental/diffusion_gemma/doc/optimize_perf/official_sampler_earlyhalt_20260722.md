# Official DiffusionGemma sampler with traced early halt (2026-07-22)

## Configuration

The official-semantics serving configuration now combines:

- IID full-vocabulary Gumbel-max through `DG_VLLM_GUMBEL_MODE=host`;
- the released `t_max=0.8` to `t_min=0.4` denoise temperature schedule;
- released entropy budget `0.1`;
- released stable-and-confident halt thresholds (`stability_threshold=1`,
  `confidence_threshold=0.005`);
- one-step traced early-halt windows through
  `DG_DENOISE_EARLY_HALT=1 DG_DENOISE_EARLY_HALT_WINDOW=1`;
- reusable up-front capture.

HTTP `temperature`, `top_k`, `top_p`, and `seed` are not checkpoint sampler
parameters and are not wired into the model-owned denoise loop. The GPQA launcher
therefore no longer presents those arguments as effective settings.

Dynamic Gumbel previously selected the fixed-budget single-step trace in the
vLLM wrapper even when early halt was enabled. It now selects
`traced_early_halt_block` when the halt window is one. A larger window fails
loudly because the Gumbel seed/noise must be refreshed between denoise steps.

`chunked` is not used for the official-quality launcher: its current
1024-wide TT RNG has a known innermost-axis distribution bias. `device` uses
the distribution-tested permuted-vocabulary workaround, but the full-vocabulary
temporary pads to an 8 GiB allocation and OOMs on this full-depth configuration.
`host` generates the same IID Gumbel distribution with torch and copies one
bounded full-vocabulary input into the persistent traced buffer. It is slower
than chunked sampling but preserves the released sampling semantics.

## Device validation

Full 30-layer DiffusionGemma on 4× Blackhole p300c, TP=4:

- `DG_UPFRONT_CAPTURE=1`
- reveal-mask `p_max=4096` (the full GPQA launcher size)
- 48-step cap
- host-generated IID Gumbel
- early-halt window 1
- two sequential prompts through one persistent capture

Results:

- request 0 halted at K=17;
- request 1 halted at K=19;
- both committed blocks, halt decisions, and realized K values exactly matched
  eager early halt under the same canvas/Gumbel/renoise seeds;
- `capture_events` stayed 48 across both requests;
- request 1 reported `gumbel_mode=materialized` and `reveal_mask_reuse`;
- both requests released normally;
- pytest result: `1 passed` in 151.25 s.

CPU coverage:

- 12 focused vLLM sampler/trace-selection tests passed;
- 36 sampling/up-front/real-Transformers parity tests passed;
- final combined up-front/vLLM regression: 39 passed, 5 device-gated skips,
  with the unrelated stale launch fixture deselected;
- formatting, shell syntax, lints, and diff whitespace checks passed.

The unrelated pre-existing
`test_server_launch_forces_optimized_trace_stack_and_full_prefill_budget`
fixture still lacks its `fixed_budget` field; it fails before reaching this
sampler selection code.

A fresh independent stage review returned `clean-pass` after the launcher moved
off biased chunked RNG and the device gate added exact eager-vs-traced
token/K/halt comparisons.
