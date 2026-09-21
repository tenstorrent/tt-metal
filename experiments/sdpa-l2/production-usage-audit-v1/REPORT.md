# Non-decode SDPA production-usage audit

2026-09-16. Read-only source audit of the current `tt-metal-blackhole` working
tree, HEAD `2daa9201e244957b95774d2a0bbe8d1db7c659e4`, including its existing
research integration changes. This is not a freshly fetched upstream-main
snapshot. No production changes, device qualification, or performance runs
were made for this audit.

## Decision

Proceed with **three PRs**, combining the originally proposed first four
into one implementation PR. Offer D/C/B/A/E/G as six recommended recipes,
but **do not make them the only representable internal configurations**.
Real models currently depend on other fidelity/destination/input-format
combinations. Preserve existing explicit configurations and omitted-config
defaults through a compatibility resolver, ultimately targeting the same
streaming machinery.

The main obstacle to deleting non-streaming code is feature/platform
coverage, not editing many Python call sites. The six new recipes can be
introduced without forcing all existing callers to change. Final deletion
requires migrating legacy configurations as well as the six recipes.

## Coverage and inventory

Three subagents audited language/shared models; vision/diffusion/audio and
other models; and APIs/tests/dispatch. The parent independently inventoried
tracked Python source and audited training, C++ references, and residual scope.

- **234 TTNN source sites across 113 files:** 134 direct calls plus four
  function-valued references under `models/` (88 files), and 96 calls under
  `tests/` (25 files). These include model-local tests and negative validation
  calls, not 234 production models. Two test calls belong to our prior
  research repro; the non-research test inventory is 94 sites in 24 files.
- Model partitions reconcile to **62 + 72 direct calls**; all four function
  references and three indirect Symbiote wrapper calls were separately traced.
- **14 TTML source sites in nine files** use independent training attention;
  C++ training modules/tests were also reviewed.
- 76 Torch reference sites were excluded from TTNN migration. The 38
  indirect/name-only census records were manually classified, including
  wrappers, Diffusion Gemma's monkeypatch, a Torch Falcon reference, and
  DeepSeek local variables rather than operator invocations.
- Zero Python AST parse failures. C++/text searches supplement the census.
  Decode APIs are excluded; shared dependencies used by decode are still
  recorded because cleanup could break them.

This is an exhaustive static inventory within the stated tracked-source scope,
not runtime instrumentation. Runtime-selected formats/configs are marked as
such, not guessed. External consumers, submodule implementations and untracked
research copies are outside the caller census. Tests admitting a configuration
are not evidence that all parametrized combinations pass.

### Detailed reports

- [Language/shared models and per-site provenance](language.md), with
  [62-call inventory](language-call-sites.json).
- [Vision, DiT, audio, wrappers and custom BGE attention](vision.md), including
  every owned direct call and function-valued reference.
- [Public APIs, defaults, tests, dispatch and deletion dependencies](tests-api.md),
  with [96-call inventory](tests-api-calls.json).
- [Training, documentation, residual scope and exclusions](training-and-scope.md).
- [Complete Python census](python-census.json) and [reproduction script](census.py).

## Observed precision families

Representative rows summarize actual call/config provenance, not every model
override. BF8/BF4 below mean `bfloat8_b`/`bfloat4_b`, not IEEE FP8. Destination
format is separate from input storage. Approximation flags are independently
selected; see per-site reports for packer, memory layout and runtime overrides.

| Current usage | Q / K / V storage | Fidelity; destination | Approximation / migration significance |
|---|---|---|---|
| DiT transformer blocks | Usually BF16 / BF16 / BF16 | HiFi2; BF16 | Math and exp approximation usually off. Not automatically identical to our frozen A recipe. |
| Wan/LTX quantized self-attention; Whisper self/prefill | Ordinary BF8 / BF8 / BF8 | HiFi2; BF16 | Wan/LTX usually exp off; Whisper exp on. Their ordinary casts are not E preparation. |
| Shared LLM attention, Gemma4, multimodal attention | BF16 or BF8 Q; BF16/BF8 KV depending on cache/config | HiFi4; FP32 | Usually both approximations off; paged, sliding, rectangular and masked users. |
| GPT-OSS / MiniMax dense; DeepSeek FlashMLA | Typically BF16 Q; BF16 or BF8 KV | **HiFi4; BF16** | Both approximations off. Important existing tuple outside the six recipes. |
| Gated-attention/GDN, Gemma encoder, several VAEs | Usually BF16 QKV | **HiFi2; FP32** | Commonly both approximations off; some VAEs have math approximation on but exp off. |
| Qwen3.6 chunked | BF8 Q and current BF8 cache | **HiFi2; FP32** | Math approximation on, exp off. Not C, which changes QK fidelity. |
| Qwen VL / BH ViT | All BF8, or BF8 Q/V with cache-dependent K | HiFi4; FP32 | Output/storage compatibility matters independently of internal accumulator precision. |
| SDXL UNet | Explicit BF16 self QKV; cross producer-dependent | **LoFi; BF16** | Both approximations off. No E/G-prescribed rounding. |
| BGE public SDPA | BF16 or BF8, shape-dependent | LoFi/HiFi2/HiFi4; BF16 or FP32 | Includes raw LoFi/FP32. Shape-specific tuning and architecture-dependent exp choice. |
| BGE custom serving SDPA | All ordinary BF4, or inherited Q with BF4 KV | LoFi; usually BF16 in BF4 serving | Own generic-op kernel, defaults non-streaming; not G's BF16-Q preparation contract. |
| DeepSeek sparse / MiniMax sparse MSA | Row-major BF16 or FP8-E4M3 Q; specialized KV formats | Often omitted compute config | Sparse-specific automatic FP32 tilization rules and gather/index contracts. |
| DIDT and numerical-control tests | BF16 plus broader suites with BF8/BF4/mixed inputs | LoFi, HiFi2, **HiFi3**, HiFi4; both destinations across suites | These controls remain compatibility obligations; no broad passing claim from static audit. |
| TT-Train fused attention | BF16 QKV | HiFi3 on WH, HiFi4 otherwise; FP32 | Independent forward/backward implementation, not an inference migration caller. |

Additional important distinctions:

1. **Omitted dense compute config resolves to HiFi2**, math approximation on,
   BF16 destination, packer L1 flag off. An **explicit empty compute config
   resolves to LoFi**. Omitted program exp selection resolves to approximation
   on. Preserve both cases, not just an apparent default constructor.
2. Existing dense APIs validate each Q/K/V independently as BF16, BF8 or BF4;
   FLOAT32 public QKV is not accepted. Mixed types occur in real callers and
   tests. Joint/ring/sparse APIs have different type constraints; see API report.
3. Sparse FP8-E4M3 and scaled-FP8 packed caches are not BFP8. Sparse default
   FP32 selection depends on the family and which input is FP8.
4. Dense output generally follows Q dtype; ring variants have separate output
   contracts. Switching an all-BF8-QKV caller to BF16-Q E/G may change output
   dtype unless handled explicitly.
5. `math_approx_mode`, program `exp_approx_mode`, and packer/L1 behavior are
   distinct. The public packer flag alone does not describe kernel recurrence
   precision: kernels explicitly configure L1 accumulation.
6. WH/BH compute config class names are aliases. They do not establish hardware
   eligibility. Source warns about WH HiFi4+FP32; training explicitly chooses
   HiFi3 there. Our Blackhole measurements do not qualify Wormhole.

## What must change, and how difficult is it?

### Straightforward: additive interface and ordinary callers

Keep the existing call signature valid. Add an optional named precision recipe;
when omitted, resolve existing compute/program configs with their current
semantics. Explicit recipe selection should reject conflicting numeric knobs
rather than silently choose precedence. Scheduling/memory choices remain
separate from numerical policy.

Common BF16 dense callers can opt in with a small Python change after
qualification. There is no reason to mechanically rewrite every existing model
to a nearest recipe. A model's HiFi4/BF16 destination request, for example, must
not silently become A/B or C/D.

### Moderate: storage/preparation and model integration

E/G must be explicit prepared-input contracts, not inference from a tensor's
BF8/BF4 dtype. Existing quantization is often fused into QKV projection,
normalization or cache writes. Preserve the original producer or deliberately
replace it; avoid hidden re-quantization and repeated cache preparation.

Model tests must verify output dtype, memory placement, preprocessing cost and
end-to-end quality. Update traced-config serialization and model quantization
profiles where they opt in. Preserve caller-selected chunks and buffering;
real model L1 pressure can invalidate a chunk configuration that wins in an
isolated benchmark.

### Substantial: complete streaming feature/platform coverage

- Ordinary FP32 support in this working tree still has a narrow experimental
  streaming eligibility guard. Existing causal, windowed, masked, chunked and
  MLA FP32 usages extend beyond it.
- Ordinary joint attention always invokes a legacy joint loop.
- Ring-distributed attention hardcodes non-streaming in reader, compute and
  writer, even for BF16 destination.
- Ring-joint FP32 still uses legacy compute. Experimental ring-joint instead
  requires streaming at compile time; a false host gate is not a working
  fallback.
- Coverage includes GQA, rectangular attention, unequal QK/V widths (e.g.
  MLA 576/512), D64/D256, non-power-of-two chunks, masks, sinks, packed windows,
  paged caches, trace-safe runtime offsets, joint tails and communication overlap.
- Sparse kernels have separate gather loops using streaming/common primitives.
  Sharing numerical policies is useful; forcing all reader/scheduling logic
  into one giant dense kernel is not.
- Quasar has separate APIs and physical kernel copies. Decide explicitly whether
  its migration is in scope; a BH refactor does not migrate them automatically.
- BGE serving uses a model-local copied implementation with compact lengths,
  GQA head-folding and direct-concat output. Migrate it or explicitly exclude
  it from the claim that all inference SDPA is streaming.
- Diffusion Gemma has a staged-matmul monkeypatch around a documented SDPA hang;
  retain until separately reproduced/qualified, rather than removing as cleanup.

### Deletion hazards

`compute_common.hpp` is not just the legacy algorithm: sparse, streaming,
decode and CCL reduce-to-root consume its helpers. Extract/retain reusable
helpers and delete only unreachable legacy loop bodies and plumbing. Decode
remains out of scope but must still compile. TT-Train's independent autograd
implementation is also outside this inference deletion project.

## Revised three-PR plan

| PR | Scope | Merge gate |
|---|---|---|
| **1. Streaming core + six recipes** (former 1–4) | Shared numerical/state primitives; A/B/C/D/E/G; required explicit preparation; API/config resolver; docs and preset tests. Internally staged commits, one reviewed PR. | Six recipes pass their declared supported envelope; old calls/defaults still work; no silent dtype or feature fallback; baseline numerical/performance tests; cache keys capture numerical policy. |
| **2. Feature/platform and caller migration** (former 5) | Port remaining joint/ring/chunked/MLA/FP32 feature routes and legacy configurations; resolve WH, Quasar and BGE scope; opt models into recipes where justified; update tracing/config tooling. | Feature × dtype × destination × architecture matrix qualified; trace/cache-hit and output contracts preserved; representative model regressions and performance measured. |
| **3. Legacy deletion** (former 6) | Remove unreachable non-streaming loops, factories/CB plumbing and obsolete dispatch; preserve shared utilities and out-of-scope training/decode. | No in-scope supported configuration still selects legacy; affected clients build/test; any explicit scope exclusions are documented. |

The simplification target is **one maintained streaming compute framework with
small numerical policies and feature-specific data movement**, not six copied
kernels, and not a forced reduction of all existing caller semantics to six
hard-coded tuples. This makes PR1 mergeable without coupling its success to a
repository-wide model behavior change; PR2 is the major migration effort.
