---
name: functional-decoder
description: Validate and harden the existing DiffusionGemma Gemma-4 backbone integration and diffusion delta. Use for dg-01 backbone parity and dg-02 diffusion-delta work; never create a greenfield functional_decoder.py.
---

# DiffusionGemma functional path

Load `diffusion-gemma` first. This skill is specific to the existing
`models/experimental/diffusion_gemma/` implementation.

## Scope

Stages dg-01 and dg-02 validate and harden code that already exists:

- reused backbone wrapper: `tt/model.py`;
- checkpoint mapping: `weight_mapping.py`;
- bidirectional canvas attention: `tt/diffusion_attention.py`;
- three-phase KV state: `kv_phase.py`;
- self-conditioning: `tt/self_conditioning.py`;
- canvas sampling: `tt/sampling.py`;
- denoise adapter glue: `tt/denoise_forward.py`.

Do not create `tt/functional_decoder.py`, a `FunctionalDecoder` class, or a new
embedding/block/LM-head stack. Do not edit `models/demos/gemma4/` or other
shared model directories.

## Stage 01: backbone parity

1. Read the HF DiffusionGemma configuration/modeling code and the current
   `plan.md`, `AGENTS.md`, and `doc/context_contract.json`.
2. Load the real checkpoint through `weight_mapping.remap_state_dict`.
   Reconcile every missing, renamed, or extra key, including
   self-conditioning weights.
3. Verify model-shape and layer-kind details: sliding versus full attention,
   dual RoPE, K=V only where configured, V norm, softcap, MoE geometry, and
   262144-token advertised context.
4. Validate the causal encoder/prefill path on QB2 with real weights for every
   meaningful layer kind. Record both logits PCC and argmax agreement.
5. Treat the documented BF16/MoE/TP=4 fidelity floor as evidence, not as
   permission to hide a new regression. The generic `PCC >= 0.995` decoder bar
   does not apply to the complete backbone; component kernels should still
   meet their own stronger bars.

## Stage 02: diffusion delta

Validate each net-new component against the torch/HF reference:

- non-causal canvas attention, including prompt visibility, symmetric local
  window geometry, long-prompt chunking, and absolute RoPE offsets;
- frozen prompt/committed KV, per-step ephemeral canvas K/V, and commit append;
- denoise-only self-conditioning, zeroed on encoder passes;
- Gumbel-max, entropy, entropy-budget accept/renoise, and random-token noise;
- trace-safe tensor-valued cutoff/index behavior.

For attention, KV, sampling, and self-conditioning component tests, use the
component acceptance bar recorded by the owning test (normally PCC >= 0.99).
For the integrated diffusion path, correctness is the injected-noise decision
trajectory: argmax agreement, entropy PCC/max error, accept/renoise IoU, canvas
agreement, and committed-token agreement. Teacher-forcing top-k is not a gate.

The production MoE path is the existing true-sparse token-gather
`tt/sparse_moe.py` implementation. Do not replace it with the generic dense
all-expert or GPT-OSS bring-up recipe.

## Capability and runtime checks

- Preserve arbitrary valid logical prompt lengths; pad/chunk internally.
- Validate short, boundary, just-over-boundary, and long non-aligned shapes.
- Keep frozen-prefix reads and canvas scratch semantics correct through 256K.
- Keep runtime forwards free of hidden torch, `ttnn.from_torch`,
  `ttnn.to_torch`, or host decision fallbacks, except explicit harness
  boundaries.
- Preserve deterministic replay by injecting the same initial canvas, Gumbel
  noise, and renoise tokens as the reference.
- Run watcher separately from profiling.

## Shared-backbone gate

Run:

```bash
DG_BASE_REF=<actual-branch-base> \
  bash models/experimental/diffusion_gemma/.agent/scripts/check_no_shared_gemma4_edits.sh
```

Choose the actual base ref; do not use a stale local `main`. Any
DiffusionGemma-owned shared-directory delta is a failure and must be moved
local or split into a separately owned upstream change.

## Evidence

Record exact commands, hardware, checkpoint revision, shapes, PCC/decision
metrics, watcher status, and limitations under:

- `models/experimental/diffusion_gemma/doc/backbone_parity/` for dg-01;
- `models/experimental/diffusion_gemma/doc/diffusion_delta/` for dg-02.

Update `doc/context_contract.json` whenever supported context, KV/cache layout,
canvas scratch, masks, or persistent allocations change.

Done means the existing path is validated on real shapes and weights, the
shared-backbone gate passes against the correct base, no hidden host fallback
is required, and the corresponding stage review is clean.
