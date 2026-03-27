# LTX-2 AV Pipeline Quality Regression Debug

## Goal
Identify and fix quality regression in the LTX-2.3 22B AV pipeline on WH LB 2x4 mesh after commits moved components from CPU to TTNN device.

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| Revert gate to fp32 host | bf16 device precision compounds over 48L×40 steps (PCC 0.92→0.997) | Keep on device with HiFi4 — insufficient precision |
| Use reference CPU Gemma encoding | TTNN Gemma hidden states explode (std ~60k vs ~2.7k by layer 47) | Fix TTNN Gemma bf16 precision — requires dedicated work |
| Set ge_gamma=0 | Good commit had no gradient estimation; not yet validated | Keep ge_gamma=2.0 — changes denoising trajectory, needs validation |
| Remove post-connector zeroing | Reference keeps register tokens alive after connector blocks | Zero only real-token positions — wrong, registers carry information |
| Switch to left-padding | Reference uses left-padding; connector register logic assumes it | Right-padding — broke register replacement packing |

## Constraints & Workarounds
- **WH LB 2x4 mesh** (8 chips), sequence parallel=2, tensor parallel=4
- **encode_prompts_reference (CPU Gemma)** — workaround for TTNN Gemma instability. ~180s vs ~5s. Permanent fix requires mixed-precision or fp32 accumulation in TTNN Gemma attention/FF across 48 layers.
- **ge_gamma=0** — workaround. Gradient estimation feature needs separate validation against reference quality before enabling.
- **Connector RoPE via host roundtrip** — TTNN connector blocks apply RoPE by all_gather → host apply → re-shard. Permanent fix: native TTNN interleaved RoPE op for connector attention.

## Surprises & Discoveries
- Gate bf16 precision loss was invisible in unit tests (single-layer PCC was fine) but compounded catastrophically over full 48L×40-step pipeline
- TTNN Gemma hidden states explode after just 1 layer (std 175 vs 7.9) even with proper attention masking — fundamental bf16 numerical issue, not a masking bug
- Post-connector zeroing destroyed 98.6% of tokens (1010/1024) — register positions were being zeroed when they should carry learned information
- Reference connector uses INTERLEAVED RoPE (not SPLIT like the DiT) with 1D positional indices
- TTNN SDPA doesn't support is_causal=True + attn_mask simultaneously

## Open Questions
- [ ] Root cause of TTNN Gemma numerical explosion — is it the RMSNorm, attention, or MLP that diverges first?
- [ ] Does ge_gamma=2.0 improve quality for some prompts? Needs A/B testing against reference
- [ ] Can connector RoPE be moved fully to device to avoid host roundtrip?

## State
- [x] Generate reference latents from good commit (749f1749f7)
- [x] Compare sigma schedule — PASS (identical)
- [x] Compare video/audio RoPE — PASS (identical)
- [x] Compare local LTX utils vs reference — PASS (identical positions/coords)
- [x] Identify gate precision regression (PCC 0.92 with bf16 device)
- [x] Fix gate: revert to fp32 host computation (PCC 0.997)
- [x] Identify TTNN Gemma encoding divergence (PCC 0.012)
- [x] Fix post-connector token zeroing (was destroying register tokens)
- [x] Add missing RoPE to connector transformer blocks
- [x] Fix padding side (right→left) matching reference
- [x] Add causal+padding attention mask to TTNN Gemma
- [x] Discover TTNN Gemma numerical instability (not fixable with masking alone)
- [x] Switch test to encode_prompts_reference as workaround
- [x] Set ge_gamma=0 matching good commit
- [x] Verify e2e test passes with good quality output
- [ ] Fix TTNN Gemma numerical stability (separate task)
- [ ] Validate gradient estimation (ge_gamma) quality impact (separate task)

## Key Measurements
| Test | Before Fix | After Fix |
|------|-----------|-----------|
| Video latent PCC (40 steps, gate only) | 0.922 | 0.997 |
| Audio latent PCC (40 steps, gate only) | 0.970 | 0.998 |
| TTNN Gemma embedding PCC vs reference | 0.012 | N/A (using reference encoding) |
| TTNN Gemma layer 0 (embedding) PCC | 0.999997 | — |
| TTNN Gemma layer 1 PCC | 0.087 | — |
| TTNN Gemma layer 47 std | 60,575 (ref: 2,740) | — |
| E2E test | PASSED | PASSED |

### Reproduction
```bash
# Full e2e test (reference encoding + gate fix + ge_gamma=0)
source python_env/bin/activate
export PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/kevinmi/.cache
python -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py::test_pipeline_av_22b -v -s
```
