# APC / frozen prompt-prefix KV reuse — work log (#47466)

Agent B. Branch `dg-vllm-apc`. Device: shared QB2 (4× Blackhole, `(1,4)` mesh) —
every device run wrapped in `flock /tmp/dg-mesh.lock timeout 900`.

## Goal

Reuse the already-computed prompt-prefix K/V across `generate` calls when two
requests share a prompt prefix, instead of re-encoding it. Prototype at the DG
serving layer behind `DG_PREFIX_CACHE` (default OFF). Do NOT wire vLLM's block pool
(that is #47488). Acceptance bar: bit-exact committed argmax with the cache ON vs
OFF for two requests sharing a long prefix, plus a logged prefill-time saving.

## What I changed

| File | Change |
|---|---|
| `tt/prefix_cache.py` (new) | `PrefixKVCache` — host-side registry of the aligned prompt tokens the contiguous cache currently holds; `plan()` decides bit-exact reuse; `prefix_cache_enabled()` reads `DG_PREFIX_CACHE`. No device state. |
| `tt/serving.py` | `BlockDiffusionServingSession.__init__` gains `prefix_cache=`; `prefill()` skips `prefill_prompt_tokens` when the aligned prompt is a full prefix of the resident cache; exposes `prefill_reused` / `prefill_time_s`. |
| `tt/generator_vllm.py` | adapter holds one shared `PrefixKVCache`, passed to every session; `model_capabilities["supports_prefix_caching"]` stays **False** (vLLM block pool not wired — #47488). |
| `demo/prefix_cache_smoke.py` (new) | reduced-surface device driver: ON-vs-OFF bit-exact committed-argmax check + prefill-saving, for exact-match and aligned-proper-prefix cases. |
| `tests/test_prefix_cache.py` (new) | 9 CPU unit tests pinning the reuse decision (which cases are bit-exact-safe vs the #47488 partial-prefix case). |
| `doc/vllm_integration/prefix_cache/README.md` (new) | design note: RoPE / three-phase-KV / sliding-window correctness argument + the #47488 boundary. |

## Correctness argument (see README for the full version)

The frozen prompt K/V is written to `tt_model.tt_kv_cache[layer]` at absolute
positions `[0:cache_len]` by causal prefill and read read-only by denoise. Because
(1) RoPE is absolute (position `i`'s rotation depends only on `i`), (2) causal
prefill makes position `i`'s K/V a pure function of `tokens[0:i]` regardless of the
total prefill length, and (3) both sliding-window and full-attention layers attend
to a subset of `[0:i]`, a new request whose *aligned* prompt is a byte-identical
leading span of the resident cache's aligned prompt has bit-identical K/V for that
span — so its prefill can be skipped. Alignment constraint: the reuse span must
match over the whole `cache_len` including 32-tile pad, so only exact-full-match
and aligned-proper-prefix qualify (enforced by `plan`).

## Where #47488 is required (documented, stopped cleanly)

The productionly-valuable **shared-prefix + differing/extending suffix** case
cannot be bit-exact at the DG serving layer: the suffix must cross-attend to the
cached prefix during prefill = chunked/prefix prefill, which the Gemma4 backbone
does not support (`models/demos/gemma4/tt/model.py:1291,1298,1385` — `del
start_pos`, "Gemma4 doesn't chunk-prefill") and which I may not add by editing the
shared backbone. A DG-local commit-decode "suffix prefill" is functionally correct
but not bit-exact to the batched-prefill matmul in bf16 (and #48291 shows small
drift can flip a diffusion decision), so it fails the bar. General APC belongs in
the #47488 paged-cache ownership path. The prototype logs
`partial-prefix miss: matched N ... → full prefill (needs chunked prefill / #47488)`
and counts `partial_prefix_misses` so the reuse a paged path could capture is
visible.

## Evidence

### CPU unit tests (device-free) — PASS

`python -m pytest models/experimental/diffusion_gemma/tests/test_prefix_cache.py -q`
→ **10 passed**. Covers: empty resident, exact match (bit-exact reuse), aligned
proper prefix classified as shorter-prefix (not reused by default) + reused only in
the approximate tier, non-aligned proper prefix (rejected), extending suffix
(partial-prefix miss / #47488), longer-than-resident (rejected), record re-anchoring,
stats/time tracking, and input validation.

### Device bit-exact check (under flock) — PASS (QB2, `bh-qbge-06`, 2026-07-06)

Important env note: `/home/zni/tt-metal-apc` is a **git worktree**; the built
`runtime/` + `libtt_metal.so` live only in the main checkout, so device runs use
`TT_METAL_HOME=/home/zni/tt-metal` with `PYTHONPATH=/home/zni/tt-metal-apc` (imports
my code, uses the built runtime). `TT_METAL_HOME=$PWD` fails the kernel link
(missing `runtime/hw/toolchain/blackhole/*.ld`).

```
flock /tmp/dg-mesh.lock timeout 900 \
  env TT_METAL_HOME=/home/zni/tt-metal TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal \
      ARCH_NAME=blackhole PYTHONPATH=/home/zni/tt-metal-apc \
      DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it TT_LOGGER_LEVEL=ERROR \
  python -m models.experimental.diffusion_gemma.demo.prefix_cache_smoke \
    --mesh P150x4 --num-layers 2 --max-seq-len 1024 \
    --num-blocks 1 --max-denoising-steps 2 --gumbel-mode argmax --case both \
    --local-files-only --metrics-json .../prefix_cache/prefix_cache_smoke_reduced.json
```

Result marker:
```
DG_PREFIX_CACHE_SMOKE_SUCCESS required_bit_exact_pass=True all_reused=True
  exact:bit_exact=True,mismatch=0,reused=True,saved_s=0.127
  prefix:bit_exact=False,mismatch=57,reused=True,saved_s=0.168
```

- **exact-match reuse: BIT-EXACT (0/256 committed-argmax mismatches)**, prefill
  skipped (`reused_on=True`, `prefill_time_on_s=0.0`). **Acceptance bar met** — two
  requests sharing the full prompt prefix produce identical committed output with the
  cache ON vs OFF. Prefill wall-time saved = 0.127 s at 2 layers (scales to the full
  ~60 s prefill at 30 layers).
- **aligned-proper-prefix reuse (approximate tier): NOT bit-exact — 57/256 tokens
  flip.** Measured evidence for the bf16 SDPA reduction-length effect; correctly gated
  off by default (`allow_shorter_prefix`), reported as `shorter_prefix_misses` when off.
- Artifact: `prefix_cache_smoke_reduced.json`. gemma4 isolation gate:
  `git status --porcelain -- models/demos/gemma4/` empty (all changes under
  `models/experimental/diffusion_gemma/`).

### Full-depth run — BLOCKED by device fault (DEVICE_PROBLEM)

A full-depth (30-layer) `--case exact` run for the realistic prefill-time saving
**failed at `ttnn.open_mesh_device`** with the recurring
`Device 0: Timed out while waiting for active ethernet core 29-25 to become active
again` fault (`llrt.cpp:581`). This is the documented recurring recoverable ARC/ERISC
flake on this box (see the parent `doc/vllm_integration/work_log.md` — "eth core 29-25
recurring reset"), triggered here because an earlier attempt was hard-killed by the
harness's 120 s tool timeout mid-device-use, leaving the ethernet core un-reset. It is
**not** a code fault (the reduced-layer run of the same code passed cleanly). Per the
task rules I did **not** run `tt-smi -r` — this needs an orchestrator-coordinated
reset. My processes are cleaned up (`pkill -9 -f prefix_cache_smoke`, 0 leftovers) and
the flock lock is free. The full-depth *saving* number is a nice-to-have; the
bit-exact correctness bar is already met by the reduced-layer run. Re-run the
full-depth command above after the mesh is reset.

## SHAs (branch dg-vllm-apc)

- `cbddeef07ca` — feat: frozen prompt-prefix KV reuse prototype behind DG_PREFIX_CACHE
  (PrefixKVCache, serving/adapter wiring, smoke, 9 CPU tests, design note).
- `09f7f075e52` — fix: restrict bit-exact reuse to exact match; add opt-in approximate
  shorter-prefix tier; measure bf16 drift; loguru import; 10 CPU tests.
- (this commit) — device evidence + docs finalization.

## OPEN QUESTIONS / BLOCKED

- General shared-prefix+suffix APC is BLOCKED on #47488 (paged-cache ownership) /
  chunked prefill — documented above; not attempted (would need a forbidden gemma4
  edit or a non-bit-exact commit-decode suffix prefill).
- Aligned proper-prefix reuse is fp32-correct but NOT bit-exact in bf16 (measured
  57/256 drift); shipped OFF. A bit-exact shorter-prefix reuse would need a
  length-independent attention reduction (kernel property, not changeable here) — so
  it too effectively belongs to the #47488 paged path where prefixes are computed once
  and never recompared against a shorter standalone prefill.
- DEVICE_PROBLEM: eth core 29-25 reset fault blocked the full-depth run; orchestrator
  reset needed (I did not reset). Reduced-layer correctness already recorded.
