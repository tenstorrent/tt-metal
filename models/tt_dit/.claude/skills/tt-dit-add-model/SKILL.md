---
name: tt-dit-add-model
description: >-
  Use when someone is standing up something new in models/tt_dit — a diffusion
  transformer, VAE, text encoder, vocoder, or any block — and needs it
  numerically correct on Tenstorrent mesh hardware. Covers the full bringup arc:
  which in-tree model to copy, where pipeline vs. module vs. tests belong,
  state-dict conversion, picking the minimum set of production shapes to test
  rather than sweeping resolutions and durations, wiring the parallel config, and
  closing PCC gaps against a torch/diffusers reference. Applies equally to
  planning and convention questions — file layout, structure, how many tests,
  which configs — for a tt_dit model that isn't written or isn't green yet, and to
  first-run failures: bad PCC, key mismatches, allocation errors, device hangs on
  a fresh port. Not for a component that already passed and then broke under
  tracing, caching, or other speed work — that's tt-dit-performance and
  tt-dit-benchmark-profile.
---

# TT-DiT Bringup

Take a component from "there is a reference implementation" to "it passes a
quality gate against that reference, at the shape that ships."

Optimizing a component that is still numerically wrong wastes every iteration
spent on it — and a component wrong in a way you haven't found yet looks
exactly like one that is right. Hence the split from `tt-dit-performance`.

## Before the first device run

Read `../shared/device-hangs.md`. Every run is timeout-gated, every kill is
followed by a reset — an unbounded run that wedges the device costs the rest of
the session.

## Two rules that save the most time

**One component at a time, all the way to end-to-end.** Unit gates green →
e2e for *that* component against the real checkpoint → next component. An e2e
pass on real weights catches key-mapping errors, tiling and stitching geometry,
and normalization constants — classes of bug synthetic-weight unit tests cannot
reach — and catches them while the component is still fresh.

**Production configs only, and the minimum number of them.** Derive test shapes
from the model's real schedule; do not sweep arbitrary `(C, T, H, W)`. Product
configs frequently collapse: one tiled VAE's four configs (two resolutions × two
durations) share exactly **two** device shapes, so testing all four tests two
shapes twice. Every test is a device run paid on
every future regression check.

Sweeping invented shapes is worse than useless: a GroupNorm hang at
`(C=128, T=5, 32×32)` cost three hangs and two resets on a shape the real
encoder never produces.

## Phases

| Phase | Do | Load |
|---|---|---|
| Orient | Plan → journal → the family's layout doc. Then find prior art | `../shared/reference-models.md` |
| Scaffold | Weight discovery, state-dict conversion, module skeleton | below |
| Gate | PCC vs the CPU reference, then e2e on real weights | `testing-and-accuracy.md` |
| Parallelize | Wire the config; re-gate sharded against unsharded | `../shared/parallelism.md` |
| Hand off | One baseline measurement per component | `tt-dit-benchmark-profile` |

Journal at every phase boundary (`../shared/journal-protocol.md`). A bringup
spanning many sessions should run under `../tt-dit-loop/`, which owns the
checkpoint, ledgers and resume path.

## Orient

| Read | Why |
|---|---|
| The plan, if one exists | Source of truth for scope and gates. Re-read every iteration — plans leave your context long before they go stale |
| The journal | What passes, what was measured, what was amended, and `Failed attempts` so you don't re-run someone else's dead ends |
| The family's layout doc (`models/<Model>.md`) | Conform to it; a parallel structure makes the merge painful for everyone |

## Scaffold

| Step | Detail |
|---|---|
| **Import the reference, don't hand-port it** | A hand-written reference is a second implementation with its own bugs; when the gate fails you can't tell which side is wrong. Priority: importable class → upstream serving PR (sglang, vLLM) → the checkpoint's own source dir |
| **Pin the reference commit** in the journal | e.g. `diffusers PR #14355 @ abc5e9bf` → `AutoencoderKLYourModel` |
| **Find weights before downloading** | Checkpoints run to hundreds of GB and are usually already on the box, often under another user's dir and occasionally misspelled. Pull only what the component needs |
| **State-dict conversion** | `utils/substate.py` for prefixed sub-trees; follow the family's `convert_<model>_state_dict`. Verify shapes from the safetensors index header before loading; cache the result (`utils/cache.py`, `TT_DIT_CACHE_DIR`) |
| **Module structure** | Match the closest in-tree model, not the reference's Python layout. Reuse from `layers/` and `utils/` — the table in `../shared/reference-models.md` |

```bash
# $SCRATCH = the shared checkpoint mount on your machine (often /data or /mnt)
ls -d "$SCRATCH"/*/*<model>* 2>/dev/null
find "$SCRATCH" -maxdepth 3 -iname "*<model>*" -type d 2>/dev/null
```

Key-mapping errors are the most common bringup bug and nearly invisible — a
swapped q/k or a transposed weight still produces plausible output at mediocre
PCC. `Module.load_torch_state_dict` is **strict by default** and raises on
missing or unexpected keys, catching most of them for free. Never silence it
with `strict=False`.

**Gate unproven behaviour behind an env flag** defaulting to current behaviour
(`LTX_TRACED`, `IDEOGRAM4_DEQUANT_CACHE` are the convention). It makes an A/B a
single env var rather than a rebuild. Flip the default when evidence arrives and
say so in the journal — a flag left `False` for six months because nobody ran
the profile is dead code with a note attached.

## When something fails

Bisect against the reference — `testing-and-accuracy.md` § "When a gate fails".

A **hang is a design signal, not a flake**: check `../shared/known-issues.md`
before treating it as random. A **precision floor is not a bug**: bf16-only
kernels cap achievable PCC; measure the floor, record it, set bars above it.

## Done

- Every component green on its unit gate and its e2e gate on real weights.
- Sharded path gated against serial — bit-exact for data parallel, within the
  quality bar for spatial sharding, seams checked separately.
- Pipeline-level sanity checks in place where there is no torch reference.
- One baseline per component, recorded with command, mesh shape, input shape and
  SHA, so optimization has a "before".
- Journal current, including any amendment where a measurement contradicted the
  plan.
