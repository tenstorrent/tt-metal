# MOSS-Image validation

Tests use repository `mesh_device` and `device_params` fixtures. Model code never
opens or closes their devices. Integrated tests reserve 128 KiB L1-small for
the full-resolution VAE; the corrected SFPU encoder recurrence runs within the
same reservation. Run from the TT-Metal checkout with its freshly
built native TTNN and the model runtime requirements installed.

## Assets

Provision this read-only directory before enabling the pipeline entry:

```text
/mnt/MLPerf/tt_dnn-models/OpenMOSS-Team/MOSS-Image/
  checkpoints/Base/{transformer,text_encoder,tokenizer,vae,scheduler}/...
  checkpoints/SFT/{transformer,text_encoder,tokenizer,vae,scheduler}/...
  cuda-Base-1024-50/manifest.json
  cuda-Base-1024-50/<manifest-declared capture files>
  cuda-SFT-1024-50/manifest.json
  cuda-SFT-1024-50/<manifest-declared capture files>
```

Shared checkpoint components may be symlinked to Base. Captures are pinned by
the manifest SHA-256 values in `capture_pins.json`; each consumed tensor is
hashed before safe loading. Checkpoint files, indexes and configs are verified
against the pinned CUDA manifest before execution. An arbitrary extra shard
cannot override an index-declared weight. These tests never download into or
write to the asset mount. Output artifacts use pytest's temporary directory.

`MOSS_IMAGE_ASSET_ROOT` overrides the asset root. Missing assets fail by default,
including in CI. A developer intentionally collecting a host-only run can set
`MOSS_IMAGE_ALLOW_MISSING_ASSETS=1`; this opt-out is rejected when `CI` is set.

## Scope and commands

```bash
# One P150: host regressions, 32 encoder layers + final conditioning in both
# branches, Base/SFT selected DiT blocks + full step, VAE, scheduler and native
# 256px, two-step functional text-to-image generation.
pytest models/experimental/moss_image/tests --timeout=600 -k 'not 2-cards'

# Two connected, explicitly reserved P150s: per-card-count DiT and functional
# generation cases. Configure hardware visibility before invoking pytest.
pytest models/experimental/moss_image/tests --timeout=600 -k '2-cards'

# Full independent generation (both variants and card counts); this is a long
# manual qualification, not the short unit-test CI job.
MOSS_IMAGE_FULL_E2E=1 pytest models/experimental/moss_image/tests/test_integrated.py --timeout=7200
```

JUnit properties record PCC, relative RMS error and maximum absolute error for
each comparison. Numerical gates are PCC >= 0.99 and relative RMS <= 2% for
DiT, <= 3% for encoder/VAE. FP32 CFG/Euler requires PCC >= 0.999999 and relative
RMS <= 0.00001. Encoder observations must include all 32 layers and final norm;
tests cannot pass with an empty observation list. Component tests explicitly
replay captured inputs; integrated tests accept raw text and generate their own
conditioning and initial noise.

Integrated tests are **functional tests, not a CUDA final-image parity gate**.
The existing 50-step integrated images exhibit substantially more accumulated
error than the isolated components, especially SFT. This remains an open
qualification limitation and must be reported with the full-generation results.
Do not interpret passing component PCC or the smoke tests as proving final-image
numerical equivalence.

## Current pipeline integration

The entry is in `tests/pipeline_reorg/models_unit_tests.yaml`, using the current
`bh_p150` cloud-MLPerf functional SKU, tier 3, with a 20-minute job timeout.
`bh_p150b_civ2` is current too, but its existing tier-3 budget is fully consumed;
no shared budget was raised. Matrix generation, command lint and the repository
wide time-budget verifier are run against the current pipeline implementation.

Before enabling upstream, maintainers must provision the pinned assets and
assign the explicitly unassigned notification owner. No Slack identity is
borrowed from another model. Two discrete P150s are not represented by the
P300 SKU, so two-card cases remain explicit manual qualification until a
matching runner lane is agreed. A new native source build and real-card test
run are required after rebasing; old bring-up evidence alone does not qualify
the current branch.
