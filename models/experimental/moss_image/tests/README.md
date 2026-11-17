# MOSS-Image validation

Tests use repository `mesh_device` and `device_params` fixtures. Model code never
opens or closes their devices. Integrated tests reserve 128 KiB L1-small for
the full-resolution VAE; the corrected SFPU encoder recurrence runs within the
same reservation. Run from the TT-Metal checkout with its freshly
built native TTNN and the model runtime requirements installed. Use Python 3.11
or newer for both TTNN and the model; Pillow 13 does not support Python 3.10.

## Assets

Provision this read-only directory for local replay and integrated validation:

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

`MOSS_IMAGE_ASSET_ROOT` overrides the asset root. Asset-dependent tests skip
when their pinned CUDA reference or checkpoint is absent, including under CI;
no opt-out environment variable is needed. Available manifests must pass their
digest and configuration checks even when the checkpoint is absent. Corrupt
manifests, missing or altered referenced tensors, and checkpoint mismatches
remain failures. A skip means the corresponding device path was not qualified.
The integrated test currently uses the same pinned capture to verify checkpoint
hashes; it also skips without that capture, although generation itself does not
consume captured activations. Host, attention and CPU-reference scheduler tests
do not need these assets.

## CPU references without CUDA captures

The attention and CFG/Euler tests compare real TT outputs against references
computed by PyTorch on CPU. They require reserved TT hardware and a matching
native TTNN build, but no checkpoint download, saved activations or CUDA runtime.
They do not provide a full-model CPU generation backend.

- `test_attention.py`: dense FP64 PyTorch attention, plus the BF16 probability
  rounding policy; covers causal GQA, masked MHA and per-head GQA masks.
- `test_scheduler_cpu.py`: CPU CFG and the pinned vendored MOSS/diffusers
  scheduler at steps 0, 24 and 49 of the 50-step schedule. Cases cover CFG 1/3,
  BF16 prediction promotion to FP32, FP32 predictions, and a padded token count.
  Both guided velocity and updated latents must meet PCC >= 0.999999 and relative
  RMS <= 0.00001. Both replicas are checked on two cards; this does not exercise
  tensor-parallel collectives.

```bash
# One reserved P150; references are calculated on the host CPU.
pytest models/experimental/moss_image/tests/test_attention.py \
  models/experimental/moss_image/tests/test_scheduler_cpu.py \
  --timeout=120 -k '1-card'

# Repeat on an explicitly reserved connected pair.
pytest models/experimental/moss_image/tests/test_attention.py \
  models/experimental/moss_image/tests/test_scheduler_cpu.py \
  --timeout=120 -k '2-cards'
```

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
# manual qualification, not a scheduled CI job.
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

## CI scope

This contribution does not register a scheduled model-unit-test CI job. CUDA
reference generation and asset provisioning remain manual validation steps.
Tests remain available for explicit runs using the commands above. Repository
workflows, shared dependency pins and other models' CI registrations are unchanged.
Two-card cases require an explicitly reserved connected pair; two discrete
P150s are not represented by the P300 SKU.

A new native source build and real-card test run are required after rebasing;
old bring-up evidence alone does not qualify a changed runtime.
