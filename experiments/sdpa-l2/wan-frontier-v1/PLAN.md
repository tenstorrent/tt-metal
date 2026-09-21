# Wan2.2 480p attention benchmark

## Fixed workload

- Wan-AI/Wan2.2-T2V-A14B-Diffusers, revision
  `5be7df9619b54f4e2667b2755bc6a756675b5cd7`.
- 832x480, 81 frames, 40 denoising steps, seed 42, guidance 4.0/3.0.
- Eight Blackhole devices on bh-lb-08, IRD 221619; mesh 2x4,
  sequence parallel 4 on axis 1, tensor parallel 2 on axis 0.
- Linear fabric, two links, dynamic expert weight loading, no FSDP.
- Converted-weight cache enabled; `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`.
- 32,760 logical video tokens (21x30x52); preserve stock padding/masking.

## Prompts

1. A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.
2. A woman sits at an outdoor café in soft daylight, turns toward the camera, smiles, and lifts a ceramic cup with both hands. Natural facial expressions, clearly visible hands, realistic skin texture, a steady camera.

The first is an existing Wan test prompt and exercises fine detail and subtle
motion. The second was selected to satisfy the user's human-subject requirement:
faces, hands, human motion and temporal consistency. It replaces the initially
proposed boxing-cats prompt. The in-progress stock smoke still uses the butterfly.

## Stock smoke

Use the existing test, without changing attention arithmetic:

```sh
WAN_CHECKPOINT=/path/to/pinned/Wan2.2-T2V-A14B-Diffusers \
python -m pytest -s -q --timeout=3600 \
  experiments/sdpa-l2/wan-frontier-v1/test_stock.py \
  -k 't2v and resolution_480p and bh_2x4_sp1tp0'
```

The wrapper imports the unchanged stock performance test, substitutes the
pinned local checkpoint path and limits host Torch threads to 16. The
pipeline constructor performs a two-step allocation/warmup run. The test
then generates the butterfly video with 40 steps and records encoder,
denoising, VAE and total pipeline time. Initial model setup is outside that
timed generation. The test also enforces repository performance thresholds;
report a threshold failure separately from successful video generation.

Stock is not identical to frozen A: stock uses ring SDPA with HiFi2,
BF16 destination and `exp_approx_mode=False`. Keep it as a separate control.

## Follow-up suite

The user authorized launch; see [STATUS.md](STATUS.md) for execution status.

Stock + D/C/B/E/F/G, two prompts, one common seed: 14 videos.
Keep F for this round to assess its long-context tradeoff.
Only replace video self-attention; keep cross-attention, text encoder,
projections, FFNs, scheduler and VAE fixed. Record video outputs, sampled-frame
CLIP, representative block timings and paired real-QKV accuracy checks.

Prerequisite: adapt the frozen kernels to Wan's sequence padding/masking and
SP4/TP2 configuration, and qualify that adapter. The existing FLUX adapter
rejects padded sequences and SP4/TP2; do not silently reuse it or drop padding
masks. Preserve compressed KV transport for E/F/G. Integration engineering
time is separate from measured benchmark execution time.

Stock smoke passed. See [REPORT.md](REPORT.md) for measurements and the
original follow-up execution budget. The full comparison is now complete;
see [suite results](suite-01/REPORT.md) and [findings](FINDINGS.md).

## Bring-up record

- Reservation extension succeeded on 2026-09-16; IRD capped the request at
  14 hours and reported approximately 14 hours remaining immediately afterward.
- Container DNS interrupted the Hugging Face download. Downloading through
  the host with `download_checkpoint.py` succeeded: 41 files, 126,199,274,206
  bytes, 406.7 seconds, with all LFS SHA256 hashes verified. TLS verification
  remained enabled and no host/container networking settings were changed.
- The stock wrapper collected exactly one selected test; both new Python
  files passed syntax compilation and Black (`--target-version py310`).
- Stock execution passed: 1 passed, 31 deselected, 622.76 seconds including
  cold conversion, compilation and allocation warmup. Timed video generation
  was 164.3348 seconds. Initial cache misses were expected for this new model;
  all three subsequent expert reloads were cache hits.
