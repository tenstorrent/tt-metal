# VoxCPM2 experimental TTNN components

This directory contains a prototype TTNN component port of [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). The native reference source is pinned to OpenBMB/VoxCPM revision `f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`; the tested checkpoint is `openbmb/VoxCPM2` revision `32279effe8c19989596f05d353d1447f51d9e915`. Checkpoint dimensions come from its own `config.json`. The reference code is distributed under [Apache-2.0](https://github.com/OpenBMB/VoxCPM/blob/f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629/LICENSE), and the external weights remain subject to the model card's terms.

## Implementation status

This is component bring-up, not qualified integrated TT text-to-speech generation. Native CUDA generation supplies captured inputs and outputs for individual TTNN components. In particular, TT waveform replay decodes CUDA-generated latents; it does not demonstrate TT text-to-speech synthesis. AR KV-cache decode, the diffusion solver, tokenizer integration, the stop loop and complete synthesis remain pending.

The current numerical gate combines PCC with explicit relative RMS error limits. The decoder must meet relative RMS error ≤ 3%; the historical approximately 7.6% result does not pass this gate. The corrected codec uses FP32 activation/output storage, nonlinear operations and residual additions, with BF16 convolution inputs and weights and FP32 accumulation. See the [accuracy report](validation/RESULTS.md) for source-build qualification; the historical measurements below use the old runtime and precision policy.

The model contains base and residual MiniCPM language models, a local audio-patch encoder, finite scalar quantizer, local diffusion transformer, and AudioVAE V2. Learned network operations execute on TT; text tokenization, checkpoint loading and reproducible input-noise creation are host boundaries. Audio quality, voice cloning, streaming, multilingual behavior and steady-state performance remain unqualified beyond the recorded component checks. See [DESIGN.md](DESIGN.md) and the [accuracy report](validation/RESULTS.md).

## Components and historical measurements

The table below preserves the original 2026-10-01 PyTorch 2.8 qualification using an installed `ttnn==0.65.1rc17.dev6333+h3.3` wheel. These are historical component measurements, not results from a build of the submitted source tree. The separate [PyTorch 2.10 report](validation/RESULTS.md#pytorch-210-compatibility--2026-10-07) records fresh CUDA captures and 16 replay cases on that same old TTNN wheel; it also does not establish current-source compatibility.

| Component | Implementation | Historical real-device CUDA PCC |
| --- | --- | --- |
| MiniCPM attention, RMSNorm, SwiGLU, residual blocks and final norm | Full-sequence TTNN forward | Covered by prefill checks |
| Base and residual LM | Embedding-input prefill; KV-cache AR decode pending | Base 0.9999093; residual 0.9999660 |
| Local patch encoder | Learned special token and noncausal transformer | 0.9999862 |
| Finite scalar quantization | TTNN inference rounding and projections | 0.9999977 |
| Local DiT V2 | mu / timestep / condition / target sequence and velocity output | 0.9999425–0.9999570 across three timesteps |
| AudioVAE V2 | Deterministic nonstreaming encode/decode graph | Encoder 0.9991306; decoder 0.9997781 |
| Integrated synthesis | Not implemented | Unqualified |

The current transformer cases use BF16 storage with HiFi4 and FP32 destination accumulation where supported. The current codec cases select `--dtype float32`: activations, nonlinear operations, residuals and outputs use FP32, while convolution inputs and weights use BF16 with FP32 accumulation. Long FP32 transpose convolutions use explicit DRAM width slices to bound L1 allocations. Explicit QK/softmax attention can differ from CUDA fused SDPA. Native BF16 timestep rounding is preserved before trigonometry. Codec replay uses 32-row activation blocks and a 256 KiB L1-small reservation. The original waveform comparison had 7.50% relative RMS error, and the PyTorch 2.10 comparison had 7.64%, despite excellent PCC; correlation alone cannot establish amplitude accuracy.

Unsupported codec options fail explicitly: streaming, noise blocks, concatenated sample-rate conditioning and stride-one transpose blocks. Replay accepts a single sample rate per batch. MiniCPM cache outputs and `forward_step` are captured by the CUDA harness but not replayed or qualified on TT.

## Environment and weights

Run the commands below from the repository root. Build TTNN from this checkout using the [repository build instructions](../../../INSTALLING.md), activate the resulting repository environment, then install the model replay dependencies:

```bash
uv pip install -r models/experimental/voxcpm2/requirements.txt
```

The runtime requirements allow the repository's Torch version (`torch>=2.10`) and constrain NumPy below 2. They do not install the native CUDA reference package or downgrade CI Torch to match the oracle. The model does not maintain a nested standalone project or dependency lock; upstream CI uses the repository environment and its source-built TTNN.

For an independent development environment, `uv venv "$HOME/data/voxcpm2/tt-venv"` followed by `uv pip install --python "$HOME/data/voxcpm2/tt-venv/bin/python" -r models/experimental/voxcpm2/requirements.txt` installs model dependencies without creating a nested project. Install the TTNN wheel built from this checkout into that environment as well, retaining its dependency constraints and the build's required source/kernel configuration. Use that environment's Python for replay; the repository environment is still required for upstream pytest fixtures and CI dependencies.

Keep native CUDA capture in a separate CUDA environment. Its [reference requirements](requirements-reference.txt) pin Torch/Torchaudio 2.10.0 and the official source revision. Choose a compatible CUDA wheel build for the host; the stored reference was generated with Torch/Torchaudio `2.10.0+cu128`.

```bash
uv venv --python 3.12 "$HOME/data/voxcpm2/reference-venv"
uv pip install --python "$HOME/data/voxcpm2/reference-venv/bin/python" \
  -r models/experimental/voxcpm2/requirements-reference.txt
```

Prepare a local snapshot of the pinned checkpoint, outside Git and preferably on the data disk. For example, after installing reference dependencies:

```bash
"$HOME/data/voxcpm2/reference-venv/bin/python" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="openbmb/VoxCPM2", revision="32279effe8c19989596f05d353d1447f51d9e915", local_dir=os.path.expanduser("~/data/voxcpm2/checkpoint"), allow_patterns=["*.json", "*.py", "*.safetensors", "*.pth"])'
```

Hugging Face credentials remain in the user's standard environment or cache. For offline replay, provide a complete local snapshot; capture and replay do not download missing weights. Captures record checkpoint hashes, and replay only selects weights recorded in the verified manifest. Missing or changed recorded files fail instead of falling back to unrecorded weights.

## CUDA capture and individual TT replay

A fresh capture can be produced with native CUDA generation:

```bash
"$HOME/data/voxcpm2/reference-venv/bin/python" \
  -m models.experimental.voxcpm2.validation.capture_reference \
  --checkpoint "$HOME/data/voxcpm2/checkpoint" \
  --checkpoint-revision 32279effe8c19989596f05d353d1447f51d9e915 \
  --device cuda:0 --seed 42 --max-len 32 --inference-timesteps 10 \
  --text "The quick brown fox jumps over the lazy dog." \
  --output "$HOME/data/voxcpm2/cuda-reference"
```

The capture stores `reference.wav`, source/device/version metadata, seeds, RNG states, valid KV prefixes, latents and checksummed tensors. Missing/incomplete captures, non-CUDA oracles, modified checkpoint bytes, nonfinite tensors, shape mismatches and missing outputs fail validation.

After reserving the TT device, replay a nonzero local-encoder event with the active source-built TTNN environment:

```bash
timeout 600 python -m models.experimental.voxcpm2.validation.replay_component \
  --reference "$HOME/data/voxcpm2/cuda-reference" \
  --checkpoint "$HOME/data/voxcpm2/checkpoint" \
  --component feat_encoder.forward --event-index 1 --device-id 0 \
  --dtype bfloat16 --l1-small-size 262144 --min-pcc 0.99 \
  --max-relative-rms 0.02 \
  --output "$HOME/data/voxcpm2/encoder-pcc.json"
```

The CLI supports base/residual prefill hidden outputs, local encoder, quantizer, local DiT estimator, LM/DiT/fusion/stop projections, stop activation and codec encode/decode. It reports PCC, relative RMS error, maximum absolute error and cold launch time. Cold launch time may include compilation and is not a steady-state benchmark. Codec replay also saves a `.pt` waveform/latent tensor beside the JSON report. Decoding the captured CUDA latents on TT is a paired codec comparison, not integrated synthesis.

## Complete component accuracy suite

The pytest gate in [test_component_device.py](tests/test_component_device.py) runs 16 explicit cases defined in [validation/cases.py](validation/cases.py): base/residual prefill, local encoder and FSQ event 1, three local-DiT events (0, 4, 8), projections and stop components, codec decoding from the text capture and codec encoding from the separate voice-reference capture. The small stop-head output uses maximum absolute error as well as PCC.

| Component group | Minimum PCC | Maximum relative RMS error | Additional bound |
| --- | --- | --- | --- |
| LMs, local encoder, FSQ, DiT and projections | 0.99 | 0.02 | — |
| Codec encoder | 0.99 | 0.05 | — |
| Codec decoder | 0.99 | 0.03 | Historical BF16 result fails; current mixed-precision case must pass |
| Stop head | 0.99 | 0.02 | Maximum absolute error 0.001 |

The suite's default asset root is `/mnt/MLPerf/tt_dnn-models/openbmb/VoxCPM2`. These assets require maintainer provisioning; the test does not download or generate an oracle.

| Asset | Default directory below the asset root | Optional relocation variable |
| --- | --- | --- |
| Checkpoint | `checkpoint` | `VOXCPM2_CHECKPOINT` |
| Text CUDA capture | `cuda-reference-torch210-20261007` | `VOXCPM2_CUDA_CAPTURE` |
| Voice CUDA capture | `cuda-reference-with-ref-torch210-20261007` | `VOXCPM2_VOICE_CUDA_CAPTURE` |

The text capture's `manifest.json` SHA256 must be `72a5bbbc9deb2b2f517bde44309bbdf978986d759a4625d6a29f49f98004d782`; the voice capture's must be `971ea6c97fe5601f21442b1bb3e6a3a981b04c2c52767f0fd79b863f0fde3393`. Relocation variables point to these exact captures. Use the individual replay CLI for newly generated or otherwise different captures.

```bash
# Optional for local copies of the exact pinned assets:
export VOXCPM2_CHECKPOINT="$HOME/data/voxcpm2/checkpoint"
export VOXCPM2_CUDA_CAPTURE="$HOME/data/voxcpm2/cuda-reference-torch210-20261007"
export VOXCPM2_VOICE_CUDA_CAPTURE="$HOME/data/voxcpm2/cuda-reference-with-ref-torch210-20261007"

# Fail on missing assets, as required in CI. Device ID follows repository pytest.
export VOXCPM2_REQUIRE_DEVICE_TESTS=1
pytest models/experimental/voxcpm2/tests --device-id 0 --timeout=600
```

Absent assets skip the device cases in ordinary local runs; with `VOXCPM2_REQUIRE_DEVICE_TESTS=1`, they fail. A manifest hash mismatch fails in either mode. Tests request the repository device fixture only after asset checks; the fixture owns device selection and teardown, with `l1_small_size=262144`. There is no CPU fallback for device results.

The model is registered in [the model unit-test matrix](../../../tests/pipeline_reorg/models_unit_tests.yaml) for Blackhole P150, tier 3. The registered job timeout is 10 minutes and fits the existing tier budget. See the accuracy report for local complete-suite execution results; this does not constitute an upstream CI run. Asset provisioning and assignment of a notification owner remain maintainer tasks. A missing owner is left unassigned rather than notifying an unrelated contributor.

## Decoder diagnostics and host checks

The stage probe compares each recorded CUDA decoder stage against both an isolated TT replay and a cumulative TT chain, reporting gain/offset fits and residual error. Fits are diagnostic only; the probe does not rescale TT outputs for acceptance.

```bash
timeout 600 python -m models.experimental.voxcpm2.validation.replay_codec_stages \
  --reference "$HOME/data/voxcpm2/cuda-reference-torch210-20261007" \
  --checkpoint "$HOME/data/voxcpm2/checkpoint" \
  --device-id 0 --dtype float32 \
  --output "$HOME/data/voxcpm2/decoder-stages.json"

# Host contracts do not exercise the parametrized device cases.
python -m unittest discover \
  -s models/experimental/voxcpm2/tests -p 'test_*.py' -v
python -m compileall -q models/experimental/voxcpm2
git diff --check
```

Host checks cover utility/configuration contracts and validation integrity; they do not substitute for native CUDA generation, current-source TTNN replay or passing the complete numerical gate. Historical “30 configured tests passed” statements refer to the old unittest setup with one selected device replay, not the current parametrized 16-case suite.
