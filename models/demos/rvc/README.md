# RVC Tenstorrent model bring-up

A bring up demo of RVC models in tenstorrent devices

## Prerequisites
- `git lfs` installed and initialized (`git lfs install`)
- Hardware: Tenstorrent N300


## Getting started
Run the following commands from the repository root (`/root/tt-metal`).

Install the package (editable is fine for local work):

```sh
uv pip install -e .
```

Install demo-specific dependencies:

```sh
uv pip install -r ./models/demos/rvc/requirements.txt
```

This repository expects model assets and configs to live outside the package.
Use the helper script to download them via Git LFS (requires `git lfs`):

```sh
chmod +x ./models/demos/rvc/assets-download.sh
./models/demos/rvc/assets-download.sh
```

Then set the required environment variables:

```sh
export RVC_CONFIGS_DIR="$PWD/models/demos/rvc/data/configs"
export RVC_ASSETS_DIR="$PWD/models/demos/rvc/data/assets"
```

The current demo uses fixed input audio files under:

- `models/demos/rvc/data/speech/sample-speech-0.wav`
- `models/demos/rvc/data/speech/sample-speech-1.wav`
- ...
- `models/demos/rvc/data/speech/sample-speech-7.wav`

Run inference using the helper script:

```sh
mkdir -p ./models/demos/rvc/data/output

# Single-audio run. The pipeline loads
# `models/demos/rvc/data/speech/sample-speech-0.wav`.
uv run --active models/demos/rvc/scripts/infer_ttnn.py -o ./models/demos/rvc/data/output/output_ttnn.wav

# torch version
# uv run --active models/demos/rvc/scripts/infer_torch.py -o ./models/demos/rvc/data/output/output_torch.wav

```

Expected layout after download:
- `RVC_CONFIGS_DIR` contains `v1/` and `v2/` config folders plus `hubert_cfg.json`.
- `RVC_ASSETS_DIR` contains `hubert.safetensors` and `pretrained/` weights.
- `models/demos/rvc/data/speech/` contains the fixed sample speech inputs and
  `sample-speech-transcript.txt`.


## Batched Processing

Batch input loading is available with `--batch-run`. This loads the hard-coded files
`sample-speech-0.wav` through `sample-speech-7.wav` from `models/demos/rvc/data/speech/`,
pads them to the longest input, and runs them as one batch:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --batch-run
```

On multi-device systems, use `--mesh-num-devices` to open a 1xN mesh and shard the
batch dimension across devices. For example, this runs the 8-file batch across 2
devices:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-run
```

For a replicated single input across 2 devices, pass an explicit batch size:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-size 2
```


## Testing

Running tests
```sh
# includes both pcc and perf tests
uv run --active pytest ./models/demos/rvc/tests
```

## Runner

For repeated fixed-shape benchmarking and validation, `rvc` provides:

- `models.demos.rvc.runner.performant_runner.RVCRunner`

This runner uses a fixed input audio path and fixed inference configuration to satisfy Cycode
security CI checks. The fixed path also keeps the performant runner shape stable for trace-oriented
benchmarking.

The runner supports data-parallel mesh execution when the caller passes a `ttnn.MeshDevice`.
The global batch size must be divisible by `device.get_num_devices()`. Runtime tensors are
split along batch dimension and outputs are concatenated back along batch dimension.

The runner is Tenstorrent-only. Torch reference generation and TT-vs-Torch comparison stay in the
separate eval flow instead of the runtime runner.

## Optimizations

The first optimization demonstrated in this demo is operation fusion in the TTNN
`Conv1d` wrapper:

- `models/demos/rvc/tt_impl/conv1d.py`

`Conv1d` accepts an optional `activation` argument, normalizes it through
`_normalize_conv2d_activation`, and forwards it into `ttnn.Conv2dConfig`. This lets
supported activations run as part of the convolution configuration instead of as a
separate TTNN op.

The relevant API surface is:

```python
Conv1d(..., activation=...)
```

and the fusion point is:

```python
ttnn.Conv2dConfig(
    ...
    activation=conv1d_config.activation,
)
```

The same wrapper also configures DRAM slicing for long sequence convolutions. RVC
changes sequence length throughout the pipeline because the audio representation is
repeatedly downsampled and upsampled. For longer intermediate lengths, the underlying
`ttnn.conv2d` call used to implement `Conv1d` may need to slice the operation through
DRAM instead of processing the full width in one kernel launch.

`models/demos/rvc/tt_impl/conv1d.py` keeps per-shape slicing parameters in
`PARAMS_TO_CONFIG_VALUES`, computes the required number of slices in
`get_conv2d_config_values`, and builds a `ttnn.Conv2dSliceConfig` when slicing is
needed:

```python
ttnn.Conv2dSliceConfig(
    num_slices=slice_num,
    slice_type=ttnn.Op2DDRAMSliceWidth,
)
```

HuBERT also switches activations into L1 once the sequence has been downsampled
enough to fit efficiently. See `models/demos/rvc/tt_impl/vc/hubert.py`, where the
feature extractor moves `x` to `ttnn.L1_MEMORY_CONFIG` at line 232 before the later
convolution layers:

```python
if i == 3:
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
```

The VITS text encoder uses relative-position multi-head attention. See
`models/demos/rvc/tt_impl/vc/synthesizer.py`, where `MultiHeadAttention` starts at
line 85. The TTNN implementation avoids a padding-heavy relative-position attention
conversion by generating device-side position index tensors and a zero mask, then
using `ttnn.scatter` to map relative positional logits into absolute attention
positions. This keeps the positional embedding path in TTNN and avoids doing the
relative-to-absolute attention transform through host-side padding logic.

The relevant methods are:

```python
_get_absolute_position_index_and_zero_mask(...)
_relative_to_absolute_position(...)
_absolute_to_relative_position(...)
```

The performant runner also uses trace capture and replay to reduce repeated
dispatch overhead for stable input shapes. Enable this path with:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner
```

On the fixed sample input of `1,905,042` samples, tracing reduced average inference
time from `5.544265s` to `4.294533s`. The corresponding RTF improved from
`0.046565` to `0.036069` for the same `119.065125s` output duration.

Mesh-device data parallelism is another throughput optimization. With
`--mesh-num-devices 2`, the batch dimension is sharded across two devices and the
outputs are concatenated back on host. This keeps per-sample latency in the same
range while doubling aggregate throughput for batch workloads:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-size 2
```

## Demo: Tracy Perf Capture

- Run the Tracy-enabled perf test to extract a CSV perf sheet:

```bash
python -m tracy -r -n rvc_e2e -m pytest models/demos/rvc/tests/perf/test_e2e_performant.py::test_rvc_e2e_no_sync
```

## Evaluation

Start with a plain TTNN run before enabling any evaluation flags. This verifies that the
model loads, runs on the Tenstorrent device, and writes converted audio.

```sh
mkdir -p ./models/demos/rvc/data/output
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  -o ./models/demos/rvc/data/output/output_ttnn.wav
```

The expected result is a playable `output_ttnn.wav` file and printed runtime metrics
such as average inference time, output duration, and real-time factor (RTF).

For batch and data-parallel checks, use:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-run
```

This loads all 8 fixed speech samples from `models/demos/rvc/data/speech/`, shards
the batch across 2 devices, and reports the output shape, batch size, input sample
count, average runtime, output duration, and RTF.

After the plain run succeeds, check token-level accuracy against the PyTorch reference and WER

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  -o ./models/demos/rvc/data/output/output_ttnn.wav \
  --check-torch-token-accuracy \
  --compute-wer
```

This command runs the TTNN pipeline, generates a PyTorch reference in validation mode,
checks waveform PCC, reports token-level content accuracy, transcribes the output
with Whisper, and computes WER against
`models/demos/rvc/data/speech/sample-speech-transcript.txt`. The targets are:

`token_accuracy > 0.95`

`wer < 2.5`

## Speaker Embedding

The original target speaker reference voice clips are not publicly available.
Those speakers are only available through `speaker_id` as can be seen in the original RVC project. So, for the time being, it is not possible to compute the similarity between speaker embeddings of reference clip and output audio.

## Validation logs
You can find them attached to the relevant PR.



## Attributions
Heavily inspired by RVC (Retrieval-based Voice Conversion) and please see the original repo
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
