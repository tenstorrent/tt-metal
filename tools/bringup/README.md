<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Bringup op-level tracing and parity tests

This directory contains a two-phase workflow for capturing per-op golden tensors
from a PyTorch model and replaying them against TTNN to check parity (PCC).

- **Phase 1** (`phase1_record_ops.py`): run the model, hook the interesting
  leaf modules, and record a `manifest.json` plus a directory of `.pt` tensor
  artifacts.
- **Phase 2** (`phase2_generate_tests.py` + `tracer_test_harness.py`): read the
  manifest, replay each op on a TT device, and compare against the recorded
  golden tensors.
- `validate_trace_manifest.py`: a fail-fast preflight that validates a manifest
  (schema, artifact existence, shape consistency) before launching a test run.

## Artifact path contract

All artifact paths in a manifest (`in_path`, `out_path`, `w_path`, `b_path`) are
**resolved relative to the directory that contains the manifest file**.

- Relative paths (the default, e.g. `tensors/00000_conv_in.pt`) are resolved as
  `<manifest_dir>/<path>`.
- Absolute paths are used verbatim. They are supported as an unambiguous escape
  hatch, but manifests produced by Phase 1 always use relative paths so an
  artifact directory can be moved or copied alongside its manifest.

There is no repo-root guessing, environment-variable override, or parent-directory
search. Resolution is exactly:

```python
def resolve_artifact_path(manifest_path, artifact_path):
    p = Path(artifact_path)
    if p.is_absolute():
        return p
    return Path(manifest_path).resolve().parent / p
```

Phase 1 writes the manifest to `<out_dir>/manifest.json` and the tensors under
`<out_dir>/tensors/`, so the stored `tensors/...` paths are manifest-relative by
construction.

## Manifest schema

```jsonc
{
  "input_shape": [B, C, H, W],   // input to the traced forward pass
  "num_records": N,              // must equal len(records)
  "records": [
    {
      "idx": 0,                  // must match the record's position in the list
      "name": "encoder.block1.conv1",
      "kind": "Conv2d",          // module type recorded by the tracer
      "params": { /* op-specific args, e.g. Conv2d kernel/stride/... */ },
      "in_shape":  [n, c, h, w], // 4D NCHW
      "out_shape": [n, c, h, w], // 4D NCHW
      "in_path":  "tensors/00000_encoder.block1.conv1_in.pt",
      "out_path": "tensors/00000_encoder.block1.conv1_out.pt",
      "w_path":   "tensors/00000_encoder.block1.conv1_w.pt",  // or null
      "b_path":   "tensors/00000_encoder.block1.conv1_b.pt"   // or null
    }
  ]
}
```

## Canonical example manifest

The following `manifest.json` sits next to a `tensors/` directory. All artifact
paths are manifest-relative:

```
bringup/artifacts/phase1/
├── manifest.json
└── tensors/
    ├── 00000_conv_in.pt
    ├── 00000_conv_out.pt
    ├── 00000_conv_w.pt
    ├── 00000_conv_b.pt
    ├── 00001_relu_in.pt
    └── 00001_relu_out.pt
```

```json
{
  "input_shape": [1, 3, 32, 32],
  "num_records": 2,
  "records": [
    {
      "idx": 0,
      "name": "conv",
      "kind": "Conv2d",
      "params": {
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": [1, 1],
        "groups": 1,
        "bias": true
      },
      "in_shape": [1, 3, 32, 32],
      "out_shape": [1, 4, 32, 32],
      "in_path": "tensors/00000_conv_in.pt",
      "out_path": "tensors/00000_conv_out.pt",
      "w_path": "tensors/00000_conv_w.pt",
      "b_path": "tensors/00000_conv_b.pt"
    },
    {
      "idx": 1,
      "name": "relu",
      "kind": "ReLU",
      "params": { "inplace": false },
      "in_shape": [1, 4, 32, 32],
      "out_shape": [1, 4, 32, 32],
      "in_path": "tensors/00001_relu_in.pt",
      "out_path": "tensors/00001_relu_out.pt",
      "w_path": null,
      "b_path": null
    }
  ]
}
```

## Usage

```bash
# Phase 1: record ops + tensors (manifest written to <out>/manifest.json)
python tools/bringup/phase1_record_ops.py --input-shape 1 3 32 32 --out bringup/artifacts/phase1

# Validate the manifest before running tests
python tools/bringup/validate_trace_manifest.py --manifest bringup/artifacts/phase1/manifest.json
python tools/bringup/validate_trace_manifest.py --manifest bringup/artifacts/phase1/manifest.json --print-resolved 5

# Phase 2: generate a pytest file (and the harness) from the manifest
python tools/bringup/phase2_generate_tests.py \
    --manifest bringup/artifacts/phase1/manifest.json \
    --out-test bringup/artifacts/phase1/test_op_pcc.py \
    --write-harness
```
