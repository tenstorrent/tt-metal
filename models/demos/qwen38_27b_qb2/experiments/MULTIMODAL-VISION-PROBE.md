# Prepared vision-only hardware probe

**Prepared, not launched.** Schedule this only after the active profiling owner
releases the Galaxy. It opens one TP4 mesh on `10.228.203.98`; it does not start
serving, load the 27B language model, reset devices or change an installation.
The current feature has passed CPU processing gates, not native vision parity.

## Frozen inputs

- Source: `/home/ttuser/qwen38-artifacts-20261007/multimodal-model-source-v4`.
- Manifest: `multimodal-vision-probe-source-manifest-v1.json` in that artifacts
  directory, also copied into `multimodal-evidence/` beside this document.
- Manifest SHA256:
  `e7efa3b257c7ce8b2462cec10335cfe70416645eca56b5d362346904ca65a21f`.
- The manifest covers 427 Python files, including the probe, Qwen3.8 runtime,
  reused vision/transformer dependencies and root fixtures. It was verified on
  the host and against this worktree. It excludes generated caches and unit tests.
- Native library: `a08819ddbe23077f8037d3802303939064868ff6` from
  `/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal` and its
  corresponding build/install directories.
- Python: `plugin-multimodal-env-v1/bin/python` under the artifacts directory,
  with Torch `2.11.0+cpu`, Transformers `5.12.1`, vLLM `0.26.0+empty`.
- Weights: `checkpoint-pinned-1d4bf0f2` under the artifacts directory. The probe
  reads only its vision parameters. Language-model constructors are guarded.

The captured CPU results cover these exact runtime source bytes. This is a
host-local experiment recipe, not a portable release or immutable container.

## Probe and bounds

`tests/vision_reference_probe.py` requires an explicit `--run-device` to open a
mesh. `--verify-only` checks source hashes without importing TTNN. Device runs
take the shared exclusive nonblocking `/tmp/tt-device.lock`; an occupied lock
fails immediately. A new output directory is mandatory. The process records
partial JSON results after every case and closes its mesh on ordinary failure.

The initial probe runs three small cases: one ragged image, two differently
sized images, and two video frames. It compares complete native vision outputs
to the selectively loaded HF reference with PCC >= 0.99 and normalized RMS error
<= 0.10. The video case perturbs the second frame and requires the first frame's
output to remain within 0.001. These are initial numerical smoke gates; they do
not qualify semantic accuracy or change existing model-evaluation thresholds.
The optional `--include-aligned-case` adds an exactly 2,048-patch image only after
the initial cases pass.

The persistent service below has a 15-minute runtime limit, 30-second stop
timeout, 32 GiB host-memory limit and four-core CPU quota. If it times out or
fails, inspect the recorded error before scheduling another run. No recovery or
reset is part of this recipe. A forced timeout may prevent final JSON cleanup;
retain the systemd exit status and log alongside the partial result.

## Launch after hardware scheduling

Run on `10.228.203.98` as `ttuser`. These commands have been prepared but **not
executed**. Use a fresh name so previous receipts remain intact.

```bash
set -euo pipefail
artifact_root=/home/ttuser/qwen38-artifacts-20261007
task_root=/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006
source_root="$artifact_root/multimodal-model-source-v4"
probe_script="$source_root/models/demos/qwen38_27b_qb2/tests/vision_reference_probe.py"
probe_manifest="$artifact_root/multimodal-vision-probe-source-manifest-v1.json"
probe_name=qwen38-mm-vision-probe-v1
probe_output="$artifact_root/$probe_name"

test ! -e "$probe_output"
"$artifact_root/plugin-multimodal-env-v1/bin/python" "$probe_script" \
  --checkpoint "$artifact_root/checkpoint-pinned-1d4bf0f2" \
  --manifest "$probe_manifest" --verify-only

systemd-run --user --unit="$probe_name" --collect \
  --property=RuntimeMaxSec=15min \
  --property=TimeoutStopSec=30s \
  --property=KillMode=control-group \
  --property=MemoryMax=32G \
  --property=CPUQuota=400% \
  --property="WorkingDirectory=$source_root" \
  --property="StandardOutput=append:$artifact_root/$probe_name.log" \
  --property="StandardError=append:$artifact_root/$probe_name.log" \
  --setenv="TT_METAL_HOME=$task_root/metal" \
  --setenv="LD_LIBRARY_PATH=$task_root/metal-install/lib:$task_root/metal-build/lib" \
  --setenv="PYTHONPATH=$source_root:$task_root/metal:$task_root/metal/ttnn:$task_root/metal/tools" \
  --setenv=ARCH_NAME=blackhole \
  --setenv=TT_METAL_INSPECTOR_RPC=0 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=HF_HUB_OFFLINE=1 \
  --setenv="QWEN_VISION_CACHE_DIR=$artifact_root/multimodal-vision-cache-v1" \
  "$artifact_root/plugin-multimodal-env-v1/bin/python" -u "$probe_script" \
  --checkpoint "$artifact_root/checkpoint-pinned-1d4bf0f2" \
  --manifest "$probe_manifest" --output "$probe_output" --run-device
```

The service persists across SSH disconnection under the existing user service
manager. After it ends, preserve `$probe_output/result.json`, its sibling log
and `journalctl --user -u "$probe_name" --no-pager`. A passing result permits
planning visual prefill-logit and mixed request tests; it does not establish
that the four original endpoint checks pass.
