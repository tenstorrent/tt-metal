# UNIAD

## Prerequisite

UniAD relies on the standard tt-metal Python environment (`./create_venv.sh`).
The host-side DCNv2 reference is implemented via `torchvision.ops.deform_conv2d`,
so no additional MMCV / Casadi install is required.

### Pretrained weights
- The pre-trained weights will be downloaded automatically during sub-module testing.
- If the weights are not downloaded automatically, you can manually fetch them using the command, `wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth`. Place it in the following path `models/experimental/uniad/`.

## Testing

Make sure `TT_METAL_HOME` points at this checkout and its Python env is active,
otherwise device init fails while building dispatch kernels.

Run everything (per-submodule PCC tests + the device DCN unit test):

```
pytest models/experimental/uniad/tests/
```

End-to-end (full UniAD; asserts planning `sdc_traj` and the seg head outputs vs
the PyTorch reference on the real BEV embedding):

```
pytest models/experimental/uniad/tests/pcc/test_ttnn_uniad.py::test_uniad
```

A single submodule (one file per submodule lives under `tests/pcc/`), e.g.:

```
pytest models/experimental/uniad/tests/pcc/test_ttnn_encoder.py
```

Notes:
- `tests/pcc/` — per-submodule PCC tests against the PyTorch reference.
- `tests/unit/` — isolated custom-op tests (currently the device modulated deformable conv).
- `test_ttnn_pan_segformer_head.py` is a structural smoke test only; the seg head's
  accuracy is asserted in the e2e on the real BEV embedding (random input makes its
  deformable-attention PCC meaningless).

## Performance

Warm-iteration wall time on Blackhole p150b (`test_ttnn_uniad.py::test_uniad`,
`TT_UNIAD_TIMING=1 TT_UNIAD_WARM_ITERS=2`, measured on `call#3`):

| Phase                              |  ms |
| ---------------------------------- | --: |
| `extract_img_feat` (ResNet101+FPN) | 394 |
| `get_bev_features` (BEV encoder)   | 392 |
| `get_detections` (DETR decoder)    |  88 |
| `simple_test_track` (subtotal)     | 951 |
| `seg_head.forward_test`            | 256 |
| `motion_head.forward_test`         | 122 |
| `planning_head.forward_test`       |  36 |
| `occ_head`                         |   1 |
| **forward total**                  | **~1367** |

Per-run noise is ±2-3 % on each phase. PCC `sdc_traj` ≥ 0.99 (gate
floor). The committed `tests/perf_baseline.json` pins each phase to
±5 % / 20 ms abs floor and asserts inter-warm variance under 3 %
whenever `TT_UNIAD_TIMING=1 TT_UNIAD_WARM_ITERS≥1`; the e2e test
fails on a regression.

### First-run startup cost

UniAD instantiates ~1143 unique ttnn kernels. On the **first run on a
host with an empty JIT cache** (`~/.cache/tt-metal-cache/<fingerprint>/`
missing) each kernel needs a fresh g++ compile against the
`riscv-tt-elf` toolchain, so a fresh start takes around **5–7 minutes
of wall time** on a 32-CPU box (≈445 s measured on an AMD EPYC 8124P,
16 phys × 2 SMT). Effective compile parallelism caps at ~5–10
g++ invocations because ttnn ops are dispatched serially from Python
and only the 5 RISC core builds × per-RISC srcs are parallelised
within a single op; ~3 cores end up busy on average regardless of how
many CPUs the host has.

Subsequent runs on the same host hit a populated JIT cache and start
in **~9 s** (the first forward pass through the warm-cache cold
path). Warm iterations after that run at the wall numbers in the
table above (~1.4 s per `forward_test`).

For production / CI deployments where the 5–7 min first-run cost is
prohibitive, the standard mitigations are:

- **Process keepalive** — keep the inference process running so the
  build cache stays warm across requests.
- **Cache bundling** — ship the populated
  `~/.cache/tt-metal-cache/9009805869222971872/` (≈2.9 GB) inside the
  container/release artifact so deployment boots at the 9 s mark.
- **Programmatic warmup** — call `ttnn_model.warmup(*args, n_iters=2)`
  on `TtUniAD` before the user-facing inference loop to absorb the 9 s
  first-call dispatch setup and prime trace replay. The method logs
  `[warmup] iteration N/N — populating ttnn caches…` so a 9 s pause
  isn't silent. Used by `test_ttnn_uniad.py::test_uniad` whenever
  `TT_UNIAD_WARM_ITERS≥1`.

### Modulated deformable convolution

The ResNet101 backbone runs 26 modulated deformable convs in
`layer3`/`layer4`. By default this routes through a device-side
`TtModulatedDeformConv2dDevice` (built on `ttnn.grid_sample`), which
removes ~3.5 sec of per-forward host CPU compute vs. the host
fallback. Set `TT_DCN_DEVICE=0` to switch back to the host
path (`torchvision.ops.deform_conv2d`) for numerical bisection.

### Environment variables

| Variable                  | Default      | Purpose |
| ------------------------- | ------------ | ------- |
| `TT_DCN_DEVICE`           | `1` (device) | Route modulated deformable conv through the device path. Set `0` for host CPU fallback. |
| `TT_DCN_TIMING`           | unset (off)  | Print per-forward DCN-path breakdown (transfer / cpu / back / device). Diagnostic only. |
| `TT_UNIAD_TIMING`         | unset (off)  | Print a per-sub-phase wall-clock breakdown of `TtUniAD.forward_test`. Enables the per-phase perf gate in `test_ttnn_uniad.py`. No-op when unset. |
| `TT_UNIAD_WARM_ITERS`     | `0`          | Test-side: extra warm-up forward passes (with `reset_test_state()` between them) before the PCC-asserted call. Use `≥1` so timing lands on a warm cache. |
| `TT_UNIAD_PERF_GATE`      | `1` (on)     | Silence the per-phase regression gate while keeping timing prints. Only consulted when `TT_UNIAD_TIMING=1` AND `TT_UNIAD_WARM_ITERS≥1`. |
| `TT_UNIAD_DISABLE_TRACE`  | unset (off)  | Bypass the BEV/DETR trace-replay capture. Required when running Tracy's device profiler, which is incompatible with `begin_trace_capture`. |

Note:
- Raised issue for fallback torch ops and added the issue links to resp
