# ACE-Step DiT matmul perf tuning (May 2026)

## Baseline (denoise loop, 1 Euler step, CFG batch=2)

From `out/perf.log` / `tt-perf-report` on device:

| Op | Device time | Ops | % |
|---|---:|---:|---:|
| **MatmulDeviceOperation** (`in0:l1_interleaved`) | ~24 ms | 445 | **~49%** |
| ReshapeViewDeviceOperation | ~5.8 ms | 305 | ~12% |
| LayerNormDeviceOperation | ~3.1 ms | 338 | ~6% |
| NLPConcatHeadsDeviceOperation | ~1.9 ms | 96 | ~4% |

Matmul rows already show **LoFi BF16 × BFP8 => BF16** with **L1 activations** and **DRAM weights** (intended policy).

### Example hot matmul (from report annotations)

`b={2} x 32 x 2048 x 2048` — fused M=2×1 tiles, ~54 μs, ~233 GB/s DRAM (~45% BW util), ~10 TFLOPs (~2% compute util).
Large-M bucket `2816×1536×2048` is dominated by im2col/VAE-style paths, not DiT patch seq.

## Memory policy (production default)

| Tensor | Placement |
|---|---|
| **in0** (activations) | **L1** interleaved (when 1D-mcast `program_config` active) |
| **in1** (weights) | **DRAM** (`ace_step_dit_weight_memory_config`) |
| **output** | **L1** when `program_config` set |

Compute:

```python
WormholeComputeKernelConfig(
    math_fidelity=LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,  # matrix_engine.md / user ref
)
```

Weights: `bfloat8_b` for linears (`ace_step_linear_weight_dtype`).

## Long-clip fallback

When `fused_M = batch × ceil(seq/32) > ACE_STEP_DIT_MAX_FUSED_M` (default **16**):

- `_mcast_1d_linear_program_config` returns `None` → matmul falls back to **DRAM** in0/out (L1 CB budget).
- `ace_step_dit_prefers_dram_activations()` moves patch tensors to DRAM in `TtAceStepDiTCore`.

**A/B knobs** (validate PCC before E2E):

| Env | Effect |
|---|---|
| `ACE_STEP_DIT_FORCE_L1_MATMUL=1` | Keep L1 in0/out even without 1D-mcast PC |
| `ACE_STEP_DIT_MAX_FUSED_M=32` | Allow larger `per_core_M` in 1D-mcast PC |
| `ACE_STEP_DIT_BFLOAT4_WEIGHTS=1` | `bfloat4_b` weights (submodule PCC first) |

## Torch fallback audit

No runtime **torch matmul fallback** on the DiT hot path. Torch appears only for:

- Host reference / PCC (`test_pcc_*`, `TorchAceStepDiTCoreRef`)
- Weight upload (`torch.from_numpy` → `ttnn.from_torch`)
- RoPE table build on host (`get_rot_mats_hf_fallback` label = HF parity, not device fallback)
- Euler/APG scalar paths in `dit_sampling_ttnn.py` (outside traced DiT body)

## Submodule tests

| Test | Purpose |
|---|---|
| `tests/test_pcc_dit_linear.py` | LoFi+BFP8 single linear PCC ≥ 0.99 |
| `tests/test_pcc_attention.py` | Full attention block |
| `perf/test_perf_attn_tracy.py` | Attention Tracy isolate |
| `perf/test_perf_matmul_tracy.py` | Single linear Tracy isolate |
| `perf/test_perf_dit_denoise_loop_tracy.py` | Full denoise loop |

## Layout experiments (reverted — do not re-enable by default)

| Change | Result |
|---|---|
| Manual concat vs `nlp_concat_heads` | Worse (more reshape/transpose) |
| `nlp_create_qkv_heads` | +NlpCreateHeads cost |
| All-manual QKV | Worst reshape/transpose counts |

## Tracy workflow

```bash
cd tt-metal
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \
  models/experimental/ace_step_v1_5/perf/test_perf_matmul_tracy.py::test_perf_ace_step_matmul_tracy_profile -v -s

tt-perf-report generated/profiler/reports/<timestamp>/cpp_device_perf_report.csv
```

Visualizer: https://github.com/tenstorrent/ttnn-visualizer

## Implemented (May 2026): fused matmuls

| Fusion | Saves per forward (24 layers) | PCC |
|---|---|---|
| Self-attn **Q+WKV** → one `w_qwkv` matmul | **24** matmuls (cross keeps separate Q / WKV) | self ≥0.998, GQA ≥0.998 |
| MLP **gate+up** → one `w_gate_up` matmul | **24** matmuls | MLP ≥0.999, layer ≥0.998 |

Re-profile denoise loop; expect **~48 fewer** `MatmulDeviceOperation` calls (~445 → ~397) and lower matmul % if memory-bound.

## Next levers (advanced)

- SiLU∘linear fusion via `fused_activation=UnaryOpType.SILU` on gate matmul only (needs split-output support)
- Core grid / `per_core_N` tuning per shape bucket (compute-bound vs memory-bound)
- Trace replay: `ace_step_dit_body_trace_safe()` false when fused_M > cap
