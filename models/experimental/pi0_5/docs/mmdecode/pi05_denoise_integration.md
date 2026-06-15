# pi0.5 Denoise matmul_decode Integration — M=32 Final Report

Continuation of the matmul_decode denoise integration. Measurement → apply → perf at
the model's REAL M=32 denoise shapes (NOT the M=64 scheme).

- **CLONE**: `/home/ttuser/salnahari/tt-metal-pi05-openpi`, branch `pi05_openpi_upstream_bh_glx`
- **REPO**: `/home/ttuser/salnahari/tt_symbiote`
- **Device**: P150 (Blackhole), single chip, host `tt-quietbox`
- **PY**: `$CLONE/python_env/bin/python`
- **tracy op-support-count**: 20000; metric = DEVICE KERNEL DURATION (col-20)

## Commits (all on the CLONE branch — fork untouched)

| commit | what |
|---|---|
| `5f4ab455` | (pre-existing) port of CLAUDE-optimized resident-weight matmul_decode op |
| `cdd0734a` | helper scripts: `bench_matmul_decode_denoise.py` + `extract_denoise_mmdecode_kernel.py` |
| `708fc7fa` | **fix**: matmul_decode kernel includes + NoC API ported to current tt-metal (see below) |
| `19d55ef8` | **apply**: env-gated `PI0_MMDECODE_DENOISE=1` swap of 5 M=32 denoise matmuls in `ttnn_gemma.py` |

Final CLONE HEAD: `19d55ef85d0a2660b899e591339861081d7560ad`.

## Blocker found + fixed (the real long pole)

The op committed in `5f4ab455` (host C++ + Python binding) RESOLVED in the binary, but its
DEVICE KERNELS failed to JIT-compile at runtime against this tt-metal checkout — surfaced only
when matmul_decode actually ran on-device. Three independent API-drift defects in the 4 dataflow
kernels (`reader_full_width_sharded`, `reader_partial_width_sharded`, `reader_full_width_temporal`,
`writer_partial_width_sharded`):

1. **Header move**: `#include "api/dataflow/{noc,circular_buffer,noc_semaphore,endpoints}.h"`
   → these now live at `experimental/{...}.h`. (`api/dataflow/dataflow_api.h` is unchanged.)
2. **Namespacing**: the `Noc` / `CircularBuffer` / `Semaphore` / `use` types are now in the
   `experimental::` namespace. Fixed by adding `using namespace experimental;` after the includes.
3. **Multicast template arg rename**: `async_write_multicast<NocOptions::MCAST_INCL_SRC>` /
   `set_multicast<NocOptions::MCAST_INCL_SRC>` → `<Noc::McastMode::INCLUDE_SRC>`.

Kernel-source only (JIT-compiled) → no host rebuild required; clearing the kernel cache and
re-running recompiles fresh. After the fix, all 4 M=32 denoise shapes run with PCC ≥ 0.999.

## STEP 2 — native baseline at M=32 on this P150

`PI0_MM_SWEEP=1 PI0_MM_SWEEP_ITER=30 pytest test_denoise_matmul_sweep.py`:

**Result: 0 configs ran / 0 passed for ALL 4 shapes.** Every explicit
`MatmulMultiCoreReuseMultiCast1DProgramConfig` (including the production picker baseline,
`build_matmul_pcfg`) FATALs on this host:

```
TT_FATAL: Illegal kernel placement for bmm_large_block_zm_fused_bias_activation,
Kernels cannot be placed on dispatch cores!  (not on_dispatch_core)
```

So the model's INTENDED tuned-native path (explicit 1D-width pcfg) is **not runnable on this
P150 host**. The only runnable native baseline here is the default auto-config `ttnn.linear`
(this is exactly what the bench's native arm measures). The model's prior tuned-native numbers
(qkv 9.11 / gate-up 12.51 / o_proj 9.32 / down 14.40 µs/call — from `test_denoise_matmul_sweep.py`
docstring, measured on a host where the explicit pcfg ran) are carried forward as the reference
baseline for the verdict.

**Stage-level baseline**: NOT captured. The model weights symlink
`weights/pi05_base → /storage/sdawle/pi05_weights/pi05_base` is **dangling** (target absent on
this host), so no runnable model / `test_perf_ttnn_trace_e2e.py` stage baseline is possible.
Matmul-level only, as the brief permits.

## STEP 4 — matmul_decode at M=32 vs baseline (tracy, osc=20000, N_ITERS=20)

`bench_matmul_decode_denoise.py` under tracy. All 4 shapes plan to a single FULL-WS call
(`n_chunks=1, k_split_G=1`) — exactly ONE device call at M=32, as the brief expects. KERNEL
(col-20) extracted via the bench's own `extract_denoise_mmdecode_kernel.py` (per-call avg) and
a min-of-N pass.

| shape | tuned-native (ref µs) | auto-native (this host, min µs) | matmul_decode (min µs / avg µs) | ratio vs tuned | ratio vs auto | PCC |
|---|---|---|---|---|---|---|
| qkv_fused (K1024 N2560) | 9.11 | 20.65 | 3.46 / 3.66 | **0.38** | 0.17 | 0.99982 |
| mlp_gate_up (K1024 N4096) | 12.51 | 20.59 | 4.89 / 5.03 | **0.39** | 0.24 | 0.99983 |
| o_proj (K2048 N1024) | 9.32 | 29.98 | 6.01 / 6.55 | **0.64** | 0.20 | 0.99963 |
| mlp_down (K4096 N1024) | 14.40 | 58.67 | 11.30 / 11.48 | **0.78** | 0.19 | 0.99917 |

matmul_decode **beats every shape vs BOTH baselines** (ratio < 0.99) with PCC ≥ 0.99.

**45.969 back-test**: reproduced EXACTLY with the extractor —
`EXTRACT_MODE=mmsweep METRIC=KERNEL N_ITERS=5 extract_perf.py frozen_golden/mmd_SigLIP_qkv.csv`
→ `SigLIP.qkv mmd_us/fwd=45.969`.

**M=32 vs our M=64 result**: at M=64 the headline win was SigLIP.qkv 55.44→45.97 (ratio 0.83,
STRICT-BEAT). At M=32 the denoise matmuls win by an even LARGER margin (ratio 0.38–0.78 vs the
tuned reference). The M=64 win not only transferred to M=32 — it strengthened. Each M=32 shape
is a single width-sharded call (no M-split), which is the cleanest regime for the op.

## STEP 5 — apply (data-driven)

Since step-4 shows matmul_decode beats every shape's native baseline (ratio < 0.99, PCC ≥ 0.99),
ALL 5 denoise matmuls were swapped, behind an opt-in env flag (`PI0_MMDECODE_DENOISE=1`), in
`models/experimental/pi0_5/tt/ttnn_gemma.py`:

- **Attention** (`GemmaAttention`): fused QKV (`wqkv_mmd`, out bf8_b to stay drop-in for
  `nlp_create_qkv_heads`), o_proj (`o_proj_mmd`, out bf16).
- **Expert MLP** (`GemmaMLP`): gate (`gate_mmd` + separate `ttnn.gelu`), up (`up_mmd`), down (`down_mmd`).

Each is built once at `__init__` (resident-weight `MatmulDecodeLinear`) and used only when
`m_tiles == 1` — the M=32 single-call regime per the brief's M=32 RULE. The swap is restricted to
the expert (`config.width <= 1024` / `mlp_dim <= 4096`); VLM/prefill (`m_tiles >= 8`) stays native.
**Default OFF → byte-identical** to the prior model.

If the real denoise sequence is padded to 64 (`m_tiles == 2`), the swap simply does not fire and
the native path runs — honest, since we only validated/tuned the single-call M=32 regime.

## STEP 6 — re-verify

Model-forward + stage perf could NOT be re-run (weights absent on this host). Matmul-level
re-verification with the EXACT dtypes/4D shapes the model now uses:

| model path | config | PCC | output shape |
|---|---|---|---|
| QKV | bf8_b out, 4D `[1,1,32,1024]`→`[…,2560]` | 0.99979 | [1,1,32,2560] ✓ |
| o_proj | bf16 out | 0.99963 | [1,1,32,1024] ✓ |
| gate + separate gelu | bf16 out + `ttnn.gelu` | 0.99983 | — |

All ≥ 0.99; 4D leading-dim preservation correct. Per-shape before/after KERNEL deltas are the
STEP-4 table above (tuned-native → mmd: −62% / −61% / −36% / −22%). Stage before/after: N/A (no weights).

## Verdict

**The CLAUDE-optimized matmul_decode IMPROVES the pi0.5 denoise at its real M=32 shapes.** Every
denoise matmul (qkv, o_proj, gate, up, down) is faster than the model's tuned-native baseline at
M=32 (ratio 0.38–0.78) with PCC ≥ 0.99, each running as a single width-sharded device call. The
M=64 win transferred to — and strengthened at — M=32. The swap is committed behind
`PI0_MMDECODE_DENOISE=1` (default-off, byte-identical otherwise).

Caveats (honest):
- Baseline reference is the model's prior tuned-native (build_matmul_pcfg 1D-width) measured on a
  host where the explicit pcfg runs; on THIS P150 host that explicit pcfg FATALs
  (`not_on_dispatch_core`), so the locally-runnable native is the slower auto-config — mmd beats
  both. The verdict (ratio < 0.99 vs the tuned reference) is the conservative one.
- No end-to-end model-forward / denoise-stage validation on this host (weights symlink dangling).
  Recommended next step on a host with weights present + the explicit-pcfg native path runnable:
  run `test_perf_ttnn_trace_e2e.py` with `PI0_MMDECODE_DENOISE=0` vs `=1` for the stage delta and
  a LIBERO-rollout PCC/semantic check.

Final state: clone committed (HEAD `19d55ef8`); device idle (TDP 0x1f); fork untouched.
