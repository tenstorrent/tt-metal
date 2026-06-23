# Running pi0.5 VLM prefill — PCC & profiler (TP=1 / TP=4 / TP=8)

Per-chip device-kernel benchmarking of the VLM prefill (Gemma-2B ×18, 3-camera /
seq=1024). All commands run from the repo root.

## Setup (once per shell)

```bash
cd <tt-metal repo root>
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
# checkpoint (HF cache snapshot):
export PI05_CHECKPOINT_DIR=$HOME/.cache/huggingface/hub/models--lerobot--pi05_libero_base/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd
# production perf flags (head-split, mm-tune, bf8, chunk=1024, num_cameras=3, ...):
set -a; source models/experimental/pi0_5/_bench_runs/pi05_production.env; set +a
unset PI0_VLM_MLP_MINIMAL   # minimal_matmul regresses the TP MLP; bf8-only is the win
```

> If the device is wedged (e.g. `Read 0xffffffff`), reset: `python_env/bin/tt-smi -glx_reset`
> (or `-r`). Always start clean.

## Tests

| | test | mesh / chips | notes |
|---|---|---|---|
| **TP=1** | `test_prefill_tp1_pcc` | 1 chip — `PI0_DEVICE_ID=1` → physical **chip 9** | single-device `PaliGemmaBackboneTTNN.forward_vlm` |
| **TP=4** | `test_prefill_tp4_pcc` | 1×4 — chips **8–11** | `StagePrefillTP4`, bf8 MLP default |
| **TP=8** | `test_prefill_tp4_pcc` + `PI0_TP=8` | 1×8 — chips **8–15** | TP degree derived from mesh |

`T=models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py`

**Checkpoint-free option:** `test_prefill_tp4_perf_dummy` runs TP=4/8 with random
weights (no checkpoint) — for profiling, op shapes/timing are identical to real
weights. It also checks PCC (torch ref on the *same* dummy weights; bar 0.97 since
random weights stress bf8 more than trained). Honors `PI0_TP` / `PI0_VLM_CHUNK_SIZE`
/ `PI0_SKIP_TORCH_REF`. Example: `TT_VISIBLE_DEVICES=8,9,10,11 python_env/bin/python
-m tracy -p -v -r --op-support-count 8000 -m pytest -sq $T::test_prefill_tp4_perf_dummy`

### PCC (correctness)

```bash
# TP=1  (chip 9 — chip 8 is the mmio chip; use DEVICE_ID=1)
TT_VISIBLE_DEVICES=8,9,10,11 PI0_DEVICE_ID=1 \
  python_env/bin/python -m pytest -sq $T::test_prefill_tp1_pcc

# TP=4
TT_VISIBLE_DEVICES=8,9,10,11 \
  python_env/bin/python -m pytest -sq $T::test_prefill_tp4_pcc

# TP=8
TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 PI0_TP=8 \
  python_env/bin/python -m pytest -sq $T::test_prefill_tp4_pcc
```
Expected PCC ≥ 0.99 (bf8 ≈ 0.9935–0.9939; bf16 path 0.995127). Prints
`Prefill TP=… stage PCC vs torch: 0.99xxxx (shape (1, 1024, 2048))`.

### Profiler (device-kernel timing, tracy)

Prefix any PCC command with the tracy wrapper:

```bash
TT_VISIBLE_DEVICES=8,9,10,11 \
  python_env/bin/python -m tracy -p -v -r --op-support-count 8000 \
  -m pytest -sq $T::test_prefill_tp4_pcc
```
CSV lands at `generated/profiler/reports/<TS>/ops_perf_results_<TS>.csv`.

> **Read device 1, not device 0.** On this box, **chip 8 (the mmio chip) has corrupt
> eager `DEVICE KERNEL DURATION`** (bogus ~3.5e12 ns on multi-core ops). Aggregate from
> a **sane device** (TP=4 → device 1; TP=8 → device 1/5/6) and **drop rows with
> `DEVICE KERNEL DURATION ≥ 1e8` ns**.
>
> **Always report the init-EXCLUDED forward.** Slice device-1 ops from the **first
> `MatmulDeviceOperation`** onward — the ~95 ops before it are one-time init
> (**~0.54 ms**: 39× `TilizeWithValPadding` + 40× `Typecast` weight-tilization,
> RoPE-table precompute, embedding setup) that a traced pipeline amortizes. Summing
> *all* device-1 ops over-counts by that ~0.54 ms (e.g. 12.10 ms all-ops = **11.56 ms
> forward** + 0.54 ms init). Parse:
> ```python
> rows=[d for d in csv.DictReader(open(f)) if d["OP TYPE"].strip()=="tt_dnn_device" and d["DEVICE ID"].strip()=="1"]
> i0=next(i for i,d in enumerate(rows) if d["OP CODE"].strip()=="MatmulDeviceOperation")
> fwd=sum(float(d["DEVICE KERNEL DURATION [ns]"] or 0) for d in rows[i0:] if 0<float(d["DEVICE KERNEL DURATION [ns]"] or 0)<1e8)
> ```
> Raise `--op-support-count` for >1 iteration to avoid profiler-buffer overflow.

## Knobs

| env | effect |
|---|---|
| `PI0_TP` | TP degree for `test_prefill_tp4_pcc` (default 4; set 8 with 8 visible chips) |
| `PI0_DEVICE_ID` | single-device chip index for TP=1 (use 1 = chip 9; chip 8 timing is corrupt) |
| `PI0_VLM_CHUNK_SIZE` | prefill seq / MLP chunk (1024 = 3-camera single-chunk; default) |
| `PI0_NUM_CAMERAS` | 3 (prod); seq = 256·N_cam + 256 (1cam→512, 3cam→1024) |
| `PI0_SKIP_TORCH_REF=1` | skip the CPU torch reference (faster profiling; PCC not checked) |
| `PI0_TP4_ATTN_HEADPAR=1` | **opt-in** head-parallel attention (default off — regresses on this Linear fabric) |

## Reference numbers (per-chip **forward**, init-excluded, prod env, bf8 default)

3-camera (seq=1024, `PI0_VLM_CHUNK_SIZE=1024`):

| TP | PCC | forward (init-excl) | all-ops sum (+~0.54 ms init) |
|---|---|---|---|
| TP=1 | 0.9942 | 18.1 ms | — |
| TP=4 | 0.9939 | **11.56 ms** | 12.10 ms |
| TP=8 | 0.9935 | 10.60 ms | — |

2-camera (seq=768, `PI0_VLM_CHUNK_SIZE=768`):

| TP | forward (init-excl) | all-ops sum |
|---|---|---|
| TP=4 | ~8.7 ms | 9.22 ms |

(TP=4 baseline before this session's opt was 20.4 ms.) Numbers drift ±10–20% run-to-run
(CCL/matmul timing jitter on this box) — the init-excluded forward is the canonical metric.
SDPA is the largest single op (~2.16 ms @ seq=1024, O(seq²) → ~0.72 ms @ seq=768) and is
at its compute floor. Head-parallel attention & fused/async CCL are gated by the fabric
(4-node Linear, no ring) — see the design spec under `docs/superpowers/specs/`.

---

# 1×8 full e2e pipeline — vision + prefill + denoise on a single mesh

`Pi0_5GLX1x8Pipeline` (pipeline_1x8.py) runs all three stages on the same 1×8
mesh (chips 8–15): SigLIP DP (3 real + 5 zero-dummy cams), Prefill TP=8,
replicated 5-step denoise. On-device CCL for cross-stage handoff (no host
bounce, no fabric sockets). TIER A precompute eliminates per-step block-mod
matmuls; staged sub-traces give true per-stage traced timing.

## Setup

```bash
cd <tt-metal repo root>
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
set -a; source models/experimental/pi0_5/_bench_runs/pi05_production.env; set +a
export TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PI0_TP=8 PI0_TP4_ATTN_HEADPAR=1 PI0_MLP_BS=1 PI0_MLP_FUSED_RS=0
# camera count (3 = production training spec; 2-cam supported but PCC drops, see notes)
export PI0_NUM_CAMERAS=3
tt-smi -glx_reset   # always start clean
```

`P=models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8.py`

## Tests

| Test | What it does | Gating |
|---|---|---|
| `test_perf_1x8_eager` | One eager `sample_actions`; asserts shape + finite. | Always on |
| `test_perf_1x8_traced` | Captures e2e trace; reports `input_upload / trace_exec / output_readback` over `PERF_ITERS=20` replays + eager per-stage proportions. | Always on |
| `test_perf_1x8_traced_staged` | Captures 3 sub-traces (vision / prefill / denoise) on the same mesh, replays each separately, reports true per-stage traced ms. Trace region bumped to 256 MiB. | Always on |
| `test_pcc_1x8_all_stages` | Vision + Prefill (isolated, same random prefix on both sides) + e2e (matched-seed noise) + estimated denoise = e2e / (vision · prefill). | `PI05_E2E_PCC=1` |

### Run all tests

```bash
# Perf — e2e single-trace breakdown (host overhead split)
python_env/bin/pytest -sq $P::test_perf_1x8_traced

# Perf — per-stage traced breakdown via 3 sub-traces
python_env/bin/pytest -sq $P::test_perf_1x8_traced_staged

# PCC — vision + prefill + e2e + estimated denoise
PI05_E2E_PCC=1 python_env/bin/pytest -sq $P::test_pcc_1x8_all_stages

# 2-cam variant: rerun with PI0_NUM_CAMERAS=2
PI0_NUM_CAMERAS=2 python_env/bin/pytest -sq $P::test_perf_1x8_traced_staged
```

Each test prints a startup `[env]` block listing which production flags are
set — confirms the pi05_production.env was sourced.

### Knobs (1×8 pipeline)

| env | effect |
|---|---|
| `PI0_NUM_CAMERAS` | Real-camera count (1..8); padded to 8 with zero dummies inside the pipeline. |
| `PERF_ITERS` | Timed iters in `test_perf_1x8_traced*` (default 20). |
| `PI05_E2E_PCC` | Enable `test_pcc_1x8_all_stages` (default off; runs CPU torch ref → slow). |
| `PI05_NUM_DENOISE_STEPS` | Override the 5-step default schedule (10 matches upstream training spec). |

## Reference numbers (commit `38ac051ee68`, 2026-06-23)

### Perf — per-stage TRACED breakdown (`test_perf_1x8_traced_staged`, PERF_ITERS=20)

| Stage | 3-cam mean | 2-cam mean | Δ (3→2 cam) |
|---|---:|---:|---:|
| input_upload (host) | 14.25 ms | 6.49 ms | — |
| **vision** (SigLIP DP, 8 chips) | **5.01 ms** | **5.00 ms** | 0 (DP pads to 8 cams regardless) |
| **prefill** (TP=8) | **9.62 ms** | **8.26 ms** | −1.36 (1024→768 prefix) |
| **denoise** (5 step, replicated) | **19.37 ms** | **18.51 ms** | −0.86 (shorter KV in cross-attn) |
| output_readback | 1.24 ms | 1.12 ms | — |
| **compute (v+p+d)** | **34.01 ms** | **31.78 ms** | −2.23 |
| traced_total | 49.49 ms | 39.39 ms | — |

### PCC (`test_pcc_1x8_all_stages`, PI05_E2E_PCC=1)

| Stage | 3-cam | 2-cam | Target |
|---|---:|---:|---:|
| vision | 0.999684 ✓ | 0.999685 ✓ | ≥ 0.997 |
| prefill | 0.994639 ✓ | 0.994757 ✓ | ≥ 0.99 |
| denoise (est) | 1.001883 | 1.001869 | — |
| **e2e** | **0.996197 ✓** | **0.996302 ✓** | ≥ 0.95 |

Both cam counts now pass; e2e PCC is essentially identical (~0.996) — both
close to the 28-chip baseline of 0.9988. The fix in commit `38ac051ee68` was
adding position-aware suffix RoPE + expert attention mask to match how torch
ref builds them in `_denoise_forward`.

## Notes

- The reported `denoise` PCC is the multiplicative-drift estimate `e2e /
  (vision · prefill)` (rough proxy; PCC isn't strictly multiplicative). Pure
  isolation would require injecting torch-computed KV into TT denoise (per-layer
  shape/layout conversion — not implemented).
- E2E noise matches via the "seed-around-both" pattern: `torch.manual_seed(SEED)`
  before `pipe.sample_actions` AND before constructing `Pi0_5Model` then calling
  its `sample_actions`. Empirically better than the 28-chip `seed(SEED+1) just
  before each call` pattern on this 1×8 path (verified during dev).
- TIER A precompute (commit 7f11f571159) brought 3-cam denoise from 24.6 → 18.8 ms
  and e2e PCC from 0.986 → 0.992 — by moving 18×N_steps mod-Dense matmuls + the
  final-norm Dense to host, fewer bf16 rounding cascades.
- TIER B (`keep_padded` + phantom mask + `fill_implicit_tile_padding`) was
  ported then reverted: +0.8 ms denoise for +0.003 PCC on this pipeline,
  because the prefix is already tile-aligned (1024 / 768) so the prefix-side
  benefit is zero. See commit notes if you want to revisit.
