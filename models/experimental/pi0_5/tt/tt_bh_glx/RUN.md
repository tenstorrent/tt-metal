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
> `DEVICE KERNEL DURATION ≥ 1e8` ns**. For per-iteration "forward" cost, slice from the
> first block `MatmulDeviceOperation` (excludes one-time weight-load/RoPE-precompute init).
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

## Reference numbers (per-chip forward, prod env, bf8 default)

| TP | PCC | per-chip forward |
|---|---|---|
| TP=1 | 0.9942 | 18.1 ms |
| TP=4 | 0.9939 | **11.56 ms** |
| TP=8 | 0.9935 | 10.60 ms |

(TP=4 baseline before this session's opt was 20.4 ms.) SDPA (~2.16 ms) is at its
compute floor. Head-parallel attention & fused/async CCL are gated by the fabric
(4-node Linear, no ring) — see the design spec under `docs/superpowers/specs/`.
