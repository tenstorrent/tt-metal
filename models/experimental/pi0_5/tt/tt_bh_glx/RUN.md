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
