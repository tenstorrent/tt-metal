# How to test CLAUDE-optimized matmul_decode on the real pi0.5 model

The change is a default-OFF opt-in flag. Testing = run the model with the flag OFF (baseline)
vs ON (optimized) and compare PCC + perf. Everything lives on the cloned branch
`pi05_openpi_upstream_bh_glx`.

## 0. What the change is
- 4 commits on top of upstream `5b6a1c2e`:
  - `5f4ab455` port the matmul_decode op (host C++ + Python binding)
  - `cdd0734a` M=32 denoise bench + kernel extractor helpers
  - `708fc7fa` fix the op's device kernels for the current tt-metal NoC/dataflow API (kernel-source)
  - `19d55ef8` env-gated swap of the 5 M=32 denoise matmuls in `tt/ttnn_gemma.py`
- Flag: **`PI0_MMDECODE_DENOISE=1`** → routes qkv, o_proj, gate, up, down through `ttnn.matmul_decode`
  **only when `m_tiles == 1` (M=32, the denoise single-tile regime), expert-only**. Default OFF =
  byte-identical to upstream. Final HEAD `19d55ef8`.

## 1. Prerequisites (what the tester needs)
- A **Blackhole** host with the device (single-chip P150 is enough for the denoise stage; the
  28-chip BH Galaxy is only for the experimental multi-chip pipeline).
- The **pi0.5 weights** present. The repo expects them at:
  `models/experimental/pi0_5/weights/pi05_base` (currently a symlink to
  `/storage/sdawle/pi05_weights/pi05_base`). Either place the real checkpoint there or repoint the
  symlink:  `ln -sfn /path/to/pi05_base models/experimental/pi0_5/weights/pi05_base`
  (On THIS host the symlink is dangling — that's why we only validated at the matmul level.)

## 2. Get the code + build
```bash
# option A — clone tt-metal and apply the 4 commits onto the branch
git clone --branch pi05_openpi_upstream_bh_glx https://github.com/tenstorrent/tt-metal.git ttm-pi05
cd ttm-pi05 && git submodule update --init --recursive
# bring in the 4 commits (push them to a remote first, or copy from this clone):
#   git -C /home/ttuser/salnahari/tt-metal-pi05-openpi format-patch 5b6a1c2e..19d55ef8 -o /tmp/mmd_patches
#   git am /tmp/mmd_patches/*.patch
# option B — just copy/reuse the existing clone: /home/ttuser/salnahari/tt-metal-pi05-openpi (HEAD 19d55ef8)

# build (the port adds a host C++ op → host build required; tracy ON for perf)
./build_metal.sh
./create_venv.sh
```

## 3. Clear the kernel cache once (so the patched device kernels JIT-recompile)
```bash
rm -rf ~/.cache/tt-metal-cache/* 2>/dev/null   # or the generated kernel cache dir
```
The `708fc7fa` fix is kernel-source; a stale cache would reuse old (broken) kernels.

## 4. Environment
```bash
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
export ARCH_NAME=blackhole          # MESH_DEVICE as appropriate for the host
PY=$PWD/python_env/bin/python
# (if a _bench_runs/pi05_production.env exists for the host, source it for pinned perf settings)
```

## 5. Correctness first — PCC with the flag ON (must pass before trusting perf)
```bash
# denoise per-step vs torch reference (the direct correctness gate for the swap):
PI0_MMDECODE_DENOISE=1 PYTHONPATH=$PWD $PY -m pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_pi05_per_step_vs_torch.py
# end-to-end / real-weights + semantic (LIBERO rollout) checks:
PI0_MMDECODE_DENOISE=1 PYTHONPATH=$PWD $PY -m pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_real_weights.py \
  models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model_libero.py
```
Expect mean PCC ≥ 0.99 (matches the upstream denoise PCC bar). If PCC drops with the flag ON, the
swap is not safe on that host/dtype — stop.

## 6. Performance — baseline (OFF) vs optimized (ON), same command
```bash
# denoise-stage breakdown:
PI0_MMDECODE_DENOISE=0 $PY -m pytest -xvs models/experimental/pi0_5/tests/perf/test_perf_ttnn_trace_e2e.py   # baseline
PI0_MMDECODE_DENOISE=1 $PY -m pytest -xvs models/experimental/pi0_5/tests/perf/test_perf_ttnn_trace_e2e.py   # optimized
# full e2e latency (~65 ms headline):
PI0_MMDECODE_DENOISE=0 $PY -m pytest -xvs models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py
PI0_MMDECODE_DENOISE=1 $PY -m pytest -xvs models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py
```
Compare denoise-stage ms and the per-call latency between the two runs.

## 7. (Optional) matmul-level sanity, no weights needed
```bash
# our M=32 matmul_decode bench (the per-call numbers in the integration report):
TT_METAL_HOME=$PWD PYTHONPATH=$PWD:/home/ttuser/salnahari/tt_symbiote \
  $PY -m tracy -p -r -v --op-support-count 20000 \
  -m "pytest models/experimental/pi0_5/tests/perf/bench_matmul_decode_denoise.py -x -s"
```
**Use `--op-support-count 20000`, never 200000** (200000 segfaults the tracy device-profiler dump).

## Caveats / gotchas to check on the test host
- **The swap only fires at `m_tiles == 1` (M=32).** Confirm the real denoise runs the action expert
  at M=32 (1 tile row). If the action sequence is padded to 64 (`m_tiles==2`) the flag is a no-op and
  the native path runs — so verify the swap actually engages (e.g. log/inspect, or check op counts).
- **Tuned-native baseline on a single P150:** on our host the model's explicit 1D-width pcfg FATALs
  (`not_on_dispatch_core`); the matmul-level win was measured vs that recorded tuned-native. On the
  real deployment host (where the explicit pcfg runs), the OFF-vs-ON perf comparison in step 6 is the
  authoritative one.
- **Trace mode:** the perf tests use trace capture/replay; matmul_decode keeps the weight L1-resident
  across the 18 blocks × N steps, which is the regime it's built for — but confirm trace capture
  succeeds with the flag ON (re-capture after clearing the cache).

## Bottom line for the tester
Build the branch (HEAD `19d55ef8`) with weights present, clear the kernel cache, then run the PCC
tests and the perf tests with `PI0_MMDECODE_DENOISE=0` vs `=1`. ON should be PCC-equal and faster on
the denoise stage; OFF is byte-identical to upstream so it's a safe, reversible A/B.
