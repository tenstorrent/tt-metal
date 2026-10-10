# Galaxy runbook: the LoudBox GLM-5.3-Flash on two Galaxy rows (+ flat expert / flat + combine)

Branch `mstaletovic/galaxy-unified`. The model is the LoudBox one, unchanged (spec
`models/demos/glm53_flash_d_p_lb/bringup/spec.yaml`, mesh 2x4); only the chips differ: two adjacent rows of the
8x4 Blackhole Galaxy. A Galaxy row (4 chips) has a wrap cable, so the 4-axis can be a ring; the LoudBox's is a line.

## 0. Once per machine

```bash
git fetch origin mstaletovic/galaxy-unified && git checkout mstaletovic/galaxy-unified
git submodule update --init --recursive
./build_metal.sh --release          # or your usual build; C++ edits need `ninja install` in build_Release
python_env/bin/activate ...         # whatever env you use
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD   # $PWD/ttnn first: an editable ttnn of another tree wins otherwise
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0     # mmap-backed uploads spin on KMD 2.9 (harmless elsewhere)
```

Which chips are which rows is per-Galaxy (a wrong list does not error, it just measures the wrong chips). Print them,
with the Galaxy idle:

```bash
python models/demos/deepseek_v3_d_p/utils/galaxy_carve.py 2x4 8x1
#  2x4 @ (0, 0): TT_VISIBLE_DEVICES=0,4,12,8,1,5,13,9      <- rows 0-1 (example from the mock Galaxy)
#  ...
```

## 1. "A Galaxy playing a LoudBox" (2x4)

```bash
export TT_VISIBLE_DEVICES=<the 2x4 line for rows 0-1>
D=models/demos/deepseek_v3_d_p/experimental_descriptors
# exactly the LoudBox (4-axis a line):
export TT_MESH_GRAPH_DESC_PATH=$PWD/$D/single_bh_galaxy_2x4_line_graph_descriptor.textproto
# or with the row's wrap links (4-axis a ring; gathers on axis 1 become rings by themselves):
#   export TT_MESH_GRAPH_DESC_PATH=$PWD/$D/single_bh_galaxy_2x4_ring_graph_descriptor.textproto BRINGUP_FABRIC=FABRIC_2D_TORUS_X
export BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml
```

Smoke check first (opens the mesh, ~2.5 min, no checkpoint): the per-layer bench below. If the mesh does not open,
check the carve list and that `TT_MESH_GRAPH_DESC_PATH` is absolute.

### Per-layer perf, fake weights (no checkpoint, ~2.5 min)

```bash
scripts/run_safe_pytest.sh --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_layer_perf.py -s
```

One layer of each type (0 kda_dense, 3 dsa_moe, 4 kda_moe), chunk 5120 at 51200: wall per call and device time per
chip / step / op (program real-time profiler), plus a whole-model estimate. Knobs: `GLM_LP_LAYERS`, `GLM_LP_CHUNK`,
`GLM_LP_START`, `GLM_LP_ITERS`, `GLM_LP_JSON=<file>`, `GLM_FAKE_HOT=n` (n hot experts), `GLM_LP_REAL=1` (checkpoint).
LoudBox reference (device ms, busiest chip; fake == real within 1%): kda_dense 10.25, dsa_moe 15.36, kda_moe 11.43;
estimate 554 ms per 5120 chunk.

### Whole model, real weights

```bash
# load + smoke ("What is the capital of France?" -> Paris)
scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_smoke.py -s
# 56k prefill perf (cold + warm, per-layer sums by block type)
scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_perf.py -s
# op-level profile of one warm chunk (GLM_PROF_LAYERS=2-4 for one layer per type)
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_profile_ops.py -s
# accuracy ladder (needs the golden)
BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py
```

Getting the weights:
- Checkpoint: `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (FP8, ~328 GB):
  `hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir /localdev/$USER/bringup/glm53_flash_d_p/hf`
  (the spec's `paths.hf`; or point `BRINGUP_HF` at any copy), or rsync it from the LoudBox
  (`/localdev/mstaletovic/bringup/glm53_flash_d_p/hf`). Needs the tokenizer files too (in the same dir).
- Device weight cache: the first load converts the routed experts to the flat bfp4 layout, ~70 s per MoE layer,
  ~1 h for all 42, ~160 GB under `generated/glm53_flash_d_p/tt_cache/flat/2x4/`. The LoudBox's cache does **not**
  carry over: the file key is the flat plan, and the Galaxy chip's grid (12x10) gives a different plan than the
  p150's (11x10). Later loads ~1 min.
- Golden (ladder only): `/localdev/$USER/bringup/glm53_flash_d_p_lb/golden` (spec `paths.golden`); rsync from the
  LoudBox (s4096 ~30 GB) or regenerate on the CPU (31 min, 30 GB):
  `python -m models.demos.common.bringup.reference.generate_golden --spec $BRINGUP_SPEC --rung s4096`.

## 2. Flat expert alone (one chip)

```bash
TT_VISIBLE_DEVICES=<any one chip> scripts/run_safe_pytest.sh --no-precompile \
  ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/test_flat_bench.py -s
```

Fake weights, real-time profiler; `FLAT_BENCH_MODEL` (glm53flash / kimi27 / glm53 / k3), `FLAT_BENCH_M`,
`FLAT_BENCH_HOT=M:factor`, ... (docstring). A 4-case sweep is about a minute.

## 3. Flat + combine overlap, one 8x1 Galaxy column

```bash
unset TT_VISIBLE_DEVICES TT_MESH_GRAPH_DESC_PATH      # opens the whole 8x4 (TORUS_XY) and takes column 0
T=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_flat_combine_overlap.py
scripts/run_safe_pytest.sh --run-all $T -k "8x1-galaxy and not perf" -rs -s   # accuracy (random weights)
scripts/run_safe_pytest.sh --run-all $T -k "8x1-galaxy and perf" -rs -s       # serial + overlap, fake weights
```

`-rs`: a case whose mesh does not match is SKIPPED and safe-pytest still prints PASS; check the skip lines.

## Gotchas

- Run device tests through `scripts/run_safe_pytest.sh` (hang detection, reset under the device lock); never a bare
  `tt-smi -r` on a shared box. `SAFE_PYTEST_DISPATCH_TIMEOUT` (alias `SAFE_PYTEST_TIMEOUT`) for long device waits.
- `--no-precompile`: the precompile pass loads the model twice.
- If `import ttnn` resolves to another tree (`python -c "import ttnn; print(ttnn.__file__)"`), fix `PYTHONPATH`.
