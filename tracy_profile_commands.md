# Tracy Profile Commands — test_conv2d_pointwise.py

## Tracy Options

| Option | Value | Effect |
|--------|-------|--------|
| `-r` | enabled | Generate ops perf report |
| `-m` | enabled | Run test via `runpy.run_module` (needed for `pytest`) |
| `--op-support-count` | 1000 | Profiler buffer supports up to 1000 ops |
| `--profile-dispatch-cores` | enabled | Capture dispatch overhead (`DISPATCH TOTAL CQ CMD OP TIME`, `GO SEND WAIT TIME`) |
| `--device-memory-profiler` | enabled | Track L1/DRAM buffer allocations |
| `--check-exit-code` | enabled | Abort report generation if test fails |

---

## Test 1 — `test_conv2d_dram_bottleneck` (1×3×1536×1536)

```bash
python3 -m tracy \
  -r -m \
  --op-support-count 1000 \
  --profile-dispatch-cores \
  --device-memory-profiler \
  --check-exit-code \
  -o profiler_output/bottleneck_conv2d_1 \
  pytest "tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py::test_conv2d_dram_bottleneck[conv2d_1_1x3x1536x1536]" -vss
```

---

## Test 2 — `test_conv2d_dram_bottleneck` (1×3×1280×2304)

```bash
python3 -m tracy \
  -r -m \
  --op-support-count 1000 \
  --profile-dispatch-cores \
  --device-memory-profiler \
  --check-exit-code \
  -o profiler_output/bottleneck_conv2d_2 \
  pytest "tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py::test_conv2d_dram_bottleneck[conv2d_2_1x3x1280x2304]" -vss
```

---

## Test 3 — `test_conv2d_only` (1×3×1536×1536)

```bash
python3 -m tracy \
  -r -m \
  --op-support-count 1000 \
  --profile-dispatch-cores \
  --device-memory-profiler \
  --check-exit-code \
  -o profiler_output/conv2d_only_1 \
  pytest "tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py::test_conv2d_only[conv2d_1_1x3x1536x1536]" -vss
```

---

## Test 4 — `test_conv2d_only` (1×3×1280×2304)

```bash
python3 -m tracy \
  -r -m \
  --op-support-count 1000 \
  --profile-dispatch-cores \
  --device-memory-profiler \
  --check-exit-code \
  -o profiler_output/conv2d_only_2 \
  pytest "tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py::test_conv2d_only[conv2d_2_1x3x1280x2304]" -vss
```
