# pytest reporting-cost investigation

Supporting material for the reporting-cost fixes in tt-metal
`mstaletovic/pytest-reporting-cost` and tt_ops_code_gen
`mstaletovic/eval-pytest-reporting-cost`. Documents only — nothing here is
built, imported or executed by CI.

All device measurements were taken on a **Blackhole quietbox with 32 cores**.

| file | contents |
| --- | --- |
| `COLLECT_PASS_PROFILE.md` | Profiling the `up_front_collect` warm pass over the full 40,828-case `rms_norm` golden suite: where the time goes, the `-rA` isolation, the device-free scaling reproduction, and the method notes. |
| `PYTEST_HARNESS_IMPROVEMENTS.md` | The proposals themselves — issue / manifestation / fix / risk / magnitude per item, with each measurement tagged by which experiment produced it. |
| `UP_FRONT_COLLECT_EAGER_COMPILE.md` | A **separate, unrelated** finding about Metal 2.0 `ProgramSpec` factories compiling eagerly. Included only because it is the precondition for reproducing the collect-pass numbers (see below). Not part of either fix PR. |

## Reproducing the device-free scaling result

No ttnn, no device, no tt-metal:

```python
# test_synth.py
import os, pytest
N = int(os.environ.get("SYNTH_N", "40000"))
_IDS = [f"1x1x32x{64+i%97}-dtype=FLOAT32-gamma_layout=ROW_MAJOR-layout=TILE-"
        f"memory_layout=BLOCK_SHARDED-rank=4-idx={i}" for i in range(N)]

@pytest.mark.parametrize("case", range(N), ids=_IDS)
def test_op(case):
    if (case % 1000) / 1000.0 < 0.833:
        pytest.skip("matches INVALID entry")
    assert True
```

```bash
SYNTH_N=40000 pytest test_synth.py -o addopts=    -q   #  36 s
SYNTH_N=40000 pytest test_synth.py -o addopts=-rA -q   # 236 s
SYNTH_N=40000 pytest test_synth.py -o addopts=-ra -q   #  36 s
```

This is the load-bearing evidence: it needs nothing from this repository, and it
is what should go upstream to pytest.

## Note on the collect-pass numbers

The 40,828-case collect-pass timings in `COLLECT_PASS_PROFILE.md` were taken with
the Metal 2.0 defer-compile change described in `UP_FRONT_COLLECT_EAGER_COMPILE.md`
applied locally. That change is **not** part of either fix PR and is not required
by them.

It matters only for reproduction: without it the collect pass JIT-compiles every
kernel inline and takes ~6,900 s, which swamps the reporting cost entirely. It is
constant across every arm of the `addopts` comparison, so the deltas attributed to
`-rA` and `--tb` are unaffected by it.
