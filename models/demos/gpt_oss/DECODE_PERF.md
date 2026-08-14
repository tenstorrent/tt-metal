# GPT-OSS 20B decode performance

Measured on Blackhole p150, single chip, 11x10 core grid.

## How to reproduce

```
python_env/bin/python -m pytest \
    models/demos/gpt_oss/demo/text_demo_signpost.py \
    -v -s --timeout=0 -k blackhole-1x1 \
    --gpt-oss-max-tokens 32 \
    --gpt-oss-decode-trace-on
```

Throughput is on the harness's own line:

```
Average decode speed: 19.89ms @ 50.27 tok/s/user (50.27 tok/s throughput) over 31 settled steps
```

**`--gpt-oss-decode-trace-on` is not optional.** The harness leaves trace off by default, because
the device profiler needs it off to capture per-op timings. An untraced decode measures about
33 ms/token instead of about 19.8. Tracing is how the program is served, so the untraced number
describes a configuration nobody runs, and omitting the flag is the easiest way to conclude
wrongly that these changes did nothing.

## What it measures

| tree | commit | ms/token | tok/s/user |
|---|---|---|---|
| harness only, before the four config commits | `376b5729330` | 32.49 | 30.78 |
| this branch | `2c3291577e9` | 19.80 | 50.49 |

That is a 64.0% gain in throughput, measured by running the command above at both commits on this
branch. The same code measured 19.89 and 19.77 ms on two earlier samples, so read the current
figure as 19.8 +/- 0.1 ms rather than as an exact quantity.

The starting point already selects the greedy token on device: the harness does that itself. Read
the full logits tensor back to the host each step instead and decode is host-bound, which hides
every device-side change behind host time. That is a property of the harness, not of the four
commits measured here.

## What produced the improvement

Four commits, each one operation, each selected by measuring every valid configuration on the
shape the model dispatches and kept only where the measurement held and the accuracy check passed:

| operation | file | device time recovered |
|---|---|---|
| expert gate/up and down projections | `tt/expert_configs.py` | see note |
| lm_head output projection | `tt/model.py` | 1.5 ms |
| RMSNorm | `tt/rms_norm.py` | 1.5 ms |
| router matmul | `tt/topk.py` | 0.68 ms |

The expert projections were tuned in two steps whose individual contributions are recorded per
commit rather than as one figure.

Device kernel time over the four commits: 19.87 ms to 11.01 ms per token.

## Accuracy

Checked with `models/demos/gpt_oss/tests/accuracy/test_model.py`.

Top-1 moves from 1.0000 to 0.9667 and top-5 holds at 1.0000, against a 0.83 floor. Different
matmul blocking and sharding change the order of floating-point accumulation, which is what this
reflects; it is not a reduction in precision.

The `pcc` figure the accuracy check prints is **not** a measurement. It is parsed from the
`comparison_mode_pcc=0.9999` default in the configuration dump the framework writes on import, and
it reads 0.9999 on every run, including runs that scored nothing at all. Only top-1 and top-5 are
real numbers here.

## What is left

Kernel time is 11.01 ms of the 19.8 ms token, so roughly 8.8 ms is host and dispatch time that no
program configuration can reach. The identified items in it are a static `nnz` on the sparse expert
matmuls, and a bfloat16 lm_head output that would remove a host-side typecast running outside the
captured trace.
