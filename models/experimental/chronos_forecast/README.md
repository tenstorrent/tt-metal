# Chronos-2 Forecast

[Chronos](https://github.com/amazon-science/chronos-forecasting) pretrained time-series models. This tree vendors **Chronos-2** inference sources as a golden PyTorch reference for a later TTNN port.

## Setup

All commands are run from the tt-metal repo root, with the tt-metal build and
Python environment active (`ttnn` comes from that build, not PyPI):

```bash
cd "$TT_METAL_HOME"   # or wherever your tt-metal checkout is
source python_env/bin/activate
git submodule update --init models/experimental/chronos_forecast/third_party/chronos-forecasting
pip install -r models/experimental/chronos_forecast/requirements.txt
pip install -U "huggingface_hub[cli]"
hf download amazon/chronos-2 --local-dir models/experimental/chronos_forecast/weights/chronos-2
```

`weights/` is gitignored. The demo and the real-checkpoint accuracy tests need
the download above; tests that need it skip or error without it. The TTNN
tests and benchmarks need Blackhole hardware (the paper-shape numbers below
were measured on p150a cards).

Pinned Chronos-2 copy: commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`. See [reference/PROVENANCE.md](reference/PROVENANCE.md).



## Demo

Single `Chronos2Model.forward` (CPU):

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py
```

Chronos-1 tokenizer only:

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py --tokenizer-only --context-length 16
```

## TTNN trace

The fixed paper benchmark (`batch=1024`, context `2048`, forecast `64`) has a
device-resident path and an address-stable TTNN trace runner. It keeps the
embeddings, 12-layer encoder, and output head on device, specializes group
attention when every series has a unique group ID, and refreshes fixed input
slots between replays.

[optimizations.md](optimizations.md) logs every optimization on the branch,
ranked by single-chip time saved, with code links, the hardware limit each one
addresses, and how data-parallel scaling holds up.

Accuracy and lifecycle:

```bash
source python_env/bin/activate
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_modules.py \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py
```

Paper-shape performance (20 replay and 20 end-to-end iterations):

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -s
```

## Data parallel (4, 8, 16, 32 chips)

Opening `TtChronos` on a mesh device runs it data parallel, with no chip-to-chip
traffic:

- Weights are copied to every chip.
- `upload_inputs` splits the series along dim 0, and `postprocess_output`
  concatenates the per-chip outputs.
- Time attention stays within a series, and group attention stays within a
  group, so each chip runs the single-chip model on its share.
- Unique-group batches are padded with dummy series to a multiple of the chip
  count.
- Grouped batches are packed into whole blocks with the block count padded to a
  multiple of the chip count, so no group spans two chips.

The PCC, accuracy and trace tests are parametrized over `1`, `4`, `(1, 8)`,
`(2, 8)` and `(4, 8)`. Shapes larger than the machine are skipped.

```bash
PYTHONPATH=. pytest -s \
  models/experimental/chronos_forecast/tests/pcc/test_paper_shape_pcc.py \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py
PYTHONPATH=. pytest -s \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -k data_parallel
```

`TtChronosTraceRunner.stream(items, prepare)` prepares batch i+1 on the host
while replay i runs, for back-to-back batches.

Paper shape (1024 series, performance precision, L1-resident, unique groups) on
p150a cards, at `86be1507afa`:

Note: benchmarks start with inputs on the host and are transferred over PCIe gen4.

| Chips | Series per chip | Replay | Serial end-to-end | Streamed per batch | Streamed series/s |
|---|---|---|---|---|---|
| 1 | 1024 | 260 ms | 322 ms | 311 ms | 3,290 |
| 4 | 256 | 69 ms | 113 ms | 97 ms | 10,570 |

On 4 chips, replay is 3.77× faster than one chip (94% scaling efficiency). With
groups of 4, replay is 106 ms (3.61×) and a streamed batch takes 129 ms. The
remaining host cost is serial readback (about 20 ms) and upload (about 9 ms).
Per-chip replay times from single-chip runs put 8, 16 and 32 chips at about
32, 16 and 9 ms, so from 16 chips up the host limits throughput unless readback
also overlaps the replay.

### After the optimization round (`10ec7ff655e`)

Same shape and configuration (`performance_l1`), measured at `10ec7ff655e` on
`chronos-forecast-experi`:

| Chips | Series per chip | Replay | Serial end-to-end | Streamed per batch | Streamed series/s |
|---|---|---|---|---|---|
| 1 | 1024 | 153 ms | 199 ms | 188 ms | 5,440 |
| 4 | 256 | 35 ms | 82 ms | 54 ms | 18,900 |

The changes since `86be1507afa`:

- Sweep-tuned L1 matmul configs, with the RMSNorm gamma folded into the
  following linears.
- LoFi attention matmuls.
- Model-local `generic_op` kernels in `ops/`: RoPE, a bank-local residual add,
  the QKV head split fused with RoPE, and an RMSNorm that writes bfloat8_b.
- The input embedding and the output head run inside the L1 chunks.

See [optimizations.md](optimizations.md) for each change's measured saving and
code references.

On 4 chips, replay is 4.36× one chip at the full batch. A chip holding 256
series is slightly more efficient per series than one holding 1024; against a
single chip at 256 series, 4 chips reach 99%.

With groups of 4:

| Chips | Replay | Streamed per batch | Replay speedup |
|---|---|---|---|
| 1 | 269 ms | 300 ms | 1× |
| 4 | 68 ms | 86 ms | 3.95× |

On 4 chips the host now limits streaming. The 35 ms replay is followed by a
serial readback of about 17 ms and an upload of about 3 ms. Host prepare
(about 17 ms) overlaps the replay. Single-chip replay at 128, 64 and 32 series
takes 17.5, 8.8 and 6.1 ms, which is roughly where 8, 16 and 32 chips would
land on replay alone.

Blackhole Galaxy bring-up:

- Run the commands above there. The perf report prints the worker grid.
- The L1 chunk budgets (`_L1_CHUNK_TOKENS_*` in `tt/program_configs.py`) were
  measured on the p150a's 11x10 grid. `TtChronos` logs a warning on any other
  grid; if you see it, re-measure the budgets on one chip with
  `sweeps/sweep_l1_chunk.py`.


## Running the tests


From the tt-metal repo root:

```bash
source python_env/bin/activate
```

Core Python/unit tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/test_*.py \
  -s
```

TTNN module PCC tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_modules.py \
  -s
```

TTNN trace PCC tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py \
  -s
```

Eager paper-shape performance test:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward.py \
  -s \
  --timeout=3600
```

Trace-replay performance test (paper shape):

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -s \
  --timeout=3600
```

Single-chip, unique groups only:

```bash
PYTHONPATH=. pytest -s \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -k "performance_l1 and not groups"
```

`TtChronosTraceRunner` is TT Metal trace capture/replay, not a resident
persistent compute kernel. A true persistent kernel would require porting the
full transformer into unified device kernels under slow dispatch.

## References
- Paper: [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Upstream: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
