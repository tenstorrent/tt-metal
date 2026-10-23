# Host IO Decoder Sweep Runbook

This guide explains how to run the DeepSeek V3 B1 Host IO decoder sweep against
converted GPU traces. The same entrypoint supports both a single decoder layer
and a rank-parallel multi-layer run:

```bash
python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep
```

The sweep loads a converted reference trace, injects `trace["input"]` into one
or more selected `HostIoDecoderStage` layers, collects the TT output, and
optionally compares against `trace["output"]` and
`kv_cache_reference_<prompt>.pt`.

## Required Trace Layout

Choose a trace root and prompt stem:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TRACE_ROOT=/path/to/converted_traces/TRACE_NAME
export PROMPT=PROMPT_NAME
export LAYERS="4 5 6 7"
```

The sweep expects one directory per converted decoder layer:

```text
$TRACE_ROOT/
  layer_00/
  layer_01/
  ...
  layer_60/
```

Each layer directory contains:

```text
$TRACE_ROOT/layer_04/
  <prompt_name>.pt
  kv_cache_reference_<prompt_name>.pt
```

The required file contract is:

```text
<prompt>.pt:
  dict with "input" and "output"
  input/output shape: (seq_len, 7168)
  input/output dtype: torch.bfloat16

kv_cache_reference_<prompt>.pt:
  tensor shape: (1, seq_len, 576)
  dtype: torch.bfloat16
```

`kv_cache_reference_<prompt>.pt` is only required when
`--validate-kv-cache-cross-trace` is enabled.

## Converted Trace Prerequisite

This runbook assumes the layer directories under `$TRACE_ROOT` have already been
converted into the `<prompt>.pt` and `kv_cache_reference_<prompt>.pt` files
shown above.

Keep `--decoder-layer-indices` aligned with the layer id used to generate each
directory. The prompt filename does not encode the layer id.

## Check Converted Traces

Run this before opening devices if you are unsure whether the traces are valid:

```bash
cd "$TT_METAL_HOME"

python_env/bin/python - <<'PY'
import os
from pathlib import Path
import torch

trace_root = Path(os.environ["TRACE_ROOT"])
prompt = os.environ["PROMPT"]

for layer in [int(x) for x in os.environ["LAYERS"].split()]:
    trace_dir = trace_root / f"layer_{layer:02d}"
    trace = torch.load(trace_dir / f"{prompt}.pt", map_location="cpu")
    kv = torch.load(trace_dir / f"kv_cache_reference_{prompt}.pt", map_location="cpu")
    print(
        f"layer_{layer:02d}: "
        f"input={tuple(trace['input'].shape)} {trace['input'].dtype}, "
        f"output={tuple(trace['output'].shape)} {trace['output'].dtype}, "
        f"kv={tuple(kv.shape)} {kv.dtype}"
    )
PY
```

Expected shapes depend on the trace length. For an 8192-token generation trace,
they look like:

```text
input=(8257, 7168) torch.bfloat16
output=(8257, 7168) torch.bfloat16
kv=(1, 8257, 576) torch.bfloat16
```

## Single-Layer Sweep

Run the single-layer validation:

```bash
cd "$TT_METAL_HOME"

TT_METAL_SLOW_DISPATCH_MODE=1 \
python_env/bin/python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \
  --decoder-layer-indices 4 \
  --hidden-states-dir "$TRACE_ROOT/layer_04" \
  --prompt "$PROMPT" \
  --num-replication-slots 1 \
  --validate-hidden-states-cross-trace \
  --validate-kv-cache-cross-trace \
  --pcc-threshold 0.97 \
  --kv-cache-pcc-threshold 0.97
```

Single-layer defaults are still useful for local determinism checks:

- `--num-replication-slots` defaults to `8`.
- `--validate-hidden-states-cross-slot` defaults to on.
- `--validate-kv-cache-cross-slot` defaults to on.
- `--validate-hidden-states-cross-trace` defaults to off.

For cross-trace correctness against one GPU reference, `--num-replication-slots 1` is usually enough.

## Multi-Layer Parallel Sweep

For independent per-layer checks in parallel, launch one rank per layer. With:

```bash
--decoder-layer-indices 4 5 6 7
```

the rank mapping is:

```text
rank 0 -> layer 4
rank 1 -> layer 5
rank 2 -> layer 6
rank 3 -> layer 7
```

Run the multi-layer test with `tt-run`:

```bash
cd "$TT_METAL_HOME"

export HOSTSP=host0:1,host1:1,host2:1,host3:1
export RANK_BINDINGS_MAPPING=decoder_verify_4x_rank_bindings_mapping.yaml
export TCP_INTERFACE=NETWORK_INTERFACE

TT_METAL_SLOW_DISPATCH_MODE=1 python_env/bin/tt-run \
  --tcp-interface "$TCP_INTERFACE" \
  --rank-bindings-mapping "$RANK_BINDINGS_MAPPING" \
  --mpi-args "--host ${HOSTSP} --map-by slot --bind-to none --oversubscribe --tag-output" \
  python_env/bin/python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \
    --decoder-layer-indices $LAYERS \
    --hidden-states-dir "$TRACE_ROOT" \
    --prompt "$PROMPT" \
    --validate-kv-cache-cross-trace \
    --pcc-threshold 0.97 \
    --kv-cache-pcc-threshold 0.97
```

Replace `HOSTSP`, `RANK_BINDINGS_MAPPING`, and `TCP_INTERFACE` with values for
your allocation. `HOSTSP` must describe the same number of ranks as `LAYERS`.
For example, four layer ids require four MPI ranks.

The rank-bindings mapping should point each launcher rank/subcontext at the
single-rank binding file that exposes that rank's local 4x2 mesh:

```yaml
subcontext_id_to_rank_bindings:
  0: decoder_verify_rank0_binding.yaml
  1: decoder_verify_rank1_binding.yaml
  2: decoder_verify_rank2_binding.yaml
  3: decoder_verify_rank3_binding.yaml
```

Each `decoder_verify_rank<N>_binding.yaml` can contain `rank: 0` internally;
the mapping file selects which binding file each global launcher rank uses.
For 8-layer or 16-layer runs, extend `LAYERS`, `HOSTSP`, and the mapping file
to the same count.

Parallel-mode defaults are tuned for independent layer checks:

- `--num-replication-slots` defaults to `1`.
- `--validate-hidden-states-cross-trace` defaults to on.
- `--validate-hidden-states-cross-slot` defaults to off.
- `--validate-kv-cache-cross-slot` defaults to off.
- `--validate-kv-cache-cross-trace` is opt-in because it pulls KV cache.

## Dumps

Hidden-state and KV-cache dumps are off by default. Enable them only when you
need artifacts:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
python_env/bin/python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \
  --decoder-layer-indices 4 \
  --hidden-states-dir "$TRACE_ROOT/layer_04" \
  --prompt "$PROMPT" \
  --num-replication-slots 1 \
  --validate-hidden-states-cross-trace \
  --dump-hidden-states \
  --dump-dir "$TRACE_ROOT/tt_dump_layer_04"
```

For parallel dumps, a template keeps each rank separate:

```bash
--dump-dir "$TRACE_ROOT/tt_dump/layer_{layer:02d}"
```

If `--dump-dir` has no format fields in parallel mode, each rank writes under
`layer_<idx>` below that root.

## Troubleshooting

- If the CLI says the number of layer ids does not match world size, launch the
  same number of ranks as `--decoder-layer-indices` values.
- If a trace is missing, check `--hidden-states-dir` and, in parallel mode, the
  per-layer subdirectory resolved internally for that rank's layer id.
- If PCC is low but not random, verify `--decoder-layer-indices` matches the layer
  used when converting the trace.
- If KV-cache validation fails while hidden-state validation passes, verify the
  converted trace used the same RoPE layout and tensor ordering expected by this
  sweep; a mismatch in RoPE application can leave hidden states close while
  causing KV-cache comparisons to fail.
- If `tt-run` is not on `PATH`, load the usual TT runtime environment before
  launching the 4-layer command.
