# MoE dispatch-column workload capture — runbook

Capture, per **dispatch invocation**, which dispatch/combine column carries the most routing
load during a chunked prefill, then turn that into replay input for
`tests/perf/test_dispatch_combine_perf.py`.

A chunked prefill calls dispatch once per **(layer, chunk)**. This measures each of those calls
separately rather than aggregating over the ISL, because a layer that spikes in one chunk gets
averaged away against its quiet chunks — for Kimi the per-invocation top case is layer 45 /
chunk 4 at 38.6%, while the aggregate ranking puts layer 11 first at 35.0% and never surfaces
layer 45 at all.

Branch: `ns/moe-column-workload-probe`.

---

## What the metric is

On an 8x4 Galaxy with `cluster_axis=0`, dispatch groups run along mesh **columns**, so column
`c` owns routed experts `[c * experts_per_col, (c + 1) * experts_per_col)` where
`experts_per_col = num_routed_experts // 4`:

| Model | Routed experts | Per column | Per chip |
|---|---|---|---|
| DeepSeek V3 | 256 | 64 | 8 |
| Kimi K2.6 | 384 | 96 | 12 |

A column's load is the number of top-k picks landing in its expert range. Uniform routing gives
25%; the hot cases measured here run 36–39%.

Tokens are sharded across mesh **rows** and replicated across **columns**
(`ShardTensor2dMesh(dims=(0, None))`), so one column's 8 devices already hold every token of a
chunk. The probe reads back only those 8 index tensors per dispatch call, not all 32.

---

## Step 1 — prerequisites on the target machine

```bash
cd <tt-metal>
git fetch origin
git checkout ns/moe-column-workload-probe
source python_env/bin/activate            # needs torch + ttnn + safetensors
export TT_METAL_HOME=$(pwd)
```

Check the four things that actually block this run:

```bash
# 1. 32-device Blackhole Galaxy (the 8x4 mesh)
ls /dev/tenstorrent | grep -c '^[0-9]'          # expect 32

# 2. Disk. Kernel JIT lands on the root fs; the capture is ~108 MB per model.
df -h / && df -h /data

# 3. Host power. Kimi's test skips below 130 W TDP (see step 3).
python3 -c "from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_tdp_limit_max, is_high_power; \
print('TDP_LIMIT_MAX', get_tdp_limit_max(), 'is_high_power', is_high_power())"

# 4. Both code_debug traces staged, each with 56320 token ids = 11 x 5120
for t in kimi_debug_55k_vllm deepseek_debug_55k_vllm; do
  python3 -c "import json; p='/mnt/models/deepseek-prefill-cache/golden/structured_traces/$t/metadata.json'; \
print('$t', len(json.load(open(p))['token_ids']))"
done
```

If root is tight, the safe thing to clear is the compiled-kernel cache — it is regenerable and
costs only a JIT recompile on the next run. Do **not** clear `~/.cache/huggingface` reflexively;
those are model weights and expensive to refetch.

```bash
du -xsh ~/.cache/*            # look at tt-metal-cache
rm -rf ~/.cache/tt-metal-cache
```

Pick an output directory with room — not the root fs:

```bash
export OUTDIR=/data/$USER/moe_workload && mkdir -p "$OUTDIR"
```

---

## Step 2 — DeepSeek V3, code_debug, one forward pass

```bash
export MESH_DEVICE=TG LOGURU_LEVEL=INFO
export DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/CI
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
export DEEPSEEK_V3_TRACE_DIR=/mnt/models/deepseek-prefill-cache
export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/deepseek_debug_55k_vllm

export TT_DS_MOE_WORKLOAD_PROBE=1
export TT_DS_MOE_ROUTING_CAPTURE=1
export TT_DS_MOE_WORKLOAD_PROBE_OUT="$OUTDIR/dsv3_code_debug"

mpirun --bind-to none --pernode --tag-output bash -lc '
  export OMP_NUM_THREADS=$(nproc)
  python3 -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_ds_prefill_transformer_chunked_no_pcc[blackhole-deepseek_v3-mesh-8x4-L61-chunks_eleven-iters1]" -xvs
'
```

DeepSeek has 3 dense layers, so 58 MoE layers x 11 chunks = **638 dispatch invocations**.

---

## Step 3 — Kimi K2.6, code_debug, one forward pass

```bash
export MESH_DEVICE=TG LOGURU_LEVEL=INFO
export KIMI_K2_6_HF_MODEL=/mnt/models/Kimi-K2_6-dequantized
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/kimi_debug_55k_vllm

export TT_DS_MOE_WORKLOAD_PROBE=1
export TT_DS_MOE_ROUTING_CAPTURE=1
export TT_DS_MOE_WORKLOAD_PROBE_OUT="$OUTDIR/kimi_k26_code_debug"

# Only if step 1 reported is_high_power False. Routing capture is not a timing measurement:
# at num_iters=1 print_duration_table bails before asserting anything, so host TDP is
# irrelevant. CI never sets this and keeps the gate.
export TT_DS_ALLOW_LOW_POWER=1

mpirun --bind-to none --pernode --tag-output bash -lc '
  export OMP_NUM_THREADS=$(nproc)
  python3 -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc[blackhole-kimi-mesh-8x4-L61-preload0-chunks_eleven-iters1-margin5pct]" -xvs
'
```

Kimi has 1 dense layer, so 60 MoE layers x 11 chunks = **660 dispatch invocations**.

### Do not set `TT_PREFILL_PROFILE_WARMUP=1`

It runs chunk 0 through every layer before the measured chunk loop starts, so those dispatch
calls land with a stale context and show up as `iter=-1` rows in the CSV.

---

## Step 4 — what each run produces

**`$OUTDIR/<name>.csv`** — one row per (chunk, layer, column); 638x4 = 2552 rows for DS,
660x4 = 2640 for Kimi:

```
iter,chunk,layer,col,in_col_picks,fabric_picks,col_share_pct,row0_picks…row7_picks
```

* `in_col_picks` — picks owned by that column (~10240 of the 40960 per invocation at uniform)
* `fabric_picks` — same, excluding picks whose expert is on the source row (those are NOC, not fabric)
* `row0..row7` — intra-column skew across the 8 chips

**`$OUTDIR/<name>_expert_routing.safetensors`** — one key per dispatch invocation,
`expert_ids_layer_{L}_chunk_{C}`, each a flat int32 of 8 chips x 640 tokens x 8 top-k = 40960
ids (~160 KB); ~108 MB total.

The run also logs the hottest `(layer, col)` pairs and per-chunk column totals inline.

---

## Step 5 — rank invocations and emit the replay file

```bash
python3 models/demos/deepseek_v3_d_p/scripts/make_captured_routing.py \
    --source capture --in "$OUTDIR/dsv3_code_debug_expert_routing.safetensors" \
    --num-routed-experts 256 \
    --out "$OUTDIR/dsv3_replay/expert_routing.safetensors"

python3 models/demos/deepseek_v3_d_p/scripts/make_captured_routing.py \
    --source capture --in "$OUTDIR/kimi_k26_code_debug_expert_routing.safetensors" \
    --num-routed-experts 384 \
    --out "$OUTDIR/kimi_replay/expert_routing.safetensors"
```

Prints the **4 worst** (highest in-col share) and **4 nominal** (closest to 25%) invocations,
formatted to paste into `_REAL_INDICES_PICKS`. Add `--rank-only` to report without writing.

Both sets are de-duplicated by layer. Without that the worst four collapse onto four chunks of
one layer and replay near-identical routing instead of four distinct cases.

`--chunk` controls what is written:

| Value | Meaning |
|---|---|
| `picks` (default) | each selected layer gets the chunk **its own pick came from**, so every case replays one real dispatch invocation |
| `<int>` | force one chunk for all layers |
| `all` | concatenate the chunks — aggregated, hides per-chunk spikes |

`--chunk picks` works because the 8 picks land on 8 distinct layers, so one file serves all
cases. Two chunks of the *same* layer cannot coexist in one file (one key per layer) — that
would need two files.

The same script also works offline on an already-flat per-layer capture, using `--chunk-size`
to recover the per-invocation view (the row axis is token position, so chunk `c` is rows
`[c*5120, (c+1)*5120)`):

```bash
python3 models/demos/deepseek_v3_d_p/scripts/make_captured_routing.py \
    --source flat --in ~/captured_expert_routing_dsv3.safetensors \
    --num-routed-experts 256 --chunk-size 5120 --rank-only
```

---

## Step 6 — wiring into the perf test

Two things in `tests/perf/test_prefill_dispatch_combine.py` need attention before the emitted
file will load.

**Token count.** The worker hardcodes its geometry:

```python
[pytest.param(3200, 7168, 256, 8, 8, 8, id="perf_real_indices")]
#              ^seq_len_per_chip
```

and `load_captured_routing` raises unless each layer tensor is exactly
`dispatch_group_size * seq_len_per_chip * top_k` = 8 x 3200 x 8 = 204800. One dispatch
invocation is 8 x 640 x 8 = **40960**, five times short. Either set `seq_len_per_chip` to 640
so one case is one real invocation (recommended — the whole point is per-invocation fidelity),
or use `--chunk all --tokens 25600`, which loads unmodified but makes each case a blend of
chunks 0–4.

**Kimi's expert space.** That same param line is 256-expert / 8-per-chip; Kimi is 384 / 12, and
`load_captured_routing` additionally hard-raises on `num_routed_experts != 256`
(`tt/moe/init_helpers.py`). `make_captured_routing.py` emits Kimi ids in the raw 384 space
unchanged and does not touch that guard.

Then, per case:

```bash
export DEEPSEEK_V3_TRACE_DIR=<dir containing longbook_qa_eng_prefill_25600_nopad/expert_routing.safetensors>
export TT_DS_CAPTURED_LAYER=29
export TT_DS_CAPTURED_COL=0
```

The **column is not stored in the file** — each key holds the full expert space and
`load_captured_routing` masks to the requested column, sentinelling out-of-column picks so the
kernel skips them. So one key can be replayed as col 0/1/2/3; the column decides how much of it
becomes real work.

Finally, each of the 8 cases needs `expected_ns` baselines for dispatch and combine across both
topologies — 32 numbers in `_DISPATCH_REAL_INDICES_EXPECTED_NS` /
`_COMBINE_REAL_INDICES_EXPECTED_NS`. These must be measured on a **Loudbox**: the worker is an
LB 8-device mesh and a Galaxy rejects it with "Blackhole only supports 32-device mesh configs
(requested 8)".

---

## Reference: picks from earlier captures

Measured offline from pre-existing captures (`~/captured_expert_routing*.safetensors`), not
from a fresh run — use as a sanity check on what the run should produce, not as final values.

| | Kimi K2.6 | DeepSeek V3 |
|---|---|---|
| worst | (45, 2) 38.6% ch4 · (48, 0) 38.1% ch2 · (44, 1) 37.4% ch2 · (50, 1) 37.1% ch2 | (19, 2) 37.2% ch1 · (29, 0) 37.0% ch10 · (42, 3) 36.9% ch2 · (24, 1) 36.7% ch1 |
| nominal | (30, 3) ch0 · (32, 1) ch4 · (20, 2) ch9 · (6, 1) ch5 — all 25.0% | (4, 2) ch3 · (8, 3) ch10 · (56, 3) ch0 · (5, 0) ch8 — all 25.0% |

Per-chunk mean share sits flat at 29–30% across all 11 chunks for both models, so column skew
here is a **layer** property rather than a chunk-position one. Both models peak below the 43.2%
of the currently hardcoded `(27, 2)` pick, which came from longbook — a more skewed dataset than
code_debug.

Chunks are indexed **0–10** (56320 / 5120 = 11).
