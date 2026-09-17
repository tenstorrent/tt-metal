# Qwen3.5-397B attention perf on QB2 (4x Blackhole, TP=4)

Per-op device perf of one **Gated DeltaNet** (linear attention) layer and one **gated full
attention** (GQA) layer, Qwen3.5-397B-A17B config. Random weights, perf only — no PCC. Timing is
`DEVICE KERNEL DURATION` from Tracy, merged across the 4 devices to the critical path
(collectives averaged, everything else MAX).

## Setup

Build with the profiler (on by default): `./build_metal.sh`. Then, from the repo root:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_CACHE=$PWD
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
export MESH_DEVICE=P150x4
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000   # default 1000 overflows past ~22 chunks

# No 397B weights needed: Qwen36ModelArgs reads only config.json. The directory MUST be named
# Qwen3.5-397B (its basename becomes model_name). The test copies tokenizer files into it from any
# cached ~/.cache/huggingface/hub/models--Qwen--Qwen3.*-27B snapshot.
cp -r models/demos/blackhole/qwen36/tests/perf/configs/Qwen3.5-397B /tmp/
export GDN_CFG_DIR=/tmp/Qwen3.5-397B HF_MODEL=/tmp/Qwen3.5-397B HF_HUB_OFFLINE=1
```

## Run

One test file, four tests. Profile **one `-k` selector per invocation** — the signpost filter keeps
every region in the CSV, so a selector matching two cases silently sums two runs. Ids end in
`-device_params0-1x4`; the `-device` suffix keeps `B1` from also matching `B16`/`B128`.

| test | ids | parser flags |
|---|---|---|
| `test_gdn_prefill` | `gdn_pf_isl{1024,4096,8192,16384,32768,65536,128000}` | `--isl <ISL> --chunks <ISL/2048, 63 for 128000> --layers 45` |
| `test_gdn_decode` | `gdn_dec_B{1,2,4,8,16,32,64,128,256,512}` | `--isl <B*16> --chunks 16 --layers 45` |
| `test_attn_prefill` | `attn_pf_isl{4096,8192,16384,32768,65536,128000}` | `--isl <ISL> --chunks <ISL/2048> --layers 15` |
| `test_attn_decode` | `attn_dec_B{1,16,32}-ctx{8k,128k}` | `--isl <B*16> --chunks 16 --layers 15` |

```bash
P=models/demos/blackhole/qwen36/tests/perf
T=$P/test_qwen35_397b_perf_TP4.py

run() {  # run <id> <parser flags...>
  python -m tracy -r -p -v -n "$1" -m pytest "$T" -k "$1-device"
  python $P/parse_gdn_perf.py generated/profiler/reports/"$1"/*/ops_perf_results_*.csv "${@:2}"
}

run gdn_pf_isl4096        --isl 4096 --chunks 2  --layers 45
run gdn_dec_B16           --isl 256  --chunks 16 --layers 45
run attn_pf_isl4096       --isl 4096 --chunks 2  --layers 15
run attn_dec_B16-ctx8k    --isl 256  --chunks 16 --layers 15

# GDN decode B=32 fits L1 only with both knobs; B>=64 does not run on main (L1 OOM).
QWEN35_GDN_DECODE_BF16=1 QWEN35_GDN_STATE_BF16=1 run gdn_dec_B32 --isl 512 --chunks 16 --layers 45
```

The parser prints a per-op table, a stage rollup, and `TOTALS`: `total device time` is the
one-layer whole-ISL total; `per-chunk total` is per 2048-token chunk (prefill) or per step (decode).

## Results — one layer, TP=4

Prefill = whole-ISL total for one layer; decode = per step. Multiply by 45 (GDN) or 15 (attention)
for all layers of that type. GDN decode does not depend on context length (fixed-size recurrent
state); attention does.

### GDN (linear attention)
`head_dim 256 · linear_key_head_dim 128 · linear_num_key_heads 16 · linear_num_value_heads 64 · linear_value_head_dim 128 · conv_kernel 4 · hidden 4096`

| phase | batch | seq_len | time (µs) |
|---|---|---|---|
| prefill | 1 | 4 096 | 7 549.9 |
| prefill | 1 | 8 192 | 15 063.6 |
| prefill | 1 | 16 384 | 30 171.6 |
| prefill | 1 | 128 000 | 236 458.5 |
| decode | 1 | any | 292.6 |
| decode | 16 | any | 772.5 |
| decode | 32 † | any | 926.8 |

† `QWEN35_GDN_DECODE_BF16=1` + `QWEN35_GDN_STATE_BF16=1`; fp32 OOMs at B=32.

### GQA (gated full attention)
`head_dim 256 · num_attention_heads 32 · num_key_value_heads 2 · hidden 4096`

| phase | batch | seq_len | time (µs) |
|---|---|---|---|
| prefill | 1 | 4 096 | 3 902.8 |
| prefill | 1 | 8 192 | 10 471.3 |
| prefill | 1 | 16 384 | 31 498.3 |
| prefill | 1 | 128 000 | 1 391 898.6 |
| decode | 1 | 8 192 | 217.8 |
| decode | 16 | 8 192 | 598.2 |
| decode | 32 | 8 192 | 988.3 |
| decode | 1 | 128 000 | 563.6 |
| decode | 16 | 128 000 | 5 333.1 |
| decode | 32 | 128 000 | 10 432.7 |

Measured 2026-09-17 on `sjc2-bhqb-e22a`, tt-metal `1c2b492ea1e`. Re-runs land within ±0.6%.

## Files

| file | what |
|---|---|
| `test_qwen35_397b_perf_TP4.py` | the four tests; config-driven via `GDN_CFG_DIR` (also runs Qwen3.8-27B) |
| `parse_gdn_perf.py` | Tracy CSV → per-op / stage tables with the 4-device critical-path merge |
| `configs/Qwen3.5-397B/config.json` | dense-shim config carrying the 397B GDN/attention dims |
| `__init__.py` | empty; pytest needs it to collect the directory |
