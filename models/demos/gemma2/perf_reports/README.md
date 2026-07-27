# Gemma2 decode profiler sheets — Blackhole P150 / P300

Evidence pack for the three Gemma2 decode configurations we support. Every number
below is reproducible from a Tracy capture using `make_perf_report.sh` in this
directory.

Captured 2026-07-27. Tool: [`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report)
1.2.8 (`pip install tt-perf-report`). All runs at ISL=128 / OSL=200, batch-1,
performance mode.

## Model names used throughout

| Name | Model | Hardware | Parallelism |
| --- | --- | --- | --- |
| `gemma2-9B_1xp150` | `google/gemma-2-9b-it` | one P150 | none (single chip) |
| `gemma2-9B_2xp150` | `google/gemma-2-9b-it` | P300 (2x P150) | TP=2 |
| `gemma2-27B_2xp150` | `google/gemma-2-27b-it` | P300 (2x P150) | TP=2 |

## Files per model

| Suffix | Contents |
| --- | --- |
| `_decode_report.txt` | **One steady-state decode iteration** — the per-token view. This is the sheet to show. |
| `_decode_summary.csv` | Per-op-category rollup for that iteration (13 rows — start here). |
| `_decode_ops.csv` | Every individual op in that iteration, machine readable. |
| `_decode_breakdown.png` | Stacked device-time chart for that iteration. |
| `_full_report.txt.gz` | The whole capture: compile, prefill, decode. For prefill questions. |

The raw Tracy `ops_perf_results_*.csv` inputs are 35–73 MB each and are **not**
committed. Regenerate them with the capture commands at the bottom of this file.

---

## Where the time goes, per decode token

Percentages are of device kernel time. On the 2-device runs `tt-perf-report`
merges devices, so absolute millisecond sums cover both chips; the percentages
and the per-op bandwidth figures are unaffected.

| | gemma2-9B_1xp150 | gemma2-9B_2xp150 | gemma2-27B_2xp150 |
| --- | --- | --- | --- |
| Matmul | **78.8 %** | **61.2 %** | **77.3 %** |
| Collectives (AllGather + ReduceScatter) | 0 % | **16.2 %** | 10.6 % |
| LayerNorm | 4.2 % | 8.0 % | 4.2 % |
| SdpaDecode | 3.4 % | 4.9 % | 1.3 % |
| ArgMax (on-device sampling) | 5.6 % | — | — |
| Everything else | ~8 % | ~9.7 % | ~6.6 % |

Two conclusions fall straight out of this table:

- **Decode is a matmul problem.** Attention (SdpaDecode) is 1–5 %. Tuning SDPA,
  norms or head ops cannot move the needle; matmul bandwidth *is* the workload.
- **TP=2 pays a 16 % communication tax.** On 9B the collectives are pure overhead
  that does not exist on one chip, and LayerNorm does not shard (8.0 % vs 4.2 %).
  That is the concrete reason 2xP150 does not double 1xP150.

---

## The matmul ceiling — the most important finding

`tt-perf-report` tags **every** decode matmul in all three configs as `SLOW`
(210 / 180 / 269 ops respectively) and reports achieved DRAM bandwidth per op.

| | time-weighted matmul BW | % of peak | cores used |
| --- | --- | --- | --- |
| `gemma2-9B_1xp150` | 278 GB/s | 54.3 % | 12 of 110 |
| `gemma2-9B_2xp150` | 209 GB/s | 40.8 % | 12 of 110 |
| `gemma2-27B_2xp150` | 263 GB/s | 51.4 % | 12 of 110 |

The decisive comparison is **within a single sheet** — same kernel, same silicon,
same iteration, only the weight dtype differs:

| Matmul in `gemma2-9B_1xp150` | weights | achieved BW |
| --- | --- | --- |
| FF1/FF3 `32 x 3584 x 14336` | bfp4 | 263 GB/s |
| FF2 `32 x 14336 x 3584` | bfp4 | 257 GB/s |
| QKV `32 x 4096 x 3584` | bfp4 | 235 GB/s |
| **LM head `32 x 3584 x 16032`** | **bfp8** | **446 GB/s** |

The bf8 LM head reaches 87 % of peak through the very same `dram_sharded` kernel.
So the ~250 GB/s our bf4 matmuls achieve is **not** a memory-system limit — it is
the bf4 dequantization path in the kernel. This is the evidence for the metal
team, and it is worth roughly 38 % end-to-end throughput if closed.

`gemma2-27B_2xp150` shows the same signature: bf8 LM head at 311 GB/s vs bf4
layers at 259–267 GB/s.

Note also **12 cores of 110**. The DRAM-sharded decode matmul is bank-limited,
not core-limited, which is why core-grid sweeps produced almost nothing and why
a 16-core FF experiment actually regressed 17 %.

---

## Other findings worth keeping

- **On-device ArgMax costs 1.3 ms/token (5.6 %)** on `gemma2-9B_1xp150` — a single
  op over the 256K vocab. Enabling it was still a large net win (it replaced a
  ~7.5 ms host readback), but it is now the third-largest line item and the next
  obvious target on that config.
- **`gemma2-9B_2xp150` matmuls are the least efficient of the three** (209 GB/s,
  40.8 %) because TP=2 halves each shard, so per-matmul fixed overhead is
  amortised over half the work. Sharding harder makes each matmul relatively worse.
- The op-to-op gap advice reports only ~327 μs (1.3 %) recoverable on
  `gemma2-9B_1xp150`, confirming dispatch overhead is already well hidden by tracing.

---

## Reproducing

### 1. Capture a Tracy profile

```bash
cd $TT_METAL_HOME && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
       TT_METAL_PROFILER_CPP_POST_PROCESS=1
MGD=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors
```

Then pick one configuration:

```bash
# gemma2-9B_1xp150
export HF_MODEL=google/gemma-2-9b-it
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 \
       TT_MESH_GRAPH_DESC_PATH=$MGD/p150_mesh_graph_descriptor.textproto
export TT_FORCE_DEVICE_SAMPLING=1

# gemma2-9B_2xp150
export HF_MODEL=google/gemma-2-9b-it
export TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 \
       TT_MESH_GRAPH_DESC_PATH=$MGD/p300_mesh_graph_descriptor.textproto
export TT_CREATE_HEADS_MD=1

# gemma2-27B_2xp150
export HF_MODEL=google/gemma-2-27b-it
export TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 \
       TT_MESH_GRAPH_DESC_PATH=$MGD/p300_mesh_graph_descriptor.textproto
export TT_CREATE_HEADS_MD=1
```

Keep the token count small so the decode region stays a manageable size; 6 tokens
yields 7 usable iterations:

```bash
python3 -m tracy -p -r -v --op-support-count 60000 -o <outdir> -t 9000 -m \
  "pytest models/demos/gemma2/demo/text_demo.py::test_demo_text \
   -k 'batch-1 and performance' -s \
   --input_prompts <isl128_prompt.json> \
   --stop_at_eos 0 --max_generated_tokens 6"
```

### 2. Turn it into these sheets

```bash
pip install tt-perf-report
./make_perf_report.sh <outdir>/reports/*/ops_perf_results_*.csv <n_layers> <model-name>
```

`<n_layers>` is 42 for 9B and 46 for 27B. The script isolates a single decode
iteration by using the SdpaDecode calls as delimiters — there are exactly
`n_layers` of them per token — and picks a middle iteration so neither first-run
compile effects nor a truncated tail can skew it.
