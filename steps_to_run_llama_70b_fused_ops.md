# Llama 70B Galaxy - Fused Ops Run Commands


## Run Commands (with both RS+MM and AG+MM fused ops)

| Seq Len | Command |
|---------|---------|
| 4K | `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "long-4k-b1" -v --timeout=0` |
| 8K | `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "long-8k-b1" -v --timeout=0` |
| 16K | `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "long-16k-b1" -v --timeout=0` |
| 32K | `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "long-32k-b1" -v --timeout=0` |
| 64K | `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "long-64k-b1" -v --timeout=0` |
| 128K | `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "long-128k-b1" -v --timeout=0` |

## TTFT Comparison (Time to First Token)

Baseline from `teja_RS+MM/tt-metal` (separate MM+RS, no fused ops). Fused Ops = RS+MM (FF1/FF3) + AG+MM (FF2).

| Seq Len | Baseline (ms) | Fused Ops (ms) | Improvement (%) |
|---------|---------------|----------------|-----------------|
| 4K | 617.96 | 587.96 | 4.9% |
| 8K | 988.80 | 945.7 | 4.5% |
| 16K | 1935.79 | 1810.16 | 6.5% |
| 32K | 4298.79 | 4042.68 | 6.0% |
| 64K | 10638.35 | 10027.41 | 5.7% |
| 128K | 29679.41 | 28867.07 | 2.7% |

**Note (128K):** Only AG+MM fused is used; RS+MM is disabled due to DRAM OOM (full MM output materialization exceeds available memory with KV cache).

## 64K Analysis

| Config | TTFT (ms) | vs Baseline |
|--------|-----------|-------------|
| Baseline (no fused) | 10638.35 | - |
| AG+MM only | 10321.74 | -317 ms |
| RS+MM only | 10168.59 | -470 ms |
| **Both combined** | **10027.41** | **-611 ms** |

**Expected if additive:** 317 + 470 = 787 ms → 9851 ms. **Actual:** 10027 ms. **Gap:** ~175 ms.

**Why the gap (consistent across runs):** MLP pipeline is FF1 → FF3 → mul → FF2 (sequential), so per-layer savings should add. With variance ruled out, likely causes: (1) **Critical-path shift** – 80 layers alternate Attention + MLP; speeding up MLP can make Attention the new bottleneck, so total time doesn’t drop linearly; (2) **Memory bandwidth saturation** – both fused ops may contend for DRAM/PCIe when run back-to-back. The combined 611 ms (5.7%) gain remains solid.

**Verification commands:** AG+MM only: `USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=0`; RS+MM only: `USE_FUSED_AG_MM=0 TT_LLAMA_USE_FUSED_MM_RS=1`.

## Profiler Sweep (64K analysis)

Scripts copied from `teja_RS+MM/tt-metal`: `scripts/run_profiler_sweep.sh`, `scripts/parse_profiler_report.py`. Run from repo root with `LLAMA_DIR` set.

**Important:** Tracy profiler has a 32K source-location limit. **64K prefill exceeds this** and the trace gets truncated (only ~19 ops through SDPA, no MLP). Use **4k or 8k** for profiling instead:

```bash
cd ~/llama-weights/tt-metal
source python_env/bin/activate
export LLAMA_DIR=/home/tvardhineni/llama-weights/Meta-Llama-3.3-70B-Instruct/original

# Baseline (use 4k or 8k - 64k hits Tracy limit)
USE_FUSED_AG_MM=0 TT_LLAMA_USE_FUSED_MM_RS=0 ./scripts/run_profiler_sweep.sh --run-name baseline --prompt-lengths 4k

# AG+MM only
USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=0 ./scripts/run_profiler_sweep.sh --run-name agmm_only --prompt-lengths 4k

# RS+MM only
USE_FUSED_AG_MM=0 TT_LLAMA_USE_FUSED_MM_RS=1 ./scripts/run_profiler_sweep.sh --run-name rsmm_only --prompt-lengths 4k

# Both
USE_FUSED_AG_MM=1 TT_LLAMA_USE_FUSED_MM_RS=1 ./scripts/run_profiler_sweep.sh --run-name both --prompt-lengths 4k
```

Comparison scripts (`compare_profiler_results.py`, `compare_ops_raw.py`, `compare_two_runs.py`, `csv_to_xlsx_with_gradient.py`) are optional; the sweep will skip them if missing. Parsed reports are still produced.
