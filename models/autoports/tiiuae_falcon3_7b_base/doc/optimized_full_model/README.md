# Falcon3-7B-Base optimized full model

| warmed batch-1 metric | completed full model | optimized full model | change |
|---|---:|---:|---:|
| TTFT, 128-token prompt | 24.658 ms | 23.562 ms | 4.4% faster |
| traced teacher-forcing decode | 75.678 t/s/u | 110.813 t/s/u | 46.4% faster |
| traced token-out decode, device-only | 75.687 t/s/u | 110.836 t/s/u | 46.4% faster |
| traced token-out decode, caller-visible | 75.418 t/s/u | 110.384 t/s/u | 46.4% faster |
| model trace | 12.417 ms/token | 8.227 ms/token | 33.7% faster |
| split sampling trace | 0.794 ms/token | 0.794 ms/token | unchanged |

The target is four Blackhole p300c devices in a 1x4 `FABRIC_1D_RING` with TP4 and two links. Measurements are warmed batch 1, prompt 128 / generation 128, real 28-layer weights, fallback exceptions enabled, and nonblocking split model/sampling trace replay. The baseline and final records are [baseline/full_model_evidence.json](results/baseline/full_model_evidence.json) and [final/full_model_evidence_dynamic_rope.json](results/final/full_model_evidence_dynamic_rope.json).

## Selected full-path changes

The decoder policy is unchanged: BFP4/LoFi weights, BFP8 attention/MLP/CCL/KV, persistent two-link asynchronous collectives, and the BF16 replicated-mesh L1 width-sharded inter-layer residual. No rejected datatype, fidelity, replicated stream, or broad datatype sweep was used.

The LM head now projects each rank's complete 32,768-column local vocabulary in one DRAM-sharded matmul. This removes three launches, three layout conversions, and the terminal concat. Real-weight one-layer candidates measured 1.358, 1.334, and 1.319 ms/token for 8,192, 16,384, and 32,768 columns respectively, with identical generated tokens.

The main stack gap was a context-sized RoPE lookup table. A nominal 32K model paid that DRAM lookup cost in every layer even for a 128-token request. The model now starts with one shared 256-row device table and grows it, before trace capture, in 256-row increments to the active request horizon. Positions remain absolute and device resident; decode still advances token, position, and RoPE state inside the traced path. The maximum-context gate grows to all 32,768 rows and passes, so advertised capability is unchanged.

Canonical split sampling is preserved. Local sampler-ready logits feed `Sampling1D`; `tt_out_tok` is copied directly into the next model-trace input; unchanged page tables cause no host copy; greedy and top-k/top-p paths are trace capable. Greedy split sampling is 0.800 ms versus 1.003 ms for force-argmax and 1.009 ms for generic full-vocabulary TT sampling, with the same token. No per-token host synchronization, token readback, or runtime fallback occurs in the measured device-only path.

## Lower bound and profiler conclusions

At the true 32K capability setting, the old fixed RoPE allocation produced a fitted 0.45405 ms/layer and 12.944 ms for 28 layers. Dynamic active-horizon RoPE produces 0.28556 ms/layer, 0.23030 ms fixed model work, and R² 0.9999999. The independently selected optimized multichip layer is 0.28660 ms. The measured 28-layer model trace is 8.2267 ms, matching `28 x layer + fixed terminal` within measurement noise; token-out adds the separately measured 0.7940 ms sampler trace.

Current-source reduced real-weight Tracy and `tt-perf-report` artifacts are under [tracy/current_source](tracy/current_source); metric reconciliation is in [perf_summary.json](perf_summary.json). The selected one-piece LM-head matmul is about 183 us and eliminates LM-head concat. The DRAM-sharded program inherits the preserved 32-core residual shard and exposes no grid/output-subblock knobs; K is 96 tiles, or three tiles/core, so the legal `in0_block_w` sweep is exactly 1 and 3. Width 3 wins (1.312 versus 1.439 ms token-out). An adapted 8-core terminal reshard is physically blocked by L1 (2.192 MB required versus 1.573 MB/core), while the runnable adapted 16-core path is 1.56% slower end-to-end (1.333 versus 1.312 ms) with identical tokens. Persistent CCL resources, ring topology, sharded residual layout, and kernel policy remain the selected optimized-multichip settings.

## Correctness and serving contract

- AIME24 prefill: top-1 92%, top-5 100%, top-100 100%.
- AIME24 teacher forcing: top-1 93%, top-5 100%, top-100 100%.
- Maximum context: 32,767 prompt tokens plus decode, all 1,024 pages and all 224 layer/rank K/V tensors checked, final device position 32,768.
- Mixed/non-aligned prompts: 33/47 and 2049/2079 pass with fixed slots and inactive rows; batch 32 passes exact slot/page permutations and page remapping across all 28 layers.
- Autoregressive: 100-token HF and TT base-model completions are coherent English continuations with no corruption or feedback regression. The exact tokenizer has no chat template, so completion prompts are correct.

Evidence is in [results/final](results/final), [results/autoregressive](results/autoregressive), and [qualitative_verdict.md](qualitative_verdict.md). `doc/context_contract.json` continues to advertise 32,768 tokens at batch 1. vLLM integration is out of scope.

## Reproduction

Run hardware commands serially. Principal commands:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_evidence.py --model-dir models/autoports/tiiuae_falcon3_7b_base --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt --output models/autoports/tiiuae_falcon3_7b_base/doc/optimized_full_model/results/final/full_model_evidence_dynamic_rope.json --weight-cache-path /tmp/falcon3-full-model-cache

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_depth_sweep.py --model-dir /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt --output models/autoports/tiiuae_falcon3_7b_base/doc/optimized_full_model/results/final/depth_sweep_dynamic_rope.json --depths 1,14,28 --iterations 64 --max-context-len 32768 --weight-cache-path /tmp/falcon3-full-model-cache

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python models/autoports/tiiuae_falcon3_7b_base/tests/full_context_coverage.py --model-dir models/autoports/tiiuae_falcon3_7b_base --output models/autoports/tiiuae_falcon3_7b_base/doc/optimized_full_model/results/final/full_context_coverage.json --weight-cache-path /tmp/falcon3-full-model-cache
```

The command ledger, rejected probes, profiler provenance, and artifact list are in [work_log.md](work_log.md).
