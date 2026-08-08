# Transformer Optimization Guidelines for Tenstorrent (Wormhole / Blackhole)

Generalized, cross-model best practices for optimizing **transformer-style models** on
Tenstorrent hardware in TT-NN. This is a synthesis of three independent optimization
campaigns, distilled into reusable rules that apply to any encoder/decoder transformer.

## Source campaigns

| Campaign | Model family | Hardware | Regime | Headline result |
|---|---|---|---|---|
| **BGE-M3** | BERT / XLM-R encoder (24 layers, H=1024, 16 heads) | Blackhole P150 | batch 1 & batch 32, seq 512 | B1 5.7→4.30 ms; B32 194.9→60.55 ms (3.22×) |
| **ViT** | Vision Transformer (12 layers, H=768, 12 heads) | Blackhole P150 + Wormhole | batch 10, seq 224; high-res seq 1024–4096 | fully block-sharded L1 pipeline |
| **Swin-L + DyHead** | Windowed-attention backbone + detection head | Wormhole B0 | trace + 2CQ, 640x640 | 5.66->6.45 FPS (+14%) |
| **TT-Transformers / Llama-70B / DeepSeek-V3** | Decoder LLMs (RoPE, GQA, KV-cache, MoE) | Wormhole / T3000 / Galaxy | prefill + decode, multi-device | the canonical generative-LLM patterns |

Four very different transformer families, multiple hardware/batch regimes - the
**common practices** that survived across all of them are what these guidelines capture.

## Scope

These guidelines focus **only on the transformer data path**:

- **[02_NORMALIZATION.md](./02_NORMALIZATION.md)** — LayerNorm, RMSNorm: sharding, fidelity, the precision-compounding bug, residual fusion.
- **[03_QKV_PROJECTION.md](./03_QKV_PROJECTION.md)** — QKV matmul program configs, memory layout, fidelity, head-split strategies.
- **[04_ATTENTION_SDPA.md](./04_ATTENTION_SDPA.md)** — SDPA vs manual attention, chunk sizing, softmax, score dtype, DRAM staging.
- **[05_MLP.md](./05_MLP.md)** — FF1/FF2/FF3 (SwiGLU) matmuls, fused activation, `minimal_matmul`, subblock tuning.
- **[08_DECODE_PREFILL_AND_MULTIDEVICE.md](./08_DECODE_PREFILL_AND_MULTIDEVICE.md)** — generative-LLM specifics: prefill vs decode, matmul-variant-by-regime, KV-cache, GQA, RoPE, multi-device weight fracturing + CCLs.

Supporting cross-cutting material:

- **[01_FOUNDATIONS.md](./01_FOUNDATIONS.md)** — hardware grid, L1/DST budget, memory configs, precision/fidelity knobs. **Read this first.**
- **[06_FUSION_AND_RESIDUALS.md](./06_FUSION_AND_RESIDUALS.md)** — op-count reduction, residual folds, reshard/dtype fusion.
- **[07_METHODOLOGY.md](./07_METHODOLOGY.md)** — how to sweep, the noise floor, single-layer vs full-model PCC, harness bugs.
- **[09_PROFILING_AND_OP_ANALYSIS.md](./09_PROFILING_AND_OP_ANALYSIS.md)** — Tracy capture, tt-perf-report, bucketing the op CSV by total device time and op count, drilling by shape, reading data-movement buckets.

## The five rules that held across all three campaigns

1. **Never hard-code the core grid.** Query `device.compute_with_storage_grid_size()`. WH is 8×8 (64 cores), BH is up to 11×10 / 10×12 (110–120 cores). A hard-coded 8×8 on Blackhole silently discards ~40% of the cores and invalidates every sweep.

2. **Walk math fidelity down per op, gated by full-model PCC.** HiFi4 is almost never needed at bf8b. The path HiFi4 → HiFi2 → LoFi was the single largest device-time lever in two of three campaigns (BGE-M3 B32: −29 ms; Swin-L: the GroupNorm/softmax/matmul LoFi drops). **Exception: normalization reductions need HiFi2 + fp32 dest accumulation** — LoFi there compounds to failure over depth.

3. **`fp32_dest_acc_en=False` unlocks the subblock cap from `h·w ≤ 4` to `h·w ≤ 8`.** This is the highest-leverage matmul knob: doubling subblock area halves pack/unpack round-trips. Try False first on every matmul, re-PCC, then re-sweep subblock with the wider ceiling.

4. **Keep activations in their native dtype through the attention path.** The single largest BGE-M3 batch-32 win (−13.7 ms) was *removing* an unnecessary Q/K/V→bf16 cast before SDPA. Audit your graph for typecasts that no op actually needs.

5. **Match producer/consumer memory layout to avoid reshards.** ViT keeps the whole encoder block-sharded in L1 so consecutive ops never round-trip to DRAM. BGE-M3 chains sharded LN outputs across blocks. Swin-L routes residuals/LN/MLP outputs to L1 where they fit. Every reshard you remove is a real win — but a reshard the *next op does internally anyway* is not (test before wiring).

## How to use these guidelines

1. Read **01_FOUNDATIONS** to internalize the hardware limits and the precision knobs.
2. Diagnose your regime (host-bound vs device-bound; small-batch vs large-batch; does the activation fit L1?). See 01 §"Regime decision".
3. Jump to the component you're optimizing (02–05).
4. Use **07_METHODOLOGY** to run every change through a sweep → PCC → in-model → wall-time loop.

## A note on generalization

Numbers in these files are **illustrative of the technique**, not targets. The right
program config is always shape- and hardware-specific — these guidelines tell you
*which knobs matter, in what order, and what the failure modes are*, so you can find
your own optimum quickly instead of rediscovering the search space.
