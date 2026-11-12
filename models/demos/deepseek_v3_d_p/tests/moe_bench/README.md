# Blackhole Galaxy MoE benchmark

Measures the two MoE paths on a Blackhole Galaxy (32 chips) for public MoE shapes (DeepSeek V3, DeepSeek V4
Flash/Pro, Kimi K2.7, Kimi K3, MiniMax M3, GLM-5.3) and places each against the per-chip memory and compute roofline.

| phase | op(s) | setup |
|---|---|---|
| prefill | `TtMoe`: gate, dispatch, routed experts (`deepseek_prefill.unified/hybrid_routed_expert_moe`), combine, shared expert | 8x4 TorusXY, 5k-token chunk (640 tokens/chip), each model's production fabric settings |
| decode | `ttnn.experimental.moe_compute` (compute_only: tilize, W0/W1, activation, W2) | one chip holding E/32 experts; 1, 8, 32 and 128 decode users per chip |

## Run

```bash
cd $TT_METAL_HOME
models/demos/deepseek_v3_d_p/tests/moe_bench/run_bench.sh prefill   # ~1-2 h
models/demos/deepseek_v3_d_p/tests/moe_bench/run_bench.sh decode    # ~1-2 h
```

Each writes profiler CSVs, logs and `roofline.md` / `roofline.json` into its results directory.

## Change the shapes

```bash
MOE_BENCH_MODELS=deepseek_v3,minimax_m3 run_bench.sh decode
MOE_BENCH_CUSTOM="my_moe:H=5120,N=1536,E=160,k=6" MOE_BENCH_MODELS=my_moe run_bench.sh prefill
MOE_BENCH_DECODE_USERS=4,16,64 run_bench.sh decode
```

`moe_models.py` holds the table; a custom row takes `H` (hidden), `N` (expert intermediate), `E` (routed experts),
`k` (experts per token) and optionally `Hr` (latent routed width), `act` (`silu`, `swiglu_oai`, `gelu`), `shared`
(shared experts) and `cf` (dispatch capacity factor). Custom prefill rows use a host gate, so any E and k work.

## Roofline

Per chip: DRAM 512 GB/s; compute = cores x 4096 FLOP/cycle (LoFi) x AICLK (read from the tt-smi snapshot). Expert
weights are BFP4 on both paths. Decode bytes count only the experts the batch actually hits; at 1 user/chip that is k
of the chip's experts. `moe_compute` runs its matmuls on a ring of 7-8 DRAM-adjacent cores, so the decode table also
gives that ring's own compute roof, which binds before DRAM at large batches.

The prefill layer time is the sum of device kernel durations per chip (ops that overlap on sub-devices are counted
in full); the traced CI gates in the prefill run give the wall-clock layer time for DeepSeek V3 and Kimi K2.7/K3.
