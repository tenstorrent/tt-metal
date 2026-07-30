# Optimized decoder work log

## Baseline and topology

- Fused-decoder checkpoint: `d5986fed723`.
- Audited packed QKV, packed gate/up, fused elementwise, RoPE/cache, SDPA,
  residual layouts, and phase-specific matmul opportunities before tuning.
- Hardware: one Blackhole p300c from a healthy four-device host; all hardware
  commands were serialized.

## Commands

Correctness and real weights:

```sh
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
```

Final paired warmed performance:

```sh
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
```

Mandatory shard advisor (separate activated shell):

```sh
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
cd "$TTMLIR_ADVISOR_HOME"
source tools/ttnn-jit/integrations/agentic-research/shard-advise/scripts/bootstrap.sh
export PYTHONPATH=/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages:/home/mvasiljevic/tt-metal:$PYTHONPATH
ttnn-advise capture /home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/tests/advise_optimized_decoder.py:decode --out /tmp/phi35-final-advice
```

Final watcher run:

```sh
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher \
  pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
```

Profiler and reports:

```sh
OPTIMIZED_PROFILE_ONLY=1 OPTIMIZED_PROFILE_ITERATIONS=3 \
  python -m tracy -r -p -v -m pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
tt-perf-report doc/optimized_decoder/tracy/ops_perf_results.csv \
  --start-signpost OPTIMIZED_DECODE_B1 \
  --end-signpost OPTIMIZED_DECODE_B1_END --no-color --no-summary
```

## Decisions and evidence

- Applied advisor DS recommendations to all four dense projections. Rejected
  its untimed configuration as a verdict and swept geometry at both B1/B32.
- Selected BFP4/LoFi after independent attention, MLP, down, real-weight, and
  HiFi2 comparisons. PCC and latency are summarized in README; raw logs are
  retained.
- Precision-locked BFP4/LoFi geometry selected QKV 12, output 12, gate/up 6,
  and down 32 at 0.5613/0.7430 ms (B1/B32 candidate means). All-width 2/4
  lost; gate/up 12 exceeded L1 (1,618,688 requested versus 1,572,864 bytes).
- Retained packed projections and fused SiLU-multiply. No measured path has
  torch/from_torch/to_torch, tilize/untilize, or host fallback.
- Adapted and retried BFP8 KV cache after the first dtype error. Cache-consuming
  PCC passed, but staging regressed prefill and violated partial-user
  ownership; the optional branch was removed.
- Split gate/up passed 12 tests. Packed was retained because it clearly wins
  prefill and the sub-percent split decode delta was inconsistent run noise.

## Optimize checklist

- [x] operation-topology audit and packed/fused candidates
- [x] mandatory shard-advisor capture with final IR and nonzero DS consideration
- [x] precision/fidelity and role-specific geometry sweep at B1/B32
- [x] DRAM-sharded decode matmuls and phase-specific prefill path
- [x] explicit SDPA, memory, program, and compute-kernel configuration
- [x] non-aligned prefill, paged cache, representative real layer, determinism
- [x] warmed prefill and traced decode before/after at B1/B32
- [x] repeated trace replay and watcher-clean optimized correctness
- [x] context contract preserved
- [x] stage-owned docs and artifacts
- [x] independent stage rereview: clean-pass
- [ ] local stage commit (SHA recorded after commit)

Limitations: this is a single-device decoder-layer stage, not multichip,
full-model, generator, or vLLM work. BF16 KV cache remains intentionally
unchanged.
