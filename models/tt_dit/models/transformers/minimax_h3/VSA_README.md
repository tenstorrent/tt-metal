# Video Sparse Attention (VSA) for MiniMax-H3 on the 4x8 Blackhole Galaxy

Checkpoint 2026-09-09 (branch `cglagovich/fast_h3_vsa`). VSA replaces the dense ring attention in the
MiniMax-H3 video transformer with a two-stage sparse attention: a coarse stage scores 64-token tiles
against pooled keys and keeps the top-k per query tile, and a fine stage (`ttnn.transformer.vsa_sdpa`)
attends only over the selected tiles. Numerics are exact to the bf16 floor of the dense kernel, results
are deterministic, and the 15 s / 768p generation runs 1.69x faster end to end.

## Where things stand

| 15 s / 768p, 4x8 Galaxy (TP=4, SP=8), sparsity 0.9 | dense | VSA |
|---|---|---|
| end-to-end t2va, 50 steps, warm, real weights | 325.9 s | 181.4 s (1.80x; 193.2 at the 09-09 checkpoint) |
| of which denoise | 294.4 s (6.01 s/step) | 151.3 s (3.03 s/step; 160.7 s at the checkpoint) |
| one transformer block period, isolated (Tracy device wall) | 78.4 ms | 59.2 ms (64.3 at the 09-09 checkpoint) |
| attention op inside the block | 51.4 ms (ring SDPA) | 16.7 ms (`vsa_sdpa`; 21.5 at the checkpoint) |
| attention oracle vs the reference implementation (4 gate/placement configs) | | PCC 99.51-99.58 % |
| `vsa_sdpa` vs fp32 reference, 1024-block rows | | PCC 0.99969, every row at the bf16 floor |
| repeated launches / trace replay | bit-exact | bit-exact |

The isolated block gap (15 %) understates the end-to-end gap (1.8x on the denoise) because the dense
block power-throttles under sustained load (median AICLK 975 MHz vs 1268 MHz for VSA); details in
`VSA_STREAM_DESIGN.md` section 8. Always compare dense vs VSA end to end. The kernel lever pass of
2026-09-09 (section 10, running log in `VSA_LEVERS_LOG.md`) took `vsa_sdpa` from 23.0 to 19.6 ms on the
real device-5 shard standalone and the denoise step from 3.28 to 3.03 s (measured 2026-09-10 with
`tests/.../test_vsa_e2e_perf_minimax_h3.py`).

## Turning it on

Pass a `MiniMaxH3VSAConfig` to the model or pipeline; with `vsa_config=None` the dense path is used and
the dense block is bit-identical to before this branch.

```python
from models.tt_dit.models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig
pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh, weights_dir=weights,
                                             vsa_config=MiniMaxH3VSAConfig(sparsity=0.9))
```

`MiniMaxH3VSAConfig` fields: `sparsity` (fraction of tiles dropped; 0.0 reproduces dense attention),
`placement` (tile-to-shard placement, default `interleaved`), `streaming` (the streaming leader/worker
kernel, default) vs the v1 per-row gather kernel, `padded_pooling` (per-shard padded coarse gathers,
default on), `distributed` (opt-in experimental window kernel, slower), `stream_order`
(default `identity`; `bstride4.16` see `VSA_STREAM_DESIGN.md` section 11). The block/attention/pipeline tests read `VSA_PLACEMENT`, `VSA_KERNEL=v1`,
`VSA_PADDED_POOLING`, `VSA_DIST` for the same knobs.

## Code map

| | |
|---|---|
| `vsa_geometry.py` (`models/tt_dit/pipelines/minimax_h3/`) | host geometry: tile ids, pad tiles, shard placement, averaging matrix |
| `vsa_stages_minimax_h3.py` | coarse stage (pooling, scores, top-k, index assembly) and `MiniMaxH3VSAConfig` |
| `attention_minimax_h3.py` | the fourth attention path: K/V all-gather, coarse stage, `vsa_sdpa`, unpack |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/vsa_sdpa*` | the `vsa_sdpa` op (host side, program factories) |
| `.../sdpa/device/kernels/{compute,dataflow}/vsa_sdpa_stream_*` | streaming kernel: leader reader, worker writer, compute |
| `.../sdpa/device/kernels/dataflow/vsa_sum_service.hpp` | exact row sums on the writer core |
| `.../sdpa/device/kernels/*/vsa_sdpa_dist_*` | distributed-window variant (opt-in) |
| `vsa_reference/` | vendored upstream reference (FastVideo VSA) used by the oracle tests |
| `VSA_SCOPE.md`, `VSA_PLAN.md`, `VSA_STREAM_DESIGN.md` | requirements, implementation journal, kernel design notes |

## Tests

Single device (`scripts/run_safe_pytest.sh <file>`; the wrapper adds a dispatch timeout and post-mortem):

- `tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa.py`: kernel vs torch, raw-selection path bit-exact
  vs host-assembled indices, stream-order (experimental), both kernels and the distributed variant.
- `test_vsa_sdpa_precision.py`: per-row numerics at 1024 listed blocks (PCC, relative error, gain).
- `test_vsa_sdpa_determinism.py`: bit-exact repeated launches.
- `test_vsa_sdpa_trace.py`: trace replay, program-cache hits, L1-fill robustness.

Host only: `models/tt_dit/tests/models/minimax_h3/test_vsa_geometry_minimax_h3.py`,
`test_vsa_oracle_minimax_h3.py` (geometry and reference-implementation agreement).

Galaxy (`SAFE=1 models/tt_dit/models/transformers/minimax_h3/scripts/run_h3_test.sh <file>`):

- `test_vsa_stages_minimax_h3.py`: coarse stage vs oracle; coarse+fine vs full oracle.
- `test_vsa_attention_minimax_h3.py`: sparsity-0 equals the ring path; attention vs torch oracle;
  production shapes.
- `test_vsa_transformer_minimax_h3.py`: full transformer, sparsity 0 vs dense; placements agree.
- `test_vsa_block_minimax_h3.py`: 15 s / 768p block, untraced and traced (L1 fit, replay bit-exact).
- `test_vsa_pipeline_minimax_h3.py`: real-weights t2va generation (needs `MINIMAX_H3_MODEL_PATH`).

## Performance tooling

- Block profiles: `MODES="dense vsa" DURS=15 scripts/profile_block.sh` (Tracy; one warm block).
- Kernel alone: `test_vsa_sdpa_perf.py` (synthetic patterns), `test_vsa_sdpa_real_perf.py` with
  `VSA_REAL_DUMP=<file>` from a model run under `VSA_DUMP_INDICES=1` (real selections).
- Kernel knobs and probes (`TT_VSA_RMAX`, `TT_VSA_DEPTH`, `TT_VSA_PROBE`, `TT_VSA_LAZY_T`, `VSA_NO_SUMS`,
  `TT_VSA_OS`): `VSA_STREAM_DESIGN.md` section 6.
- The Tracy device profiler keeps roughly 1000 programs per run, so a full pipeline cannot be
  op-profiled; instrument with synced host timers instead (section 8).

## Known limits and next steps

- Power: under sustained load the galaxy throttles; VSA runs at ~94 % of nominal clock, dense at ~72 %.
- `vsa_sdpa` is 21 ms of the 64 ms block; the two K/V all-gathers before it (8 ms) run alone and could
  overlap the coarse stage (largest remaining block-level lever). The coarse stage's nine transposes
  (3 ms) can be folded into the pooling layout. The remaining 25 ms are dense-identical CCL matmuls.
- The distributed-window kernel is experimental and slower (behind a flag). The KV stream order default is
  `identity`; `bstride4.16` (interleaved spatial segments) is 9.5 % faster standalone but hangs the full
  pipeline until the leader/worker progress race is fixed (design doc section 11).
- fp32 DEST accumulation was measured and rejected (22-43 % slower on the dense kernel, does not fit L1).
