# DeepSeek-V4-Flash: ttnn prefill for disaggregated serving

This package is the prefill half of DeepSeek-V4-Flash disaggregated serving on Tenstorrent Blackhole. The decode half is the
tt-blaze decode ring. Every measurement and defect behind this code is in the tt-blaze postmortem
(`docs/deepseek_v4_flash_postmortem.md`), cited here by entry id (`DS4F-NNNN`).

## What it is

- **Where:** one Blackhole galaxy, 32 chips as an SP 8 x TP 4 mesh (host 41 on the DC16-2 pod), one runner rank.
- **What it computes:** all 43 layers over the prompt. Layers 0-1 are sliding-window (SWA), even layers 2-42 compressed
  sparse with the lightning indexer (CSA), odd layers 3-41 heavily compressed (HCA). mHC hyper-connections carry 4 residual
  streams. The 256-expert MoE is hash-routed on layers 0-2. The schedule is derived in `tt/v4/layer_kinds.py`.
- **Chunks:** the engine feeds `PREFILL_CHUNK_SIZE` tokens per chunk. Each layer keeps per-slot state across chunks (window
  carry, compressor priors, entry counts). Only the last chunk of a request may be ragged.
- **Trace islands:** with `PREFILL_USE_TRACE=1`, the position-independent parts of every layer (mHC sites, norms, MoE) are
  captured once and replayed per chunk. With path A3 the attention also runs as two islands per slot, with eager glue for
  the position-dependent writes (`tt/v4/trace_island.py`, `TtV4PrefillBlock.enable_trace_islands`; DS4F-0246/0247/0300).
- **KV export:** every chunk mirrors its KV into engine-owned caches in the decode ring's layout (`tt/v4/kv_contract.py`:
  4 migrated configs, 2 pending). `tt/v4/kv_table.py` builds the `KvChunkAddressTable`; the migration driver copies the
  rows into the ring through the tt-llm-engine migration endpoints (DS4F-0263).
- **First token:** the last layer runs in full. The runner publishes the final-norm row and the migration driver applies
  the LM head on the host; the ring decodes from the prompt's end with that token. The lane means of layers 40-42 can be
  exported for the ring's DSpark drafter too (DS4F-0273).

## Code map (files this branch adds or changes)

Paths are under `models/demos/deepseek_v3_d_p/` unless they start with `common/` (= `models/demos/common/prefill/`),
`ttnn/` or `tt_metal/`. `git diff --name-only 20778bdbcda HEAD` lists every file.

| File | Role |
|---|---|
| `tt/runners/adapters/deepseek_v4_flash.py`, `common/adapter.py` | the `PrefillModelAdapter` for `PREFILL_MODEL=deepseek_v4_flash`, and its registration |
| `tt/runners/manifests/deepseek_v4_flash.json` | runner env manifest (`PREFILL_MANIFEST`): model, 43 layers, full last layer, gate mode |
| `tt/v4/runtime.py` | `TtV4PrefillRuntime`: compile (warms every chunk index), `capture_trace`, `prefill_chunk`, layer acks, KV table, tail row |
| `tt/v4/transformer.py`, `tt/v4/pp_pack.py` | embedding -> 4 streams -> this rank's blocks -> `TtHyperHead` -> final norm; stream packing across a PP boundary |
| `tt/v4/block.py` | `TtV4PrefillBlock` (production) and its islands; `TtV4Block` at the bottom is main's |
| `tt/v4/trace_island.py`, `utils/sub_device_trace.py` | segmented capture / replay of one island; trace release under its sub-device manager (DS4F-0258) |
| `tt/v4/hyper_connection.py`, `tt/v4/mhc_math.py` | mHC in ttnn, and the same op sequence in torch (the host algebra lock) |
| `tt/v4/attention/swa.py`, `csa.py`, `csa_math.py` | `TtSWA` (`TtHCA` without a compressor); `TtCSA`: two-series compressor, indexer, paths B / A / A2 / A3, capacity buckets; torch algebra lock |
| `tt/mla/heavily_compressed_attention.py` | main's `TtHCA`, extended: unified-cache export, the A3 protocol, shared scalars |
| `tt/v4/moe.py`, `tt/moe/tt_moe.py`, `tt_moe_gate_prefill.py`, `tt_shared_expert.py` | V4 MoE over `TtMoe`: sqrtsoftplus, hash layers, activation override, SwiGLU clamp, gate-bias centring |
| `tt/v4/layer_kinds.py`, `reference/deepseek_v4_flash_config.py` | layer schedule and per-kind KV geometry; the hand-built `DeepseekV4Config` with Flash's schedule |
| `tt/v4/kv_contract.py` | the prefill <-> decode KV contract as plain data; tt-blaze `blaze/models/deepseek_v4_flash/kv_migration.py` asserts against it |
| `tt/v4/kv_table.py`, `tt/v4/kv_cache.py`, `tt/runners/kv_caches.py` | chunk-table walk and build; cache allocation; `V4FlashKvCaches` (one SP-replicated tensor per group) |
| `tt/v4/golden.py` | `v4_groups_v1` golden KV traces from the torch reference (read by the producer's PCC) |
| `tt/v4/weights/` | checkpoint dequant (FP4 experts, FP8 blocks), layer taxonomy, native -> reference names |
| `common/runners/prefill_runner.py` | `PREFILL_KV_ONLY_LAST_LAYER`, first-token / drafter-window publish, `flush_acks`, `on_services_ready` hook |
| `common/runners/prefill_producer.py` | the V4 multi-config KV read-back + PCC; per-request sidecars |
| `common/runners/migration_driver.py` | host LM head, row-range mapping, multi-round sequences, streaming migrate, source-row hashing |
| `common/runners/producer_manifests/producer_manifest_deepseek_v4_flash_*.yaml` | driver / producer manifests, below |
| `ttnn/.../transformer/sdpa/` (sparse_sdpa) | attention sink in `sparse_sdpa`, used by CSA path A (DS4F-0255) |
| `ttnn/.../unified_routed_expert_ffn/`, `ttnn/.../swiglu_sfpu.h`, `tt_metal/hw/.../ckernel_sfpu_softplus.h` | `SiluClamped` routed experts (DS4F-0251); bf16 softplus returned 0 for t < -5 |
| `tt_metal/fabric/`, `pipeline_builder.hpp`, `sd_mesh_command_queue.cpp` | decode-side support carried on this branch (per the commit messages): multi-mesh link assignment, galaxy corner pinning, DSpark drafter coexistence |

## How to run

The measured runs use the tt-blaze launchers, which render the full runner environment: tt-blaze branch
`sdawle/dsv4-flash-prefill-ttnn`, `tests/blaze/models/deepseek_v4_flash/launchers/dc16_2_quad29/disagg/README.md`
(`launch_e2e.sh` end to end, `prefill_bench.sh` prefill only, env templates `runner_env_e2e.sh.in`, `runner_env_prefill.sh.in`).

1. **Runner** (prefill galaxy): `python -m models.demos.common.prefill.runners.prefill_runner` with `PREFILL_MODEL=deepseek_v4_flash`
   (or `PREFILL_MANIFEST=models/demos/deepseek_v3_d_p/tt/runners/manifests/deepseek_v4_flash.json`). It compiles, captures
   the islands, publishes the KV table and device map, and with `PREFILL_ENABLE_MIGRATION=1` waits for its migration endpoint.
2. **Producer** (prefill only): `python -m models.demos.common.prefill.runners.prefill_producer` pushes the token ids of a
   trace dir (`PREFILL_PRODUCER_SLOT_TRACES`) over H2D and drains the layer acks. `PREFILL_PRODUCER_CHECK_PCC=1` reads the KV
   back over UMD and compares it with a `v4_groups_v1` golden.
3. **Migration driver** (end to end): `python -m models.demos.common.prefill.runners.migration_driver --manifest
   models/demos/common/prefill/runners/producer_manifests/producer_manifest_deepseek_v4_flash_e2e.yaml --verify-migration off`.
   It prefills, migrates slot 0 to the ring's endpoint 2, and writes the handoff and DONE sentinel the ring waits for
   (tt-blaze `e2e_pd.sh` runs it this way). The other V4 manifests are transport gates with 4 real layers: `_loopback`
   (loopback endpoint, source PCC, `--verify-migration both`), `_xep` and `_xfer` (to a runner on another host).

**Knobs a user sets.** Defaults are the code's; "measured runs" means the tt-blaze env templates above.

| Knob | Default | Effect |
|---|---|---|
| `PREFILL_CHUNK_SIZE` | 5120 | tokens per chunk; a multiple of 256 (32 x SP). Also sizes the CSA write headroom. 1M fits only at 4096 (DS4F-0300) |
| `PREFILL_MAX_SEQ_LEN`, `PREFILL_NUM_USERS` | 11 x chunk, 2 | per-slot capacity, a multiple of the chunk; slots (measured runs: 1) |
| `PREFILL_USE_TRACE`, `PREFILL_TRACE_REGION_SIZE` | 0, 256 MiB | 1 = capture the islands after compile (measured runs: 1, 1 GiB) |
| `PREFILL_KV_ONLY_LAST_LAYER` | 1 (the JSON manifest sets 0) | 0 runs layer 42 in full plus the head's final norm, needed for the first token |
| `PREFILL_FIRST_TOKEN_DIR` | unset | the last rank writes `slot<s>_end<n>.pt` (final-norm row) per chunk for the driver's host LM head |
| `PREFILL_DRAFTER_WINDOW`, `PREFILL_DRAFTER_LAYERS` | 0, `40,41,42` | rows of lane means exported for the ring's DSpark drafter (measured runs: 128) |
| `PREFILL_KV_TABLE_PENDING` | 1 | 0 drops the 2 pending configs; the blaze ring registers 4, so e2e needs 0 (DS4F-0267) |
| `PREFILL_CSA_SPARSE` | 0 (path B) | 1 = path A: `sparse_sdpa` over [window, top-k] once the context holds 2048 tokens; chunk 0 stays on B. Needed by A2 / A3, buckets, compact cache (measured runs: 1) |
| `PREFILL_CSA_FUSED_INDEXER` | 1 | path A2 fused indexer scorer |
| `PREFILL_CSA_ISLANDS`, `PREFILL_HCA_ISLANDS`, `PREFILL_SWA_ISLANDS` | 1 | path A3 per kind; needs `PREFILL_USE_TRACE=1`, CSA also path A + A2 |
| `PREFILL_TRACED_CHUNK0`, `PREFILL_TRACED_CHUNK0_RAGGED`, `PREFILL_TRACED_MIN_LEN` | 1, 1, 1 | chunk 0's attention on its own A2 island; a ragged first chunk on islands A + B; shortest ragged chunk on the islands (DS4F-0300) |
| `PREFILL_TRACED_RAGGED` | 0 | 1 = a ragged final chunk rides the full-chunk islands (DS4F-0268; measured runs: 1) |
| `PREFILL_CSA_CAP_BUCKETS`, `PREFILL_CSA_BUCKET_STEP` | 1, 32768 | extra A2 islands on a ladder of scorer widths; a chunk replays the smallest that holds its live entries (DS4F-0300) |
| `PREFILL_CSA_COMPACT_DENSE` | 0 | 1 = the path-B cache holds only chunk 0's width; needed for runners >= 256k (DS4F-0295). Needs path A, chunk >= 2048 |
| `PREFILL_MHC_STACKED` | 0 | 1 = the 4 streams batched in one tensor inside mHC, fewer programs (measured runs: 1) |
| `PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE`, `PREFILL_SKIP_CACHED_EXPERTS` | pod NFS paths, 1 | checkpoint; cache at `<cache>/deepseek_v4_flash_bh_<N>dev/<sp>x<tp>`; skip the checkpoint read of fully cached experts |
| `DSV4_FLASH_CACHE_BF4` | unset | 1 = SWA / HCA caches in bfp4; must match the ring (the same variable) |
| `V41_PREFILL_GATE_BIAS_CENTER` | `mid` for sqrtsoftplus gates | centres the gate bias before the bf16 cast (DS41F-0040); `0` = raw. New with the main merge (`fc098e3c95a` did not centre); UNMEASURED on V4 |
| `PREFILL_MIGRATION_ROW_BASE`, `PREFILL_MIGRATION_ROW_DIVISOR` | unset (rows = tokens) | driver: migrate `base + ceil(S / div)` rows; V4 needs 128 / 4 (DS4F-0270) |

- Others: `PREFILL_CSA_LIVE_QUANTUM` (4096, live-width rounding), `PREFILL_MOE_TRACED_PADDING` (1),
  `PREFILL_LAYER_ACK_MODE` (`chunk`, `layer` or `lag1`), `PREFILL_MIGRATION_CONNECT_TIMEOUT_MS` (300000; measured runs 3600000).
- Measurement and A/B only: `PREFILL_ISLAND_TIMING`, `PREFILL_A1_OPTIME`, `PREFILL_HIDDEN_PROBE_STARTS`,
  `PREFILL_HIDDEN_PROBE_DETAIL`, `PREFILL_DRAM_ADDR_DUMP`, `PREFILL_CHUNK_LOG_EVERY`, `PREFILL_SHARED_SCALARS`,
  `PREFILL_RAGGED_FULL_CUT`, `PREFILL_WARM_TAIL_ROWS`, `PREFILL_MIGRATION_SRC_HASH`, `PREFILL_MIGRATION_SRC_HASH_ALL`,
  `PREFILL_MIGRATION_SRC_HASH_OUT`.
- The common runner files also carry V4.1-only paths that stay off for V4: `PREFILL_MIGRATION_STREAM*`,
  `PREFILL_REQUEST_SIDECAR_DIR`.

## Tests

**Host-only, `tests/v4/`** (`test_block.py`, `test_block_chunked.py`, `golden.py` there are main's). No device;
`test_v4_weights.py` reads ~200 MB of the checkpoint and skips without it. The package conftest (`tests/conftest.py`)
imports ttnn, and two tests import modules that do (`tt/v4/runtime.py`, `tt/v4/moe.py`): on a pod host where importing
ttnn opens the chips, run them with the chips free.

| Test | Checks |
|---|---|
| `test_v4_flash_config.py` | the hand-built config, the layer schedule, the KV contract and geometry, the adapter registry |
| `test_v4_kv_table.py` | the linear chunk-table walk against a host model of the ND-shard placement; the populate loop |
| `test_v4_golden_schema.py` | the golden generator's schema and shapes against the reference modules (small random model) |
| `test_mhc_host.py`, `test_csa_math_host.py` | the mHC and CSA compressor / indexer decompositions against the reference, fp32 |
| `test_v4_pp_pack.py`, `test_v4_weights.py` | the stream pack round trip; the weight loader against the real checkpoint's names and shapes |

**Device, `tests/pcc/` and `tests/perf/`.** Most run on a 2x4 mesh (8 chips; see the docstrings for the `TT_VISIBLE_DEVICES` / MGD
setup), some also on the full 8x4.

| Test | Checks |
|---|---|
| `test_ttnn_mhc.py`, `test_ttnn_v4_moe.py` | mHC (PCC >= 0.999) and the V4 MoE (learned and hash-routed layers) vs the reference |
| `test_ttnn_swa.py`, `test_ttnn_csa.py` (our cases) | `tt/v4/attention` SWA / CSA single-shot and chunked vs the reference; path A vs B; the DS4F-0262 traced-vs-eager pinpoint |
| `test_ttnn_hca_runtime_cache.py` | HCA / SWA entries and window ring in the unified caches vs the reference |
| `test_v4_block.py`, `test_v4_transformer.py` | one block per kind; two chunks through the transformer; PCC >= 0.99 |
| `test_v4_kv_table_device.py` | every config read back over UMD through the KV table vs the device tensors |
| `test_v4_decode_kv_parity.py` | the exported KV vs the decode ring's seed for a real layer (needs a tt-blaze probe dump, `V4_KV_SEED`) |
| `perf/test_v4_block_perf.py`, `perf/test_v4_block_trace_probe.py` | host issue / device time per layer kind; one chunk traced vs eager |
| `tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa.py` | `sparse_sdpa` with the attention sink |

## Performance

Not copied here. The measured TTFT tables (disaggregated prefill vs the ring's own prefill-by-decode, and prefill only) are
in tt-blaze branch `sdawle/dsv4-flash-prefill-ttnn`: `docs/deepseek_v4_flash/README.md` (Table 4) and
`tests/blaze/models/deepseek_v4_flash/launchers/dc16_2_quad29/disagg/README.md` (sections 1.1, 1.2). Each number cites its
postmortem entry (DS4F-0300 for TTFT). They were measured on the pre-merge tt-metal `fc098e3c95a`.

## Branches

| Repo | Branch | Role |
|---|---|---|
| tt-metal | `sdawle/dsv4-flash-prefill-ttnn` | this prefill (pre-merge head `fc098e3c95a`) |
| tt-blaze | `sdawle/dsv4-flash-prefill-ttnn` | the decode ring, the decode side of the migration, the launchers |

- Main merge: tt-metal main `20778bdbcda` (tt-blaze main's tt-metal pin), merge commit `b1860c1403a`, built from the V4.1
  merge minus the V4.1-only delta (DS4F-0310). The merged pair has NOT run on a device yet; the tt-blaze tables are pre-merge.
- DeepSeek-V4.1-Flash builds on this package: tt-metal `sdawle/dsv41-prefill-ttnn`, `tt/v41/README.md`.
