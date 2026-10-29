# Sparse checkpoint ownership: installed-source proof

This is a source-only diagnosis of the recorded512-block serving configuration. It neither enables a checkpoint option nor qualifies a new runtime, page layout, cache policy or hit-rate claim.

## Exact runtime inspected

Corrected source-readback run [40920236280](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40920236280) succeeded using wrapper`03c152146eb14fa03307f332a3238f1d50736e31`; serving/device jobs were skipped. Its artifact`qwen38-runtime-overlay-40920236280/readback/installed-cache-sources.zip` reads the actual installed vLLM0.30.0+empty and TT plugin0.1.0 from image`41aa0687bc668186baf9c4ed7942e135b60cf8828d9bbd67418cf53993a3d612`. All8 source-file sizes/SHA256s and ZIP members were checked. Archive SHA256:

`1c327709dc8b1bce978ee9a22e47afb0923d017d1a1e1afea933e70a85ad3504`

The earlier40913533837 readback failed under the system interpreter;40919989533 was canceled during a flag check, without a device job. Source-overlay mode correctly requires`build-qwen38-overlay=true, build-only=false`; the normal serving dispatch skips that combination. The corrected helper uses the serving virtualenv and imports only stdlib modules.

## What the installed code proves

Metal502 advertises zero internal-prefill checkpoint blocks. The adapter subdivides a scheduled prefill into512-token pieces but saves only where the per-layer scheduler tables contain nonzero IDs. Physical splitting therefore does not create checkpoint ownership. In the recovered shared worker log, all32 cold16K requests saved only4096,8192,12288,16384, while the next-turn eligible prefix is15872 and all32 report12288 cached. Missing15872 state is a sufficient barrier even if the attention tail is resident. The available events do not establish each miss's exact cause.

Actual installed-source locations within the archive:

- `vllm/v1/core/sched/scheduler.py:353`: all recurrent groups must advertise the internal-checkpoint option. Lines448–466 govern its final-query behavior;495–525 also stop at shared-prefix junctions.
- `vllm/v1/core/single_type_kv_cache_manager.py:1682`: the manager's eligibility check. Lines1775–1807 reserve the penultimate column;1827–1933 pad null table entries and allocate physical IDs;2046–2073 hash the exported checkpoint.
- `vllm/v1/kv_cache_interface.py:1034`: align-mode memory allowance is2+speculative+prefill-checkpoint blocks. Lines1073–1106 select and validate the last hash boundary before query end.
- `vllm_tt_plugin/worker.py:818`: existing worker reservations already include the checkpoint field.
- `vllm_tt_plugin/model_runner.py:1082`: running requests append the returned logical table delta. Returned null gaps are intentional; append alone is not a proven bug.

For equal512 block/hash granularity without speculation, one checkpoint would let the current adapter write the penultimate state: a cold0→4096 query uses checkpoint3584/column6 and running tail4096/column7. A query ending16384 can expose15872/column30. It does not provide every internal512 boundary. Ragged-end column arithmetic was checked for17,487 end positions.

## Reservation tradeoff

AtB32 with three recurrent groups, total sizing reservation is`32 × (1 + 3 × (2 + C)) + 1`, whereC is the advertised checkpoint count. The1 terms are output and global null reservations. These are worst-case sizing allowances, not permanently occupied IDs or guaranteed conversation tokens.

| Configuration | Actual pool IDs | Reserved sizing IDs | Remaining hook units | Virtual hook-token units |
|---|---:|---:|---:|---:|
| Packed control, C=0 | 1379 | 225 | 1154 | 590,848 |
| Shared, C=0 | 1470 | 225 | 1245 | 637,440 |
| Shared, C=1 | 1470 | 321 | 1149 | 588,288 |

Thus sharedC1 is equivalent to1470−96=1374 IDs under the old sizing policy, five below control1379. Physical allocation remains1470 IDs/25,409,617,920 bytes per device. Keeping the old hook while advertisingC1 would request1566 IDs,96 extra:1,659,371,520 more backing bytes at this geometry. The existing exact-count allocator guard would reject that mismatch. The real upstream admission floor also increases; the recorded native-context point remains above it, but the CPU worker fixture deliberately does not certify framework admission.

## New opt-in blocker: predictor versus sparse allocation

The installed manager has a checkpoint-specific physical-count overestimate on an already-running multi-block query. The CPU repro executes its exact extracted method ASTs, with explicit logical block/spec/pool fixtures and no inference-package imports:

| Checkpoints | Query | Predicted physical IDs | Actual new IDs | Returned table entries | Null entries |
|---|---|---:|---:|---:|---:|
| 0 | 4096→8192 | 1 | 1 | 8 | 7 |
| 1 | 0→4096 | 2 | 2 | 8 | 6 |
| 1 | 4096→8192 | 8 | 2 | 8 | 6 |
| 1 | 12288→15872 | 7 | 2 | 7 | 5 |
| 1 | 15872→16896 | 2 | 2 | 2 | 0 |

`get_num_blocks_to_allocate` bypasses its sparse cap when both checkpoint and already-running are true. `allocate_new_blocks` nevertheless inserts null placeholders and calls the physical pool for only checkpoint+running IDs. This can cause false scarcity if the option is enabled. It **did not cause the recorded checkpoint0 misses**. A spec-only flip is therefore not a complete remedy.

Reproduce using the original artifact, without vLLM/TTNN/Torch imports:

```bash
python3 models/autoports/qwen_qwen3_8_27b/tests/probe_installed_checkpoint_contract.py \
  --source-archive /path/to/readback/installed-cache-sources.zip \
  --output /outside/git/checkpoint-source-proof.json
```

The script pins the archive/image, verifies all source hashes, runs8 manager cases with a checkpoint0 regression, checks the ragged-end index invariant, and executes the installed worker sizing body for four reservation cases. Logical fixtures are not genuine vLLM schema admission, end-to-end caching, state-copy, or device tests. Partial-hit/CoW/speculative behavior is deliberately unqualified.

## Minimal coordinated remedy and remaining gates

1. First align the opt-in manager's physical predictor with actual sparse allocation. Cover fresh/running, multi-block, partial-hit/CoW, speculation, reused checkpoint columns and overallocated rows before accepting a package patch. Do not add a second reservation in the TT worker: it already accounts forC.
2. A model-local default-off0/1 option must drive the MambaSpec, token-hook inversion, single-request admission check and exact allocator-count guard together. Values above1 are unsupported: installed manager behavior is boolean and does not allocate denser checkpoints for larger values.
3. The current512/512 adapter can address the new penultimate column. Larger state pages with a finer hash unit need explicit producer/consumer metadata; the current boundary-only `_snapshot_ids` is insufficient for an arbitrary sub-page checkpoint. Coordinate this contract with Claude's union backing/equal1536 pages. This work does not implement or compete with unpadded prefill or that layout.
4. The readback lacks actual InputBatch/append-row definitions and execution modules that might consume CoW copies. Supplied TT worker/runner files do not reference upstream`kv_cache_block_copies`; do not claim unsupported partial-hit copying works. A narrow runtime integration must verify returned null/ID placement, preemption/resume, exact FP32/BF16 state ownership and asynchronous lifetime.
5. Prefer a jointly owned, eviction-freeA1 continuation proof with the same16K/32K arrays before the unchangedB32 study. Keep trueLCP and a fixed512 eligibility denominator for geometry comparisons. No standalone checkpoint hardware run is authorized by this report.

## Why checkpoint coverage alone is insufficient

The shared run cached1,479,168 of3,112,960 eligible tokens (47.5164%). Even making all currently positive-hit requests perfect, while leaving zero-hit requests zero, yields only1,663,488/3,112,960=53.4375%. Zero-hit cohorts must also recover; this conditional bound does not assign them all to eviction. Separately,1470 global IDs cannot hold2048 attention IDs for32 private32K histories simultaneously, before any recurrent ownership.

Normal workflow artifacts recovered all three worker logs despite failure of the stricter final HTTP+worker exporter. Executed logical prefill was4,619,776/reference,4,580,864/control,4,199,424/shared; repeated work beyond HTTP uncached input was542,208;521,216;368,128. These counts exclude padding and do not substitute for a stable final exporter or complete group/block census. Source/worker evidence remains outside Git under`/home/mvasiljev/build/qwen38-swe/memory-prefix/`; no raw logs, arrays, ZIPs or installed package copies are committed here.
