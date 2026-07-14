# AutoDebug: ring full-attention prefill L1 collision

## Scope and verdict

This was an inspection-only analysis of the TP4 ring candidate. No source was
changed and no hardware was used.

The failure is a deterministic L1-capacity collision between persistent decode
all-reduce scratch that was materialized by earlier traced-decode tests and the
static circular buffers of the later full-attention prefill SDPA. It is not an
attention-correctness failure or a failed ring-fabric bring-up. The smallest
safe first adaptation is to reduce the ring/full-attention prefill K chunk from
128 to 64 while retaining Q=128. Do not deallocate the shared persistent CCL
pool before prefill: captured decode traces require those buffer addresses to
remain stable.

## Evidence-ranked findings

### 1. Confirmed: resident persistent decode scratch collides with full-prefill SDPA by 2,112 bytes

Confidence: high.

- Ring fabric initializes on all four devices before the tests
  (`ring_topology_final_graph.log:46-52`). Both traced-decode PCC cases then
  pass (`:57-60`), and the ring sliding-layer warmed benchmark also passes
  (`:61`). This rules out a general ring-fabric, TP4, persistent-all-reduce, or
  trace-replay failure.
- The later full-attention warmed test fails on its first prefill call, before
  its own decode capture (`test_multichip_decoder.py:1353` versus decode at
  `:1382-1387`). The validator reports a globally allocated L1 buffer starting
  at 1,241,792 and an SDPA static-CB region ending at 1,243,904 on the SDPA
  8x4 core range, an overlap of exactly 2,112 bytes
  (`ring_topology_final_graph.log:63,105-123,143-145`).
- The persistent pool is module-global (`multichip_decoder.py:63`), shared by
  mesh identity, and holds its buffer dictionary for the module-scoped mesh
  lifetime (`:556-570`; the fixture is
  `test_multichip_decoder.py:325-335`). Therefore scratch created by an earlier
  parameterized test remains live in the later test.
- `_tp_allreduce` keys stable scratch by role, ping-pong slot, shape, dtype, and
  shard grid (`multichip_decoder.py:138-155`). On a miss it allocates an L1
  WIDTH_SHARDED intermediate with a per-core width four times the reduced
  output shard and stores it in the pool (`:156-186`). It deallocates only the
  transient communicated and reduced tensors (`:187-201`), not the stable
  scratch.
- Both decode O and decode MLP reductions select the persistent path
  (`multichip_decoder.py:908-919` and `:301-314`). Prefill deliberately selects
  the ordinary nonpersistent all-reduce (`:1030-1038`), so the retained scratch
  is idle during the failing SDPA but still consumes L1.
- The test ordering is an essential part of the repro. The module-scoped mesh
  survives the earlier traced-decode tests, and each warmed test itself runs
  prefill before decode (`test_multichip_decoder.py:1333-1413`). Running only
  the final full-attention warmed node on a fresh process would omit the
  resident-scratch condition and would not close this regression.

### 2. Confirmed: K=64 removes the larger, decisive static-CB family

Confidence: high for the code-level capacity reduction; hardware is still
required to prove the final fit and latency.

- The failing lowered config is full attention with head_dim=512, seq=128,
  grid=8x4, Q chunk=128, and K chunk=128
  (`ring_topology_final_graph.log:120-123`).
- `prefill_sdpa_program_config` chooses exactly that configuration for
  head_dim>=512 and already exposes `GEMMA4_PREFILL_SDPA_QCHUNK` and
  `GEMMA4_PREFILL_SDPA_KCHUNK` as sweep overrides
  (`models/demos/gemma4/tt/attention/operations.py:176-206`).
- In the lowered SDPA factory, K and V are both double-buffered:
  `k_tiles = Sk_chunk_t * DHt * 2` and
  `v_tiles = Sk_chunk_t * vDHt * 2`; the QK intermediate also scales with
  `Sk_chunk_t`
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:247,379-390`).
  At head_dim=512, K=128 to K=64 removes 64 K BF16 tiles, 64 V BF16 tiles,
  and 8 QK intermediate tiles per core--far more than the 2,112-byte deficit.
- Q=64 is also likely to fit, but it is not the cleaner first choice. With
  seq=128 it changes `q_num_chunks` from one to two, which raises
  `q_buffer_factor` from one to two under the factory's global-Q scheduling;
  consequently the Q input CB remains 64 tiles rather than shrinking
  (`sdpa_program_factory.cpp:247-251,331-353,379`). Q=64 still reduces QK,
  output-intermediate, output, and statistics CBs, but K=64 provides the larger
  directly-accounted capacity reduction while preserving one-Q-chunk
  scheduling.

### 3. Recommended intervention boundary

Confidence: medium-high, pending the focused hardware run.

First prove the existing no-source override with Q=128/K=64 under the exact
four-node ordering below. If ring is retained as a production candidate, make
the same choice local to `MultichipDecoder` only when its CCL topology is Ring
and `head_dim >= 512`; keep the shared Gemma helper's Q=128 default and the
MultichipDecoder Linear default unchanged. This preserves:

- the TP4 mesh and replicated BF16 residual contract;
- the shared, stable persistent async-CCL scratch and semaphore pool;
- the accepted decode graph and traced warmed decode path;
- the existing Linear topology behavior; and
- mathematical SDPA semantics (subject to the required PCC rerun because a
  different reduction blocking can still change low-order numerical results).

Do not clear or deallocate `pool["buffers"]` at a prefill boundary. A trace
captures device buffer addresses, so reclamation/reallocation can leave an
existing trace referring to stale storage. Test-only pool clearing would also
hide the full-model coexistence requirement rather than satisfy it.

If K=64 fits but causes a material prefill regression, the next coherent
adaptation is Q=64/K=128 with measured PCC/latency; if either single-axis
adaptation remains too tight, Q=64/K=64 is the safer exact pair. Moving the MLP
decode grid and its scratch away from the SDPA grid is a substantially more
invasive fallback: it changes matmul/NoC placement and needs full decode
performance evidence.

## Focused verification

Run the exact ordering that first populates persistent scratch and then reaches
the full-attention prefill:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} \
PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools \
GEMMA4_MC_TOPOLOGY=ring \
GEMMA4_MULTICHIP_BENCH=1 \
GEMMA4_PREFILL_SDPA_QCHUNK=128 \
GEMMA4_PREFILL_SDPA_KCHUNK=64 \
timeout 600 pytest -q -s -x \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k 'paged_decode_trace_matches_optimized_baseline or warmed_latency'
```

Acceptance requires all four selected tests to pass, both layer-kind PCCs to
remain at the accepted baseline, and both warmed prefill/traced-decode records
to be captured. Compare the resulting full-attention prefill median against the
ring Q=128 fresh-pool measurement if available; do not use a fresh-pool-only
pass as evidence that the coexistence bug is fixed.

## Remaining uncertainty and test gap

Static inspection proves that K=64 substantially shrinks the colliding CB
family, but only the target Blackhole mesh can prove final allocator placement,
PCC, and latency. The regression suite also lacks a named test that explicitly
materializes both persistent reduction roles, releases a trace, and then runs
full-attention prefill on the same mesh. The four-node command above currently
provides that stress sequence indirectly.
