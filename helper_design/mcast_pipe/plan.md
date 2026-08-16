# Remaining `mcast_pipe` rollout plan

Date: 2026-08-16
Branch: `sjovic/mcast-migration` at `db246d49b89978d436d944d57f0ba326ef698416`
Baseline: `origin/llk_helper_library` at `dc9282be7d5e9d5a4b9137c1bf327de8d923e18e`
Materialized helper: `MCAST_PIPE_API_VERSION = 11`
Status: **planning only; no migration is authorized by this document**

## Decision requested

Approve the tiering, gates, and execution order below. Implementation starts only after explicit user approval.
Each migration unit stops independently if its source, correctness, coverage, LOC, or performance gate fails.

## Audit result and scope correction

The current ledger has 91 entries: 17 `migrated`, 3 `pending`, and 71 `deferred`. This plan re-read all 74
non-migrated ledger paths and their live factories instead of carrying the old `deferred` verdicts forward.

The exact primitive recall also found two real kernel call sites omitted by the ledger:

- `ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_full_width_sharded.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_partial_width_sharded.cpp`

They implement a two-hub gather/broadcast: both hubs multicast disjoint regions into the same destination CB and
then increment one completion counter. Three unmatched host hits add no new non-migrated kernel call site:
`moe_compute_program_factory.cpp` matched a comment, while `groupnorm_sharded_program_factory.cpp` and
`exp_ring_joint_sdpa_program_factory.cpp` contain live host artifacts already associated with inventoried migrated/deferred
units. Substrate headers and the emulator were also correctly excluded.

Factory traversal exposed receiver-side companions that primitive grep cannot see and that the ledger omitted:

- `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp`
- `tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp`
- `ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp`
- Quasar Matmul receiver twins: `reader_bmm_tile_layout_in0_receiver{,_metal2}.cpp` and
  `reader_bmm_tile_layout_in1_receiver_writer_padding{,_metal2}.cpp`
- Quasar Conv2D receiver twins: the 1D and 2D weights receiver kernels, both original and `_metal2`
- Shared Conv support has two independent sibling headers, each resolved by quoted include and each requiring a disposition:
  production `operations/conv/conv2d/device/kernels/conv_reader_common.hpp` (D1 support) and Quasar
  `operations/experimental/quasar/conv2d/device/kernels/conv_reader_common.hpp` (D4 support)

Before an apply run, `reconcile-dm-helper` must present the 13 call-site/receiver inventory additions and the regenerated
host-binding map for user approval. The shared Conv support header stays an atomic-scope dependency rather than a false
call-site ledger row. This plan does not mutate the ledger.

## Non-negotiable gates for every migration unit

### Atomic scope

A unit contains every changed kernel face, factory/descriptor emitter, legacy and descriptor path, runtime override,
shared-header consumer, semaphore allocation, and program-cache patch site. A helper-neutral protocol companion may
move in the same ABI commit, but it is not marked migrated unless it actually calls `mcast_pipe`.

### Required-behavior and API gate

Before editing, record the required observable behavior independently of the old implementation: destination set,
sender membership, self-delivery, ACK population, source lifetime, flag/counter semantics, ordering/fence point, and
CB ownership. Treat old argument layouts and factory decisions as changeable unless code or a test proves otherwise.

Use API v11 as-is whenever it can express the behavior. If it cannot, stop that unit before production edits. An API
extension is eligible only if all of the following are true:

1. The missing semantic capability is stated precisely, and changing the factory or surrounding kernel cannot express it.
2. At least two independent production operation families (not clones or `_metal2` twins) benefit from the same invariant.
3. The surface is a protocol abstraction, not a raw primitive passthrough, callback/config blob, or wrapper around one call site.
4. Each adopting host file and kernel file is projected to become smaller.
5. Focused host/device helper tests and at least two production adopters can validate it.
6. Matched performance is within the 2% limit for every affected case.

If any condition fails, defer the unit for now. Any eligible extension gets its own revised plan, Claude review, and
explicit user approval before implementation. No speculative helper extension is authorized here.

### LOC gate

Use `git diff --numstat <unit-base> -- <scoped paths>` immediately before committing. For production code:

- every touched host/factory/descriptor source or header must individually have more deleted lines than added lines;
- every touched kernel or shared kernel header must have more deleted lines than added lines;
- reductions in tests, comments, generated artifacts, or helper-library implementation do not offset a larger caller;
- moving the old block into a call-site-local wrapper does not count;
- no new preprocessor define or parallel legacy/helper branch is allowed.

Failure is a design failure: simplify the formulation or defer the unit. Do not waive the gate by summing one larger file
with a different smaller file, pairing a growing override header with a shrinking `.cpp`, or crediting an older migrated
unit's reduction. The gate applies prospectively to every new edit in this plan; older completed migrations are historical
evidence, not retroactively rewritten to satisfy a rule adopted on 2026-08-16.

### Correctness and coverage gate

Before editing each unit:

1. Collect every test that dispatches the changed factory/header and store the collected node list in `test_map.json`.
2. Run one exact parametrization through `scripts/run_safe_pytest.sh --dev` from a fresh JIT cache and confirm all intended
   kernel sources appear. Avoid `=` in `-k` expressions.
3. Run the complete mapped inventory, including cache, dtype, layout, geometry, NoC, degenerate, and optional-path routes.
4. After host changes, run `./build_metal.sh` before device validation.
5. Run `McastHostFixture.*`, the complete `test_mcast_pipe.py`, and the source/opaque-boundary audit.
6. Run device tests sequentially after activating `/localdev/sjovic/tt-metal/python_env/bin/activate`.

“All tests” means all collected tests that can dispatch the changed production route, not one hand-picked smoke node.
The first exact node is only the compilation/JIT proof.

### Performance gate

Record the baseline before changing source, on the same checkout, device, firmware, dispatch mode, AICLK, environment,
and profiler configuration. For every unit run at least two relevant cases before and after:

- Prefer an existing operation perf test or the real-time migration suite.
- If none exists, run two representative correctness nodes with
  `python -m tracy -r -m pytest <node>` and compare `Kernel duration (ns)` for the affected DM envelope.
- Use warmups plus at least 20 measured iterations; compare the median of at least three run medians.
- Each case must be no worse than +2.00%. A result from +1.50% through +2.00% is repeated for five run medians.
- A noisy or non-comparable envelope is not a pass; add `DeviceZoneScoped` around the affected region or defer.

## Tiers and execution order

### Tier 0 — source-integrated; verification and ledger write-back only

These are easiest because the production migration already exists in the tree. Do not rewrite them unless verification
finds a defect.

1. **Matmul in0 interleaved**
   - Kernels: `reader_bmm_tile_layout_in0_sender_padding.cpp`, `reader_bmm_tile_layout_in0_receiver.cpp`.
   - Host: 1D and 2D reuse-mcast factories plus
     `sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`; all legacy/descriptor bindings.
   - The actual unit commits `1d18a2ca59a` + `a1c9c1f68bc` are LOC-negative in every production caller they touched:
     both production factory `.cpp` files, sparse factory `.cpp`, sender, and receiver. They did not touch the later-growing
     factory override headers; unrelated branch-wide diff growth is not attributed to this unit.
   - Correctness: all `MM-IN0-INTERLEAVED` and `MM-SPARSE-IN0`; guard `MM-BLOCK-SHARDED-HYBRID`.
   - Perf prerequisite: the 2026-08-07 established cases do **not** execute this interleaved in0 multicast route. Add one
     1D and one 2D real-time case with interleaved input and `mcast_in0=true`, asserting both sender and receiver sources.
     Record matched baselines at the pre-unit parent `45033178088b` and compare to current on the same system. Because the
     source is already integrated and worktrees are prohibited, baseline acquisition requires a separately authorized,
     reversible historical-checkout procedure that preserves the dirty submodule; without it, Tier-0 write-back stops.
     The existing 1x1 case is only a current drift/route guard, and sparse needs its own operation-matched profile.

2. **Matmul in0 block-sharded rotating hybrid**
   - Kernel: `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`.
   - Host: all four 1D/2D legacy/descriptor block-sharded bindings.
   - The actual unit commits `1d18a2ca59a` + `a1c9c1f68bc` are LOC-negative in the kernel and both factory `.cpp` files.
   - Correctness: all `MM-BLOCK-SHARDED-HYBRID`, with interleaved and sparse guards.
   - Perf evidence is inherited from the completed 2026-08-07 record in
     `archive/plans/matmul_feedback_plan_2026-08-07.md`: matched pre/post medians at 800 MHz for the block-sharded 2D SDXL
     and transposed cases were +0.643% and -0.045%. The sender-span case has a current absolute record and remains a
     drift/route guard, not a regression claim because no matched historical baseline exists for it.

Prerequisite: source is v11 while `ledger.json.current_api_version` is still 10. Finish API-v11 verification/write-back for
the migrated v10 fleet and clear all **12** current `needs_recheck` entry flags. This is verification only, not a reason to
rewrite unchanged production callers.

### Tier 1 — direct production API-v11 mappings with small, fixed topology

Units 3-5 are intentionally absent from the active rollout: the TT-Metal programming examples and TTNN lab example moved
to D5 by user direction. Existing unit numbers remain stable so the audited path mapping does not churn.

6. **DeepSeek B1 single-device sampling barrier**
   - Kernel: `models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp` and every host/fused emitter.
   - Formulation: no-data Counter/Flag control pipe (preserve the single-device coordinate transformation); mesh mode stays
     untouched and is not used for validation.
   - Correctness: all single-device argmax and top-k sampling nodes that compile this kernel, including program-cache paths.
     The ledger's old 101-core coverage gap is resolved on the current Blackhole system, whose 11×10 worker grid has 110
     cores; record that device evidence before clearing the flag.
   - Perf: profile two single-device sampling nodes (argmax 101-core and one top-k case).

### Tier 2 — API v11 as-is, but a larger host/ABI or protocol refactor

7. **DRAM-sharded Matmul in0**
   - Kernel/factory: `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` and
     `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`.
   - One binary has sender-only, sender+compute loopback, and receiver roles. Use the existing divergent ACK count,
     loopback inference, degenerate local copy, and raw-L1 source support; preserve asymmetric `SKIP_MCAST` behavior.
   - Correctness: all `test_matmul_dram_sharded.py` routes after exact JIT confirmation.
   - Perf: profile two DRAM-sharded shapes with different sender membership.

8. **Group-attention Matmul rotating multicast**
   - Kernel/factory: `reader_mcast_transformer_group_attn_matmul.cpp` and
     `group_attn_matmul_program_factory.cpp`.
   - Use rotating sender/receiver faces. First prove whether the historical post-flag full barrier expresses a required
     completion point; do not preserve it merely because it exists.
   - Correctness: every `test_group_attn_matmul*` route, including sharded, dtypes, program cache, and exhaustive nodes.
   - Perf: profile two existing `num_loops=20`/representative dtype-shape routes.

9. **Conv3D multicast weight-sharing branch**
   - Kernel/factory: `experimental/conv3d/device/kernels/writer.cpp` and its dispatch factory.
   - Migrate only `McastSender`/`McastReceiver` behavior. The unicast chain and unrelated reduction semaphores remain
     operation-owned. Express passive drain with ordinary repeated receives if behavior matches; do not add an ack-only API.
   - Correctness: all Conv3D tests that select multicast sharing, then the complete `test_conv3d.py` inventory and cache tests.
   - Perf: two representative Conv3D shapes, one grouped and one non-grouped.

10. **Sharded LayerNorm post-allgather**
    - Kernels: sender/receiver `reader_mcast_*_unary_sharded_ln_post_allgather.cpp`.
    - Host: `sharded_layernorm_factory_helpers.cpp` and all dispatch/cache paths.
    - Preserve explicit sender self-delivery without enlarging the receiver rectangle. First try a local operation-owned copy
      plus the existing remote pipe; add no one-off loopback knob.
    - Correctness: all `LN-POST-ALLGATHER`, plus `LN-PRE-ALLGATHER` and `LN-SHARDED` ABI guards.
    - Perf: profile two post-allgather nodes (LayerNorm and RMSNorm; distinct geometries).

11. **Plain sharded LayerNorm two-phase reduction**
    - Kernels: sender/receiver `reader_mcast_*_unary_sharded_ln.cpp`.
    - Host: the variant-specific builders in `sharded_layernorm_factory_helpers.cpp`.
    - Compose distinct existing pipes for the bounded gather-ready phase and monotone block phase; gather reads and second-stage
      synchronization remain operation-owned.
    - Correctness: all `LN-SHARDED`, with complete pre/post-allgather guards.
    - Perf: two plain-sharded nodes covering one-stage and two-stage geometries.

12. **Interleaved GroupNorm, legacy and Welford**
    - Eight kernels: sender/receiver pairs for `reader_mcast_*_unary_gn.cpp` and
      `welford_reader_mcast_*_unary_gn.cpp`.
    - Host: `groupnorm_mcast_program_factory.cpp` and no-mcast/cache variants that share arguments.
    - Reuse the already migrated sharded-v2 precedent: compose mid/first/last `Mcast2D` wires, preserve gather tails, and use
      separate existing calls for go and data signals. Do not invent a destination-set API.
    - Correctness: the complete GroupNorm inventory for legacy and Welford, fixed/default routing, DRAM, cache, dtype, and
      zero/one/two-edge geometry.
    - Perf prerequisite: add two **interleaved** GroupNorm cases (legacy and Welford) to the real-time migration suite and
      confirm they JIT the non-v2 kernels from this unit. The existing `groupnorm_sdxl_1920_{legacy,welford}` cases use
      block-sharded memory and dispatch already-migrated `*_unary_sharded_gn_v2.cpp`; they are guards, not matched evidence.

13. **Single-device SDPA and SDPA-decode star paths**
    - `transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` plus every including reader/writer factory.
    - Migrate only `read_k`'s fixed-column star. Preserve the proven BH completion point and audit every header consumer.
    - `transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp` is not in this unit; it is part of the Tier-3 relay study.
    - Correctness: all single-device `test_sdpa_decode.py` and nightly/cache routes that compile `read_k`.
    - Perf: profile two decode shapes with different `q_heads_parallel_factor`, one sharded.

14. **Argmax multicore control**
    - Kernel/factory: `reader_argmax_interleaved_multicore.cpp` and `argmax_multi_core_program_factory.cpp`.
    - Compose two existing control pipes for the two rectangles and preserve the independent unicast fan-in/done counter.
    - Correctness: all multicore Argmax routes, shapes, dimensions, dtypes, and cache cases after exact path confirmation.
    - Perf: profile two multicore shapes, including `[64,128]`, `dim=-1`.

15. **Move overlap, tiled and stick**
    - Kernels: production `move_interleaved_with_overlap.cpp` and `move_stick_layout_interleaved_with_overlap.cpp`.
    - Host: `move_overlap_program_factory.cpp` and cache override.
    - Compose the existing per-rectangle signal wires; keep the operation-owned return counter. The source must first be ported
      cleanly from legacy primitives without a parallel branch.
    - Correctness: every overlap case in `test_move.py`, both TILE and ROW_MAJOR, all mapped shapes/dtypes/memory configs, plus
      program-cache coverage.
    - Perf: profile one TILE and one ROW_MAJOR overlap case.

16. **DeepSeek single-device ordinary multicast units**
    - Unit 16a: `experimental/deepseek_prefill/unified_routed_expert_ffn/.../unified_routed_expert_ffn_reader.cpp` and its factory.
    - Unit 16b: `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp` with the H2D emitter.
    - Unit 16c: `persistent_d2h_reader.cpp` with the D2H emitter.
    - These are three independent atomic units. Preserve non-multicast routing and socket/metadata synchronization outside the pipe.
    - Correctness: complete routed-expert/bias tests. The exact H2D/D2H worker-sync service routes require Blackhole
      Galaxy/UBB in the current tree; on a single-card machine record the coverage gap rather than crediting the skipped tests.
    - Perf: two operation-matched cases per unit; profile regular nodes where no perf test exists.

### Tier 3 — prototype against API v11; extension allowed only through the generality gate

18. **Experimental decode Matmul two-hub broadcast (new recall)**
    - Kernels: `reader_full_width_sharded.cpp`, `reader_partial_width_sharded.cpp`.
    - Host: `full_width_sharded_program_factory.cpp`, `partial_width_sharded_program_factory.cpp`.
    - Required behavior is two producers writing disjoint regions to the same receiver CB followed by a count-of-two completion.
      This violates the helper's documented single-sender-per-receiver precondition. First prototype composition of two existing
      no-pre-handshake Counter pipes without production edits and prove no early CB publication or semaphore race.
    - If composition is invalid and no second unrelated production family needs the exact multi-producer invariant, defer; do not
      add a decode-only `send_data_only`/multi-hub wrapper.
    - Correctness: all full- and partial-width cases in `test_matmul_decode.py`; batched-width is an ABI guard.
    - Perf: one full-width and one partial-width case from the same file.

19. **DeepSeek preprogrammed multicast family**
    - `models/demos/deepseek_v3_b1/unified_kernels/kv_cache_update.hpp` and `flash_mla.hpp`, including every micro-op/fused
      kernel that includes them.
    - Their preprogrammed data/sem transaction state is a performance behavior, not automatically a semantic requirement.
      Prototype API-v11 ordinary pipes and measure first.
    - If performance exceeds +2%, an extension is considered only if one stable preprogrammed backend serves both independent
      KV-cache and MLA paths and simplifies both host and kernel callers. Do not move each header's current block verbatim into
      `mcast_pipe`.
    - Correctness: all B1 KV-cache update and Flash-MLA micro/fused tests, with JIT confirmation for every including kernel.
    - Perf: at least two KV-cache and two Flash-MLA operation cases because the shared headers have multiple consumers.

20. **Single-card `moe_compute` persistent value signaling**
    - Kernels: `experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp` and `tilize_writer.cpp`.
    - Host: `moe_compute_program_factory.cpp`, single-card `(1,1)` dispatch only.
    - API v11 is **not** an as-is match. `num_matmul_cores` is compile-time, but receivers maintain exact/`wait_min`
      thresholds advanced by that stride without clearing. The sender also broadcasts arbitrary accumulated values and
      sometimes rebroadcasts an observed value. Flag receive always clears once, while Counter signaling is fixed `+1`;
      neither preserves this persistent value protocol. The gap spans both arbitrary-value broadcast and non-clearing
      value observation, not a `send_signal(delta)` convenience.
    - State the complete persistent-value invariant and look for a second unrelated production family needing the same
      semantics. A `moe_compute`/`moe_gpt` clone pair is not independent evidence. If none exists, defer rather than moving
      `set_multicast_safe` and its waits into a MoE-only helper surface.
    - Coverage precondition: the `(1,1)` Blackhole route is known-red for PCC under issue #50038 (skipped in CI, failing
      off-CI). No production edit or performance claim begins until that issue is resolved or a supported device produces
      a green unchanged-source baseline. Then run every `(1,1)` test in `test_moe_compute_single_card.py`, including
      DeepSeek, GPT-OSS, non-tile tokens, bias, compute-only/fused-local, and full-local B1.
    - Perf after the coverage precondition: profile one DeepSeek and one GPT-OSS single-card case.

21. **Cross-ID relay family: SDPA chain plus Indexer Score**
    - Sources: `transformer/sdpa/device/kernels/dataflow/chain_link.hpp`,
      `transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`, and
      `experimental/indexer_score/device/kernels/reader_indexer_score.cpp`.
    - Host: all single-device SDPA and `indexer_score_program_factory.cpp` bindings; the ring indexer factory is a guard only.
    - Required behavior is a write-once source semaphore relayed into a different receiver semaphore ID. Unlike the first
      draft, this now has two independent single-device production families (SDPA and Indexer Score), so it may satisfy the
      generality threshold. Prototype a narrow relay-channel abstraction against existing `Semaphore::relay_multicast` and
      require both families' host/kernel callers to shrink; do not copy `ChainLink` wholesale into the star helper.
    - Correctness: every single-device SDPA route that compiles `reader_interleaved.cpp` and the complete Blackhole-only
      `test_indexer_score.py` inventory through the ordinary indexer factory. Multi-device ring/fabric routes are guards/deferred.
    - Perf: two single-device SDPA and two single-device Indexer Score nodes.

### Tier 4 — no active units

Former units 22-24 moved to D4 by user direction. Quasar paths remain inventoried, but none are part of this rollout.

### Tier 5 — real single-device callers blocked by coverage or likely gate failure

These remain tiered rather than being mislabeled multi-device. They do not enter production editing until their named
precondition is resolved.

25. **DeepSeek B1 `unified_kernels/mcast.hpp` facade**
    - Inspect every single-device consumer and prototype deleting the facade in favor of direct `mcast_pipe` wires, or a
      demonstrably thinner adapter. The expected blocker is the red-flag outcome: wrapping one reusable abstraction in another
      grows layering/LOC. If every affected host and kernel/header cannot shrink independently, stop at the LOC/generality gate
      and defer with measurements rather than a pre-judged label.
    - Correctness/perf: every single-device B1 fused/micro-op consumer, with at least two KV/MLA-independent perf cases.

26. **Single-device Matmul ring-all-gather start barrier**
    - Active caller: production `reader_bmm_tile_layout_in1_ring_all_gather.cpp` (flag-only on-chip start barrier). Its in0
      companion is ring unicast with no rectangle multicast and remains helper-neutral. The Quasar counterpart is D4.
    - Formulation attempt: existing no-data handshaked Flag/Counter pipe; keep ring forwarding and return counters outside.
    - Coverage blocker: `test_ring_matmul.py::test_multi_core_matmul_1d_wh_minimal` is a single-device route but is skipped on
      the current Blackhole system. Establish a runnable single-device route on supported hardware, then collect the full test
      and two perf baselines before edits. Lack of a runnable baseline stops the unit; it is not called multi-device.

27. **Single-device `moe_gate_mm/dm1.cpp`**
    - Kernel/factory: `experimental/deepseek/moe/moe_gate_mm/device/kernels/dm1.cpp` and `moe_gate_mm_program_factory.cpp`.
    - Preserve the `partial_semaphore` multiplexing with sender-to-collector reduce-increment phases; a helper-owned reset/ctor
      store may corrupt the live counter.
    - Coverage blocker: every `test_moe_mm` case skips on Blackhole and Wormhole under issue #44858, leaving no correctness or
      performance evidence on this system. Resolve or obtain a runnable supported route before production edits, then run the
      complete test file and two matched perf cases.

## Deferred now

### D1 — explicitly requested convolution block-sharded activation reader

Defer all live twins of this protocol:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`

Reason: `mcast_block_chunked` deliberately interleaves `wait_front` with per-burst multicasts so transmission begins before
the complete source block is ready. API v11 only auto-chunks an already-ready block. Moving this loop into a Conv-only helper
would be exactly the prohibited one-call-site wrapper. Revisit only when another unrelated producer-overlapped multicast user
establishes a general streaming-source contract and both caller layers become smaller.

### D2 — multi-device/fabric operations (no valid device coverage now)

These are deferred before migration, as requested, even where a local rectangle leg looks simple:

- Matmul ring/all-gather: the `experimental/ccl/llama_all_gather_matmul_async` in1 ring reader and its multi-device factory.
- SDPA ring/fabric: `exp_ring_joint_reader.cpp`.
- `experimental/transformer/all_reduce_create_qkv_heads/.../worker_writer.cpp`.
- `experimental/deepseek_prefill/combine/.../reader_combine.cpp`.
- CCL RMS all-gather: `rms_sender_reader.cpp`, `rms_writer.cpp`, and receiver companions.
- CCL Llama all-gather Matmul: `worker_receiver.cpp` and its paired kernels/factory.
- CCL `moe_gpt`: `tilize_reader.cpp`, `tilize_writer.cpp`.
- CCL selective-reduce-combine: `reader.cpp`, `writer.cpp`.
- CCL all-gather-concat: `llama_all_gather_concat_writer.cpp` and receiver companions.
- CCL all-to-all: `all_to_all_sender_writer.cpp` and its fabric companion.
- Persistent D2D: `ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp`, `persistent_d2d_sender.cpp`.

The single-card `(1,1)` `moe_compute` path is not in this deferral; it is Tier 3.

### D3 — not a `mcast_pipe` caller

- `data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp`: already-closed helper-neutral done-counter
  companion of the migrated `sort-single-row-control` atomic unit; it is not a remaining standalone migration.
- Production `reader_bmm_tile_layout_in0_ring_all_gather.cpp`: peer-to-peer ring unicast with no rectangle multicast.
- `models/demos/deepseek_v3_b1/unified_kernels/dataflow_utils.hpp`: primitive substrate used by several protocols.

Keep these in the audit record, but do not claim a migration. If later ABI cleanup makes a helper-neutral file smaller, record
it as a companion only.

### D4 — all Quasar operations (user-directed scope deferral)

Defer every Quasar `mcast_pipe` candidate, including all original and `_metal2` variants and their factory-atomic companions:

- Matmul DRAM-sharded, fixed in0, block-sharded hybrid, fixed in1, sparse bindings, and the in1 ring-all-gather start barrier.
- Conv2D width-sharded activation, block-sharded activation, 1D/2D weights sender/receiver kernels, and the Quasar
  `conv_reader_common.hpp` support header.
- Move overlap tile- and stick-layout kernels.

This is a rollout-scope decision, not a claim that the helper cannot express these protocols. Preserve the inventory and
revisit only after explicit user direction; no Quasar baseline, migration, build, correctness test, or perf run is authorized.

### D5 — programming and lab examples (user-directed scope deferral)

Defer the example-only multicast callers:

- TT-Metal reuse-mcast Matmul programming example: its host and all four in0/in1 sender/receiver combinations.
- TT-Metal contributed multicast programming example: `coordinator_kernel.cpp`, `inbound_kernel.cpp`, and its host;
  `outbound_kernel.cpp` remains helper-neutral.
- TTNN lab multicast example: `mcast_sender.cpp`, `mcast_receiver.cpp`, and its host; `write_tiles.cpp` remains helper-neutral.

Keep the ledger and receiver-companion inventory coverage, but do not migrate, build, run, or profile these examples unless
the user explicitly reopens them.

## Per-unit execution protocol

For each approved unit, in tier order:

1. Freeze the scoped baseline: source hashes, `git diff --numstat`, collected tests, exact JIT source list, and two-or-more perf records.
2. Write the required-behavior map and existing-v11 formulation into the unit log.
3. Make the host/factory and kernel change atomically; no unrelated cleanup.
4. Run the per-file LOC gate before building. Stop immediately if any production caller grows.
5. Build host code with `./build_metal.sh`.
6. Run one exact `--dev` compile/JIT node, then every mapped correctness test sequentially.
7. Run helper host/device/source-audit suites.
8. Run matched performance. Stop on any case above +2.00%.
9. Update ledger, test map, unit log, report, and plan outcome in the same commit as the source.
10. Re-run primitive recall and factory companion discovery so the next unit starts from a complete inventory.

No rebase, push, reset, worktree, migration, build, or device test is authorized by approval of this planning artifact alone.

## Appendix A — path-level disposition of every non-migrated ledger entry

This table is the completeness check for the 74 current ledger rows. `Tn.m` refers to the numbered unit above.

| Disposition | Ledger path |
|---|---|
| T0.1 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` |
| T0.1 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp` |
| T2.7 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` |
| T0.2 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` |
| T2.8 | `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp` |
| T5.26 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` |
| D3 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp` |
| D5 | `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp` |
| D5 | `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp` |
| D5 | `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp` |
| D1 | `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` |
| T2.9 | `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp` |
| T2.10 | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` |
| T2.10 | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` |
| T2.12 | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_gn.cpp` |
| T2.12 | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_gn.cpp` |
| T2.11 | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp` |
| T2.11 | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp` |
| T2.12 | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp` |
| T2.12 | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp` |
| T3.21 | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp` |
| T3.21 | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp` |
| T2.13 | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp` |
| T2.14 | `ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp` |
| D3 | `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp` |
| T2.15 | `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp` |
| T2.15 | `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp` |
| D5 | `tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp` |
| T5.25 | `models/demos/deepseek_v3_b1/unified_kernels/mcast.hpp` |
| D3 | `models/demos/deepseek_v3_b1/unified_kernels/dataflow_utils.hpp` |
| T3.19 | `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` |
| T3.19 | `models/demos/deepseek_v3_b1/unified_kernels/kv_cache_update.hpp` |
| T1.6 | `models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp` |
| T5.27 | `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm1.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_sender_reader.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/worker_receiver.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_reader.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_writer.cpp` |
| T3.20 | `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp` |
| T3.20 | `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_writer.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp` |
| T2.16 | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp` |
| T2.16 | `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp` |
| D2 | `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp` |
| T2.16 | `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_d2h_reader.cpp` |
| D2 | `ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp` |
| D2 | `ttnn/core/tensor/kernels/persistent_d2d_sender.cpp` |
| T3.21 | `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/reader_indexer_score.cpp` |
| D5 | `ttnn/examples/lab_multicast/kernels/dataflow/mcast_sender.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/activation_reader_width_sharded.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/activation_reader_width_sharded_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp` |

## Appendix B — proposed inventory and shared-scope additions

| Disposition | Newly discovered, paired, or shared-scope path |
|---|---|
| T3.18 | `ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_full_width_sharded.cpp` |
| T3.18 | `ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_partial_width_sharded.cpp` |
| D5 | `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp` |
| D5 | `tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp` |
| D5 | `ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks_metal2.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` |
| D4 | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks_metal2.cpp` |
| D1 support | `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp` |
| D4 support | `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/conv_reader_common.hpp` |

## Review record

Claude review: previous revision **APPROVED** on 2026-08-16; the user waived an additional Claude pass for the scope-only D4/D5 revision.
User approval: **APPROVED** on 2026-08-16, including the D4 Quasar and D5 programming/lab example deferrals.
