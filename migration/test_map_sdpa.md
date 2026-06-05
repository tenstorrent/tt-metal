# SDPA op-family — static kernel→test map (Phase 2)

STATIC analysis only — NO device runs were performed. The `verification_command`s below are
the commands a later phase should run to confirm coverage; "confirm kernel built" means grep the
kernel basename inside the post-run JIT artifacts (`generated/` — the build cache was empty at map
time, so no path is hard-coded; after a run the JIT tree appears under `generated/` and the kernel
`.cpp`/`.hpp` basename will be present once that program factory dispatches).

Note on tests: sdpa tests live under `tests/ttnn/unit_tests/operations/sdpa/` and
`tests/ttnn/nightly/unit_tests/operations/sdpa/` (NOT `.../transformer/`).

---

### ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp
- role / tag: ref / prior-art. EXISTING two-sided ChainLink helper (flush + `linked=true`). Design template, NOT a migration target — document where used + which tests exercise it.
- consumers: included ONLY by `ring_joint_reader.cpp` (`#include "chain_link.hpp"`, line 8), which instantiates `ChainLink<head_mcast_enabled, true> head_chain` (ring_joint_reader.cpp:138) and `ChainLink<batch_mcast_enabled, false> batch_chain` (ring_joint_reader.cpp:162). NOT included by `exp_ring_joint_reader.cpp` (that kernel open-codes the chain-link).
- factory: `RingJointSdpaProgramFactory` in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp` (reader kernel_source set at :1367). Selected when: `ttnn.prim::ring_joint_scaled_dot_product_attention` is dispatched (ring + joint attention path).
- op: `ttnn.transformer.ring_joint_scaled_dot_product_attention` (host wrapper `ring_joint_scaled_dot_product_attention`, sdpa.cpp:176 → `ttnn::prim::ring_joint_scaled_dot_product_attention`, sdpa.cpp:201).
- candidate_validation_set: ring-joint requires a multi-device mesh (CCL ring). No single-chip fast case exists in the unit-test tree; nearest exerciser is the ring-distributed/joint nightly suite. Smallest joint shape generally: `test_joint_sdpa[ ... seq_len=15 joint_seq_len=19 ... ]` BUT that hits the NON-ring `joint_sdpa` factory (joint_reader.cpp), not chain_link. ChainLink itself is only reachable via a true ring-joint multi-device run.
- candidate_regression_set: ring-joint multi-device nightly (mesh-gated). `tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py` covers the *exp* variant, NOT this one — exp uses open-coded chain-link, so it does not build chain_link.hpp.
- verification_command: (mesh-gated; needs ≥2-device ring) `scripts/run_safe_pytest.sh <ring_joint test>`; then confirm `grep -rl "ring_joint_reader" generated/` AND `grep -rl "chain_link" generated/`. No suitable single-chip param substring available in the inspected suites.
- coverage_confidence: low
- gaps: This helper is the design template, not a migration target, so low confidence is acceptable. But note: chain_link.hpp is reachable ONLY through the multi-device ring_joint factory; there is no single-chip unit test that builds it. If a later phase needs to prove ChainLink behavior on-device it must run the ring_joint multi-device nightly. Do NOT assume `test_exp_ring_joint_*` exercises chain_link.hpp — it does not.

---

### ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp
- role / tag: refactor, hybrid — open-coded chain-link, K+V (linked write + companion semaphore mcast back-to-back; flush via `noc_async_writes_flushed()`, `true /*linked*/` at :457, :616).
- factory: `SDPAOperation::SDPAProgramFactory::create_descriptor` in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:152` (reader kernel_source = reader_interleaved.cpp at sdpa_program_factory.cpp:1241). ALSO used by `RingDistributedSdpaProgramFactory::create_descriptor`, `ring_distributed_sdpa_program_factory.cpp:75` (reader path at :461). Selected when: main interleaved SDPA forward (prefill) — both causal and non-causal, chunked, sliding-window, attention-sink all route through this reader (it is THE prefill reader).
- op: `ttnn.transformer.scaled_dot_product_attention` (host `scaled_dot_product_attention`, sdpa.cpp:21 → `ttnn::prim::sdpa`). Chunked variant → `chunked_scaled_dot_product_attention` (sdpa.cpp:80/116). Ring-distributed variant → `ring_distributed_scaled_dot_product_attention` (sdpa.cpp:355).
- candidate_validation_set: `test_sdpa_tt[bfp8-dram_interleaved-q128-k128-...]` (b=1,nh=8,nkv=1,s=2048,d=128 — causal) in `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py:575`; plus `test_sdpa_noncausal[bf16-q128-k128-...]` (same shape, non-causal) at :603 for the non-causal dispatch branch.
- candidate_regression_set: full `test_sdpa_prefill.py` (causal/noncausal/sliding-window/attention-sink/program-cache); `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_prefill.py`; chunked coverage in `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py`; ring-distributed in `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_ring_distributed.py`.
- verification_command: `source python_env/bin/activate && scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py -k "sdpa_tt and bfp8 and q128"` (note: filter on value substrings only — no `name=value`). Then `grep -rl "reader_interleaved" generated/` to confirm the kernel was JIT-built.
- coverage_confidence: high
- gaps: The mcast (linked-write + companion-semaphore) branch at :443-:459 / :616 only fires when K/V are multicast across cores (multi-core grid sharing a kv head). The single-head s=2048 case is multi-core and exercises it, but if a later tier wants to isolate the non-mcast branch, pick a 1-core config. Causal vs non-causal both covered by the two validation cases.

---

### ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp
- role / tag: refactor, hybrid — open-coded chain-link + ring sync. The RING part is OUT of scope; only the chain-link (linked write + flush, `true /*linked*/` at :309, :391; `noc_async_writes_flushed()` at :316,:398,:434) is in scope.
- factory: `ExpRingJointSDPAProgramFactory::create_descriptor` in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.cpp:107` (reader_kernel.kernel_source = exp_ring_joint_reader.cpp at :1382; kernel pushed at :1687). Selected when: `ttnn.prim::exp_ring_joint_scaled_dot_product_attention` is dispatched (experimental ring + joint attention with persistent K/V output buffers and AllGather). Requires `mesh_dispatch_coordinate` (exp_ring_joint_sdpa_program_factory.cpp:116) — multi-device only.
- op: `ttnn.transformer.experimental ...` exp ring joint (host `ExecuteExpRingJointAttention::invoke`, sdpa.cpp:231 → `ttnn::prim::exp_ring_joint_scaled_dot_product_attention`, sdpa.cpp:255).
- candidate_validation_set: `tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py::test_exp_ring_joint_attention_sdpa_accuracy` (test_exp_ring_joint_sdpa.py:533). Shapes are auto-generated from detected device count (`generate_test_configs`, :118) — needs ≥4 devices (`if num_devices < 4: skip`, :128). No fixed-small single-chip case.
- candidate_regression_set: same file — `test_exp_ring_joint_attention_sdpa_accuracy`, `..._determinism` (:569), `..._sweep_perf_impl` (:509).
- verification_command: (mesh-gated, ≥4 devices, Blackhole) `source python_env/bin/activate && scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py -k "accuracy"`; then `grep -rl "exp_ring_joint_reader" generated/`.
- coverage_confidence: low
- gaps: Multi-device Galaxy/Blackhole-mesh only; cannot run on a single chip. The chain-link portion (in scope) is interleaved with ring sync (out of scope) in the same loop body — migration must surgically wrap only the linked-write+flush pairs (:307-:316, :389-:398) and leave the ring semaphore logic untouched. No single-chip fast validation case exists.

---

### ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp
- role / tag: refactor, both — `read_k` vertical-column K multicast (`noc_async_write_multicast(..., /*linked=*/false)` at dataflow_common.hpp:635-:640; BARRIER + `linked=false` — the UNLINKED F4 path). NOTE: this is the *sdpa/* dataflow_common.hpp; the *sdpa_decode/* dataflow_common.hpp (sdpa_decode/.../dataflow_common.hpp:10) `#include`s it, so the decode kernels reach `read_k` transitively.
- factory: `SdpaDecodeDeviceOperation::create_descriptor` in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:29` (reader kernel reader_decode_all.cpp at :795, writer writer_decode_all.cpp at :802 — both include dataflow_common.hpp). The mcast (`use_mcast=true`) branch of `read_k` is gated by `use_col_major_group_indexing` (sdpa_decode_program_factory.cpp:254-255) passed as compile-time `use_k_mcast` (reader_decode_all.cpp:53). Selected when: `q_heads_parallel_factor > 1` AND `grid_size.y >= num_cores_per_head` AND not on subcore grid AND Q locally available — i.e. GQA decode with multiple Q heads parallelized per KV head on a sharded grid. The `use_mcast=false` (no mcast, plain read) branch runs in the common decode path.
- op: `ttnn.transformer.scaled_dot_product_attention_decode` / `paged_scaled_dot_product_attention_decode` (host `scaled_dot_product_attention_decode` sdpa_decode.cpp:41, paged at :101 → `ttnn::prim::sdpa_decode`).
- candidate_validation_set: non-mcast `read_k` path: `test_sdpa_decode[kv_bfp8-...]` (b=4,nh=32,nkv=8,s=8192,d=128, GQA) in `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py:50`. For the UNLINKED mcast (use_k_mcast=true) F4 path specifically: needs the sharded col-major-group config → `test_sdpa_decode_sharded` (test_sdpa_decode.py:234) is the candidate that can satisfy `q_heads_parallel_factor>1` with sharded Q.
- candidate_regression_set: full `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py` (decode, non_causal, paged, sharded, kv_head_core_divisibility, non_tile_aligned_heads) + `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode.py` + `test_sdpa_decode_cache.py` + `test_sdpa_decode_sink.py`.
- verification_command: non-mcast: `source python_env/bin/activate && scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py -k "sdpa_decode and kv_bfp8"`; mcast F4 path: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py -k "sharded"`. Confirm with `grep -rl "reader_decode_all\|dataflow_common" generated/`.
- coverage_confidence: med
- gaps: The in-scope target is the UNLINKED (`linked=false`) mcast branch, which only compiles/runs when `use_col_major_group_indexing` resolves true. I confirmed the host gate (sdpa_decode_program_factory.cpp:254-255) statically but did NOT confirm which exact decode test param drives `q_heads_parallel_factor>1` on-device — that needs a run to verify via the JIT compile-time arg. `test_sdpa_decode_sharded` is the best candidate but the specific shape that flips the gate is unconfirmed. Flag for the device-verification subagent to check `use_k_mcast=1` in the built reader's compile args.

---

### ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp
- OUT OF SCOPE — fabric cross-chip writer (all_reduce_create_qkv_heads, CCL fabric). Not part of the sdpa chain-link/mcast helper rollout.
