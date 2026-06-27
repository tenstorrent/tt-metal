# CCL / deepseek / examples — STATIC kernel→test map (Phase 2)

Static analysis only — NO device runs. Repo root: /localdev/sjovic/tt-metal

Conventions:
- "multi-device only" = the kernel is only ever created when the op runs on a mesh
  (≥2 devices) fixture, so a single-device migration-validation run cannot reach it.
  These are coverage GAPS.
- pytest `-k` cannot use `name=value`; filters below use a value-substring only.
- Verify the kernel built: after a run, grep the kernel basename under
  `generated/` JIT artifacts (e.g. `find generated -name '<basename>'`), or grep the
  JIT'd source/hash dirs. (Static map — not yet confirmed against a build.)

---

## IN-SCOPE CLEAN

### models/demos/deepseek_v3_b1/unified_kernels/kv_cache_update.hpp
- role / tag: sender (NOPE NCRISC mcast, preprogram-state) — CLEAN
- factory: deepseek_v3_b1 is a model with host-side fused-op program builders, not a
  TTNN program factory. KVCacheUpdate::Op<IsNopeSenderCore,IsNopeCore,IsRopeCore> is
  instantiated by the kv-cache-branch fused op. (selected when: NOPE sender core →
  `IsNopeSenderCore=true` mcasts rmsnorm output to the knope grid; receivers
  `IsNopeCore`; ROPE cores `IsRopeCore` skip the mcast).
- op/model: deepseek_v3_b1 KV-cache update branch (attention block).
- candidate_validation_set: models/demos/deepseek_v3_b1/tests/unit_tests/test_kv_cache_branch.py::test_kv_cache_branch[1-True-1e-06] (single `device`; position_id=1)
- candidate_regression_set: test_kv_cache_branch (all position_id 0/1/5/7) + test_kv_cache_branch.py::test_kv_cache_dram_shard (position_id 0/1/34/128/1130) + test_attention_block.py / test_decoder_block.py
- verification_command: `scripts/run_safe_pytest.sh models/demos/deepseek_v3_b1/tests/unit_tests/test_kv_cache_branch.py -k "test_kv_cache_branch"` then `find generated -name 'kv_cache_update*' -o -name '*kv_cache*'` (kernel is header-included into the fused-op kernel .cpp; grep the fused attention/decoder kernel basename instead)
- coverage_confidence: med (single-device reachable; but kernel is a .hpp pulled into a fused-op .cpp — must confirm the including kernel is JIT-built by the chosen test)
- gaps: header-only kernel; basename grep must target the *including* fused-op kernel, not kv_cache_update.hpp. Confirm test_kv_cache_branch actually drives the NOPE-sender mcast path (single device, knope grid is intra-chip — OK).

### models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp
- role / tag: sender (flag-only single-device loop barrier; final core mcasts release) — CLEAN
- factory: micro-op TopKSampling::Op; single-device loop barrier guarded by
  `!SamplingReaderCTArgs::mesh_mode` and `Core::is_final_core` (mcast release) vs
  non-final (wait). (selected when: `sampling_mesh_mode==0` AND num_internal_iterations>1
  AND `sampling_loop_num_dests>0`; final core does the multicast set).
- op/model: deepseek_v3_b1 LM-head top-k sampling (single-device 101-core argmax).
- candidate_validation_set: models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_argmax_single_device_101_cores (single `device`; smallest seed/final_core_idx param)
- candidate_regression_set: test_sampling_argmax_single_device_101_cores (all params) + test_lm_head_sampling.py
- verification_command: `scripts/run_safe_pytest.sh models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py -k "single_device_101"` then `find generated -name 'sampling_kernel*'`
- coverage_confidence: high (single-device test directly named, single-device loop-barrier path is the `!mesh_mode` branch this test hits)
- gaps: the loop barrier only fires when num_internal_iterations>1 — confirm the single-device test uses >1 internal iterations (else only the body runs, not the barrier mcast).

### ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_sender_reader.cpp
- role / tag: sender (INCLUDE_SRC loopback + barrier; all-to-all mcast sender) — CLEAN
- factory: rms_allgather_program_factory.cpp:753 `reader_mcast_sender_kernels_id = CreateKernel(... sender_cores ...)` RISCV_0. (selected when: core ∈ sender_cores; always created. SKIP_WRITE_BACK define set when `skip_write_back = (output.shard_spec()==a.shard_spec())`, factory:88).
- op/model: ttnn.experimental fused RMSNorm + all-gather (rms_allgather).
- candidate_validation_set: multi-device only — see gaps. (tests/ttnn/unit_tests/operations/ccl/test_minimals.py drives rms_allgather but on a mesh fixture)
- candidate_regression_set: tests/ttnn/unit_tests/operations/ccl/test_minimals.py (rms-allgather cases)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/test_minimals.py -k "rms"` then `find generated -name 'rms_sender_reader*'`
- coverage_confidence: low (CCL all-gather — needs mesh; the intra-chip mcast leg of the reader still runs per-chip, but op entry requires multi-device)
- gaps: MULTI-DEVICE ONLY — no single-device validation. Coverage gap for single-device migration validation.

### ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/worker_receiver.cpp
- role / tag: receiver (INCLUDE_SRC partner; loopback-src mcast to mm cores) — CLEAN
- factory: llama_all_gather_matmul_async_program_factory.cpp:289 `worker_receiver_kernel_id = CreateKernel(... worker_receiver.cpp ...)`. (selected when: always created in the fused AG+matmul path; per-core mm_core_offset / next_core ids set in loop).
- op/model: ttnn.experimental all_gather_matmul_async (llama, fused AG + matmul).
- candidate_validation_set: multi-device only — tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py (mesh / ring_size>1)
- candidate_regression_set: test_llama_all_gather_matmul.py + test_ccl_async_TG_llama.py (TG)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py -k "all_gather_matmul"` then `find generated -name 'worker_receiver*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (ring_size from mesh). Coverage gap.

---

## REFACTOR

### ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/reader_dispatch.cpp
- role / tag: refactor — data + COUNTER inc_multicast + wait_min
- factory: dispatch_program_factory.cpp:616/1242 (two creation sites; RISCV_1 reader). (selected when: deepseek_prefill `dispatch` op program; both sites create reader_dispatch.cpp on the relevant core sets — distinct grid configs).
- op/model: ttnn.experimental.deepseek_prefill.dispatch (MoE token dispatch, deepseek_v3_d_p).
- candidate_validation_set: multi-device only — models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_dispatch.py::test_ttnn_dispatch (min mesh 2x2 from ALL_MESH_CONFIGS)
- candidate_regression_set: test_prefill_dispatch.py (all mesh configs) + op_unit_tests/test_dispatch_combine_l1_small_semaphores.py
- verification_command: `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_dispatch.py -k "predictable"` then `find generated -name 'reader_dispatch*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (mesh_device, mesh_config, sp_axis; min 2x2). Intra-chip COUNTER inc_multicast leg unreachable on a single card. BLOCKER: COUNTER inc_multicast + wait_min (needs atomic-barrier counter semantics in helper).

### ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp
- role / tag: refactor — data + COUNTER
- factory: combine_program_factory.cpp:911 `CreateKernel(... reader_combine.cpp ...)` RISCV_1 (one per sender). (selected when: deepseek_prefill `combine` op program).
- op/model: ttnn.experimental.deepseek_prefill.combine (MoE token combine, deepseek_v3_d_p).
- candidate_validation_set: multi-device only — models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_combine.py::test_ttnn_combine (min mesh 2x2)
- candidate_regression_set: test_prefill_combine.py (all mesh configs) + test_dispatch_combine_l1_small_semaphores.py
- verification_command: `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_combine.py -k "predictable"` then `find generated -name 'reader_combine*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (mesh). BLOCKER: COUNTER pattern.
  NOTE: do NOT confuse with deepseek_prefill `extract`/`insert` ops (separate factories,
  separate kernels reader_extract/reader_insert) — those have single-`device` tests
  (test_deepseek_prefill_extract.py / _insert.py) but are OUT OF SCOPE here.

### ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm1.cpp
- role / tag: refactor — one_packet mcast collector → 7 cores
- factory: moe_gate_mm_program_factory.cpp:171 `dm1_kernel_handle = CreateKernel(... dm1.cpp ..., all_cores ...)` RISCV_0 / NOC_1. (selected when: always created for moe_gate_mm op).
- op/model: ttnn.experimental.deepseek.moe.moe_gate_mm.
- candidate_validation_set: tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_mm.py::test_moe_mm[...] — single `device`; param M,K,N,L,C = (32,7168,256,1,1) (only SHAPE2TIME key), device_params id "dispatch_row", check_accuracy_True.
- candidate_regression_set: test_moe_mm.py (full SHAPE2TIME × device_params); also exercised by deepseek_v3_b1 decoder/moe fused tests.
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_mm.py -k "check_accuracy_True"` then `find generated -name 'dm1*'`
- coverage_confidence: high (single `device`, kernel always created, op directly called at test_moe_mm.py:347)
- gaps: SINGLE-DEVICE OK. dm1.cpp basename is generic — confirm the right op's dm1 by checking the JIT dir is under the moe_gate_mm program hash.

### ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp
- role / tag: refactor, sender — INCLUDE_SRC loopback
- factory: rms_allgather_program_factory.cpp:796 `writer_mcast_sender_kernels_id = CreateKernel(... rms_writer.cpp ..., all_to_all_cores ...)`. (selected when: core ∈ all_to_all_cores; SKIP_WRITE_BACK define as above).
- op/model: ttnn.experimental rms_allgather.
- candidate_validation_set: multi-device only (same op as rms_sender_reader)
- candidate_regression_set: tests/ttnn/unit_tests/operations/ccl/test_minimals.py (rms cases)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/test_minimals.py -k "rms"` then `find generated -name 'rms_writer*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY. Coverage gap.

### ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_reader.cpp
- role / tag: refactor — value-carrying flag (token counts packed in sem word)
- factory: moe_gpt_program_factory.cpp:857 `tilize_reader_kernel_id = CreateKernel(... tilize_reader.cpp ..., tilize_core_range_set ...)` RISCV_1 / NOC_1 / DM_DYNAMIC_NOC. (selected when: tilize stage of moe_gpt op — created unconditionally in the tilize block; tilize_core_range_set non-empty).
- op/model: ttnn.experimental moe_gpt (gpt-oss MoE, fused dispatch+tilize+compute).
- candidate_validation_set: multi-device only — tests/ttnn/nightly/.../test_moe_gpt_e2e.py::test_moe_gpt_e2e (mesh 4x8) / test_dispatch_compute (4x8)
- candidate_regression_set: test_moe_gpt_e2e.py (e2e + dispatch_compute + dispatch_compute_combine + full_pipeline_multi_iter) + tests/nightly/tg/ccl/moe/test_moe_compute_6U.py
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py -k "dispatch_compute"` then `find generated -name 'tilize_reader*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (4x8 mesh). Coverage gap. basename `tilize_reader.cpp` is shared with moe_compute — disambiguate by JIT program hash / op dir.

### ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_writer.cpp
- role / tag: refactor — value-carrying flag
- factory: moe_gpt_program_factory.cpp:870 `tilize_writer_kernel_id = CreateKernel(... tilize_writer.cpp ...)`. (selected when: same tilize block as tilize_reader).
- op/model: ttnn.experimental moe_gpt.
- candidate_validation_set: multi-device only — test_moe_gpt_e2e.py::test_dispatch_compute (4x8)
- candidate_regression_set: test_moe_gpt_e2e.py suite
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py -k "dispatch_compute"` then `find generated -name 'tilize_writer*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY. Shared basename with moe_compute tilize_writer.

### ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp
- role / tag: refactor
- factory: moe_compute_program_factory.cpp:1042 `tilize_reader_kernel_id = CreateKernel(... tilize_reader.cpp ..., tilize_core_range_set ...)`. (selected when: moe_compute op tilize block — created in the compute path; reached even when compute_only=True).
- op/model: ttnn.experimental.moe_compute (deepseek_v3 / gpt-oss MoE compute).
- candidate_validation_set: SINGLE-CARD — tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py::test_moe_compute_single_card_gpt_oss (1x1 mesh, compute_only=True) — smallest; also _deepseek variant.
- candidate_regression_set: test_moe_compute_single_card.py (gpt_oss + deepseek) + tests/nightly/tg/ccl/moe/test_moe_compute_6U.py (TG)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py -k "single_card_gpt_oss"` then `find generated -name 'tilize_reader*'`
- coverage_confidence: med (1x1 mesh single-card test exists and runs the tilize path; must confirm tilize block is reached under compute_only=True — the tilize block is part of the compute path, not the bypassed combine)
- gaps: SINGLE-CARD reachable (good). Confirm tilize_core_range_set non-empty for the single-card grid; shared basename with moe_gpt — disambiguate by program hash.

### ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_writer.cpp
- role / tag: refactor
- factory: moe_compute_program_factory.cpp:1055 `tilize_writer_kernel_id = CreateKernel(... tilize_writer.cpp ...)`. (selected when: same tilize block).
- op/model: ttnn.experimental.moe_compute.
- candidate_validation_set: SINGLE-CARD — test_moe_compute_single_card.py::test_moe_compute_single_card_gpt_oss (1x1 mesh)
- candidate_regression_set: test_moe_compute_single_card.py + test_moe_compute_6U.py
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py -k "single_card_gpt_oss"` then `find generated -name 'tilize_writer*'`
- coverage_confidence: med
- gaps: SINGLE-CARD reachable. Shared basename with moe_gpt.

### ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp
- role / tag: refactor — flag-only sync
- factory: selective_reduce_combine_program_factory.cpp:448 `ternary_reader_kernel_id = CreateKernel(... reader.cpp ...)` RISCV_1. (selected when: selective_reduce_combine op program; runs as the combine stage of moe_compute when compute_only=False / fused, or standalone via ttnn.experimental.selective_reduce_combine).
- op/model: ttnn.experimental.selective_reduce_combine (MoE combine reduce).
- candidate_validation_set: multi-device only — tests/nightly/tg/ccl/moe/test_selective_combine_6U.py (galaxy 1x8/1x16). NOTE: test_moe_compute_single_card.py runs compute_only=True which BYPASSES selective_reduce_combine, so it does NOT cover this kernel.
- candidate_regression_set: test_selective_combine_6U.py + test_moe_gpt_e2e.py (dispatch_compute_combine / full_pipeline) + models/demos/deepseek_v3/tests/tg_moe_tests/individual_ops/test_combine_tg.py
- verification_command: `scripts/run_safe_pytest.sh tests/nightly/tg/ccl/moe/test_selective_combine_6U.py -k "combine"` then `find generated -name 'reader.cpp' | grep selective` (basename `reader.cpp` is extremely generic — MUST grep by program hash / source path in JIT dir)
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (galaxy/TG). Single-card moe_compute bypasses it via compute_only=True. Coverage gap. Generic basename `reader.cpp`.

### ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp
- role / tag: refactor — flag-only sync
- factory: selective_reduce_combine_program_factory.cpp:520 `unary_writer_kernel_id = CreateKernel(... writer.cpp ...)` RISCV_0.
- op/model: ttnn.experimental.selective_reduce_combine.
- candidate_validation_set: multi-device only — tests/nightly/tg/ccl/moe/test_selective_combine_6U.py
- candidate_regression_set: same as reader.cpp above
- verification_command: `scripts/run_safe_pytest.sh tests/nightly/tg/ccl/moe/test_selective_combine_6U.py -k "combine"` then grep JIT by program hash for writer.cpp
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY. compute_only single-card bypass. Generic basename.

### ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp
- role / tag: refactor — flag-only
- factory: all_gather_concat_program_factory.cpp:368 `worker_sender_writer_kernel_id = CreateKernel(... llama_all_gather_concat_writer.cpp ...)`. (selected when: always created in fused AG+concat-heads path).
- op/model: ttnn.experimental.all_gather_concat (llama fused all-gather + concat heads).
- candidate_validation_set: multi-device only — tests/ttnn/unit_tests/operations/ccl/fusion_subtests/concat_fuse_test.py (mesh_device) and test_ag_rs_llama_prefill_TG.py / test_ccl_async_TG_llama.py
- candidate_regression_set: test_ag_rs_llama_prefill_TG.py + test_ccl_async_TG_llama.py (TG)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py -k "concat"` then `find generated -name 'llama_all_gather_concat_writer*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (TG/mesh). Coverage gap.

### ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp
- role / tag: refactor — INCLUDE_SRC loopback intra-chip leg (fabric leg OOS)
- factory: all_to_all_async_generic_program_factory.cpp:254 `sender_writer_kernel_id = CreateKernel(... all_to_all_sender_writer.cpp ...)`. (selected when: always created for all_to_all_async_generic op).
- op/model: ttnn.experimental.all_to_all_async_generic.
- candidate_validation_set: multi-device only — tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_all_to_all.py::test_all_to_all (bh_1d_mesh_device; num_devices = mesh shape[0])
- candidate_regression_set: blackhole_CI box all_post_commit + nightly test_all_to_all.py
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_all_to_all.py -k "all_to_all"` then `find generated -name 'all_to_all_sender_writer*'`
- coverage_confidence: low
- gaps: MULTI-DEVICE ONLY (bh_1d_mesh). Only the intra-chip loopback leg is in scope; the fabric leg is OOS. Coverage gap for single-device validation.

---

## REF / OOS (not mapped deeply)

- models/demos/deepseek_v3_b1/unified_kernels/mcast.hpp — REF: existing two-sided Mcast prior-art (header, no own test; covered transitively wherever Mcast is used).
- models/demos/deepseek_v3_b1/unified_kernels/dataflow_utils.hpp — REF: primitive layer (header).
- models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp — REFACTOR (k-chunk data+flag mixed counter/flag): tested by models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla.py (single `device`); not mapped deeply here.

---

## COVERAGE SUMMARY

Single-device reachable (good for migration validation):
- kv_cache_update.hpp (med — header-include caveat)
- sampling_kernel.cpp (high)
- moe_gate_mm/dm1.cpp (high)
- moe_compute tilize_reader.cpp / tilize_writer.cpp (med — 1x1 mesh single-card test)

MULTI-DEVICE ONLY (coverage GAPS — no single-card validation path):
- rms_allgather: rms_sender_reader.cpp, rms_writer.cpp
- llama_all_gather_matmul_async: worker_receiver.cpp
- deepseek_prefill dispatch/combine: reader_dispatch.cpp, reader_combine.cpp
- moe_gpt: tilize_reader.cpp, tilize_writer.cpp
- selective_reduce_combine: reader.cpp, writer.cpp (single-card moe_compute BYPASSES via compute_only=True)
- all_gather_concat_heads_fused: llama_all_gather_concat_writer.cpp
- all_to_all_async_generic: all_to_all_sender_writer.cpp
