# Attention DRAM-sharded decode candidates

This is CPU/source preparation for the remaining dense attention DRAM-sharding advice in [optimization_advice_audit.md](optimization_advice_audit.md). Neither patch has been applied and no hardware has been used for these candidates. Source compatibility and CPU adapter checks do not establish numerical correctness or performance. The earlier shared-MLP DRAM loss does not reject these different projection roles.

## Artifacts and scope

- [attention_dram.patch](attention_dram.patch): optional `attention_dram=None`, `"qkv"` or `"output"`, with one setup-only bank-sharded weight copy for the chosen role.
- [attention_dram_runner.patch](attention_dram_runner.patch): `--attention-dram {qkv,output}`, factory forwarding, result metadata, incompatible-QKV/AGMM guard and added-copy capacity accounting.
- [attention_dram_provenance.json](attention_dram_provenance.json): exact base/candidate hashes and candidate snapshots.
- [attention_dram_cpu_checks.json](attention_dram_cpu_checks.json) and [CPU check source](attention_dram_cpu_checks.py.txt): actual model geometry, extra bytes, prefill branch preservation and forwarding of live compute-policy overrides.
- Complete proposed files: [runtime candidate](attention_dram_candidate.py.txt), [runner candidate](attention_dram_runner_candidate.py.txt).

The base runtime is `20151de9802f92af989cdad8f27cb764c79c8b8fad6c80939f5392edb8575cd4`; the candidate is `49cd4b3e47aeca24389fd58c29262556d05ec35cf0f0d80c8077c02af51f88e5`. The runner base is `beabf408847488fa8f9daaadc04c913c5b6bd1de681edfcbc5a3fb565b6128ca`, containing the parent's separate QKV/WO fidelity overrides. Candidate runner is `acf7c7991775248983d779bc6e782d6f803516528c998213d01aeb4a05138876`. Both patches pass `git apply --check` against those bases; recheck after concurrent edits.

The helper uses `ttnn.to_memory_config` on the existing BFP8 matrix during construction. No Torch repacking, host readback, re-upload or weight typecast is added. The original matrix remains available to the cached prefill projection. The four local N dimensions already divide eight banks times 32 columns, so these candidates need no extra weight padding or output cropping. Weight mesh ownership and packed Q/K/V order are preserved by the layout conversion.

`qkv` changes only `_Projection`'s logical single-token branch. Its existing `MinimalPrefillProjection` remains the first branch for multirow inputs. `output` changes only `_LocalAttention.project(..., decode=True)`: it reproduces the baseline one-core head-concat setup and logical/physical reshape, reshares the resulting tiled BF16 tensor directly to the working width-sharded input, performs the DRAM matmul, and restores `self.output_memory` before the original reduction. This avoids adding an intermediate interleaved copy between head concat and the working shard. Prefill continues through the existing superclass path.

The modes are separate candidates, not an implicit QKV+WO combination. QKV DRAM and fused AGMM are explicitly rejected together because AGMM otherwise bypasses `_Projection.__call__`, leaving a silently unused DRAM copy. No default, shared-MLP, active-expert, router, cache, collective placement, residual layout or norm policy is changed.

## Geometry and precision

All four use eight L1 storage cores in row-major `(8,1)`, eight DRAM banks in `(8,1)`, one worker per bank, tiled layout and `per_core_M=1`. Logical decode M1 retains physical M32; no tiny-tile path is introduced.

| Role / kind | Local K,N | DRAM shard | L1 input shard | L1 output shard | K block, tiles | Program N, tiles |
| --- | --- | --- | --- | --- | ---: | ---: |
| QKV sliding | 2816,2048 | 2816,256 | 32,352 | 32,256 | 11 | 8 |
| QKV full | 2816,3072 | 2816,384 | 32,352 | 32,384 | 11 | 12 |
| WO sliding | 1024,2816 | 1024,352 | 32,128 | 32,352 | 4 | 11 |
| WO full | 2048,2816 | 2048,352 | 32,256 | 32,352 | 8 | 11 |

QKV retains FP32 normalized input, BFP8 weight, FP32 output and FP32 destination accumulation. WO retains BF16 SDPA/head-concat input, BFP8 weight, FP32 output and FP32 destination accumulation. The helper takes `compute` as a call argument from `projection.decode_compute` or `attention.output_compute`; it does not cache an earlier config. Consequently the parent's post-construction `--qkv-fidelity` and `--output-fidelity` selections, approximate-math setting, packer setting and destination-sync setting reach the candidate unchanged. With no overrides, the current source uses HiFi2 sliding/LoFi full for QKV and HiFi4 for WO. The measured candidate must use the same selected overrides as its control.

All four have eight K blocks, so requested packer accumulation is not disabled by a one-block special case. These projections currently request packer accumulation off. Changed K blocking and the different kernel can still alter rounding even when every requested dtype/fidelity field is preserved.

An optional later WO variant can use four storage cores, input shards `[32,256]`/`[32,512]`, output shard `[32,704]`, program N22 and K blocks 8/16 for sliding/full. It retains eight DRAM reader workers. That variant is source-legal by the same divisibility rules but is not exposed by this patch; measure the explicit eight-core candidate first.

## Source support

Paths below are relative to `ttnn/cpp/ttnn/operations/`:

- `matmul/device/matmul_device_operation.cpp:1302-1375` validates the DRAM-sharded path: one physical M tile; width-sharded row-major activation; sharded output with matching buffer/layout; width-sharded B; global K and local activation-shard K divisible by the K block. QKV divides 88/11 and 11/11; sliding WO divides 32/4 and 4/4; full WO divides 64/8 and 8/8. The code rejects tile heights below 16, which these ordinary 32-high tiles avoid. `:1216-1230` permits the different input/output shard widths because it compares buffer type and memory layout.
- `matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:124-167` derives bank reader workers and per-worker N separately from storage grids. These cases need no padding-only banks. `:170-209` selects FP32-safe destination subblocks and FP32 intermediates; `:386-387` emits `FP32_DEST_ACC_EN`; `:525` passes it to the compute descriptor. `:950-969` derives A/B/output formats from their actual tensor dtypes, and `:1040-1058` forwards the selected compute config. There is no BF16-only A restriction in this program's validator. This is explicit source support for QKV's FP32 input/output/accumulation, not a measured result.
- `data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:13-114` validates standard tiled transfers without a BF16-only dtype whitelist. Its low-precision restriction is that BFP8/BFP4 require tiled layout; the DRAM restriction applies to legacy BLOCK_SHARDED, not WIDTH_SHARDED. The associated `interleaved_to_sharded_program_factory.cpp:78-100` derives tile sizes/data formats and both source/destination DRAM flags from tensors; `:221` and `:303` handle a DRAM destination. Thus the setup BFP8 bank copy and runtime FP32 activation sharding are supported by source.
- `core/to_memory_config/to_memory_config_op.cpp` preserves dtype when none is requested, selects the supported sharding/resharding operation, and routes oversized transfer buffers through its general copy fallback. It rejects legacy DRAM block sharding; this proposal uses width sharding. Its explicit no-op check cannot alias the interleaved original to the different bank-sharded destination config.
- `data_movement/sharded/reshard/device/reshard_device_operation.cpp:22-109` selects L1-to-L1 tiled generic reshard for the WO head-concat tensor. Its `validate_inputs` at `:125` checks device allocation, shard/layout constraints and matching dtype for a preallocated destination; it does not prohibit FP32. This candidate requests no dtype conversion. The row-major width-alignment special case does not apply to tiled inputs.
- `models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py:1344-1367` is the existing decode head-concat/project sequence copied into the optional WO branch. The head count, head dimension, logical reshape and reduction boundary remain the same.

The DRAM factory can allocate primary worker CBs of approximately 90,112 bytes A plus 287,232/430,848 bytes B for sliding/full QKV, or 16,384/32,768 bytes A plus 143,616/287,232 bytes B for sliding/full WO. These use actual FP32 or BF16 A tile bytes, BFP8 B tile bytes, A double buffering and B triple buffering. They exclude activation storage, output/intermediate CBs, other live tensors and kernel/runtime overhead; they are not an allocator-peak proof. Setup conversion of the largest QKV bank shard covers 1,148,928 BFP8 bytes; `to_memory_config`'s source-level capacity fallback is relevant if its temporary copy buffer cannot fit.

## Added weight residency

The original prefill matrix and the new bank copy coexist. There is no weight padding, but the entire BFP8 matrix is additional persistent device storage. Payload counts use 1088 bytes per 32x32 BFP8 tile, which already satisfies 64-byte DRAM alignment.

| Role | Extra sliding layer bytes | Extra full layer bytes | Extra bytes/device for 25 sliding + 5 full |
| --- | ---: | ---: | ---: |
| QKV | 6,127,616 | 9,191,424 | 199,147,520 |
| WO | 3,063,808 | 6,127,616 | 107,233,280 |

The runner derives these dimensions from the actual config, asserts the current-layer amount against the helper's byte count, adds all layer copies to resident reservations, and adds the current-layer copy to its weight bound. Existing shared/hybrid reservations remain intact. Allocation metadata, setup temporary buffers and fragmentation remain covered only by the existing independent reserve, not by these payload figures. If a candidate is selected, the parent must update the persistent-memory/context contract and rerun the relevant capacity gates; this preparation does not claim that gate passed.

## Completed checks and parent-owned experiment

Both complete candidate Python files parse with `ast.parse`; inserted classes were Black-formatted without reformatting live source. Both patches pass applicability checks. The CPU check imports no TTNN and executes only AST-extracted adapter classes against a small stand-in API. It verifies the four real geometries, divisibility, extra-copy accounting, preserved input dtype, unchanged prefill branch selection, and that replacing the compute config after setup reaches both candidate roles. It does not execute a TT kernel or compare numerical results.

After applying/rebasing, compare `--attention-dram qkv` and `--attention-dram output` separately against the best passing grouped-reduction/geometry 1 control with identical selected projection-fidelity flags. Keep hybrid experts, fused tail, replicated residuals, Linear topology, actual real weights and the paired 4096/128 trace/cache harness fixed. Exercise layers 0 and 5 for each role. A first API validation error warrants a source-backed contract adaptation and retry, not a blanket rejection of DRAM advice. Reset/list/mesh-smoke after a failed multichip run; collect triage before killing a hang.

Compare complete-layer traced host medians, including activation conversion, head concat, output conversion and unchanged reduction. Native profile evidence should include those transfers as well as the matmul. Preserve cache ownership, exact replica/replay and no-host-fallback checks. Only a passing, faster whole-layer candidate proceeds to stack, batch, maximum-context and final-profile/Watcher gates. This artifact does not declare the overall path accepted.
