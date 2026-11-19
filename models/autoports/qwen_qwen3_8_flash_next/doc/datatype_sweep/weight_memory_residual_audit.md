# Persistent-weight versus constructor-memory residual audit

CPU-only source/provenance audit, 2026-11-16. No TTNN import, device use or runtime edits. Inputs: `weight_memory_recheck_v1.json`, `inherited_e4_lofi_v3.json`, `kv8_v3.json`, `head8_lofi_v3.json`, `head8_hifi2_v3.json` and corresponding provenance. The recheck's 11 exact matches remain exact; its three nonzero residuals do not demonstrate incorrect weight-byte arithmetic.

`memory_loaded` is **total allocated DRAM after `build_generator`**, before `prepare`. It includes model tensors, full private KV and other generator state, plus any resident constructor helper/program allocations. `run_full_model_accuracy.memory` reads `get_memory_view(DRAM)` without tagging allocation owners. It is not a weights-only measurement. `static_weight_delta` intentionally models persistent tiled weights only.

## Recomputed observations

All values below are bytes per rank; allocator fields are per bank and were multiplied by eight.

| Candidate | Predicted weight delta | Predicted full-context KV delta | Observed loaded delta | Loaded residual | Prepared residual | Final residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kv8_v3` | 0 | -1509949440 | -1509933056 | +16384 | +32768 | +32768 |
| `head8_lofi_v3` | -149299200 | 0 | -149364736 | -65536 | -65536 | -65536 |
| `head8_hifi2_v3` | -149299200 | 0 | -149364736 | -65536 | -65536 | -65536 |

Prepared/final residuals use the corresponding inherited phase as the comparison, with the same predicted persistent weight+KV delta. Both head fidelities have identical allocated bytes in all three phases. KV8 adds another 16384 bytes relative to inherited during `prepare`; its later measured growth then equals inherited. Therefore these small differences are already established during construction/preparation and are not decode throughput or head fidelity effects.

The recorded autoport `tt/` source hashes are identical between inherited and all three candidates; environment maps and git HEAD are identical. Differing hashes are test/analysis harness files (including the later trace-inventory diagnostics), not model runtime. This rules out an autoport-runtime source mismatch in the recorded provenance, but does not independently fingerprint every shared TTNN C++ binary. Candidate capacities are all 262144 with one slot, and no nonzero reserved trace region is requested. The absent baseline `trace_region_size` field comes from the older harness schema; its command has no override and the default is zero.

## The persistent byte calculations are correct

The head's recorded actual shape is `[1,1,2560,62080]` in both dtypes. `tt/dram_projection.py:_PreparedGroup` bank-shards it across eight DRAM banks, with per-bank padded columns `ceil(62080/(32*8))*32 = 7776`. Thus it has `(2560/32)*(7776/32)*8 = 155520` physical tiles per rank. BF16 tiles cost 2048 bytes; BFP8 tiles cost 1088 bytes. The exact persistent change is `155520*(1088-2048) = -149299200`. It is neither -149364736 nor a logical-shape-only estimate. Decode/rows share the prepared weights; there is no duplicate head copy to remove in the candidate.

The private KV stack has 12 QSA layers, separate K and V, 8192 pages per tensor at context 262144, and eight 32×32 tiles per page. Its exact delta is `12*2*8192*8*(1088-2048) = -1509949440`. All shapes are tile/page aligned; raw history and compressed keys remain BF16. No padding-rounding change in this tensor geometry explains +16384.

## Source-backed explanation and attribution limit

There are concrete dtype-dependent constructor program paths:

- `ttnn/core/tensor/py_to_tt_tensor.cpp:create_tt_tensor_from_host_data` disables device conversion for BFP8/BFP4 unless `enable_bfloat_opt=true`. The model does not enable it. BFP8 head packing therefore follows the host-layout path. BF16 mesh uploads can instead construct in row-major and perform device tilize when geometry/memory checks permit; `convert_python_tensor_to_tt_tensor` calls `to_layout` if needed. This can retain a BF16 upload/tilize program that the BFP8 head never needs. It is a credible source for the head's **smaller constructor overhead**, not a change to head tensor bytes.
- `ttnn/cpp/ttnn/operations/creation/creation.cpp:full_impl` allocates sufficiently large tiled BF16 zeros on device and fills them with `ttnn::fill`. Nonempty BFP8 zero fills also use the device path. The full-context KV tensors qualify in both cases, but BFP8 introduces a different dtype specialization. During preparation, `tt/generator.py:_reset_cache` additionally executes in-place `multiply(...,0)` on actual K/V dtype. These are concrete opportunities for distinct BFP8 constructor/reset program storage.
- `tt_metal/impl/program/program.cpp:ProgramImpl::allocate_kernel_bin_buf_on_device` allocates persistent kernel binary buffers in DRAM. Their page size is 2048 bytes (`tt_metal/impl/dispatch/device_command.hpp:PROGRAM_PAGE_SIZE`). Eight-bank allocator reservation has a 16384-byte rank quantum for such pages. The residuals are exactly +1, +2 or -4 of these quanta, consistent with program-buffer differences.

**The exact allocation owner and compiled binary sizes are not in these artifacts.** No program-cache inventory, allocation-owner trace or constructor-by-constructor memory checkpoints were captured. The current evidence supports “small dtype-dependent constructor/preparation overhead, consistent with cached helper-program storage”; it does **not** prove “the missing head tilize binary is exactly 65536 bytes” or “this particular KV fill binary owns exactly 16384 bytes.” Do not label those hypotheses as measured attribution. Largest-contiguous-free differences describe fragmentation and likewise cannot identify the owner or replace allocated-byte accounting.

The shared source's conversion/fill paths explain why equality between persistent tensor deltas and whole-generator allocator deltas is not an invariant. Retain the residual column instead of modifying correct tile counts to force equality. An exact owner investigation, if later needed, would require a separate controlled device run with allocation/binary inventory around head upload, KV zeros and cache reset; it is unnecessary to assert that attribution now and was not run during the parent's active GR trial.

## Public cache-budget consequences

The default planner charges `KNOWN_STATIC_BYTES + static_weight_delta(selected_policy)`, exact selected KV page cost, explicit non-KV state, and a separate runtime reserve. At B1 the reserve is 2147483648 bytes. These residuals must stay outside static weight arithmetic:

- Head8's -65536 residual is **conservative** for admission: the planner claims 65536 fewer bytes of savings than the observed whole-constructor allocation. It cannot over-admit from this discrepancy.
- KV8's +16384 loaded/+32768 prepared residual is additional runtime overhead. It is small relative to the separate reserve; do not call the weight+KV-only model an exact whole-runtime forecast. KV8 remains independently rejected by its full-model PCC capability gate, so it is not currently a selectable policy.
- At context 262144/B1, baseline/KV8 `known_resident_before_pool_bytes=28610184192`. Observed loaded non-KV allocation is actually **3284480 bytes below** that listed amount for baseline and **3268096 below** for KV8. Head8's corresponding listed amount is 28460884992, with actual loaded non-KV **3350016 below** it. The listed ledger is not a perfect constructor allocation inventory, even before runtime reserve.
- Measured loaded→final growth is 685991424 bytes for baseline and both head variants; KV8 is 686007808. These particular workloads remain well below the 2 GiB B1 runtime reserve. This is supporting evidence for the current allowance, not validation of every possible retained trace or shared-pool shape.

Pure-CPU planner arithmetic yields full-stack pool blocks/remainders of 8700/83968 bytes (inherited BF16), 16376/182272 (KV8) and 9079/354304 (head8 BF16). Even hypothetically charging the additional 32768 KV8 overhead separately would not reduce its integer block count at this B1/full-context point because the remainder exceeds it. This does not establish the same claim at every batch/context admission boundary; reserve coverage and actual selected-config capability checks remain necessary.

No evidence here requires changing `static_weight_delta`, reducing advertised context, or altering the public budget for a selected weights-only policy. The 11 exact coarse weight/fidelity/activation/CCL deltas and both exact expert-upgrade deltas support current persistent arithmetic. A head8 selection would be conservatively budgeted by this residual; other selections still need their own measured full-context/trace acceptance and selected-context smoke. No winner or vLLM capability is inferred by this audit.
