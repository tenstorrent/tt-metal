# B32 preparation allocation hypothesis

**Follow-up resolved the dominant owner:** `reshape_mapping_reconciliation.md` reproduces six major allocation histogram bins exactly from cached reshape segment-map tensors. The preliminary binary-ownership discussion below proves a real category but does not identify the dominant category; the reduced measurement rules out binaries as the main growth source.

Source-only AutoFix investigation, 2026-11-14. No device execution, production edit, new capacity claim, or artifact hashing. Evidence: `slots32_all64_pool.log`; current `tt/bounded_prefill.py`, `tt/model.py`, inherited decoder and native sources cited below.

## What the failure establishes

The all64 model loaded at 13,857,710,080 bytes/device. B1 preparation reached every length 1..512. Uniform B32 preparation printed through 160/512, then failed in `BoundedPrefill.prepare -> Model.terminal -> ttnn.matmul -> MatmulDeviceOperation::create_output_tensors`.

The requested output was 762,839,040 bytes/device, or 95,354,880 per bank. This equals `32 * 192 * 62080 * 2`: a BF16 local-vocabulary tensor with each user's sequence axis tile-padded to 192. The first logical length after the printed checkpoint is 161; that attribution is derived from the loop and size, not a separately printed length marker. The failure was allocation of the all-position LM-head output, before that matmul could run.

At failure allocated DRAM was `8 * 4,081,165,120 = 32,649,320,960` bytes/device, free `8 * 178,675,264 = 1,429,402,112`, but largest free block only 51,515,008 per bank. Total free capacity exceeded the requested output, while the necessary contiguous block did not. This is evidence of fragmentation plus a large retained preparation footprint. It does **not** establish a physical context limit. The mesh closed normally.

## Ownership proven by source

| Allocation family | Exact owner / creation point | Does it grow with new length? |
| --- | --- | --- |
| Real model weights/embedding/head/RoPE | `Model` and layer weight dictionaries | No; already in loaded baseline. |
| KV pool, published GDN, deduplicated reset sources | Generator/cache fields | No; fixed at generator construction. |
| FP32 live GDN worksets, BF16 convolution worksets, 512 ID/position banks, runtime page tables/indices | Each `BoundedPrefill` instance (`__init__`, lines 78–115) | No; all allocated before `prepare`. |
| Native compiled kernel binaries | `ProgramCache -> CachedMeshWorkload.workload -> MeshWorkloadImpl.kernel_bin_buf_` | **Yes, on each new cached workload.** Exact DRAM owner and allocation proven below. |
| Short persistent collective output/intermediate buffers | Each layer's `_ccl_buffers` dictionary | Yes while discovering distinct short signatures; B32 flat shapes above one token fail the persistence predicate. B1 discovered buffers are copied to all layers after B1's full loop. |
| Large nonpersistent AG/RS outputs and intermediate tensors | Operation return values, Python locals / expression temporaries | No retained per-length owner in the inspected native factories or Python layer dictionaries. |
| Diagnostic all-position LM-head output | Return value of `Model.terminal(result)` | No Python retention: preparation discards the return immediately. The failed output was never allocated. |
| 512 distinct synthetic attention-head outputs used for concat warmup | Local `concat_inputs` list in `prepare` | Fixed after its first creation (length 2), released on helper return; not a new 512-item list per length. |
| Current embeddings, RoPE, representative result, row/head outputs | `prepare` frame locals | Current/previous iteration overlap only; not a history list. |

### Native program binaries are a real, persistent DRAM category

`tt_metal/api/tt-metalium/program_cache.hpp:67` makes the cached workload own a `MeshWorkload`. `tt_metal/distributed/mesh_workload_impl.hpp:77` gives that workload a strong `shared_ptr<MeshBuffer> kernel_bin_buf_`. `tt_metal/distributed/mesh_workload.cpp:170–249` allocates one replicated, top-down DRAM buffer on the workload's first enqueue; its size is the maximum `program_transfer_info.binary_data.size()*4` across the mesh's programs. It then installs non-owning/address views into each program. A cache hit reuses that owner. Clearing Python tensor locals cannot release it.

The framework allocates this buffer separately for each cached workload. It does not share the DRAM buffer merely because host JIT compilation reused the same kernel source/binary. The log's high JIT cache-hit count therefore does not bound this device-memory category. The number and byte sizes of these owners in this failed process were not recorded, so attributing *all* unexplained bytes to binaries would exceed the evidence.

`WorkloadDescriptor` can also retain typed buffer owners and semaphores (`tt_metal/api/tt-metalium/workload_descriptor.hpp:16–58`). For the relevant inspected paths:

- `experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:1958–2010` constructs the Line descriptor from input/intermediate/output references and retains program descriptors only; it does not park the large tensors in `workload.buffers`.
- `experimental/ccl/all_gather_async/device/all_gather_async_llama_sharded_program_factory.cpp:349–361` likewise retains descriptors, not its temporary `vector<Tensor>` used to derive operation configuration. The broadcast variant (`all_gather_via_broadcast_factory.cpp:294–306`) retains two workload-scoped semaphores, not large DRAM output tensors.
- The accepted fused prefill projection factory, `experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_program_factory.cpp:916–954`, retains runtime-argument references and updates A/W/output addresses on hits. Its mesh-global mailbox and semaphores (`:957–1012`) are allocated once per mesh in L1_SMALL/L1; they are not sequence-sized DRAM histories.
- The ordinary terminal matmul factory's shared variables contain kernel/CB handles, cores and configuration (`operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp:15–24`), not an owning logits tensor. This rules out the proposed Python/native matmul-output retention mechanism in that path; it does not replace a native allocation inventory for every dispatched factory.

### Python collective retention is bounded differently from program-cache retention

`MultichipDecoder._collective` (`tt/multichip_decoder.py:448–517`) reshapes flat collectives to M=B*S and uses persistent buffers only when M<=32. Dictionary keys include the exact logical shape, dtype, memory configuration and reduction flag. Thus B1 lengths 1..32 can retain physically tile-padded copies under 32 distinct logical keys. `_prepare_layer_buffers` (`bounded_prefill.py:278–297`) allocates equivalent owners on every layer, preserving existing per-layer ownership. B1 completed that step before B32 started.

For uniform B32, the main flat-prefill path at S>1 has M>32 and does not add these large persistent dictionary outputs. The unaligned-head preparation always executes S=1, M=32, so those same keys repeat. Final norm gathers similarly flatten B*S. B32's all-layer `_prepare_layer_buffers` call is after its entire length loop and had not run at this failure. Consequently ongoing large B32 `_ccl_buffers` growth on every new length is refuted for these paths. An actual byte inventory must still include all B1 short keys copied across the 64 layers; the old one-signature capacity estimate is insufficient.

## Diagnostic peaks and numerical size checks

The synthetic concat list holds 512 distinct BF16 `[32,1,1280]` TILE outputs. Each has padded shape `[32,32,1280]`, costing 2,621,440 bytes; the complete list is **1,342,177,280 bytes/device**. It remains alive while every all-position terminal call executes. This is avoidable warmup overlap, not an ongoing leak. The single-batch helper's corresponding list is 41,943,040 bytes.

At B32/S512, the all-position terminal output alone is **2,034,237,440 bytes/device**. At S161 it is the observed 762,839,040. These outputs are discarded after each warmup call, but their allocation requires large contiguous holes among persistent program buffers. Final normalization's gathered hidden and current/previous iteration tensors add smaller temporary peaks. Merely assigning and then deleting the discarded terminal return cannot cure persistent growth.

The failed allocation is 18,791,610,880 bytes/device above the loaded-model baseline. Known source categories include the 8224-page KV pool/reset (~2.36 GB), fixed B32 state/helper storage (~2.01 GB), common sampler arrays, the 1.34 GB concat list, copied B1 short CCL buffers, all cached program binaries, and current live temporaries. The existing log does not provide enough observations to split the remainder exactly. No unsupported residual-byte attribution is made here.

## Smallest useful next measurement

Use one isolated **diagnostic** run with real layers `[0,1,3]`, B32, the same 8224 physical pages and virtual `[32,8192]` tables. Three layers cover the I/O GDN, ordinary GDN, and full-attention representative signatures. Keep all accepted precision, projection, CCL and cache policies. No full64 reload is needed to determine representative-program growth; all64 per-layer CCL replication is counted separately.

At generator construction, after B1 preparation, and around selected B32 lengths, record:

1. Synchronized DRAM allocated/free/largest-free, and `num_program_cache_entries`.
2. Live buffer address/size/type snapshots from `ttnn._ttnn.reports.get_buffers(mesh)` (bound in `ttnn/cpp/ttnn-nanobind/reports.cpp:28–36`). Deduplicate `(device_id,address)` views; report one device, not four-device sums.
3. For each layer, `_ccl_buffers` key count and unique tensor buffer addresses/allocated padded bytes. Record concat-list allocated bytes separately.
4. Phase boundaries: after representative decoder work, after concat, after all-position terminal, after last-position terminal, and after temporary references leave a scope. Do not retain readback/logit tensors in instrumentation.

A small `[1,2,31,32,33,64,128,160,161,512]` diagnostic repertoire establishes per-operation slopes and peaks; it does not establish full-repertoire fit. A dense 1..161 run gives the relevant accumulation. Repeat a selected length immediately: unchanged persistent bytes and cache count distinguish cache-miss growth from a per-call leak.

For attribution only, after the diagnostic helper returns and **before capturing any trace**, snapshot memory, call `clear_program_cache`, synchronize, and snapshot again. The released amount isolates cached workload-owned storage (including binaries/resources), while `_ccl_buffers`, KV, weights and helper fields remain live. This is a disposable diagnostic experiment, **never** a production fix and never allowed after capture. If exact per-operation binary attribution is needed, temporarily log workload binary allocation size at `MeshWorkloadImpl::load_binaries`; native trace-allocation tracking alone does not provide a pre-capture allocation census.

## Repairs ordered by evidence and scope

1. **Minimal, policy-preserving warmup peak repair:** run distinct-buffer concat preparation as a separate phase, release its 512 clones before decoder/terminal shape warmup, and release loop temporaries at clear scope boundaries. All concat arities 1..512 remain prepared, and real runtime keeps the identical optimized decoder, sequence repertoire and outputs. This removes 1.34 GB of artificial overlap at B32. It may postpone rather than solve the failure if cached binaries dominate; no fit claim is earned without measurement. Allocating clones incrementally only saves early-loop memory and still reaches the same maximum, so separate phases are preferable.
2. **If cached workload binaries dominate:** preserve logical/program signatures but deduplicate identical immutable binary payload allocations per mesh, or repair overly specific *native op* cache keys where logical differences are already runtime arguments and generated programs are proven identical. Either needs native ownership/address/capture tests; neither may drop required compile-time distinctions. The source establishes why this can recover real DRAM without changing math or TP strategy, but does not establish a ready-made safe small patch. Do not disable cache, evict programs after capture, or call the preparation list complete with missing signatures.
3. **If copied B1 short CCL owners materially dominate:** first consider model-lifetime canonical short scratch by exact physical specification while keeping native call geometry unchanged. Current implementation deliberately assigns separate ownership to every layer; sharing is not a drop-in deletion. It requires a single-queue lifetime/dependency proof across captured traces and all within-layer uses, stable retained buffers, and trace/accuracy rereview. No such change is recommended without that evidence.
4. **Terminal-only diagnostic peak:** all-position logits can potentially be produced in a bounded device block schedule, preserving output ordering and BF16/HiFi4. This would change the diagnostic head execution contract and needs numerical and trace validation. It must preserve requested all-position support and must not replace normal token-out sampling with host logits. Simply omitting full-logit warmup leaves the postcapture all-logits path unsafe and is not a repair.

Do not replace uniform B32 optimized prefill with B1 row loops, shrink the advertised 262144 context, change FP32 live GDN to BF16, or replace 512 chunks with short recurrent steps to make preparation fit. Powers-of-two decomposition is not a free signature reduction: subsequent partial chunks become unaligned under the accepted 512-token cache/SDPA geometry, changing the selected fast path. Native SDPA validates start alignment against q/k chunk geometry (`operations/transformer/sdpa/device/sdpa_device_operation.cpp:389–401`).

**Conclusion:** persistent native program-binary growth is proven by ownership; a large fixed synthetic concat peak and all-position-head peak are proven by shape; repeated retention of every head output and unbounded large Python CCL outputs is unsupported/refuted on the inspected paths. The present failure is preparation memory/fragmentation evidence, not evidence that the requested context or 8224-page physical pool intrinsically cannot fit. Measure ownership before choosing a broader repair.
