# AutoDebug: TP4 B32/S33 prefill

## Leading finding

**The failing `e95f…` TP4 revision imports a CCL manager that initializes global semaphores on only an 8×8 core set, while its collectives select workers from the device's full 11×10 grid. Some selected workers therefore read semaphore locations that setup never initialized.** This is a confirmed source contract mismatch and the strongest causal hypothesis for the intermittent prefill failure. Hardware evidence has not yet established that it caused the reported NaN. A concurrent setup repair appeared during report review; its revision and passing artifacts are recorded separately below.

The specific hypothesis is that the first attention or MLP reduce-scatter consumes an unwritten intermediate page after an uninitialized ready semaphore prematurely satisfies its wait. That happens after the conv/recurrent state updates, explaining how output can fail while those states remain accurate. The end-of-kernel semaphore resets can also make subsequent decode safer than first prefill, subject to worker-set changes discussed below. This is a synchronization/initialization defect, not a reason to suppress nonfinite values.

**Important limit:** this defect also exposes some passing configurations. It predicts a first-use and memory-content-sensitive failure, not a deterministic rule that every B32 prefill fails and every smaller batch is safe. The actual number of fabric links and initial contents of the uncovered semaphore cells were not recorded in the failing artifact.

## Scope and evidence

Inspection only: no TTNN/model imports, device operations, resets, target execution, or source edits. Only this report was written. Three independent investigators audited the GDN adapter/output path, GDN chunk kernels, and collectives. Findings below were revisited against the code after the initial investigation.

Paths below are relative to `/home/ttuser/qwen36-27b/tt-metal` unless stated otherwise. Abbreviations:

- **M:** `models/autoports/qwen_qwen3_6_27b/tt/multichip_decoder.py`.
- **O:** `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py`.
- **D:** `models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/`.
- **RS:** `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/`.
- **AG:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/`.
- **GDN:** `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/`.

Source hashes at the start of inspection match the failing JSON. **All M line references below refer to this original revision**, not the concurrently repaired file:

| Source | SHA256 |
|---|---|
| M | `e95f63f2477d5d38b5876ba0440d3a88abd8d5d56a2643816703035e415e69ac` |
| O | `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22` |

Direct artifact observations, separated from interpretations:

| Artifact under D | Observed result |
|---|---|
| `correctness_v1_sharded_l0_b32_s33.json` | Prefill/output-contribution PCC NaN; decode PCC 0.9999604811; prefill conv/recurrent PCC 0.9999953852/0.9999794606; decode trace output/state bit equal to eager. |
| `correctness_v1_sharded_l0_b2_s33.json` | Prefill PCC 0.9999782418, decode 0.9999901238; passes. |
| `diagnose_b32.json` and `.log` | Same M/O and input hashes; prefill PCC 0.9999744170; finite logical values at the instrumented chunk output, concat output, projections, collectives, and final output. Decode/state PCCs match the failing report. |
| `b32_repro.json` | Same M/O and input hashes; prefill PCC 0.9999744170 and trace checks pass. Thus even an existing rerun without component instrumentation passes. |
| `topology_probe.log:22` | `MESH_OPEN [0, 1, 2, 3] 11-10 8-1`: compute grid 11×10, DRAM grid 8×1. This is an earlier topology observation, not a failing-run dump of active workers. |

B1/S33 passing is also requester-provided evidence; the B2 comparison above is directly available in the named correctness artifacts.

The original failing log/JSON records **PCC NaN**, not a finite-count/coordinate dump of the TP4 tensor. The helper is `torch.corrcoef` on flattened double tensors (`tests/run_decoder.py:42–43`). Independently check both reference and actual finiteness in the next failing occurrence; the original metric alone does not distinguish NaN/Inf data from another degenerate correlation input. No bad-index correspondence is available to infer a rank, head, or token pattern.

The current runner has additional diagnosis code and shifted line numbers compared with the failing traceback. Its diagnosis wrappers must not be treated as having executed in the original failure. The focus-source hashes and recorded tensor/input metadata are consistent.

### Concurrent repair and newly available artifacts

During final review, M changed to SHA256 `270ec411e158954f5ea796eaf0f5b4090876721a065683a77789acbef526cebc`. Static, in-memory reconstruction proved that removing only the new `_BlackholeCCLManager` class and restoring `self.ccl = CCLManager(...)` recovers **exactly** the original `e95f…` hash. The added class derives semaphore coverage from `mesh_device.compute_with_storage_grid_size()`; current M:28–36 implements it and current M:96 selects it. Original M references after that addition have shifted by +11 lines. O remains unchanged at `315187…`. This investigator made none of those source changes.

Owner-produced `watcher_b32_full_semaphores.json` and `watcher_cold_b32_v2.json` both identify `270e…`, record `finite: true`, prefill PCC 0.9999744170, and passing decode/state/trace checks. The cold log records `0/476` JIT cache hits. These artifacts support the proposed setup intervention and show a passing first-forward check without a populated JIT cache. They do **not** measure the original uncovered-cell contents or establish the first bad producer in the original revision. Unchanged-source B32 reruns already passed, so two repaired passes alone do not prove causality. No device run was performed by this investigator.

## Confirmed CCL initialization mismatch

### Caller setup versus worker selection

1. M:85 constructs `CCLManager(mesh_device, ttnn.get_num_links(mesh_device, 1), Topology.Linear)`.
2. `models/demos/gpt_oss/tt/ccl.py:29–40` hardcodes `CoreCoord(8,8)` and constructs a local `SubDevice`, but never creates/loads a subdevice manager. Constructing that descriptor does not restrict the mesh's active workers. Lines 42–58 create all six RS, four AG, and two barrier semaphore handles on that 8×8 set.
3. M:249–269 passes raw semaphore handles, `dim=3`, `cluster_axis=1`, topology Linear, and no worker count or subdevice restriction. Moving a tensor to `L1_MEMORY_CONFIG` here means interleaved L1, not an 8×8 worker restriction.
4. The RS public wrapper resolves defaults and selects the Linear factory (`RS/reduce_scatter_minimal_async.cpp:43–79`; `RS/device/reduce_scatter_minimal_async_op_device_operation.cpp:47–54`). The Linear factory chooses worker counts from the physical buffer size and then calls `choose_worker_cores` (`RS/device/reduce_scatter_minimal_async_program.cpp:1264–1277,1315–1345`).
5. `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:403–456` selects from `mesh_device.worker_cores(default_subdevice)` and fills rows across the actual range. A count below 64 does **not** imply containment in `(0,0)..(7,7)`: on an 11-column grid, the ninth selected core is `(8,0)`.
6. AG's default factory independently selects from the active worker grid too (`AG/device/all_gather_async_default_program_factory.cpp:42–87,149–164,221–222`). Its core set is not constrained by the semaphore's construction grid.

Sibling preparation supports this contract: `tests/ttnn/unit_tests/operations/ccl/test_reduce_scatter_fp32_precision.py:60–71,89` derives semaphore coverage from the actual device grid and explicitly loads its subdevice manager. The TP4 path duplicates neither guarantee.

### Why this is a real initialization bug

`tt_metal/impl/buffers/global_semaphore.cpp:84–107` creates a height-sharded buffer using exactly the supplied `cores_`. Its initial-value/reset operation writes `cores_.num_cores()` words through that buffer, not every Tensix core (`:60–78`). A scalar address returned by `GlobalSemaphore.address()` is not evidence that every other core has received the initial zero at that address.

The Linear RS factory passes `semaphore[0].address()` and the barrier address as common runtime arguments (`RS/device/reduce_scatter_minimal_async_program.cpp:445–460`). Worker physical coordinates are supplied separately (`:1658–1670,1737–1743`). Its kernel consumes those locations before clearing them:

| Phase | Actual state transition |
|---|---|
| Setup | Zero only the declared 8×8 semaphore instances. |
| RS startup writer | Increment the neighbor's barrier; wait for local barrier ≥1; **then** set local barrier to 0 (`line_reduce_scatter_minimal_async_writer.cpp:229–252`). There is no initial local clear before this wait. |
| RS incoming data reader | Wait for increasing `out_ready_sem` targets; then load the intermediate DRAM page (`line_reduce_scatter_minimal_async_reader.cpp:262–285,397–420`). |
| RS sender | Send intermediate data, then send ready credits (`line_reduce_scatter_minimal_async_writer.cpp:353–384`). Correct ordering cannot compensate for credits that were already present. |
| RS completion reader | Set local ready semaphore to zero **at the end** (`line_reduce_scatter_minimal_async_reader.cpp:429–430`). This cannot initialize the first wait retroactively. |
| AG | Likewise waits on the barrier before clearing it (`minimal_default_writer.cpp:306–307`); ready consumers wait before subtracting consumed credits (`minimal_default_reader.cpp:390–418`). |

A positive stale ready value can satisfy a wait before the corresponding remote data write. The reader can then fetch stale/unwritten intermediate bytes, and `line_reduction.cpp:40–55` adds those bytes to valid input. This permits NaN or wrong finite output. A stale barrier value can also bypass the intended protection against cross-device allocation/dispatch skew.

This finding does **not** require overlapping L1 allocations. The allocator's default lockstep behavior explicitly accounts for accesses outside a buffer's shard grid (`tt_metal/impl/allocator/allocator.cpp:198–206`); no aliasing/overwrite claim is made here. The proven defect is missing initialization of used semaphore instances.

### Concrete passing/failing geometry

For BF16 attention/down-projection reduce-scatter, M builds logical `[1,B,S,5120]`, padded `[1,B,ceil(S/32)*32,5120]`. The per-chip packed sizes are:

| Case | Buffer bytes | Width tiles input → output |
|---|---:|---:|
| B1/S33 | 655,360 | 160 → 40 |
| B2/S33 | 1,310,720 | 160 → 40 |
| B32/S33 | 20,971,520 | 160 → 40 |
| B32/decode S1 | 10,485,760 | 160 → 40 |

`reduce_scatter_program_utils.cpp:50–117` selects workers from moved bytes `size * 3/4 / links` for a four-device line. For the listed non-single-packet cases, thresholds are >4,000,000→8 workers/direction/link; <500,000→2; otherwise 4. There is also a ≤4,096-byte→1-worker special case that does not affect this table. Each direction also reserves one mux core. The derived `(workers, total selected cores)` values are:

| Links | B1/S33 | B2/S33 | B32/S33 | B32/decode |
|---:|---:|---:|---:|---:|
| 1 | (2,6) | (4,10) | (8,18) | (8,18) |
| 2 | (2,12) | (2,12) | (8,36) | (4,20) |
| 4 | (2,24) | (2,24) | (4,40) | (4,40) |

These are explicitly conditional examples, not an assertion of the failing run's link count. `get_num_links` obtains routing-plane availability at runtime (`ttnn/cpp/ttnn/operations/ccl/common/host/moe_utils.cpp:98–131`); that returned value is absent from the supplied correctness log.

For **two links**, the two smaller prefills use uncovered RS workers `(8,0),(10,0)`. B32 prefill additionally uses `(8,1),(9,1),(10,1),(8,2),(9,2),(10,2)`. Decode uses uncovered `(8,0),(9,0),(8,1)`. For **one link**, B2 already exposes `(8,0),(9,0)`, while B32 prefill/decode use `(8,0),(10,0)`. Therefore smaller-batch and decode successes do not prove correct semaphore setup.

For a concrete one-link B32 prefill work item, each directional worker owns 10 of the 80 output tiles per batch channel. With BF16 and no passed compute config, granularity is `min(4 * floor(channel_buffer_bytes / 2048), 8)` (`RS/device/reduce_scatter_minimal_async_program.cpp:1350–1365`). At the default eight-tile granularity that is an 8-page chunk followed by a 2-page chunk, over 32 channels. B2 with one link owns 20 tiles/channel, giving 8+8+4 over two channels. The runtime channel-buffer setting is not recorded, so the chunk example is conditional on that default. The reader/compute/writer reserve and advance full CB slots but operate only on the valid page count. No partial-CB-count contradiction was found. The missing initialization occurs before those otherwise coherent transfers.

### Fit to all observations, including counterevidence

- Conv and recurrent state updates occur at M:401,439 before `_finish` performs attention RS at M:294. An output-path RS error can leave those states accurate. High state PCC does not, by itself, prove that every earlier input/gather was perfect.
- A fresh prefill invokes RS ping-pong groups 0 then 1. Decode reuses 0 then 1 (`ccl.py:61–71`; M:293–303). RS completion clears worker ready values. With one link, prefill and decode use the same worker set, so a failed first-use prefill can prime the exact cells later decode needs. With two links the sets differ; `(9,0)` is a prefill mux and later decode worker, so this reset history does **not** guarantee decode correctness.
- Accurate, bit-equal decode trace tests exercise the subsequent decode path; they do not validate first-use prefill semaphore contents.
- Component readbacks add completion boundaries and can change scheduling. Nevertheless, the uninstrumented `b32_repro` also passes: instrumentation cannot be the sole explanation for recovery. Different initial uncovered-cell contents are a possible explanation, not an observed fact.
- B1/B2 passing and B32 reruns passing are compatible with uninitialized values accidentally being zero or waits being satisfied only after data arrives. Their actual initial values and timing remain unknown.

## One discriminating check for the parent

**Prioritize the CCL boundary and initialization contract.** Record the actual `num_links`, active worker coordinates, semaphore construction grid, and whether every RS/AG consumer lies inside that grid. This geometry check does not require reading device tensor data. Also record `TT_METAL_CLEAR_L1`: device opening does not promise globally zero L1 by default (`tt_metal/llrt/rtoptions.hpp:281`; `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:166–170`); an explicit clear would weaken the stale-first-use explanation and require revisiting its prerequisites.

For the next failing occurrence, establish whether the local projected attention is finite **before** attention reduce-scatter and whether its result is the first bad tensor. Include the MLP down-projection/second RS if the first RS is clean. Save finite counts for both the reference and final output, and identify rank/batch/token/hidden coordinates. A finite raw GDN output in a different, passing invocation does not settle this boundary.

Then make one setup-only controlled contrast: create **all RS, AG, and barrier semaphores on the actual full worker core set, initialized to zero before the first forward**, retaining shapes, default worker counts, math, and collective topology. Run the first B32/S33 prefill in fresh invocations and retain an uninstrumented path; do not warm it through an earlier prefill and then call that the first-use check. Merely calling `reset_global_semaphores()` on the existing 8×8 handles leaves the coverage defect intact.

If pre-RS projection is already nonfinite, this finding is secondary for that occurrence. If the first bad values appear after RS, and full semaphore coverage removes the first-use failure across repetitions, the causal case strengthens substantially. A single passing rerun is insufficient because unchanged-source reruns already pass. Initial uncovered-cell values, when obtainable without changing the critical execution, provide the most direct confirmation of the proposed mechanism.

The smallest semantic repair is to align semaphore coverage with the workers selected by the collectives in TP4 setup. That can be done locally to the multichip implementation or by an explicit grid parameter to the shared manager; O need not change. Actually installing a constrained subdevice and consistently using it would be a larger alternative. Supplying only `SubDeviceId(0)`, or lowering a worker count without checking the selected coordinates, does not establish the invariant.

## Other potential issues and paths ruled out as headlines

| Candidate | Source evidence and adjudication |
|---|---|
| GDN adapter reshape/head ordering | No contradiction found. Per-rank widths Q/K/V=512/512/1536 give Hk=4, HV=12, K=V=128. S33 is explicitly padded to T64; NC=2 and BH=12B. Native output `[12B,64,128]` becomes logical `[B,12,33,128]` with padded `[B,12,64,128]`, preserving physical pages (M:376–449; GDN public wrapper:145–200,477–480). |
| Conv padding or TP/SP confusion | The conv writer emits all full-64 output rows. Default sequence-parallel axis 0 has extent 1 on mesh `(1,4)`, so there is no cross-TP history exchange. No unwritten q/k/v tail was found. |
| B32 fused GDN output/readout | Real branch difference: B1/B2 BH12/24 use phased execution; B32 BH384 exceeds the 110-core grid and uses 16 fused waves of 24 heads. At NC2, each wave's producer ring depth3 holds its work without wrap. Full output/state allocation and runtime offset refresh cover all heads (GDN public wrapper:319–328,456–464; device operation:295–338). Scan writer barriers cover all output and state tiles (`writer_chunk_gdn_scan.cpp:44–75`). Missing/stale head-offset stories would normally affect both. No direct kernel contract violation found. |
| Output-only Q corruption | Still possible: Q feeds q-decay/intra/readout, not recurrent-state update (`chunk_gdn_math.hpp:989–1001,1111–1178`). If local projected attention is bad before RS, check raw Q and chunk output `[384,64,128]` before cast/norm, retaining all 64 token rows. This is secondary because no concrete defective instruction/setup was identified. |
| Head RMS repeated-row path | B32 processes 768 tile rows: 108 cores×7 and 2×6; B1/B2 process 24/48, one per core. Gamma is BF16 here. The compute loop resets input/pack formats each iteration, retains gamma, and balances its CB counts (`normalization/layernorm/device/kernels/compute/layernorm.cpp:237–240,445+`). No repeated-row lifecycle bug established. |
| Concatenate-heads scheduling | Uses full 11×10 device grid, not policy `prefill_grid=(8,8)`. B32 has 64 tile-row blocks, still <110, so head-parallel selection remains enabled for B1/B2/B32. The apparent B32-only 64-core branch is false for this hardware (`nlp_concat_heads_program_factory.cpp:37,54–80`). |
| Undefined sequence padding after projection reshape | Confirmed caveat: BF16/FP32 native reshape fills implicit padding only when `pad_value` is explicit (`data_movement/reshape_view/reshape.cpp:541–550`). Z SiLU/multiply can propagate bad padding. However, no path was shown to promote those lanes into logical data: `_linear_raw` flattens logical B×S with real data movement for S33, and width-scatter preserves sequence rows. Therefore undefined padding alone is not the diagnosis. Logical-only finite probes do not inspect it. |
| B32 projection splitting | O:499–609 flattens 32×33=1056 rows and splits at the TP4 cap512 into 512+512+32. The last 32-row part takes the DRAM-sharded small-M matmul path also used by B32 decode. B1/B2 have 33/66 rows and no cap split. This is a probe distinction; no wrong extent or consumed-padding calculation was established. |
| CCL width/shape arithmetic | Correctly uses padded S64, input width160 tiles and output40. Intermediate allocation doubles the appropriate padded leading extent for the line's forward/backward regions. No logical-S33 versus physical-S64 sizing contradiction was found. |

## Completion boundaries and confidence

The forwards enqueue device work; reaching `PREFILL_DONE` alone would not prove completion. Here the initial result and state are read back before that print (`tests/run_multichip_decoder.py:153–160`), providing completion for those reads. Diagnosis adds reads after individual operations (`:92–116`), so it changes completion boundaries. Eager timing explicitly synchronizes; correctness trace replays are blocking; timed replays synchronize only after the loop (`:277–352`). None of those later boundaries fixes a semaphore wait that admitted an unwritten page earlier.

The confirmed finding is **worker/semaphore initialization coverage drift at TP4 CCL setup**. Its connection to this particular NaN remains a causal hypothesis requiring the boundary check above. GDN/reshape alternatives were demoted after checking their lowered shapes, writes, and queue counts rather than being presented as additional established bugs.

Cheap checks performed: read-only source/hash checks; independent arithmetic of padded buffer sizes, default worker thresholds, mux/worker selection and uncovered coordinates for links 1/2/4; line-by-line audits of the relevant conv, GDN, concat, RMS and CCL paths. No device correctness claim or fix validation is made.
