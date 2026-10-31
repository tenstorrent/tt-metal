# Stage 07 LM-head L1 investigation

**Diagnosis:** the two failures are explained by the LM-head's explicit buffer geometry. The original 32K vocabulary split has a static footprint that overlaps live L1 allocations. The current 16K split fixes that small-M footprint, but its explicit 2D prefill configuration lets the output block grow with `per_core_M`; at M=544 this exceeds physical L1. Neither failure requires a numerical, synchronization, or allocator-corruption hypothesis.

**Recommended same-precision candidate:** use 8K vocabulary chunks, 1024-wide DRAM-bank shards, `per_core_N=32`, the default full-height output block, decode K-block 3, and prefill K-block 1. The maximum prefill endpoint through M1024 becomes 984,192 bytes; its live-allocation margin still needs verification. A 4K split with K-block 3 ends at 658,560 and provides more margin. **K-block 2 is illegal with decode's existing 21-tile-wide activation shards**, although it is legal for the DRAM-interleaved prefill input.

**Final-review correction:** setting only `out_block_h=1` reduces the CB footprint but exposes an activation-reader bounds bug on partial core rows, including M544. It is not a complete-workload fix without physical row padding or a reader repair. During inspection, `model.py` briefly acquired this override and then changed again to 8K chunks with default output blocks. The report separates those later edits from the causes of the two supplied logs. No proposed fix was tested on hardware by this investigator.

## Scope and evidence provenance

Inspection only, 2026-10-31. Read `AGENTS.md`, `bringup/run_contract.json`, and `bringup/agentic_evals/inputs.json`. No device commands, resets, target execution, source edits, external publication, or AutoDebug relaunch. This report is the only file written. No removed Gemma implementation or Gemma-specific framework branch was consulted or copied. Three independent xhigh investigations checked DRAM geometry, prefill geometry, and the passing/failing lifecycle contrast.

Checkout HEAD: `d3dcd354e6df404724721b2e0e3af2e1ff7db71d`, with pre-existing working-tree changes. The initial inspected `model.py` already used width 16,384; the first log represents an earlier width-32,768 experiment, as recorded in `doc/optimized_full_model/work_log.md:27`. That historical source snapshot is not present in the two logs. Do not assume every current lifetime detail existed in the earlier run.

During the final audit, `model.py` changed independently of this investigation twice. Snapshot `ffbc2182bcd6913ea898b09e59996898bbcf2d1ceee81181d7b3bfa00fa913ab` added `out_block_h=1, out_block_w=64` at line159. The subsequently observed snapshot `568a8fa1babb6d54b41329947484add83375c835a5fdb41930b5d44efa0d7a99` instead uses 8K chunks, 1024-wide bank shards, per-core N32, default output blocks, and K-block3 in both phases. Its predicted maximum prefill endpoint is 1,156,224 bytes, as tabulated below; no later run is evaluated here. `generator.py` retained its initial hash. The table and main L1 diagnosis below identify the **initial inspected snapshot**, which agrees with the retry geometry. I did not make or revert either source edit. References to the 16K implementation describe that initial snapshot, not a guarantee about the actively changing working tree.

Paths below are relative to the checkout. `A/` means `models/autoports/google_gemma_4_31b_it/`; `MM/` means `ttnn/cpp/ttnn/operations/matmul/`. Line citations refer to the inspected working tree.

| Artifact | SHA-256 |
|---|---|
| `A/tt/model.py` | `eaac31c3ef905ee5143c58616e38cf18d1be576172a98b3a6d25dcc7d03b575c` |
| `A/tt/generator.py` | `b871c79595956defc710cc06f93f9c2a4e1182a8f0bbcdfec02b91926ee012e3` |
| `A/doc/optimized_full_model/buffered_head_b1.log` | `2fa449b1a874156c1a39fdaf485ab1f550a696be525e89b38b2b65a3990b41ba` |
| `A/doc/optimized_full_model/buffered_head_b1_retry.log` | `f9db97a0b95989502b986c5f0c4c6beda7eb3ed9b1a87edcdbac49a19eda4b40` |

## Direct observations versus interpretation

| Direct observation | Supported interpretation / limit |
|---|---|
| First log: `WARM_PREFILL 32`, then static end **1,143,936**, lowest live L1 address **912,640**, range `[0-0 - 10-9]` (`buffered_head_b1.log:24–45`). | Static/live overlap of **231,296 bytes**. This static region is below physical L1. The log does not identify the live buffer's owner. |
| First traceback reaches `prepare_decode → all_logits = terminal(hidden) → linear`; it includes a matmul program-cache hit (`:35–53`). | Failure occurs during host-side compile/allocation validation on enqueue. It need not be the first vocabulary chunk; no chunk index is recorded. A cache hit does not guarantee the same live allocation frontier. |
| Retry advances through M32,64,…,512, then fails at M544 with end **1,746,048 > 1,572,864**, range `[0-0 - 7-5]` (`buffered_head_b1_retry.log:24–61`). | A different, absolute capacity failure. At M544, the 2D per-core M block increases from two to three tile rows. |
| Sweep log records 32K/eight senders/two readers/K-block3 at 1023.23 µs and PCC≈0.99999994 (`lm_head_candidates.log:133`). | An LM-head-only traced-host timing and eager-output comparison against the interleaved TT baseline, not full-model performance or HF accuracy. |
| Both sweep and integration construct `Gemma4Generator(layer_indices=[0,5])` (`A/tests/sweep_lm_head.py:19`, `A/tests/probe_buffered_generation.py:22`). | “The sweep had no live generator storage” is false for the current harness. Exact allocation/lifetime differences remain unresolved; construction alone cannot explain the contrast. |

The warmup prints precede execution and are not synchronized correctness assertions. The supplied retry proves earlier calls cleared their synchronous allocation checks; it does not prove all their queued numerical work completed correctly. `prepare_decode` synchronizes only after warmup/decode/sampling and before trace capture (`A/tt/generator.py:142–158`). Both reported exceptions occur before that boundary, in `validate_dataflow_buffer_region`, so a late device hang is not needed to explain them.

## Lowered contract and allocation timeline

`A/tt/model.py:61–83,138–167` establishes:

- TP4 vocabulary ownership: global 262,144 → local 65,536. Each local chunk retains the full K=5,376, or **168 tiles**. Current chunks are four contiguous 16,384-column tensors with DRAM-bank shards `(5376,2048)`.
- Normalized activation BF16, weight BFP8_B, output BF16; HiFi2, `fp32_dest_acc_en=True`, `packer_l1_acc=True`, approximation disabled. There is no bias or fused activation in the matmul. Softcap follows concatenation and is not reached by the failing linear.
- M≤32: convert activation to L1 width-sharded storage over eight cores, use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, and produce an L1 width-sharded tensor before copying it to DRAM.
- M>32: retain DRAM-interleaved activation, use `MatmulMultiCoreReuseMultiCastProgramConfig` with `(8,gy)` and DRAM output. This is the **2D** factory despite the weights remaining DRAM-sharded.

Explicit configs bypass automatic capacity-based selection (`MM/device/config/matmul_program_config.cpp:1022–1023`; `MM/device/matmul_device_operation.cpp:3229–3247`). Factory dispatch is exact at `matmul_device_operation.cpp:2508–2526`. Thus these failures belong at the model's explicit geometry selection boundary; there is no evidence that an automatic planner selected a different configuration or silently shrank these buffers.

The time-ordered capacity ledger is:

1. Model/generator construction allocates persistent state, including two L1 gather workspaces (`A/tt/multichip_decoder.py:247–285`).
2. Warmup produces hidden, normalizes it, and, for M≤32, allocates the width-sharded activation.
3. Matmul output storage is created before workload execution (`MM/device/matmul_device_operation.cpp:3049–3052`, `ttnn/api/ttnn/device_operation.hpp:622`). In the DRAM-head path that output is another live L1 allocation, separate from the static output FIFO.
4. The factory builds static dataflow buffers. Their base comes from the persistent L1 arena's high-water mark (`tt_metal/impl/program/program.cpp:474–483`; `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2995–3033`).
5. The validator re-queries the current lowest occupied compute-L1 address, first checks physical L1, then checks static/live overlap (`dataflow_buffer.cpp:3071–3096`). It catches both failures correctly.
6. Only after a successful linear does current code copy its output to DRAM and `del part` (`A/tt/model.py:162–166`). That release cannot rescue the allocation required to enter the same linear.

No capacity check may count only the static CBs and ignore the input/output tensor allocations. Conversely, don't double-count the output tensor as the static output FIFO: they are separate regions in the DRAM-head factory (`MM/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:496–507`).

For eight activation-storage cores, each live BF16 activation shard is `21×2048 = 43,008` bytes. With eight output-storage cores, a 32K chunk allocates `128×2048 = 262,144` bytes per output core; a 16K chunk allocates `64×2048 = 131,072`. Output shard construction uses the configured storage width (`MM/device/matmul_device_operation.cpp:2836–2876`). These are additional live tensor allocations, not extra terms in the static-CB formulas below. The exact allocator address also depends on the other live allocations and their placement.

## Finding 1: 32K/two-reader/K-block3 consumes too much of the live L1 budget

For 32×32 tiles: BF16=2,048 bytes, BFP8_B=1,088 bytes, FP32=4,096 bytes (`tt_metal/impl/data_format/tile.cpp:136–170`). These sizes already satisfy the relevant 64-byte DRAM/DFB alignment. FP32 partials cannot alias the BF16 output FIFO because their formats differ.

The DRAM factory assigns **eight banks × two readers = 16 compute/read workers** and recalculates their compute width from N (`MM/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:174–216`). Its `per_core_N` argument is output **storage** width, not the per-reader compute width. At local width32K, the workers compute 64 tiles each, while eight output-storage cores each hold 128 tiles. Current16K uses compute width32 and storage width64.

With `k=in0_block_w`, `n=compute N tiles per reader`, M=1, the static allocation is:

```
activation FIFO       = 2 × k × 2048
weight FIFO           = 3 × k × n × 1088
BF16 output FIFO      = n × 2048
FP32 partials         = n × 4096
```

The activation is double-buffered, the weight is **triple-buffered**, and output/partials each hold a complete output block (`...dram_sharded_program_factory.cpp:267–308,479–555`). There is no bias or transpose scratch allocation for this case.

At 32K/k3/n64:

```
12,288 + 626,688 + 131,072 + 262,144 = 1,032,192 bytes
observed endpoint 1,143,936 − CB bytes 1,032,192 = base 111,744
```

This reconciles the first log exactly. **111,744 is an inferred effective base for these observations, not a universal architecture constant.** Candidate endpoints below use this same base; fresh runtime state must still satisfy the final validator.

| Local chunk | Compute N/reader | k1 endpoint | k3 endpoint | k3 vs observed 912,640 frontier |
|---:|---:|---:|---:|---:|
|32,768|64|717,952|1,143,936|231,296-byte overlap|
|16,384|32|416,896|633,984|278,656-byte margin|
|8,192|16|266,368|379,008|533,632-byte margin|
|4,096|8|191,104|251,520|661,120-byte margin|

The original 32K decode/small-M path could therefore be reduced by changing **only k3→k1**, without changing weights. Current 16K/k3 already has the smaller footprint observed to get past M32 in the retry.

The reported rectangular range is not a count of compute workers. The DRAM factory constructs a bounding rectangle and places kernels with active/inactive flags across it (`...dram_sharded_program_factory.cpp:375–380,1229–1232`); `[0,0]–[10,9]` does not mean 110 independent N partitions.

**K-block legality:** `_working_memory(x,8)` makes activation shards 672 elements = **21 tiles** (`A/tt/optimized_decoder.py:664–674`). The shard width and K-block width must divide one another (`MM/device/matmul_device_operation.cpp:1640–1648`). Blocks 1 and 3 are legal; block 2 is not, even though total Kt=168 is divisible by 2. Changing the sender layout to seven cores gives 24-tile shards and permits 2, but is a larger intervention. At 32K, k2's calculated endpoint 930,944 would still exceed the old frontier by 18,304 bytes.

### What the sweep/integration contrast does not establish

Both harnesses construct caches, sampler state, prepared inputs, logits/history, and persistent gather storage. The sweep uses an interleaved head at construction, M192→last-row selection, an eager reference readback, and several traced candidates before the32K candidate (`A/tests/sweep_lm_head.py:19–43,51–67`). Integrated warmup resets caches and calls the all-row terminal on M32 first (`A/tt/generator.py:141–150`). Those are different allocation/execution lifecycles, but the logs do not record their corresponding L1 allocation maps.

Do not attribute 912,640 to retained prefill hidden: current prefill explicitly returns hidden in DRAM (`A/tt/optimized_decoder.py:635–637`). Do not attribute it to a newly captured sampling trace: no capture or sampling warmup has executed yet. The first traceback's cache hit also prevents claiming it is necessarily the first vocabulary chunk. Current code releases `part`; the earlier log does not prove the same release existed then. The **overlap is proven; its exact live-buffer owner and why the sweep had more usable headroom are unresolved**.

## Finding 2: 2D output-block defaults explain the exact M544 failure

For prefill, the initial inspected code computes `Mt=ceil(M/32)`, `gy=min(8,Mt)`, and `m=ceil(Mt/gy)` and passes `per_core_M=m`, `per_core_N=64`, K-block3 (`A/tt/model.py:153–159` in the initial snapshot). It omits `out_block_h/w`. Python bindings turn these omissions into **`out_block_h=per_core_M`, `out_block_w=per_core_N`** (`MM/matmul_nanobind.cpp:167–168`). `out_subblock_h=1` does **not** make the whole output buffer one row high.

The 2D factory allocates two activation/weight blocks plus one full BF16 output block and one full FP32 partial block (`MM/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:370–443,797–837`). With output-block dimensions h×n and K-block k:

```
CB bytes = 2 × h × k × 2048
         + 2 × n × k × 1088
         + h × n × (2048 + 4096)
```

At M544: `Mt=17`, `gy=8`, `m=h=ceil(17/8)=3`, `n=64`, `k=3`:

| Buffer | Tiles | Bytes |
|---|---:|---:|
|Activation|2×3×3 =18|36,864|
|Weights|2×64×3 =384|417,792|
|BF16 output|3×64 =192|393,216|
|FP32 partials|3×64 =192|786,432|
|Total static CBs||**1,634,304**|
|Plus inferred base111,744||**1,746,048**|

This exactly matches the retry and exceeds physical L1 by **173,184 bytes**. Even the CB payload alone exceeds1,572,864, so relocating existing tensors cannot fix this geometry.

Actual active grid height is `ceil(Mt/per_core_M)`, not the requested maximum gy (`...2d_program_factory.cpp:460–470`). At M544 it is `ceil(17/3)=6`, explaining `[0,0]–[7,5]`. The last row of cores computes a partially padded M block, but allocates the same full three-row buffers.

| Physical M | Mt | per-core M / default h | Active grid | k3 endpoint |
|---:|---:|---:|---|---:|
|64–256|2–8|1|8×Mt|935,040|
|288–512|9–16|2|8×ceil(Mt/2)|1,340,544|
|544–768|17–24|3|8×ceil(Mt/3)|1,746,048|
|800–1024|25–32|4|8×ceil(Mt/4)|2,151,552|

The current warmup contract is **32–1024 in steps of32**, not2048 (`A/tt/prefill.py:14`). M32 and selected single rows use the separate DRAM-head factory. Fixing only544 does not cover the remaining800–1024 shapes.

The earliest problematic choice is letting a per-core work assignment also determine the entire buffered output block, without checking its footprint. The bindings and factory implement that explicit/defaulted configuration consistently; the allocator is the messenger.

## Same-precision interventions

All options below retain BF16 activations/output, BFP8_B weights, HiFi2, FP32 destination accumulation, packer accumulation, full K, vocabulary ownership/order, and advertised context. They require integrated correctness/performance validation; equal precision is not a guarantee of bitwise equality after changing reduction blocking.

### 1. Conditional small change: keep16K weights and bound the prefill output block

Setting `out_block_h=1` in the 2D prefill config, retaining `per_core_M` as the work assignment and `out_block_w=per_core_N=64`, is a small **capacity** intervention. The arithmetic below is correct, but the final reader audit rejected it as a standalone repair for all warmup buckets.

| Prefill configuration | Endpoint for every current M>32 | Interpretation |
|---|---:|---|
|h1, k3|935,040|One added field; below physical L1 and below the already accepted M512 footprint. **22,400 bytes above** the first log's separate small-M frontier.|
|h1, k2|791,680|One added field plus a prefill-only K-block change;120,960 below that historical frontier.|
|h1, k1|648,320|More headroom;264,320 below that historical frontier.|

The first log's 912,640 is **not a universal ceiling**: retry M512 already gets past a 1,340,544 static endpoint, and 2D output is DRAM. All three h1 configurations fit physical L1, with k1/k2 providing more margin. None of these comparisons proves the later full-model allocation frontier or resolves the input-reader issue below.

Config validation passes: h1 divides all per-core M values 1–4, Kt=168 is divisible by k1/k2/k3, and width 64 remains divisible by subblock width 4 (`MM/device/matmul_device_operation.cpp:545–586`). The weight sender explicitly loops over output-height blocks and resets its DRAM K offset for each (`MM/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:384–400`); the activation sender and compute kernel also loop over height blocks. However, passing config validation and matching FIFO counts do not prove correct handling of fully padded height blocks.

For a concrete M544 h1/k2 program: each core processes three height blocks, each with 84 K blocks. Each K block produces/consumes 2 activation tiles and 128 weight tiles; its two-slot FIFOs hold 4 and 256 tiles. Output/partials each hold 64 tiles. The existing h3/k3 program instead handles one height block with 56 K blocks, 9 activation tiles and 192 weight tiles per K block, and 192-tile output/partial buffers. The FIFO counts match, but the input address for the third h1 block on the final core row is invalid.

**Concrete bounds discrepancy:** at M544, Mt=17 and per-core M=3. The final active core row starts at tile row15 and h1 iterates over rows15,16,17. Only rows15 and16 exist. The factory supplies `last_block_h=1`, since `last_per_core_M=2` is divisible by h1 (`...2d_program_factory.cpp:101–114,1152–1162`). The activation reader's guard is only `bh < num_blocks_h_dim - 1 || h < last_block_h` (`MM/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp:322–331`). On bh2/h0 it still reads row17, starting at page `17×168=2856`, exactly beyond this tensor's 2856 pages. Output masking suppresses the corresponding output write, but cannot prevent that input read. The original h3 program handles these rows in one block and supplies `last_block_h=2`, so its input guard avoids row17.

For the current mapping, h1 exposes this problem when `Mt % per_core_M != 0`: **M288,352,416,480,544,608,640,704,736,800,832,864,928,960,992**. Changing K-block size does not fix it. This is a source-level bounds violation in the proposed/later h1 path, **not the cause of the supplied static-allocation failures**; the runtime manifestation remains untested.

To retain h1 safely by construction, one candidate is to physically pad the normalized DRAM activation to `ceil(Mt/per_core_M)×per_core_M×32` rows before projection, then trim logits back to the original M. M544 would use576 physical rows. Another is a generic reader repair that passes valid height-block counts and suppresses/zero-fills all out-of-range input blocks while keeping FIFO traffic balanced. Both require additional work and validation; neither is included in the one-field shortcut. Repeated height blocks also reread weights, so performance needs measurement.

Leave the M≤32 branch at16K/k3. A prefill-only k2 does not require changing the decode activation layout.

### 2. Recommended bounded alternative: narrow physical vocabulary chunks

Narrow splits with the default `out_block_h=per_core_M` avoid the newly exposed input-block path. Change **both** the physical weight sharding and `per_core_N`. Remove the later explicit h1/w64 override: omit both output-block fields, or set them consistently to `per_core_M/per_core_N`. For eight DRAM banks/eight output-storage cores, use:

| Local chunk width | Number of chunks | DRAM shard shape | `per_core_N` |
|---:|---:|---|---:|
|16,384|4|(5376,2048)|64|
|8,192|8|(5376,1024)|32|
|4,096|16|(5376,512)|16|

Keep ascending slices and final concatenation to preserve local vocabulary order. Merely slicing narrower while leaving `per_core_N=64` does not reduce the 2D block width or its dominant buffers.

With default h=per-core M, worst-caseM1024:

| Width | Prefill k | Endpoint | Capacity interpretation |
|---:|---:|---:|---|
|16,384|1|1,840,256|Still exceeds physical L1.|
|8,192|3|1,156,224|Fits physical L1; live frontier must exceed this.|
|8,192|2|1,070,208|Legal prefill choice with more margin.|
|8,192|1|984,192|Smaller, but still71,552 above the first log's small-M frontier.|
|4,096|3|658,560|Fits below that historical frontier by254,080.|

Thus 8K splits plus prefill K-block1 are the smaller recommended repartition candidate; decode can retain K-block3. The 4K/K-block3 option provides greater static margin. Neither is a proof against every live-allocation state. Combining these splits with h1 would reintroduce the same input bounds problem unless separately addressed.

### 3. Changes that do not solve the complete problem

- **Only k3→k2 at16K/h3:** M544 still ends at1,594,496, over physical L1.
- **Only k3→k1 at16K:** M544 fits at1,442,944, but M800/h4 fails at1,840,256.
- **Only reduce output subblock dimensions:** these control destination-register work, not the full h×n output/partial CB extents.
- **Only move final logits to DRAM:** prefill already does this; its static output FIFO and FP32 partials remain necessary.
- **Disable packer accumulation while retaining FP32 destination:** both factories still select Float32 partial storage. This is not the requested same-policy geometry repair.
- **Skip all-token terminal warmup or project only the last row everywhere:** hides the failing supported path; `return_all_logits=True` still needs it (`A/tt/generator.py:207–210`).

## Other Potential Issues / limits on proposed fixes

**Do not shrink `out_block_w` alone on unchanged16K DRAM-sharded weights as a shortcut.** The 2D DRAM reader explicitly assumes one output-width block (`reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:399`). Its bank walker is assigned per-core N-wide spans (`...2d_program_factory.cpp:1201–1205`), while the FIFO is sized from `out_block_w`. Setting w32 with per-core N64 can make those widths disagree. This is not implicated in the observed logs, which use w=N64; it is a reason to prefer h1 or actual narrower weights rather than this unvalidated alternative.

No reported observation identifies the owner of912,640. Persistent-workspace fragmentation, an older chunk-output lifetime, or other transient allocations could affect that frontier; none is established sufficiently to headline as a separate bug. The current explicit config also makes a stale automatic-planner snapshot unnecessary to explain the failures.

Changing output storage `per_core_N` in the DRAM-head factory can distribute its tensor across more cores, but does not change its independently calculated per-reader static buffers. This is separate from changing `per_core_N` in the 2D factory.

## Claim recheck and remaining validation

After drafting, the headline claims were checked again against the bindings, factory allocation branches, validator, and logs, followed by an independent adversarial check of the report. A pure-Python integer calculation, with no TTNN import or hardware access, reproduced both observed endpoints exactly and checked the candidate tables. Claims rejected during review were: the sweep lacked generator residency; retained prefill hidden necessarily occupied L1; 912,640 was a constant ceiling; k2 was legal in both phases; smaller K blocking alone fixed every prefill shape; and h1 with matching FIFO counts was safe across all partial-row shapes. The final reader audit supplied a concrete counterexample to the last claim, so h1 was withdrawn as a standalone recommendation.

The causal chain accounts for both error families, exact byte counts, M512→544 boundary, and the six-row active grid. It does **not** identify the historical live buffer or establish new runtime correctness/performance. The inspected current source and log geometry agree, but no binary/source build manifest was provided for the historical32K run.

Suggested follow-up for the implementing agent, not performed here: try 8K chunks with matching1024-wide bank shards and per-core N32, default full output blocks, decode k3 and prefill k1. Use 4K/default-block/k3 if more L1 margin is required. Do not leave the later h1/w64 overrides in either configuration. Record effective config, chunk index, allocator frontier after output allocation, and static endpoint. Exercise all 32 warmup buckets through1024, especially512/544 and768/800; test all-token and last-row logits plus batches1/8/16/32 with the real persistent state. Synchronize and compare outputs, then repeat with full-model storage and trace replay. Only then use integrated timing to choose between these same-precision configurations. If preserving16K weights with h1, first implement physical row padding/output trimming or repair the reader, and explicitly cover the15 affected buckets above.
