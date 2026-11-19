# Long-context full-attention numerical investigation

## Findings

**The leading diagnosis is loss of precision in the native SDPA running state, despite `fp32_dest_acc_en=True`.** This is a confirmed property of the selected kernels and a plausible explanation of both prefill and decode length sensitivity; its responsibility for the recorded PCC is not yet established by a matched-input device experiment.

- **Decode stores scores, exponentials, running softmax sums, and output numerators in BF16.** At the failing extent each core performs 1,024 serial KV-chunk iterations before a three-round tree reduction. The FP32 destination flag does not preserve FP32 state across those iterations.
- **Prefill retains its numerator in BF16 across up to 8,192 iterations.** Its sum buffers are Float32, but ordinary FPU reloads of those buffers pass through TF32 Src registers. Describing this path as a fully FP32 online reduction would also be incorrect.
- **The long-context harness accepts prefill using a flattened PCC over 38 selected decoder-output rows, without enforcing the contribution-row threshold.** Ten selected contribution rows are below .995. The numerical discrepancy therefore already exists before decode capture.

No concrete position, page mapping, last-token masking, CB-count, or capture-state contradiction was found for the reported configuration. The custom HF oracle has the expected logical operations, but its default SDPA backend is not an exact oracle: small CPU checks select CPU FlashAttention, which also has reduced-precision boundaries. A same-input FP64 attention control is needed before attributing the entire discrepancy to TTNN.

**Scope:** inspection only; no TT hardware operations, resets, reproductions, or source changes. Only this report was written. Three independent xhigh investigations covered native decode, native prefill, and the oracle/harness. Small CPU arithmetic/backend checks were permitted inspection aids, not reproductions of the recorded model failure. The required 262,144-token context remains unchanged.

## Evidence and revision consistency

Repository: `/home/ttuser/qwen36-27b/tt-metal` at `a0b0bd54e198d81676662b5a2d3a489e29bc9ab9`. The model directory is untracked in this checkout; the native transformer sources are tracked. Current decoder/config hashes exactly match the retry JSON's provenance.

Direct observations from `doc/functional_decoder/full_context_262143_retry.json` and its matching log:

| Measurement | Observed value | What was actually measured |
|---|---:|---|
| Prefill input length | 262143 | Logical positions 0 through 262142 |
| Prefill PCC | .9984699000628736 | Flattened full decoder outputs at 38 selected positions |
| Minimum contribution PCC | .9692672122154313 | Position **262140**; correlation of decoder output minus input |
| Contribution rows below .995 | 10 / 38 | Positions 262113, 262118, 262121, 262124, 262126, 262128, 262137, 262138, 262140, 262141 |
| Decode context | 262144 | New token at absolute position 262143 |
| Traced decode PCC | .9563047020222268 | One full decoder output after blocking replay |
| Prefill wall time | 150.317 s | Includes a device synchronization after prefill |

The log reports Blackhole. Model geometry is batch 1, 24 Q heads, four KV heads, head width 256, six Q heads per KV head, BF16 activation/cache tensors, page size 32, grid 8×4, Q/K chunks 32, HiFi4, accurate math and exponential settings, FP32 destination accumulation. See `functional_decoder.py:47–51,252–264,279–350`.

The earlier `full_long_harness.json` records prefill length4097 PCC .999979768505389 and traced decode context4098 PCC .9999799757180245. **It has no source provenance.** `work_log.md:62–72` explicitly identifies that artifact as predating the FP32 folded-norm repair. It is a useful historical passing observation, not a same-revision controlled length comparison. Furthermore, `run_long_context.py:52–53` draws the decode token after generating the length-dependent prefill tensor; the short and long commands do not hold the decode token fixed. Current short/long kernel work counts below are derived from current source, not claimed as recorded runtime telemetry for the older run.

The retry provenance records decoder/config hashes and checkout HEAD, but not the harness/reference hashes or installed package versions (`tests/evidence.py:12–32`). The harness acquired optional `--attention-chunk` / `--probe-tail-only` diagnostics between the initial read and the hashed inspection snapshot. Neither option appears in the retry command. Harness line references below use the later243-line snapshot, whose hash is recorded at the end; they are not a claim that the retry recorded that harness hash. The default allocation and attention/capture sequence remain consistent with the initially inspected210-line version. Current installation: Torch 2.11.0+cpu and Transformers 5.12.1. The config's `transformers_version=4.57.1` is checkpoint metadata, not the installed reference version.

## Source map

Line references in this report use these abbreviations, all relative to the repository root unless specified:

| Name | File |
|---|---|
| Model | `models/autoports/qwen_qwen3_6_27b/tt/functional_decoder.py` |
| Harness | `models/autoports/qwen_qwen3_6_27b/tests/run_long_context.py` |
| HF | `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` |
| PF | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` |
| PC | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` |
| DF | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` |
| DC | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` |
| DR | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp` |

## Native prefill: where precision is lost

### Selected lowering and concrete dimensions

Model:326–350 pads each physical prefill chunk to a multiple of32, fills paged K/V, and invokes the **scalar** `chunked_scaled_dot_product_attention` entry point. `sdpa/sdpa.cpp:104–141` passes Q/K/V unchanged into `prim::sdpa`, always causal, with the supplied scalar absolute start. There is no operand swap or dense finite mask in this wrapper. The implicit scale is `1/sqrt(256)=1/16`, exactly representable; see PF:1159–1161 and Model:314 for the explicit equivalent decode scale.

PF:80–82,1398 selects **legacy/standard compute**, because `fp32_dest_acc_en=True` disables the streaming implementation. The executable path is `sdpa/device/kernels/compute/sdpa.cpp`, the standard branch at :197–220 and the shared `sdpa_inner_loop` in PC. Streaming-only math, K/V reuse and ring/distributed algorithms are not this execution.

| Quantity | Near-max prefill | Derived current-source short case |
|---|---:|---:|
| Logical input length | 262143 | 4097 |
| Full physical input chunks | 511 ×512 | 8 ×512 |
| Final physical chunk start | 261632 | 4096 |
| Final logical / padded query length | 511 /512 | 1 /32 |
| Final absolute query tile | 8191 | 128 |
| KV chunks processed by that tile | 8192 | 129 |

For the long final physical chunk, each head has 16 query tiles; total384 query/head work items are distributed over32 cores, ordinarily12 items/core. These are **query** work items: a given late query tile's KV scan is not split among those32 cores. The final query tile processes all8192 KV tiles serially. Reader/compute bounds agree (`reader_interleaved.cpp:575–583,662`; PC:1716–1722,1751–1753). Logical last output262142 is causal and cannot see padded row262143; the padded row's own output is sliced away.

### Actual stored formats

PF:1750–1754 and1858–1866 determine the following formats with FP32 destination enabled:

| State | Format in L1 | Important boundary |
|---|---|---|
| Q/K/V inputs | BF16 | Model projections/cache |
| QK scores, then exp weights (`qk_im`) | Float32 | FPU consumers reload through TF32 Src |
| Current/previous row maximum | BF16 | Rounded after each maximum reduction |
| Maximum correction factor | BF16 | Rounded `exp((m_old-m_new)/16)` |
| Current/previous partial exponential sum | Float32 | TF32 FPU reload on rescale/add |
| Current/previous output numerator (`out_im_A/B`) | **BF16** | Repacked for every KV chunk |
| Final output | BF16 | Output normalization/storage |

The sequence is explicit:

1. QK is computed and packed; row maximum is reduced and packed to BF16 (PC:1802–1816,1943–1945).
2. `sub_exp_block_bcast_cols_inplace` calculates exponentials and partially reduces them into the sum CB (PC:1957–1958; helper:388–451).
3. `P @ V_chunk` is packed into the BF16 current numerator (PC:1963–1978).
4. For every later KV chunk, rescale previous sum/numerator by the packed correction factor, add their new contributions, and pack again (PC:1984–2007). In particular, the numerator merge uses packer L1 accumulation **onto a BF16 CB**. This cannot recover precision lost at preceding packs.
5. Swap state buffers and repeat. Final sum reduction, reciprocal and numerator normalization occur after the KV loop (PC:2010–2021,2133–2141).

The Float32 sum buffers deserve a separate caveat. PF:2010–2015 leaves `ComputeConfigDescriptor.unpack_to_dest_mode` at its default. `tt_metal/jit_build/genfiles.cpp:1069–1086` chooses TF32 intake for FP32 destination mode with B-family formats; `tt_metal/jit_build/data_format.cpp:147–154,211–217` actually assigns that conditional format to Float32 buffers unless an explicit nondefault unpack mode overrides it. `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:311–322,339–340` distinguishes full FP32 direct-to-destination intake from FP32-to-TF32 Src intake (10 fraction bits). Sum rescale/add use ordinary broadcast/binary Src operations (PC:643–653,676–688; `tt_metal/hw/inc/api/compute/bcast.h:440–446,463–467` and neighboring `eltwise_binary.h:38–45`). **Float32 storage therefore does not mean 23 mantissa bits survive the recurrence.** Likewise the Float32 exponential weights feed a TF32-input PV matmul.

This establishes repeated quantization, not the amount of error in the real attention distribution. Actual Q/K/V, attention maxima, effective support, and intermediate numerator/sum errors were not saved. Different queries/heads can have very different sensitivity, so variation among adjacent late contribution PCCs does not falsify this mechanism. It also does not prove it.

## Native decode: 1,024 serial BF16 updates before reduction

### Selected lowering and workload

Model:291–322 projects/normalizes/rotates Q/K/V, permutes K/V into `[1,batch,4,256]`, places them on a one-core-per-request height shard, and calls `paged_update_cache` at the supplied position. Q is separately permuted to interleaved tiled `[1,1,24,256]`, physically padded to32 head rows. `sdpa_decode/sdpa_decode.cpp` preserves Q/K/V ordering and forwards the program and compute configs into `prim::sdpa_decode`; DF installs `reader_decode_all.cpp`, `sdpa_flash_decode.cpp`, and `writer_decode_all.cpp` (:930–950).

`SDPAProgramConfig.max_cores_per_head_batch` defaults to16 (`sdpa_config.hpp:21`). DF:202–216 computes:

```text
cores_per_batch = min(32, 16 * 1 * 4) / 1 = 32
cores_per_KV_head = 32 / 4 = 8
heads_per_core = 1; active cores = 32; KV-head reducers = 4
tree rounds = ceil(log2(8)) = 3
```

DR:60–89 rounds `cur_pos+1` upward to a whole K chunk and assigns contiguous ranges in reverse core order:

| Decode extent | Total KV chunks | Serial chunks/core/KV head | Tree rounds |
|---|---:|---:|---:|
| 262144 tokens, position262143 | 8192 | 1024 | 3 |
| 4098 tokens, position4097 | 129 | 16, with17 on core7 | 3 |

For the failing extent, core0 owns chunks `[7168,8192)` and core7 owns `[0,1024)`. Tree reduction only combines results **after** these serial recurrences. It is not an8192-way balanced softmax reduction.

### Actual stored formats and recurrence

DF:507–508 hardcodes both `im_df` and `stats_df` to `Float16_b`. DF:637–674,694–720 allocates:

| CBs | Meaning | Format |
|---|---|---|
| c24 | QK, then exponentials | BF16 |
| c25, c26, c23 | Chunk numerator, running numerator, child numerator | BF16 |
| c27/c28 | Current/previous maximum | BF16 |
| c29/c30/c21 | Current/previous/child sum | BF16 |
| c31/c22 | Correction factors | BF16 |
| c6/c7; c16/c17/c18; c19 | Incoming stats, outgoing numerator/stats, tree staging | BF16 |
| c20 | Final output | BF16 |
| c8/c15; c9 | Writer/compute positions; page table | Raw INT32 |

DC:415–527 packs QK to BF16, finds a maximum, calculates and packs exponential weights, reduces their sum, and computes a chunk numerator. Every subsequent iteration rescales and repacks old sum/output and adds and repacks the new sum/output (DC:543–574; PC:636–662 confirms addition's pack boundary). Thus there are1023 serial state-update merges per core at maximum context versus15–16 at the derived short case.

Tree correction also packs maxima, sums and factors to BF16 (PC:828–847; DC:618–646); tree transfers retain that representation. Reciprocal and final normalized numerator are also packed to BF16 (DC:687–714). This is a stronger precision restriction than prefill's Float32 score/sum CBs, even though prefill has the longer sequential scan.

DF:219 extracts `packer_l1_acc`, but DF:946–951 does not propagate that field in the compute descriptor. Helpers explicitly toggle packer L1 accumulation where used. The caller's `packer_l1_acc=True` should not be interpreted as a promise to retain these state tensors at higher precision. This ignored config field is a discrepancy, not a separate established cause of the PCC failure.

### Small CPU check of the proposed mechanism

A cheap CPU-only recurrence check demonstrates why this storage boundary can matter. For each chunk, add32 to a BF16 sum. Set a scalar V to0 for the first half of chunks and1 for the second half, adding either0 or32 to a BF16 numerator. Max correction is1. With round-to-nearest BF16 after every addition:

| Local chunks | Exact sum | Repacked BF16 sum | Exact numerator | BF16 numerator | Exact mean / BF16 ratio |
|---|---:|---:|---:|---:|---:|
| 16 | 512 | 512 | 256 | 256 | .5 / .5 |
| 1024 | 32768 | 8192 | 16384 | 8192 | .5 / 1 |
| 8192 | 262144 | 8192 | 131072 | 8192 | .5 / 1 |

At8192, a BF16 spacing of64 makes another32 a round-to-even tie. This is a constructed arithmetic example, **not** a simulator of the native packer, prefill's TF32 sum reload, cross-core layout, or the real model's data. It proves the generic vulnerability and gives exact-output synthetic discriminators; it does not predict the observed .9563 PCC. Constant V alone can conceal errors when numerator and denominator lose the same scale. A global halfzero/halfone pattern can also align with decode's eight core partitions and conceal the local effect; a device version of this particular example must put the transition inside each core's owned range.

Reproducible CPU check, no TTNN import:

```python
import torch
for n in (16, 1024, 8192):
    denominator = torch.tensor(0., dtype=torch.bfloat16)
    numerator = torch.tensor(0., dtype=torch.bfloat16)
    for i in range(n):
        denominator = (denominator.float() + 32).bfloat16()
        numerator = (numerator.float() + (32 if i >= n//2 else 0)).bfloat16()
    print(n, float(denominator), float(numerator),
          float(numerator) / float(denominator))
```

## Oracle and module-by-module comparison

No logical discrepancy was found in the full-attention equations:

| Stage | HF/reference | TT model | Assessment |
|---|---|---|---|
| Input / head RMSNorm | HF:736–750 | Model:54–74,119–127 | FP32 normalize and folded `(1+weight)` stored FP32, cast BF16 afterward; not the older folded-scale bug |
| Q and gate projection | HF:683–690 | Model:97–100,279–283,285–289 | Correct per-head interleaving split; GQA ratio6 |
| K/V projection and cache | Harness:81–95,112–120 | Model:279–283,301–306,333–336 | All real prefix tokens are projected and cached; current decode token added once logically |
| Partial RoPE | HF:134–167,562–605 | Model:272–277 | Rotate first64 dimensions using half32 split; pass remaining192 unchanged |
| Attention | Harness:97–109 | Model:308–322,337–350 | Same causal domain and scale1/16; materially different numerical reduction boundaries |
| Output gate / projection | HF:713–716 | Model:285–289 | Sigmoid gate before output projection |
| Residual / MLP | HF decoder/MLP, HF:720–733,756 onward | Model:129–134 | Same order, with BF16 operator boundaries |

HF constructs cos/sin using FP32 absolute positions and casts them to the activation dtype (HF:149–167); the harness uploads those same rotations to TT. There is no independent BF16 position generation or TT high-angle trigonometry here. On text inputs all MRoPE axes have the same position. This rules out several plausible high-position RoPE stories at the preparation boundary, while leaving ordinary projection/norm/RoPE arithmetic error to be measured.

The custom selected-query oracle uses a boolean mask `key_absolute_position <= query_absolute_position`, which is necessary for noncontiguous selected query positions; replacing it with an ordinary local triangular mask would be wrong. It includes the current decode K/V explicitly, uses GQA and no dropout, and defaults to scale1/16 (Harness:97–120). A small CPU check with synthetic BF16 tensors showed selected-query output bit-equal to selecting those same rows from a full causal call; this verifies the test construction for that small case, not its full-context numerical accuracy.

**Do not assume this CPU oracle uses the math backend.** Current Torch2.11 CPU profiling of small BF16 Q24/KV4 signatures with a boolean mask and `enable_gqa=True` reports `aten::_scaled_dot_product_flash_attention_for_cpu`. In one short synthetic check default CPU Flash versus forced math gave PCC .9999993282 (max absolute difference .015625); this is not an error bound for the long real input. The original long execution did not record its selected backend.

The installed Torch git revision's [CPU FlashAttention source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L313) uses FP32 working score/max/sum/numerator buffers for BF16 inputs, but rounds exponential weights to BF16 for PV. Its short-query specialization uses K blocks512, which differs from TT's32-token chunks (source lines313,370–417,633–640,1097–1109). A forced FP64 same-input attention oracle is therefore the strongest local adjudicator. Cast the **already computed BF16 Q/K/V** to FP64; do not regenerate the entire model in FP64 and thereby change the inputs under test.

`contribution_pcc` is correlation of `(decoder_output - decoder_input)` in FP32 with FP64 centering/reduction (`run_contract.py:20–25`; Harness:182). The current helper explicitly asserts finite row metrics. It includes attention, MLP, and residual-add rounding. It is neither raw attention PCC nor a direct measurement of the SDPA numerator. The common residual can make flattened full-output PCC look good while a small contribution is inaccurate. No saved intermediate norms establish how much amplification or masking occurred here.

## Page, boundary, queue and capture audit

### Instantiated page/position boundaries

- Harness:151–154, with no diagnostic override, allocates `floor((length+32)/32)=8192` page entries, a permutation of0…8191. Cache shape is `[8192,4,32,256]`, 512MiB per BF16 K or V tensor. Each request's page-table row occupies32768 bytes. The decode token uses virtual page8191, offset31, within that allocation.
- The final prefill call starts261632 and contains511 real rows, padded to512. Its sliced fill table starts at virtual page8176 and has16 entries. The padded zero K/V row occupies final-page offset31; decode overwrites precisely that row. At short length4097, final prefill starts4096, is padded1→32, and decode overwrites final-page offset1. These are corresponding safe padding/update patterns, not evidence of an extra context token.
- The shared page mapper computes physical tile index `physical_page*32 + kv_head*8 + hidden_tile` using uint32 (`sdpa/device/kernels/dataflow/dataflow_common.hpp:53–68`). At physical page8191, KV head3, hidden tile7 the index is262143. The maximum is safe for these index widths. A permutation changes physical page ownership, not logical causal position.
- Reader, compute and writer derive decode work from the same INT32 position and `get_workload_for_core` (reader:170–197; DC:182–209; writer:145–172). Position262143 is column31 of the last tile; its mask is allzero, as required for a completely valid last tile (`sdpa_decode/.../dataflow_common.hpp:53–55,253–278`). Core0 owns that last chunk.
- Q24 disables half-tile decode (DF:513), so accurate full-tile exponential is selected (PC:397–407). The half-tile path that uses approximate exponentials despite `exp_approx_mode=False` is not reached. Scale truncation to BF16 is exact for1/16.

### Queue ledger

For a prefill Q tile, each KV iteration produces/consumes eight K tiles and eight V tiles. QK produces one tile; after exponentiation PV consumes it and the explicit pop drains it (PC:1980). K is consumed by the QK matmul; V by the PV matmul (matmul helper PC:1066–1155). First-iteration state is initialized rather than read from the previous query; aliases reset at PC:1742–1748 and `processed_k_chunks=0` at1771. Later merges consume old stats/output and produce new ones; final reciprocal/normalization drains retained output/sum and pops the final maximum (PC:2133–2141). No unmatched last iteration was found.

The longest prefill scan moves `8192*8=65536` tiles through each K/V CB, making counter wrap worth checking. Blackhole wait logic deliberately uses uint16 modular differences (`tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h:23–33`, neighboring `llk_io_pack.h:33–43`), as does dataflow reserve logic (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:411–424`). Each outstanding CB occupancy remains bounded by its small capacity. An exact65536 cumulative count alone is not a source defect. Decode moves only8192 K/V tiles per core at this extent.

Decode has distinct position CBs for the writer and compute (c8/c15), filled by the reader (reader:124–151; DF:650–664); the historical two-consumer position race is already addressed. GQA6 output writing loops through individual rows (`sdpa_decode/.../dataflow_common.hpp:496–523`), so the writer's stale power-of-two comment does not establish a live six-head write bug.

### Completion and trace boundaries

Harness:168–181 synchronizes before and after prefill; selected outputs are downloaded afterward. `TT_PROGRESS ... enqueued` alone is not completion evidence, but the subsequent synchronization and `to_torch` calls are. The metric therefore does not merely inspect queued work.

Harness:221–232 runs a warm decode, synchronizes, discards its output, captures another decode, then replays with `blocking=True` and downloads the output. The full-attention branch saves/restores no cache tensors. Here warm/capture/replay use the same input, RoPE, INT32 position and request state; each overwrites the same K/V slot with the same projections. There is no append-style position advance, so the absence of cache restoration is not itself a semantic bug for this fixed-input full-attention test. Relevant Q/K/V/position/page-table addresses are registered `Buffer*` bindings, not literal stale pointers (DF:1040–1045).

The long harness nevertheless **does not compare eager decode with captured/replayed decode**: warm output is discarded. A trace-specific additional error remains possible, but cannot explain the prefill discrepancy. Saved short trace-reuse tests do not establish this long boundary.

Target-specific math audit: full-tile accurate exp applies scale then initializes exp immediately before use (PC:401–407). Blackhole FP32 accurate exp selects the replay implementation unless disabled (`ckernel_sfpu_exp.h:707–735`); its replay region is statically bounded away from matmul replay (`:566–596`). The maximum-correction first-column polynomial and tree-correction helper are distinct paths (`experimental/llk_sfpu/ckernel_sfpu_sdpa.h:193–244`). Reciprocal explicitly initializes its own constants before consuming them (PC:317–331). No concrete stale-state/format/constant contradiction was established in these selected phases. Source inspection cannot certify silicon timing or instruction scheduling.

## Candidate ledger after claim review

| Candidate | Confirmed fact / reached preconditions | Fit to observations | Remaining uncertainty / smallest falsifier |
|---|---|---|---|
| Native running-state quantization | BF16 numerator both paths; BF16 sum/scores decode; TF32 reload of Float32 prefill state; thousands of reached serial merges | Predicts context sensitivity and can affect prefill before tracing; distinct formats/inputs permit different prefill/decode error magnitudes | No real same-input intermediate comparison. Freeze TT Q/K/V and compare standalone native attention to FP64, then shorten only the reduction chain |
| HF oracle numerical error/backend variation | Default backend unpinned; current small CPU signatures dispatch Flash; oracle has BF16 boundaries | Can bias both metrics, especially long attention | No source logical bug found. Compare default/forced math/FP64 on exactly the same projected tensors and selected absolute rows |
| Page/last-position defect | Padding and position arithmetic are reached but align consistently | Could in principle affect maximum context | No contradiction identified. Compare unpaged native decode on gathered same K/V, and inspect distinctive final-page markers |
| Trace-specific error | Only traced long decode metric saved | Could add decode-only loss | Cannot explain bad prefill rows. Compare saved eager/capture/replay outputs with identical frozen inputs |
| Upstream projection/norm/RoPE or downstream gate/MLP sensitivity | Numerical differences possible at BF16 boundaries; equations match | Can perturb logits or amplify a small attention error | No local failing intermediate saved. Compare matched-input stages before assigning blame to SDPA alone |

The first row is the leading explanation because it has explicit reached storage/reload boundaries and a direct length-dependent work count. It remains a diagnosis to test, not a proven successful repair. The report does not promote a speculative address, replay, half-tile, count-wrap, or HF-causality story merely because it can cause generic numerical error.

## Smallest discriminators and intervention boundary

These are recommendations for the hardware-owning caller; none was run in this investigation. Keep the advertised context and .995 threshold.

1. **Localize with identical inputs.** Preserve BF16 Q/K/V immediately before the native SDPA call (plus page table and position), and the native output before gate/o-projection. Compare attention alone with FP64 attention over those exact values and the correct absolute causal mask. First check the oracle on HF's own projected BF16 Q/K/V using default CPU SDPA, forced `SDPBackend.MATH`, and explicit/forced FP64. Process a few queries or one KV head at a time to avoid a quadratic full-output reference. Record per-head relative L2, absolute error, output norm/scale and PCC. Then compare TT-vs-HF Q, K and V separately; distinguish cache content divergence from native attention math.

2. **Retain the same full extent; change only K chunk32→128→256.** For native decode the local serial chain becomes1024→256→128 with the same eight cores/head and three tree rounds. These sizes are valid with32-token pages: the wrapper requires a power-of-two multiple32 and the reader maps each tile across pages (`sdpa_decode.cpp:135–143`; device validation:454–467; decode dataflow helpers:755–763,837–845). For prefill use an aligned physical query chunk, e.g. final chunk start261632, divisible by all three sizes; scalar chunk-start validation requires divisibility by both Q and K chunk sizes (`sdpa_device_operation.cpp:389–403`). Check increased L1 demand. Improvement supports reduction sensitivity but does not isolate numerator versus denominator, nor prove a final repair.

3. **Use exact-output native tests independent of the model.** With zero Q/K, all logits are0. With V allones, every output must be1. With V equal1 only in the first32 keys and0 elsewhere, the output at absolute query position p must be `min(32,p+1)/(p+1)`, or `32/(p+1)` for late queries. After verifying marker/cache reads and transfers, the late-query marker keeps the unnormalized numerator exactly32, removing growing numerator accumulation as a confound and targeting denominator/normalization handling. A failure can still involve reciprocal or final normalization rather than the sum recurrence alone. Allones separately exercises numerator accumulation and cancellation. Keep batch/head/page/grid geometry, test both native prefill and decode at full extent, and vary chunk count. Use absolute/relative error, **not PCC for constant vectors**. Marker values or page-index encodings also test physical-page gathering. An identity/permuted table comparison must permute physical K/V consistently so logical attention stays identical.

4. **Compare eager and trace directly.** Save the synchronized warm result, compare it with capture/replay on the same frozen values, and repeat replay without changing position. Full attention's same-slot K/V update is idempotent here. This can adjudicate a decode-only trace contribution cheaply after prefill; it does not require another full-model prefill for every variant if cache/input snapshots are retained.

5. **Only then test a precision repair at the native reduction boundary.** A candidate state-precision repair should consider persistent numerator, maxima/correction state, and denominator intake together; decode additionally needs consistent child transfer/receive buffers. This is an experiment, not a finding that every state must change to meet .995. Widening CBs alone still permits TF32 FPU reload quantization: verify unpack modes and the consuming copy/add/rescale primitive together, or use an explicitly full-FP32 destination/SFPU update. `UnpackToDestFp32` is not a descriptor-only remedy: its contract makes that buffer incompatible with SrcA/B (`tt_metal/api/tt-metalium/base_types.hpp:33–38`), which the existing binary/matmul helpers consume. Transfer offsets, byte sizes, writer/reader CB formats and L1 sizing must agree. Do not simply upload FP32 Q/K/V: native decode validation supports BF16/BF8/BF4 inputs, and its intermediates are hardcoded regardless. Changing only `fp32_dest_acc_en`, only the denominator CB, or merely enabling `packer_l1_acc` does not address the demonstrated storage chain. Larger K chunks or blockwise/tree partial reductions are narrower configuration/algorithm experiments that preserve full context; neither is certified sufficient without evidence.

Once a local change is justified, rerun the current-source4097/4098 and262143/262144 cases with a fixed shared input prefix and fixed decode token, report all selected contribution rows, and retain the current finite-row check while adding a threshold for every required contribution row. Also compare matched position-modulo32 cases, e.g. decode positions4095 and262143, to distinguish context length from final-tile occupancy. Compare raw attention and full decoder separately. The current long prefill acceptance gate should enforce whichever per-row contribution contract the caller intends rather than relying on the flattened residual-containing statistic.

## Other potential issues and limits

- The long prefill assertion checks `prefill_pcc >= .995` and finite output, not per-row contribution fidelity (Harness:198). The decode assertion checks PCC only (Harness:237). These are evidence/acceptance gaps; they do not generate the observed finite numerical error.
- `WormholeComputeKernelConfig` is a compatibility alias of the universal config (`compute_kernel_config.hpp:18–39`), so its name is not evidence of a wrong-architecture config on Blackhole.
- Nearby native tests do not establish this exact configuration. `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_prefill.py:348–375` has a128K high-precision case with Q256/K128 (1024 KV chunks), approximate exp in its helper, and an RMSE gate. Its Q32/K32 case only has32 tokens (:311–325). `test_sdpa_chunked.py:241–278` uses16K, larger chunks and BF8 KV; `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_standard_exp.py:19–29` exercises only one/two KV chunks. These cover useful subcontracts, not8192 serial chunk updates at262K with this paged BF16 setup.
- Older linear-attention inverse, gated-normalization, or folded-scale reports in the parent evidence directory concern different paths or earlier source snapshots. They do not establish the cause of this full-attention failure.
- The current evidence contains no intermediate native attention values, exact runtime backend selection, long eager decode comparison, or stage-level sensitivity measurements. It cannot quantify the relative contribution of BF16 numerator, denominator reload, score/maximum rounding, upstream errors or oracle error. No hardware fix or passing result is claimed.

## Inspection hashes

```text
99046b9c3cf1b4a2ec3e2cdfa1332e327a57a5d899f3ae98144e796b7948dd43  tt/functional_decoder.py
41af9795c2672ee4ed1827172b0de228e45a8965af16594111752a1e44b34410  tests/run_long_context.py
2095b71249564dcf97701ad96fe0f66faa51e67941591b9ef366bcde419d3ec7  tests/evidence.py
65c044642f541f8573d95ab67d1be2e6f73eb78d5fbcd19eaa39fb495cc8a6b5  doc/hf_config.json
6996d3390b976f9c1e5cbe99acde63fedf2120fe23d5b78f6576328cf3f2a0e4  doc/functional_decoder/full_context_262143_retry.json
```

The first draft was revisited against the selected code paths; findings without a reached contradiction or a sufficient causal chain were retained as alternatives/test gaps rather than promoted to established root causes.
