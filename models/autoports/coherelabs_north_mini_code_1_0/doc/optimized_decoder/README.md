# North-Mini-Code-1.0 optimized decoder

This stage adds the single-device optimized decoder for
`CohereLabs/North-Mini-Code-1.0` revision
`d11e61a842617a22dc328552fa5bb86231ee4f37`. Scope is limited to
`tt/optimized_decoder.py`, its tests/performance harness, and decoder
documentation. No multichip, full-model, or vLLM work is included.

## Result

The functional sparse-MoE path evaluated all 128 experts. The optimized path
routes on device and executes only the active top-8 expert union with
`ttnn.sparse_matmul`. Decode packs same-input gate/up weights into one sparse
projection and splits on device. Packed QKV, paged cache operations, native
SDPA, per-token sigmoid routing scores, and trace-safe stable inputs remain.

The selected policy is:

- BFP8/LoFi attention weights and BFP8 KV cache;
- BFP4/LoFi dense-layer gate/up and BFP8/LoFi down weights and matmuls;
- two separately tuned 48-core, 8x6 dense gate/up projections at batch 1,
  where they beat every correct packed candidate; a packed 64-core 1x3
  projection at serving batch 32; dense down uses a 32-core interleaved 1x2
  program at batch 1 and a DRAM-width-sharded program at batch 32;
- BFP8/HiFi2 decode and small-prefill experts;
- BF16/HiFi4 large-prefill expert matrices with FP32 destination accumulation;
- packed decode gate/up and shared decode/prefill down storage;
- batch-specific expert programs, plus a 32-core width-sharded residual/RMSNorm
  chain at both decode batches;
- an exact `nnz=8` sparse-expert count at one-token decode, derived from an
  on-device top-k presence mask; batch-32 unions remain dynamic;
- DRAM-width-sharded QKV/O weights and decode matmul programs at batch 1;
  serving batch 32 retains interleaved attention because its sharded candidate
  failed the exact correctness gate;
- arbitrary public logical sequence lengths, internally padded/chunked by 32;
- a 1024-token dense-expert composite for total prefill M at least 1024, with
  the functional-compatible router accumulation policy.

There is no functional-runtime fallback. Measured `prefill_forward` and
`decode_forward` contain no Torch conversion, host round trip, or CPU fallback.

## Warmed latency

Official real layer-1 weights were used. Decode is traced replay; prefill is
warmed and untraced. The pre-review final JSONs record argv, effective
policy/program configuration, Git state, model revision, and source SHA256.
Their manually attached PCC fields are not treated as exact correctness
bindings unless the evidence workload matches.

| workload | functional | optimized | change |
|---|---:|---:|---:|
| prefill, batch 1, sequence 128 | 14.848 ms | 13.228 ms | 10.91% faster |
| prefill, batch 32, sequence 128 | 147.841 ms | 135.064 ms | 8.64% faster |
| traced decode, batch 1 | 9.570 ms | 0.604 ms | 93.69% faster |
| traced decode, batch 32 | 11.155 ms | 3.846 ms | 65.52% faster |

The primary batch-1 result beats the best correct batch-1 baseline and does
not regress either prefill batch or serving decode. Final records are
`artifacts/final_{prefill,decode}_b{1,32}.json`; all four bind to optimized
source SHA256
`21d222646cffcb8c09c0fba1b60b0c1f30f117e6dc6a53b49b3f1af340602ecd`.
Their correctness bindings are deliberately null: PCC is reported from the
exact test logs below, never borrowed across workload or batch.

## Correctness, determinism, and capacity

The final aggregate watcher run passes 30 tests with five explicit opt-in
skips in 245.36 s; `watcher_clean_final.txt` contains the complete log.
It was captured from source `d4665e8a`; the committed source adds only the
explicit `MODEL_ID` module export required by the performance harness. No
runtime method changed, and the import smoke plus all final warmed/profile
runs bind exactly to `21d22264`.
The 4096-token populated-history trace is also watcher-clean in its required
fresh process. Three other skips are isolated precision probes and one is the
costly exact-context prefill, each with a dedicated retained log.

| coverage | result |
|---|---|
| dense/full/forced-RoPE prefill, logical lengths 1/31/33/65 | PCC 0.99914 / 0.99629 / 0.99631 / 0.99528 |
| sliding/RoPE/sparse-MoE prefill, lengths 33/65/1025 | PCC 0.99922 / 0.99933 / 0.99996 |
| full/no-RoPE/sparse-MoE prefill, length 33 | PCC 0.99925 |
| selected real-weight traced decode, batch 1 | PCC 0.998268 |
| selected real-weight traced decode, batch 32 | PCC 0.997431 |
| populated 4096-token sliding history | PCC 0.999130; watcher-clean fresh process |
| official real-weight trace | five bitwise-identical replays |
| exact real-weight batch-32, sequence-128 sparse prefill | optimized/functional 0.999871; optimized/HF 0.989082; functional/HF 0.988945 |
| paged-cache nonzero slots | key/value PCC at least 0.99980 |
| advertised context | optimized batch-1 prefill at 500000 returns expected finite output; batch-32 cache+weights coexist and last-position sparse decode executes; batch-1 all layer kinds execute |

The official router has close eighth/ninth expert boundaries. The prior
HiFi4/FP32 router changed some expert memberships and reached only 0.982107
direct optimized/functional PCC. Retaining the optimized 1024-token expert
program but using the functional-compatible router kernel restores direct PCC
to 0.999871. TTNN/CPU route-set agreement is 0.87179, while optimized/HF
still exceeds functional/HF.

BFP8 cache passes the official-weight populated-history control at PCC
0.999373 (matched BF16 cache: 0.999373). Its physical batch-32/context-500000
footprint is 17.408 GB. Cache plus all optimized sparse and phase-specific
BF16 prefill weights is 19.702 GB on a 34.179 GB device. BF16 cache would
exceed physical capacity with those required weights. The advertised
500000-token capability is preserved; exact accounting is in
`../context_contract.json`.

The canonical exact optimized 500000-token batch-1 prefill watcher test
completed its pytest call in 331.01 s (332.92 s elapsed). The context contract
records both its evidence hashes and the final source/test hashes, together
with their audited import/export-only equivalence. The public contract is
unchanged, including nonaligned logical lengths.

## Operation-topology audit

| current topology | candidate | action | evidence |
|---|---|---|---|
| packed same-input Q/K/V | split Q/K/V | retain packed QKV | one QKV matmul in final Tracy rows |
| all-BFP8/HiFi2 dense MLP | BFP4/LoFi gate/up and BFP8/LoFi down | select lower precision | PCC 0.995275/0.998849/0.998981 remains above the 0.995 functional floor; faster at all four b1/b32 prefill/decode workloads |
| packed same-input dense gate/up | two equivalently tuned projections | select separate at b1, packed at b32 | b1 0.19694 ms versus best correct packed 0.19884 ms; native packed wins b32 at 0.76639 ms |
| generic dense down matmul | 32-core 1x2, block-w 8 | select b1 | 55.6 us to 18.8 us; warmed dense b1 0.2436 to 0.2075 ms |
| interleaved dense down | DRAM-width-sharded down | select b32; reject b1 | 1.4983 to 1.0044 ms b32; 0.2066 to 0.2212 ms b1 |
| packed dense gate/up K blocks 2/4/8/16/32 | native block 2 | select b32 | b1 legal block-4 retry 0.19884 ms still loses to separate; b32 block 4 is 0.97227 ms and blocks 8+ exceed L1 after corrected DRAM retries |
| separate same-input expert gate/up | packed sparse gate/up | select for decode | one active-expert `2048x1536` projection |
| dense all-128-expert decode | routed sparse experts | select | 9.570 ms to 0.604 ms at batch 1 |
| runtime-inferred sparse active count | exact top-8 presence at one-token decode | select b1; retain dynamic b32 | 0.679 ms to 0.604 ms b1 at PCC 0.998268; official b32 union is 87 |
| repeated routing | one top-k/scatter feeding both expert projections | select | one routing group in final path |
| primitive attention sequence | native paged SDPA | retain | native SDPA/cache rows in final profile |
| interleaved decode RMSNorm/residual | 32-core width-sharded RMSNorm/residual chain | select at b1 and b32 | PCC 0.998268/0.997431 in selected chain; 0.604/3.846 ms final |
| interleaved QKV/O weights and generic decode matmuls | DRAM-sharded QKV/O weights with decode program configs | select b1; reject b32 | b1 PCC 0.998264; b32 legal retry reached 3.583 ms but failed PCC at 0.982460 |
| separate same-input large-prefill gate/up | packed BF16 gate/up plus on-device split | reject | legal 8x8 retry: 142.395 ms b32 versus 135.011 ms split; b1 path is below the 1024-token applicability threshold |
| routed sparse expert matmuls | dense DRAM-sharded expert family | reject only for the expert path | family has no active-expert operand and restores all-128-expert traffic; measured dense baseline is 9.564 ms |
| generic decode router matmul | explicit 2-core 1-D router program | reject | PCC 0.998225/0.998236; 0.710 ms b1, slower than selected chain; combining it with the sharded chain requires batch fusion |
| fixed public chunk multiple | internal padding/chunking | select | logical 1, 31, 33, 65, and 1025 pass |
| optimized HiFi4/FP32 prefill router | functional-compatible router with optimized M=1024 experts | select | direct optimized/functional PCC 0.982107 to 0.999871; 135.064 ms final |
| sparse large-M prefill | chunked sparse versus dense composite | select dense composite | sparse chunk-1024 646.509 ms; selected 135.064 ms |

Required tilize/untilize and reshapes remain only at routing/top-k layout
boundaries. There is no unnecessary runtime `torch`, `from_torch`,
`to_torch`, `.cpu()`, `.numpy()`, or functional decoder call.

## Candidate evidence

Decode candidates were measured independently at batch 1 and serving batch
32. A faster timing is not accepted without its batch-specific correctness
gate.

Official dense layer-0 precision evidence is batch-specific and uses the
current topology. The selected BFP4/LoFi gate/up plus BFP8/LoFi down policy
scores 0.995275 prefill and 0.998849/0.998981 decode b1/b32. The
BFP8/HiFi2 control raises PCC to 0.999475/0.999343/0.999490, but the selected
policy still meets the functional 0.995 floor and is faster at all four
workloads: 0.582/12.278 ms prefill and 0.197/0.767 ms decode, versus
0.581/12.311 and 0.207/0.843 ms for the control. Functional timing is
0.629/13.691 and 0.356/6.648 ms. The material prefill PCC delta is accepted
because it remains above the stated bar while delivering the required
lowest-correct-precision policy. Exact evidence is in
`artifacts/final_dense_selected_precision_correctness.txt`,
`final_dense_bfp8_hifi2_control_correctness.txt`, and their source-bound JSONs.

Dense topology/program evidence:

| candidate | batch 1 | batch 32 | decision |
|---|---:|---:|---|
| selected separate BFP4/LoFi, 48-core 8x6, block 16 | 0.19694 ms | — | select b1 |
| native packed BFP4/LoFi block 2 | 0.20669 ms | 0.76530 ms | select b32 |
| packed block 4 with interleaved input and DRAM output | 0.19884 ms | 0.97227 ms | best packed b1 control, still slower; reject b32 |
| packed block 8 | 0.19892 ms | L1 allocation failure after DRAM retry | reject |
| packed block 16 | 0.19994 ms | L1 allocation failure after DRAM retry | reject |
| packed block 32 | 0.20110 ms | L1 allocation failure after DRAM retry | reject |
| selected final full topology | 0.19694 ms | 0.76639 ms | select |

The packed dense composite was not rejected on its first error:
`ttnn.split` failed in the device compiler, while two legal `ttnn.slice`
operations passed prefill and traced decode. Larger packed K blocks were
retried with an interleaved input and DRAM output; block 4 passed exact PCC
(0.995275/0.998898/0.998974), while the larger serving retries exposed a hard
L1 circular-buffer allocation limit. The best correct packed batch-1 control
still loses to the equivalently tuned separate family. Likewise, the
DRAM-down retry fixed the dense K dimension, explicitly sharded the M-axis
activation, and was measured at both batches before batch-32-only selection.

| candidate | batch 1 | batch 32 | decision |
|---|---:|---:|---|
| functional all-expert | 9.570 ms | 11.155 ms | correctness/performance baseline |
| prior correct routed default | 0.724 ms | 4.924 ms | baseline before reviewer follow-up |
| sharded residual/RMSNorm | 0.692 ms | 3.845 ms | correct both batches; selected |
| sharded residual plus DRAM attention | 0.678 ms | 3.583 ms | select b1; reject b32 at PCC 0.982460 |
| final batch-aware selected chain | 0.604 ms | 3.846 ms | select: PCC 0.998268/0.997431 |
| BFP8/HiFi2 experts, 16-core | 0.725 ms | 4.214 ms | b32 failed PCC 0.953752; historical b32 JSON marked unbound |
| BFP8/HiFi2 gate 24-core/down 32-core | 0.740 ms | 4.049 ms | reject; historical b32 JSON has no exact bound PCC |
| BFP8/HiFi2 8-core | 0.777 ms | 5.505 ms | reject: slower; historical PCC metadata cleared |
| BFP4/LoFi experts with BF16 cache | faster historical kernels | — | reject: populated-history PCC 0.98240 |
| BFP4 attention on selected chain | 0.602 ms | 3.832 ms | reject: exact official-weight PCC 0.981934 b1 / 0.980513 b32 despite small speed gains |
| BF16 cache control | — | — | PCC 0.999373; reject on capacity |

Large-prefill 2-D program candidates:

| candidate | batch-32 sequence-128 | decision |
|---|---:|---|
| 8x4 grid | 193.055 ms | reject |
| 8x8 grid, BF16/HiFi4/FP32 expert accumulation | 135.064 ms final | select with functional-compatible router |
| packed gate/up, 8x8 | 142.395 ms | reject: slower than separate projections |

The first API error for a block/subblock candidate was not treated as a
rejection. A corrected legal retry was run; it stalled beyond 60 seconds, the
exact process was terminated, and `tt-smi` showed all four devices healthy.

## Profiler conclusions

Final batch-1 sparse decode contains 0.5755 ms device operations plus
0.0657 ms op-to-op gaps under profiling (0.6412 ms total) versus 0.6037 ms
unprofiled wall latency. Matmuls use 0.2323 ms (36.23% of the profiled
window). The selected DRAM-sharded QKV/O rows are explicitly marked
`DRAM Sharded=True`; complete layout conversion costs about 0.0097 ms.
Modeled roofline utilization is 19.4% (99 GB/s).

Batch-32 prefill contains 129.726 ms device operations plus 1.220 ms gaps
(130.946 ms profiled) versus 135.064 ms unprofiled wall latency. Dense expert
matmuls use 81.413 ms (62.17%). Modeled roofline utilization is 20.9%
(107 GB/s). Operation rows, stacked reports, exact commands, and source
hashes are under `tracy/`; `final_profile_manifest.json` binds the collection.

The official dense layer-0 batch-1 decode profile contains 0.173 ms device
operations plus 0.029 ms gaps (0.202 ms total); matmuls account for 0.107 ms.
Its modeled DRAM roofline utilization is 36.4% (186 GB/s). The necessary
sharded-to-interleaved transition before the winning separate MLP family costs
1.464 us and pays for itself versus packed. At batch 32, the packed gate/up
row is 129.1 us and the selected DRAM-sharded down row is 25.7 us; the
complete window is 0.766 ms. Retained rows and stacked reports are
`tracy/final_dense_decode_b{1,32}_rows*`.

## Reproduction

```bash
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py

NORTH_MINI_LONG_HISTORY_TRACE=1 TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py::test_optimized_sliding_moe_populated_history_dynamic_trace_replay_matches_reference

python -m models.autoports.coherelabs_north_mini_code_1_0.tests.optimized_decoder_perf \
  --mode decode --batch 1 --layer 1 --candidate default --real-weights \
  --warmups 10 --iterations 50
```

See `work_log.md` for the complete optimize checklist, command patterns,
review record, and local commit SHAs.
