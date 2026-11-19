# Prefill correction pair: legal interleaved width blocking

Source-only follow-up to the measured rejection of decode packing (`pair99`). No device was opened. Candidate: `candidates/prefill_correction_pair.py`, frozen over `delivery118.py` by `make_prefill_correction_pair.py`. Original/main files remain untouched.

## Why the proposed DRAM-sharded width33 configuration is unsafe

The existing defect diagnosed in `AUTODEBUG_blocks.md` remains present. The 2D factory sets B buffer width to `out_block_w`, but `DramWidthShardWalker` provides spans covering `per_core_N`. The DRAM-width-sharded reader reads all those spans with the smaller destination stride and resets the same origin for each internal N block. Its explicit comment says one width block is assumed. Thus width33/per-core99 is not a legal repair for DRAM WIDTH_SHARDED B, regardless of validation succeeding. No shared kernel changes are proposed.

A concrete alternative exists: preserve the packed pair as **DRAM-interleaved** B. In `reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:514`, the interleaved branch invokes `read_in1_block` with the actual block width and current N-block origin. At line731 it advances that origin by the width-block stride. This supports the intended three N blocks and final partial width. `matmul_device_operation.cpp:1609` also requires `out_block_h=1` when width is blocked; the candidate complies. `per_core_M=2` still processes512 rows over eight active grid rows; it is not an output height-block size of2.

## Exact setup and runtime changes

Setup concatenates the existing already-quantized TILE BFP8 primary and residual (each5120x16512), then pads columns33024→34816 with zeros. Both term boundaries are whole tiles. There is no dequantize/requantize step. The pair is held outside `self.weights` so the normal setup sharding loop cannot convert it to the unsafe reader layout. Optional `prefill_pair_verify_setup=True` reads it back only at setup and requires exact dequantized equality of both terms and zero padding. Existing terms remain intact for decode and controls.

Only the longer-sequence `_gdn` input projection calls the new helper. Compact prefill S1 returns earlier; `_gdn_compact`, `_linear_decode`, `_linear`, public prefill/decode methods, all state/cache math and the full-attention branch remain unchanged. The opt-in is `prefill_pack_input_correction=True`; defaultFalse restores the original projection call. No mutable phase flag is used.

Runtime flattens logical B*S once, chunks at a configurable512 logical rows, executes one FP32-output packed matmul per chunk, slices primary/residual halves, and adds them in original order to BF16. Concatenation occurs after that BF16 addition. Tails are internally padded by the existing tiled ops; there is no logical sequence alignment restriction. The helper supports B1/B3/B32 and arbitrary positive logical lengths, including1/17/33/65/511/513. Existing public prefix chunking still applies.

Default `prefill_pair_config`: grid[11,10], physical_n34816, row_cap512, in0_block_w4, out_block_w33, subblock_w1. This yields per_core_N99, three internal N blocks, out_block_h1, and per_core_M2 for512 rows. The final active grid column has98 real N tiles, so its tail occurs in the third internal block. K8 is the first additional candidate. Width99/full-worker and width11 are legal controls under the interleaved reader. Changing grid/physicalN requires choosing a width dividing the resulting per_core_N. The old >20000→256 row cap is bypassed only in this dedicated helper.

The additional persistent packed BFP8 tensor occupies180.625MiB (160*1088 tiles at1088bytes/tile). It is an opt-in experiment, not a context-contract change. For512 rows, the joined FP32 output is68MiB; two cropped FP32 halves add64.5MiB and BF16 result16.125MiB. This runtime movement is the principal risk: packing saves a dispatch and repeated input setup but does not guarantee an end-to-end win. The factory-derived per-worker CB formulas below are estimates, not hardware allocation claims:

- B double buffer: 2*Kblock*33*1088bytes.
- A double buffer: 2*Kblock*2048bytes.
- Two33-tile FP32 output/partial buffers: 2*33*4096bytes.

These total573,952 bytes atK4 and877,568 atK8 before other factory buffers; actual allocation must be measured. No latency or numerical pass is claimed.

## Required comparison and evidence

Run the existing decoder harness with delivery118's exact selected precision/compute policies, changing only this candidate and opt-in. Enable setup verification. Compare B1/S512 and B32/S33 warmed prefill against the tuned **separate** delivery118 control, not against older untuned rows. First test K4,width33; then K8,width33 and full-width99 if legal L1 allocation permits. Keep identical attention fidelity and packer policy. Measure slices/add in profiler as part of the complete path. Decode must retain the existing traced control; it should be source-identical.

Require unchanged prefill/decode PCC, state gates, repeated equality, and the512-step stress gate because a different K-block or accumulation schedule can shift prefix state despite identical weight pages. Include B1/S513 and B3/S33 (or B32/S35) for arbitrary tails. Reject packing if it fails those gates or loses total warmed prefill. A first API/config error warrants source-based repair, not dismissal of the idea.

CPU verification: `python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_prefill_correction_pair_source.py` passes180 geometry/tail cases, deterministic generation/hash checks, exact untouched decode/compact/prefill-method AST, and restoration of original `_gdn` AST by replacing the one call. `py_compile` passes. Results: `prefill_correction_pair_cpu_checks.json`. Device concat equality, correctness and performance remain pending.
