# Minimal prefill projection integration patch

`minimal_prefill_projection.patch` is **unapplied**. Base runtime SHA256 is `3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9`; applying exactly this patch produces `032ef220d04a565203993feb4d9c68706cc2b10eb73180a76e7f874469861dbe`. Defaults remain unchanged while the parent completes numerical and timing controls.

The factory gains independent `prefill_qkv_minimal=False` and `prefill_output_minimal=False`, with independent `prefill_qkv_minimal_block_w=16` and `prefill_output_minimal_block_w=16`. Tested K-block choices 4, 8 and 16 are accepted. Example override: `{"prefill_qkv_minimal":true,"prefill_qkv_minimal_block_w":16}`; the output backend can be enabled separately or combined after validation.

`MinimalPrefillProjection` owns four setup-time config objects: M block 1/2/3/4, selected as `min(4, padded_M_tiles)`, K selected independently per role, N block 8, subblock 1×4, grid 11×8. Config coverage does not impose a maximum public sequence length or require runtime config creation. The existing decoder chunks prefill; the helper can also select the same M4 config for larger operands. Weight and compute objects are retained by reference, without recasting or uploading tensors.

`MinimalPrefillQKV` delegates logical M=1 to the exact existing projection callable. At M>1 it reads the original packed BroadcastQKV/TiedQKV weight and its prefill compute config. Thus default QKV remains FP32 activation × BFP8 weight → FP32 with original **HiFi4**, rather than accidentally inheriting decode HiFi2/LoFi. The full-attention TiedQKV tail is appended after projection exactly as before, in DRAM. The wrapper also works around existing lane/compensated decode objects and separate/DRAM decode configuration because their original prefill packed weight is retained. The minimal option explicitly supersedes an optional old dense-prefill program; decode choices remain independent.

The output backend reuses the existing prefill head concatenation, optional input-L1 placement, selected prefill output compute config (default LoFi/FP32 destination), BFP8 output weight and FP32 DRAM output. The existing decode `project` path is unchanged. Both backends pass the caller's memory policy directly; there are no test imports, global patches, source execution or host tensor reads in the runtime patch.

## Source legality and capacity

`minimal_matmul_device_operation.cpp:52` accepts BF16/BFP8/BFP4/FP32 tiled operands, singleton leading weight dimensions and matching logical K. Config validation at line 253 requires positive blocks, subblock divisibility and destination volume, plus a grid fitting the device. N8/sub1×4 fits FP32 destination. Unlike the previous 1D decode matmul config, K16 need not divide H/32=88: `minimal_matmul_program_factory.cpp:269` rounds the K tile count up to a block multiple, so K88 uses six K16 blocks with a partial last block. H2816 and full packed N9216 are legal shapes; no physical weight tensor padding is introduced by this wrapper.

The factory's explicit circular-buffer inventory is at lines 309–355. Without bias/ternary/SwiGLU, it allocates double-buffered A and B, double-buffered output, and one intermediate block. With M4/N8, FP32 output/intermediate tiles are 4096 bytes, BF16 input tiles 2048 bytes, and BFP8 weight tiles 1088 bytes:

| Role/config | A CB bytes | B CB bytes | Output CB bytes | Intermediate CB bytes | Total per participating core |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV FP32×BFP8, K16 | 524,288 | 278,528 | 262,144 | 131,072 | 1,196,032 |
| QKV FP32×BFP8, K8 | 262,144 | 139,264 | 262,144 | 131,072 | 794,624 |
| Output BF16×BFP8, K16 | 262,144 | 278,528 | 262,144 | 131,072 | 933,888 |
| Output BF16×BFP8, K8 | 131,072 | 139,264 | 262,144 | 131,072 | 663,552 |

These are source-derived CB totals, not measured total L1 use: kernel code, semaphores, runtime arguments and other live L1 allocations are additional. Smaller M blocks reduce A/output/intermediate CBs; the B CB is unchanged. Explicit alternative weight dtypes change B tile bytes and require their own capacity check. Existing real-weight headline probes establish that the selected BFP8 K16 shapes compile and execute; no claim of arbitrary-dtype capacity is made.

Persistent expert weights, cache pages, RoPE tables and context capacity are unchanged. Both roles reuse existing weight tensors; the only additional persistent objects are small Python wrappers/configs. Output logical shapes and DRAM tensor bytes are unchanged, including full-attention tied K duplication. Final maximum/near-maximum tests must still validate the cumulative selected numerical policy after integration.

## Verification and evidence status

`git apply --check` passed. Candidate syntax was parsed without importing TTNN. New code conforms to Black at line length 120; an unrelated pre-existing normalization ternary reflow was deliberately omitted from the patch. `minimal_prefill_projection_cpu_checks.json` records CPU-only helper mocks covering 28 tied/untied row cases from 2 through 262144, tile tails, setup-only config construction, exact M1 decode delegation, weight/compute identity, requested memory and tied duplication. These mocks verify orchestration only; they are not accelerator or numerical evidence.

The parent's `minimal_qkv_k16_layer{0,5}.json` probes pass the actual headline workload and improve whole-prefill host medians versus matched `minimal_baseline_layer{0,5}.json`; full K8 fails one headline decode check near PCC 0.9949893. QKV stress and output alternating timing controls are separate parent-owned evidence. This patch keeps both new defaults off until that selection is complete.
