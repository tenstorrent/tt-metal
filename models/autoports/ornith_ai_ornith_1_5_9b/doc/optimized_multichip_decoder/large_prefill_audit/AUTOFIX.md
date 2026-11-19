# 16K/32K prefill chunks with DRAM activation residency

Status: source-only candidate prepared; no device compilation, reference generation, execution or measurements performed here. Parent reports lifetime-only 4096 passes all 13 lengths through 16385 and DRAM-first normalization makes 8192 pass both layer kinds. Selective 8192 and max-context jobs remain separately owned by the parent.

## Diagnosis and source constraints

Simply returning a DRAM BF8 norm tensor is insufficient. Frozen `tests/multichip_sdpa_reference.py:538–545` immediately converts every nondecode projection input to L1. A coherent large-chunk family must keep the norm output in DRAM **and** remove that projection-side copy for input, attention-output and MLP-down projections. The latter two inputs remain BF16; this candidate does not reduce their precision.

`minimal_matmul_device_operation.cpp:165–182` requires TILE inputs and explicitly accepts BF16/BF8/BF4/FP32. It imposes no L1-only activation constraint. `device/kernels/dm_in0_sender.cpp:75–77` constructs its input TensorAccessor; `matmul_dataflow_common.hpp:91–151` reads tiles through that accessor into bounded L1 CBs. DRAM BF8 is a supported source configuration, rather than an API workaround.

`minimal_matmul_program_factory.cpp:265–267,355–368` sizes scratch by M/K/N blocks, independent of total sequence length. At M4/N8, K16 -> K8 reduces in0/in1 CBs while retaining output/intermediate sizes:

| Input / projection | K16 CB bytes/core | K8 CB bytes/core |
|---|---:|---:|
| BF16 input, BF8 weights, ordinary BF16 output | 802816 | 532480 |
| BF8 input, BF8 weights, ordinary BF16 output | 679936 | 471040 |
| BF8 input, BF8 weights, fused SwiGLU BF16 output | 614400 | 405504 |

These are the four principal CBs, not total static L1 allocation; semaphore/other overhead is additional. Formula: `2*M*K*input_tile_bytes + 2*K*N*1088 + 2*M*N*2048 + M*N*4096`, with the output term halved for SwiGLU. BF8 tile storage is 1088 bytes. Changing K can change intermediate accumulation/packing behavior, so each K choice needs the unchanged PCC gates.

A 4096-wide normalized BF8 tensor alone occupies 71303168 bytes at 16384 rows and 142606336 at 32768 rows. The corresponding BF16 norm intermediate is twice the element count, before quantization. Keeping both in DRAM avoids imposing these full-tensor allocations on L1 alongside static matmul CBs. This changes bounded transient residency, not cache/recurrent storage or advertised context.

The factory transposes its grid when M>N (`:291–310`) and includes that choice in the program-cache hash (`minimal_matmul_device_operation.cpp:104–126`). Large and tail chunks cross this boundary, so boundary tests must exercise cached programs across both orientations; do not hardcode a square-grid assumption.

## Other consumers

- Packed MLP calls minimal_matmul directly on `h` (`multichip_sdpa_reference.py:614–622`), so it already accepts the new DRAM BF8 norm without a second edit. Inherited `from_state_dict` applies the actual `prefill_program` K8/K16 to this call and all ordinary projection calls. The old `--output-k-block` option alone does not change the control role's minimal config.
- Attention consumers see the same BF16 DRAM projection result as before. The QKV convolution requires BF16 TILE (`qkv_causal_conv1d_silu_device_operation.cpp:45–46`); its public implementation defaults outputs to DRAM (`qkv_causal_conv1d_silu.cpp:38`). No BF8 tensor is passed directly to that convolution.
- Linear GDN scratch and default output use DRAM (`chunk_gated_delta_rule.cpp:56,62,85,115,295`); the recurrence chunk remains 32 even when the outer decoder chunk is 16384/32768. No recurrent arithmetic or internal chunk-size change is made.
- Full-attention Q/K/V remain interleaved. SDPA accepts DRAM/L1 interleaved operands (`sdpa_device_operation.cpp:50`). RoPE/cache fill, the existing q32/k128 SDPA program, head concatenation and gating remain unchanged. They are not promised to scale optimally: a new failure or dominant cost in these consumers requires a separate measured adaptation.
- The existing large-prefill reduction uses BF8 payloads in DRAM, then restores BF16 residuals (`multichip_sdpa_reference.py:390–407`). This avoids reinstating a large L1 residual contract after the projection.

## Isolated implementation

New file: `tests/multichip_dram_prefill_candidate.py`.

- Inherits the existing lifetime candidate's exact prefill loop, releasing input norm `h` before finish and `mixed/chunk` at each iteration end. Public nonaligned lengths, full-attention prefix alignment, padding/masking and output slicing are inherited unchanged.
- Norm rows <=4096 use the frozen implementation. Rows 4097..8192 use DRAM RMSNorm -> DRAM BF8 -> L1, matching the selective 8192 control. Rows >8192 keep BF8 in DRAM (`:38–51`). Head norms remain unchanged.
- Projection rows >8192 use minimal_matmul with DRAM input/output (`:53–66`). Existing compute fidelity and output BF16 are retained. Smaller tails use the prior implementation; one-row tails retain decode dispatch.
- `--minimal-k-block 8|16` sets the actual inherited prefill program. `--dram-threshold` defaults to 8192 and must be >=4096.
- The wrapper requires `--roles control`, replicated residuals and no fused-prefill alternative. These guard against unintentionally mixing other geometry families. It patches bindings only within its process; no production, frozen reference or queued module was edited.
- OriginalGeometry is captured before patching the module binding, avoiding recursive fallback from norm into this subclass.

## Jobs and gates

`../dram_prefill_jobs.json` contains 12 fresh jobs, in dependency order:

1. Two unmodified `multichip_probe --baseline` jobs generate fresh **OptimizedDecoder single-chip** references, one per layer kind. They stop the queue on failure. These are correctness references only; they do not constitute multichip completion evidence.
2. Two matched 8192/K16 control jobs on the target TP4 mesh.
3. Four 16384/32768 K16 jobs and four corresponding K8 jobs, both layer kinds.

Every job uses the same lengths and seeded input stream:

`8191,8192,8193,16383,16384,16385,32767,32768,32769,65535,65536,65537`

The new references are `generated/multichip/dram_prefill_boundaries_{0,3}.pt`; they do not exist yet. No existing reference is overwritten. Every candidate retains full-row prefill PCC, contribution PCC, every replicated rank's output/contribution PCC, four decode steps, and recurrent/conv or KV state checks at .995. Candidate timing uses 3 prefill warmups, 5 measured repeats, and the existing traced warmed decode benchmark. Baseline generation skips redundant timing; all candidate comparisons have identical measurement settings. Existing trace controls and device-only guards are unchanged. Watcher and final max-context promotion gates are still required separately after a candidate passes.

Exact launch command for the parent's serialized hardware queue (not executed here):

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/dram_prefill_jobs.json
```

Do not launch until the parent's queue/STOP protocol permits. The stage runner snapshots sources, serializes devices and refuses existing command records. Each job has a 3600-second bound; failed candidates may continue through the established recovery protocol. The long full-row references intentionally consume host RAM/disk; they are not sampled approximations.

Host verification passed: AST, wrapper `--help` before mesh opening, inherited prefill function identity, 14 mocked norm/projection boundary cases, 12 unique fresh job labels, identical reference length streams, and correct baseline-before-candidate ordering. See `host_validation.json` and `source_manifest.json`.

No supported-context reduction is proposed. 65537 is a discriminating boundary test, not a replacement for the 262144 contract. A passing candidate must still be measured on the final default, rerun at the existing max-context/nonaligned gates, and have transient memory reflected in the context evidence before promotion. Any L1/API failure is evidence for the next adaptation, not grounds to reject this family after one attempt.
