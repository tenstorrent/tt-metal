# AutoDebug: PLE layer1 seq34 failure predates the writer and compact2 changes

## Finding

**The matched controls rule out a numerical regression introduced by either new feature for this fixture.** The functional baseline also fails row 5. `best_exact_qkv_cleanup`, writer-only, tail-only, and combined produce **byte-for-byte identical complete prefill outputs and all three decode outputs**. Reverting either new feature cannot repair this particular failure.

The leading investigation boundary is the shared PLE path. Its first known intentional numerical difference from literal HF is the packed embedding reconstruction: HF reads exact checkpoint BF16 rows, while TT reads the four-bit/group10/scale8 cache. Whether that difference causes the row 5 gate failure, or whether shared TT arithmetic or BFP8 expert computation contributes materially, is **not yet established**. The available caches contain final outputs, not intermediate device tensors. Row5 is the first threshold failure reported, not an identified first divergent calculation.

No implementation defect in the new writer or compact2 path was established by inspection. All five seq34 acceptance runs remain failures under the unchanged gate; equality to a failing baseline is diagnostic evidence, not acceptance.

## Scope and revision

Investigation performed on 2026-11-13, without running a decoder/reference forward pass, importing TTNN, opening hardware, or editing implementation or quality gates. Work consisted of source/log inspection and cheap CPU analysis of already-produced tensor caches using `torch.load(..., map_location="cpu", weights_only=True)`.

Unless otherwise stated, source paths below are relative to `/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next`; low-level `ttnn/` and `tt_metal/` paths are relative to the checkout root.

- Checkout HEAD: `70dc6f5dd24298741e2f108a281160677e986155`. The fused implementation is untracked, so HEAD alone does not identify it.
- Current `tt/fused_decoder.py`, its original failure source snapshot, and every seq34 cache identify SHA256 `8f8a4e2adccf66cb43135a6d1f11d8ed65d472ce1c3ce46f8e815ffb20f145ea`.
- `tests/probe_ple_decoder.py`: `28e0067b06e8a0db1da4e01332a77fd0fd7bfbf743a4986fcd5f98c756abd86b`.
- Every runtime file listed in the cache identity was rehashed and matched the current model tree. The five caches have identical identity dictionaries, including HF source, config, manifest, probe, and runtime source hashes.
- Actual HF source is `python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, SHA256 `e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df`. The separate `/home/ttuser/qwen38-flash/reference/modeling_qwen4_exp.py` has a different hash; it was not used to establish runtime line references.
- Model source provenance does not freeze every low-level TTNN/LLK header. Lower-level audit findings describe the prepared checkout, with that limitation.

## Direct observations

The original `doc/fused_decoder/integrated_writer_layer1_hf_v1.json` and its log agree: seq34, layer1, `bfloat8_b` experts, resident `packed_ple_lloyd_lsq_group10_scale8_wide`, threshold 0.995, no program preparation. Exactly row 5 fails whole and contribution PCC. Global PCC passes. Decode positions 34–36, reset/replay equality, and repeated-prefill equality pass.

The parent collection completed during inspection. `doc/fused_decoder/ple34_control_collection_v1.json` records exit 1 for all five controls. Their selection sidecars confirm the intended model/policy; the baseline constructs `FunctionalDecoder`.

| Control | Row5 whole PCC | Row5 contribution PCC | Prefill bytes versus old fused policy | All three decode outputs versus old fused policy |
|---|---:|---:|---|---|
| `baseline` | 0.9944669133989147 | 0.9936475639549838 | Different | Different |
| `best_exact_qkv_cleanup` | 0.9944641348820654 | 0.9936444270418033 | Reference | Reference |
| `best_writer_v5` | 0.9944641348820654 | 0.9936444270418033 | Identical | Identical |
| `best_tail2` | 0.9944641348820654 | 0.9936444270418033 | Identical | Identical |
| `best_writer_tail2` | 0.9944641348820654 | 0.9936444270418033 | Identical | Identical |

Each JSON is `doc/fused_decoder/ple34_control_<control>_v1.json`; each tensor cache is `/home/ttuser/qwen38-flash/ple34_control_<control>_v1.pt`. The full prefill tensors are BF16 `[1,1,34,10240]`. Equality above was checked on contiguous `uint8` views, not inferred from equal PCC or from floating-point `torch.equal` alone. Prompt, token IDs, literal expected prefill, decode inputs, and expected decode match across the controls. All five reports have `[prefill,decode0,decode1,decode2]` pass flags `[false,true,true,true]`, and all reset checks pass.

The baseline and old fused outputs differ slightly, but their row 5 error is almost the same:

- Maximum baseline-versus-fused row 5 difference: `0.001953125`; row 5 PCC between them: `0.9999963229401435`.
- Row5 error L2 versus HF: baseline approximately `0.598970`, fused `0.599091`; baseline-versus-fused difference L2 approximately `0.015397`.
- Cosine between their row 5 HF-error vectors: approximately `0.999670`.

These observations support a shared error source, without identifying its location. The new combined cache also reproduces every original reported metric; the original failure did not export tensors, so byte equality with that historical run itself is not claimed.

Cache SHA256 identities, for reproducibility:

```text
baseline                a00d3a2e8850b8a68e467cf24023b2a211e8dcc4b569cb1ed30abe167993d136
best_exact_qkv_cleanup  ec0574f0b11a15a9c89047ff4d97822c1686fbcc6eb8803cf900829b2af5e6c0
best_writer_v5          854146cc6e822a9253e5ba69b56327a361a6c021bcfa9d2f15939c09e2b1398b
best_tail2              d61e376e3492fdd4127a69d83cfbc9cb17f967993dd2484cdcc0bfc866ce8ea0
best_writer_tail2       c4050cce9fd32f1dba0a1a8672bdd28a59b1a16dde332bb3247bf99e62b7449b
```

## Why the earlier passing lengths do not isolate this change

`tests/probe_ple_decoder.py:183–197` seeds once, loads HF, draws a `[1,seq_len,10240]` prompt, draws three decode inputs, then draws all raw token IDs. Changing sequence length changes the RNG consumption before IDs and decode inputs; EOS placement also changes (`seq_len-2`, hence row31 versus row32). Seq33 and seq34 must not be treated as the same lexical prefix plus one extra token. In particular, a row 5 comparison across these runs has no established matching PLE input.

`final_v5_layer1_hf.json` and `final_v5_baseline_layer1_hf.json` are passing seq33 runs with minimum whole-row PCC approximately 0.996964 and 0.996958. Their success is consistent with a fixture-dependent shared error. The stated seq4768/4767 passes likewise cannot exclude it without matched hidden states, raw IDs, reference outputs and revisions. None establishes that seq34 passed before the writer/tail changes.

## Actual execution geometry and completion boundaries

`tt/functional_decoder.py:123–142` executes:

```text
raw IDs → device n-gram hash → packed embedding → PLE → add original hidden
        → attention residual read → GDN → attention residual write
        → MLP residual read → router/experts/shared expert → MLP residual write
```

`FusedDecoder` inherits this orchestration. The selected policy has `fuse_ple=False`, so the PLE object is the same implementation as the functional baseline (`tt/fused_decoder.py:2587,2654–2658`).

`tt/functional_decoder.py:157–177` chunks at 32:

| Logical request | Chunk starts/counts | Padded chunk height | Compact MoE under old/new policy |
|---|---|---|---|
| seq33 | `(0,32),(32,1)` | 32,32 | expanded32 + compact1 / same |
| seq34 | `(0,32),(32,2)` | 32,32 | expanded32 + expanded2 / expanded32 + compact2 |

Row5 is in the first chunk. The limit change only affects the last two-row MoE call (`tt/fused_decoder.py:988–995`), and stores a Python attribute without new limit-dependent setup allocations (`2598–2604`; existing compact allocations at`924–949`). The outer writer maps tail rows 0/1 to output rows 32/33, tile row 1, not row 5 (`tt/chunk_output.py:29–36`). A tail explanation therefore requires a nonlocal corruption mechanism. The matched byte-equal controls supply no evidence of one here.

Python returns enqueue work. The first failure is measured only after `synchronize_device` and `to_torch` (`tests/probe_ple_decoder.py:362–367`), before decode capture starts. Decode replays are blocking and read back (`395–408`); repeated prefill is also read back (`428–432`). Thus the failure is a completed prefill numerical result, not an inference from merely reaching a host assertion or teardown. Reset equality demonstrates repeatability and state restoration, not accuracy against HF.

## New-path audit: no contradictory contract found

### Exact residual writer

`tt/fused_decoder.py:262–344` accepts BF16 TILE interleaved DRAM tensors: residual`[1,1,T,10240]`, block`[1,1,T,2560]`, injection`[1,4,T,1]`. Each integrated seq33/34 invocation has `mt=ceil(T/32)=1` and 320 output tiles, including the tail. For output tile`j`, the reader uses block tile`j%80`, injection tile`floor(j/80)`, residual tile`j` (`194–203`). For example tile85 consumes block tile5 and injection stream1; logical row 5 uses that stream's row 5 injection.

Core `i` receives `[floor(320i/C),floor(320(i+1)/C))`, where `C=min(320,grid.x*grid.y)`, with the same count for reader, compute, writer (`288–301`). Actual physical grid dimensions were not in the failure evidence; no particular grid is assumed. The intervals exactly partition 320 tiles for any valid `C`.

| Phase, per tile | Buffer transition |
|---|---|
| Reader | reserve 1 in each CB0/1/2; read block/injection/residual; NoC read barrier; push 1 each |
| Compute | wait 1 each; copy index 0 into DST0/1/2; multiply, BF16 round, residual add, BF16 round; reserve/pack/push 1 CB3; pop 1 each input |
| Writer | wait 1 CB3; write corresponding output tile; NoC write barrier; pop 1 |

Each CB has capacity 2; no partial-count branch or unconsumed boundary item was found (`196–204,222–238,252–255,335–341`). `generic_op` preserves the supplied descriptor/operand arrangement and uses the supplied last tensor as output (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:40–49,138–153`).

The Blackhole primitive sequence agrees with the two rounding boundaries in `tt/decoder_components.py:99–110`: FP32 multiply with column-broadcast injection, round to BF16, FP32 residual add, round to BF16. The current Blackhole typecast explicitly masks the low 16 bits and stores FP32 DST (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:273–287`). Initialization is repeated locally; no missing mode transition was established. This is supported more directly by the complete writer-only output equality on the failing fixture.

Correction to the supplied component-coverage shorthand: `tests/probe_fused_residual_write.py:192` tests **11 counts** `{1,2,3,4,7,16,17,31,32,33,65}`, not every integer 1–65. `integrated_writer_clean_v1.json` and `integrated_writer_watcher_v1.json` report exact equality and zero error for both cases at every listed count. `--integrated` uses the actual production writer (`175–176`); twelve alternating blocking trace replays are checked at `240–248`. Counts 32 and 2 cover the integrated geometry. These component inputs are host-uploaded; the full matched controls additionally exercise device-produced operands.

### Compact2 MoE

The compact route packet retains expert-ID/score pairing, sorts unique IDs, and pads unused slots with expert-ID-zero/zero scores (`tt/fused_decoder.py:431–526`). Seq33's one-row compact path uses 10 slots; seq34's two-row path uses 20 slots with physical height 32. The latter's score scratch stride is 32 uint16 elements, its row stores are 40 bytes, and the maximum scratch endpoint is 8320 bytes within 32768.

The indexed sparse path consumes exactly the fixed slot count. Up projection broadcasts A; down projection consumes A by compact slot; B selects weights by `indices[slot]`; output remains associated with that slot. Relevant lowered code is `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:366–425` and the dataflow kernels under `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/`: `reader_bmm_tile_layout_in0_sender_padding.cpp:96–104,197` and `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:305–323`.

The fused reducer chooses the largest divisor≤8, so both 10 and 20 slots use granularity 5: respectively two and four complete groups per output, with no partial CB drain (`ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp:123–140`). On this 1×1 mesh, ownership mapping cannot drop a remote expert. `compact_repeated_routes_layer1_watcher_v1.json` additionally reports exact count 2 packets and outputs with 10 unique experts in 20 slots, exercising zero-score padding.

## Ranked remaining explanations

These are hypotheses about the shared failure, not diagnoses proved by final-output PCC.

| Candidate | Direct evidence and required preconditions | What remains unresolved / smallest discriminant | Implied intervention if confirmed |
|---|---|---|---|
| Packed PLE reconstruction propagated through decoder | Exact HF rows versus compressed TT rows is explicit in source; both functional/fused use the same table; fixture changes selected rows | Need literal HF versus HF using only actual packed lookup, on the exact seq34 fixture | Revisit packed-table accuracy policy within the existing memory contract; no new-writer change |
| Shared PLE/GR/GDN arithmetic, possibly amplified by routing | Shared code exists before/through router; baseline and fused have nearly identical error vectors | Compare device stages to packed-HF stages, then compare suspect operation on identical operands | Repair the first demonstrated arithmetic mismatch; no speculative broad rewrite |
| BFP8 expert arithmetic or route-sensitive MoE error | All controls use `bfloat8_b` expert weights; literal HF uses checkpoint weights | Need equal MLP input and expert-ID-aligned routes/weights before precision isolation | Target expert precision/arithmetic only if the residual error localizes there |
| New residual writer or compact2 | Reachable, but no source contradiction and all changed-policy outputs equal old fused byte-for-byte | Already falsified as the source of a changed result for this fixture | Toggling either cannot repair this failure |

### Earliest likely boundary to measure

`tests/probe_ple_decoder.py:137–139` installs `SelectedRows(...,"absmax")`, but `SelectedRows.quantized` defaults to **False** (`tests/probe_ple_quantization.py:62,79`). The literal reference is exact; the constructor string does not quantize it.

TT loads a manifest-owned LSQ group10 cache with 16 scales encoded in four words per 160-value embedding row. `tt/packed_embedding.py:251–282` gathers codes/scales, reconstructs their product, rounds to BF16, and returns `[1,1,T,2560]`. That is the earliest known intended HF/TT numerical divergence after the hash boundary. Hash/addressing errors are not established; verifying exact hash IDs and selected-byte reconstruction should precede blaming compression quality.

The PLE logical operations agree with installed HF (`modeling_qwen4_exp.py:1236–1248`): four 2560-wide grouped norms, shared value broadcast, signed square-root gate, dilation 3 and nine state rows, then residual addition (`1277–1280`). No semantic omission was found.

For fresh row 5, PLE's four convolution taps read normalized rows `[-4,-1,2,5]`; the negative rows are zero history. GDN can carry earlier rows 0–5 differences into row 5. Capture at least rows 0–5, not only the final failing row. Later rows 32–33 cannot explain row 5 through ordinary causal arithmetic.

## Precise follow-up evidence needed

### 1. Run the existing matched CPU counterfactual outside this inspection session

This command is a proposed diagnostic and **was not run here**. From the checkout root:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_prefill_backend \
  --snapshot /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa \
  --cache-dir /home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide \
  --seq-len 34 --row 5 \
  --device-output-cache /home/ttuser/qwen38-flash/ple34_control_best_exact_qkv_cleanup_v1.pt \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/ple34_cpu_controls.json \
  --tensor-output /home/ttuser/qwen38-flash/ple34_cpu_controls.pt
```

This diagnostic reproduces the probe's fixture/loading order and 16 CPU threads (`tests/probe_ple_prefill_backend.py:71,102–111`). It runs literal, packed-only, packed-plus-six-FP64-projections, and literal-repeat controls (`122–136`). It asserts prompt/IDs and literal expected-prefill equality with the saved device fixture (`240–246`). `CachedRows` reads actual cache words/scales and independently checks them against the production quantizer (`tests/probe_ple_batch_quantization.py:33–75`); this is not merely a regenerated quantizer surrogate.

Inspect row 5's whole/contribution PCC and complete error vector. If packed-only reproduces most of the literal-HF error and TT is close to packed-HF, embedding compression is supported as the leading cause. If packed-only passes or leaves a substantial device-versus-packed residual, continue isolating shared device arithmetic. Both mechanisms can contribute; packed-only failure alone does not prove TT adds no further error.

The diagnostic’s tensor artifact includes selected-row PLE, both hyper-connections, GDN, MLP, six projection outputs, router outputs, and full router indices (`150–168,228–237`). Its CPU experts remain BF16; neither packed-only nor the six-FP64-projection control models device BFP8 experts. The six projections are attention/MLP mix down/up plus GDN QKV/out, not a universal arithmetic oracle.

### 2. If necessary, capture the first 32-token device chunk at these boundaries

Use the saved exact prompt/IDs and fresh/reset state. Record logical row/stream/head, dtype, padded shape, layout, memory config, tensor provenance, and synchronization point for each capture:

1. Hash IDs; selected packed words/scales; reconstructed BF16 embeddings. Compare hash IDs exactly with HF and embeddings bitwise with CPU decoding the **same selected actual cache bytes**.
2. PLE key/value projections, group norms, gate, normalized gated values, pre-SiLU convolution, PLE output, and PLE-added residual. Capture rows 0–5 and full feature width.
3. Attention mixing/injection, GDN output, and post-attention residual.
4. MLP mixing, all 512 router logits, paired top 10 IDs/weights, routed/shared outputs, and final writer operands/output.

Compare each device stage with both literal HF and packed HF to separate expected compression differences from additional TT arithmetic. Stop at the first unexplained divergence; rerun that operator with matched operands and its actual dtype/layout/broadcast/provenance. A device-produced-input comparison is preferable to assuming that a host-uploaded surrogate selects identical behavior. Any instrumented diagnostic should first retain/reconfirm the uninstrumented final failure, because readbacks alter queue timing and allocation lifetime.

For routes, compare sorted expert sets and weights aligned by expert ID. `router_changed_rows` in the existing CPU diagnostic compares ordered arrays (`220–225`); a permutation is not a changed expert set. A legitimate top-k boundary crossing can amplify upstream error without a router defect. A labeled fixed-route counterfactual can distinguish routing amplification from direct MLP-input/expert error; it must not replace the acceptance oracle.

### 3. Only reopen writer/tail suspicion if new evidence contradicts matched equality

If a later run shows earlier output corruption, compare row 5 before outer chunk write, after first-chunk write, and after tail write. If a real writer mismatch emerges, compare old/new writers on the same original device `(residual,block,injection)` tensors at attention write and MLP write, then against the two BF16 rounding boundaries. These are conditional controls; the present results do not justify prioritizing them.

## Other potential issues and evidence limits

- **Shared PLE convolution rounding:** `tt/ple.py:64–65` materializes BF16 tap products and successive sums; HF calls one convolution before SiLU (`installed modeling_qwen4_exp.py:1210–1224`). Those tensor rounding boundaries differ. This is a concrete preexisting arithmetic difference, but no matched intermediate evidence makes it the row 5 cause. Keep it secondary pending step 2.
- **Writer validation gap:** the new writer checks shape/dtype/layout/memory/device but lacks the explicit standard-tile/padded-shape validation in `tt/chunk_output.py:71–75`. No production input violating those assumptions was observed. This is not a diagnosis of the reported failure.
- **Component pass versus exactness:** writer reports explicitly check equality. Some compact-component overall gates use PCC; inspect the individual `exact` and `max_abs` fields before claiming exactness. The cited repeated-route count 2 case explicitly has zero error.
- **GDN full-HF passing evidence:** a GDN-only pass does not test the same PLE-produced inputs or downstream router sensitivity. It is compatible with, but cannot establish, a PLE-first diagnosis.
- **No tolerance workaround:** global PCC≥0.995 does not rescue a rowwise failure. The FP64 metric at `tests/probe_ple_decoder.py:32–117` examines every row, and the final gate at `433–438` correctly remains false.

## Post-draft claim review

The headline was checked again against the raw five-cache byte comparison, completed collection outcomes, selected policies, and source hashes. Writer/tail explanations were tested against actual 32+2 execution and demoted because the changed-policy tensors are identical. Packed PLE was retained as the earliest known intentional difference and the leading measurement boundary, **not** promoted to a proven root cause. Extra PLE convolution rounding and missing writer validation remain in this secondary section because their causal prerequisites have not been demonstrated. No source, functional implementation, or gate change is recommended before the matched packed-only and intermediate evidence resolves that remaining uncertainty.
