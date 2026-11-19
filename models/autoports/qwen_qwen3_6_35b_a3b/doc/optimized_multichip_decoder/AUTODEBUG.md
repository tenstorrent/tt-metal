# AutoDebug: stage 5 combined fused sharded-residual K replicas

## Finding

**The candidate uses a rank-dependent fused AG–matmul accumulation to produce K/V heads that its caller requires to be bit-identical across rank pairs.** The fused operation visits the same K blocks in different orders on ranks 0/1 and 2/3. Its FP32 intermediate accumulation does not make those orders bit-equivalent. This is a concrete caller/operation contract mismatch and the leading source-supported explanation for the saved one-ULP K differences. It is not a rank-dependent distributed-RMSNorm statistic reduction.

The source proves the ordering discrepancy. The saved artifact proves the final K discrepancy. No intermediate tensors were saved, so inspection alone cannot establish that a particular accumulator rounded at a particular instruction to produce each observed value. A same-input separate-projection control or intermediate readbacks would close that remaining attribution gap. No device code was run for this investigation, and no implementation was edited.

The smallest candidate intervention is **`gather_projection="separate"`**, keeping sharded residuals, distributed normalization, batched attention/projections, and fused output reduce-scatter. This restores a common full-input projection traversal without changing the exact KV equality assertion or the PCC gates. If retaining `gather_projection="fused"` is itself mandatory, the fused implementation needs a rank-invariant reduction mode, or the model needs single-owner KV computation with pair broadcast; changing RMSNorm precision/order is not the repair for this contract mismatch.

## Scope and revision evidence

Investigated `models/autoports/qwen_qwen3_6_35b_a3b/tt/multichip_decoder.py` and the saved `doc/optimized_multichip_decoder/combined_fused_l3.*` evidence. Paths below are relative to this repository; `model/` abbreviates `models/autoports/qwen_qwen3_6_35b_a3b/`.

The run provenance records HEAD `5ebe699604445e7b30ea0813e15aa913efbd3188`, timestamp `2026-11-13T21:44:15Z`, and model source SHA256 `b99b6a963a97928b253830838b5feee6d153ee3160ca1930ab2f5b91524d18b8`. The parent is actively changing the working-tree model file for other candidates; its current contents are not the exact archived run source. Model line references below describe inspected function bodies, with function names supplied to survive shifting lines. The fused factory/receiver/kernel and normalization/collective source files discussed here have no working-tree edits. The selected fused algorithm and its rank-dependent scheduling are already present at recorded HEAD. The saved command, policy, and candidate source describe the multirow branch under investigation; unrelated subsequent precision/shared-projection options are not attributed to this run.

The validation harness SHA256 is `20c887456a2852ffb52add853c55a19ffdffaf09b3dd44e0aad7f6e91e3ed3b3`, matching the candidate and the stage 5 passing controls inspected here.

Native-build provenance is less complete: the recorded TTNN version is `0.1.dev1+ga464944`; that abbreviated commit is not present in this checkout. A clean C++ working tree does not prove that the installed native library was built from the inspected factories. The lowering below is the prepared source's executable contract, supported by the candidate's API/configuration and log, rather than a verified binary-to-source correspondence. The header/kernel analysis and all saved-tensor observations stand independently of that limitation.

Effective candidate policy:

```json
{"batch_attention_reduce":true,"batch_full_projections":true,"residual_layout":"sharded","full_norm":"distributed","attention_collective":"fused_rs","gather_projection":"fused","gather_rows":32}
```

The saved command uses layer 3, 32 requests, prefill length 32, eight changing decode steps, 60 timing replays, three prefill repetitions, three stress repetitions, natural activations, and the saved paired single-chip baseline. Mesh is 1×4 Blackhole, FABRIC_1D, **Linear topology**. `CollectiveWorkspace.__init__` explicitly chooses `ttnn.Topology.Linear` (`model/tt/multichip_decoder.py:768–771`); a Ring-only alternating schedule must not be substituted into this diagnosis.

## Direct observations from saved evidence

The log reaches `PREFILL_DONE`, completes `_run_side`, and fails `run_multichip_decoder.py:371`:

```python
assert torch.equal(parts[0], parts[1]) and torch.equal(parts[2], parts[3]), key
```

The exception is `AssertionError: k`. The `.pt` artifact exists because the harness saves it at lines 456–457 before running `_state_checks` at line 459; the final JSON does not exist because that assertion interrupts its construction. This is a real gate failure, not a completed passing candidate.

CPU-only `torch.load(..., weights_only=True, map_location="cpu")` inspection finds **exactly two unequal paired-K elements** across the 32 requests, and no paired-V differences:

| Request row | Rank pair | Physical cache index `[page, head, row, channel]` | Logical token | Rank 0 value | Rank 1 value | Absolute delta |
|---|---|---|---|---|---|---|
| 6 | 0 / 1 | `[4,0,15,107]` | 15, prefill-owned cell | 0.01318359375 | 0.01324462890625 | 0.00006103515625 |
| 21 | 0 / 1 | `[7,0,5,198]` | 37, decode step 5 | -0.1318359375 | -0.130859375 | 0.0009765625 |

Each is one adjacent BF16 representable value. All ranks 2/3 paired K cells agree; all other ranks 0/1 K cells agree. Both affected channels are outside the 64-channel rotary prefix.

The logical-token mapping is established, not inferred from a log label: row 6 has `pages[0]=4`; row 21 has `pages[1]=7`; page length is 32. Therefore the physical rows map to `0*32+15=15` and `1*32+5=37`. Only final states are saved, so the row-6 observation is a final disagreement in a cell owned by prefill, not a direct post-prefill snapshot.

Using the harness's centered-double PCC definition on saved tensors gives prefill PCC `0.9999583914039581` and decode PCCs:

```text
0.9999600812894678  0.9999078013771664  0.9998730543702083  0.9999635050336493
0.9999122343434764  0.9998813550439363  0.9998500960283638  0.9999133288736465
```

These comfortably exceed 0.995 but do not waive pair identity. Diagnostic reconstruction of written logical state using ranks 0/2 has minimum PCC `0.999985651601417`; that is not an accepted state check because the preceding pair-equality assertion fails.

The artifact summary records deterministic output, all-step eager/trace output equality, live-trace prefill output equality, one capture, and 24 changing-input stress replays. Those checks compare model outputs, not paired state replicas at every replay. They support repeatability but do not prove that K replicas were always identical.

Completion boundaries are explicit: timed prefill synchronizes the mesh; changing-input trace execution uses `blocking=True`; final state extraction restores the prefill snapshots, executes all eight decode steps with blocking replay, then converts every device state to host (`run_multichip_decoder.py:200–208,279–300,343–350`). Mesh teardown finishes before `_state_checks` runs. Thus the host assertion inspects completed saved state, not merely a successfully enqueued cache update. This does not rule out an earlier device defect, but there is no unobserved final queue drain needed to explain the assertion.

## Causal chain and lowered execution

### 1. The loader intentionally duplicates K/V within each pair

`MultichipDecoder.from_state_dict` packs each rank's local weights as `[Q1024, K256, V256, gate1024]`. It selects both K and V with `r // 2`, while Q/gate use `r*4:(r+1)*4` (`model/tt/multichip_decoder.py:951–970`). Thus ranks 0/1 own identical head-0 K/V weights and ranks 2/3 own identical head-1 K/V weights. Each local matrix is `[2048,2560]`, uploaded BFP8. K starts at output column 1024 and V at 1280, both tile-aligned; their duplicate tiles do not straddle different Q/gate weight regions. The pair-equality gate reflects this GQA ownership correctly.

For this single-layer runner, decode inputs are independently refreshed from the saved paired `inputs["xd"]`, not fed back from the candidate layer's previous output (`run_multichip_decoder.py:110–125,181–193`). Residual/output reduce-scatter is downstream of the current K/V writes. It cannot alter the current projection through a hidden activation feedback loop in this repro.

### 2. Distributed RMSNorm assembles statistics in a common order

`_normalized_input` converts the hidden shard to DRAM-interleaved `[1,1,32,512]`, runs pre-all-gather RMSNorm with explicit BF16 statistics, gathers those statistics, and runs post-all-gather RMSNorm (`model/tt/multichip_decoder.py:1208–1221`). At both 32-token prefill and batched decode, local width is 16 tiles, local height is one tile row.

The lowered path is:

| Phase | Concrete data and source contract |
|---|---|
| Pre | `layernorm_pre_all_gather_device_operation.cpp:87–101` specifies `[1,1,32,32]` stats. `layernorm_pre_all_gather_program_factory.cpp` selects `rmsnorm_pre_allgather.cpp`; its lines 89–112 square/reduce the 16 local tiles. |
| Stats AG | `_gather` uses `dim=3`, `cluster_axis=1`, DRAM. The aligned TILE/interleaved path selects minimal-default AG. `all_gather_async/device/kernels/minimal_default_writer.cpp:326–333` places local source `r` at `r*input_tensor_Wt`; lines 586–587 place forwarded data at its source's index. For stats `input_tensor_Wt=1`: every rank receives slots `[s0,s1,s2,s3]`, shape `[1,1,32,128]`. Arrival order does not become reduction order. |
| Post read | `reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:120–140` reads the four statistics pages sequentially from the common row offset. |
| Post compute | `layernorm_post_all_gather_program_factory.cpp:270–285` sets reduction factor `logical_W*num_devices=512*4=2048` and selects **`rmsnorm_post_allgather_metal2.cpp`**. That kernel lines 96–123 reduces `row(stats_tiles_cols=4)`, adds epsilon, and computes rsqrt. No rank argument selects a different sum order. |

The source therefore gives all ranks the same normalization scalar for a token, assuming collective data integrity. The normalized *local hidden slices* and gamma slices intentionally differ; gathering those slices gives the same global normalized vector on every rank. BF16 local sums and BF16 gamma can differ numerically from the ordered full-width baseline, but this is a baseline-accuracy issue, not a source of different per-rank global statistics ordering.

The fused branch returns the normalized local `[1,32,512]` shard instead of a separately gathered full vector (`_normalized_input:1222–1229`). The crucial change occurs in its projection consumer.

### 3. Fused AG–matmul deliberately consumes K in rank-relative order

`linear(...,"attention_packed")` reshapes local activations to `[1,1,32,512]` and calls `strided_all_gather_minimal_matmul_async` (`model/tt/multichip_decoder.py:1090–1115`). Its settings are:

| Property | Value |
|---|---|
| Gather dimension / cluster axis | 3 / 1 |
| Activation / gathered / weight | BF16 `[1,1,32,512]` / BF16 `[1,1,32,2048]` / BFP8 `[1,1,2048,2560]` per rank |
| Memory | DRAM-interleaved inputs, persistent gathered buffer, and matmul output |
| Matmul config | grid 8×2; M block 1 tile; K block 8 tiles; N block 10 tiles; subblock 1×2 |
| Compute | inherited attention HiFi2, `math_approx_mode=False`, **`fp32_dest_acc_en=True`** |
| CCL | Linear; one link, one worker/link, one buffer/channel; `read_local_slice_from_input=False`; one output chunk; aggregator Off |

The compute config is `candidate_compute["attention_packed"]`, constructed by `_ProjectionDecoder.from_state_dict` in `model/tt/optimized_decoder.py:417–428`. Do not confuse it with `self.fused_projection_compute`, which belongs to the downstream attention-output matmul/reduce-scatter.

The public C++ wrapper directly forwards these values (`strided_all_gather_minimal_matmul_async.cpp:35–60`). The selected program factory linearizes the mesh coordinate into the rank, calls `MinimalMatmulFusedOpSignaler::init_all_gather`, then **`minimal_matmul_fabric_bound_factory_helper_common`** (`.../device/strided_all_gather_minimal_matmul_async_program.cpp:56–100`). Consequently the selected kernels are `fabric_bound_dm_in0_sender.cpp`, `fabric_bound_dm_in1_sender_out.cpp`, and `fabric_bound_compute.cpp`, not the similarly named standalone/Metal-2 minimal-matmul kernels.

Concrete partition: `K_tiles=2048/32=64`, `input_tensor_Wt=512/32=16`, `K_block_tiles=8`, `K_num_blocks=8`. Each source device owns exactly two complete K blocks; no block straddles a device boundary. `compute_device_chunk_stats` in `strided_all_gather_async/device/kernels/fused_receiver_utils.hpp:15–53` gives starts `[0,2,4,6]`, counts `[2,2,2,2]`, and one expected producer per block.

The receiver resets to its own device (`fused_receiver_utils.hpp:168–185`), then the Linear branch of `process_chunk` expands backward/forward from that rank (`:188–251`). `compute_actual_k_block_iter` stores/reuses this order (`:333–386`). The actual sequences are:

| Rank | Source-device order | Global 256-element K-block order |
|---|---|---|
| 0 | 0,1,2,3 | **0,1,2,3,4,5,6,7** |
| 1 | 1,0,2,3 | **2,3,0,1,4,5,6,7** |
| 2 | 2,1,3,0 | **4,5,2,3,6,7,0,1** |
| 3 | 3,2,1,0 | **6,7,4,5,2,3,0,1** |

This is a fixed rank-relative ordering, not a claim that packet arrival races randomly reorder arithmetic. Ring-only `AG_ALTERNATE_MIDDLE` is inactive for this Linear topology. `read_local_slice_from_input=False` changes the local source/readiness requirement; it does **not** disable the rank-relative ordering.

A cheap host-only C++ check extracted the actual `compute_device_chunk_stats` and `process_chunk` bodies from the inspected header (SHA256 `fe2e1f9cb8d5f1b0a46041cd2518bc2b23b1ff0cbf6af0c24f10eb802705dc7d`) and supplied only host stubs for the enum/state. It confirmed all four sequences above, device counts/starts, exactly one consumption of each block 0–7, and matching independently initialized A/B receiver sequences. This checks the source schedule, not device arithmetic.

Both A and B readers use `compute_actual_k_block_iter`. A reads the selected global K interval (`fabric_bound_dm_in0_sender.cpp:514–565`); B reads the matching weight interval (`fabric_bound_dm_in1_sender_out.cpp:355–377`). Thus A/B pairing is coherent. All eight blocks contribute once; no missing slice or mismatched weight interval is required to explain this failure.

For one concrete compute core, K occupies output tiles 32–39, within N block 30–39 on x=3. The two affected channels are K output tiles 35 and 38, both in that block. M has one real tile row; `IN0_SUB_CHUNKS=1` for the supported one-M-tile setup. Each K iteration presents 8 A tiles and 80 B tiles, computes ten output tiles, then pops the corresponding A/B entries. The intermediate holds ten FP32 tiles. Eight iterations match the eight-block producer schedule; there is no partial K-block boundary in either the passing or failing rows. Which request row happens to cross a BF16 rounding boundary is data-dependent, not a queue-count change.

The selected CB ledger is c0 BF16 capacity 16 tiles (8 pushed/popped per K block), c1 BFP8 capacity 160 tiles (80 pushed/popped per K block), c7 BFP8 scratch 80 tiles (read and re-presented for each block), c3 FP32 capacity 10 tiles (reserved once, accumulated in place, pushed/popped once), and c2 BF16 capacity 10 tiles (final copy and writer drain). There are no optional bias/ternary/split-output buffers on this path. Relevant allocation/flow sources are `minimal_matmul_fabric_bound_program_factory.cpp:360–456`, `fabric_bound_dm_in1_sender_out.cpp:355–428`, and `fabric_bound_compute.cpp:566–638,673–682`.

### 4. The accumulation preserves that ordering through rounding and K normalization

The fabric-bound factory explicitly chooses Float32 intermediate tiles when FP32 destination accumulation is enabled (`minimal_matmul_fabric_bound_program_factory.cpp:233–238,454–457`). In `fabric_bound_compute.cpp`, `matmul_blocks` accumulates the eight inner tiles and packs to the intermediate; the outer K loop consumes inputs in stream order (`:432–469,569–638`). After the first K block, the kernel enables packer L1 accumulation at lines 632–634. Subsequent FP32 block contributions are added into that intermediate in the rank-relative sequence. The final copy packs the BF16 output.

Output dtype defaults to the BF16 input in `minimal_matmul_device_operation.cpp:401–424`. The selected final branch calls `copy_block_to_cb` (`fabric_bound_compute.cpp:673–682`); that helper reconfigures the output pack format and calls `pack_tile` (`:25–47`). Thus the argument does not rely on an assumed output dtype or on inspecting an unselected standalone kernel.

The Python config's `packer_l1_acc=False` is therefore **not** evidence that this kernel avoids L1 accumulation: this selected kernel explicitly enables it internally. That is not by itself a bug; it explains how the order propagates to finite-precision sums.

For equal K weight columns, the two ranks compute the same real-valued dot product, but sum the eight block contributions in different orders. FP32 addition is not associative. Near a BF16 rounding midpoint, a much smaller internal difference can become one BF16 output ULP. No part of the fused API promises bit-identical duplicate projections across ranks.

`_full_attention` then splits Q/K/V, calls the common per-head `self.norm(k,"self_attn.k_norm.weight")`, applies RoPE only to channels 0–63, and fills/updates local K/V caches (`model/tt/multichip_decoder.py:1590–1645`). K's 256-channel RMSNorm can transmit/amplify an upstream projection difference to a different K channel; V has no corresponding head norm. At channels 107 and 198 the RoPE branch is not arithmetic on the observed cell. The cache receives each rank's own K; there is no pair reconciliation. Hence the caller has no mechanism that restores equality after the differing projection reductions.

This chain explains how high aggregate output/state PCC and deterministic execution can coexist with the strict K-only pair failure. It does not predict the exact two cells from source alone. In particular, different rank schedules make either pair *capable* of differing; pair 2/3 and V remaining equal on these data is compatible with both sides rounding to the same BF16 value, but that specific rounding history is not saved evidence.

## Competing explanations and controls

| Candidate | Direct code evidence | Observations explained | Remaining assumptions / falsifier |
|---|---|---|---|
| **Rank-dependent AGMM reduction used for duplicated KV** | Proven different block sequences; same pair K/V weights; finite-precision accumulation; no pair reconciliation | Prefill and decode cells, rare one-ULP K mismatches, high PCC, repeatable outputs | Exact first unequal intermediate is not saved. Compare fused output's duplicate K/V columns before head norm with canonical projection of the returned gathered input. |
| Distributed RMSNorm sums stats in local-rank order | **Refuted for this selected path:** common source-indexed stats, common sequential post reduction | Can explain baseline rounding differences, not a rank-order discrepancy | Would require unobserved AG corruption or a lower-level numeric defect, not the alleged sum order. Read back gathered stats per rank to test integrity. |
| Fused output reduce-scatter changes current K | Downstream of current K/V writes; no activation feedback for these independent single-layer decode inputs | Can affect residual output PCC | Cannot account for the earliest prefill-owned K cell through the inspected control flow. |
| Batched row slicing / RoPE / paged cache defect | No demonstrated index, ownership, or count contradiction | Broadly could corrupt state | Row-6 prefill cell does not use batched decode slicing; both observed channels bypass rotary arithmetic; exact page mapping is valid. Intermediate cache-source comparison would test remaining corruption hypotheses. |
| Trace allocation/lifetime corruption | Log contains generic active-trace allocation warning, but no identified overlapping allocation | Generic possible corruption | No matched address/lifetime violation shown. Blocking replay/host read boundaries and output repeatability prevent treating the warning alone as a diagnosis. |

Stage 5 `before_l3_b32`, `batch_ar_l3_v3`, `batch_proj_l3`, and `batch_proj_decodeonly_l3` artifacts all have zero paired K/V disagreements and share identical input tensors/signatures. The closest source-matched control, `batch_proj_decodeonly_l3`, has the same model source hash as `combined_fused_l3`, but changes several policies together: residual replication, normalization, projection gather, output collective, and direct-AR choice. It proves batching alone need not violate pair identity; it does **not** isolate RMSNorm or AGMM as an experimental single-variable contrast.

Historical B1 random-input controls with separate projection pass for both ordered and distributed norms on the same historical source. Historical fused gather-row 32/64 candidates also pass. Those differ in source/input/batching from stage 5, so they show the problem is not an unconditional AGMM failure, not that this natural-B32 branch is correct. Their passing values do not falsify a data-dependent rounding-boundary failure.

The generic CCL test `tests/nightly/t3000/ccl/test_strided_all_gather_minimal_matmul_async.py:416–460` compares gathered/matmul outputs with PCC, including the replicated-weight case. It does not assert duplicate columns are bit-identical between ranks. Therefore its success would not cover this model's stronger KV contract.

## Smallest fix and unchanged validation contract

1. **Smallest scoped candidate change:** set `gather_projection="separate"` and retain the other six listed policy values. `_normalized_input` then gathers the normalized hidden slices into a common full `[1,32,2048]` activation. `linear` avoids the 512-wide fused branch. Batched decode passes `batch_decode=True` and selects the configured batched DRAM projection; 32-row prefill does not pass that flag and uses the ordinary `ttnn.linear` fallback. Within each phase, paired ranks have the same projection geometry and K traversal for duplicated K/V columns. This addresses the earliest proven divergent arithmetic boundary without replacing distributed RMSNorm, gathering residual outputs, changing precision, or modifying the gates. Its performance and full correctness still require the parent's device validation.
2. **If the exact fused candidate must be retained:** add an explicit canonical K-reduction mode to the fused operation, ensuring both readers and compute consume blocks in the same global order on all ranks. Readiness must remain correct: the existing semaphores count rank-relative arriving chunks, so merely changing `my_chip_id`, returning `k_block_iter`, or removing waits is unsafe. A valid implementation needs to wait until the requested canonical block is available, or wait for full gather before running a canonical matmul. Alternatively compute K/V once per pair and distribute them to both owners. This is a larger implementation change than using the supported separate path.

Do not relax `torch.equal` to PCC/allclose, average disagreeing replicas, or switch to ordered normalization as a purported pair-equality fix. A higher fidelity or different K-block size may hide these particular rounding events but does not remove rank-relative accumulation and therefore supplies no general pair-identity guarantee. FP32 accumulation is already enabled. The existing separate path is a deliberate precision/ordering boundary repair, not a tolerance workaround.

Keep `run_multichip_decoder.py:371` exact pair equality and all existing `>=0.995` output/state PCC gates (`:460–467`) unchanged, along with changing-input eager/trace checks, state ownership, natural inputs, and replay counts.

## Follow-up validation for the parent; not executed here

- Run the same natural-B32 candidate with only `gather_projection="separate"` changed, retaining distributed norm and fused output RS. Check all K/V rank pairs in both post-prefill and post-decode states, not only aggregate output PCC. Include the row-6/token-15/channel-107 and row-21/token-37/channel-198 cells as diagnostics, while checking the entire state.
- If continued fused optimization is desired, use its returned gathered tensor for a diagnostic canonical matmul with the same uploaded weight and compute config. Compare **exact duplicate K/V columns** on every rank before K norm, after K norm, and immediately before/after the cache write. The returned gathered activations should themselves be bit-equal across all ranks. These readbacks belong outside timed execution.
- Verify that all gathered statistics are equal across destinations before spending effort changing RMSNorm. A common stats tensor plus unequal fused K columns would directly separate norm-order and AGMM-order explanations.
- After an implementation repair, repeat the existing full gates and measure its complete next-consumer path. Do not report this candidate accepted on the basis of the saved high PCC alone.

## Review of the draft's claims

The diagnosis was rechecked against actual Linear topology, selected fabric-bound kernels, effective FP32 compute config, concrete K-block sizes, both reader schedules, statistics placement/read order, exact cache-page mapping, and prefill versus decode ownership. Claims that distributed RMSNorm reduces in rank-local order, that FP32 accumulation is disabled, that this is decode-only, or that passing PCC establishes acceptance were rejected. No direct queue-count, cache-index, or lifetime violation was found that is needed to explain the observed failure. Hardware-level attribution of the two final rounded values remains explicitly unverified.
