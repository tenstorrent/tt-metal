# AutoDebug: mixed versus solo greedy decode, reduced Qwen layers 0/3

## Finding and confidence

**The strongest observed boundary is batch-sensitive grouped-MoE output; stochastic FPU scheduling is the leading source-backed mechanism candidate.** Solo physical slot 0 uses a different decoder implementation from mixed execution. Solo slots 1/2 use the same three-row implementation as mixed execution, but grouped MoE enables stochastic FPU rounding once per launch and associates subsequent work with a batch-dependent expert schedule. Thus using the same trace does not establish that a request sees the same arithmetic when other requests become active.

This is a **ranked diagnosis with strong later support, not a demonstrated complete repair**. The original failure artifact lacks the failing slot and intermediate values. Later diagnostics appeared during review: CPU inspection identifies differing model logits before generated-token feedback diverges, with grouped-MoE output differences despite equal recorded target inputs. A subsequent SR-off-labelled run restores bit-exact solo/mixed logits for slots 1/2 across all five calls, while exposing a slot-0 mismatch between B1 and B3. That run has an additional generator revision change and no kernel-hash sidecar, so attribution to an isolated SR-only intervention remains qualified. No obvious token/slot permutation, live-page collision, recurrent batch-axis error, or causal-mask coordinate error was found.

The next decisive observation is the earliest differing grouped-path intermediate across all chips, followed by a controlled rounding-policy comparison. Preserve the final greedy equality requirement. Scope is reduced-stack serving correctness; these fixtures do not support a qualitative model verdict.

## Inspection scope and revision consistency

- Inspection only: source, existing artifacts, and cheap CPU bookkeeping. No model execution, device tests, reset, target imports, or implementation edits. The AutoDebug launcher was not invoked.
- Git HEAD: `121ab5df0f494787236fff2b371e422e6edb599d`. The prepared tree includes uncommitted full-model work.
- Model-relative paths below are under `models/autoports/qwen_qwen3_6_35b_a3b/`. Low-level paths start at the repository root.
- Primary runtime evidence: `doc/full_model/logs/generator_replay_probe.log:274–310` and `doc/full_model/generator_replay_probe/result.json` (`status=FAIL`, context 1024, capacity 3, layers `[0,3]`, four resident replay steps after initial admission).
- At initial inspection, seven recorded non-probe source hashes matched: the prefill-equivalence helper plus model, generator, prefill, prefill transport, slot state, and multichip decoder. The artifact's probe hash is `e2d920184662c32d15004020d10b9ba4975f1dfcfc60fa455d148389712a08d4`; initially inspected probe hash was `f4301e37cc8b7b5184561a895a203c5b8021458b02325cf10be27b92936dfaf8`.
- The old traceback points to probe line 311; the analogous assertion is now line 320. The current probe saves `mixed_diagnostic` before that assertion (`tests/run_generator_replay_probe.py:315–321`); the failing artifact has no such field. Do not attribute the newer diagnostic persistence to the recorded run. Low-level C++/LLK sources are not individually pinned by that artifact; their behavior here describes the prepared checkout.

**Concurrent-work boundary:** model, generator, probe, and the grouped kernel changed externally while this investigator was reviewing the report. No investigator edited those files. Unless a later artifact is named explicitly, Python line references and the `StochRndType::Fpu` source excerpt below refer to the initial snapshot, not the moving final checkout. Later code temporarily changed the kernel to `StochRndType::None` for a diagnostic. Its subsequently completed artifact is analyzed separately below rather than spliced into the original run.

Relevant initial matching implementation hashes:

| File | SHA-256 |
|---|---|
| `tt/model.py` | `c541b1448264efb26880e39a3335565622462e62d247f55ddbcb4c32ee803e1e` |
| `tt/generator.py` | `d0aed4fe1f65f3f03a044adb4c2e667f83303191da27842bd859e9b6a9245d5f` |
| `tt/multichip_decoder.py` | `f7e44a88e57e58f10d0e1cd6a1ffd68390295eb541e9958312e3db5a5c909e9a` |
| `tt/prefill.py` | `1979a8ad1e4c2cf4fd3adb1c50f99ffa047d4d7c18eae5d86c95d33643b58a15` |
| `tt/slot_state.py` | `8265153a603c37ff6705e315a4e3efc60942f81930d0da1fd5d2d9db2f135350` |

## Direct observations versus interpretations

### Directly supported observations

1. The log records 22 singleton checkpoints, covering lengths `1,31,33,63,65,255,257` and continuations `(start,end)=(31,33),(31,64),(255,257),(255,288)`, twice each. Every checkpoint reports 8,487 program-cache entries and delta 0. The JSON contains 11 corresponding passing `public_length` cases.
2. The three `mixed_reference/slot=0,1,2` checkpoints also report delta 0. The following failure is `Mixed versus solo final greedy token changed`.
3. The current corresponding control flow puts public mixed-prefill metric assertions and replay position/seed checks before this final-token assertion. The first mixed iteration has `changed_pages=False`; the changed-unused-pages and inactive-middle controls, and the later sampling section, have no passing evidence from this run.
4. The saved artifact contains singleton trajectories but no mixed trajectory or per-slot mixed-prefill metrics. The assertion string alone does not reveal which physical slot failed or when its sequence first diverged.

### Interpretations with limits

- “All singleton repeats pass” means **physical slot 0** in the length sweep (`tests/run_generator_replay_probe.py:258–279`). Solo slots 1 and 2 are each run once as references (`:293–302`); those trajectories are not independently compared with an HF or single-row reference.
- “Public prefill logits pass mixed equals solo” is the check's success criterion, not proof of bitwise equality. The current check requires finite logits, PCC >= 0.99999, and reference-top-5/top-100 membership of the candidate argmax (`:309–314`). It does not require `exact`, `top1`, or `greedy_token_agreement`. The metric's `top5` is **argmax membership in the reference's top five**, not equality of both top-five sets (`tests/run_prefill_equivalence_probe.py:70–77`). No actual failed-run metric values were persisted.
- Public prefill returns final-row logits; this check does not compare every cached K/V token, recurrent state, or convolution history. Matching final logits cannot establish matching decode inputs.
- Stable program counts and trace IDs establish reuse. They do not prove output correctness or equality of runtime routing/counts/data.

## Later diagnostic evidence inspected during report review

The parent continued its own work while this investigation ran. I read the resulting JSON and `.pt` artifacts on CPU using `torch.load(..., map_location="cpu", weights_only=True)`; I did not run the generating device code. `mixed_diagnostic.json` retains the initial model/generator hashes and the initially inspected probe hash. `mixed_logits.json` retains the initial model/generator hashes with a newer probe. `mixed_hidden.json` and `mixed_moe.json` record instrumented model `cce70243...`, generator `e4555358...`, and probes `d5a88909...` / `5b791b23...` respectively; their multichip decoder, prefill, transport and slot-state hashes remain the initial ones. These are distinct runs and source revisions.

All four JSON diagnostics preserve the same token trajectories:

| Physical slot | Initial sampled token, solo/mixed | Four resident-step tokens, solo | Four resident-step tokens, mixed |
|---|---|---|---|
| 0 | 242684 / 242684 | 18409,11927,58366,245458 | 18409,11927,58366,245458 |
| 1 | 209290 / 209290 | 450,28433,94353,109160 | 45878,55073,132666,169484 |
| 2 | 205456 / 205456 | 138501,71729,83859,10167 | 138501,71729,83859,28433 |

For the `mixed_logits`, `mixed_hidden` and `mixed_moe` saved logits, CPU comparison gives the same values at these early points:

| Target / consumed position | Solo/mixed argmax | Max absolute logit delta | Interpretation |
|---|---|---:|---|
| Slot 1 / 65, initial call | 209290 / 209290 | 0.140625 | Model outputs already differ, same chosen feedback token |
| Slot 1 / 66, first resident step | 450 / 45878 | 0.1875 | First token divergence; target input token still agrees |
| Slot 2 / 257, initial call | 205456 / 205456 | 0.15625 | Model outputs already differ before any generated input differs |
| Slot 2 / 261, final resident step | 10167 / 28433 | 0.1640625 | First token divergence for slot 2 |

`mixed_moe.slots_1.pt`, `mixed_moe.slots_2.pt`, and `mixed_moe.slots_0_1_2.pt` further narrow the boundary:

- **Slot 2, initial position 257, layer 0, chip 0:** pre-MoE input and shared-expert output are bit-exact; `_grouped_moe` output differs in 1,685/2,048 values, maximum `0.00006103515625`. Recorded conv/recurrent state on chip 0 is exact.
- **Slot 1, position 66, layer 0, chip 0:** pre-MoE input and shared-expert output are bit-exact; grouped output differs in 1,727/2,048 values, maximum `0.000091552734375`. Recorded conv/recurrent state on chip 0 is exact.
- Slot 1's layer-0 chip-0 MoE output is exact on the initial call even though the layer output/logits differ. This is not counterevidence without inspecting the other three chips: the layer subsequently reduces their partial outputs.

The tap reads only **chip 0**, and `_grouped_moe` includes dispatch, expert arithmetic, combine and post-combine weighting. Equal tapped input does not prove equal target router values/IDs/weights, dispatch contents, or other-chip inputs. It localizes a real output difference to that composite boundary on the recorded chip. Attribution specifically to stochastic rounding still requires a controlled contrast or finer taps.

These later observations demote the B1-versus-B3 split as an explanation of the reproduced token failure: slot 0 agrees, while failing slots 1/2 use the same B3 trace. They also demote a sampler-only explanation because model logits already differ before feedback diverges. The original uninstrumented artifact alone could establish neither fact.

For snapshot identity, the three inspected MoE tensor-file SHA-256 values are: solo slot 1 `6f1d0bb7c4821c896942af906a78e32d1fdbea49c0ab4265542cf820250de2ba`; solo slot 2 `4545d78ac1f469dcb66d7fc5fef4ad0ff488fb282fb396d434d74cd5e7a156dd`; mixed `0cd8d315d6371efdbda0eb4ed961789510926dea1a1e2e04c96e7627e49907d7`.

### Later SR-off-labelled diagnostic: fixed-B3 equality restored, overall test still fails

Before finalization, `mixed_sr_off.json` completed with `status=FAIL`. CPU comparison of its three corresponding slot-1/slot-2/mixed tensor files shows **bit-exact full-vocabulary logits at all five calls for both slots 1 and 2**. Every recorded chip-0 MoE tap in both layers is also exact. Their complete tokens agree:

- Slot 1: initial 209290, then 45878,55073,132666,169484 in both arms.
- Slot 2: initial 205456, then 138501,71729,83859,28433 in both arms.
- Slot 0: solo remains initial 242684 then 18409,11927,58366,245458; mixed is initial 242684 then 184997,152864,92842,158538. Hence the overall mixed-versus-solo acceptance assertion still fails.

The source observed while this diagnostic ran changed the grouped kernel's explicit mode from `Fpu` to `None`. The run records the same model/probe/multichip hashes as `mixed_moe.json`, but generator hash `746a47bf...` differs from that run's `e4555358...`; low-level kernel hashes are absent from both sidecars. This is strong supporting evidence for the grouped rounding mechanism, **with an isolation/provenance caveat**, and direct evidence that simply obtaining fixed-B3 equality does not satisfy the full three-slot matrix. Reconcile the generator delta and repeat a source-pinned SR-only contrast before calling it causal proof. The new slot-0 failure leaves the separately audited B1/B3 comparison requiring localization if deterministic grouped arithmetic is chosen. It does not justify changing the correctness threshold.

## Exact execution and ownership ledger

`QwenGenerator.decode_forward` selects `model` only when `active_slots == (0,)`; otherwise it selects `model_fixed` (`tt/generator.py:513–527`). `model_fixed` always computes physical rows `[0,1,2]` with runtime activity (`:434–440`); it does not compact live requests into a different Python batch.

| Comparison arm | Model trace / physical rows | Activity | Consumed positions: admission plus four steps | Position after admission → final |
|---|---|---|---|---|
| Solo slot 0, length 31 | `model`, `[0]` | unmasked | 31,32,33,34,35 | 32 → 36 |
| Solo slot 1, length 65 | `model_fixed`, `[0,1,2]` | `[0,1,0]` | slot 1: 65..69 | 66 → 70 |
| Solo slot 2, length 257 | `model_fixed`, `[0,1,2]` | `[0,0,1]` | slot 2: 257..261 | 258 → 262 |
| Mixed | `model_fixed`, `[0,1,2]` | `[1,1,1]` | respective ranges above | `[32,66,258]` → `[36,70,262]` |

Each active seed advances from 2 after admission to 6 after the four replay steps. The initial input is deliberately `fixture[slot][length]`, and the next four inputs are device-sampled feedback (`tests/run_generator_replay_probe.py:195–246`). The test does not use the public prefill argmax as the first decode token. Therefore public-prefill argmax agreement alone cannot explain or validate the initial decode output.

Context 1024 gives 32 physical pages, a `(1,32)` table per slot, and one shared K/V pool per full-attention layer (`tt/model.py:186–201`; `tt/multichip_decoder.py:1646–1682`). The probe rolls the identity mapping by `slot * floor(32/3)`, hence offsets 0,10,20 (`tests/run_generator_replay_probe.py:47–58`).

| Slot | Prefill live physical pages | Live pages including all five decode calls | KV page/offset on admission |
|---|---|---|---|
| 0 | 0 | 0,1 | physical 0, offset 31 |
| 1 | 10,11,12 | 10,11,12 | physical 12, offset 1 |
| 2 | 20..28 | 20..28 | physical 28, offset 1 |

Live ownership is disjoint. Reversing the requested slot order during replay pairs `pages[list(order)]` with those slots; computation remains canonical and readback is restored by physical slot (`tests/run_generator_replay_probe.py:216–230`; `tt/generator.py:261–278,489–492`). No evidence supports treating this reorder as request-state migration.

## Leading candidate: grouped MoE has a batch-dependent rounding schedule

### Lowered operation chain

1. Fixed decode combines three post-attention rows, normalizes them, then invokes `_moe(..., batched=True)` (`tt/multichip_decoder.py:1634–1639`).
2. `_moe` explicitly pads `[1,3,2048]` to `[1,32,2048]`, including inactive and padding rows in router/top-k/count preparation (`:1322–1347`). Runtime activity does not remove those rows from the grouped operation. It masks hidden rows around layer boundaries and protects request-state commits (`tt/model.py:242–247,260–267`).
3. Top-eight IDs produce a selection mask, per-expert counts, 32-row-aligned regions, and offsets. Local ownership masks retain the chip's 64 experts. Dispatch compacts rows into those regions (`tt/multichip_decoder.py:1396–1438`). Other requests can change both the nonempty expert set and a target row's compacted row index. Dispatch's token-order offset increments are in `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/reader_worker_dispatch.cpp:359–387`.
4. Production passes LoFi, `fp32_dest_acc_en=False`, `math_approx_mode=False`, and BF16 preallocated output to `moe_fused_swiglu` (`tt/multichip_decoder.py:919–925,1439–1451,1475–1485`). The wrapper resolves the supplied compute config, output dtype and output memory from those arguments; it does not reorder operands or substitute FP32 accumulation (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/moe_fused_swiglu.cpp:68–87,95–117`). The factory passes these values to `moe_fused_swiglu_compute.cpp` (`.../device/moe_fused_swiglu_program_factory.cpp:591–609`).
5. **That compute kernel independently enables stochastic FPU rounding and seeds once per participating core per launch**, immediately after hardware startup (`.../device/kernels/moe_fused_swiglu_compute.cpp:446–455`):

   ```cpp
   UNPACK((_llk_unpack_configure_stoch_rnd_<ckernel::StochRndType::Fpu>()));
   MATH((ckernel::init_prng_seed(0x5EED1234u)));
   ```

   `math_approx_mode=False` does not disable this explicit override. Blackhole defines FPU mode as stochastic rounding for every FPU accumulation (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h:150–164`); the helper sets FPU SR on and gasket/packer SR off (`.../llk_lib/llk_unpack_common.h:96–107`). `init_prng_seed` writes a single PRNG seed register (`.../common/inc/ckernel.h:812–822`).
6. The reader derives `m_t=ceil(count/32)` and `m_blocks` from runtime counts (`.../moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp:574–581,611`). Compute loops through all 64 local experts, executing math only for each expert's `m_blocks` (`.../moe_fused_swiglu_compute.cpp:520–552`). For this padded 32-row workload, each expert has 0 or 1 tile row: adding one previously unselected expert changes zero compute blocks into one compute block, even without changing the program cache key.
7. The inner matmul initialization and format reconfiguration do not reseed per expert or reset FPU SR (`.../moe_fused_swiglu_compute_helpers.hpp:145–146`; `tt_metal/hw/inc/api/compute/matmul.h:208–227`; Blackhole `llk_unpack_common.h:211–261`). The seed is associated with the whole invocation, not a stable request/expert/logical coordinate.

### Concrete contrasting schedule

For an unchanged target in physical slot 1, compare the same B3 trace with masks `[0,1,0]` and `[1,1,1]`. Its router input can be equal at this boundary while other rows select additional experts. A possible count contrast on one chip is:

```text
                         preceding local expert a    target's local expert b, a < b
solo target only:                  count 0                      count 1
other requests active:             count 1                      count 1
reader m_t:                         0 → 1                       1 → 1
compute blocks before b:            adds one expert block
PRNG initialization:               once at launch in both arms, no reset at b
```

This example requires `a` not already selected by the zero rows; there are at most eight zero-row IDs and 64 local experts, so such schedules are representable. It is **not a claim that the recorded fixture selected these specific experts**. Within an expert selected by multiple live rows, token-order compaction can additionally move the target to a different lane.

**Strong inference:** the changed FPU work/lane association can change which pseudorandom rounding decisions a target accumulation receives. Fixed seeding can make a fixed workload repeatable while failing batch-membership invariance. The hardware PRNG's exact advancement/period is not modeled sufficiently in the inspected source to prove a particular changed element, and actual route IDs/counts were not saved.

This candidate covers the otherwise difficult solo-slot-1/2 versus mixed comparison, which uses identical B3 shapes and trace IDs. It also distinguishes serial prefill: public prefill processes requests one at a time with their own same-length arithmetic (`tt/generator.py:316–346`), so other requests do not enter the same MoE count vector. Singleton repetition evidence exercises B1 slot 0 and does not validate this grouped path.

**Smallest diagnostic intervention:** from equal target input/state, compare pre-MoE and routed-expert outputs in the same B3 graph, with other rows inactive/live; then repeat a source-pinned diagnostic with only FPU SR disabled. The later artifact supplies the predicted fixed-B3 contrast, subject to the provenance caveat above. Separately test FP32 DEST if needed. Changes must occur before fresh preparation/capture. They are falsifying experiments, not proposed acceptance waivers or a validated repair. A divergence already before this layer's MoE refutes it as that step's first divergence. A successful fixed-B3 contrast still requires the original complete greedy equality bar and numerical accuracy before adopting a production policy.

## Additional concrete split: slot 0 solo uses different arithmetic

If the failing physical slot is 0, several earlier or larger implementation changes exist independently of stochastic stream placement:

| Stage | Solo slot 0: B1 `model` | Mixed slot 0: B3 `model_fixed` |
|---|---|---|
| Linear-attention state | Direct recurrent output into persistent DRAM | Transient recurrent candidate then byte-preserving conditional DRAM commit |
| Linear post-attention norm | `normalized_singleton` compensated epilogue | Ordered/refined multirow graph from `FusedDecoder.norm` |
| Router/top-k | Linear singleton ordered scan at cache boundary, custom FP32 selector | Multirow projection, native top-k after padding to 32 |
| Routed experts | TP4 indexed sparse matmuls, FP32 DEST | EP grouped fused FFN, BF16 DEST, explicit stochastic FPU rounding |
| Shared expert projections | Explicit singleton DRAM projection programs | Generic matmul with default `batch_shared_projections=False` |
| Full-attention input normalization/projection | Singleton input can use the selected L1-sharded norm path | Three-row normalization and packed projection, then per-row attention |

Evidence: `tt/multichip_decoder.py:375–414,498–513,834,1143–1202,1263–1290,1322–1473,1493–1531,1583–1644`; sparse compute config has FP32 DEST enabled at `tt/optimized_decoder.py:44–49`. The linear norm fallback is **also an ordered/refined implementation**, not an ordinary generic RMSNorm (`tt/fused_decoder.py:1014–1104`; singleton `tt/mesh_compensated_norm.py:213–242`). Both norm formulas intend the same operation; different execution is not itself proof of an error.

These are concrete precision/layout/program differences. They can explain a slot-0 mismatch without explaining a slot-1/2 mismatch. The original artifact does not identify its failing slot; the later diagnostics reproduce failure in slots 1/2 while slot 0 agrees, so this split is insufficient for those runs. Terminal normalization already moves hidden state to DRAM and uses FP32 norm then BF16 head input/output (`tt/model.py:167–184`); do not reuse the earlier singleton terminal-storage diagnosis as if it were still uncorrected.

**Smallest distinguishing control:** force slot 0 alone through the existing masked B3 execution signature in a diagnostic, then compare with all three live using identical target inputs/state. If the target still differs, a B1-versus-B3 explanation is insufficient. If only B1 differs, compare the first linear attention output, postnorm, routing IDs/weights, routed expert output, shared expert output, full attention projections, and terminal output in that order to identify the earliest actual discrepancy.

## Paging and mask audit: plausible contrast, no established violation

### Each attention kernel still receives B1

Batched full-attention decode combines only normalization/projection/output work. It slices `projected[:, i:i+1, :]` and invokes `_full_attention` separately per slot (`tt/multichip_decoder.py:1590–1613`). Each call creates four local Q heads, one local KV head, dimension 256, and receives its own scalar current position and `(1,32)` page table (`:1697–1709,1721–1739`). The shared pool is `(32,1,32,256)`, BF16. There is no B3 paged-SDPA batch dimension to explain by itself.

### Causal padding can contain another request's data

The configured K chunk is 256 tokens. At positions 31..35 and 65..69, decode reads one 256-token chunk. At 257..261, it reads two chunks, rounded to 512 tokens. For slot 2, those 16 virtual pages map to physical `[20..31,0..3]`. The reset/cache-update contract predicts that physical pages 0/1 contain slot 0's live tokens in mixed execution and zeros in the slot-2-only run (`tt/generator.py:148–154`); no actual KV dump verifies those bytes. These are **future, causally masked positions**, not colliding live ownership.

For position 257, the selected decode scheduling gives reducer core 0 chunk `[1,2)` and core 1 chunk `[0,1)`; the other per-head cores have no work. In chunk 1, only keys 256 and 257 are visible: tile 0 columns 0/1 have zero mask, columns 2..31 and the remaining seven K tiles have negative infinity. Chunk 0 has 256 visible keys. Compute applies the mask before MAX/exp, reduces across all eight K tiles, and combines the two partial softmax statistics. Neither working core has an entirely masked local chunk with finite valid inputs.

Lowered evidence:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:130,449–453,487–527,598–646,687–696`;
- workload rounding/reversed core assignment in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:35–91`, and mask generation in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:251–304`;
- half-tile exp/masking helper in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:363–454`.

The low-head-count BF16 path internally uses 16x32 tiles and unclamped approximate exp, even with the caller's approximate-math flag false. The helper explicitly enables packer ReLU around exp packing and disables it afterward; the API documents negative outputs for very negative unclamped inputs, which this ReLU clears (`tt_metal/hw/inc/api/compute/eltwise_unary/exp.h:15–19`). Format reconfiguration preserves the ReLU setting (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h:434–445`; `.../common/inc/cpack_common.h:509–591`; ReLU configuration at `.../llk_lib/llk_pack_common.h:167–183`). No phase-local reconfiguration contradiction was found.

The external output remains **32x32 tiles**: the factory selects full output tiles, and output specs use a default tiled page configuration (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:544–551`; `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:624–634`). Consequently `copy_if(o,o,...)`'s 1024-element page size matches this output. Internal half tiles do not establish a conditional-copy bug.

**Adjudication:** the source contract intends exactly zero weight for finite future K/V. A mixed-only leak needs a mask/exp/packing defect, nonfinite data, or corruption not yet observed. This is weaker evidence than the concrete grouped precision dependency. If the first difference localizes to slot 2's SDPA with equal Q and equal live K/V, compare identical SDPA inputs with masked future pages zero versus finite/nonzero. Compare only visible values first; dump mapped page IDs so “equal KV” has an explicit logical/physical correspondence. This isolated contrast would make further mask investigation justified.

## Other inspected areas and demoted issues

- **Prefill cache-state equivalence is unmeasured.** Prefill is serial over slots, resets linear conv/recurrent state on `start=0`, uses identical per-request chunk planning and refreshes the selected page tables (`tt/generator.py:314–346`; `tt/prefill.py:25–40,126–143,195–219`). This supports equivalence but does not replace an exact all-chip state comparison. Save conv, recurrent, and live logical KV immediately after each request and after complete mixed prefill.
- **Recurrent layout changes have no index violation found.** Linear decode loops per request; recurrent state stays local `[1,8,128,128]`, conv `[1,3,2048]` (`tt/multichip_decoder.py:612–630,1615–1620`). Masked execution computes the candidate used for the output, then conditionally copies it to persistent state (`:498–513`). Direct-versus-transient output can select different layout/memory kernels for slot 0, but there is no source evidence that one request reads another's state.
- **Activity predicates are runtime device data.** Fixed capture occurs while only slot 0 is active (`tt/generator.py:419–437`), but the trace contains position/activity readers for all physical slots. Inactive current position becomes -1 while RoPE uses 0; active positions are advanced only under the scalar predicate (`tt/slot_state.py:47–80,202–228`). No Python conditional specializes the fixed trace to the capture-time active payload. Capturing with all rows active is a secondary diagnostic only if eager-versus-traced fixed execution disagrees.
- **Token/sampler commits are coherently ordered in source.** `_sampling_forward` copies old tokens, samples into the stable token buffer, merges active rows, and increments active seeds (`tt/generator.py:383–390`). `_ROWS` finishes all three reads before its possibly aliasing write (`tt/slot_state.py:83–108`). Masks and scalar position helper scratch extents fit their allocations. Distinct trace use alone does not establish a sampler bug; compare host argmax of the actual full-vocabulary logits with sampled tokens first.
- **Generic-op program reuse does not imply stale pointers.** Its hash omits runtime values, but cache-hit descriptor handling reapplies runtime arguments (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:52–75`; `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:820–824`). New slot pointer values need no new program. This rules out a diagnosis based only on equal program-cache counts.
- **Grouped capacity is sufficient by the current formula.** At 32 padded input rows, `dispatch_capacity(32,64)=32*8+64*31=2240` rows (`tt/multichip_decoder.py:801–805,1422`). There are 256 assignments and at most 64 local aligned regions. No planner/runtime capacity contradiction was identified. Counts/offsets should still be included in a routing-localized dump; their actual fixture values are unknown.
- **Top-k implementation/order differences alone are not a root cause.** IDs and values must remain paired; reordered expert sums can be mathematically equivalent. Diagnose actual expert-set/weight/combined-output differences rather than assuming a custom selector or different sort order is wrong.

## Completion boundaries and smallest useful follow-up

All host refreshes, model replay, sampling replay, and subsequent feedback are enqueued on CQ 0. Model replay and sampling replay use `blocking=False` (`tt/generator.py:486,527`). In this failing mixed arm `read_each=True`, token readback occurs after every resident model+sample pair; admission and final snapshots read tokens/positions/seeds, and checkpoints explicitly synchronize (`tests/run_generator_replay_probe.py:155–170,195–246`). The final-token assertion compares host-read values, so this is stronger evidence than merely reaching a host enqueue point. Counter advancement still proves only the checked state transitions, not attention/recurrent correctness.

Complete the remaining comparisons without changing the original acceptance check; the late diagnostic taps already cover some of these boundaries on chip 0:

1. **Persist correspondence first:** physical slot, logical request/fixture, layer index, chip coordinate, step, consumed token, current position, activity, trace ID, page-table prefix and stage name. Read all four chips for TP4 recurrent/KV/projection tensors; the probe's chip-0-only token helper is insufficient for sharded state. Gather all vocabulary shards for logits.
2. **Compare initial states before feedback:** exact per-chip conv/recurrent, and K/V indexed by live logical position through each slot's page table, after public prefill and before initial admission. Compare final-row logits with exact/top1 metrics in addition to the existing thresholds.
3. **Separate model from sampling for all five calls:** `decode_forward(..., return_logits=True)` executes the model and advances positions, but skips sampling (`tt/generator.py:526–532`). Read logits/state, invoke `_sample()` exactly once, then read tokens. Do not call model decode a second time for the same step. Save greedy IDs, top-two margins, chosen token logits and sampled tokens.
4. **Localize only the earliest differing operation:** for layer 0 compare normalized input, recurrence/conv candidate and committed state, attention partial/reduced/residual, postnorm, router IDs/weights/counts/offsets, routed and shared FFN outputs, and layer output. Then inspect the same boundaries for layer 3, including projected Q/K/V and pre/post-SDPA output. Later differences after the sampled token changes are consequences until demonstrated otherwise.
5. **Use equal-input controls:** teacher-force the target with the same solo token sequence and state to prevent autoregressive amplification from hiding the first difference. Compare the same B3 graph with other rows inactive/live, and separately B1 versus B3 slot 0. Only after localization run SR-off, FP32-DEST, masked-future-page, or eager-versus-traced variants. Compile/prepare diagnostic tensor layouts before fresh trace capture so readback instrumentation does not introduce new programs into the frozen-cache test.

Expected discrimination: equal model logits but different tokens points to sampling/feedback; a difference in prefetched cache state predates decode; equal input/state until routed MoE makes the grouped precision/SR hypothesis substantially stronger; equal Q/liveKV followed by differing SDPA output promotes the masked-tail hypothesis. A repeatable final-token difference alone does not choose among these causes.

## Report review

The initial findings were rechecked against production branches, the saved run's hashes, concrete page/position arithmetic, and C++/LLK consumers. Two independent xhigh reviewers checked lifecycle/evidence and paging/mask claims. Review corrected two tempting overstatements: the multirow linear norm is also ordered/refined, and the half-tile SDPA implementation still emits full-tile output. Both were removed as alleged direct bugs. Mask generation, online softmax reduction and packer-ReLU ownership showed coherent source contracts, so cache-mask leakage remains secondary. Later artifacts identify failing slots and grouped output differences; the SR-off-labelled run strengthens the rounding diagnosis but also exposes the separate B1/B3 boundary. An isolated source-pinned causal intervention and a complete passing correctness matrix remain unverified.
