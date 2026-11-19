# AutoDebug: sequential GDN decode PCC, optimized decoder stage 3

## Findings

**Two inherited numerical-contract mismatches are confirmed: HF persists the recurrent cache in BF16, while TT persists it in FP32; HF also performs q/k L2 normalization in BF16 before widening to FP32, while the functional and native TT paths do not.** Changing projection weights to BF16 does not control either difference. The recent in-place state update and batched convolution are not necessary for the failure: disabling both leaves the same 21 failing user-step coordinates.

These are the strongest concrete discrepancies found, not a proven complete explanation of the minimum PCC. Static inspection found no clear sign, scale, GQA mapping, padding, wave-count, or harness state-reset error. A cheap synthetic recurrence experiment did **not** reproduce the observed .97 dips. Actual checkpoint activations and TT intermediates are needed to separate cache rounding, normalization, native operand precision, and downstream amplification. Do not lower the .995 threshold or declare the failure benign.

This investigation was inspection-only: no devices opened, no hardware reproductions, no source/test/threshold edits, and no AutoDebug relaunch. Only this report and its requested copy were written.

## Revision and evidence boundaries

Checkout: `4c79322a4c142187308864977f00cbe5012c0352`. Measurements use pinned checkpoint `Qwen/Qwen3.6-27B`, revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`, Transformers `5.12.1`, Torch `2.11.0+cpu`. Paths below are relative to this checkout. Define:

- `M = models/autoports/qwen_qwen3_6_27b`
- `D = M/doc/optimized_decoder`
- `N = ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule`
- `HF = python_env/lib/python3.12/site-packages/transformers`

The first four hashes below were compared with recorded provenance, and match. Thus the measured optimized path is the immutable candidate, **not** the actively changing `M/tt/optimized_decoder.py`.

| Inspected source | SHA256 |
|---|---|
| `M/tests/run_decode_sequence.py` | `8ddbf0ba13196b2b05498720d1b3ce66157e9e4b001dd0b877cd7a25bd16f68e` |
| `D/candidates/queued_search.py` | `bf5aa8131896232fea2ecc717544d53bf1b0aa4420acfba98993b8e083a4e1b7` |
| `M/tt/fused_decoder.py` | `6e2429f7440438719c1c3bd42e8dc12d34c22281018602d2eb00a291ac739a31` |
| `HF/models/qwen3_5/modeling_qwen3_5.py` | `395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6` |
| `M/tt/functional_decoder.py` | `476f9d99fa90a1268f264964037dbc6837b320adc9f48d172dc1156b8b35962d` |
| `HF/cache_utils.py` | `2783b2702910dbb6e70aa05a093f71e1c51c110d7fa73c904fc38d12e5882545` |

Provenance hashes the HF modeling file but does not separately hash `cache_utils.py`, the TT binary, or every imported helper (`M/tests/evidence.py:23-42`). The cache conclusion is directly verified in the installed dependency, with this historical provenance limitation. No applicable `AGENTS.md` was found for these paths.

## Direct observations from the JSON files

All cases have a 128-token prefix, 128 independently sampled teacher-forced hidden inputs, and layer-local real checkpoint weights. Step indices below are zero-based; position is `128 + step`. PCC is over one user's 5120-dimensional decoder output, not the complete batch (`run_decode_sequence.py:115-135`).

| Record in `D` | Prefix PCC | Minimum user PCC | Worst `(step,user)` | Cells below .995 |
|---|---:|---:|---|---:|
| `sequence_early_0_b32.json` | .99986860 | .97039646 | (58,30) | 21/4096 |
| `sequence_control_oldpath.json` | .99986860 | .97039407 | (58,30) | 21/4096 |
| `sequence_fused_0_b32.json` | .99996375 | .95983924 | (99,27) | 14/4096 |
| `sequence_control_bf16attention.json` | .99995334 | .95477284 | (99,27) | 14/4096 |
| `sequence_control_hifi4attention.json` | .99988806 | .96224893 | (99,27) | 20/4096 |
| `sequence_early_0_b1.json` | .99986013 | .99520265 | (119,0) | 0/128 |
| `sequence_early_3_b1.json`, full attention | .99985224 | .99935560 | (103,0) | 0/128 |
| `sequence_early_3_b32.json`, full attention | .99984549 | .99943929 | (1,15) | 0/4096 |

The stress controls explicitly use `queued_search`, and `oldpath` omits both `inplace_gdn` and `batched_conv` (`D/stress_controls.py:4-8`). Early versus oldpath has exactly the same 21 failing coordinates, mean absolute PCC difference approximately `9.84e-7`, maximum difference `.00056495336`, and 1793 bit-identical PCC cells. They are very close, not entirely bit-identical. Their first failure is step 19/user 20. Fused first fails at step 41/user 27; BF16 attention first fails at step 56/user 6.

The worst dips are isolated readouts rather than a monotonic PCC collapse. For user 27, steps 98/99/100 are approximately `.99984/.97573/.99979` in early optimized and `.99985/.95477/.99978` in BF16 attention. For user 30, early steps 57/58/59 are `.99996/.97040/.99990`. Medians remain approximately `.99989` optimized and `.99993` fused/BF16 attention. Shared bad coordinates establish input correspondence between these B32 runs; the JSON does not contain the head-level values needed to explain each dip.

**B1 is not the same stream executed with a different batch kernel.** Prefix shape changes RNG consumption before the decode tokens are generated (`run_decode_sequence.py:28-37`), and B32 tests 32 times as many user-step cells. The HF layer constructor also consumes RNG before the fixture is sampled. Reproducing the measured inputs requires preserving this construction/RNG sequence or saving the inputs. B1's near-threshold pass cannot by itself establish a batch-specific native defect.

## Confirmed discrepancy 1: recurrent cache storage and rounding

The FP32 recurrence in the HF function is only the **working dtype**, not the persistent cache dtype.

1. The harness constructs a BF16 HF layer and `DynamicCache(config=c)` (`run_decode_sequence.py:32-35`). The cache maps `linear_attention` to `LinearAttentionLayer` (`HF/cache_utils.py:1024-1035,1473`).
2. During the prefix, HF calls `update_conv_state` before `update_recurrent_state` (`HF/models/qwen3_5/modeling_qwen3_5.py:485-487,548-550`). The convolution projection is BF16.
3. `LinearAttentionLayer.lazy_initialization` sets `self.dtype = conv_states.dtype`, hence BF16 (`HF/cache_utils.py:925-930`). It allocates recurrent storage with **`dtype=self.dtype`**, even when its incoming recurrent tensor is FP32 (`:935-937`).
4. `update_recurrent_state` uses `self.recurrent_states.copy_(recurrent_states)` (`:986-990`), rounding the FP32 final state into the BF16 buffer after the prefix and after every decode call.
5. The next HF recurrent calculation widens that already-rounded state to FP32 (`modeling_qwen3_5.py:334-349`), computes the output/update, and caches BF16 again.

TT allocates persistent state as FP32: `functional_decoder.py:108-114`, `fused_decoder.py:162-170`, and measured `queued_search.py:264-272`. Functional `_delta_step` copies an FP32 update to that buffer (`functional_decoder.py:150-155`); functional prefill likewise copies FP32 (`:393-417`). Native initial-state conversion and final-state specs are FP32 (`N/chunk_gated_delta_rule.cpp:281-290`; `N/device/chunk_gdn_device_operation.cpp:270-276`). Measured phased, public, and in-place paths retain FP32 storage (`queued_search.py:805-815,819-850`).

Ignoring the other arithmetic differences, the state machines are therefore:

```text
HF: S_next = BF16(R(FP32(S_cache), q, k, v, beta, g))
TT: S_next = native_FP32_stored_update(S_cache, q, k, v, beta, g)
```

BF16 projection weights do not change this distinction. The harness snapshot also preserves `t.dtype`, so restoring TT state does not accidentally emulate HF's BF16 cache (`run_decode_sequence.py:66-67`).

Why this fits the sequence boundary: prefix outputs are computed before final cache storage rounding can influence a later call. One-step comparison sees only the initial discrepancy. Long decode repeatedly feeds differently rounded state into subsequent updates. Full attention does not use this recurrent cache. All failing GDN implementations inherit the FP32 storage choice, and B1 exercises different inputs/fewer cells.

This is a confirmed numerical mismatch, not evidence that FP32 persistent state is mathematically wrong for GDN or that the installed HF cache should be changed. Whether the pinned BF16 HF execution is the target determines the required decoder behavior.

## Confirmed discrepancy 2: q/k normalization before widening

HF `l2norm` executes the following operations in the input dtype (`HF/models/qwen3_5/modeling_qwen3_5.py:240-243`):

```text
p = BF16(x * x)
r = BF16(sum(p))
r = BF16(r + 1e-6)
u = BF16(rsqrt(r))
x_normalized = BF16(x * u)
then widen to FP32 for recurrence
```

This ordering is explicit for both chunk and recurrent functions (`:258-264,330-336`). Torch may accumulate a BF16 sum internally in higher precision; the returned BF16 reduction and subsequent BF16 stages remain material rounding boundaries. HF applies `K**-0.5` to the widened normalized query afterward (`:340-341`).

Functional `_l2` instead widens the original input first, then squares/reduces/normalizes (`functional_decoder.py:136-144,379`). Thus replacing native decode with the current `_delta_step` is a useful native-versus-composed control but does **not** independently reproduce HF normalization or cache precision.

The native flat-input path enables normalization automatically for chunk size 32 (`N/chunk_gated_delta_rule.cpp:202-208,239-245`). The selected helper obtains squared row norms from the diagonal of `x @ x.T`, adds epsilon and takes rsqrt in the native FP32 destination path, then normalizes using diagonal matrix multiplication. It folds query scale into that norm (`N/device/kernels/compute/chunk_gdn_math.hpp:426-491,797-829`). It does not implement HF's BF16 product/reduction/rsqrt/output boundaries. Both B1 and B32 use this math.

The native public flag `use_qk_l2norm` is rejected at entry (`chunk_gated_delta_rule.cpp:141`), but flat input automatically selecting internal normalization is intentional (`:239-245`); that apparent API discrepancy is not an omission of normalization in the repro.

## Cheap CPU checks and their limits

A CPU-only probe imported the installed cache class and HF `l2norm`, with no TTNN or checkpoint execution:

- After initializing a BF16 convolution cache, passing a tiny FP32 recurrent tensor into `LinearAttentionLayer.update_recurrent_state` stored BF16. An input element `1.001` was stored as `1.0`; maximum probe storage error was `.00100004673`.
- For BF16 `x = [.25,.125,.5,-.75]`, the product, sum result, epsilon addition, rsqrt, and final HF normalized tensor were all BF16. HF produced `[.263671875,.1318359375,.52734375,-.7890625]`; widening before normalization produced approximately `[.264906317,.132453158,.529812634,-.794718981]`. The difference is real even with identical upstream input.

A separate CPU synthetic recurrence used seed 27, four Torch threads, `B=4,H=48,K=V=128`, 128 prefix steps and 128 decode steps, identical random upstream tensors, and constant decay gates derived from real layer-0 A/dt values at `a=0`. It took about 4.8 seconds. Comparing BF16-persisted versus FP32-persisted state alone gave minimum normalized readout cosine about `.9999257` and final relative state RMS error about `1.408%`. Changing normalization as well gave minimum cosine about `.9999186` and state error about `1.441%`. BF16 state rounding occurred once after the prefix, then after each decode output. Readouts used unweighted per-head RMS normalization, then cosine over flattened user/head/value dimensions; this was not centered PCC and omitted output BF16 casting, Z gate, norm weights, projection, and MLP. The wider normalization branch used `F.normalize` (norm clamp), rather than adding epsilon under the square root; random norms were far above epsilon. **This generic check did not recreate a sub-.995 readout.** It is illustrative, not a fixture-matched PCC result, native arithmetic simulation, or proof of attribution.

The checkpoint permits very different retention by head. `A_log` spans approximately `[-5.5625,-1.08594]`, `dt_bias` approximately `[-5.71875,19.25]`; at `a=0`, illustrative `exp(g)` is approximately `.999938` in head 7 and `.006594` in head 32. These are conditional values, not recovered gates at the failed coordinates.

Do not describe this as inevitable recurrent instability. For ideal normalized k, the homogeneous state map is `exp(g) * (I - beta*k*k.T)`. With `g<=0` and `0<=beta<=1`, it is non-expansive. Quantization differences and small readouts can nevertheless matter. The core readout can contain cancellation or small projection onto q, followed by per-head RMS normalization; a small absolute core error may then produce a larger angular error downstream. No captured head readout, denominator, or state at `(58,30)` / `(99,27)` establishes that amplification here. This remains a falsifiable hypothesis.

## Native dispatch, math, and state audit

### Concrete runtime geometry

The measured decoder supplies flat tile q/k/v with widths `2048/2048/6144`, `Hk=16`, `HV=48`, `K=V=128`. Decode has one live row, padded to 32; gates have row 0 live and tails zero (`queued_search.py:443-476,477-484,817-849`). The public oldpath sees `T=32`, `NC=1`; the in-place primitive receives the same effective chunk. GQA repeats each key head three times.

- **B1:** explicit phased prep/scan, `BH=48`, `Ct=1,Kt=Vt_full=4` (`fused_decoder.py:476-517`, measured `queued_search.py:775-816`). Phased scan chooses its own value split; on an 11x10 grid, `NV=2,Vtl=2` because four receivers/head cannot fit but two can (`N/device/chunk_gdn_phased_program_factory.cpp:149-172`).
- **B32 optimized:** `BH=1536`; configured `wave_heads=48,num_receivers=1,producer_pool=True` produces 32 waves, each owning one user's 48 value heads (`queued_search.py:819-843`; public wave setup in `N/chunk_gated_delta_rule.cpp`). Each wave has `NC=1,Ct=1,Kt=4,Vt=4,NV=1,Vtl=4`.
- **B32 fused baseline:** public native path with its default wave planner, not the optimized path's explicit policy. Its continued failure supports a shared numerical discrepancy but does not make the two schedules identical.

For the configured optimized producer pool, 48 heads times one chunk supplies exactly 48 jobs, so the pool has 48 producers, one home producer per head and no extra chunk-splitting producer. Each receiver consumes one chunk. There is no partially filled head wave at B32 and no partial value tile: `1536 % 48 == 0`, `128 % 32 == 0`.

For example, value head 30 of user b maps to key head 10. Flat q/k tile pages are `64*b + 40..43`; v pages are `192*b + 120..123`; gate page is `2*b` with source column 30; recurrent state starts at page `768*b + 480` and has 16 pages. With wave offset `48*b`, the output's four pages start at `192*b+120` and final state's 16 pages start at `768*b+480`. This agrees with the reader/writer maps; runtime offsets are refreshed for producer, state reader, and output writer (`N/device/chunk_gdn_device_operation.cpp:302-338`). No concrete cross-user addressing defect was found.

### The recurrence is algebraically consistent

For one live token, both HF and functional TT implement:

```text
H = exp(g) * S
delta = beta * (v - k.T @ H)
S_new = H + k @ delta
o = (q / sqrt(128)).T @ S_new
```

See HF `modeling_qwen3_5.py:352-367` and functional `_delta_step:141-156`. Native prep/scan expresses the same recurrence through `v_new`, `q_decay`, `intra`, `k_dec_t`, and `dl` (`chunk_gdn_math.hpp:1058-1178`). Both output and state are formed as sums of two matrix products. No missing decay, beta, query scale, or sign was identified.

Padding is neutral in exact arithmetic: zero beta tails make their state updates zero; zero g tails preserve decay. The strictly lower WY matrix is zero for this one-live-token case, so its inverse is identity. Query/key normalization of finite padded inputs does not make zero beta nonzero. Consequently an inherently ill-conditioned 32-row inverse is not a good explanation of the **per-token** failures. A prefill state discrepancy is still possible and should be measured, rather than eliminated by a globally high prefix-output PCC.

For the selected SFPU solve prep path, normalized q/k buffers each reserve/push four tiles. The norm diagonal scratch waits/pops one tile for q and one for k; k's normalized output is waited on at its first `k_beta` consumer and q at its first q consumer. For the configured **B32 NV1 scan receiver**, input state waits for 16 tiles; `nkd/vbeta` wait for four each, `Tinv` for one; `vnew` produces four. Output reserves/pushes four; final state reserves/pushes 16. B1 phased scan with `NV=2` instead uses eight state/final tiles and two `vbeta/vnew/output` tiles per receiver; shared K-dependent `nkd/qdecay` remain four and `Tinv` one. `vnew` and old state are popped only after both output and state blocks have issued their consumption (`chunk_gdn_math.hpp:797-847,1058-1178`). The fused handoff has launch-reset semaphores, producer credit initialization, receiver reservation before credit, and a data-write barrier before VALID; at `NC=1` its ring does not wrap (`N/device/chunk_gdn_fused_program_factory.cpp:187-209,256-276`; `N/device/kernels/dataflow/writer_chunk_gdn_fused.cpp:181-201,383-420`). This audit found no unmatched count at the passing/failing geometries. It is not a hardware proof of all ordering or target-specific helpers.

### FP32 storage is not IEEE FP32 matrix operand precision

Every native program requires HiFi4, FP32 destination accumulation, and approximate math disabled (`N/device/chunk_gdn_compute_config.hpp:17-40`). That does not give IEEE FP32 operands to all matrix operations. `tt_metal/jit_build/genfiles.cpp:1069-1075` selects the Tf32 unpack destination under FP32 destination accumulation for this format mix; `tt_metal/jit_build/data_format.cpp:147-155` assigns it to Float32 source buffers. Blackhole's `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:311-322` documents Float32-to-TF32 source-register conversion. The GDN compute config supplies no unpack-to-destination override. Native scan repeatedly consumes state through matrix operations, including `(dl*I) @ S` (`chunk_gdn_math.hpp:1143-1168`). Thus use the precise description **FP32 state storage and destination accumulation with matrix-engine operand precision**, not “full FP32 recurrence.” The recorded early B32 log identifies the architecture as Blackhole (`D/sequence_early_0_b32.log:6`).

This is a shared native precision difference worth controlling if cache/norm parity is insufficient. No target-specific intrinsic, register-mode, reconfiguration, or queue contract violation was established that explains the measured dips. Claiming a low-level native bug from dtype names or these PCCs alone would overreach.

## Harness and synchronization adjudication

The visible save/restore sequence is coherent:

1. HF evaluates its prefix once (`run_decode_sequence.py:38-41`). TT evaluates the prefix, and a blocking host read obtains both state snapshots (`:65-67`). The host snapshots preserve each TT buffer's dtype and layout.
2. TT warmup completes at `synchronize_device` (`:95-96`); capture uses the stable input and existing state buffers (`:97-99`).
3. Both writable request buffers are restored (`:100-101`). The code copies into existing tensors; the decoders do not replace `state.conv` or `state.recurrent`, so `buffers` does not retain stale Python tensor references.
4. Each input refresh and pending state restoration completes before timing at `synchronize_device` (`:104-109`). Trace replay is blocking (`:111`), followed by output readback (`:113`). CQ0 is used consistently. Host enqueue completion alone is not being treated as a device result.
5. HF then advances exactly once with that same tested token (`:115-119`). TT warmup/capture does not advance HF cache. There is no apparent off-by-two reference or request-state update.

Shared convolution exchange buffers and semaphores are not snapshotted. A hidden scratch lifecycle issue is not proven absent by the harness; however, the new batched-convolution path bypasses that decode exchange flow and has the same failure coordinates as oldpath. It therefore cannot explain the shared symptom on its own. Eager-versus-traced multi-step state equivalence is still an unperformed decisive control.

The different history sizes are harmless on this fixture: HF retains four convolution inputs while TT retains three. A width-4 causal convolution producing the newest token uses the previous three plus the current input; HF takes the last output from the concatenated history (`modeling_qwen3_5.py:229-236`). The extra oldest HF history item is not used in that output. Prefix length 128 avoids the documented shorter-than-kernel HF edge case.

## Competing explanations, ranked after claim review

| Candidate | Direct code evidence | What it explains | Unverified part / smallest falsifier |
|---|---|---|---|
| **HF BF16 recurrent-cache boundary versus TT FP32 storage** | Confirmed allocation/copy dtype difference | Shared failure across native schedules/weight precisions; long versus short; full-attention contrast | No measured state delta or exact-output ablation; match only the cache boundary on identical upstream tensors |
| **BF16 HF q/k normalization versus TT wider normalization** | Confirmed operation/dtype ordering difference | Shared functional/native discrepancy; weight precision control cannot remove it | Also applies in prefix and B1; 2x2 cache/norm parity control needed |
| Native matrix-engine operand precision / upstream and downstream rounding | Confirmed different arithmetic paths, no proven erroneous kernel contract | Can interact with cache differences and small readouts | Needs shared-input native-versus-composed recurrence and head-level error measurements |
| Harness restore / trace hidden state | No visible restore contradiction | Could explain sequence-only behavior if state differs after restore/replay | Eager-versus-traced state comparison from identical snapshots; currently no such evidence |
| Recent in-place GDN / batched convolution | Removed in oldpath with same failed cells | Can perturb individual values slightly | Not necessary for failure; cannot headline as its cause |
| BFP8 weights or HiFi2 alone | BF16/HiFi4 and fused BF16 also fail | Contributes some error; controls reduce bad-cell count | Cannot explain shared failure alone |
| Inherently unstable exact delta recurrence or one-token WY inverse | One-token inverse is identity; ideal recurrence is non-expansive | No established fit | Demoted; do not infer growth from a low output PCC |

This ranking distinguishes observed discrepancies from explanations. The exact .9704/.9598/.9548 extrema remain unisolated. In particular, a synthetic check that shows small cosine differences does not prove the real fixture insensitive, and matching bad coordinates does not prove which head or arithmetic stage causes them.

## Narrow discriminating controls for the caller

These are proposed follow-ups, not experiments completed here. Keep the original acceptance gate, original sequence fixture, and measured source snapshot intact; run diagnostic variants separately.

1. **Log actual cache dtypes and compare prefill final states.** Confirm HF convolution/recurrent storage is BF16 and TT recurrent storage FP32 immediately after prefix and after step 0. Compare per-user/per-head recurrent values, not only the prefix's aggregate output PCC. Separate a different starting state from subsequent per-step drift.
2. **Run a shared-upstream 2x2 precision control.** Save exact fixture prefix/tokens or preserve the full constructor/RNG sequence. Obtain identical q/k/v/g/beta from one upstream implementation. Independently toggle (a) HF BF16 versus wider q/k normalization and (b) BF16 round-trip versus FP32 persistent recurrence. For each, pass core output through the same HF downstream norm/gate/projection/residual/MLP. Preserve the 128-step result grid and inspect `(19,20),(58,30),(99,27),(123,13)`. This distinguishes cache rounding from L2 effects without confounding projections. A separately labeled HF FP32-cache reference can help attribute the mismatch; it must not silently replace the acceptance reference.
3. **Split native recurrence from the complete decoder.** With identical upstream inputs and initial state, compare native one-token scan against composed `_delta_step`, then compare both after explicitly matching HF normalization/cache semantics. Current `_delta_step` alone is not an HF-precision oracle. Snapshot old state, normalized q/k, exp(g), prediction, delta, new state, and pre-RMS readout at a failing coordinate. If shared-input native differs materially after matching semantics, investigate its TF32/matrix/SFPU arithmetic before changing projection policies.
4. **Compare eager and traced sequences.** From the same saved device state/input tensors, run 128 eager tokens versus 128 captured replays, recording recurrent and convolution state after each token. Check exact restoration before step 0. Divergence confined to trace redirects investigation to runtime arguments, lifetimes, hidden scratch, or queue state. Agreement directs it back to math/reference precision.
5. **Make B1 a matched-user control.** Extract the exact B32 prefix, token stream, and state for user 30 or 27, run that same stream as B1, and compare raw core outputs/states as well as final PCC. Where supported, compare B1 phased versus explicit one-wave fused scheduling. A new `--batch 1` RNG stream is not this control.
6. **Measure the proposed amplification.** At the shared bad cells, record per-head `||q.T @ S_new||`, RMS denominator, normalized core angular error, and downstream attention/residual/MLP errors. A small denominator or cancellation must be demonstrated, not inferred from the final PCC. Also compare final output and residual-subtracted contribution to locate which branch changes.

## Smallest decoder-local corrections if controls confirm attribution

For fidelity to this pinned BF16 HF execution, preserve stable device buffers and the native FP32 state ABI. A minimal cache-parity correction is a device-only `FP32 -> BF16 -> FP32` round-trip of the completed recurrent state, copied back into the existing request-state buffer **once per externally visible HF-equivalent cache-update boundary**. The first boundary is after prefix state generation; each decode adds one boundary. For an in-place wave loop, apply it after all disjoint head waves, before the next token consumes state. Do not round the state before computing the current token's output: HF returns that output from the FP32 working update, then rounds only the cached state. Do not insert BF16 rounding at every internal 32-token prefill chunk, because HF does not persist between those internal chunks. Longer internally split prefills need the same public-call boundary discipline.

For normalization parity, add a decoder-local HF-compatible q/k helper: square in BF16, return the sum in BF16, add epsilon in BF16, rsqrt in BF16, multiply in BF16, then widen for recurrence and apply query scale afterward. This must occur per 128-element head, not across the 2048-element flattened width. Pre-normalized inputs must bypass the native flat-input automatic normalization (or use an explicitly defined local adapter); normalizing twice would invalidate the control. A rank-4/pre-normalized public path may introduce its own packing/scale rounding, so verify effective dtypes there rather than assuming parity.

Start with these boundaries independently, assess all original sequence cases, then combine only if justified. They are proposed interventions, not verified fixes. Keep changes decoder-local: do not change global GDN semantics, HF dependency source, thresholds, or move into full-model work. If matching cache/norm semantics does not remove the failure, retain the evidence and pursue the shared-input native arithmetic comparison rather than claiming the discrepancy fixed.

## Other potential issues and test gaps

- HF gated RMSNorm applies BF16 normalization/weight rounding and evaluates SiLU on FP32 gate before its final cast (`modeling_qwen3_5.py:193-202`); TT's fused normalization/gate chain has different packing/fusion boundaries (`fused_decoder.py:266-276`, measured `queued_search.py:484-494`). These are further numerical differences, not a demonstrated explanation of the sparse dips.
- HF fallback convolution packs BF16 before separate SiLU (`modeling_qwen3_5.py:234-236,497`); native convolution and measured batched convolution target FP32 accumulation plus SiLU before the BF16 result pack (`queued_search.py:496-519`). This can change q/k/v directions. It is shared arithmetic behavior, not proof of the newer batched implementation being wrong.
- Flat native unit tests explicitly widen BF16 q/k before passing them to the golden (`tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_flat_gates.py:98-110`), use global output/state PCC, and do not exercise this 128-call BF16 HF-cache sequence. Native-versus-phased bit equality verifies shared execution math, not equivalence to the HF cache's rounding behavior. Test success there cannot settle this failure.
- Global prefix-output PCC does not validate per-head final state precision. Likewise, short one-step comparisons are insufficient coverage for a persistent rounding contract.
- Binary/source correspondence and dependency-file provenance could be recorded more completely for future diagnoses. No evidence of a stale binary or changed cache dependency was found; neither is promoted as the cause.

The recommended next intervention boundary is the decoder's HF-equivalent cache publication and q/k preparation. Source inspection supports that boundary strongly; hardware attribution and acceptance remain open until the discriminating controls are measured.
