# Native recurrence persistence precision diagnosis

CPU diagnosis, 2026-11-13. **Persistence rounding alone reproduces the failing cross-dtype quality gate.** With the exact B10/rank inputs from `native_recurrent_perf19_v2.json`, two otherwise identical FP64 reference recurrences produce minimum per-user output PCC **0.9817125822** after 702 identical-input updates when one persists BF16 and the other FP32. After 50 changed-input updates, the corresponding minimum is **0.9999213869**. The original hardware run remains `fail`; no threshold, failed artifact, helper, probe, production source, or default was changed.

This explains why the public FP32 path can pass its FP32 oracle and still fail against the current BF16-state decoder. It does not claim that native FPU error is zero: native versus its FP32 oracle is itself 0.9985420974 at the worst user. The CPU result isolates persistence as a sufficient cause of cross-dtype rejection; the observed hardware cross-PCC of 0.9897577436 combines persistence and native arithmetic differences and need not equal the oracle-to-oracle score.

## Exact failed hardware evidence

`native_recurrent_perf19_v2.json` uses B1/3/9/10/17/32, normal inputs, FP32 log decay, `custom_variant=cumulative_exp_scalar`, three changed-input banks, two warmups, seven samples of 100 replays, and 50 queued changed steps. B1/3/9 finished; B10 stopped in `check_step` after `timed_steps=2+7*100=702`. The worst failed cross-control user is rank0/user0. The dictionary comprehension raises before assigning `native_vs_controls`, so `timed_final` contains dtype-matched scores but no completed cross-control table; the failure values are retained in the traceback.

| Hardware comparison at B10 | Output PCC minimum | State PCC minimum |
|---|---:|---:|
| Custom BF16 versus BF16-persistence FP64 oracle, 702 fixed-input steps | 0.9999999139 | 0.9999999785 |
| Custom FP32 versus FP32-persistence FP64 oracle, 702 fixed-input steps | 0.999999999998 | 0.9999999999999 |
| Public native FP32 versus FP32-persistence FP64 oracle, 702 fixed-input steps | 0.9985420974 | 0.9999159293 |
| Public native FP32 versus custom BF16, 702 fixed-input steps | **0.9897577436 — fail** | Not evaluated after output assertion |
| Public native FP32 versus custom BF16, 50 changed-input steps | 0.9998741049 | 0.9999511859 |
| Public native FP32 versus custom FP32, 50 changed-input steps | 0.9999593107 | 0.9999812995 |

The B10 latencies (custom BF16 90.08178 us, custom FP32 96.45876 us, public native FP32 468.24093 us) were collected before the final gate failed. They are failed-candidate evidence, not valid kept-path performance. Earlier fully finished B1/3/9 comparisons also show public-wrapper overhead; none establishes a performance rejection of the compact prepared endpoint.

## Controlled CPU reproduction

Used `python_env/bin/python`, Torch `2.11.0+cpu`, four CPU threads, no TTNN import. The default system `python3` lacks Torch; no installation or environment mutation was needed. Runtime was 58.46 seconds. No hardware functions were called.

Recreated the exact generator sequence: generators `52813+97*10+rank`; initial `[10,12,128,128]` random state times 0.015625 plus rank*0.001, rounded to BF16; then all three packed QKV input banks using the original probe's random draws, user/rank/step offsets, FP32 log g and BF16 beta. Both trajectories start from the same BF16 state bytes. Q/K normalization and decay are precomputed in FP64 once per fixed bank, equivalent to the deterministic reference recomputing them each step. Every state update and current-step readout uses FP64. Only the persistent state cast differs: BF16 versus FP32; readout occurs before that cast, then outputs cast to FP32 as in the existing reference. For changed inputs, bank index is `step%3`; 50 steps end with bank1.

| Input trajectory | Minimum per-user output PCC | Minimum per-user state PCC | Minimum per-head output PCC |
|---|---:|---:|---:|
| 1 fixed step | 1.0000000000 | 0.9999985527 | 1.0000000000 |
| 4 fixed steps | 0.9999957490 | 0.9999949598 | 0.9999928165 |
| 50 fixed steps | 0.9979957443 | 0.9997767681 | **0.9684378883** |
| 100 fixed steps | **0.9858159863** | 0.9990456250 | 0.5399643187 |
| 200 fixed steps | **0.9817708974** | 0.9987612973 | 0.1521281841 |
| 512 fixed steps | **0.9817125822** | 0.9987570693 | 0.0343782066 |
| 702 fixed steps | **0.9817125822** | 0.9987570693 | 0.0343782066 |
| 50 changed steps, bank0/1/2 cycling | 0.9999213869 | 0.9999683863 | 0.9989561174 |

At 702, rank0/user0 again gives the minimum per-user score. Rankwise minimum output PCC is 0.9817125822 / 0.9981588875 / 0.9979997059 / 0.9991874402. The largest absolute output discrepancy divided by that user's maximum FP32-reference output is 0.197835. At 50 changed steps all 480 rank/user/head output comparisons remain above 0.995.

The low-signal head is informative. Rank0/user0/heads0..2 share normalized scaled `q dot k = -9.473201549e-6`; the worst head at 702 is head2, PCC 0.0343782. With constant input, exact decay contracts state components outside K's direction, and the stationary state lies in K's direction:

```
S_star = k[:,None] * (beta*v)[None,:] / (1 - decay + decay*beta*dot(k,k))
y_star = dot(q_scaled,k) * beta*v / (1 - decay + decay*beta*dot(k,k))
```

Thus a nearly orthogonal Q/K pair has very small exact output. Repeated BF16 rounding can retain a different quantized residual around that fixed point; projecting it through Q changes output correlation substantially even while state correlation remains high. The measured saturation from 512 to 702 and small Q/K dot support this interpretation. This is not evidence of a NoC, cache ownership, stale replay, or native ring bug. Those still need their own device gates, but no such mechanism is required to reproduce this quality failure.

## Candidate and harness implications without changing acceptance

1. **Public FP32 state is not an accepted drop-in replacement for the current BF16 persistence contract.** Keep this failed gate and do not promote its B10 timing. Its matched FP32 oracle pass demonstrates correct execution of a different precision policy, not preservation of the accepted default's result. This failure alone does not reject compact preparation, multiple-head scheduling, direct output packing, or native arithmetic with BF16 persistence.
2. **The prepared BF16 branch is the coherent adaptation to test next.** It keeps BF16 state throughout, expands one head locally, computes live update/readout in FP32, and rounds only persistence. There is no full DRAM FP32-to-BF16 restoration. Its frozen implementation already follows this ordering. Passing the same default-BF16 and BF16-persistence reference gates at unchanged thresholds would address the persistence mismatch, but is not guaranteed: TF32-class native contractions and its FP32 state-seed copies can still differ from the custom SFPU implementation.
3. If matched BF16 long-drift gates fail, inspect native arithmetic separately instead of changing the state policy again. A concrete bounded adaptation is direct FP32 unpack for `cb_sd` when native `rank1_update` seeds DST via `copy_tile`, preventing an extra FP32-to-TF32 truncation before adding the rank-one update. Compare that isolated fidelity change; native FPU prediction/readout still have TF32-class operands. A hybrid precise readout/decay is a further candidate if material. Do not classify a first configuration/compiler failure as a family rejection.
4. Improve diagnostic collection by computing/storing the complete cross-comparison matrix and accumulating failures before raising; keep the final failure status and 0.995 threshold. The current early assertion discarded later useful cross-control scores. Keep explicit dtype/trajectory labels, and separate collected timing from kept results. Timing may reset state before each sample for identical starting conditions, with reset outside measurement, **but the existing 702-step semantic failure must remain a separate retained stress test**. Resetting samples is not a quality repair or a way to erase the failure.
5. The frozen prepared probe currently compares every native candidate against both custom dtypes at per-head granularity. The CPU references themselves violate that cross-dtype expectation: a faithful BF16 recurrence is not supposed to equal the FP32 trajectory. At the worst fixed-input head the two oracle correlations are 0.0344; two vectors both having PCC >=0.995 to a third require mutual PCC >=`2*0.995**2-1 = 0.98005`. Therefore those oracle targets cannot both be universal per-head acceptance targets for this trajectory. This is a reference-contract incompatibility, not permission to silently loosen a gate. **No frozen probe gate was changed.** The initial two-step smoke and 50-changed-step stress remain applicable. Before long fixed-input ranking, parent must explicitly document the roles of the unchanged BF16 default/oracle acceptance targets and the FP32 precision-control diagnostic; until then retain a failing/incompatible all-controls result, while collecting matched-family evidence. Do not claim an overall pass by dropping a comparison or excluding the low-signal head.
6. Real-weight decoder prefill/decode PCC, trace ownership, watcher stress, context support, and final default-path performance remain required. Synthetic matched-family evidence cannot substitute for them. A full-model or vLLM task is unnecessary for this diagnosis.

## Reproduction/provenance

The CPU tool invocation was `python_env/bin/python - <<'PY' ... PY`, using the original probe's host generation and exact reference recurrence below; its output printed each rank's two trajectory minima and the summary table. Full numeric results and input hashes are embedded below, so the diagnosis does not depend on a transient Python process. Source-derived precomputation only hoists deterministic normalization/exp for identical bank tensors.

```python
# B=10; rank generators and initial states are created before drawing any packed bank.
gens = [torch.Generator().manual_seed(52813 + 97*10 + r) for r in range(4)]
initial = [(torch.randn(10,12,128,128,generator=g)*.015625+r*.001).bfloat16()
           for r,g in enumerate(gens)]
users = torch.arange(10).reshape(10,1,1).float()
heads = torch.arange(12).reshape(1,1,12).float()
bank = []
for step in range(3):
    packed = [(torch.randn(10,1,2560,generator=g)*.125+users*.002+r*.01+step*.02).bfloat16()
              for r,g in enumerate(gens)]
    gs = [(-.04-users*.001-heads*.002-r*.003-step*.004).float() for r in range(4)]
    betas = [(.125+users*.003+heads*.015+r*.007+step*.01).bfloat16() for r in range(4)]
    bank.append(list(zip(packed,gs,betas)))

def prepare(host):
    packed,g,beta = host
    q = packed[:,:,:512].reshape(10,4,128).double()
    k = packed[:,:,512:1024].reshape(10,4,128).double()
    q = q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)
    k = k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
    return (q.repeat_interleave(3,1)*128**-.5, k.repeat_interleave(3,1),
            packed[:,:,1024:].reshape(10,12,128).double(),
            g.double().reshape(10,12,1,1).exp(), beta.double().reshape(10,12,1))

def update(inputs,state,persistence_dtype):
    q,k,v,decay,beta = inputs
    decayed = state.double()*decay
    delta = (v-(decayed*k.unsqueeze(-1)).sum(-2))*beta
    live = decayed+k.unsqueeze(-1)*delta.unsqueeze(-2)
    return (live*q.unsqueeze(-1)).sum(-2).float(), live.to(persistence_dtype)

# For each rank, run BF16 and FP32 states from the same initial bytes:
#   702 updates with prepare(bank[0][rank]); then reset and 50 with bank[step%3].
# PCC: flatten all 12 heads per user, center each vector, normalized dot;
# separately repeat PCC per head. No native/TTNN operations enter this experiment.
```

Failed hardware JSON SHA-256: `5f0bb9fe37d986e7fa911d7258b7d44d4d74ac35c4ed736d85ac38a91e94bd4e` (verified unchanged after CPU work). Its recorded probe SHA-256 is `3ca331f01cac4d9533055c46f2a06eb3ff8db204bebbfcecc13b5a8e1e891865`. The implementing subagent did not edit either prepared helper/probe; those remain queued for the parent-owned bounded hardware run.

## Follow-up: first-step zero-state cancellation and complete failure collection

Parent subsequently ran `prepared_native_b1_smoke_v1` successfully. `prepared_native_wide22.json` then completed B1 normal and zero-QKV cases but failed B1 zero-state, eager step0, in the public-native FP32 control: rank1/user0/head6 output PCC **0.9892162833519499**. Both custom controls had passed. The old immediate assertion prevented evaluating the prepared paths, so this artifact does not establish whether either prepared family passes this case. The overall `fail` remains unchanged.

This is a **separate arithmetic-sensitive failure**, not a consequence of persistent BF16 versus FP32 state: both references start from exactly zero and current-step output is formed before the first persistence cast. Recreating this case's exact B1 generator consumption on CPU gives rank1 grouped `q_scaled dot k` values:

```
[-0.004210725189858209, 0.006268990258470655,
 -0.000018695203753874012, -0.004153072749401894]
```

Head6 uses group2. Its dot-product cancellation ratio `abs(sum(q*k))/sum(abs(q*k))` is **0.00033188515**. At zero initial state, the ideal first output is `beta*(q_scaled dot k)*v`; the near-cancelled dot makes native operand/outer-product/readout rounding especially visible. These source/CPU facts strongly motivate testing arithmetic precision, but do not alone identify which native contraction contributes most.

BF16 persistence by itself cannot fix this first-step discrepancy. Direct FP32 unpack for the rank-one state's seed also cannot help this particular zero-state first step, because the seed is zero. A concrete next arithmetic adaptation, if prepared paths share the failure, is a precise SFPU FP32 readout from the live native update, measured as part of the same boundary. If the native outer-product operand precision still fails, adapt that update's precision as well. Keep the failing head and threshold; do not hide this with per-user aggregation or omit zero-state coverage.

Only `tests/probe_prepared_native_recurrent.py` was changed for instrumentation after parent authorization; the helper stayed frozen. `metric` now records every per-user/per-head score and all sub-threshold/undefined-PCC locations. `check_step` collects all five dtype-matched path metrics and all six native-versus-custom pairs (22 output/state comparisons total), saves the complete record, then raises if **any** gate fails. Tensor shape/finiteness assertions are recorded as failed metrics; device/runtime errors are not swallowed. The 0.995 threshold, zero/constant-vector behavior, reference arithmetic, final run failure, and every cross-control gate remain intact.

CPU-only instrumentation validation extracted these two functions with Python AST and supplied host Torch tensors. A deliberately bad public-control head produced three failed comparisons; all five path records and all cross-control records were saved, both prepared path results remained visible, and the final assertion still fired. The clean case still passed. Unequal constant vectors still failed undefined PCC and serialized without NaN. No TTNN import or device operation was used. Parent can rerun the failed case under a fresh artifact name to learn prepared-family outcomes without the rejected public control hiding them.

Failed zero-state artifact SHA-256: `29b92b9c77fce96ecb22b325e1d2e34ec85d31529bc723bb1da79bb97cd5edeb`.

Instrumentation probe SHA-256: `a26715520ad0dbd4696138f5d145f76d8c024e39ee654ac6e7007784e2e73cd1`. Frozen helper SHA-256: `0e5991945bc595b784ece95a7c04db1395aa3abf81cab9f83b5b859b576d96f4`.

## Full CPU evidence

```json
{
  "torch": "2.11.0+cpu",
  "cpu_threads": 4,
  "batch": 10,
  "ranks": 4,
  "seed_formula": "52813 + 97*batch + rank",
  "same_input_checkpoints": {
    "1": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9999999999999999,
            0.9999999999999996,
            1.0000000000000002,
            0.9999999999999997,
            1.0,
            1.0,
            1.0000000000000002,
            1.0000000000000002,
            1.0000000000000004,
            0.9999999999999997
          ],
          "pcc_min": 0.9999999999999996,
          "worst_user": 1,
          "per_head_pcc_min": 0.9999999999999996,
          "worst_user_head": [
            1,
            4
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999986245370691,
            0.9999986180135297,
            0.9999986197104214,
            0.9999986207423119,
            0.9999986233552302,
            0.9999986240554716,
            0.9999986246022389,
            0.9999986198178192,
            0.9999986212284223,
            0.9999986311405955
          ],
          "pcc_min": 0.9999986180135297,
          "worst_user": 1,
          "per_head_pcc_min": 0.999998584053732,
          "worst_user_head": [
            8,
            9
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0034588202986810543,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9999999999999999,
            0.9999999999999997,
            0.9999999999999999,
            1.0,
            1.0,
            1.0,
            1.0000000000000002,
            0.9999999999999998,
            1.0000000000000002,
            1.0
          ],
          "pcc_min": 0.9999999999999997,
          "worst_user": 1,
          "per_head_pcc_min": 0.9999999999999996,
          "worst_user_head": [
            0,
            7
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999986171687214,
            0.9999986247269552,
            0.9999986143139816,
            0.9999986109340265,
            0.999998616468404,
            0.99999862512441,
            0.9999986122091261,
            0.9999986196075676,
            0.9999986098668758,
            0.9999986131637709
          ],
          "pcc_min": 0.9999986098668758,
          "worst_user": 8,
          "per_head_pcc_min": 0.9999985720315026,
          "worst_user_head": [
            8,
            10
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0034151755521453896,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.9999999999999999,
            0.9999999999999999,
            1.0000000000000007,
            1.0000000000000004,
            1.0000000000000002,
            1.0000000000000002,
            1.0000000000000002,
            0.9999999999999999,
            1.0000000000000002,
            0.9999999999999999
          ],
          "pcc_min": 0.9999999999999999,
          "worst_user": 0,
          "per_head_pcc_min": 0.9999999999999993,
          "worst_user_head": [
            8,
            10
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999985912653009,
            0.9999985991260539,
            0.9999986019462636,
            0.999998597194369,
            0.9999985902651719,
            0.9999985903132554,
            0.9999985831773633,
            0.9999985989478719,
            0.999998601324989,
            0.9999985918789425
          ],
          "pcc_min": 0.9999985831773633,
          "worst_user": 6,
          "per_head_pcc_min": 0.999998542398949,
          "worst_user_head": [
            4,
            6
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0037002780455425645,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            1.0000000000000002,
            1.0000000000000002,
            0.9999999999999999,
            0.9999999999999999,
            1.0000000000000004,
            0.9999999999999999,
            1.0000000000000002,
            0.9999999999999997,
            1.0000000000000004,
            1.0000000000000002
          ],
          "pcc_min": 0.9999999999999997,
          "worst_user": 7,
          "per_head_pcc_min": 0.9999999999999996,
          "worst_user_head": [
            6,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999985551472259,
            0.9999985696377854,
            0.9999985703010315,
            0.9999985610887178,
            0.9999985701042521,
            0.9999985658522174,
            0.9999985692004952,
            0.9999985584522905,
            0.9999985558852281,
            0.9999985527174624
          ],
          "pcc_min": 0.9999985527174624,
          "worst_user": 9,
          "per_head_pcc_min": 0.9999985057261976,
          "worst_user_head": [
            6,
            11
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0034224318030530074,
          "per_head_pcc_below_995": 0
        }
      }
    ],
    "4": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9999957489717778,
            0.9999962861128856,
            0.999996626478954,
            0.9999969208846725,
            0.9999962429603477,
            0.9999966771677407,
            0.9999965634435571,
            0.9999966560850082,
            0.9999959663568144,
            0.9999962486657654
          ],
          "pcc_min": 0.9999957489717778,
          "worst_user": 0,
          "per_head_pcc_min": 0.9999938066422176,
          "worst_user_head": [
            6,
            3
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0034878383816788487,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999951144756595,
            0.9999951496058255,
            0.9999951215066453,
            0.9999951272676912,
            0.9999951442291021,
            0.999995123406091,
            0.9999951334186298,
            0.9999951658334972,
            0.9999952058421725,
            0.9999951886737805
          ],
          "pcc_min": 0.9999951144756595,
          "worst_user": 0,
          "per_head_pcc_min": 0.9999947787374668,
          "worst_user_head": [
            4,
            3
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.006395786463469703,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9999971116450438,
            0.999995774002588,
            0.9999961278052618,
            0.9999970904722384,
            0.9999966000417521,
            0.9999980831806268,
            0.9999961644692228,
            0.9999958314996458,
            0.9999962900907108,
            0.9999960218146363
          ],
          "pcc_min": 0.999995774002588,
          "worst_user": 1,
          "per_head_pcc_min": 0.9999935737827276,
          "worst_user_head": [
            3,
            10
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0034615524690093462,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999951206497765,
            0.9999951404860292,
            0.9999951472041639,
            0.9999951834237109,
            0.999995163245695,
            0.999995145537816,
            0.9999951243647965,
            0.9999951762288324,
            0.9999951982698847,
            0.9999951409901108
          ],
          "pcc_min": 0.9999951206497765,
          "worst_user": 0,
          "per_head_pcc_min": 0.9999947786645542,
          "worst_user_head": [
            6,
            2
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.006810979396583583,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.9999958342235018,
            0.9999962287129678,
            0.9999966954545444,
            0.9999965815295532,
            0.9999970201847894,
            0.9999965276873951,
            0.9999963204358234,
            0.9999965575829115,
            0.9999976049366059,
            0.9999965743500699
          ],
          "pcc_min": 0.9999958342235018,
          "worst_user": 0,
          "per_head_pcc_min": 0.9999935113887863,
          "worst_user_head": [
            3,
            7
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0034817354961239853,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999950635672088,
            0.9999951396513022,
            0.9999950850090317,
            0.9999951155012955,
            0.999995129186685,
            0.9999951377786124,
            0.9999950844547231,
            0.9999951142762605,
            0.9999950737799135,
            0.9999951342899018
          ],
          "pcc_min": 0.9999950635672088,
          "worst_user": 0,
          "per_head_pcc_min": 0.9999947596545464,
          "worst_user_head": [
            6,
            3
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.006304192094022838,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            0.9999972929086738,
            0.9999965278174252,
            0.9999966004933968,
            0.9999969317895153,
            0.9999976942148822,
            0.9999981737212394,
            0.9999981422096363,
            0.9999976513176139,
            0.9999981257049592,
            0.9999967812202631
          ],
          "pcc_min": 0.9999965278174252,
          "worst_user": 1,
          "per_head_pcc_min": 0.9999928164965318,
          "worst_user_head": [
            2,
            9
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0032691876339424283,
          "per_head_pcc_below_995": 0
        },
        "state": {
          "pcc_by_user": [
            0.9999950026843255,
            0.9999949843040384,
            0.9999950243555216,
            0.9999949997758635,
            0.9999949598350039,
            0.9999950701646618,
            0.9999951104177163,
            0.9999950076725532,
            0.9999950260561278,
            0.9999950506441501
          ],
          "pcc_min": 0.9999949598350039,
          "worst_user": 4,
          "per_head_pcc_min": 0.999994717138079,
          "worst_user_head": [
            1,
            1
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.007850193582018335,
          "per_head_pcc_below_995": 0
        }
      }
    ],
    "50": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9979957442918006,
            0.99980796627039,
            0.9999244627843631,
            0.9999074160674909,
            0.9997210505961532,
            0.9998984556838758,
            0.9998443179453199,
            0.9998699332034038,
            0.9996733457272917,
            0.9997174702028486
          ],
          "pcc_min": 0.9979957442918006,
          "worst_user": 0,
          "per_head_pcc_min": 0.9813118417252435,
          "worst_user_head": [
            9,
            1
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.07378418390382556,
          "per_head_pcc_below_995": 12
        },
        "state": {
          "pcc_by_user": [
            0.9998092798575348,
            0.9998075113334135,
            0.9998061390065018,
            0.999808875460772,
            0.9998024345908644,
            0.9997985047026243,
            0.9997947659680783,
            0.9998030544181765,
            0.9997931258532698,
            0.9997980215681684
          ],
          "pcc_min": 0.9997931258532698,
          "worst_user": 8,
          "per_head_pcc_min": 0.9997547836048953,
          "worst_user_head": [
            5,
            11
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.042319170020126355,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9999117771539655,
            0.9996235991520486,
            0.9997365583198359,
            0.9999178261167674,
            0.9998096448964156,
            0.9999613646215958,
            0.9996770355800056,
            0.9997262584181295,
            0.9997563579643185,
            0.9996785283601353
          ],
          "pcc_min": 0.9996235991520486,
          "worst_user": 1,
          "per_head_pcc_min": 0.9684378882529022,
          "worst_user_head": [
            3,
            10
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.055232458854847406,
          "per_head_pcc_below_995": 14
        },
        "state": {
          "pcc_by_user": [
            0.9998020669497272,
            0.9998021803361604,
            0.9997929373686825,
            0.9997940102791721,
            0.9997947920749957,
            0.9997944775883828,
            0.9997933141210789,
            0.9997922896842072,
            0.9997911124487685,
            0.9997916751854977
          ],
          "pcc_min": 0.9997911124487685,
          "worst_user": 8,
          "per_head_pcc_min": 0.9997511678808051,
          "worst_user_head": [
            2,
            10
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.04220603113399068,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.9996202673630227,
            0.9997905691539891,
            0.9997981123477473,
            0.9998532404398223,
            0.9999351291150564,
            0.9998708768470821,
            0.999808793052032,
            0.9998164034186339,
            0.9999395415872492,
            0.9998715597774543
          ],
          "pcc_min": 0.9996202673630227,
          "worst_user": 0,
          "per_head_pcc_min": 0.9734414317864754,
          "worst_user_head": [
            9,
            4
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.04711542811012164,
          "per_head_pcc_below_995": 12
        },
        "state": {
          "pcc_by_user": [
            0.9997957623318827,
            0.9997975282513334,
            0.9997934073596327,
            0.999784338313864,
            0.9997927306727789,
            0.9997900689814591,
            0.9997893181597537,
            0.9997862025631179,
            0.9997816888597878,
            0.9997858816929885
          ],
          "pcc_min": 0.9997816888597878,
          "worst_user": 8,
          "per_head_pcc_min": 0.9997438482651057,
          "worst_user_head": [
            8,
            8
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.039188701363243754,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            0.9999254972106839,
            0.9998016060833482,
            0.9998488227881392,
            0.999883246787846,
            0.999935534296771,
            0.9999466803374287,
            0.9999665131863898,
            0.9999202988494486,
            0.9999675998879766,
            0.9999028924722948
          ],
          "pcc_min": 0.9998016060833482,
          "worst_user": 1,
          "per_head_pcc_min": 0.9932401941900323,
          "worst_user_head": [
            2,
            9
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.02817601957788322,
          "per_head_pcc_below_995": 2
        },
        "state": {
          "pcc_by_user": [
            0.9997918037176224,
            0.9997956904735038,
            0.9997955232456273,
            0.9997864899818484,
            0.9997926172281933,
            0.9997850374066661,
            0.9997804277008101,
            0.9997816958470115,
            0.9997768893929302,
            0.9997767681245573
          ],
          "pcc_min": 0.9997767681245573,
          "worst_user": 9,
          "per_head_pcc_min": 0.9997465708483887,
          "worst_user_head": [
            9,
            11
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.041578701873485883,
          "per_head_pcc_below_995": 0
        }
      }
    ],
    "100": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9858159863277464,
            0.9990527885377166,
            0.999576896413325,
            0.9995072763439997,
            0.9985759096252093,
            0.9995449246479376,
            0.999255754596082,
            0.999425401986786,
            0.9987651855367556,
            0.998761243965989
          ],
          "pcc_min": 0.9858159863277464,
          "worst_user": 0,
          "per_head_pcc_min": 0.617999371189537,
          "worst_user_head": [
            9,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.1909388141548603,
          "per_head_pcc_below_995": 39
        },
        "state": {
          "pcc_by_user": [
            0.9990456250401976,
            0.9990618352150062,
            0.999078337321631,
            0.999095951822127,
            0.9991200476350317,
            0.9991339628127741,
            0.9991596841015492,
            0.9991722544636081,
            0.9991884609690967,
            0.9992072387044019
          ],
          "pcc_min": 0.9990456250401976,
          "worst_user": 0,
          "per_head_pcc_min": 0.9988575555309196,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.06083563297054575,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9996005381030518,
            0.998464932335718,
            0.9987566622984069,
            0.9996600669825895,
            0.9991745417853891,
            0.9998366932563547,
            0.9986953645343746,
            0.9989894351918355,
            0.9990882482477336,
            0.9987713048258753
          ],
          "pcc_min": 0.998464932335718,
          "worst_user": 1,
          "per_head_pcc_min": 0.8409964664518425,
          "worst_user_head": [
            3,
            11
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0813230712593906,
          "per_head_pcc_below_995": 31
        },
        "state": {
          "pcc_by_user": [
            0.9991059404336738,
            0.9991316179181037,
            0.999142161707649,
            0.9991676377944165,
            0.9991702163836417,
            0.9991868003609577,
            0.9992092511990414,
            0.9992262729909374,
            0.9992508826021873,
            0.9992761815666856
          ],
          "pcc_min": 0.9991059404336738,
          "worst_user": 0,
          "per_head_pcc_min": 0.9988907188933888,
          "worst_user_head": [
            2,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05533454880923305,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.998328174903218,
            0.9990489188479716,
            0.9991778715827853,
            0.9994687949033977,
            0.9997515051541231,
            0.999484487969743,
            0.9993791006130307,
            0.9994021113040004,
            0.999758150908354,
            0.9996016443733945
          ],
          "pcc_min": 0.998328174903218,
          "worst_user": 0,
          "per_head_pcc_min": 0.5399643187242743,
          "worst_user_head": [
            2,
            4
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.06794039354391816,
          "per_head_pcc_below_995": 23
        },
        "state": {
          "pcc_by_user": [
            0.999151789064478,
            0.9991800126297601,
            0.9991920748202352,
            0.9992123253076773,
            0.9992276510025071,
            0.9992545657325658,
            0.9992809671314529,
            0.9992799410465177,
            0.9992931679341918,
            0.9993137031378564
          ],
          "pcc_min": 0.999151789064478,
          "worst_user": 0,
          "per_head_pcc_min": 0.9989329663401648,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.049965216375050615,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            0.9996957327207701,
            0.9992763740479418,
            0.9994073980701639,
            0.9995588350960352,
            0.9997097353702166,
            0.9998243540552194,
            0.99988551327605,
            0.9997617960736751,
            0.9998965784664555,
            0.9996801818659196
          ],
          "pcc_min": 0.9992763740479418,
          "worst_user": 1,
          "per_head_pcc_min": 0.9291908652813238,
          "worst_user_head": [
            1,
            1
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05539013595286799,
          "per_head_pcc_below_995": 19
        },
        "state": {
          "pcc_by_user": [
            0.999217645460356,
            0.9992365402494208,
            0.9992483573846964,
            0.9992635993918213,
            0.999282455283193,
            0.9992910178505326,
            0.9993226822444771,
            0.9993290596371169,
            0.9993454704173804,
            0.9993661193827161
          ],
          "pcc_min": 0.999217645460356,
          "worst_user": 0,
          "per_head_pcc_min": 0.9989941986207511,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05348328192653112,
          "per_head_pcc_below_995": 0
        }
      }
    ],
    "200": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9817708973660428,
            0.9988026506519,
            0.9994591389075005,
            0.9993741211296797,
            0.9982395089511937,
            0.9994452616955769,
            0.999099333523105,
            0.9993045349348381,
            0.9985836562862153,
            0.998585397625213
          ],
          "pcc_min": 0.9817708973660428,
          "worst_user": 0,
          "per_head_pcc_min": 0.1521281841199681,
          "worst_user_head": [
            0,
            2
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.197817475810258,
          "per_head_pcc_below_995": 41
        },
        "state": {
          "pcc_by_user": [
            0.998761297326771,
            0.998801881933639,
            0.998855257656071,
            0.9988819863710491,
            0.9989307858517498,
            0.9989664116278731,
            0.9990081741050727,
            0.9990280081841079,
            0.999067569168687,
            0.9990965101510565
          ],
          "pcc_min": 0.998761297326771,
          "worst_user": 0,
          "per_head_pcc_min": 0.9981339517737674,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.06636545282734393,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9995060221854681,
            0.9981617728849561,
            0.9984891382725181,
            0.9996065660369311,
            0.999049738709678,
            0.9998096492556809,
            0.9985051311228279,
            0.9988544005375856,
            0.9989787016425041,
            0.9986398330208958
          ],
          "pcc_min": 0.9981617728849561,
          "worst_user": 1,
          "per_head_pcc_min": 0.8282901194937067,
          "worst_user_head": [
            3,
            9
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.08188046174412086,
          "per_head_pcc_below_995": 34
        },
        "state": {
          "pcc_by_user": [
            0.998905752702329,
            0.9989494587574098,
            0.9989782617319866,
            0.9990274948972161,
            0.9990369580232992,
            0.9990651271712228,
            0.9991003680000845,
            0.9991268193912871,
            0.9991661736943689,
            0.9991977455096314
          ],
          "pcc_min": 0.998905752702329,
          "worst_user": 0,
          "per_head_pcc_min": 0.9983701193320963,
          "worst_user_head": [
            1,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05653982201739043,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.9980020899060754,
            0.9988776142614118,
            0.9990355030374826,
            0.9994060801136387,
            0.9997169945498452,
            0.9994202948870592,
            0.999320358733374,
            0.9993401708018339,
            0.9997346924974404,
            0.9995695421242984
          ],
          "pcc_min": 0.9980020899060754,
          "worst_user": 0,
          "per_head_pcc_min": 0.36349634053935587,
          "worst_user_head": [
            2,
            3
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.07155972128046155,
          "per_head_pcc_below_995": 23
        },
        "state": {
          "pcc_by_user": [
            0.9990043014074176,
            0.9990415690910622,
            0.9990715663351538,
            0.9991090984289542,
            0.9991308920095407,
            0.9991693100988709,
            0.9992056335796337,
            0.9992105482206673,
            0.9992308966291195,
            0.9992577771313074
          ],
          "pcc_min": 0.9990043014074176,
          "worst_user": 0,
          "per_head_pcc_min": 0.9985277445391841,
          "worst_user_head": [
            1,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.051838917916178125,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            0.9996509801354009,
            0.9991878329276712,
            0.9993292350771819,
            0.9995055801577134,
            0.999672415442024,
            0.9998093088505813,
            0.9998745734517799,
            0.99974353380734,
            0.9998898633613115,
            0.9996586385306824
          ],
          "pcc_min": 0.9991878329276712,
          "worst_user": 1,
          "per_head_pcc_min": 0.911443523751632,
          "worst_user_head": [
            1,
            1
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0575815406492055,
          "per_head_pcc_below_995": 21
        },
        "state": {
          "pcc_by_user": [
            0.9991132299661055,
            0.9991383809060888,
            0.9991605424911468,
            0.9991859940361817,
            0.9992104716316291,
            0.9992286510271398,
            0.9992708085970033,
            0.9992788826456765,
            0.9993033248107684,
            0.9993280695353957
          ],
          "pcc_min": 0.9991132299661055,
          "worst_user": 0,
          "per_head_pcc_min": 0.9986753022367637,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05505121093132528,
          "per_head_pcc_below_995": 0
        }
      }
    ],
    "512": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9817125822066459,
            0.9987995191534944,
            0.999457741351336,
            0.9993727403421078,
            0.9982364581581079,
            0.999444453310891,
            0.9990980658632586,
            0.999303705028643,
            0.9985825693523117,
            0.9985844904990288
          ],
          "pcc_min": 0.9817125822066459,
          "worst_user": 0,
          "per_head_pcc_min": 0.03437820655575595,
          "worst_user_head": [
            0,
            2
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.19783505084729536,
          "per_head_pcc_below_995": 41
        },
        "state": {
          "pcc_by_user": [
            0.9987570693075861,
            0.998798384142873,
            0.9988527339692455,
            0.9988797536217703,
            0.998929033504991,
            0.9989650521708936,
            0.9990070505608072,
            0.9990270568765884,
            0.9990668832953281,
            0.9990959437851893
          ],
          "pcc_min": 0.9987570693075861,
          "worst_user": 0,
          "per_head_pcc_min": 0.9981144553924677,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.06646723648203147,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9995050183664943,
            0.9981588875394991,
            0.9984868872817086,
            0.999606225193034,
            0.9990489637436576,
            0.9998094947517165,
            0.9985041650701069,
            0.998853789557269,
            0.9989782391282712,
            0.9986393882167605
          ],
          "pcc_min": 0.9981588875394991,
          "worst_user": 1,
          "per_head_pcc_min": 0.8282662432410146,
          "worst_user_head": [
            3,
            9
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.08187843335770666,
          "per_head_pcc_below_995": 34
        },
        "state": {
          "pcc_by_user": [
            0.9989037162061469,
            0.9989477973094393,
            0.9989769116134566,
            0.9990265087104407,
            0.9990361131376326,
            0.9990644320247146,
            0.9990998104045412,
            0.9991263624786297,
            0.9991658368071212,
            0.9991974642717054
          ],
          "pcc_min": 0.9989037162061469,
          "worst_user": 0,
          "per_head_pcc_min": 0.9983616523200477,
          "worst_user_head": [
            1,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05655705302165457,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.9979997059241921,
            0.9988764788422897,
            0.9990346419256704,
            0.9994057781748231,
            0.9997168267584745,
            0.9994200382141118,
            0.9993201559548593,
            0.999339969055881,
            0.9997346292265443,
            0.9995694660629182
          ],
          "pcc_min": 0.9979997059241921,
          "worst_user": 0,
          "per_head_pcc_min": 0.3624455660162789,
          "worst_user_head": [
            2,
            3
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.07158706816753314,
          "per_head_pcc_below_995": 23
        },
        "state": {
          "pcc_by_user": [
            0.9990032601269471,
            0.9990406646214625,
            0.9990708737068041,
            0.9991085790458507,
            0.9991304570002405,
            0.999168969661179,
            0.9992053662153643,
            0.9992103309607328,
            0.9992307236357922,
            0.9992576421492735
          ],
          "pcc_min": 0.9990032601269471,
          "worst_user": 0,
          "per_head_pcc_min": 0.9985228392381362,
          "worst_user_head": [
            1,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0518577961689704,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            0.9996507533189591,
            0.999187440157148,
            0.9993289204164555,
            0.9995053841722537,
            0.9996722812300072,
            0.9998092664905818,
            0.9998745455503273,
            0.9997434959042683,
            0.999889853046118,
            0.9996586002098443
          ],
          "pcc_min": 0.999187440157148,
          "worst_user": 1,
          "per_head_pcc_min": 0.9113446507623759,
          "worst_user_head": [
            1,
            1
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05758827123349925,
          "per_head_pcc_below_995": 21
        },
        "state": {
          "pcc_by_user": [
            0.9991127136506287,
            0.9991379288713534,
            0.9991601916255773,
            0.9991857178510157,
            0.9992102433222447,
            0.9992284772029908,
            0.99927068573421,
            0.9992787736415748,
            0.9993032481788974,
            0.9993280075294975
          ],
          "pcc_min": 0.9991127136506287,
          "worst_user": 0,
          "per_head_pcc_min": 0.9986723744576221,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.055057566867085156,
          "per_head_pcc_below_995": 0
        }
      }
    ],
    "702": [
      {
        "rank": 0,
        "output": {
          "pcc_by_user": [
            0.9817125822065067,
            0.9987995191534952,
            0.9994577413513343,
            0.9993727403421082,
            0.9982364581581079,
            0.999444453310891,
            0.9990980658632586,
            0.999303705028643,
            0.9985825693523117,
            0.9985844904990288
          ],
          "pcc_min": 0.9817125822065067,
          "worst_user": 0,
          "per_head_pcc_min": 0.034378206568465255,
          "worst_user_head": [
            0,
            2
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.19783505084729536,
          "per_head_pcc_below_995": 41
        },
        "state": {
          "pcc_by_user": [
            0.9987570693075857,
            0.9987983841428728,
            0.9988527339692455,
            0.9988797536217707,
            0.998929033504991,
            0.9989650521708936,
            0.9990070505608072,
            0.9990270568765884,
            0.9990668832953281,
            0.9990959437851893
          ],
          "pcc_min": 0.9987570693075857,
          "worst_user": 0,
          "per_head_pcc_min": 0.9981144553924665,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.06646723648203147,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 1,
        "output": {
          "pcc_by_user": [
            0.9995050183664014,
            0.9981588875394999,
            0.998486887281708,
            0.999606225193034,
            0.9990489637436576,
            0.9998094947517165,
            0.9985041650701069,
            0.998853789557269,
            0.9989782391282712,
            0.9986393882167605
          ],
          "pcc_min": 0.9981588875394999,
          "worst_user": 1,
          "per_head_pcc_min": 0.8282662432410146,
          "worst_user_head": [
            3,
            9
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.08187843335770666,
          "per_head_pcc_below_995": 34
        },
        "state": {
          "pcc_by_user": [
            0.9989037162061472,
            0.9989477973094395,
            0.9989769116134564,
            0.9990265087104407,
            0.9990361131376326,
            0.9990644320247146,
            0.9990998104045412,
            0.9991263624786297,
            0.9991658368071212,
            0.9991974642717054
          ],
          "pcc_min": 0.9989037162061472,
          "worst_user": 0,
          "per_head_pcc_min": 0.9983616523200483,
          "worst_user_head": [
            1,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05655705302165457,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 2,
        "output": {
          "pcc_by_user": [
            0.9979997059241921,
            0.9988764788422897,
            0.9990346419256704,
            0.9994057781748231,
            0.9997168267584745,
            0.9994200382141118,
            0.9993201559548593,
            0.999339969055881,
            0.9997346292265443,
            0.9995694660629182
          ],
          "pcc_min": 0.9979997059241921,
          "worst_user": 0,
          "per_head_pcc_min": 0.3624455660162789,
          "worst_user_head": [
            2,
            3
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.07158706816753314,
          "per_head_pcc_below_995": 23
        },
        "state": {
          "pcc_by_user": [
            0.9990032601269471,
            0.9990406646214625,
            0.9990708737068041,
            0.9991085790458505,
            0.9991304570002405,
            0.999168969661179,
            0.9992053662153643,
            0.9992103309607328,
            0.9992307236357922,
            0.9992576421492735
          ],
          "pcc_min": 0.9990032601269471,
          "worst_user": 0,
          "per_head_pcc_min": 0.9985228392381362,
          "worst_user_head": [
            1,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.0518577961689704,
          "per_head_pcc_below_995": 0
        }
      },
      {
        "rank": 3,
        "output": {
          "pcc_by_user": [
            0.9996507533189591,
            0.999187440157148,
            0.9993289204164555,
            0.9995053841722537,
            0.9996722812300072,
            0.9998092664905818,
            0.9998745455503273,
            0.9997434959042683,
            0.999889853046118,
            0.9996586002098443
          ],
          "pcc_min": 0.999187440157148,
          "worst_user": 1,
          "per_head_pcc_min": 0.9113446507623759,
          "worst_user_head": [
            1,
            1
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.05758827123349925,
          "per_head_pcc_below_995": 21
        },
        "state": {
          "pcc_by_user": [
            0.9991127136506287,
            0.9991379288713534,
            0.9991601916255773,
            0.9991857178510157,
            0.9992102433222447,
            0.9992284772029908,
            0.99927068573421,
            0.9992787736415748,
            0.9993032481788974,
            0.9993280075294975
          ],
          "pcc_min": 0.9991127136506287,
          "worst_user": 0,
          "per_head_pcc_min": 0.9986723744576221,
          "worst_user_head": [
            0,
            0
          ],
          "maximum_absolute_relative_error_vs_fp32": 0.055057566867085156,
          "per_head_pcc_below_995": 0
        }
      }
    ]
  },
  "changed50": [
    {
      "rank": 0,
      "output": {
        "pcc_by_user": [
          0.9999827445756966,
          0.9999693873265608,
          0.9999828415901033,
          0.9999213868560342,
          0.9999768923665415,
          0.9999855722749945,
          0.9999778853812152,
          0.9999873058307014,
          0.9999834167873448,
          0.9999847594919008
        ],
        "pcc_min": 0.9999213868560342,
        "worst_user": 3,
        "per_head_pcc_min": 0.9989561174107887,
        "worst_user_head": [
          9,
          2
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.015265737741443432,
        "per_head_pcc_below_995": 0
      },
      "state": {
        "pcc_by_user": [
          0.999971868051097,
          0.9999717788038345,
          0.9999721567750586,
          0.9999716876732149,
          0.9999720160464933,
          0.9999712108161095,
          0.9999716468822912,
          0.9999714538685747,
          0.9999716074020967,
          0.9999713902141466
        ],
        "pcc_min": 0.9999712108161095,
        "worst_user": 5,
        "per_head_pcc_min": 0.9999675031840183,
        "worst_user_head": [
          8,
          10
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.024581392543140065,
        "per_head_pcc_below_995": 0
      }
    },
    {
      "rank": 1,
      "output": {
        "pcc_by_user": [
          0.9999829920670923,
          0.9999781854823071,
          0.9999865654320661,
          0.9999860680764425,
          0.9999814716450678,
          0.9999853025204124,
          0.9999892275899419,
          0.9999897026512554,
          0.9999931426224179,
          0.9999910228033091
        ],
        "pcc_min": 0.9999781854823071,
        "worst_user": 1,
        "per_head_pcc_min": 0.9996211877881884,
        "worst_user_head": [
          2,
          2
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.013092210901687247,
        "per_head_pcc_below_995": 0
      },
      "state": {
        "pcc_by_user": [
          0.9999719882104361,
          0.9999708560974988,
          0.9999712452246532,
          0.9999712656035351,
          0.9999713320810631,
          0.999971160399134,
          0.9999709241091794,
          0.9999708266714679,
          0.999971024579847,
          0.9999707072481753
        ],
        "pcc_min": 0.9999707072481753,
        "worst_user": 9,
        "per_head_pcc_min": 0.9999664936371121,
        "worst_user_head": [
          1,
          9
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.025987325754742597,
        "per_head_pcc_below_995": 0
      }
    },
    {
      "rank": 2,
      "output": {
        "pcc_by_user": [
          0.9999831671104399,
          0.9999854479877199,
          0.9999907713873563,
          0.999982400815542,
          0.9999917006558543,
          0.999992282371899,
          0.9999911988467168,
          0.9999909699533512,
          0.9999956549695513,
          0.9999906821772323
        ],
        "pcc_min": 0.999982400815542,
        "worst_user": 3,
        "per_head_pcc_min": 0.9999607075531893,
        "worst_user_head": [
          0,
          1
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.010051491190942043,
        "per_head_pcc_below_995": 0
      },
      "state": {
        "pcc_by_user": [
          0.9999711422026558,
          0.9999708607989616,
          0.9999713386505636,
          0.9999704720603939,
          0.9999712993407011,
          0.9999700044375207,
          0.9999711964156197,
          0.999969986672523,
          0.9999702165965427,
          0.999969881750387
        ],
        "pcc_min": 0.999969881750387,
        "worst_user": 9,
        "per_head_pcc_min": 0.99996499176468,
        "worst_user_head": [
          5,
          11
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.028262472417156806,
        "per_head_pcc_below_995": 0
      }
    },
    {
      "rank": 3,
      "output": {
        "pcc_by_user": [
          0.9999919293211141,
          0.9999904344075624,
          0.9999889740805019,
          0.9999923343096729,
          0.9999926109649206,
          0.9999947870400868,
          0.9999951973675897,
          0.9999963754890989,
          0.9999896581159782,
          0.9999953440342888
        ],
        "pcc_min": 0.9999889740805019,
        "worst_user": 2,
        "per_head_pcc_min": 0.9999351599676811,
        "worst_user_head": [
          1,
          4
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.0065334135235070485,
        "per_head_pcc_below_995": 0
      },
      "state": {
        "pcc_by_user": [
          0.999970187453686,
          0.9999704898769749,
          0.9999705074211909,
          0.9999700348495594,
          0.9999705986103914,
          0.999970319486332,
          0.9999695349291423,
          0.999968386300486,
          0.9999684304339809,
          0.9999685718644026
        ],
        "pcc_min": 0.999968386300486,
        "worst_user": 7,
        "per_head_pcc_min": 0.9999637414002092,
        "worst_user_head": [
          9,
          11
        ],
        "maximum_absolute_relative_error_vs_fp32": 0.02166364922658766,
        "per_head_pcc_below_995": 0
      }
    }
  ],
  "input_sha256": {
    "initial": [
      "5332a1d56308f644cd34aae3fd98bbedf558bc325e908d38594c620b15e95dbb",
      "0e7f82b1c8ea1b04e892ec149fcec2b4024d117de031cd28625a34abefb88c94",
      "c32d89fe60f43c5ea8197360d80541dd737854d01ff885af582ec68050d565f0",
      "7013d8b8793cc4420fc73420bb6be35e6759a73cd6772f74e62e47799d2f9573"
    ],
    "bank0_packed": [
      "aa7064b251281c116e339d51576338df7449e561ea598f76806c3f7cadeb674f",
      "f6f50ae4acbd848ff40acc0729c2dfdcb68ee317b2ab57d5b49a919cb18ad18e",
      "fd70a9a7acd5ac20c155bb9e750255253243e341e6dd7f29eedc4de789e84804",
      "3f90f022d5ccecf6385542a1379beed084933674ec93683204163e43a3b3c45b"
    ]
  },
  "source_artifact_sha256": "5f0bb9fe37d986e7fa911d7258b7d44d4d74ac35c4ed736d85ac38a91e94bd4e",
  "same_input_q_dot_k": [
    {
      "rank": 0,
      "minimum_abs": 9.473201549316593e-06,
      "rank_user0_heads": [
        -9.473201549316593e-06,
        -9.473201549316593e-06,
        -9.473201549316593e-06,
        0.00046153743275553294,
        0.00046153743275553294,
        0.00046153743275553294,
        -0.0017062941840244195,
        -0.0017062941840244195,
        -0.0017062941840244195,
        0.003090397746839845,
        0.003090397746839845,
        0.003090397746839845
      ]
    },
    {
      "rank": 1,
      "minimum_abs": 0.0003958100585158492,
      "rank_user0_heads": [
        0.00830480270522283,
        0.00830480270522283,
        0.00830480270522283,
        0.016922936094480433,
        0.016922936094480433,
        0.016922936094480433,
        0.005488932778501636,
        0.005488932778501636,
        0.005488932778501636,
        0.009298583050935609,
        0.009298583050935609,
        0.009298583050935609
      ]
    },
    {
      "rank": 2,
      "minimum_abs": 0.0001262707275664389,
      "rank_user0_heads": [
        0.0045335749895251964,
        0.0045335749895251964,
        0.0045335749895251964,
        0.0005589870128494315,
        0.0005589870128494315,
        0.0005589870128494315,
        -0.003708244968695483,
        -0.003708244968695483,
        -0.003708244968695483,
        0.008063439579256235,
        0.008063439579256235,
        0.008063439579256235
      ]
    },
    {
      "rank": 3,
      "minimum_abs": 0.001016945590259305,
      "rank_user0_heads": [
        0.018770989735892894,
        0.018770989735892894,
        0.018770989735892894,
        -0.0017531584560655507,
        -0.0017531584560655507,
        -0.0017531584560655507,
        0.012296623594231934,
        0.012296623594231934,
        0.012296623594231934,
        0.007094476083604544,
        0.007094476083604544,
        0.007094476083604544
      ]
    }
  ],
  "ttnn_imported": false,
  "cpu_seconds": 58.455672482028604,
  "summary": [
    {
      "trajectory": 1,
      "output_min": 0.9999999999999996,
      "state_min": 0.9999985527174624,
      "head_output_min": 0.9999999999999993
    },
    {
      "trajectory": 4,
      "output_min": 0.9999957489717778,
      "state_min": 0.9999949598350039,
      "head_output_min": 0.9999928164965318
    },
    {
      "trajectory": 50,
      "output_min": 0.9979957442918006,
      "state_min": 0.9997767681245573,
      "head_output_min": 0.9684378882529022
    },
    {
      "trajectory": 100,
      "output_min": 0.9858159863277464,
      "state_min": 0.9990456250401976,
      "head_output_min": 0.5399643187242743
    },
    {
      "trajectory": 200,
      "output_min": 0.9817708973660428,
      "state_min": 0.998761297326771,
      "head_output_min": 0.1521281841199681
    },
    {
      "trajectory": 512,
      "output_min": 0.9817125822066459,
      "state_min": 0.9987570693075861,
      "head_output_min": 0.03437820655575595
    },
    {
      "trajectory": 702,
      "output_min": 0.9817125822065067,
      "state_min": 0.9987570693075857,
      "head_output_min": 0.034378206568465255
    },
    {
      "trajectory": "50 changed",
      "output_min": 0.9999213868560342,
      "state_min": 0.999968386300486,
      "head_output_min": 0.9989561174107887
    }
  ]
}
```
