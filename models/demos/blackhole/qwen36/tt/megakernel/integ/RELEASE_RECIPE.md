# Megakernel release recipe (integration step 2 of RELEASE_PLAN.md)

## One tt-metal SHA, no overlay
`integ/make_rc.sh <tt-metal checkout>` builds `anatarajan/qwen38-r16-mk-rc` locally:

| piece | ref | what it carries |
|---|---|---|
| base | prod r13 tt-metal `8eb9af536d8` | prod (ancestor of everything below) |
| multi-image | `c8c90fe4acb` | QWEN36_MM_MAX_IMAGES (already an ancestor of integ2) |
| vision cache | `8c2faf2ec95` (anatarajan/qwen38-r16-mm-vcache) | QWEN36_MM_VISION_CACHE (default off) |
| engine | `anatarajan/qwen38-r16-mk-int` | M3 engine (v12 MLP, AR, GDN, attention land here by token) |
| integration | `anatarajan/qwen38-r16-mk-integ2` | QWEN36_R16_MEGAKERNEL hook, engine/build.py, Stage1Verify, weights_mk.repack |

Everything vs r13 is python under `models/demos/blackhole/qwen36` (the script warns otherwise), so the in-pod r13
ttnn build matches the RC's C++.

## Image (main triggers the in-cluster build; tt-shield r13 recipe, see .claude/tmp/r16/mm/RELEASE_PLAN.md)
- tt-metal: the RC SHA above (must be fetchable by the build pod: pushing the RC branch needs the user's OK)
- tt-inference-server: `1177235e` (scripts/build_single_docker.sh --build)
- vllm-tt-plugin: `66980cb`
- Tag `0.25.0-<rc sha>-66980cb-mk…`, deployed by `@sha256` digest; chart/values change committed in tt-orchestration.

## RC runtime env (spec v1.10) = prod r13 runtime-model-spec env + the lines below
Base: the prod `qwen38-tt-inference-server-runtime-model-spec` env_vars (64 vars; integ's prod_env.sh matches it exactly,
the step harness adds only `QWEN36_FABRIC_RING=1`). KV stays BFP8 (no `QWEN36_R16_KV_BFP4`). Added on top:

| var | value | where it acts | in RC tree? |
|---|---|---|---|
| `QWEN36_R16_MEGAKERNEL` | `1` (Stage1Verify, engine kind mk, no probe) | traced verify layer loop | yes (integ2) |
| `QWEN36_R16_MK_BUCKETS` / `QWEN36_R16_MK_PAD_MIN` | `8` / `1` (also the code defaults) | engine bucket; every live width padded onto it | yes (-mk-on) |
| `QWEN36_MM_MAX_IMAGES` | `4` | multi-image | yes (c8c90fe) |
| `QWEN36_MTP_LAYER_BFP4` | `1` | drafter (always active) | yes (r13) |
| `QWEN36_DRAFT_VOCAB_TOPK` | `131072` | drafter (always active) | yes (r13) |
| `QWEN36_MTP_SDPA_WINDOW` | `8192` | drafter (always active) | yes (r13) |
| `QWEN36_BFP4_LAYOUT_GATEUP` | `dsmc:3:2:4` | op path (prefill, kv-split variants, out-of-scope buckets, engine-off fallback) | yes (r13 tp_common `{BFP4_LAYOUT_ENV}_{TARGET}` override; needs `QWEN36_BFP4_LAYOUT=1`, already prod) |
| ~~`QWEN36_R16_VERIFY_SDPA_MULTI_CORES`~~ | ~~`16`~~ | **DROPPED from the RC env (main, 11-11)**: +1.08 ms verify at 128K / +0.19 at 64K on 3 users vs prod; helps only at 8x32K | (code stays, default unset) |
| `QWEN36_GLUE` | `kvwo,guin,kvt` | op path; md5-identical | yes (integ2). The GSNK gate ran WITH it (spec_GSNK.json), so keep it in the RC env |

The GSNK gate (length ratio 1.027, CI 0.958-1.096; GPQA 87.1 vs 86.4; needle 9/9) ran these on d26u08's in-pod tree,
not on this RC SHA. **The binding check is the RC's own flag-on length gate: RC SHA + this env (engine on) vs prod r13,
≥ 2 replicates.**

## Ship config
- Megakernel: see the RC runtime env above — off (unset) = r13 graph (gate 1: md5 identity PASSED, see below).
- Multi-image: `QWEN36_MM_MAX_IMAGES=4`, vllm `limit-mm-per-prompt {"image": 4, "video": 1}` (ConfigMap
  qwen38-dev-mm4-spec), router `VLLM_ROUTER_MAX_IMAGES=4`.

## Gates (dev hosts only)
1. Flag-off md5 identity vs prod r13 (stock in-pod r13 tree vs RC tree, prod runtime-spec env, greedy + sampled ident).
   PASSED 2026-11-10 on d26u02, RC `bb5f961b63e`: greedy c522fcae4e83cd38 (2341 accepted, 2.7126/row-step) and sampled
   a85b8ed4407707ae (2358, 2.7938) identical to r13 `8eb9af536d8`. Re-run on the final RC SHA.
2. Flag-on (not md5-identical by design: split-K order, deferred norm): acceptance, GPQA accuracy + paired length
   (±5% of r13), needle / long context, OR-style replay (primary), 13 router standard tests.
3. Multi-image co-batched decode: 3 clients × 4×1024px images, flag on vs off (dev baseline 234 → 126 tok/s).

## RC engine scope (spec v1.73, user decisions 11-10)
- Engine built and traced ONLY at the 8-user verify bucket (QWEN36_R16_MK_BUCKETS default "8"), R = 32 (build_program
  asserts R == 32 under MK_REAL_AR: real-AR residual blow-up at R < 32).
- Every live width 1..8 is padded onto it (QWEN36_R16_MK_PAD_MIN default 1: prod buckets "1,2,4,8" -> "8"; idle rows
  kvpos -1 / GDN ctrl HOLD). Fallback if the rowind gate fails at k = 1 but passes k = 2..7: QWEN36_R16_MK_PAD_MIN=2
  (bucket 1 op-by-op). No KV-split verify variant at engine widths.
- Gates: flag-off md5 = r13; rowind (tests/test_mk_row_independence.py, k = 1, 2, 5, 7); length / accuracy / OR replay
  with the engine on.

## Engine config (in code: engine/program.py ENGINE_ENV; the vLLM runtime spec sets only QWEN36_R16_* knobs)
Applied by build_program around setup_bench AND the lazy per-device builds (DM opt levels are read there). Override per key
with QWEN36_R16_MK_ENV="K=V ..." or QWEN36_R16_MK_ENV_FILE (debug only; the RC uses the code defaults).

| kind | value |
|---|---|
| MK_DEFINES | `MKA_GUPRE=1,MK_DNPIPE=1,MK_FLAG_FLUSH=1,MKA_MIXTRID=1,MKA_RQKV=1,MKA_PQGDN=1,MKA_PQDELAY=27000,MKA_RQNR=1,MKA_RMASK=1,MKA_QSK=1,MK_V12=1,MK_REAL_AR=1,MK_OWNER_RESID=0,MKA_RWO=1,MK_REAL_AAR=1,MK_REAL_GAR=1,MKA_RQPOS=1,MKA_RQBAL=1,MKA_RQBAL_HOST=1,MK_V12_NOCBAL=60,MK_V12_NOCBAL_ATT=1,MK_GDN_INPROJ=1,MK_GIN_PERCORE=1,MK_GAMMA_CONS=1` |
| env `QWEN36_R16_MK_ENV` | `MK_DEFINES=...` |
| env `MK_SGDN_US` | `64` |
| env `MK_SAR_US` | `18` |
| env `MKA_SWO_US` | `10` |
| env `MKA_CTX` | `262144` |
| env `MKA_KVDT` | `bfp8` |
| env `MKA_NBB` | `1` |
| env `MKA_BUDGET` | `16384` |
| env `MKA_EXT` | `1` |
| env `MK_MIXN` | `1` |
| env `MKA_GDN24` | `1` |
| env `MKA_GDN_WAIT` | `1` |
| env `MKA_GDNOUT` | `1` |
| env `MKA_UPAR` | `1` |
| env `MKA_KVA` | `1` |
| env `E1_HOLD_SKIP` | `1` |
| env `MKA_GDN_CHECK` | `0` |
| env `E1_GO_DBG` | `0` |
| env `MKA_KVPROD` | `1` |
| env `MKA_NKVH` | `1` |
| env `MKA_KVH` | `0` |
| env `MKA_BS` | `64` |
| env `MK_STREAM_DM_OS` | `1` |
| env `MK_AR_RSTD` | `1` |
| env `MK_AR_RSTD_RED` | `gdn` |
| set per build | `FM_NL` (pass layers), `MKA_NU` (4), `V12_R` (32), `MK_AMASK` (from the layer types), `MKA_ATT0=0`, `MKA_KVA_NU` (n_users), KV injection: `MKA_KVPROD=1 MKA_NKVH MKA_KVH=0 MKA_BS MKA_KVDT`, `MKA_REPLAY_L1 / MKA_ATTN_L1 / MKA_QKG_ADDR`, `MKA_AUX_* / MKA_ROPE_* / MKA_PT_*` |

vLLM-side knobs (code defaults = RC): `QWEN36_R16_MEGAKERNEL=1`, `QWEN36_R16_MK_BUCKETS=8`, `QWEN36_R16_MK_PAD_MIN=1`,
`QWEN36_R16_MK_KV_INJECT=1`, `QWEN36_R16_MK_KVA=1`, `QWEN36_R16_MK_W_ATTN=1`, `QWEN36_R16_MK_L1POOL=1`, `QWEN36_R16_MK_SEMPOOL` (0 until
verified), `QWEN36_R16_MK_MAX_CTX` (default the engine's MKA_CTX; contexts above it replay the op-by-op split fallback).
