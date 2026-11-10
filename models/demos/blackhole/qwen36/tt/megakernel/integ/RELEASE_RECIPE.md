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

## Ship config
- Megakernel: `QWEN36_R16_MEGAKERNEL=1` (real engine, kind=mk) — off (unset) = r13 graph.
- Multi-image: `QWEN36_MM_MAX_IMAGES=4`, vllm `limit-mm-per-prompt {"image": 4, "video": 1}` (ConfigMap
  qwen38-dev-mm4-spec), router `VLLM_ROUTER_MAX_IMAGES=4`.

## Gates (dev hosts only)
1. Flag-off md5 identity vs prod r13 (stock in-pod r13 tree vs RC tree, prod runtime-spec env, greedy + sampled ident).
2. Flag-on (not md5-identical by design: split-K order, deferred norm): acceptance, GPQA accuracy + paired length
   (±5% of r13), needle / long context, OR-style replay (primary), 13 router standard tests.
3. Multi-image co-batched decode: 3 clients × 4×1024px images, flag on vs off (dev baseline 234 → 126 tok/s).
