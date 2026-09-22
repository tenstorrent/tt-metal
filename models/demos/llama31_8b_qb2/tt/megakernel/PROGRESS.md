# Resumable experiment checkpoint

Updated 2026-09-22 13:27 UTC. **Verified partial MLP prototype; full megakernel incomplete.**

- Base: `b8915544692d8f9feb2c890afbc2f22791560cd2` (origin/main at setup).
- Earlier implementation HEAD: `4214810a674a018b256f4dd0f776974ba919bec1`; subsequent
  checkpoint commits include this note. Exact transfer HEAD is recorded in
  the run artifact `PARENT_CHECKPOINT.md` and obtainable with `git rev-parse HEAD`.
- Branch: `codex/llama31-qb2-megakernel`. Mark authorized commits/pushes to this
  branch only on `tenstorrent/tt-metal`; no PR or other external posts. Remote
  GitHub authentication is unavailable. Parent transfers the saved bundle and
  pushes using its existing access; do not start OAuth or copy credentials.
- Owned node/job: `qb2-120-p03t06` / `113796`; expires 2026-09-22 19:22:10 UTC.
  Parent supervises hourly. Fresh checkpoint due by 17:22:10 UTC and before
  stopping. Stop this run's workloads before expiry; preserve the allocation.
- Run root: `/home/moconnor/llama-megakernel-113796`; artifacts in its `artifacts`
  directory and backed up to `/data/moconnor/llama-megakernel-113796`.

## Environment and reproducible commands

Current runtime/extension built in `build-current` from the base SHA using
installed Clang 20.1.8, CMake 4.0.2, Python 3.12.3 and pinned SFPI 7.80.0[956].
Isolated `../venv-current`: Torch 2.11.0 CPU, Transformers 5.12.1,
torchvision 0.26.0 CPU. Do not use the unrelated older Gemma runtime for model
measurements. System toolchain was preserved. Pinned checkpoint revision:
`0e9e39f249a16976918f6564b8830bc894c89659` (local snapshot; `LLAMA_MODEL_PATH`).

From the checkout, reuse the existing environment/build:

```bash
source ../artifacts/run-env.sh
export CCACHE_DIR=/home/moconnor/llama-megakernel-113796/cache/ccache
cmake --build build-current --target ttnn test_system_health tracy_profiler_cli_tools --parallel 16
```

Original configure: `build_metal.sh --build-dir build-current --enable-ccache
--cpm-source-cache ../cache/cpm --configure-only --build-metal-tests`.
Exact configure/install commands are saved in `../artifacts/commands`;
installation includes `tar`, `tt_pybinds`, `umd-runtime`, direct tt_stl runtime
install, and editable Python package. Source-only changes need no host rebuild;
changed device C++ must compile and execute on hardware before qualification.

## Checks, results, and blocker

- Current host runtime/extension/health/profiler build passed (1280 actions).
- SwiGLU and MLP Blackhole RISC compile/link passed with explicit mock UMD.
  A 32-row address table reuses one cached program for indices 31,1,16,0 with
  cache misses forbidden (cache entries 2 → 2). This is not numerical evidence.
- Compiler telemetry: SwiGLU code/config 6,144 bytes; MLP 16,016 bytes, including
  14,768 bytes of kernel text. See `compile-mock-reuse.log` and `footprint.json`.
- Python syntax/help pass; six real-model pytest cases collect, including
  focused MLP intermediate outputs and distinct layer-0/layer-31 weight rows.
  Hardware assertions now pass as detailed below.
- A CPU-only HF BF16 real-checkpoint reference completed at context 128 with
  32 predictions. Logits [32,128256] and every layer's K/V [1,8,159,128] are
  finite; cache lengths grow 128→159. Artifacts: `hf-reference-128/reference.pt`
  and `.json`; preparation took 139.33 s. This is not TT correctness/performance.
  Benchmark `--hf-reference` enforces the same teacher stream and adds HF
  logit/cache comparisons while retaining the selected TT baseline precision.
- Earlier missing chip 2↔3 link and isolated ERISC initialization failures are
  preserved in artifacts. Mark explicitly authorized repeated device resets
  during the owned allocation; this supersedes earlier approval restrictions.
- Bounded serialized reset at 13:11 UTC restored all four devices and both
  internal links. Current-runtime full-connectivity tests pass, and the
  four-chip FABRIC_1D_RING mesh opens and closes successfully. Evidence:
  `reset-once.log`, `system-health-after-reset.log`, `mesh-after-reset.log`.
- Original real-weight layer HF PCC 0.997647 passes. Fused MLP stages are
  bitwise identical for real layers 0/31, two token embeddings and five replays.
- SwiGLU, MLP, and reduced shared-scratch MLP complete-layer tests pass 12
  position/remapping checks each, including 127→128 and 255→256 boundaries,
  exact full-cache equality and repeated trace replay. The larger shared
  allocation collided with prefill RMSNorm L1; one-block shared weights fix it.
- All-32-layer B1/context128/32-output-token default MLP: teacher logits and all
  64 K/V tensors bitwise equal to matched traced baseline; greedy agreement100%.
  Selected precision and sampling unchanged. Three warmed host-generation runs:
  baseline median8.7855 ms/token; MLP9.2462 ms/token, a5.24% regression.
- BF16 HF diagnostic: baseline and prototype logit PCC0.977533, relativeL2
  0.208836, teacher top1 agreement100%; 0.99 HF target is explicitly false.
  Do not conflate matched TT correctness with BF16 HF accuracy qualification.
- Device profiler/memory capture is starting. No measured device latency,
  DRAM/synchronization or serving result yet. No speedup claim.

## Implemented scope and next actions

One local gate/up→SwiGLU→down program, BF16 intermediate rounding and original
BFP4/BFP8 weights, reusable scratch/address table; explicit all-32-layer
integration and real-weight layer/model comparison harnesses. This is **MLP
fusion only**: prefill, normalization, attention/KV, collectives and terminal
embedding/head/sampling remain native TTNN; no complete decoder/device layer
loop or vLLM qualification. B1 only. Current Blackhole supports 64 CB indices.

1. Hardware recovered; context-128 CPU reference is prepared. Compiler success
   does not qualify the body.
2. Initial bounded hardware layer/model checks completed. Preserve evidence
   in `numerical`, `numerical-shared-single`, `model-baseline-128-complete`,
   and `model-mlp-128` artifacts. Collect matched separate device profiles next.
3. Fix failures and extend context to 2048. Check page writes, remapping within
   captured traces, repeated replay, position growth and final token/logit/KV.
4. Collect separate matched device-profiler runs, phase/synchronization and
   DRAM evidence, concise performance reports; keep serving separate.
5. Extend verified code to normalization/fabric collectives and attention/KV,
   then a complete one-token device layer loop and model boundary. Persistent
   multi-token execution is optional. Record concrete constraints if incomplete.

All device operations are serial and bounded. On a hang, save tt-triage before
terminating only this run's process. Credentials, weights, builds, caches and
bulk logs remain outside Git.
