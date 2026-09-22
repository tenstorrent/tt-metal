# Resumable experiment checkpoint

Updated 2026-09-22 12:58 UTC. **Incomplete and not hardware-validated.**

- Base: `b8915544692d8f9feb2c890afbc2f22791560cd2` (origin/main at setup).
- Implementation HEAD: `1222558c5f1d52491dce2b23f4819727068445be`; subsequent
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
  **Hardware assertions have not run.**
- A CPU-only HF BF16 real-checkpoint reference completed at context 128 with
  32 predictions. Logits [32,128256] and every layer's K/V [1,8,159,128] are
  finite; cache lengths grow 128→159. Artifacts: `hf-reference-128/reference.pt`
  and `.json`; preparation took 139.33 s. This is not TT correctness/performance.
  Benchmark `--hf-reference` enforces the same teacher stream and adds HF
  logit/cache comparisons while retaining the selected TT baseline precision.
- Current-build full connectivity fails: chip 2↔3 has one internal link instead
  of two, missing chip 2 `(0,8)` / chip 3 `(0,3)`. Four-chip ring mesh fails.
- Isolating chips 0/1 lets topology map but ERISC firmware init times out.
  Runtime exits status 1 after 33.36 seconds; no process killed. Inspector and
  focused ARC/Ethernet/version triage are saved in `isolated-mesh-triage`.
- **Reset approval remains pending. No reset/reboot/firmware change performed.**
  `RECOVERY_REQUIRED` prevents device execution. Prepared
  `commands/recovery-one-reset.sh` requires explicit authorization and checks
  node/allocation/owners, performs at most one reset, then checks connectivity
  and mesh. Do not run it until approval arrives.
- No device latency, correctness, profiler, measured DRAM/synchronization,
  serving result, or performance improvement exists yet. See artifact REPORT.md.

## Implemented scope and next actions

One local gate/up→SwiGLU→down program, BF16 intermediate rounding and original
BFP4/BFP8 weights, reusable scratch/address table; explicit all-32-layer
integration and real-weight layer/model comparison harnesses. This is **MLP
fusion only**: prefill, normalization, attention/KV, collectives and terminal
embedding/head/sampling remain native TTNN; no complete decoder/device layer
loop or vLLM qualification. B1 only. Current Blackhole supports 64 CB indices.

1. Continue independent review while reset approval is pending; the context-128
   CPU reference is now prepared. Preserve this scope; compiler success does not qualify the body.
2. After approved recovery and passing health/mesh, run the serial bounded
   `commands/validate-after-recovery.sh`: existing HF layer comparison, both
   fused-layer modes, then full-model baseline/prototype at context 128.
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
