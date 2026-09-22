# Resumable experiment checkpoint

Updated 2026-09-22 15:06 UTC. **Verified partial prototype; full decode megakernel incomplete.**

Base origin/main: `b8915544692d8f9feb2c890afbc2f22791560cd2`.
Branch: `codex/llama31-qb2-megakernel`. Last hardware-qualified checkpoint:
`0414386dff2ae867395f99c2f6357524aa19e7de`. Later commits add explicitly
unqualified composition stages. Exact transfer SHA is in artifact
`PARENT_CHECKPOINT.md`; parent can transfer the git bundle and push this branch.
Mark authorized branch pushes only, no PR/posts. Do not copy credentials or
start remote OAuth. Node/job `qb2-120-p03t06` / `113796` expires 19:22:10 UTC;
fresh checkpoint due by 17:22:10 and before stopping. Do not change allocations.

## Immediate hardware ownership blocker

At 14:31:52 another user's container started a live `VLLM::EngineCor` holding
`CHIP_IN_USE_0_PCIe`: host PID1571890, namespace PID642, user `ttuser`, container
`e300d7509f7348e66843bac1dced2fa0cd6f652ffe070adabf19d4eaab6aca6d`.
Our allocation is still RUNNING/exclusive. The parent/operator has been asked
to have its owner release the hardware. Do not kill/reset underneath it.
Our blocked mesh-opening process was terminated only after preserving evidence.
`artifacts/OWNERSHIP_BLOCKED` guards the serialized runner and reset script.
Check `/proc/*/status` NSpid when interpreting a container PID; fuser without
root cannot enumerate all another user's FDs. Latest check15:00 still live.

Mark permits unlimited device resets during this allocation; no further reset
approval is needed after ownership is clear. Last reset13:44 passed all4-device
enumeration, current-runtime full connectivity and ring mesh open/close.
No hardware work has run since the conflicting workload took the device lock.

## Environment and commands

Run root `/home/moconnor/llama-megakernel-113796`; artifacts below `artifacts/`,
mirrored to `/data/moconnor/llama-megakernel-113796`. Matching current host/runtime
built with Clang20.1.8, CMake4.0.2, Python3.12.3, pinned SFPI7.80.0[956].
Torch2.11 CPU, Transformers5.12.1, NumPy<2 in `venv-current`; independent
`venv-report` has tt-perf-report1.3.0. System toolchain unchanged. Real checkpoint
revision `0e9e39f249a16976918f6564b8830bc894c89659`, selected `gu4_head8_lm_head_hifi2`
policy retained: GU BFP4, QKV/O/down/head and KV BFP8, BF16 activations/CCL,
LoFi projections, HiFi4/FP32 norm/RoPE/SDPA, HiFi2 head.

```bash
source ../artifacts/run-env.sh
export CCACHE_DIR=/home/moconnor/llama-megakernel-113796/cache/ccache
cmake --build build-current --target ttnn test_system_health tracy_profiler_cli_tools --parallel 16
cmake --install build-current --component tt_pybinds
# Physical execution, only after resolving OWNERSHIP_BLOCKED:
python ../artifacts/run_device.py --name UNIQUE --timeout 900 -- python -m pytest \
  'models/demos/llama31_8b_qb2/tests/test_megakernel.py::test_fused_layer_real_weights[gather_norm_mlp_tail]' -x -s
```

Device operations must run serially. Runner records exact commands, PIDs and
elapsed time; on a kernel hang, capture tt-triage before terminating only this
experiment's process. The unrelated existing Gemma runtime is used only for
the reset utility, never model measurements. Initial current build1280actions
passed; new SDPA descriptor binding built/installed successfully.

## Qualified results

Real original HF layer PCC0.997647. Local MLP intermediates exact at layers0/31
for two real token embeddings and repeated replay. Complete layers pass page
writes, physical-page migration, in-place captured page-table remapping,
positions127→129 and255→257 and repeated replay. Native RMSNorm standalone
also bitwise exact across3 embeddings/8replays. All32-layer B1/context128,
32generated tokens, selected precision and greedy k1/p0/T1/seed42:

| Mode | Host median ms/decode token | TT teacher logits / all64 KV / greedy |
|---|---:|---|
| Traced baseline |8.785510|reference|
| mlp |9.246204|bitwise exact / exact /32of32|
| mlp_reduce |9.161502|exact / exact /32of32|
| mlp_reduce, GU16 |9.383494|exact / exact /32of32|
| mlp_tail |9.133671|exact / exact /32of32|
| norm_mlp_tail |9.336578|exact / exact /32of32|

Each uses three stable warmed generations,31decode steps per trial. No speedup.
HF BF16 diagnostic is identical for baseline and prototypes: logitPCC0.977533,
relativeL2 .208836, teacher top1 agreement100%, worstKV PCC.925991. The .99 HF
criterion fails for both; original failure/tensors retained. This is not task
accuracy qualification. Context2048 baseline completed:9.223623ms/token median;
matched prototype pending. Independent HF2048 reference completed, all finite.

Separate qualified device profiles use explicit buffer drains after warmup and
each window. `profile-baseline-drained/` and `profile-gu16-drained/` contain3
complete windows/all4devices/no missing durations, raw Tracy/device CSV,
`coverage.json` and concise `decode-*-summary.csv`. Baseline999ops/device/window.
Device3 at1350MHz median kernel sum8.525387ms / firmware span9.393458ms;
GU16 MLP+RS9.091880ms /9.853521ms. Summed firmware durations overlap and are not
latency. Earlier captures without drains lose records and are provisional.
Measured GU16 coordinator barrier median/p95: GU4.029/9.388us, down.348/.351us.
Phase durations overlap; do not sum them. No measured DRAM traffic or serving
benchmark yet. Source-counted per-chip/layer GU16,515,072B/down15,597,568B.

## Implementation and compiler-only additions

Verified one body reuses a32-row DRAM weight-address table and scratch for
RMSNorm→gate/up→BF16SiLU→multiply→down→four-chip reduce-scatter→residual.
Two fabric workers reuse native BF16 reduction order with program-local receive
CBs. Opt-in LOCAL_STAGING_CB native writer hook is hardware-qualified.
Default GU8 workers; GU16 is exact but slower. Optional single-block persistent
scratch passes stages/layers, full-model timing unqualified. Prior larger
persistent scratch collided with prefill norm L1. Four-worker fabric attempt
hung; triage preserved before reset; two workers passed.

New unqualified modes compile/link with explicit mock UMD, execute no physical
hardware, and reuse program cache for layer rows31,1,16,0:

- `gather_norm_mlp_tail`: fabric all-gather before the verified tail, same links.
- `post_attention`: O projection→RS/add→AG/norm→MLP→RS/add in one mesh program.
- `attention_tail`: current native paged SDPA on32 separate cores, BF16 head
  concatenation and projection-ready flags before `post_attention`. Preserves
  native attention compute configuration and reduction grouping. Uses CB32;
  Blackhole supports64 CB indices. A small private Python binding builds the
  native SDPA descriptor with its validation/output-spec checks.

Latest successful compiler logs: `compile-mock-gather-v4.log`,
`compile-mock-post-attention.log`, `compile-mock-attention-v3.log`.
First real all-gather attempt failed host semaphore allocation, fixed by
reserving norm-only IDs only on participating cores. Retry was blocked in mesh
opening by the ownership conflict, so no new-stage numerical claim is made.
Code/config mock maxima: gather18,000B; postattention23,088B. Default local
MLP projection scratch690,176B/core plus57,344B packed output. These are compile
and layout facts, not measured device bandwidth.

## Next work

After ownership release, verify enumeration/connectivity/mesh; recover if needed.
Test gather tail, postattention then attention tail serially with real layer
remap/replay checks, fullcontext128 exact comparison and matchedcontext2048.
Collect actual device traffic and final mode device profiles separately from
host/serving runs. Continue QKV/RoPE/paged-KV composition, then a device32-layer
loop with weight/KV tables and embedding/head/sampling boundary. Current32
layer calls are captured by a host trace: this is not yet a full decoder
megakernel. Prefill remains native; batch>1, serving, persistent multi-token
execution and broad accuracy are unqualified.

## Compiler-only complete layer (15:20 UTC)

Uncommitted `decoder` mode additionally includes first all-gather/RMSNorm,
QKV projection, native RoPE arithmetic and native BFP8 paged-cache update
conversion sequence. One mesh program now contains the complete single layer.
It consults device position/page table on every invocation. Current SFPI mock
compile/link/cache-row reuse passes (`compile-mock-decoder-v4.log`); maximum
configuration49,488B including47,872B kernel text. **Still no hardware execution
of new stages due to the ownership conflict.** Layer-loop/model boundary remains
to implement; source progress continues using current cross-RISC CB-reset APIs.
Parent receipt15:16 confirms compiler checkpoint8ee5264a pushed to the requested
branch and conflict escalated to Mark.
