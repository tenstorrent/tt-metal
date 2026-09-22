# Resumable experiment checkpoint

Updated 2026-09-22 17:16 UTC. **Full token-to-logits prototype verified at B1/context128; slower than baseline.**

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
root cannot enumerate all another user's FDs. Released by16:57; full connectivity and ring mesh open/close passed16:58.
The guard was archived after verifying release; device testing resumed.

Mark permits unlimited device resets during this allocation; no further reset
approval is needed after ownership is clear. Last reset13:44 passed all4-device
enumeration, current-runtime full connectivity and ring mesh open/close.
No reset was needed after the conflict ended. See the latest qualification below.

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

## Compiler-only layer loop and embedding (15:38 UTC)

`DecoderLoop` now binds all layer K/V addresses in the same128-byte weight table
and wraps every participating RISC in a device layer loop. It snapshots current
CB interfaces, uses current DeepSeek64-CB reset helpers, and places global
per-chip plus cross-RISC barriers around each layer. Native fabric initialization
coordinates chips. Local semaphore masks come from the actual descriptors.
Separate start/end barrier words fix a source-review race in consecutive packed
barriers. Native math kernels/configurations are retained. `decoder_loop` is
integrated at the original model boundary; `decoder_loop_embedding` also reads
BF16 embedding rows directly from device token IDs. Final norm/head/sampling
stay native. Immutable KV allocation binding is enforced; page remapping and
positions remain dynamic device inputs.

Both variants pass explicit mock SFPI compile/link and repeated cached calls;
latest logs `compile-mock-loop-v3.log` and `compile-mock-loop-embedding.log`.
Loop max configuration54,160B, including50,320B kernel text; adds4KiB persistent
state per participating core. `test_megakernel_loop.py` collects real layer0/31
one-/two-layer checks with distinct cache arenas and page migration. **All these
new paths remain numerically unqualified because the unrelated device owner is
still live.** Next hardware order: gather tail, postattention, attention tail,
complete layer, loop1/2, full32model; preserve triage on any hang. The fullmodel
benchmark accepts both loop variants and verifies against saved baseline.

## 16:10 token-to-logits compiler checkpoint

Added a standalone native-math final RMSNorm/HiFi2 head and a composed
`decode_token` mode: BF16 embedding lookup, all32 decoder layers, final
all-gather, final RMSNorm and BFP8 vocabulary projection in one mesh program.
Native sampling remains outside the program and owns token/position feedback.
`decoder_loop_head` is an intermediate debugging mode with a separate final
all-gather and norm/head program. Head output shape is the existing per-chip
BF16 padded vocabulary tensor; final RMSNorm affine stays folded in the weight.

The complete body uses86 layer workers plus16 head and8 terminal norm workers.
The final gather runs only after the last layer's second reduction, borrows
its completed sixteen-tile output CB, and reuses open fabric connections without
changing reduction generations. Terminal norm reuses the layer norm output.
This requires active B1 positions; negative/inactive rows and serving remain
unsupported. KV allocations bind before warmup; no mid-trace allocation rebinding.

Real SFPI mock-UMD compilation and model.decode entry/cache-reuse smoke passed,
with32 distinct KV pairs and no cache misses on four further calls. Max program
code/config remains54,208B; descriptor-derived max local CB845,824B/core,
head835,584B/core, loop state4096B on each of86 cores. These exclude firmware,
other tensor allocations and profiler overhead; they are not physical L1 or
execution measurements. Actual Blackhole grid is11x10; mock compilation used
its available13x10 grid while reserving terminal workers inside11x10.
New real-checkpoint traced norm/head comparison exercises3 inputs and8 replays;
all13 decoder/loop/head tests collect, but the head test has NOT run on hardware.
Current host TTNN/health/Tracy build and tt_pybinds install pass. Logs and JSONs:
`compile-mock-head.log`, `compile-mock-token-final.log`,
`token-descriptor-footprint.json`, `collect-token.log`, `build-token-checkpoint.log`.
No newer hardware accuracy or speed claim replaces the0414386d results.

## 16:31 traffic instrumentation and matching-grid compiler checks

Parent verified cfbc1a67 on GitHub16:17 and preserved a curated287-entry evidence
archive locally. Hardware conflict remains; no physical device work since14:31.

A mock descriptor with two harvested columns now yields the actual11x10 worker
layout and NUM_L1_BANKS=110. Full token model entry/cache reuse compiles and passes
there too (`compile-mock-token-110.log`). A reviewed loop optimization reads the
KV address table only on34 KV/attention reader cores;52 other workers did not
consume those shared columns. Projection readers retain their own weight-table
reads. This removes212,992 source-derived request bytes/chip/token, not a measured
speedup. Compiler-only validation remains explicitly separate from hardware.

Focused real-weight native/fused MLP traffic harness and coverage-gated analyzer
are ready. NoC profiler metadata was found to clamp payloads at8,160B; host test
reproduced8192->8160. Extended the payload into seven reserved bits, preserving
wire size8B and old low-field encoding. Boundary/legacy tests pass through the
real CMake target; host runtime/TTNN rebuild and install pass. First SFPI attempt
caught an unsigned-int/uint32_t template mismatch, fixed with explicit template
type; profiling-enabled head/full-token compilation now passes. Instrumented
full-token maxcode/config59,936B (13x10 mock), uninstrumented54,208B. Actual
hardware traffic and physical validation of this profiler change remain pending.
Logs: `noc-event-metadata-before.log`, `noc-event-metadata-cmake-v2-test.log`,
`build-noc-metadata-v2.log`, `compile-mock-head-noc-v2.log`,
`compile-mock-token-noc.log`, `traffic-analyzer-host-check.log`.

## 16:46 alignment and inactive warmup review fixes

Current code fixes two defects in the previously compiler-only full-layer path:
page-table scalar reads and second RoPE half-face reads could violate Blackhole
DRAM source/destination low6-bit congruence. New scratch helper preserves that
alignment; page entries remain4B reads rather than reading beyond short tables.
RoPE loads each256B row into scratch then copies halves locally. CPU audit1042
address cases and64word face-layout roundtrip pass; SFPI11x10 model-entry/cache
reuse passes. No physical DMA/numerical validation is implied.

The current generator explicitly warms decode at position-1 after prefill. The
full layer now handles that sentinel without page-table/KV access: a separate
UINT32 CB30 and read_tile_value mailbox synchronization give every TRISC the
same branch, then active paths call the original cache untilize/update/tilize
body. Both inactive cache writers still signal attention readiness. Native SDPA
skips the inactive query; its wrapper emits zero concatenated scratch for the
ignored warmup logits. RoPE maps the inactive rotary sentinel tozero. Native
sampling restores saved state afterward. This supersedes the earlier statement
that negative positions are wholly unsupported; only sentinel-1 warmup is now
implemented, still hardware-unqualified. Other negative values/serving unsupported.

Input metadata layout/dtype and native head block4/two-reader geometry are now
checked explicitly. Real loop test preserves prefilled KV across3 inactive
warmups and saves failure tensors before active replay/remapping tests. Compiler
max code/config54,240B, local CB peak unchanged. Logs:
`compile-mock-token-alignment.log`, `compile-mock-token-contract.log`,
`compile-mock-token-inactive.log`, `read-alignment-audit.json`.
Use pytest --timeout=0 under the bounded device runner so its triage runs before
terminating a hung experiment, rather than the inner pytest timeout intervening.

## Hardware composition qualification (17:06 UTC)

Access resumed16:58. Gather tail and output-projection tail pass. Initial
attention-tail PCC.99001 exposed native read_q's local-Q optimization on its
output core: relocation must set reader is_output_core=false as well as patch
Q-source coordinates. Corrected attention tail and complete decoder both pass
12checks with bitwise output/allKV,127→129/255→257, physicalpage migration and
in-place table remap/replay. Logs layer-attention-tail-hw-v2/layer-decoder-hw-1703.

Loop initially produced corrupt residuals despite exactKV: its public entry
accepted DRAM input while the second residual phase uses L1 with the same
TensorAccessor layout. Added native-equivalent to_memory_config conversion.
One- and two-layer device loops (real layer0/31) now pass with exact output/KV,
three inactive position-1 warmups preserving allKV, positions127→129, in-place
remapping and eight repeated replays. Failed output/cache tensors preserved.
Logs decoder-loop-one-hw-v2/decoder-loop-two-hw-1705. Full32-layer and terminal
head qualification are next; no full-megakernel latency claim yet.

## Full32-layer token-to-logits qualification (17:16 UTC)

`decode_token` now runs BF16 embedding, all32decoder layers through one reused
body/weight-KV table andscratch, final gather/native norm/HiFi2 head in one
four-chip program. Prefill andnative sampler remain separate. Full realmodel
B1/context128/32tokens/selected precision andgreedy settings: teacherlogits and
all64KV bitwise exact,32/32 greedy outputs,3stable repeated generations.
`model-decode-token-128-v2`:11.153716ms/token median versus8.785510baseline
(26.96% slower). Fullprogram is numerically qualified at this configuration,
not a performance improvement or serving qualification.

First32-layer run exhausted PacketHeaderPool in nativefabricwriter. Full
tt-triage/lightweight assertion saved before terminating ownPID2176497;
reset17:12 plus4-chipenumeration/connectivity/ringmesh allpassed. Reset pool
before each nativephase after previous writes drain/connections close, using
currentAPI; realSFPI compilation andfullmodel retry passed. Originalfailedrun
andallrecovery evidence retained. Currentcontext2048baseline rerun fixes
teacherstream toHF reference (earlier2048baseline useddifferentstream).
Deviceprofiles/traffic andcurrentbaseline refresh remain next work.
