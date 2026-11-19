# Fused all-gather / packed MLP candidate

Status: CPU implementation/audit with parent-run device results added below.
Parent measured separate/minimal/native-SwiGLU boundaries and found conventional
fusion blocked by its Linear collective builder. No TTNN import, mesh open,
device experiment, or production-default change was performed by this worker.
Parent owns device validation and acceptance. This is an independent
implementation audit, not a stage review. Qwen3.6-27B is dense; MoE is N/A.

The new `tt/fused_ag_candidate.py` implements the complete decode boundary:
attention partial reduce-scatter -> hidden-sharded residual add -> distributed
RMSNorm (stats gather only) -> fused hidden AG/packed gate-up -> local SwiGLU ->
existing local down projection/reduce-scatter -> hidden-sharded residual add.
No full hidden gather is inserted between the sharded norm and fused AGMM.
The output remains `[1,B,1280]` and can feed the next decoder directly.

## Why the previous evidence does not settle decode

`doc/multichip_decoder/final_packed_agminimal.json` measures 512 rows, packed
gate/up with separate SwiGLU: 0.753429 ms fused versus 0.496683 ms separate,
PCC 0.999992667, with its device admission fence included. It does not measure
the full residual/norm/down/RS boundary, decode rows 1/32, or the native fused
SwiGLU interface now exposed by this checkout. It is valid rejection evidence
for that measured prefill candidate, not for every decode family.

The frozen `optimized_decoder.py` remains untouched. Its default GDN hidden
norm uses a replicated FP32/SFPU reduction tree. Distributed BF16 stats change
the numerical tree, so the new probe records per-user and MLP-contribution PCC
in addition to full-output PCC. Passing this synthetic topology probe is not
sufficient real-input or full-layer accuracy evidence.

## Contracts and candidate table

All routes use TP=4 on mesh 1x4, Linear topology, cluster axis 1, BF16
activations/CCL and the decoder's unchanged MLP weight dtype/fidelity. Logical
B=1/32 is represented as M in `[1,1,B,1280]`, with exactly one physical 32-row
tile. It is never represented as 32 separate one-row batches.

| Boundary | Local shape / memory | Adaptation |
|---|---|---|
| residual and attention reduction | `[1,B,1280]` BF16 | retain hidden sharding through both adds |
| pre-norm stats | `[1,1,B,32]`, tile padded | gather stats to width128; do not gather hidden data here |
| normalized activation | `[1,1,B,1280]`, interleaved DRAM | gather fused with gate/up |
| gathered scratch | `[1,1,B,5120]`, physical M32, interleaved DRAM | explicit persistent buffer per prepared logical batch |
| contiguous packed weight | `[1,1,5120,8704]`, interleaved DRAM | setup converts DRAM-sharded weight and crops reader padding, preserving gate4352/up4352 ordering |
| paired packed weight | `[1,1,5120,8704]`, interleaved DRAM | reuse `swiglu_weight`: alternating gate32/up32 tiles, not contiguous halves |
| separate SwiGLU | `[1,B,4352]` | preserve `round_silu` policy |
| fused SwiGLU | `[1,B,4352]` | native pair-wise epilogue; arithmetic/rounding change must pass contribution/full-layer gates |
| down weight | existing `[4352,5120]` local K shard | selected DRAM-sharded decoder path unchanged |
| output | `[1,B,1280]` BF16 | down RS and residual add; gather only outside measured function for reference comparison |

| Family | Residual before/after | Hidden collective payload | Persistence / expected tradeoff |
|---|---|---|---|
| current best `_finish` | hidden shard / hidden shard | hidden AG + output RS | existing selected norm and DRAM projection; measured baseline in every probe |
| `separate` | hidden shard / hidden shard | stats AG + hidden AG + output RS | distributed-norm control isolates norm-tree change before fused APIs |
| `minimal` | hidden shard / hidden shard | stats AG + fused hidden AG + output RS + tiny admission AG | one persistent hidden buffer, dedicated semaphores; 10-column compute, single useful M row |
| `minimal --fuse-swiglu` | hidden shard / hidden shard | same | removes separate gate/up slicing and elementwise kernels; consumes paired weights |
| `mcast` | hidden shard / hidden shard | stats AG + fused hidden AG + output RS | blocked: conventional collective builder rejects Linear fusion, independent of 2D matmul geometry |
| `mcast1d` | hidden shard / hidden shard | same | device-confirmed setup rejection: collective builder rejects Linear fusion; 1D matmul builder support is insufficient |
| replicated residual / all-reduce control | replicated / replicated | output RS + immediate AG, then norm | already represented by parent's replicated contract controls; not equivalent to these sharded families |
| fused down MMRS | hidden shard / hidden shard | fused output RS | existing `fused_mmrs_decode` policy can be crossed with any candidate after correctness; default remains parent's selected down path |

Per physical decode tile, local hidden data is 81,920 bytes, complete gathered
hidden scratch is 327,680 bytes, and each rank receives 245,760 useful remote
hidden bytes. Stats are 2,048 bytes/rank (32x32 BF16), receiving 6,144 remote
bytes; down RS reduces a 327,680-byte full hidden partial to 81,920 bytes.
These are logical tiled payload counts, not a claim about measured link
traffic or forwarded hops. Minimal adds a 2,048-byte/rank admission AG.
Its fence input/output occupy 10,240 bytes/device. Each prepared B signature
owns 327,680 scratch bytes; preparing B1 and B32 owns two buffers even though
both have the same physical capacity. BFP8 packed 5120x8704 adds 47,349,760
bytes/device (43,520 tiles x1088 bytes) while the existing decode weight is
retained. Paired-weight mode reuses the already-owned paired weight instead.
Global semaphore storage and runtime transient outputs are additional.
If selected, parent must include the actual persistent delta in the context
contract and 64-owner resource check; do not allocate every rejected variant
for the final model.

## API/source audit

The native minimal validator allows folded physical rows M32 even when
logical M=1. Weight K5120 = localK1280 x4; local K has40 tiles. Default candidate
uses Mblock1/Kblock8/Nblock4, subblock1x2 and grid10x2. Kblock8 divides40;
the FP32 destination footprint is two tiles. N272 tiles is internally padded
over ten N owners; for fused SwiGLU the op partitions136 gate/up pairs and
keeps every pair together. Explicit `transpose_core_grid=False` fixes M on
grid.y and N on grid.x. Ten compute columns leave the eleventh device column
free for native mux cores. Two links and one worker/link satisfy
`ceil(in0_axis / workers_per_link) == num_links` with in0_axis2.
Eight-by-two is the first geometry control, then Kblock4/8. These configurations
pass source arithmetic checks, not hardware compilation/latency checks.

`all_gather_minimal_matmul_async` returns a vector whose first tensor is the
matmul/SwiGLU result. Its `persistent_output_buffer` is **gather scratch**;
it is not persistent matmul output. `fuse_swiglu=True` requires paired gate/up
tiles and returns width4352. No FSDP axis is used.

Correction after parent device test: conventional AGMM is blocked on this
Linear contract. The matmul portion alone is not restricted to 2D. Its factory
calls the dedicated experimental `ccl_fusion_mcast_1d.cpp` builder, which
supports `mcast_in0=True` and explicitly rejects `gather_in0` and multicast-in1.
It does not accept the DRAM-sharded matmul program config. Candidate `mcast1d`
uses interleaved inputs/weights, grid8x4, per-core M1/N9, Kblock4, subblock1x1;
272 N tiles need31 cores with an N tail on the last core. A 2D8x1 control
uses per-core M1/N34 and subblock1x2 (272=8x34). Conventional AG workers start
at (0,5) for1D or (0,2) for2D, with one link and two workers/link, outside the
compute rectangle. Its return order is `(gathered, matmul_output)` and its
persistent keyword is also `persistent_output_buffer`. It has no cluster-axis
argument; on the required1x4 mesh its whole-mesh group is the intended TP group.

However, the subsequent `all_gather_async_default_program_factory.cpp:220`
explicitly asserts `!((topology == ccl::Topology::Linear) && fuse_op)`.
Parent's `ag_mcast1d_l0_b32.log` reproduces that exact assertion before candidate
execution. Both conventional 1D and 2D routes share this builder; switching
geometry cannot repair the rejection. The initial audit missed this downstream
collective restriction and overstated conventional-route viability. Do not
repeat the Linear 2D command as if it were an independent workaround.

Source anchors:

- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/{all_gather_minimal_matmul_async_nanobind.cpp,device/all_gather_minimal_matmul_async_device_operation.cpp,device/all_gather_minimal_matmul_async_program_factory.cpp}`
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/{all_gather_matmul_async_nanobind.cpp,device/all_gather_matmul_async_program_factory.cpp}`
- `ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/{ccl_fusion_mcast_1d.cpp,ccl_fusion_mcast_2d.cpp}`

## Trace ownership and exact parent hooks

The candidate allocates all dedicated global semaphores on the entire actual
compute/storage core grid, not an assumed8x8 region. Dedicated fused and fence
semaphores never enter the decoder's ordinary CCL ping-pong pool. `prepare`
allocates all remote-write targets before warmup/capture. The same owner must
remain alive until traces using it are released. One command queue is assumed;
concurrent queues sharing this owner are unsupported. Ordinary CCL state is
owned by the decoder and follows the parent's current persistence policy.

Minimal keeps the stage4 tiny ordinary-AG admission fence inside every traced
invocation. Stage4 unfenced persistent-scratch launches hung under queued
replay; a barrier argument alone did not fix that. The candidate does not
reinterpret that failure as solved by larger semaphores or by host fences.
The complete-boundary timing includes admission overhead. Conventional fusion
would use its native all-gather barrier and persistent scratch, but is blocked
before execution for this Linear topology.

Parent integration requires no main-file edit to begin:

```python
from models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate import FusedAGMLPCandidate

candidate = FusedAGMLPCandidate(model, family="minimal")
candidate.prepare((1, 32))
candidate.install()  # only this explicit hook changes this decoder instance
# Existing harness: run exact-shape eager warmups, then candidate.seal().
# Existing harness: capture/replay complete decoder, holding candidate alive.
# Revert only after traces are released: model._finish = candidate.original_finish.
```

`finish` preserves the original prefill function and handles decode only. It
accepts the existing compact `[1,B,H/4]` interface and legacy `[B,1,H/4]`, returning
the caller's shape. Other logical prompt lengths remain on the original path.
To isolate norm effects, install `family="separate"` first. Do not replace the
input attention norm with this MLP-only hook. Extending fusion to input QKV/GDN
requires separate head/BA ownership and GDN correction handling.

## Owner-run commands and acceptance order

Use the parent's environment/hardware wrapper and live-hang capture policy.
Run commands sequentially; this worker has not executed any of them.
The standalone probe takes the same policy JSON as the selected decoder; pass
the parent's integrated policy with `--policy '{...}'` when different from
current defaults. Every command compares against the current selected `_finish`
with that same policy, not a generic slow matmul control.

```bash
python -m models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate --layer 0 --batch 32 --family separate --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/ag_separate_l0_b32.json
python -m models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate --layer 0 --batch 32 --family minimal --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/ag_minimal_l0_b32.json
python -m models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate --layer 0 --batch 32 --family minimal --fuse-swiglu --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/ag_swiglu_l0_b32.json
```

Repeat surviving candidates with `--batch 1` and both `--layer 0` and `--layer 3`.
Minimal geometry controls are `--grid 8 2 --k-block 8` and `--grid 10 2 --k-block 4`.
The probe records eager, per-user and contribution PCC >=0.995, finiteness,
exact eager/first/last replay equality, and50 queued replays timed only with
endpoint synchronization. It writes partial results before numerical assertions.
It uses actual model weights and deterministic synthetic residual/attention
partials; this is a topology probe, not a real-input quality gate. Layer-level
integration must still use parent's recorded inputs, changed-input traces,
allocation checks, watcher, native profiler/tt-perf-report, and all original
accuracy/cache/context gates. Run watcher separately from profiler. A favorable
boundary result alone cannot select a production path.

CPU validation by this worker: Python AST/bytecode compilation, Black/isort and
git whitespace checks. Parent device results follow; they do not establish
watcher, changed-input replay or memory-pressure acceptance for these rejected
candidate routes.

## Parent results and the conventional Ring question

Parent ran the full adapted boundary at layer0/B32 with source SHA256
`01de0c6d43e5f6305811945e86ee3de2419c1bfba8928e9a04d13f66cd6c5d85`.
These are parsed results from the named JSON artifacts, not estimates:

| Artifact | Baseline ms | Candidate ms | Contribution PCC | Outcome |
|---|---:|---:|---:|---|
| `ag_separate_l0_b32.json` | 0.264523 | 0.305686 | 0.9999935195 | distributed norm control passes, slower |
| `ag_minimal_l0_b32.json` | 0.264881 | 0.533441 | 0.9999935195 | fused AGMM passes, slower |
| `ag_swiglu_l0_b32.json` | 0.264407 | 0.536692 | 0.9999939427 | fused AGMM plus native SwiGLU passes, slower |
| `ag_mcast1d_l0_b32.json` / `.log` | 0.264560 | unavailable | unavailable | baseline passes; candidate rejects Linear fusion at factory line220 |

All three successful candidate runs report finite eager/replay outputs,
per-user PCC above0.999995, and exact eager/first/last replay equality after
50 queued replays. Distributed norm consumes the sharded residual, and fusion
directly gathers the normalized local1280 into the K5120 projection. There is
no forced replicated-residual restoration before the candidate. Thus minimal
and native-SwiGLU runs provide valid **adapted-family rejection evidence for
the measured layer0/B32 configuration**. They establish that fusion was not
discarded after its first API/layout error. They do not prove every geometry,
B1, or full-attention layer is slower. Conventional fusion is an exact
API/topology blocker, not a measured latency loss.

Do not change only `Topology.Linear` to `Topology.Ring` on the current mesh.
Parent's log records `FABRIC_1D`; `fabric_context.cpp:139-140` maps this to
Linear and maps `FABRIC_1D_RING` to Ring. A valid Ring attempt needs a newly
opened mesh with appropriate fabric configuration and a verified physical
closing link between first and last ranks in the **actual fabric order**.
Four P300c devices, a1x4 logical shape, two links on axis1, and ordinary Linear
success do not prove that closing link or compatible ring routing.

This audit found no saved peer-to-peer link map establishing that closing edge
for the current chips. Stage4 triage port-health tables show four live
ports/device but do not identify their peers; they do not establish a specific
cycle. Repository QB Ring tests are hardware-family coverage, not this host's
connectivity evidence. Source also requires care:
`ccl_common.cpp::get_axis_topology` checks both fabric wrap configuration and
physical endpoint adjacency. The no-cluster-axis neighbor path can derive
WRAP from logical extent alone; getting past argument setup does not prove
physical Ring legality. Conventional AGMM passes no cluster axis.

If parent later obtains a saved UMD/physical-discovery peer map and fabric-node
ordering proving the cycle, a controlled Ring evaluation would require:

1. Close the old mesh and open a new1x4 mesh with `FABRIC_1D_RING`; confirm all
   four intended chips and the closing edge are retained. No claim is made that
   a different logical shape is necessary; fabric/order need independent validation.
2. Pass an ordinary small Ring AG smoke on that exact ordering, then recreate
   weights and dedicated buffers/semaphores on the new mesh. Do not reuse
   resources from the prior Linear mesh.
3. Adapt conventional candidate and control to that consistent fabric contract,
   then run the same complete-boundary eager/queued-replay comparison. Current
   CLI hardcodes FABRIC_1D/Linear and has no Ring flag; there is no valid existing
   one-line Ring retry. Do not bypass the native assertion.

Absent that connectivity evidence, no Ring device launch is recommended.
Minimal Linear already supplies executable fused comparison without changing
the hardware/fabric contract. If more coverage is required within the parent's
stage budget, exact supported retries are:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate --layer 0 --batch 1 --family minimal --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/ag_minimal_l0_b1.json
python -m models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate --layer 3 --batch 32 --family minimal --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/ag_minimal_l3_b32.json
python -m models.autoports.qwen_qwen3_6_27b.tt.fused_ag_candidate --layer 0 --batch 32 --family minimal --grid 8 2 --k-block 8 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/ag_minimal_g8x2_l0_b32.json
```

Pass the same integrated `--policy` as parent baseline when applicable.
The conventional failure is already reproducible via its preserved command
sidecar and log; repeating it or running the2D variant is unnecessary.
