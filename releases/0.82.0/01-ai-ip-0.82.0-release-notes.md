<!--
AI-IP release notes for v0.81.0 + v0.82.0 (since v0.80.0), the Nov-13 AI-IP release.
v0.81.1 is a patch of v0.81.0 with one packaging change (PR 61961) and no Quasar change.
Feature set: Jira epic AIIPSW-18 (Nov-13 Release) and its child tickets AIIPSW-27, 35,
39, 78, 80, 82, 83, 85, 86, 100 and 128. AIIPSW-82 and AIIPSW-83 (Fabric) were closed
because the Horizon design does not need them; they have no PRs here.

Built with gen_release_notes_0.82.0.py from the tt-metal main history.
  release:  v0.82.0, commit bfc3f70fd49 (stable). Its code is the same as a74e6aca315:
            the two later commits (PRs 65908, 65903) change only CI workflow files.
  window:   the 1,494 PRs merged to main between the v0.80.0 cut (4732ed4672e,
            2026-10-07) and the v0.82.0 cut (8b53dfa87b6, 2026-11-03): 651 up to the
            v0.81.0 cut (85b69f498da, 2026-10-20) and 843 after it. The GitHub release
            bodies list each window as one squashed "Promote main -> stable" PR (61691,
            64009), so the PRs are enumerated here from the main history. The 8 PRs that are
            only on the stable branch (61961, 62481, 59470, 62515, 62531, 64268, 65908,
            65903) change no Quasar code.
  scope:    231 Quasar-related tt-metal PRs: 28 against the Nov-13 features,
            203 supporting. Plus 15 PRs in other Tenstorrent repositories. The other
            1,263 PRs of the window are not Quasar-related and are not listed.

Selection: a PR is Quasar-related when its title names Quasar, Horizon, Trinity, the
emulators, ATT or the IP variants, or when a meaningful share of its files sits under a
Quasar-specific path (tt_llk_quasar/, ckernels/quasar/, tt-2xx/, experimental/quasar/,
the Quasar model and op-test trees, tests/scripts/quasar/, impl/emulation/).
Repository-wide refactors that touch a few Quasar files in passing are left out.
A PR goes under a feature when its description says it was made for that feature.
Every selected PR is listed once.

PR titles are reproduced as authored, with formatting-only passes so they render outside
GitHub: leading "#NNNNN:" issue prefixes and leading [bracket] tags removed, Markdown-active
characters escaped, first letter capitalised.
-->

# Quasar, Trinity and Horizon Release Notes - v0.81.0 + v0.82.0

> Combined release notes for **v0.81.0 + v0.82.0**, since **v0.80.0** (the Oct-15 AI-IP release). Delivered for the **Nov-13 AI-IP release**. v0.81.1 is a packaging patch of v0.81.0 and has no Quasar, Trinity or Horizon change.

## Summary

- **TTNN/Kernel Ops: Qwen3-VL-2B related Kernel Ops** (AIIPSW-85) - 14 PR(s). TTNN kernel ops and tests for the Qwen3-VL vision-language model on Quasar.
- **TTNN/Kernel Ops: YOLO related Kernel Ops** (AIIPSW-86) - 8 PR(s). TTNN kernel ops and tests for the YOLO vision model on Quasar.
- **TTNN/Kernel Ops: Port Llama and YOLO Kernel Ops to Trinity, and Qwen3-VL-2B Kernel Ops to Horizon** (AIIPSW-27, AIIPSW-35, AIIPSW-128) - 2 PR(s) in tt-metal, 1 in other repositories. Trinity and Horizon use the Quasar kernel ops from `ttnn/cpp/ttnn/operations/experimental/quasar/`.
- **LLK: Quasar OFT related LLK features** (AIIPSW-100) - 4 PR(s). Op tests for the OFT (orthographic feature transform) model on Quasar, and the op fixes they found.
- **Horizon: ATT / flat address UD, UE, and runtime verification across partitions** (AIIPSW-78, AIIPSW-80) - 11 PR(s) in other repositories. Flat addresses and ATT checks per access point, and IP-partitioned Horizon builds in which each partition opens as its own device.
- **Debug Tools: tt-npe on Trinity** (AIIPSW-39) - 3 PR(s) in other repositories. tt-npe models a Quasar-family device from a device-model YAML file.
- **Supporting work** - 203 PR(s) in 6 areas. See section 2.
- **Known issues** - 7 open issues, 5 of them on Trinity. See section 3.

## 1. Quasar, Trinity and Horizon changes

### TTNN/Kernel Ops: Qwen3-VL-2B related Kernel Ops (AIIPSW-85)  (14)

TTNN kernel ops and tests for the Qwen3-VL vision-language model on Quasar.

#### Model and op tests

- Model op tests: TTNN\_GRAPH\_OPS\_BF8\_AS\_BF16 switch for Quasar runs ([PR 59575](https://github.com/tenstorrent/tt-metal/pull/59575))
- Qwen3\_vl\_ops: adapt the Wormhole capture cases to Quasar ([PR 62796](https://github.com/tenstorrent/tt-metal/pull/62796))
- Qwen3-VL e2e test on Quasar: craq-sim, emulator and gap log ([PR 63013](https://github.com/tenstorrent/tt-metal/pull/63013))
- Qwen3-VL demo.py runs on Quasar ([PR 63354](https://github.com/tenstorrent/tt-metal/pull/63354))

#### Op fixes found in the Qwen3-VL bring-up

- Scatter: Quasar bring-up (self-loop DFBs to Scratchpads, compile fixes) ([PR 59571](https://github.com/tenstorrent/tt-metal/pull/59571))
- Layer\_norm: retarget the Quasar packer on the non-RMS interleaved path (#54630) ([PR 59573](https://github.com/tenstorrent/tt-metal/pull/59573))
- Rotary\_embedding\_llama (Quasar): re-init packer and unpacker in the prefill kernel ([PR 59903](https://github.com/tenstorrent/tt-metal/pull/59903))
- Concat (Quasar): build only the current input's accessor in the interleaved readers ([PR 59906](https://github.com/tenstorrent/tt-metal/pull/59906))
- Quasar layernorm: keep the reduce-scaler DFB at kernel scope ([PR 62427](https://github.com/tenstorrent/tt-metal/pull/62427))
- Slice (Quasar): keep the readers' index odometer under a scoped\_lock ([PR 62543](https://github.com/tenstorrent/tt-metal/pull/62543))
- Quasar sdpa readers: invalidate NoC-read page tables ([PR 62544](https://github.com/tenstorrent/tt-metal/pull/62544))
- Scatter (Quasar): write back the bf16 fp32 accumulator ([PR 62545](https://github.com/tenstorrent/tt-metal/pull/62545))
- Quasar: unpack int32 to Dest in permute and transpose ([PR 62744](https://github.com/tenstorrent/tt-metal/pull/62744))
- Plus\_one: stage interleaved sticks in a scratchpad for Quasar ([PR 62787](https://github.com/tenstorrent/tt-metal/pull/62787))

#### Known at the v0.82.0 tag:

- The op tests are in `models/experimental/ops/quasar/tests/qwen3_vl_ops/`.
- The end-to-end demo is `models/experimental/ops/quasar/qwen3_vl/demo/demo.py` (PR 63354).

### TTNN/Kernel Ops: YOLO related Kernel Ops (AIIPSW-86)  (8)

TTNN kernel ops and tests for the YOLO vision model on Quasar.

#### Model and op tests

- Yolo ops Quasar uplift ([PR 60455](https://github.com/tenstorrent/tt-metal/pull/60455))
- Quasar yolo/qwen3\_vl op tests: add small-grid cases for 1-2 compute nodes ([PR 63000](https://github.com/tenstorrent/tt-metal/pull/63000))

#### Op fixes found in the YOLO bring-up

- Quasar conv2d: L1 estimate misses the un-overlapped ACT\_ROW\_MAJOR DFB (2-core emulator OOM) ([PR 60631](https://github.com/tenstorrent/tt-metal/pull/60631))
- Compute\_kernel\_lib: Mask and MaskPosInf on Quasar ([PR 60639](https://github.com/tenstorrent/tt-metal/pull/60639))
- Moreh\_common: compile out the helpers Quasar can't build ([PR 60640](https://github.com/tenstorrent/tt-metal/pull/60640))
- Quasar tiled permute: cb\_in entries are exactly one tile ([PR 60684](https://github.com/tenstorrent/tt-metal/pull/60684))
- Yolo Quasar ops: fix pool window gather, untilize\_with\_unpadding implicit-sync overflow, div output dtype ([PR 62730](https://github.com/tenstorrent/tt-metal/pull/62730))
- Moreh\_softmax (Quasar): kernel-scope DFBs by reference and retarget the packer ([PR 63003](https://github.com/tenstorrent/tt-metal/pull/63003))

#### Known at the v0.82.0 tag:

- The op tests are in `models/experimental/ops/quasar/tests/yolo_ops/`.

### TTNN/Kernel Ops: Port Llama and YOLO Kernel Ops to Trinity, and Qwen3-VL-2B Kernel Ops to Horizon (AIIPSW-27, AIIPSW-35, AIIPSW-128)  (2)

Trinity and Horizon use the Quasar kernel ops from `ttnn/cpp/ttnn/operations/experimental/quasar/`. These changes make fast dispatch and IP-variant selection work on the Horizon (2.0.1) and Trinity (2.0.2) variants.

- Support FDS on all Quasar IP variants ([PR 59544](https://github.com/tenstorrent/tt-metal/pull/59544))
- Use IP variant supplied by UMD ([PR 61696](https://github.com/tenstorrent/tt-metal/pull/61696))

#### Delivered in other repositories (released separately, referenced here)

- Add an optional ip\_variant field to the SoC descriptor — `tt-umd` ([PR 3587](https://github.com/tenstorrent/tt-umd/pull/3587))

#### Known at the v0.82.0 tag:

- tt-metal reads the IP variant from UMD (PR 61696, with tt-umd PR 3587).
- Fast dispatch uses FDS on all three IP variants (PR 59544).
- The Qwen3-VL op suite passes on Horizon and on Trinity in the release pipeline.
- YOLO does not run on Trinity yet (known issue RELEASE-121, section 3).

### LLK: Quasar OFT related LLK features (AIIPSW-100)  (4)

Op tests for the OFT (orthographic feature transform) model on Quasar, and the op fixes they found.

- Add per-op OFT model tests for Quasar (Feature, Test Only) ([PR 60669](https://github.com/tenstorrent/tt-metal/pull/60669))
- Adjust reads on reshape to stay within range ([PR 61449](https://github.com/tenstorrent/tt-metal/pull/61449))
- Fix grid\_sample sharded grid stick size for unaligned shard widths ([PR 61465](https://github.com/tenstorrent/tt-metal/pull/61465))
- Add more oft tests ([PR 61658](https://github.com/tenstorrent/tt-metal/pull/61658))

#### Known at the v0.82.0 tag:

- The op tests are in `models/experimental/ops/quasar/tests/oft_ops/`.

### Horizon: ATT / flat address UD, UE, and runtime verification across partitions (AIIPSW-78, AIIPSW-80)  (11)

Flat addresses and ATT checks per access point, and IP-partitioned Horizon builds in which each partition opens as its own device. This work is in UMD.

#### Delivered in other repositories (released separately, referenced here)

- Expose devices' flat addresses for fabric, by access point — `tt-umd` ([PR 3647](https://github.com/tenstorrent/tt-umd/pull/3647))
- Give each access point its own flat addresses, with the horizon\_x2 map — `tt-umd` ([PR 3653](https://github.com/tenstorrent/tt-umd/pull/3653))
- Check each access point's flat\_address against UMD's ATT map — `tt-umd` ([PR 3724](https://github.com/tenstorrent/tt-umd/pull/3724))
- Check the build's ATT against UMD's map when a device opens — `tt-umd` ([PR 3725](https://github.com/tenstorrent/tt-umd/pull/3725))
- Let a device list several access points, and pick the one it talks over — `tt-umd` ([PR 3645](https://github.com/tenstorrent/tt-umd/pull/3645))
- Parse a build's ip\_layout.yaml once, in Cluster — `tt-umd` ([PR 3652](https://github.com/tenstorrent/tt-umd/pull/3652))
- Reset Horizon cores from UMD, through each tile's TSC — `tt-umd` ([PR 3654](https://github.com/tenstorrent/tt-umd/pull/3654))
- Let devices share a socket, taking turns on one channel per socket — `tt-umd` ([PR 3655](https://github.com/tenstorrent/tt-umd/pull/3655))
- Split horizon\_x2 by column, with the horizon\_x2\_columns map — `tt-umd` ([PR 3691](https://github.com/tenstorrent/tt-umd/pull/3691))
- Rename IpLayout to IpBlockLayout — `tt-umd` ([PR 3718](https://github.com/tenstorrent/tt-umd/pull/3718))
- Require each access point's IP block in ip\_layout.yaml — `tt-umd` ([PR 3723](https://github.com/tenstorrent/tt-umd/pull/3723))

#### Known at the v0.82.0 tag:

- tt-metal v0.82.0 pins tt-umd commit `7b508db`, which includes all the tt-umd PRs above.

### Debug Tools: tt-npe on Trinity (AIIPSW-39)  (3)

tt-npe models a Quasar-family device from a device-model YAML file.

#### Delivered in other repositories (released separately, referenced here)

- Support Custom Device Models — `tt-npe` ([PR 137](https://github.com/tenstorrent/tt-npe/pull/137))
- Add initial Quasar support to custom device model — `tt-npe` ([PR 139](https://github.com/tenstorrent/tt-npe/pull/139))
- Add mesh X-then-Y routing for Quasar via device model YAML — `tt-npe` ([PR 141](https://github.com/tenstorrent/tt-npe/pull/141))

#### Known at the v0.82.0 tag:

- tt-npe is released separately from tt-metal.


## 2. Supporting Quasar, Trinity and Horizon work

Quasar, Trinity and Horizon changes in this release that are not tied to a
named Nov-13 feature. They are part of the same delivery and are listed so
every tt-2xx PR in the release can be accounted for.

### LLK (`tt_llk_quasar`, Quasar ckernels and compute API, LLK tests)  (86)

- Reduce TRISC\_ID to the per-Neo thread id ([PR 55821](https://github.com/tenstorrent/tt-metal/pull/55821))
- Unpack\_to\_dest API ([PR 56543](https://github.com/tenstorrent/tt-metal/pull/56543))
- Implement ema for Quasar ([PR 57129](https://github.com/tenstorrent/tt-metal/pull/57129))
- Sfpi unified api ([PR 57605](https://github.com/tenstorrent/tt-metal/pull/57605))
- Remove hardcoding DEST to 32bit state during unpack-to-dest ([PR 58374](https://github.com/tenstorrent/tt-metal/pull/58374))
- Add dst\_idx param to unpack APIs ([PR 58416](https://github.com/tenstorrent/tt-metal/pull/58416))
- Add Quasar narrow-row pack-untilize via iDMA. ([PR 58428](https://github.com/tenstorrent/tt-metal/pull/58428))
- Add Int4/UInt4 support to Quasar unpack and pack test infra ([PR 58597](https://github.com/tenstorrent/tt-metal/pull/58597))
- Declare data\_format\_config\_set thread\_local in metal build ([PR 58840](https://github.com/tenstorrent/tt-metal/pull/58840))
- Same core compute semaphore implementation ([PR 58850](https://github.com/tenstorrent/tt-metal/pull/58850))
- Add dest index bounds asserts to math LLK APIs ([PR 58852](https://github.com/tenstorrent/tt-metal/pull/58852))
- Speed up Quasar LLK test compilation ([PR 59022](https://github.com/tenstorrent/tt-metal/pull/59022))
- Make pack clear\_dest\_bank\_id per-TRISC thread\_local ([PR 59062](https://github.com/tenstorrent/tt-metal/pull/59062))
- Bug fix: unpack-to-dest section handshake per tile\_regs… ([PR 59248](https://github.com/tenstorrent/tt-metal/pull/59248))
- Gate unpack-to-dest on a DEST occupancy semaphore (#58903) ([PR 59254](https://github.com/tenstorrent/tt-metal/pull/59254))
- Implement gcd for Quasar ([PR 59258](https://github.com/tenstorrent/tt-metal/pull/59258))
- Add quasar gather variant header overrides ([PR 59271](https://github.com/tenstorrent/tt-metal/pull/59271))
- Carry DEST bank where the unpacker and packer read it ([PR 59290](https://github.com/tenstorrent/tt-metal/pull/59290))
- Quasar SFPU: SrcS op types with SfpuIssue flag, shared math/loops for exp, square, add ([PR 59355](https://github.com/tenstorrent/tt-metal/pull/59355))
- Implement binary\_bcast for Quasar ([PR 59434](https://github.com/tenstorrent/tt-metal/pull/59434))
- Implement lcm for Quasar ([PR 59435](https://github.com/tenstorrent/tt-metal/pull/59435))
- Implement dropout for Quasar ([PR 59436](https://github.com/tenstorrent/tt-metal/pull/59436))
- Implement rand for Quasar ([PR 59437](https://github.com/tenstorrent/tt-metal/pull/59437))
- Implement reshuffle\_rows for Quasar ([PR 59438](https://github.com/tenstorrent/tt-metal/pull/59438))
- Implement unary\_max\_min for Quasar ([PR 59439](https://github.com/tenstorrent/tt-metal/pull/59439))
- Implement welfords for Quasar ([PR 59440](https://github.com/tenstorrent/tt-metal/pull/59440))
- Remove direct load/store builtin use ([PR 59472](https://github.com/tenstorrent/tt-metal/pull/59472))
- Feat: add alternative signbit algorithm that uses loadmacro ([PR 59543](https://github.com/tenstorrent/tt-metal/pull/59543))
- Quasar 2x column reduce only with an MxFp4 scaler ([PR 59597](https://github.com/tenstorrent/tt-metal/pull/59597))
- Remove unwanted fuser test entry ([PR 59624](https://github.com/tenstorrent/tt-metal/pull/59624))
- Quasar: fix test drivers that fail to compile under… ([PR 59644](https://github.com/tenstorrent/tt-metal/pull/59644))
- Finish Quasar eltwise binary broadcast tiny-tile and acc\_to\_dest ([PR 59673](https://github.com/tenstorrent/tt-metal/pull/59673))
- Quasar LLK tests multiply Float16\_b and MxFp4\_2x at LoFi ([PR 59719](https://github.com/tenstorrent/tt-metal/pull/59719))
- Fix Quasar fuser fidelity validation and unpacker configuration ([PR 59820](https://github.com/tenstorrent/tt-metal/pull/59820))
- Refactor: stop using deprecated function in llk quasar tests ([PR 59830](https://github.com/tenstorrent/tt-metal/pull/59830))
- Compare Blackhole and Quasar sweeps and record perf accounting ([PR 59833](https://github.com/tenstorrent/tt-metal/pull/59833))
- Fix: disable failing Quasar LLK tests ([PR 59926](https://github.com/tenstorrent/tt-metal/pull/59926))
- Quasar: metal-layer test for the 2x column reduce and its uninit ([PR 59958](https://github.com/tenstorrent/tt-metal/pull/59958))
- Clear dvalid wait masks at kernel boot ([PR 59976](https://github.com/tenstorrent/tt-metal/pull/59976))
- Validate math fidelity against source formats and skip redundant phases ([PR 60002](https://github.com/tenstorrent/tt-metal/pull/60002))
- Unify Copy Block and Pack Block Implementations ([PR 60151](https://github.com/tenstorrent/tt-metal/pull/60151))
- Add experimental ClusterSemaphore for cross-NEO PACK/UNPACK sync ([PR 60188](https://github.com/tenstorrent/tt-metal/pull/60188))
- Add Quasar experimental kernels ([PR 60293](https://github.com/tenstorrent/tt-metal/pull/60293))
- Startup barrier for the unpack-to-dest semaphore seeding ([PR 60388](https://github.com/tenstorrent/tt-metal/pull/60388))
- Quasar fuser: sfpu\_unary\_sweep unpacks only Float32 into its 32-bit Dest ([PR 60427](https://github.com/tenstorrent/tt-metal/pull/60427))
- Remove implied math format from Quasar tests ([PR 60443](https://github.com/tenstorrent/tt-metal/pull/60443))
- Quasar reduce test kernels: the scaler DFB outlives the reads and gets popped ([PR 60467](https://github.com/tenstorrent/tt-metal/pull/60467))
- Quasar: re-point the packer in tilize\_init ([PR 60490](https://github.com/tenstorrent/tt-metal/pull/60490))
- Align eltwise unary SFPU tests across WH/BH and Quasar ([PR 60533](https://github.com/tenstorrent/tt-metal/pull/60533))
- Shared reduce perf core matrix across WH/BH and Quasar (2/3, #53957) ([PR 60557](https://github.com/tenstorrent/tt-metal/pull/60557))
- Reduce perf: block length, reduce-to-one and reconfig cost (3/3, #53957) ([PR 60558](https://github.com/tenstorrent/tt-metal/pull/60558))
- Fix: \[Quasar\] use the same constant for reserved field of SFP\_STOCH\_RND ([PR 60564](https://github.com/tenstorrent/tt-metal/pull/60564))
- Quasar compute semaphores don't wait on unpacker 2 or packer 1 ([PR 60616](https://github.com/tenstorrent/tt-metal/pull/60616))
- Reduce logaddexp2's fp32 exponential exactly in base 2 ([PR 60714](https://github.com/tenstorrent/tt-metal/pull/60714))
- QuasarDatacopyToDestWriter unpacks Int32 to Dest, not just Float32 ([PR 60760](https://github.com/tenstorrent/tt-metal/pull/60760))
- Sweep tile shapes ([PR 60905](https://github.com/tenstorrent/tt-metal/pull/60905))
- Drop Float16 from the Quasar test and perf sweeps (stacked on #60911) ([PR 60920](https://github.com/tenstorrent/tt-metal/pull/60920))
- Remove racy experimental compute ckernel::Semaphore (Quasar) ([PR 60946](https://github.com/tenstorrent/tt-metal/pull/60946))
- Unpack-tilize into DEST and one SrcA MOP run per tile row ([PR 61421](https://github.com/tenstorrent/tt-metal/pull/61421))
- Fix: use LCONST\_0 for unused field of TTI\_SFP\_STOCH\_RND on Quasar ([PR 61423](https://github.com/tenstorrent/tt-metal/pull/61423))
- Feat: add PR gate for LLK Quasar tests using ttsim as a backend ([PR 61429](https://github.com/tenstorrent/tt-metal/pull/61429))
- Minor GCD/LCM fixes ([PR 61473](https://github.com/tenstorrent/tt-metal/pull/61473))
- Future-proof stall test against compiler optimizations ([PR 61733](https://github.com/tenstorrent/tt-metal/pull/61733))
- Quasar LLK tests: share the single-node DFB test program ([PR 61795](https://github.com/tenstorrent/tt-metal/pull/61795))
- Remove EN\_IMPLIED\_MATH\_FORMAT parameter from alu\_config ([PR 61865](https://github.com/tenstorrent/tt-metal/pull/61865))
- Plumb the codegen SFPU kernels through the compute API (1/3) ([PR 61879](https://github.com/tenstorrent/tt-metal/pull/61879))
- Plumb sfpu\_binary\_bcast through the compute API (2/3) ([PR 61880](https://github.com/tenstorrent/tt-metal/pull/61880))
- Plumb welford through the compute API (3/3) ([PR 61881](https://github.com/tenstorrent/tt-metal/pull/61881))
- Move Quasar SFPU sigmoid, silu, sqrt, operand kernels into llk\_sfpu ([PR 62040](https://github.com/tenstorrent/tt-metal/pull/62040))
- Drop the UNPACK\_INIT startup-barrier semaphore from the unpack-to-dest handshake ([PR 62071](https://github.com/tenstorrent/tt-metal/pull/62071))
- LLK tests: centralize unpacks\_to\_dest() helper ([PR 62192](https://github.com/tenstorrent/tt-metal/pull/62192))
- Add gather/compress LLKs ([PR 62201](https://github.com/tenstorrent/tt-metal/pull/62201))
- Ci: remove skipped Quasar tests for unpacking and packing formats ([PR 62395](https://github.com/tenstorrent/tt-metal/pull/62395))
- LLK tests: five more shared test helpers ([PR 62418](https://github.com/tenstorrent/tt-metal/pull/62418))
- Use new rvtt\_ttinsn builtin ([PR 62420](https://github.com/tenstorrent/tt-metal/pull/62420))
- LLK tests: shared test utilities, round 3 ([PR 62502](https://github.com/tenstorrent/tt-metal/pull/62502))
- Quasar layer\_norm large-tensor: no unpack-to-dest, retarget packer ([PR 62518](https://github.com/tenstorrent/tt-metal/pull/62518))
- Fix Quasar unpack to dest ([PR 62637](https://github.com/tenstorrent/tt-metal/pull/62637))
- Remove unpack-to-dest from the unary operand unpack family ([PR 62650](https://github.com/tenstorrent/tt-metal/pull/62650))
- Topk: compare values with the fp32 SFPSWAP mode ([PR 62653](https://github.com/tenstorrent/tt-metal/pull/62653))
- Quasar DFB intra test: order the PACK thread's push after its direct write ([PR 62717](https://github.com/tenstorrent/tt-metal/pull/62717))
- Fix gather\_compress test for the new \_llk\_math\_srcAB\_hw\_configure\_ signature ([PR 62912](https://github.com/tenstorrent/tt-metal/pull/62912))
- Fix precision loss in reduce row ([PR 62918](https://github.com/tenstorrent/tt-metal/pull/62918))
- Unpack 32-bit tiles to DEST in pack\_untilize and clear the UNP\_A face index after each tilize-to-dest run ([PR 62968](https://github.com/tenstorrent/tt-metal/pull/62968))
- Fix: disable hanging Quasar LLK metal tests ([PR 63372](https://github.com/tenstorrent/tt-metal/pull/63372))
- LLK tests: stop adding the deleted Quasar common/inc/sfpu as an include root ([PR 63713](https://github.com/tenstorrent/tt-metal/pull/63713))

### Kernel Ops (TT-NN `experimental/quasar` ops: matmul, `binary_ng`, SDPA, conv, pool, softmax, data movement)  (65)

- Matmul: Reuse config writes every block when per\_core\_M \< Mt ([PR 57957](https://github.com/tenstorrent/tt-metal/pull/57957))
- Add MX data types to TTNN ([PR 58175](https://github.com/tenstorrent/tt-metal/pull/58175))
- Matmul: honor packer\_l1\_acc with two K blocks ([PR 58178](https://github.com/tenstorrent/tt-metal/pull/58178))
- Matmul: don't share a sharded output with partials across N blocks ([PR 58180](https://github.com/tenstorrent/tt-metal/pull/58180))
- Minimal\_matmul: Quasar uplift of the Metal 2.0 port ([PR 58909](https://github.com/tenstorrent/tt-metal/pull/58909))
- Paged\_cache fused update\_cache: retarget the packer before the tilize on Quasar ([PR 58913](https://github.com/tenstorrent/tt-metal/pull/58913))
- Reshape\_view tiled: RawUInt32 mapping DFB on Quasar ([PR 58914](https://github.com/tenstorrent/tt-metal/pull/58914))
- Binary\_ng: borrow L1-interleaved operands and admit DRAM shards on the native factory ([PR 59064](https://github.com/tenstorrent/tt-metal/pull/59064))
- \#59399, #59400: Remove dead enough\_space\_width from quasar tilize and tilize\_with\_val\_padding ([PR 59401](https://github.com/tenstorrent/tt-metal/pull/59401))
- UINT16 to FP32 Typecast ([PR 59546](https://github.com/tenstorrent/tt-metal/pull/59546))
- Quasar Matmul 3 (DM Optimization) ([PR 59549](https://github.com/tenstorrent/tt-metal/pull/59549))
- \#59593: Reserve pending L1 output in Quasar tilize L1 budget check ([PR 59613](https://github.com/tenstorrent/tt-metal/pull/59613))
- Quasar device compile fix ([PR 59659](https://github.com/tenstorrent/tt-metal/pull/59659))
- Quasar SDPA: fix windowed fork and mask bugs after #59055 ([PR 59667](https://github.com/tenstorrent/tt-metal/pull/59667))
- Matmul: dst\_full\_sync doesn't raise the subblock limit ([PR 59703](https://github.com/tenstorrent/tt-metal/pull/59703))
- Fix Quasar pool2d nightly tests on N300: use single-chip device fixture ([PR 59709](https://github.com/tenstorrent/tt-metal/pull/59709))
- Restore NCRISC read cmd buf NOC\_CTRL VC default on exit in matmul in1 readers ([PR 59777](https://github.com/tenstorrent/tt-metal/pull/59777))
- Quasar SFPU: include ckernel\_sfpu\_converter.h in rpow and rdiv ([PR 59900](https://github.com/tenstorrent/tt-metal/pull/59900))
- Quasar Int32 -\> Float16\_b typecast reads Dest as sign-magnitude ([PR 59967](https://github.com/tenstorrent/tt-metal/pull/59967))
- Matmul: Fix Broadcast-Batch in0 Reuse Hang ([PR 60175](https://github.com/tenstorrent/tt-metal/pull/60175))
- Quasar binary\_ng: fix fallback hangs when a core makes several calls ([PR 60207](https://github.com/tenstorrent/tt-metal/pull/60207))
- \#59596: Reserve pending L1 output in Quasar untilize\_with\_unpadding L1 budget check ([PR 60250](https://github.com/tenstorrent/tt-metal/pull/60250))
- \#59595: Reserve pending L1 output in Quasar untilize L1 budget check ([PR 60251](https://github.com/tenstorrent/tt-metal/pull/60251))
- Fix gather\_in0 CB stride and fp32 unpack, RM pad value/fence, rank-1 tile pad, host to\_layout, untilize units (#57683) ([PR 60261](https://github.com/tenstorrent/tt-metal/pull/60261))
- Fix sdpa\_decode subblock width, DHT granularity and default core cap (#58408) ([PR 60270](https://github.com/tenstorrent/tt-metal/pull/60270))
- Add pad-region coverage for odd-width tilize\_with\_val\_padding (#60285) ([PR 60358](https://github.com/tenstorrent/tt-metal/pull/60358))
- Data\_movement: route codegen ops to native on Quasar ([PR 60367](https://github.com/tenstorrent/tt-metal/pull/60367))
- Quasar typecast test: Int32 -\> Float16\_b feeds sign-magnitude like every other integer typecast ([PR 60422](https://github.com/tenstorrent/tt-metal/pull/60422))
- Quasar tilize\_with\_val\_padding's reader builds for RV64 and pads an odd-width row ([PR 60423](https://github.com/tenstorrent/tt-metal/pull/60423))
- Quasar binary\_ng: the scalar SFPU kernel includes binary\_remainder.h ([PR 60425](https://github.com/tenstorrent/tt-metal/pull/60425))
- Add quasar op tests for vad variant ([PR 60472](https://github.com/tenstorrent/tt-metal/pull/60472))
- Binary\_ng: every layout mix runs native, each operand borrowed where it sits ([PR 60476](https://github.com/tenstorrent/tt-metal/pull/60476))
- DRAM-sharded matmul in1 reader builds on Quasar again: restore the read VC through the NOC API ([PR 60657](https://github.com/tenstorrent/tt-metal/pull/60657))
- Minimal\_matmul: the Metal 2.0 compute kernel builds on Quasar again ([PR 60660](https://github.com/tenstorrent/tt-metal/pull/60660))
- Quasar 2D-multicast matmul: multicast rectangles are always low corner to high ([PR 60680](https://github.com/tenstorrent/tt-metal/pull/60680))
- \#60693, #60694: Fix unsafe native L1 sharding in Quasar binary\_ng ([PR 60720](https://github.com/tenstorrent/tt-metal/pull/60720))
- Embedding: Metal 2.0 post-port passes for Quasar ([PR 60812](https://github.com/tenstorrent/tt-metal/pull/60812))
- Binary\_ng: subtract, multiply, divide and fused activations run native ([PR 60847](https://github.com/tenstorrent/tt-metal/pull/60847))
- Quasar Matmul 4 ([PR 60926](https://github.com/tenstorrent/tt-metal/pull/60926))
- Remove dead experimental compute kernels ([PR 61487](https://github.com/tenstorrent/tt-metal/pull/61487))
- Binary\_ng: fp32 runs native, on the SFPU kernel. Native factory default on. ([PR 61498](https://github.com/tenstorrent/tt-metal/pull/61498))
- Quasar: fix the kernel\_lib eltwise chain build and the TRISC stack overflow from DataflowBuffer (#61354, #61482) ([PR 61760](https://github.com/tenstorrent/tt-metal/pull/61760))
- Quasar pool2d: template the release\_out\_pages DFB helper ([PR 61783](https://github.com/tenstorrent/tt-metal/pull/61783))
- Binary\_ng: bound sharded FPU DST batch by fp32 dest accumulation ([PR 61825](https://github.com/tenstorrent/tt-metal/pull/61825))
- Cap matmul auto-config per-core block to output size ([PR 62001](https://github.com/tenstorrent/tt-metal/pull/62001))
- Move Quasar SFPU typecast, binary\_comp, mul\_int32 and fill kernels from tt-llk into llk\_sfpu ([PR 62008](https://github.com/tenstorrent/tt-metal/pull/62008))
- \#61887: Fix Quasar multicast matmul hangs in the DFB exit wait ([PR 62081](https://github.com/tenstorrent/tt-metal/pull/62081))
- Quasar-native softmax and batch\_norm (4 reader / 4 NEO / 2 writer) ([PR 62095](https://github.com/tenstorrent/tt-metal/pull/62095))
- Remove unused kernels/tests (21 experimental/quasar dataflow kernels) ([PR 62163](https://github.com/tenstorrent/tt-metal/pull/62163))
- Migrate Unified Matmul To Resnet and Llama on Quasar ([PR 62332](https://github.com/tenstorrent/tt-metal/pull/62332))
- \#62387: Quasar: fix DataflowBuffer destructor-drain hangs and resnet weight OOM ([PR 62388](https://github.com/tenstorrent/tt-metal/pull/62388))
- Bug fix: \[quasar\] Stop batched DRAM-sharded matmul from dropping partial sums with \>1 batch per core ([PR 62463](https://github.com/tenstorrent/tt-metal/pull/62463))
- Fix matmul 1D mcast\_in1 reading wrong in0 data with height-sharded in0 and multiple output blocks ([PR 62493](https://github.com/tenstorrent/tt-metal/pull/62493))
- Binary\_ng tests: read the arch from the device, follow the hardware's rounding ([PR 62532](https://github.com/tenstorrent/tt-metal/pull/62532))
- Quasar pool2d: slice DRAM inputs with the quasar ops ([PR 62563](https://github.com/tenstorrent/tt-metal/pull/62563))
- Binary\_ng: remove the experimental descriptor factory and kernels for wh/bh ([PR 62591](https://github.com/tenstorrent/tt-metal/pull/62591))
- Slice: validate preallocated output layout, key end\_tensor in program hash (#58407 items 19, 20, 22) ([PR 62614](https://github.com/tenstorrent/tt-metal/pull/62614))
- Models: give Quasar the Blackhole device names ([PR 62714](https://github.com/tenstorrent/tt-metal/pull/62714))
- Quasar transpose: handle RM block/width-sharded tensors with sub-aligned shard rows ([PR 62905](https://github.com/tenstorrent/tt-metal/pull/62905))
- Binary\_ng: batch the native factory beside DRAM ([PR 62954](https://github.com/tenstorrent/tt-metal/pull/62954))
- Fix ImportError in test\_binary\_ng\_mx.py: use is\_quasar() ([PR 62982](https://github.com/tenstorrent/tt-metal/pull/62982))
- Minimal\_matmul (Quasar): compile out the pack-thread SwiGLU path ([PR 63007](https://github.com/tenstorrent/tt-metal/pull/63007))
- \`ttnn.slice\` uplift to Quasar ([PR 63023](https://github.com/tenstorrent/tt-metal/pull/63023))
- Route RM width/block-sharded slice with sub-aligned shard rows through L1 interleaved ([PR 63286](https://github.com/tenstorrent/tt-metal/pull/63286))
- Quasar 2D-multicast matmul: give its DM kernels the multicast NoC ([PR 63891](https://github.com/tenstorrent/tt-metal/pull/63891))

### Runtime: fast dispatch, data movement, DFB, NoC/ATT, firmware, HAL  (33)

- Re-land: Add Blocked Access Pattern To DFB ([PR 52360](https://github.com/tenstorrent/tt-metal/pull/52360))
- Quasar: bring up for tt-metal on the qsr.s1 emulation model ([PR 57580](https://github.com/tenstorrent/tt-metal/pull/57580))
- Runtime \[Feature\] :Move DFB finish and write\_barrier into the DataflowBuffer destructor ([PR 57646](https://github.com/tenstorrent/tt-metal/pull/57646))
- Phase bit optimization + Line Invalidation for Quasar fast dispatch ([PR 58907](https://github.com/tenstorrent/tt-metal/pull/58907))
- Enable RVV on Quasar tt-metal ([PR 59063](https://github.com/tenstorrent/tt-metal/pull/59063))
- Quasar NOC API v1: serialize command-buffer issue across DM cores ([PR 59519](https://github.com/tenstorrent/tt-metal/pull/59519))
- Quasar NOC API v1: lock the EXTERNAL semaphore down() ([PR 59520](https://github.com/tenstorrent/tt-metal/pull/59520))
- Enable TensorAccessor to use HW AddrGen ([PR 59773](https://github.com/tenstorrent/tt-metal/pull/59773))
- Quasar: access the dispatch telemetry control block through the uncached alias ([PR 59927](https://github.com/tenstorrent/tt-metal/pull/59927))
- Quasar: no MAX\_BYTES\_IN\_PACKET write on the simple command buffer ([PR 59963](https://github.com/tenstorrent/tt-metal/pull/59963))
- Quasar: the iDMA zero device does not set AXI\_OPT\_1.decouple\_aw ([PR 59968](https://github.com/tenstorrent/tt-metal/pull/59968))
- Quasar event test: the spin kernel reads the release word through uncached L1 ([PR 60176](https://github.com/tenstorrent/tt-metal/pull/60176))
- Quasar: fix DFB drain hangs after #57646 ([PR 60187](https://github.com/tenstorrent/tt-metal/pull/60187))
- Quasar HAL: report dfb::NUM\_DFBS dataflow buffers per core ([PR 60437](https://github.com/tenstorrent/tt-metal/pull/60437))
- Quasar DFB config: allow gaps in a core's device slots ([PR 60439](https://github.com/tenstorrent/tt-metal/pull/60439))
- Quasar NOC atomic tests: build the raw-atomic probe under NOC API v1 ([PR 60440](https://github.com/tenstorrent/tt-metal/pull/60440))
- CreateMultipleGlobalSemaphoresOnSameCore keeps its third core range on a Quasar grid ([PR 60653](https://github.com/tenstorrent/tt-metal/pull/60653))
- NOC.TensixSingleDeviceHarvestingPrints knows Quasar's unharvested grid ([PR 60654](https://github.com/tenstorrent/tt-metal/pull/60654))
- Quasar DM mailbox test: drop the Zebu debug markers ([PR 60682](https://github.com/tenstorrent/tt-metal/pull/60682))
- Quasar PrefetcherPipe: write back and invalidate the local credit lines instead of discarding them ([PR 60689](https://github.com/tenstorrent/tt-metal/pull/60689))
- Updating WaitOnEventIgnoresLaterWork to work with slower device ([PR 60758](https://github.com/tenstorrent/tt-metal/pull/60758))
- Ensure Quasar launch messages are committed before firmware reads them ([PR 60769](https://github.com/tenstorrent/tt-metal/pull/60769))
- Use hardware counters for Quasar dispatch write-completion checks ([PR 60838](https://github.com/tenstorrent/tt-metal/pull/60838))
- Ensure trisc3 doesn't use any dfb path until it is officially supported ([PR 60860](https://github.com/tenstorrent/tt-metal/pull/60860))
- Convert done/go from single state variable to a 2d array of pending go signals ([PR 60924](https://github.com/tenstorrent/tt-metal/pull/60924))
- Quasar implicit read guard tests: order the preloads, and leave the ring balanced at exit ([PR 60935](https://github.com/tenstorrent/tt-metal/pull/60935))
- Quasar NOC atomic tests: skip the CAS4 tests under NOC API v1 ([PR 61893](https://github.com/tenstorrent/tt-metal/pull/61893))
- Fix quasar cache test ([PR 61924](https://github.com/tenstorrent/tt-metal/pull/61924))
- Quasar: helpers take the kernel's DataflowBuffer by reference ([PR 62092](https://github.com/tenstorrent/tt-metal/pull/62092))
- Reorganize Fabric API Headers for Quasar ([PR 62319](https://github.com/tenstorrent/tt-metal/pull/62319))
- Quasar: checks and comments for six errata ([PR 62331](https://github.com/tenstorrent/tt-metal/pull/62331))
- Support slow dispatch on QSR3.A2 ([PR 62516](https://github.com/tenstorrent/tt-metal/pull/62516))
- Fix failing/hanging DFB tests on Quasar ([PR 62541](https://github.com/tenstorrent/tt-metal/pull/62541))

### Debug tools and profiler (watcher, NoC sanitize, perf counters, triage)  (5)

- Perf counters: Quasar on the shared tt-llk counter infrastructure ([PR 56322](https://github.com/tenstorrent/tt-metal/pull/56322))
- Quasar: watcher NoC sanitizer under ATT ([PR 57794](https://github.com/tenstorrent/tt-metal/pull/57794))
- Enabling tt-triage on simulators/emulators ([PR 57904](https://github.com/tenstorrent/tt-metal/pull/57904))
- Quasar: report watcher-test completion via DM0 FW ([PR 59344](https://github.com/tenstorrent/tt-metal/pull/59344))
- QSR: Rewind the watcher ring buffer head before its semaphore saturates ([PR 62910](https://github.com/tenstorrent/tt-metal/pull/62910))

### Trinity and Horizon variants  (2)

- Fix: disable mxfp4\_2x tests on 4-row FPU arch ([PR 60703](https://github.com/tenstorrent/tt-metal/pull/60703))
- Quasar Int4 support in LLK. ([PR 60759](https://github.com/tenstorrent/tt-metal/pull/60759))

### Emulation, simulation and CI (tt-emule, ttsim, craq-sim, emulators)  (12)

- Emule: marshal Metal 2.0 tensor binding sequences and the compile-time vararg count ([PR 58035](https://github.com/tenstorrent/tt-metal/pull/58035))
- Emule: fast dispatch and sub-devices on emulated devices ([PR 58062](https://github.com/tenstorrent/tt-metal/pull/58062))
- Emule: per-dispatch core-state descriptor, dynamic-shape bindings, move emule gtests out ([PR 58478](https://github.com/tenstorrent/tt-metal/pull/58478))
- Kernel fixes found by tt-emule: combine CB pop, group norm over-pop, ring joint SDPA guard ([PR 58910](https://github.com/tenstorrent/tt-metal/pull/58910))
- Emule: marshal Metal 2.0 PrefetcherPipe binding handles ([PR 60214](https://github.com/tenstorrent/tt-metal/pull/60214))
- Quasar: Keep tt-metal inside the DRAM the emulator models back ([PR 60444](https://github.com/tenstorrent/tt-metal/pull/60444))
- Add Quasar sim ttnn coverage to Sanity tests ([PR 61950](https://github.com/tenstorrent/tt-metal/pull/61950))
- CI setup-ttsim: run Quasar simulator on the default NoC API (V2) ([PR 62033](https://github.com/tenstorrent/tt-metal/pull/62033))
- CI: apply the ttsim skip list's quasar section to the ttnn sim\_quasar legs ([PR 62222](https://github.com/tenstorrent/tt-metal/pull/62222))
- LLK perf input filter: skip Quasar-only changes, and stop guessing which functional tests matter ([PR 62694](https://github.com/tenstorrent/tt-metal/pull/62694))
- Quasar: emulator option to address the full DRAM the soc descriptors declare ([PR 62723](https://github.com/tenstorrent/tt-metal/pull/62723))
- Updating Quasar nightly ops tests & making metal tests 2x3\_DISPATCH ([PR 63491](https://github.com/tenstorrent/tt-metal/pull/63491))

## 3. Known issues

Open issues in this release. The milestones document ([02-ai-ip-0.82.0-milestones-and-tests.md](02-ai-ip-0.82.0-milestones-and-tests.md#known-issues)) gives the impact and the requirements that each issue affects.

| Issue | Program | Ticket | Status |
| --- | --- | --- | --- |
| YOLO does not run on Trinity yet: the YOLO op suite hangs in its first conv2d case. | Trinity | RELEASE-121 | Under investigation |
| ResNet50 does not run on Trinity yet: the ResNet50 op tests hang. | Trinity | RELEASE-107 | Fix in progress |
| The ResNet50 end-to-end test gives a wrong top-1 result on Horizon and Trinity. It passes on Quasar. | Horizon, Trinity | RELEASE-120 | Fix in review |
| DM firmware: a NoC transaction ID left by an earlier kernel can make later plain NoC reads end before the data is in L1. | Trinity | RELEASE-111 | Fix in review |
| The LLK SFPU column-reduce test hangs on Trinity. The other Trinity LLK entries pass. | Trinity | RELEASE-103 | Fix in review |
| ResNet50 `test_linear` can stop on a TRISC1 hardware fault in some runs. | Quasar, Horizon | RELEASE-118 | Fixed after v0.82.0 |
| The YOLO op suite does not pass in the Quasar release pipeline yet. The owner run passes. | Quasar |  | Under investigation |

## 4. Related repositories

Versions that tt-metal v0.82.0 pins or uses.

- **tt-umd** (submodule `tt_metal/third_party/umd`): commit `7b508db` (2026-11-02). This is v0.9.15 plus 32 commits, one commit before v0.9.16. It includes the tt-umd PRs listed in section 1.
- **tt-exalens**: 0.4.1 in `tools/triage/requirements.txt` and in `tt_metal/tt-llk/tests/requirements.txt`.
- **tt-npe**: not pinned. It is released separately.
- **ttsim**: v1.15.1 (`tt_metal/ttsim-version`).
- **sfpi**: 7.93.0 (`tt_metal/sfpi-version`).
