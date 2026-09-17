<!--
AI-IP release notes for v0.77.0 + v0.78.0 (since v0.76.0), the Sep-15 AI-IP release.
Feature set: Jira epic AIIPSW-10 (Sept-15 Release) and its six child tickets.

Built with tools/gen_release_notes.py from the v0.77.0 and v0.78.0 GitHub
release bodies, then edited down to the Quasar scope.
  release tags: v0.77.0 (2026-08-18), v0.78.0 (2026-09-05, commit a3a9fb4229a)
  scope:        the 71 Quasar-related PRs of the 1,103 in the combined
                v0.77.0 + v0.78.0 changelog -- 16 against named Sep-15 features,
                55 supporting. The remaining 1,032 are not reproduced here; they
                are in the GitHub release bodies for the two tags.

Two of the six AIIPSW-10 features are not covered: AIIPSW-26 (Trinity port of
ResNet Kernel Ops) and AIIPSW-55 (Trinity runtime basic functionality). Neither
landed in v0.78.0 -- Trinity LLK support was still on a branch at the cut-off.

PR titles are reproduced as authored, with three formatting-only passes so they
render outside GitHub and cannot be read as Markdown:
  1. leading "#NNNNN:" issue prefixes removed
  2. Markdown-active characters escaped (a PR title is raw text)
  3. leading [bracket] tags removed
Some titles are truncated with an ellipsis -- that is how GitHub stores them.
-->

# Quasar Release Notes - v0.77.0 + v0.78.0

> Combined release notes for **v0.77.0 + v0.78.0**, since **v0.76.0** (the Aug-15 AI-IP release). Delivered for the **Sep-15 AI-IP release**.

## Summary

- **TTNN/Kernel Ops: Quasar ResNet Kernel Ops** - 10 PR(s). TTNN kernel ops required to run a ResNet graph on Quasar, extended this release to conv2D, pooling and linear.
- **Debug tools: Exalens advanced debugging** - 3 PR(s) in tt-metal, 5 in other repositories. Call-stack retrieval and step-by-step debugging in TT-Exalens. TT-Exalens ships as its own PyPI package; the tt-metal side of this release is the version uplift and the worker-list integration.
- **LLK: Qwen3-VL-2B related LLK features** - 2 PR(s). LLK features required by the Qwen3-VL-2B model on Quasar.
- **Debug tools: dynamic visualizer / NPE support** - 1 PR(s) in tt-metal, 3 in other repositories. Render a device from a supplied SoC descriptor rather than a baked-in one, so a licensee part can be visualised. Delivered across tt-metal, tt-npe and ttnn-visualizer.

## 1. Quasar changes

### TTNN/Kernel Ops: Quasar ResNet Kernel Ops  (10)

TTNN kernel ops required to run a ResNet graph on Quasar, extended this release to conv2D, pooling and linear.

**Convolution, pooling & reduction**

- fix experimental quasar pool scalar\_face value ([PR 54194](https://github.com/tenstorrent/tt-metal/pull/54194))
- Use up to 8 tiles per reduction and remove STALLWAIT workaround ([PR 54284](https://github.com/tenstorrent/tt-metal/pull/54284))

**Compute-API migration & cleanup**

- Pass1 Cleanup: Call compute API, drop packer SFPU\_ACTIVATION, unguard fast\_tilize ([PR 54585](https://github.com/tenstorrent/tt-metal/pull/54585))
- Pass2 Cleanup: Call compute API, unguard fast\_tilize in compute pool ([PR 54713](https://github.com/tenstorrent/tt-metal/pull/54713))
- Use compute APIs instead of llk\_ in binary and binary\_ng, call hw\_startup once, drop redundant pack reconfig ([PR 54964](https://github.com/tenstorrent/tt-metal/pull/54964))

**Model & op updates**

- update quasar resnet model and ops ([PR 52535](https://github.com/tenstorrent/tt-metal/pull/52535))
- Remove TT\_METAL\_QSR\_TILIZE\_UNPACK\_TO\_DEST ([PR 53339](https://github.com/tenstorrent/tt-metal/pull/53339))
- fix up resnet e2e run on quasar issues ([PR 53373](https://github.com/tenstorrent/tt-metal/pull/53373))

**Build & firmware fixes**

- ifdef out function calls not defined on quasar in kernel lib helper ([PR 52512](https://github.com/tenstorrent/tt-metal/pull/52512))
- Increase Quasar compute MOP timeout in TRISC firmware ([PR 53617](https://github.com/tenstorrent/tt-metal/pull/53617))

### Debug tools: Exalens advanced debugging  (3)

Call-stack retrieval and step-by-step debugging in TT-Exalens. TT-Exalens ships as its own PyPI package; the tt-metal side of this release is the version uplift and the worker-list integration.

**TT-Metal integration & version uplift**

- Update tt-exalens version to 0.3.30 ([PR 53505](https://github.com/tenstorrent/tt-metal/pull/53505))
- update tt-exalens version to 0.3.31 in LLK test harness ([PR 53514](https://github.com/tenstorrent/tt-metal/pull/53514))
- extract list of tensix functional workers from exalens ([PR 53642](https://github.com/tenstorrent/tt-metal/pull/53642))

**Delivered in other repositories** (released separately, referenced here)

- Implementing step functionality for rocket cores — `tt-exalens` ([commit 9edee97](https://github.com/tenstorrent/tt-exalens/commit/9edee9793a051fbe179f11e99f9d6b456daf1b2e))
- Adding new classes for debugging risc cores — `tt-exalens` ([commit d4de84b](https://github.com/tenstorrent/tt-exalens/commit/d4de84bab7472cb1dfa4734172d106cdb27893ea))
- Adding rocket core support — `tt-exalens` ([PR 1006](https://github.com/tenstorrent/tt-exalens/pull/1006))
- Adding read and write memory bytes to rocket core debug — `tt-exalens` ([PR 1117](https://github.com/tenstorrent/tt-exalens/pull/1117))
- Skipping is\_ebreak\_hit in callstack for rocket cores — `tt-exalens` ([PR 1126](https://github.com/tenstorrent/tt-exalens/pull/1126))

### LLK: Qwen3-VL-2B related LLK features  (2)

LLK features required by the Qwen3-VL-2B model on Quasar.

**Qwen3-VL bring-up on Quasar (may be incomplete)**

- add qwen3 copy for quasar work ([PR 54588](https://github.com/tenstorrent/tt-metal/pull/54588))
- add qwen3\_vl ops tests for quasar ([PR 54625](https://github.com/tenstorrent/tt-metal/pull/54625))

### Debug tools: dynamic visualizer / NPE support  (1)

Render a device from a supplied SoC descriptor rather than a baked-in one, so a licensee part can be visualised. Delivered across tt-metal, tt-npe and ttnn-visualizer.

**TT-Metal side**

- Dump soc descriptor in noc tracing ([PR 54976](https://github.com/tenstorrent/tt-metal/pull/54976))

**Delivered in other repositories** (released separately, referenced here)

- Add device SoC descriptor to the visualizer timeline file — `tt-npe` ([PR 133](https://github.com/tenstorrent/tt-npe/pull/133))
- Render Cluster without a baked SoC descriptor — `ttnn-visualizer` ([PR 1840](https://github.com/tenstorrent/ttnn-visualizer/pull/1840))
- Let an NPE report supply its own SoC descriptor — `ttnn-visualizer` ([PR 1955](https://github.com/tenstorrent/ttnn-visualizer/pull/1955))


## 2. Supporting Quasar work

Quasar-related changes in this release that are not tied to a named Sep-15
feature. They are part of the same delivery and are listed so every Quasar PR
in the release can be accounted for.

### Virtual device / emulation (tt-emule)  (10)

- emule: add wormhole+blackhole smoke-test script ([PR 49636](https://github.com/tenstorrent/tt-metal/pull/49636))
- emule: resolve mcast semaphore increments only to worker cores ([PR 51504](https://github.com/tenstorrent/tt-metal/pull/51504))
- emule: enable multi-dispatch socket pipelines under host-interleaved dispatch ([PR 52334](https://github.com/tenstorrent/tt-metal/pull/52334))
- emule: fabric route ordering race, $TMPDIR JIT scratch, and two triage diagnostics ([PR 52355](https://github.com/tenstorrent/tt-metal/pull/52355))
- emule: bind per-core CB config by core\_ranges membership (tt-emule-blaze#153) ([PR 52726](https://github.com/tenstorrent/tt-metal/pull/52726))
- populate per-CB face geometry at CB/DFB setup (tt-emule-blaze#191) ([PR 52742](https://github.com/tenstorrent/tt-metal/pull/52742))
- emule: resolve a fabric connection's direction per worker, not per chip ([PR 52888](https://github.com/tenstorrent/tt-metal/pull/52888))
- emule: converge the runner with the emule-blaze fork ([PR 53162](https://github.com/tenstorrent/tt-metal/pull/53162))
- emule tests modernization ([PR 53452](https://github.com/tenstorrent/tt-metal/pull/53452))
- Fix emule fabric routing and source shadows ([PR 54631](https://github.com/tenstorrent/tt-metal/pull/54631))

### Fast dispatch & dispatch engine on Quasar  (7)

- Updating experimental::quasar DM CreateKernel to skip DM0 & DM1 ([PR 52095](https://github.com/tenstorrent/tt-metal/pull/52095))
- Add Quasar fast dispatch stress tests ([PR 52104](https://github.com/tenstorrent/tt-metal/pull/52104))
- Fix Quasar unicast and iDMA VC assignments ([PR 52240](https://github.com/tenstorrent/tt-metal/pull/52240))
- Give Quasar dispatch engines their own L1 memory layout ([PR 53054](https://github.com/tenstorrent/tt-metal/pull/53054))
- Resolve Quasar dispatch core type before choosing core descriptor ([PR 53085](https://github.com/tenstorrent/tt-metal/pull/53085))
- Use NoC for Local L1 copies in FD on Quasar ([PR 53569](https://github.com/tenstorrent/tt-metal/pull/53569))
- Validate the Quasar FDS go/done wiring between dispatch engines and worker cores ([PR 54341](https://github.com/tenstorrent/tt-metal/pull/54341))

### LLK correctness, coverage & performance on Quasar  (18)

- Enable Quasar selection in LLK ttsim regression script ([PR 51649](https://github.com/tenstorrent/tt-metal/pull/51649))
- Experimental Quasar reduce\_block\_max\_row LLK kernel + metal integration. ([PR 51739](https://github.com/tenstorrent/tt-metal/pull/51739))
- Add llk-wave-debug skill for Quasar FSDB waveform diagnosis ([PR 51977](https://github.com/tenstorrent/tt-metal/pull/51977))
- Add QuaSAR per-DFB TDMA guard to detect wait/pop and reser… ([PR 52369](https://github.com/tenstorrent/tt-metal/pull/52369))
- Implement atan2 for Quasar ([PR 52449](https://github.com/tenstorrent/tt-metal/pull/52449))
- guard REDUCE\_OP parameter use in Quasar tilizeA\_B api ([PR 52510](https://github.com/tenstorrent/tt-metal/pull/52510))
- Allocate buffer descriptors during operation initialization ([PR 52762](https://github.com/tenstorrent/tt-metal/pull/52762))
- Have DFB get\_read/write\_ptr() APIs return the uncached address ranges on Quasar DM ([PR 52769](https://github.com/tenstorrent/tt-metal/pull/52769))
- Implement cumsum for Quasar ([PR 52935](https://github.com/tenstorrent/tt-metal/pull/52935))
- enable layernorm test for quasar ([PR 52950](https://github.com/tenstorrent/tt-metal/pull/52950))
- Add Quasar parallel FPU and SFPU perf coverage ([PR 53072](https://github.com/tenstorrent/tt-metal/pull/53072))
- Improve Quasar LLK perf coverage and stability ([PR 53128](https://github.com/tenstorrent/tt-metal/pull/53128))
- migrate tt-llk tests to the BFD allocator (and remove \`construct\_tdma\_desc\`) ([PR 53598](https://github.com/tenstorrent/tt-metal/pull/53598))
- Move Quasar parquet perf test into python\_tests/quasar ([PR 53812](https://github.com/tenstorrent/tt-metal/pull/53812))
- add support for perf testing on Quasar ([PR 53910](https://github.com/tenstorrent/tt-metal/pull/53910))
- Add WH/BH perf\_eltwise\_unary\_datacopy (Quasar-style shared kernel) ([PR 53942](https://github.com/tenstorrent/tt-metal/pull/53942))
- Add WH/BH perf\_pack (Quasar-style shared kernel) ([PR 53944](https://github.com/tenstorrent/tt-metal/pull/53944))
- Add Quasar ternary SFPU where performance tests ([PR 53952](https://github.com/tenstorrent/tt-metal/pull/53952))

### Watcher, DPRINT & runtime plumbing  (8)

- Cover Quasar in runtime arch branches that gate on Blackhole ([PR 49415](https://github.com/tenstorrent/tt-metal/pull/49415))
- Quick fix for DEVICE\_PRINT dispatch for Quasar ([PR 49582](https://github.com/tenstorrent/tt-metal/pull/49582))
- Use monotonic semaphores in Quasar pipeline test kernels ([PR 51601](https://github.com/tenstorrent/tt-metal/pull/51601))
- Fixing watcher to always analyze unmapped TCs ([PR 52087](https://github.com/tenstorrent/tt-metal/pull/52087))
- Changing Quasar DPRINT tests to use Metal 2.0 API ([PR 53005](https://github.com/tenstorrent/tt-metal/pull/53005))
- Cache hw\_thread\_idx in a thread\_local variable ([PR 53384](https://github.com/tenstorrent/tt-metal/pull/53384))
- Add Watcher Ringbuffer on Quasar and Blackhole ([PR 53417](https://github.com/tenstorrent/tt-metal/pull/53417))
- Update tt-2xx get\_timestamp and get\_timestamp\_32b ([PR 53714](https://github.com/tenstorrent/tt-metal/pull/53714))

### Grid bring-up & multi-node  (4)

- Add Quasar 8x4 full-grid integration tests ([PR 51620](https://github.com/tenstorrent/tt-metal/pull/51620))
- Add multi-node tests for Quasar ([PR 53539](https://github.com/tenstorrent/tt-metal/pull/53539))
- Bring up \`tt-metal\` on the Quasar 8x4 grid in \`craq-sim\` ([PR 54374](https://github.com/tenstorrent/tt-metal/pull/54374))
- Revert "\[Feature\] Bring up \`tt-metal\` on the Quasar 8x4 grid in \`craq… ([PR 54773](https://github.com/tenstorrent/tt-metal/pull/54773))

### Early model bring-up on Quasar (Llama, GPT-OSS ops tests)  (4)

- create prototype copy of llama quasar ops tests ([PR 52663](https://github.com/tenstorrent/tt-metal/pull/52663))
- add graph tracing based llama quasar ops pytests ([PR 54428](https://github.com/tenstorrent/tt-metal/pull/54428))
- create copy of gpt-oss for quasar testing ([PR 54653](https://github.com/tenstorrent/tt-metal/pull/54653))
- add ops tests for quasar testing based on a gpt oss graph trace ([PR 54755](https://github.com/tenstorrent/tt-metal/pull/54755))

### CI & build  (4)

- trigger quasar build on fuser changes ([PR 52069](https://github.com/tenstorrent/tt-metal/pull/52069))
- Update Quasar program versions in documentation ([PR 53195](https://github.com/tenstorrent/tt-metal/pull/53195))
- add quasar to llk ttsim weekly workflow ([PR 53674](https://github.com/tenstorrent/tt-metal/pull/53674))
- fix quasar compile tests action ([PR 55073](https://github.com/tenstorrent/tt-metal/pull/55073))
