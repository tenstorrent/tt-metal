# Quasar, Trinity and Horizon Release Notes - v0.79.0 + v0.80.0

> Combined release notes for **v0.79.0 + v0.80.0**, since **v0.78.0** (the Sep-15 AI-IP release). Delivered for the **Oct-15 AI-IP release**.

## Summary

- **TTNN/Kernel Ops: Quasar Llama related Kernel Ops** - 7 PR(s). TTNN kernel ops required to run the Llama 3.2 1B model on Quasar.
- **LLK: Trinity port LLK from Quasar to Trinity** - 2 PR(s). LLK ported from Quasar to Trinity.
- **Horizon: ATT / flat address UA, UB** - 2 PR(s) in tt-metal, 1 in other repositories. ATT and flat-address buffer support for the Horizon UA and UB builds.
- **Horizon: Overlay registers** - 1 PR(s). Overlay register support for Horizon.
- **Supporting work** - 198 PR(s) in 6 areas. See section 2.

## 1. Quasar, Trinity and Horizon changes

### TTNN/Kernel Ops: Quasar Llama related Kernel Ops  (7)

TTNN kernel ops required to run the Llama 3.2 1B model on Quasar.

- Port \`transformer/sdpa\_decode\` ([PR 54249](https://github.com/tenstorrent/tt-metal/pull/54249)) -- The PR description was not updated when the scope changed: the merged change adds a Quasar copy of the op under `experimental/quasar/transformer/sdpa_decode/` and does not change the Wormhole/Blackhole op. It switches the Llama-on-Quasar model and its SDPA-decode graph_ops tests to that Quasar copy.
- Port \`transformer/sdpa\` (SDPA + JointSDPA factories) ([PR 54468](https://github.com/tenstorrent/tt-metal/pull/54468))
- Quasar: bring up sdpa\_decode on craq-sim + emulator ([PR 57038](https://github.com/tenstorrent/tt-metal/pull/57038))
- Update Quasar Llama tests ([PR 58213](https://github.com/tenstorrent/tt-metal/pull/58213))
- Fix up Llama Quasar issues ([PR 58490](https://github.com/tenstorrent/tt-metal/pull/58490))
- Matmul DRAM-sharded: Quasar uplift of the Metal 2.0 factory ([PR 58912](https://github.com/tenstorrent/tt-metal/pull/58912))
- Quasar Llama test updates ([PR 58971](https://github.com/tenstorrent/tt-metal/pull/58971))

### LLK: Trinity port LLK from Quasar to Trinity  (2)

#### LLK ported from Quasar to Trinity.

- Round the direct-indexed eltwise replay length up, not down ([PR 55667](https://github.com/tenstorrent/tt-metal/pull/55667)).-- This fix also covers the 4-row math configuration (`ELTWISE_MATH_ROWS` = 4) used for the Trinity port.
- 4 row math architecture (MATH\_ROWS\=4) support ([PR 56606](https://github.com/tenstorrent/tt-metal/pull/56606))

#### Known at the v0.80.0 tag:

- Trinity LLK is selected with `QUASAR_ARCH_VARIANT=quasar_4row`.
- The folder `tt_metal/tt-llk/tt_llk_quasar/arch/quasar_4row` is present at v0.80.0.

### Horizon: ATT / flat address UA, UB  (2)

#### ATT and flat-address buffer support for the Horizon UA and UB builds.

- Let TT\_VISIBLE\_DEVICES pick the device of a partitioned simulator build ([PR 58808](https://github.com/tenstorrent/tt-metal/pull/58808))
- Add the horizon\_2x3 NOC ATT map (TT\_METAL\_NOC\_ATT\=horizon\_2x3) ([PR 58809](https://github.com/tenstorrent/tt-metal/pull/58809))

#### Delivered in other repositories (released separately, referenced here)

- Read and write any bytes in flat-address mode over whole-word simulator accesses (top of a stack of 17 PRs) — `tt-umd` ([PR 3566](https://github.com/tenstorrent/tt-umd/pull/3566))

### Horizon: Overlay registers  (1)

Overlay register support for Horizon.

- Ayaacob/add Horizon headers ([PR 57976](https://github.com/tenstorrent/tt-metal/pull/57976))


## 2. Supporting Quasar, Trinity and Horizon work

Quasar, Trinity and Horizon changes in this release that are not tied to a
named Oct-15 feature. They are part of the same delivery and are listed so
every tt-2xx PR in the release can be accounted for.

### LLK (`tt_llk_quasar`, Quasar ckernels and compute API, LLK tests)  (94)

- Create an API wrapper for Unary SFPU LLK kernels + SrcS indices ([PR 48955](https://github.com/tenstorrent/tt-metal/pull/48955))
- Quasar exp: restore the fp32 overflow/underflow guards ([PR 51544](https://github.com/tenstorrent/tt-metal/pull/51544))
- Stable sort and topk ([PR 53557](https://github.com/tenstorrent/tt-metal/pull/53557))
- Add Quasar SFPU reduce kernel and tests ([PR 54437](https://github.com/tenstorrent/tt-metal/pull/54437))
- Fix(SFPU): invert Quasar softplus boundary predicate to match Wormhole/Blackhole ([PR 54858](https://github.com/tenstorrent/tt-metal/pull/54858))
- Fix operand-derived binary TensorShape handling ([PR 54946](https://github.com/tenstorrent/tt-metal/pull/54946))
- Matmul tiny tiles ([PR 54989](https://github.com/tenstorrent/tt-metal/pull/54989))
- Reduce register pressure for clreg -1.0f loss ([PR 55097](https://github.com/tenstorrent/tt-metal/pull/55097))
- Add dummy\_pack + fix TEN-4746 pack/unpack ordering in tests ([PR 55426](https://github.com/tenstorrent/tt-metal/pull/55426))
- BFD allocation and index tracking ([PR 55437](https://github.com/tenstorrent/tt-metal/pull/55437))
- Add full\_ct\_dim divisible by block\_ct\_dim untilize asserts ([PR 55454](https://github.com/tenstorrent/tt-metal/pull/55454))
- Remove unnecessary stallwaits around edge mask config ([PR 55476](https://github.com/tenstorrent/tt-metal/pull/55476))
- Declare unimplemented Quasar p\_stall wait resources and STALL\_XMOV as non-integers ([PR 55686](https://github.com/tenstorrent/tt-metal/pull/55686))
- Fix Quasar fused INIT collisions and pack/datacopy perf CSV keys ([PR 55772](https://github.com/tenstorrent/tt-metal/pull/55772))
- Tests: golden fixes for math fidelity phases and Dest flush to zero ([PR 55779](https://github.com/tenstorrent/tt-metal/pull/55779))
- Tests: read buffer C from the Tensix the test ran on ([PR 55781](https://github.com/tenstorrent/tt-metal/pull/55781))
- Align eltwise binary functional and performance tests across BH and Quasar ([PR 55921](https://github.com/tenstorrent/tt-metal/pull/55921))
- Bug fix: block the RISC in llk\_wait\_tiles until WAIT\_TILES resolves so wait\_front() returns with tiles resident ([PR 55938](https://github.com/tenstorrent/tt-metal/pull/55938))
- Drain the math pipeline in the Dest-to-source move guards ([PR 55966](https://github.com/tenstorrent/tt-metal/pull/55966))
- Re-initalization guard ([PR 56082](https://github.com/tenstorrent/tt-metal/pull/56082))
- Fixing llk\_pack\_dummy ([PR 56145](https://github.com/tenstorrent/tt-metal/pull/56145))
- Feature Quasar - Implemented F32 \<-\> Uint16 Typecast ([PR 56152](https://github.com/tenstorrent/tt-metal/pull/56152))
- Fix(LLK): scale strided-pack per-tile index by stride\_size\_tiles on Quasar ([PR 56195](https://github.com/tenstorrent/tt-metal/pull/56195))
- Name the SFPSTORE format field in the SFPU relu kernel ([PR 56211](https://github.com/tenstorrent/tt-metal/pull/56211))
- SrcA transpose support for Matmul ([PR 56285](https://github.com/tenstorrent/tt-metal/pull/56285))
- Emit the semaphore pre-stall for every wait-resource slot ([PR 56328](https://github.com/tenstorrent/tt-metal/pull/56328))
- Feat: add compute api for Quasar relu\_min, relu\_max ([PR 56489](https://github.com/tenstorrent/tt-metal/pull/56489))
- Fix: refactor Quasar softplus to avoid register spill issue ([PR 56516](https://github.com/tenstorrent/tt-metal/pull/56516))
- Feat: add SFPU rounding ops for Quasar ([PR 56540](https://github.com/tenstorrent/tt-metal/pull/56540))
- Write LOADMACRO CONTROL from SFPCONFIG's own immediate ([PR 56546](https://github.com/tenstorrent/tt-metal/pull/56546))
- Tests: underflow on the Src exponents, not the product ([PR 56553](https://github.com/tenstorrent/tt-metal/pull/56553))
- Fix the x/x shortcut in the SFPU divide ([PR 56579](https://github.com/tenstorrent/tt-metal/pull/56579))
- Add DISABLE\_SFPLOADMACRO Quasar exp kernel ([PR 56589](https://github.com/tenstorrent/tt-metal/pull/56589))
- Tests: mask reuse-dest operands in the Src registers' format ([PR 56590](https://github.com/tenstorrent/tt-metal/pull/56590))
- Import TestConfig in the Quasar parallel-matmul exp test ([PR 56591](https://github.com/tenstorrent/tt-metal/pull/56591))
- Add Quasar SFPU accuracy plotting harness ([PR 56618](https://github.com/tenstorrent/tt-metal/pull/56618))
- Make Float16\_b to Float\_32 typecast an SFPU no-op ([PR 56662](https://github.com/tenstorrent/tt-metal/pull/56662))
- Fix: improve Quasar's SFPU log accuracy ([PR 56837](https://github.com/tenstorrent/tt-metal/pull/56837))
- Fix tiny-tile eltwise binary ([PR 56846](https://github.com/tenstorrent/tt-metal/pull/56846))
- Feat: add copy\_dest\_values Quasar SFPU kernel ([PR 56905](https://github.com/tenstorrent/tt-metal/pull/56905))
- Implemented INT32 -\> FP16b Typecast ([PR 56911](https://github.com/tenstorrent/tt-metal/pull/56911))
- Fix reduce scalar for 32bit dest ([PR 56971](https://github.com/tenstorrent/tt-metal/pull/56971))
- Add Unary Broadcast support for 32bit Dest mode, unpack to Source registers combinations ([PR 57005](https://github.com/tenstorrent/tt-metal/pull/57005))
- Make Fp32 -\> Fp16b typecast an SFPU no-op ([PR 57019](https://github.com/tenstorrent/tt-metal/pull/57019))
- Quasar: rename calculate\_reduce NTTP off the REDUCE\_DIM macro ([PR 57079](https://github.com/tenstorrent/tt-metal/pull/57079))
- Align BH/QSR eltwise binary SFPU functional and perf tests ([PR 57137](https://github.com/tenstorrent/tt-metal/pull/57137))
- Fix reduce scalar scratch overlap for 16x16 tiny tiles ([PR 57187](https://github.com/tenstorrent/tt-metal/pull/57187))
- Remove unpack to dest math hw config ([PR 57203](https://github.com/tenstorrent/tt-metal/pull/57203))
- Refactor math hw configure ([PR 57226](https://github.com/tenstorrent/tt-metal/pull/57226))
- Fix HiFi ELWMUL overwrite when acc\_to\_dest is false ([PR 57344](https://github.com/tenstorrent/tt-metal/pull/57344))
- Fix: remove superfluous DataFormat combinations in Quasar LLK tests ([PR 57428](https://github.com/tenstorrent/tt-metal/pull/57428))
- Clear pack/pack\_matmul FACE/ROW\_SEL state at pack/pack\_matmul\_init ([PR 57530](https://github.com/tenstorrent/tt-metal/pull/57530))
- Test fewer input dimensions and test bank switching for unpack unary, pack, binary bcast and datacopy ([PR 57544](https://github.com/tenstorrent/tt-metal/pull/57544))
- Fix: collapse Quasar datacopy MX formats to one smoke pair ([PR 57548](https://github.com/tenstorrent/tt-metal/pull/57548))
- Use new sfpi::DataLayout enums ([PR 57562](https://github.com/tenstorrent/tt-metal/pull/57562))
- Test fewer input dimensions and test bank switching for eltwise binary ([PR 57669](https://github.com/tenstorrent/tt-metal/pull/57669))
- Add MxFp8 to the pack and unpack\_unary tests ([PR 57730](https://github.com/tenstorrent/tt-metal/pull/57730))
- Fix: do not use high fidelities for fp16b data in LLK Quasar tests ([PR 57774](https://github.com/tenstorrent/tt-metal/pull/57774))
- Fix: cap MX and Float16b math fidelity at LoFi ([PR 57880](https://github.com/tenstorrent/tt-metal/pull/57880))
- Fix eltwise binary dest index for tiny tiles ([PR 57928](https://github.com/tenstorrent/tt-metal/pull/57928))
- Remove redundant ZEROACC from eltwise binary reuse dest ([PR 58108](https://github.com/tenstorrent/tt-metal/pull/58108))
- Reduce kernel .text section size by marking \`ckernel\_template\` functions inline ([PR 58112](https://github.com/tenstorrent/tt-metal/pull/58112))
- Quasar unary broadcast: set SrcA dvalid before the SrcB faces ([PR 58115](https://github.com/tenstorrent/tt-metal/pull/58115))
- Add Quasar unpack tilize tests for Fp32, UInt8 and Int8 ([PR 58140](https://github.com/tenstorrent/tt-metal/pull/58140))
- Add quasar-test-coverage skill for choosing functional-test sweep axes ([PR 58158](https://github.com/tenstorrent/tt-metal/pull/58158))
- Quasar matmul strided Dest: drop the counter-to-CR SETRWC ([PR 58191](https://github.com/tenstorrent/tt-metal/pull/58191))
- Port Blackhole SFPI SFPU kernels with fuser coverage ([PR 58209](https://github.com/tenstorrent/tt-metal/pull/58209))
- Enable RVV for Quasar in tt-llk ([PR 58236](https://github.com/tenstorrent/tt-metal/pull/58236))
- Do not allow 2x data format when using GMPOOL ([PR 58451](https://github.com/tenstorrent/tt-metal/pull/58451))
- Feat: wire Quasar cumsum to Compute API ([PR 58537](https://github.com/tenstorrent/tt-metal/pull/58537))
- Implement max\_pool\_indices for Quasar ([PR 58538](https://github.com/tenstorrent/tt-metal/pull/58538))
- Plumb the ported SFPI kernels through the compute API ([PR 58555](https://github.com/tenstorrent/tt-metal/pull/58555))
- Implement signbit for Quasar ([PR 58561](https://github.com/tenstorrent/tt-metal/pull/58561))
- Reduce BFD initialization code size ([PR 58574](https://github.com/tenstorrent/tt-metal/pull/58574))
- Add matmul support for 32x16 and 16x32 input tiles ([PR 58595](https://github.com/tenstorrent/tt-metal/pull/58595))
- Test only: set Quasar Src dvalids with UNPACR\_NOP, not SETDVALID ([PR 58604](https://github.com/tenstorrent/tt-metal/pull/58604))
- Tests: add the matmul operand face dims to the Quasar perf schema ([PR 58641](https://github.com/tenstorrent/tt-metal/pull/58641))
- Quasar unary broadcast test: balance SrcA in the perf isolate loops ([PR 58870](https://github.com/tenstorrent/tt-metal/pull/58870))
- LLK: Quasar pack test: Int16 takes no ReLU ([PR 58876](https://github.com/tenstorrent/tt-metal/pull/58876))
- Quasar typecast to UInt16: leave CC\_en on, re-widen with SFPENCC(0, 0) ([PR 58880](https://github.com/tenstorrent/tt-metal/pull/58880))
- Tests: let the Quasar unpack-reconfig test reach its comparison ([PR 58881](https://github.com/tenstorrent/tt-metal/pull/58881))
- Tests: read the Quasar unary broadcast input through its tensor accessor ([PR 58882](https://github.com/tenstorrent/tt-metal/pull/58882))
- Quasar unpack-to-Dest: hold UNPACK\_MATH until the tile is committed ([PR 58892](https://github.com/tenstorrent/tt-metal/pull/58892))
- LLK: Quasar p\_unpacr names UNPACR\_NOP's types by Quasar's encoding ([PR 58895](https://github.com/tenstorrent/tt-metal/pull/58895))
- Add output tolerances and fix Quasar SigmoidAppx test failures ([PR 58966](https://github.com/tenstorrent/tt-metal/pull/58966))
- Skip TensixComputePackUntilizeDstTinyTile on Quasar ([PR 59011](https://github.com/tenstorrent/tt-metal/pull/59011))
- Improve SFPU log accuracy, use fp32 ln2 in \_calculate\_log\_body\_no\_init\_ ([PR 59014](https://github.com/tenstorrent/tt-metal/pull/59014))
- Quasar integer-average reduce drops its SFPENCC(1, 2) ([PR 59036](https://github.com/tenstorrent/tt-metal/pull/59036))
- Port and plumb the logaddexp, logaddexp2, shift, abs and fill SFPI kernels ([PR 59267](https://github.com/tenstorrent/tt-metal/pull/59267))
- Quasar \_configure\_alu\_formats\_ sets the ALU Src formats under implied math format too ([PR 59314](https://github.com/tenstorrent/tt-metal/pull/59314))
- Quasar MxFp4\_2x tests configure the ALU with the 2x register fo… ([PR 59315](https://github.com/tenstorrent/tt-metal/pull/59315))
- Fix: correct issues with Quasar trigonometry functions ([PR 59327](https://github.com/tenstorrent/tt-metal/pull/59327))
- Fix: let Quasar HiFi kernels call any\_tiles\_bcast with add/sub ([PR 59459](https://github.com/tenstorrent/tt-metal/pull/59459))
- Fix potential false scoreboard hazard on Quasar SFPMUL24 ([PR 59536](https://github.com/tenstorrent/tt-metal/pull/59536))

### Kernel Ops (TT-NN `experimental/quasar` ops, ResNet, SDPA, matmul, `binary_ng`)  (50)

- Replace TEN-4746 copy\_tile drain with dummy\_unpack() helper ([PR 54948](https://github.com/tenstorrent/tt-metal/pull/54948))
- binary\_ng: Quasar-native multi-DM / multi-Tensix support and typical case perf measurement ([PR 55000](https://github.com/tenstorrent/tt-metal/pull/55000))
- Quasar Pool: Use All 4 NEOs Per Cluster (num\_threads\=4) ([PR 55001](https://github.com/tenstorrent/tt-metal/pull/55001))
- Complete Metal 2.0 CB-\>DFB port of Quasar transformer/sdpa kernel helpers ([PR 55219](https://github.com/tenstorrent/tt-metal/pull/55219))
- Metal 2.0: finish sdpa\_decode quasar-fork port (CB-\>DFB) ([PR 55254](https://github.com/tenstorrent/tt-metal/pull/55254))
- Clean up fused conv compute kernel and use cfg register to tilize/untilize more than one row of tiles ([PR 55363](https://github.com/tenstorrent/tt-metal/pull/55363))
- Adjust experimental Quasar ops and ResNet model to use less SRAM and make block sharded conv pass ([PR 55399](https://github.com/tenstorrent/tt-metal/pull/55399))
- BMM kernels: remove inline ISA, gate unsupported SFPU off on Quasar ([PR 55467](https://github.com/tenstorrent/tt-metal/pull/55467))
- Remove llk api hw config calls for BH in binary\_ng/ and experimental/quasar/binary\_ng ([PR 55474](https://github.com/tenstorrent/tt-metal/pull/55474))
- Honour memory\_config in Quasar outer ([PR 55613](https://github.com/tenstorrent/tt-metal/pull/55613))
- Close the gap with experimental/quasar in binary and binary\_ng kernels ([PR 55645](https://github.com/tenstorrent/tt-metal/pull/55645))
- Fix Quasar SDPA with fp32 dest accumulation ([PR 55734](https://github.com/tenstorrent/tt-metal/pull/55734))
- Remove stale tests and compute kernels ([PR 55924](https://github.com/tenstorrent/tt-metal/pull/55924))
- Quasar SDPA: convert page\_table DM self-loop DFB to a scratchpad ([PR 55978](https://github.com/tenstorrent/tt-metal/pull/55978))
- Skip ResNet Quasar tests that are not supported yet ([PR 56005](https://github.com/tenstorrent/tt-metal/pull/56005))
- Quasar binary\_ng: remove ARCH\_QUASAR copy\_init ifdefs from the SFPU DFB kernels ([PR 56101](https://github.com/tenstorrent/tt-metal/pull/56101))
- Fix statically audited bugs in experimental Quasar conv2d ([PR 56141](https://github.com/tenstorrent/tt-metal/pull/56141))
- Quasar reshard: dest stride and split scratch on unaligned same-width ([PR 56450](https://github.com/tenstorrent/tt-metal/pull/56450))
- Quasar slice: stage tensor-args indices in a scratchpad ([PR 56458](https://github.com/tenstorrent/tt-metal/pull/56458))
- Quasar tilize/untilize: skip DRAM-sharded tilize zero-copy and size HS unpad shards ([PR 56481](https://github.com/tenstorrent/tt-metal/pull/56481))
- Quasar matmul: hoist fused bias out of the batched DRAM-sharded loop ([PR 56597](https://github.com/tenstorrent/tt-metal/pull/56597))
- Quasar matmul: do not deref end() on gather\_in0 DRAM bank miss ([PR 56624](https://github.com/tenstorrent/tt-metal/pull/56624))
- Quasar prefill SDPA: functional on craq-sim + emulator ([PR 56795](https://github.com/tenstorrent/tt-metal/pull/56795))
- Quasar Matmul ([PR 56813](https://github.com/tenstorrent/tt-metal/pull/56813))
- Fix moreh/moreh\_mean unpack\_modes skipped on Quasar (post-port semantic pass) ([PR 56829](https://github.com/tenstorrent/tt-metal/pull/56829))
- Stop relying on the default device: Quasar outer takes the device from its operands, validation-tools test no longer installs one (T3K common unit tests segfault) ([PR 56968](https://github.com/tenstorrent/tt-metal/pull/56968))
- binary\_ng: uneven tile counts and dataflow batching at any (R,C,W) on the native factory ([PR 57065](https://github.com/tenstorrent/tt-metal/pull/57065))
- Add local copy and inits for layernorm on Quasar ([PR 57202](https://github.com/tenstorrent/tt-metal/pull/57202))
- Update rotary\_embedding\_llama for Quasar support ([PR 57210](https://github.com/tenstorrent/tt-metal/pull/57210))
- Add Quasar hardware configs for tilize and sharded data-movement operations ([PR 57214](https://github.com/tenstorrent/tt-metal/pull/57214))
- Adjust noc async read on nlp\_concat\_heads for Quasar ([PR 57216](https://github.com/tenstorrent/tt-metal/pull/57216))
- Enable experimental paged\_update\_cache and paged\_fill\_cache on Quasar ([PR 57229](https://github.com/tenstorrent/tt-metal/pull/57229))
- ifndef code in eltwise kernel\_lib not yet supported on Quasar ([PR 57253](https://github.com/tenstorrent/tt-metal/pull/57253))
- Add Quasar support to rotary\_embedding\_indexed ([PR 57275](https://github.com/tenstorrent/tt-metal/pull/57275))
- Enable embedding scratchpads on Quasar ([PR 57558](https://github.com/tenstorrent/tt-metal/pull/57558))
- tilize/tilize\_with\_val\_padding: Quasar DM + coherence fixes ([PR 57616](https://github.com/tenstorrent/tt-metal/pull/57616))
- Quasar matmul: stop aliasing in0 as the inert sparsity tensor ([PR 57886](https://github.com/tenstorrent/tt-metal/pull/57886))
- Quasar matmul: read the in0 receiver semaphore through value() ([PR 57898](https://github.com/tenstorrent/tt-metal/pull/57898))
- Data movement Quasar adjustments and disable non-ported (e.g. codegen) variations ([PR 58195](https://github.com/tenstorrent/tt-metal/pull/58195))
- Fix Quasar synchronization in rotary embedding and QKV-head decode ([PR 58196](https://github.com/tenstorrent/tt-metal/pull/58196))
- Quasar specific fixes for matmul ([PR 58197](https://github.com/tenstorrent/tt-metal/pull/58197))
- Experimental Quasar tilize/sdpa/sdpa\_decode changes ([PR 58201](https://github.com/tenstorrent/tt-metal/pull/58201))
- binary\_ng: borrow L1-sharded operands on the native factory ([PR 58221](https://github.com/tenstorrent/tt-metal/pull/58221))
- Quasar Matmul 2 (Multi-NEO) ([PR 58470](https://github.com/tenstorrent/tt-metal/pull/58470))
- Fix remainder/outer and binary\_ng high-rank bugs (#57362 items 1-4) ([PR 58544](https://github.com/tenstorrent/tt-metal/pull/58544))
- Fix Quasar matmul, layer norm, and ArgMax execution ([PR 58967](https://github.com/tenstorrent/tt-metal/pull/58967))
- Reject unsupported Quasar typecasts and wide FP32 tilize ([PR 58970](https://github.com/tenstorrent/tt-metal/pull/58970))
- Quasar SDPA: attention sink and sliding window for GPT-OSS and Qwen3-VL ([PR 59015](https://github.com/tenstorrent/tt-metal/pull/59015))
- Restrict Quasar kernel GLOB\_RECURSE to .cpp/.hpp ([PR 59233](https://github.com/tenstorrent/tt-metal/pull/59233))
- Fix Quasar matmul auto-config fuse\_batch for sharded output with batched A ([PR 59241](https://github.com/tenstorrent/tt-metal/pull/59241))

### Runtime: fast dispatch, data movement, DFB, NoC/ATT, firmware, HAL  (40)

- Hide Noc zeroing APIs on Quasar V1 ([PR 51597](https://github.com/tenstorrent/tt-metal/pull/51597))
- Quasar single-device tests modernization ([PR 52770](https://github.com/tenstorrent/tt-metal/pull/52770))
- Quasar: host-selected semaphore mechanisms ([PR 53687](https://github.com/tenstorrent/tt-metal/pull/53687))
- Initialize Quasar DM NoC command buffers at boot ([PR 54172](https://github.com/tenstorrent/tt-metal/pull/54172))
- Quasar NOC non-blocking API V3: the ATT-native transport ([PR 54415](https://github.com/tenstorrent/tt-metal/pull/54415))
- Update headers files to use those generated from RDL ([PR 54512](https://github.com/tenstorrent/tt-metal/pull/54512))
- Quasar ATT maps: window math, product map configurations, typed resolution, bring-up replay ([PR 54566](https://github.com/tenstorrent/tt-metal/pull/54566))
- Enable the ATT address backend for Quasar ([PR 54576](https://github.com/tenstorrent/tt-metal/pull/54576))
- Make Quasar compute sync\_threads() barrier all TRISCs using HW semaphores and add sync\_all\_cores(). ([PR 54628](https://github.com/tenstorrent/tt-metal/pull/54628))
- Fix idle-eth UNRESERVED/KERNEL\_CONFIG overlap on Blackhole/Quasar ([PR 54963](https://github.com/tenstorrent/tt-metal/pull/54963))
- Support zeroing APIs under V1 NOC APIs ([PR 55103](https://github.com/tenstorrent/tt-metal/pull/55103))
- Use cached L1 on FD command path on Quasar ([PR 55122](https://github.com/tenstorrent/tt-metal/pull/55122))
- Bring up \`tt-metal\` on the Quasar 8x4 grid in \`craq-sim\` ([PR 55384](https://github.com/tenstorrent/tt-metal/pull/55384))
- Add cache flush/invalidate in dm common.h for Quasar ([PR 55498](https://github.com/tenstorrent/tt-metal/pull/55498))
- Increase dispatch engine DM stack space ([PR 55513](https://github.com/tenstorrent/tt-metal/pull/55513))
- Support the cache ops in DFB scoped read/write locks for CoreLocalMem/Scratchpad/LocalTensorAccessor ([PR 55570](https://github.com/tenstorrent/tt-metal/pull/55570))
- Fix leftover old constant name after Quasar NOC virtual channel rename ([PR 55845](https://github.com/tenstorrent/tt-metal/pull/55845))
- Add Quasar FDS interrupt tests ([PR 55956](https://github.com/tenstorrent/tt-metal/pull/55956))
- Removed stale/duplicate instruction macros from Quasar tensix.h ([PR 55999](https://github.com/tenstorrent/tt-metal/pull/55999))
- Quasar: keep DM local region size a multiple of 2 KB ([PR 56009](https://github.com/tenstorrent/tt-metal/pull/56009))
- Add initial support for PrefetcherPipe on Quasar (single threaded) ([PR 56018](https://github.com/tenstorrent/tt-metal/pull/56018))
- Reduce local dfb footprint on pack by creating a global shared TC pool for all packer dfbs to allow increasing MAX\_ACTIVE\_DFBS\_PACK ([PR 56539](https://github.com/tenstorrent/tt-metal/pull/56539))
- Adding support for multi-threaded PrefetcherPipe on Quasar ([PR 56584](https://github.com/tenstorrent/tt-metal/pull/56584))
- Fix the Quasar V2 inline dword write register recipe ([PR 56704](https://github.com/tenstorrent/tt-metal/pull/56704))
- Use FDS for go/done signalling on Quasar ([PR 56800](https://github.com/tenstorrent/tt-metal/pull/56800))
- Uplift dispatch unit tests to Metal 2.0, fix mislabeled data movement skips ([PR 56830](https://github.com/tenstorrent/tt-metal/pull/56830))
- Use cached upstream credit semaphores on Quasar FD ([PR 56888](https://github.com/tenstorrent/tt-metal/pull/56888))
- Quasar: fast dispatch under ATT ([PR 56912](https://github.com/tenstorrent/tt-metal/pull/56912))
- Fix NOC\_V3\_STATE\_CMD\_BUFS  on Quasar ([PR 56957](https://github.com/tenstorrent/tt-metal/pull/56957))
- Adjust Quasar cache behaviour in data movement common header ([PR 57036](https://github.com/tenstorrent/tt-metal/pull/57036))
- Use Metal 2.0 in tests ([PR 57228](https://github.com/tenstorrent/tt-metal/pull/57228))
- Fix implicit sync check that entry available in the dfb has not previously been claimed by the kernel ([PR 57296](https://github.com/tenstorrent/tt-metal/pull/57296))
- Enable dfb.get\_tile\_address/dfb.read\_tile\_value on Quasar ([PR 57411](https://github.com/tenstorrent/tt-metal/pull/57411))
- Overlay cmdbuf/addrgen API: builtins, template ids, argument cleanup ([PR 57602](https://github.com/tenstorrent/tt-metal/pull/57602))
- Don't have dm0/1 pull dfb config info if nothing will run on that node ([PR 57980](https://github.com/tenstorrent/tt-metal/pull/57980))
- Add default dram-to-worker assignment for Quasar ([PR 58474](https://github.com/tenstorrent/tt-metal/pull/58474))
- Update Quasar l1 banks to be row major order ([PR 58644](https://github.com/tenstorrent/tt-metal/pull/58644))
- Stop dispatch DM firmware from accessing the DFB remapper ([PR 58920](https://github.com/tenstorrent/tt-metal/pull/58920))
- Replay each trace with the DFB config from when it was captured ([PR 59386](https://github.com/tenstorrent/tt-metal/pull/59386))
- Quasar: access the dispatch telemetry control block through… ([PR 60421](https://github.com/tenstorrent/tt-metal/pull/60421))

### Debug tools and profiler (DPRINT, NoC sanitize, NoC tracing, RT profiler)  (6)

- Update device print buffer handling to reset positions during startup ([PR 54852](https://github.com/tenstorrent/tt-metal/pull/54852))
- Add dprint\_tensix DEST-register read + dbg\_halt rendezvous ([PR 56580](https://github.com/tenstorrent/tt-metal/pull/56580))
- Quasar\] dprint\_tensix DEST print: Int32, Int8, Float16 and MX block formats ([PR 57588](https://github.com/tenstorrent/tt-metal/pull/57588))
- Bring up NoC event tracing on Quasar ([PR 58232](https://github.com/tenstorrent/tt-metal/pull/58232))
- Debounce NoC Sanitize records on Quasar ([PR 58680](https://github.com/tenstorrent/tt-metal/pull/58680))
- RT profiler: keep the dispatch\_s wait timer out of Quasar builds ([PR 59856](https://github.com/tenstorrent/tt-metal/pull/59856))

### Trinity and Horizon variants  (2)

- Add header files that describe Trinity regitsters ([PR 58640](https://github.com/tenstorrent/tt-metal/pull/58640))
- Disable FDS use on Trinity ([PR 59057](https://github.com/tenstorrent/tt-metal/pull/59057))

### Emulation, simulation and CI (tt-emule, ttsim, craq-sim, evidence)  (6)

- CI: account tt-umd-horizon results as AIIPSW-15 test evidence (MINFRA-1623) ([PR 53855](https://github.com/tenstorrent/tt-metal/pull/53855))
- Update Quasar regression test list ([PR 54777](https://github.com/tenstorrent/tt-metal/pull/54777))
- CI: add ttsim's Quasar simulator unit-test list to the sanity pipeline ([PR 54793](https://github.com/tenstorrent/tt-metal/pull/54793))
- Add Quasar ttsim tests to the merge gate ([PR 55385](https://github.com/tenstorrent/tt-metal/pull/55385))
- Fidelity defines, Quasar phases, fiber scheduler fairness ([PR 56708](https://github.com/tenstorrent/tt-metal/pull/56708))
- Read Quasar and Horizon emulator evidence from the tt-umd-simulators release checks ([PR 59215](https://github.com/tenstorrent/tt-metal/pull/59215))

## 3. Related repositories

Versions that tt-metal v0.80.0 pins or uses.

- **tt-umd** (submodule `tt_metal/third_party/umd`): commit `5437fe3` (2026-10-02). This is v0.9.12 plus 31 commits. It has no release tag. It includes tt-umd PR 3566.
- **tt-exalens**: 0.3.32 in `tools/triage/requirements.txt`; 0.3.31 in `tt_metal/tt-llk/tests/requirements.txt`. tt-exalens 0.4.x is not pinned.
- **tt-npe**: not pinned. It has no release.
- **ttsim**: v1.11.1 (`tt_metal/ttsim-version`).
- **sfpi**: 7.85.0 (`tt_metal/sfpi-version`).
