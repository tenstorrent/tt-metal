# Decode convolution, activation, and carry fusion

This isolated candidate replaces the current singleton convolution adapter with one model-local generic operation. It reads the existing BF16 TILE QKV projection, three BF16 ROW_MAJOR carry rows, and the existing four FP32 TILE tap tensors. It emits BF16 TILE convolution output and updates the existing carry allocation. No persistent tensor, duplicate tap table, or new model layout is introduced. Prefill and public singleton-prefill paths are not changed by this component.

## Previous options and the new boundary

`AUTOFIX_native_conv_reassessment.md` and `AUTOFIX_conv_contract_followup.md` reject the specialized causal-convolution op as an exact replacement: it applies SiLU before the required BF16 preactivation boundary. Stock depthwise Conv1D requires channel partitioning, halo/layout work, and separate activation. Prior stock and structural adapter results are retained in `stock_conv_precision*.json`, `v4_conv_adapter_probe.json`, and the associated memos. This candidate does not retry those APIs.

The selected decoder already includes the validated mixed-input first multiply, folded tilize/widen for the remaining three inputs, and fused native unary chain. The remaining opportunity is to remove the materialized four-row history, repeated row slices/layout conversions, and dispatches around that same arithmetic. The generic operation deliberately reuses native compute APIs:

- `mul_binary_tile<true>` for the initial FP32 product;
- `addcmul_tile<DataFormat::Float32,true>` with scalar1 for each subsequent tap;
- the same native FP32-to-BF16 typecast, BF16-to-FP32 no-op, `silu_tile<true>`, and final BF16 typecast.

The native addcmul performs a scalar multiply then SFPMAD; its full-FP32 behavior is not an IEEE-FMA contract. No CPU-FMA or separate-add substitute is used. All four tap results are packed to FP32 local feedback and unpacked before their next consumer, including the last result before activation. This preserves the original arithmetic boundaries while removing external tensor/dispatch overhead.

Authoritative local sources are `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`, `.../eltwise_unary/addcmul.h`, `.../eltwise_unary/typecast.h`, `.../compute_kernel_api.h`, and Blackhole `ckernel_sfpu_addcmul.h`/`ckernel_sfpu_silu.h`. Native unary factory uses `math_approx_mode=false`; native SiLU explicitly initializes its non-approximate sigmoid path.

## Layout and conversion controls

The reader assigns disjoint32-channel tiles to workers, with up to64 workers. It reads two aligned blocks for the current QKV tile, each old carry row's64-byte channel segment, and the two FP32 half-rows of each tap. It creates row-zero FP32 operand tiles in local CBs, leaving padding zero. After all source history words are loaded, it writes old carry1→0, old carry2→1, and the current raw QKV row→2. Writes for different workers cover disjoint64-byte blocks. DRAM source/destination alignment is explicit; the scratch base is biased to64-byte alignment.

The prior complete conversion capture `v4_conv_widen_boundary_probe.conversion.pt` proves the selected folded RM→FP32-TILE route maps finite exponent-zero BF16 inputs to positive zero and widens all other finite codes by shifting their bits16 places. The new CPU setup reproduces this map with zero differing bits over that capture. This is a bounded source/CPU preparation check, **not** a proof of the first mixed BF16 product or QKV untilize/carry behavior. The strict device component gate includes every FP32 partial and carry word precisely to catch differences at those boundaries. Neither signed zero nor near-underflow cases are dropped.

Temporary CB storage is30KiB/core in normal mode and34KiB/core with FP32 partial capture. There are no persistent L1 buffers. Inputs retain their existing placement: QKV may be interleaved L1 or DRAM, carry stays DRAM ROW_MAJOR, and taps retain their setup allocation. Output is interleaved DRAM TILE as in the current convolution.

## Frozen component and gates

`tests/v6_decode_conv_fusion_probe.py` SHA256 `1f1ccbfb6e0ff5893ee4c558829e5144c0837e5f5adf02a93300075c1edf0c1a` is frozen after pinned autoflake/isort/Black. Source snapshots and CPU setup are recorded in `v6_decode_conv_fusion_setup.json` and its sidecars. The native oracle is AST-extracted from immutable ab28, including history creation, tap adapters, arithmetic, activation, Q/K/V slices, and carry write. Instrumentation only retains the four accumulator handles. The candidate harness includes the same Q/K/V slice adapters after its kernel, so a later complete-adapter timing must count those costs on both paths.

CPU setup loads the actual layer34 saved inputs/carry at positions3271,227,738,5799; seeded random and signed-zero histories; both halves of all65,280 finite BF16 codes; and four small-value/sign controls. Real checkpoint tap weights are used, and source/tensor hashes are retained. The pos3271 capture originated in an oracle-assisted state diagnostic; it is used here only as a same-operand convolution fixture, not as decoder acceptance or runtime oracle injection.

First parent-owned device command:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_decode_conv_fusion_probe.py --run-device --cases actual3271 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_decode_conv_fusion_device.json
```

After that passes, use a fresh output and `--cases actual3271,actual227,actual738,actual5799,random,zeros,finite,tiny`. The gate requires zero differing bits for all four FP32 accumulators, final BF16 convolution, and all three BF16 carry rows. All reports are saved before failure. Runtime executes inside the no-host-computation/conversion guard. Compilation, synchronization, and numerical equivalence remain unverified until the parent runs it.

Only a passing component may advance to warmed/traced complete-adapter timing, with all placement and carry writes included, changing-input repeated calls, and output/carry determinism. A future decoder snapshot must patch only the decode convolution region, preserve prefill, retain all public sequence/context contracts, and pass the existing every-token layer-kind/stress/watcher/performance gates. Fewer layout operations alone is not a selection result.

## Complete-adapter timing harness (prepared, not executed)

`tests/v6_decode_conv_timing_probe.py` SHA256 `f58127154248e8f31ced1b6869c7872a6e5a8cb51a49c0be70b6a477a1d14ac8` is a separate frozen harness, formatted with pinned hooks. Its CPU-only setup passed and is recorded in `v6_decode_conv_timing_setup.json` with a source snapshot. It requires the complete 12-case `component_exact` device result for the original component source SHA before opening a device. It does not modify the component.

The timed native function has the identical extracted original block AST, without diagnostic accumulator retention. The fused function uses `debug=False`. Both retain the final Q/K/V slice adapters; all history conversions, carry writes, and runtime allocations are inside the adapter. The input QKV remains L1 TILE, the existing carry remains DRAM ROW_MAJOR, and taps retain their setup placement. Setup uploads and host checks are outside measured execution. Three cycles of six actual/random/zero QKV inputs compare both evolving caches and every output; this is explicitly a synthetic sequence, not a chronological decoder stream.

An in-place three-row cache cannot be treated as immutable during replay. The timing harness first executes at least four eager warmups with a fixed QKV, replacing all three history rows. It compares that steady-state native output/carry with each candidate, then reports warmed eager groups (default three groups of20 calls) and traced groups (three groups of100 replays), synchronized around each group. It compares the output, all Q/K/V slices, and the carry before and after timing and checks first-versus-last replay determinism. These controls avoid comparing a warmed cache with the initial capture's different history.

Parent-owned command after the expanded component gate succeeds:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_decode_conv_timing_probe.py --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_decode_conv_timing_device.json
```

The default gate path is `v6_decode_conv_fusion_all_device.json`; `--component-gate` accepts a different preserved exact gate artifact. The diagnostic trace reserve is32MiB and does not change the decoder's context contract. There is no timing or correctness claim for this harness until device execution completes.

## Expanded gate failure and isolated mixed-transport revision

The parent-owned first actual3271 case passed all four accumulator, output, and carry bit comparisons (`v6_decode_conv_fusion_device.json`). The expanded twelve-case gate then failed and closed cleanly at11:29:55 (`v6_decode_conv_fusion_all_device.json` and `.device.pt`). All four actual captures and the random fixture are exact. Signed-zero input differs in4,165 newly written carry entries; the finite0 case differs in128 newly written carry entries. Each tiny case differs in2,560 new carry entries and10–13 final BF16 outputs. First-tap partials differ in290/304 entries for finite0/1 and769–785 entries for the tiny controls. These are failures of the unchanged bitexact gate, not waived underflow differences.

`v6_decode_conv_transport_cpu.json` compares every saved carry word. A general map applied only to the new QKV row—every BF16 exponent-zero code becomes positive zero—reproduces every native carry word in all twelve cases. The first two shifted carry rows were already exact. The first-tap product differences concern small normal inputs and cannot be corrected by the folded FP32 widening map: the selected first tap consumes a BF16 tile through native mixed-format unpack, whereas the initial generic reader had widened it manually. Earlier first-tap and widened-boundary evidence already distinguishes those paths. No CPU FMA equivalence is assumed.

The new independent `tests/v6_decode_conv_mixed_transport_probe.py`, SHA256 `4f1e2a409a889c40f2f15d0edbe8492f505049058f3152d9f7becab61f59d44f`, adds a dedicated BF16 first-tap CB and the same native mixed-format unpack/SrcA reconfiguration sequence used by the stock SFPU binary kernel. Taps1–3 retain the selected folded FP32 operand map. The new QKV row uses the observed native transport map before carry write; existing carry shifts remain raw copies. The extra BF16 page raises temporary CB storage to32KiB/core (36KiB debug), with no persistent allocation. The source and all twelve CPU setup fixtures are frozen in `v6_decode_conv_mixed_transport_setup.json` and its snapshots. This revision remains an unverified device hypothesis until its complete gate passes.

The separate matching timing wrapper `tests/v6_decode_conv_mixed_transport_timing_probe.py`, SHA256 `d0014b2ce77e862c6e22641328e46af99938535d9537a038b1f91ff80c4a599f`, preserves the original timing protocol and pins the new component. Its default mandatory gate is `v6_decode_conv_mixed_transport_all_device.json`; CPU setup is `v6_decode_conv_mixed_transport_timing_setup.json`. Both new scripts passed pinned formatting before freeze. Original sources, setup snapshots, device failures, and original timing wrapper remain untouched. No decoder integration was built after the failed expanded gate.

## Mixed transport gate passed; non-debug descriptor compile repair

The repaired4f1 component passed the full twelve-fixture device gate with every accumulator, output, and carry bit equal, and cleanly closed at11:33:01 (`v6_decode_conv_mixed_transport_all_device.json`). This establishes the BF16 first-tap unpack and carry transport correction under the component's debug capture mode. The subsequent timing harness exercised `debug=False` for the first time and failed compilation before timing (`v6_decode_conv_mixed_transport_timing_device.json/.log`, clean close11:33:52): C++ still instantiates the unused writer `TensorAccessorArgs<po>` expression inside its `if constexpr(debug)` block, and the non-debug argument list lacked a descriptor at `po`.

The new `tests/v6_decode_conv_complete_probe.py`, SHA256 `dd93111d7f9f7d8f5db6a13de7f1a62dd9701c9bfefdb86e90b3a20012717642`, supplies the existing output's descriptor as a valid unused partial descriptor when debugging is disabled. It adds no tensor allocation and performs no partial write in non-debug mode. The debug argument list is unchanged. `v6_decode_conv_complete_source_transfer.json` proves this single assignment is the only executable AST difference and that all three C++ strings are identical. The new complete twelve-case gate and non-debug timing are still required; the failure does not justify assuming an uncompiled path is correct.

The matching new timing helper is `tests/v6_decode_conv_complete_timing_probe.py`, SHA256 `6b76de4b6326600228a74915283273e2808fab0051415748b7a20ecc3f2a93a6`. CPU-only setup and frozen snapshots are recorded in `v6_decode_conv_complete_setup.json` and `v6_decode_conv_complete_timing_setup.json`. It requires `v6_decode_conv_complete_all_device.json` before opening hardware. All earlier sources and results remain preserved.

A source-only independent integration had just been emitted before the non-debug compile failure arrived: `v6_ab28_decode_conv_integration.py.snapshot`, SHA256 `19e91dfc39ce84546e88d93e870923d6ee638d9e2da2a1c3bd2af70007be8301`, with `v6_ab28_decode_conv_integration_build.json`. **This unmeasured snapshot is superseded and must not be promoted** because it embeds4f1's incomplete non-debug descriptor list. No combined transport snapshot was emitted from that version. `tests/v6_build_decode_conv_complete_snapshot.py` prepares the corrected standalone helper and precise decode-only branch replacement, but requires the new component gate before emission. It can separately build on ab28 and13f78, proving unchanged setup/convolution source before composition and exact full-module AST restoration afterward. Prefill, including internal singleton calls with `cache_boundary=False`, retains the original convolution block.

## Corrected complete component and timing passed

The parent-owned dd931 complete component passed all twelve fixtures with zero FP32 partial, BF16 output, and BF16 carry differences (`v6_decode_conv_complete_all_device.json`). The non-debug complete-adapter timing then passed and closed cleanly at11:36:23 (`v6_decode_conv_complete_timing_device.json`). All18 changing-input comparisons and every warmed/first/last output, Q/K/V slice, and cache comparison are bitexact. The measured normal path therefore compiled and ran; correctness is not extrapolated from debug mode.

| Complete singleton convolution adapter | Native | Fused | Reduction |
|---|---:|---:|---:|
| Warmed eager median, ms | 0.730223790742 | 0.206243502907 | 71.76% |
| Traced median, ms | 0.124941091053 | 0.060842284001 | 51.30% |

The trace result uses three groups of100 replays and includes input layout adaptation, all four tap accumulations, activation, Q/K/V slicing, and in-place carry writes. The fixed-QKV steady-state protocol and separate changing-input controls are described above. These are complete-convolution-adapter measurements, not full-decoder latency or acceptance.

## Frozen standalone candidates for whole-decoder qualification

`tests/v6_build_decode_conv_complete_snapshot.py` emits independent and composed candidates from immutable inputs, embeds the measured class and three literal kernels, and runs pinned autoflake/isort/Black followed by read-only checks. It proves exact full-module AST restoration by removing only the inserted setup statement, replacing only the new conditional with its unchanged original else block, and removing four appended helper definitions. Formatting preserves the candidate AST. The full-attention setup branch and all public method ASTs are unchanged. The new guard is `decode and cache_boundary`, so all prefill calls, including public/internal singleton-prefill, execute the original convolution block. Setup adds a nonallocating object; runtime adds no persistent L1 or host conversion. Temporary storage remains32KiB/core.

- Independent ab28 candidate: `v6_ab28_decode_conv_complete_integration.py.snapshot`, SHA256 `b55e57aaf3350bbccb59ba68bfe4ad864a35b63e191da32cbf2cbb84ee8d0c8c`. Proof: `v6_ab28_decode_conv_complete_integration_build.json`.
- Combined with13f78 transport candidate: `v6_transport_decode_conv_complete_integration.py.snapshot`, SHA256 `b253be4403326047c86d94b5e460f6a6e32bfb43373ba85af6f491cd05df1257`. Proof: `v6_transport_decode_conv_complete_integration_build.json`.

`v6_decode_conv_complete_composition_transfer.json` independently proves that the entire combined AST equals the independent candidate plus only13f78's existing `_moe` and `_forward` method ASTs. Convolution setup, runtime branch, and all four helper definitions are identical between candidates. The transport composition does not silently alter the convolution evidence. Both standalone files passed all pinned hooks before their hashes were frozen.

The parent will run independent and combined complete-decoder correctness/performance gates. No whole-decoder gain, new capacity proof, watcher result, or final acceptance is claimed from these source and component results. Original failed sources, device logs, superseded19e91d snapshot, and all diagnostic evidence remain preserved.
