# Scoped prefill SwiGLU adapter proposal

Unapplied CPU-only proposal; no hardware/build/TTNN import or production edit. `prefill_swiglu27.patch` changes only the proposed decoder method and its actual flat-prefill call site. `prefill_swiglu27.json` is an18-job whole-decoder manifest using exact `norm_whole26` minimal controls/reference files, GDN column1 retained. `prefill_swiglu27_cpu.json` records source/patch hashes, 1,404 chunk-geometry checks and normalized AST equality of both original default minimal branches. `git apply --check` passes. `prepare_prefill_swiglu27_patch.py` regenerates only these D artifacts.

**Important topology correction:** bulk prefill executes `flat_prefill._finish_flat`, not `MultichipDecoder._finish`. A hook solely in the latter would miss the measured M16384 work. The patch routes the former into the same explicit helper only when opted in; its default body is unchanged. B32/S33 is one1056-row flat chunk, not a1024-row bulk plus32-row tail. B32/S513 is16384 rows plus32-row tail. This corrects the short-shape prose in my geometry27 audit; that manifest remains legal, but c640 at S33 splits1056 into640+416. All tail timing attribution must use the actual S513 dispatch.

## Narrow policies and matched comparisons

`prefill_swiglu_options` applies only to **prefill M>512**. It accepts:

- `{"k_block":8}`: M4/K8/N16, subblock1×4, grid11×10, same entire bulk call, existing paired weight and `matmul_configs["mlp"]`.
- `{"k_block":4,"chunk_rows":2048}`: matched DRAM-chunk control, including slices, eight native calls and output concat at M16384.
- `{"k_block":4,"chunk_rows":2048,"input_l1":true}`: the same chunk family including every DRAM→L1 ingress and output concat. L1 chunks are explicitly bounded to≤2048. K8+L1 is supported but deferred until the two individual families show a reason to compose them.

The native config and output dtype stay BF16, activation stays BF16, weight stays the already-resident BFP8 paired tensor, and fidelity/FP32-DST/packer policy is passed through unchanged. Per-call code owns padding to32 rows, unpadding of the final result, and restoration of logical B/S shape. Chunk temporaries have function-local lifetimes, so admitting a new L1 chunk does not intentionally retain the previous one. No new persistent weight/activation cache or host forward is introduced.

`prefill_swiglu_tail_packed=true` applies only to **prefill M≤32**. It uses existing DRAM-sharded contiguous gate/up weights through `_linear`, followed by the existing `packed_swiglu` epilogue; it owns flattening `[B,1,H]` or `[1,B,H]` and restoring logical output axes. The weight's DRAM location does not require a DRAM activation restore: the epilogue preserves whichever interleaved/width-sharded layout `_linear` chooses, and the next down projection performs its existing admission. Existing `packed_mlp` and `fused_packed_swiglu` must both be enabled. Decode dispatch and middle M33..512 remain unchanged.

The manifest contains, for each kind, S513 control plus K8, DRAM-chunk, L1-chunk and tail-packed; S33 control plus the three bulk variants. Tail-only S33 and B1/S512 rows are deliberately omitted because those shapes do not exercise this tail hook or bulk override. A future B1/3/17/S513 tail stress needs matching frozen references; do not infer that coverage from B32.

Run only after the parent applies/reviews the proposal:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_search models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_swiglu27.json --resume
```

The M4/K4/N16 full bulk is only the required matched control; no redundant broad search of previously optimized settings is proposed. Full decoder prefill includes all admission, slices, intermediate allocations, epilogue, concat and downstream MMRS. Standalone matmul-only timings cannot decide the L1 family. Reprofile only a successful promising whole family and verify the expected op/config/placement changes.

## Native legality and physical estimate

Native minimal matmul partitions the272 weight tile columns on **136 gate/up pairs**, rounding pairs per N-worker, so pairs never split (`minimal_matmul_program_factory.cpp:319–347`). Full M16384 transposes the parallel axes: M over11, N over10, giving per-core M47 and weight N28; M2048 uses M over10,N over11, giving M7,N26. The same block16 N is legal because native masks partial final blocks. K160 tile columns divide K4/K8; blockM4 and N16 divide subblock1×4; N blocks and per-core N stay even. Merely changing row partition can alter native arithmetic reduction order and must pass quality gates.

For BF16 input/output, BFP8 weight tiles1088 bytes and FP32 intermediate tiles4096 bytes, native CB allocations give:

| Per worker, excluding live tensors/runtime reservations | K4 | K8 |
|---|---:|---:|
| Double-buffered A |65,536 B|131,072 B|
| Double-buffered B |139,264 B|278,528 B|
| Double-buffered half-width output |131,072 B|131,072 B|
| Full intermediate block |262,144 B|262,144 B|
| Total |598,016 B|802,816 B|

An M2048 input is20,971,520 bytes total;110-bank tiled interleaving gives an estimated192,512 bytes/bank. Thus K4 plus one chunk is790,528 bytes before resident owner buffers, semaphores, runtime/kernel reserve and allocator alignment; K8 plus chunk995,328 bytes. These are **estimates, not measured free-L1 admission**. The live minimal norm owners remain resident. If native capacity rejects2048, retry1024 then512 with complete cost and unchanged gates, rather than rejecting L1 after one error. Do not place the whole M16384 input in L1.

## Numerical evidence required

Do not claim tail-packed is bitwise identical to fused minimal. The existing native `swiglu_sfpu.hpp` uses accurate FP32 sigmoid and unrounded gate/up DST when FP32 accumulation is enabled. Packed projection instead first emits BF16 gate/up and the already-audited epilogue uses native BF16 SiLU and multiply roundings. This is an explicitly compared arithmetic family, with the same selected external dtype/fidelity and **unchanged layer quality gates**, not a silent relaxation. K-block and chunk changes preserve the existing native arithmetic functions but may alter summation order.

Before a focused comparison, save actual normalized BF16 inputs per rank and decode both resident BFP8 weight representations once during setup. Reorder paired columns as `gate tile,up tile` → contiguous halves and assert decoded-value equality to the contiguous packed weights; fail on mismatch rather than changing quantization. Form an independent CPU FP64 matmul from those actual decoded operands, then compute a mathematical SwiGLU diagnostic. For the packed path also retain a separate BF16-projection-boundary CPU diagnostic; native bitwise epilogue comparison must use the same saved BF16 gate/up values with the unchanged native BF16 SiLU+multiply reference. Do not require FP64 sigmoid and native SFPU approximations to be bit-identical. Report per-user/per-rank PCC, absolute error, finite masks and differences before assertions. Constant unequal outputs remain fatal; no gate changes.

The authoritative end-to-end checks remain the exact frozen-reference output/contribution/per-user/state gates in `run_multichip_decoder`; HF-contract evidence remains separate. Standalone changed-input, rebound-buffer, queued50 and trace equality compare each path to its own eager output, while numerical comparisons use independent inputs/weights and the frozen decoder. Watcher/allocation-tracker positive controls are separate from timing. Padding poison must test unaligned public batches and sequences; source legality alone does not close that gate.

## Gather-first fused norm: supported name and necessary flat-path admission

The existing **local fusion hook is directly supported**: `sfpu_prefill_norm=true`, with `gather_norm=true`, selects `prefill_hidden_rmsnorm_sfpu` before explicit-sum norm in `OptimizedDecoder._norm`.

However, `flat_prefill._can_flatten` currently excludes literal `gather_norm=True`. My preceding audit missed that guard. Setting only those two flags changes the entire prefill scheduler/physical-batch path, so it is not the intended matched norm-placement comparison. `flat_gather_prefill27.patch` supplies a separate off-default `flat_gather_prefill=true` admission for the explicit gather-first case. It retains the same flat bulk, postnorm caller and per-user cache/attention layout; the source already passes logical_batch where needed. Both patches pass combined `git apply --check` and are unapplied.

After that tiny admission is independently checked, matched jobs can use (a) current control, (b) `gather_norm=true,flat_gather_prefill=true`, and (c) the same plus `sfpu_prefill_norm=true`. Preserve minimal decode norm: its branch precedes gather selection. The independent actual-input/gamma FP64 oracle and whole-layer gates in `long_prefill_geometry27_audit.md` still apply. Capture actual dispatch/op provenance to prove these jobs retain the flat topology. Do not mix this comparison with the SwiGLU adapter before isolating its effects.
