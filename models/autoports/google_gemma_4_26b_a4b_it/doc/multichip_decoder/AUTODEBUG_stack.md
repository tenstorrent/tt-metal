# AutoDebug: TP1/TP4 two-layer stack divergence

## Scope and evidence

Fresh isolated, source-only investigation under AutoFix. No device was opened,
no target code was run, and no implementation was edited. Source hashes were
checked with `sha256sum` and match `stack_hybrid_tail.json`:

- `tt/multichip_decoder.py`: `3f3fc2ed3ad8ff21bcc19cb37519438187dbac5fd1e4cfdcfcf6aacd66504a20`
- `tt/optimized_decoder.py`: `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`
- `tests/test_multichip_stack.py`: `3c015c53fee4c049c29cea2ad43e541ff3fdf76ea6d2906ad2bf4a18aa8e0d14`

The preserved `runtime_hybrid_tail.py.txt` has the same `3f3fc2ed...` hash;
runtime source references below describe that frozen body. The parent began
a separate opt-in shared-decode-policy experiment during this investigation.
Its results were not seen or used in this diagnosis.

The report and log record `hybrid_experts=true`, `fused_tail=true`, replicated
residuals, exact output replicas, exact repeat traces and a clean device-only
guard. Layer 0 prefill/decode PCC is 0.99996067/0.99995707/0.99993129. Layer 5
prefill is 0.99939822; decode positions 33 and 34 fail the unchanged 0.995 gate
at 0.99479943 and 0.99310276. This is deterministic numerical divergence, not
an observed hang or nondeterministic replay failure. Repeated equality does
not prove that the first execution is correct.

The equivalent reproduction arguments are:

```sh
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.test_multichip_stack \
  --hybrid-experts --fused-tail --output <new-stack-report.json>
```

This investigator did not execute that command.

## First divergence and leading hypotheses

The earliest **observed** difference is already layer 0 prefill. The earliest
failing boundary is layer 5 decode. There are no saved module boundaries or
route IDs in the failing report, so attributing the failure to a particular
op is currently unsupported. A low aggregate prefill error also does not
bound the worst individual token's error.

The leading source-backed candidates are:

1. **Different shared-MLP policies, especially full-layer decode.** TP1 uses
   BFP4 shared weights and an explicit LoFi decode program; TP4 retains BF16
   loader weights and default `linear` kernels. This is a concrete policy
   mismatch, not a suggestion to raise precision. It occurs after routing
   within each layer, so it can directly alter layer 5's final output and can
   perturb layer 0 outputs that later feed layer 5's attention/router. Check
   same-input normalized shared outputs before changing any dtype.
2. **Small upstream errors amplified by discrete top-8 selection.** Both
   decoders route on the attention residual before their shared MLP. The
   router centers FP32 scores and then rounds them to BF16 *before* the native
   top-8 gate (`optimized_decoder.py:917-976`). A small residual change can
   move the eighth/ninth expert across the cutoff or create a tie. There is
   no small-output-error guarantee after swapping experts; the selected
   contribution also has learned per-expert scaling and a separate output
   norm. This is plausible, but no current route IDs or score margins prove
   it. The older `doc/fused_decoder/AUTODEBUG_tail65.md` contains a measured
   example of that mechanism, not proof about this fixture.
3. **Full-attention arithmetic/cache-update differences preceding routing.**
   Full TP4 changes RoPE implementation, QKV prefill fidelity, output decode
   fidelity, and fused versus separate cache writes relative to TP1. Locate
   a first difference in post-RoPE Q/K, logical cache, SDPA output or projected
   attention before deciding which control matters.
4. **TP expert partial rounding/reduction.** Decode splits intermediate
   width 704 into padded local widths 192 and reduces BF16 partial expert
   mixtures. TP1 mixes after a complete width-704 projection. Hybrid prefill
   instead reduces complete local experts across four groups of 32. These
   algebraically equivalent expressions have different BF16 rounding points.
   Identical-route, identical-input expert tests distinguish this from a
   routing change.

The fused tail is almost a verbatim copy of the TP1 default tail
(`multichip_decoder.py:511-544`, `fused_decoder.py:186-227`). It is therefore a
lower-priority suspect until its input branches have been compared. TP4 CCL
still changes those inputs. No supported reason to remove the threshold,
discard either position, or call this an inherent limitation exists yet.

## Fixture meaning

`test_multichip_stack.py:42-47` selects actual checkpoint layers `(0, 5)` and
uses the first 33 rows of the layer-0 4096/128 fixture, followed by the first
two rows of its separate decode tensor. Fixture generation saves decode from
`hidden[:, length:length+steps]`
(`tests/create_optimized_activation_fixture.py:130-152`). Thus the test uses
embeddings for tokens originally at 4096/4097 as tokens at positions 33/34.
Layer-0 inputs are embeddings, so there is no imported deeper-layer history
hidden inside those decode activations; this is a spliced token sequence.

Layer 5 then consumes layer 0 output directly, skipping layers 1 through 4.
It is a useful synthetic mixed-attention layout/cache/trace composition test,
and both TP paths are given the same computation to perform. It is not a
consecutive model-stack accuracy test or a faithful continuation of the saved
text. Its synthetic distribution may expose sensitivities absent from the
single-layer actual-layer-5 4096/128 fixture. That explains why the earlier
pass does not cover this failure; it does not waive this failure.

After localization, add a consecutive `(4, 5)` pair driven by a genuine
layer-4 fixture, or layers 0 through 5 driven by the layer-0 fixture. For a
coherent 33+2 sequence at the layer-0 boundary, use prefill rows 33:35 as
decode input. Keep the present failing fixture as a stress regression; do
not replace it with the easier case and claim the original passed.

## Cache, position and layout audit

- Both paths inherit `OptimizedDecoder.prefill_forward/decode_forward`.
  Each layer owns independent K/V allocations, page table and two position
  buffers. The random page permutation is created once before the TP loop,
  so corresponding TP1/TP4 layers use identical page ownership.
- Cache storage is BFP8, 32 rows per block, four physical pages, capacity 128.
  TP1 uses Q16/KV8/D256 for layer 0 and Q16/KV2/D512 for layer 5. TP4 uses
  Q4/KV2/D256 and Q4/KV1/D512 locally. Loader mapping assigns full-layer KV0
  to ranks 0/1 and KV1 to ranks 2/3
  (`models/demos/gemma4/tt/attention/weights.py:83-102`). The harness allocates
  using the actual local config (`test_multichip_stack.py:92-101`), rather
  than allocating global heads and interpreting them as local heads.
- Logical prefill 33 is padded to 64 before both layers, and each layer
  returns exactly 33 logical rows. `OptimizedAttention.prefill` writes the
  tile-rounded 64 cache rows; later decode overwrites rows 33/34. The paged
  attention mask must exclude future padding. Layer 5 receives the real
  device output and independently pads its input; no host handoff changes
  precision or memory layout.
- `current_pos` is uint32 for RoPE embedding; `cache_pos` is int32 for both
  cache update and SDPA. Every layer refreshes both to 33 or 34 before replay
  (`test_multichip_stack.py:163-173`). Warmup/capture overwrite row 33, and
  repeated trace replay overwrites the same current row. No source-level
  off-by-one or automatic hidden counter is apparent.
- Position 33 is logical block 1, row 1; position 34 is block 1, row 2. They
  are not page boundaries. Native SDPA uses dynamic `k_chunk_size=0`; the
  kernel computes two tiles at these positions, rounding reads to 64 rows
  (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:95`,
  `get_workload_for_core` in the same file). The 128-row allocation covers
  that read window. An over-allocation test is still cheap if exact cache
  readback disagrees, but there is no present under-allocation finding.
- Both final tails return BF16 DRAM tensors. TP1 stages decode residuals in
  per-core sharded L1; TP4 keeps mesh replicas and normalizes through the
  inherited per-core norm path. `sharded_residual=false` here. The latter
  flag describes mesh partitioning, not the per-core sharding used by norms.
- Full TP1 uses two `paged_update_cache` calls and two-step cast/shard; TP4
  uses `paged_fused_update_cache` with disjoint update cores and unary
  cast/shard (`fused_decoder.py:609-642`, `multichip_decoder.py:335-339`).
  The stack harness does not compare cache content. A cache audit must
  reconstruct logical pages and compare corresponding global/local heads,
  including valid rows and preservation of other rows, not compare raw
  physical buffers across topologies.

## Precision-policy ledger

Source locations: TP1 auto-policy is `optimized_decoder.py:280-440`, setup is
`:492-642`; TP4 setup is `multichip_decoder.py:281-427`. S/F below means sliding/full.

| Boundary | TP1 baseline | TP4 hybrid candidate |
| --- | --- | --- |
| Public input/output | BF16 | BF16 replicated |
| QKV weights/output | BFP8 / FP32 | BFP8 / FP32 |
| Decode QKV fidelity | S HiFi2; F LoFi; FP32 accumulation | Same named fidelity/accumulation; different local program/width |
| Prefill QKV fidelity | S HiFi4; F HiFi2 | HiFi4 both kinds |
| Full RoPE | Separate FP32 multiply/add (`precision_ops.rotary`) | Fused `rotary_embedding_hf` |
| Sliding RoPE | Separate prefill, fused decode | Same policy |
| K/V cache | BFP8, page32, BF16 update values, replicated page IDs | Same format; local KV heads; full decode update op differs as above |
| SDPA fidelity | Decode S HiFi4/F LoFi; prefill S LoFi/F HiFi2 | Same policy, fewer local heads |
| Output projection | BFP8 weights, FP32 output; decode S HiFi4/F LoFi; prefill LoFi | BFP8 weights, FP32 partials; decode HiFi4 both kinds; prefill LoFi; FP32 CCL |
| Router | BF16 weight, FP32 scaled input/output, decode S HiFi4/F LoFi, max-centering then BF16 native gate | Same policy and replicated weights |
| Expert weights | Gate/up S BFP8/F BFP4; down BFP4; LoFi; BF16 outputs | Same formats; EP prefill and TP decode; BF16 CCL after local mixtures |
| Expert decode input | S BF16, F BFP8 | Same policy |
| Shared decode | S BFP8/F BFP4 weights; explicit LoFi, BF16 output, FP32 accumulation off, packer accumulation on | BF16 loader weights; default `ttnn.linear` config/output; BF16 branch/CCL expected, verify lowered args |
| Shared prefill | BF16 inherited fused shared MLP | BF16 local shared MLP plus reduction |
| Tail | Sharded decode branch norms + residual-input norm + BF16 fused residual/scalar add | Same structure with `fused_tail=true` |

`_SharedMLP.compute` is stored but never passed to either projection
(`multichip_decoder.py:74-87`). Do not label those default matmuls HiFi4 merely
because the wrapper receives the HiFi4 config. This distinction should be in
the stage's policy manifest before comparing or changing precision.

## Smallest discriminating test ladder

1. **Instrument the exact failing stream, without altering runtime defaults.**
   Retain tensor references and read after execution; avoid host operations
   inside the guarded forward/trace. Save per-layer/per-position input,
   input norm, post-RoPE Q/K/V, projected attention, attention residual,
   normalized shared output, normalized routed output, combined tail and
   final output. Record PCC, relative RMS and max error, plus per-token
   prefill metrics. Preserve final-output identity against the original
   failure so observation cannot silently change the case.
2. **Capture routing at the first large jump.** Save FP32 router scores,
   centered BF16 scores, selected IDs and route weights on both paths.
   Record rank-8/rank-9 margin and top-set overlap. A `ttnn.topk` hook alone
   misses decode: production uses `generalized_moe_gate`. Wrap the router's
   projection/gate boundary or read its retained `decode_indices`. If sets
   differ, evaluate the CPU router on each saved TT residual and compare
   with CPU routing on each saved normalized/scaled input. This separates
   inherited input drift, norm/projection drift and gate quantization.
3. **Isolate propagation from a local layer-5 discrepancy.** Replay layer 5
   TP1/TP4 with the *same saved* layer-0 prefill outputs and decode outputs,
   rebuilding their own caches from that common sequence. Repeat with the
   other saved sequence only if needed. This diagnostic host reinjection
   does not replace the required direct-handoff test. If it resolves the
   failure, tiny layer-0 differences are necessary for this case; if not,
   localize layer 5 with the saved module boundaries.
4. **Test only the first divergent branch.** For matching routes and a bad
   shared branch, run that branch on identical saved inputs and compare
   both outputs against the same CPU BF16-weight oracle. A focused TP1
   control `shared_dtype=None` exposes the existing BF16 shared backend;
   changing that flag alone is diagnostic, not an accepted baseline swap.
   For matching shared output but bad experts, supply identical route IDs
   **and weights** and compare local projections/mix/CCL. A fixed-routing
   control must update the indexed router IDs as well as dense routing,
   since indexed decode consumes both (`optimized_decoder.py:210-215`).
5. **If attention/cache diverges first**, compare actual logical K/V after
   prefill and positions 33/34, then CPU SDPA on the saved TT Q/K/V. Run
   separate one-factor controls for full RoPE backend, full QKV prefill
   fidelity, full WO decode fidelity, or update op according to the first
   changed boundary. Retain BFP8 cache for the first controls. Any BF16
   cache experiment needs a same-BFP8 arithmetic control and an exact-shape
   cache probe; a BF16 pass alone does not prove a cache-kernel defect.

After a verified cause and minimal fix, rerun this unchanged stack gate and
the affected single-layer 4096/128 gate. A coherent consecutive-layer control
adds model realism but cannot retroactively pass this recorded failure.

## Status

Still failing. The source audit narrows the test ladder but establishes no
root cause or fix. Parent agent owns experiments and the stage work-log
update. No blanket higher-precision change or acceptance exception is
recommended.
