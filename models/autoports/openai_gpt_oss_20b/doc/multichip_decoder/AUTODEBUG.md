# AutoDebug: multichip decoder round-1 review

## Scope

This source-only investigation addressed the six findings in
`stage_review_round1.md`. The normal fresh-context AutoDebug wrapper could
not start because `bubblewrap` is unavailable on this host, so a
fresh-context subagent performed the source and evidence audit directly.
This report records hypotheses; measured resolutions are in `AUTOFIX.md`.

## H1: optimized precision policies were not preserved

The first implementation used native prefill attention and one LoFi/BFP8
expert policy for every layer and length. The current `OptimizedDecoder`
needs precision-sensitive policies for full attention:

- exact FP32 manual attention at S=128;
- higher-fidelity active experts for full prefill; and
- distinct sliding/full long-context projection fidelity.

Required experiment: compare real layer 12 and 13 weights at S=128, S=129,
and S=2048 against isolated current `OptimizedDecoder` captures. Check
attention, routing/top-4, final output, logical K/V, and following decode.
Keep EP4 gate-selected execution and avoid replicated dense experts.

## H2: long trace captured a host-specialized position

The first manual long-decode helper derived gather ranges and shapes from a
Python cache position. Replaying one fixed position proved determinism but
not sequential generation.

Required experiment: first test native paged SDPA across 128-131 and
191-193. If it is inaccurate, build fixed-shape, device-masked 64-token page
banks so one capture advances within a bank and recapture occurs only at a
bank boundary. Compare every replay with eager and verify the exact local
physical K/V write.

## H3: profiler overwrote accepted wall timing

Wall timing and Tracy used the same JSON name. A one-prefill/three-replay
profile therefore replaced the accepted 20/500 artifact.

Required fix: support a profile-specific result path, freeze source/test
hashes, rerun all 1x1/1x4 timing rows, and keep profiler provenance separate.

## H4: topology rejections were historical

The first report selected all-reduce + EP4 but rejected alternatives using
older-branch evidence. Required current controls:

- ordinary ring all-reduce;
- padded H=2880→2944 reduce-scatter plus all-gather;
- a 736-wide residual carried through distributed RMSNorm, router, and the
  next real packed QKV;
- fused attended all-gather + local O, plus the Blackhole fused
  matmul/reduce-scatter API audit; and
- TP4 versus EP4 gate-selected experts.

Each control must record shapes, padding, buffers, PCC/top-4, and warmed
whole-layer latency.

## H5: context endpoint covered only sliding attention

The first 131072 gate allocated and wrote a sliding-layer cache but did not
qualify the full-attention path or compare endpoint output.

Required experiment: run both layer kinds with reverse page tables, verify
finite replicated output and exact physical endpoint K/V, and compare with
both the default-native and exact-manual single-chip endpoint controls.

## H6: Ethernet watcher disablement lacked current justification

Historical runs indicated that full watcher instrumentation made the
Blackhole ACTIVE_ETH program 27,920 bytes, exceeding the 25,600-byte kernel
configuration buffer before model execution.

Required experiment: reproduce without disabling ETH on final source. If the
same platform limit occurs, run the maximal worker/Tensix watcher with
`TT_METAL_WATCHER_DISABLE_ETH=1`, then compensate with CCL-heavy trace,
fabric/ring runs, ARC/Ethernet triage where supported, and a final device
health listing.

## Repair order

1. Make long decode sequentially traceable.
2. Restore real-weight prefill/endpoint precision.
3. Separate and regenerate performance/profiler provenance.
4. Re-run topology alternatives.
5. Requalify full context and watcher evidence.
6. Request a fresh independent stage review.
