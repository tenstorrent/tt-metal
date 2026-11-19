# BFP8 KV short-prefix drift audit

CPU-only inspection of the paired full-model captures found no concrete cache
construction, update, gather, input, or policy mismatch explaining the failure.
The BFP8 candidate still **fails** the required per-user vocabulary PCC floor
of 0.99. Matching top-1 tokens does not waive that floor. Accumulated numerical
drift is consistent with the evidence, but these final-logit captures do not
localize the first divergence or prove a quantization-only cause.

The raw artifacts are `kv_b32_inherited_e4_lofi_v1.pt` and
`kv_b32_kv8_v1.pt`, with matching JSON/provenance sidecars in this directory.
Their inputs are exactly equal `[32,35]` integer tensors; every user receives
the same 33-token prefix and two forced next tokens. Their policies differ only
in `config_id` and `kv_cache_dtype`. Checkpoint, environment, capacity128,
phase order, final positions35, and decode counters agree. Runtime source
digests agree; the only provenance source difference is the unrelated
`tests/report_datatype_sweep.py`. Each run has finite `[32,3,248320]` logits,
and all 32 users are bit-equal within that run.

Recomputed from the raw tensors, using float64 centered Pearson correlation:

| Phase | Vocabulary PCC | RMS logit difference | Top-1 BF16 / BFP8 | Top-100 set overlap |
| --- | ---: | ---: | --- | ---: |
| Prefill33 | 0.971848264 | 0.443313 | 257 / 257 | 87 / 100 |
| Forced position33 | 0.972965253 | 0.552158 | 14540 / 14540 | 69 / 100 |
| Forced position34 | 0.984140623 | 0.373217 | 17082 / 17082 | 89 / 100 |

The shared-pool allocator directly creates BFP8 zero K/V tensors and binds
the same dtype into all request/scratch cache views (`tt/cache_pool.py`).
`MultichipDecoder._create_base_state` likewise directly constructs the
selected dtype for private caches. `MeshQSA.build` selects the BFP8 gather
only for BFP8 storage. Q/K/V projections and cache-update input tensors remain
BF16; both prefill and decode sequentially call `QSA._update` /
`paged_update_cache`. Native validation permits BFP8 destinations. No
`paged_fill_cache` dtype adaptation is needed on this path. Indexer compressed
keys and raw history remain BF16; recurrent state remains FP32.

`tt/kernels/local_selected_paged_kv_bfp8.cpp` uses the 1088-byte BFP8 tile
stride, matching exponent/face offsets, page-table translation, invalid-ID
zeroing, and BF16 selected-row outputs. The stage-5
`../optimized_multichip_decoder/bfp8_kv_component.json` records exact native
unpack agreement for Q=1,2,31,32 on all four ranks, changed selections, page
permutations, and six trace replays per shape. The corresponding Python wrapper
and probe digests still match. The kernel digest differs; the stage-5 work log
explicitly records its post-measurement clang-format-only change. Current
normal-construction layer-3 evidence in
`kv_short_prefix_b32_comparison_v2.json` independently passes the paired
decoder checks (minimum prefix row PCC 0.999030874 and B32 PCC 0.999735122).
That layer result cannot establish full-stack numerical agreement.

One concrete provenance correction is needed in the surrounding report:
token offset256 is the book's **table of contents**, not narrative prose
after front matter. Decoding the saved tokens starts “the Second--the Golden
Thread” and continues chapter titles. This is still a valid matched short-prefix
cache diagnostic, but it is not the separate 128+512 natural-prose readiness
reference. Also, the single-layer B32 fixture uses distinct checkpoint embedding
rows at empty-cache position0; it does not duplicate the full-model B32
same-input, cached-prefix regime. Neither distinction explains away the
full-model failure.

No implementation fix or extra hardware rerun is justified by this inspection
alone. Retain KV8 as rejected by the recorded diagnostic while completing its
planned full-reference accuracy/performance measurements. Reconsidering KV8
would require locating the first divergent real-model state and validating any
repair; this audit does not grant acceptance or classify the failure as an HF
limitation. No TTNN import, device access, or source edit was performed.
