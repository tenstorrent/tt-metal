# Decode RoPE producer layout

The candidate uploads the two decode RoPE tables directly in row-major layout
at harness setup. Prefill keeps its separate TILE uploads; embedding still
returns TILE rows to the existing rotary path. No forward host work, table
content cache, or runtime conversion is introduced.

This is a source-only proposal against runtime SHA
`0aabcac2109a35b436c78ca6322ba4e88331abdab39e8271e14ea5235af9938a`.
No runtime or existing harness file was edited. Numerical, trace, and timing
acceptance belongs to the hardware owner's actual-input runs.

## Source contract

- `ttnn/cpp/ttnn/operations/embedding/embedding.cpp:28-35` converts every TILE
  weight table to ROW_MAJOR before lookup and unsqueezes it to rank4. Supplying
  ROW_MAJOR skips that table conversion; this is not a content-dependent cache.
- `embedding.cpp:45-68,75-77` retains explicitly requested TILE output.
  Row-major indices with width32 and head dimension256/512 satisfy the fused
  output-tilization conditions. The RoPE row values, BF16 table dtype, absolute
  tensor-valued positions, and gathered-row layout remain unchanged.
- `embedding/device/embedding_device_operation.cpp:32-44` requires BF16
  ROW_MAJOR interleaved weights, precisely the proposed decode producer format.
- `tt/fused_decoder.py:609-616` consumes caller tables through
  `ttnn.embedding(..., layout=ttnn.TILE_LAYOUT)`. Optimized decode inherits
  this path. The public decoder imposes no TILE-only input-table restriction,
  so legacy TILE callers remain accepted through embedding's conversion.
- `tests/run_decoder.py:164,255` currently uploads separate 4D prefill tables
  and 2D decode tables with its TILE default. The candidate affects only the
  latter. B32, request-reuse, and long-context harnesses use the same pattern.

The old path converts the full two tables during each recorded decoder call.
Removing those operations is the source hypothesis. No latency improvement
is claimed here; parent-owned profiles establish the actual cost and benefit.
At the headline extent5120, the BF16 pair holds 5,242,880 bytes for D256 or
10,485,760 bytes for D512. Both shapes are tile aligned, so changing the
producer layout does not reduce the pair's tensor payload. It avoids the
repeated conversion work and its temporary tensors.

## Runnable probe

[probe_optimized_decode_rope_layout.py](../../tests/probe_optimized_decode_rope_layout.py)
uses the unchanged audited `run_optimized_decoder` harness. It records exact
storage identities of the actual HF-generated cos/sin tensors. The setup
upload wrapper changes layout only for their 2D views; the 4D prefill views
and unrelated same-shaped tensors retain their requested layouts. Runtime
`device_only()` guards still replace `ttnn.from_torch` while the decoder runs.
The successful probe requires exactly one cos/sin upload for each phase and
records actual output dtype/layout/memory descriptors in JSON.

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_decode_rope_layout \
  --decode-rope-layout row_major --defaults --layer 0 --length 4096 --real \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --decode --steps 128 --timing --prefill-timing --verify-program-cache \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_decode_rope_rm_layer0.json
```

Use the corresponding layer5 fixture and output for full attention.
`--decode-rope-layout tile` is the same probe's producer-layout control.
[decode_rope_layout_command_plan.json](decode_rope_layout_command_plan.json)
contains unexecuted command templates for both kinds at4096/128 and1025/512,
with fused baselines retaining their old TILE producer default. Stress
commands save already-read outputs for the existing direct preservation
comparison. These are command plans, not run journals or passing evidence.

## Unapplied integration

[decode_rope_producer_layout.patch](decode_rope_producer_layout.patch) adds
`OptimizedDecoder.decode_rope_layout = ttnn.ROW_MAJOR_LAYOUT` as a caller-facing
setup preference. Decode still accepts TILE. Four owned harnesses consume
`getattr(decoder, "decode_rope_layout", ttnn.TILE_LAYOUT)` (using their local
`layer` variable where appropriate) when uploading only the 2D decode tables:
`run_decoder.py`, `batched.py`, `request_reuse.py`, and `long_context.py`.
`run_decoder` also records the chosen layout in its result JSON. Functional
and fused decoders advertise no preference and retain TILE by default.

This is a producer/consumer layout contract, not an internal cache keyed on
arbitrary caller tensors. The caller creates and owns immutable RoPE tables
before capture; existing trace position buffers remain device supplied.
Prefill-prefix continuation may still reshape caller TILE prefill tables for
its internal decode path; no hidden persistent replacement table is created.

`git apply --check` passes against the recorded constituent file hashes in
[decode_rope_layout_source_checks.json](decode_rope_layout_source_checks.json).
The patch is intentionally unapplied while parent-owned device trials run.

## CPU validation and limits

The new probe passes Black (line length120, targetpy310) and syntax checks.
Eight CPU-only setup cases execute its actual class AST using real Torch
storage views at contexts5120/2048, head dimensions256/512, and TILE/ROW_MAJOR
choices. They verify exact cos/sin view identity, unchanged prefill uploads,
and no effect on unrelated same-shaped tensors or shorter same-storage views.
Eight further cases execute the proposed harness capability lookup: optimized
chooses ROW_MAJOR and an ordinary fused/functional object chooses TILE in each
of the four harnesses. All proposed files parse; the runtime file stayed
unchanged during these checks. No TTNN import or device access was used.

These checks establish setup selection and shape/layout intent. They do not
simulate device rounding, test trace execution, or provide synthetic-PCC
acceptance evidence. The 0.995 actual-input prefill/decode threshold remains
unchanged in the delegated hardware harness.
