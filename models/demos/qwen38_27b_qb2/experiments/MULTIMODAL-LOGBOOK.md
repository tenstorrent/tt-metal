# Image and video feature branch

October 10, 2026. Implementation and CPU checks only; **not hardware or endpoint
qualification**. No serving process, allocation, native installation or deployment
has been changed for this work. The screenshot reports image/video URL/base64
errors, but it contains no response body or endpoint revision. The particular
errors' cause remains unproven.

## Isolation and revisions

- Metal worktree: `/private/tmp/tt-metal-qwen38-multimodal-20261010`.
- Branch: `anatarajan/qwen38-multimodal-20261010`.
- Base: `4d4c6db8d30527d74a0fbbf42e1106422e546ba4`, the published performance
  implementation. Subsequent performance profiling is separate.
- Native library reference: `a08819ddbe23077f8037d3802303939064868ff6`.
- Transformers: `5.12.1`; vLLM: `0.26.0`.
- Companion plugin: `anatarajan/qwen38-multimodal-20261010` in
  `tenstorrent/vllm-tt-plugin`, based on `e5b02d58` and the existing stable
  prefill-state-slot contract. Its local implementation commit is
  `f06225fbb4dfece4a6494517a9850c23db880615`; both SSH and HTTPS publication
  were denied for that repository. It is not a verified remote artifact.
  Resolve and pin a published plugin SHA before deployment.
- Checkpoint: host-local `checkpoint-pinned-1d4bf0f2`, with
  `Qwen3_5ForConditionalGeneration`. Its 333 vision tensors contain 460,730,096
  BF16 parameters (921,460,192 bytes), all in the first safetensors shard.

## Implemented

`tt/generator_vllm_multimodal.py` adds the opt-in architecture
`TTQwen38ForConditionalGeneration`. The existing text architecture and launch
script stay unchanged. The new adapter retains TP4, the configured batch size,
asynchronous device sampling and the existing logical paged KV layout. Eight DP
workers would each load their own vision encoder; this has not been launched.

`tt/vision_weights.py` reads only `model.visual.*` tensors, constructs only the
vision reference on the meta device, assigns its checkpoint parameters directly,
and initializes its nonpersistent rotary buffer. It never constructs or loads a
second 27B language model. The current CPU reference retains the vision weights:
about 0.92 GB per worker before temporary buffers, roughly 7.37 GB for eight
workers. Actual device memory and startup time still require measurement.

`tt/vision.py` reuses the existing Qwen3.5 TT attention, MLP, distributed norm and
patch-merger implementations. Patch embedding and position interpolation run on
CPU, matching the inherited implementation. The inherited DropIn forward did not
forward its calculated cumulative sequence lengths. This adapter passes explicit
per-temporal-frame windows plus a separate padding window into **every** vision
block. Actual patch count is cropped before merging. This fixes the wiring in
the new path without changing the shared Qwen3.6 implementation.

`tt/multimodal.py` validates media grids, placeholder spans, feature counts and
finite encoder outputs. It builds complete-prompt 3-axis M-RoPE positions and
chunk-specific embedding replacements. Video timestamps remain text tokens;
each frame receives its own spatial grid, matching the pinned HF implementation.
No media is silently dropped or converted into a text-only request.

Model prefill splices the encoded visual features into token embeddings and uses
the request's 3-axis cos/sin. Logical token offsets still control KV writes,
causal attention and recurrent continuation. Decode uses separate rotary indices
(`logical position + request delta`); KV positions remain unchanged.

The adapter implements all four explicit plugin reload commands. Page-table or
rotary-only refreshes preserve advancing device sampling state. Host slot/delta
metadata is committed only after accepted submission. Request-specific encoding
is retained for live chunks and follows an explicit decode slot permutation;
the plugin preserves a continuing prefill request's physical owner slot.

## Plugin contract

Only the new adapter advertises `supports_video_inputs=True` and
`supports_multimodal_chunked_prefill=True`. It keeps `supports_prefix_caching=False`.

For each scheduled prefill row the plugin supplies:

- `mm_request_ids`: request ID in actual submitted row order.
- `mm_prompt_token_ids`: the complete original prompt, including all visual
  placeholders and video timestamp tokens, rather than only the current chunk.
- `mm_item_spans`: items with `modality`, `identifier`, `offset`, `length` in
  feature order; modality pixel/grid lists preserve occurrence order.
- `pixel_values`, `image_grid_thw`, `pixel_values_videos`, `video_grid_thw`:
  outer lists in request order, inner lists in modality occurrence order;
  absent modalities are `None`.
- Video timestamps and `second_per_grid_ts` may be forwarded for provenance.
  The pinned reference derives rotary positions from the already-expanded
  prompt and frame grids; this adapter does not regenerate timestamps.

`input_tokens` still contains the prefix through the scheduled chunk end.
`empty_slots` identifies physical recurrent-state owners. The adapter returns a
rotary delta for each submitted row. The plugin associates those deltas with its
submitted request snapshot and later passes decode-order deltas only when needed.

A missing cached processor payload is usable only for the same live plan, exact
request ID, prompt and item identities. A fresh or preempted-from-zero request
without pixels fails explicitly. No cross-request feature or prefix reuse is
implemented. Prefix/SSD integration must first include media content identity,
processor settings, grids/timestamps and rotary state in its checkpoint contract.

## Current limits and performance tradeoffs

- Limits are four images and one video per request, at most 32,768 raw patches
  per modality encoding. Native cumulative boundaries are bounded to 1,024
  entries. Frontend processor limits must bound decoded media before execution;
  arbitrary input sizes are not qualified. A conservative processor override is
  `--mm-processor-kwargs '{"max_pixels":2097152}'`: at patch size 16, four
  images fit the aggregate raw-patch budget, and video is bounded more tightly.
  This operator setting is not yet a validated request-enforced hard cap.
- Media prefill is sequential across scheduled physical slots; mixed batches use
  this same conservative path. Text-only batches retain the existing fast paths.
- Vision uploads and transient visual prefill inputs release parked decode
  traces; decode is recaptured afterward. This prioritizes buffer correctness
  and may add TTFT/turn-boundary cost. No visual TSU or TTFT result is claimed.
- Vision output currently returns through host memory before embedding upload.
  Keeping it on device is a later optimization once reference parity is proven.
- Weight-cache directories include checkpoint metadata identity and worker PID
  under `QWEN_VISION_CACHE_DIR`, then `TT_METAL_CACHE`, then `/tmp`. This prevents
  DP workers racing on tensorbin creation but does not reuse files across process
  restarts; all are disposable host files. Measure capacity before deployment.
- The extra vision weights and activations may reduce usable KV capacity. No
  claim that prior text-only capacity remains available is made.

## Validation and remaining gates

The CPU suite checks pinned-reference image/video/mixed M-RoPE positions and
deltas; chunk reconstruction; malformed or missing media; fresh request/slot
identity isolation; frame/padding attention windows; exact tiny-reference vision
loader output for FP32 and BF16; and that language tensors/constructors are never
accessed. Adapter tests execute actual production method bodies using AST loading
and explicit CPU substitutes at TTNN/vLLM boundaries. They cover continuation,
slot reorder, all reload commands, sampler preservation, KV/rotary separation,
and native window-argument forwarding. **These mocks do not validate TT kernels.**

Run on a normal supported Metal environment:

```bash
PYTHONPATH=. python -m pytest \
  models/demos/qwen38_27b_qb2/tests/unit/test_multimodal.py \
  models/demos/qwen38_27b_qb2/tests/unit/test_multimodal_adapter.py -q
```

Local macOS tests use the same repository `expect_error` fixture extracted by
AST to avoid importing root device fixtures, with `--confcutdir` pointing at the
unit-test directory. The raw CPU report and source hashes are recorded in
`multimodal-evidence/`. Repository pre-commit checks also run before publication.

Before enabling this architecture in a deployment:

1. Import the real adapter and registry using the exact pinned native library,
   Transformers, plugin and vLLM environment. Confirm processor/dummy limits and
   start-up memory accounting; the macOS suite cannot check those imports.
2. On one allocated TP4 mesh, compare encoder output and prefill logits against
   the pinned HF visual reference for one image, multiple images, and video,
   including ragged frame boundaries and padding. Gate numerical error before
   acceptance. A CPU equality result is not a TT encoder equality result.
3. Check chunked versus whole prefill and multi-turn continuity with mixed text,
   image and video requests, partial visual chunks, slot remaps, cancellation,
   and enough decode steps to exercise async reloads. Confirm text regression.
4. Exercise the actual OpenAI-compatible API for image URL, image data URL,
   video URL and video data URL. Capture HTTP errors and model revision. Reuse
   existing media-fetch policy; do not add an alternate unaudited URL fetcher.
5. Measure memory headroom, first/cached media TTFT and text B16 throughput on one
   TP4 replica before eight-worker deployment. Update immutable release pins only
   after these checks pass. The existing launcher does not select this prototype.

## Timeline

- Scope: identified the text-only adapter and reusable native vision blocks;
  verified checkpoint vision geometry and read exact installed HF source.
- Implementation: added selective loader, frame-window forwarding, request plans,
  chunk embedding replacement, separate rotary/KV positions and opt-in adapter.
- CPU checks: the first reference suite passed 25 tests; expanded production-body
  lifecycle tests reached 40. Further checks added no-identity-payload rejection
  and the BF16 loader case. Final counts and exact source hashes are in evidence.
- Repository checks: replaced direct exception assertions with the repository
  `expect_error` fixture after its pre-commit rule identified the convention.
- Hardware and endpoint qualification: pending; no device work was started.
