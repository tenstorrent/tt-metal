# BFP4 KV-cache applicability audit

2026-11-13, independent CPU-only source inspection. No source edits, TTNN/device
imports, builds or hardware results. This is a proposed experiment, not evidence
that BFP4 satisfies accuracy or improves latency.

**Applicable:** this checkout explicitly admits BFP4 caches throughout the paged
attention path. At initial inspection the TP4 decoder did not honor its `kv_dtype` policy when
allocating K/V, and its batched prefill helper separately restricted dtype to BFP8.
Changing only a JSON policy would therefore have produced misleading BFP4 evidence.
At handoff, a fresh source read confirms the parent has applied the two hooks
below: `create_paged_state` now allocates both caches using validated BF8/BF4
policy dtype, and batched fill admits matching BF8/BF4 tensors. Their hardware
quality and performance remain to be established; this audit did not run them.

## Minimal coherent hook (items 1–2 now applied by parent)

1. In `tt/multichip_decoder.py:588–599`, resolve a bounded cache dtype from
   `self.policy.get("kv_dtype", "bfloat8_b")` and use it for **both** key and value
   allocations. Start with BF8/BF4 (BF16 may be a diagnostic control). Keep
   `[physical_pages,1,32,256]`, TILE, DRAM, local one-KV-head ownership and the
   current page-table padding/lease contracts unchanged. Log actual resulting
   tensor dtype, padded shape, tile/page bytes, physical pages and allocated
   buffer size before running numerical gates.
2. In `tt/batched_cache_fill_candidate.py:32–35`, replace the hardcoded BF8
   assertion with a supported cache-dtype check plus equality of all four
   K/V/cache tensor dtypes. The helper calls native raw tile-copy fill and has
   no hardcoded byte stride. Keep certified ownership and batch-index checks.
   Disabling batched fill would compare a different family and is unnecessary.
3. Existing `tt/flat_prefill.py:87–88` and inherited
   `tt/optimized_decoder.py:1390` already cast padded prefill K/V to
   `state.key.dtype`. Both K and V must therefore use the same dtype in this
   minimal candidate. Do **not** cast decode update inputs to BFP4: native update
   takes BF16/FP32 activations and repacks the destination cache.
4. Update `tests/optimized_multichip_capacity.py:104` to count the actual
   full-attention policy dtype instead of literal BFP8 before publishing a
   capacity result. Do not change the frozen optimized decoder or accepted
   reference outputs. Keep all other selected precision/fidelity policies fixed
   for the first paired comparison.

## Native support and physical contract

Paths in this section are below `ttnn/cpp/ttnn/operations/`.

- `experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:31–46`
  explicitly accepts BF4 and requires input dtype equal to cache dtype. It is a
  raw tile copy; its geometry checks compare complete physical page element
  counts, with 32-aligned effective block size and head width. Current local
  page32/head256 satisfies these rules.
- `experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:63–67`
  accepts BF4 cache; lines205–210 require BF16/FP32 update input; lines221–246
  require matching K/V cache dtypes and disjoint equal-sized input core sets.
  Current Q/K/RoPE output and V remain BF16, with K on offset32 owners and V on
  first-B owners, so this needs no layout workaround.
- `.../paged_tiled_fused_update_cache_program_factory.cpp:147–158,241–284`
  derives cache tile size/format independently from input/intermediate format.
  `experimental/paged_cache/device/kernels/compute/paged_fused_update_cache.cpp`
  untilizes the old cache block, lets the writer update its selected row, then
  tilizes to the destination cache format. Exercise repeated updates within a
  page, page transitions and untouched-row preservation; a successful prefill
  pack alone does not validate this repack path.
- Prefill `transformer/sdpa/device/sdpa_device_operation.cpp:43–49` and decode
  `transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:40–46` admit
  BF4 K/V. Decode factory lines498–554 and prefill factory lines1740–1764 derive
  K and V formats/tile sizes separately. This is actual format-aware lowering,
  not just a permissive front-end enum. It still needs device/profiler proof.
- A BF8 tile is1088B and BF4 tile576B, including64B shared exponents. A physical
  page has8 tiles:8704B versus4608B per K or V buffer. The page table continues
  to contain logical physical-page indices, not byte offsets; do not halve its
  indices or change page32 when switching dtype.

## Quality and adaptation plan

Use the current TP4 selected-policy BF8 path as the paired control, with exactly
the same real weights and recorded text activations. The existing harness gates
output, contribution, every user's output and **both persistent states** at
PCC>=.995 (`tests/run_multichip_decoder.py:220–239`), plus changed-input trace
state comparisons (lines464–465) and stress state comparisons (lines549–550).
Keep these thresholds and references unchanged. Record K/V PCC separately by
rank and active user/page as useful diagnostics; padded empty capacity must not
hide a specific failing KV component.

Start with full-attention layer3 B1/S512 and B32/S33. Existing recorded inputs
are `doc/optimized_decoder/text_activations/text_l3_b1_s512.pt` and
`text_l3_b32_s33.pt`; an existing short text reference is
`doc/multichip_decoder/bfp4_v2_ref_l3_32x33.pt`. Verify input hashes against each
chosen reference; do not pair a text fixture with a random-input performance
reference. Extend to B3/S31, B17/S33, B32/S513 (generate a matching recorded-text
fixture/reference if absent), continuation lengths1/31/32/33, shuffled pages,
page31→32 and511→512 transitions, mixed positions and changed traced input.
Use B1/128-step and B32/512-step repeated decode stress to expose cache-repack
drift. GDN has no KV cache; its actual selected path remains a control, not a
reason to skip full-attention cache gates.

If an API/layout error occurs, retain it and adapt before rejection:

- Raw-fill dtype mismatch: cast the already internally padded BF16 K/V to the
  cache dtype; do not alter public sequence alignment.
- Batched-fill assertion: apply the matching-dtype helper admission above, then
  prove both certified bulk and exposed-table ordered branches. Keep cache
  ownership/alias tests.
- Decode update input dtype rejection: preserve BF16 inputs and let update pack
  into BF4; never pass packed low-precision input to its untilize path.
- Native kernel/CB failure: compare isolated native BF4 fill, update and SDPA
  with the exact local shapes; tune legal SDPA chunk/grid and supported update
  compute config, retaining same BF4 cache and whole-chain comparison. Use
  AutoDebug/AutoFix if a genuine lowering failure remains.

A numerical cache PCC failure is different from an API failure. Record direct
BF16→BF4 versus BF8 cache reconstruction on the same actual post-RoPE K/V,
then check fresh prefill and repeated native decode packing. Increasing SDPA
fidelity cannot recover bits already lost in the persistent cache. If only one
of K/V fails, an asymmetric family is technically possible: separate policy
types, independent prefill casts, separate native `paged_update_cache` calls
(each accepts BF4), and SDPA's distinct K/V formats. That family must include
the extra update dispatch and any helper/lease changes in its timing; the fused
update requires equal dtypes. If both components demonstrably fail the unchanged
cache gate across representative real inputs, do not adopt BF4 based solely on
passing decoder output or relax cache-state comparison. Such reconstruction
evidence is a meaningful precision rejection; no speedup is worth a failed gate.

## Capacity and profiler evidence required

At B1/context262144 there are 8192 physical pages per layer, with no extra
16-column rounding. Across 16 full-attention layers and 2 caches, per-device KV
payload changes from 2,281,701,376 B to 1,207,959,552 B: **exactly 1 GiB saved per
device**. This is source-derived payload arithmetic, not an allocator peak or a
new feasible context claim. Recurrent state, weights, page tables and activation
reserve do not shrink. BF4/BF8 pure-KV ratio is 576/1088, not exactly one half.

The model already supports its advertised 262144 context with BF8. Keep that
public maximum; do not infer a larger maximum from payload savings. If BF4 is
kept, update `doc/context_contract.json` with actual cache dtype/layout, measured
per-device allocations and fresh validation. The previous unchanged-dtype 32K
optimization smoke alone cannot establish long-context BF4 quality. Validate
near-max 262143, max 262144 and high-position decode with the same meaningful
reservation and state gates, or explicitly leave the new dtype unaccepted.

Collect paired profiler reports using `tests/profile_multichip_search.py`,
separate prefixes/manifests for BF8 and BF4 so artifact names do not collide.
Use real text inputs and unchanged policy except KV dtype. Confirm the profiled
fill/update/SDPA operands and descriptors actually contain BFP4, allocated cache
pages are 576 B tiles, and input updates remain BF16. Report typecast, raw fill,
fused update, prefill SDPA and decode SDPA rows plus aggregate whole-decoder
prefill/traced-decode latency. Inspect bytes/bandwidth advice with actual packed
tile bytes; compare short and long contexts because short decode may not be
cache-bandwidth dominated. Run watcher/allocation tracking separately. A policy
label, nominal half-byte estimate, or isolated cache-copy timing is insufficient.
