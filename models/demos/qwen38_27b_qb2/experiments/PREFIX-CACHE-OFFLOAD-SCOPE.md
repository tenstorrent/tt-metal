# Prefix reuse and local SSD offload scope

October 10, 2026. This is an implementation scope, not a supported feature or
an enabled serving change. Compact GPQA and kernel qualification stay ahead
of this work. Preserve BFP8 KV, FP32 recurrent state and existing trace safety.

## Current implementation

- `tt/generator_vllm.py` declares `supports_prefix_caching=False`, and
  `demo/galaxy_serving.py` passes `--no-enable-prefix-caching`.
- `ModelCache` owns full-attention KV pages plus per-request GDN recurrent
  and convolution state. KV pages already use scheduler-owned page tables.
- The generator supports nonzero prefill positions, recurrent-slot reset and
  on-device slot permutation. These support continued resident execution;
  they do not provide retained prefix checkpoints or eviction/restore.
- The actually pinned plugin is `b7e4292e4193cba20abe9c7c68ce489201b2e36b`,
  using vLLM 0.26.0. Its `src/vllm_tt_plugin/platform.py:1752` gates prefix
  caching on the model capability. `model_runner.py:491` accepts attention
  specs and uniform attention-spec groups; it does not accept a recurrent
  state group there. The old local inference-server submodule is a different
  checkout and must not be mistaken for the deployed pin.
- No TT KV-to-host/SSD connector or cross-replica prefix placement is
  implemented in this Qwen adapter. Existing capture backups are transient
  warmup protection, not a reusable prefix cache.

## Prefix cache: required work

1. Expose both full-attention and recurrent-state cache groups to the plugin
   and scheduler. Reuse vLLM's hybrid cache management and block hashing where
   the pinned APIs permit; extend the TT runner's allocation/lifecycle hooks.
   A capability flag alone cannot supply the missing state.
2. Add snapshot, fork and restore at an exact consumed-token frontier. A hit
   requires both the retained KV prefix and all 48 GDN layers' FP32 state and
   BF16 convolution history for that same frontier on all four TP ranks.
   Restore into stable per-request storage before appending the suffix.
3. Retain immutable shared KV blocks with reference counts and copy-on-write
   for mutable tails. GDN state is cloned into each continuing request; two
   branches cannot mutate a shared recurrent checkpoint. Align first-stage
   checkpoints to supported prefill/token-block boundaries and fall back to
   an earlier complete checkpoint when necessary.
4. Coordinate async decode completion, state packing/scattering, slot reuse,
   cancellation and eviction. A finished response may contain one sampled
   token beyond the consumed model state: record that frontier explicitly.
   Recompute a final token or retain the necessary output for exact-prefix
   hits; never claim nonexistent logits. New requests get their own RNG;
   paused-generation restore also needs RNG, positions and token history.
5. Add bounded checkpoint admission/eviction and cache-aware replica routing.
   Keys must distinguish exact token IDs, model revision, numerical policy,
   cache layout/TP geometry and tenant/cache-salt namespace. Do not silently
   reuse an incompatible snapshot after an implementation or precision change.

Upstream documents block-aligned hybrid-state prefix reuse, but support in
upstream GPU paths is not proof of this TT integration. Newer optional
sub-block Mamba checkpoint modes have additional restrictions; they are not
prerequisites for an initial aligned, non-speculative implementation.
[vLLM prefix-cache documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

One FP32 recurrent checkpoint has
`48 layers * 4 ranks * 12 heads * 128 * 128 * 4 bytes = 144 MiB`.
BF16 convolution history adds
`48 * 4 * 3 * (2*4*128 + 12*128) * 2 bytes = 2.8125 MiB`.
Thus each retained frontier costs about **146.8 MiB across TP4**, in addition
to the KV pages. These are state payload sizes, excluding metadata and staging.
Saving a frontier every 32 tokens of a 32K prompt would consume about 146.8 GiB
of recurrent/conv snapshots alone. Prefer a small number of useful complete
frontiers, not one state snapshot per attention page.

At 32K, the 16 attention layers' TP4 BFP8 KV payload is approximately
`16 * 2 * 4 * (32768/32) * (256/32) * 1088 bytes = 1.0625 GiB`, using the
current 32x32 BFP8 tile size including exponents. One complete prefix plus one
state checkpoint is therefore approximately **1.206 GiB** before allocator,
metadata, staging and duplicate-copy overhead. Shared KV pages can be retained
without copying them, but pinned checkpoints consume capacity otherwise usable
for active requests.

## Host RAM and SSD: additional work

- Export/import selected KV pages in their TT physical encoding, plus state
  checkpoints. Preserve BFP8 packed bytes and FP32 state exactly. Page IDs are
  allocation-local and must be remapped on restore; serialize logical order,
  layout/version and rank identity rather than treating old IDs as addresses.
- Add bounded pinned host staging and an asynchronous TT device transfer
  adapter. Fence completion before publishing a cache hit or reusing source
  buffers. Pack adjacent pages to avoid thousands of tiny transfers; measure
  transfer bandwidth and decode contention rather than assuming line rate.
- Reuse a storage backend for host RAM/local files, checksums, quotas and
  eviction. LMCache supplies CPU/disk storage infrastructure, but its TT
  device transfer and this hybrid checkpoint representation require separate
  integration validation. [LMCache storage configuration](https://docs.lmcache.ai/api_reference/configurations.html).
- vLLM's documented native OffloadingConnector currently supports CUDA,
  ROCm and XPU. It stages secondary-tier transfers through host RAM. Its
  filesystem tier is useful reference infrastructure, not a TT launch flag.
  [vLLM offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/).
- Store only in an explicitly bounded, deletable task-owned host directory.
  Use atomic completion records and reject partial/corrupt/incompatible cache
  entries. On restore failure, recompute the prompt rather than exposing a
  partially restored cache. No NFS, raw-disk formatting or native-install edits.

Keep active attention state resident during decode. SSD is for idle
conversations and reusable prefixes; reading active KV from SSD every token
would create a new bandwidth bottleneck. It does not increase native active
batch capacity at unchanged TSU merely by adding disk.

## Suggested order and acceptance

Device-resident prefix reuse first (substantial plugin/state work), host-RAM
round-trip next, then SSD using that same transfer/serialization interface.
Prefer one TP4 implementation/correctness gate before eight-replica routing.

Required checks: cold versus restored logits/tokens/state; common-prefix
branches with divergent suffixes; different hit lengths and chunk boundaries;
slot remaps, cancellation, eviction/reload and process restart; shuffled KV
pages, all ranks and inactive-user isolation; transfer interruption and stale
model/layout rejection; unchanged fixed-batch decode performance; fresh versus
warm TTFT and all-in throughput under realistic prefix hit rates. Restore
correctness must include long recurrent continuation, not only the first token.

Benefit is avoided repeated prefill and more retained inactive conversations.
It does not remove decode's per-token KV reads or turn the measured 20.03 TSU
into 30 TSU. No measured hit-rate, restore latency or engineering completion
date is claimed. Measure a host round trip before promising an SSD speedup.
