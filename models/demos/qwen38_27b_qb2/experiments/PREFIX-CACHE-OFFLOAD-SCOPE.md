# Prefix reuse and local SSD offload scope

October 10, 2026. This is an implementation scope, not a supported feature or
an enabled serving change. Compact GPQA and kernel qualification stay ahead
of this work. Preserve BFP8 KV, FP32 recurrent state and existing trace safety.

## October 11, 03:35 UTC: lifecycle coordinator and batched-transfer candidate

The isolated branch now contains [`tt/prefix_serving.py`](../tt/prefix_serving.py),
an execution-lease coordinator around the existing checkpoint codec and local
store. It accepts scheduler-owned private pages and slots, checks request
generations and tenant/cache-salt identity, follows complete slot permutations,
and publishes an exact consumed frontier only after all-rank completion.
Cancellation can mark a request during a transfer. A corrupt-storage restore
resets and fences its private recurrent slot before cold fallback; a device
failure instead quarantines the cache and retains outstanding host buffers.
Multimodal prefix reuse is explicitly rejected until media/processor identity
and M-RoPE state are represented in the checkpoint.

An opt-in `PackedCacheTransfer(..., batched=True)` candidate submits shuffled
page windows in groups with at most 1 MiB of host transfer buffers before each
fence. It retains the existing exact packed-byte path and convolution neighbour
preservation. The qualified serial default is unchanged. This candidate still
issues individual page copies and reads before restoring writes; it does not
yet implement a device gather/scatter mover or prove PCIe/SSD bandwidth.

**38 CPU tests pass.** The asynchronous fake runtime verifies a 32-window read
with one fence instead of 32, exact all-rank restore through arbitrary codec
chunks, and buffer retention after failed completion. This is a mechanism
test, not measured TT transfer performance. No new hardware run has started.
The existing physical transfer test accepts `QWEN_PREFIX_BATCHED_TRANSFER=1`;
the bounded continuation controller accepts `--batched-transfer` and requires
the physical receipt to prove the same mode and source before continuation.

The coordinator is **not wired into the serving runner yet**. The pinned TT
runner has slot-move/release hooks but no KVConnector execution path. Its
attention-only prefix capability cannot represent this hybrid checkpoint.
Long-prefix hits under chunked prefill still need scheduler matched-token
admission and complete private allocations before the worker restores KV/GDN.
Copying a longer prefix into only the first chunk's allocated pages would be
incorrect. Reuse vLLM block allocation and an existing storage connector where
possible; keep scheduler policy separate from the TT transfer lease.

Merge/default enablement remains conditional on repeated serving correctness,
including cancellation, eviction/reload, independent suffixes and slot reuse.
AgentX still waits for both prefix caching and SSD offload through serving.
The user now gates another full GPQA on measured B16/32K/TP4 decode reaching
25 TSU. Prefix reuse targets repeated-prompt TTFT and idle-session capacity;
it does not increase native steady-state decode TSU.

## October 11 update: physical transfer and full-model continuation passed

The isolated branch now includes the opaque-byte TT adapter in
[`tt/prefix_transfer.py`](../tt/prefix_transfer.py). Twenty-five CPU tests pass.
All-rank hardware transfer, shuffled physical pages and neighbour preservation
passed, followed by four-layer and **full-64-layer TP4 continuation**. The full
test restored a 4096-token prefix into another slot, appended 32 tokens, and
matched every logit exactly through 32 teacher-forced decode steps. A second
restore at the 4160-token frontier preserved the existing trace, resident
addresses and neighbouring state, again with exact logits.

This proves the tested exclusive-lease boundary, not a serving integration or
a long-conversation eval. `supports_prefix_caching` remains disabled. There is
no scheduler admission/eviction/replica-affinity adapter or production storage
connector yet, and AgentX must wait for both caching and offload through serving.

The 282.8-MiB checkpoint took **2.207 s to capture and 3.542 s to restore**;
prefilling the same 4K prefix took **1.185 s**. Thus this first conservative
adapter is slower than recomputation in that case. Randomly shuffled pages
required 16,768 transfer windows. Coalescing, async transfers and resident
prefix reuse need measurement before claiming a speedup. File-backed restore
may hit the host page cache; these are not physical SSD-bandwidth measurements.

[Receipts, failures and reproduction details](../galaxy-evidence/prefix-transfer-continuation-v1/README.md).
The original scope below is retained with the updated completion boundaries.

## Parallel implementation, October 10

The isolated branch `anatarajan/qwen38-prefix-offload-20261010` starts from
`6462756f9159d3d97b40c9781153b30eeedde7ad`. The 30-TSU decode goal and its hardware
queue retain priority. No active source snapshot or serving launch is changed.

- [Checkpoint codec and transfer leases](../tt/prefix_checkpoint.py) now define
  a complete KV/GDN/conv frontier across all TP ranks. The format keys exact
  consumed token IDs, model/implementation/config identity, layout and tenant
  namespace. It streams opaque packed bytes with per-segment SHA256 checks;
  physical page IDs and pointers never enter the checkpoint.
- [Local-file reference backend](../tt/prefix_storage.py) implements immutable,
  atomically published blobs, an explicit shared quota, restart-readable files
  and explicit eviction. Incomplete crash files count against capacity and
  cannot become hits. The operator supplies a deletable directory on local
  disk or a bounded RAM filesystem. This is a correctness backend, not an
  LMCache connector or a replacement for vLLM's scheduler/block manager.
- [CPU tests](../tests/unit/test_prefix_checkpoint.py) cover independent request
  branches, all-rank opaque-byte round trips with destination page remapping,
  bad/truncated payloads, missing completion records, export/transfer failure,
  cancellation, stale identity, concurrent quota admission, eviction leases
  and reading from a new process. These use a fake device and real filesystem;
  they do not qualify TT DMA, numerical continuation, traces or model accuracy.

The reference backend serializes transfer/eviction while holding a file lease
to avoid undercounting an unlinked-but-still-open inode against its quota.
Concurrent storage leases and asynchronous TT transfers remain production
integration work. Codec staging is at most 1 MiB per active operation, excluding
storage buffers, device adapters and source snapshots. SSD bandwidth/latency is
not measured by these small CPU tests.

Next implementation gates:

1. Implement the TT capture/restore adapters against these leases (the initial
   serialized adapter and exclusive-lease hardware boundary now pass). Fence and
   publish resident decode-bucket state before capture, retain the exact
   consumed frontier, gather logical KV pages without BFP8 conversion, and
   restore into private pages and stable recurrent slots. Cancellation must
   abort or quarantine partially written destinations.
2. Extend the passing all-rank hardware test to long cold/restored continuations, divergent
   suffixes, shuffled pages, slot reuse and existing decode traces. Keep the
   capability disabled until these pass.
3. Extend the pinned plugin's hybrid allocation/lifecycle hooks, reusing vLLM
   prefix/block management. Then connect the checkpoint store to an existing
   host/SSD backend and add admission/eviction policy and replica affinity.

Expected benefit is avoided repeated prefill, not a native decode speedup.
For context, the fixed B16/32K test measured about 95.8 seconds of prefill for
the batch. Hit benefit is the work skipped minus export/restore, suffix prefill
and scheduling costs; no hit-rate or TTFT speedup is qualified yet. Keep SSD
traffic off the per-token decode path.

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
- A standalone TT transfer/checkpoint adapter now exists, but no serving
  connector or cross-replica prefix placement is implemented in this Qwen
  adapter. Existing capture backups are transient warmup protection.

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
into 30 TSU. No measured serving hit-rate or engineering completion date is
claimed. The measured restore latency above does not yet justify an SSD speedup.
