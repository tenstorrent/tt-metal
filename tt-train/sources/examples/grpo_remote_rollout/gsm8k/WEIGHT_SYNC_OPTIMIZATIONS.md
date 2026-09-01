# Cutting `MeshSocketWeightBridge` sync time toward 0.5 s

Context: on Qwen3-0.6B, `WeightSyncCallback_time_s` currently averages ~0.85 s
per push with `MeshSocketWeightBridge` (down from ~3.9 s with `HostWeightBridge`
over MPI). Target: **≤ 0.5 s per push**.

Getting from 0.85 s → ≤ 0.5 s means cutting ~350 ms. That's realistically 3-4
stacked optimizations, not one silver bullet. Ranked list with rough savings
estimates and concrete change shape for each.

## Where the ~0.85 s is going (estimate)

Rough per-push breakdown, both ranks:

- `qwen3_weights_ref_hf_dict` on sender: ~150-250 ms (56 `ttnn.slice` device
  ops + ~200 Python param lookups).
- `_send_manifest` (JSON over MPI): ~5-20 ms.
- Loop of ~200 `send_async` calls: ~100-150 ms of Python + pybind dispatch.
- Sender `synchronize_device` (wire drain): ~150-300 ms.
- Receiver `_recv_manifest`: ~5-20 ms.
- Receiver 200 × `allocate_tensor_on_device`: ~50-100 ms.
- Receiver 200 × `recv_async` dispatch: ~100-150 ms.
- Receiver `synchronize_device` (mostly overlaps with sender's drain).
- Receiver `worker.update_weights` (the `copy_to_buffer` convert cascade):
  ~200-400 ms.
- `bridge.barrier()`: small (< 5 ms).

Total ~ 800-1400 ms brackets the observed 850 ms. The two big fat bars are
`worker.update_weights` and the sender's dict-build + slice ops.

## The high-leverage stack (do these, in order)

### 1. Skip `worker.update_weights` — write directly into the model's weight buffers

**Expected saving: 200-400 ms.** The single biggest win, and probably the
difference between 0.85 s and ~0.5 s all on its own.

Currently the flow is:

- `recv_async(scratch_tensor, socket)` → scratch tensor allocated per key
  per push.
- `worker.update_weights` walks `hf_dict` and calls
  `copy_to_buffer(scratch, self.wqkv, ...)` per param, which fires up to 4
  device ops per tensor (`to_layout` / `typecast` / `reshape` /
  `to_memory_config` — the last one is the `interleaved_to_sharded` that
  triggered the L1 clash we saw during the FIFO bump; see
  [`models/common/utility_functions.py`](../../../../../models/common/utility_functions.py)
  around line 1257).

For Qwen3-0.6B that's ~200 tensors × several ms of program dispatch each.
Reproducibly hundreds of ms per push.

Shape of the fix:

- At `connect()` time (or first `send_weights`), build a **destination map**
  `hf_key → live_ttnn.Tensor` from `worker.models[0]` on the receiver. That
  map is the *actual* model weight handles: `self.wqkv`, `self.wo`, per-layer
  norms, etc.
- Change `receive_weights` at [`mesh_socket_bridge.py:210-241`](../utils/mesh_socket_bridge.py)
  to `recv_async(dest_map[key], self._socket)` — the socket writes straight
  into the model's buffer.
- On the sender, `qwen3_weights_ref_hf_dict` has to produce tensors in
  **exactly** the layout/dtype/shard that `dest_map[key]` expects (e.g.
  sharded L1 for `wqkv`, DRAM interleaved for norms). One-time conversion on
  the sender caches the shape; subsequent pushes reuse the shaped output
  buffers.
- `on_weights_received` shrinks to a no-op (or bookkeeping only). No
  `copy_to_buffer` cascade.

Caveats:

- Sharding metadata has to survive the socket. Today the receiver
  `allocate_tensor_on_device(spec, target)` allocates a plain DRAM interleaved
  tensor; you'd need the destination buffer's memory config, not a fresh one.
  The socket API writes bytes into whatever buffer you hand it, so this should
  just work as long as the sender's byte layout matches the destination's byte
  layout.
- Sender has to be reshuffled to produce tensors in destination layout. For
  most params that's TILE bf16 DRAM (matches today). For `wqkv`, the
  destination is sharded L1 — the sender has to shard-out before sending.
  That shard-out is a one-time program compile + cached; per-push cost is a
  single `to_memory_config(cached_wqkv_send_buffer)` call.

This is the single change with the largest expected impact. Worth doing
before anything else.

### 2. Cache manifest + receive-side allocations across pushes

**Expected saving: 80-150 ms.** Trivial change; do it in the same commit as
#1.

Current per-push cost on the receiver at
[`mesh_socket_bridge.py:210-231`](../utils/mesh_socket_bridge.py):

- `_recv_manifest` — one MPI round-trip for a JSON payload. Skip after first.
- `ttnn.allocate_tensor_on_device(spec, target)` × ~200 tensors, per push.
  Each allocation is a device call.

Fix:

- Send the manifest once during `connect()` (or lazy first-time), receiver
  caches the parsed spec + the allocated destination tensors.
- Subsequent `send_weights` skips `_send_manifest`; subsequent
  `receive_weights` skips both `_recv_manifest` and the allocation loop and
  goes straight to `recv_async(cached_tensor)`.

This also composes with #1: if #1 already replaced allocations with
model-weight handles, this optimization mostly captures the manifest MPI
round-trip and validation costs.

### 3. Kill the 56 `ttnn.slice` ops in `qwen3_weights_ref_hf_dict`

**Expected saving: 80-200 ms.** Sender-side.

Right now the ttml side has a fused `kv_proj` (K rows then V rows), the HF
export splits it into `k_proj` and `v_proj` via two `ttnn.slice` calls per
layer = 56 slice programs per push at
[`qwen3_overrides.py:127-128`](../utils/qwen3_overrides.py). Then on the
receiver, `Attention._update_wqkv` re-fuses q/k/v back together. Fuse →
split → re-fuse is pure waste.

Two options:

- **Send `kv_proj` as one tensor**, invent a bridge-only key like
  `model.layers.{i}.self_attn.qwen3_kv_proj.weight`, and teach the receiver's
  Attention update path to accept the ttml-native fused kv directly. Zero
  slice ops. Requires a small tt-transformers update.
- **In-place slice into cached destination buffers.** Allocate the two output
  tensors once, then per push `ttnn.copy` slices from `kv_proj` into the
  cached `k_out` / `v_out` (or use `ttnn.slice` with `output_tensor=cached_k`
  if that overload exists). Same output, no fresh allocation per push,
  program cache hits every time.

The first is a cleaner architectural fix; the second is easier if you're
time-boxed.

### 4. Concatenate small tensors before send

**Expected saving: 50-100 ms.** Sender + receiver.

There are ~200 `send_async` calls per push in
[`mesh_socket_bridge.py:201-202`](../utils/mesh_socket_bridge.py). At
~0.5-1 ms Python + pybind dispatch each, that's 100-200 ms of pure dispatch
overhead. Most of it is for tiny norm gammas (a few KB each).

Group by natural chunks — e.g. all norms across all layers into one buffer,
all `q_proj` across all layers into one buffer, etc. — and issue ~20-30
`send_async`s instead of 200. Same total bytes on the wire, ~10× less
dispatch. Receiver splits back on landing, which can also be `ttnn.slice`
into cached destination buffers (see #1).

Cleaner variant: pack everything into **one flat buffer per push** (or a
small handful) and send that. Receiver has a static offset table (established
at `connect`) that says which bytes belong to which model weight. Requires
disciplined byte-layout agreement, but this is where you take dispatch
overhead to ~zero.

### 5. Move the socket endpoint to an eth core (not the Tensix `(0,0)`)

**Expected saving: 20-100 ms, unlocks bigger FIFO.** Only meaningful if wire
time is a real chunk of the remaining budget.

Once you've done #1-#4, the callback might be at 0.4-0.5 s of which maybe
100-200 ms is genuine wire transfer. At that point:

- Use `mesh_device.get_active_ethernet_cores()` (or the equivalent on the
  receiver) to pick an ETH core for the socket endpoint at
  [`mesh_socket_bridge.py:171-172`](../utils/mesh_socket_bridge.py) instead
  of `CoreCoord(0, 0)`.
- Then bump `_DEFAULT_FIFO_SIZE_BYTES` back up — 128 KiB or 256 KiB fits
  comfortably on ERISC L1 without touching Tensix compute programs.
- Same-shape bump also lets you experiment with **multiple `SocketConnection`s
  in one MeshSocket** — one per eth core, all serving the same socket.
  Doesn't hit the multi-socket-corruption bug (which was about multiple
  `MeshSocket` objects, per the comment at
  [`mesh_socket_bridge.py:13-15`](../utils/mesh_socket_bridge.py)), so if the
  fabric routes traffic across all provided connections you can get 2-4×
  the throughput on the wire without another API change.

This is the wire-side win. Do it last because it only matters once framework
overhead is squeezed.

## Structural moves (bigger conceptual changes)

### 6. Fire-and-forget push — overlap install with next training step

**Expected wallclock saving: hides 100-400 ms behind trainer work, doesn't
change per-push CPU time.**

Right now `push_weights` blocks on `bridge.barrier()` which waits for the
receiver's `update_weights` to complete. But the GRPO trainer's *next* action
is a backward + optimizer step, not another generation. So the receiver's
install could happen in parallel with the trainer's local compute, and only
the *next* `remote_generate` would need to wait for install.

Rough shape:

- `bridge.send_weights` returns as soon as the wire drain completes (no
  barrier).
- Track a `pending_install` future on both sides.
- `remote_generate` on the client fences on `pending_install` before
  submitting a request; `serve_forever` on the server fences before invoking
  `generate_fn`.

If the trainer's backward/optimizer takes ≥ update_weights time (typical),
install becomes free wallclock.

Combines multiplicatively with #1 (smaller install → less to overlap → still
a win if install lands anywhere > 0 ms).

### 7. Sparse or delta transfer

**Expected saving: proportional to unchanged bytes.**

Not all params change appreciably every step — norm gammas move slowly,
embeddings move slowly with LR = 1e-5. If you track a per-param staleness
bound (e.g. only push params whose L∞ delta since last push exceeds ε, or
only every N steps for slow-moving params), you send strictly less. Requires
bookkeeping, and correctness discipline (staleness bounds have to be smaller
than what breaks GRPO on-policy assumption).

Realistically this is a bigger project, only worth it after #1-#5 are
exhausted.

### 8. Increase `weight_sync_every`

**Expected saving: linear in the divisor, at the cost of policy staleness.**

`weight_sync_every: 2` cuts the effective per-step sync cost in half but
makes rollouts one step older. On GRPO with small-batch-per-step, going
from 1 → 2 is usually fine (the policy improvement per step is tiny),
1 → 4 might affect KL / advantage estimates. Worth pairing with the other
optimizations rather than substituting for them.

## Concrete plan of attack for ≤ 0.5 s

Do this in one branch:

1. **Instrument first.** Add `time.perf_counter()` brackets around each of
   the six phases (build hf_dict, send_manifest, send_async loop, sender
   sync, receiver recv loop, receiver update_weights, barrier) and print
   one line per push. 5-minute change. Rerun for 10 pushes, get the real
   breakdown. Everything above is estimated at ~35 ms accuracy; real
   numbers will refine the priority.

2. **Land items #1 + #2 together.** Direct-into-model recv + cached
   buffers. Expected: 0.85 s → ~0.45 s.

3. **If that's not enough**, land #3 (fuse kv) and #4 (concat small
   tensors). Expected: → ~0.30 s.

4. **If you still want more**, land #5 (eth-core endpoint + bigger FIFO)
   and consider #6 (fire-and-forget).

5. Only reach for #7 or #8 if 0.3-0.5 s isn't small enough for your rollout
   cadence.

Realistic outcome: after #1-#4, weight sync should be in the
**0.25-0.45 s / push** range, which translates to another 2-3 s / step of
trainer wallclock on top of the 2 s already gained by switching bridges.
