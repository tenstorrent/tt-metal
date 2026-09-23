# UMD remote-transfer core selection and fabric routers

This directory documents a latent bug in how host reads and writes reach remote
(non-MMIO) Wormhole chips while fabric routers are running, the workaround on
this branch, and the alternative that was landed instead. It exists so the
mechanism is not rediscovered from scratch the next time the symptom appears.

## Symptom

After PR #48280 (express-link fabric routing), T3K tests with the watcher
enabled slowed down about 30x and hit the 300 s pytest timeout. Watcher polls of
remote chips 4 to 7 took 30 to 50 s each instead of 30 to 70 ms. A watcher-on
upload of an 8192x131072 bfp8 weight to a 1x8 `FABRIC_1D_RING` mesh took 199 s
instead of 5 s. Fabric init, the data path and host-side work were all normal.
The slowdown was constant per device until the next fabric re-init.

## Mechanism

Every host read or write to a remote chip goes through UMD's
`RemoteCommunication` for that chip, which posts the request into the command
queue of one Ethernet core on the gateway MMIO chip and spins on that core's
response queue. The base Ethernet firmware on that core only services the
queue when the fabric router occupying the core yields to it.

Three properties combine into the bug.

1. **The core list is a mid-scan snapshot.** During topology discovery UMD
   walks each MMIO chip's Ethernet channels in index order and adds each
   trained one to a set. When the walk reaches a link to a chip it has not seen
   yet, it creates that chip's `RemoteCommunication` with a copy of the set as
   it stands at that moment (`topology_discovery.cpp`, `create_remote_device`).
   On T3K the remote partner sits on channel 8, so each remote chip's list is
   {0, 1, 8} or {6, 7, 8}. Only channel 8 links to the remote chip. Channels
   0, 1, 6 and 7 link to a neighbouring MMIO chip, so a request placed there is
   forwarded across further hops through cores running wait-for-idle routers
   that yield about every 45 ms. Channel 9, the other direct link, is scanned
   after the snapshot and never appears.

2. **Only writes move the index, and only by filling the queue.**
   `RemoteCommunication::active_eth_core_idx` selects the core. Reads never
   change it. `write_to_non_mmio` advances it when one call pushes four
   commands into the 4-entry queue. A write to a destination that is not
   32-byte aligned is split into 4-byte commands until it reaches alignment, so
   a write 16 bytes off alignment is 5 commands and rotates the index once.
   Replacing the list does not reset the index.

3. **The count is a pure function of the build's bring-up writes.** Every
   bring-up write starts with an empty queue (measured on three builds), so
   firmware speed plays no part. The final index is decided by the sizes and
   destination addresses of all remote writes issued during device init.

Measured per remote chip, watcher on, same repro:

| Build                                   | Queue fills | Index | Core            | Upload |
|-----------------------------------------|-------------|-------|-----------------|--------|
| last good commit in PR, `57ecf98b9a6`   | 177         | 0     | channel 8       | 5.9 s  |
| `origin/main` `d9c2c92d05c` (PR reverted) | 177       | 0     | channel 8       | 6.2 s  |
| #48280 `eaa4a2ac610`                    | 181         | 1     | channel 1 or 7  | 199 s  |

Read latency by local core, median: channel 8, 25 us; channels 0 and 6, 100 ms;
channels 1 and 7, 150 ms. A watcher poll is a few hundred reads per chip.

The four extra fills on #48280 come from one change. The
`FabricEriscDatamoverConfig` constructor made the multi-TXQ counter reservation
unconditional to keep the L1 layout stable. The same block sets
`router_buffer_clear_size_words`, so on Wormhole the three router-word clears
in `configure_fabric_cores` grew from 4 to 64 bytes. The clear of
`edm_local_sync_address` sits 16 bytes past a 32-byte boundary, so the wider
write is 5 commands, and four routers per remote chip receive it during the
phase that counts. Nothing about router timing is involved.

## Workaround on this branch: tell UMD which cores to use

The fabric builder already configures the dispatch-link routers to yield every
16 iterations "to service slow dispatch / UMD / debug tools"
(`ComputeMeshRouterBuilder::configure_for_dispatch`). Nothing told UMD to use
those cores. This branch closes that gap.

tt-metal changes (7 files):

- `tt_metal/fabric/fabric_builder.cpp`: record each device's dispatch-link
  channels in the builder context after routers are created.
- `tt_metal/fabric/fabric_builder_context.{hpp,cpp}`: store and expose them
  (`set_dispatch_router_chans` / `get_dispatch_router_chans`).
- `tt_metal/llrt/tt_cluster.{hpp,cpp}`: new
  `Cluster::configure_ethernet_cores_for_remote_transfers(mmio_device_id,
  channels)`. Keeps the requested channels whose Ethernet link connects
  directly to a non-MMIO chip, using the cluster descriptor's connection map,
  and passes those cores to UMD's existing
  `configure_active_ethernet_cores_for_mmio_device`. Falls back to all
  requested channels, then to all active channels, if the filter leaves
  nothing. An empty request restores all active channels. Wormhole silicon
  only, and only for MMIO devices that tunnel to remote chips.
- `tt_metal/impl/device/firmware/fabric_firmware_initializer.{hpp,cpp}`: after
  `wait_for_fabric_router_sync`, restrict each MMIO device to its dispatch
  channels. In `teardown`, after the router termination signal is written,
  restore all active channels.

On T3K this leaves exactly channel 8 for every remote chip. Every post-init
remote read goes through it regardless of how many queue fills bring-up
produced.

UMD prerequisite (`umd_reset_active_eth_core_idx.patch`, apply inside
`tt_metal/third_party/umd`):

```cpp
// RemoteCommunication::set_remote_transfer_ethernet_cores
remote_transfer_eth_cores_.assign(remote_transfer_eth_cores.begin(), remote_transfer_eth_cores.end());
// The active index was chosen against the previous list; a shorter list would make it out of range.
active_eth_core_idx = 0;
```

Without it the process aborts: the old index (1 or 2) is out of range for the
new one-entry list and `get_remote_transfer_ethernet_core` throws from `.at()`.
Confirmed on hardware:

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
```

The tt-metal change must therefore land after a tt-umd PR carrying this line
and a submodule bump. The same UMD PR should also take the `NON_MMIO` mutex in
the setter: `read_non_mmio` and `write_to_non_mmio` read the list and index
under that mutex, and the watcher thread can be mid-poll when fabric init swaps
the list.

Results with both applied, watcher on: upload 199.5 s to 5.2 s on #48280 and
199.8 s to 5.2 s on the subtorus-routing branch; zero reads over 20 ms after
init; three fabric config swaps (1D, 1D ring, 1D) with `all_gather` correct.

## What was landed instead

Commit "fix wh lb watcher timeout" on `nnyamagoudar/subtorus-routing-ccl-fix`
gates the clear-size assignment back under the TXQ check in
`FabricEriscDatamoverConfig`:

```cpp
size_t num_words_consumed_per_counter = tt::align(sizeof(uint32_t) * num_sender_channels, field_size);
if (this->sender_txq_id != this->receiver_txq_id) {
    this->router_buffer_clear_size_words = num_words_consumed_per_counter;
}
```

This keeps the layout reservation and returns the count to 177, so UMD lands on
channel 8 again. It needs no UMD change. It does not remove the dependence on
the count: any future change to the size, count or alignment of remote writes
during bring-up re-rolls a one-in-three draw on T3K. If the symptom returns,
check the rotation count first (see below), then consider landing this branch.

## Further UMD changes worth upstreaming

- Build each remote chip's core list from the channels that connect directly
  to that chip (`get_directly_connected_ethernet_channels_between_chips`),
  falling back to all active channels. This removes the multi-hop cores for
  every UMD user. On T3K alone it is not sufficient: it leaves channels 8 and
  9, and channel 9 runs a wait-for-idle router, so half the time reads would
  still cost about 100 ms. Combined with this branch it is complete.
- Reset the index and take the mutex in `set_remote_transfer_ethernet_cores`
  (above).

## Reproducing and diagnosing

Repro: open a 1x8 `FABRIC_1D_RING` mesh, upload a `[1,1,8192,131072]` bfp8
tensor sharded on dim 3, run `ttnn.linear`, time the upload. Run with the CI
watcher variables set in a subshell so they do not leak:

```
(export TT_METAL_WATCHER=15 TT_METAL_WATCHER_DISABLE_ETH=1 \
        TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1
 timeout 420 python3 upload_linear_repro.py)
```

Good is about 5 s, bad is about 200 s.

To see which core UMD is on, add a `log_info` in
`RemoteCommunication::update_active_eth_core_idx` printing `this`, the new
index and the core, rebuild with `ninja -C build_Release install` (plain
`ninja` does not refresh `build/lib`, which is where the runtime path points),
and count the lines per `this` before `Fabric initialized on N devices`. Map
translated coordinates to channels with the Wormhole SoC descriptor: channel 8
is (25,17), channels 0, 1, 6, 7 are (25,16), (18,16), (22,16), (21,16). If the
final index is not on channel 8, this is the bug.

A regression guard that times one remote read per remote chip after fabric init
and after a config swap, failing above a few milliseconds, would have caught
this at the first commit that changed the count. It does not exist yet.
