# D2H Metadata Forwarding

How `D2HStreamService` ships a small, per-transfer **inline metadata** blob from the device to the
host alongside the payload, and keeps it consistent across every socket in the mesh.

This complements [`d2h_stream_service.md`](d2h_stream_service.md) (the service overview) and
[`d2h_stream_service_architecture.md`](d2h_stream_service_architecture.md) (the layered design). It is
the reverse of the metadata path in `H2DStreamService`: H2D fans metadata **out** from the service
core to the workers, whereas D2H fans it **in** from a worker to the service core (host-ward).

## What it's for

Each `read_from_tensor()` can carry an extra `metadata_size_bytes` of out-of-band data next to the
payload — e.g. a shape/offset header, a sequence counter, or any side-channel the consumer needs per
transfer. Metadata is **opt-in**: `Config::metadata_size_bytes == 0` disables the whole path and the
single-argument `read_from_tensor(bytes)` is used.

Constraints (enforced at construction):

- `metadata_size_bytes > 0` requires `Config::worker_cores` **and** `Config::metadata_master_core`.
- `metadata_master_core` must lie inside `worker_cores` (it may be the only worker — a single-worker
  grid is allowed; the lone worker is then its own metadata master).
- `metadata_size_bytes <= socket_page_size` — the metadata travels as a **single trailing socket
  page** on the wire. Multi-page metadata is a future extension.

## Data flow

This is the **reverse** of `H2DStreamService`'s metadata path. In H2D the metadata originates on the
host and is fanned **out** from the service core to the workers. In D2H the metadata originates
on-device (replicated on the worker cores) and is fanned **in**: the designated metadata master worker
reads its own local copy and writes it to the service-core staging region the sender ships from.

```
device produces metadata, replicated on each worker core's L1 (worker_metadata_addr)
                                  │
                metadata master worker reads its OWN local copy and writes it
                                  ▼
                    service-core L1 (metadata_input_addr, one region per mesh coord)
                                  │
                    persistent sender reads → PCIe → D2HSocket → host
                                  │
                    read_from_tensor(bytes, metadata):
                    copies socket[0]'s metadata page out, then
                    asserts every other socket's page is identical
```

### 1. The metadata is produced on-device (replicated)
Each worker core holds an identical copy of the per-transfer blob in its own L1 at
`worker_metadata_addr`. (In the tests this is `write_worker_metadata()`, which calls
`detail::WriteToDeviceL1` for every worker core on every mesh coord — standing in for a real producer
kernel that would emit the bytes as a side effect of its work.)

### 2. The metadata master worker fans it in to the service core
Every worker core runs the **same** kernel (`persistent_d2h_worker.cpp`); the only role difference is a
runtime arg, `is_master`. There is **no cross-talk between worker cores** — no roll-call, no
peer-release multicast. Per transfer, every worker:

1. waits for `transfer_done` (backing unlocked),
2. writes its own backing-tensor slice,
3. (master only, `is_master == 1`) reads its **own replicated metadata copy** from local worker L1
   (`worker_metadata_addr`) and **writes** it to the service-core staging region
   (`metadata_input_addr`),
4. atomic-increments the service core's `write_ack_counter`.

Because the metadata is identical on every core, only one core needs to forward it — electing the
master avoids every worker racing to write the same bytes to the service core. The service core waits
for **all `num_workers` acks** before it streams, so the master writing the metadata *before* its own
ack is sufficient ordering: by the time the sender runs, the staging region is populated. No
inter-worker semaphore is needed.

### 3. Sender ships it to the host
The persistent sender (`persistent_d2h_sender.cpp`) reads the same service-core L1 region
(`metadata.metadata_l1_addr = metadata_input_addrs_.at(coord)`), pads it out to a full socket page,
and PCIe-writes it as the **trailing page** after the payload pages. The staging region is already a
full, zero-padded, NoC-accessible socket page on the sender's own core, so the sender writes it
straight to the host FIFO with no intermediate scratch-CB copy.

### 4. Host reads back + consistency check
`read_from_tensor(bytes, metadata)` drains the payload, then reads the trailing metadata page from
**socket 0** into the caller's `metadata` span, and for every other socket reads its trailing page and
`TT_FATAL`s on any mismatch:

> `D2HStreamService::read_from_tensor: metadata mismatch across sockets (socket index N)`

Because the metadata is replicated identically across every device's worker cores, every socket must
deliver byte-identical metadata. This guard is what the "D2H metadata consistency across sockets" work
added — it catches a device/worker that shipped a stale or divergent blob instead of silently
returning socket 0's copy.

## API surface

```cpp
D2HStreamService::Config cfg{
    .global_spec = global_spec,
    .mapper = create_mesh_mapper(...),
    .fifo_size_bytes = ...,
    .scratch_cb_size_bytes = ...,
    .worker_cores = worker_cores,                 // required when metadata > 0
    .metadata_master_core = metadata_master,      // required when metadata > 0; inside worker_cores
    .metadata_size_bytes = 64,
};

// Per transfer (metadata span size must exactly equal Config::metadata_size_bytes):
std::vector<std::byte> payload(service.payload_size_bytes());
std::vector<std::byte> metadata(service.metadata_size_bytes());
service.read_from_tensor(payload, metadata);   // or read_from_tensor(host_tensor, metadata)
```

Owner-only getters used to wire up the device workload (see the test harness for a worked example):
`get_metadata_master_core`, `get_worker_metadata_addr`, `get_metadata_input_addr(coord)`,
`get_write_ack_counter_addr(coord)`, `get_transfer_done_sem_addr`, `metadata_size_bytes`.

## Cross-process

The descriptor carries `metadata_size_bytes`, so a connector built via `D2HStreamService::connect()`
sizes its socket pages and metadata scratch automatically and reads metadata with the same
`read_from_tensor(bytes, metadata)` call. The connector has no MeshDevice — it only drains sockets —
so the **owner** is responsible for running the worker workload that produces the metadata.
Cross-process metadata is supported for **replicated** placements (per-shard size == global, identical
metadata on every socket); sharded cross-process recompose is out of scope.

## Tests

| Test | File | Coverage |
|------|------|----------|
| `D2HStreamServiceTest.Replicated_WorkerSync_Metadata` | `tests/ttnn/unit_tests/gtests/tensor/test_d2h_stream_service.cpp` | Baseline 2-worker replicated, 64 B metadata |
| `D2HStreamServiceTest.Replicated_WorkerSync_Metadata_SingleWorker` | same | 1-worker grid (lone worker is its own metadata master), 64 B metadata |
| `D2HStreamServiceTest.Replicated_WorkerSync_Sweep` (`*_meta_*` rows) | same | 16 B / 256 B / near-page (2544 B) metadata × chunking budgets on a 4-worker grid |
| `D2HStreamServiceTest.Sharded_WorkerSync_Sweep` (`*_meta_*` row) | same | 128 B metadata under sharded placements (per-shard verify) |
| `CrossProcessD2HStreamServiceFixture.ReplicatedMetadata` | `tests/tt_metal/distributed/test_cross_process_d2h_stream_service.cpp` | Owner produces metadata via worker-sync; connector reads payload + metadata over PCIe and verifies both |

Both the device-side fill pattern and the metadata pattern are deterministic
(`make_worker_fill_pattern`, `make_metadata_pattern`), so the host/connector reconstruct the expected
bytes without any side-channel. Run on a Tenstorrent device node:

```bash
cmake --build build_Release --target unit_tests_ttnn_tensor cross_process_d2h_stream_service_test
./build_Release/test/ttnn/unit_tests_ttnn_tensor --gtest_filter='D2HStreamServiceTest.*'
mpirun --oversubscribe -np 2 ./build_Release/test/tt_metal/distributed/cross_process_d2h_stream_service_test
```
