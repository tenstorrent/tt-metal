# D2HStreamService

Device-to-host mirror of `H2DStreamService`. Worker cores write a fixed **backing device tensor**; a persistent **sender kernel** on each service core streams pages to host FIFOs via **D2HSocket**; the host calls **`read_from_tensor()`**.

```
Workers → Backing DRAM → persistent_d2h_sender → host FIFO → read_from_tensor()
```

| H2D | D2H |
|-----|-----|
| `forward_to_tensor` | `read_from_tensor` |
| `persistent_h2d_receiver` | `persistent_d2h_sender` |
| `worker_cores` / `data_ready_sem` | `worker_cores` / `transfer_done_sem` |
| `consumed_counter` (workers ack after consume) | `write_ack_counter` (workers ack after produce) |

## Worker-sync handshake

When `Config::worker_cores` is set:

1. **Service core unlocks workers** — multicast `transfer_done_sem` so workers may write backing DRAM.
2. **Workers write** their page slices. Every core runs the same `persistent_d2h_worker.cpp` kernel;
   role is selected by an `is_master` runtime arg, and there is **no cross-talk between worker cores**.
3. **If metadata is enabled** — the designated **metadata master** worker (`metadata_master_core`)
   additionally, before its own ack:
   - reads its **local replicated** metadata copy from worker L1,
   - writes it **in** to the service-core staging region the sender ships from (fan-in, host-ward).
4. **Every worker acks** — atomic-inc `write_ack_counter` on the service core (one inc per worker).
5. **Service core streams** backing (+ metadata page) to host FIFO; host calls `read_from_tensor()` + `barrier()`.

The service core waits for all `num_workers` acks before streaming, so the master writing metadata
before its ack is sufficient ordering — no inter-worker handshake is required. When
`metadata_size_bytes == 0`, no worker is the master — every worker just writes + acks.

## Host-only path

When `worker_cores` is unset, the host writes backing via `copy_to_device`, then `read_from_tensor()` (which calls `notify_backing_ready()` internally to bump `write_ack`).

## Python API

```python
service = ttnn.D2HStreamService(
    mesh_device=mesh_device,
    global_spec=global_spec,
    fifo_size_bytes=fifo_size_bytes,
    scratch_cb_size_bytes=scratch_cb_size_bytes,
    worker_cores=None,
    metadata_master_core=None,  # required only when metadata_size_bytes > 0
    metadata_size_bytes=0,
)
```

## Cross-process connector

Owner calls `export_descriptor(service_id)`. Connector attaches with `D2HStreamService::connect(service_id)`.

Descriptor path: `/dev/shm/tt_d2h_stream_service_<id>.bin`

## Hardware tests

C++ (from repo root, on a device node):

```bash
cmake --build build_Release --target unit_tests_ttnn_tensor cross_process_d2h_stream_service_test
tt-smi -r
./build_Release/test/ttnn/unit_tests_ttnn_tensor --gtest_filter='D2HStreamServiceTest.*'
mpirun --oversubscribe -np 2 ./build_Release/test/tt_metal/distributed/cross_process_d2h_stream_service_test
```

Python (host-only replicated sweep; needs `build_Release` ttnn + a venv on the **compute node**):

```bash
cmake --build build_Release --target ttnn
pytest tests/ttnn/unit_tests/base_functionality/test_d2h_stream_service.py --tt-arch=blackhole
```
