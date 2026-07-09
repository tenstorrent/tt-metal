# Socket-transfer debug repros

Minimal, self-contained reproducers for debugging cross-rank tensor transfer over
Tenstorrent **fabric MeshSockets** and **host-staged MPISockets**. Each `debug_*` folder is a
2-rank job launched by its own `runner.sh` via `tt-run` (rank 0 = sender, rank 1 = receiver),
on a T3000-class host (4× N300 = 8 chips, a 2×4 mesh).

All are configurable with `REPRO_NUM_TENSORS` and `REPRO_TENSOR_SHAPE` (defaults below).

| Folder | Devices (send → recv) | Tensors sent | Result | How the impl works |
|---|---|---|---|---|
| **`debug_mesh_socket`** | 4 → 4 | 4 | ❌ **CORRUPT** (`all tensors correct: False`) | **Fabric, 1 mesh + 4 sockets ("bigmesh_Nsock").** Both ranks open one `[1,4]` mesh. Four `ttnn.MeshSocket`s, socket *i* = single connection `(0,i)→(0,i)`. Sender streams sharded tensors over all 4 sockets via `send_async`; receiver issues **all** `recv_async` into `[1,4]` templates, one `synchronize_device`, then verifies each `(device,tensor)` shard. Reproduces the transfer corruption. |
| **`debug_mesh_socket_and_broadcast`** | 4 → 4 | 100 | ✅ works | **Fabric, single socket + broadcast.** One `ttnn.MeshSocket`, connection `(0,0)→(0,0)`: sender sends only device-0's shard (`send_async`); receiver `recv_async` onto device 0, then `ttnn.broadcast` fans it across its whole `[1,4]` mesh. Times all recvs + `synchronize_device`; `to_torch`/verify only after all recvs. |
| **`debug_mpi_socket`** | 4 → 4 | 100 | ✅ works *(needs the `create_socket`/`SocketType` `_ttnn` build)* | **Host-staged MPI (MPISocket), genuine 4→4.** `create_socket(SocketType.MPI)`; `sock.send(replicated [1,4] tensor)` emits one MPI message per device shard (4 total), `sock.recv(template)` fills all 4 — addressed by **rank**, no fabric data path. Two-phase: all receives, then `to_torch`; times recvs + `synchronize_device`. |
| **`debug_mesh_socket_2_6`** | 2 → 6 | 100\* | 💥 **FATAL at `open_mesh_device`** | **Fabric, single socket + broadcast — asymmetric.** Same 1→1+broadcast logic as `…_and_broadcast`, but sender `[1,2]` (1 board) / receiver `[1,6]` (3 boards). Dies during control-plane bring-up: `control_plane.cpp:2996 "one src to multiple dst chips … not supported yet"` — the lone sender board's local chip cables to two receiver boards. \*Never reaches the transfer, so no tensors are actually sent. |

## Notes

- **Transport:** `debug_mpi_socket` is host-staged MPI; the other three are fabric
  `MeshSocket` (`ttnn.experimental.send_async` / `recv_async`).
- **Why 4/4 works but 2/6 doesn't:** on a T3000 the four PCIe/local chips form a ring, so each
  local chip has two off-board ethernet links. A 2-board / 2-board (4/4) cut lands where each
  boundary chip has exactly one cross-mesh peer; a single-board mesh (2/6) fans out to two
  receiver-mesh chips, which the fabric control plane rejects — before any socket is created.
- **`debug_mesh_socket` corruption:** the "issue all `recv_async`, then one `synchronize`"
  cadence over 4 per-device sockets on one mesh returns a wrong shard (historically the 3rd
  streamed tensor on devices 0/1).

## Running

From `tt-train/sources/examples/grpo` (requires `TT_METAL_HOME` set):

```bash
bash debug_mesh_socket/runner.sh                 # 4→4, 4 sockets  → CORRUPT
bash debug_mesh_socket_and_broadcast/runner.sh   # 4→4, 1→1 + broadcast, timed
bash debug_mpi_socket/runner.sh                  # 4→4, MPISocket, timed
bash debug_mesh_socket_2_6/runner.sh             # 2→6  → FATAL (routing)

# override defaults, e.g.:
REPRO_NUM_TENSORS=8 REPRO_TENSOR_SHAPE=1,1,8192,4096 bash debug_mesh_socket/runner.sh
```

The three fabric folders use existing ttnn ops (no rebuild). **`debug_mpi_socket`** needs a
`_ttnn` build that exposes `create_socket`/`SocketType`:

```bash
cd $TT_METAL_HOME && cmake --build build_Release --target ttnn
python3 -c "from ttnn._ttnn.multi_device import create_socket, SocketType; print('ok')"
```

Each rank opens its mesh per `configurations/local8/{rank_bindings.yaml,mgd.textproto}`
(hardware-specific templates — adjust `TT_VISIBLE_DEVICES` / `device_topology` to your wiring).
