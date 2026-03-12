# §7.1 — Topology and Links

This section explains the two physical topology modes (`Ring` and `Linear`), how `num_links` affects throughput and L1 cost, how `cluster_axis` selects a plane in a 2D mesh, and practical guidance for choosing each parameter.

---

## Topology

Every CCL op accepts an optional `topology` keyword argument of type `ttnn.Topology`. For the complete `Topology` enum definition (all 5 values and their internal meanings), see [Ch1 §1.3 — `tt::tt_fabric::Topology`](../ch1_introduction/ttnn_ecosystem.md#tttt_fabrictopology). For most CCL operations you will only use `Linear` and `Ring`; the `NeighborExchange`, `Mesh`, and `Torus` values are used internally or by lower-level fabric ops.

For the physical Ring and Linear topology definitions and cable requirements, see [Ch1 §1.2 — Hardware Topology](../ch1_introduction/hardware_topology.md#inter-chip-communication-erisc-and-edm).

### When to choose each

| Condition | Recommended topology |
|-----------|---------------------|
| Hardware has wrap-around cables | `Ring` |
| No wrap-around cables | `Linear` |
| Large tensors, bandwidth-bound | `Ring` (bidirectional halves latency) |
| Small tensors, latency-bound | `Linear` (fewer setup costs) |
| AllReduce async (always bidirectional) | `Ring` |
| Pipeline-stage boundary (no wrap needed) | `Linear` |
| Default (let tt-metal decide) | `None` (auto-selects based on mesh shape) |

When `topology=None` (the default for most ops), `ccl_common.cpp:get_usable_topology()` inspects the tensor's mesh shape and the `cluster_axis` to pick the most appropriate mode automatically.

---

## num_links

`num_links` specifies how many ERISC (Ethernet RISC-V) channels are used in parallel for the CCL operation.

### Physical meaning

Each Wormhole device has up to 16 Ethernet ports, each served by one ERISC core. A single ERISC core manages one EDM (Ethernet Data Mover) instance. When `num_links=2`, two independent ERISC cores on each device participate, each owning a separate L1 buffer and semaphore set.

```
                    num_links=1            num_links=2
                   ─────────────         ─────────────────
Device A ETH ──►  [ERISC ch0]           [ERISC ch0]
                                         [ERISC ch1]
Device B ETH ◄──  [ERISC ch0]           [ERISC ch0]
                                         [ERISC ch1]
```

With `num_links=2`, tensor chunks are split across the two ERISC channels, doubling the peak Ethernet bandwidth available to the CCL op.

### Bandwidth scaling

Peak bandwidth scales approximately linearly with `num_links` up to the physical link count of the device. For a transfer of size `B` bytes across `N` devices with `L` links, the CCL transmission time is approximately:

```
t_ccl ≈ (N-1) × B / (L × link_bw_per_port)
```

where `link_bw_per_port` is the single-port Ethernet bandwidth (~12.5 GB/s for 100 GbE).

### L1 cost

Each active ERISC channel consumes L1 memory for its buffer. From `EriscDatamoverConfig`:

```cpp
std::size_t total_l1_buffer_space = hal::get_erisc_l1_unreserved_size();
std::size_t usable_l1_base_address = hal::get_erisc_l1_unreserved_base();
static constexpr std::size_t eth_buffer_size_bytes < 163000;  // hard assertion
```

For `num_links=L` the total ERISC L1 consumed is approximately `L × eth_buffer_size_bytes` per device side (sender and receiver each have their own buffers). This is ERISC L1, not Tensix L1 — it does not compete with your tensor storage.

### Choosing num_links

| Scenario | Suggested num_links |
|----------|-------------------|
| Small tensors (< 1 MB total) | 1 — link overhead dominates |
| Medium tensors (1–32 MB) | 1–2 — test both |
| Large tensors (> 32 MB) | 2–4 — bandwidth-bound |
| Already fusion-pipelined (Ch5) | 1 — fused pipeline fills one link |
| Debug / correctness testing | 1 — simpler trace |
| `None` (default) | Auto-selected; usually 1 |

Increasing `num_links` beyond the hardware physical link count has no effect — the runtime clamps it.

---

## cluster_axis

In a 2D device mesh (e.g., 4×8 = 32 devices arranged as 4 rows × 8 columns), a CCL op can run along rows or columns independently on each "strip."

`cluster_axis` selects which mesh dimension the CCL traverses:

- `cluster_axis=0` — CCL runs along dimension 0 (rows). Devices in each column perform an independent ring/linear gather.
- `cluster_axis=1` — CCL runs along dimension 1 (columns). Devices in each row perform an independent ring/linear gather.
- `cluster_axis=None` — CCL runs over all devices in the mesh as a flat ring/linear (default for single-axis meshes).

### Example: tensor parallelism on a 4×8 mesh

```
Mesh layout (row × col):

  col0  col1  col2  col3  col4  col5  col6  col7
row0:  D00   D01   D02   D03   D04   D05   D06   D07
row1:  D10   D11   D12   D13   D14   D15   D16   D17
row2:  D20   D21   D22   D23   D24   D25   D26   D27
row3:  D30   D31   D32   D33   D34   D35   D36   D37

cluster_axis=1 → 4 independent rings, each of length 8 (along columns)
  Ring 0: D00 → D01 → ... → D07
  Ring 1: D10 → D11 → ... → D17
  ...

cluster_axis=0 → 8 independent rings, each of length 4 (along rows)
  Ring 0: D00 → D10 → D20 → D30
  Ring 1: D01 → D11 → D21 → D31
  ...
```

This is how pipeline-parallel + tensor-parallel (PP+TP) training partitions the mesh: one axis for TP CCL operations and the other for PP data movement.

### cluster_axis and topology interaction

When `cluster_axis` is specified, `get_usable_topology()` in `ccl_common.cpp` uses the axis length to determine topology:

- If the axis has a physical wrap-around cable (Ring), `Ring` is selected.
- Otherwise `Linear` is selected.

You can always override with an explicit `topology=ttnn.Topology.Ring` or `topology=ttnn.Topology.Linear`.

---

## Topology selection gotchas

**Gotcha 1: Ring without wrap-around cable hangs indefinitely.**

The ERISC kernel sends the last chunk to the first device expecting an ACK. Without the physical wrap-around port, no ACK arrives. The EDM kernel busy-waits in its state machine until a timeout or reset. Always verify physical cabling before using `topology=ttnn.Topology.Ring`.

**Gotcha 2: Linear with num_links > 1 is safe but may not help.**

Linear topology routes only in one direction, so both ERISC channels carry the same directional traffic. For large tensors this still doubles bandwidth, but for small tensors the overhead of two channel setups dominates.

**Gotcha 3: Auto-topology can select Ring when you expect Linear.**

If your mesh is configured with wrap-around cables on some axes and not others, and you rely on `topology=None`, verify that `get_usable_topology()` is selecting what you intend by checking logs at `tt::LogOp` trace level.

**Gotcha 4: num_links=None does not mean "use maximum links."**

`None` selects the op's internal default, typically 1. It does not auto-maximize.

---

## Performance measurement tips

- Profile at the op level using `ttnn.tracer` or Metal device profiler to isolate CCL time from compute time.
- For bandwidth-bound ops, the bottleneck is `num_links × link_bw`; increase `num_links` until you stop seeing speedup.
- For latency-bound ops (small tensors), reducing `num_links` to 1 and switching to `Linear` often yields lower latency than `Ring` with multiple links.
- Measure end-to-end iteration time, not just CCL time, to capture overlap effects (see §7.2).
- When using fused ops (Ch5), `num_links=1` is almost always optimal because the pipeline already saturates one link while compute runs on the other half.

---

*Back to [Chapter 7 Index](index.md)*
