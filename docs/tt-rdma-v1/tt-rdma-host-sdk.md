# TT-RDMA-over-Ethernet — Host-Side SDK Design

**Status:** design, post Phase 3.2. New compilation unit `tt_metal/llrt/tt_rdma_endpoint.{hpp,cpp}` sits alongside `ExternalIfaceSender` and reuses its TX WQE-ring, hugepage mapping, and reliability layer. `ExternalIfaceSender` stays as the L2-raw primitive; `TtRdmaEndpoint` is the verbs façade.

---

## 1. API surface

```cpp
// tt_metal/llrt/tt_rdma_endpoint.hpp
#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <memory>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace tt::llrt {

class ExternalIfaceSender;

enum class AccessFlags : uint32_t {
    kLocalRead     = 1u << 0,
    kLocalWrite    = 1u << 1,
    kRemoteRead    = 1u << 2,
    kRemoteWrite   = 1u << 3,
    kRemoteAtomic  = 1u << 4,
};
constexpr AccessFlags operator|(AccessFlags a, AccessFlags b) {
    return AccessFlags(uint32_t(a) | uint32_t(b));
}

struct MrHandle {
    uint32_t rkey         = 0;   // network-scoped key
    uint32_t lkey         = 0;   // local-scoped key
    uint8_t  fw_table_idx = 0;   // FW MR-table slot (0..15)
    uint8_t  generation   = 0;   // bumped on dereg
    uint16_t _rsvd        = 0;
    bool valid() const { return rkey != 0; }
};

// Values MUST match the shipped wire protocol (tt-rdma-wire-protocol-v1.md §2).
// Family-ranged, NOT sequential: 0x0X SEND, 0x1X WRITE, 0x2X READ, 0x4X ctrl.
enum class WireOpcode : uint8_t {
    kSend     = 0x01, kSendImm  = 0x02,
    kWrite    = 0x10, kWriteImm = 0x11,
    kReadReq  = 0x20, kReadResp = 0x21,
    kAck      = 0x40, kControl  = 0xF0,
};

enum class CqStatus : uint8_t {
    kOk = 0, kBadRkey = 1, kBadLen = 2,
    kFlushed = 3, kRetryExc = 4, kRemoteErr = 5,
};

struct TxCqe {
    uint32_t   seq;
    uint32_t   cookie;
    WireOpcode opcode;
    CqStatus   status;
    uint16_t   _rsvd;
};

struct RxCqe {
    uint32_t   peer_seq;
    uint32_t   length;
    WireOpcode opcode;          // kSend or kWriteImm
    CqStatus   status;
    uint16_t   _rsvd;
    uint32_t   immediate;       // valid iff opcode == kWriteImm
    void*      data;            // landing pointer
    MrHandle   mr;
};

struct EndpointConfig {
    uint8_t  mr_table_size  = 16;
    uint32_t rx_ring_slots  = 64;
    uint32_t rx_slot_stride = 1536;
    bool     auto_reliable_one_sided = true;
};

class TtRdmaEndpoint {
public:
    TtRdmaEndpoint(ChipId chip_id, CoreCoord virtual_eth_core,
                   EndpointConfig cfg = {});
    ~TtRdmaEndpoint();

    bool bring_up(uint32_t link_timeout_ms = 10000);

    // Memory registration (addr must be hugepage-backed).
    MrHandle register_mr(void* addr, size_t len, AccessFlags flags);
    void     deregister_mr(MrHandle mr);

    // One-sided ops
    std::optional<uint32_t> post_send_write(
        std::span<const uint8_t> buf, MrHandle remote,
        uint64_t remote_offset, uint32_t cookie = 0);
    std::optional<uint32_t> post_send_write_imm(
        std::span<const uint8_t> buf, MrHandle remote,
        uint64_t remote_offset, uint32_t immediate, uint32_t cookie = 0);
    std::optional<uint32_t> post_send_read(
        MrHandle remote, uint64_t remote_offset, size_t len,
        MrHandle local_landing, uint64_t local_offset, uint32_t cookie = 0);

    // Two-sided ops
    void post_recv(MrHandle mr, uint64_t offset, size_t max_len, uint32_t cookie = 0);
    void post_recv_any(size_t max_len, uint32_t cookie = 0);
    std::optional<RxCqe> poll_rx_completion();
    std::optional<RxCqe> wait_rx_completion(uint32_t timeout_ms);
    void release_rx_slot(const RxCqe& cqe);

    // TX completion (typed)
    std::optional<TxCqe> poll_completion();
    bool wait_completion(uint32_t seq, uint32_t timeout_ms);

    // Back-compat: raw SEND
    std::optional<uint32_t> post_send(std::span<const uint8_t> buf, uint32_t cookie = 0);

    // Reliability
    void set_reliability(bool enabled);
    bool tick_retx();

    ExternalIfaceSender& underlying();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::llrt
```

Notable refinements over the initial sketch:
- `register_mr` requires hugepage-backed memory (NoC DMA needs IOMMU-pinnable pages).
- `MrHandle::generation` packed into low byte of `rkey` — detects dereg/reregister-into-same-slot race in late completions.
- `post_send_read` takes an MR + offset for landing, not a raw `void*`.
- `poll_completion` returns typed `TxCqe`; the bare-uint32 version stays on `ExternalIfaceSender` for soak tests.
- `post_recv_any` for the 90% case; `post_recv(mr, offset, len)` is verbs-faithful for exact-address recvs.
- `release_rx_slot` makes consume-then-release lifetime explicit.

---

## 2. MR registration mechanism

FW MR-table sized for 16 entries (covers control plane + a handful of bulk buffers; fits in 256 B; trivial expansion later).

L1 addition after RCB+0x40 (**32 B per entry — matches `tt-rdma-wire-protocol-v1.md §3` and the shipped FW MR table**; an earlier draft used 16 B):

```
0x8500–0x86FF  MR table: 16 entries × 32 B = 512 B
   +0x00 u64 base_noc_addr    target NoC address (PCIe/NoC tile encoding)
   +0x08 u64 length           bytes
   +0x10 u32 rkey             (slot<<24)|(rand16<<8)|generation; 0 = free
   +0x14 u32 access_flags     bit0=LOCAL_WRITE bit1=REMOTE_WRITE
                              bit2=REMOTE_READ bit3=REMOTE_ATOMIC
   +0x18 u32 pd               protection domain (0 = default)
   +0x1C u32 reserved         (SDK may stash state: 0=free,1=valid,2=zombie)
```

The `generation` byte lives in the low byte of `rkey`; the SDK's `state`/zombie
bookkeeping is host-side and can ride in the reserved word.

RCB additions at `+0x28`:

```
+0x28 u32 mr_table_addr        (= 0x8500, informational)
+0x2C u32 rx_host_noc_lo
+0x30 u32 rx_host_noc_hi
+0x34 u32 rx_ring_log2n        (= 6 for 64 slots)
+0x38 u32 rx_slot_stride       (= 1536)
+0x3C u32 reserved
```

Lifecycle:

1. `mmap(MAP_HUGETLB)` the buffer; SDK calls `PCIDevice::map_hugepage_to_noc` (same path as `enable_fw_dma_pull`).
2. SDK picks first `state==free` slot, fills entry, bumps `generation`, sets `state=1`. Single `cluster().write_core` for ordering.
3. `rkey = (slot_idx << 24) | (random16 << 8) | generation`. Random middle bits make rkey-spoofing harder; FW validates `slot_idx` + `generation` only.
4. `deregister_mr`: `state=2` first, wait one ring depth's worth of completions, then `state=0`. `generation` bump invalidates any straggling wire frame.

16-entry cap; no automatic eviction.

### How many MRs do I actually need?

**The FW MR table only holds *local* regions that a *remote* peer will target
(inbound WRITE / READ / ATOMIC).** Regions you *initiate against* do **not**
consume a slot: a `RemoteMr` (`tt-rdma-mesh-addressing-spec.md §5.1`) is a
host-side handle built from the peer's advertised `{rkey, base_vaddr}` and never
touches the FW table. So a node doing purely *outbound* WRITE/READ to many peers
needs **zero** FW MR slots — the requirement is driven by *how many distinct
local buffers this node exposes as targets*, not by how many peers it talks to.

| Deployment | FW slots required | Why |
|---|---|---|
| MVP / smoke | **1** | only slot 0 is regression-tested (`README.md` open-Q #5) |
| Disaggregated inference (KV-cache/weights target) | **~1–3** | one DRAM buffer per exposed region (`runtime-workload-coexistence.md`) |
| Native verbs provider, usable | **~12 of 16** | kernel reserves 4 for RxWqeRing + CQ ring + control (`tt-rdma-verbs-provider.md §5.3`) |
| NCCL / UCX | **> 16 → needs 64** | these pool/register per-buffer and exhaust 16 (`tt-rdma-verbs-provider.md` R1) |
| Mesh node, many peers | scales with (local regions exposed), **not** peer count | imported remote MRs are host-side only |

#### Why "many peers" doesn't multiply the count

An `rkey` is a password to a region: register once → one rkey → **one slot** →
anyone holding that rkey may access the region. How you hand the password out
decides whether peer count multiplies slots:

- **Shared-rkey (default):** register the buffer once, advertise the *same* rkey
  to every peer. **1 slot regardless of peer count.** No isolation between peers
  (all share the region), but that's fine because the gateway resolves peer
  identity from its `rkey → endpoint` table (`mesh-spec §3`, rkey pass-through) —
  so you know who sent a frame without spending a slot per peer.
- **Per-peer isolated:** register the same region once *per peer*, each with a
  distinct rkey; peer B never learns peer A's key, so the FW rejects cross-peer
  access (`rkey_miss` drop). **1 slot per peer × regions-per-peer** — this is the
  only model where peers multiply slots. It's a multi-tenancy choice, not a
  requirement of scaling peers.

| 10 peers, 1 exposed buffer each | Registrations | FW slots | Isolation |
|---|---|---|---|
| Shared-rkey | 1 | **1** | none (all 10 share) |
| Per-peer isolated | 10 | **10** | full (each walled off) |

When per-peer isolation *does* run out, the answer isn't a bigger table — it's
gateway-side rkey namespacing (`mesh-spec` open-Q #1), which keeps the TT table
small regardless of peer count.

**Target 64 for anything multi-tenant or NCCL-shaped.** The 16→64 bump is
free — the lookup is already O(1) direct-index (`slot = (rkey>>24)&0x0F`), so it
is just widening the mask to `&0x3F` and growing the table to 64 × 32 B = 2 KB
L1 (`0x8500–0x8CFF`, still below the WQE payload base at `0x9000`); **0 ns on the
data path** (`tt-rdma-mesh-addressing-spec.md §7.2`, milestone M7). 16 is fine for
inference-target and point-to-point; NCCL/UCX MR pools are the one workload that
demands the wider table.

---

## 3. RX consumer ring

`RxWqeRing` lives entirely in host hugepage memory; FW pushes via NoC→PCIe (reverse of Phase 3.2 pull).

**Sizing:** 64 slots × 1536 B = 96 KB. Same 2 MB hugepage as TX (TX 96 KB + RX 96 KB + MR metadata). Single `map_hugepage_to_noc`.

**RCB additions** (host-readable, FW-writable):

```
0x8440 u32 rx_prod_idx        ← FW writes
0x8444 u32 rx_cons_idx        ← host writes
0x8448 u32 rx_cq_head         ← FW writes (parallel to TX cq_head)
0x844C u32 rx_overflow_drops  ← FW increments on full-ring drop
```

**Per-slot header** (first 32 B of each 1536-B slot):

```
+0x00 u32 peer_seq
+0x04 u32 length
+0x08 u8  opcode
       u8  status
       u16 _rsvd
+0x0C u32 immediate           (kWriteImm only)
+0x10 u32 cookie
+0x14 u8  mr_table_idx        (0xFF for ring-slot)
       u8  flags              (bit0 = OWNED_BY_HOST)
       u16 _rsvd
+0x18 u32 reserved
+0x1C u32 reserved
+0x20 ... payload (up to 1504 B)
```

`OWNED_BY_HOST` written **last** by FW — host's poll on `rx_prod_idx` is safe even under PCIe write reordering. Mirrors TX `OWNED_BY_FW`.

**WRITE bypasses the ring.** FW dispatches by opcode: WRITE → rkey validation → NoC write to `mr.host_noc + remote_offset`. No ring slot, no CQE.

**WRITE_IMM exception:** consumes one ring slot for the immediate; payload still lands at `mr + offset`. Slot's `length=0`, `mr_table_idx` points at target MR.

---

## 4. Completion semantics

**TX (`TxCqe`):** completes when FW confirms wire-TX done (today's `cq_head`).

**RX (`RxCqe`):**
- `kSend`: matched against oldest unconsumed `post_recv` or `post_recv_any`. `data` → payload location.
- `kWriteImm`: ring slot consumed for immediate; `data` → `mr + remote_offset`; `length` = actual write length.
- `kReadResp`: NOT delivered to app. SDK consumes internally, completes originating `post_send_read`'s TxCqe.

**READ correlation:** SDK side table `seq → {local_mr, local_offset, len}`. Better: FW writes payload directly to landing MR via the lkey+offset from READ_REQ, posts only a header into the ring.

**Plain WRITE has no receiver visibility.** Verbs contract. Use SEND or WRITE_IMM for synchronisation.

---

## 5. Error model

| Failure | Detected by | Surfaced as |
|---|---|---|
| Bad rkey / wrong generation | FW MR-table lookup | RX-side drop + counter. TX-side: peer NACK → `kRemoteErr` when reliable; silent otherwise. |
| `remote_offset + len > mr.len` | FW pre-DMA | `kBadLen` on TX |
| TX ring full at sender | SDK | `post_send_*` → `nullopt` |
| RX ring full at receiver | FW | `rx_overflow_drops++`, drop, sender retx kicks in |
| CRC error on wire | CMAC | Drop pre-FW; reliability handles |
| Local landing MR not registered (READ) | SDK at post time | `nullopt`, no FW state change |
| MR-table full | SDK | invalid handle |
| Dereg-while-in-flight | SDK (generation) | Late frames dropped silently in FW; late TX CQEs → `kFlushed` |

---

## 6. Reliability — policy

**Recommendation:**

| Op | Reliability default | Rationale |
|---|---|---|
| `post_send` (legacy) | **off** | Matches today; raw CMAC testing |
| `post_send` matched with `post_recv` | **off** | Receiver gets CQE; app-level retry via cookie |
| `post_send_write` / `_imm` | **on** | One-sided; no receiver CQE for plain WRITE → silent loss = silent corruption. **Non-negotiable.** |
| `post_send_read` | **on** | REQ or RESP loss = stuck app |

Implementation: `auto_reliable_one_sided = true` (default). First one-sided post auto-enables reliability on underlying sender. Once on, stays on (no clean way to drop window without quiescing).

**Justification:** verbs contract says WRITE is reliable. Period. Cost: ACK frames ~one per 64 posts at 1500 B = 0.1 Gbps overhead at 6 Gbps — in the noise.

---

## 7. Concrete code locations

| File | Change |
|---|---|
| `tt_metal/llrt/tt_rdma_endpoint.hpp` | **new** — class above |
| `tt_metal/llrt/tt_rdma_endpoint.cpp` | **new** — implementation; composes `std::unique_ptr<ExternalIfaceSender>` |
| `tt_metal/llrt/external_iface_sender.hpp` | Add typed `poll_completion_typed()`; RX-ring publish helpers; RCB offset constants |
| `tt_metal/llrt/external_iface_sender.cpp` | Extend `enable_fw_dma_pull` → `_bidir` (TX + RX + MR metadata in one map) |
| `tt_metal/llrt/CMakeLists.txt` | add `tt_rdma_endpoint.cpp` |
| `tests/tt_metal/llrt/test_tt_rdma_loopback.cpp` | **new** loopback test |
| `tests/tt_metal/llrt/CMakeLists.txt` | add new test |

Existing `test_external_cmac_soak.cpp`, smoke tests, `test_external_cmac_rx_idle` **don't change**.

---

## 8. Example usage

**One-sided WRITE:**

```cpp
#include "tt_metal/llrt/tt_rdma_endpoint.hpp"
#include <sys/mman.h>

using namespace tt::llrt;
constexpr size_t kHugepageBytes = 2u << 20;

void* src = mmap(nullptr, kHugepageBytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
void* dst = mmap(nullptr, kHugepageBytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

TtRdmaEndpoint ep(0, virtual_core);
TT_FATAL(ep.bring_up(), "bring_up failed");

MrHandle src_mr = ep.register_mr(src, kHugepageBytes, AccessFlags::kLocalRead);
MrHandle dst_mr = ep.register_mr(dst, kHugepageBytes, AccessFlags::kRemoteWrite);

std::memset(src, 0xA5, 4096);

std::span<const uint8_t> payload(static_cast<uint8_t*>(src), 4096);
auto seq = ep.post_send_write(payload, dst_mr, 0x1000);
TT_FATAL(seq && ep.wait_completion(*seq, 1000), "WRITE timeout");

auto cqe = ep.poll_completion();
TT_FATAL(cqe && cqe->status == CqStatus::kOk, "WRITE failed");
```

**Two-sided SEND/recv:**

```cpp
// Receiver
for (int i = 0; i < 8; ++i) ep.post_recv_any(1500, /*cookie=*/i);

// Sender
std::array<uint8_t, 128> msg{};
std::memcpy(msg.data(), "hello rdma", 11);
auto seq = ep.post_send(std::span(msg));
ep.wait_completion(*seq, 100);

// Receiver poll
auto rx = ep.wait_rx_completion(100);
TT_FATAL(rx && rx->opcode == WireOpcode::kSend && rx->length == 128);
ep.release_rx_slot(*rx);
```

**WRITE_IMM (one-sided + sync):**

```cpp
ep.post_recv_any(0);  // Receiver pre-posts for immediate
ep.post_send_write_imm(payload, dst_mr, 0x2000, /*imm=*/0xCAFEBABE);

auto rx = ep.wait_rx_completion(1000);
assert(rx->opcode == WireOpcode::kWriteImm);
assert(rx->immediate == 0xCAFEBABE);
// rx->data points at dst_mr+0x2000; payload already there
ep.release_rx_slot(*rx);
```

---

## 9. Migration plan

1. **Phase A** — opcode field on wire; FW dispatches; magic-byte fallback for legacy SEND. Soak tests don't change.
2. **Phase B** — MR table + WRITE. SDK exposes `register_mr` / `post_send_write`. No RX ring yet.
3. **Phase C** — RX ring + SEND/WRITE_IMM. Legacy `kRxWpAddr` ping-pong stays for rx_idle test but verbs path doesn't use it.
4. **Phase D** — READ + ATOMIC. Reliability mandatory.

`ExternalIfaceSender::post_send` keeps working unchanged in every phase. Internally opcode=`kSend` with `rkey=0`; FW treats `rkey=0` as "no MR lookup, legacy SEND."
