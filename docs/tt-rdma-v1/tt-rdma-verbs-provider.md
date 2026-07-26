# TT-RDMA libibverbs Provider — Production Roadmap

**Status:** design / build plan. Not yet implemented.
**Target:** ship `libtt_rdma-rdmav<N>.so` (userspace provider) + `ib_tt_rdma` (kernel driver) as a native [rdma-core](https://github.com/linux-rdma/rdma-core) provider so that **unmodified `libibverbs` apps** — `ib_*_bw`, MPI (Open MPI, MPICH), UCX, NCCL, libfabric, NVMe-oF initiators — reach a Tenstorrent chip over TT-RDMA-v1 wire, with **FPGA-TT-Link** as the partner NIC (no Mellanox, no BF3 gateway, no PCIe-bound HCA in the data path).

This is the **software-side** alternative to `bf3-gateway-design.md`. The two solve overlapping problems with opposite cost profiles:

| Solution | App changes | Hardware | Wire-rate ceiling | Where translation lives |
|---|---|---|---|---|
| BF3 gateway (`bf3-gateway-design.md`) | none | + BF3 SmartNIC per cluster | RoCE wire (gated by BF3 PCIe) | SmartNIC Arm cores |
| **This doc: verbs provider** | none (relink) | none beyond FPGA-TT-Link | TT-RDMA-v1 wire (line rate) | host kernel + userspace lib |

For an FPGA-TT-Link deployment the provider is strictly better — the BF3 would re-introduce a PCIe bottleneck the FPGA path was built to remove.

---

## 1. Goals and non-goals

### Goals (v1)

1. **API:** Standard `libibverbs` 1.x. App code that runs against `mlx5` runs against `tt_rdma` with only the device-name string changing.
2. **Wire:** Stock TT-RDMA-v1 frames (ethertype `0x1AF6`, 32 B header) — **no FW change for the initial subset**.
3. **Verbs subset:** `RDMA_WRITE`, `RDMA_WRITE_WITH_IMM`, `SEND`, `SEND_WITH_IMM`, `RDMA_READ`, plus `reg_mr` / `dereg_mr` / `post_send` / `post_recv` / `poll_cq` / `create_qp` / `modify_qp` / `destroy_qp`.
4. **Connection setup:** Standard `librdmacm` (`rdma_resolve_addr`, `rdma_connect`, etc.) — OOB TCP for QP-info exchange, same pattern as `rxe`.
5. **Performance:** Within 10 % of the existing `TtRdmaEndpoint` SDK (`tt-rdma-host-sdk.md`). Target ~25 Gbps over PCIe Gen3 x4, line rate over FPGA-TT-Link.
6. **Upstreamable:** Code style and ABI clean enough to submit to `linux-rdma/rdma-core` + the kernel RDMA tree without a fork.

### Non-goals (v1)

- ATOMIC ops (CAS/FAA) — deferred to v1.2, wire opcodes `0x30`/`0x31` already reserved.
- SRQ, XRC, multicast, UD, RoCE-CM TCP modes.
- Multi-port HCAs (one port per WH chip).
- Kernel-bypass for MR registration (mmap fast path is fine; reg_mr stays in slow path).
- IB management datagrams (MADs) — TT-RDMA has no fabric manager concept.
- Memory windows, FMR — deprecated even in upstream.

---

## 2. End-to-end architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HOST USERSPACE                                                          │
│                                                                          │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐    │
│  │ App (MPI/UCX/NCCL/  │───▶│ libibverbs / librdmacm  (unchanged)  │    │
│  │ ib_write_bw/...)    │    └────────────────┬─────────────────────┘    │
│  └─────────────────────┘                     │ dlopen                   │
│                                              ▼                          │
│                          ┌──────────────────────────────────────────┐   │
│                          │ libtt_rdma-rdmav<N>.so  (this provider)  │   │
│                          │  fast path:                              │   │
│                          │    post_send  → write WQE to L1 ring     │   │
│                          │    post_recv  → host RxWqeRing slot      │   │
│                          │    poll_cq    → read host CQ ring        │   │
│                          │  slow path:                              │   │
│                          │    reg_mr / create_qp → ioctl(ib_tt_rdma)│   │
│                          └────────────────┬─────────────────────────┘   │
└───────────────────────────────────────────┼─────────────────────────────┘
                                            │ /dev/infiniband/uverbsN
                                            │ + mmap fast-path regions
                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ HOST KERNEL                                                             │
│                                                                          │
│  ┌─ ib_core (subsystem) ─────────────────────────────────────────────┐  │
│  │  ib_register_device("tt_rdma_0")                                 │  │
│  └─────────────────────────────┬─────────────────────────────────────┘  │
│                                ▼                                        │
│  ┌─ ib_tt_rdma.ko (this driver) ────────────────────────────────────┐   │
│  │  ib_device_ops:                                                  │   │
│  │   • alloc_pd / dealloc_pd                                        │   │
│  │   • create_qp / modify_qp / destroy_qp                           │   │
│  │   • reg_user_mr (ib_umem_get → register_mr_slot doorbell)        │   │
│  │   • create_cq / destroy_cq                                       │   │
│  │   • mmap (expose WQE ring + CQ ring to userspace)                │   │
│  │   • post_send/recv/poll_cq stubs (userspace handles fast path)   │   │
│  │  background workqueue:                                           │   │
│  │   • drain host RxWqeRing → push CQEs into per-QP CQ              │   │
│  │   • handle CONTROL frames (MR ack, error)                        │   │
│  └────────────┬───────────────────────────────────────────────┬────┘   │
│               │ kernel handle to tt-metal MMIO                 │        │
│               ▼                                                ▼        │
│  ┌─ tt-kmd (existing) ─────────┐    ┌─ existing TtRdmaEndpoint kernel  │
│  │  PCIe BAR access to WH/BH    │    │  glue (hugepage pinning, NoC    │
│  │  chip; doorbell at RCB+0x28  │    │  PCIe tile addressing)          │
│  └─────────────┬────────────────┘    └─────────────────────────────────┘ │
└────────────────┼────────────────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TT CHIP — erisc_cmac_simple FW v1.0 (md5 43bdcb46, UNCHANGED)           │
│  WQE ring @ L1:0x9000  •  RX BUF_WRAP @ L1:0x4000–0x7800                │
│  MR table @ L1:0x8500  •  CMAC PFC priority 3 lossless                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ 100 GbE, ethertype 0x1AF6
                             ▼
                  ┌────────── FPGA-TT-Link NIC ──────────┐
                  │ open-nic-shell + QDMA                │
                  │ TT-RDMA-v1 partner (mirrors WH FW)   │
                  └──────────────────────────────────────┘
```

**Load-bearing properties:**
- Fast path is **fully userspace** — `post_send` mmaps the WQE ring and writes directly; no syscall per WR. Matches mlx5 fast-path cost.
- Slow path (`reg_mr`, `create_qp`) goes through standard uverbs ioctl into `ib_tt_rdma.ko`, which doorbells the erisc at `RCB+0x28` (Phase P7).
- The FW does **not change** for the v1 subset — same dispatcher, same MR table, same opcode handlers. The provider is pure host-side software above the existing wire format.

---

## 3. The wire-protocol gap: QPN

> **Superseded in part (2026 addendum):** the "`tag` = QPN for everything"
> framing below (Option B) collides with READ correlation, which also uses
> `tag` (`main_cmac.cc:762-764`), and assumes a 32-bit `tag` when the wire
> field is 16-bit. `tt-rdma-mesh-addressing-spec.md §2` resolves this: route
> **one-sided ops by `rkey`** (a remote MR implies its endpoint) and **only
> two-sided SEND-family by `tag`-as-QPN** (which never needs correlation).
> Prefer that model for both the gateway and this provider.

The single design problem that bleeds beyond the v1 wire is **how to multiplex multiple verbs QPs onto one CMAC port**. v1's 32 B header has no QPN field (see `tt-rdma-wire-protocol-v1.md:14-36`):

```
 0      1     opcode
 1      1     version_flags    ← bit 7 is reserved
 2      2     tag              ← could double as QPN?
 4      4     length
 8      4     seq
12      4     rkey
16      8     remote_offset
24      4     imm_data
28      4     header_cksum
```

Three options, pick one per phase:

| Option | Wire change? | Max QPs | When |
|---|---|---|---|
| **A. Single-QPN** | none | 1 | Phases 0–3, smoke tests, single-stream `ib_write_bw` |
| **B. `tag` doubles as QPN** | none, semantic reuse | 65 536 | Phase 4+, MPI multi-rank, NCCL collectives |
| **C. v1.x extension: dedicated QPN field** | yes, ethertype `0x1AF7`, 36 B hdr | 2³² | v2 / production |

**Recommendation:** ship Phases 0–3 with Option A (no wire change, prove the stack). Move to Option B when MPI lands (Phase 4); `tag` is currently "opaque cookie for SEND/WRITE" so repurposing the low 16 bits as QPN costs nothing on shipped FW. Option C is the long-term clean answer and pairs naturally with v1.2's multi-erisc-per-chip work.

The QPN/`tag` split for Option B:

```
 2      2     tag    →    qpn      (16 bits, src QPN on TX, dst QPN on RX)
                          tag_hi   (was upper byte) → completion correlation only
```

FW change for Option B: **none** — FW already passes `tag` through opaquely. The provider library on both ends agrees to interpret it as QPN.

---

## 4. Verbs → TT-RDMA-v1 opcode translation

Complete mapping, every WR field accounted for:

| `ibv_send_wr` field | TT-RDMA-v1 header bytes | Notes |
|---|---|---|
| `wr.opcode` | `[0] opcode` | per table below |
| `wr.send_flags & IBV_SEND_INLINE` | inline payload follows hdr | payload ≤ inline_size (provider caps at 240 B) |
| `wr.send_flags & IBV_SEND_SIGNALED` | `version_flags.REQ_ACK` | always set for RDMA_READ |
| `wr.send_flags & IBV_SEND_SOLICITED` | `version_flags.SOLICITED` | hints receiver to raise CQE-event |
| `wr.imm_data` | `[24..27] imm_data` + `version_flags.IMM_PRESENT` | only for `*_WITH_IMM` |
| `wr.wr_id` | host-side WQE table | kept in userspace `wqe_slot[i].wr_id` |
| `wr.wr.rdma.remote_addr` | `[16..23] remote_offset` (via rkey base subtract) | provider computes `addr - mr.base` |
| `wr.wr.rdma.rkey` | `[12..15] rkey` | rkey encoding matches IB |
| `wr.sg_list[]` | DMA-gather into TX WQE slot | provider does the gather (FW WQE is contiguous) |
| (qp_num) | `tag` low 16b (Option B) | Phase 4+ |

Opcode crosswalk:

| `ibv_wr_opcode` | TT-v1 opcode | Verb status | FW status |
|---|---|---|---|
| `IBV_WR_SEND` | `0x01 SEND` | ✅ v1 | ✅ shipped |
| `IBV_WR_SEND_WITH_IMM` | `0x02 SEND_IMM` | ✅ v1 | ✅ shipped (Q(b)) |
| `IBV_WR_RDMA_WRITE` | `0x10 WRITE` | ✅ v1 | ✅ shipped (P3/P4) |
| `IBV_WR_RDMA_WRITE_WITH_IMM` | `0x11 WRITE_IMM` | ✅ v1 | ✅ shipped (Q(b)) |
| `IBV_WR_RDMA_READ` | `0x20 READ_REQ` → `0x21 READ_RESP` | ✅ v1 | ✅ shipped (P6, Phase I) |
| `IBV_WR_ATOMIC_CMP_AND_SWP` | `0x30 ATOMIC_CAS` | ⏳ v1.2 | ⏳ reserved |
| `IBV_WR_ATOMIC_FETCH_AND_ADD` | `0x31 ATOMIC_FAA` | ⏳ v1.2 | ⏳ reserved |
| `IBV_WR_LOCAL_INV` | n/a | ❌ | not in scope |
| `IBV_WR_BIND_MW` | n/a | ❌ deprecated | not in scope |

WC (completion) crosswalk:

| `ibv_wc_opcode` | Source | Provider derives from |
|---|---|---|
| `IBV_WC_SEND` | local | WQE drain (ACK 0x40 received for SEND seq) |
| `IBV_WC_RDMA_WRITE` | local | ACK 0x40 |
| `IBV_WC_RDMA_READ` | local | matching `0x21 READ_RESP` lands |
| `IBV_WC_RECV` | remote | FW pushes RxWqeRing slot to host hugepage |
| `IBV_WC_RECV_RDMA_WITH_IMM` | remote | `0x11 WRITE_IMM` raises receiver-side CQE |

---

## 5. Kernel driver: `ib_tt_rdma.ko`

Lives in `drivers/infiniband/hw/tt_rdma/` (upstream path) or `drivers/tt_rdma/` (out-of-tree during development).

### 5.1 Files

```
drivers/infiniband/hw/tt_rdma/
├── main.c          # module init, ib_register_device, sysfs
├── verbs.c         # ib_device_ops dispatch
├── qp.c            # QP state machine (RESET→INIT→RTR→RTS→ERR)
├── mr.c            # ib_umem_get → MR slot reservation → RCB doorbell
├── cq.c            # CQ ring backing, drain workqueue
├── uar.c           # mmap of WQE ring + CQ ring into userspace
├── cm.c            # rdma_cm passive/active listener glue
├── tt_chip.c       # bridge to existing tt-kmd: BAR mapping, doorbell, MMIO
├── tt_rdma-abi.h   # uABI shared with userspace provider
├── tt_rdma.h       # internal
└── Kconfig, Makefile
```

### 5.2 `ib_device_ops` implementation table

| Verb | Hot path? | Implementation |
|---|---|---|
| `alloc_pd` / `dealloc_pd` | no | host-side counter, PD-id allocated from idr |
| `create_qp` | no | allocate QPN (Option A=0, B=idr), allocate WQE slot range in shared L1 region |
| `modify_qp` | no | state machine; `INIT→RTR` writes peer mac/QPN into per-QP TX header template; `RTR→RTS` is a no-op for v1 |
| `destroy_qp` | no | flush WQEs, free QPN, return slots |
| `reg_user_mr` | no | `ib_umem_get` pin pages → find free MR slot (16 entries) → write entry via tt-kmd → doorbell `RCB+0x28` → wait for FW ack via CONTROL 0xF0 |
| `dereg_mr` | no | invalidate slot (`rkey=0`), doorbell, wait for ack |
| `create_cq` | no | allocate kernel CQ ring (4 KB), expose via mmap, register with rxq drain wq |
| `mmap` | no | maps: (1) shared WQE ring fragment, (2) CQ ring, (3) doorbell page |
| `post_send` / `post_recv` / `poll_cq` | yes | **userspace handles directly** — kernel only provides the shared rings |

### 5.3 MR registration flow

```
userspace                    kernel                       FW
─────────                    ──────                       ──
ibv_reg_mr(addr, len, flags)
  │
  └──▶ tt_rdma_reg_mr_uABI(fd, addr, len, flags)
            │
            └──▶ ib_umem_get(addr, len)             [pin pages]
                  │
                  └──▶ tt_chip_alloc_mr_slot()      [16-entry table]
                        │
                        └──▶ MMIO write to L1:0x8500 + slot_off:
                              base_noc_addr  = umem.dma_addr_in_NoC
                              length         = len
                              rkey           = (slot<<24)|rand16<<8|gen
                              access_flags   = from flags
                              ◀──── (doorbell RCB+0x28) ────▶
                                                            FW reads MR slot
                                                            FW posts CONTROL 0xF0
                                                            "MR_ACK slot=N gen=M"
                              ◀──── poll ack, ≤ 100 µs ────
                  ◀──── ib_mr { lkey, rkey, addr } ────
  ◀──── ibv_mr* ────
```

The 16-entry FW table is the bottleneck. Userspace gets a fair-share quota (~12 slots), kernel reserves 4 for internal use (RxWqeRing, CQ ring, control). When userspace exhausts slots, `reg_mr` returns `-ENOMEM` — apps should pool MRs (NCCL/UCX both do this).

### 5.4 QP creation and the WQE ring

With Option B (Phase 4+), each QP gets:
- A 16-bit QPN (`tag` low bits)
- A per-QP TX template (peer MAC, peer QPN, local QPN, starting `seq`)
- A slice of the L1 WQE ring (initially: shared 32-slot ring, software arbitration)
- An RX filter on the host RxWqeRing (drained by per-CQ kthread)

WQE ring shared-arbitration scheme:

```
Userspace QPs              Shared L1:0x9000 WQE ring (32 × 4 KB)
─────────────              ───────────────────────────────────
QP_A.post_send ──┐
                 ├─ host-side mutex around ring tail ──▶ slot[i]
QP_B.post_send ──┘                                       ↓
                                                    CMAC TX (FIFO order)
```

This is the simplest correct implementation. Per-QP rings come with v1.2 multi-erisc (`README.md` line 129).

---

## 6. Userspace provider: `libtt_rdma`

Lives in `rdma-core/providers/tt_rdma/`.

### 6.1 Files

```
providers/tt_rdma/
├── tt_rdma.c           # ibv_context_ops, ibv_driver_init_func
├── tt_rdma.h           # internal types
├── verbs.c             # post_send / post_recv / poll_cq fast path
├── translate.c         # WR → v1 hdr packing, WC unpacking
├── cq.c                # CQ ring producer-consumer
├── qp.c                # QP create / modify / destroy uABI calls
├── mr.c                # reg_mr / dereg_mr uABI calls
├── tt_rdma-abi.h       # SHARED with kernel — sym-linked from kernel tree
├── CMakeLists.txt
└── tt_rdma.driver      # rdma-core driver registration shim
```

### 6.2 `post_send` fast path (the only hot one)

```c
int tt_rdma_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                      struct ibv_send_wr **bad) {
    struct tt_rdma_qp *q = to_tqp(qp);

    for (; wr; wr = wr->next) {
        // 1. Reserve a WQE slot (atomic increment of shared tail)
        uint32_t slot = __atomic_fetch_add(&q->ring->tail, 1, __ATOMIC_ACQ_REL)
                        & 31;
        struct v1_wqe *w = &q->ring->slots[slot];

        // 2. Build the 32 B header
        w->hdr.opcode        = wr_to_v1_opcode(wr->opcode);
        w->hdr.version_flags = 1
            | (wr->send_flags & IBV_SEND_INLINE     ? 0 : 0)  // payload by ref handled elsewhere
            | (wr->imm_data                          ? 0x10 : 0)
            | (wr->send_flags & IBV_SEND_SOLICITED  ? 0x40 : 0);
        w->hdr.tag           = (q->qpn & 0xffff)          // Option B
                             | ((wr->wr_id & 0xffff) << 16);
        w->hdr.length        = wr_payload_len(wr);
        w->hdr.seq           = q->next_tx_seq++;
        w->hdr.rkey          = wr->wr.rdma.rkey;
        w->hdr.remote_offset = wr->wr.rdma.remote_addr - q->peer_mr_base[wr->wr.rdma.rkey];
        w->hdr.imm_data      = be32toh(wr->imm_data);
        w->hdr.header_cksum  = crc32(&w->hdr, 28);  // CRC-32 (0x04C11DB7, ETH-CTRL ICRC poly)

        // 3. Gather sg_list into payload
        gather_sges(w->payload, wr->sg_list, wr->num_sge);

        // 4. Track for completion
        q->outstanding[slot] = (struct tt_wqe_meta){
            .wr_id   = wr->wr_id,
            .opcode  = wr->opcode,
            .signaled= !!(wr->send_flags & IBV_SEND_SIGNALED),
        };

        // 5. Doorbell the FW (one mmio write to the chip's TX kick reg)
        __atomic_store_n(q->doorbell, slot + 1, __ATOMIC_RELEASE);
    }
    return 0;
}
```

No syscall on the fast path. `q->ring` and `q->doorbell` are mmap'd in `create_qp`.

### 6.3 `poll_cq` fast path

```c
int tt_rdma_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc) {
    struct tt_rdma_cq *c = to_tcq(cq);
    int i = 0;
    uint32_t head = __atomic_load_n(&c->ring->head, __ATOMIC_ACQUIRE);

    while (i < n && c->next != head) {
        struct v1_cqe *e = &c->ring->slots[c->next & (c->depth - 1)];
        wc[i].wr_id    = e->wr_id;
        wc[i].opcode   = v1_to_wc_opcode(e->op);
        wc[i].status   = e->status;        // IBV_WC_SUCCESS / IBV_WC_RETRY_EXC / etc.
        wc[i].byte_len = e->length;
        wc[i].imm_data = e->imm_data;
        wc[i].qp_num   = e->qpn;
        c->next++;
        i++;
    }
    return i;
}
```

The CQ ring is populated by the kernel drain workqueue (5.2) — kernel watches the host RxWqeRing for SEND/WRITE_IMM/READ_RESP/ACK, builds CQE, advances `head`. Userspace just reads.

### 6.4 Connection establishment via librdmacm

`librdmacm` handles `rdma_resolve_addr` / `rdma_connect` over TCP, then calls into the provider via `modify_qp(INIT→RTR)` once it has the peer's QPN/PSN/MAC. Copy the `rxe` glue almost verbatim — the only TT-specific change is that the v1 wire has no PSN per se (we use `seq` directly).

CM message exchange (TCP, JSON for clarity; production: protobuf or REQ/REP TLV):

```json
// REQ:  initiator → target
{
  "qpn":          0x4321,
  "starting_psn": 0,
  "src_mac":      "00:11:22:33:44:55",
  "tt_rdma_ver":  1
}
// REP:  target → initiator
{
  "qpn":          0x8765,
  "starting_psn": 0,
  "src_mac":      "00:aa:bb:cc:dd:ee",
  "tt_rdma_ver":  1
}
```

---

## 7. Phase plan

Modeled on the existing v1.0 FW phase table (`README.md`). Each phase ends with a validation gate that gives a binary go/no-go.

| Phase | Deliverable | Effort | FW change? | Validation gate |
|---|---|---|---|---|
| **0** | Skeleton kernel module, registers `ib_device("tt_rdma_0")`, all verbs return `-EOPNOTSUPP` | 3 d | no | `ibv_devices` lists it; `ib_show_dev tt_rdma_0` prints port-attr |
| **1** | `alloc_pd`, `create_cq`, `create_qp` (Option A: QPN=0); userspace provider loads, opens context | 4 d | no | `ibv_open_device` succeeds; QP transitions to INIT |
| **2** | `reg_mr` end-to-end with `RCB+0x28` doorbell; mmap WQE ring into userspace | 5 d | no | `ibv_reg_mr(1 MB)` returns valid lkey/rkey; FW MR-table slot populated (verify via tt-exalens) |
| **3** | `post_send(IBV_WR_RDMA_WRITE)` + WQE-ring fast path + CQ drain workqueue | 1 w | no | `ib_write_bw -d tt_rdma_0` runs against an FPGA-TT-Link partner echo target; bytes match; ≥ 20 Gbps over Gen3 x4 |
| **4** | Option B: `tag`-as-QPN; multiple QPs share the ring; `ib_write_bw -t 4` | 1 w | no | 4 concurrent QPs each delivering ≥ 5 Gbps; no cross-QP completions |
| **5** | `RDMA_READ` (post-side + completion correlation table) | 4 d | no | `ib_read_bw -d tt_rdma_0` matches existing `post_send_read` ceiling |
| **6** | `SEND` / `SEND_WITH_IMM` + `post_recv` + RxWqeRing → CQE drain | 1 w | no | `ib_send_bw`; `IBV_WR_SEND_WITH_IMM` raises receiver CQE with correct imm_data |
| **7** | `WRITE_WITH_IMM` receiver-side completion | 3 d | no | imm_data raises CQE on receiver-side CQ without app polling MR |
| **8** | `librdmacm` integration — `rdma_listen` / `rdma_connect` / `rdma_disconnect` | 1 w | no | `rdma_*` examples in `rdma-core/librdmacm/examples` pass |
| **9** | OFED-style perftest pass: `ib_*_bw` and `ib_*_lat` all green | 3 d | no | `ib_write_bw / ib_read_bw / ib_send_bw / ib_write_lat / ib_read_lat / ib_send_lat` all complete |
| **10** | MPI smoke: Open MPI compiled with `--with-verbs`; `mpirun osu_bw / osu_latency` | 1 w | no | osu_bw ≥ 20 Gbps; osu_latency reports a credible RTT |
| **11** | UCX provider integration (UCX has its own selection logic) | 4 d | no | `ucx_perftest -t tag_bw` passes |
| **12** | NCCL smoke: `nccl-tests/build/all_reduce_perf` over `tt_rdma_0` | 1 w | no | `all_reduce` completes; correctness pass |
| **13** | Wire-protocol v1.x: dedicated 32-bit QPN field, ethertype `0x1AF7`, 36 B header | 2 w | **yes** | dual-stack: v1 and v1.x coexist on same FW; provider selects by negotiation |
| **14** | ATOMIC ops (CAS/FAA) — pairs with FW v1.2 ATOMIC handlers | 1 w | yes (v1.2) | `ib_atomic_bw` passes |
| **15** | Upstream submission to `linux-rdma/rdma-core` + kernel maintainers | 4 w cal | no | merged or in-review on `linux-rdma` lists |

**Critical-path total:** Phases 0–9 = ~7 weeks of focused engineering. Phases 0–12 = ~12 weeks. Upstreaming runs in parallel from Phase 9.

### Dependencies between phases

```
Phase 0 ─▶ 1 ─▶ 2 ─▶ 3 ─▶ 5 ─▶ 9 ─▶ 10 ─▶ 11 ─▶ 12 ─▶ 15
                  └─▶ 4 ─▶ 6 ─▶ 7 ─▶ 8 ─┘
                                          └─▶ 13 ─▶ 14
```

Phases 4 and 5 are independent and can be done in parallel by two engineers. Phase 13 (wire-protocol extension) is independent of Phases 5–12 and can be designed in parallel.

---

## 8. uABI between kernel and userspace

Shared header `tt_rdma-abi.h`:

```c
#define TT_RDMA_ABI_VERSION 1

struct tt_rdma_alloc_ucontext_resp {
    __u32 max_qp;
    __u32 max_mr;
    __u32 max_cqe;
    __u32 max_send_wr;
    __u32 max_recv_wr;
    __u32 max_send_sge;
    __u32 max_inline_data;     // 240 (4 KB - 32 B hdr, minus reserve)
    __u32 doorbell_offset;     // mmap offset for doorbell page
};

struct tt_rdma_create_qp_resp {
    __u32 qpn;
    __u32 send_wqe_offset;     // mmap offset for this QP's WQE slice
    __u32 send_wqe_count;
    __u32 db_offset;           // doorbell offset within doorbell page
};

struct tt_rdma_create_cq_resp {
    __u32 cqn;
    __u32 ring_offset;         // mmap offset for this CQ's ring
    __u32 ring_size;
};

struct tt_rdma_reg_mr_resp {
    __u32 lkey;
    __u32 rkey;
    __u32 mr_slot;             // FW MR-table slot (debug only)
    __u32 _pad;
};
```

ABI versioning: bump `TT_RDMA_ABI_VERSION` on every breaking change; provider library checks at `ibv_open_device`.

---

## 9. Test plan

Per-phase smoke tests are in the table above. Cross-phase test suites:

### 9.1 Unit (in-tree)
- `providers/tt_rdma/tests/translate_test.c` — every WR → v1 header field-by-field
- `drivers/infiniband/hw/tt_rdma/tests/mr_lifecycle_test.c` — register/dereg 1k times, no leak
- `tests/ttkmd/tt_rdma_uabi_test.c` — ABI version mismatch handling

### 9.2 Integration (rig-required, FPGA-TT-Link partner)
- `tests/tt_metal/llrt/test_verbs_provider_smoke.cpp` — open ctx, alloc_pd, reg_mr, create_qp, single WRITE, verify on FPGA partner side
- `tests/tt_metal/llrt/test_verbs_provider_perf.sh` — wraps `ib_write_bw` / `ib_send_bw` / `ib_read_bw`, asserts ≥ 20 Gbps
- `tests/tt_metal/llrt/test_verbs_provider_multi_qp.sh` — 4–16 QPs, asserts per-QP isolation + aggregate BW
- `tests/tt_metal/llrt/test_verbs_provider_soak.sh` — 1 h run, no drops, no FW counter drift

### 9.3 Upstream-friendly (rdma-core CI compatible)
- `rdma-core/tests/test_tt_rdma_basic.py` — pyverbs test using rdma-core's existing harness
- Use `rxe` test patterns as a template — same test files, swap device name

### 9.4 Correctness oracles
- Cross-check every `ib_write_bw` run against the existing `TtRdmaEndpoint` SDK (`tt-rdma-host-sdk.md`) — same payload, same MR; bytes must match.
- Use `test_external_cmac_smoke_mlx.cpp` repointed at FPGA-TT-Link as the partner-side echo target.

---

## 10. Performance targets and budgets

| Metric | Target | Reference |
|---|---|---|
| `ib_write_bw -s 4080` over PCIe Gen3 x4 | ≥ 22 Gbps | matches existing SDK ceiling |
| `ib_write_bw -s 4080` over FPGA-TT-Link to remote chip MR | ≥ 90 Gbps | line-rate ceiling, no PCIe |
| `ib_send_bw` | ≥ 20 Gbps | host hugepage RxWqeRing limit |
| `ib_read_bw -s 4080` | ≥ 20 Gbps | Phase I baseline |
| `ib_write_lat` (1 B, 1 QP, polled) | ≤ 8 µs | userspace fast path + FW ~1 µs iter |
| `post_send` CPU cost (per WR, 1 thread) | ≤ 200 ns | matches mlx5 inline path |
| MR registration cost | ≤ 50 µs | dominated by `ib_umem_get` pinning |
| QP create | ≤ 200 µs | host-side allocation only |
| Sustained 1 h soak, all phases | 0 drops, ≤ 1 ppm CQE-loss | v1.2 1 h soak gate |

Budget breakdown for `post_send` fast path (the dominant cost):

| Step | Cycles @ 3 GHz | Notes |
|---|---|---|
| WQE slot reserve (atomic fetch-add) | ~15 | uncontended; contended grows linearly |
| Header build (28 B word stores) | ~10 | scalar stores; SIMD doesn't help |
| CRC32C over 28 B | ~30 | `_mm_crc32_u64` x 4 |
| sg_list gather (1 SGE, 4 KB, NT stream) | ~120 | bandwidth-bound, not latency |
| Outstanding-tracking write | ~3 | one store |
| Doorbell (uncached mmio) | ~400 | dominant; PCIe write fence |
| **Total** | **~580 cycles** ≈ **190 ns** | meets target |

---

## 11. Risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | FW 16-MR-slot ceiling — apps that pool many MRs (NCCL especially) exhaust it | high | document; ship a userspace MR-cache + LRU eviction; long-term: bump FW MR table to 64 slots in v1.2 |
| R2 | Single shared 32-slot WQE ring — head-of-line blocking between QPs under burst | medium | per-QP slot quotas; bump to per-QP rings in v1.2 multi-erisc |
| R3 | No native QPN — Option B reuses `tag`, breaks any app that uses `tag` as cookie | low (no app does today) | Phase 13 ships v1.x extension before any prod deployment |
| R4 | RDMA-CM TCP setup is slow (~ms) — bad for short-lived connections | low | match `rxe` behaviour; long-lived connections are the verbs norm |
| R5 | `ib_umem_get` pinning blows past `/proc/sys/vm/max_user_watches` for small-MR-heavy workloads | medium | document `MEMLOCK` rlimit + hugepage MRs; same advice as upstream HCAs |
| R6 | Kernel ABI churn — `ib_device_ops` changed in 5.x, may change in 7.x | medium | track LTS kernels (6.6, 6.12); CI matrix |
| R7 | Upstream review may demand FW changes (e.g., a real PSN field) | low | Phase 13 already plans v1.x; submit upstream after Phase 13 ships |
| R8 | NCCL's IB plugin makes RoCE-specific assumptions (GID, PKey) | medium | provider returns sensible RoCE-style GIDs synthesized from MAC; copy `rxe` GID generation |
| R9 | Bursty small-frame WRITEs overflow the 14 KB RX ring (open Q #3 in v1 README) | high | apply same coalescing the SDK already does; document min frame size 1 KB for high-rate streams |

---

## 12. Upstreaming plan

Two parallel submissions:

### Kernel (`linux-rdma`)

1. Phase 9 deliverable code in `drivers/infiniband/hw/tt_rdma/`
2. RFC patch to `linux-rdma@vger.kernel.org`
3. Address review (typically 2–4 rounds, 6–10 weeks)
4. Merge to `rdma-for-next` (Jason Gunthorpe's tree)
5. Lands in next kernel release (LTS targeted: 6.12)

### Userspace (`rdma-core`)

1. Provider in `providers/tt_rdma/` matching kernel uABI
2. PR to `linux-rdma/rdma-core` on GitHub
3. CI must pass: provider build on every distro `rdma-core` supports
4. Maintainer review (typically 1–2 rounds, 2–4 weeks)
5. Merge; ships in next `rdma-core` release; distros pick it up within ~6 months

### Documentation

- `man 3 ibv_open_device` already lists providers; rdma-core build picks it up automatically.
- Provider-specific notes: ship `tt_rdma.7` man page (file/dev paths, env vars, FW version requirements).

---

## 13. Open questions

1. **MR pinning vs ODP (On-Demand Paging).** v1 is pin-only. ODP would let apps register huge address ranges lazily. Cost: needs FW page-fault handling, big lift. Defer to v2.
2. **GID format.** RoCE GID is 128 b — synthesize from MAC + scope, or invent a TT-specific format? Recommend MAC-derived for `librdmacm` compatibility.
3. **Multi-host topology.** What does `ibv_query_port` return for `link_layer` — `IBV_LINK_LAYER_ETHERNET` (true) or `IBV_LINK_LAYER_UNSPECIFIED` (safer for tooling)? Recommend `ETHERNET` for tooling acceptance.
4. **Per-QP ordering vs cumulative-ACK.** v1's ACK is port-level cumulative, not per-QP. Verbs spec is per-QP. Phase 13 should either add per-QP seq or document that ordering is per-port not per-QP (which violates strict verbs semantics but matches what apps actually rely on).
5. **inline_size negotiation.** 240 B chosen because 4096-32 = 4064 minus reserve, but `mlx5` uses 60 B as default. Pick a value that matches existing apps' assumptions — likely 220 B for compatibility.
6. **Endianness.** v1 is little-endian on the wire (`tt-rdma-wire-protocol-v1.md:14`). IB BTH is big-endian. Provider does the byte-swap on TX/RX. Cost: ~4 ns, negligible.
7. **Doorbell write-combining.** PCIe doorbell store is the dominant `post_send` cost (~400 cycles). Worth using a write-combining region? Probably yes; copy `mlx5`'s WC BlueFlame approach for ≥ 10 ns improvement on small ops.

---

## 14. Critical files in the repo when this lands

```
# Kernel (out-of-tree initially → linux-rdma upstream)
drivers/infiniband/hw/tt_rdma/main.c
drivers/infiniband/hw/tt_rdma/verbs.c
drivers/infiniband/hw/tt_rdma/qp.c
drivers/infiniband/hw/tt_rdma/mr.c
drivers/infiniband/hw/tt_rdma/cq.c
drivers/infiniband/hw/tt_rdma/cm.c
drivers/infiniband/hw/tt_rdma/tt_chip.c
include/uapi/rdma/tt_rdma-abi.h

# Userspace (rdma-core)
providers/tt_rdma/tt_rdma.c
providers/tt_rdma/verbs.c
providers/tt_rdma/translate.c
providers/tt_rdma/cq.c
providers/tt_rdma/CMakeLists.txt
providers/tt_rdma/man/tt_rdma.7

# Existing tt-metal-external-eth integration points
tt_metal/llrt/external_iface_sender.{hpp,cpp}   # may grow a "verbs mode" flag
tests/tt_metal/llrt/test_verbs_provider_*.cpp   # new

# FW (no v1 change required for Phases 0–12; v1.x in Phase 13)
budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/main_cmac.cc
```

---

## 15. References

- [rdma-core](https://github.com/linux-rdma/rdma-core) — upstream userspace
- [Linux RDMA tree](https://git.kernel.org/pub/scm/linux/kernel/git/rdma/rdma.git) — kernel side
- [`rxe` provider](https://github.com/linux-rdma/rdma-core/tree/master/providers/rxe) — pure-software template
- [`siw` provider](https://github.com/linux-rdma/rdma-core/tree/master/providers/siw) — soft-iWARP template
- [`efa` provider](https://github.com/linux-rdma/rdma-core/tree/master/providers/efa) — AWS Elastic Fabric Adapter, closest custom-wire analogue
- `README.md` (this directory) — v1.0 shipped status
- `tt-rdma-wire-protocol-v1.md` — header format, opcode set
- `tt-rdma-host-sdk.md` — existing `TtRdmaEndpoint` API (this provider replaces it for verbs apps)
- `tt-rdma-fw-arch-rx.md` — FW dispatch flow (unchanged)
- `bf3-gateway-design.md` — the hardware-side alternative
- `runtime-workload-coexistence.md` — how this coexists with a Tensix runtime workload
- IBTA Spec Volume 1 release 1.6 — verbs semantics reference
