# Runtime Workload Coexistence over FPGA-TT-Link

How a Tensix inference workload (Qwen3-0.6B persistent decoder stack) shares a
single TT chip with the `erisc_cmac_simple` external-eth FW, with **FPGA-TT-Link**
as the wire partner. The picture below is a "where do bytes physically move"
diagram — not an API call graph — because the load-bearing insight is that
both the runtime and the FW dispatch through the **same NoC** via
`mr.base_noc_addr` (see `README.md` §"Architecture, one diagram").

## One picture

```
                Host (CPU)                                                          Remote TT chip
        ┌────────────────────────────┐                                       ┌────────────────────────┐
        │ greedy_driver.py            │                                      │ peer Qwen3 shard /      │
        │  • prompt → token_ids       │                                      │ KV cache target / etc.  │
        │  • TtRdmaEndpoint SDK       │                                      └───────────▲─────────────┘
        │    register_mr / post_*     │                                                  │
        └──┬───────────┬───────────┬──┘                                                  │ NoC fabric over wire
           │ PCIe      │ hugepage  ┊ control plane:                                      │ ~line rate, no host
           │ Gen3 x4   │ MR DMA    ┊ register_mr_slot                                    │
           │ ~25 Gbps  │ ~25 Gbps  ┊ doorbell @ RCB+0x28                                 │
           │ (data)    │ (data)    ┊ (one-time, off data path)                           │
           ▼           ▼           ┊                                                     │
   ╔═══════════════════════════════╧═════════════════════════════════════════════════╗   │
   ║  Wormhole chip (single die)                                                     ║   │
   ║                                                                                 ║   │
   ║  ┌──────── Tensix grid — Qwen3 runtime ────────┐  ┌──── erisc core ──────────┐  ║   │
   ║  │  Qwen3DecoderStack  (persistent while-loop) │  │ erisc_cmac_simple FW     │  ║   │
   ║  │   ┌─ qkv_norm_rope_kvwrite ─┐               │  │   RX: BUF_WRAP 14 KB     │  ║   │
   ║  │   ├─ flash_gqa              │ × 28 layers   │  │   ├─ WRITE 0x10 → NoC W  │  ║   │
   ║  │   ├─ o_residual             │               │  │   ├─ READ  0x20 → NoC R  │  ║   │
   ║  │   └─ swiglu_ffn             │               │  │   ├─ SEND  0x01 → host   │  ║   │
   ║  │   + lm_head_sampling                        │  │   └─ ACK   0x40 / retx   │  ║   │
   ║  └────┬─────────────────────────────┬──────────┘  │   TX: 32 × 4 KB WQE ring │  ║   │
   ║       │ weights (1.2 GB BF16)       │             └──┬───────────────▲───────┘  ║   │
   ║       │ KV cache (4.8 GB @ 40k pos) │                │ ~line rate    │          ║   │
   ║       ▼                             ▼                ▼               │          ║   │
   ║  ┌──── NoC fabric  —  mr.base_noc_addr decides where the byte lands ────────┐   ║   │
   ║  │ ░░░ host hugepage ░░░ │ ▒▒▒ tensix L1 ▒▒▒ │ ▓▓▓ DRAM banks ▓▓▓ │ ███ ETH-NoC ███ │ ║
   ║  │  via PCIe NoC tile     local NoC            DRAM controllers     to/from CMAC    │ ║
   ║  │  ≤ 25 Gbps             ~TB/s peak           ~512 GB/s aggregate   ~100 Gbps wire │ ║
   ║  │  (PCIe-bound)          (line rate)          (line rate)           (line rate)    │ ║
   ║  └───────────────────────────────────────────────────────────────────┬──────────┘   ║   │
   ║                                                                      │ CMAC TX/RX   ║   │
   ║                                                                      ▼              ║   │
   ║                                                        ┌──────── CMAC PHY ───────┐  ║   │
   ╚════════════════════════════════════════════════════════╧═════════════╤═══════════╧══╝   │
                                                                          │                  │
                                                100 GbE wire — ethertype 0x1AF6              │
                                                PFC priority 3 lossless                      │
                                                ~100 Gbps line rate                          │
                                                                          ▼                  │
                                                        ┌─────── FPGA-TT-Link NIC ──────┐    │
                                                        │  open-nic-shell + QDMA         │   │
                                                        │  TT-RDMA v1 partner            │   │
                                                        │   (mirrors WH FW wire side)    │   │
                                                        │  PFC pause native              │   │
                                                        │  ───────────────────────       │   │
                                                        │  egress A: host RAM (QDMA)     │   │
                                                        │  egress B: another TT chip ────┼───┘
                                                        └────────────────────────────────┘
                                                              (NoC fabric over Ethernet,
                                                               line rate, host bypassed)
```

Legend for the NoC-fabric strip (the "where does the byte land" overlay,
since `mr.base_noc_addr` is just a 64-bit NoC-encoded address that resolves
to one of four destinations):

| Glyph | Region | Bandwidth | Typical Qwen3 use | Typical RDMA use |
|---|---|---|---|---|
| `░` | Host hugepage (via PCIe NoC tile) | **~25 Gbps** (PCIe Gen3 x4) | token-out, sampled logits | `SEND` payloads, host-side MR |
| `▒` | Tensix L1 | **~TB/s peak** local | per-layer scratch, q/k_norm intermediates | RDMA scratchpads, inline RPC buffers |
| `▓` | DRAM banks | **~512 GB/s aggregate** | weights (1.2 GB), KV cache (4.8 GB) | bulk WRITE/READ MRs |
| `█` | ETH-NoC tile (to/from CMAC) | **~100 Gbps wire** | (not directly addressed by Qwen3) | wire ingress/egress |

## Reading the diagram

1. **The chip is the heavy black box.** Everything inside is one Wormhole
   die. The Qwen3 runtime (Tensix grid) and the external-eth FW (erisc core)
   are *peer producers/consumers* of the same NoC, not a stack.
2. **Three on-chip lanes meet at the NoC fabric strip.** That strip is the
   abstraction `mr.base_noc_addr` exposes — the same FW handler services a
   write to host RAM, to tensix L1, to DRAM, or out the wire, depending on
   which region the registered MR points at.
3. **The control plane is dotted on purpose.** `register_mr_slot` is a
   one-time doorbell at `RCB+0x28` (Phase P7). After it lands, the host is
   **not in the data path** for WRITE/READ — only `SEND`-family ops touch the
   host RxWqeRing. The diagram would lie if the control arrow were solid.
4. **Wire bandwidth is the bottleneck the FPGA path fixes.** PCIe Gen3 x4
   caps the host-RDMA lane at ~25 Gbps regardless of wire speed. The whole
   point of FPGA-TT-Link is that *neither end of the wire is PCIe-bound*: the
   FPGA's two egresses can either spill into local host RAM (QDMA) **or**
   relay straight to another TT chip's NoC, in which case the wire runs at
   line rate end-to-end with the host fully bypassed.
5. **Coexistence is L1/DRAM-budget-driven, not bandwidth-driven.** The Qwen3
   runtime owns ~6 GB of DRAM (weights + KV cache) and most of the L1; the
   erisc FW owns ~0x29000 of L1 for its RX ring + TX WQE pool + MR table at
   `L1:0x8500`. They never compete for the same NoC tile in steady state —
   Qwen3 traffic is Tensix↔DRAM, FW traffic is ETH-NoC↔(L1|DRAM|PCIe).

## Why FPGA-TT-Link is the right partner for this diagram

Cross-referencing the partner table in `README.md`:

| Aspect | FPGA-TT-Link path |
|---|---|
| WH FW | unchanged (`erisc_cmac_simple`) |
| Wire protocol | unchanged (TT-RDMA v1) |
| Host SDK | unchanged (`TtRdmaEndpoint`) |
| Partner ingress | QDMA → host RAM **or** NoC fabric → remote TT |
| Sustained Gbps | **~100 Gbps wire** (FPGA-dependent, not PCIe-bound) |
| PFC | FPGA-programmable, native |
| Host involvement (data path) | optional — bypassable via egress B |

The diagram's right-hand side reflects this: the FPGA NIC has **two egress
arrows**, and the chip-to-chip one loops back into another TT chip's NoC
fabric. That is the TT-Link vision in one picture.

## When *not* to use this diagram

- Don't use it to reason about the FW dispatch state machine — see
  `tt-rdma-fw-arch-rx.md` for the opcode-handler flow.
- Don't use it to reason about wire framing — see
  `tt-rdma-wire-protocol-v1.md` for the 32-byte header.
- Don't use it to reason about reliability — Phase R cumulative-ACK +
  v1.1 selective-ACK live in the FW spec, not on this picture.

This figure exists to answer one question: **"if I'm running an inference
workload on this chip, what does the external-eth path look like next to it,
and where does the FPGA-TT-Link partner sit?"**
