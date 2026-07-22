// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Mellanox-side counterpart to test_external_cmac_smoke.cpp.
//
// Replaces the FPGA bridge (open-nic-shell-tt-link) with a stock Mellanox NIC
// for first WH bring-up. Receives raw L2 Ethernet frames with EtherType
// 0x1AF4 (TT-link) on the configured DPDK port, prints the first frame, then
// echoes it back (dst/src MAC swapped) for as many frames as --count requests.
//
// Run model: terminal 1 runs this binary, terminal 2 runs
// test_external_cmac_smoke (with TT_FPGA_BAR unset so the FPGA stat check is
// skipped). Verifies the WH erisc emission lands on the wire byte-for-byte
// without needing the FPGA bitstream.
//
// Usage:
//   sudo ./setup_hugepages_mlx_smoke.sh
//   sudo ./test_external_cmac_smoke_mlx -l 0,1 -n 4 -a 0000:02:00.0 -- --count 1 --ethertype 0x1AF4
//
// Args after `--` are app args; everything before is EAL.
//
// Requirements:
//   - DPDK 22.11+ (Mellanox build at /opt/mellanox/dpdk)
//   - ConnectX-5 or newer
//   - Hugepages configured (1024 × 2MB is plenty)
//   - Root or CAP_NET_RAW + CAP_NET_ADMIN
//   - Interface in UP state (ip link set <ifname> up) — mlx5 is bifurcated
//     so the kernel netdev and DPDK can coexist on the same port

#include <arpa/inet.h>
#include <chrono>
#include <signal.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

namespace {

constexpr uint16_t kDefaultEtherType = 0x1AF4;
constexpr uint16_t kRxQueue = 0;
constexpr uint16_t kTxQueue = 0;
constexpr uint16_t kNbRxDesc = 4096;
constexpr uint16_t kNbTxDesc = 1024;
constexpr uint32_t kMbufPoolSize = 16384;
constexpr uint32_t kMbufCacheSize = 256;
constexpr uint16_t kBurstSize = 64;

volatile sig_atomic_t g_stop = 0;

void on_sigint(int) { g_stop = 1; }

struct AppArgs {
    uint16_t port_id = 0;
    uint16_t ether_type = kDefaultEtherType;
    int count = 1;
    // Reverse-direction sanity check: when >0, skip RX/flow setup and TX N
    // raw L2 broadcast frames at --ethertype, then exit. Used to verify
    // MLX→WH direction independently of the WH→MLX path that's currently
    // being silently dropped inside mlx5's MAC.
    int tx_probe = 0;
    // Payload size in bytes for the L2 frame body (excluding 14B Ethernet
    // header and 4B FCS). 256 mirrors the WH gw frame's host buffer size,
    // giving a 274-byte wire frame that lands in the WH CMAC's
    // ETH_RXQ_PKT_END_CNT[0|1] regs (which the firmware sums into
    // RESULTS_BUF_ADDR+32 / 0x1E20 every burst-loop iteration, surfaced by
    // tt-exalens eth_packet_test.py --check-status as "RX count").
    int tx_probe_size = 256;
    // Unicast dst MAC for --tx-probe. Default ff:ff:ff:ff:ff:ff (broadcast),
    // but broadcast is consumed inside mlx5's local eSwitch on the bifurcated
    // ConnectX-5 port and never reaches the wire-egress to WH. For pure-RX
    // experiments use the WH src MAC ("aa:bb:cc:dd:ee:00") which CMAC's RX
    // filter accepts (verified via the echo loop at 994k/1M admit rate).
    uint8_t tx_probe_dst[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    // Phase B (TT-RDMA v1): when set, frames carry a 32-byte RDMA header
    // [opcode|version_flags|tag|length|seq|rkey|remote_offset|imm|hdr_crc]
    // at wire-payload offset 0 (after the 14-byte L2 the DPDK app builds).
    // WH FW sees the header at BUF offset 0 (CMAC RX strips wire L2) and
    // dispatches by opcode. opcode=0x01=SEND, length=payload bytes after the
    // header. Wire ethertype unchanged (0x1AF4 today) — header detection on
    // WH is heuristic (first-byte opcode lookup) until ethertype 0x1AF6 lands.
    bool tx_probe_rdma = false;
    // P3 (TT-RDMA v1 §2.3): when set with --tx-probe-rdma, emit opcode 0x10
    // (WRITE) instead of 0x01 (SEND). Header carries rkey + remote_offset so
    // WH FW deposits payload at mr.base + remote_offset.
    bool tx_probe_write = false;
    bool tx_probe_write_imm = false;  // Q(b): opcode 0x11
    bool tx_probe_send_imm = false;   // Q(b): opcode 0x02
    bool tx_probe_ack = false;        // Phase R: opcode 0x40 (standalone ACK)
    bool rx_echo_read = false;        // Phase I: respond to incoming opcode 0x20 with READ_RESP (0x21)
    // P6: emit opcode 0x20 (READ_REQ). `length` field carries BYTES REQUESTED;
    // payload is empty on wire. Cannot combine with --tx-probe-write.
    bool tx_probe_read = false;
    uint32_t tx_probe_rkey = 0x00000001u;  // matches FW MVP MR slot 0
    uint64_t tx_probe_remote = 0;          // base remote_offset for frame 0
    uint64_t tx_probe_stride = 256;        // remote_offset += stride per frame
    // Soak mode: skip per-frame stdout, probe flows, catchall, and COUNT
    // actions on the main flow. The diagnostic mode prints/hexdumps every
    // frame in the gw size bucket — that's fine for a one-shot trigger but
    // self-DOSes the receiver under soak load (~150kfps). With soak mode on,
    // the receiver only logs counters at heartbeat and lets DPDK + NIC_RX
    // hardware steering run unimpeded.
    bool soak_mode = false;

    // Phase 2 reliability: send cumulative ACK frames back to WH every K
    // matched frames or every T_us microseconds (whichever first). Off by
    // default — Phase 1 soaks don't expect ACK return traffic.
    bool ack_enable = false;
    int ack_every_frames = 64;        // matches sender's ring N
    int ack_every_us = 1000;          // 1 ms idle-flush bound
    uint16_t ack_ethertype = 0x1AF4;  // Phase R: same as data; FW uses opcode 0x40, not ethertype

    // R-validation: drop K% of incoming data frames before they reach the
    // ack-tracking logic. From the WH sender's POV those seq#s never ack,
    // so its retx timer fires and it re-sends the unacked window. Exercises
    // the reliability path end-to-end. 0 (default) = no drops.
    int ack_drop_pct = 0;
};

bool parse_app_args(int argc, char** argv, AppArgs& out) {
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--port-id") && i + 1 < argc) {
            out.port_id = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if (!std::strcmp(a, "--ethertype") && i + 1 < argc) {
            out.ether_type = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 0));
        } else if (!std::strcmp(a, "--count") && i + 1 < argc) {
            out.count = std::atoi(argv[++i]);
        } else if (!std::strcmp(a, "--tx-probe") && i + 1 < argc) {
            out.tx_probe = std::atoi(argv[++i]);
        } else if (!std::strcmp(a, "--tx-probe-size") && i + 1 < argc) {
            out.tx_probe_size = std::atoi(argv[++i]);
        } else if (!std::strcmp(a, "--tx-probe-rdma")) {
            out.tx_probe_rdma = true;
        } else if (!std::strcmp(a, "--tx-probe-write")) {
            out.tx_probe_write = true;
        } else if (!std::strcmp(a, "--tx-probe-read")) {
            out.tx_probe_read = true;
        } else if (!std::strcmp(a, "--tx-probe-write-imm")) {
            out.tx_probe_write_imm = true;
        } else if (!std::strcmp(a, "--tx-probe-send-imm")) {
            out.tx_probe_send_imm = true;
        } else if (!std::strcmp(a, "--tx-probe-ack")) {
            out.tx_probe_ack = true;
        } else if (!std::strcmp(a, "--rx-echo-read")) {
            out.rx_echo_read = true;
        } else if (!std::strcmp(a, "--tx-probe-rkey") && i + 1 < argc) {
            out.tx_probe_rkey = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
        } else if (!std::strcmp(a, "--tx-probe-remote") && i + 1 < argc) {
            out.tx_probe_remote = std::strtoull(argv[++i], nullptr, 0);
        } else if (!std::strcmp(a, "--tx-probe-stride") && i + 1 < argc) {
            out.tx_probe_stride = std::strtoull(argv[++i], nullptr, 0);
        } else if (!std::strcmp(a, "--tx-probe-dst") && i + 1 < argc) {
            unsigned int b[6];
            if (std::sscanf(argv[++i], "%x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
                std::fprintf(stderr, "Bad --tx-probe-dst MAC (expected aa:bb:cc:dd:ee:ff)\n");
                return false;
            }
            for (int k = 0; k < 6; ++k) {
                out.tx_probe_dst[k] = static_cast<uint8_t>(b[k]);
            }
        } else if (!std::strcmp(a, "--soak-mode")) {
            out.soak_mode = true;
        } else if (!std::strcmp(a, "--ack-enable")) {
            out.ack_enable = true;
        } else if (!std::strcmp(a, "--ack-every-frames") && i + 1 < argc) {
            out.ack_every_frames = std::atoi(argv[++i]);
        } else if (!std::strcmp(a, "--ack-every-us") && i + 1 < argc) {
            out.ack_every_us = std::atoi(argv[++i]);
        } else if (!std::strcmp(a, "--ack-ethertype") && i + 1 < argc) {
            out.ack_ethertype = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 0));
        } else if (!std::strcmp(a, "--ack-drop-pct") && i + 1 < argc) {
            out.ack_drop_pct = std::atoi(argv[++i]);
        } else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            std::printf(
                "Usage: %s <EAL args> -- [--port-id N] [--ethertype 0xNNNN] [--count N]\n"
                "                            [--tx-probe N] [--tx-probe-size B]\n"
                "  --port-id        DPDK port ID (default 0)\n"
                "  --ethertype      EtherType for receive filter / TX probe (default 0x1AF4)\n"
                "  --count          Echo this many frames then exit (default 1; 0 = forever)\n"
                "  --tx-probe N     TX-only mode: send N broadcast L2 frames and exit.\n"
                "                   Skips RX queue + flow rule setup. Use to validate\n"
                "                   MLX→WH direction (the reverse of the gw frame path).\n"
                "  --tx-probe-size  L2 payload bytes (default 256). Wire frame =\n"
                "                   14 (eth hdr) + payload + 4 (FCS). 256 mirrors\n"
                "                   the WH gw-frame size for symmetric counter checking.\n",
                argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown app arg: %s\n", a);
            return false;
        }
    }
    return true;
}

void hexdump(const uint8_t* data, size_t len, size_t max = 64) {
    size_t n = len < max ? len : max;
    for (size_t i = 0; i < n; i++) {
        std::printf("%02x", data[i]);
        if ((i + 1) % 2 == 0) {
            std::printf(" ");
        }
        if ((i + 1) % 16 == 0) {
            std::printf("\n");
        }
    }
    if (n % 16 != 0) {
        std::printf("\n");
    }
    if (len > max) {
        std::printf("... (%zu more bytes)\n", len - max);
    }
}

void print_mac(const char* tag, const struct rte_ether_addr& a) {
    std::printf(
        "%s %02x:%02x:%02x:%02x:%02x:%02x",
        tag,
        a.addr_bytes[0],
        a.addr_bytes[1],
        a.addr_bytes[2],
        a.addr_bytes[3],
        a.addr_bytes[4],
        a.addr_bytes[5]);
}

int port_init(uint16_t port_id, struct rte_mempool* mbuf_pool, int probe_payload_bytes) {
    struct rte_eth_conf port_conf = {};
    // Q(a) jumbo: lift MTU when probe payload is bigger than standard MTU so
    // the NIC + driver build descriptors for the larger frame. Cap at 9216 B
    // which is what most NICs / partner switches handle.
    const uint16_t effective_mtu =
        static_cast<uint16_t>(std::min(9216, std::max<int>(RTE_ETHER_MTU, probe_payload_bytes + 128)));
    port_conf.rxmode.mtu = effective_mtu;

    int ret = rte_eth_dev_configure(port_id, /*nb_rx*/ 1, /*nb_tx*/ 1, &port_conf);
    if (ret != 0) {
        std::fprintf(stderr, "rte_eth_dev_configure(%u) failed: %d\n", port_id, ret);
        return ret;
    }

    uint16_t nb_rxd = kNbRxDesc;
    uint16_t nb_txd = kNbTxDesc;
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rxd, &nb_txd);
    if (ret != 0) {
        std::fprintf(stderr, "rte_eth_dev_adjust_nb_rx_tx_desc(%u) failed: %d\n", port_id, ret);
        return ret;
    }

    ret = rte_eth_rx_queue_setup(port_id, kRxQueue, nb_rxd, rte_eth_dev_socket_id(port_id), nullptr, mbuf_pool);
    if (ret < 0) {
        std::fprintf(stderr, "rte_eth_rx_queue_setup(%u) failed: %d\n", port_id, ret);
        return ret;
    }

    ret = rte_eth_tx_queue_setup(port_id, kTxQueue, nb_txd, rte_eth_dev_socket_id(port_id), nullptr);
    if (ret < 0) {
        std::fprintf(stderr, "rte_eth_tx_queue_setup(%u) failed: %d\n", port_id, ret);
        return ret;
    }

    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        std::fprintf(stderr, "rte_eth_dev_start(%u) failed: %d\n", port_id, ret);
        return ret;
    }

    rte_eth_promiscuous_enable(port_id);

    struct rte_ether_addr mac;
    rte_eth_macaddr_get(port_id, &mac);
    print_mac("Port MAC", mac);
    std::printf("\n");

    struct rte_eth_link link;
    if (rte_eth_link_get_nowait(port_id, &link) != 0) {
        std::printf("Port %u link: query failed\n", port_id);
    } else {
        std::printf(
            "Port %u link: %s, speed %u Mbps, %s-duplex\n",
            port_id,
            link.link_status ? "UP" : "DOWN",
            link.link_speed,
            link.link_duplex == RTE_ETH_LINK_FULL_DUPLEX ? "full" : "half");
    }

    return 0;
}

// Install rte_flow rule: ETH(type=ether_type) → [COUNT] + QUEUE(rx_queue).
// COUNT lets us read how many frames the NIC's flow engine matched
// independent of whether queue 0 ever delivered them to DPDK rx_burst.
// Skipping COUNT (with_count=false) keeps the rule minimal — useful in
// soak mode since flow counters can be a steering-pressure resource.
struct rte_flow* install_flow(uint16_t port_id, uint16_t ether_type, uint16_t rx_queue, bool with_count = true) {
    struct rte_flow_attr attr = {};
    attr.ingress = 1;

    struct rte_flow_item_eth eth_spec = {};
    struct rte_flow_item_eth eth_mask = {};
    eth_spec.type = rte_cpu_to_be_16(ether_type);
    eth_mask.type = 0xFFFF;

    struct rte_flow_item pattern[2] = {};
    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[0].spec = &eth_spec;
    pattern[0].mask = &eth_mask;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_END;

    struct rte_flow_action_count count_action = {};
    struct rte_flow_action_queue queue_action = {};
    queue_action.index = rx_queue;

    struct rte_flow_action actions[3] = {};
    int ai = 0;
    if (with_count) {
        actions[ai].type = RTE_FLOW_ACTION_TYPE_COUNT;
        actions[ai].conf = &count_action;
        ai++;
    }
    actions[ai].type = RTE_FLOW_ACTION_TYPE_QUEUE;
    actions[ai].conf = &queue_action;
    ai++;
    actions[ai].type = RTE_FLOW_ACTION_TYPE_END;

    struct rte_flow_error error = {};
    int ret = rte_flow_validate(port_id, &attr, pattern, actions, &error);
    if (ret != 0) {
        std::fprintf(
            stderr,
            "rte_flow_validate failed: type=%d msg=%s\n",
            error.type,
            error.message ? error.message : "(no message)");
        return nullptr;
    }

    struct rte_flow* flow = rte_flow_create(port_id, &attr, pattern, actions, &error);
    if (flow == nullptr) {
        std::fprintf(
            stderr,
            "rte_flow_create failed: type=%d msg=%s\n",
            error.type,
            error.message ? error.message : "(no message)");
        return nullptr;
    }

    std::printf("Flow installed: ETH(type=0x%04x) → %squeue %u\n", ether_type, with_count ? "COUNT + " : "", rx_queue);
    return flow;
}

bool query_flow_count(uint16_t port_id, struct rte_flow* flow, uint64_t& hits, uint64_t& bytes) {
    struct rte_flow_action_count count_action = {};
    struct rte_flow_action actions[2] = {};
    actions[0].type = RTE_FLOW_ACTION_TYPE_COUNT;
    actions[0].conf = &count_action;
    actions[1].type = RTE_FLOW_ACTION_TYPE_END;

    struct rte_flow_query_count query = {};
    query.reset = 0;
    struct rte_flow_error error = {};
    int ret = rte_flow_query(port_id, flow, &actions[0], &query, &error);
    if (ret != 0) {
        std::fprintf(
            stderr,
            "rte_flow_query failed: type=%d msg=%s\n",
            error.type,
            error.message ? error.message : "(no message)");
        return false;
    }
    hits = query.hits;
    bytes = query.bytes;
    return true;
}

// Phase R (2026-05-13): build + TX a cumulative ACK frame back to WH as a
// proper v1 RDMA opcode 0x40 frame (replaces the old magic-prefix path).
//
// FW handler (main_cmac.cc case 0x40): reads ack_seq at the standard v1
// header's `seq` field (offset +8) and updates rcb[2] (acked_idx) using a
// wrap-safe monotonic compare.
//
// Wire format (256 B excl. FCS):
//   [dst=02:00:00:00:00:01]   - unicast (avoid mlx5 eswitch broadcast drops)
//   [src=NIC MAC]
//   [ethertype=0x1AF4]        - same as data path; WH CMAC strips L2
//   [u32: opcode=0x40 | ver=0x01 | tag(u16)]
//   [u32: length=0]           - ACK carries no payload
//   [u32: seq=ack_seq]        - the field FW reads
//   [u32: rkey=0]
//   [u64: remote_offset=0]
//   [u32: imm=ack_flags]      - opportunistic carry for flags
//   [u32: hdr_crc=0]
//   [padding zeros to kAckFrameBytes]
int send_ack_frame(
    uint16_t port_id,
    struct rte_mempool* mbuf_pool,
    const struct rte_ether_addr& src_mac,
    uint16_t ack_ethertype,
    uint32_t ack_seq,
    uint32_t ack_flags) {
    constexpr int kAckFrameBytes = 256;  // small but safely above Ethernet min
    struct rte_mbuf* m = rte_pktmbuf_alloc(mbuf_pool);
    if (m == nullptr) {
        return 0;
    }
    uint8_t* buf = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(m, kAckFrameBytes));
    if (buf == nullptr) {
        rte_pktmbuf_free(m);
        return 0;
    }
    std::memset(buf, 0, kAckFrameBytes);

    struct rte_ether_hdr* eh = reinterpret_cast<struct rte_ether_hdr*>(buf);
    // Unicast dst — broadcast at small sizes worked historically but
    // we standardise on the same dst the data path uses.
    eh->dst_addr.addr_bytes[0] = 0x02;
    eh->dst_addr.addr_bytes[1] = 0x00;
    eh->dst_addr.addr_bytes[2] = 0x00;
    eh->dst_addr.addr_bytes[3] = 0x00;
    eh->dst_addr.addr_bytes[4] = 0x00;
    eh->dst_addr.addr_bytes[5] = 0x01;
    eh->src_addr = src_mac;
    eh->ether_type = rte_cpu_to_be_16(ack_ethertype);

    uint8_t* p = buf + sizeof(*eh);
    p[0] = 0x40;  // opcode = ACK
    p[1] = 0x01;  // version_flags
    uint16_t tag = static_cast<uint16_t>(ack_seq & 0xFFFFu);
    std::memcpy(p + 2, &tag, 2);
    uint32_t length = 0;
    std::memcpy(p + 4, &length, 4);
    std::memcpy(p + 8, &ack_seq, 4);     // <-- FW reads here
    std::memset(p + 12, 0, 12);          // rkey + remote_offset
    std::memcpy(p + 24, &ack_flags, 4);  // imm carries flags
    std::memset(p + 28, 0, 4);           // hdr_crc

    return rte_eth_tx_burst(port_id, kTxQueue, &m, 1) == 1 ? 1 : (rte_pktmbuf_free(m), 0);
}

// Phase I: build + TX a READ_RESP (opcode 0x21) in response to a READ_REQ
// (opcode 0x20). Echoes the request's tag, sets length to the request's
// `length` field (which is bytes-requested), fills payload with a
// deterministic pattern (byte i = (i ^ tag) & 0xFF) so the host can verify
// landing correctness.
int send_read_resp(
    uint16_t port_id,
    struct rte_mempool* mbuf_pool,
    const struct rte_ether_addr& src_mac,
    const struct rte_ether_addr& dst_mac,
    uint16_t ethertype,
    uint16_t tag,
    uint32_t length,
    uint32_t req_seq) {
    // Wire frame = 14 B L2 + 32 B RDMA hdr + length B payload, rounded up to
    // a reasonable minimum. Cap at the mbuf body size minus L2.
    const uint32_t wire_payload = 32u + length;
    const uint32_t wire_min = 64u;  // Ethernet min
    const uint32_t wire_size = (wire_payload < wire_min) ? wire_min : wire_payload;

    struct rte_mbuf* m = rte_pktmbuf_alloc(mbuf_pool);
    if (m == nullptr) {
        return 0;
    }
    uint8_t* buf = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(m, sizeof(rte_ether_hdr) + wire_size));
    if (buf == nullptr) {
        rte_pktmbuf_free(m);
        return 0;
    }
    std::memset(buf, 0, sizeof(rte_ether_hdr) + wire_size);

    struct rte_ether_hdr* eh = reinterpret_cast<struct rte_ether_hdr*>(buf);
    eh->dst_addr = dst_mac;
    eh->src_addr = src_mac;
    eh->ether_type = rte_cpu_to_be_16(ethertype);

    uint8_t* p = buf + sizeof(*eh);
    p[0] = 0x21;                      // opcode READ_RESP
    p[1] = 0x01;                      // version_flags
    std::memcpy(p + 2, &tag, 2);      // tag (echoed)
    std::memcpy(p + 4, &length, 4);   // length (bytes that follow)
    std::memcpy(p + 8, &req_seq, 4);  // seq (echo)
    // bytes +12..+31: rkey/remote_off/imm/hdr_crc all already zero
    // Payload: deterministic pattern.
    for (uint32_t i = 0; i < length; ++i) {
        p[32 + i] = static_cast<uint8_t>((i ^ tag) & 0xFFu);
    }

    return rte_eth_tx_burst(port_id, kTxQueue, &m, 1) == 1 ? 1 : (rte_pktmbuf_free(m), 0);
}

// Echo: swap dst↔src MAC in place, re-transmit on tx_queue.
// Frame stays on the same mbuf — single-segment small frames, no chain handling.
int echo_frame(uint16_t port_id, struct rte_mbuf* m) {
    if (m->nb_segs != 1) {
        std::fprintf(stderr, "WARN: multi-segment mbuf (%u segs) — dropping\n", m->nb_segs);
        rte_pktmbuf_free(m);
        return 0;
    }
    struct rte_ether_hdr* eh = rte_pktmbuf_mtod(m, struct rte_ether_hdr*);
    struct rte_ether_addr tmp = eh->dst_addr;
    eh->dst_addr = eh->src_addr;
    eh->src_addr = tmp;

    uint16_t sent = rte_eth_tx_burst(port_id, kTxQueue, &m, 1);
    if (sent != 1) {
        rte_pktmbuf_free(m);
        return -1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int eal_args = rte_eal_init(argc, argv);
    if (eal_args < 0) {
        std::fprintf(stderr, "rte_eal_init failed\n");
        return 1;
    }
    argc -= eal_args;
    argv += eal_args;

    AppArgs args;
    if (!parse_app_args(argc, argv, args)) {
        rte_eal_cleanup();
        return 1;
    }

    if (rte_eth_dev_count_avail() == 0) {
        std::fprintf(stderr, "No DPDK ports available — check -a <pci-bdf> EAL arg\n");
        rte_eal_cleanup();
        return 1;
    }

    // Q(a) jumbo TX path: bump mbuf data size when the user picks --tx-probe-size
    // > the default mbuf body (2176 - headroom ≈ 2048). For 4080 B jumbo we need
    // 9216 B mbuf bodies. RTE_PKTMBUF_HEADROOM is 128 by default.
    const uint16_t kJumboMbufSize = static_cast<uint16_t>(std::max<int>(
        RTE_MBUF_DEFAULT_BUF_SIZE,
        args.tx_probe_size + static_cast<int>(sizeof(rte_ether_hdr)) + RTE_PKTMBUF_HEADROOM + 512));
    struct rte_mempool* mbuf_pool =
        rte_pktmbuf_pool_create("MBUF_POOL", kMbufPoolSize, kMbufCacheSize, 0, kJumboMbufSize, rte_socket_id());
    if (mbuf_pool == nullptr) {
        std::fprintf(stderr, "rte_pktmbuf_pool_create failed: %s\n", rte_strerror(rte_errno));
        rte_eal_cleanup();
        return 1;
    }

    if (port_init(args.port_id, mbuf_pool, args.tx_probe_size) != 0) {
        rte_eal_cleanup();
        return 1;
    }

    // TX-probe (reverse-direction sanity): MLX → WH only. Build N broadcast L2
    // frames at args.ether_type with port MAC as src, hand to rte_eth_tx_burst,
    // then tear down. WH-side observation: eth_packet_test.py --check-status
    // before/after — RX count delta should equal N if the MLX→WH wire is good
    // (independent of the WH→MLX drop currently under investigation).
    if (args.tx_probe > 0) {
        struct rte_ether_addr src_mac;
        rte_eth_macaddr_get(args.port_id, &src_mac);
        const int payload_bytes = args.tx_probe_size;
        const int frame_bytes = static_cast<int>(sizeof(struct rte_ether_hdr)) + payload_bytes;

        std::printf(
            "TX-probe: sending %d frame(s), %d B wire (excl. FCS), ethertype 0x%04x\n",
            args.tx_probe,
            frame_bytes,
            args.ether_type);
        print_mac("  src", src_mac);
        std::printf("\n");
        std::printf(
            "  dst %02x:%02x:%02x:%02x:%02x:%02x\n",
            args.tx_probe_dst[0],
            args.tx_probe_dst[1],
            args.tx_probe_dst[2],
            args.tx_probe_dst[3],
            args.tx_probe_dst[4],
            args.tx_probe_dst[5]);

        int sent_total = 0;
        int alloc_fail = 0;
        int tx_fail = 0;
        auto tx_t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < args.tx_probe; ++i) {
            struct rte_mbuf* m = rte_pktmbuf_alloc(mbuf_pool);
            if (m == nullptr) {
                alloc_fail++;
                continue;
            }

            uint8_t* buf = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(m, frame_bytes));
            if (buf == nullptr) {
                rte_pktmbuf_free(m);
                alloc_fail++;
                continue;
            }

            struct rte_ether_hdr* eh = reinterpret_cast<struct rte_ether_hdr*>(buf);
            std::memcpy(&eh->dst_addr, args.tx_probe_dst, 6);
            eh->src_addr = src_mac;
            eh->ether_type = rte_cpu_to_be_16(args.ether_type);

            uint8_t* p = buf + sizeof(*eh);
            uint32_t seq = static_cast<uint32_t>(i);

            if (args.tx_probe_rdma) {
                // Phase B (TT-RDMA v1): write 32-byte RDMA header at wire
                // payload offset 0. CMAC RX strips the 14-byte wire L2, so WH
                // FW sees this header at BUF[0]. Layout matches
                // /tmp/TT-RDMA-v1.md §2.2.
                //   +0    opcode = 0x01 (SEND)
                //   +1    version_flags = 0x01 (ver=1, no flags)
                //   +2    tag = low 16 bits of seq# (correlation cookie)
                //   +4    length = payload bytes AFTER the 32-byte header
                //   +8    seq = i
                //   +12   rkey = 0 (no MR; SEND only)
                //   +16   remote_offset = 0
                //   +24   imm_data = 0
                //   +28   header_cksum = 0 (Phase B: CRC validation deferred)
                constexpr int kRdmaHdrBytes = 32;
                int rdma_payload = payload_bytes - kRdmaHdrBytes;
                if (rdma_payload < 0) {
                    rdma_payload = 0;
                }

                uint8_t op = 0x01;  // default SEND
                if (args.tx_probe_write) {
                    op = 0x10;
                }
                if (args.tx_probe_read) {
                    op = 0x20;
                }
                if (args.tx_probe_send_imm) {
                    op = 0x02;
                }
                if (args.tx_probe_write_imm) {
                    op = 0x11;
                }
                if (args.tx_probe_ack) {
                    op = 0x40;
                }
                p[0] = op;
                p[1] = 0x01;  // ver 1
                uint16_t tag = static_cast<uint16_t>(seq & 0xFFFF);
                std::memcpy(p + 2, &tag, 2);
                uint32_t plen = static_cast<uint32_t>(rdma_payload);
                std::memcpy(p + 4, &plen, 4);
                std::memcpy(p + 8, &seq, 4);
                if (args.tx_probe_write || args.tx_probe_read || args.tx_probe_write_imm) {
                    // WRITE / WRITE_IMM / READ_REQ all carry rkey + remote_off.
                    std::memcpy(p + 12, &args.tx_probe_rkey, 4);
                    uint64_t remote = args.tx_probe_remote + static_cast<uint64_t>(i) * args.tx_probe_stride;
                    std::memcpy(p + 16, &remote, 8);
                    // Q(b): WRITE_IMM carries imm_data at +24 — use seq so the
                    // host can verify which frame's imm landed in which slot.
                    if (args.tx_probe_write_imm) {
                        std::memcpy(p + 24, &seq, 4);
                        std::memset(p + 28, 0, 4);
                    } else {
                        std::memset(p + 24, 0, 8);
                    }
                } else if (args.tx_probe_send_imm) {
                    // SEND_IMM: no rkey/remote_offset, but imm at +24.
                    std::memset(p + 12, 0, 12);
                    std::memcpy(p + 24, &seq, 4);
                    std::memset(p + 28, 0, 4);
                } else {
                    std::memset(p + 12, 0, 20);  // rkey..hdr_crc = 0
                }

                // Mark first payload byte after the header so we can confirm
                // FW stride-walked to the right offset under pile-up.
                if (rdma_payload > 0) {
                    p[kRdmaHdrBytes] = 0xA5;
                    // P3: put seq in next 4 bytes of payload so tt-exalens can
                    // verify which frame landed where on the WRITE path.
                    if (rdma_payload >= 5) {
                        std::memcpy(p + kRdmaHdrBytes + 1, &seq, 4);
                    }
                }
            } else {
                // Legacy probe: 16-byte magic + 32-bit sequence at payload offset 0.
                const char kMagic[16] = {
                    'M', 'L', 'X', 'T', 'X', 'P', 'R', 'O', 'B', 'E', '\0', '\0', '\0', '\0', '\0', '\0'};
                std::memcpy(p, kMagic, 16);
                std::memcpy(p + 16, &seq, 4);
            }

            // Q(a) follow-up: retry until the NIC TX ring accepts the frame
            // (or a bounded retry budget is exhausted). At jumbo the TX ring
            // backpressures heavily — without retry, ~96% of frames hit
            // tx_fail and the measured Gbps is a small fraction of wire rate.
            uint16_t sent = 0;
            for (int r = 0; r < 1000; r++) {
                sent = rte_eth_tx_burst(args.port_id, kTxQueue, &m, 1);
                if (sent == 1) {
                    break;
                }
                // Tiny back-off — let TX ring drain. ~1 µs per spin.
                rte_pause();
            }
            if (sent == 1) {
                sent_total++;
            } else {
                rte_pktmbuf_free(m);
                tx_fail++;
            }
        }

        // Drain TX completions before close — DPDK PMDs are asynchronous; without
        // this the last few frames can sit in the TX ring and never make it to
        // the wire when rte_eth_dev_stop runs.
        rte_delay_ms(50);

        auto tx_t1 = std::chrono::steady_clock::now();
        double tx_secs = std::chrono::duration<double>(tx_t1 - tx_t0).count();
        // Compute observed wire-rate Gbps based on sent_total + frame_bytes
        // (excludes FCS but includes the 14 B L2 + 32 B RDMA hdr + payload).
        double gbps = (sent_total > 0 && tx_secs > 0)
                          ? (static_cast<double>(sent_total) * frame_bytes * 8.0) / (tx_secs * 1e9)
                          : 0.0;
        double mfps = (sent_total > 0 && tx_secs > 0) ? (static_cast<double>(sent_total) / (tx_secs * 1e6)) : 0.0;
        std::printf(
            "TX-probe done: sent=%d alloc_fail=%d tx_fail=%d elapsed=%.3fs rate=%.3f Mfps = %.3f Gbps (wire frame %d "
            "B)\n",
            sent_total,
            alloc_fail,
            tx_fail,
            tx_secs,
            mfps,
            gbps,
            frame_bytes);
        std::fflush(stdout);

        rte_eth_dev_stop(args.port_id);
        rte_eth_dev_close(args.port_id);
        rte_eal_cleanup();
        return sent_total == args.tx_probe ? 0 : 2;
    }

    // Multi-ethertype probe: install separate COUNT rules for each candidate
    // ethertype. The gw frame's wire ethertype is unknown — could be 0x1AF4
    // (intended), 0x88B5 (burst USE_TYPE not cleared), 0x88B6 (B688 from
    // post-trigger register dump), 0x88B7, or in length-mode the raw frame
    // length 0x0100=256. After triggers, the counter that incremented =
    // the wire ethertype.
    struct ProbeFlow {
        uint16_t et;
        struct rte_flow* flow;
    };
    ProbeFlow probes[] = {
        {0x1AF4, nullptr},
        {0x88B5, nullptr},
        {0x88B6, nullptr},
        {0x88B7, nullptr},
        {0x0100, nullptr},
        {0xB588, nullptr},
        {0xB688, nullptr},
        {0x0001, nullptr},
    };
    constexpr int kNumProbes = sizeof(probes) / sizeof(probes[0]);
    if (args.soak_mode) {
        // Only install the main flow (the configured ethertype). No COUNT —
        // we measure delivery via rx_burst tally and ethtool xstats deltas.
        probes[0].et = args.ether_type;
        probes[0].flow = install_flow(args.port_id, args.ether_type, kRxQueue, /*with_count=*/false);
        if (probes[0].flow == nullptr) {
            std::fprintf(stderr, "Failed to install soak flow for 0x%04x\n", args.ether_type);
        }
    } else {
        for (int i = 0; i < kNumProbes; ++i) {
            probes[i].flow = install_flow(args.port_id, probes[i].et, kRxQueue);
            if (probes[i].flow == nullptr) {
                std::fprintf(stderr, "Failed to install probe flow for 0x%04x\n", probes[i].et);
            }
        }
    }
    struct rte_flow* flow = probes[0].flow;  // main flow = 0x1AF4 (queue 0 echo target)

    // Broadcast-dst probe: the gw frame's wire dst MAC is 0xFFFFFFFFFFFF
    // (set by init_tx_burst's DEST_MAC_ADDR). Match by dst MAC instead of
    // ethertype so we catch it regardless of what wire ethertype the CMAC
    // emits. mlx5's wildcard ETH (mask=0) doesn't actually match all frames
    // in practice — needs a real field with a non-zero mask.
    struct rte_flow* catchall = nullptr;
    if (!args.soak_mode) {
        struct rte_flow_attr attr = {};
        attr.ingress = 1;
        struct rte_flow_item_eth z_spec = {};
        struct rte_flow_item_eth z_mask = {};
        for (int k = 0; k < 6; ++k) {
            z_spec.dst.addr_bytes[k] = 0xFF;
            z_mask.dst.addr_bytes[k] = 0xFF;
        }
        struct rte_flow_item pattern[2] = {};
        pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
        pattern[0].spec = &z_spec;
        pattern[0].mask = &z_mask;
        pattern[1].type = RTE_FLOW_ITEM_TYPE_END;
        struct rte_flow_action_count count_action = {};
        struct rte_flow_action_queue queue_action = {};
        queue_action.index = kRxQueue;
        struct rte_flow_action actions[3] = {};
        actions[0].type = RTE_FLOW_ACTION_TYPE_COUNT;
        actions[0].conf = &count_action;
        actions[1].type = RTE_FLOW_ACTION_TYPE_QUEUE;
        actions[1].conf = &queue_action;
        actions[2].type = RTE_FLOW_ACTION_TYPE_END;
        struct rte_flow_error error = {};
        if (rte_flow_validate(args.port_id, &attr, pattern, actions, &error) == 0) {
            catchall = rte_flow_create(args.port_id, &attr, pattern, actions, &error);
        }
        std::printf(
            "Broadcast-dst rule: %s %s\n",
            catchall ? "installed" : "NOT installed",
            error.message ? error.message : "");
    }

    signal(SIGINT, on_sigint);

    std::printf(
        "Listening on port %u for EtherType 0x%04x; will echo %d frame(s)...\n",
        args.port_id,
        args.ether_type,
        args.count);

    int echoed = 0;
    uint64_t total_frames = 0;
    uint64_t last_report = 0;
    auto last_report_time = std::chrono::steady_clock::now();

    // ── Phase 2: ACK return path state ──
    // Cumulative-ACK sliding window. last_acked = highest contiguous seq#
    // observed (UINT32_MAX = sentinel = "no acks yet"; first match seq=0
    // makes last_acked=0). reorder_buf is a small circular table indexed by
    // seq mod kReorderN — covers the sender's ring window.
    constexpr uint32_t kAckSeed = 0xFFFFFFFFu;
    uint32_t last_acked = kAckSeed;
    constexpr int kReorderN = 256;  // > sender ring size (64) for headroom
    bool reorder_present[kReorderN] = {};
    uint32_t reorder_seq[kReorderN] = {};
    int frames_since_last_ack_send = 0;
    uint64_t acks_sent = 0;
    auto last_ack_send_time = std::chrono::steady_clock::now();

    struct rte_ether_addr port_mac;
    rte_eth_macaddr_get(args.port_id, &port_mac);

    auto try_send_ack = [&](uint32_t flags) {
        if (!args.ack_enable) {
            return;
        }
        if (last_acked == kAckSeed) {
            return;  // nothing yet to ack
        }
        int rc = send_ack_frame(args.port_id, mbuf_pool, port_mac, args.ack_ethertype, last_acked, flags);
        if (rc) {
            acks_sent++;
            if (acks_sent <= 3) {
                std::printf("[ack] sent ack_seq=%u (acks_sent=%llu)\n", last_acked, (unsigned long long)acks_sent);
                std::fflush(stdout);
            }
        } else if (acks_sent == 0) {
            std::printf("[ack] WARN: rte_eth_tx_burst returned 0 for ACK frame\n");
            std::fflush(stdout);
        }
        frames_since_last_ack_send = 0;
        last_ack_send_time = std::chrono::steady_clock::now();
    };
    // Track unique ethertypes seen so we can confirm whether the matched
    // ethertype is in the queue at all (vs. silently filtered upstream).
    constexpr int kEtTableSize = 32;
    uint16_t et_table[kEtTableSize] = {};
    uint64_t et_counts[kEtTableSize] = {};
    int et_n = 0;
    while (!g_stop && (args.count == 0 || echoed < args.count)) {
        struct rte_mbuf* bufs[kBurstSize];
        uint16_t n = rte_eth_rx_burst(args.port_id, kRxQueue, bufs, kBurstSize);

        // Phase 2: idle-bound ACK send. Even with no incoming match frames in
        // this burst, an ACK should be sent every ack_every_us so the sender
        // doesn't time out waiting for ack progress when load drops.
        if (args.ack_enable && last_acked != kAckSeed) {
            auto now = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_ack_send_time).count();
            if (us >= args.ack_every_us) {
                try_send_ack(0);
            }
        }

        if (n == 0) {
            // Periodic heartbeat so a silent stdout doesn't look like a hang.
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 2) {
                std::printf("[heartbeat] total_frames=%llu echoed=%d", (unsigned long long)total_frames, echoed);
                for (int i = 0; i < kNumProbes; ++i) {
                    if (probes[i].flow == nullptr) {
                        continue;
                    }
                    uint64_t h = 0, b = 0;
                    if (query_flow_count(args.port_id, probes[i].flow, h, b) && (h > 0 || b > 0)) {
                        std::printf(" 0x%04x=%llu(%lluB)", probes[i].et, (unsigned long long)h, (unsigned long long)b);
                    }
                }
                std::printf("\n");
                std::fflush(stdout);
                last_report_time = now;
                last_report = total_frames;
            }
            continue;
        }
        total_frames += n;
        // Periodic frame-rate report so we can see whether queue 0 is delivering anything.
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 2) {
            std::printf(
                "[heartbeat] total_frames=%llu (delta=%llu) echoed=%d",
                (unsigned long long)total_frames,
                (unsigned long long)(total_frames - last_report),
                echoed);
            for (int i = 0; i < kNumProbes; ++i) {
                if (probes[i].flow == nullptr) {
                    continue;
                }
                uint64_t h = 0, b = 0;
                if (query_flow_count(args.port_id, probes[i].flow, h, b) && (h > 0 || b > 0)) {
                    std::printf(" 0x%04x=%llu(%lluB)", probes[i].et, (unsigned long long)h, (unsigned long long)b);
                }
            }
            if (catchall != nullptr) {
                uint64_t h = 0, b = 0;
                if (query_flow_count(args.port_id, catchall, h, b)) {
                    std::printf(" BCAST=%llu(%lluB)", (unsigned long long)h, (unsigned long long)b);
                }
            }
            std::printf("\n");
            std::fflush(stdout);
            last_report_time = now;
            last_report = total_frames;
        }

        for (uint16_t i = 0; i < n; i++) {
            struct rte_mbuf* m = bufs[i];
            struct rte_ether_hdr* eh = rte_pktmbuf_mtod(m, struct rte_ether_hdr*);
            uint16_t et = rte_be_to_cpu_16(eh->ether_type);
            bool match = (et == args.ether_type);
            // Track ethertype distribution.
            bool found = false;
            for (int k = 0; k < et_n; ++k) {
                if (et_table[k] == et) {
                    et_counts[k]++;
                    found = true;
                    break;
                }
            }
            if (!found && et_n < kEtTableSize) {
                et_table[et_n] = et;
                et_counts[et_n] = 1;
                et_n++;
                std::printf("[new ethertype] 0x%04x first seen (total=%llu)\n", et, (unsigned long long)total_frames);
                std::fflush(stdout);
            }
            // Diagnostic-only: per-frame size-bucket hexdump and MATCH dump.
            // Gated off in soak mode — at 150kfps each printf+hexdump self-DOSes
            // the receiver, back-pressuring DPDK rx_burst and inflating the
            // measured drop rate by ~30%.
            if (!args.soak_mode) {
                if (m->pkt_len >= 256 && m->pkt_len <= 300) {
                    std::printf("[gw-size %u B] ethertype=0x%04x match=%d\n", m->pkt_len, et, (int)match);
                    hexdump(rte_pktmbuf_mtod(m, const uint8_t*), m->pkt_len, 64);
                    std::fflush(stdout);
                }
                if (eh->src_addr.addr_bytes[0] == 0xAA && eh->src_addr.addr_bytes[1] == 0xBB &&
                    eh->src_addr.addr_bytes[2] == 0xCC) {
                    std::printf("[src=AABB... at offset 6] frame %u bytes ethertype=0x%04x\n", m->pkt_len, et);
                    std::fflush(stdout);
                }
                if (match) {
                    std::printf("RX frame %u bytes ethertype=0x%04x MATCH\n", m->pkt_len, et);
                    print_mac("  dst", eh->dst_addr);
                    std::printf("\n");
                    print_mac("  src", eh->src_addr);
                    std::printf("\n");
                    hexdump(rte_pktmbuf_mtod(m, const uint8_t*), m->pkt_len);
                }
            }

            if (match) {
                // Phase I: if rx_echo_read is on and this is a READ_REQ
                // (opcode 0x20 at the first byte after the wire L2 hdr),
                // craft a READ_RESP and TX it back. The WH FW that issued
                // the READ_REQ will get the response, look up the correlation
                // entry by tag, and NoC-write the payload to the host's
                // landing MR.
                if (args.rx_echo_read && m->pkt_len >= sizeof(rte_ether_hdr) + 32) {
                    const uint8_t* p = rte_pktmbuf_mtod(m, const uint8_t*);
                    const uint8_t* rdma = p + sizeof(rte_ether_hdr);
                    if (rdma[0] == 0x20) {
                        uint16_t req_tag;
                        std::memcpy(&req_tag, rdma + 2, 2);
                        uint32_t req_length;
                        std::memcpy(&req_length, rdma + 4, 4);
                        uint32_t req_seq;
                        std::memcpy(&req_seq, rdma + 8, 4);
                        // Cap response payload at 4080 B (WH FW's req_len cap).
                        if (req_length > 4080u) {
                            req_length = 4080u;
                        }
                        // Reply destination: the sender of the request.
                        const struct rte_ether_hdr* eh = reinterpret_cast<const struct rte_ether_hdr*>(p);
                        struct rte_ether_addr dst_mac = eh->src_addr;
                        struct rte_ether_addr nic_mac;
                        rte_eth_macaddr_get(args.port_id, &nic_mac);
                        send_read_resp(
                            args.port_id, mbuf_pool, nic_mac, dst_mac, args.ether_type, req_tag, req_length, req_seq);
                    }
                }
                // Phase 2: extract seq# from the host's payload. Wire layout:
                //   [wire L2 hdr: 14B from CMAC prepend][host fake L2 hdr: 14B]
                //   [u32 seq @ offset 28][...]
                // CMAC raw-TX prepends its own 14-byte L2 header before the
                // host's buffer, so the seq# the host wrote at frame[14] sits
                // at wire byte 28.
                if (args.ack_enable && m->pkt_len >= sizeof(rte_ether_hdr) + 32) {
                    const uint8_t* p = rte_pktmbuf_mtod(m, const uint8_t*);
                    // TT-RDMA v1 seq lives at RDMA header offset +8, which
                    // is wire offset 14 + 8 = 22 (after the 14 B L2 the
                    // wire carries from WH CMAC's prepend).
                    uint32_t seq;
                    std::memcpy(&seq, p + sizeof(rte_ether_hdr) + 8, 4);

                    // R-validation: probabilistic drop. From the WH sender's
                    // POV this seq# never gets acked; the cumulative-ack
                    // window stalls at the drop point until host's tick_retx
                    // re-fires the slot and we receive it again.
                    if (args.ack_drop_pct > 0 && (std::rand() % 100) < args.ack_drop_pct) {
                        rte_pktmbuf_free(m);
                        continue;  // skip seq tracking, ack tx, echo — fully drop
                    }

                    if (last_acked == kAckSeed) {
                        // First frame: bootstrap last_acked. If the first frame
                        // is seq>0, treat seq-1 as implicitly "received before
                        // we started" — best-effort start for v1 reliability.
                        // Sender's retx covers any genuinely lost prefix.
                        last_acked = (seq == 0) ? 0u : (seq - 1u);
                        std::printf("[ack] bootstrap last_acked=%u (first seq=%u)\n", last_acked, seq);
                        std::fflush(stdout);
                        // Fall through to advance ack pointer + reorder logic.
                    }
                    if (static_cast<int32_t>(seq - last_acked) > 0) {
                        if (seq == last_acked + 1) {
                            last_acked = seq;
                            // Drain any contiguous run buffered in reorder_buf.
                            while (true) {
                                uint32_t want = last_acked + 1;
                                int idx = want % kReorderN;
                                if (reorder_present[idx] && reorder_seq[idx] == want) {
                                    reorder_present[idx] = false;
                                    last_acked = want;
                                } else {
                                    break;
                                }
                            }
                        } else {
                            // Out-of-order, future seq — buffer.
                            reorder_present[seq % kReorderN] = true;
                            reorder_seq[seq % kReorderN] = seq;
                        }
                    }
                    // else: duplicate or stale — drop quietly.
                    frames_since_last_ack_send++;
                    if (frames_since_last_ack_send >= args.ack_every_frames) {
                        try_send_ack(0);
                    }
                }
                if (echo_frame(args.port_id, m) == 0) {
                    echoed++;
                }
            } else {
                // Free non-match mbufs without echoing. Echoing the ~6 Mfps
                // 0x88B5 burst storm saturates the TX ring and head-of-line
                // blocks the rare matched frame's echo path, so the receiver
                // never increments echoed past 0 even when the frame arrives.
                rte_pktmbuf_free(m);
            }
        }
    }

    // Flush a final ACK so the sender can confirm receipt of the last batch.
    try_send_ack(0);
    rte_delay_ms(50);  // let TX drain before stop

    std::printf(
        "Echoed %d frame(s). Tearing down. acks_sent=%llu last_acked=%u\n",
        echoed,
        (unsigned long long)acks_sent,
        last_acked == kAckSeed ? 0 : last_acked);

    rte_flow_error error{};
    for (int i = 0; i < kNumProbes; ++i) {
        if (probes[i].flow != nullptr) {
            rte_flow_destroy(args.port_id, probes[i].flow, &error);
        }
    }
    if (catchall != nullptr) {
        rte_flow_destroy(args.port_id, catchall, &error);
    }
    (void)flow;
    rte_eth_dev_stop(args.port_id);
    rte_eth_dev_close(args.port_id);
    rte_eal_cleanup();
    return echoed > 0 ? 0 : 2;
}
