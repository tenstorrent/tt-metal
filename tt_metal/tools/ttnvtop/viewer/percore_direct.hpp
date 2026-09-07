// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Read per-core utilisation straight from tt-kmd, with no collector and no /dev/shm.
//
// WHY THIS CAN EXIST NOW. The collector/viewer split was the right call when sampling meant
// UMD topology discovery: measured on an n300, that is 7.8 s before the first sample, which
// nobody can pay per TUI launch. The driver's PERCORE ioctls cost 0.5 ms for open, arm and
// a first read -- about 15000x less -- so for that source the separate writer has stopped
// earning its place.
//
// It is also MORE correct interactively. The sweep is armed by an open fd and released by
// the kernel when it closes, so sampling lasts exactly as long as somebody is watching:
// open the TUI and it arms, quit (or get killed) and it stops. No daemon, no stale files,
// no "did I remember to start the collector".
//
// WHAT IS GIVEN UP, and it is real: the shm path needs no device permissions at all, and
// the viewer's own header says "Zero UMD dependency -- safe to run as many instances".
// Direct mode opens /dev/tenstorrent, so a user who can read shm but not the device loses
// this. That is why it is a mode and not a replacement.
//
// THE FRAME LAYOUT IS NOT COMPILED IN. Geometry comes from the ioctl, which got it from
// firmware tags. The header moved once already (16 -> 64 B) and broke every tool that had
// assumed it; only the field OFFSETS WITHIN the header are assumed here, and a frame whose
// header is smaller than those offsets is rejected rather than decoded.

#pragma once

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/shm_schema.hpp"

namespace ttnvtop::direct {

constexpr unsigned long kIoctlRead = (0xFAUL << 8) | 20UL;
constexpr unsigned long kIoctlCtl = (0xFAUL << 8) | 21UL;
constexpr uint32_t kArmDram = 1u << 1;
constexpr int kMaxDies = 2;

// Offsets inside the frame header. These are a layout dependency and the only one here;
// everything else (frame size, core count, sample stride) comes from the ioctl.
constexpr uint32_t kOffRowMask = 8;
constexpr uint32_t kOffEpoch = 10;
constexpr uint32_t kOffCores = 12;
constexpr uint32_t kOffAiclk8 = 13;
constexpr uint32_t kOffFlags = 14;
constexpr uint32_t kOffDramRd = 4;
constexpr uint32_t kOffDramWr = 6;
// telem[] begins right after the original 16 B header; 12 dwords, mirroring
// percore_frame.py's TELEM_OFF/NTELEM and src/arc/tensix_percore.h.
constexpr uint32_t kOffTelem = 16;
constexpr uint32_t kTelemThrottler = 3;
constexpr uint32_t kTelemVcore = 4;
constexpr uint32_t kTelemAsicTemp = 5;
constexpr uint32_t kTelemVregTemp = 6;
constexpr uint32_t kTelemBoardTemp = 7;
constexpr uint32_t kTelemTdp = 8;
constexpr uint32_t kTelemTdc = 9;

inline uint32_t telem_dword(const uint8_t* f, uint32_t i) {
    uint32_t v = 0;
    std::memcpy(&v, f + kOffTelem + i * 4u, sizeof(v));
    return v;
}
constexpr uint32_t kMinHeaderBytes = 16;
constexpr uint16_t kP1000Invalid = 0xFFFF;
constexpr uint8_t kFlagSfpu = 1u << 4;

struct DieDesc {
    uint8_t location;
    uint8_t published;
    uint8_t reserved0[2];
    uint32_t frame_offset;
    uint32_t rx_frames;
    uint32_t frames;
};

struct ReadIn {
    uint32_t output_size_bytes;
    uint32_t frames_size;
    uint64_t frames_ptr;
    uint32_t reserved[4];
};

struct ReadOut {
    uint32_t output_size_bytes;
    uint32_t geometry;
    uint32_t header_bytes;
    uint32_t sample_bytes;
    uint32_t cores;
    uint32_t frame_bytes;
    uint32_t dies;
    uint32_t frames_needed;
    DieDesc die[kMaxDies];
    // Board-level, the PCIe edge rails only -- see ioctl.h. Not total board power.
    uint32_t slot_12v_w;
    uint32_t slot_3v3_w;
    uint32_t rest_of_chip_w;
    uint32_t mvddq_w;
};

struct ReadArg {
    ReadIn in;
    ReadOut out;
};

struct CtlIn {
    uint32_t output_size_bytes;
    uint8_t arm;
    uint8_t reserved0[3];
    uint32_t flags;
    uint32_t reserved[4];
};

struct CtlOut {
    uint32_t output_size_bytes;
    uint8_t armed;
    uint8_t driver_owns_arm;
    uint8_t reserved0[2];
    uint32_t flags;
    uint32_t reserved[4];
};

struct CtlArg {
    CtlIn in;
    CtlOut out;
};

inline uint16_t rd16(const uint8_t* p, uint32_t off) {
    uint16_t v;
    std::memcpy(&v, p + off, sizeof(v));
    return v;
}

// The sweep's (col, row) -> Wormhole Tensix NOC coordinates. Ported verbatim from the
// publisher so both sources label a core identically.
inline void noc_xy(uint32_t col, uint32_t row, uint8_t& x, uint8_t& y) {
    auto fx = [](uint32_t p, uint32_t n) -> uint8_t {
        const uint32_t q = (p + 1) / 2;
        return static_cast<uint8_t>(((p + 1) % 2 == 0) ? q : (n - 1 - q));
    };
    x = fx(col, 10);
    y = fx(row, 12);
}

// One die, presented in exactly the shm layout the renderer already consumes, so nothing
// downstream has to know where the data came from.
struct DieView {
    UtilShmHeader header{};
    std::vector<PerCoreView> cores;
};

class Source {
public:
    ~Source() { close_fd(); }

    // A GLOBALLY unique id, which "location" is not. The driver reports 0/1 per card, so on
    // a multi-card host every card's local die would claim 0 -- and the viewer SORTS and
    // filters (--chip N) on this field, so four n300s would collapse into two apparent
    // chips. Not the UMD unique id (this path never talks to UMD), but unique and stable,
    // which is all the field is used for here.
    //
    // The board serial comes from the driver's own sysfs, which is already exposed and
    // already unique per card. Shifted left one so the die bit cannot collide with a
    // neighbouring serial; Wormhole serials have a zero top byte, so nothing is lost.
    static uint64_t serial_for(const std::string& dev_path) {
        const size_t slash = dev_path.find_last_of('/');
        const std::string n = (slash == std::string::npos) ? dev_path : dev_path.substr(slash + 1);
        const std::string p = "/sys/class/tenstorrent/tenstorrent!" + n + "/tt_serial";
        FILE* f = std::fopen(p.c_str(), "r");
        if (f == nullptr) {
            return 0;
        }
        char buf[64] = {0};
        const size_t got = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        if (got == 0) {
            return 0;
        }
        return std::strtoull(buf, nullptr, 16);
    }

    // Opens, learns geometry, and arms. Returns false with `why` set on any of: no such
    // device, a driver without the ioctls, or firmware that publishes no per-core tags.
    bool open(const std::string& path, std::string& why) {
        fd_ = ::open(path.c_str(), O_RDWR);
        if (fd_ < 0) {
            why = "cannot open " + path + ": " + std::strerror(errno);
            return false;
        }
        ReadArg a{};
        a.in.output_size_bytes = sizeof(ReadOut);
        if (::ioctl(fd_, kIoctlRead, &a) != 0) {
            why = (errno == EINVAL || errno == ENOTTY)
                      ? "this tt-kmd has no PERCORE_READ ioctl"
                      : "per-core surface unavailable: " + std::string(std::strerror(errno));
            close_fd();
            return false;
        }
        geom_ = a.out;
        if (geom_.header_bytes < kMinHeaderBytes || geom_.cores == 0 || geom_.dies == 0) {
            why = "firmware reported a frame this build cannot decode";
            close_fd();
            return false;
        }
        buf_.resize(geom_.frames_needed);

        CtlArg c{};
        c.in.output_size_bytes = sizeof(CtlOut);
        c.in.arm = 1;
        c.in.flags = kArmDram;
        if (::ioctl(fd_, kIoctlCtl, &c) != 0) {
            why = "could not arm the sweep: " + std::string(std::strerror(errno));
            close_fd();
            return false;
        }
        serial_ = serial_for(path);
        views_.resize(geom_.dies);
        for (uint32_t i = 0; i < geom_.dies; ++i) {
            views_[i].cores.resize(geom_.cores);
        }
        return true;
    }

    uint32_t dies() const { return geom_.dies; }
    const DieView& view(uint32_t i) const { return views_[i]; }

    // Refill every die from one ioctl. Returns false only on a hard read failure; a die
    // with nothing published keeps its previous contents and is reported unpublished.
    bool update(uint64_t now_us) {
        ReadArg a{};
        a.in.output_size_bytes = sizeof(ReadOut);
        a.in.frames_size = static_cast<uint32_t>(buf_.size());
        a.in.frames_ptr = reinterpret_cast<uint64_t>(buf_.data());
        if (::ioctl(fd_, kIoctlRead, &a) != 0) {
            return false;
        }
        for (uint32_t d = 0; d < a.out.dies && d < views_.size(); ++d) {
            const DieDesc& desc = a.out.die[d];
            if (!desc.published) {
                continue;
            }
            decode(views_[d], buf_.data() + desc.frame_offset, a.out, desc, now_us);
        }

        // BOARD power, which needs every die and so cannot be done inside decode().
        //
        // A die's TDP does not include its VP/VPH/GDDR rails; the ARC keeps that as a
        // separate fixed figure (25 W on an n300). So the card's draw is the sum over its
        // dies of TDP + rest_of_chip + MVDDQ -- roughly 180 W under load against ~47 W on
        // the slot rails, the difference being the auxiliary connector.
        //
        // Attached to the LOCAL die only, like slot power, so summing the rendered chips
        // cannot double-count one card.
        if (a.out.rest_of_chip_w > 0) {
            uint32_t board_w = 0;
            for (uint32_t d = 0; d < a.out.dies && d < views_.size(); ++d) {
                if (!a.out.die[d].published) {
                    continue;
                }
                board_w += views_[d].header.tdp_w + a.out.rest_of_chip_w + a.out.mvddq_w;
            }
            for (uint32_t d = 0; d < a.out.dies && d < views_.size(); ++d) {
                if (a.out.die[d].location == 0) {
                    views_[d].header.board_power_w = static_cast<uint16_t>(board_w);
                }
            }
        }
        return true;
    }

private:
    void close_fd() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void decode(DieView& v, const uint8_t* f, const ReadOut& o, const DieDesc& desc, uint64_t now_us) {
        const uint16_t row_mask = rd16(f, kOffRowMask);
        const uint8_t flags = f[kOffFlags];
        const bool slot3_is_sfpu = (flags & kFlagSfpu) != 0;

        UtilShmHeader& h = v.header;
        std::memcpy(h.magic, kShmMagic, sizeof(h.magic));
        h.version = kShmVersion;
        h.struct_size = sizeof(PerCoreView);
        // See serial_for(): unique across cards, which desc.location alone is not.
        h.asic_id = (serial_ << 1) | desc.location;
        h.arch_id = 0;
        // Declare exactly what the ARC sweep samples -- the same contract the publisher
        // uses, so the viewer labels these columns identically whichever source it read.
        h.signal_sources =
            SIGNAL_SRC_COMPUTE | SIGNAL_SRC_ACTIVITY | SIGNAL_SRC_PACK | (slot3_is_sfpu ? 0u : SIGNAL_SRC_UNPACK);
        h.num_cores = o.cores;
        h.collector_pid = static_cast<uint32_t>(::getpid());
        h.aiclk_mhz = static_cast<uint32_t>(f[kOffAiclk8]) * 8u;
        // Per-die thermals and power, straight out of this frame. For a relayed frame
        // these are the REMOTE die's own values -- see UtilShmHeader for why the scaling
        // differs field by field.
        const uint32_t t_asic = telem_dword(f, kTelemAsicTemp);
        h.asic_temp_c16 = static_cast<uint16_t>(t_asic & 0xFFFFu);
        h.asic_max_c16 = static_cast<uint16_t>(t_asic >> 16);
        h.vreg_temp_c = static_cast<uint16_t>(telem_dword(f, kTelemVregTemp) & 0xFFFFu);
        h.board_temp_c = static_cast<uint16_t>(telem_dword(f, kTelemBoardTemp) & 0xFFFFu);
        h.vcore_mv = static_cast<uint16_t>(telem_dword(f, kTelemVcore) & 0xFFFFu);
        h.tdp_w = static_cast<uint16_t>(telem_dword(f, kTelemTdp) & 0xFFFFu);
        h.tdc_a = static_cast<uint16_t>(telem_dword(f, kTelemTdc) & 0xFFFFu);
        h.throttler = telem_dword(f, kTelemThrottler);

        // Slot power describes the CARD, not this die, so it is attached only to the local
        // die. Putting it on both would invite a viewer to sum eight dies and report twice
        // the board's slot draw.
        if (desc.location == 0) {
            h.slot_12v_w = static_cast<uint16_t>(o.slot_12v_w);
            h.slot_3v3_w = static_cast<uint16_t>(o.slot_3v3_w);
        }

        // FRESHNESS IS THE FRAME'S EPOCH, NOT THE FACT THAT WE READ IT. `published` is
        // sticky: a die whose sweep has stopped keeps published=1 and keeps handing back
        // the LAST frame it produced, forever (see tenstorrent_percore_die in ioctl.h).
        // Stamping last_update_us on every read therefore told the viewer a frozen frame
        // was live, and main.cpp's chip_stale check -- which exists precisely to catch
        // this -- could never fire in --direct mode.
        //
        // Observed on a quad-n300: one die rendered 75% activity and a 69-82% per-core
        // gradient while its own telemetry on the same row read 8 W, 11 A and 0.70 V,
        // against 26 W and 0.89 V on genuinely idle dies. The utilisation was real, just
        // minutes old -- left over from a matmul that had already finished.
        const uint16_t epoch = rd16(f, kOffEpoch);
        const bool fresh = (epoch != last_epoch_[desc.location]);
        if (fresh) {
            h.last_update_us = now_us;
        }
        if (h.epoch_us == 0) {
            h.epoch_us = now_us;
        }

        // DRAM is MB in THIS frame, so a rate needs the frame count over wall time rather
        // than an assumed 9.9 Hz -- a missed frame would otherwise inflate it.
        if (fresh) {
            dram_rd_acc_[desc.location] += rd16(f, kOffDramRd);
            dram_wr_acc_[desc.location] += rd16(f, kOffDramWr);
            last_epoch_[desc.location] = epoch;
        }
        const uint64_t since = now_us - dram_t0_[desc.location];
        if (since >= 1000000ULL) {
            h.dram_rd_mbps = static_cast<uint32_t>(dram_rd_acc_[desc.location] * 1000000ULL / since);
            h.dram_wr_mbps = static_cast<uint32_t>(dram_wr_acc_[desc.location] * 1000000ULL / since);
            dram_rd_acc_[desc.location] = 0;
            dram_wr_acc_[desc.location] = 0;
            dram_t0_[desc.location] = now_us;
        }

        // Samples are packed in sweep order over the rows the harvesting fuse left alive,
        // so the mapping is only recoverable with row_mask. Bit r set means row r skipped.
        const uint8_t* s = f + o.header_bytes;
        uint32_t slot = 0;
        for (uint32_t row = 0; row < 10 && slot < o.cores; ++row) {
            if ((row_mask >> row) & 1u) {
                continue;
            }
            for (uint32_t col = 0; col < 8 && slot < o.cores; ++col, ++slot) {
                const uint16_t m = rd16(s, slot * o.sample_bytes + 0);
                const uint16_t act = rd16(s, slot * o.sample_bytes + 2);
                const uint16_t u = rd16(s, slot * o.sample_bytes + 4);
                const uint16_t p = rd16(s, slot * o.sample_bytes + 6);
                PerCoreView& c = v.cores[slot];
                c = PerCoreView{};
                noc_xy(col, row, c.noc_x, c.noc_y);
                c.logical_x = static_cast<uint8_t>(col);
                c.logical_y = static_cast<uint8_t>(row);
                c.is_remote = desc.location;
                const bool invalid = (m == kP1000Invalid);
                c.dispatched = invalid ? 0 : 1;
                if (invalid) {
                    continue;
                }
                c.compute_busy_p1000 = m;
                c.dispatch_busy_p1000 = act;  // ACTIVITY; declared via SIGNAL_SRC_ACTIVITY
                c.pack_busy_p1000 = p;
                c.sfpu_busy_p1000 = slot3_is_sfpu ? u : 0;
                c.unpack_busy_p1000 = slot3_is_sfpu ? 0 : u;
                c.samples_seen = epoch;
            }
        }
    }

    int fd_ = -1;
    uint64_t serial_ = 0;
    ReadOut geom_{};
    std::vector<uint8_t> buf_;
    std::vector<DieView> views_;
    uint16_t last_epoch_[kMaxDies] = {0xFFFF, 0xFFFF};
    uint64_t dram_rd_acc_[kMaxDies] = {0, 0};
    uint64_t dram_wr_acc_[kMaxDies] = {0, 0};
    uint64_t dram_t0_[kMaxDies] = {0, 0};
};

}  // namespace ttnvtop::direct
