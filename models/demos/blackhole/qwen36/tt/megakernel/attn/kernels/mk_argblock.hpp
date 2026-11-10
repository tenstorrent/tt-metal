// Megakernel per-core arg blocks (engine/argblock.py, spec v1.25). The runtime-arg region and the CB-config block are
// program-wide (dispatch sizes them by the largest core / max CB index), so per-core tables do not travel as runtime
// args: each kind (phase table, AR roles, v12 slot tables, GDN page maps, attention) is one HEIGHT_SHARDED L1 tensor
// over the cores that use it, one shard (= this core's block) per core. A sharded tensor has the same L1 address on
// every core of its grid (and on every device of the mesh), so the address arrives as ONE common runtime arg: common
// arg index = kind id. Kernels read their block once at init.
//
// Block layout (u32 words): [0] magic 0x4D4B4142 "MKAB", [1] kind id, [2] payload words n, [3] format version,
// [4 .. 4+n) payload (owner-defined per kind; padded with zeros to a multiple of 4 words = 16 B).
#pragma once
#include <cstdint>

namespace mk {
constexpr uint32_t ARGBLOCK_MAGIC = 0x4D4B4142u;
constexpr uint32_t ARGBLOCK_HEADER = 4;
// kind ids = common runtime arg slots (keep in sync with engine/argblock.py KINDS)
constexpr uint32_t AB_PHASE = 0, AB_AR = 1, AB_V12 = 2, AB_GDN = 3, AB_ATTN = 4, AB_REPLAY = 5, AB_NKINDS = 6;

struct ArgBlock {
    volatile tt_l1_ptr uint32_t* p;
    FORCE_INLINE uint32_t operator[](uint32_t i) const { return p[ARGBLOCK_HEADER + i]; }
    FORCE_INLINE uint32_t size() const { return p[2]; }
    FORCE_INLINE uint32_t version() const { return p[3]; }
    FORCE_INLINE bool ok(uint32_t kind) const { return p[0] == ARGBLOCK_MAGIC && p[1] == kind; }
    FORCE_INLINE uint32_t addr() const { return reinterpret_cast<uint32_t>(p); }
};

// This core's block of `kind` (the block's L1 base address is common runtime arg `kind`).
FORCE_INLINE ArgBlock argblock(uint32_t kind) {
    return ArgBlock{reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(kind))};
}
}  // namespace mk
