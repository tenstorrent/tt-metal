// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side X280 ISS test harness. Embeds Spike's sim_t with the same
// Blackhole L2CPU map as scripts/x280_iss.sh, then:
//   * loads any RISC-V ELF and runs it to HTIF exit
//   * copies host files into guest physical memory after ELF load
//   * dumps guest physical memory to host files after the run
//
// Usage:
//   x280_harness [options] <elf>
//   --load  FILE@ADDR       write FILE bytes to guest physical ADDR
//   --dump  ADDR+LEN:FILE   after run, write LEN bytes from ADDR to FILE
//   --dump  ADDR:LEN:FILE   same (alternate spelling)

#include <riscv/cfg.h>
#include <riscv/debug_module.h>
#include <riscv/devices.h>
#include <riscv/sim.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDefaultIsa = "rv64gcv_zicsr_zifencei_zvl512b";
constexpr const char* kDefaultPriv = "MSU";
constexpr size_t kDefaultHarts = 4;
constexpr uint64_t kDefaultMemBase = 0x08000000ULL;
constexpr uint64_t kDefaultMemSize = 0x001E0000ULL;
constexpr uint64_t kDefaultPc = 0x08001000ULL;
constexpr unsigned long long kDefaultInsnLimit = 100000000ULL;

struct LoadOp {
    std::string path;
    uint64_t addr = 0;
    std::vector<uint8_t> bytes;
};

struct DumpOp {
    std::string path;
    uint64_t addr = 0;
    uint64_t len = 0;
};

struct Options {
    std::string elf;
    std::string isa = kDefaultIsa;
    std::string priv = kDefaultPriv;
    size_t nprocs = kDefaultHarts;
    uint64_t mem_base = kDefaultMemBase;
    uint64_t mem_size = kDefaultMemSize;
    uint64_t pc = kDefaultPc;
    unsigned long long insn_limit = kDefaultInsnLimit;
    bool has_insn_limit = true;
    std::vector<LoadOp> loads;
    std::vector<DumpOp> dumps;
};

void usage(const char* argv0, int rc) {
    std::cerr << "Usage: " << argv0 << " [options] <elf>\n"
              << "\n"
              << "Run a RISC-V ELF on a simulated Blackhole L2CPU X280 (Spike).\n"
              << "Host files can be copied into guest physical memory, and guest\n"
              << "memory can be dumped to host files after HTIF exit.\n"
              << "\n"
              << "Options:\n"
              << "  --load FILE@ADDR         Copy FILE into guest memory at ADDR\n"
              << "                           (applied after the ELF is loaded)\n"
              << "  --dump ADDR+LEN:FILE     After run, write LEN bytes at ADDR to FILE\n"
              << "  --dump ADDR:LEN:FILE     Same as ADDR+LEN:FILE\n"
              << "  --isa STRING             ISA string [" << kDefaultIsa << "]\n"
              << "  --priv STRING            Privilege modes [" << kDefaultPriv << "]\n"
              << "  -p N                     Hart count [" << kDefaultHarts << "]\n"
              << "  -m BASE:SIZE             Physical memory map ["
              << "0x" << std::hex << kDefaultMemBase << ":0x" << kDefaultMemSize << std::dec << "]\n"
              << "  --pc ADDR                Start PC [0x" << std::hex << kDefaultPc << std::dec << "]\n"
              << "  --instructions N         Retire at most N instructions [" << kDefaultInsnLimit << "]\n"
              << "  --no-instruction-limit   Run until HTIF exit\n"
              << "  -h, --help               This help\n"
              << "\n"
              << "ADDR/LEN/N accept decimal or 0x-prefixed hex.\n";
    std::exit(rc);
}

uint64_t parse_u64(const std::string& s, const char* what) {
    if (s.empty()) {
        throw std::runtime_error(std::string("empty ") + what);
    }
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 0);
    if (end == s.c_str() || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + what + ": " + s);
    }
    return static_cast<uint64_t>(v);
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open file for read: " + path);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    if (n < 0) {
        throw std::runtime_error("cannot stat file: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (n > 0 && !in.read(reinterpret_cast<char*>(buf.data()), n)) {
        throw std::runtime_error("short read: " + path);
    }
    return buf;
}

void write_file(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open file for write: " + path);
    }
    if (!buf.empty() &&
        !out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()))) {
        throw std::runtime_error("short write: " + path);
    }
}

LoadOp parse_load(const std::string& spec) {
    const auto at = spec.rfind('@');
    if (at == std::string::npos || at == 0 || at + 1 == spec.size()) {
        throw std::runtime_error("--load expects FILE@ADDR, got: " + spec);
    }
    LoadOp op;
    op.path = spec.substr(0, at);
    op.addr = parse_u64(spec.substr(at + 1), "load address");
    op.bytes = read_file(op.path);
    return op;
}

DumpOp parse_dump(const std::string& spec) {
    // ADDR+LEN:FILE  or  ADDR:LEN:FILE
    const auto colon = spec.rfind(':');
    if (colon == std::string::npos || colon + 1 == spec.size()) {
        throw std::runtime_error("--dump expects ADDR+LEN:FILE or ADDR:LEN:FILE, got: " + spec);
    }
    DumpOp op;
    op.path = spec.substr(colon + 1);
    const std::string left = spec.substr(0, colon);
    const auto plus = left.find('+');
    const auto mid = left.find(':');
    if (plus != std::string::npos) {
        op.addr = parse_u64(left.substr(0, plus), "dump address");
        op.len = parse_u64(left.substr(plus + 1), "dump length");
    } else if (mid != std::string::npos) {
        op.addr = parse_u64(left.substr(0, mid), "dump address");
        op.len = parse_u64(left.substr(mid + 1), "dump length");
    } else {
        throw std::runtime_error("--dump expects ADDR+LEN:FILE or ADDR:LEN:FILE, got: " + spec);
    }
    if (op.len == 0) {
        throw std::runtime_error("--dump length must be > 0");
    }
    return op;
}

void parse_mem(const std::string& spec, Options& opt) {
    const auto colon = spec.find(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("-m expects BASE:SIZE, got: " + spec);
    }
    opt.mem_base = parse_u64(spec.substr(0, colon), "memory base");
    opt.mem_size = parse_u64(spec.substr(colon + 1), "memory size");
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires an argument");
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage(argv[0], 0);
        } else if (a == "--load") {
            opt.loads.push_back(parse_load(need("--load")));
        } else if (a == "--dump") {
            opt.dumps.push_back(parse_dump(need("--dump")));
        } else if (a == "--isa") {
            opt.isa = need("--isa");
        } else if (a == "--priv") {
            opt.priv = need("--priv");
        } else if (a == "-p") {
            opt.nprocs = static_cast<size_t>(parse_u64(need("-p"), "hart count"));
            if (opt.nprocs == 0) {
                throw std::runtime_error("hart count must be > 0");
            }
        } else if (a == "-m") {
            parse_mem(need("-m"), opt);
        } else if (a == "--pc") {
            opt.pc = parse_u64(need("--pc"), "pc");
        } else if (a == "--instructions") {
            opt.insn_limit = parse_u64(need("--instructions"), "instructions");
            opt.has_insn_limit = true;
        } else if (a == "--no-instruction-limit") {
            opt.has_insn_limit = false;
        } else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown option: " + a);
        } else if (opt.elf.empty()) {
            opt.elf = a;
        } else {
            throw std::runtime_error("unexpected extra argument: " + a);
        }
    }
    if (opt.elf.empty()) {
        usage(argv[0], 1);
    }
    return opt;
}

class x280_sim_t : public sim_t {
public:
    using sim_t::sim_t;
    std::vector<LoadOp> loads;

    void start() override {
        htif_t::start();
        for (const auto& load : loads) {
            if (load.bytes.empty()) {
                continue;
            }
            memif().write(load.addr, load.bytes.size(), load.bytes.data());
            std::cerr << "[harness] loaded " << load.bytes.size() << " bytes from " << load.path << " @ 0x" << std::hex
                      << load.addr << std::dec << "\n";
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        cfg_t cfg;
        cfg.isa = opt.isa.c_str();
        cfg.priv = opt.priv.c_str();
        cfg.mem_layout = {mem_cfg_t(opt.mem_base, opt.mem_size)};
        cfg.start_pc.set_global(opt.pc);
        cfg.explicit_hartids = false;
        cfg.hartids.clear();
        cfg.hartids.reserve(opt.nprocs);
        for (size_t i = 0; i < opt.nprocs; ++i) {
            cfg.hartids.push_back(i);
        }

        std::vector<std::unique_ptr<mem_t>> owned_mems;
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems;
        for (const auto& region : cfg.mem_layout) {
            owned_mems.emplace_back(std::make_unique<mem_t>(region.get_size()));
            mems.emplace_back(region.get_base(), owned_mems.back().get());
        }

        debug_module_config_t dm_config;
        const std::vector<device_factory_sargs_t> plugins;
        std::vector<std::string> htif_args = {opt.elf};
        std::optional<unsigned long long> insn_limit;
        if (opt.has_insn_limit) {
            insn_limit = opt.insn_limit;
        }

        std::cerr << "[harness] elf=" << opt.elf << " isa=" << opt.isa << " harts=" << opt.nprocs << " mem=0x"
                  << std::hex << opt.mem_base << "+0x" << opt.mem_size << " pc=0x" << opt.pc << std::dec
                  << " loads=" << opt.loads.size() << " dumps=" << opt.dumps.size() << "\n";

        x280_sim_t sim(
            &cfg,
            false,
            mems,
            plugins,
            false,
            htif_args,
            dm_config,
            nullptr,
            false,
            nullptr,
            false,
            nullptr,
            insn_limit);
        sim.loads = std::move(opt.loads);

        const int rc = sim.run();

        for (const auto& dump : opt.dumps) {
            std::vector<uint8_t> buf(static_cast<size_t>(dump.len));
            sim.memif().read(dump.addr, buf.size(), buf.data());
            write_file(dump.path, buf);
            std::cerr << "[harness] dumped " << buf.size() << " bytes @ 0x" << std::hex << dump.addr << std::dec
                      << " -> " << dump.path << "\n";
        }

        std::cerr << "[harness] exit=" << rc << "\n";
        return rc;
    } catch (const std::exception& e) {
        std::cerr << "x280_harness: " << e.what() << "\n";
        return 2;
    }
}
