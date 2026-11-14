// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "generic_op.hpp"

namespace hpcg_tt {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::UnpackToDestMode;

void Op::operator()() const { ttnn::generic_op(io, prog); }

namespace {

const std::string SR_ROOT = "models/experimental/hpcg/tt/custom_ops/sparse_rows/kernels";
const std::string CG_ROOT = "models/experimental/hpcg/tt/custom_ops/cg_vector/kernels";

// L1_BUDGET, F_*, NGATH, NGRP, VDEP, GDEP and the CB tables: tt/shared_constants.h (shared with sparse_rows/op.py)
const std::vector<std::pair<std::string, uint32_t>> SR_CB = {HPCG_SPARSE_ROWS_CBS(HPCG_CB_PAIR)};
const std::vector<int> NOC_FLIP = {std::begin(NOC_FLIP_RECT), std::end(NOC_FLIP_RECT)};

uint8_t sr_cb(const char* name) {
    for (const auto& [k, v] : SR_CB) {
        if (k == name) {
            return (uint8_t)v;
        }
    }
    throw std::logic_error(name);
}

ComputeConfigDescriptor fp32_config(const std::vector<uint8_t>& fp32_cbs) {
    ComputeConfigDescriptor cfg;
    cfg.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    cfg.fp32_dest_acc_en = true;
    cfg.dst_full_sync_en = true;
    cfg.math_approx_mode = false;
    cfg.unpack_to_dest_mode.assign(64, UnpackToDestMode::Default);
    for (uint8_t i : fp32_cbs) {
        cfg.unpack_to_dest_mode[i] = UnpackToDestMode::UnpackToDestFp32;
    }
    return cfg;
}

std::vector<uint32_t> accessors(std::initializer_list<const Tensor*> ts) {
    std::vector<uint32_t> out;
    for (const Tensor* t : ts) {
        const auto a = accessor_args(*t);
        out.insert(out.end(), a.begin(), a.end());
    }
    return out;
}

const tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors& semaphores(int nc, int gx, const CoreRangeSet& cores) {
    static std::map<std::pair<int, int>, ProgramDescriptor::SemaphoreDescriptors> made;
    auto it = made.find({nc, gx});
    if (it == made.end()) {
        ProgramDescriptor::SemaphoreDescriptors s;
        for (uint32_t i = 0; i < 16; ++i) {
            s.push_back(SemaphoreDescriptor{.id = i, .core_ranges = cores, .initial_value = 0});
        }
        it = made.emplace(std::make_pair(nc, gx), std::move(s)).first;
    }
    return it->second;
}

}  // namespace

Op sparse_row_op(
    const Dev& D,
    Plan& plan,
    const Tensor& values,
    const Tensor& offsets,
    const Tensor& x,
    const Tensor& out,
    const SparseRowArgs& a,
    const Tensor& dbg) {
    if (a.tpw != 256 && a.tpw != TILE) {
        throw std::invalid_argument("tpw must be 256 or 1024");
    }
    const std::vector<int64_t>& table = a.table_phases ? *a.table_phases : a.phases;
    const Meta& meta = plan.meta(D, table, a.tpw);
    std::vector<int64_t> nseg = meta.nseg, nseg_gs = meta.nseg_gs, nseg_r = meta.nseg_r;
    uint32_t nph = 0;
    if (a.table_phases) {
        if (!std::equal(a.phases.begin(), a.phases.end(), table.begin())) {
            throw std::invalid_argument("phases must be a prefix of table_phases");
        }
        nph = (uint32_t)a.phases.size();
        const bool gs_only = std::all_of(a.phases.begin(), a.phases.end(), [](int64_t p) { return p < 8; });
        const bool spmv_rest = std::all_of(table.begin() + nph, table.end(), [](int64_t p) { return p >= 8; });
        if (gs_only && spmv_rest) {  // the table's Gauss-Seidel counts are this launch's; no RESTF segments
            nseg = nseg_gs;
            nseg_r.assign(plan.nc, 0);
        } else {
            plan.segment_counts(a.phases, a.tpw, nseg, nseg_gs, nseg_r);
        }
    }
    const uint32_t meta_b = (uint32_t)(meta.w * 4);
    const int64_t ntlv = plan.nslot == 27 ? slot_pages(plan.pout / TILE) : plan.pout / TILE;
    const bool offw = plan.nslot == 27;
    std::vector<std::pair<std::string, uint32_t>> defs = SR_CB;
    defs.emplace_back("NG", NGATH);
    defs.emplace_back("NSLOT", plan.nslot);
    defs.emplace_back("NGRP", std::min(NGRP, plan.nslot));
    defs.emplace_back("TPW", (uint32_t)a.tpw);
    if (a.gs) {
        defs.emplace_back("GS", 1);
    }
    if (offw) {
        defs.emplace_back("OFFW", 1);
        defs.emplace_back("FASTMAC", 1);
    }
    const uint32_t flags =
        (a.wsplit ? F_WSPLIT : 0) | (a.zerox ? F_ZEROX : 0) | (a.restf && a.out2 ? F_RESTF : 0) | (a.addb ? F_ADDB : 0);
    const Tensor& b = a.b ? *a.b : x;
    const Tensor& o2 = a.out2 ? *a.out2 : out;
    const CoreRangeSet cores = core_ranges(plan.nc, D.gx);
    const std::vector<uint32_t> reader_common = {
        addr(x),
        addr(values),
        addr(offsets),
        addr(meta.t),
        addr(b),
        (uint32_t)D.gx,
        addr(dbg),
        meta_b,
        (uint32_t)ntlv,
        (uint32_t)plan.win_b,
        flags,
        nph};
    const std::vector<uint32_t> writer_common = {
        addr(out),
        addr(meta.t),
        (uint32_t)D.gx,
        addr(offsets),
        0,
        0,
        addr(dbg),
        addr(values),
        addr(o2),
        addr(x),
        meta_b,
        (uint32_t)ntlv,
        flags,
        nph};
    // META's accessor takes the values' compile-time arguments (4 KB pages): the page size is a compile-time
    // argument, only a default the kernels override with META's row size at run time
    const auto reader_ct = accessors({&x, &values, &offsets, &values, &b, &dbg});
    const auto writer_ct = accessors({&out, &values, &offsets, &dbg, &values, &o2, &x});

    const uint32_t pg = (uint32_t)a.tpw * 4;
    const bool face = a.tpw == 256;
    std::vector<int> own(plan.nslot);
    for (int j = 0; j < plan.nslot; ++j) {
        own[j] = j % NGATH;
    }
    const int c2 = (int)std::count(own.begin(), own.end(), 2), c3 = (int)std::count(own.begin(), own.end(), 3);
    const int ring = std::max(2 * c2, 1);
    defs.emplace_back("RING", ring);
    const int ngrp = std::min(NGRP, plan.nslot);
    const int gsz = (plan.nslot + ngrp - 1) / ngrp;
    int mg[2];
    for (int o = 0; o < 2; ++o) {
        int most = 0;
        for (int lo = 0; lo < plan.nslot; lo += gsz) {
            int cnt = 0;
            for (int j = lo; j < std::min(lo + gsz, plan.nslot); ++j) {
                cnt += own[j] == o;
            }
            most = std::max(most, cnt);
        }
        mg[o] = std::max(1, most);
    }
    int vpages = 0, nvw = 0;
    if (offw) {
        const int nvr = (int)std::count_if(own.begin(), own.end(), [](int o) { return o != 1; });
        nvw = plan.nslot - nvr;
        vpages = nvr * std::max(1, VDEP / nvr);
    } else {
        vpages = std::min(VDEP, 2 * plan.nslot);
    }
    const uint32_t win_b = (uint32_t)plan.win_b;
    ProgramDescriptor::CBDescriptors cbs;
    uint64_t total = 0;
    for (int gd : GDEP) {  // large grids: fewer gathered pages in flight so the circular buffers fit L1
        cbs.clear();
        auto add = [&](const char* name, uint32_t nbytes, uint32_t page, bool tile) {
            cbs.push_back(cb(cores, sr_cb(name), nbytes, page, tile && face));
        };
        add("CB_V", vpages * pg, pg, true);
        add("CB_G", gd * pg, pg, true);
        add("CB_G2", gd * pg, pg, true);
        add("CB_RDY", 64, 32, false);
        add("CB_OUT", 2 * pg, pg, true);
        add("CB_WIN", win_b, win_b, false);
        add("CB_OFF", 2 * mg[0] * pg, pg, false);
        add("CB_META", meta_b, meta_b, false);
        add("CB_META2", meta_b, meta_b, false);
        add("CB_OFF2", 2 * mg[1] * pg, pg, false);
        add("CB_G3", gd * pg, pg, true);
        add("CB_OFF3", ring * pg, pg, false);
        add("CB_MB", 128, 128, false);
        add("CB_G4", 4 * pg, pg, true);
        add("CB_OFF4", 2 * std::max(1, c3) * pg, pg, false);
        if (a.gs || a.addb) {
            add("CB_B", 2 * pg, pg, true);
        }
        if (offw) {
            add("CB_V2", 2 * nvw * pg, pg, true);
        }
        total = 0;
        for (const auto& c : cbs) {
            total += c.total_size;
        }
        if (total <= L1_BUDGET) {
            break;
        }
    }
    if (total > L1_BUDGET) {
        throw std::runtime_error("L1: circular buffers exceed the budget; grid too large");
    }
    const std::vector<int>& flip = a.noc_flip ? *a.noc_flip : ((!a.gs || a.out2) ? NOC_FLIP : std::vector<int>{});
    KernelDescriptor::Defines define_list;
    for (const auto& [k, v] : defs) {
        define_list.emplace_back(k, std::to_string(v));
    }
    ProgramDescriptor prog;
    for (const KernelGroup& g : kernel_groups(plan.nc, D.gx, flip)) {
        KernelDescriptor r;
        r.kernel_source = SR_ROOT + "/reader.cpp";
        r.source_type = KernelDescriptor::SourceType::FILE_PATH;
        r.core_ranges = g.cores;
        r.compile_time_args = reader_ct;
        r.defines = define_list;
        r.common_runtime_args = reader_common;
        r.config = g.reader;
        prog.kernels.push_back(r);
        KernelDescriptor w;
        w.kernel_source = SR_ROOT + "/writer.cpp";
        w.source_type = KernelDescriptor::SourceType::FILE_PATH;
        w.core_ranges = g.cores;
        w.compile_time_args = writer_ct;
        w.defines = define_list;
        w.common_runtime_args = writer_common;
        w.config = g.writer;
        prog.kernels.push_back(w);
    }
    KernelDescriptor c;
    c.kernel_source = SR_ROOT + "/compute.cpp";
    c.source_type = KernelDescriptor::SourceType::FILE_PATH;
    c.core_ranges = cores;
    c.defines = define_list;
    for (int k = 0; k < plan.nc; ++k) {
        c.runtime_args.emplace_back(
            D.logical[k],
            KernelDescriptor::CoreRuntimeArgs{
                (uint32_t)nseg[k], (uint32_t)nseg_gs[k], (uint32_t)nseg_r[k], flags, nph});
    }
    c.config = fp32_config(
        {sr_cb("CB_V"), sr_cb("CB_G"), sr_cb("CB_B"), sr_cb("CB_G2"), sr_cb("CB_G3"), sr_cb("CB_G4"), sr_cb("CB_V2")});
    prog.kernels.push_back(c);
    prog.semaphores = semaphores(plan.nc, D.gx, cores);
    prog.cbs = cbs;
    Op op;
    op.prog = std::move(prog);
    // generic_op's last tensor is the launch's output; gs updates x in place
    if (!a.gs) {
        op.io = {x, values, offsets, meta.t, out};
    } else {
        op.io = {values, offsets, meta.t, b};
        if (a.out2) {
            op.io.push_back(*a.out2);
        }
        op.io.push_back(x);
    }
    op.keep = {dbg};
    return op;
}

namespace {
const std::vector<std::pair<std::string, uint32_t>> VEC_CB = {HPCG_CG_VECTOR_CBS(HPCG_CB_PAIR)};  // VB: shared header
const std::vector<std::pair<std::string, uint32_t>> FINISH_CB = {HPCG_DOT_FINISH_CBS(HPCG_CB_PAIR)};

uint32_t cb_index(const std::vector<std::pair<std::string, uint32_t>>& table, const char* name) {
    for (const auto& [k, v] : table) {
        if (k == name) {
            return v;
        }
    }
    throw std::logic_error(name);
}

CBDescriptor page_cb(const CoreRangeSet& cores, uint32_t idx, uint32_t pages) {
    return cb(cores, (uint8_t)idx, pages * 4096, 4096, false);
}

struct VecParts {
    CoreRangeSet cores;
    ProgramDescriptor::CBDescriptors cbs;
    KernelDescriptor::Defines defs;
    KernelDescriptor compute;
};

const VecParts& vec_parts(const Dev& D, int nc, int base, int rem, int mcode) {
    static std::map<std::tuple<int, int, int, int>, VecParts> made;
    const auto key = std::make_tuple(nc, base, rem, mcode);
    auto it = made.find(key);
    if (it != made.end()) {
        return it->second;
    }
    VecParts p;
    p.cores = core_ranges(nc, D.gx);
    for (const char* k : {"CB_A", "CB_B", "CB_C", "CB_D"}) {
        for (const auto& [n, v] : VEC_CB) {
            if (n == k) {
                p.cbs.push_back(page_cb(p.cores, v, 2 * VB));
            }
        }
    }
    for (const char* k : {"CB_ACC", "CB_OUT", "CB_OUT2", "CB_OUT3"}) {
        for (const auto& [n, v] : VEC_CB) {
            if (n == k) {
                p.cbs.push_back(page_cb(p.cores, v, 2));
            }
        }
    }
    p.cbs.push_back(page_cb(p.cores, cb_index(VEC_CB, "CB_S"), 1));
    for (const auto& [k, v] : VEC_CB) {
        p.defs.emplace_back(k, std::to_string(v));
    }
    p.defs.emplace_back("VB", std::to_string(VB));
    p.compute.kernel_source = CG_ROOT + "/compute.cpp";
    p.compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    p.compute.core_ranges = p.cores;
    p.compute.defines = p.defs;
    for (int i = 0; i < nc; ++i) {
        p.compute.runtime_args.emplace_back(
            D.logical[i], KernelDescriptor::CoreRuntimeArgs{(uint32_t)(base + (i < rem ? 1 : 0)), (uint32_t)mcode});
    }
    p.compute.config = fp32_config({0, 1, 2, 3, 4, 5});  // CB_A .. CB_ACC
    return made.emplace(key, std::move(p)).first->second;
}
}  // namespace

VecOp cg_vector_op(
    const Dev& D,
    VecMode mode,
    const Tensor& a,
    const Tensor& b,
    const Tensor* c,
    const Tensor* d,
    const Tensor* s1,
    const Tensor* s2,
    const Tensor* part,
    int64_t P) {
    const int64_t nt = P / TILE;
    const int nc = (int)std::min<int64_t>(D.nc, nt);
    const int base = (int)(nt / nc), rem = (int)(nt % nc);
    const Tensor& C = c ? *c : a;
    const Tensor& Dd = d ? *d : a;
    const Tensor& S1 = s1 ? *s1 : a;
    const Tensor& S2 = s2 ? *s2 : a;
    const Tensor& PART = part ? *part : a;
    const Tensor& o1 = mode == VecMode::axpy ? b : a;  // axpy writes over b; xr writes x (a) and r (c)
    const Tensor& o2 = mode == VecMode::dot || mode == VecMode::zero ? b : C;  // dot: the writer reads b
    const uint32_t mcode = (uint32_t)mode;
    const VecParts& p = vec_parts(D, nc, base, rem, (int)mcode);
    std::vector<uint32_t> reader_common = {
        addr(a), addr(b), addr(C), addr(Dd), addr(S1), addr(S2), (uint32_t)base, (uint32_t)rem, (uint32_t)D.gx, mcode};
    std::vector<uint32_t> writer_common = {
        addr(o1), addr(o2), addr(PART), (uint32_t)base, (uint32_t)rem, (uint32_t)D.gx, addr(Dd), mcode};
    KernelDescriptor r;
    r.kernel_source = CG_ROOT + "/reader.cpp";
    r.source_type = KernelDescriptor::SourceType::FILE_PATH;
    r.core_ranges = p.cores;
    r.compile_time_args = accessors({&a, &b, &C, &Dd, &S1, &S2});
    r.defines = p.defs;
    r.common_runtime_args = reader_common;
    r.config = tt::tt_metal::ReaderConfigDescriptor{};
    KernelDescriptor w;
    w.kernel_source = CG_ROOT + "/writer.cpp";
    w.source_type = KernelDescriptor::SourceType::FILE_PATH;
    w.core_ranges = p.cores;
    w.compile_time_args = accessors({&o1, &o2, &PART, &Dd});
    w.defines = p.defs;
    w.common_runtime_args = writer_common;
    w.config = tt::tt_metal::WriterConfigDescriptor{};
    VecOp op;
    op.prog.kernels.push_back(r);
    op.prog.kernels.push_back(w);
    op.prog.kernels.push_back(p.compute);
    op.prog.cbs = p.cbs;
    op.io = {a, b, C, Dd, S1, S2, mode != VecMode::axpy && mode != VecMode::zero ? PART : o1};
    op.nc = nc;
    return op;
}

Op dot_finish_op(
    const Dev& D, const Tensor& part, int nparts, const Tensor& out, const Tensor* old, bool keep_old, bool sqrt) {
    if (keep_old && old == nullptr) {
        throw std::invalid_argument("keep_old needs the old tile");
    }
    const Tensor& O = old ? *old : out;
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    KernelDescriptor::Defines defs;
    for (const auto& [k, v] : FINISH_CB) {
        defs.emplace_back(k, std::to_string(v));
    }
    defs.emplace_back("VB", std::to_string(VB));
    const auto cfg = fp32_config({0});
    const CoreCoord c0(0, 0);
    KernelDescriptor r;
    r.kernel_source = CG_ROOT + "/finish_reader.cpp";
    r.source_type = KernelDescriptor::SourceType::FILE_PATH;
    r.core_ranges = cores;
    r.compile_time_args = accessors({&part});
    r.defines = defs;
    r.runtime_args = {{c0, {addr(part), (uint32_t)nparts}}};
    r.config = tt::tt_metal::ReaderConfigDescriptor{};
    KernelDescriptor w;
    w.kernel_source = CG_ROOT + "/finish_writer.cpp";
    w.source_type = KernelDescriptor::SourceType::FILE_PATH;
    w.core_ranges = cores;
    w.compile_time_args = accessors({&out, &O});
    w.defines = defs;
    w.runtime_args = {{c0, {addr(out), addr(O), (uint32_t)keep_old}}};
    w.config = tt::tt_metal::WriterConfigDescriptor{};
    KernelDescriptor c;
    c.kernel_source = CG_ROOT + "/finish_compute.cpp";
    c.source_type = KernelDescriptor::SourceType::FILE_PATH;
    c.core_ranges = cores;
    c.defines = defs;
    c.runtime_args = {{c0, {(uint32_t)nparts, (uint32_t)sqrt}}};
    c.config = cfg;
    Op op;
    op.prog.kernels.push_back(r);
    op.prog.kernels.push_back(w);
    op.prog.kernels.push_back(c);
    op.prog.cbs.push_back(page_cb(cores, cb_index(FINISH_CB, "CB_A"), 2 * VB));
    op.prog.cbs.push_back(page_cb(cores, cb_index(FINISH_CB, "CB_OUT"), 1));
    op.prog.cbs.push_back(page_cb(cores, cb_index(FINISH_CB, "CB_T"), 1));
    op.io = {part, O, out};
    return op;
}

}  // namespace hpcg_tt
