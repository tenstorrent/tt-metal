// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDN decode step 1a (K2): the FPU passes and the one-token orchestration of
// ttnn/.../fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gdn_math.hpp on one Blackhole
// core, K = V = 128 (Kt = Vt = 4), HiFi4, fp32 DEST, against double references and the padded 32x32 forms, with
// MATH-thread cycle counts per pass and per token. Kernel: test_kernels/compute/gdn_decode_math.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace unit_tests::compute::gdn_decode_math {

constexpr uint32_t KT = 4, VT = 4, KD = KT * 32, VD = VT * 32, KV = KT * VT;
constexpr uint32_t kBank = 0;
constexpr uint32_t kStatsWords = 16;
constexpr uint32_t kRepsPass = 16;  // single-pass variants (the output CB holds every repetition)
constexpr uint32_t kRepsToken = 8;  // the token variant also keeps Kt*Vt state tiles per repetition in L1
const float kNaN = std::numeric_limits<float>::quiet_NaN();

// Circular buffer ids of the contract (section 3).
namespace cbid {
constexpr uint32_t qk16 = 0, kT16 = 1, qcol32 = 2, v16 = 3, gcol32 = 4, bcol16 = 5, S = 6, ones32 = 7, zero32 = 8;
constexpr uint32_t ecol32 = 9, r16 = 10, craw16 = 11, ccol32 = 12, u16 = 13, s2 = 14, s3 = 15, out16 = 16;
constexpr uint32_t state_out = 17;
constexpr uint32_t ebc16 = 19, S_mm = 20, s2_mm = 21;  // section 8 and the op's fused_recurrent_gdn_cb.hpp
}  // namespace cbid

enum Variant : uint32_t { kRead = 1, kUpdate = 2, kConstants = 3, kToken = 4 };

struct GdmTileSpec {
    uint32_t r, c;
    uint32_t fr() const { return std::min<uint32_t>(r, 16); }
    uint32_t fc() const { return std::min<uint32_t>(c, 16); }
    uint32_t elems() const { return r * c; }
    uint32_t bytes() const { return r * c * 4; }
};
constexpr GdmTileSpec T32{32, 32}, T16x32{16, 32}, T32x16{32, 16};

// Row-major [r x c] -> device tile order: faces row-major over the face grid, rows within a face.
std::vector<uint32_t> tilize(const std::vector<float>& rm, GdmTileSpec t) {
    std::vector<uint32_t> out;
    out.reserve(t.elems());
    for (uint32_t fy = 0; fy < t.r / t.fr(); fy++) {
        for (uint32_t fx = 0; fx < t.c / t.fc(); fx++) {
            for (uint32_t rr = 0; rr < t.fr(); rr++) {
                for (uint32_t cc = 0; cc < t.fc(); cc++) {
                    out.push_back(std::bit_cast<uint32_t>(rm[(fy * t.fr() + rr) * t.c + fx * t.fc() + cc]));
                }
            }
        }
    }
    return out;
}

std::vector<float> untilize(const uint32_t* tile, GdmTileSpec t) {
    std::vector<float> rm(t.elems());
    uint32_t n = 0;
    for (uint32_t fy = 0; fy < t.r / t.fr(); fy++) {
        for (uint32_t fx = 0; fx < t.c / t.fc(); fx++) {
            for (uint32_t rr = 0; rr < t.fr(); rr++) {
                for (uint32_t cc = 0; cc < t.fc(); cc++) {
                    rm[(fy * t.fr() + rr) * t.c + fx * t.fc() + cc] = std::bit_cast<float>(tile[n++]);
                }
            }
        }
    }
    return rm;
}

constexpr uint32_t kNoAlias = 0xFFFFFFFFu;

// One circular buffer of the program. `data` (tilized tiles, concatenated) is fed by the reader when non-empty;
// `count` tiles are allocated. l1_backed CBs sit on a host-known L1 buffer and are read back after the run. A spec
// with `alias_of` set is a second id over the memory of that CB (one CircularBufferConfig with two buffer indices,
// each with its own utd flag and bookkeeping); its `data` makes the reader push the same tiles under that id too.
struct GdmCb {
    uint32_t id;
    GdmTileSpec tile;
    uint32_t count;
    bool utd = false;
    std::vector<uint32_t> data;
    bool l1_backed = false;
    uint32_t alias_of = kNoAlias;
};

struct GdmOut {
    uint32_t id;
    uint32_t ntiles;
};

struct GdmRunConfig {
    uint32_t variant;
    uint32_t reps;
    bool emit_state = false;
    bool full_sync = false;
    bool startup_reverse = true;  // compute_kernel_hw_startup<SrcOrder::Reverse> (false: the regular order)
    bool startup_on_utd = false;  // token: start up on the utd S instead of its plain alias
    std::vector<GdmCb> cbs;
    std::vector<GdmOut> outs;
};

struct GdmRunResult {
    std::vector<std::vector<uint32_t>> out_words;  // per output: all tiles, device order
    std::vector<std::vector<float>> out_rm;        // per output: all tiles row-major, concatenated
    std::map<uint32_t, std::vector<uint32_t>> l1;  // l1_backed CBs: raw words after the run
    std::vector<uint32_t> stats;
};

struct GdmInputs {
    std::vector<float> S;     // [KD x VD]
    std::vector<float> k, q;  // k^ (unit), q~ = scale * q^
    std::vector<float> v, u;  // [VD]
    float beta = 0, g = 0;
};

GdmInputs make_inputs(uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ub(0.1f, 0.9f), ug(-2.0f, 0.0f);
    GdmInputs x;
    x.S.resize(KD * VD);
    for (auto& s : x.S) {
        s = 0.05f * nd(gen);
    }
    x.q.resize(KD);
    x.k.resize(KD);
    double nq = 0, nk = 0;
    for (uint32_t i = 0; i < KD; i++) {
        x.q[i] = nd(gen);
        x.k[i] = nd(gen);
        nq += double(x.q[i]) * x.q[i];
        nk += double(x.k[i]) * x.k[i];
    }
    for (uint32_t i = 0; i < KD; i++) {
        x.q[i] = float(x.q[i] / std::sqrt(nq) / std::sqrt(double(KD)));
        x.k[i] = float(x.k[i] / std::sqrt(nk));
    }
    x.v.resize(VD);
    x.u.resize(VD);
    for (auto& e : x.v) {
        e = nd(gen);
    }
    for (auto& e : x.u) {
        e = 0.1f * nd(gen);
    }
    x.beta = ub(gen);
    x.g = ug(gen);
    return x;
}

// ---- tile builders (tilized, concatenated) -----------------------------------------------------------------------
std::vector<uint32_t> cat(std::vector<uint32_t> a, const std::vector<uint32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// [k^; q~] in rows 0 and 2 of Kt tiles of `rows` x 32, the other rows `pad`.
std::vector<uint32_t> qk_tiles(const GdmInputs& x, uint32_t rows, float pad) {
    std::vector<uint32_t> out;
    const GdmTileSpec spec{rows, 32};
    for (uint32_t i = 0; i < KT; i++) {
        std::vector<float> rm(spec.elems(), pad);
        for (uint32_t c = 0; c < 32; c++) {
            rm[0 * 32 + c] = x.k[i * 32 + c];
            rm[2 * 32 + c] = x.q[i * 32 + c];
        }
        out = cat(out, tilize(rm, spec));
    }
    return out;
}

// k^ in column 0 of Kt tiles of 32 x (padded ? 32 : 16).
std::vector<uint32_t> kT_tiles(const GdmInputs& x, bool padded) {
    std::vector<uint32_t> out;
    const GdmTileSpec spec = padded ? T32 : T32x16;
    for (uint32_t i = 0; i < KT; i++) {
        std::vector<float> rm(spec.elems(), 0.0f);
        for (uint32_t r = 0; r < 32; r++) {
            rm[r * spec.c] = x.k[i * 32 + r];
        }
        out = cat(out, tilize(rm, spec));
    }
    return out;
}

// q~ in column 0 of Kt 32x32 tiles.
std::vector<uint32_t> qcol_tiles(const GdmInputs& x) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < KT; i++) {
        std::vector<float> rm(32 * 32, 0.0f);
        for (uint32_t r = 0; r < 32; r++) {
            rm[r * 32] = x.q[i * 32 + r];
        }
        out = cat(out, tilize(rm, T32));
    }
    return out;
}

// a [VD] vector in row 0 of Vt tiles of (padded ? 32 : 16) x 32.
std::vector<uint32_t> row_tiles(const std::vector<float>& vec, bool padded) {
    std::vector<uint32_t> out;
    const GdmTileSpec spec = padded ? T32 : T16x32;
    for (uint32_t j = 0; j < VT; j++) {
        std::vector<float> rm(spec.elems(), 0.0f);
        for (uint32_t c = 0; c < 32; c++) {
            rm[c] = vec[j * 32 + c];
        }
        out = cat(out, tilize(rm, spec));
    }
    return out;
}

// `value` in column 0 of every row of one rows x 32 tile, `rest` elsewhere.
std::vector<uint32_t> col_tile(float value, uint32_t rows, float rest = 0.0f) {
    const GdmTileSpec spec{rows, 32};
    std::vector<float> rm(spec.elems(), rest);
    for (uint32_t r = 0; r < rows; r++) {
        rm[r * 32] = value;
    }
    return tilize(rm, spec);
}

std::vector<uint32_t> const_tile(float value) { return tilize(std::vector<float>(32 * 32, value), T32); }

std::vector<uint32_t> s_tiles(const GdmInputs& x) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < KT; i++) {
        for (uint32_t j = 0; j < VT; j++) {
            std::vector<float> rm(32 * 32);
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t c = 0; c < 32; c++) {
                    rm[r * 32 + c] = x.S[(i * 32 + r) * VD + j * 32 + c];
                }
            }
            out = cat(out, tilize(rm, T32));
        }
    }
    return out;
}

// ---- the program --------------------------------------------------------------------------------------------------
GdmRunResult run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const GdmRunConfig& c) {
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    auto make_buffer = [&](uint32_t bytes, BufferType type) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = type, .bottom_up = false},
            mesh_device.get());
    };

    std::vector<std::shared_ptr<distributed::MeshBuffer>> keep_alive;
    std::map<uint32_t, std::pair<std::shared_ptr<distributed::MeshBuffer>, uint32_t>> l1_backing;
    std::map<uint32_t, GdmTileSpec> tile_of;
    std::vector<UnpackToDestMode> utd(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    std::vector<uint32_t> reader_args;
    uint32_t num_in = 0;
    for (const GdmCb& s : c.cbs) {
        tile_of[s.id] = s.tile;
        if (s.utd) {
            utd[s.id] = UnpackToDestMode::UnpackToDestFp32;
        }
        if (s.alias_of == kNoAlias) {
            const uint32_t bytes = s.tile.bytes();
            const uint32_t total = s.count * bytes;
            std::map<uint8_t, tt::DataFormat> fmt{{uint8_t(s.id), tt::DataFormat::Float32}};
            std::vector<const GdmCb*> members{&s};
            for (const GdmCb& a : c.cbs) {
                if (a.alias_of == s.id) {
                    TT_FATAL(a.count * a.tile.bytes() == total, "alias CB {} must match the size of CB {}", a.id, s.id);
                    fmt[uint8_t(a.id)] = tt::DataFormat::Float32;
                    members.push_back(&a);
                }
            }
            std::optional<CircularBufferConfig> cfg;
            if (s.l1_backed) {
                auto buf = make_buffer(total, BufferType::L1);
                keep_alive.push_back(buf);
                l1_backing[s.id] = {buf, total};
                cfg.emplace(total, fmt, *buf->get_device_buffer(zero_coord));
            } else {
                cfg.emplace(total, fmt);
            }
            for (const GdmCb* m : members) {
                const auto id = static_cast<CBIndex>(m->id);
                cfg->set_page_size(id, m->tile.bytes()).set_tile_dims(id, Tile({m->tile.r, m->tile.c}));
            }
            CreateCircularBuffer(program_, core, *cfg);
        }
        if (!s.data.empty()) {
            TT_FATAL(s.data.size() % s.tile.elems() == 0, "CB {} data is not whole tiles", s.id);
            const uint32_t ntiles = s.data.size() / s.tile.elems();
            TT_FATAL(ntiles <= s.count, "CB {}: {} tiles of data but {} allocated", s.id, ntiles, s.count);
            auto buf = make_buffer(s.data.size() * 4, BufferType::DRAM);
            keep_alive.push_back(buf);
            std::vector<uint32_t> data = s.data;
            distributed::WriteShard(cq, buf, data, zero_coord);
            reader_args.insert(reader_args.end(), {s.id, uint32_t(buf->address()), kBank, ntiles});
            num_in++;
        }
    }

    std::vector<uint32_t> writer_args;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> out_buffers;
    for (const GdmOut& o : c.outs) {
        const uint32_t bytes = tile_of.at(o.id).bytes();
        auto buf = make_buffer(o.ntiles * bytes, BufferType::DRAM);
        out_buffers.push_back(buf);
        writer_args.insert(writer_args.end(), {o.id, uint32_t(buf->address()), kBank, o.ntiles});
    }
    auto stats_buffer = make_buffer(kStatsWords * 4, BufferType::L1);

    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_gdn_decode_math.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {num_in}});
    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_gdn_decode_math.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {uint32_t(c.outs.size())}});
    auto compute = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/gdn_decode_math.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = c.full_sync,
            .unpack_to_dest_mode = utd,
            .math_approx_mode = false,
            .compile_args = {
                c.variant,
                c.reps,
                KT,
                VT,
                c.emit_state ? 1u : 0u,
                c.startup_reverse ? 1u : 0u,
                c.startup_on_utd ? 1u : 0u}});

    SetRuntimeArgs(program_, reader, core, reader_args);
    SetRuntimeArgs(program_, writer, core, writer_args);
    SetRuntimeArgs(program_, compute, core, {uint32_t(stats_buffer->address())});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    GdmRunResult res;
    for (size_t k = 0; k < c.outs.size(); k++) {
        std::vector<uint32_t> words;
        distributed::ReadShard(cq, words, out_buffers[k], zero_coord);
        const GdmTileSpec spec = tile_of.at(c.outs[k].id);
        std::vector<float> rm;
        for (uint32_t t = 0; t < c.outs[k].ntiles; t++) {
            const auto tile = untilize(words.data() + t * spec.elems(), spec);
            rm.insert(rm.end(), tile.begin(), tile.end());
        }
        res.out_words.push_back(std::move(words));
        res.out_rm.push_back(std::move(rm));
    }
    auto* device = mesh_device->get_devices().at(0);
    for (const auto& [id, backing] : l1_backing) {
        std::vector<uint32_t> words;
        detail::ReadFromDeviceL1(device, core, uint32_t(backing.first->address()), backing.second, words);
        res.l1[id] = std::move(words);
    }
    detail::ReadFromDeviceL1(device, core, uint32_t(stats_buffer->address()), kStatsWords * 4, res.stats);
    return res;
}

// ---- checks -------------------------------------------------------------------------------------------------------
bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float f) { return std::isfinite(f); });
}

bool bit_equal(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) { return a == b; }

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

// max |got - ref| / max |ref|
double max_rel_err(const std::vector<float>& got, const std::vector<double>& ref) {
    TT_FATAL(got.size() == ref.size(), "size mismatch {} vs {}", got.size(), ref.size());
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        num = std::max(num, std::fabs(double(got[i]) - ref[i]));
        den = std::max(den, std::fabs(ref[i]));
    }
    return den > 0 ? num / den : num;
}

// The repetitions of an output are the equal slices of its words; true when every one equals the last.
bool reps_identical(const std::vector<uint32_t>& words, uint32_t reps) {
    const size_t per = words.size() / reps;
    for (uint32_t r = 0; r + 1 < reps; r++) {
        if (!std::equal(words.begin() + r * per, words.begin() + (r + 1) * per, words.begin() + (reps - 1) * per)) {
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> last_rep(const std::vector<uint32_t>& words, uint32_t reps) {
    const size_t per = words.size() / reps;
    return {words.end() - per, words.end()};
}

std::vector<float> last_rep(const std::vector<float>& rm, uint32_t reps) {
    const size_t per = rm.size() / reps;
    return {rm.end() - per, rm.end()};
}

// Row `row` of each of the Vt consecutive tiles in `rm` (row-major, `spec` each), concatenated.
std::vector<float> row_of_tiles(const std::vector<float>& rm, GdmTileSpec spec, uint32_t row, uint32_t ntiles) {
    std::vector<float> out;
    for (uint32_t t = 0; t < ntiles; t++) {
        const float* tile = rm.data() + t * spec.elems();
        out.insert(out.end(), tile + row * spec.c, tile + (row + 1) * spec.c);
    }
    return out;
}

int32_t ulp_distance(float a, float b) {
    return std::abs(int32_t(std::bit_cast<uint32_t>(a)) - int32_t(std::bit_cast<uint32_t>(b)));
}

// ---- references ---------------------------------------------------------------------------------------------------
struct GdmRef {
    double e = 0, c = 0;
    std::vector<double> r_k, r_q, u, o;  // [VD]
    std::vector<double> S_new;           // [KD x VD]
};

GdmRef reference(const GdmInputs& x) {
    GdmRef ref;
    ref.e = std::exp(double(x.g));
    for (uint32_t i = 0; i < KD; i++) {
        ref.c += double(x.q[i]) * x.k[i];
    }
    ref.r_k.assign(VD, 0.0);
    ref.r_q.assign(VD, 0.0);
    for (uint32_t kk = 0; kk < KD; kk++) {
        for (uint32_t c = 0; c < VD; c++) {
            ref.r_k[c] += double(x.k[kk]) * x.S[kk * VD + c];
            ref.r_q[c] += double(x.q[kk]) * x.S[kk * VD + c];
        }
    }
    ref.u.resize(VD);
    ref.o.resize(VD);
    for (uint32_t c = 0; c < VD; c++) {
        ref.u[c] = double(x.beta) * (double(x.v[c]) - ref.e * ref.r_k[c]);
        ref.o[c] = ref.e * ref.r_q[c] + ref.c * ref.u[c];
    }
    ref.S_new.resize(KD * VD);
    for (uint32_t kk = 0; kk < KD; kk++) {
        for (uint32_t c = 0; c < VD; c++) {
            ref.S_new[kk * VD + c] = ref.e * x.S[kk * VD + c] + double(x.k[kk]) * ref.u[c];
        }
    }
    return ref;
}

// e * S + k^T u in tile order (i, j), row-major within the tile, for a given u.
std::vector<double> update_ref(const GdmInputs& x, double e, const std::vector<float>& u) {
    std::vector<double> ref;
    for (uint32_t i = 0; i < KT; i++) {
        for (uint32_t j = 0; j < VT; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t c = 0; c < 32; c++) {
                    const uint32_t kk = i * 32 + r, vv = j * 32 + c;
                    ref.push_back(e * x.S[kk * VD + vv] + double(x.k[kk]) * u[vv]);
                }
            }
        }
    }
    return ref;
}

// rows 0 (r_k) and 2 (r_q) of the Vt read-pass tiles as one vector, matching row_of_tiles(..., 0) + (..., 2)
std::vector<double> read_ref(const GdmRef& ref) {
    std::vector<double> out(ref.r_k);
    out.insert(out.end(), ref.r_q.begin(), ref.r_q.end());
    return out;
}

std::vector<float> read_rows(const std::vector<float>& rm_last, GdmTileSpec spec) {
    auto out = row_of_tiles(rm_last, spec, 0, VT);
    const auto q = row_of_tiles(rm_last, spec, 2, VT);
    out.insert(out.end(), q.begin(), q.end());
    return out;
}

// ---- tests --------------------------------------------------------------------------------------------------------

// (a) read pass: 16x32 [k^; q~] x 32x32 S, bit-equal to the padded 32x32 in0, within 5e-3 of double; a NaN in the
// padding rows of qk16 leaves the live rows unchanged. (f) probe: the same with S flagged UnpackToDestFp32 as the
// matmul in1 (reported, and the reason the state CBs carry a plain alias id for the matmul), and with the plain alias
// over a utd S (the form the token uses), which must equal the plain S bit for bit.
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeMathReadPass) {
    const GdmInputs x = make_inputs(11);
    const GdmRef ref = reference(x);
    const uint32_t reps = kRepsPass;
    // s_mode: 0 = S plain, 1 = S utd (the probe), 2 = S utd under c_6 with the matmul on a plain alias id (c_20).
    auto config = [&](uint32_t rows, float pad, uint32_t s_mode) {
        const GdmTileSpec out_spec{rows, 32};
        GdmRunConfig c{
            .variant = kRead,
            .reps = reps,
            .cbs =
                {{.id = cbid::qk16, .tile = out_spec, .count = KT, .data = qk_tiles(x, rows, pad)},
                 {.id = cbid::S, .tile = T32, .count = KV, .utd = s_mode != 0, .data = s_tiles(x)},
                 {.id = cbid::out16, .tile = out_spec, .count = VT * reps}},
            .outs = {{.id = cbid::out16, .ntiles = VT * reps}}};
        if (s_mode == 2) {
            c.cbs.push_back({.id = cbid::S_mm, .tile = T32, .count = KV, .data = s_tiles(x), .alias_of = cbid::S});
            c.variant = kRead + 16;  // the kernel reads S through c_20
        }
        return c;
    };
    auto tiny = run(this->devices_.at(0), config(16, 0.0f, 0));
    auto padded = run(this->devices_.at(0), config(32, 0.0f, 0));
    auto poisoned = run(this->devices_.at(0), config(16, kNaN, 0));
    auto s_utd = run(this->devices_.at(0), config(16, 0.0f, 1));
    auto s_alias = run(this->devices_.at(0), config(16, 0.0f, 2));

    const auto t_rows = read_rows(last_rep(tiny.out_rm[0], reps), T16x32);
    const auto p_rows = read_rows(last_rep(padded.out_rm[0], reps), T32);
    const auto n_rows = read_rows(last_rep(poisoned.out_rm[0], reps), T16x32);
    const auto u_rows = read_rows(last_rep(s_utd.out_rm[0], reps), T16x32);
    const double err_t = max_rel_err(t_rows, read_ref(ref));
    const double err_p = max_rel_err(p_rows, read_ref(ref));
    const double err_utd = max_rel_err(u_rows, read_ref(ref));
    const bool utd_same = bit_equal(tiny.out_words[0], s_utd.out_words[0]);
    const bool alias_same = bit_equal(tiny.out_words[0], s_alias.out_words[0]);
    EXPECT_TRUE(all_finite(t_rows));
    EXPECT_TRUE(bit_equal(t_rows, p_rows)) << "16x32 in0 differs bitwise from the padded 32x32 in0";
    EXPECT_LT(err_t, 5e-3);
    EXPECT_TRUE(bit_equal(n_rows, t_rows)) << "NaN in the padding rows of qk16 reached the live rows";
    EXPECT_TRUE(reps_identical(tiny.out_words[0], reps)) << "read pass is not bit-identical across repetitions";
    EXPECT_TRUE(alias_same) << "the plain alias of a utd S gives different matmul bits than a plain S";
    const double cyc = double(tiny.stats[1] - tiny.stats[0]) / reps;
    log_info(
        tt::LogTest,
        "read pass: 16x32 in0 bit-equal to padded {}; max rel err vs double {:.2e} (padded {:.2e}); NaN padding "
        "inert {}; identical across {} reps {}; cycles per pass {:.0f} (budget 1900), per product {:.1f}; padded pass "
        "{:.0f}",
        bit_equal(t_rows, p_rows),
        err_t,
        err_p,
        bit_equal(n_rows, t_rows),
        reps,
        reps_identical(tiny.out_words[0], reps),
        cyc,
        cyc / (VT * KT),
        double(padded.stats[1] - padded.stats[0]) / reps);
    log_info(
        tt::LogTest,
        "read pass, matmul in1 = S flagged UnpackToDestFp32: bit-equal to the plain S {}, max rel err vs double "
        "{:.2e}; plain alias id over the utd S: bit-equal to the plain S {}",
        utd_same,
        err_utd,
        alias_same);
}

// (b) update pass: exact seed + SFPU decay + 32x16 x 16x32 rank-1, bit-equal to the padded 32x32 x 32x32 form,
// within 5e-3 of e S + k^T u; a slot packed twice gives the same bits; a fresh acquire accumulates from zero.
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeMathUpdatePass) {
    const GdmInputs x = make_inputs(12);
    const double e = std::exp(double(x.g));
    const float e_f = std::exp(x.g);
    const std::vector<double> ref = update_ref(x, e, x.u);
    auto config = [&](bool padded, uint32_t reps, bool emit) {
        GdmRunConfig c{
            .variant = kUpdate,
            .reps = reps,
            .emit_state = emit,
            .cbs =
                {{.id = cbid::kT16, .tile = padded ? T32 : T32x16, .count = KT, .data = kT_tiles(x, padded)},
                 {.id = cbid::u16, .tile = padded ? T32 : T16x32, .count = VT, .data = row_tiles(x.u, padded)},
                 {.id = cbid::S, .tile = T32, .count = KV, .utd = true, .data = s_tiles(x)},
                 {.id = cbid::ecol32, .tile = T32, .count = 1, .utd = true, .data = col_tile(e_f, 32, 1.0f)},
                 {.id = cbid::out16, .tile = T32, .count = KV * reps},
                 {.id = cbid::state_out, .tile = T32, .count = emit ? KV * reps : 1}},
            .outs = {{.id = cbid::out16, .ntiles = KV * reps}}};
        if (emit) {
            c.outs.push_back({.id = cbid::state_out, .ntiles = KV * reps});
        }
        return c;
    };
    auto tiny = run(this->devices_.at(0), config(false, kRepsPass, false));
    auto padded = run(this->devices_.at(0), config(true, kRepsPass, false));
    auto twice = run(this->devices_.at(0), config(false, 4, true));
    // Cost decomposition (the kernel's variants 5 and 6): the same loop without the decay / without the rank-1.
    auto cfg_no_decay = config(false, kRepsPass, false);
    cfg_no_decay.variant = kUpdate + 3;
    auto cfg_no_rank1 = config(false, kRepsPass, false);
    cfg_no_rank1.variant = kUpdate + 4;
    auto no_decay = run(this->devices_.at(0), cfg_no_decay);
    auto no_rank1 = run(this->devices_.at(0), cfg_no_rank1);

    const auto t = last_rep(tiny.out_rm[0], kRepsPass), p = last_rep(padded.out_rm[0], kRepsPass);
    const double err_t = max_rel_err(t, ref), err_p = max_rel_err(p, ref);
    const bool twice_same = bit_equal(twice.out_words[0], twice.out_words[1]);
    const bool twice_vs_tiny = bit_equal(last_rep(twice.out_words[0], 4), last_rep(tiny.out_words[0], kRepsPass));
    EXPECT_TRUE(all_finite(t));
    EXPECT_TRUE(bit_equal(t, p)) << "32x16 x 16x32 rank-1 differs bitwise from the padded 32x32 x 32x32 form";
    EXPECT_LT(err_t, 5e-3);
    EXPECT_TRUE(reps_identical(tiny.out_words[0], kRepsPass)) << "update pass is not bit-identical across reps";
    EXPECT_TRUE(twice_same) << "a slot packed to S_next and to state_out gave different bits";
    EXPECT_TRUE(twice_vs_tiny) << "the double-pack run differs from the single-pack run";
    const double cyc = double(tiny.stats[1] - tiny.stats[0]) / kRepsPass;
    log_info(
        tt::LogTest,
        "update pass: tiny bit-equal to padded {}; max rel err vs double {:.2e} (padded {:.2e}); identical across "
        "{} reps {}; slot packed twice identical {} (and equal to the single pack {}); cycles per pass {:.0f} "
        "(budget 4500), per S tile {:.1f}; padded pass {:.0f}, double pack {:.0f}",
        bit_equal(t, p),
        err_t,
        err_p,
        kRepsPass,
        reps_identical(tiny.out_words[0], kRepsPass),
        twice_same,
        twice_vs_tiny,
        cyc,
        cyc / KV,
        double(padded.stats[1] - padded.stats[0]) / kRepsPass,
        double(twice.stats[1] - twice.stats[0]) / 4);
    log_info(
        tt::LogTest,
        "update pass decomposition, cycles per S tile (groups of 3): seed + rank-1 + pack {:.1f}; seed + decay + pack "
        "{:.1f}; seed + decay + rank-1 + pack {:.1f}",
        double(no_decay.stats[1] - no_decay.stats[0]) / (kRepsPass * KV),
        double(no_rank1.stats[1] - no_rank1.stats[0]) / (kRepsPass * KV),
        cyc / KV);
}

// (c) constants pass: e within 2 ulp of fp32 std::exp in column 0 of ecol32 (all 32 rows); ebc16 carries e / beta /
// c in column 0 rows 0-3 / 4-7 / 8-11 and +0.0 everywhere else; c within 2e-3 of the double dot product relative to
// |q~| |k^|.
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeMathConstantsPass) {
    const GdmInputs x = make_inputs(13);
    const GdmRef ref = reference(x);
    const uint32_t reps = kRepsPass;
    GdmRunConfig c{
        .variant = kConstants,
        .reps = reps,
        .cbs =
            {{.id = cbid::qk16, .tile = T16x32, .count = KT, .data = qk_tiles(x, 16, 0.0f)},
             {.id = cbid::qcol32, .tile = T32, .count = KT, .data = qcol_tiles(x)},
             {.id = cbid::gcol32, .tile = T32, .count = 1, .utd = true, .data = col_tile(x.g, 32)},
             {.id = cbid::bcol16, .tile = T16x32, .count = 1, .utd = true, .data = col_tile(x.beta, 16)},
             {.id = cbid::ones32, .tile = T32, .count = 1, .data = const_tile(1.0f)},
             {.id = cbid::ecol32, .tile = T32, .count = 1, .utd = true, .l1_backed = true},
             {.id = cbid::craw16, .tile = T16x32, .count = 1},
             {.id = cbid::out16, .tile = T16x32, .count = reps}},
        .outs = {{.id = cbid::out16, .ntiles = reps}}};
    auto res = run(this->devices_.at(0), c);
    const auto ecol = untilize(res.l1.at(cbid::ecol32).data(), T32);  // the last repetition's ecol32, from L1
    const auto ebc = last_rep(res.out_rm[0], reps);                   // ebc16, 16x32
    const float e_ref = std::exp(x.g);
    int32_t max_ulp = 0;
    for (uint32_t r = 0; r < 32; r++) {
        max_ulp = std::max(max_ulp, ulp_distance(ecol[r * 32], e_ref));
    }
    const float c_dev = ebc[8 * 32];
    bool e_rows = true, b_rows = true, c_rows = true, rest_zero = true;
    for (uint32_t r = 0; r < 16; r++) {
        for (uint32_t col = 0; col < 32; col++) {
            const uint32_t bits = std::bit_cast<uint32_t>(ebc[r * 32 + col]);
            if (col == 0 && r < 4) {
                e_rows &= bits == std::bit_cast<uint32_t>(ecol[0]);
            } else if (col == 0 && r < 8) {
                b_rows &= bits == std::bit_cast<uint32_t>(x.beta);
            } else if (col == 0 && r < 12) {
                c_rows &= bits == std::bit_cast<uint32_t>(c_dev);
            } else {
                rest_zero &= bits == 0u;
            }
        }
    }
    // c = q~ . k^ of two independent random directions is a heavily cancelling sum (|c| ~ |q~| |k^| / sqrt(K)), so
    // its error relative to |c| is not bounded by the operand precision; the bound that is: |dc| against |q~| |k^|.
    double nq = 0, nk = 0;
    for (uint32_t i = 0; i < KD; i++) {
        nq += double(x.q[i]) * x.q[i];
        nk += double(x.k[i]) * x.k[i];
    }
    const double c_scale = std::sqrt(nq) * std::sqrt(nk);
    const double c_err = std::fabs(double(c_dev) - ref.c) / std::fabs(ref.c);
    const double c_err_scaled = std::fabs(double(c_dev) - ref.c) / c_scale;
    EXPECT_LE(max_ulp, 2) << "e is more than 2 ulp from fp32 std::exp(g)";
    EXPECT_TRUE(e_rows) << "ebc16 rows 0-3 of column 0 are not e";
    EXPECT_TRUE(b_rows) << "ebc16 rows 4-7 of column 0 are not beta";
    EXPECT_TRUE(c_rows) << "ebc16 rows 8-11 of column 0 are not one value c";
    EXPECT_TRUE(rest_zero) << "ebc16 has non-zero bits outside column 0 rows 0-11";
    EXPECT_LT(c_err_scaled, 2e-3);
    EXPECT_TRUE(reps_identical(res.out_words[0], reps)) << "constants pass is not bit-identical across repetitions";
    log_info(
        tt::LogTest,
        "constants pass: g {:.5f} e {:.8f} (std::exp {:.8f}, {} ulp max over 32 rows of ecol32); ebc16 column 0: e "
        "rows 0-3 {}, beta rows 4-7 {}, c rows 8-11 {}, rest +0.0 {}; c {:.7f} (double {:.7f}, err {:.2e} relative "
        "to c, {:.2e} relative to |q~||k^| = {:.4f}); identical across {} reps {}; cycles per pass {:.0f} (budget 900)",
        x.g,
        ecol[0],
        e_ref,
        max_ulp,
        e_rows,
        b_rows,
        c_rows,
        rest_zero,
        c_dev,
        ref.c,
        c_err,
        c_err_scaled,
        c_scale,
        reps,
        reps_identical(res.out_words[0], reps),
        double(res.stats[1] - res.stats[0]) / reps);
}

// (d, e) one token end to end on one core, Kt = Vt = 4: u, o and S_new within 5e-3 of the double recurrence; u16 rows
// 1-15 exactly zero in L1; bit-identical across repetitions; the state packed to S_next and to state_out identical;
// cycles per pass and per token, half-sync and full-sync.
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeMathToken) {
    const GdmInputs x = make_inputs(14);
    const GdmRef ref = reference(x);
    const uint32_t reps = kRepsToken;
    auto config = [&](bool full_sync, bool startup_reverse = true, bool startup_on_utd = false) {
        return GdmRunConfig{
            .variant = kToken,
            .reps = reps,
            .emit_state = true,
            .full_sync = full_sync,
            .startup_reverse = startup_reverse,
            .startup_on_utd = startup_on_utd,
            .cbs =
                {{.id = cbid::qk16, .tile = T16x32, .count = KT, .data = qk_tiles(x, 16, 0.0f)},
                 {.id = cbid::kT16, .tile = T32x16, .count = KT, .data = kT_tiles(x, false)},
                 {.id = cbid::qcol32, .tile = T32, .count = KT, .data = qcol_tiles(x)},
                 {.id = cbid::v16, .tile = T16x32, .count = VT, .utd = true, .data = row_tiles(x.v, false)},
                 {.id = cbid::gcol32, .tile = T32, .count = 1, .utd = true, .data = col_tile(x.g, 32)},
                 {.id = cbid::bcol16, .tile = T16x32, .count = 1, .utd = true, .data = col_tile(x.beta, 16)},
                 {.id = cbid::S, .tile = T32, .count = KV, .utd = true, .data = s_tiles(x)},
                 {.id = cbid::S_mm, .tile = T32, .count = KV, .data = s_tiles(x), .alias_of = cbid::S},
                 {.id = cbid::ones32, .tile = T32, .count = 1, .data = const_tile(1.0f)},
                 {.id = cbid::zero32, .tile = T32, .count = 1, .utd = true, .data = const_tile(0.0f)},
                 {.id = cbid::ecol32, .tile = T32, .count = 1, .utd = true},
                 {.id = cbid::r16, .tile = T16x32, .count = VT, .utd = true},
                 {.id = cbid::craw16, .tile = T16x32, .count = 1},
                 {.id = cbid::ebc16, .tile = T16x32, .count = 1, .utd = true},
                 {.id = cbid::u16, .tile = T16x32, .count = VT, .l1_backed = true},
                 {.id = cbid::s2, .tile = T32, .count = KV, .utd = true, .l1_backed = true},
                 {.id = cbid::s2_mm, .tile = T32, .count = KV, .alias_of = cbid::s2},
                 {.id = cbid::out16, .tile = T16x32, .count = VT * reps},
                 {.id = cbid::state_out, .tile = T32, .count = KV * reps}},
            .outs = {{.id = cbid::out16, .ntiles = VT * reps}, {.id = cbid::state_out, .ntiles = KV * reps}}};
    };
    auto half = run(this->devices_.at(0), config(false));
    auto full = run(this->devices_.at(0), config(true));
    auto regular = run(this->devices_.at(0), config(false, /*startup_reverse=*/false));
    auto on_utd = run(this->devices_.at(0), config(false, /*startup_reverse=*/false, /*startup_on_utd=*/true));
    auto cfg_fused = config(false);
    cfg_fused.variant = kToken + 3;  // decode_token<Kt, Vt, FuseReadVector = true>
    auto fused = run(this->devices_.at(0), cfg_fused);

    const auto out_last = last_rep(half.out_rm[0], reps);
    const auto o = row_of_tiles(out_last, T16x32, 0, VT);
    const auto S_new = last_rep(half.out_rm[1], reps);
    std::vector<float> u(VD);
    bool u_pad_zero = true, o_pad_zero = true;
    const auto& u_words = half.l1.at(cbid::u16);
    for (uint32_t j = 0; j < VT; j++) {
        const auto tile = untilize(u_words.data() + j * T16x32.elems(), T16x32);
        for (uint32_t c = 0; c < 32; c++) {
            u[j * 32 + c] = tile[c];
        }
        for (uint32_t r = 1; r < 16; r++) {
            for (uint32_t c = 0; c < 32; c++) {
                u_pad_zero &= std::bit_cast<uint32_t>(tile[r * 32 + c]) == 0u;
                o_pad_zero &= std::bit_cast<uint32_t>(out_last[j * T16x32.elems() + r * 32 + c]) == 0u;
            }
        }
    }
    // The device state is in (i, j) tile order; the reference matrix is [KD x VD].
    std::vector<double> S_new_ref;
    for (uint32_t i = 0; i < KT; i++) {
        for (uint32_t j = 0; j < VT; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t cc = 0; cc < 32; cc++) {
                    S_new_ref.push_back(ref.S_new[(i * 32 + r) * VD + j * 32 + cc]);
                }
            }
        }
    }
    const double err_u = max_rel_err(u, ref.u), err_o = max_rel_err(o, ref.o), err_S = max_rel_err(S_new, S_new_ref);
    const bool reps_out = reps_identical(half.out_words[0], reps), reps_S = reps_identical(half.out_words[1], reps);
    const bool ring_same = bit_equal(half.l1.at(cbid::s2), last_rep(half.out_words[1], reps));
    const bool full_same =
        bit_equal(full.out_words[0], half.out_words[0]) && bit_equal(full.out_words[1], half.out_words[1]);
    const bool regular_same =
        bit_equal(regular.out_words[0], half.out_words[0]) && bit_equal(regular.out_words[1], half.out_words[1]);
    const bool on_utd_same =
        bit_equal(on_utd.out_words[0], half.out_words[0]) && bit_equal(on_utd.out_words[1], half.out_words[1]);
    const bool fused_same =
        bit_equal(fused.out_words[0], half.out_words[0]) && bit_equal(fused.out_words[1], half.out_words[1]);
    EXPECT_TRUE(fused_same) << "the fused read + vector form changes the bits";
    EXPECT_TRUE(all_finite(o) && all_finite(u) && all_finite(S_new));
    EXPECT_LT(err_u, 5e-3);
    EXPECT_LT(err_o, 5e-3);
    EXPECT_LT(err_S, 5e-3);
    EXPECT_TRUE(u_pad_zero) << "u16 rows 1-15 are not exactly zero in L1";
    EXPECT_TRUE(reps_out && reps_S) << "token outputs differ across repetitions";
    EXPECT_TRUE(ring_same) << "S_new packed to the ring CB and to state_out differ";
    EXPECT_TRUE(full_same) << "dst_full_sync_en changes the bits";
    EXPECT_TRUE(regular_same) << "the regular compute_kernel_hw_startup source order changes the bits";
    const auto& st = half.stats;
    const double per_token = double(st[1] - st[0]) / reps;
    log_info(
        tt::LogTest,
        "token: max rel err vs double u {:.2e} o {:.2e} S_new {:.2e}; u16 rows 1-15 zero {}; out16 rows 1-15 zero {}; "
        "identical across {} reps: o {} S {}; ring == state_out {}; full-sync bits == half-sync {}; regular startup "
        "order bits == reverse {}; startup on the utd S (today's op kernel) bits == plain {}",
        err_u,
        err_o,
        err_S,
        u_pad_zero,
        o_pad_zero,
        reps,
        reps_out,
        reps_S,
        ring_same,
        full_same,
        regular_same,
        on_utd_same);
    log_info(
        tt::LogTest,
        "token cycles (MATH thread, half-sync): per token {:.0f}; constants {:.0f} (budget 900), read {:.0f} (1900), "
        "vector {:.0f} (2400), update {:.0f} (4500); full-sync per token {:.0f}: constants {:.0f} read {:.0f} vector "
        "{:.0f} update {:.0f}",
        per_token,
        double(st[2]) / reps,
        double(st[3]) / reps,
        double(st[4]) / reps,
        double(st[5]) / reps,
        double(full.stats[1] - full.stats[0]) / reps,
        double(full.stats[2]) / reps,
        double(full.stats[3]) / reps,
        double(full.stats[4]) / reps,
        double(full.stats[5]) / reps);
    log_info(
        tt::LogTest,
        "token, fused read + vector form (half-sync): bits == two-pass form {}; per token {:.0f}; constants {:.0f}, "
        "read + vector {:.0f}, update {:.0f}",
        fused_same,
        double(fused.stats[1] - fused.stats[0]) / reps,
        double(fused.stats[2]) / reps,
        double(fused.stats[3]) / reps,
        double(fused.stats[5]) / reps);
}

}  // namespace unit_tests::compute::gdn_decode_math
}  // namespace tt::tt_metal
